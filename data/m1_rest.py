"""
Fugle REST API 分K 收集器（替代 WebSocket 版本）

優點：
- 無連線數/訂閱數限制（不受 Fugle WebSocket plan 限制）
- 每分鐘 poll 一次，無需斷線重連邏輯
- 可同時監控大量股票（concurrent requests）

使用方式：
    def on_minute(minute_str, df):
        pass

    poller = M1RestPoller(on_minute=on_minute, stocks=stock_list_or_callable)
    poller.start()   # 阻塞直到 poller.stop()
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from fugle_marketdata import RestClient

from data.fugle_tickers import fugle_stocks

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_TW = timezone(timedelta(hours=8))
_LIVE_DIR = _ROOT / "db/m1_live"
_MAX_WORKERS = int(os.environ.get("REST_WORKERS", "10"))
_REQ_INTERVAL = float(os.environ.get("REST_INTERVAL", "0.15"))  # 每個 request 最小間隔（秒）
_RATE_LIMIT_BACKOFF = int(os.environ.get("REST_RATE_LIMIT_BACKOFF", "300"))
_BACKFILL_INTERVAL = float(os.environ.get("REST_BACKFILL_INTERVAL", "2.1"))
_BACKFILL_MAX_RETRIES = int(os.environ.get("REST_BACKFILL_MAX_RETRIES", "2"))
_BASE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"

_POLL_START_HOUR = int(os.environ.get("REST_POLL_START_HOUR", "9"))
_POLL_START_MIN = int(os.environ.get("REST_POLL_START_MIN", "1"))
_POLL_END_HOUR = int(os.environ.get("REST_POLL_END_HOUR", os.environ.get("SESSION_END_HOUR", "13")))
_POLL_END_MIN = int(os.environ.get("REST_POLL_END_MIN", os.environ.get("SESSION_END_MIN", "0")))
_POLL_START = (_POLL_START_HOUR, _POLL_START_MIN)
_POLL_END = (_POLL_END_HOUR, _POLL_END_MIN)


def _live_path(date_str: str) -> Path:
    return _LIVE_DIR / f"{date_str}.parquet"


def _minute_tuple(dt: datetime) -> tuple[int, int]:
    return dt.hour, dt.minute


def _in_poll_window(closed_minute: datetime) -> bool:
    return _POLL_START <= _minute_tuple(closed_minute) <= _POLL_END


def _before_poll_start(dt: datetime) -> bool:
    return _minute_tuple(dt) < _POLL_START


def _after_poll_end(dt: datetime) -> bool:
    return _minute_tuple(dt) > _POLL_END


def _first_poll_time(day: datetime) -> datetime:
    return day.replace(
        hour=_POLL_START_HOUR,
        minute=_POLL_START_MIN,
        second=5,
        microsecond=0,
    )


def _next_poll_window_start(now: datetime) -> datetime:
    today_first = _first_poll_time(now)
    if now < today_first:
        return today_first
    return _first_poll_time(now + timedelta(days=1))


def _sleep_until(dt: datetime, stop_check, label: str):
    while not stop_check():
        wait = (dt - datetime.now(_TW)).total_seconds()
        if wait <= 0:
            return
        print(f"{label}，下次啟動 {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(min(wait, 300))


def _parse_rest_bars(symbol: str, bars: list, date_str: str) -> pd.DataFrame:
    """將 Fugle REST API 回傳的 bars 轉換為標準分K格式"""
    rows = []
    for b in bars:
        # date format: '2026-07-02T09:00:00.000+08:00'
        dt = pd.to_datetime(b["date"]).tz_convert("Asia/Taipei").tz_localize(None)
        if dt.strftime("%Y-%m-%d") != date_str:
            continue
        rows.append({
            "stock_id": symbol,
            "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(b["open"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "close": float(b["close"]),
            "volume": int(b["volume"]),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float32")
    df["volume"] = df["volume"].astype("int64")
    return df[["stock_id", "date", "open", "high", "low", "close", "volume"]]


def _atomic_save(df: pd.DataFrame, file_path: Path):
    os.makedirs(file_path.parent, exist_ok=True)
    if file_path.exists():
        old = pd.read_parquet(file_path)
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values(["date", "stock_id"], inplace=True)
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float32")
    df["volume"] = df["volume"].astype("int64")
    tmp = str(file_path) + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, file_path)


class M1RestPoller:
    """
    每分鐘透過 Fugle REST API 抓取分K，觸發 on_minute callback。
    介面與 M1Collector（WebSocket 版）相同，可直接替換。
    """

    def __init__(self, on_minute=None, stocks=None):
        self._on_minute = on_minute
        self._stocks = stocks
        self._stop = False
        self._client = None
        self._rate_limited_until = 0.0
        self._rate_limit_lock = threading.Lock()
        self._historical_backfilled_dates = set()

    def _get_stocks(self) -> list:
        if callable(self._stocks):
            return self._stocks()
        if self._stocks:
            return list(self._stocks)
        return fugle_stocks()

    def _is_rate_limited(self) -> bool:
        return time.time() < self._rate_limited_until

    def _mark_rate_limited(self, symbol: str, error: Exception):
        with self._rate_limit_lock:
            until = time.time() + _RATE_LIMIT_BACKOFF
            if until > self._rate_limited_until:
                self._rate_limited_until = until
                retry_at = datetime.fromtimestamp(until, tz=_TW).strftime("%H:%M:%S")
                print(f"  REST 觸發 rate limit（{symbol}），暫停到 {retry_at}: {error}")

    def _fetch_one(self, symbol: str, date_str: str) -> pd.DataFrame:
        if self._is_rate_limited():
            return pd.DataFrame()
        try:
            r = self._client.stock.intraday.candles(symbol=symbol, timeframe=1)
            bars = r.get("data", [])
            return _parse_rest_bars(symbol, bars, date_str)
        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                self._mark_rate_limited(symbol, e)
                return pd.DataFrame()
            print(f"  REST 取資料失敗 {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_historical_one(self, symbol: str, date_str: str, api_key: str) -> pd.DataFrame:
        params = {
            "timeframe": "1",
            "fields": "open,high,low,close,volume",
            "sort": "asc",
            "adjusted": "true",
        }
        headers = {"X-API-KEY": api_key}
        url = f"{_BASE_URL}/historical/candles/{symbol}"

        for attempt in range(_BACKFILL_MAX_RETRIES + 1):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=15)
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After", _RATE_LIMIT_BACKOFF))
                    if attempt >= _BACKFILL_MAX_RETRIES:
                        print(f"  historical backfill rate limit {symbol}，略過")
                        return pd.DataFrame()
                    print(f"  historical backfill rate limit {symbol}，等待 {retry_after:.0f}s 後重試")
                    time.sleep(retry_after + 1)
                    continue
                if r.status_code == 404:
                    return pd.DataFrame()
                r.raise_for_status()
                bars = r.json().get("data", [])
                return _parse_rest_bars(symbol, bars, date_str)
            except Exception as e:
                if attempt >= _BACKFILL_MAX_RETRIES:
                    print(f"  historical backfill 失敗 {symbol}: {e}")
                    return pd.DataFrame()
                time.sleep(_BACKFILL_INTERVAL)

        return pd.DataFrame()

    def _historical_backfill_once(self, stocks: list, date_str: str, api_key: str):
        if date_str in self._historical_backfilled_dates:
            print(f"[{date_str}] historical backfill 已執行過，略過")
            return
        self._historical_backfilled_dates.add(date_str)

        if not stocks:
            print(f"[{date_str}] historical backfill 無股票清單")
            return

        print(f"[{date_str}] 收盤後 historical backfill 開始，共 {len(stocks)} 支")
        _BATCH = 100  # 每 100 支存一次，避免全量 concat 超過記憶體上限
        frames = []
        saved = 0
        t0 = time.time()
        for idx, symbol in enumerate(stocks, 1):
            if self._stop:
                break
            df = self._fetch_historical_one(symbol, date_str, api_key)
            if not df.empty:
                frames.append(df)
            if idx % 20 == 0 or idx == len(stocks):
                print(f"  historical backfill 進度 {idx}/{len(stocks)}，待存 {len(frames)} 支")
            if len(frames) >= _BATCH or (idx == len(stocks) and frames):
                _atomic_save(pd.concat(frames, ignore_index=True), _live_path(date_str))
                saved += len(frames)
                frames.clear()
            if idx < len(stocks):
                time.sleep(_BACKFILL_INTERVAL)

        elapsed = time.time() - t0
        if saved == 0 and not frames:
            print(f"[{date_str}] historical backfill 沒有取得今日資料")
            return
        print(f"[{date_str}] historical backfill 完成，{saved} 支（{elapsed:.1f}s）")

    def _fetch_all(self, stocks: list, date_str: str) -> pd.DataFrame:
        if not stocks:
            return pd.DataFrame()

        frames = []
        _throttle = threading.Semaphore(_MAX_WORKERS)
        _last_req = [0.0]
        _lock = threading.Lock()
        _rate_limited = threading.Event()

        def _throttled_fetch(symbol):
            if _rate_limited.is_set() or self._is_rate_limited():
                return pd.DataFrame()
            with _throttle:
                if _rate_limited.is_set() or self._is_rate_limited():
                    return pd.DataFrame()
                with _lock:
                    elapsed = time.time() - _last_req[0]
                    wait = _REQ_INTERVAL - elapsed
                    if wait > 0:
                        time.sleep(wait)
                    _last_req[0] = time.time()
                df = self._fetch_one(symbol, date_str)
                if self._is_rate_limited():
                    _rate_limited.set()
                return df

        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(stocks))) as ex:
            futs = {ex.submit(_throttled_fetch, s): s for s in stocks}
            for fut in as_completed(futs):
                df = fut.result()
                if not df.empty:
                    frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def start(self):
        api_key = os.environ.get("FUGLE", "")
        if not api_key:
            raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")

        os.makedirs(_LIVE_DIR, exist_ok=True)
        self._client = RestClient(api_key=api_key)
        self._stop = False

        print(
            "REST Poller 啟動，每分鐘 poll 一次..."
            f"（有效分K {_POLL_START_HOUR:02d}:{_POLL_START_MIN:02d}"
            f"~{_POLL_END_HOUR:02d}:{_POLL_END_MIN:02d}）"
        )

        while not self._stop:
            now = datetime.now(_TW)
            current_closed_minute = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
            if _after_poll_end(current_closed_minute):
                date_str = current_closed_minute.strftime("%Y-%m-%d")
                # 只在收盤後 3 小時內補資料，之後直接 sleep 到明天
                if now.hour < _POLL_END_HOUR + 3:
                    stocks = self._get_stocks()
                    self._historical_backfill_once(stocks, date_str, api_key)
                else:
                    print(f"[{now.strftime('%H:%M')}] 收盤超過 3 小時，略過 backfill")
                wake_at = _next_poll_window_start(now)
                _sleep_until(
                    wake_at,
                    lambda: self._stop,
                    f"[{now.strftime('%H:%M:%S')}] 收盤後 live poll 停止",
                )
                continue

            # 等到下一分鐘的 :05 秒，確保分K已收盤
            next_poll = now.replace(second=5, microsecond=0) + timedelta(minutes=1)
            wait = (next_poll - datetime.now(_TW)).total_seconds()
            if wait > 0:
                time.sleep(wait)

            if self._stop:
                break

            now = datetime.now(_TW)
            # 剛收盤的分鐘（:00 的那根）
            closed_minute = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
            minute_str = closed_minute.strftime("%Y-%m-%d %H:%M:%S")
            date_str = closed_minute.strftime("%Y-%m-%d")

            if not _in_poll_window(closed_minute):
                if _before_poll_start(closed_minute):
                    wake_at = _next_poll_window_start(now)
                    _sleep_until(
                        wake_at,
                        lambda: self._stop,
                        f"[{now.strftime('%H:%M:%S')}] 盤前，不呼叫 Fugle REST",
                    )
                    continue
                if _after_poll_end(closed_minute):
                    stocks = self._get_stocks()
                    self._historical_backfill_once(stocks, date_str, api_key)
                    wake_at = _next_poll_window_start(now)
                    _sleep_until(
                        wake_at,
                        lambda: self._stop,
                        f"[{now.strftime('%H:%M:%S')}] 收盤後 live poll 停止",
                    )
                    continue

            if self._is_rate_limited():
                retry_at = datetime.fromtimestamp(self._rate_limited_until, tz=_TW)
                wait = (retry_at - datetime.now(_TW)).total_seconds()
                print(f"[{now.strftime('%H:%M:%S')}] Fugle rate limit backoff 到 {retry_at.strftime('%H:%M:%S')}")
                if wait > 0:
                    time.sleep(wait)
                continue

            stocks = self._get_stocks()
            t0 = time.time()
            df = self._fetch_all(stocks, date_str)
            elapsed = time.time() - t0

            if df.empty:
                print(f"[{now.strftime('%H:%M:%S')}] {minute_str} 無資料")
                continue

            _atomic_save(df, _live_path(date_str))
            minute_df = df[df["date"] == minute_str]
            ts = now.strftime("%H:%M:%S")
            print(f"[{ts}] {minute_str} 存檔，{len(df['stock_id'].unique())} 支股票"
                  f"（fetch {elapsed:.2f}s，該分鐘 {len(minute_df)} 支）")

            if self._on_minute and not minute_df.empty:
                try:
                    self._on_minute(minute_str, minute_df.copy())
                except Exception as e:
                    print(f"on_minute 錯誤: {e}")

    def stop(self):
        self._stop = True
