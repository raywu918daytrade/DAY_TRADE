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
from dotenv import load_dotenv
from fugle_marketdata import RestClient

from tay_trade.fugle_tickers import fugle_stocks

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_TW = timezone(timedelta(hours=8))
_LIVE_DIR = _ROOT / "db/m1_live"
_MAX_WORKERS = int(os.environ.get("REST_WORKERS", "10"))
_REQ_INTERVAL = float(os.environ.get("REST_INTERVAL", "0.15"))  # 每個 request 最小間隔（秒）


def _live_path(date_str: str) -> Path:
    return _LIVE_DIR / f"{date_str}.parquet"


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

    def _get_stocks(self) -> list:
        if callable(self._stocks):
            return self._stocks()
        if self._stocks:
            return list(self._stocks)
        return fugle_stocks()

    def _fetch_one(self, symbol: str, date_str: str) -> pd.DataFrame:
        try:
            r = self._client.stock.intraday.candles(symbol=symbol, timeframe=1)
            bars = r.get("data", [])
            return _parse_rest_bars(symbol, bars, date_str)
        except Exception as e:
            print(f"  REST 取資料失敗 {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_all(self, stocks: list, date_str: str) -> pd.DataFrame:
        frames = []
        _throttle = threading.Semaphore(_MAX_WORKERS)
        _last_req = [0.0]
        _lock = threading.Lock()

        def _throttled_fetch(symbol):
            with _throttle:
                with _lock:
                    elapsed = time.time() - _last_req[0]
                    wait = _REQ_INTERVAL - elapsed
                    if wait > 0:
                        time.sleep(wait)
                    _last_req[0] = time.time()
                return self._fetch_one(symbol, date_str)

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

        print("REST Poller 啟動，每分鐘 poll 一次...")

        while not self._stop:
            now = datetime.now(_TW)
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
