"""
Fugle WebSocket 即時分K收集器
- 今日盤中資料存到 db/m1_live/{date}.parquet（不影響歷史 db/m1/）
- 每分鐘收盤後觸發 on_minute(minute_str, df) callback 供模型推論
"""
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
from dotenv import load_dotenv
from fugle_marketdata import WebSocketClient

from tay_trade.fugle_tickers import fugle_stocks

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_TW = timezone(timedelta(hours=8))
_LIVE_DIR = _ROOT / "db/m1_live"


def _live_path(date_str: str) -> Path:
    return _LIVE_DIR / f"{date_str}.parquet"


def _parse_bars(symbol: str, raw_bars: list) -> pd.DataFrame:
    if not raw_bars:
        return pd.DataFrame()
    df = pd.DataFrame(raw_bars)
    df["stock_id"] = symbol
    date = pd.to_datetime(df["date"])
    if date.dt.tz is not None:
        date = date.dt.tz_convert(_TW).dt.tz_localize(None)
    df["date"] = date.dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["stock_id", "date", "open", "high", "low", "close", "volume"]
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float32")
    df["volume"] = df["volume"].astype("int64")
    return df[cols]


def _atomic_save(df: pd.DataFrame, file_path: str):
    """讀取現有資料 merge 後原子寫入。
    volume 是每分鐘的累計值（per-trade 推播），同(stock_id, date)保留最後一筆即為該分鐘完整 bar。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        old = pd.read_parquet(file_path)
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values(["date", "stock_id"], inplace=True)
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float32")
    df["volume"] = df["volume"].astype("int64")
    tmp = f"{file_path}.tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, file_path)


class M1Collector:
    """
    使用方式：
        def on_minute(minute_str, df):
            # minute_str: '2026-06-30 09:32:00'
            # df: 該分鐘所有股票的分K（可直接餵給模型）
            pass

        collector = M1Collector(on_minute=on_minute)
        collector.start()   # 阻塞直到 KeyboardInterrupt 或 collector.stop()
    """

    def __init__(self, on_minute=None, stocks=None):
        self._on_minute = on_minute
        self._stocks = stocks   # list | callable() -> list | None（None = 全部）
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}   # {minute_str: {stock_id: row_dict}}
        self._client = None
        self._stop = False
        self._last_error = ""

    # ─────────────────── 訊息處理 ───────────────────

    def _handle(self, raw):
        try:
            msg = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return

        event = msg.get("event")
        data = msg.get("data", {})

        if event == "snapshot":
            # 訂閱後立刻收到今日所有分K（09:00 ~ 現在）
            symbol = data.get("symbol", "")
            df = _parse_bars(symbol, data.get("data", []))
            if df.empty:
                return
            date_str = df["date"].str[:10].iloc[0]
            with self._lock:
                _atomic_save(df, _live_path(date_str))

        elif event == "data":
            # 每分K收盤後推一根
            symbol = data.get("symbol", "")
            df = _parse_bars(symbol, [data])
            if df.empty:
                return
            minute_str = df.iloc[0]["date"]
            row = df.iloc[0].to_dict()
            with self._lock:
                # 直接覆蓋：volume 是累計值，保留最新一筆即正確
                self._pending.setdefault(minute_str, {})[symbol] = row
                self._flush_elapsed(minute_str)

    def _flush_elapsed(self, current_minute: str):
        """把比 current_minute 舊的分鐘存檔並觸發 callback"""
        old_minutes = sorted(m for m in self._pending if m < current_minute)
        for minute in old_minutes:
            rows = list(self._pending.pop(minute).values())
            df = pd.DataFrame(rows)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype("float32")
            df["volume"] = df["volume"].astype("int64")
            date_str = minute[:10]
            _atomic_save(df, _live_path(date_str))
            ts = datetime.now(_TW).strftime("%H:%M:%S")
            print(f"[{ts}] {minute} 存檔，{len(df)} 支股票")
            if self._on_minute:
                try:
                    self._on_minute(minute, df.copy())
                except Exception as e:
                    print(f"on_minute 錯誤: {e}")

    # ─────────────────── 啟動 / 停止 ───────────────────

    def _connect_once(self, api_key: str, stocks: list):
        self._client = WebSocketClient(api_key=api_key)
        ws = self._client.stock
        authenticated = threading.Event()
        subs_reply = threading.Event()
        existing_ids: list[str] = []

        self._last_error = ""

        def _on_message(raw):
            msg = json.loads(raw) if isinstance(raw, str) else raw
            event = msg.get("event")
            if event == "authenticated":
                authenticated.set()
            elif event == "subscriptions":
                # server 回報目前所有訂閱，收集 id 準備清除
                for sub in msg.get("data", []):
                    sid = sub.get("id")
                    if sid:
                        existing_ids.append(sid)
                subs_reply.set()
            self._handle(raw)

        def _on_error(e):
            self._last_error = str(e)
            print(f"WS 錯誤: {e}")

        ws.on("message", _on_message)
        ws.on("error", _on_error)

        def _subscribe_all():
            if not authenticated.wait(timeout=10):
                print("認證逾時，跳過訂閱")
                return

            # Maximum connections 斷線 → 查詢並取消 server 端殘留訂閱
            if "Maximum number of connections" in self._last_error:
                print("查詢舊訂閱...")
                ws.subscriptions()
                subs_reply.wait(timeout=5)
                if existing_ids:
                    print(f"取消 {len(existing_ids)} 個舊訂閱...")
                    for sub_id in existing_ids:
                        try:
                            ws.unsubscribe({"id": sub_id})
                        except Exception:
                            pass
                    time.sleep(2)
                    print("舊訂閱已清除，重新訂閱...")

            print("開始訂閱...")
            for i, symbol in enumerate(stocks):
                try:
                    ws.subscribe({"channel": "candles", "symbol": symbol})
                except Exception as e:
                    print(f"訂閱中斷（已送出 {i}/{len(stocks)}）: {e}")
                    return
                if (i + 1) % 500 == 0:
                    print(f"  已送出 {i+1}/{len(stocks)} 筆訂閱...")
            print(f"訂閱送出完畢（{len(stocks)} 支）")

        threading.Thread(target=_subscribe_all, daemon=True).start()
        ws.connect()  # 阻塞，斷線後返回

    def start(self):
        api_key = os.environ.get("FUGLE", "")
        if not api_key:
            raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")

        os.makedirs(_LIVE_DIR, exist_ok=True)
        self._stop = False
        retry = 0

        while not self._stop:
            # 每次重連都重新取得標的清單（daily refresh 可能已更新）
            if callable(self._stocks):
                stocks = self._stocks()
            elif self._stocks:
                stocks = self._stocks
            else:
                stocks = fugle_stocks()
            print(f"共 {len(stocks)} 支股票，開始連線...")
            try:
                if retry > 0:
                    # Fugle server 釋放舊連線需要 60-120s，一律等 120s 起跳
                    wait = min(120 * retry, 600)
                    print(f"第 {retry} 次重連，等待 {wait} 秒...")
                    time.sleep(wait)
                self._connect_once(api_key, stocks)
            except Exception as e:
                if self._stop:
                    break
                print(f"連線中斷: {e}")
            if not self._stop:
                retry += 1

    def stop(self):
        self._stop = True
        if self._client:
            self._client.stock.disconnect()


# ─────────────────── 直接執行 ───────────────────

def _default_on_minute(minute_str: str, df: pd.DataFrame):
    print(f"  → 推論觸發: {minute_str}，{len(df)} 支股票")


if __name__ == "__main__":
    collector = M1Collector(on_minute=_default_on_minute)
    try:
        collector.start()
    except KeyboardInterrupt:
        collector.stop()
        print("已停止")
