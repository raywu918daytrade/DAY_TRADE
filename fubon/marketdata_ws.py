"""
富邦即時分K收集器：開最多 5 條 WebSocket 連線，訂閱 candles channel，
把收到的 1分K 寫進 db/m1_live/{日期}.parquet（欄位/存檔邏輯對齊
data/m1_rest.py，方便共用 data/query.py 的 load_m1_live()）。

股票清單來自 fubon/subscribe_list.py 存好的
db/fubon_subscribe/subscribe_list.parquet（開盤前先跑一次
`python -m fubon.subscribe_list` 產生），這裡只負責讀檔訂閱，不重算排序。

⚠️ 帳號目前還在等富邦開通 API 使用權限，candles 訊息的實際欄位／推送頻率
（每筆成交都推、還是分鐘收盤才推一次）尚未用真實連線驗證過。等權限開通、
第一次連線務必先看 log 確認 payload 長相，必要時調整 _parse_candle()。

使用方式：
    python -m fubon.marketdata_ws
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import orjson
import pandas as pd
from dotenv import load_dotenv

from fubon import trade_api
from fubon.subscribe_list import load_subscribe_batches

try:
    from api import append_system_log as _log_sys
except Exception:
    def _log_sys(msg, level="info"): pass  # type: ignore

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_TW = timezone(timedelta(hours=8))
_LIVE_DIR = _ROOT / "db/m1_live"
_FLUSH_INTERVAL = float(os.environ.get("FUBON_WS_FLUSH_INTERVAL", "5"))


def _live_path(date_str: str) -> Path:
    return _LIVE_DIR / f"{date_str}.parquet"


def _atomic_save(df: pd.DataFrame, file_path: Path):
    """存檔邏輯對齊 data/m1_rest.py：merge 舊檔 + dedup（stock_id, date）keep last。"""
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


def _parse_candle(raw: bytes | str) -> dict | None:
    """解析 candles channel 推送訊息，不是分K資料（如 authenticated/subscribed ack）回傳 None。

    date 欄位跟 Fugle REST 一樣是帶時區字串（例：'2026-07-02T09:00:00.000+08:00'），
    轉成台北 naive 字串存檔，不能直接用 UTC 解讀（見 CLAUDE.md）。
    """
    msg = orjson.loads(raw)
    if not isinstance(msg, dict):
        return None
    data = msg.get("data")
    if not isinstance(data, dict) or "open" not in data or "symbol" not in data:
        return None

    dt = pd.to_datetime(data["date"])
    if dt.tzinfo is not None:
        dt = dt.tz_convert("Asia/Taipei").tz_localize(None)
    return {
        "stock_id": str(data["symbol"]),
        "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": float(data["open"]),
        "high": float(data["high"]),
        "low": float(data["low"]),
        "close": float(data["close"]),
        "volume": int(data["volume"]),
    }


class FubonM1Collector:
    """管理最多 5 條 candles WebSocket 連線，收到的分K先 buffer，定期存檔。"""

    def __init__(self, on_minute=None):
        self._on_minute = on_minute
        self._sdk = None
        self._clients: list = []
        self._buffer: dict[tuple[str, str], dict] = {}  # (stock_id, minute_str) -> row
        self._buffer_lock = threading.Lock()
        self._flushed_minutes: dict[str, set] = {}  # date_str -> 已呼叫過 on_minute 的分鐘
        self._stop = False

    # ── 連線 ──────────────────────────────────────────────────────────────

    def start(self):
        print("富邦登入...", flush=True)
        self._sdk, accounts = trade_api.login()
        print(f"登入成功：{[a.name for a in accounts]}", flush=True)

        trade_api.init_market_data(self._sdk)  # candles channel 只支援 Normal mode（預設值）
        token = trade_api.realtime_token(self._sdk)

        batches = load_subscribe_batches()
        if not batches:
            raise RuntimeError("訂閱清單是空的，請先跑 python -m fubon.subscribe_list")

        total = 0
        for i, batch in enumerate(batches, 1):
            stock = trade_api.open_candles_connection(token)
            stock.on("message", self._make_handler(i))
            stock.on("disconnect", lambda code, msg, i=i: print(f"[連線{i}] disconnected: {code} {msg}", flush=True))
            stock.on("error", lambda e, i=i: print(f"[連線{i}] error: {e}", flush=True))
            stock.connect()  # 內部會 block 到 auth 完成（或失敗直接 raise）
            print(f"[連線{i}] 已連線，訂閱 {len(batch)} 支...", flush=True)

            for sid in batch:
                try:
                    trade_api.subscribe_candles(stock, sid)
                    total += 1
                except Exception as e:
                    print(f"[連線{i}] 訂閱 {sid} 失敗: {e}", flush=True)

            self._clients.append(stock)
            print(f"[連線{i}] 訂閱完成：{len(batch)} 支", flush=True)

        print(f"共訂閱 {total} 支（{len(self._clients)} 條連線），開始接收分K...", flush=True)
        _log_sys(f"富邦 WebSocket 訂閱完成：{total} 支（{len(self._clients)} 條連線）")

        self._flush_loop()

    def stop(self):
        self._stop = True
        for stock in self._clients:
            try:
                stock.disconnect()
            except Exception:
                pass
        if self._sdk is not None:
            trade_api.logout(self._sdk)

    # ── 訊息處理 ──────────────────────────────────────────────────────────

    def _make_handler(self, conn_id: int):
        def handler(raw):
            try:
                row = _parse_candle(raw)
            except Exception as e:
                print(f"[連線{conn_id}] 訊息解析失敗: {e}｜{raw!r:.200}", flush=True)
                return
            if row is None:
                return
            with self._buffer_lock:
                self._buffer[(row["stock_id"], row["date"])] = row
        return handler

    # ── 定期存檔 ──────────────────────────────────────────────────────────

    def _flush_loop(self):
        while not self._stop:
            time.sleep(_FLUSH_INTERVAL)
            self._flush()

    def _flush(self):
        with self._buffer_lock:
            if not self._buffer:
                return
            rows = list(self._buffer.values())

        df = pd.DataFrame(rows)
        for date_str, g in df.groupby(df["date"].str[:10]):
            _atomic_save(g.copy(), _live_path(date_str))

        if self._on_minute:
            now_minute_str = datetime.now(_TW).replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            for minute_str, g in df.groupby("date"):
                if minute_str >= now_minute_str:
                    continue  # 這分鐘可能還在形成中，先不當已收盤處理
                done = self._flushed_minutes.setdefault(minute_str[:10], set())
                if minute_str in done:
                    continue
                done.add(minute_str)
                try:
                    self._on_minute(minute_str, g.copy())
                except Exception as e:
                    print(f"on_minute 錯誤: {e}", flush=True)

        print(f"[flush] 存檔 {len(df)} 筆（{df['stock_id'].nunique()} 支）", flush=True)


if __name__ == "__main__":
    collector = FubonM1Collector()
    try:
        collector.start()
    except KeyboardInterrupt:
        print("停止中...", flush=True)
    finally:
        collector.stop()
