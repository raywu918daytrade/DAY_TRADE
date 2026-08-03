"""
富邦即時成交明細（tick）收集器：訂閱 trades channel（跟 fubon/marketdata_ws.py
的 candles channel 不同，是逐筆成交，不是分K），把收到的 tick 寫進
db/tick_live/{日期}.parquet，schema 跟 db/tick（見 finmind/tick_api.py）
完全一致，供策略即時推論用（見 strategy/breakout_retest_ml/predict.py 的
ticks_by_stock 參數）。

跟 FubonM1Collector 刻意分開的原因：
1. 只做「收→buffer→定期存檔」，不補歷史缺口（_backfill_intraday）、不驅動
   on_minute() 的每分鐘觸發時機——目前只需要「能收 tick」，不需要這些。
2. 策略還在開發中，tick 解析可能有 bug；獨立成一支檔案，出問題不會拖累
   已經上線在跑其他策略的 candles collector。

不自己 login()：重用 FubonM1Collector 已經拿到的 realtime_token()（見
FubonM1Collector.__init__ 的 on_token_ready callback），避免重複呼叫富邦
登入——這是已經修過的真實 bug（_startup()/_daily_refresh() 同時登入），
tick collector 不該再犯一次。

WS trades channel 的原始 payload 欄位還沒逐欄核對過（跟 _parse_candle() 當初
一樣的狀況，見該函式的說明），_parse_trade() 先用 REST intraday/trades 的
欄位命名法（price/size/bid/ask/time，見 fubon/tick_api.py::fetch_trades_today()）
去試著解析，並且無論解析成功與否都會印出前 _DEBUG_LOG_COUNT 筆原始 payload，
方便對照調整。

使用方式：
    python -m fubon.tick_ws
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
from dotenv import load_dotenv

from fubon import fubon_api as trade_api
from fubon.subscribe_list import load_subscribe_batches

try:
    from api import append_system_log as _log_sys
except Exception:
    def _log_sys(msg, level="info"): pass  # type: ignore

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))
_LIVE_DIR = _ROOT / "db/tick_live"
# tick 量遠大於分K，每次 flush 都是整個檔案讀出來 concat 再整個寫回去，
# 間隔太短會隨著一天累積量增加而越寫越久，所以預設比 candles 的 5 秒長。
_FLUSH_INTERVAL = float(os.environ.get("FUBON_TICK_WS_FLUSH_INTERVAL", "10"))
_DEBUG_LOG_COUNT = int(os.environ.get("FUBON_TICK_DEBUG_LOG_COUNT", "3"))


def _live_path(date_str: str) -> Path:
    return _LIVE_DIR / f"{date_str}.parquet"


def _infer_tick_type(price, bid, ask) -> int:
    """跟 fubon/tick_api.py::_infer_tick_type() 同一套規則（price 相對
    bid/ask 的關係推斷買賣方主動性，1=買方主動/外盤、2=賣方主動/內盤、
    0=無法判定）——如果 WS payload 直接有方向欄位（例如 "side"/"tradeType"），
    以後在這裡改成直接讀那個欄位，不用推斷。"""
    if bid is None or ask is None or price is None:
        return 0
    if price >= ask:
        return 1
    if price <= bid:
        return 2
    return 0


_debug_logged = 0


def _parse_trade(raw: bytes | str) -> dict | None:
    """解析 trades channel 推送訊息，不是成交資料（例如 authenticated/subscribed
    ack）回傳 None。欄位猜測依據 REST intraday/trades 的命名法，第一次實際
    收到訊息時務必對照 log 印出來的原始 payload 確認，不對再調整這裡。"""
    global _debug_logged
    msg = orjson.loads(raw)
    if _debug_logged < _DEBUG_LOG_COUNT:
        print(f"[tick 原始payload] {msg}", flush=True)
        _debug_logged += 1
    if not isinstance(msg, dict):
        return None
    data = msg.get("data")
    if not isinstance(data, dict) or "symbol" not in data:
        return None
    price = data.get("price")
    size = data.get("size")
    if price is None or size is None:
        return None

    if "date" in data:
        dt = pd.to_datetime(data["date"])
        if dt.tzinfo is not None:
            dt = dt.tz_convert("Asia/Taipei").tz_localize(None)
    elif "time" in data:
        # REST intraday/trades 的 time 是微秒(us)精度的 unix epoch
        dt = pd.to_datetime(int(data["time"]), unit="us", utc=True).tz_convert("Asia/Taipei").tz_localize(None)
    else:
        return None

    tick_type = data.get("tick_type")
    if tick_type is None:
        tick_type = _infer_tick_type(price, data.get("bid"), data.get("ask"))

    return {
        "stock_id": str(data["symbol"]),
        "date": dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "deal_price": float(price),
        "volume": int(size),
        "tick_type": int(tick_type),
    }


def _atomic_save_tick(df: pd.DataFrame, file_path: Path):
    """讀舊檔 merge 新資料、dedup（stock_id, date）keep last，再原子性寫回去
    ——跟 finmind/tick_api.py::save_tick() 同一套 dedupe 慣例，schema 跟
    db/tick 一致，方便之後如果要把 tick_live 併回 db/tick 直接相容。呼叫端
    自己要加鎖，這裡不處理併發（同 data/m1_utils.py::_atomic_save() 的說明）。
    """
    os.makedirs(file_path.parent, exist_ok=True)
    if file_path.exists():
        old = pd.read_parquet(file_path)
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values(["date", "stock_id"], inplace=True)
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    df["deal_price"] = df["deal_price"].astype("float32")
    df["volume"] = df["volume"].astype("int64")
    df["tick_type"] = df["tick_type"].astype("int8")
    tmp = str(file_path) + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, file_path)


class FubonTickCollector:
    """管理最多 5 條（扣掉 candles 用掉的）trades WebSocket 連線，收到的
    tick 先 buffer，定期存檔。只負責收集，不驅動策略推論時機。"""

    def __init__(self, token: str):
        self._token = token
        self._clients: list = []
        # key 用 (stock_id, date) 而不是單純累加 list，理由跟 FubonM1Collector
        # 一樣：同一支股票同一個時間戳記重複推送時（WS 斷線重連常見）用最新
        # 那筆覆蓋，避免無限累積重複列。date 是微秒精度字串，正常情況下同一支
        # 股票不會有兩筆完全相同時間戳記的成交，dedupe key 足夠。
        self._buffer: dict[tuple[str, str], dict] = {}
        self._buffer_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._stop = False

    def start(self):
        batches = load_subscribe_batches()
        if not batches:
            raise RuntimeError("訂閱清單是空的，請先跑 python -m fubon.subscribe_list")

        total = 0
        for i, batch in enumerate(batches, 1):
            stock = trade_api.open_candles_connection(self._token)  # channel-agnostic，見 subscribe_trades() 的說明
            stock.on("message", self._make_handler(i))
            stock.on("disconnect", lambda code, msg, i=i: print(f"[tick連線{i}] disconnected: {code} {msg}", flush=True))
            stock.on("error", lambda e, i=i: print(f"[tick連線{i}] error: {e}", flush=True))
            stock.connect()
            print(f"[tick連線{i}] 已連線，訂閱 {len(batch)} 支...", flush=True)

            for sid in batch:
                try:
                    trade_api.subscribe_trades(stock, sid)
                    total += 1
                except Exception as e:
                    print(f"[tick連線{i}] 訂閱 {sid} 失敗: {e}", flush=True)

            self._clients.append(stock)
            print(f"[tick連線{i}] 訂閱完成：{len(batch)} 支", flush=True)

        print(f"共訂閱 {total} 支 tick（{len(self._clients)} 條連線），開始接收成交明細...", flush=True)
        _log_sys(f"富邦 tick WebSocket 訂閱完成：{total} 支（{len(self._clients)} 條連線）")

        self._flush_loop()  # 主執行緒 block 在這裡，直到 stop() 被呼叫

    def stop(self):
        self._stop = True
        for stock in self._clients:
            try:
                stock.disconnect()
            except Exception:
                pass

    def _make_handler(self, conn_id: int):
        def handler(raw):
            try:
                row = _parse_trade(raw)
            except Exception as e:
                print(f"[tick連線{conn_id}] 訊息解析失敗: {e}｜{raw!r:.200}", flush=True)
                return
            if row is None:
                return
            with self._buffer_lock:
                self._buffer[(row["stock_id"], row["date"])] = row
        return handler

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
        with self._save_lock:
            for date_str, g in df.groupby(df["date"].str[:10]):
                _atomic_save_tick(g.copy(), _live_path(date_str))
        print(f"[tick flush] 存檔 {len(df)} 筆（{df['stock_id'].nunique()} 支）", flush=True)


if __name__ == "__main__":
    print("富邦登入（tick collector 獨立測試用，正式運作時會重用 candles collector 的 token）...", flush=True)
    sdk, accounts = trade_api.login()
    print(f"登入成功：{[a.name for a in accounts]}", flush=True)
    trade_api.init_market_data(sdk)
    token = trade_api.realtime_token(sdk)

    collector = FubonTickCollector(token=token)
    try:
        collector.start()
    except KeyboardInterrupt:
        print("停止中...", flush=True)
    finally:
        collector.stop()
        trade_api.logout(sdk)
