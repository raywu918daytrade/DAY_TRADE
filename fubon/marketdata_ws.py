"""
富邦即時分K收集器：開最多 5 條 WebSocket 連線，訂閱 candles channel，
把收到的 1分K 寫進 db/m1_live/{日期}.parquet（欄位/存檔邏輯對齊
data/m1_rest.py，方便共用 data/query.py 的 load_m1_live()）。

on_minute 觸發機制：存檔（_flush_loop，每 FUBON_WS_FLUSH_INTERVAL 秒）跟
觸發推論（_minute_tick_loop，每分鐘固定一次，用真實時鐘驅動）是兩條分開的
迴圈——不管這分鐘 WebSocket 有沒有推新訊息，_minute_tick_loop 都保證每分鐘
呼叫一次 on_minute（讀 db/m1_live/ 當下最新資料），跟 M1RestPoller 的保底
行為一致，確保收盤後的 SL/TP reconcile 監控不會因為行情安靜而跳過整分鐘。

股票清單來自 fubon/subscribe_list.py 存好的
db/fubon_subscribe/subscribe_list.parquet（開盤前先跑一次
`python -m fubon.subscribe_list` 產生），這裡只負責讀檔訂閱，不重算排序。

補資料（backfill）：WebSocket 只會推「連線之後」的分K，如果不是一開盤就連線
（例如中途 9:05 才啟動），連線前那段會整段缺資料。start() 會在開 WebSocket
連線之前，先用富邦 REST intraday/candles/{symbol}（rate limit 300次/分鐘，
這裡節流得保守一點）把當天到目前為止的分K全部補進 db/m1_live/，補完才開始
連線收即時資料，確保「WebSocket 連上後才推論」時當天資料是完整的。

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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

import orjson
import pandas as pd
from dotenv import load_dotenv

from data.m1_rest import _atomic_save, _parse_rest_bars
from data.query import load_m1_live
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
# 富邦 intraday/candles rate limit 300次/分鐘（官方文件），節流到 ~240次/分鐘留緩衝。
# 這個間隔控制的是「發出下一個 request」的頻率，不是單一 worker 的間隔——
# 多執行緒下大家共用同一個節流時鐘，各自的網路等待時間可以互相重疊
# （寫法對齊 data/m1_rest.py::_fetch_all 的 _throttled_fetch）。
_BACKFILL_INTERVAL = float(os.environ.get("FUBON_REST_INTERVAL", "0.25"))
_BACKFILL_WORKERS = int(os.environ.get("FUBON_BACKFILL_WORKERS", "10"))


def _live_path(date_str: str) -> Path:
    return _LIVE_DIR / f"{date_str}.parquet"


def _backfill_intraday(sdk, symbols: list[str], date_str: str):
    """連線前先用 REST 補齊當天已經產生、但 WebSocket 連上前收不到的分K。
    多執行緒併發抓取，全域節流頻率控制在 ~1/_BACKFILL_INTERVAL 次/秒，
    避免逐支序列等待網路延遲拖慢整體時間。
    """
    print(f"[backfill] 開始補 {len(symbols)} 支 {date_str} 的分K（連線前資料，"
          f"{_BACKFILL_WORKERS} 併發）...", flush=True)
    t0 = time.time()
    frames: list[pd.DataFrame] = []
    frames_lock = threading.Lock()
    throttle = threading.Semaphore(_BACKFILL_WORKERS)
    rate_lock = threading.Lock()
    last_req = [0.0]
    done = [0]

    def fetch_one(sid: str):
        with throttle:
            with rate_lock:
                wait = _BACKFILL_INTERVAL - (time.time() - last_req[0])
                if wait > 0:
                    time.sleep(wait)
                last_req[0] = time.time()
            try:
                bars = trade_api.intraday_candles(sdk, sid)
                df = _parse_rest_bars(sid, bars, date_str)
                if not df.empty:
                    with frames_lock:
                        frames.append(df)
            except Exception as e:
                print(f"[backfill] {sid} 失敗: {e}", flush=True)
        with frames_lock:
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(symbols):
                print(f"[backfill] 進度 {done[0]}/{len(symbols)}", flush=True)

    with ThreadPoolExecutor(max_workers=_BACKFILL_WORKERS) as ex:
        list(ex.map(fetch_one, symbols))

    if frames:
        _atomic_save(pd.concat(frames, ignore_index=True), _live_path(date_str))
    elapsed = time.time() - t0
    msg = f"[backfill] 完成，{len(frames)}/{len(symbols)} 支有資料（{elapsed:.1f}s）"
    print(msg, flush=True)
    _log_sys(f"富邦 backfill 完成 {date_str}：{len(frames)}/{len(symbols)} 支（{elapsed:.1f}s）")


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
        self._stop = False

    # ── 連線 ──────────────────────────────────────────────────────────────

    def start(self):
        print("富邦登入...", flush=True)
        self._sdk, accounts = trade_api.login()
        print(f"登入成功：{[a.name for a in accounts]}", flush=True)

        trade_api.init_market_data(self._sdk)  # candles channel 只支援 Normal mode（預設值）

        batches = load_subscribe_batches()
        if not batches:
            raise RuntimeError("訂閱清單是空的，請先跑 python -m fubon.subscribe_list")

        date_str = datetime.now(_TW).strftime("%Y-%m-%d")
        all_symbols = [sid for batch in batches for sid in batch]
        _backfill_intraday(self._sdk, all_symbols, date_str)

        token = trade_api.realtime_token(self._sdk)

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

        threading.Thread(target=self._flush_loop, daemon=True).start()
        self._minute_tick_loop()  # 主執行緒 block 在這裡，跟 M1RestPoller.start() 同樣角色

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

    # ── 定期存檔（只負責把 buffer 寫進 db/m1_live/，不觸發 on_minute）──────

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
        print(f"[flush] 存檔 {len(df)} 筆（{df['stock_id'].nunique()} 支）", flush=True)

    # ── 每分鐘固定觸發一次 on_minute（保底機制）────────────────────────────
    #
    # 跟 data/m1_rest.py::M1RestPoller 對齊：不管這分鐘 WebSocket 有沒有推新
    # 訊息（連線斷線、行情安靜、還沒開盤都可能發生），每分鐘都要固定呼叫一次
    # on_minute，否則 on_minute() 裡收盤後的 SL/TP reconcile 監控會因為某幾
    # 分鐘沒有新訊息就整段被跳過。用真實時鐘（跟 REST 版本一樣卡在每分鐘 :05
    # 秒）驅動，不是靠 buffer 裡有沒有「新完成的一分鐘」來判斷。

    def _minute_tick_loop(self):
        if not self._on_minute:
            return
        while not self._stop:
            now = datetime.now(_TW)
            next_tick = now.replace(second=5, microsecond=0) + timedelta(minutes=1)
            wait = (next_tick - datetime.now(_TW)).total_seconds()
            if wait > 0:
                time.sleep(wait)
            if self._stop:
                break
            self._flush()  # 先把 buffer 存檔，確保等一下讀到的是最新資料
            self._tick_once()

    def _tick_once(self):
        closed_minute = datetime.now(_TW).replace(second=0, microsecond=0) - timedelta(minutes=1)
        minute_str = closed_minute.strftime("%Y-%m-%d %H:%M:%S")
        date_str = minute_str[:10]

        day_df = load_m1_live(date_str)
        minute_df = day_df[day_df["date"] == minute_str] if not day_df.empty else day_df

        try:
            self._on_minute(minute_str, minute_df)
        except Exception as e:
            print(f"on_minute 錯誤: {e}", flush=True)


if __name__ == "__main__":
    collector = FubonM1Collector()
    try:
        collector.start()
    except KeyboardInterrupt:
        print("停止中...", flush=True)
    finally:
        collector.stop()
