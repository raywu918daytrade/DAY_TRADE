"""
富邦即時分K收集器：開最多 5 條 WebSocket 連線，訂閱 candles channel，
把收到的 1分K 寫進 db/m1_live/{日期}.parquet（欄位/存檔邏輯共用
data/m1_utils.py，方便共用 data/query.py 的 load_m1_live()）。

on_minute 觸發機制：存檔（_flush_loop，每 FUBON_WS_FLUSH_INTERVAL 秒）跟
觸發推論（_minute_tick_loop，每分鐘固定一次，用真實時鐘驅動）是兩條分開的
迴圈——不管這分鐘 WebSocket 有沒有推新訊息，_minute_tick_loop 都保證每分鐘
呼叫一次 on_minute（讀 db/m1_live/ 當下最新資料），確保收盤後的 SL/TP
reconcile 監控不會因為行情安靜而跳過整分鐘（Fugle REST 輪詢器年代就是這樣
設計的，2026-07-13 移除那支檔案時保留了這個保底邏輯）。

股票清單來自 fubon/subscribe_list.py 驗證好的
db/tickers/tick_universe.parquet（daytrade_ok=True 的子集，2026-08-19
合併進候選母體檔案本身，不再另存 db/fubon_subscribe/subscribe_list.parquet，
見該檔案的說明——開盤前先跑一次 `python -m fubon.subscribe_list` 產生），
這裡只負責讀檔訂閱，不重算排序。

補資料（backfill）：WebSocket 只會推「連線之後」的分K，如果不是一開盤就連線
（例如盤中重啟），連線前那段會整段缺資料。start() 會**先**開好 WebSocket 連線
開始收即時資料，**再**用富邦 REST intraday/candles/{symbol}（rate limit
300次/分鐘，這裡節流得保守一點）在背景執行緒補「連線前」缺的分K，兩者不會
互相卡住——盤中重啟不會因為要等 backfill 跑完（880支約3~5分鐘）才開始收
新資料。backfill 補到的資料寫進跟 WebSocket 共用的同一個 buffer（由
_flush_loop 統一存檔），不會各自獨立呼叫存檔函式，避免兩個執行緒同時
讀寫同一個檔案的 race condition。

富邦 API 使用權限已開通（2026-07-14）。/health 已確認 collector 能實際連線、
持續收到即時資料（coverage 逐步從 0 補到 1000 支）。但 candles 訊息的實際
欄位／推送頻率（每筆成交都推、還是分鐘收盤才推一次）目前是靠下游 cumsum
算出來的 cum_vol_today 間接推斷，還沒直接對照過原始 payload log 逐欄確認，
如果之後對到某支股票的累積量兜不起來，先看 log 裡的原始 payload 長相，
必要時調整 _parse_candle()。

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

from data.m1_utils import _atomic_save, _parse_rest_bars
from data.query import load_m1_live
from fubon import fubon_api as trade_api
from fubon.subscribe_list import load_subscribe_batches

try:
    from api import append_system_log as _log_sys, set_collector_coverage
except Exception:
    def _log_sys(msg, level="info"): pass  # type: ignore
    def set_collector_coverage(arrived, total): pass  # type: ignore

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))
_LIVE_DIR = _ROOT / "db/m1_live"
_FLUSH_INTERVAL = float(os.environ.get("FUBON_WS_FLUSH_INTERVAL", "5"))
# 觸發推論的時機：收盤後固定等 :05 秒是「最短」等待，之後改成看涵蓋率——
# 涵蓋率達標（大部分股票的 candle 都到齊了）就提早觸發，資料慢的話最多等到
# FUBON_WS_MAX_WAIT_SEC 秒還是會觸發（保底，不會無限等下去）。
_COVERAGE_THRESHOLD = float(os.environ.get("FUBON_WS_COVERAGE_THRESHOLD", "0.95"))
_MAX_WAIT_SEC = float(os.environ.get("FUBON_WS_MAX_WAIT_SEC", "15"))
_COVERAGE_POLL_SEC = 1.0
# 富邦 intraday/candles rate limit 300次/分鐘（官方文件），節流到 ~240次/分鐘留緩衝。
# 這個間隔控制的是「發出下一個 request」的頻率，不是單一 worker 的間隔——
# 多執行緒下大家共用同一個節流時鐘，各自的網路等待時間可以互相重疊。
_BACKFILL_INTERVAL = float(os.environ.get("FUBON_REST_INTERVAL", "0.25"))
_BACKFILL_WORKERS = int(os.environ.get("FUBON_BACKFILL_WORKERS", "10"))


def _live_path(date_str: str) -> Path:
    return _LIVE_DIR / f"{date_str}.parquet"


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

    def __init__(self, on_minute=None, backfill_done=None, on_token_ready=None):
        self._on_minute = on_minute
        # threading.Event（見 main/state.py::AppState.backfill_done 的說明），
        # _backfill_intraday() 補完才 set()，讓呼叫端（on_minute）知道資料
        # 缺口補齊了沒有。留空（手動測試/單獨跑這個檔案時）就不做這個追蹤。
        self._backfill_done = backfill_done
        # 選填 callback(token)，在拿到 realtime_token() 之後立刻呼叫（見 start()）。
        # 給 fubon/tick_ws.py::FubonTickCollector 重用同一個 token 開連線，不用
        # 再登入一次富邦帳號——重複登入是已經修過的真實 bug（見 main/live_trader.py
        # 的 _startup_done 說明），tick collector 不該再犯一次。
        self._on_token_ready = on_token_ready
        self._sdk = None
        self._clients: list = []
        self._buffer: dict[tuple[str, str], dict] = {}  # (stock_id, minute_str) -> row
        self._buffer_lock = threading.Lock()
        # 保護 _atomic_save()：_flush_loop 跟 _minute_tick_loop 都會呼叫 _flush()，
        # 兩邊同時寫檔會搶同一個 .tmp 檔名，os.replace() 會噴 FileNotFoundError
        # （2026-07-13 實際發生過，導致 collector 整個掛掉、不會自動重連）。
        self._save_lock = threading.Lock()
        self._stop = False
        # 涵蓋率追蹤：只記股票代號，不存報價內容（報價內容仍在 self._buffer）。
        # _target_minute 是目前這一輪在等的那根已收盤分鐘，只有訊息剛好屬於這
        # 一分鐘才會被算進涵蓋率，避免跨分鐘的訊息把下一輪的涵蓋率灌水。
        self._arrived_this_minute: set[str] = set()
        self._target_minute: str | None = None
        self._total_subscribed = 0

    # ── 連線 ──────────────────────────────────────────────────────────────

    def start(self):
        print("富邦登入...", flush=True)
        self._sdk, accounts = trade_api.login()
        print(f"登入成功：{[a.name for a in accounts]}", flush=True)

        trade_api.init_market_data(self._sdk)  # candles channel 只支援 Normal mode（預設值）

        batches = load_subscribe_batches()
        if not batches:
            raise RuntimeError("訂閱清單是空的，請先跑 python -m fubon.subscribe_list")

        token = trade_api.realtime_token(self._sdk)
        if self._on_token_ready:
            self._on_token_ready(token)

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

        # WebSocket 已經在收即時資料了，backfill 補「連線前」的缺口丟到背景執行緒，
        # 不卡住即時資料流入（尤其是盤中重啟的情境）。先 clear()：每次重新連線
        # （包含 main/collector.py 自動重試時建立的新 collector 實例）都要重新
        # 走一次「還沒補完」的狀態，不能沿用上一輪（可能是舊 process 或舊連線）
        # 留下的、已經 set() 過的舊狀態。
        if self._backfill_done:
            self._backfill_done.clear()
        date_str = datetime.now(_TW).strftime("%Y-%m-%d")
        all_symbols = [sid for batch in batches for sid in batch]
        self._total_subscribed = len(all_symbols)
        threading.Thread(target=self._backfill_intraday, args=(all_symbols, date_str), daemon=True).start()

        threading.Thread(target=self._flush_loop, daemon=True).start()
        self._minute_tick_loop()  # 主執行緒 block 在這裡，直到 stop() 被呼叫

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
                if row["date"] == self._target_minute:
                    self._arrived_this_minute.add(row["stock_id"])
        return handler

    # ── 背景補歷史缺口（寫進共用 buffer，不直接存檔）─────────────────────

    def _backfill_intraday(self, symbols: list[str], date_str: str):
        """用 REST 補「WebSocket 連線前」缺的分K，寫進跟即時資料共用的
        self._buffer（由 _flush_loop 統一存檔），不直接呼叫存檔函式——避免
        這個背景執行緒跟 _flush_loop 同時讀寫同一個檔案造成 race condition。
        多執行緒併發抓取，全域節流頻率控制在 ~1/_BACKFILL_INTERVAL 次/秒，
        避免逐支序列等待網路延遲拖慢整體時間。
        """
        print(f"[backfill] 開始補 {len(symbols)} 支 {date_str} 的分K（背景執行，"
              f"{_BACKFILL_WORKERS} 併發）...", flush=True)
        t0 = time.time()
        rate_lock = threading.Lock()
        last_req = [0.0]
        done = [0]
        got = [0]

        def fetch_one(sid: str):
            with rate_lock:
                wait = _BACKFILL_INTERVAL - (time.time() - last_req[0])
                if wait > 0:
                    time.sleep(wait)
                last_req[0] = time.time()
            try:
                bars = trade_api.intraday_candles(self._sdk, sid)
                df = _parse_rest_bars(sid, bars, date_str)
                if not df.empty:
                    with self._buffer_lock:
                        for row in df.to_dict("records"):
                            key = (row["stock_id"], row["date"])
                            # WebSocket 即時資料優先：這個 key 已經被即時訊息
                            # 寫過就不覆蓋，backfill 只補真正缺的部分
                            if key not in self._buffer:
                                self._buffer[key] = row
                    got[0] += 1
            except Exception as e:
                print(f"[backfill] {sid} 失敗: {e}", flush=True)
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(symbols):
                print(f"[backfill] 進度 {done[0]}/{len(symbols)}", flush=True)

        with ThreadPoolExecutor(max_workers=_BACKFILL_WORKERS) as ex:
            list(ex.map(fetch_one, symbols))

        elapsed = time.time() - t0
        msg = f"[backfill] 完成，{got[0]}/{len(symbols)} 支有資料（{elapsed:.1f}s）"
        print(msg, flush=True)
        _log_sys(f"富邦 backfill 完成 {date_str}：{got[0]}/{len(symbols)} 支（{elapsed:.1f}s）")
        if self._backfill_done:
            self._backfill_done.set()

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
        with self._save_lock:
            for date_str, g in df.groupby(df["date"].str[:10]):
                _atomic_save(g.copy(), _live_path(date_str))
        print(f"[flush] 存檔 {len(df)} 筆（{df['stock_id'].nunique()} 支）", flush=True)

    # ── 每分鐘固定觸發一次 on_minute（保底機制）────────────────────────────
    #
    # 不管這分鐘 WebSocket 有沒有推新訊息（連線斷線、行情安靜、還沒開盤都
    # 可能發生），每分鐘都要固定呼叫一次 on_minute，否則 on_minute() 裡
    # 收盤後的 SL/TP reconcile 監控會因為某幾分鐘沒有新訊息就整段被跳過。
    # 用真實時鐘卡在每分鐘 :05 秒驅動，不是靠 buffer 裡有沒有「新完成的
    # 一分鐘」來判斷。

    def _minute_tick_loop(self):
        if not self._on_minute:
            return
        while not self._stop:
            now = datetime.now(_TW)
            min_tick = now.replace(second=5, microsecond=0) + timedelta(minutes=1)
            closed_minute = (min_tick - timedelta(minutes=1)).replace(second=0, microsecond=0)
            target_minute_str = closed_minute.strftime("%Y-%m-%d %H:%M:%S")
            with self._buffer_lock:
                self._target_minute = target_minute_str
                self._arrived_this_minute = set()

            wait = (min_tick - datetime.now(_TW)).total_seconds()
            if wait > 0:
                time.sleep(wait)
            if self._stop:
                break

            # 涵蓋率達標或等到上限，先到先觸發：資料到得快就不用死等 15 秒，
            # 到得慢也不會像原本固定 5 秒那樣太早截斷、漏掉一大半股票。
            deadline = min_tick + timedelta(seconds=_MAX_WAIT_SEC)
            total = self._total_subscribed or 1
            while not self._stop:
                with self._buffer_lock:
                    arrived = len(self._arrived_this_minute)
                if arrived / total >= _COVERAGE_THRESHOLD or datetime.now(_TW) >= deadline:
                    break
                time.sleep(_COVERAGE_POLL_SEC)
            if self._stop:
                break

            with self._buffer_lock:
                arrived = len(self._arrived_this_minute)
                self._arrived_this_minute = set()  # 送出這一輪後清空，重新計下一分鐘
            total = self._total_subscribed or 0
            pct = (arrived / total * 100) if total else 0
            print(f"[minute_tick] {target_minute_str} 涵蓋率 {arrived}/{total}（{pct:.0f}%）", flush=True)
            try:
                set_collector_coverage(arrived, total)
            except Exception:
                pass

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
