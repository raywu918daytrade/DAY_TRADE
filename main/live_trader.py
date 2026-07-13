"""
即時交易進入點（Render web service）

架構：
    主執行緒  → uvicorn（FastAPI + SSE）
    背景執行緒 → 分K收集器（main/collector.py，REST 或富邦 WebSocket）
    背景執行緒 → _daily_refresh（每天 06:00 更新當沖標的 + 日K，08:45 重算策略盤前快取）
    背景執行緒 → _force_close_eod（每天 13:25 強制平倉，當沖不過夜）

模組分工（這支檔案只管流程編排：呼叫順序、on_minute、排程）：
    main/config.py          → .env 讀出來的設定常數
    main/state.py           → AppState：跨執行緒共用的執行期狀態
    main/strategy_loader.py → 動態載入策略模組（切換模型／策略）
    main/premarket.py       → 盤前資料準備（當沖候選清單、日K、策略快取）
    main/backfill.py        → 開機補載推論（填 SSE monitoring，不用等下一分鐘）
    main/collector.py       → 分K收集器（REST／富邦 WebSocket 切換）

資料取用時段：
    盤前（<09:01）  : 日K 從 HF Hub / 本地載入；分K 從既有 parquet 讀取（昨日 backfill）
    盤中（09:01~13:00）: 分K收集器每分鐘 poll/推送，寫入 db/m1_live/
    盤後（>13:00）  : REST /historical/candles 補齊完整今日分K，再 sleep 到隔日

流程：
    on_minute → predict_live → push_monitoring → SSE monitoring event
                             → push_signals    → API /signals/today
                             → push_candles    → API /chart/{id}/candles
                             → reconcile       → 永豐 / paper / 富邦 下單
"""

import builtins as _builtins
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

_TW_TS = timezone(timedelta(hours=8))
_orig_print = _builtins.print


def _ts_print(*args, **kwargs):
    ts = datetime.now(_TW_TS).strftime("%H:%M:%S")
    _orig_print(f"[{ts}]", *args, **kwargs)


_builtins.print = _ts_print

import pandas as pd
import uvicorn

from api import (
    append_system_log as _log_sys,
    get_uvicorn_config,
    push_candles,
    push_inference_log,
    push_monitoring,
    push_signals,
    tw_naive_to_epoch,
    update_positions_price,
)

from main import collector as _collector
from main import premarket as _premarket
from main.backfill import run_startup_backfill
from main.config import (
    FORCE_CLOSE_HOUR as _FORCE_CLOSE_HOUR,
    FORCE_CLOSE_MIN as _FORCE_CLOSE_MIN,
    STRATEGY_MODULE,
    THRESHOLD,
    TOTAL_CAPITAL,
    TRADE_MODE,
)
from main.state import AppState
from main.strategy_loader import load_strategy

from data.data_manager import Phase
from data.query import load_m1_live

_TW = timezone(timedelta(hours=8))

state = AppState()

# strategy/base/date_trade_model.py 已刪除（2026-07-09，特徵整合進
# strategy/rally），STRATEGY_MODULE 預設值改成 rally；每個策略模組都要暴露
# 同一組介面：load_model() / predict_live(...) / SESSION_START / SESSION_END
state.strategy_module = load_strategy(STRATEGY_MODULE)
state.session_start = state.strategy_module.SESSION_START
state.session_end = state.strategy_module.SESSION_END
state.load_model = state.strategy_module.load_model
state.predict_live = state.strategy_module.predict_live
print(f"[策略] 使用 {STRATEGY_MODULE}", flush=True)
_log_sys(f"策略模組：{STRATEGY_MODULE}")

print("載入模型...")
sys.stdout.flush()
try:
    state.model = state.load_model()
    print("✓ 模型載入成功", flush=True)
    _log_sys(f"模型載入成功（策略：{STRATEGY_MODULE}，model={type(state.model).__name__}）")
except Exception as e:
    print(f"✗ 模型載入失敗: {e}", flush=True)
    raise

print("更新當沖標的清單...", flush=True)
try:
    _premarket.refresh_tickers(state)
    print(f"  當沖標的（API）：{len(state.tickers)} 支", flush=True)
except Exception as e:
    print(f"✗ 取得當沖標的失敗: {e}", flush=True)
    state.tickers = {}
    state.day_trade_stocks = None

# 盤中才有標的清單，非盤中跳過日K載入（省記憶體）；_daily_refresh 在 06:00 補載
if state.day_trade_stocks:
    print(f"[D1] 載入日K（{Phase.PRE_MARKET.value}）...", flush=True)
    try:
        _premarket.refresh_day(state)
        _d1_last = state.day["date"].max() if not state.day.empty else "無"
        print(
            f"✓ [D1] {len(state.day):,} 筆，{state.day['stock_id'].nunique():,} 支，最新日期：{_d1_last}",
            flush=True,
        )
        _log_sys(f"D1 載入完成：{state.day['stock_id'].nunique() if not state.day.empty else 0} 支，最新 {_d1_last}")
    except Exception as e:
        print(f"✗ [D1] 載入失敗: {e}", flush=True)
        _log_sys(f"D1 載入失敗: {e}", "error")
        state.day = pd.DataFrame()
else:
    state.day = pd.DataFrame()
    print("[D1] 非盤中，跳過日K載入（_daily_refresh 06:00 更新）", flush=True)

print("盤前預算快取...", flush=True)
try:
    _premarket.refresh_prewarm(state)
    print(f"✓ 盤前快取完成：{list(state.prewarm_cache.keys())}", flush=True)
except Exception as e:
    print(f"✗ 盤前快取失敗，改用 predict_live() 內建 fallback: {e}", flush=True)
    state.prewarm_cache = {}

# 啟動補載：若今日已有 m1_live，立刻跑推論填 _monitoring（不用等下一分鐘）
run_startup_backfill(state, THRESHOLD)

print(f"就緒，等待盤中訊號（門檻={THRESHOLD}）...", flush=True)

if TRADE_MODE != "off":
    try:
        from trade.run_execute import make_executor

        state.executor = make_executor(
            TRADE_MODE,
            TOTAL_CAPITAL,
            name_lookup=lambda sid: state.tickers.get(sid, sid),
        )
        print(f"交易模式：{TRADE_MODE}，資金={TOTAL_CAPITAL:,.0f}", flush=True)
        _log_sys(f"交易引擎啟動：{TRADE_MODE} 模式，資金={TOTAL_CAPITAL:,.0f}")
        if hasattr(state.executor, "sync_from_broker"):
            state.executor.sync_from_broker()
        if hasattr(state.executor, "startup_sltp_check"):
            state.executor.startup_sltp_check()
        if hasattr(state.executor, "close_stock_now"):
            from api import register_close_now

            register_close_now(state.executor.close_stock_now)
    except Exception as e:
        print(f"[WARN] 交易模組載入失敗，改為僅推訊號: {e}", flush=True)
        _log_sys(f"交易模組載入失敗: {e}", "error")


def _daily_refresh():
    """每天 06:00 更新當沖清單與日K；08:45（開盤前15分）重算盤前策略快取
    （Render 24小時常駐用）"""
    last_refresh = None
    last_prewarm = None
    while True:
        now = datetime.now(_TW)
        today = now.date()
        need_refresh = last_refresh != today and now.hour == 6 and now.minute >= 0
        # 啟動時若日K是空的（非盤中跳過）且已過 06:00，立即補載
        need_refresh = need_refresh or (last_refresh is None and state.day.empty and now.hour >= 6)
        if need_refresh:
            print(f"[{now.strftime('%H:%M')}] 每日更新：當沖標的 + 日K...")
            try:
                _premarket.refresh_tickers(state)
                if state.day_trade_stocks:
                    _premarket.refresh_day(state)
                last_refresh = today
                n = len(state.day_trade_stocks) if state.day_trade_stocks else 0
                print(f"  更新完成，當沖標的：{n} 支")
                _log_sys(f"每日更新完成（{Phase.PRE_MARKET.value}）：{n} 支標的")
            except Exception as e:
                print(f"  更新失敗: {e}")

        need_prewarm = last_prewarm != today and (now.hour, now.minute) >= (8, 45)
        if need_prewarm:
            print(f"[{now.strftime('%H:%M')}] 盤前策略快取重算...")
            try:
                _premarket.refresh_prewarm(state)
                last_prewarm = today
                print(f"  快取完成：{list(state.prewarm_cache.keys())}")
                _log_sys(f"盤前策略快取重算完成：{list(state.prewarm_cache.keys())}")
            except Exception as e:
                print(f"  快取失敗: {e}")
                _log_sys(f"盤前策略快取失敗: {e}", "error")
        time.sleep(60)


def on_minute(minute_str: str, df: pd.DataFrame):
    """分K收集器每分鐘回呼：推論、推送監控/K線/訊號，並觸發下單。"""
    dt = pd.Timestamp(minute_str)
    h, m = dt.hour, dt.minute

    if (h, m) < state.session_start:
        return

    # SESSION_END 後：停止開倉，但繼續跑 reconcile 做 SL/TP 監控直到收盤
    if (h, m) > state.session_end:
        if state.executor is not None:
            price_map = {}
            if not df.empty and "stock_id" in df.columns and "close" in df.columns:
                price_map = dict(zip(df["stock_id"].astype(str), df["close"].astype(float)))
                update_positions_price(price_map)
            try:
                state.executor.reconcile([], prices=price_map)  # signals=[] 不開倉，只跑 SL/TP
            except Exception as e:
                print(f"[TRADE ERROR after session] {e}")
        return

    # 載入今日分K（只載一次，下面 push_candles 和 predict_live 共用）
    date_str = minute_str[:10]
    m1_live = load_m1_live(date_str)
    print(
        f"[on_minute] {minute_str[11:16]}  M1:{m1_live['stock_id'].nunique() if not m1_live.empty else 0} 支 {len(m1_live):,} 筆",
        flush=True,
    )

    if not m1_live.empty:
        for sid, g in m1_live.groupby("stock_id"):
            candles = []
            for _, row in g.iterrows():
                dt = datetime.strptime(str(row["date"]), "%Y-%m-%d %H:%M:%S")
                ts = tw_naive_to_epoch(dt)
                candles.append(
                    {
                        "time": ts,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                )
            push_candles(str(sid), candles)

    # 模型推論：threshold=0 取得所有股票機率，共用已載入的 m1_live
    all_results = state.predict_live(
        minute_str,
        state.day,
        model=state.model,
        threshold=0,
        day_trade_stocks=state.day_trade_stocks,
        m1_live=m1_live if not m1_live.empty else None,
        **state.prewarm_cache,
    )
    for r in all_results:
        r["name"] = state.tickers.get(r["stock_id"], r["stock_id"])
    # 每分鐘從 settings 讀信心度，允許前端即時調整
    from api import get_setting

    threshold = float(get_setting("threshold") or THRESHOLD)

    push_monitoring(minute_str, all_results, threshold)
    push_inference_log(minute_str, all_results, threshold)

    # 用最新分K收盤價更新持倉卡片浮動損益（今日損益只計已實現，不含此處）
    price_map = {r["stock_id"]: r["price"] for r in all_results}
    update_positions_price(price_map)

    # 只有達門檻才產生交易訊號
    signals = [r for r in all_results if r["proba"] >= threshold]
    push_signals(minute_str, signals)

    top = sorted(all_results, key=lambda x: -x["proba"])[:5]
    top_str = " ".join(f"{r['stock_id']}={r['proba']:.2f}" for r in top)
    print(
        f"  推論:{len(all_results)} 支  訊號:{len(signals)} 支（門檻={threshold:.2f}）  top5:[{top_str}]",
        flush=True,
    )
    _log_sys(
        f"推論 {minute_str}：{len(all_results)} 支 → {len(signals)} 個訊號（門檻={threshold:.2f}）  top5:[{top_str}]"
    )
    if signals:
        sig_str = " ".join(f"{s['stock_id']}={s['proba']:.2f}" for s in signals)
        print(f"  → 訊號: {sig_str}", flush=True)
        _log_sys(f"訊號: {sig_str}")

    if state.executor is not None:
        try:
            state.executor.reconcile(signals, prices=price_map)
        except Exception as e:
            print(f"[TRADE ERROR] {e}")
            _log_sys(f"reconcile 錯誤: {e}", "error")


_force_close_done_date = None  # 防止同一天重複觸發


def _force_close_eod():
    """每天強制平倉所有當沖部位（不過夜）。
    時間優先讀 settings 的 force_close_time（HH:MM），否則用 .env 的 FORCE_CLOSE_HOUR/MIN。
    每 30 秒檢查一次，允許前端即時改時間。
    """
    global _force_close_done_date
    from api import get_setting

    while True:
        now = datetime.now(_TW)
        today = now.date()
        # 每次都重新讀，允許前端即時改
        fc_str = get_setting("force_close_time") or f"{_FORCE_CLOSE_HOUR}:{_FORCE_CLOSE_MIN:02d}"
        try:
            fc_h, fc_m = map(int, fc_str.split(":"))
        except Exception:
            fc_h, fc_m = _FORCE_CLOSE_HOUR, _FORCE_CLOSE_MIN
        if (now.hour, now.minute) == (fc_h, fc_m) and _force_close_done_date != today and state.executor is not None:
            _force_close_done_date = today
            print(f"[{now.strftime('%H:%M:%S')}] 強制平倉：本系統當沖倉位（{fc_str}）", flush=True)
            try:
                state.executor.force_close_own_positions()
            except Exception as e:
                print(f"[FORCE CLOSE] 錯誤: {e}", flush=True)
            time.sleep(10)
            if hasattr(state.executor, "sync_from_broker"):
                try:
                    state.executor.sync_from_broker()
                    print(f"[FORCE CLOSE] 盤後同步完成", flush=True)
                except Exception as e:
                    print(f"[FORCE CLOSE] 盤後同步失敗: {e}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    # 每日 08:45 更新排程
    threading.Thread(target=_daily_refresh, daemon=True).start()
    # 每日 13:25 強制平倉（當沖不過夜）
    threading.Thread(target=_force_close_eod, daemon=True).start()
    # 分K收集器在背景執行緒
    threading.Thread(target=lambda: _collector.start_collector(state, on_minute), daemon=True).start()

    # uvicorn 跑主執行緒（阻塞）
    config = get_uvicorn_config(host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    server.run()
