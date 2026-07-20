"""
即時交易進入點（Render web service）

架構：
    主執行緒  → uvicorn（FastAPI + SSE）
    背景執行緒 → 分K收集器（main/collector.py，富邦 WebSocket）
    背景執行緒 → _daily_refresh（每天 06:00 更新當沖標的 + 日K，08:45 重算策略盤前快取）
    背景執行緒 → _force_close_eod（每天 13:25 強制平倉，當沖不過夜）

模組分工（這支檔案只管流程編排：呼叫順序、on_minute、排程）：
    main/config.py          → .env 讀出來的設定常數
    main/state.py           → AppState：跨執行緒共用的執行期狀態
    main/strategy_loader.py → 動態載入策略模組（切換模型／策略）
    main/premarket.py       → 盤前資料準備（當沖候選清單、日K、策略快取）
    main/backfill.py        → 開機補載推論（填 SSE monitoring，不用等下一分鐘）
    main/collector.py       → 分K收集器（富邦 WebSocket，fubon/marketdata_ws.py）

資料取用時段：
    盤前（<09:01）  : 日K 從 HF Hub / 本地載入；分K 從既有 parquet 讀取（昨日 backfill）
    盤中（09:01~13:00）: 富邦 WebSocket 即時推送，寫入 db/m1_live/
    盤後（>13:00）  : 持續收 WebSocket 資料到收盤，SL/TP reconcile 監控不中斷

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
    push_consensus_signals,
    push_inference_log,
    push_monitoring,
    push_quote,
    push_signals,
    set_strategies,
    tw_naive_to_epoch,
    update_positions_price,
)

from main import collector as _collector
from main import premarket as _premarket
from main.backfill import run_startup_backfill
from main.config import (
    CONSENSUS_TOP_N,
    FORCE_CLOSE_HOUR as _FORCE_CLOSE_HOUR,
    FORCE_CLOSE_MIN as _FORCE_CLOSE_MIN,
    STRATEGY_MODULES,
    TOTAL_CAPITAL,
    TRADE_MODE,
    WATCHLIST_QUOTES,
)
from main.state import AppState, StrategyState
from main.strategy_loader import load_strategies

from data.data_manager import Phase
from data.query import load_m1_live

_TW = timezone(timedelta(hours=8))

state = AppState()

# 可同時載入多個策略模組（見 main/config.py 的 STRATEGY_MODULES）；每個策略
# 模組都要暴露同一組介面：load_model() / predict_live(...) / SESSION_START /
# SESSION_END，包成 StrategyState 存進 state.strategies（key=策略名）。
for _name, _module in load_strategies(STRATEGY_MODULES).items():
    state.strategies[_name] = StrategyState(_name, _module)
print(f"[策略] 使用 {list(state.strategies.keys())}（{STRATEGY_MODULES}）", flush=True)
_log_sys(f"策略模組：{list(state.strategies.keys())}")

# 登記給前端 GET /strategies 查詢用，讓前端開機就知道有幾個模型、各自的
# session範圍，不用等第一筆 monitoring/signals 事件才知道（見 api.py 的
# set_strategies() 說明）。
set_strategies(
    [
        {
            "name": _s.name,
            "session_start": f"{_s.session_start[0]:02d}:{_s.session_start[1]:02d}",
            "session_end": f"{_s.session_end[0]:02d}:{_s.session_end[1]:02d}",
        }
        for _s in state.strategies.values()
    ],
    consensus_top_n=CONSENSUS_TOP_N,
)

print("載入模型...")
sys.stdout.flush()
try:
    for _s in state.strategies.values():
        _s.model = _s.load_model()
        print(f"✓ [{_s.name}] 模型載入成功（{type(_s.model).__name__}）", flush=True)
    _log_sys(f"模型載入成功：{[(s.name, type(s.model).__name__) for s in state.strategies.values()]}")
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
    print(f"✓ 盤前快取完成：{ {s.name: list(s.prewarm_cache.keys()) for s in state.strategies.values()} }", flush=True)
except Exception as e:
    print(f"✗ 盤前快取失敗，改用 predict_live() 內建 fallback: {e}", flush=True)

# 啟動補載：若今日已有 m1_live，立刻跑推論填 _monitoring（不用等下一分鐘）
run_startup_backfill(state)

_threshold_str = ", ".join(f"{s.name}={s.threshold}" for s in state.strategies.values())
print(f"就緒，等待盤中訊號（各策略門檻預設：{_threshold_str}）...", flush=True)

if TRADE_MODE != "off":
    try:
        from sinopac.run_execute import make_executor

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
                _cache_keys = {s.name: list(s.prewarm_cache.keys()) for s in state.strategies.values()}
                print(f"  快取完成：{_cache_keys}")
                _log_sys(f"盤前策略快取重算完成：{_cache_keys}")
            except Exception as e:
                print(f"  快取失敗: {e}")
                _log_sys(f"盤前策略快取失敗: {e}", "error")
        time.sleep(60)


_prev_close_cache: dict[str, tuple[str, float]] = {}  # {stock_id: (date_str, prev_close)}


def _watchlist_prev_close(stock_id: str, date_str: str) -> float | None:
    """前一交易日收盤價，一天只查一次本機 db/fugle_day/（不是每分鐘都重讀 parquet）。
    跟策略候選股的均量篩選/當沖資格判斷無關，0050 這種 ETF 不會進策略候選池，
    也要能查得到，所以直接查 data/query.py 的日K，不依賴 state.day。"""
    cached = _prev_close_cache.get(stock_id)
    if cached and cached[0] == date_str:
        return cached[1]

    from data.query import load_day_by_stock

    df = load_day_by_stock(stock_id)
    if df.empty:
        return None
    df = df[df["date"] < pd.Timestamp(date_str)]
    if df.empty:
        return None
    val = float(df.iloc[-1]["close"])
    _prev_close_cache[stock_id] = (date_str, val)
    return val


def on_minute(minute_str: str, df: pd.DataFrame):
    """分K收集器每分鐘回呼：每個策略各自推論、推送監控/訊號，並觸發下單。

    2026-07-13 改成多策略：SESSION_START/SESSION_END 現在是每個策略各自的
    範圍（StrategyState.session_start/session_end），不是單一全域邊界——
    候選/K線資料共用一份 m1_live，但要不要對某個策略呼叫 predict_live() 是
    逐一策略各自判斷。目前 TRADE_MODE 還是關的（交易先暫停，等實盤穩定），
    reconcile() 收到的 all_signals 是把所有策略達門檻的訊號直接合併，還沒
    處理「兩個策略同時看好同一支股票」的衝突，見 main/config.py 的說明。
    """
    dt = pd.Timestamp(minute_str)
    h, m = dt.hour, dt.minute

    # 載入今日分K（只載一次，下面 push_candles 和每個策略的 predict_live 共用）
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
                dt2 = datetime.strptime(str(row["date"]), "%Y-%m-%d %H:%M:%S")
                ts = tw_naive_to_epoch(dt2)
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

            # 頁首固定追蹤清單（WATCHLIST_QUOTES，如 0050）：跟策略候選股無關，
            # 收到 m1 就先查昨收、算漲跌幅，傳到前端之前先算好（不是前端自己拉
            # candles 再算），見 main/config.py 的說明與 api.py 的 push_quote()。
            if str(sid) in WATCHLIST_QUOTES and candles:
                prev_close = _watchlist_prev_close(str(sid), date_str)
                push_quote(str(sid), candles[-1]["close"], prev_close, minute_str)

    from api import get_setting

    # price_map 先用這分鐘全部收到的分K收盤價打底（涵蓋所有目前有持倉的股票，
    # 不受任何策略候選清單過濾影響），下面各策略的 predict_live 結果再疊上去
    # （通常是同一個值，只是保險起見以最新推論結果優先）。
    price_map = {}
    if not df.empty and "stock_id" in df.columns and "close" in df.columns:
        price_map = dict(zip(df["stock_id"].astype(str), df["close"].astype(float)))

    # volume_map：這一分鐘各股票自己的成交量（不是累積量），給前端監控參考用，
    # 跟 price_map 同樣的來源（這分鐘收到的分K），不影響任何策略的推論輸入。
    volume_map = {}
    if not df.empty and "stock_id" in df.columns and "volume" in df.columns:
        volume_map = dict(zip(df["stock_id"].astype(str), df["volume"].astype(int)))

    all_signals = []
    top_by_strategy: dict[str, list[dict]] = {}  # 各策略前N名（依proba排名，不管有沒有過門檻），給下面重疊比對用
    for s in state.strategies.values():
        if (h, m) < s.session_start or (h, m) > s.session_end:
            continue  # 這個策略還沒開始/已經過了進場窗口，不產生新訊號（既有持倉SL/TP由下面reconcile統一監控，不受這裡影響）

        # 每分鐘從 settings 讀信心度，允許前端即時調整；沒設定才 fallback 該
        # 策略自己的預設值（見 main/state.py::StrategyState 的說明——三個
        # 模型的機率校準不一定一樣，2026-07-21 從全域共用一個門檻拆成各策略
        # 各自一個）。
        threshold = float(get_setting("threshold") or s.threshold)

        all_results = s.predict_live(
            minute_str,
            state.day,
            model=s.model,
            threshold=0,
            day_trade_stocks=state.day_trade_stocks,
            m1_live=m1_live if not m1_live.empty else None,
            **s.prewarm_cache,
        )
        for r in all_results:
            r["name"] = state.tickers.get(r["stock_id"], r["stock_id"])
            r["strategy"] = s.name  # 保留來源策略，reconcile() 收到的合併清單才分得出是誰產生的
            r["volume"] = volume_map.get(r["stock_id"])

        push_monitoring(minute_str, all_results, threshold, strategy=s.name)
        push_inference_log(minute_str, all_results, threshold, strategy=s.name)
        price_map.update({r["stock_id"]: r["price"] for r in all_results})

        signals = [r for r in all_results if r["proba"] >= threshold]
        push_signals(minute_str, signals, strategy=s.name)
        all_signals.extend(signals)

        top = sorted(all_results, key=lambda x: -x["proba"])[:5]
        top_by_strategy[s.name] = sorted(all_results, key=lambda x: -x["proba"])[:CONSENSUS_TOP_N]
        top_str = " ".join(f"{r['stock_id']}={r['proba']:.2f}" for r in top)
        print(
            f"  [{s.name}] 推論:{len(all_results)} 支  訊號:{len(signals)} 支（門檻={threshold:.2f}）  top5:[{top_str}]",
            flush=True,
        )
        _log_sys(
            f"[{s.name}] 推論 {minute_str}：{len(all_results)} 支 → {len(signals)} 個訊號（門檻={threshold:.2f}）  top5:[{top_str}]"
        )
        if signals:
            sig_str = " ".join(f"{r['stock_id']}={r['proba']:.2f}" for r in signals)
            print(f"  [{s.name}] → 訊號: {sig_str}", flush=True)
            _log_sys(f"[{s.name}] 訊號: {sig_str}")

    # 多策略共識訊號：等所有策略這分鐘都跑完，比對誰的前N名彼此重疊（見
    # main/config.py 的 CONSENSUS_TOP_N）。2個以上策略同時看好同一支股票，
    # 才算共識，跟各策略自己的 monitoring/signals 分開推送，不取代它們。
    if len(top_by_strategy) >= 2:
        stock_hits: dict[str, dict[str, float]] = {}  # stock_id -> {策略名: proba}
        stock_names: dict[str, str] = {}
        for name, top in top_by_strategy.items():
            for r in top:
                stock_hits.setdefault(r["stock_id"], {})[name] = r["proba"]
                stock_names[r["stock_id"]] = r["name"]
        consensus = [
            {
                "stock_id": sid,
                "name": stock_names[sid],
                "strategies": sorted(probas.keys()),
                "probas": probas,
            }
            for sid, probas in stock_hits.items()
            if len(probas) >= 2
        ]
        if consensus:
            consensus.sort(key=lambda c: -sum(c["probas"].values()) / len(c["probas"]))
            push_consensus_signals(minute_str, consensus)
            print(f"  [共識] {len(consensus)} 支同時進入 {CONSENSUS_TOP_N} 名內: "
                  + " ".join(f"{c['stock_id']}({'+'.join(c['strategies'])})" for c in consensus), flush=True)

    if price_map:
        update_positions_price(price_map)

    if state.executor is not None:
        try:
            state.executor.reconcile(all_signals, prices=price_map)
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
    threading.Thread(target=lambda: _collector.start_collector(on_minute), daemon=True).start()

    # uvicorn 跑主執行緒（阻塞）
    config = get_uvicorn_config(host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    server.run()
