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

# 2026-07-25發現並實測驗證：process 會 100% 重現的 SIGSEGV（macOS crash log
# 顯示 faulting thread 卡在 libomp.dylib 的 __kmp_launch_worker/
# __kmp_fork_barrier，不是 Python 例外，不會印 traceback，看起來像「安靜地
# 自動結束」）。根本原因是 import 順序：strategy/cnn 用 PyTorch（torch），
# 其他策略用 XGBoost/LightGBM，這幾個套件各自帶了一份 OpenMP 執行期
# （libomp）。main/strategy_loader.py::load_strategies() 依 STRATEGY_MODULES
# 順序把所有策略模組一次 import 完，torch 是 module 頂層 import、立刻載入；
# xgboost/lightgbm 是各自 train.py 裡的 function-local import，要等真的呼叫
# load_model() 才觸發——如果 torch 先進程序、lightgbm/xgboost 後進，就會
# SIGSEGV（實測：對調順序，先 import lightgbm/xgboost 再讓 torch 進來，
# 6個模型全部正常載入；跟多執行緒/_startup()無關，單執行緒同步跑一樣會
# 炸，已排除是 threading 造成的）。修法：搶在任何策略模組被 import 之前，
# 這裡先 import lightgbm/xgboost，確保它們永遠比 torch 早進程序，不管
# STRATEGY_MODULES 怎麼排、以後策略怎麼加都不會踩到這個順序問題。
# KMP_DUPLICATE_LIB_OK=TRUE 是額外的保險（常見 OpenMP 重複載入建議設定），
# 實測單獨設這個沒有解決順序問題本身，但保留著無害。
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm  # noqa: F401  只是為了搶在 torch 之前把 libomp 執行期建立好，見上方說明
import xgboost  # noqa: F401

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
import numpy as np
import uvicorn

from api import (
    append_system_log as _log_sys,
    get_uvicorn_config,
    push_candles,
    push_conflict_signals,
    push_consensus_signals,
    push_inference_log,
    push_monitoring,
    push_quote,
    push_signals,
    push_sr_vwap_cross,
    push_vwap_breakout,
    push_vwap_macd_div,
    register_vwap_sr_catchup_hook,
    set_strategies,
    tw_naive_to_epoch,
    update_positions_price,
)

from main import collector as _collector
from main import premarket as _premarket
from main.backfill import run_startup_backfill
from main.config import (
    CONFLICT_THRESHOLD,
    CONSENSUS_MIN_PROBA,
    CONSENSUS_TOP_N,
    FORCE_CLOSE_HOUR as _FORCE_CLOSE_HOUR,
    FORCE_CLOSE_MIN as _FORCE_CLOSE_MIN,
    MARKET_CLOSE_HOUR as _MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MIN as _MARKET_CLOSE_MIN,
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


def _on_vwap_sr_catchup(vwap: list, sr: list):
    for e in vwap:
        state.vwap_crossed_today.add(str(e["stock_id"]))
    for e in sr:
        state.vwap_crossed_today.add(str(e["stock_id"]))


register_vwap_sr_catchup_hook(_on_vwap_sr_catchup)
# _startup() 跑完（成功或失敗）就會 set，_daily_refresh() 的立即補載判斷要
# 等這個，避免兩邊背景執行緒同時搶著呼叫 refresh_tickers()（見 _startup()、
# _daily_refresh() 的說明）。
_startup_done = threading.Event()

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
            "directions": sorted(_s.directions),  # 前端依此上色（見 dashboard.html）
        }
        for _s in state.strategies.values()
    ],
    consensus_top_n=CONSENSUS_TOP_N,
)


def _startup():
    """開機序列（模型載入／當沖標的清單／日K／盤前快取／交易引擎），搬進背景
    執行緒跑（2026-07-25討論）：refresh_tickers() 拿掉 DAILY_REFRESH_TICKERS
    開關後，每次開機都要重新跑一次均量排序+逐支查 canBuyDayTrade（811支×
    sleep(0.25)≈3~4分鐘，見 fubon/subscribe_list.py::_filter_day_tradable()
    的說明），這段邏輯寫在模組最外層、卡在 uvicorn.Server().run() 前面時，
    整個 API 都還沒起來、前端連 SSE 一律「連線中斷」，要等這幾分鐘跑完
    才能連上。搬成背景執行緒後 uvicorn 立刻起來，這些資料背景慢慢補齊。

    可以放心搬的原因：各策略 predict_live() 本來就有 model=None 時自動
    load_model_by_type() 的 fallback（見 strategy/*/predict.py），collector
    提早開始收資料、on_minute() 提早呼叫 predict_live() 也不會出錯，只是
    model 還沒背景載完的那幾次呼叫會各自重新讀一次模型檔，效能小損失換
    前端能立刻連上。
    """
    print("載入模型...")
    sys.stdout.flush()
    try:
        for _s in state.strategies.values():
            _s.model = _s.load_model()
            print(f"✓ [{_s.name}] 模型載入成功（{type(_s.model).__name__}）", flush=True)
        _log_sys(f"模型載入成功：{[(s.name, type(s.model).__name__) for s in state.strategies.values()]}")
    except Exception as e:
        print(f"✗ 模型載入失敗: {e}", flush=True)
        _startup_done.set()
        return

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
            _log_sys(
                f"D1 載入完成：{state.day['stock_id'].nunique() if not state.day.empty else 0} 支，最新 {_d1_last}"
            )
            _refresh_sr_levels(datetime.now(_TW).strftime("%Y-%m-%d"))
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
        print(
            f"✓ 盤前快取完成：{ {s.name: list(s.prewarm_cache.keys()) for s in state.strategies.values()} }", flush=True
        )
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

    _startup_done.set()


def _daily_refresh():
    """每天 06:00 更新當沖清單與日K；08:45（開盤前15分）重算盤前策略快取
    （Render 24小時常駐用）。

    2026-07-25 修正：啟動時的立即補載判斷要等 _startup() 跑完（_startup_done
    這個 threading.Event）才檢查——_startup() 現在是背景執行緒，跟這支函式
    幾乎同時啟動，如果不等，兩邊會在同一瞬間都判斷「day是空的、已過06:00」
    而同時各自呼叫 refresh_tickers()，變成兩個執行緒同時登入富邦、同時查
    候選清單（實際發生過：log 裡「每日更新：當沖標的 + 日K...」跟
    「載入模型...」交錯出現）。等 _startup() 完成後再檢查，state.day 才會
    是最終狀態，這裡的判斷才準確。
    """
    last_refresh = None
    last_prewarm = None
    while True:
        now = datetime.now(_TW)
        today = now.date()
        need_refresh = last_refresh != today and now.hour == 6 and now.minute >= 0
        # 啟動時若日K是空的（非盤中跳過）且已過 06:00，立即補載——一定要等
        # _startup() 完成，避免跟它同時搶著呼叫 refresh_tickers()。
        need_refresh = need_refresh or (
            last_refresh is None and _startup_done.is_set() and state.day.empty and now.hour >= 6
        )
        if need_refresh:
            print(f"[{now.strftime('%H:%M')}] 每日更新：當沖標的 + 日K...")
            try:
                _premarket.refresh_tickers(state)
                if state.day_trade_stocks:
                    _premarket.refresh_day(state)
                    _refresh_sr_levels(now.strftime("%Y-%m-%d"))
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
    """前一交易日收盤價，一天只查一次本機 db/d1/（不是每分鐘都重讀 parquet）。
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


def _refresh_sr_levels(date_str: str):
    """換日／開機：用 D 之前日K 算橫向壓力／支撐，盤中不重算。"""
    from data.adjustment_query import load_pattern_day
    from pattern.horizontal_sr import horizontal_sr_prices

    stocks = {str(s) for s in (state.day_trade_stocks or state.tickers.keys())}
    if not stocks:
        state.sr_levels = {}
        state.sr_levels_date = date_str
        return
    hist_start = (pd.Timestamp(date_str) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    print(f"[SR水位] 預先計算橫向壓力/支撐 {len(stocks)} 檔（日K < {date_str}）…", flush=True)
    day = load_pattern_day(start_date=hist_start, end_date=date_str)
    if day.empty:
        state.sr_levels = {}
        state.sr_levels_date = date_str
        print("  [SR水位] 無日K", flush=True)
        return
    day["stock_id"] = day["stock_id"].astype(str)
    cutoff = pd.Timestamp(date_str)
    levels = {}
    for sid, g in day.groupby("stock_id", sort=False):
        sid = str(sid)
        if sid not in stocks:
            continue
        hist = g.loc[g["date"] < cutoff]
        res, sup = horizontal_sr_prices(hist)
        if res is not None or sup is not None:
            levels[sid] = (res, sup)
    state.sr_levels = levels
    state.sr_levels_date = date_str
    print(f"  [SR水位] {len(levels)} 檔有壓力/支撐", flush=True)


def _ensure_sr_vwap_day(date_str: str):
    if state.sr_levels_date == date_str:
        return
    state.vwap_crossed_today.clear()
    state.sr_vwap_fired_today.clear()
    _refresh_sr_levels(date_str)


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
    _ensure_sr_vwap_day(date_str)
    m1_live = load_m1_live(date_str)
    print(
        f"[on_minute] {minute_str[11:16]}  M1:{m1_live['stock_id'].nunique() if not m1_live.empty else 0} 支 {len(m1_live):,} 筆",
        flush=True,
    )

    # VWAP突破/跌破偵測（2026-08-12加，見前端「VWAP突破」欄）：跟下面
    # push_candles 共用同一份已經依 stock_id/date 排好序的 m1_live（見
    # data/query.py::load_m1_live()），不用另外重查。這一批（這一分鐘）
    # 偵測到的全部股票，等迴圈跑完一次性推送，跟 push_consensus_signals()/
    # push_conflict_signals() 同一種「這分鐘的結果一次送一批」節奏。
    vwap_breakouts = []
    sr_vwap_hits = []
    macd_by_sid: dict = {}

    if not m1_live.empty:
        from pattern.vwap_sr_scan import events_for_group
        from pattern.vwap_macd_div import all_divs_for_group

        hhmm = minute_str[11:16]
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
            sid = str(sid)

            macd_evs = all_divs_for_group(g)
            if macd_evs:
                macd_by_sid[sid] = {"events": macd_evs}

            # VWAP 只算一次：突破欄每次變號一筆；壓力/支撐同一套規則可重複。
            if len(g) >= 2:
                ve, se = events_for_group(
                    sid,
                    state.tickers.get(sid, sid),
                    g,
                    state.sr_levels.get(sid),
                )
                for e in ve:
                    if e.get("time") != hhmm:
                        continue
                    vwap_breakouts.append(e)
                    state.vwap_crossed_today.add(sid)
                for e in se:
                    if e.get("time") == hhmm:
                        sr_vwap_hits.append(e)

            # 頁首固定追蹤清單（WATCHLIST_QUOTES，如 0050）：跟策略候選股無關，
            # 收到 m1 就先查昨收、算漲跌幅，傳到前端之前先算好（不是前端自己拉
            # candles 再算），見 main/config.py 的說明與 api.py 的 push_quote()。
            # 2026-08-14加：收盤（MARKET_CLOSE_HOUR:MIN，預設13:30）後不再推
            # ——WebSocket 收盤後還是會繼續收資料（SL/TP reconcile 監控不能
            # 中斷，見模組頂端說明），但報價已經是收盤價、不會再變，沒必要
            # 讓前端SSE一直收到「新」的報價更新。
            if str(sid) in WATCHLIST_QUOTES and candles and (h, m) <= (_MARKET_CLOSE_HOUR, _MARKET_CLOSE_MIN):
                prev_close = _watchlist_prev_close(str(sid), date_str)
                push_quote(str(sid), candles[-1]["close"], prev_close, minute_str)

    if vwap_breakouts:
        push_vwap_breakout(minute_str, vwap_breakouts)
        print(
            f"  [VWAP突破] {len(vwap_breakouts)} 支跨過VWAP: "
            + " ".join(f"{b['stock_id']}({'突破' if b['direction']=='up' else '跌破'})" for b in vwap_breakouts),
            flush=True,
        )
    if sr_vwap_hits:
        push_sr_vwap_cross(minute_str, sr_vwap_hits)
        print(
            f"  [VWAP+壓力支撐] {len(sr_vwap_hits)} 筆: "
            + " ".join(f"{b['stock_id']}({b['sr_kind']})" for b in sr_vwap_hits),
            flush=True,
        )
    if not m1_live.empty:
        push_vwap_macd_div(minute_str, macd_by_sid)

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
    # 各策略前N名，依方向分開存（見 main/config.py::CONSENSUS_TOP_N 的說明），
    # 2026-07-23討論：mkt 同時有 up/down 兩種訊號，共識比對不能把兩個方向
    # 混在一起排名——「A策略看漲、B策略看跌」不是共識，是分歧，一定要同方向
    # 才算「多個模型都看好同一件事」。
    top_by_direction: dict[str, dict[str, list[dict]]] = {"up": {}, "down": {}}
    # 多空衝突（反轉）訊號用：每支股票分別記錄「目前看過的所有策略」裡最高
    # 的多方信心度、最高的空方信心度（不限前N名，全部推論結果都算——見
    # main/config.py::CONFLICT_THRESHOLD 的說明）。等所有策略跑完，兩邊都
    # 超過門檻的股票，代表模型之間對這支股票方向嚴重分歧。
    stock_best_by_direction: dict[str, dict[str, dict]] = {}
    # 同一分鐘內，如果兩個策略共用同一個 predict_live 函式「且」同一個 model
    # 物件（例如 mkt_up/mkt_down，見 strategy/mkt/up、strategy/mkt/down/live.py
    # ——同一顆3分類模型，只是各自濾不同方向），代表算出來的東西完全一樣，
    # 只是後面篩的方向不同，不用重跑一次推論。key 用 (predict_live函式本身,
    # model物件) 的 id：orb_xgb/orb_lgbm 這種真的不同模型的，model物件不同、
    # id不同，不會被誤判成可以共用（見 strategy/mkt/train.py::load_model_by_type()
    # 的說明——那邊加了快取讓同一個 model_type 只從磁碟讀一次、回傳同一個
    # 物件，這裡的判斷才會生效）。快取只在這個 on_minute() 呼叫內有效，下一
    # 分鐘重新算，不用煩惱資料過期。
    _infer_cache: dict[tuple[int, int], list[dict]] = {}
    # 盤中重啟時 db/m1_live/ 可能有缺口，collector 的 _backfill_intraday()
    # 補完前先跳過整段推論（2026-07-25討論）——資料不完整時貿然推論，算出來
    # 的機率不可靠。K線/報價（上面已經推送）不受影響照常更新；既有持倉的
    # SL/TP 監控（下面 reconcile()）也不受影響，只有「這一分鐘要不要產生
    # 新訊號」被擋住。見 main/state.py::AppState.backfill_done 的說明。
    _backfill_pending = not state.backfill_done.is_set()
    if _backfill_pending:
        print(f"[on_minute] {minute_str[11:16]} 分K缺口還沒補完，這一分鐘先跳過推論", flush=True)

    # 即時 tick 資料（fubon/tick_ws.py 收集，見 data/query.py::load_tick_live()
    # 的說明）：只有真的有策略需要（module.USES_TICKS，見 main/state.py::
    # StrategyState.uses_ticks）才讀檔組字典，避免沒人要用就白白讀
    # db/tick_live/。目前沒有任何策略啟用 USES_TICKS，這段實際上不會執行。
    ticks_by_stock = None
    if any(s.uses_ticks for s in state.strategies.values()):
        from data.query import load_tick_live

        tick_live_df = load_tick_live(date_str)
        if not tick_live_df.empty:
            ticks_by_stock = {sid: g for sid, g in tick_live_df.groupby("stock_id")}

    for s in state.strategies.values():
        if (h, m) < s.session_start or (h, m) > s.session_end:
            continue  # 這個策略還沒開始/已經過了進場窗口，不產生新訊號（既有持倉SL/TP由下面reconcile統一監控，不受這裡影響）
        if _backfill_pending:
            continue

        # 每分鐘從 settings 讀信心度，允許前端即時調整；沒設定才 fallback 該
        # 策略自己的預設值（見 main/state.py::StrategyState 的說明——三個
        # 模型的機率校準不一定一樣，2026-07-21 從全域共用一個門檻拆成各策略
        # 各自一個）。
        threshold = float(get_setting("threshold") or s.threshold)

        cache_key = (id(s.predict_live), id(s.model))
        if cache_key in _infer_cache:
            raw_results = _infer_cache[cache_key]
        else:
            predict_kwargs = dict(
                model=s.model,
                threshold=0,
                day_trade_stocks=state.day_trade_stocks,
                m1_live=m1_live if not m1_live.empty else None,
                **s.prewarm_cache,
            )
            # 只有宣告 USES_TICKS 的策略才傳 ticks_by_stock——orb/mkt/cnn/
            # vwap_ml/vwap_dl 的 predict_live() 沒有 **kwargs，無條件塞這個
            # 參數會直接 TypeError（見 main/state.py::StrategyState.uses_ticks
            # 的說明）。
            if s.uses_ticks:
                predict_kwargs["ticks_by_stock"] = ticks_by_stock
            raw_results = s.predict_live(minute_str, state.day, **predict_kwargs)
            _infer_cache[cache_key] = raw_results

        # 只保留這個策略設定允許的方向（module.DIRECTIONS，見 main/state.py
        # 的說明）——orb/rally 目前只送做多，即使哪天 predict_live() 不小心
        # 也算出跌訊號，這裡會直接濾掉，不會送到前端。沒有 direction 欄位
        # 的（orb/rally 現況）視為 "up"。dict(r) 淺拷貝一份，下面對 all_results
        # 逐筆補欄位時才不會動到快取原始資料/另一個策略正在用的那份。
        all_results = [dict(r) for r in raw_results if r.get("direction", "up") in s.directions]
        for r in all_results:
            r["direction"] = r.get("direction", "up")
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
        for direction in ("up", "down"):
            # 信心度沒到 CONSENSUS_MIN_PROBA 就不列入共識比對（2026-08-04要求），
            # 避免排名前面但信心度其實很低的股票被誤判成共識。
            dir_results = [
                r for r in all_results if r["direction"] == direction and r["proba"] >= CONSENSUS_MIN_PROBA
            ]
            if dir_results:
                top_by_direction[direction][s.name] = sorted(dir_results, key=lambda x: -x["proba"])[:CONSENSUS_TOP_N]
        for r in all_results:
            slot = stock_best_by_direction.setdefault(r["stock_id"], {})
            best = slot.get(r["direction"])
            if best is None or r["proba"] > best["proba"]:
                slot[r["direction"]] = r
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
    # main/config.py 的 CONSENSUS_TOP_N）。2個以上策略同時看好同一支股票、
    # 且方向相同，才算共識，跟各策略自己的 monitoring/signals 分開推送，
    # 不取代它們。up/down 分開跑，各自可能各自產生一批共識（見上面
    # top_by_direction 的說明）。
    for direction, top_by_strategy in top_by_direction.items():
        if len(top_by_strategy) < 2:
            continue
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
                "direction": direction,
                "strategies": sorted(probas.keys()),
                "probas": probas,
            }
            for sid, probas in stock_hits.items()
            if len(probas) >= 2
        ]
        if consensus:
            consensus.sort(key=lambda c: -sum(c["probas"].values()) / len(c["probas"]))
            push_consensus_signals(minute_str, consensus)
            print(
                f"  [共識-{direction}] {len(consensus)} 支同時進入 {CONSENSUS_TOP_N} 名內: "
                + " ".join(f"{c['stock_id']}({'+'.join(c['strategies'])})" for c in consensus),
                flush=True,
            )

    # 多空衝突（反轉）訊號：同一支股票，多方最高信心度、空方最高信心度都
    # 超過門檻（見 main/config.py::CONFLICT_THRESHOLD），代表模型之間對這支
    # 股票方向嚴重分歧，跟上面「共識」（同方向）邏輯相反，分開推送，不取代。
    conflicts = []
    for sid, best in stock_best_by_direction.items():
        up_r, down_r = best.get("up"), best.get("down")
        if up_r and down_r and up_r["proba"] >= CONFLICT_THRESHOLD and down_r["proba"] >= CONFLICT_THRESHOLD:
            conflicts.append(
                {
                    "stock_id": sid,
                    "name": up_r["name"],
                    "up_proba": up_r["proba"],
                    "up_strategy": up_r["strategy"],
                    "down_proba": down_r["proba"],
                    "down_strategy": down_r["strategy"],
                }
            )
    if conflicts:
        conflicts.sort(key=lambda c: -(c["up_proba"] + c["down_proba"]))
        push_conflict_signals(minute_str, conflicts)
        print(
            f"  [多空衝突] {len(conflicts)} 支同時滿足多空門檻（≥{CONFLICT_THRESHOLD:.0%}）: "
            + " ".join(f"{c['stock_id']}(多{c['up_proba']:.2f}/空{c['down_proba']:.2f})" for c in conflicts),
            flush=True,
        )

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


def _run_collector_after_startup():
    """等 _startup() 完全跑完才啟動 collector（2026-07-25發現的crash）：
    _startup() 最後一步 run_startup_backfill() 會在自己這個背景執行緒裡呼叫
    predict_live()（XGBoost/LightGBM 推論，底層用 OpenMP 平行運算）；collector
    一開機就自己開始跑、on_minute() 每分鐘固定觸發也會呼叫 predict_live()。
    _startup() 因為 refresh_tickers() 要跑好幾分鐘，這段時間如果 collector
    已經自己觸發過 on_minute()，兩個背景執行緒會同時呼叫 XGBoost/LightGBM
    推論——OpenMP 的執行緒池初始化不是完全執行緒安全的，實測會讓整個
    process 直接 SIGSEGV（macOS crash log 證實：faulting thread 在
    libomp.dylib 的 __kmp_launch_worker，不是 Python 例外、不會印
    traceback，看起來像「安靜地自動結束」）。等 _startup_done 之後才讓
    collector 開始，確保兩邊不會同時摸到同一個 model。
    """
    _startup_done.wait()

    def _on_token_ready(token: str):
        state.fubon_ws_token = token
        state.fubon_ws_token_ready.set()

    _collector.start_collector(on_minute, backfill_done=state.backfill_done, on_token_ready=_on_token_ready)


def _run_tick_collector_after_startup():
    """成交明細（tick）收集器，等分K collector 先登入拿到 realtime_token()
    才開始（見 state.fubon_ws_token_ready、main/collector.py::
    start_tick_collector() 的說明）——全程只用富邦帳號登入一次，重用同一個
    token 開 tick 訂閱連線，不再另外呼叫一次富邦登入。"""
    _startup_done.wait()
    _collector.start_tick_collector(state.fubon_ws_token_ready, lambda: state.fubon_ws_token)


if __name__ == "__main__":
    # 開機序列（模型/當沖清單/日K/盤前快取/交易引擎）背景執行緒，讓 uvicorn
    # 能立刻起來接受連線，不用等這些跑完（見 _startup() 的說明）。
    threading.Thread(target=_startup, daemon=True).start()
    # 每日 08:45 更新排程
    threading.Thread(target=_daily_refresh, daemon=True).start()
    # 每日 13:25 強制平倉（當沖不過夜）
    threading.Thread(target=_force_close_eod, daemon=True).start()
    # 分K收集器在背景執行緒，等 _startup() 完全跑完才開始（見
    # _run_collector_after_startup() 的說明，避免跟 _startup() 同時呼叫模型
    # 推論導致 OpenMP crash）。
    threading.Thread(target=_run_collector_after_startup, daemon=True).start()
    # 成交明細（tick）收集器：等分K collector 登入完成、拿到 token 後才開始，
    # 全程只登入富邦帳號一次（見 _run_tick_collector_after_startup() 的說明）。
    threading.Thread(target=_run_tick_collector_after_startup, daemon=True).start()

    # uvicorn 跑主執行緒（阻塞）
    config = get_uvicorn_config(host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    server.run()
