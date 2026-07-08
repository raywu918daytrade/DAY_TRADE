"""
即時交易進入點（Render web service）

架構：
    主執行緒  → uvicorn（FastAPI + SSE）
    背景執行緒 → M1RestPoller（Fugle REST API，每分鐘 poll）
    背景執行緒 → _daily_refresh（每天 06:00 更新當沖標的 + 日K）
    背景執行緒 → _force_close_eod（每天 13:25 強制平倉，當沖不過夜）

資料取用時段：
    盤前（<09:01）  : 日K 從 HF Hub / 本地載入；分K 從既有 parquet 讀取（昨日 backfill）
    盤中（09:01~13:00）: Fugle REST /intraday/candles 每分鐘 poll，寫入 db/m1_live/
    盤後（>13:00）  : Fugle REST /historical/candles 補齊完整今日分K，再 sleep 到隔日

流程：
    on_minute → predict_live → push_monitoring → SSE monitoring event
                             → push_signals    → API /signals/today
                             → push_candles    → API /chart/{id}/candles
                             → reconcile       → 永豐 / paper 下單
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
    set_collector_status,
    tw_naive_to_epoch,
    update_positions_price,
)
from strategy.base.date_trade_model import SESSION_END, SESSION_START, load_model, predict_live
import os

from data.fugle_tickers import update_tickers
from data.m1_rest import M1RestPoller
from data.query import load_m1_live
from data.data_manager import Phase, load_d1  # noqa: F401 Phase 供外部 import 參考

TRADE_MODE = os.environ.get("TRADE_MODE", "off")  # off | paper | sim | live
_TOTAL_CAPITAL_ENV = float(os.environ.get("TOTAL_CAPITAL", "1000000"))


# settings.json 存在且有 total_capital → 優先使用，否則回落 .env
def _resolve_capital() -> float:
    # api.py 已初始化 _settings_cache（含 HF Hub fallback），直接用 get_setting
    try:
        from api import get_setting

        v = get_setting("total_capital")
        if v is not None:
            print(f"[設定] 總額度從 settings 載入：{float(v):,.0f}")
            return float(v)
    except Exception:
        pass
    return _TOTAL_CAPITAL_ENV


TOTAL_CAPITAL = _resolve_capital()


_TW = timezone(timedelta(hours=8))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.55"))
_FORCE_CLOSE_HOUR = int(os.environ.get("FORCE_CLOSE_HOUR", "13"))
_FORCE_CLOSE_MIN = int(os.environ.get("FORCE_CLOSE_MIN", "25"))


print("載入模型...")
sys.stdout.flush()
try:
    model = load_model()
    print("✓ 模型載入成功", flush=True)
except Exception as e:
    print(f"✗ 模型載入失敗: {e}", flush=True)
    raise

print("更新當沖標的清單...", flush=True)
try:
    _tickers_df = update_tickers()
    if _tickers_df.empty:
        print("  警告：無法取得當沖標的（非盤中），不過濾股票", flush=True)
        _tickers = {}
    else:
        _tickers = _tickers_df.set_index("stock_id")["name"].to_dict()
    _day_trade_stocks = set(_tickers.keys()) or None  # None = 不過濾
    print(f"  當沖標的（API）：{len(_tickers)} 支", flush=True)
except Exception as e:
    print(f"✗ 取得當沖標的失敗: {e}", flush=True)
    _tickers = {}
    _day_trade_stocks = None

# 盤中才有標的清單，非盤中跳過日K載入（省記憶體）；_daily_refresh 在 06:00 補載
if _day_trade_stocks:
    print(f"[D1] 載入日K（{Phase.PRE_MARKET.value}）...", flush=True)
    try:
        _day, _day_trade_stocks = load_d1(_day_trade_stocks)
        _d1_last = _day["date"].max() if not _day.empty else "無"
        print(f"✓ [D1] {len(_day):,} 筆，{_day['stock_id'].nunique():,} 支，最新日期：{_d1_last}", flush=True)
        _log_sys(f"D1 載入完成：{_day['stock_id'].nunique() if not _day.empty else 0} 支，最新 {_d1_last}")
    except Exception as e:
        print(f"✗ [D1] 載入失敗: {e}", flush=True)
        _log_sys(f"D1 載入失敗: {e}", "error")
        _day = pd.DataFrame()
else:
    _day = pd.DataFrame()
    print("[D1] 非盤中，跳過日K載入（_daily_refresh 06:00 更新）", flush=True)

# 啟動補載：若今日已有 m1_live，立刻跑推論填 _monitoring（不用等下一分鐘）
if not _day.empty:
    try:
        _today_str = datetime.now(_TW).strftime("%Y-%m-%d")
        print(f"[M1] 補載今日分K（{_today_str}）...", flush=True)
        _m1_now = load_m1_live(_today_str)
        if not _m1_now.empty:
            _last_min = str(_m1_now["date"].max())
            print(f"  → {len(_m1_now):,} 筆，{_m1_now['stock_id'].nunique():,} 支，最新分鐘：{_last_min}", flush=True)
            _init_results = predict_live(
                _last_min,
                _day,
                day_trade_stocks=_day_trade_stocks,
                m1_live=_m1_now,
            )
            push_monitoring(_last_min, _init_results, THRESHOLD)
            print(f"✓ [M1] 補載監控完成：{len(_init_results)} 支訊號", flush=True)
        else:
            print("  → 今日無分K資料（尚未開盤或非交易日）", flush=True)
        del _m1_now
    except Exception as _e:
        print(f"✗ [M1] 補載失敗: {_e}", flush=True)

print(f"就緒，等待盤中訊號（門檻={THRESHOLD}）...", flush=True)

_executor = None
if TRADE_MODE != "off":
    try:
        from trade.run_execute import make_executor

        _executor = make_executor(
            TRADE_MODE,
            TOTAL_CAPITAL,
            name_lookup=lambda sid: _tickers.get(sid, sid),
        )
        print(f"交易模式：{TRADE_MODE}，資金={TOTAL_CAPITAL:,.0f}", flush=True)
        _log_sys(f"交易引擎啟動：{TRADE_MODE} 模式，資金={TOTAL_CAPITAL:,.0f}")
        if hasattr(_executor, "sync_from_broker"):
            _executor.sync_from_broker()
        if hasattr(_executor, "startup_sltp_check"):
            _executor.startup_sltp_check()
        if hasattr(_executor, "close_stock_now"):
            from api import register_close_now

            register_close_now(_executor.close_stock_now)
    except Exception as e:
        print(f"[WARN] 交易模組載入失敗，改為僅推訊號: {e}", flush=True)
        _log_sys(f"交易模組載入失敗: {e}", "error")


def _daily_refresh():
    """每天 06:00 更新當沖清單與日K（Render 24小時常駐用）"""
    global _tickers, _day_trade_stocks, _day
    last_refresh = None
    while True:
        now = datetime.now(_TW)
        today = now.date()
        need_refresh = last_refresh != today and now.hour == 6 and now.minute >= 0
        # 啟動時若日K是空的（非盤中跳過）且已過 06:00，立即補載
        need_refresh = need_refresh or (last_refresh is None and _day.empty and now.hour >= 6)
        if need_refresh:
            print(f"[{now.strftime('%H:%M')}] 每日更新：當沖標的 + 日K...")
            try:
                df = update_tickers()
                if not df.empty:
                    _tickers = df.set_index("stock_id")["name"].to_dict()
                    _day_trade_stocks = set(_tickers.keys()) or None
                if _day_trade_stocks:
                    _day, _day_trade_stocks = load_d1(_day_trade_stocks)
                last_refresh = today
                n = len(_day_trade_stocks) if _day_trade_stocks else 0
                print(f"  更新完成，當沖標的：{n} 支")
                _log_sys(f"每日更新完成（{Phase.PRE_MARKET.value}）：{n} 支標的")
            except Exception as e:
                print(f"  更新失敗: {e}")
        time.sleep(60)


def on_minute(minute_str: str, df: pd.DataFrame):
    """M1RestPoller 每分鐘回呼：推論、推送監控/K線/訊號，並觸發下單。"""
    dt = pd.Timestamp(minute_str)
    h, m = dt.hour, dt.minute

    if (h, m) < SESSION_START:
        return

    # SESSION_END 後：停止開倉，但繼續跑 reconcile 做 SL/TP 監控直到收盤
    if (h, m) > SESSION_END:
        if _executor is not None:
            price_map = {}
            if not df.empty and "stock_id" in df.columns and "close" in df.columns:
                price_map = dict(zip(df["stock_id"].astype(str), df["close"].astype(float)))
                update_positions_price(price_map)
            try:
                _executor.reconcile([], prices=price_map)  # signals=[] 不開倉，只跑 SL/TP
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
    all_results = predict_live(
        minute_str,
        _day,
        model=model,
        threshold=0,
        day_trade_stocks=_day_trade_stocks,
        m1_live=m1_live if not m1_live.empty else None,
    )
    for r in all_results:
        r["name"] = _tickers.get(r["stock_id"], r["stock_id"])
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

    if _executor is not None:
        try:
            _executor.reconcile(signals, prices=price_map)
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
        if (now.hour, now.minute) == (fc_h, fc_m) and _force_close_done_date != today and _executor is not None:
            _force_close_done_date = today
            print(f"[{now.strftime('%H:%M:%S')}] 強制平倉：本系統當沖倉位（{fc_str}）", flush=True)
            try:
                _executor.force_close_own_positions()
            except Exception as e:
                print(f"[FORCE CLOSE] 錯誤: {e}", flush=True)
            time.sleep(10)
            if hasattr(_executor, "sync_from_broker"):
                try:
                    _executor.sync_from_broker()
                    print(f"[FORCE CLOSE] 盤後同步完成", flush=True)
                except Exception as e:
                    print(f"[FORCE CLOSE] 盤後同步失敗: {e}", flush=True)
        time.sleep(30)


def _get_stocks():
    """每次重連都取最新當沖標的；非盤中無清單時回退到所有可交易股票

    固定加入 0050：它本身不是當沖候選股，但 rally 策略的 idx_* 特徵
    （大盤 1分K 相對強弱）需要它當天的即時分K，若當沖候選清單剛好沒選到
    0050，db/m1_live/ 就不會有它的資料，predict_live() 算 idx_* 特徵時
    會直接 KeyError。
    """
    from data.fugle_tickers import fugle_stocks

    stocks = list(_day_trade_stocks) if _day_trade_stocks else fugle_stocks()
    if "0050" not in stocks:
        stocks.append("0050")
    return stocks


def _on_rate_limited():
    """Fugle 429 時推送前端警示。"""
    try:
        from api import push_alert

        push_alert("Fugle REST API 限流，使用快取分K繼續監控（SL/TP 仍有效）", level="warning")
    except Exception:
        pass


def _start_collector():
    """M1RestPoller 收集器執行緒，異常時更新 collector 狀態供 /health 回傳。"""
    collector = M1RestPoller(on_minute=on_minute, stocks=_get_stocks, on_rate_limited=_on_rate_limited)
    try:
        set_collector_status("running")
        collector.start()
    except Exception as e:
        set_collector_status("error")
        print(f"Collector 中斷: {e}")
    else:
        set_collector_status("stopped")


if __name__ == "__main__":
    # 每日 08:45 更新排程
    threading.Thread(target=_daily_refresh, daemon=True).start()
    # 每日 13:25 強制平倉（當沖不過夜）
    threading.Thread(target=_force_close_eod, daemon=True).start()
    # M1RestPoller 在背景執行緒
    threading.Thread(target=_start_collector, daemon=True).start()

    # uvicorn 跑主執行緒（阻塞）
    config = get_uvicorn_config(host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    server.run()
