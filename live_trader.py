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

import calendar
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import uvicorn

from api import (
    get_uvicorn_config,
    push_candles,
    push_monitoring,
    push_signals,
    set_collector_status,
    update_positions_price,
)
from strategy.date_trade_model import SESSION_END, SESSION_START, load_model, predict_live
import os

from data.fugle_tickers import update_tickers
from data.m1_rest import M1RestPoller
from data.query import load_day, load_m1_live

_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")  # 設在 Render 環境變數

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


def _load_day_from_hf() -> "pd.DataFrame":
    """從 HF Hub 下載 fugle_day.parquet，存到本地後用 load_day() 載入"""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    print(f"從 HF Hub 下載日K：{_HF_REPO_ID}...")
    local_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename="day_trade/day/fugle_day.parquet",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN") or None,
    )
    df = pd.read_parquet(local_path)
    # live trading 只需要最近 60 天日K特徵，節省記憶體
    cutoff = (pd.Timestamp.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    df = df[df["date"] >= cutoff].copy()
    print(f"  {len(df):,} 筆，{df['stock_id'].nunique():,} 支（最近 60 天）")
    return df


_TW = timezone(timedelta(hours=8))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.55"))
MIN_AVG_VOL_LOTS = int(os.environ.get("MIN_AVG_VOL_LOTS", "1000"))  # 20日均量門檻（張）
MAX_SUBSCRIPTIONS = int(os.environ.get("MAX_SUBSCRIPTIONS", "500"))  # Fugle 方案訂閱上限
_FORCE_CLOSE_HOUR = int(os.environ.get("FORCE_CLOSE_HOUR", "13"))
_FORCE_CLOSE_MIN = int(os.environ.get("FORCE_CLOSE_MIN", "25"))


def _volume_filter(stocks: set, day: "pd.DataFrame") -> list:
    """20日均量過濾，回傳按均量排序（高→低）的 stock_id list，最多 MAX_SUBSCRIPTIONS 支"""
    if day.empty or not stocks:
        result = list(stocks)[:MAX_SUBSCRIPTIONS]
        print(f"  均量過濾：無日K，取前 {len(result)} 支")
        return result
    recent = day[day["stock_id"].isin(stocks)].copy()
    recent["date"] = pd.to_datetime(recent["date"])
    last20 = recent.sort_values("date").groupby("stock_id").tail(20)
    avg_vol = last20.groupby("stock_id")["volume"].mean()
    qualified = avg_vol[avg_vol >= MIN_AVG_VOL_LOTS * 1000].sort_values(ascending=False)
    result = list(qualified.index[:MAX_SUBSCRIPTIONS])
    print(f"  均量過濾（≥{MIN_AVG_VOL_LOTS}張）: {len(stocks)} → {len(qualified)} 支 → 訂閱前 {len(result)} 支")
    return result


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

# HF_REPO_ID 有設定 → 從 HF 下載（Render）；否則用本地（本機開發）
print("載入日K資料...", flush=True)
try:
    if _HF_REPO_ID:
        _day = _load_day_from_hf()
    else:
        _day = load_day()
    print(f"✓ 日K 載入完成：{len(_day)} 筆", flush=True)
except Exception as e:
    print(f"✗ 日K 載入失敗: {e}", flush=True)
    _day = pd.DataFrame()

# 條件③：20日均量過濾
if _day_trade_stocks:
    _day_trade_stocks = _volume_filter(_day_trade_stocks, _day)

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
        if hasattr(_executor, "sync_from_broker"):
            _executor.sync_from_broker()
    except Exception as e:
        print(f"[WARN] 交易模組載入失敗，改為僅推訊號: {e}", flush=True)


def _daily_refresh():
    """每天 08:45 更新當沖清單與日K（Render 24小時常駐用）"""
    global _tickers, _day_trade_stocks, _day
    last_refresh = None
    while True:
        now = datetime.now(_TW)
        today = now.date()
        if last_refresh != today and now.hour == 6 and now.minute >= 0:
            print(f"[{now.strftime('%H:%M')}] 每日更新：當沖標的 + 日K...")
            try:
                df = update_tickers()
                if not df.empty:
                    _tickers = df.set_index("stock_id")["name"].to_dict()
                    _day_trade_stocks = set(_tickers.keys()) or None
                _day = _load_day_from_hf() if _HF_REPO_ID else load_day()
                if _day_trade_stocks:
                    _day_trade_stocks = _volume_filter(_day_trade_stocks, _day)
                last_refresh = today
                print(f"  更新完成，當沖標的：{len(_day_trade_stocks) if _day_trade_stocks else 0} 支")
            except Exception as e:
                print(f"  更新失敗: {e}")
        time.sleep(60)


def on_minute(minute_str: str, df: pd.DataFrame):
    """M1RestPoller 每分鐘回呼：推論、推送監控/K線/訊號，並觸發下單。"""
    dt = pd.Timestamp(minute_str)
    h, m = dt.hour, dt.minute

    if not (SESSION_START <= (h, m) <= SESSION_END):
        return

    # 推入 K 線資料（每股今日所有分K，供圖表使用）
    date_str = minute_str[:10]
    m1_live = load_m1_live(date_str)
    if not m1_live.empty:
        for sid, g in m1_live.groupby("stock_id"):
            candles = []
            for _, row in g.iterrows():
                dt = datetime.strptime(str(row["date"]), "%Y-%m-%d %H:%M:%S")
                ts = calendar.timegm(dt.timetuple())  # 以 TW 本地時間當 UTC，圖表顯示正確
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

    # 模型推論：threshold=0 取得所有股票機率，供監控面板顯示
    all_results = predict_live(
        minute_str,
        _day,
        model=model,
        threshold=0,
        day_trade_stocks=_day_trade_stocks,
    )
    for r in all_results:
        r["name"] = _tickers.get(r["stock_id"], r["stock_id"])
    # 每分鐘從 settings 讀信心度，允許前端即時調整
    from api import get_setting

    threshold = float(get_setting("threshold") or THRESHOLD)

    push_monitoring(minute_str, all_results, threshold)

    # 用最新分K收盤價更新持倉卡片浮動損益（今日損益只計已實現，不含此處）
    price_map = {r["stock_id"]: r["price"] for r in all_results}
    update_positions_price(price_map)

    # 只有達門檻才產生交易訊號
    signals = [r for r in all_results if r["proba"] >= threshold]
    push_signals(minute_str, signals)

    print(
        f"[{minute_str}] 推論 {len(all_results)} 支，信心度: "
        + " ".join(f"{r['stock_id']}={r['proba']:.2f}" for r in all_results)
    )
    if signals:
        print(f"  → 訊號（{len(signals)} 支）: " + " ".join(f"{s['stock_id']} {s['proba']:.2f}" for s in signals))

    if _executor is not None:
        try:
            _executor.reconcile(signals)
        except Exception as e:
            print(f"[TRADE ERROR] {e}")


_force_close_done_date = None  # 防止同一天重複觸發


def _force_close_eod():
    """每天 FORCE_CLOSE_HOUR:FORCE_CLOSE_MIN 強制平倉所有當沖部位（不過夜）。
    reconcile([]) 傳空訊號 → 所有現倉視為「不在目標」而被賣出。
    """
    global _force_close_done_date
    while True:
        now = datetime.now(_TW)
        trigger = now.replace(
            hour=_FORCE_CLOSE_HOUR,
            minute=_FORCE_CLOSE_MIN,
            second=0,
            microsecond=0,
        )
        wait = (trigger - now).total_seconds()
        if wait > 0:
            time.sleep(wait)
            now = datetime.now(_TW)
        today = now.date()
        if _force_close_done_date != today and _executor is not None:
            _force_close_done_date = today
            print(
                f"[{now.strftime('%H:%M:%S')}] 強制平倉：本系統當沖倉位（{_FORCE_CLOSE_HOUR}:{_FORCE_CLOSE_MIN:02d}）"
            )
            try:
                _executor.force_close_own_positions()  # 只平本系統追蹤的倉，長期持倉不動
            except Exception as e:
                print(f"[FORCE CLOSE] 錯誤: {e}")
            # 等委託成交後，從永豐同步一次最終狀態（確保 dashboard 與 broker 一致）
            time.sleep(10)
            if hasattr(_executor, "sync_from_broker"):
                try:
                    _executor.sync_from_broker()
                    print(f"[FORCE CLOSE] 盤後同步完成")
                except Exception as e:
                    print(f"[FORCE CLOSE] 盤後同步失敗: {e}")
        # 等到明天同一時間
        next_trigger = trigger + timedelta(days=1)
        time.sleep(max((next_trigger - datetime.now(_TW)).total_seconds(), 60))


def _get_stocks():
    """每次重連都取最新當沖標的；非盤中無清單時回退到所有可交易股票"""
    from data.fugle_tickers import fugle_stocks

    return list(_day_trade_stocks) if _day_trade_stocks else fugle_stocks()


def _start_collector():
    """M1RestPoller 收集器執行緒，異常時更新 collector 狀態供 /health 回傳。"""
    collector = M1RestPoller(on_minute=on_minute, stocks=_get_stocks)
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
