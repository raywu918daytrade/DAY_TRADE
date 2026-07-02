"""
即時交易進入點（Render web service）

架構：
    主執行緒  → uvicorn（FastAPI + WebSocket）
    背景執行緒 → M1Collector（Fugle WebSocket，阻塞式）

流程：
    on_minute → predict_live → push_signals → API /signals/today
                             → push_candles  → API /chart/{id}/candles
"""

import calendar
import threading
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import uvicorn

from api import get_uvicorn_config, push_candles, push_monitoring, push_signals, set_collector_status, update_positions_price
from date_trade_model import SESSION_END, SESSION_START, load_model, predict_live
import os

from tay_trade.fugle_tickers import update_tickers
from tay_trade.m1_rest import M1RestPoller
from tay_trade.query import load_day, load_m1_live

_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")   # 設在 Render 環境變數

TRADE_MODE = os.environ.get("TRADE_MODE", "off")      # off | paper | sim | live
TOTAL_CAPITAL = float(os.environ.get("TOTAL_CAPITAL", "1000000"))


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
    cutoff = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    df = df[df["date"] >= cutoff].copy()
    print(f"  {len(df):,} 筆，{df['stock_id'].nunique():,} 支（最近 60 天）")
    return df

_TW = timezone(timedelta(hours=8))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.55"))
MIN_AVG_VOL_LOTS = int(os.environ.get("MIN_AVG_VOL_LOTS", "1000"))  # 20日均量門檻（張）
MAX_SUBSCRIPTIONS = int(os.environ.get("MAX_SUBSCRIPTIONS", "500"))  # Fugle 方案訂閱上限


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
model = load_model()

print("更新當沖標的清單...")
_tickers_df = update_tickers()
if _tickers_df.empty:
    print("  警告：無法取得當沖標的（非盤中），不過濾股票")
    _tickers = {}
else:
    _tickers = _tickers_df.set_index("stock_id")["name"].to_dict()
_day_trade_stocks = set(_tickers.keys()) or None   # None = 不過濾
print(f"  當沖標的（API）：{len(_tickers)} 支")

# HF_REPO_ID 有設定 → 從 HF 下載（Render）；否則用本地（本機開發）
if _HF_REPO_ID:
    _day = _load_day_from_hf()
else:
    _day = load_day()

# 條件③：20日均量過濾
if _day_trade_stocks:
    _day_trade_stocks = _volume_filter(_day_trade_stocks, _day)

print(f"就緒，等待盤中訊號（門檻={THRESHOLD}）...")

_executor = None
if TRADE_MODE != "off":
    try:
        from trade.run_execute import make_executor
        _executor = make_executor(TRADE_MODE, TOTAL_CAPITAL)
        print(f"交易模式：{TRADE_MODE}，資金={TOTAL_CAPITAL:,.0f}")
    except Exception as e:
        print(f"[WARN] 交易模組載入失敗，改為僅推訊號: {e}")


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
                candles.append({
                    "time": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                })
            push_candles(str(sid), candles)

    # 模型推論：threshold=0 取得所有股票機率，供監控面板顯示
    all_results = predict_live(
        minute_str, _day,
        model=model,
        threshold=0,
        day_trade_stocks=_day_trade_stocks,
    )
    for r in all_results:
        r["name"] = _tickers.get(r["stock_id"], r["stock_id"])
    push_monitoring(minute_str, all_results, THRESHOLD)

    # 用最新分K收盤價更新持倉卡片浮動損益（今日損益只計已實現，不含此處）
    price_map = {r["stock_id"]: r["price"] for r in all_results}
    update_positions_price(price_map)

    # 只有達門檻才產生交易訊號
    signals = [r for r in all_results if r["proba"] >= THRESHOLD]
    push_signals(minute_str, signals)

    print(f"[{minute_str}] 推論 {len(all_results)} 支，信心度: " +
          " ".join(f"{r['stock_id']}={r['proba']:.2f}" for r in all_results))
    if signals:
        print(f"  → 訊號（{len(signals)} 支）: " +
              " ".join(f"{s['stock_id']} {s['proba']:.2f}" for s in signals))

    if _executor is not None:
        try:
            _executor.reconcile(signals)
        except Exception as e:
            print(f"[TRADE ERROR] {e}")


def _get_stocks():
    """每次重連都取最新當沖標的；非盤中無清單時回退到所有可交易股票"""
    from tay_trade.fugle_tickers import fugle_stocks
    return list(_day_trade_stocks) if _day_trade_stocks else fugle_stocks()


def _start_collector():
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
    # M1Collector 在背景執行緒
    threading.Thread(target=_start_collector, daemon=True).start()

    # uvicorn 跑主執行緒（阻塞）
    config = get_uvicorn_config(host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    server.run()
