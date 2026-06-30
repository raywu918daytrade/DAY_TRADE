"""
即時交易進入點（Render web service）

架構：
    主執行緒  → uvicorn（FastAPI + WebSocket）
    背景執行緒 → M1Collector（Fugle WebSocket，阻塞式）

流程：
    on_minute → predict_live → push_signals → API /signals/today
                             → push_candles  → API /chart/{id}/candles
"""

import threading
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import uvicorn

from api import get_uvicorn_config, push_candles, push_signals, set_collector_status
from date_trade_model import SESSION_END, SESSION_START, load_model, predict_live
import os

from tay_trade.fugle_tickers import update_tickers
from tay_trade.m1_websocket import M1Collector
from tay_trade.query import load_day, load_m1_live

_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")   # 設在 Render 環境變數


def _load_day_from_hf() -> "pd.DataFrame":
    """從 HF Hub 下載 fugle_day.parquet，存到本地後用 load_day() 載入"""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    print(f"從 HF Hub 下載日K：{_HF_REPO_ID}...")
    local_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename="fugle_day.parquet",
        repo_type="dataset",
        token=None,   # public repo
    )
    df = pd.read_parquet(local_path)
    print(f"  {len(df):,} 筆，{df['stock_id'].nunique():,} 支")
    return df

_TW = timezone(timedelta(hours=8))
THRESHOLD = 0.55

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
print(f"  當沖標的：{len(_tickers)} 支")

# HF_REPO_ID 有設定 → 從 HF 下載（Render）；否則用本地（本機開發）
if _HF_REPO_ID:
    _day = _load_day_from_hf()
else:
    _day = load_day()

print(f"就緒，等待盤中訊號（門檻={THRESHOLD}）...")


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
                last_refresh = today
                print(f"  更新完成，當沖標的：{len(_day_trade_stocks)} 支")
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
            candles = [
                {
                    "time": str(row["date"])[11:16],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
                for _, row in g.iterrows()
            ]
            push_candles(str(sid), candles)

    # 模型推論 → 推入訊號
    signals = predict_live(
        minute_str, _day,
        model=model,
        threshold=THRESHOLD,
        day_trade_stocks=_day_trade_stocks,
    )
    # 補上股票名稱
    for s in signals:
        s["name"] = _tickers.get(s["stock_id"], s["stock_id"])
    push_signals(minute_str, signals)

    if signals:
        print(f"\n[{minute_str}] 訊號（{len(signals)} 支）:")
        for s in signals:
            print(f"  {s['stock_id']:8s}  機率={s['proba']:.3f}  價={s['price']:.2f}")
    else:
        print(f"[{minute_str}] 無訊號")


def _start_collector():
    collector = M1Collector(on_minute=on_minute)
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
