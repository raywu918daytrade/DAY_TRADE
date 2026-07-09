"""
即時推論 — predict()（批次機率矩陣）、predict_live()（正式即時推論入口）

breakout_signal 硬過濾（use_breakout_filter）預設關閉，經驗證會拉低勝率，
細節見 predict_live() 的 docstring 跟 experiments/breakout_filter_eval.py。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from data.query import load_day, load_m1_live
from strategy.rally.features import FEATURES, compute_m3, compute_m5, load_features, make_features
from strategy.rally.train import load_model


def predict(
    model=None,
    breakout_filter: bool = False,
    test_days: int = 10,
    test_only: bool = True,
) -> pd.DataFrame:
    """
    對全天資料產生預測機率矩陣（index=datetime, columns=stock_id）。

    不再限制早盤時段——模型有 minutes_since_open 特徵自己判斷時段，
    要進場時段的限制交給呼叫端（例如 backtest/intraday_backtest.py 的
    first_entry_time/last_entry_time）。

    test_only=True（預設）時，只回傳最後 test_days 天的資料——跟
    train.py 切分訓練/測試集用的是同一個 cutoff 邏輯，這幾天是模型訓練時
    沒看過的樣本外資料。拿全部資料（含訓練集）去跑回測，績效會被模型
    「背過答案」的部分灌水，沒有參考價值。test_only=False 才會回傳全部。

    breakout_filter=True 時，只保留 breakout_signal=True 的樣本
    （強過濾：先跌後漲破底翻才納入預測）。
    """
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES)

    if breakout_filter:
        df = df[df["breakout_signal"]]

    if test_only:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        df = df[df["date"] > cutoff]

    df["proba"] = model.predict_proba(df[FEATURES])[:, 1]
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    model=None,
    threshold: float = 0.55,
    use_breakout_filter: bool = False,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論（可選 breakout_signal 硬過濾，預設關閉）。

    day: 已載入的日K，傳入可避免每次呼叫都重讀整個 db/fugle_day/（live_trader.py
        開機時載一次、_daily_refresh() 每天 06:00 更新，呼叫端應該傳這份快取）。
        留空則沿用舊行為，內部自己 load_day()。
    day_trade_stocks: 當沖標的 set，若提供則只推論這些股票。

    use_breakout_filter 預設 False（2026-07-09 改的）：`experiments/
    breakout_filter_eval.py` 用洩漏修正後的模型重測過，breakout_signal
    當硬過濾**會拉低勝率**（XGB/LGBM 加了這道規則後勝率各掉 8-10 個百分點，
    breakout=True 單獨看的勝率 34.5% 甚至低於全體基準 37.2%）。
    breakout_signal 現在只是模型的普通輸入之一（而且重要性只有 0.12%，
    幾乎沒用），不是可靠的獨立訊號，不該再拿來當硬性篩選規則。
    若要開回去，設 use_breakout_filter=True，只有 breakout_signal=True 的
    樣本才會進入模型預測。

    交易時段限制交給呼叫端（跟 SESSION_START/END 比對），這裡不再額外用
    BREAKOUT_TRADE_START/END 鎖死 9:14~9:30 黃金窗口。

    回傳格式：[{"stock_id": ..., "proba": ..., "price": ..., "breakout": ...}, ...]
    """
    if model is None:
        model = load_model()

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    # 只保留當沖標的
    if day_trade_stocks:
        m1_live = m1_live[m1_live["stock_id"].isin(day_trade_stocks)]
    if m1_live.empty:
        return []

    # make_features 以 day_date 做 merge；今日 day 資料尚不存在，補一行今日摘要
    if day is None:
        day = load_day()
    day = day.copy()
    day["date"] = pd.to_datetime(day["date"])
    today_ts = pd.Timestamp(date_str)
    if not (day["date"] == today_ts).any():
        rows = []
        for sid, g in m1_live.groupby("stock_id"):
            g_s = g.sort_values("date")
            rows.append(
                {
                    "stock_id": sid,
                    "date": today_ts,
                    "open": float(g_s.iloc[0]["open"]),
                    "high": float(g["high"].max()),
                    "low": float(g["low"].min()),
                    "close": float(g_s.iloc[-1]["close"]),
                    "volume": int(g["volume"].sum()),
                }
            )
        if rows:
            day = pd.concat([day, pd.DataFrame(rows)], ignore_index=True)
            day["date"] = pd.to_datetime(day["date"])
            day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # m3/m5 現算：db/m3、db/m5 是批次預算（scripts/build_m3_m5.py 從歷史
    # db/m1/ 算好存檔），不包含「今天」的資料。make_features() 若沒收到
    # m3/m5 會 fallback 去讀那兩個批次 cache，merge 會完全對不上、
    # m3_*/m5_*（連帶靠 m5_ret 算出來的 breakout_signal）整批變 NaN。
    # 即時推論資料量小（單日、當沖候選清單），現算成本可忽略，直接對
    # m1_live 呼叫跟批次同一套邏輯，保證跟訓練用的公式一致。
    # 排序/型別跟 make_features() 內部對 m1 做的預處理完全一致，確保
    # m3_live/m5_live 的 index 跟 make_features() 重新整理過的 m1 對得上
    # （後面 m3["open"] / day_open 是用 index 對齊，不是 merge）。
    m1_live_sorted = m1_live.copy()
    m1_live_sorted["date"] = pd.to_datetime(m1_live_sorted["date"])
    m1_live_sorted = m1_live_sorted.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1_live_sorted["day_date"] = m1_live_sorted["date"].dt.date
    m3_live = compute_m3(m1_live_sorted)
    m5_live = compute_m5(m1_live_sorted)

    df = make_features(m1_live, m3=m3_live, m5=m5_live, day=day, compute_labels=False)
    current = df[df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []

    valid = current.dropna(subset=FEATURES)
    if valid.empty:
        return []

    # ── 強過濾：硬規則（預設強制開啟）─────────────────────────────
    if use_breakout_filter:
        valid = valid[valid["breakout_signal"]]
        if valid.empty:
            return []

    proba = model.predict_proba(valid[FEATURES])[:, 1]
    signals = [
        {
            "stock_id": row["stock_id"],
            "proba": float(p),
            "price": float(row["close"]),
            "breakout": bool(row["breakout_signal"]),
        }
        for (_, row), p in zip(valid.iterrows(), proba)
        if p >= threshold
    ]
    return sorted(signals, key=lambda x: -x["proba"])
