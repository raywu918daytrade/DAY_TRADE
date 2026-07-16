"""
即時推論 — predict()（批次機率矩陣，回測用）、predict_live()（正式即時推論入口）

預設用 LGBM——見 strategy/orb/validate.py 的 compare_report()，同一份測試集上
AUC、精確率門檻表現都比 XGB 好，且 XGB 沒有任何門檻精確率能過50%打平線。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from data.query import load_day, load_m1, load_m1_live
from strategy.orb.config import DEFAULT_TEST_DAYS
from strategy.orb.features import (
    FEATURES,
    apply_liquidity_filter,
    build_history_tables,
    compute_m3,
    compute_m5,
    load_features,
    make_features,
    to_model_input,
)
from strategy.orb.train import load_model_lgbm


def predict(
    model=None,
    test_days: int = DEFAULT_TEST_DAYS,
    test_only: bool = True,
) -> pd.DataFrame:
    """
    對全天資料產生預測機率矩陣（index=datetime, columns=stock_id），回測用。

    test_only=True（預設）時，只回傳最後 test_days 天的資料——這幾天是模型
    訓練時沒看過的樣本外資料，跟 train.py 切分訓練/測試集用同一套 cutoff
    邏輯（見 strategy/orb/train.py 的 _prepare_train_test()）。test_only=False
    才會回傳全部（含訓練集），訓練集算出來的績效會被模型「背過答案」灌水，
    沒有參考價值，只用來除錯。
    """
    if model is None:
        model = load_model_lgbm()

    # cache 新鮮度只比對 test_days 這段窗口涵蓋的月份檔案（見
    # load_features() docstring）：test_only=True 時我們本來就只看最後
    # test_days 天，背景 finmind.backfill_history 這類程式補很久以前的
    # 歷史資料、動到舊月份檔案 mtime 不該讓這裡整批重算特徵。多留10天緩衝
    # 給 atr_hour_surprise 等需要「過去N天」歷史的特徵。
    since = pd.Timestamp.now().normalize() - pd.Timedelta(days=test_days + 10) if test_only else None
    df = load_features(since=since)
    df = df.dropna(subset=FEATURES)
    df = apply_liquidity_filter(df)

    if test_only:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        df = df[df["date"] > cutoff]

    df = df.copy()
    df["proba"] = model.predict_proba(to_model_input(df))[:, 1]
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def build_prewarm_cache() -> dict:
    """
    盤前預算快取 — 給 strategy/prewarm.py 統一呼叫的介面（見該檔案說明）。

    open_vol_history/hourly_tr_history 只吃 db/m1（歷史資料，盤中不會變），
    同一個交易日內不管幾點算結果都一樣，開盤前算一次存起來、整天重複用，
    不要讓 predict_live() 每分鐘都自己重跑一次 build_history_tables()
    （那是留給沒有傳快取時的 fallback，不是正常路徑該走的）。

    回傳的 dict key 要跟 predict_live() 接受的參數名一致，因為
    main/live_trader.py 會直接 **cache 展開傳進 predict_live()。
    """
    open_vol_history, hourly_tr_history = build_history_tables(load_m1())
    return {"open_vol_history": open_vol_history, "hourly_tr_history": hourly_tr_history}


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    open_vol_history: pd.DataFrame | None = None,
    hourly_tr_history: pd.DataFrame | None = None,
    model=None,
    threshold: float = 0.55,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論。

    day: 已載入的日K（db/fugle_day/），傳入可避免每次呼叫都重讀整個目錄
        （live_trader.py 開機時載一次、之後定期更新，呼叫端應該傳這份快取）。
        留空則沿用舊行為，內部自己 load_day()。
    open_vol_history/hourly_tr_history: build_history_tables(load_m1()) 算出的
        兩張表，atr_hour_surprise、open_vol_m1~m5_1~5 這兩組特徵需要「過去N天」
        歷史，m1_live 只有今天一天資料算不出來，這兩個參數留空的話這裡會自己
        call build_history_tables(load_m1())（對全歷史 db/m1/ 算，比較慢）；
        production 環境應該跟 day 一樣快取、開盤前算一次傳進來，不要每分鐘
        重算。詳見 strategy/orb/features.py 的 build_history_tables() docstring。
    day_trade_stocks: 當沖標的 set，若提供則只推論這些股票。

    ORB 的日K背景特徵（day_ret_N/day_atr/ma_dev_N/idx_day_*/close_vs_close_lag_N）
    全部是 shift(1) 以上的落後值，不需要「今天自己」全天收完的 OHLC；但
    day_atr/idx_day_atr 的分母是「今天自己的開盤價」，且 make_features() 用
    day_date 當 merge key，today 這個 day_date 在 db/fugle_day 完全沒有列
    （日K要收盤後才有）就沒東西可以 merge——所以還是要補一列「今天摘要」，
    理由跟 strategy/rally/predict.py 的 predict_live() 完全一樣。

    回傳格式：[{"stock_id": ..., "proba": ..., "price": ...}, ...]
    """
    if model is None:
        model = load_model_lgbm()

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    if day_trade_stocks:
        m1_live = m1_live[m1_live["stock_id"].isin(day_trade_stocks)]
    if m1_live.empty:
        return []

    if open_vol_history is None or hourly_tr_history is None:
        open_vol_history, hourly_tr_history = build_history_tables(load_m1())

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

    # m3/m5 現算：db/m3、db/m5 是批次預算，不含「今天」資料，即時推論要對
    # m1_live 現算，理由同 strategy/rally/predict.py 的 predict_live()。
    m1_live_sorted = m1_live.copy()
    m1_live_sorted["date"] = pd.to_datetime(m1_live_sorted["date"])
    m1_live_sorted = m1_live_sorted.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1_live_sorted["day_date"] = m1_live_sorted["date"].dt.date
    m3_live = compute_m3(m1_live_sorted)
    m5_live = compute_m5(m1_live_sorted)

    df = make_features(
        m1_live,
        m3=m3_live,
        m5=m5_live,
        day=day,
        open_vol_history=open_vol_history,
        hourly_tr_history=hourly_tr_history,
        compute_labels=False,
    )
    current = df[df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []

    valid = current.dropna(subset=FEATURES)
    valid = apply_liquidity_filter(valid)
    if valid.empty:
        return []

    proba = model.predict_proba(to_model_input(valid))[:, 1]
    signals = [
        {"stock_id": row["stock_id"], "proba": float(p), "price": float(row["close"])}
        for (_, row), p in zip(valid.iterrows(), proba)
        if p >= threshold
    ]
    return sorted(signals, key=lambda x: -x["proba"])
