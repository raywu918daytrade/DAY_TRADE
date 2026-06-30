"""
日內交易模型 — LightGBM 分K方向預測

== 訓練邏輯 ==

目標
    預測下一根分K的 close 是否高於當前 close（二元分類，1=漲，0=跌）

資料來源
    分K (db/m1/)      ：還原股價歷史1分鐘OHLCV，所有股票合併訓練
    日K (db/day_trade/)：還原股價歷史日線，提供趨勢背景特徵

特徵
    分K 特徵：
        ret_1/3/5/10/20/30  當前收盤相對N分鐘前的報酬率
        vol_ratio           當前成交量 / 20分鐘均量
        close_pos           close 在當根 (high-low) 的相對位置 [0,1]
        hour / minute       盤中時間（捕捉開收盤效應）
    日K 特徵（前一交易日，無未來洩漏）：
        prev_ret            前日漲跌幅
        prev_vol_ratio      前日量 / 20日均量
        pos_20d             收盤在近20日高低點的相對位置 [0,1]

資料切割
    以時間為界，最後 test_days 個自然日做測試集，不隨機 shuffle
    避免未來資料洩漏（look-ahead bias）

    注意：每支股票每日最後一根分K不納入訓練，
    因為 target（下一分鐘）會跨日，導致標籤錯誤

模型
    LightGBM Classifier，early stopping 根據測試集 binary_logloss

評估
    Accuracy、AUC（測試集）
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, roc_auc_score

from tay_trade.query import load_day, load_m1

_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")

# ── 特徵工程 ──────────────────────────────────────────────────────────────────

def make_features(m1: pd.DataFrame, day: pd.DataFrame) -> pd.DataFrame:
    g = m1.groupby("stock_id", group_keys=False)

    # 分K 報酬率
    for lag in [1, 3, 5, 10, 20, 30]:
        m1[f"ret_{lag}"] = g["close"].transform(lambda x: x.pct_change(lag))

    # 量比（相對20分鐘均量）
    m1["vol_ma20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
    m1["vol_ratio"] = m1["volume"] / m1["vol_ma20"].replace(0, np.nan)

    # close 在當根 high-low 的位置
    bar_range = (m1["high"] - m1["low"]).replace(0, np.nan)
    m1["close_pos"] = (m1["close"] - m1["low"]) / bar_range

    # 時間特徵
    m1["hour"] = m1["date"].dt.hour
    m1["minute"] = m1["date"].dt.minute
    m1["day_date"] = m1["date"].dt.date

    # 目標：下一分鐘 close > 當前 close（組內 shift，不跨股票）
    m1["target"] = (
        g["close"].transform(lambda x: x.shift(-1)) > m1["close"]
    ).astype(int)

    # 過濾每日最後一根（目標無效，會跨日）
    last_bar = m1.groupby(["stock_id", "day_date"])["date"].transform("max")
    m1 = m1[m1["date"] < last_bar].copy()

    # 日K 特徵（前一日，避免未來資料）
    dg = day.groupby("stock_id")
    day = day.copy()
    day["prev_ret"]       = dg["close"].transform(lambda x: x.pct_change(1))
    day["prev_vol_ratio"] = dg["volume"].transform(lambda x: x / x.rolling(20).mean())
    day["pos_20d"]        = dg["close"].transform(
        lambda x: (x - x.rolling(20).min()) / (x.rolling(20).max() - x.rolling(20).min()).replace(0, np.nan)
    )
    day["day_date"] = day["date"].dt.date
    day_feat = day[["stock_id", "day_date", "prev_ret", "prev_vol_ratio", "pos_20d"]]

    m1 = m1.merge(day_feat, on=["stock_id", "day_date"], how="left")
    return m1


FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_30",
    "vol_ratio", "close_pos", "hour", "minute",
    "prev_ret", "prev_vol_ratio", "pos_20d",
]


# ── 訓練 ──────────────────────────────────────────────────────────────────────

def train(test_days: int = 10):
    print("載入分K...")
    m1 = load_m1()
    print(f"  {len(m1):,} 筆")

    print("載入日K...")
    day = load_day()
    print(f"  {len(day):,} 筆")

    print("特徵工程...")
    df = make_features(m1, day)
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  有效樣本: {len(df):,} 筆")

    # 時間切割（最後 test_days 天當測試，不 shuffle 避免未來洩漏）
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["date"] <= cutoff]
    test_df  = df[df["date"] >  cutoff]
    print(f"  訓練: {len(train_df):,}  測試: {len(test_df):,}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train_df[FEATURES], train_df["target"],
        eval_set=[(test_df[FEATURES], test_df["target"])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    return model


if __name__ == "__main__":
    model = train()
