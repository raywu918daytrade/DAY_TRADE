"""
當沖策略 — RandomForest 簡單模型

== 策略邏輯 ==

只用 1 分鐘線收盤價與成交量 2 個特徵，預測未來 30 根分K內：
  +3% 停利先碰到 → label = 1（漲）
  -3% 停損先碰到 → label = 0（跌）

== 資料 ==

載入 db/m1/ 歷史分K（與 date_trade_model.py 共用）
只用 close 與 volume 兩個欄位做特徵。

== Main 模式 ==

train      訓練 RandomForest 模型
validate   驗證模型（信心度分析 + 召回率分析）
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 確保能從根目錄導入
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from data.query import load_day, load_m1

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_MODEL_PATH = _ROOT / "models/m1_rfc.pkl"

# ── Triple Barrier 參數 ──────────────────────────────────────────────────────
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── 交易時段 ──────────────────────────────────────────────────────────────────
SESSION_START = (9, 1)
_end_h = int(os.environ.get("SESSION_END_HOUR", "10"))
_end_m = int(os.environ.get("SESSION_END_MIN", "0"))
SESSION_END = (_end_h, _end_m)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Triple Barrier Label
# ═══════════════════════════════════════════════════════════════════════════════


def _barrier_label_group(closes: np.ndarray) -> np.ndarray:
    """每日每股，逐 bar 往前看最多 HOLD_BARS 根，判斷先碰到 tp 或 sl。"""
    n = len(closes)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        entry = closes[i]
        tp_price = entry * (1 + TP_PCT)
        sl_price = entry * (1 - SL_PCT)
        future = closes[i + 1 : i + HOLD_BARS + 1]
        tp_idx = np.argmax(future >= tp_price) if (future >= tp_price).any() else len(future)
        sl_idx = np.argmax(future <= sl_price) if (future <= sl_price).any() else len(future)
        if tp_idx < sl_idx:
            labels[i] = 1
        elif sl_idx < tp_idx:
            labels[i] = 0
        elif len(future) == HOLD_BARS:
            labels[i] = 1 if future[-1] > entry else 0
    return labels


def _make_barrier_labels(m1: pd.DataFrame) -> pd.Series:
    result = m1.groupby(["stock_id", "day_date"], group_keys=False).apply(
        lambda g: pd.Series(
            _barrier_label_group(g["close"].values),
            index=g.index,
        ),
        include_groups=False,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 特徵工程（只用 close 與 volume）
# ═══════════════════════════════════════════════════════════════════════════════


FEATURES = [
    # 1分鐘K OHLCV（價格除以當日開盤正規化，量除以當日累積均量）
    "m1_open",
    "m1_high",
    "m1_low",
    "m1_close",
    "m1_volume",
    # 3分鐘K OHLCV（當前 + 前2根，每根間隔3分鐘）
    "m3_open",
    "m3_high",
    "m3_low",
    "m3_close",
    "m3_volume",
    "m3_ret",  # 3分鐘K報酬率（K棒間變化%）
    "m3_open_lag1",
    "m3_high_lag1",
    "m3_low_lag1",
    "m3_close_lag1",
    "m3_volume_lag1",
    "m3_open_lag2",
    "m3_high_lag2",
    "m3_low_lag2",
    "m3_close_lag2",
    "m3_volume_lag2",
    # 5分鐘K OHLCV（當前 + 前2根，每根間隔5分鐘）
    "m5_open",
    "m5_high",
    "m5_low",
    "m5_close",
    "m5_volume",
    "m5_ret",  # 5分鐘K報酬率（K棒間變化%）
    "m5_open_lag1",
    "m5_high_lag1",
    "m5_low_lag1",
    "m5_close_lag1",
    "m5_volume_lag1",
    "m5_open_lag2",
    "m5_high_lag2",
    "m5_low_lag2",
    "m5_close_lag2",
    "m5_volume_lag2",
    # 報酬率與量比
    "ret_1",
    "vol_ratio",
    "tf3_ret",
    "tf3_vol_ratio",
    "tf5_ret",
    "tf5_vol_ratio",
]


def make_features(
    m1: pd.DataFrame,
    compute_labels: bool = True,
) -> pd.DataFrame:
    """
    簡單特徵工程：只用 close 與 volume 衍生特徵。

    回傳欄位：
      ret_1, vol_ratio, tf3_ret, tf3_vol_ratio, tf5_ret, tf5_vol_ratio,
      target（若 compute_labels=True）
    """
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)

    # ── 當日開盤價（第一根 open）用於正規化 ──────────────────────────
    day_open = g_day["open"].transform("first").replace(0, np.nan)

    # ── 1分鐘K OHLCV（價格 / 當日開盤，量用當日累積均量正規化） ─────
    m1["m1_open"] = m1["open"] / day_open
    m1["m1_high"] = m1["high"] / day_open
    m1["m1_low"] = m1["low"] / day_open
    m1["m1_close"] = m1["close"] / day_open
    # 量比（當前量 / 當日累積平均量）
    m1["_cum_vol"] = g_day["volume"].transform("cumsum")
    m1["_bar_count"] = g_day["volume"].transform("cumcount") + 1
    m1["m1_volume"] = m1["volume"] / (m1["_cum_vol"] / m1["_bar_count"]).replace(0, np.nan)

    # ── 3分鐘K OHLCV（由1分K滾動聚合，不跨日）───────────────────────
    # 當前3分鐘K
    m1["m3_open"] = g_day["open"].transform(lambda x: x.shift(2)) / day_open
    m1["m3_high"] = g_day["high"].transform(lambda x: x.rolling(3).max()) / day_open
    m1["m3_low"] = g_day["low"].transform(lambda x: x.rolling(3).min()) / day_open
    m1["m3_close"] = m1["close"] / day_open
    m1["m3_volume"] = g_day["volume"].transform(lambda x: x.rolling(3).sum().pct_change(3))
    # 3分鐘K報酬率（K棒間變化%）
    m1["m3_ret"] = g_day["close"].transform(lambda x: x.pct_change(3))
    # 前1根3分鐘K（shift 3）
    m1["m3_open_lag1"] = g_day["m3_open"].transform(lambda x: x.shift(3))
    m1["m3_high_lag1"] = g_day["m3_high"].transform(lambda x: x.shift(3))
    m1["m3_low_lag1"] = g_day["m3_low"].transform(lambda x: x.shift(3))
    m1["m3_close_lag1"] = g_day["m3_close"].transform(lambda x: x.shift(3))
    m1["m3_volume_lag1"] = g_day["m3_volume"].transform(lambda x: x.shift(3))
    # 前2根3分鐘K（shift 6）
    m1["m3_open_lag2"] = g_day["m3_open"].transform(lambda x: x.shift(6))
    m1["m3_high_lag2"] = g_day["m3_high"].transform(lambda x: x.shift(6))
    m1["m3_low_lag2"] = g_day["m3_low"].transform(lambda x: x.shift(6))
    m1["m3_close_lag2"] = g_day["m3_close"].transform(lambda x: x.shift(6))
    m1["m3_volume_lag2"] = g_day["m3_volume"].transform(lambda x: x.shift(6))

    # ── 5分鐘K OHLCV（由1分K滾動聚合，不跨日）───────────────────────
    # 當前5分鐘K
    m1["m5_open"] = g_day["open"].transform(lambda x: x.shift(4)) / day_open
    m1["m5_high"] = g_day["high"].transform(lambda x: x.rolling(5).max()) / day_open
    m1["m5_low"] = g_day["low"].transform(lambda x: x.rolling(5).min()) / day_open
    m1["m5_close"] = m1["close"] / day_open
    m1["m5_volume"] = g_day["volume"].transform(lambda x: x.rolling(5).sum().pct_change(5))
    # 5分鐘K報酬率（K棒間變化%）
    m1["m5_ret"] = g_day["close"].transform(lambda x: x.pct_change(5))
    # 前1根5分鐘K（shift 5）
    m1["m5_open_lag1"] = g_day["m5_open"].transform(lambda x: x.shift(5))
    m1["m5_high_lag1"] = g_day["m5_high"].transform(lambda x: x.shift(5))
    m1["m5_low_lag1"] = g_day["m5_low"].transform(lambda x: x.shift(5))
    m1["m5_close_lag1"] = g_day["m5_close"].transform(lambda x: x.shift(5))
    m1["m5_volume_lag1"] = g_day["m5_volume"].transform(lambda x: x.shift(5))
    # 前2根5分鐘K（shift 10）
    m1["m5_open_lag2"] = g_day["m5_open"].transform(lambda x: x.shift(10))
    m1["m5_high_lag2"] = g_day["m5_high"].transform(lambda x: x.shift(10))
    m1["m5_low_lag2"] = g_day["m5_low"].transform(lambda x: x.shift(10))
    m1["m5_close_lag2"] = g_day["m5_close"].transform(lambda x: x.shift(10))
    m1["m5_volume_lag2"] = g_day["m5_volume"].transform(lambda x: x.shift(10))

    # ── 報酬率與量比 ────────────────────────────────────────────────
    # 前1分鐘報酬率
    m1["ret_1"] = g_day["close"].transform(lambda x: x.pct_change(1))

    # 量比（當前量 / 前1分鐘量）
    m1["vol_ratio"] = g_day["volume"].transform(lambda x: x / x.shift(1).replace(0, np.nan))

    # 3分鐘K報酬率
    m1["tf3_ret"] = g_day["close"].transform(lambda x: x.pct_change(3))

    # 3分鐘K量比
    m1["tf3_vol_ratio"] = g_day["volume"].transform(
        lambda x: x.rolling(3).sum() / x.rolling(3).sum().shift(3).replace(0, np.nan)
    )

    # 5分鐘K報酬率
    m1["tf5_ret"] = g_day["close"].transform(lambda x: x.pct_change(5))

    # 5分鐘K量比
    m1["tf5_vol_ratio"] = g_day["volume"].transform(
        lambda x: x.rolling(5).sum() / x.rolling(5).sum().shift(5).replace(0, np.nan)
    )

    # 清除暫存欄位
    m1 = m1.drop(columns=["_cum_vol", "_bar_count"], errors="ignore")

    # 時間特徵（用於過濾時段，不作為模型輸入）
    m1["hour"] = m1["date"].dt.hour
    m1["minute"] = m1["date"].dt.minute

    # 目標標籤
    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    return m1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 訓練
# ═══════════════════════════════════════════════════════════════════════════════


def train(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    訓練 RandomForest 模型。

    Parameters
    ----------
    test_days : int
        以最後 N 天為測試集（預設 10）。若指定 start_date/end_date 則忽略此參數。
    start_date : str
        訓練資料起日，格式 "YYYY-MM-DD"。留空表示不限制。
    end_date : str
        訓練資料迄日，格式 "YYYY-MM-DD"。留空表示不限制。
    """
    print("載入分K...")
    m1 = load_m1()
    print(f"  {len(m1):,} 筆")

    print("特徵工程...")
    df = make_features(m1, compute_labels=True)
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵: {FEATURES}")

    # 只保留早盤時段
    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()
    print(f"  早盤有效樣本: {len(df):,} 筆")

    # 日期過濾
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    print(f"  日期區間: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

    # 時間切割（若有指定明確日期範圍則以日期切割，否則用 test_days）
    if start_date and end_date:
        cutoff = df["date"].quantile(0.8)  # 前 80% 訓練、後 20% 測試
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    else:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    print(
        f"  訓練: {len(train_df):,} ({train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"  測試: {len(test_df):,} ({test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')})"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}")

    return model


def load_model():
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型，請先執行 train()")
    return joblib.load(_MODEL_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 驗證
# ═══════════════════════════════════════════════════════════════════════════════


def confidence_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """依信心度區間顯示樣本數與勝率。"""
    if model is None:
        model = load_model()

    m1 = load_m1()
    df = make_features(m1, compute_labels=True)
    df = df.dropna(subset=FEATURES + ["target"])

    # 日期過濾
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    bins = [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.01]
    labels = [
        "<0.30",
        "0.30-0.35",
        "0.35-0.40",
        "0.40-0.45",
        "0.45-0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.70",
        "≥0.70",
    ]
    test_df["bucket"] = pd.cut(test_df["proba"], bins=bins, labels=labels, right=False)

    report = (
        test_df.groupby("bucket", observed=True)
        .agg(樣本數=("target", "count"), 勝率=("target", "mean"))
        .assign(勝率=lambda x: (x["勝率"] * 100).round(1).astype(str) + "%")
    )
    print("\n── 信心度分析（測試集）──")
    print(report.to_string())
    return report


def coverage_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """召回率分析：在不同門檻下的精確率與召回率。"""
    if model is None:
        model = load_model()

    m1 = load_m1()
    df = make_features(m1, compute_labels=True)
    df = df.dropna(subset=FEATURES + ["target"])

    # 日期過濾
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    total_pos = test_df["target"].sum()
    total_neg = (test_df["target"] == 0).sum()

    print(f"\n── 召回率分析（測試集）──")
    print(f"  實際漲（label=1）: {total_pos:,} 筆")
    print(f"  實際跌（label=0）: {total_neg:,} 筆")
    print()
    print(f"  {'門檻':>6}  {'訊號數':>7}  {'精確率':>7}  {'召回率':>7}  {'F1':>6}")
    print("  " + "-" * 45)

    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        flagged = test_df["proba"] >= thr
        tp = (flagged & (test_df["target"] == 1)).sum()
        precision = tp / flagged.sum() if flagged.sum() > 0 else 0
        recall = tp / total_pos if total_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {thr:.2f}  {flagged.sum():>7,}  {precision*100:>6.1f}%  {recall*100:>6.1f}%  {f1:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 特徵重要性
# ═══════════════════════════════════════════════════════════════════════════════


def feature_importance(model=None, top_n: int = 10):
    """顯示 RandomForest 特徵重要性。"""
    if model is None:
        model = load_model()

    print("\n── 特徵重要性 ──")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s}  {imp:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 主程式
# ═══════════════════════════════════════════════════════════════════════════════


def main(
    mode: str = "",
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    當沖策略 RandomForest 主程式。

    支援兩種用法：
      1. 直接傳參數：main(mode="train")
      2. CLI 執行：python strategy/date_trade_rfc_model.py train

    Parameters
    ----------
    mode : str
        執行模式。留空則從 CLI 讀取。
        train / validate / importance
    test_days : int
        測試集天數（預設 10）
    start_date : str
        資料起日，格式 "YYYY-MM-DD"。留空不限制。
    end_date : str
        資料迄日，格式 "YYYY-MM-DD"。留空不限制。
    """
    if not mode:
        parser = argparse.ArgumentParser(
            description="當沖策略 — RandomForest（只用 close + volume）",
        )
        parser.add_argument(
            "mode",
            choices=["train", "validate", "importance"],
            help="執行模式",
        )
        parser.add_argument("--test_days", type=int, default=10, help="測試集天數")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        start_date = args.start_date
        end_date = args.end_date

    if mode == "train":
        train(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "validate":
        confidence_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        coverage_report(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "importance":
        feature_importance()

    else:
        print(f"未知模式: {mode}，可用: train / validate / importance")


if __name__ == "__main__":
    # ══════════════════════════════════════════════════════════════════════
    #  在這裡直接改 mode，不用每次打 CLI
    #
    #  可用 mode:
    #    "train"       訓練模型
    #    "validate"    信心度 & 召回率分析
    #    "importance"  顯示特徵重要性
    #    ""            走 CLI argparse（terminal 下參數）
    #
    #  start_date / end_date 可指定日期區間（留空 = 全部資料）
    # ══════════════════════════════════════════════════════════════════════
    mode = "train"
    test_days = 10
    start_date = ""
    end_date = ""

    main(mode=mode, test_days=test_days, start_date=start_date, end_date=end_date)
