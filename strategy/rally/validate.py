"""
模型驗證報表 — 信心度分析、召回率分析、模型×時段×信心度交叉報表、特徵重要性

破底翻（breakout_signal）相關的驗證報表在 strategy/rally/experiments/
breakout_filter_eval.py——那是還沒有定論、需要持續觀察的問題，不是這裡
「已經決定要長期用」的核心驗證報表。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from strategy.rally.features import FEATURES, load_features
from strategy.rally.train import (
    _MODEL_PATH,
    _MODEL_PATH_LGBM,
    _MODEL_PATH_XGB,
    load_model,
    load_model_lgbm,
    load_model_xgb,
)

_CONFIDENCE_BINS = [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.01]
_CONFIDENCE_LABELS = [
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


def confidence_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """依信心度區間顯示樣本數與勝率。"""
    if model is None:
        model = load_model()

    df = load_features(start_date=start_date)
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
    test_df["bucket"] = pd.cut(test_df["proba"], bins=_CONFIDENCE_BINS, labels=_CONFIDENCE_LABELS, right=False)

    report = (
        test_df.groupby("bucket", observed=True)
        .agg(樣本數=("target", "count"), 勝率=("target", "mean"))
        .assign(勝率=lambda x: (x["勝率"] * 100).round(1).astype(str) + "%")
    )
    print("\n── 信心度分析（測試集）──")
    print(report.to_string())
    return report


def model_hour_confidence_report(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    多模型 × 時段 × 信心度 交叉報表。

    同一份測試集，RFC/XGB/LGBM（有訓練好的才顯示）分別依「小時（9~13）」與
    「信心度區間」交叉列出樣本數與勝率，方便比較不同模型在不同時段/不同
    信心度下的表現差異。模型現在是全天訓練的（見 features.py 的
    minutes_since_open），這份報表用來檢查全天各時段實際表現是否均衡，
    還是集中在特定時段才準。
    """
    models = {
        "RFC ": (load_model, _MODEL_PATH),
        "XGB ": (load_model_xgb, _MODEL_PATH_XGB),
        "LGBM": (load_model_lgbm, _MODEL_PATH_LGBM),
    }
    loaded = {}
    for name, (loader, path) in models.items():
        if not path.exists():
            print(f"  （跳過 {name.strip()}：模型不存在）")
            continue
        loaded[name] = loader()

    if not loaded:
        print("  無可用模型，請先訓練 RFC / XGB / LGBM。")
        return

    df = load_features(start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    for name, model in loaded.items():
        d = test_df.copy()
        d["proba"] = model.predict_proba(d[FEATURES])[:, 1]
        d["信心度"] = pd.cut(d["proba"], bins=_CONFIDENCE_BINS, labels=_CONFIDENCE_LABELS, right=False)

        print(f"\n══════════ {name.strip()} — 依時段 × 信心度 ══════════")
        for h in [9, 10, 11, 12, 13]:
            sub = d[d["hour"] == h]
            if sub.empty:
                continue
            report = sub.groupby("信心度", observed=True).agg(樣本數=("target", "count"), 勝率=("target", "mean"))
            report = report[report["樣本數"] > 0]
            if report.empty:
                continue

            # 召回率：跟 coverage_report/舊 compare 一致的定義——若只在
            # proba >= 該區間下界時進場，能抓到這個時段內多少比例的實際上漲
            # （不是「這個區間內」的召回率，是「這個門檻以上」的累積召回率）。
            total_pos = sub["target"].sum()
            recalls = {}
            for lo, label in zip(_CONFIDENCE_BINS[:-1], _CONFIDENCE_LABELS):
                tp_cum = sub.loc[sub["proba"] >= lo, "target"].sum()
                recalls[label] = tp_cum / total_pos if total_pos else float("nan")
            report["召回率"] = [recalls.get(idx, float("nan")) for idx in report.index]
            report["召回率"] = (report["召回率"] * 100).round(1).astype(str) + "%"
            report["勝率"] = (report["勝率"] * 100).round(1).astype(str) + "%"
            print(f"\n  -- {h} 點（實際上漲 {total_pos:,} 筆）--")
            print("  " + report.to_string().replace("\n", "\n  "))


def hour_signal_report(
    threshold: float = 0.55,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    每個時段：門檻以上的訊號數 vs 實際抓到幾筆上漲（原始筆數，不是百分比）。

    跟 model_hour_confidence_report 的差別：那份報表是按信心度「區間」分桶
    （<0.30/0.30-0.35/...），召回率是「≥某信心度」的累積算法，容易被誤讀成
    「這個區間內抓到多少」。這份報表直接固定一個門檻（預設 0.55，跟
    predict_live() 的實單門檻一致），每個時段只看一行：達門檻的訊號數、
    其中真的上漲幾筆、時段內總共實際上漲幾筆。
    """
    models = {
        "RFC ": (load_model, _MODEL_PATH),
        "XGB ": (load_model_xgb, _MODEL_PATH_XGB),
        "LGBM": (load_model_lgbm, _MODEL_PATH_LGBM),
    }
    loaded = {}
    for name, (loader, path) in models.items():
        if not path.exists():
            print(f"  （跳過 {name.strip()}：模型不存在）")
            continue
        loaded[name] = loader()

    if not loaded:
        print("  無可用模型，請先訓練 RFC / XGB / LGBM。")
        return

    df = load_features(start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}  門檻: {threshold}")

    for name, model in loaded.items():
        d = test_df.copy()
        d["proba"] = model.predict_proba(d[FEATURES])[:, 1]

        print(f"\n══════════ {name.strip()} — 每小時訊號數 vs 抓到筆數（門檻 ≥{threshold}） ══════════")
        print(f"  {'時段':>4}  {'訊號數':>7}  {'抓到':>6}  {'實際上漲':>8}  {'精確率':>7}  {'召回率':>7}")
        print("  " + "-" * 55)
        for h in [9, 10, 11, 12, 13]:
            sub = d[d["hour"] == h]
            if sub.empty:
                continue
            total_pos = int(sub["target"].sum())
            flagged = sub["proba"] >= threshold
            n_signal = int(flagged.sum())
            tp = int((flagged & (sub["target"] == 1)).sum())
            precision = tp / n_signal if n_signal else float("nan")
            recall = tp / total_pos if total_pos else float("nan")
            print(f"  {h:>4}  {n_signal:>7,}  {tp:>6,}  {total_pos:>8,}  {precision*100:>6.1f}%  {recall*100:>6.1f}%")


def coverage_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """召回率分析：在不同門檻下的精確率與召回率。"""
    if model is None:
        model = load_model()

    df = load_features(start_date=start_date)
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


def feature_importance(model=None, top_n: int = 10):
    """顯示 RandomForest 特徵重要性。"""
    if model is None:
        model = load_model()

    print("\n── 特徵重要性 ──")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s}  {imp:.4f}")
