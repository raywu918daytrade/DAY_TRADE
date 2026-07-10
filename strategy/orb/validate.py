"""
模型驗證報表 — 信心度分析、召回率分析、時段交叉報表、特徵重要性、LGBM vs XGB 比較

比照 strategy/rally/validate.py 的做法。confidence_report/coverage_report/
hour_confidence_report/feature_importance 都是單一模型的報表（model=None 時
預設 LGBM），entry.py 的 validate 模式用 available_models() 對每個已訓練好
的模型各跑一輪；compare_report() 是兩個模型在同一份測試集上的 Accuracy/AUC
對照表。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from strategy.orb.config import DEFAULT_TEST_DAYS
from strategy.orb.features import FEATURES, apply_liquidity_filter, load_features, to_model_input
from strategy.orb.train import _MODEL_PATH_LGBM, _MODEL_PATH_XGB, load_model_lgbm, load_model_xgb


def available_models() -> dict:
    """回傳目前已訓練好的模型 {名稱: model}，還沒訓練過的會跳過。"""
    models = {}
    if _MODEL_PATH_LGBM.exists():
        models["LGBM"] = load_model_lgbm()
    else:
        print("  （跳過 LGBM：模型不存在，請先執行 train_lgbm）")
    if _MODEL_PATH_XGB.exists():
        models["XGB"] = load_model_xgb()
    else:
        print("  （跳過 XGB：模型不存在，請先執行 train_xgb）")
    return models

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


def _warn_if_train_test_overlap(model, test_df: pd.DataFrame) -> None:
    """
    檢查驗證用的測試區間，是不是跟這個模型實際訓練時的切點重疊。

    train.py 的 train_lgbm()/train_xgb() 會把訓練切點存在 model._orb_train_cutoff
    上；如果驗證時傳的 test_days 跟訓練時不一樣，測試集可能有一部分其實是
    模型訓練時看過的資料，算出來的 AUC/勝率會虛高、不能信（2026-07-10
    實測過：訓練用 test_days=5、驗證卻用 test_days=10，AUC 從 0.65 假漲到
    0.73，兩批資料重疊了快一半）。舊模型（加這個機制之前存的）沒有這個
    屬性，跳過檢查，不報錯。
    """
    train_cutoff = getattr(model, "_orb_train_cutoff", None)
    if train_cutoff is None:
        return
    test_start = test_df["date"].min()
    if test_start <= train_cutoff:
        print(
            f"  ⚠️  警告：測試區間起點（{test_start.strftime('%Y-%m-%d')}）"
            f"沒有晚於這個模型的訓練切點（{train_cutoff.strftime('%Y-%m-%d')}）——"
            f"目前這份「測試集」有一部分是模型訓練時已經看過的資料，"
            f"算出來的指標會虛高，不能信。請用跟訓練時一致的 test_days，"
            f"或用這個 test_days 重新訓練一次。"
        )


def _load_test_df(
    model,
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    df = apply_liquidity_filter(df)

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")
    _warn_if_train_test_overlap(model, test_df)

    test_df["proba"] = model.predict_proba(to_model_input(test_df))[:, 1]
    return test_df


def confidence_report(
    model=None,
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
):
    """依信心度區間顯示樣本數與勝率。"""
    if model is None:
        model = load_model_lgbm()

    test_df = _load_test_df(model, test_days, start_date, end_date)
    test_df["bucket"] = pd.cut(test_df["proba"], bins=_CONFIDENCE_BINS, labels=_CONFIDENCE_LABELS, right=False)

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
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
):
    """召回率分析：在不同門檻下的精確率與召回率。"""
    if model is None:
        model = load_model_lgbm()

    test_df = _load_test_df(model, test_days, start_date, end_date)

    total_pos = test_df["target"].sum()
    total_neg = (test_df["target"] == 0).sum()

    print("\n── 召回率分析（測試集）──")
    print(f"  實際漲（label=1）: {total_pos:,} 筆")
    print(f"  實際跌（label=0）: {total_neg:,} 筆")
    print()
    print(f"  {'門檻':>6}  {'訊號數':>7}  {'抓到':>7}  {'實際漲':>8}  {'精確率':>7}  {'召回率':>7}  {'F1':>6}")
    print("  " + "-" * 65)

    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        flagged = test_df["proba"] >= thr
        tp = (flagged & (test_df["target"] == 1)).sum()
        precision = tp / flagged.sum() if flagged.sum() > 0 else 0
        recall = tp / total_pos if total_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(
            f"  {thr:.2f}  {flagged.sum():>7,}  {tp:>7,}  {total_pos:>8,}  "
            f"{precision*100:>6.1f}%  {recall*100:>6.1f}%  {f1:.3f}"
        )


def hour_confidence_report(
    model=None,
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
):
    """依時段（9~13點）× 信心度區間 交叉列出樣本數、勝率、累積召回率。"""
    if model is None:
        model = load_model_lgbm()

    test_df = _load_test_df(model, test_days, start_date, end_date)
    test_df["信心度"] = pd.cut(test_df["proba"], bins=_CONFIDENCE_BINS, labels=_CONFIDENCE_LABELS, right=False)

    for h in [9, 10, 11, 12, 13]:
        sub = test_df[test_df["hour"] == h]
        if sub.empty:
            continue
        report = sub.groupby("信心度", observed=True).agg(樣本數=("target", "count"), 勝率=("target", "mean"))
        report = report[report["樣本數"] > 0]
        if report.empty:
            continue

        # 召回率：≥該區間下界的門檻能抓到這個時段內多少比例的實際上漲（累積算法，跟 rally 一致）
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


def compare_report(
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
):
    """LGBM vs XGB：同一份測試集上的 Accuracy / AUC 對照表。"""
    models = {"LGBM": load_model_lgbm(), "XGB ": load_model_xgb()}

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    df = apply_liquidity_filter(df)
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")
    for name, model in models.items():
        _warn_if_train_test_overlap(model, test_df)

    print("\n── LGBM vs XGB（同一份測試集）──")
    print(f"  {'模型':>6}  {'Accuracy':>9}  {'AUC':>7}")
    print("  " + "-" * 28)
    X_test = to_model_input(test_df)
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(test_df["target"], y_pred)
        auc = roc_auc_score(test_df["target"], y_prob)
        print(f"  {name}  {acc:>9.4f}  {auc:>7.4f}")


def feature_importance(model=None):
    """顯示 LightGBM 特徵重要性。"""
    if model is None:
        model = load_model_lgbm()

    print("\n── 特徵重要性 ──")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:26s}  {imp:.4f}")
