"""
LGBM/XGB 訊號合併實驗 — 驗證能不能把兩個模型合併成一個訊號來源。

背景：strategy/orb/xgb/live.py 2026-07-22 討論過「不要平均/投票，2個模型
各自獨立進出場」，這次重新驗證這個決定站不站得住腳。只做離線驗證，不改
train.py/predict.py/rfc|lgbm|xgb/live.py/main/config.py 任何正式 pipeline，
跑完看結果再決定要不要推翻原本的決定，不在這支檔案的範圍內。

測試四種做法，同一份 test_df 上比較：
    baseline    RFC/LGBM/XGB 各自單獨的門檻對照表（現有已訓練模型）
    stacking    XGB 機率當一個新特徵，餵給重新訓練的 meta-LGBM（GroupKFold
                out-of-fold，避免同一(股票,日)多筆候選互相洩漏——專案已知
                同一(股票,日)候選 83.6% target 一致，普通 KFold 隨機切會讓
                高度相關的候選跨 fold 外洩，做出來的 out-of-fold 預測其實
                看過自己的近親，不是真的獨立）
    average     LGBM/XGB 機率簡單平均 + 一組加權平均（XGB 0.6 / LGBM 0.4，
                比照上次驗證 XGB 表現持續變好、比 LGBM 穩的結果）
    voting      LGBM 跟 XGB 都要過門檻，訊號才算數

用法：
    python -m strategy.orb.experiments.verify_ensemble
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from strategy.orb.config import DEFAULT_TEST_DAYS
from strategy.orb.features import to_model_input
from strategy.orb.train import _prepare_train_test, load_model_lgbm, load_model_rfc, load_model_xgb

_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def _threshold_table(name: str, proba: np.ndarray, target: pd.Series) -> None:
    """印門檻 vs 精確率/召回率/F1，格式比照 validate.py::coverage_report()。"""
    total_pos = int(target.sum())
    y = target.to_numpy()
    print(f"\n── {name} ──")
    print(f"  {'門檻':>6}  {'訊號數':>7}  {'抓到':>7}  {'精確率':>7}  {'召回率':>7}  {'F1':>6}")
    print("  " + "-" * 55)
    for thr in _THRESHOLDS:
        flagged = proba >= thr
        n = int(flagged.sum())
        tp = int((flagged & (y == 1)).sum())
        precision = tp / n if n > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {thr:.2f}  {n:>7,}  {tp:>7,}  {precision*100:>6.1f}%  {recall*100:>6.1f}%  {f1:.3f}")


def _fit_lgbm(X: pd.DataFrame, y: pd.Series):
    import lightgbm as lgb

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=n_neg / n_pos,
    )
    model.fit(X, y)
    return model


def _fit_xgb(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBClassifier

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
        enable_categorical=True,
        scale_pos_weight=n_neg / n_pos,
    )
    model.fit(X, y)
    return model


def _oof_xgb_proba(train_df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """對 train_df 做 GroupKFold out-of-fold XGB 機率，group 用 (stock_id, day_date)
    ——見檔頭說明，不能用普通 KFold。"""
    groups = train_df["stock_id"].astype(str) + "_" + train_df["date"].dt.date.astype(str)
    X = to_model_input(train_df)
    y = train_df["target"]
    oof = np.full(len(train_df), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        model = _fit_xgb(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"  fold {fold}/{n_splits} 完成（train={len(tr_idx):,} / val={len(va_idx):,}）")
    return oof


def run(test_days: int = DEFAULT_TEST_DAYS, start_date: str = "2024-07-01", end_date: str = ""):
    print("載入 train/test（跟正式訓練同一套切分）...")
    train_df, test_df = _prepare_train_test(
        test_days=test_days, start_date=start_date, end_date=end_date, use_cache=True
    )
    X_test = to_model_input(test_df)
    y_test = test_df["target"]

    # ── 基準線：現有已訓練模型 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("基準線（現有已訓練模型）")
    print("=" * 70)
    rfc_proba = load_model_rfc().predict_proba(X_test)[:, 1]
    lgbm_proba = load_model_lgbm().predict_proba(X_test)[:, 1]
    xgb_proba = load_model_xgb().predict_proba(X_test)[:, 1]
    _threshold_table("RFC", rfc_proba, y_test)
    _threshold_table("LGBM", lgbm_proba, y_test)
    _threshold_table("XGB", xgb_proba, y_test)

    # ── 做法1：Stacking（XGB 機率當 LGBM 特徵） ────────────────────────
    print("\n" + "=" * 70)
    print("做法1：Stacking（XGB 機率當 LGBM 特徵，GroupKFold out-of-fold）")
    print("=" * 70)
    xgb_oof = _oof_xgb_proba(train_df)
    valid_mask = ~np.isnan(xgb_oof)
    if valid_mask.sum() < len(train_df):
        print(f"  警告：{len(train_df) - int(valid_mask.sum())} 筆沒有 out-of-fold 預測，已排除")

    X_meta_train = to_model_input(train_df.loc[valid_mask]).copy()
    X_meta_train["xgb_proba"] = xgb_oof[valid_mask]
    y_meta_train = train_df.loc[valid_mask, "target"]
    meta_lgbm = _fit_lgbm(X_meta_train, y_meta_train)

    full_xgb = load_model_xgb()  # 用完整 train_df 訓練出來的正式 XGB，對 test_df 沒有洩漏疑慮
    X_meta_test = X_test.copy()
    X_meta_test["xgb_proba"] = full_xgb.predict_proba(X_test)[:, 1]
    stacking_proba = meta_lgbm.predict_proba(X_meta_test)[:, 1]
    _threshold_table("Stacking (meta-LGBM + XGB proba)", stacking_proba, y_test)

    # ── 做法2：機率平均 ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("做法2：機率平均")
    print("=" * 70)
    avg_proba = (lgbm_proba + xgb_proba) / 2
    _threshold_table("Average (LGBM 0.5 / XGB 0.5)", avg_proba, y_test)
    weighted_proba = lgbm_proba * 0.4 + xgb_proba * 0.6
    _threshold_table("Weighted average (LGBM 0.4 / XGB 0.6)", weighted_proba, y_test)

    # ── 做法3：投票（LGBM 跟 XGB 都要過門檻） ──────────────────────────
    print("\n" + "=" * 70)
    print("做法3：投票（LGBM 跟 XGB 都要過門檻）")
    print("=" * 70)
    total_pos = int(y_test.sum())
    y_test_arr = y_test.to_numpy()
    print(f"\n  {'門檻':>6}  {'訊號數':>7}  {'抓到':>7}  {'精確率':>7}  {'召回率':>7}  {'F1':>6}")
    print("  " + "-" * 55)
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        flagged = (lgbm_proba >= thr) & (xgb_proba >= thr)
        n = int(flagged.sum())
        tp = int((flagged & (y_test_arr == 1)).sum())
        precision = tp / n if n > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {thr:.2f}  {n:>7,}  {tp:>7,}  {precision*100:>6.1f}%  {recall*100:>6.1f}%  {f1:.3f}")


if __name__ == "__main__":
    run(test_days=DEFAULT_TEST_DAYS, start_date="2024-07-01")
