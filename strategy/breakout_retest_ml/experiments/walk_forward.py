"""
簡易 walk-forward：固定訓練窗 / 測試窗向前滾動，印出各窗止盈 precision。

用法：
    python -m strategy.breakout_retest_ml.experiments.walk_forward --use_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import precision_score

from strategy.breakout_retest_ml.features import FEATURES
from strategy.breakout_retest_ml.train import _prepare_data


def run(
    start_date: str = "2025-07-01",
    train_days: int = 60,
    test_days: int = 15,
    step_days: int = 15,
    threshold: float = 0.6,
    use_cache: bool = True,
):
    df = _prepare_data(use_cache=use_cache, start_date=start_date)
    if df.empty:
        print("無資料")
        return

    df = df.sort_values("date")
    t_min, t_max = df["date"].min(), df["date"].max()
    cursor = t_min + pd.Timedelta(days=train_days)
    fold = 0
    while cursor + pd.Timedelta(days=test_days) <= t_max + pd.Timedelta(days=1):
        train_end = cursor
        test_end = cursor + pd.Timedelta(days=test_days)
        train_df = df[(df["date"] >= train_end - pd.Timedelta(days=train_days)) & (df["date"] < train_end)]
        test_df = df[(df["date"] >= train_end) & (df["date"] < test_end)]
        fold += 1
        if len(train_df) < 50 or len(test_df) < 5:
            print(f"fold {fold}: 樣本不足 train={len(train_df)} test={len(test_df)}，跳過")
            cursor += pd.Timedelta(days=step_days)
            continue

        model = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(train_df[FEATURES], train_df["target"])
        proba = model.predict_proba(test_df[FEATURES])
        class_idx = {c: i for i, c in enumerate(model.classes_)}
        p_tp = proba[:, class_idx[2]] if 2 in class_idx else [0.0] * len(test_df)
        y_pred = (pd.Series(p_tp) >= threshold).astype(int).to_numpy()
        y_true = (test_df["target"] == 2).astype(int).to_numpy()
        prec = precision_score(y_true, y_pred, zero_division=0)
        n_sig = int(y_pred.sum())
        print(
            f"fold {fold}: train~{train_end.date()} test~{test_end.date()} "
            f"n_train={len(train_df)} n_test={len(test_df)} signals={n_sig} "
            f"tp_precision@{threshold}={prec * 100:.1f}%"
        )
        cursor += pd.Timedelta(days=step_days)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start_date", default="2025-07-01")
    p.add_argument("--train_days", type=int, default=60)
    p.add_argument("--test_days", type=int, default=15)
    p.add_argument("--step_days", type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--use_cache", action="store_true")
    args = p.parse_args()
    run(**vars(args))
