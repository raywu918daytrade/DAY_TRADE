"""
模型訓練 — LightGBM / XGBoost（只用 ORB 特徵，兩個模型互相比較用）
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.orb.features import FEATURES, load_features

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH_LGBM = _ROOT / "models/m1_orb_lgbm.pkl"
_MODEL_PATH_XGB = _ROOT / "models/m1_orb_xgb.pkl"


def _prepare_train_test(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """載入特徵並切分訓練/測試集（LGBM/XGB 共用）。"""
    print("特徵工程...")
    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵數: {len(FEATURES)}")
    print(f"  全天有效樣本: {len(df):,} 筆")

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    print(f"  日期區間: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

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
    return train_df, test_df


def train_lgbm(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """訓練 LightGBM 模型（只用 ORB 特徵）。"""
    import lightgbm as lgb

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_LGBM.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_LGBM)
    print(f"模型已存至 {_MODEL_PATH_LGBM}")
    return model


def train_xgb(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """訓練 XGBoost 模型（跟 LGBM 共用 FEATURES/切分方式，方便比較）。"""
    from xgboost import XGBClassifier

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date)

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
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_XGB.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_XGB)
    print(f"模型已存至 {_MODEL_PATH_XGB}")
    return model


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError("找不到 LGBM 模型，請先執行 train_lgbm()")
    return joblib.load(_MODEL_PATH_LGBM)


def load_model_xgb():
    if not _MODEL_PATH_XGB.exists():
        raise FileNotFoundError("找不到 XGB 模型，請先執行 train_xgb()")
    return joblib.load(_MODEL_PATH_XGB)
