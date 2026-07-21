"""
模型訓練 — RandomForest / XGBoost / LightGBM

三個模型共用 features.py 的 FEATURES 與 triple barrier 標籤，用同一份
_prepare_train_test() 切分全天訓練/測試集，方便公平比較。
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.rally.features import FEATURES, load_features

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "models/m1_rfc.pkl"
_MODEL_PATH_XGB = _ROOT / "models/m1_xgb.pkl"
_MODEL_PATH_LGBM = _ROOT / "models/m1_lgbm_breakout.pkl"


def train(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """
    訓練 RandomForest 模型。

    Parameters
    ----------
    test_days : int
        以最後 N 天為測試集（預設 10）。若指定 start_date/end_date 則忽略此參數。
    start_date : str
        訓練資料起日，格式 "YYYY-MM-DD"。留空表示不限制。會往下傳給
        load_features()，在算特徵之前就篩掉更早的資料，減少計算量（見
        features.load_features() 的說明），不是算完全部歷史才篩。
    end_date : str
        訓練資料迄日，格式 "YYYY-MM-DD"。留空表示不限制（只在算完特徵後篩選，
        不影響特徵計算量）。
    use_cache : bool
        False 時不管 cache 現在是什麼狀態，一律重新計算特徵（見 features.load_features()）。
    """
    print("特徵工程...")
    df = load_features(use_cache=use_cache, start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵: {FEATURES}")
    print(f"  全天有效樣本: {len(df):,} 筆")

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


def load_model_xgb():
    if not _MODEL_PATH_XGB.exists():
        raise FileNotFoundError(f"找不到 XGB 模型，請先執行 train_xgb()")
    return joblib.load(_MODEL_PATH_XGB)


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError(f"找不到 LGBM 模型，請先執行 train_lgbm()")
    return joblib.load(_MODEL_PATH_LGBM)


_MODEL_LOADERS = {
    "rfc": load_model,
    "xgb": load_model_xgb,
    "lgbm": load_model_lgbm,
}


def load_model_by_type(model_type: str):
    """依 config.MODEL_TYPE（"rfc"/"xgb"/"lgbm"）載入對應模型，
    run_backtest.py 跟 live.py 共用這支，切換模型只要改 config.py 一個地方。"""
    if model_type not in _MODEL_LOADERS:
        raise ValueError(f"未知 model_type: {model_type!r}，可用: {list(_MODEL_LOADERS)}")
    return _MODEL_LOADERS[model_type]()


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost / LightGBM 訓練（與 RFC 共用同一份 FEATURES 與標籤）
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare_train_test(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """載入特徵並切分全天訓練/測試集（三模型共用）。"""
    print("特徵工程...")
    df = load_features(use_cache=use_cache, start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵數: {len(FEATURES)}")
    print(f"  全天有效樣本: {len(df):,} 筆")

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    print(f"  日期區間: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

    if start_date and end_date:
        cutoff = df["date"].quantile(0.8)
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


def train_xgb(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """訓練 XGBoost 模型（與 RFC 共用 FEATURES）。"""
    from xgboost import XGBClassifier

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date, use_cache)

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


def train_lgbm(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """訓練 LightGBM 模型（與 RFC 共用 FEATURES）。"""
    import lightgbm as lgb

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date, use_cache)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,  # LightGBM sklearn API 沒設這個 subsample 不會生效（bagging_freq
        # 預設0，bagging_fraction 會被忽略當1.0處理）——2026-07-21 發現，見
        # experiments/tune_lgbm.py 的說明，補上才讓 subsample=0.8 真的有作用
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
