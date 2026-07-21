"""
模型訓練 — RandomForest / LightGBM / XGBoost（只用 ORB 特徵，三個模型互相比較用）
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.orb.config import DEFAULT_TEST_DAYS, MIN_VOL_MA20
from strategy.orb.features import FEATURES, apply_liquidity_filter, load_features, to_model_input

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH_RFC = _ROOT / "models/m1_orb_rfc.pkl"
_MODEL_PATH_LGBM = _ROOT / "models/m1_orb_lgbm.pkl"
_MODEL_PATH_XGB = _ROOT / "models/m1_orb_xgb.pkl"


def _prepare_train_test(
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """載入特徵並切分訓練/測試集（RFC/LGBM/XGB 共用）。

    start_date 會往下傳給 load_features()，只算/只讀 start_date 所在月份到
    db/m1 現有最新月份這段範圍，更早的月份完全不會被讀取或比對新鮮度——
    2019 年起的歷史 backfill 不會讓這裡的 cache 誤判過期（見 features.py
    load_features() 的按月分區說明）。

    use_cache: 見 features.load_features() 的說明——預設 True，逐月檢查
    新鮮度，新鮮的月份直接沿用 cache、過期的月份才重算。False 時不管每個月
    分區現在是什麼狀態，目標範圍內全部月份都重算，犧牲速度換正確性保證，
    只在你明確懷疑 cache 壞掉或改了 FEATURES 想強制全部重算時才用。
    """
    print("特徵工程...")
    df = load_features(start_date=start_date, use_cache=use_cache)
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵數: {len(FEATURES)}")
    print(f"  全天有效樣本: {len(df):,} 筆")

    n_before = len(df)
    df = apply_liquidity_filter(df)
    print(f"  流動性篩選（20日均量 ≥ {MIN_VOL_MA20:,}股）: {n_before:,} → {len(df):,} 筆")

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


def train_rfc(
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """訓練 RandomForest 模型（只用 ORB 特徵）。use_cache 見 _prepare_train_test()。"""
    train_df, test_df = _prepare_train_test(test_days, start_date, end_date, use_cache)
    X_train, X_test = to_model_input(train_df), to_model_input(test_df)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # target 不平衡（漲約35% vs 跌約65%），理由同
        # train_lgbm()/train_xgb() 的 scale_pos_weight 校準邏輯，RFC 用 sklearn
        # 原生的 class_weight 達到同樣效果
    )
    model.fit(X_train, train_df["target"])
    model._orb_train_cutoff = train_df["date"].max()  # 理由同 train_lgbm()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_RFC.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_RFC)
    print(f"模型已存至 {_MODEL_PATH_RFC}")
    return model


def train_lgbm(
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """訓練 LightGBM 模型（只用 ORB 特徵）。use_cache 見 _prepare_train_test()。"""
    import lightgbm as lgb

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date, use_cache)
    X_train, X_test = to_model_input(train_df), to_model_input(test_df)

    # target 不平衡（漲約35% vs 跌約65%），scale_pos_weight 校準機率分佈，
    # 讓 0.5 這個門檻對應到的召回率/精確率取捨點跟著調整（不會提升 AUC，
    # 只是重新校準機率，等同於用不同方式選門檻，但不用手動改門檻數字）。
    n_pos = (train_df["target"] == 1).sum()
    n_neg = (train_df["target"] == 0).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}（正樣本{n_pos:,} / 負樣本{n_neg:,}）")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,  # LightGBM sklearn API 沒設這個 subsample 不會生效（bagging_freq
        # 預設0，bagging_fraction 會被忽略當1.0處理）——見 strategy/rally/train.py 同一個坑
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, train_df["target"])
    # 記錄實際訓練切點，讓 validate.py 之後能檢查「驗證用的 test_days 是否
    # 跟訓練時不一致」——不一致會讓部分「測試集」其實是訓練時看過的資料，
    # 驗證指標虛高（2026-07-10 發現：訓練用 test_days=5、驗證卻傳 test_days=10，
    # AUC從0.65假摔／假漲到0.73，兩批資料重疊了快一半才是真正原因）。
    model._orb_train_cutoff = train_df["date"].max()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_LGBM.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_LGBM)
    print(f"模型已存至 {_MODEL_PATH_LGBM}")
    return model


def train_xgb(
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """訓練 XGBoost 模型（跟 LGBM 共用 FEATURES/切分方式，方便比較）。use_cache 見 _prepare_train_test()。"""
    from xgboost import XGBClassifier

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date, use_cache)
    X_train, X_test = to_model_input(train_df), to_model_input(test_df)

    # 同 train_lgbm() 的 scale_pos_weight 校準邏輯
    n_pos = (train_df["target"] == 1).sum()
    n_neg = (train_df["target"] == 0).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}（正樣本{n_pos:,} / 負樣本{n_neg:,}）")

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
        enable_categorical=True,  # hour 是 category dtype（見 features.to_model_input）
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, train_df["target"])
    # 記錄實際訓練切點，理由同 train_lgbm()
    model._orb_train_cutoff = train_df["date"].max()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_XGB.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_XGB)
    print(f"模型已存至 {_MODEL_PATH_XGB}")
    return model


def load_model_rfc():
    if not _MODEL_PATH_RFC.exists():
        raise FileNotFoundError("找不到 RFC 模型，請先執行 train_rfc()")
    return joblib.load(_MODEL_PATH_RFC)


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError("找不到 LGBM 模型，請先執行 train_lgbm()")
    return joblib.load(_MODEL_PATH_LGBM)


def load_model_xgb():
    if not _MODEL_PATH_XGB.exists():
        raise FileNotFoundError("找不到 XGB 模型，請先執行 train_xgb()")
    return joblib.load(_MODEL_PATH_XGB)


_MODEL_LOADERS = {
    "rfc": load_model_rfc,
    "lgbm": load_model_lgbm,
    "xgb": load_model_xgb,
}


def load_model_by_type(model_type: str):
    """依 config.MODEL_TYPE（"rfc"/"lgbm"/"xgb"）載入對應模型，
    run_backtest.py 跟 live.py 共用這支，切換模型只要改 config.py/.env 一個地方。"""
    if model_type not in _MODEL_LOADERS:
        raise ValueError(f"未知 model_type: {model_type!r}，可用: {list(_MODEL_LOADERS)}")
    return _MODEL_LOADERS[model_type]()
