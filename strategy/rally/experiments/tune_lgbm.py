"""
LightGBM 超參數搜尋（Optuna），比照 strategy/rally/experiments/tune_xgb.py 的
做法（train/val/test 三段式切分避免污染、固定門檻評分、n_pred 太少判 0 分
避免退化解），只是換成 LGBMClassifier。

⚠️ subsample 在 LightGBM 的 sklearn API 裡要搭配 subsample_freq>=1 才會真的
生效（bagging_freq 預設 0，bagging_fraction 會被忽略、當作 1.0 處理）——
train.py::train_lgbm() 目前沒設 subsample_freq，代表現在的 subsample=0.8
從頭到尾都沒真的起作用過（跟 strategy/mkt/experiments/tune_lgbm.py 2026-07-21
發現的是同一個坑）。這裡的搜尋空間跟最終 refit 都補上 subsample_freq=1，
讓 subsample 這個維度是有意義的搜尋，不是浪費 trial。

跟 tune_xgb.py 共用 train/val/test 切分邏輯（_train_val_test_split）、
precision 計算（_up_precision/_precision_at_thresholds），不重複寫一份。

用法：
    python strategy/rally/experiments/tune_lgbm.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.rally.experiments.tune_xgb import (
    _MIN_PRED,
    _THRESHOLD,
    _precision_at_thresholds,
    _train_val_test_split,
    _up_precision,
)
from strategy.rally.features import FEATURES


def _fit_lgbm(train_df: pd.DataFrame, params: dict):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])
    return model


def _objective(trial, train_df: pd.DataFrame, val_df: pd.DataFrame):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,  # 見檔頭說明：不設這個 subsample 不會生效，不是搜尋維度
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    )
    model = _fit_lgbm(train_df, params)
    precision, n = _up_precision(model, val_df)
    trial.set_user_attr("n_pred", n)
    if n < _MIN_PRED:
        return 0.0
    return precision


def run(
    test_days: int = 45,
    val_days: int = 45,
    n_trials: int = 40,
    start_date: str = "2026-01-01",
    use_cache: bool = True,
):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df, val_df, test_df = _train_val_test_split(test_days, val_days, start_date, use_cache)
    print(
        f"train: {len(train_df):,} ({train_df['date'].min().strftime('%Y-%m-%d')}~"
        f"{train_df['date'].max().strftime('%Y-%m-%d')})  "
        f"val: {len(val_df):,} ({val_df['date'].min().strftime('%Y-%m-%d')}~"
        f"{val_df['date'].max().strftime('%Y-%m-%d')})  "
        f"test: {len(test_df):,} ({test_df['date'].min().strftime('%Y-%m-%d')}~"
        f"{test_df['date'].max().strftime('%Y-%m-%d')})"
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: _objective(t, train_df, val_df), n_trials=n_trials, show_progress_bar=True)

    print(f"\n=== Val集最佳結果（threshold={_THRESHOLD}）===")
    print(f"precision: {study.best_value:.2%}  預測數: {study.best_trial.user_attrs.get('n_pred')}")
    print(f"最佳參數: {study.best_params}")

    # 用完全沒被調參過程碰過的test集，重新算一次未經挑選污染的precision，
    # 才是可信的、能拿去跟 train.py::train_lgbm() 現有參數比較的數字。
    print("\n用最佳參數在train+val（trainval，即test之前全部資料）上重新訓練，評估test集...")
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    final_params = dict(study.best_params)
    final_params["subsample_freq"] = 1
    final_model = _fit_lgbm(trainval_df, final_params)
    precision, n = _up_precision(final_model, test_df, threshold=_THRESHOLD)
    total_up = int((test_df["target"] == 1).sum())
    print(f"=== Test集最終結果（threshold={_THRESHOLD}，未被調參污染）===")
    print(f"precision: {precision:.2%}  預測數: {n}  測試集實際漲樣本數: {total_up}")

    return study, final_model


if __name__ == "__main__":
    run()
