"""
LightGBM 超參數搜尋（Optuna），邏輯完全比照
strategy/vwap_ml/experiments/tune_xgb.py（訓練/驗證/測試三段式45天切分
避免污染、固定門檻=0.6評分「回歸」class=0的precision、預測數太少判0分
避免退化解），只是換成 LGBMClassifier。LGBMClassifier 原生支援多分類的
class_weight="balanced"（不用像XGB那樣自己算sample_weight），
train.py::train_lgbm() 已經在用，這裡沿用同一套。

⚠️ subsample 在 LightGBM 的 sklearn API 裡要搭配 subsample_freq>=1 才會真的
生效（bagging_freq預設0，bagging_fraction會被忽略、當作1.0處理），比照
strategy/mkt/experiments/tune_lgbm.py 檔頭的同樣提醒——搜尋空間跟最終refit
都補上 subsample_freq=1，讓 subsample 這個維度是有意義的搜尋。

跟 tune_xgb.py 共用 train/val/test 切分邏輯（_train_val_test_split）、
precision計算（_revert_precision/_precision_at_thresholds），不重複寫
一份。

用法：
    python -m strategy.vwap_ml.experiments.tune_lgbm
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.vwap_ml.experiments.tune_xgb import (
    _MIN_PRED,
    _THRESHOLD,
    _precision_at_thresholds,
    _revert_precision,
    _train_val_test_split,
    print_direction_breakdown,
)
from strategy.vwap_ml.features import FEATURES

_HIGH_CHECK_THRESHOLD = 0.7  # 高信心度門檻可行性檢查，比照mkt的同名參數
_MIN_HIGH_PRED = 5  # 該門檻至少要有這麼多筆預測，不然視為「機率衝不上去」判0分


def _fit_lgbm(train_df: pd.DataFrame, params: dict):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
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
    by_thr = _precision_at_thresholds(model, val_df, [_THRESHOLD, _HIGH_CHECK_THRESHOLD])
    precision, n = by_thr[_THRESHOLD]
    _, n_high = by_thr[_HIGH_CHECK_THRESHOLD]
    trial.set_user_attr("n_pred", n)
    trial.set_user_attr("n_pred_high", n_high)
    if n < _MIN_PRED or n_high < _MIN_HIGH_PRED:
        return 0.0
    return precision


def run(
    test_days: int = 45,
    val_days: int = 45,
    n_trials: int = 40,
    use_cache: bool = True,
    start_date: str | None = "2024-01-01",
):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df, val_df, test_df = _train_val_test_split(test_days, val_days, use_cache=use_cache, start_date=start_date)
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

    print(f"\n=== Val集最佳結果（threshold={_THRESHOLD}，回歸/class=0）===")
    print(
        f"precision: {study.best_value:.2%}  預測數: {study.best_trial.user_attrs.get('n_pred')}  "
        f"（門檻{_HIGH_CHECK_THRESHOLD}預測數: {study.best_trial.user_attrs.get('n_pred_high')}）"
    )
    print(f"最佳參數: {study.best_params}")

    # 用完全沒被調參過程碰過的test集，重新算一次未經挑選污染的precision。
    print("\n用最佳參數在train+val（trainval，即test之前全部資料）上重新訓練，評估test集...")
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = _fit_lgbm(trainval_df, {**study.best_params, "subsample_freq": 1})
    precision, n = _revert_precision(final_model, test_df, threshold=_THRESHOLD)
    total_revert = int((test_df["target"] == 0).sum())
    print(f"=== Test集最終結果（threshold={_THRESHOLD}，未被調參污染）===")
    print(f"precision: {precision:.2%}  預測數: {n}  測試集實際回歸樣本數: {total_revert}")
    print(f"\n── 依 trigger_side 拆多空（threshold={_THRESHOLD}）──")
    print_direction_breakdown(final_model, test_df, threshold=_THRESHOLD)

    return study, final_model


if __name__ == "__main__":
    run()
