"""
XGBoost 超參數搜尋（Optuna），只優化「漲」(class=2) 在固定門檻下的precision
（2026-07-21討論：day_ret_vs_idx日K特徵實驗失敗後，改往「調整既有模型的
超參數」這個方向試，因為目前3個模型都還在用最初隨便設的參數，從沒針對
mkt這組資料調過）。

獨立腳本，不動 train.py::train_xgb() 的預設參數——這裡只是拿去試、找出
比較好的組合，覺得有幫助才手動貼回 train_xgb()，不會自動覆寫（比照
strategy/mkt/experiments/ 其他驗證腳本的慣例：一次性假設驗證不混進核心
pipeline檔案）。

⚠️ 三段式切分，不是只切train/test：如果直接拿 train.py::_split_data() 的
test集當作Optuna調參的評分依據，等於用「之後要拿來報告最終績效」的那份
資料去挑參數，會讓最後在同一份test集上算出來的precision虛高（間接洩漏）。
這裡改成 train / val / test 三段：Optuna只在val上挑參數，最後才用從頭到尾
沒被調參過程碰過的test集，重新算一次「未經挑選污染」的precision，才是
可信的數字。

固定門檻=0.6評分（不是掃描），避免把「門檻」也變成搜尋維度之一，模糊掉
真正想調的是模型超參數本身；n_pred太少（模型學會「乾脆都不猜漲」來逃避
低precision這種退化解）時直接判0分，逼模型至少要猜出一定數量才有意義。

2026-07-21：test_days/val_days 從20天拉大到45天——用20天跑walk_forward_xgb.py
時發現5個窗口的precision在4.75%~22.16%大幅擺盪（std跟mean同量級），代表
20天（約13個交易日）太短，容易被單一窗口剛好遇到的市場狀況主導，不是穩定
估計。45天（約30個交易日）換取比較穩定的val/test precision，訓練資料到
2024-08都有，拉大不會吃緊。

用法：
    python strategy/mkt/experiments/tune_xgb.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from strategy.mkt.features import FEATURES
from strategy.mkt.train import _prepare_data

_THRESHOLD = 0.6
_MIN_PRED = 20


def _train_val_test_split(test_days: int = 45, val_days: int = 45, use_cache: bool = True):
    """跟 train.py::_split_data() 的date cutoff邏輯一致，只是多切一段val出來
    （見檔頭說明，避免Optuna調參污染最終報告用的test集）。"""
    df = _prepare_data(use_cache=use_cache)
    test_cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    trainval_df = df[df["date"] <= test_cutoff]
    test_df = df[df["date"] > test_cutoff]

    val_cutoff = trainval_df["date"].max() - pd.Timedelta(days=val_days)
    train_df = trainval_df[trainval_df["date"] <= val_cutoff]
    val_df = trainval_df[trainval_df["date"] > val_cutoff]
    return train_df, val_df, test_df


def _up_precision(model, df: pd.DataFrame, threshold: float = _THRESHOLD) -> tuple[float, int]:
    """漲(class=2)在固定門檻下的precision，跟train.py::_predict_with_threshold()
    的P(漲)判法一致，但這裡只需要單一class=2的precision分數，不用完整3類
    判法。回傳 (precision, 預測數) ——預測數太少時呼叫端要自己判斷是否為
    退化解。"""
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    proba = model.predict_proba(df[FEATURES])[:, class_idx[2]]
    pred = proba >= threshold
    n = int(pred.sum())
    if n == 0:
        return 0.0, 0
    tp = int(((df["target"] == 2) & pred).sum())
    return tp / n, n


def _precision_at_thresholds(model, df: pd.DataFrame, thresholds: list[float]) -> dict[float, tuple[float, int]]:
    """跟 _up_precision() 邏輯一樣，但 predict_proba() 只算一次、多個門檻共用，
    給 walk_forward_xgb.py 同時掃好幾個門檻用（不用每個門檻各自重跑一次
    predict_proba）。回傳 {threshold: (precision, 預測數)}。"""
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    proba = model.predict_proba(df[FEATURES])[:, class_idx[2]]
    is_up = (df["target"] == 2).to_numpy()
    out = {}
    for threshold in thresholds:
        pred = proba >= threshold
        n = int(pred.sum())
        if n == 0:
            out[threshold] = (0.0, 0)
            continue
        tp = int((is_up & pred).sum())
        out[threshold] = (tp / n, n)
    return out


def _fit_xgb(train_df: pd.DataFrame, params: dict) -> XGBClassifier:
    sample_weight = compute_sample_weight("balanced", train_df["target"])
    model = XGBClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(train_df[FEATURES], train_df["target"], sample_weight=sample_weight)
    return model


def _objective(trial, train_df: pd.DataFrame, val_df: pd.DataFrame):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 5, 100),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )
    model = _fit_xgb(train_df, params)
    precision, n = _up_precision(model, val_df)
    trial.set_user_attr("n_pred", n)
    if n < _MIN_PRED:
        return 0.0
    return precision


def run(test_days: int = 45, val_days: int = 45, n_trials: int = 40, use_cache: bool = True):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df, val_df, test_df = _train_val_test_split(test_days, val_days, use_cache=use_cache)
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
    # 才是可信的、能拿去跟 train.py::train_xgb() 現有參數比較的數字。
    print("\n用最佳參數在train+val（trainval，即test之前全部資料）上重新訓練，評估test集...")
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = _fit_xgb(trainval_df, study.best_params)
    precision, n = _up_precision(final_model, test_df, threshold=_THRESHOLD)
    total_up = int((test_df["target"] == 2).sum())
    print(f"=== Test集最終結果（threshold={_THRESHOLD}，未被調參污染）===")
    print(f"precision: {precision:.2%}  預測數: {n}  測試集實際漲樣本數: {total_up}")

    return study, final_model


if __name__ == "__main__":
    run()
