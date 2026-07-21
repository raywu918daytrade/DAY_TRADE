"""
XGBoost 超參數搜尋（Optuna）— 比照 strategy/mkt/experiments/tune_xgb.py 的做法
（train/val/test 三段式切分避免污染、固定門檻評分、n_pred 太少判 0 分避免
退化解），train.py::train_xgb() 現在的參數是一開始隨便設的，從沒針對這組
資料調過。

跟 mkt 版的差異：rally 是二分類（target=1漲/0跌），predict_proba()[:, 1]
直接拿 P(漲)，不用像 mkt 三分類那樣查 class_idx；也沒用 sample_weight=
"balanced"，維持跟 train.py::train_xgb() 現有做法一致，不在調參腳本裡引入
額外差異。

獨立腳本，不動 train.py::train_xgb() 的預設參數——這裡只是拿去試、找出比較
好的組合，覺得有幫助才手動貼回 train_xgb()，不會自動覆寫（比照 experiments/
其他驗證腳本的慣例：一次性假設驗證不混進核心 pipeline 檔案）。

⚠️ 三段式切分，不是只切 train/test：如果直接拿 train.py 的 test 集當作
Optuna 調參的評分依據，等於用「之後要拿來報告最終績效」的那份資料去挑參數，
會讓最後在同一份 test 集上算出來的 precision 虛高（間接洩漏）。這裡改成
train / val / test 三段：Optuna 只在 val 上挑參數，最後才用從頭到尾沒被
調參過程碰過的 test 集，重新算一次「未經挑選污染」的 precision，才是可信
的數字。

start_date 預設 "2026-01-01"：跟現在 entry.py 訓練用的範圍一致（2026-07-21
改成按月分區 cache 之後的安全範圍，見 features.py 的說明），避免調參腳本
不小心觸發從 db/m1 最早月份（2024-08）開始的大量重算。

用法：
    python strategy/rally/experiments/tune_xgb.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
from xgboost import XGBClassifier

from strategy.rally.features import FEATURES, load_features

_THRESHOLD = 0.6
_MIN_PRED = 20


def _train_val_test_split(
    test_days: int = 45,
    val_days: int = 45,
    start_date: str = "2026-01-01",
    use_cache: bool = True,
):
    """train/val/test 三段式切分：Optuna 只在 val 上挑參數，最後用完全沒被
    調參碰過的 test 集重新算一次可信的 precision（理由見檔頭說明）。"""
    df = load_features(use_cache=use_cache, start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])

    test_cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    trainval_df = df[df["date"] <= test_cutoff]
    test_df = df[df["date"] > test_cutoff]

    val_cutoff = trainval_df["date"].max() - pd.Timedelta(days=val_days)
    train_df = trainval_df[trainval_df["date"] <= val_cutoff]
    val_df = trainval_df[trainval_df["date"] > val_cutoff]
    return train_df, val_df, test_df


def _up_precision(model, df: pd.DataFrame, threshold: float = _THRESHOLD) -> tuple[float, int]:
    """target=1（漲）在固定門檻下的 precision。回傳 (precision, 預測數)——
    預測數太少時呼叫端要自己判斷是否為退化解。"""
    proba = model.predict_proba(df[FEATURES])[:, 1]
    pred = proba >= threshold
    n = int(pred.sum())
    if n == 0:
        return 0.0, 0
    tp = int(((df["target"] == 1) & pred).sum())
    return tp / n, n


def _precision_at_thresholds(model, df: pd.DataFrame, thresholds: list[float]) -> dict[float, tuple[float, int]]:
    """跟 _up_precision() 邏輯一樣，但 predict_proba() 只算一次、多個門檻共用，
    給 walk_forward_xgb.py 之類要同時掃好幾個門檻的腳本用。"""
    proba = model.predict_proba(df[FEATURES])[:, 1]
    is_up = (df["target"] == 1).to_numpy()
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
    model = XGBClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(train_df[FEATURES], train_df["target"])
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
    # 才是可信的、能拿去跟 train.py::train_xgb() 現有參數比較的數字。
    print("\n用最佳參數在train+val（trainval，即test之前全部資料）上重新訓練，評估test集...")
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = _fit_xgb(trainval_df, study.best_params)
    precision, n = _up_precision(final_model, test_df, threshold=_THRESHOLD)
    total_up = int((test_df["target"] == 1).sum())
    print(f"=== Test集最終結果（threshold={_THRESHOLD}，未被調參污染）===")
    print(f"precision: {precision:.2%}  預測數: {n}  測試集實際漲樣本數: {total_up}")

    return study, final_model


if __name__ == "__main__":
    run()
