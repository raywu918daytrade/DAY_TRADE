"""
XGBoost 超參數搜尋（Optuna），只優化「回歸」(class=0) 在固定門檻下的
precision——回歸是目前唯一決定要交易的類別（無訊號/延續都不進場，見
predict.py::_direction_probas() 2026-07-26 的決定），比照
strategy/mkt/experiments/tune_xgb.py 的做法（train/val/test三段式45天
切分避免污染、固定門檻評分、n_pred太少判0分避免退化解），只是把目標
類別從mkt的「漲」(class=2) 換成這裡的「回歸」(class=0)。

⚠️ 三段式切分，不是只切train/test：如果直接拿 train.py::_split_data() 的
test集當作Optuna調參的評分依據，等於用「之後要拿來報告最終績效」的那份
資料去挑參數，會讓最後在同一份test集上算出來的precision虛高（間接洩漏）。
這裡改成 train / val / test 三段：Optuna只在val上挑參數，最後才用從頭到尾
沒被調參過程碰過的test集，重新算一次「未經挑選污染」的precision，才是
可信的數字。

固定門檻=0.6評分（不是掃描）——2026-07-26 walk-forward 驗證（5個45天
窗口）顯示這個門檻的回歸precision平均67%（std±6.5%），是目前正式設定
（見 .env::VWAP_ML_THRESHOLD），拿它當調參評分依據，跟現行門檻直接可比。

用法：
    python -m strategy.vwap_ml.experiments.tune_xgb
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from strategy.vwap_ml.features import FEATURES
from strategy.vwap_ml.train import _prepare_data

_THRESHOLD = 0.6
_MIN_PRED = 15


def _train_val_test_split(
    test_days: int = 45,
    val_days: int = 45,
    use_cache: bool = True,
    start_date: str | None = "2024-01-01",
):
    """跟 train.py::_split_data() 的date cutoff邏輯一致，只是多切一段val
    出來（見檔頭說明，避免Optuna調參污染最終報告用的test集）。"""
    df = _prepare_data(use_cache=use_cache, start_date=start_date)
    test_cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    trainval_df = df[df["date"] <= test_cutoff]
    test_df = df[df["date"] > test_cutoff]

    val_cutoff = trainval_df["date"].max() - pd.Timedelta(days=val_days)
    train_df = trainval_df[trainval_df["date"] <= val_cutoff]
    val_df = trainval_df[trainval_df["date"] > val_cutoff]
    return train_df, val_df, test_df


def _revert_precision(model, df: pd.DataFrame, threshold: float = _THRESHOLD) -> tuple[float, int]:
    """回歸(class=0)在固定門檻下的precision，跟
    train.py::_predict_with_threshold() 的P(回歸)判法一致，但這裡只需要
    單一class=0的precision分數。回傳 (precision, 預測數)。"""
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    proba = model.predict_proba(df[FEATURES])[:, class_idx[0]]
    pred = proba >= threshold
    n = int(pred.sum())
    if n == 0:
        return 0.0, 0
    tp = int(((df["target"] == 0) & pred).sum())
    return tp / n, n


def _precision_at_thresholds(model, df: pd.DataFrame, thresholds: list[float]) -> dict[float, tuple[float, int]]:
    """跟 _revert_precision() 邏輯一樣，但 predict_proba() 只算一次、多個
    門檻共用。回傳 {threshold: (precision, 預測數)}。"""
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    proba = model.predict_proba(df[FEATURES])[:, class_idx[0]]
    is_revert = (df["target"] == 0).to_numpy()
    out = {}
    for threshold in thresholds:
        pred = proba >= threshold
        n = int(pred.sum())
        if n == 0:
            out[threshold] = (0.0, 0)
            continue
        tp = int((is_revert & pred).sum())
        out[threshold] = (tp / n, n)
    return out


_SIDE_DIRECTION = {"upper": "做空", "lower": "做多"}  # 上軌回歸=做空、下軌回歸=做多，見 predict.py::_direction_probas()


def print_direction_breakdown(model, test_df: pd.DataFrame, threshold: float = _THRESHOLD) -> None:
    """把回歸(class=0)在test集上的precision/recall依 trigger_side 拆開印
    （upper=上軌觸發=做空、lower=下軌觸發=做多），確認調完參數後多空兩側
    是不是都可靠，不是只看合併後的單一數字——如果其中一側樣本明顯拖累
    整體，合併數字會掩蓋這個問題（2026-07-26 討論：只交易回歸之後，
    「回歸」本身還是含多空兩個方向，需要分開檢視）。

    df 需已有 trigger_side 欄位（_prepare_data() 產生，不在 FEATURES 裡
    但沒被 dropna() 濾掉）。
    """
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    proba = model.predict_proba(test_df[FEATURES])[:, class_idx[0]]
    pred = proba >= threshold
    actual = (test_df["target"] == 0).to_numpy()
    for side in ("upper", "lower"):
        mask = (test_df["trigger_side"] == side).to_numpy()
        n = int((pred & mask).sum())
        tp = int((pred & mask & actual).sum())
        total_actual = int((actual & mask).sum())
        precision = tp / n if n else float("nan")
        recall = tp / total_actual if total_actual else float("nan")
        direction = _SIDE_DIRECTION[side]
        print(
            f"  {side}（{direction}）: 預測數={n:,}  猜中={tp:,}  precision={precision * 100:.2f}%  "
            f"recall={recall * 100:.2f}%  （實際回歸數={total_actual:,}）"
        )


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
    precision, n = _revert_precision(model, val_df)
    trial.set_user_attr("n_pred", n)
    if n < _MIN_PRED:
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
    print(f"precision: {study.best_value:.2%}  預測數: {study.best_trial.user_attrs.get('n_pred')}")
    print(f"最佳參數: {study.best_params}")

    # 用完全沒被調參過程碰過的test集，重新算一次未經挑選污染的precision，
    # 才是可信的、能拿去跟 train.py::train_xgb() 現有參數比較的數字。
    print("\n用最佳參數在train+val（trainval，即test之前全部資料）上重新訓練，評估test集...")
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = _fit_xgb(trainval_df, study.best_params)
    precision, n = _revert_precision(final_model, test_df, threshold=_THRESHOLD)
    total_revert = int((test_df["target"] == 0).sum())
    print(f"=== Test集最終結果（threshold={_THRESHOLD}，未被調參污染）===")
    print(f"precision: {precision:.2%}  預測數: {n}  測試集實際回歸樣本數: {total_revert}")
    print(f"\n── 依 trigger_side 拆多空（threshold={_THRESHOLD}）──")
    print_direction_breakdown(final_model, test_df, threshold=_THRESHOLD)

    return study, final_model


if __name__ == "__main__":
    run()
