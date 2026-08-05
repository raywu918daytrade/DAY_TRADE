"""
limitup_fade_ml 訓練 — LightGBM 三分類（0=止損 / 1=震盪 / 2=止盈）

用法：
    python -m strategy.limitup_fade_ml.train train --start_date 2024-01-01 --end_date 2026-07-31
    python -m strategy.limitup_fade_ml.train evaluate --use_cache --start_date 2024-01-01 --end_date 2026-07-31
    python -m strategy.limitup_fade_ml.train confidence --use_cache --start_date 2024-01-01 --end_date 2026-07-31
    python -m strategy.limitup_fade_ml.train importance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from strategy.limitup_fade_ml_my.config import MODEL_DIR, MODEL_TYPE, prepared_cache_path
from strategy.limitup_fade_ml_my.dataset import build_events
from strategy.limitup_fade_ml_my.features import FEATURES, make_features

_MODEL_PATH_LGBM = MODEL_DIR / "limitup_fade_ml_lgbm.pkl"
_TARGET_NAMES = ["止損", "震盪", "止盈"]
_DEFAULT_START = "2024-01-01"
_DEFAULT_END = "2026-07-31"


def _prepare_data(
    use_cache: bool = False,
    start_date: str | None = _DEFAULT_START,
    end_date: str | None = _DEFAULT_END,
) -> pd.DataFrame:
    cache_path = prepared_cache_path(start_date, end_date)
    if use_cache and cache_path.exists():
        print(f"讀取 cache [{cache_path.name}]", flush=True)
        return pd.read_parquet(cache_path)

    ev = build_events(start_date or _DEFAULT_START, end_date or _DEFAULT_END, with_labels=True)
    if ev.empty:
        return ev
    df = make_features(ev)
    df = df.dropna(subset=FEATURES + ["target"])
    df["target"] = df["target"].astype(int)
    print(f"有效樣本: {len(df):,}", flush=True)
    print(f"標籤分佈:\n{(df['target'].value_counts(normalize=True) * 100).round(2)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"cache → {cache_path}", flush=True)
    return df


def _split_data(
    test_frac: float = 0.2,
    use_cache: bool = False,
    start_date: str | None = _DEFAULT_START,
    end_date: str | None = _DEFAULT_END,
):
    df = _prepare_data(use_cache=use_cache, start_date=start_date, end_date=end_date)
    if df.empty:
        raise ValueError("無訓練資料")
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1.0 - test_frac))
    cut = max(1, min(cut, len(df) - 1)) if len(df) > 1 else len(df)
    return df.iloc[:cut], df.iloc[cut:]


def train_lgbm(
    test_frac: float = 0.2,
    start_date: str | None = _DEFAULT_START,
    end_date: str | None = _DEFAULT_END,
    use_cache: bool = False,
):
    import lightgbm as lgb

    train_df, test_df = _split_data(test_frac, use_cache, start_date, end_date)
    print(
        f"\n訓練: {len(train_df):,} ({train_df['date'].min()} ~ {train_df['date'].max()})",
        flush=True,
    )
    print(
        f"測試: {len(test_df):,} ({test_df['date'].min()} ~ {test_df['date'].max()})",
        flush=True,
    )
    print(f"訓練標籤:\n{(train_df['target'].value_counts(normalize=True) * 100).round(2)}", flush=True)

    model = lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    if len(test_df) > 0:
        y_pred = model.predict(test_df[FEATURES])
        print(f"\nAccuracy: {accuracy_score(test_df['target'], y_pred):.4f}", flush=True)
        print("混淆矩陣（止損/震盪/止盈）:", flush=True)
        print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]), flush=True)
        print(
            classification_report(
                test_df["target"], y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0
            ),
            flush=True,
        )

    _MODEL_PATH_LGBM.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_LGBM)
    print(f"模型 → {_MODEL_PATH_LGBM}", flush=True)
    return model


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError(f"找不到模型 {_MODEL_PATH_LGBM}，請先 train")
    return joblib.load(_MODEL_PATH_LGBM)


_LOAD_MODEL_BY_TYPE = {"lgbm": load_model_lgbm}
_TRAIN_BY_TYPE = {"lgbm": train_lgbm}
_model_cache: dict[str, object] = {}


def load_model_by_type(model_type: str):
    if model_type not in _LOAD_MODEL_BY_TYPE:
        raise ValueError(f"未知 model_type: {model_type!r}")
    if model_type not in _model_cache:
        _model_cache[model_type] = _LOAD_MODEL_BY_TYPE[model_type]()
    return _model_cache[model_type]


def _predict_with_threshold(model, test_df, threshold: float | None):
    if threshold is None:
        return model.predict(test_df[FEATURES]), "未設信心度門檻"
    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    p_tp = proba[:, class_idx[2]] if 2 in class_idx else np.zeros(len(test_df))
    y_pred = np.ones(len(test_df), dtype=int)
    y_pred[p_tp >= threshold] = 2
    return y_pred, f"止盈信心度門檻={threshold:.2f}"


def evaluate(
    model=None,
    test_frac: float = 0.2,
    threshold=None,
    use_cache: bool = False,
    start_date: str | None = _DEFAULT_START,
    end_date: str | None = _DEFAULT_END,
):
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    _, test_df = _split_data(test_frac, use_cache, start_date, end_date)
    thresholds = threshold if isinstance(threshold, list) else [threshold]
    if thresholds == [None]:
        thresholds = [None, 0.5, 0.6, 0.7, 0.8]
    for thr in thresholds:
        y_pred, thr_label = _predict_with_threshold(model, test_df, thr)
        print(f"\n{'=' * 60}\n{thr_label}\n{'=' * 60}", flush=True)
        print(f"Accuracy: {accuracy_score(test_df['target'], y_pred):.4f}", flush=True)
        print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]), flush=True)
        print(
            classification_report(
                test_df["target"], y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0
            ),
            flush=True,
        )


def confidence_report(
    model=None,
    test_frac: float = 0.2,
    thresholds: list[float] | None = None,
    use_cache: bool = False,
    start_date: str | None = _DEFAULT_START,
    end_date: str | None = _DEFAULT_END,
):
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    _, test_df = _split_data(test_frac, use_cache, start_date, end_date)
    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    test_df = test_df.copy()
    test_df["proba_tp"] = proba[:, class_idx[2]] if 2 in class_idx else 0.0
    total_actual = int((test_df["target"] == 2).sum())
    print("── 止盈（class=2）信心度門檻掃描 ──", flush=True)
    print(f"  實際止盈: {total_actual:,}", flush=True)
    print(f"  {'門檻':>6}  {'預測數':>7}  {'猜中':>7}  {'precision':>9}  {'recall':>6}", flush=True)
    for thr in thresholds:
        sub = test_df[test_df["proba_tp"] >= thr]
        n = len(sub)
        if n == 0:
            print(f"  {thr:.2f}  {0:>7,}  {0:>7,}  {'--':>9}  {'--':>6}", flush=True)
            continue
        tp = int((sub["target"] == 2).sum())
        print(
            f"  {thr:.2f}  {n:>7,}  {tp:>7,}  {tp / n * 100:>8.2f}%  "
            f"{(tp / total_actual * 100 if total_actual else 0):>5.2f}%",
            flush=True,
        )


def feature_importance(model=None, top_n: int = 20):
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    print(f"\n── 特徵重要性 ──", flush=True)
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:top_n]:
        print(f"  {name:24s}  {imp:.4f}", flush=True)


def main(
    mode: str = "",
    test_frac: float = 0.2,
    threshold: float | list[float] | None = None,
    model_type: str = "lgbm",
    use_cache: bool = False,
    start_date: str | None = _DEFAULT_START,
    end_date: str | None = _DEFAULT_END,
):
    """CLI / IDE 共用進入點（比照 breakout_retest_ml / vwap_ml）。

    兩種用法：
      1. IDE 按 F5：改 __main__ 裡寫死的變數，不帶 CLI 參數。
      2. 終端機：python -m strategy.limitup_fade_ml.train train --use_cache ...
    """
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="limitup_fade_ml — LightGBM")
        parser.add_argument(
            "mode",
            nargs="?",
            default="train",
            choices=["train", "importance", "evaluate", "confidence", "prepare"],
        )
        parser.add_argument("--test_frac", type=float, default=0.2)
        parser.add_argument("--threshold", type=float, nargs="*", default=None)
        parser.add_argument("--model_type", type=str, default="lgbm", choices=["lgbm"])
        parser.add_argument("--use_cache", action="store_true")
        parser.add_argument("--start_date", type=str, default=_DEFAULT_START)
        parser.add_argument("--end_date", type=str, default=_DEFAULT_END)
        args = parser.parse_args()
        mode = args.mode
        test_frac = args.test_frac
        threshold = args.threshold
        model_type = args.model_type
        use_cache = args.use_cache
        start_date = args.start_date
        end_date = args.end_date
    elif not mode:
        mode = "train"

    if mode == "prepare":
        _prepare_data(use_cache=False, start_date=start_date, end_date=end_date)
    elif mode == "train":
        _TRAIN_BY_TYPE[model_type](
            test_frac=test_frac,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
        )
    elif mode == "importance":
        feature_importance(model=_LOAD_MODEL_BY_TYPE[model_type]())
    elif mode == "evaluate":
        evaluate(
            model=_LOAD_MODEL_BY_TYPE[model_type](),
            test_frac=test_frac,
            threshold=threshold if threshold is not None else [None, 0.5, 0.6, 0.7, 0.8],
            use_cache=use_cache,
            start_date=start_date,
            end_date=end_date,
        )
    elif mode == "confidence":
        confidence_report(
            model=_LOAD_MODEL_BY_TYPE[model_type](),
            test_frac=test_frac,
            use_cache=use_cache,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        print(f"未知模式: {mode}", flush=True)


if __name__ == "__main__":
    # IDE 直接跑：改這裡即可，不必打 CLI
    mode = "train"  # train / prepare / evaluate / confidence / importance
    test_frac = 0.2
    threshold = None
    model_type = "lgbm"
    use_cache = False
    start_date = "2024-01-01"
    end_date = "2026-07-31"
    main(
        mode=mode,
        test_frac=test_frac,
        threshold=threshold,
        model_type=model_type,
        use_cache=use_cache,
        start_date=start_date,
        end_date=end_date,
    )
