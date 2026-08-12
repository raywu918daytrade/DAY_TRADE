"""
ret5_pullback_ml 模型訓練 — LightGBM 三分類（0=止損 / 1=震盪 / 2=止盈）

== Main 模式 ==

train        訓練模型，存至 models/ret5_pullback_ml_lgbm.pkl
importance   顯示特徵重要性（讀已存模型，不重訓）
evaluate     混淆矩陣 + 分類報告（讀已存模型，不重訓）
confidence   止盈類別信心度門檻掃描（讀已存模型，不重訓）

用法：
    python -m strategy.ret5_pullback_ml.train train \\
        --start_date 2024-01-01 --end_date 2026-07-31
    python -m strategy.ret5_pullback_ml.train evaluate --use_cache \\
        --start_date 2024-01-01 --end_date 2026-07-31
    python -m strategy.ret5_pullback_ml.train confidence --use_cache \\
        --start_date 2024-01-01 --end_date 2026-07-31
    python -m strategy.ret5_pullback_ml.train importance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from strategy.ret5_pullback_ml.config import MODEL_DIR, MODEL_TYPE, events_cache_path
from strategy.ret5_pullback_ml.dataset import build_events
from strategy.ret5_pullback_ml.features import FEATURES

_MODEL_PATH_LGBM = MODEL_DIR / "ret5_pullback_ml_lgbm.pkl"
_TARGET_NAMES = ["止損", "震盪", "止盈"]


def _prepare_data(
    use_cache: bool = False,
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    use_tick_universe: bool = False,
) -> pd.DataFrame:
    cache_path = events_cache_path(start_date, end_date)
    if use_cache and cache_path.exists():
        print(f"讀取 cache [{cache_path.name}]")
        return pd.read_parquet(cache_path)

    df = build_events(
        start_date=start_date,
        end_date=end_date,
        use_tick_universe=use_tick_universe,
        attach_features=True,
    )
    if df.empty:
        print("無有效事件樣本")
        return df

    df["target"] = df["target"].astype(int)
    print(f"有效樣本: {len(df):,} 筆")
    print(f"標籤分佈:\n{(df['target'].value_counts(normalize=True) * 100).round(2)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"cache 已存至 {cache_path}")
    return df


def _split_data(
    test_days: int = 90,
    use_cache: bool = False,
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    use_tick_universe: bool = False,
):
    df = _prepare_data(
        use_cache=use_cache,
        start_date=start_date,
        end_date=end_date,
        use_tick_universe=use_tick_universe,
    )
    if df.empty:
        raise ValueError("無訓練資料")
    df = df.copy()
    df["trigger_ts"] = pd.to_datetime(df["trigger_ts"])
    cutoff = df["trigger_ts"].max() - pd.Timedelta(days=test_days)
    return df[df["trigger_ts"] <= cutoff], df[df["trigger_ts"] > cutoff]


def train_lgbm(
    test_days: int = 90,
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    use_cache: bool = False,
    use_tick_universe: bool = False,
):
    import lightgbm as lgb

    train_df, test_df = _split_data(
        test_days,
        use_cache=use_cache,
        start_date=start_date,
        end_date=end_date,
        use_tick_universe=use_tick_universe,
    )
    print(
        f"\n訓練: {len(train_df):,} ({train_df['trigger_ts'].min().strftime('%Y-%m-%d')} ~ "
        f"{train_df['trigger_ts'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"測試: {len(test_df):,} ({test_df['trigger_ts'].min().strftime('%Y-%m-%d')} ~ "
        f"{test_df['trigger_ts'].max().strftime('%Y-%m-%d')})"
        if len(test_df)
        else "測試: 0"
    )
    print(f"\n訓練集標籤分佈:\n{(train_df['target'].value_counts(normalize=True) * 100).round(2)}")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=30,
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
        print(f"\nAccuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
        print("\n混淆矩陣（列=實際，欄=預測，順序 止損/震盪/止盈）:")
        print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
        print("\n分類報告:")
        print(
            classification_report(
                test_df["target"], y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0
            )
        )
        # 純規則 vs 模型止盈門檻子集（mean pnl）
        proba = model.predict_proba(test_df[FEATURES])
        class_idx = {c: i for i, c in enumerate(model.classes_)}
        p_tp = proba[:, class_idx[2]] if 2 in class_idx else np.zeros(len(test_df))
        print("\n── test 純規則 vs 止盈機率門檻子集（做多 mean pnl）──")
        print(
            f"  全交易: n={len(test_df)}  mean={100 * test_df['pnl_pct'].mean():.3f}%  "
            f"TP={100 * (test_df['target'] == 2).mean():.1f}%  "
            f"SL={100 * (test_df['target'] == 0).mean():.1f}%"
        )
        for thr in (0.4, 0.5, 0.6, 0.7):
            sub = test_df[p_tp >= thr]
            if sub.empty:
                print(f"  p_tp≥{thr:.1f}: n=0")
                continue
            print(
                f"  p_tp≥{thr:.1f}: n={len(sub)}  mean={100 * sub['pnl_pct'].mean():.3f}%  "
                f"TP={100 * (sub['target'] == 2).mean():.1f}%  "
                f"SL={100 * (sub['target'] == 0).mean():.1f}%"
            )

    _MODEL_PATH_LGBM.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_LGBM)
    print(f"模型已存至 {_MODEL_PATH_LGBM}")
    return model


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError("找不到 LGBM 模型，請先執行 train")
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


_DEFAULT_THRESHOLDS = [None, 0.5, 0.6, 0.7]


def evaluate(
    model=None,
    test_days: int = 90,
    threshold: float | None | list[float | None] = _DEFAULT_THRESHOLDS,
    use_cache: bool = False,
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    use_tick_universe: bool = False,
):
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    _, test_df = _split_data(
        test_days,
        use_cache=use_cache,
        start_date=start_date,
        end_date=end_date,
        use_tick_universe=use_tick_universe,
    )
    if test_df.empty:
        print("無測試樣本")
        return test_df
    print(
        f"測試: {len(test_df):,} 筆 ({test_df['trigger_ts'].min().strftime('%Y-%m-%d')} ~ "
        f"{test_df['trigger_ts'].max().strftime('%Y-%m-%d')})"
    )
    thresholds = threshold if isinstance(threshold, list) else [threshold]
    for thr in thresholds:
        y_pred, thr_label = _predict_with_threshold(model, test_df, thr)
        print(f"\n{'=' * 60}\n{thr_label}\n{'=' * 60}")
        print(f"Accuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
        print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
        print(
            classification_report(
                test_df["target"], y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0
            )
        )
    return test_df


def confidence_report(
    model=None,
    test_days: int = 90,
    thresholds: list[float] | None = None,
    use_cache: bool = False,
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    use_tick_universe: bool = False,
):
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    _, test_df = _split_data(
        test_days,
        use_cache=use_cache,
        start_date=start_date,
        end_date=end_date,
        use_tick_universe=use_tick_universe,
    )
    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    test_df = test_df.copy()
    test_df["proba_tp"] = proba[:, class_idx[2]] if 2 in class_idx else 0.0

    total_actual = int((test_df["target"] == 2).sum())
    print("── 止盈（class=2）信心度門檻掃描 ──")
    print(f"  實際止盈樣本: {total_actual:,}")
    print(f"  {'門檻':>6}  {'預測數':>7}  {'猜中數':>7}  {'precision':>9}  {'mean%':>8}")
    for thr in thresholds:
        sub = test_df[test_df["proba_tp"] >= thr]
        n = len(sub)
        if n == 0:
            print(f"  {thr:.2f}  {0:>7,}  {0:>7,}  {'--':>9}  {'--':>8}")
            continue
        tp = int((sub["target"] == 2).sum())
        precision = tp / n
        print(
            f"  {thr:.2f}  {n:>7,}  {tp:>7,}  {precision * 100:>8.2f}%  "
            f"{100 * sub['pnl_pct'].mean():>7.3f}%"
        )
    return test_df


def feature_importance(model=None, top_n: int = 20):
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    print(f"\n── 特徵重要性（FEATURES: {FEATURES}）──")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:top_n]:
        print(f"  {name:24s}  {imp:.4f}")


def main(
    mode: str = "",
    test_days: int = 90,
    threshold: float | list[float] | None = None,
    model_type: str = "lgbm",
    use_cache: bool = False,
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    use_tick_universe: bool = False,
):
    """CLI 進入點，比照 strategy/vwap_ml/train.py::main()。

    兩種用法：
      1. VS Code 按 F5：__main__ 裡寫死 mode 等變數。
      2. 終端機帶參數：會覆蓋 __main__ 寫死的值。
    """
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="ret5_pullback_ml — LightGBM")
        parser.add_argument(
            "mode",
            nargs="?",
            default="train",
            choices=["train", "importance", "evaluate", "confidence"],
            help="執行模式（預設 train）",
        )
        parser.add_argument("--test_days", type=int, default=90)
        parser.add_argument("--threshold", type=float, nargs="*", default=None)
        parser.add_argument("--model_type", type=str, default="lgbm", choices=["lgbm"])
        parser.add_argument("--use_cache", action="store_true")
        parser.add_argument("--start_date", type=str, default="2024-01-01")
        parser.add_argument("--end_date", type=str, default="2026-07-31")
        parser.add_argument("--use_tick_universe", action="store_true")
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        threshold = args.threshold
        model_type = args.model_type
        use_cache = args.use_cache
        start_date = args.start_date
        end_date = args.end_date
        use_tick_universe = args.use_tick_universe
    elif not mode:
        mode = "train"

    kwargs = dict(
        test_days=test_days,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        use_tick_universe=use_tick_universe,
    )
    if mode == "train":
        _TRAIN_BY_TYPE[model_type](**kwargs)
    elif mode == "importance":
        feature_importance(model=_LOAD_MODEL_BY_TYPE[model_type]())
    elif mode == "evaluate":
        evaluate(
            model=_LOAD_MODEL_BY_TYPE[model_type](),
            threshold=threshold if threshold is not None else _DEFAULT_THRESHOLDS,
            **kwargs,
        )
    elif mode == "confidence":
        confidence_report(model=_LOAD_MODEL_BY_TYPE[model_type](), **kwargs)
    else:
        print(f"未知模式: {mode}，可用: train / importance / evaluate / confidence")


if __name__ == "__main__":
    """
    VS Code 按 F5（開發時常用）：直接改下面這幾行變數，不用打字。
    終端機帶參數會覆蓋掉下面寫死的值，見 main() 的說明。
    """
    mode = "train"  # train / importance / evaluate / confidence
    test_days = 90
    threshold = None  # mode="evaluate" 用得到；留 None = 用預設門檻列表
    model_type = "lgbm"
    use_cache = True
    start_date = "2024-01-01"
    end_date = "2026-07-31"
    use_tick_universe = False
    main(
        mode=mode,
        test_days=test_days,
        threshold=threshold,
        model_type=model_type,
        use_cache=use_cache,
        start_date=start_date,
        end_date=end_date,
        use_tick_universe=use_tick_universe,
    )
