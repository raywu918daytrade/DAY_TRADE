"""
vwap_ml walk-forward 驗證 — 回歸(class=0)在多個測試窗口下的 precision/
recall 穩定性驗證，一次性驗證腳本，不是核心pipeline的一部分（比照
strategy/mkt/experiments/ 的慣例）。

2026-07-26 討論：單一測試窗口（例如最近30天）測到的 precision 可能只是
那段期間市場狀況剛好配合，不能只看一個窗口就下結論——同樣的教訓見
strategy/mkt/README.md「單一窗口的評估數字常常比多窗口平均高很多」的
說明。這裡跑 N 個連續的 45 天窗口（expanding window：每個窗口的訓練集
都是「這個窗口開始之前的全部資料」，窗口越晚訓練集越大），各自訓練＋
測試，只關注「回歸」(class=0) 這個唯一會拿來交易的類別，看 precision
在不同窗口下是不是穩定，不是像 mkt 那樣同時看三個類別。

用法：
    python -m strategy.vwap_ml.experiments.walk_forward
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.vwap_ml.features import FEATURES
from strategy.vwap_ml.train import _prepare_data

_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]


def _fit(train_df: pd.DataFrame):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])
    return model


def run(
    start_date: str = "2024-01-01",
    window_days: int = 45,
    n_windows: int = 5,
    min_train_rows: int = 5000,
    use_cache: bool = True,
):
    """
    expanding window：從最新資料往回切 n_windows 個不重疊的 window_days
    天窗口當測試集，每個窗口的訓練集都是「該窗口開始之前的全部資料」
    （越早的窗口訓練集越小，越晚的窗口訓練集越大，模擬「實際部署時只能
    用過去資料訓練」的情境）。

    只看「回歸」(class=0) 的 precision/recall——這是目前唯一決定要交易
    的類別（見討論：無訊號/延續現階段都不進場），不用像 mkt 那樣同時報
    三個類別。

    min_train_rows：訓練集筆數低於這個門檻的窗口直接跳過並警告（太早期
    的窗口可能資料量不足，precision數字統計上不穩定，見
    strategy/mkt/README.md 對這個問題的說明）。
    """
    df = _prepare_data(use_cache=use_cache, start_date=start_date)
    df = df.sort_values("date").reset_index(drop=True)

    max_date = df["date"].max()
    rows = []
    for i in range(n_windows):
        test_end = max_date - pd.Timedelta(days=window_days * i)
        test_start = test_end - pd.Timedelta(days=window_days)
        train_df = df[df["date"] < test_start]
        test_df = df[(df["date"] >= test_start) & (df["date"] < test_end)]

        window_label = f"{test_start.date()}~{test_end.date()}"
        if len(train_df) < min_train_rows or test_df.empty:
            print(f"窗口 {i + 1}（{window_label}）資料量不足，跳過（train={len(train_df):,}, test={len(test_df):,}）")
            continue

        model = _fit(train_df)
        proba = model.predict_proba(test_df[FEATURES])
        class_idx = {c: idx for idx, c in enumerate(model.classes_)}
        p_revert = proba[:, class_idx[0]]
        actual_revert = (test_df["target"] == 0).to_numpy()
        total_actual = int(actual_revert.sum())

        print(
            f"\n窗口 {i + 1}: {window_label}（train={len(train_df):,}, test={len(test_df):,}, "
            f"實際回歸={total_actual:,}）"
        )
        for thr in _THRESHOLDS:
            pred = p_revert >= thr
            n = int(pred.sum())
            tp = int((pred & actual_revert).sum())
            precision = tp / n if n else float("nan")
            recall = tp / total_actual if total_actual else float("nan")
            rows.append(
                {"window": window_label, "threshold": thr, "n": n, "tp": tp, "precision": precision, "recall": recall}
            )
            print(f"  門檻{thr:.1f}: 預測數={n:>5,}  猜中={tp:>5,}  precision={precision * 100:6.2f}%  recall={recall * 100:6.2f}%")

    if not rows:
        print("\n沒有任何窗口跑出結果（可能是資料量太少或window_days/n_windows設定不合理）")
        return pd.DataFrame()

    results_df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}\n各門檻在 {results_df['window'].nunique()} 個窗口的平均（回歸/class=0）\n{'=' * 60}")
    summary = results_df.groupby("threshold").agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        n_mean=("n", "mean"),
    )
    print(summary.round(4))
    return results_df


if __name__ == "__main__":
    run()
