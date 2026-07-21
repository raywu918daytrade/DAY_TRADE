"""
Walk-forward驗證：LGBM版，邏輯完全比照 strategy/rally/experiments/
walk_forward_xgb.py（訓練永遠在測試之前、expanding window、多窗口、同時掃
_THRESHOLDS 整組門檻），只是換成 LGBMClassifier。

2026-07-21 背景：experiments/tune_lgbm.py 搜出來的候選（val=62.83%→
test=60.11%，掉幅小、generalize得好）已經貼進 train.py::train_lgbm() 取代
原本隨便設的參數。這支腳本拿「原始參數」當基準、「已經貼進去的新參數」當
candidate，用多個 walk-forward 窗口確認這次替換是不是穩健的決定，不是只
靠 tune_lgbm.py 那一次 train/val/test 切分的運氣。

⚠️ _BASELINE_PARAMS 補了 subsample_freq=1——原始 train_lgbm() 沒設這個，
subsample=0.8 其實從沒真的生效過（見 tune_lgbm.py 檔頭說明），這裡如實模擬
「假設 subsample 有生效」的版本當基準，才是公平的比較對象，不是重現原本
帶 bug 的行為。

用法：
    python strategy/rally/experiments/walk_forward_lgbm.py
"""

import statistics
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.rally.experiments.tune_lgbm import _fit_lgbm
from strategy.rally.experiments.tune_xgb import _precision_at_thresholds
from strategy.rally.features import FEATURES, load_features

# train.py::train_lgbm() 原本的參數，補上 subsample_freq=1 當公平比較基準
# （理由見檔頭說明）。
_BASELINE_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)

# tune_lgbm.py 2026-07-21 搜出來、已經貼進 train_lgbm() 的候選：
# val precision=62.83%/n=30346，test precision=60.11%/n=54362。
_CANDIDATE_PARAMS = dict(
    n_estimators=200,
    num_leaves=101,
    max_depth=3,
    learning_rate=0.03657843946200939,
    min_child_samples=28,
    subsample=0.8600035132620599,
    subsample_freq=1,
    colsample_bytree=0.8319671794969992,
    reg_lambda=0.5187813029919602,
)

# rally 的機率分布比 mkt 集中在較低區間（見 validate.py 的 coverage_report，
# RFC/XGB/LGBM 過了 0.65~0.70 之後訊號就很稀薄），門檻組取比較密集在這個範圍。
_THRESHOLDS = [0.45, 0.5, 0.55, 0.6, 0.65]


def _walk_forward_windows(df: pd.DataFrame, n_windows: int, window_days: int) -> list[tuple]:
    """從最新日期往回切 n_windows 個連續、不重疊的 window_days 天測試窗口，
    回傳依時間由舊到新排序的 (test_start, test_end) list。"""
    max_date = df["date"].max()
    windows = []
    for i in range(n_windows):
        test_end = max_date - pd.Timedelta(days=window_days * i)
        test_start = test_end - pd.Timedelta(days=window_days)
        windows.append((test_start, test_end))
    return list(reversed(windows))


def run(
    n_windows: int = 4,
    window_days: int = 30,
    min_train_days: int = 30,
    start_date: str = "2026-01-01",
    use_cache: bool = True,
    train_window_days: int | None = None,
):
    """train_window_days：見 walk_forward_xgb.py::run() 的說明，用法完全一致
    （預設None＝expanding訓練資料；設數字則每個窗口只留最近這麼多天）。"""
    df = load_features(use_cache=use_cache, start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    windows = _walk_forward_windows(df, n_windows, window_days)

    results: dict[float, list[dict]] = {t: [] for t in _THRESHOLDS}

    for test_start, test_end in windows:
        train_df = df[df["date"] < test_start]
        if train_window_days is not None:
            train_start = test_start - pd.Timedelta(days=train_window_days)
            train_df = train_df[train_df["date"] >= train_start]
        test_df = df[(df["date"] >= test_start) & (df["date"] < test_end)]
        label = f"{test_start.date()}~{test_end.date()}"

        if train_df.empty or test_df.empty:
            print(f"[{label}] 跳過：訓練或測試資料為空")
            continue
        train_days_span = (train_df["date"].max() - train_df["date"].min()).days
        if train_days_span < min_train_days:
            print(f"[{label}] 跳過：訓練資料只橫跨{train_days_span}天，未達min_train_days={min_train_days}")
            continue

        baseline_model = _fit_lgbm(train_df, _BASELINE_PARAMS)
        candidate_model = _fit_lgbm(train_df, _CANDIDATE_PARAMS)

        base_by_thr = _precision_at_thresholds(baseline_model, test_df, _THRESHOLDS)
        cand_by_thr = _precision_at_thresholds(candidate_model, test_df, _THRESHOLDS)
        total_up = int((test_df["target"] == 1).sum())

        print(f"[{label}] 實際漲={total_up}")
        for thr in _THRESHOLDS:
            p_base, n_base = base_by_thr[thr]
            p_cand, n_cand = cand_by_thr[thr]
            results[thr].append(
                {
                    "window": label,
                    "actual_up": total_up,
                    "baseline_precision": p_base,
                    "baseline_n": n_base,
                    "candidate_precision": p_cand,
                    "candidate_n": n_cand,
                }
            )
            print(
                f"    threshold={thr:.2f}  舊參數: precision={p_base:>6.2%}(n={n_base:>4})  "
                f"新參數: precision={p_cand:>6.2%}(n={n_cand:>4})"
            )

    if not results[_THRESHOLDS[0]]:
        print("沒有任何窗口跑得動，檢查 n_windows/window_days/min_train_days 是否合理")
        return results

    print(f"\n=== 總結（{len(results[_THRESHOLDS[0]])}個窗口，各門檻分別統計） ===")
    for thr in _THRESHOLDS:
        rows = results[thr]
        base_p = [r["baseline_precision"] for r in rows]
        cand_p = [r["candidate_precision"] for r in rows]
        win = sum(1 for r in rows if r["candidate_precision"] > r["baseline_precision"])
        print(f"\n-- threshold={thr:.2f} --")
        print(f"舊參數 precision： mean={statistics.mean(base_p):.2%}  std={statistics.pstdev(base_p):.2%}")
        print(f"新參數 precision： mean={statistics.mean(cand_p):.2%}  std={statistics.pstdev(cand_p):.2%}")
        print(f"新參數贏過舊參數的窗口數： {win}/{len(rows)}")
        if win < len(rows) * 0.6:
            print(f"⚠️ threshold={thr:.2f} 新參數沒有穩定贏過半數窗口，不建議在這個門檻使用新參數。")

    return results


if __name__ == "__main__":
    run()
