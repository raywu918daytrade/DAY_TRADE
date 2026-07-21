"""
Walk-forward驗證：LGBM版，邏輯完全比照 strategy/mkt/experiments/
walk_forward_xgb.py（訓練永遠在測試之前、expanding window、45天窗口、
同時掃 _THRESHOLDS 整組門檻），只是換成LGBMClassifier，理由跟
walk_forward_xgb.py檔頭一致：單一test集的高分可能只是運氣好，要看多個
窗口是否穩定贏過現有參數，才決定要不要貼回 train.py::train_lgbm()。

⚠️ _CANDIDATE_PARAMS 目前是佔位用（先複製 _BASELINE_PARAMS），還沒有真的
調過——要先跑一次 strategy/mkt/experiments/tune_lgbm.py，把它印出來的
「最佳參數」貼到下面 _CANDIDATE_PARAMS（記得補上 subsample_freq=1，理由見
tune_lgbm.py檔頭），這支腳本才有意義（不然就是拿baseline跟自己比）。

用法：
    python strategy/mkt/experiments/walk_forward_lgbm.py
"""

import statistics
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.mkt.experiments.tune_lgbm import _fit_lgbm
from strategy.mkt.experiments.tune_xgb import _precision_at_thresholds
from strategy.mkt.train import _prepare_data

# train.py::train_lgbm() 目前實際在用的參數，當作比較基準。⚠️ 這裡補了
# subsample_freq=1——原本的 train_lgbm() 沒設，subsample=0.8 其實從沒真的
# 生效過（見 tune_lgbm.py 檔頭說明），這裡如實模擬「假設subsample有生效」
# 的版本當基準，不是重現原本那個帶bug的行為。
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

# tune_lgbm.py 2026-07-21第二次搜出來的候選（修正learning_rate下限+高門檻
# 可行性檢查後）：val集precision=11.97%/n=309，test集precision=17.45%/
# n=361（val→test變好，跟XGB第二輪同方向，第一次那組learning_rate太低
# 導致0.7/0.8完全沒預測的問題應該已經修正，先跑walk-forward確認）。
_CANDIDATE_PARAMS = dict(
    n_estimators=800,
    num_leaves=101,
    max_depth=8,
    learning_rate=0.17376155767832735,
    min_child_samples=16,
    subsample=0.8372082477437949,
    subsample_freq=1,
    colsample_bytree=0.6699451641790991,
    reg_lambda=1.1052479063154046,
)

_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]


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
    n_windows: int = 5,
    window_days: int = 45,
    min_train_days: int = 60,
    use_cache: bool = True,
    train_window_days: int | None = None,
):
    """
    train_window_days：見 walk_forward_xgb.py::run() 的說明，用法完全一致
    （預設None＝expanding訓練資料；設數字則每個窗口只留最近這麼多天）。
    """
    df = _prepare_data(use_cache=use_cache)
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
        total_up = int((test_df["target"] == 2).sum())

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
                f"    threshold={thr:.1f}  舊參數: precision={p_base:>6.2%}(n={n_base:>4})  "
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
        print(f"\n-- threshold={thr:.1f} --")
        print(f"舊參數 precision： mean={statistics.mean(base_p):.2%}  std={statistics.pstdev(base_p):.2%}")
        print(f"新參數 precision： mean={statistics.mean(cand_p):.2%}  std={statistics.pstdev(cand_p):.2%}")
        print(f"新參數贏過舊參數的窗口數： {win}/{len(rows)}")
        if win < len(rows) * 0.6:
            print(f"⚠️ threshold={thr:.1f} 新參數沒有穩定贏過半數窗口，不建議在這個門檻使用新參數。")

    return results


if __name__ == "__main__":
    run()
