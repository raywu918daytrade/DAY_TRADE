"""
Walk-forward驗證：訓練永遠在測試之前，依序切出多個(train,test)窗口，比較
「舊參數」（train.py::train_xgb() 現在用的原始參數）跟「新參數」
（experiments/tune_xgb.py 搜出來的候選）在每個窗口的「漲」precision，看新
參數是不是每個窗口都穩定比較好，還是只在單一窗口運氣好。

2026-07-21 背景：tune_xgb.py 的三段式切分已經抓到這組候選 val=62.12%→
test=44.29%（掉了17.8個百分點，幾乎跟瞎猜差不多），初步判斷是過擬合到 val
期間、沒有貼回 train_xgb()。這支腳本用來確認這個判斷在多個窗口下是否一致
（也可能candidate只是那一個test窗口運氣特別差，多窗口才看得出來）。

⚠️ 每個窗口都是「訓練資料只用這個窗口test_start之前的全部歷史」（expanding
window，不是固定寬度往前滑），理由：正式上線後模型永遠是用「當下能拿到的
全部歷史」重新訓練，不會刻意丟掉更早的資料，這裡跟上線情境保持一致。

start_date 預設 "2026-01-01"：跟現有 cache 範圍一致，expanding window 不需要
額外往回抓資料。如果要測 train_window_days（固定回看N天，rolling而非
expanding），第一個測試窗口往前推算的訓練起點可能超出這個範圍，需要先把
cache 往回擴充到更早的月份（見 features.py 的按月分區說明）。

用法：
    python strategy/rally/experiments/walk_forward_xgb.py
"""

import statistics
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.rally.experiments.tune_xgb import _fit_xgb, _precision_at_thresholds
from strategy.rally.features import FEATURES, load_features

# train.py::train_xgb() 現在實際在用的原始參數，當作比較基準。
_BASELINE_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    reg_lambda=1.0,
)

# tune_xgb.py 2026-07-21 搜出來的候選：val precision=62.12%/n=23519，
# test precision=44.29%/n=136152（val→test掉了17.8個百分點，初步判斷過擬合，
# 這裡用 walk-forward 多窗口確認）。
_CANDIDATE_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.013674939977956298,
    subsample=0.603928001148621,
    colsample_bytree=0.8222495158859673,
    min_child_weight=25,
    reg_lambda=1.9326921144453604,
    gamma=2.5044241331307036,
)

# rally 的機率分布比 mkt 集中在較低區間（見 validate.py 的 coverage_report，
# RFC/XGB 過了 0.65~0.70 之後訊號就很稀薄），門檻組取比較密集在這個範圍。
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
    """
    train_window_days（見檔頭說明）：預設 None＝每個窗口的訓練資料用「test_start
    之前的全部歷史」（expanding，跟正式上線情境一致）。設數字（例如90/180）
    則每個窗口都只留訓練資料裡最近這麼多天（rolling），用來驗證「訓練資料是
    不是塞越多歷史越好，還是舊regime的資料反而稀釋掉近期訊號」。
    """
    df = load_features(use_cache=use_cache, start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    windows = _walk_forward_windows(df, n_windows, window_days)

    # results[threshold] = list of per-window dict
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

        baseline_model = _fit_xgb(train_df, _BASELINE_PARAMS)
        candidate_model = _fit_xgb(train_df, _CANDIDATE_PARAMS)

        # predict_proba 各自只算一次，_THRESHOLDS 裡每個門檻共用，不用重跑模型。
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
