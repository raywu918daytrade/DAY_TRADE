"""
Walk-forward驗證：流動性過濾 top_n=100（現在的正式設定）vs top_n=300，
哪個「漲」precision比較好/比較穩（2026-07-22討論）。

⚠️ 跟 walk_forward_xgb.py 不同的比較維度：那支比較的是「同一份top_n=100
資料，兩組不同超參數」；這支固定用同一組已經驗證過的XGB超參數（見
_XGB_PARAMS，跟 strategy/mkt/train.py::train_xgb() 現在用的一致），只改
top_n本身——這樣precision差異才能歸因給「流動性過濾範圍」，不會混進
「順便換了參數」這個變數。

top_n=100/300 各自有獨立cache（cache/mkt_prepared.parquet /
cache/mkt_prepared_top300.parquet，見 strategy/mkt/train.py::
_cache_path_for() 的說明），不會互相覆蓋、也不影響正式pipeline用的
top_n=100那份。

用法：
    python strategy/mkt/experiments/walk_forward_top_n.py
"""

import statistics
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.mkt.experiments.tune_xgb import _fit_xgb, _precision_at_thresholds
from strategy.mkt.train import _prepare_data

# strategy/mkt/train.py::train_xgb() 現在正式在用的參數（2026-07-21第二輪
# walk-forward驗證過的那組），固定不變，只換 top_n。
_XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.12770505395846093,
    subsample=0.9541586311700712,
    colsample_bytree=0.9978337571109724,
    min_child_weight=22,
    reg_lambda=1.847987791882633,
    gamma=0.2824283269109099,
)

_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]


def _walk_forward_windows(df: pd.DataFrame, n_windows: int, window_days: int) -> list[tuple]:
    max_date = df["date"].max()
    windows = []
    for i in range(n_windows):
        test_end = max_date - pd.Timedelta(days=window_days * i)
        test_start = test_end - pd.Timedelta(days=window_days)
        windows.append((test_start, test_end))
    return list(reversed(windows))


def run(n_windows: int = 5, window_days: int = 45, min_train_days: int = 60, use_cache: bool = True):
    print("載入 top_n=100（現行設定）資料...")
    df_100 = _prepare_data(top_n=100, use_cache=use_cache)
    print("載入 top_n=300 資料...")
    df_300 = _prepare_data(top_n=300, use_cache=use_cache)

    # 兩邊用同一套時間窗口（以df_100的日期範圍為準——兩邊來源m1相同，日期
    # 範圍理論上一致，只有股票數不同）。
    windows = _walk_forward_windows(df_100, n_windows, window_days)

    results: dict[float, list[dict]] = {t: [] for t in _THRESHOLDS}

    for test_start, test_end in windows:
        label = f"{test_start.date()}~{test_end.date()}"

        train_100 = df_100[df_100["date"] < test_start]
        test_100 = df_100[(df_100["date"] >= test_start) & (df_100["date"] < test_end)]
        train_300 = df_300[df_300["date"] < test_start]
        test_300 = df_300[(df_300["date"] >= test_start) & (df_300["date"] < test_end)]

        if train_100.empty or test_100.empty or train_300.empty or test_300.empty:
            print(f"[{label}] 跳過：訓練或測試資料為空")
            continue
        train_days_span = (train_100["date"].max() - train_100["date"].min()).days
        if train_days_span < min_train_days:
            print(f"[{label}] 跳過：訓練資料只橫跨{train_days_span}天，未達min_train_days={min_train_days}")
            continue

        model_100 = _fit_xgb(train_100, _XGB_PARAMS)
        model_300 = _fit_xgb(train_300, _XGB_PARAMS)

        by_thr_100 = _precision_at_thresholds(model_100, test_100, _THRESHOLDS)
        by_thr_300 = _precision_at_thresholds(model_300, test_300, _THRESHOLDS)
        actual_100 = int((test_100["target"] == 2).sum())
        actual_300 = int((test_300["target"] == 2).sum())

        print(f"[{label}] 實際漲：top100={actual_100}  top300={actual_300}")
        for thr in _THRESHOLDS:
            p100, n100 = by_thr_100[thr]
            p300, n300 = by_thr_300[thr]
            results[thr].append(
                {
                    "window": label,
                    "top100_precision": p100,
                    "top100_n": n100,
                    "top300_precision": p300,
                    "top300_n": n300,
                }
            )
            print(
                f"    threshold={thr:.1f}  top100: precision={p100:>6.2%}(n={n100:>4})  "
                f"top300: precision={p300:>6.2%}(n={n300:>4})"
            )

    if not results[_THRESHOLDS[0]]:
        print("沒有任何窗口跑得動，檢查 n_windows/window_days/min_train_days 是否合理")
        return results

    print(f"\n=== 總結（{len(results[_THRESHOLDS[0]])}個窗口，各門檻分別統計） ===")
    for thr in _THRESHOLDS:
        rows = results[thr]
        p100_list = [r["top100_precision"] for r in rows]
        p300_list = [r["top300_precision"] for r in rows]
        win = sum(1 for r in rows if r["top300_precision"] > r["top100_precision"])
        print(f"\n-- threshold={thr:.1f} --")
        print(f"top100 precision： mean={statistics.mean(p100_list):.2%}  std={statistics.pstdev(p100_list):.2%}")
        print(f"top300 precision： mean={statistics.mean(p300_list):.2%}  std={statistics.pstdev(p300_list):.2%}")
        print(f"top300贏過top100的窗口數： {win}/{len(rows)}")
        if win < len(rows) * 0.6:
            print(f"⚠️ threshold={thr:.1f} top300沒有穩定贏過半數窗口。")

    return results


if __name__ == "__main__":
    run()
