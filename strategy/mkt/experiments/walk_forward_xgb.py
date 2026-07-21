"""
Walk-forward驗證：訓練永遠在測試之前，依序切出多個(train,test)窗口，比較
「舊參數」（train.py::train_xgb() 現在用的）跟「新參數」（tune_xgb.py 搜出
來的候選）在每個窗口的「漲」precision，看新參數是不是每個窗口都穩定比較
好，還是只在單一窗口運氣好（2026-07-21討論：tune_xgb.py那次val=7.57%/
test=20.24%落差太大，需要多窗口才能判斷穩不穩定）。

⚠️ 每個窗口都是「訓練資料只用這個窗口test_start之前的全部歷史」（expanding
window，不是固定寬度往前滑），理由：正式上線後模型永遠是用「當下能拿到的
全部歷史」重新訓練，不會刻意丟掉更早的資料，這裡跟上線情境保持一致。

2026-07-21 追加：一開始只驗證 threshold=0.6 這個單點，換上新參數貼回
train_xgb()後用 evaluate() 對最近一個窗口重新看整組門檻，發現0.5/0.6兩個
低門檻新參數持平或略贏，但0.7/0.8這兩個「高信心度」門檻新參數反而輸給
舊參數（0.8：51%→32%），只驗過0.6這一點看不出這個高門檻的退步，所以改成
每個窗口同時掃 _THRESHOLDS 這一整組門檻，各自比較新舊參數在每個門檻上
分別穩不穩定。

2026-07-21 再追加：window_days 從20天拉大到45天——原本20天（約13個交易日）
跑出來5個窗口precision在4.75%~22.16%大幅擺盪，std跟mean幾乎同量級，太容易
被單一窗口剛好遇到的市場狀況主導。45天（約30個交易日）換取比較穩定的
per-window估計，訓練資料到2024-08都有，n_windows=5、window_days=45合計只
佔用最近225天當測試，不會讓最早那個窗口的訓練資料不夠長。

用法：
    python strategy/mkt/experiments/walk_forward_xgb.py
"""

import statistics
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.mkt.experiments.tune_xgb import _fit_xgb, _precision_at_thresholds
from strategy.mkt.train import _prepare_data

# train.py::train_xgb() 目前實際在用的參數（2026-07-21第一輪tune_xgb.py，
# window=20天那次的候選，5個walk-forward窗口4個贏過最原始參數後貼回去的），
# 當作這一輪的比較基準——原始未調參的參數已經被這組取代，不用再跟它比。
_BASELINE_PARAMS = dict(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.10036535618307245,
    subsample=0.9225827336328982,
    colsample_bytree=0.5564310861998877,
    min_child_weight=74,
    reg_lambda=1.9262387751617027,
    gamma=0.3957709979902845,
)

# tune_xgb.py 2026-07-21第二輪（window改45天後）搜出來的候選：val集
# precision=11.30%/n=584，test集precision=16.42%/n=597（val/test落差只有
# 5個百分點，比第一輪20天窗口的13個百分點落差小很多，估計比較可信）。要
# 驗證別組候選，改這裡就好，不用動下面run()的邏輯。
_CANDIDATE_PARAMS = dict(
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
    train_window_days（2026-07-21討論）：預設None＝每個窗口的訓練資料用
    「test_start之前的全部歷史」（expanding，見檔頭說明，跟正式上線情境
    一致）。設數字（例如365/540）則每個窗口都只留訓練資料裡最近這麼多天
    （rolling），用來驗證「訓練資料是不是塞越多歷史越好」，還是舊regime
    的資料反而稀釋掉近期訊號——把這個參數設不同值分別跑一次run()，比較
    哪種訓練資料範圍在多個窗口下precision比較穩定/比較高。
    """
    df = _prepare_data(use_cache=use_cache)
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
