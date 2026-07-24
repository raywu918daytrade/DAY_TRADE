"""
Walk-forward驗證：目前正式門檻(p90附近，0.00748) vs 更高的ATR5門檻
（p95/p97/p99），固定top_n=300（2026-07-25正式設定），哪個「漲」precision
比較好/比較穩（2026-07-25討論）。

跟 walk_forward_top_n.py 同樣的比較邏輯：固定同一組已驗證過的XGB超參數
（_XGB_PARAMS，跟 train.py::train_xgb() 現在用的一致），只改atr5_threshold
本身——這樣precision差異才能歸因給「ATR5門檻高低」，不會混進其他變數。

⚠️ 每個候選門檻都用 train.py::_prepare_data(atr5_threshold=...) 傳入非
預設值，這個函式設計成非預設atr5_threshold會跳過cache讀寫（見train.py的
說明），所以每個候選門檻都會重新跑一次完整pipeline，比較慢，屬於一次性
診斷腳本，不影響正式cache。

用法：
    python strategy/mkt/experiments/walk_forward_atr5_threshold.py
"""

import statistics
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.mkt.experiments.tune_xgb import _fit_xgb, _precision_at_thresholds
from strategy.mkt.train import _prepare_data

# strategy/mkt/train.py::train_xgb() 現在正式在用的參數，固定不變，只換
# atr5_threshold。
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

_MODEL_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]

# 候選ATR5門檻：目前正式設定(0.00748，2026-07-23對top_n=100算出來的p90，
# 沿用至今) + 對top_n=300全體樣本重算的p95/p97/p99（見2026-07-25density
# 診斷，atr5_pr_check.py::run_absolute_uniform()）。
_ATR5_CANDIDATES = {
    "目前(0.00748)": 0.00748,
    "p95(0.00856)": 0.00856,
    "p97(0.01000)": 0.01000,
    "p99(0.01318)": 0.01318,
}


def _walk_forward_windows(df: pd.DataFrame, n_windows: int, window_days: int) -> list[tuple]:
    max_date = df["date"].max()
    windows = []
    for i in range(n_windows):
        test_end = max_date - pd.Timedelta(days=window_days * i)
        test_start = test_end - pd.Timedelta(days=window_days)
        windows.append((test_start, test_end))
    return list(reversed(windows))


def run(n_windows: int = 5, window_days: int = 45, min_train_days: int = 60):
    dfs = {}
    for label, thr in _ATR5_CANDIDATES.items():
        print(f"載入 atr5_threshold={thr}（{label}）資料...")
        dfs[label] = _prepare_data(atr5_threshold=thr)

    windows = _walk_forward_windows(dfs["目前(0.00748)"], n_windows, window_days)

    results: dict[float, list[dict]] = {t: [] for t in _MODEL_THRESHOLDS}

    for test_start, test_end in windows:
        wlabel = f"{test_start.date()}~{test_end.date()}"

        splits = {}
        skip = False
        for label, df in dfs.items():
            train_df = df[df["date"] < test_start]
            test_df = df[(df["date"] >= test_start) & (df["date"] < test_end)]
            if train_df.empty or test_df.empty:
                skip = True
                break
            span = (train_df["date"].max() - train_df["date"].min()).days
            if span < min_train_days:
                skip = True
                break
            splits[label] = (train_df, test_df)

        if skip:
            print(f"[{wlabel}] 跳過：訓練或測試資料為空，或訓練資料橫跨天數不足{min_train_days}天")
            continue

        models = {label: _fit_xgb(train_df, _XGB_PARAMS) for label, (train_df, _) in splits.items()}
        by_thr = {label: _precision_at_thresholds(models[label], test_df, _MODEL_THRESHOLDS) for label, (_, test_df) in splits.items()}
        actual = {label: int((test_df["target"] == 2).sum()) for label, (_, test_df) in splits.items()}

        print(f"[{wlabel}] 實際漲：" + "  ".join(f"{label}={actual[label]}" for label in dfs))
        for mthr in _MODEL_THRESHOLDS:
            row = {"window": wlabel}
            line = f"    threshold={mthr:.1f}  "
            for label in dfs:
                p, n = by_thr[label][mthr]
                row[f"{label}_precision"] = p
                row[f"{label}_n"] = n
                line += f"{label}: precision={p:>6.2%}(n={n:>4})  "
            results[mthr].append(row)
            print(line)

    if not results[_MODEL_THRESHOLDS[0]]:
        print("沒有任何窗口跑得動，檢查 n_windows/window_days/min_train_days 是否合理")
        return results

    print(f"\n=== 總結（{len(results[_MODEL_THRESHOLDS[0]])}個窗口，各門檻分別統計） ===")
    for mthr in _MODEL_THRESHOLDS:
        rows = results[mthr]
        print(f"\n-- threshold={mthr:.1f} --")
        for label in dfs:
            plist = [r[f"{label}_precision"] for r in rows]
            print(f"{label:>16s} precision： mean={statistics.mean(plist):.2%}  std={statistics.pstdev(plist):.2%}")

    return results


if __name__ == "__main__":
    run()
