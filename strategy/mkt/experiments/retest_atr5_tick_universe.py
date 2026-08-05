"""
2026-07-25：改用tick_universe固定400支之後，重新驗證ATR5門檻該用多少。

背景：舊的 ATR5_FILTER_THRESHOLD（p97=0.01000）是在發現「top_n動態排名
母體隨時間漂移」這個bug之前，對top_n=300（母體歷史上曾經到2710支）算出來
的，數字不可信，改用tick_universe固定400支之後要重新算一次。

⚠️ 只呼叫一次 _prepare_data(atr5_threshold=極小值) 拿到「時段過濾後、ATR5
過濾前」的完整population（這一步很貴，要跑完整條pipeline，約近30分鐘），
存下來後，密度診斷（不同百分位門檻的跌/平/漲比例）跟walk-forward precision
比較都直接對這份存好的df做in-memory篩選，不再重跑pipeline，避免4個候選
門檻各自重跑一次。

用法：
    python strategy/mkt/experiments/retest_atr5_tick_universe.py
"""

import statistics
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from strategy.mkt.experiments.tune_xgb import _fit_xgb, _precision_at_thresholds
from strategy.mkt.train import _prepare_data

_ROOT = Path(__file__).parent.parent.parent.parent
_UNFILTERED_CACHE = _ROOT / "cache/mkt_prepared_unfiltered_debug.parquet"

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


def load_unfiltered(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and _UNFILTERED_CACHE.exists():
        print(f"讀已存的未過濾population... [{_UNFILTERED_CACHE.name}]")
        return pd.read_parquet(_UNFILTERED_CACHE)
    print("重跑完整pipeline，atr5_threshold設極小值＝不過濾（這步很貴，約30分鐘）...")
    df = _prepare_data(atr5_threshold=-1.0)
    _UNFILTERED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_UNFILTERED_CACHE)
    print(f"已存至 {_UNFILTERED_CACHE}，之後重跑這支腳本可以直接讀取，不用再重算")
    return df


def density_table(df: pd.DataFrame, percentiles=(50, 75, 90, 95, 97, 99)):
    print(f"\n總樣本數（ATR5過濾前）: {len(df):,}")
    print(f"\n{'門檻(全體pN)':>14}  {'atr5值':>10}  {'樣本數':>10}  {'保留%':>7}  {'跌%':>7}  {'平%':>7}  {'漲%':>7}")
    total = len(df)
    thresholds = {}
    for p in percentiles:
        threshold = float(np.percentile(df["atr5"], p))
        sub = df[df["atr5"] >= threshold]
        n = len(sub)
        keep_pct = n / total * 100
        vc = sub["target"].value_counts(normalize=True) * 100
        print(
            f"{'p' + str(p):>14}  {threshold:>10.5f}  {n:>10,}  {keep_pct:>6.2f}%  "
            f"{vc.get(0, 0):>6.2f}%  {vc.get(1, 0):>6.2f}%  {vc.get(2, 0):>6.2f}%"
        )
        thresholds[f"p{p}"] = threshold
    return thresholds


def _walk_forward_windows(df: pd.DataFrame, n_windows: int, window_days: int) -> list[tuple]:
    max_date = df["date"].max()
    windows = []
    for i in range(n_windows):
        test_end = max_date - pd.Timedelta(days=window_days * i)
        test_start = test_end - pd.Timedelta(days=window_days)
        windows.append((test_start, test_end))
    return list(reversed(windows))


def walk_forward_compare(df: pd.DataFrame, candidates: dict, n_windows: int = 5, window_days: int = 45, min_train_days: int = 60):
    dfs = {label: df[df["atr5"] >= thr].copy() for label, thr in candidates.items()}
    for label, d in dfs.items():
        print(f"{label}: {len(d):,} 筆")

    windows = _walk_forward_windows(df, n_windows, window_days)
    results: dict[float, list[dict]] = {t: [] for t in _MODEL_THRESHOLDS}

    for test_start, test_end in windows:
        wlabel = f"{test_start.date()}~{test_end.date()}"
        splits = {}
        skip = False
        for label, d in dfs.items():
            train_df = d[d["date"] < test_start]
            test_df = d[(d["date"] >= test_start) & (d["date"] < test_end)]
            if train_df.empty or test_df.empty:
                skip = True
                break
            span = (train_df["date"].max() - train_df["date"].min()).days
            if span < min_train_days:
                skip = True
                break
            splits[label] = (train_df, test_df)
        if skip:
            print(f"[{wlabel}] 跳過：訓練/測試資料不足")
            continue

        models = {label: _fit_xgb(train_df, _XGB_PARAMS) for label, (train_df, _) in splits.items()}
        by_thr = {
            label: _precision_at_thresholds(models[label], test_df, _MODEL_THRESHOLDS) for label, (_, test_df) in splits.items()
        }
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
        print("沒有任何窗口跑得動")
        return results

    print(f"\n=== 總結（{len(results[_MODEL_THRESHOLDS[0]])}個窗口） ===")
    for mthr in _MODEL_THRESHOLDS:
        rows = results[mthr]
        print(f"\n-- threshold={mthr:.1f} --")
        for label in dfs:
            plist = [r[f"{label}_precision"] for r in rows]
            print(f"{label:>10s} precision： mean={statistics.mean(plist):.2%}  std={statistics.pstdev(plist):.2%}")
    return results


if __name__ == "__main__":
    df = load_unfiltered()
    thresholds = density_table(df)
    candidates = {"p90": thresholds["p90"], "p95": thresholds["p95"], "p97": thresholds["p97"], "p99": thresholds["p99"]}
    walk_forward_compare(df, candidates)
