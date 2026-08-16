"""
09:00–09:05 量能相對自己歷史同段的 PR vs 當日高低幅 (high-low)/open。

用法：
    python -m strategy_test.open5_vol_pr_range.verify \\
        --start_date 2024-01-01 --end_date 2026-08-14
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import time as dtime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from data.query import load_day, load_m5_std
from finmind.tick_universe import load_tick_universe

ENTRY_T = dtime(9, 5)
BUCKETS = (
    ("lt50", lambda p: p < 0.50),
    ("50to80", lambda p: (p >= 0.50) & (p < 0.80)),
    ("80to90", lambda p: (p >= 0.80) & (p < 0.90)),
    ("ge90", lambda p: p >= 0.90),
)
EXTRA = (("ge95", lambda p: p >= 0.95),)


def _mw_greater(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(a[m], b[m])
    return float(r), float(p)


def _print_rng(label: str, rng: pd.Series, min_n: int) -> None:
    rng = rng.dropna()
    n = len(rng)
    if n < min_n:
        print(f"  {label}: n={n:,}  (< min_n={min_n}，略)", flush=True)
        return
    print(
        f"  {label}: n={n:,}  "
        f"mean={100 * rng.mean():.3f}%  "
        f"median={100 * rng.median():.3f}%  "
        f"p75={100 * rng.quantile(0.75):.3f}%  "
        f"≥3%={100 * (rng >= 0.03).mean():.1f}%  "
        f"≥5%={100 * (rng >= 0.05).mean():.1f}%",
        flush=True,
    )


def _vol5_pr(vol: pd.Series, lookback: int, min_hist: int) -> pd.Series:
    """當天 vol vs 過去 lookback 日（不含當天）的 empirical CDF。"""

    def _at_end(x: np.ndarray) -> float:
        hist = x[:-1]
        today = x[-1]
        if len(hist) < min_hist or not np.isfinite(today):
            return np.nan
        return float(np.mean(hist < today))

    return vol.rolling(lookback + 1, min_periods=min_hist + 1).apply(_at_end, raw=True)


def _attach_pr(ev: pd.DataFrame, lookback: int, min_hist: int) -> pd.DataFrame:
    ev = ev.sort_values(["stock_id", "day"]).copy()
    parts = []
    for _, g in ev.groupby("stock_id", sort=False):
        g = g.copy()
        g["vol5_pr"] = _vol5_pr(g["vol5"], lookback, min_hist)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else ev


def run(
    start_date: str,
    end_date: str,
    lookback: int = 20,
    min_hist: int = 10,
    min_n: int = 30,
) -> pd.DataFrame:
    t0 = time.time()
    stocks = set(str(s) for s in load_tick_universe())
    load_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=max(lookback * 3, 45))
    ).strftime("%Y-%m-%d")
    end_excl = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    print(f"載入 m5_std（{load_start}～{end_date}，09:05 vol5）...", flush=True)
    m5 = load_m5_std(start_date=load_start)
    m5["stock_id"] = m5["stock_id"].astype(str)
    m5 = m5[m5["stock_id"].isin(stocks) & (m5["date"] < end_excl)]
    m5 = m5[m5["date"].dt.time == ENTRY_T].copy()
    m5["day"] = m5["date"].dt.strftime("%Y-%m-%d")
    m5 = m5.drop_duplicates(["stock_id", "day"], keep="last")
    print(f"  09:05 根 {len(m5):,}", flush=True)

    print(f"載入 day（{load_start}～{end_date}）...", flush=True)
    day = load_day(start_date=load_start)
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].isin(stocks) & (day["date"] < end_excl)]
    day = day[day["open"] > 0].copy()
    day["day"] = day["date"].dt.strftime("%Y-%m-%d")
    day["rng"] = (day["high"] - day["low"]) / day["open"]
    print(f"  日K {len(day):,}", flush=True)

    vol = m5[["stock_id", "day", "volume"]].rename(columns={"volume": "vol5"})
    ev = vol.merge(day[["stock_id", "day", "rng"]], on=["stock_id", "day"], how="inner")
    ev = ev.dropna(subset=["vol5", "rng"])
    print(f"  stock-day {len(ev):,}（merge vol5+rng）", flush=True)
    if ev.empty:
        print("無資料", flush=True)
        return ev

    print(f"算 vol5 PR（lookback={lookback}, min_hist={min_hist}）...", flush=True)
    ev = _attach_pr(ev, lookback, min_hist)
    ev = ev[(ev["day"] >= start_date) & (ev["day"] <= end_date)]
    n_pr = ev["vol5_pr"].notna().sum()
    print(f"  統計窗 {start_date}～{end_date}：{len(ev):,} 列，有 PR {n_pr:,}", flush=True)

    ok = ev.dropna(subset=["vol5_pr"])
    print("\n" + "=" * 60, flush=True)
    print("09:05 vol5 自身歷史 PR vs 當日 (high-low)/open", flush=True)
    print("=" * 60, flush=True)
    _print_rng("all", ok["rng"], min_n=1)

    for name, pred in BUCKETS + EXTRA:
        _print_rng(name, ok.loc[pred(ok["vol5_pr"]), "rng"], min_n)

    print("\n高 PR vs 其餘（Mann-Whitney greater on rng）", flush=True)
    for label, thr in (("ge80", 0.80), ("ge90", 0.90), ("ge95", 0.95)):
        hi = ok.loc[ok["vol5_pr"] >= thr, "rng"]
        rest = ok.loc[ok["vol5_pr"] < thr, "rng"]
        if len(hi) < min_n or len(rest) < min_n:
            print(
                f"  {label} vs rest: n_hi={len(hi):,} n_rest={len(rest):,}  (< min_n，略)",
                flush=True,
            )
            continue
        lift = hi.median() / rest.median() if rest.median() > 0 else float("nan")
        p = _mw_greater(hi, rest)
        print(
            f"  {label} vs rest: n_hi={len(hi):,} n_rest={len(rest):,}  "
            f"median {100 * hi.median():.3f}% / {100 * rest.median():.3f}%  "
            f"×{lift:.2f}  p={p:.2e}",
            flush=True,
        )

    r_v, p_v = _spearman(ok["vol5"], ok["rng"])
    r_p, p_p = _spearman(ok["vol5_pr"], ok["rng"])
    print("\nSpearman", flush=True)
    print(f"  vol5 vs rng:  ρ={r_v:.3f}  p={p_v:.2e}  n={ok['vol5'].notna().sum():,}", flush=True)
    print(f"  PR   vs rng:  ρ={r_p:.3f}  p={p_p:.2e}  n={n_pr:,}", flush=True)
    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return ev


def main() -> None:
    p = argparse.ArgumentParser(description="開盤 5 分鐘量能 PR vs 當日高低幅")
    p.add_argument("--start_date", default="2024-01-01")
    p.add_argument("--end_date", default="2026-08-14")
    p.add_argument("--lookback", type=int, default=20, help="自身歷史交易日窗")
    p.add_argument("--min_hist", type=int, default=10, help="至少幾日歷史才算 PR")
    p.add_argument("--min_n", type=int, default=30, help="分桶最少樣本")
    args = p.parse_args()
    run(
        args.start_date,
        args.end_date,
        lookback=args.lookback,
        min_hist=args.min_hist,
        min_n=args.min_n,
    )


if __name__ == "__main__":
    main()
