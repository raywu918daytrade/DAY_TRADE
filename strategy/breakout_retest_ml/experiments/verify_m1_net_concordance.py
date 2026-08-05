"""
M1 K 線方向 vs 該分鐘淨大單方向：同向率（小樣本描述統計，不接勝率）。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_m1_net_concordance \\
        --start_date 2026-07-01 --end_date 2026-07-07 --max_rows 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import time as dtime
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from data.adjustment_query import load_pattern_m1
from data.query import load_tick_by_stock
from finmind.tick_universe import load_tick_universe
from strategy.breakout_retest_ml.config import SESSION_END, TICK_LARGE_LOT
from strategy.breakout_retest_ml.experiments.verify_m5_tick import (
    SESSION_START,
    _tick_in_window,
)

RNG = np.random.default_rng(42)
PER_DAY_MAX = 3  # 每個 stock×day 最多抽幾根非平盤 M1


def _collect_candidates(
    m1: pd.DataFrame,
    max_rows: int,
    per_day_max: int = PER_DAY_MAX,
) -> pd.DataFrame:
    """決策窗內非平盤 M1；每 stock×day 隨機抽 per_day_max，再截到 max_rows。"""
    t0 = dtime(*SESSION_START)
    t1 = dtime(*SESSION_END)
    rows: list[dict] = []

    for (sid, day), g in m1.groupby(["stock_id", "day_str"], sort=False):
        g = g.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
        day_cands: list[dict] = []
        for _, r in g.iterrows():
            ts = pd.Timestamp(r["date"])
            bar_end = ts + pd.Timedelta(minutes=1)
            tm = bar_end.time()
            if tm < t0 or tm > t1:
                continue
            o = float(r["open"])
            c = float(r["close"])
            if not np.isfinite(o) or not np.isfinite(c) or c == o:
                continue
            day_cands.append(
                {
                    "stock_id": str(sid),
                    "trade_date": str(day),
                    "win_start": ts,
                    "win_end": bar_end,
                    "m1_dir": 1 if c > o else -1,
                    "open": o,
                    "close": c,
                }
            )
        if not day_cands:
            continue
        if len(day_cands) > per_day_max:
            idx = RNG.choice(len(day_cands), size=per_day_max, replace=False)
            day_cands = [day_cands[i] for i in sorted(idx)]
        rows.extend(day_cands)
        if len(rows) >= max_rows:
            break

    ev = pd.DataFrame(rows[:max_rows])
    return ev


def run(start_date: str, end_date: str, max_rows: int = 2000) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("M1 方向 × 淨大單方向 同向率（小樣本）", flush=True)
    print(
        f"母體: tick400；決策窗 {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}",
        flush=True,
    )
    print(
        f"M1: close≠open；淨大單 net=大買占比−大賣占比（>{TICK_LARGE_LOT}張）；"
        f"每 stock×day 抽≤{PER_DAY_MAX} 根，max_rows={max_rows}",
        flush=True,
    )
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    print(f"載入 pattern M1（start={start_date})...", flush=True)
    m1 = load_pattern_m1(start_date=start_date)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(stock_ids)].copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[(m1["day_str"] >= start_date) & (m1["day_str"] <= end_date)]
    print(f"M1 {len(m1):,} / stocks={m1['stock_id'].nunique()}", flush=True)

    ev = _collect_candidates(m1, max_rows=max_rows)
    print(f"抽樣列={len(ev):,}（up={(ev['m1_dir']==1).sum()} down={(ev['m1_dir']==-1).sum()}）", flush=True)
    if ev.empty:
        return ev

    print("讀取 tick（按 stock×day 快取）...", flush=True)
    nets: list[float] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for j, r in enumerate(ev.itertuples(index=False), start=1):
        key = (str(r.stock_id), str(r.trade_date)[:10])
        if key not in cache:
            try:
                cache[key] = load_tick_by_stock(key[0], date=key[1])
            except Exception:
                cache[key] = pd.DataFrame()
        feat = _tick_in_window(cache[key], r.win_start, r.win_end)
        nets.append(float(feat["bar_large_net_ratio"]))
        if j % 400 == 0 or j == len(ev):
            print(
                f"  [tick] {j}/{len(ev)} cache_days={len(cache)} " f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    ev = ev.copy()
    ev["net"] = nets

    m1_up = ev["m1_dir"] == 1
    m1_dn = ev["m1_dir"] == -1
    net_buy = ev["net"] > 0
    net_sell = ev["net"] < 0
    net_zero = ev["net"] == 0

    same = (m1_up & net_buy) | (m1_dn & net_sell)
    opp = (m1_up & net_sell) | (m1_dn & net_buy)
    n_dir = int((~net_zero).sum())  # 有明確淨大單方向

    print("\n" + "=" * 64)
    print("同向／反向（僅 net≠0）")
    print("=" * 64)
    if n_dir:
        print(
            f"n_dir={n_dir}  同向={int(same.sum())} ({100*same.sum()/n_dir:.1f}%)  "
            f"反向={int(opp.sum())} ({100*opp.sum()/n_dir:.1f}%)  "
            f"net=0={int(net_zero.sum())}",
            flush=True,
        )
    else:
        print("無明確淨大單方向樣本", flush=True)

    print("\n分邊占比", flush=True)
    hdr = f"{'M1':>6} {'n':>6} {'net>0':>8} {'net<0':>8} {'net=0':>8} {'mean(net)':>10}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for label, mask in (("上", m1_up), ("下", m1_dn)):
        sub = ev[mask]
        n = len(sub)
        if n == 0:
            continue
        print(
            f"{label:>6} {n:>6} "
            f"{100*(sub['net']>0).mean():>7.1f}% "
            f"{100*(sub['net']<0).mean():>7.1f}% "
            f"{100*(sub['net']==0).mean():>7.1f}% "
            f"{sub['net'].mean():>10.4f}",
            flush=True,
        )

    mean_up = float(ev.loc[m1_up, "net"].mean()) if m1_up.any() else float("nan")
    mean_dn = float(ev.loc[m1_dn, "net"].mean()) if m1_dn.any() else float("nan")
    print(
        f"\nmean(net|M1上)={mean_up:.4f}  mean(net|M1下)={mean_dn:.4f}  " f"差={mean_up - mean_dn:.4f}",
        flush=True,
    )
    print(
        f"解讀: 同向率≈50% 或兩側 mean(net)接近 → 幾乎脫鉤。" f" 耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return ev


def main():
    p = argparse.ArgumentParser(description="M1 × 淨大單同向率（小樣本）")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-07-07")
    p.add_argument("--max_rows", type=int, default=2000)
    args = p.parse_args()
    run(args.start_date, args.end_date, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
