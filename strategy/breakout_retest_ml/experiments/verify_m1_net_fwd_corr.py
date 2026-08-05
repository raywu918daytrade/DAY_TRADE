"""
淨大單比例 × 後續漲幅相關（小樣本）。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_m1_net_fwd_corr \\
        --start_date 2026-07-01 --end_date 2026-07-07 --max_rows 4000
"""

from __future__ import annotations

import argparse
import sys
import time
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
from strategy.breakout_retest_ml.experiments.verify_m1_net_concordance import (
    PER_DAY_MAX,
    _collect_candidates,
)

HORIZONS = (5, 15, 30)


def _fwd_ret(m1_day: pd.DataFrame, win_start: pd.Timestamp, entry: float, minutes: int) -> float:
    if entry <= 0 or m1_day is None or m1_day.empty:
        return float("nan")
    t = pd.Timestamp(win_start) + pd.Timedelta(minutes=minutes)
    hit = m1_day.loc[m1_day["date"] == t, "close"]
    if hit.empty:
        return float("nan")
    c = float(hit.iloc[0])
    if not np.isfinite(c):
        return float("nan")
    return (c - entry) / entry


def _corr_pair(x: pd.Series, y: pd.Series) -> tuple[int, float, float]:
    ok = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(ok)
    if n < 3:
        return n, float("nan"), float("nan")
    pear = float(ok["x"].corr(ok["y"], method="pearson"))
    spear = float(ok["x"].corr(ok["y"], method="spearman"))
    return n, pear, spear


def _print_corr_block(title: str, sub: pd.DataFrame) -> None:
    print(f"\n[{title}] n={len(sub)}", flush=True)
    if sub.empty:
        return
    hdr = f"{'H':>4} {'n':>6} {'Pearson':>10} {'Spearman':>10}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for h in HORIZONS:
        n, pear, spear = _corr_pair(sub["net"], sub[f"ret_{h}"])
        print(f"{h:>4} {n:>6} {pear:>10.4f} {spear:>10.4f}", flush=True)

    print("net 五分位 mean(ret)%", flush=True)
    hdr2 = f"{'Q':>4} {'n':>6} {'mean(net)':>10}" + "".join(f"{'ret'+str(h):>10}" for h in HORIZONS)
    print(hdr2, flush=True)
    print("-" * len(hdr2), flush=True)
    # 五分位；若唯一值太少（大量 net=0）qcut 會失敗 → 改用 rank
    try:
        q = pd.qcut(sub["net"], 5, labels=False, duplicates="drop")
    except ValueError:
        q = pd.Series(np.nan, index=sub.index)
    if q.notna().sum() < len(sub) * 0.5:
        # 大量重複時改三分：<0 / =0 / >0
        bins = pd.Series(
            np.where(sub["net"] < 0, 0, np.where(sub["net"] > 0, 2, 1)),
            index=sub.index,
        )
        labels = {0: "net<0", 1: "net=0", 2: "net>0"}
        for b in (0, 1, 2):
            g = sub[bins == b]
            if g.empty:
                continue
            ret_s = "".join(
                f"{100*g[f'ret_{h}'].mean():>9.3f}%" if np.isfinite(g[f"ret_{h}"].mean()) else f"{'nan':>10}"
                for h in HORIZONS
            )
            print(
                f"{labels[b]:>4} {len(g):>6} {g['net'].mean():>10.4f}{ret_s}",
                flush=True,
            )
        return

    for qi in sorted(q.dropna().unique()):
        g = sub[q == qi]
        ret_s = "".join(
            f"{100*g[f'ret_{h}'].mean():>9.3f}%" if np.isfinite(g[f"ret_{h}"].mean()) else f"{'nan':>10}"
            for h in HORIZONS
        )
        print(
            f"Q{int(qi)+1:>3} {len(g):>6} {g['net'].mean():>10.4f}{ret_s}",
            flush=True,
        )


def run(start_date: str, end_date: str, max_rows: int = 4000) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("淨大單比例 × 後續漲幅相關（小樣本）", flush=True)
    print(
        f"母體 tick400；窗 {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}；"
        f"淨大單 >{TICK_LARGE_LOT}張；每 stock×day≤{PER_DAY_MAX}；max_rows={max_rows}",
        flush=True,
    )
    print(f"ret_H = close(t+H)/close(t) − 1；區間 {start_date} ~ {end_date}\n", flush=True)

    print(f"載入 pattern M1（start={start_date})...", flush=True)
    m1 = load_pattern_m1(start_date=start_date)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(stock_ids)].copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[(m1["day_str"] >= start_date) & (m1["day_str"] <= end_date)]
    print(f"M1 {len(m1):,}", flush=True)

    m1_by_day = {
        (str(sid), str(day)): g.sort_values("date").reset_index(drop=True)
        for (sid, day), g in m1.groupby(["stock_id", "day_str"], sort=False)
    }

    ev = _collect_candidates(m1, max_rows=max_rows)
    print(
        f"抽樣列={len(ev):,} up={(ev['m1_dir']==1).sum()} down={(ev['m1_dir']==-1).sum()}",
        flush=True,
    )
    if ev.empty:
        return ev

    print("讀取 tick + forward...", flush=True)
    rows: list[dict] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for j, r in enumerate(ev.itertuples(index=False), start=1):
        key = (str(r.stock_id), str(r.trade_date)[:10])
        if key not in cache:
            try:
                cache[key] = load_tick_by_stock(key[0], date=key[1])
            except Exception:
                cache[key] = pd.DataFrame()
        feat = _tick_in_window(cache[key], r.win_start, r.win_end)
        net = float(feat["bar_large_net_ratio"])
        rec = {
            "m1_dir": int(r.m1_dir),
            "net": net,
            "entry": float(r.close),
        }
        day_df = m1_by_day.get(key)
        for h in HORIZONS:
            rec[f"ret_{h}"] = _fwd_ret(day_df, r.win_start, float(r.close), h)
        rows.append(rec)
        if j % 500 == 0 or j == len(ev):
            print(f"  {j}/{len(ev)} elapsed={time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    print("\n" + "=" * 64)
    print("相關與分位")
    print("=" * 64)

    _print_corr_block("全體", out)
    _print_corr_block("僅 M1 上", out[out["m1_dir"] == 1])
    _print_corr_block("全體且 net≠0", out[out["net"] != 0])
    _print_corr_block("M1上且 net≠0", out[(out["m1_dir"] == 1) & (out["net"] != 0)])

    print(
        f"\n解讀: Spearman 明顯>0 且分位 mean(ret) 隨 net 上升 → 正相關；"
        f"否則淨大單不預測後續漲幅。耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return out


def main():
    p = argparse.ArgumentParser(description="淨大單比例 × 後續漲幅相關")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-07-07")
    p.add_argument("--max_rows", type=int, default=4000)
    args = p.parse_args()
    run(args.start_date, args.end_date, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
