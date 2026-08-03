"""
反向後誰往誰對齊：淨大單→M1 vs M1→淨大單。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_m1_net_realign \\
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

HORIZONS = (1, 3, 5, 15)


def _bar_m1_dir(m1_day: pd.DataFrame, win_start: pd.Timestamp) -> float:
    """該分鐘 M1 方向：+1/-1；平盤或缺棒 → nan。"""
    if m1_day is None or m1_day.empty:
        return float("nan")
    hit = m1_day.loc[m1_day["date"] == pd.Timestamp(win_start)]
    if hit.empty:
        return float("nan")
    o = float(hit.iloc[0]["open"])
    c = float(hit.iloc[0]["close"])
    if not np.isfinite(o) or not np.isfinite(c) or c == o:
        return float("nan")
    return 1.0 if c > o else -1.0


def _bar_net(ticks: pd.DataFrame, win_start: pd.Timestamp) -> float:
    win_end = pd.Timestamp(win_start) + pd.Timedelta(minutes=1)
    return float(_tick_in_window(ticks, win_start, win_end)["bar_large_net_ratio"])


def _net_sign(net: float) -> float:
    if not np.isfinite(net) or net == 0:
        return 0.0
    return 1.0 if net > 0 else -1.0


def _classify_h(m1_h: float, net_h: float) -> str:
    if not np.isfinite(m1_h) or _net_sign(net_h) == 0:
        return "nosig"
    ns = _net_sign(net_h)
    if m1_h == ns:
        return "align"
    return "opp"


def run(start_date: str, end_date: str, max_rows: int = 4000) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("反向後誰往誰對齊（淨大單→M1 vs M1→淨大單）", flush=True)
    print(
        f"母體 tick400；窗 {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}；"
        f"淨大單 >{TICK_LARGE_LOT}張；每 stock×day≤{PER_DAY_MAX}；max_rows={max_rows}",
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

    print("讀取 tick + 各 H 狀態...", flush=True)
    records: list[dict] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for j, r in enumerate(ev.itertuples(index=False), start=1):
        key = (str(r.stock_id), str(r.trade_date)[:10])
        if key not in cache:
            try:
                cache[key] = load_tick_by_stock(key[0], date=key[1])
            except Exception:
                cache[key] = pd.DataFrame()
        ticks = cache[key]
        day_df = m1_by_day.get(key)
        net0 = _bar_net(ticks, r.win_start)
        m1_0 = float(r.m1_dir)
        ns0 = _net_sign(net0)
        if ns0 == 0:
            grp = "zero"
        elif ns0 == m1_0:
            grp = "same"
        else:
            grp = "opp"

        row: dict = {
            "stock_id": key[0],
            "trade_date": key[1],
            "win_start": r.win_start,
            "m1_0": m1_0,
            "net_0": net0,
            "ns_0": ns0,
            "grp": grp,
        }
        for h in HORIZONS:
            ws = pd.Timestamp(r.win_start) + pd.Timedelta(minutes=h)
            m1_h = _bar_m1_dir(day_df, ws)
            net_h = _bar_net(ticks, ws)
            st = _classify_h(m1_h, net_h)
            row[f"m1_{h}"] = m1_h
            row[f"net_{h}"] = net_h
            row[f"st_{h}"] = st
            if st == "align" and grp == "opp":
                align_dir = m1_h
                if align_dir == m1_0:
                    row[f"who_{h}"] = "net_to_m1"
                elif align_dir == ns0:
                    row[f"who_{h}"] = "m1_to_net"
                else:
                    row[f"who_{h}"] = "other"
            else:
                row[f"who_{h}"] = ""
        records.append(row)
        if j % 500 == 0 or j == len(ev):
            print(f"  {j}/{len(ev)} elapsed={time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(records)
    n_opp = int((out["grp"] == "opp").sum())
    n_same = int((out["grp"] == "same").sum())
    n_zero = int((out["grp"] == "zero").sum())
    print("\n" + "=" * 72)
    print(f"t0 分組: 反向={n_opp} 同向={n_same} net=0={n_zero}")
    print("=" * 72)

    opp = out[out["grp"] == "opp"]
    print("\n【主表】t0 反向 → 各 H", flush=True)
    hdr = (
        f"{'H':>4} {'n':>5} {'對齊%':>8} {'仍反向%':>8} {'無訊號%':>8} "
        f"{'淨大單→M1':>12} {'M1→淨大單':>12}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for h in HORIZONS:
        col = f"st_{h}"
        who = f"who_{h}"
        m1c = f"m1_{h}"
        valid = opp[np.isfinite(opp[m1c])]
        n = len(valid)
        if n == 0:
            print(f"{h:>4} {0:>5}", flush=True)
            continue
        n_al = int((valid[col] == "align").sum())
        n_op = int((valid[col] == "opp").sum())
        n_ns = int((valid[col] == "nosig").sum())
        al = valid[valid[col] == "align"]
        n_nm = int((al[who] == "net_to_m1").sum()) if n_al else 0
        n_mn = int((al[who] == "m1_to_net").sum()) if n_al else 0
        print(
            f"{h:>4} {n:>5} {100*n_al/n:>7.1f}% {100*n_op/n:>7.1f}% {100*n_ns/n:>7.1f}% "
            f"{(100*n_nm/n_al if n_al else 0):>10.1f}%/{n_nm} "
            f"{(100*n_mn/n_al if n_al else 0):>10.1f}%/{n_mn}",
            flush=True,
        )

    print("\n【反向組】首次對齊累積（一路看到 15）", flush=True)
    first_who: list[str] = []
    for _, r in opp.iterrows():
        found = ""
        for h in HORIZONS:
            if not np.isfinite(r[f"m1_{h}"]):
                continue
            if r[f"st_{h}"] == "align":
                found = str(r[f"who_{h}"])
                break
        first_who.append(found)
    opp = opp.copy()
    opp["first_who"] = first_who
    n_base = len(opp)
    n_first = int((opp["first_who"] != "").sum())
    n_nm = int((opp["first_who"] == "net_to_m1").sum())
    n_mn = int((opp["first_who"] == "m1_to_net").sum())
    print(
        f"n={n_base}  15分內曾對齊={n_first} ({100*n_first/n_base if n_base else 0:.1f}%)  "
        f"淨大單→M1={n_nm} ({100*n_nm/n_first if n_first else 0:.1f}% of aligned)  "
        f"M1→淨大單={n_mn} ({100*n_mn/n_first if n_first else 0:.1f}% of aligned)",
        flush=True,
    )

    print("\n【基線】同向／net=0 在各 H 的狀態（不拆誰遷就誰）", flush=True)
    for gname, gkey in (("同向", "same"), ("net=0", "zero")):
        sub = out[out["grp"] == gkey]
        print(f"\n{gname} n={len(sub)}", flush=True)
        hdr2 = f"{'H':>4} {'n':>5} {'同向/對齊%':>12} {'反向%':>8} {'無訊號%':>8}"
        print(hdr2, flush=True)
        print("-" * len(hdr2), flush=True)
        for h in HORIZONS:
            m1c = f"m1_{h}"
            col = f"st_{h}"
            valid = sub[np.isfinite(sub[m1c])]
            n = len(valid)
            if n == 0:
                continue
            n_al = int((valid[col] == "align").sum())
            n_op = int((valid[col] == "opp").sum())
            n_ns = int((valid[col] == "nosig").sum())
            print(
                f"{h:>4} {n:>5} {100*n_al/n:>11.1f}% {100*n_op/n:>7.1f}% {100*n_ns/n:>7.1f}%",
                flush=True,
            )

    print(
        f"\n解讀: 對齊少→不常合併；淨大單→M1 高→大單跟K；"
        f"M1→淨大單高→K跟大單（大單較像領先）。耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return out


def main():
    p = argparse.ArgumentParser(description="反向後淨大單/M1 誰往誰對齊")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-07-07")
    p.add_argument("--max_rows", type=int, default=4000)
    args = p.parse_args()
    run(args.start_date, args.end_date, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
