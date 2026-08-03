"""
M1⊥淨大單：偏離有多大、跟著淨大單做有無利潤（小樣本）。

偏離：|net|、M1 實體幅度 |(c-o)/o|
利潤：棒收進場，方向＝sign(net)（做多淨買／放空淨賣），看 5/15/30 分報酬。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_m1_net_divergence_pnl \\
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


def _print_div(name: str, sub: pd.DataFrame) -> None:
    n = len(sub)
    print(f"\n[{name}] n={n}", flush=True)
    if n == 0:
        return
    body = sub["body_pct"].astype(float)
    abs_net = sub["abs_net"].astype(float)
    print(
        f"  |net|    mean={abs_net.mean():.3f}  p50={abs_net.median():.3f}  "
        f"p75={abs_net.quantile(0.75):.3f}  p90={abs_net.quantile(0.90):.3f}",
        flush=True,
    )
    print(
        f"  M1實體%  mean={100*body.mean():.3f}%  p50={100*body.median():.3f}%  "
        f"p75={100*body.quantile(0.75):.3f}%  p90={100*body.quantile(0.90):.3f}%",
        flush=True,
    )


def _print_pnl(name: str, sub: pd.DataFrame) -> None:
    """pnl_signed = ret * sign(net)：跟著淨大單。"""
    n = len(sub)
    print(f"\n[{name}] 跟著淨大單 n={n}", flush=True)
    if n == 0:
        return
    hdr = f"{'H':>4} {'n':>5} {'勝%':>7} {'敗%':>7} {'mean':>9} {'p50':>9} {'p75':>9}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for h in HORIZONS:
        col = f"pnl_{h}"
        ok = sub[np.isfinite(sub[col])]
        if ok.empty:
            print(f"{h:>4} {0:>5}", flush=True)
            continue
        x = ok[col]
        win = 100.0 * (x > 0).sum() / len(ok)
        lose = 100.0 * (x < 0).sum() / len(ok)
        print(
            f"{h:>4} {len(ok):>5} {win:>6.1f}% {lose:>6.1f}% "
            f"{100*x.mean():>8.3f}% {100*x.median():>8.3f}% "
            f"{100*x.quantile(0.75):>8.3f}%",
            flush=True,
        )


def run(start_date: str, end_date: str, max_rows: int = 4000) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("反向偏離幅度 × 跟著淨大單利潤（小樣本）", flush=True)
    print(
        f"母體 tick400；窗 {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}；"
        f"淨大單 >{TICK_LARGE_LOT}張；max_rows={max_rows}",
        flush=True,
    )
    print("進場=棒收；方向=sign(net)；pnl=ret×sign(net)", flush=True)
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    m1 = load_pattern_m1(start_date=start_date)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(stock_ids)].copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[(m1["day_str"] >= start_date) & (m1["day_str"] <= end_date)]
    m1_by_day = {
        (str(sid), str(day)): g.sort_values("date").reset_index(drop=True)
        for (sid, day), g in m1.groupby(["stock_id", "day_str"], sort=False)
    }

    ev = _collect_candidates(m1, max_rows=max_rows)
    print(f"抽樣列={len(ev):,}", flush=True)
    if ev.empty:
        return ev

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
        m1_dir = int(r.m1_dir)
        o = float(r.open)
        c = float(r.close)
        body = abs(c - o) / o if o > 0 else float("nan")
        if net == 0:
            grp = "zero"
            ns = 0
        elif (net > 0 and m1_dir > 0) or (net < 0 and m1_dir < 0):
            grp = "same"
            ns = 1 if net > 0 else -1
        else:
            grp = "opp"
            ns = 1 if net > 0 else -1

        rec = {
            "grp": grp,
            "m1_dir": m1_dir,
            "net": net,
            "abs_net": abs(net),
            "ns": ns,
            "body_pct": body,
            "entry": c,
        }
        day_df = m1_by_day.get(key)
        for h in HORIZONS:
            ret = _fwd_ret(day_df, r.win_start, c, h)
            # 跟著淨大單；net=0 無方向 → nan
            rec[f"pnl_{h}"] = (ret * ns) if ns != 0 and np.isfinite(ret) else float("nan")
            rec[f"ret_{h}"] = ret
        rows.append(rec)
        if j % 500 == 0 or j == len(ev):
            print(f"  {j}/{len(ev)} elapsed={time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    print("\n" + "=" * 64)
    print("一、偏離有多大")
    print("=" * 64)
    _print_div("反向", out[out["grp"] == "opp"])
    _print_div("同向（對照）", out[out["grp"] == "same"])

    # 反向：再按 |net| 分位看偏離
    opp = out[out["grp"] == "opp"]
    if len(opp) >= 20:
        thr = float(opp["abs_net"].median())
        print(f"\n反向再切 |net|：中位={thr:.3f}", flush=True)
        _print_div(f"反向 |net|≥中位", opp[opp["abs_net"] >= thr])
        _print_div(f"反向 |net|<中位", opp[opp["abs_net"] < thr])

    print("\n" + "=" * 64)
    print("二、跟著淨大單有無利潤")
    print("=" * 64)
    _print_pnl("反向", out[out["grp"] == "opp"])
    _print_pnl("同向（對照：跟淨大單=跟M1）", out[out["grp"] == "same"])
    if len(opp) >= 20:
        thr = float(opp["abs_net"].median())
        _print_pnl(f"反向且|net|≥中位({thr:.2f})", opp[opp["abs_net"] >= thr])
        # 大正體偏離
        bmed = float(opp["body_pct"].median())
        _print_pnl(
            f"反向且實體≥中位({100*bmed:.2f}%)",
            opp[opp["body_pct"] >= bmed],
        )

    print(
        f"\n解讀: 若反向組 mean(pnl)>0 且勝率明顯>50%，才像有利潤；"
        f"|net|/實體越大越賺才像偏離可交易。耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return out


def main():
    p = argparse.ArgumentParser(description="反向偏離幅度與跟著淨大單利潤")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-07-07")
    p.add_argument("--max_rows", type=int, default=4000)
    args = p.parse_args()
    run(args.start_date, args.end_date, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
