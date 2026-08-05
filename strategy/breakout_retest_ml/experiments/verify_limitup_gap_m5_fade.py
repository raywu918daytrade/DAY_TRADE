"""
前1日實體漲停 → 今日開高 → 首根分K下跌 → 做空基線。

正式策略已改 m3（09:03），請用：
    python -m strategy.limitup_fade_ml.experiments.verify_rule_baseline

本檔保留 m5_std@09:05 歷史對照。
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

from data.adjustment_query import load_pattern_day, load_pattern_m5_std

LIMIT_UP_RET = 0.095
MIN_BODY = 0.50
MAX_UPPER = 0.20
FIRST_M5_TIME = dtime(9, 5)


def _is_stock_id(sid: str) -> bool:
    s = str(sid)
    return len(s) == 4 and s.isdigit() and not s.startswith("00")


def _summarize(label: str, df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print(f"  {label}: n=0", flush=True)
        return
    p_oc = 100.0 * df["close_lt_open"].mean()
    p_pc = 100.0 * df["close_lt_prev"].mean()
    print(
        f"  {label}: n={n:,}  收<開={p_oc:.1f}%  收<昨收={p_pc:.1f}%  "
        f"開→收 mean={100 * df['oc_ret'].mean():.3f}%  "
        f"median={100 * df['oc_ret'].median():.3f}%",
        flush=True,
    )


def run(start_date: str, end_date: str) -> pd.DataFrame:
    t0 = time.time()
    print("前日實體漲停 → 今開高 → 首5分下跌 → 收盤跌？", flush=True)
    print(
        f"漲停: 日報酬>={LIMIT_UP_RET:.1%} 陽線 body>={MIN_BODY:.0%} 上影<={MAX_UPPER:.0%}",
        flush=True,
    )
    print("開高: open>前日close；首5分: m5_std@09:05 close<open", flush=True)
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    print(f"載入 pattern day（start={hist})...", flush=True)
    day = load_pattern_day(start_date=hist)
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].map(_is_stock_id)].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)
    print(f"day {len(day):,}", flush=True)

    g = day.groupby("stock_id", sort=False)
    day["prev_close"] = g["close"].shift(1)
    day["prev_open"] = g["open"].shift(1)
    day["prev_high"] = g["high"].shift(1)
    day["prev_low"] = g["low"].shift(1)

    po = day["prev_open"].astype(float)
    ph = day["prev_high"].astype(float)
    pl = day["prev_low"].astype(float)
    pc = day["prev_close"].astype(float)
    ppc = g["close"].shift(2)
    prev_ret = pc / ppc - 1.0
    rng = (ph - pl).replace(0, np.nan)
    body = (pc - po) / rng
    upper = (ph - pc) / rng
    prev_limit_solid = (
        (prev_ret >= LIMIT_UP_RET)
        & (pc > po)
        & (body >= MIN_BODY)
        & (upper <= MAX_UPPER)
    )
    gap_up = day["open"].astype(float) > pc

    cands = day[
        (day["date"] >= start_date)
        & (day["date"] <= end_date)
        & prev_limit_solid
        & gap_up
    ].copy()
    print(f"前日實體漲停且今日開高: {len(cands):,}", flush=True)
    if cands.empty:
        return cands

    cands["day_str"] = cands["date"].dt.strftime("%Y-%m-%d")
    need_sids = set(cands["stock_id"].unique())
    need_days = set(cands["day_str"].unique())

    print("載入 pattern m5_std（只取 09:05 首棒）...", flush=True)
    m5 = load_pattern_m5_std(start_date=start_date)
    m5["stock_id"] = m5["stock_id"].astype(str)
    m5["date"] = pd.to_datetime(m5["date"], format="mixed")
    m5 = m5[
        m5["stock_id"].isin(need_sids)
        & (m5["date"].dt.time == FIRST_M5_TIME)
        & (m5["date"].dt.strftime("%Y-%m-%d").isin(need_days))
    ].copy()
    m5["day_str"] = m5["date"].dt.strftime("%Y-%m-%d")
    print(f"首5分棒 {len(m5):,}", flush=True)

    m5 = m5.drop_duplicates(subset=["stock_id", "day_str"], keep="last")
    first = m5.set_index(["stock_id", "day_str"])[["open", "close", "high", "low"]]
    first.columns = ["m5_open", "m5_close", "m5_high", "m5_low"]

    ev = cands.merge(first, left_on=["stock_id", "day_str"], right_index=True, how="left")
    has_m5 = ev["m5_open"].notna()
    ev_ok = ev[has_m5].copy()
    ev_ok["m5_bear"] = ev_ok["m5_close"].astype(float) < ev_ok["m5_open"].astype(float)
    ev_ok["close_lt_open"] = ev_ok["close"].astype(float) < ev_ok["open"].astype(float)
    ev_ok["close_lt_prev"] = ev_ok["close"].astype(float) < ev_ok["prev_close"].astype(float)
    o = ev_ok["open"].astype(float)
    c = ev_ok["close"].astype(float)
    m5c = ev_ok["m5_close"].astype(float)
    ev_ok["oc_ret"] = (c - o) / o.replace(0, np.nan)
    # 09:05 首5分收確認後做空，收盤平：勝 = 收盤 < 進場價(m5_close)
    ev_ok["short_entry"] = m5c
    ev_ok["short_win"] = c < m5c
    ev_ok["short_ret"] = (m5c - c) / m5c.replace(0, np.nan)

    bear = ev_ok[ev_ok["m5_bear"]].copy()
    bull = ev_ok[~ev_ok["m5_bear"]].copy()

    print(
        f"\n候選開高={len(cands):,} 無首5分={int((~has_m5).sum()):,} "
        f"有首5分={len(ev_ok):,} 首5分跌={len(bear):,} 首5分非跌={len(bull):,}",
        flush=True,
    )

    print("\n" + "=" * 56)
    print("收盤為跌的機率（vs 開盤，非實際進場）")
    print("=" * 56)
    _summarize("開高+首5分跌（目標）", bear)
    _summarize("開高（有首5分，不限方向）", ev_ok)
    _summarize("開高+首5分漲/平", bull)

    print("\n" + "=" * 56)
    print("做空勝率：進場=首5分收(09:05) → 出場=日收")
    print("=" * 56)
    if len(bear):
        n = len(bear)
        wins = int(bear["short_win"].sum())
        print(
            f"  開高+首5分跌: n={n:,}  勝率={100 * bear['short_win'].mean():.1f}%  "
            f"({wins}/{n})  short mean={100 * bear['short_ret'].mean():.3f}%  "
            f"median={100 * bear['short_ret'].median():.3f}%",
            flush=True,
        )
        print(
            f"  （對照）收盤 < 開盤: {100 * bear['close_lt_open'].mean():.1f}%",
            flush=True,
        )
    else:
        print("  n=0", flush=True)

    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return bear


def main():
    p = argparse.ArgumentParser(description="漲停後開高首5分跌→收盤跌機率")
    p.add_argument("--start_date", default="2026-06-01")
    p.add_argument("--end_date", default="2026-07-31")
    args = p.parse_args()
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
