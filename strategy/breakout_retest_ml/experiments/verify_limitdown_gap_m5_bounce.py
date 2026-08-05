"""
前1日實體跌停 → 今日開低 → 拉回再買（首5分續跌後做多）。

定義：
- 前日跌停：close/prev_close-1 <= -9.5%，且陰線實體（body>=50%、下影<=20%）
- 今日開低：open < 前日 close
- 拉回：m5_std 首棒（09:05）close < open（再開低後繼續往下）
- 做多：進場=首5分收 → 出場=日收；勝 = close > entry

對照：首5分漲（反彈追多）、開低不限方向。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_limitdown_gap_m5_bounce \\
        --start_date 2026-06-01 --end_date 2026-07-31
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

LIMIT_DOWN_RET = -0.095
MIN_BODY = 0.50
MAX_LOWER = 0.20
FIRST_M5_TIME = dtime(9, 5)


def _is_stock_id(sid: str) -> bool:
    s = str(sid)
    return len(s) == 4 and s.isdigit() and not s.startswith("00")


def _summarize(label: str, df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print(f"  {label}: n=0", flush=True)
        return
    p_oc = 100.0 * df["close_gt_open"].mean()
    p_pc = 100.0 * df["close_gt_prev"].mean()
    print(
        f"  {label}: n={n:,}  收>開={p_oc:.1f}%  收>昨收={p_pc:.1f}%  "
        f"開→收 mean={100 * df['oc_ret'].mean():.3f}%  "
        f"median={100 * df['oc_ret'].median():.3f}%",
        flush=True,
    )


def run(start_date: str, end_date: str) -> pd.DataFrame:
    t0 = time.time()
    print("前日實體跌停 → 今開低 → 首5分續跌（拉回）→ 做多？", flush=True)
    print(
        f"跌停: 日報酬<={LIMIT_DOWN_RET:.1%} 陰線 body>={MIN_BODY:.0%} 下影<={MAX_LOWER:.0%}",
        flush=True,
    )
    print("開低: open<前日close；拉回: m5_std@09:05 close<open", flush=True)
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
    # 陰線實體：body = (open-close)/range；下影 = (close-low)/range（收在實體下緣附近）
    body = (po - pc) / rng
    lower = (pc - pl) / rng
    prev_limit_solid = (
        (prev_ret <= LIMIT_DOWN_RET)
        & (pc < po)
        & (body >= MIN_BODY)
        & (lower <= MAX_LOWER)
    )
    gap_down = day["open"].astype(float) < pc

    cands = day[
        (day["date"] >= start_date)
        & (day["date"] <= end_date)
        & prev_limit_solid
        & gap_down
    ].copy()
    print(f"前日實體跌停且今日開低: {len(cands):,}", flush=True)
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
    ev_ok["m5_bull"] = ev_ok["m5_close"].astype(float) > ev_ok["m5_open"].astype(float)
    ev_ok["close_gt_open"] = ev_ok["close"].astype(float) > ev_ok["open"].astype(float)
    ev_ok["close_gt_prev"] = ev_ok["close"].astype(float) > ev_ok["prev_close"].astype(float)
    o = ev_ok["open"].astype(float)
    c = ev_ok["close"].astype(float)
    m5c = ev_ok["m5_close"].astype(float)
    ev_ok["oc_ret"] = (c - o) / o.replace(0, np.nan)
    # 09:05 拉回確認後做多，收盤平：勝 = 收盤 > 進場價(m5_close)
    ev_ok["long_entry"] = m5c
    ev_ok["long_win"] = c > m5c
    ev_ok["long_ret"] = (c - m5c) / m5c.replace(0, np.nan)

    pullback = ev_ok[ev_ok["m5_bear"]].copy()  # 目標：拉回再買
    bounce = ev_ok[ev_ok["m5_bull"]].copy()  # 對照：反彈追多

    print(
        f"\n候選開低={len(cands):,} 無首5分={int((~has_m5).sum()):,} "
        f"有首5分={len(ev_ok):,} 首5分跌(拉回)={len(pullback):,} "
        f"首5分漲={len(bounce):,}",
        flush=True,
    )

    print("\n" + "=" * 56)
    print("收盤為漲的機率（vs 開盤，非實際進場）")
    print("=" * 56)
    _summarize("開低+首5分跌／拉回（目標）", pullback)
    _summarize("開低（有首5分，不限方向）", ev_ok)
    _summarize("開低+首5分漲（反彈追多）", bounce)

    print("\n" + "=" * 56)
    print("做多勝率：進場=首5分收(09:05) → 出場=日收")
    print("=" * 56)

    def _long_line(label: str, df: pd.DataFrame) -> None:
        if df.empty:
            print(f"  {label}: n=0", flush=True)
            return
        n = len(df)
        wins = int(df["long_win"].sum())
        print(
            f"  {label}: n={n:,}  勝率={100 * df['long_win'].mean():.1f}%  "
            f"({wins}/{n})  long mean={100 * df['long_ret'].mean():.3f}%  "
            f"median={100 * df['long_ret'].median():.3f}%",
            flush=True,
        )
        print(
            f"    （對照）收盤 > 開盤: {100 * df['close_gt_open'].mean():.1f}%",
            flush=True,
        )

    _long_line("開低+首5分跌（拉回再買）", pullback)
    _long_line("開低+首5分漲（對照）", bounce)

    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return pullback


def main():
    p = argparse.ArgumentParser(description="跌停後開低拉回再買→做多勝率")
    p.add_argument("--start_date", default="2026-06-01")
    p.add_argument("--end_date", default="2026-07-31")
    args = p.parse_args()
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
