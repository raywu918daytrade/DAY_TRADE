"""
盤整期淨大單能否預測接下來突破方向（上／下）。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_consol_net_breakout_dir \\
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

CONSOL_BARS = 15
FWD_MINUTES = 30
MIN_NONFLAT = 10
DEFAULT_RANGE_PCT = 0.01


def _first_consol(
    m1_day: pd.DataFrame,
    range_pct: float = DEFAULT_RANGE_PCT,
) -> dict | None:
    """決策窗內最早一個 15 分盤整窗。"""
    if m1_day is None or m1_day.empty:
        return None
    m1 = (
        m1_day.dropna(subset=["open", "high", "low", "close"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(m1) < CONSOL_BARS:
        return None

    t0 = dtime(*SESSION_START)
    t1 = dtime(*SESSION_END)

    for i in range(CONSOL_BARS - 1, len(m1)):
        window = m1.iloc[i - (CONSOL_BARS - 1) : i + 1]
        dates = pd.to_datetime(window["date"], format="mixed")
        # 連續分鐘
        deltas = dates.diff().dt.total_seconds().iloc[1:]
        if not (deltas == 60).all():
            continue
        win_start = pd.Timestamp(dates.iloc[0])
        last_min = pd.Timestamp(dates.iloc[-1])
        win_end = last_min + pd.Timedelta(minutes=1)
        # 盤整結束時刻落在決策窗（之後還要 30 分標籤，結束不宜太晚）
        if win_end.time() < t0 or win_end.time() > t1:
            continue

        o0 = float(window.iloc[0]["open"])
        if not np.isfinite(o0) or o0 <= 0:
            continue
        hi = float(window["high"].astype(float).max())
        lo = float(window["low"].astype(float).min())
        if (hi - lo) / o0 > range_pct:
            continue
        opens = window["open"].astype(float)
        closes = window["close"].astype(float)
        nonflat = int((closes != opens).sum())
        if nonflat < MIN_NONFLAT:
            continue
        return {
            "win_start": win_start,
            "win_end": win_end,
            "range_high": hi,
            "range_low": lo,
            "open_first": o0,
            "range_pct": (hi - lo) / o0,
        }
    return None


def _breakout_label(
    m1_day: pd.DataFrame,
    win_end: pd.Timestamp,
    range_high: float,
    range_low: float,
) -> int:
    """+1 先觸上、-1 先觸下、0 未突破、nan 資料不足。"""
    fut = m1_day[
        (m1_day["date"] >= win_end)
        & (m1_day["date"] < win_end + pd.Timedelta(minutes=FWD_MINUTES))
    ]
    if fut.empty:
        return 0
    # 需要足夠未來棒才算「未突破」，否則仍標 0（未突破）以簡化
    for _, row in fut.iterrows():
        h = float(row["high"])
        l = float(row["low"])
        up = h >= range_high
        dn = l <= range_low
        if up and dn:
            # 同棒雙觸：用收盤相對中軸
            mid = 0.5 * (range_high + range_low)
            c = float(row["close"])
            return 1 if c >= mid else -1
        if up:
            return 1
        if dn:
            return -1
    return 0


def _hit_stats(sub: pd.DataFrame) -> dict:
    """有方向淨大單且有突破者的命中率。"""
    if sub is None or sub.empty:
        return {"n": 0, "hit": 0.0, "n_up": 0, "n_dn": 0}
    # pred: sign(net); label: breakout dir
    use = sub[(sub["net"] != 0) & (sub["brk"] != 0)].copy()
    n = len(use)
    if n == 0:
        return {"n": 0, "hit": 0.0, "n_up": 0, "n_dn": 0}
    pred = np.where(use["net"] > 0, 1, -1)
    hit = float((pred == use["brk"].to_numpy()).mean() * 100)
    return {
        "n": n,
        "hit": hit,
        "n_up": int((use["brk"] == 1).sum()),
        "n_dn": int((use["brk"] == -1).sum()),
    }


def run(
    start_date: str,
    end_date: str,
    max_rows: int = 2000,
    range_pct: float = DEFAULT_RANGE_PCT,
) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("盤整期淨大單 → 預測突破方向", flush=True)
    print(
        f"母體 tick400；決策窗 {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}",
        flush=True,
    )
    print(
        f"盤整: {CONSOL_BARS}根 M1、振幅≤{range_pct:.1%}、非平盤≥{MIN_NONFLAT}；"
        f"突破: 結束後{FWD_MINUTES}分先觸 high/low；淨大單 >{TICK_LARGE_LOT}張",
        flush=True,
    )
    print(f"每 stock×day 取最早1窗；max_rows={max_rows}；區間 {start_date}~{end_date}\n", flush=True)

    print(f"載入 pattern M1（start={start_date})...", flush=True)
    m1 = load_pattern_m1(start_date=start_date)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(stock_ids)].copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[(m1["day_str"] >= start_date) & (m1["day_str"] <= end_date)]
    print(f"M1 {len(m1):,}", flush=True)

    rows: list[dict] = []
    grouped = m1.groupby(["stock_id", "day_str"], sort=False)
    for i, ((sid, day), g) in enumerate(grouped, start=1):
        g = g.sort_values("date").reset_index(drop=True)
        consol = _first_consol(g, range_pct=range_pct)
        if consol is None:
            continue
        brk = _breakout_label(g, consol["win_end"], consol["range_high"], consol["range_low"])
        rows.append(
            {
                "stock_id": str(sid),
                "trade_date": str(day),
                "win_start": consol["win_start"],
                "win_end": consol["win_end"],
                "range_high": consol["range_high"],
                "range_low": consol["range_low"],
                "range_pct": consol["range_pct"],
                "brk": brk,
            }
        )
        if len(rows) >= max_rows:
            break
        if i % 2000 == 0:
            print(f"  [scan] {i} found={len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)

    ev = pd.DataFrame(rows)
    print(f"盤整窗={len(ev):,}", flush=True)
    if ev.empty:
        return ev

    print("讀取 tick...", flush=True)
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
            print(f"  [tick] {j}/{len(ev)} elapsed={time.time()-t0:.0f}s", flush=True)
    ev = ev.copy()
    ev["net"] = nets
    ev["abs_net"] = ev["net"].abs()

    n_up = int((ev["brk"] == 1).sum())
    n_dn = int((ev["brk"] == -1).sum())
    n_flat = int((ev["brk"] == 0).sum())
    n_brk = n_up + n_dn
    maj = max(n_up, n_dn) / n_brk * 100 if n_brk else 0.0

    print("\n" + "=" * 64)
    print("突破標籤分布")
    print("=" * 64)
    print(
        f"n={len(ev)}  向上={n_up} ({100*n_up/len(ev):.1f}%)  "
        f"向下={n_dn} ({100*n_dn/len(ev):.1f}%)  "
        f"未突破={n_flat} ({100*n_flat/len(ev):.1f}%)",
        flush=True,
    )
    print(f"有突破 n={n_brk}；多數類別基線={maj:.1f}%；隨機≈50%", flush=True)

    print("\n" + "=" * 64)
    print("sign(net) 預測突破方向")
    print("=" * 64)
    st = _hit_stats(ev)
    print(
        f"全體(net≠0且有突破): n={st['n']}  命中率={st['hit']:.1f}%  "
        f"(標籤上={st['n_up']} 下={st['n_dn']})",
        flush=True,
    )
    n0 = int(((ev["net"] == 0) & (ev["brk"] != 0)).sum())
    print(f"net=0 且有突破: n={n0}（無方向可預測）", flush=True)

    # |net| 分位（僅有突破且 net≠0）
    use = ev[(ev["net"] != 0) & (ev["brk"] != 0)].copy()
    if len(use) >= 20:
        try:
            use["q"] = pd.qcut(use["abs_net"], 3, labels=["小", "中", "大"], duplicates="drop")
        except ValueError:
            use["q"] = "all"
        print("\n|net| 三分位命中率", flush=True)
        hdr = f"{'|net|':>6} {'n':>6} {'命中%':>8} {'mean|net|':>10} {'上':>5} {'下':>5}"
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)
        for qv, g in use.groupby("q", observed=True):
            hs = _hit_stats(g)
            print(
                f"{str(qv):>6} {hs['n']:>6} {hs['hit']:>7.1f}% "
                f"{g['abs_net'].mean():>10.3f} {hs['n_up']:>5} {hs['n_dn']:>5}",
                flush=True,
            )

    # net>0 vs net<0 各自後驗突破方向
    print("\n條件分布 P(突破方向 | sign(net))", flush=True)
    for label, mask in (("net>0", ev["net"] > 0), ("net<0", ev["net"] < 0)):
        g = ev[mask & (ev["brk"] != 0)]
        n = len(g)
        if n == 0:
            continue
        print(
            f"  {label}: n={n}  →上={100*(g['brk']==1).mean():.1f}%  "
            f"→下={100*(g['brk']==-1).mean():.1f}%",
            flush=True,
        )

    print(
        f"\n解讀: 命中率需明顯高於多數類別基線({maj:.1f}%)且|net|大時更高，"
        f"才像有預測力。耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return ev


def main():
    p = argparse.ArgumentParser(description="盤整期淨大單→突破方向")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-07-07")
    p.add_argument("--max_rows", type=int, default=2000)
    p.add_argument("--range_pct", type=float, default=DEFAULT_RANGE_PCT)
    args = p.parse_args()
    run(args.start_date, args.end_date, max_rows=args.max_rows, range_pct=args.range_pct)


if __name__ == "__main__":
    main()
