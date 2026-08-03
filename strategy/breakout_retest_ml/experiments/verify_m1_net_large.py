"""
M1 上漲 × 有／無淨大單 對照。

母體：tick400、決策窗內每日每檔第一根實體陽線 M1、ATR5≥門檻、±3%/30分。
淨大單 = 該分鐘大單買占比 − 大單賣占比（單筆 > TICK_LARGE_LOT 張）。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_m1_net_large \\
        --start_date 2026-07-01 --end_date 2026-07-31 --min_atr5 0.01
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
from strategy.breakout_retest_ml.config import (
    LABEL_HORIZON_MINUTES,
    MAX_UPPER_SHADOW_RATIO,
    MIN_BODY_RATIO,
    SESSION_END,
    TICK_LARGE_LOT,
    TP_PCT,
)
from strategy.breakout_retest_ml.features import _passes_m1_body_trigger, _triple_barrier_label
from strategy.breakout_retest_ml.experiments.verify_m5_tick import (
    SESSION_START,
    _add_atr5_day,
    _atr5_at,
    _tick_in_window,
)

# 「大量」淨大單：淨占比達此值
NET_LARGE_THR = 0.10


def _first_solid_m1(m1_day: pd.DataFrame) -> dict | None:
    """09:05～10:00 內第一根實體陽線 M1。"""
    if m1_day is None or m1_day.empty:
        return None
    m1 = (
        m1_day.dropna(subset=["open", "high", "low", "close"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    t0 = dtime(*SESSION_START)
    t1 = dtime(*SESSION_END)
    for _, row in m1.iterrows():
        ts = pd.Timestamp(row["date"])
        # 棒結束時刻落在決策窗（與 M5 一致：用 bar 結束）
        bar_end = ts + pd.Timedelta(minutes=1)
        tm = bar_end.time()
        if tm < t0 or tm > t1:
            continue
        ok, meta = _passes_m1_body_trigger(row)
        if not ok:
            continue
        return {
            "win_start": ts,
            "win_end": bar_end,
            "last_min": ts,
            "trigger_ts": bar_end,
            "entry_price": float(row["close"]),
            **meta,
        }
    return None


def _row_stats(sub: pd.DataFrame) -> dict:
    if sub is None or sub.empty:
        return {"n": 0, "n_tp": 0, "n_sl": 0, "n_flat": 0, "win": 0.0, "fake": 0.0}
    n = len(sub)
    n_tp = int((sub["target"] == 2).sum())
    n_sl = int((sub["target"] == 0).sum())
    n_flat = int((sub["target"] == 1).sum())
    return {
        "n": n,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_flat": n_flat,
        "win": 100.0 * n_tp / n,
        "fake": 100.0 * n_sl / n,
    }


def _print_row(label: str, st: dict) -> None:
    print(
        f"{label:>22} {st['n']:>8} {st['n_tp']:>6} {st['n_sl']:>6} {st['n_flat']:>6} "
        f"{st['win']:>7.1f}% {st['fake']:>7.1f}%",
        flush=True,
    )


def run(start_date: str, end_date: str, min_atr5: float = 0.01) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("M1 上漲 × 有／無淨大單", flush=True)
    print(f"母體: tick400 第一根實體陽線 M1；ATR5>={min_atr5}", flush=True)
    print(
        f"決策窗: {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}；"
        f"body≥{MIN_BODY_RATIO:.0%} 上影≤{MAX_UPPER_SHADOW_RATIO:.0%}",
        flush=True,
    )
    print(
        f"淨大單: 該分鐘 (大買/總量 − 大賣/總量)，單筆>{TICK_LARGE_LOT}張；"
        f"大量門檻 net≥{NET_LARGE_THR:.0%}",
        flush=True,
    )
    print(f"標籤 ±{TP_PCT:.0%}/{LABEL_HORIZON_MINUTES}分；區間 {start_date} ~ {end_date}\n", flush=True)

    print(f"載入 pattern M1（start={start_date})...", flush=True)
    m1 = load_pattern_m1(start_date=start_date)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(stock_ids)].copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[(m1["day_str"] >= start_date) & (m1["day_str"] <= end_date)]
    print(f"M1 {len(m1):,} / stocks={m1['stock_id'].nunique()}", flush=True)

    grouped = m1.groupby(["stock_id", "day_str"], sort=False)
    n_pairs = grouped.ngroups
    rows: list[dict] = []
    n_hit = 0
    for i, ((sid, trade_day), m1_day) in enumerate(grouped, start=1):
        trig = _first_solid_m1(m1_day)
        if trig is None:
            continue
        n_hit += 1
        m1_atr = _add_atr5_day(m1_day)
        atr5 = _atr5_at(m1_atr, trig["last_min"])
        if not np.isfinite(atr5) or atr5 < min_atr5:
            continue
        label_raw = _triple_barrier_label(m1_day, trig["trigger_ts"], trig["entry_price"])
        if not np.isfinite(label_raw):
            continue
        rows.append(
            {
                "stock_id": str(sid),
                "trade_date": str(trade_day),
                "win_start": trig["win_start"],
                "win_end": trig["win_end"],
                "trigger_ts": trig["trigger_ts"],
                "entry_price": trig["entry_price"],
                "atr5": atr5,
                "target": {-1.0: 0, 0.0: 1, 1.0: 2}[float(label_raw)],
            }
        )
        if i % 3000 == 0:
            print(
                f"  [m1] {i}/{n_pairs} labeled={len(rows)} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    ev = pd.DataFrame(rows)
    print(f"\nstock×day={n_pairs:,} M1 hit={n_hit:,} ATR+標籤={len(ev):,}", flush=True)
    if ev.empty:
        return ev

    print("讀取 tick...", flush=True)
    feats: list[dict] = []
    for j, r in enumerate(ev.itertuples(index=False), start=1):
        try:
            ticks = load_tick_by_stock(str(r.stock_id), date=str(r.trade_date)[:10])
        except Exception:
            ticks = pd.DataFrame()
        feats.append(_tick_in_window(ticks, r.win_start, r.win_end))
        if j % 200 == 0 or j == len(ev):
            print(f"  [tick] {j}/{len(ev)} elapsed={time.time()-t0:.0f}s", flush=True)
    ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(feats)], axis=1)

    net = ev["bar_large_net_ratio"].astype(float)
    print("\n" + "=" * 72)
    print("有淨大單 vs 無淨大單")
    print("=" * 72)
    print(f"net 分布: mean={net.mean():.3f} p50={net.median():.3f} "
          f"p75={net.quantile(0.75):.3f} %>0={(net>0).mean()*100:.1f}%", flush=True)

    hdr = f"{'條件':>22} {'進場總數':>8} {'勝':>6} {'敗':>6} {'持平':>6} {'勝率':>8} {'騙線%':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    _print_row("僅M1上漲+ATR", _row_stats(ev))
    _print_row("有淨大單(net>0)", _row_stats(ev[net > 0]))
    _print_row("無淨大單(net≤0)", _row_stats(ev[net <= 0]))
    _print_row(f"大量淨大(net≥{NET_LARGE_THR:.0%})", _row_stats(ev[net >= NET_LARGE_THR]))
    _print_row(f"非大量(net<{NET_LARGE_THR:.0%})", _row_stats(ev[net < NET_LARGE_THR]))
    buy_gt = ev["bar_large_buy_ratio"] > ev["bar_large_sell_ratio"]
    _print_row("大買>大賣", _row_stats(ev[buy_gt]))
    _print_row("非大買>大賣", _row_stats(ev[~buy_gt]))

    print(
        f"\n解讀: 若「有／大量淨大單」勝率明顯高於「無／非大量」，才值得當過濾。"
        f" 耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return ev


def main():
    p = argparse.ArgumentParser(description="M1 上漲 × 有／無淨大單")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument("--min_atr5", type=float, default=0.01)
    args = p.parse_args()
    run(args.start_date, args.end_date, min_atr5=args.min_atr5)


if __name__ == "__main__":
    main()
