"""
tick_universe（~400 檔）無支撐硬過濾：滾動 5 分實體陽線 + ATR5 + 大買>大賣。

與 verify_m5_tick 相同盤中條件／標籤，但母體改為每日每檔（非 breakout 支撐候選）。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_m5_tick_universe \\
        --start_date 2026-05-01 --end_date 2026-07-31 --min_atr5 0
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
from strategy.breakout_retest_ml.config import (
    LABEL_HORIZON_MINUTES,
    MAX_UPPER_SHADOW_RATIO,
    MIN_BODY_RATIO,
    SESSION_END,
    TICK_LARGE_LOT,
    TP_PCT,
)
from strategy.breakout_retest_ml.features import _triple_barrier_label
from strategy.breakout_retest_ml.experiments.verify_m5_tick import (
    DEFAULT_MIN_ATR5,
    ROLL_MINUTES,
    SESSION_START,
    _add_atr5_day,
    _atr5_at,
    _first_rolling_solid_m5,
    _label_stats,
    _print_direction_table,
    _print_outcome_row,
    _tick_in_window,
)


def run(start_date: str, end_date: str, min_atr5: float = DEFAULT_MIN_ATR5) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("tick400 無支撐：滾動5分實體 + ATR5 + 大買>大賣", flush=True)
    print(f"母體: tick_universe {len(stock_ids)} 檔（無 breakout／支撐硬過濾）", flush=True)
    print(
        f"合成門檻: 連續 {ROLL_MINUTES} 根 M1、body≥{MIN_BODY_RATIO:.0%}、" f"上影≤{MAX_UPPER_SHADOW_RATIO:.0%}",
        flush=True,
    )
    print(
        f"決策窗: 窗結束落在 {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}；每日每檔第一根",
        flush=True,
    )
    print(
        f"ATR5 過濾: atr5>= {min_atr5:.5f}（收集時門檻；表內另掃 0.006/0.008/0.010）",
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
    trading_days = sorted(m1["day_str"].unique())
    print(
        f"M1 {len(m1):,} / stocks={m1['stock_id'].nunique()} / days={len(trading_days)}",
        flush=True,
    )

    grouped = m1.groupby(["stock_id", "day_str"], sort=False)
    n_pairs = grouped.ngroups
    print(f"待掃 stock×day: {n_pairs:,}", flush=True)

    # 第一輪：只算 M5 + ATR + 標籤（不讀 tick）
    rows: list[dict] = []
    n_hit = 0
    n_atr_drop = 0
    n_no_label = 0
    for i, ((sid, trade_day), m1_day) in enumerate(grouped, start=1):
        trig = _first_rolling_solid_m5(m1_day)
        if trig is None:
            if i % 2000 == 0:
                print(
                    f"  [m5] {i}/{n_pairs} hit={n_hit} labeled={len(rows)} " f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
            continue
        n_hit += 1
        m1_atr = _add_atr5_day(m1_day)
        atr5 = _atr5_at(m1_atr, trig["last_min"])
        if min_atr5 > 0 and (not np.isfinite(atr5) or atr5 < min_atr5):
            n_atr_drop += 1
            continue
        label_raw = _triple_barrier_label(m1_day, trig["trigger_ts"], trig["entry_price"])
        if not np.isfinite(label_raw):
            n_no_label += 1
            continue
        rows.append(
            {
                "stock_id": str(sid),
                "trade_date": str(trade_day),
                "win_start": trig["win_start"],
                "win_end": trig["win_end"],
                "trigger_ts": trig["trigger_ts"],
                "entry_price": trig["entry_price"],
                "body_ratio": trig["body_ratio"],
                "upper_shadow_ratio": trig["upper_shadow_ratio"],
                "atr5": atr5,
                "target": {-1.0: 0, 0.0: 1, 1.0: 2}[float(label_raw)],
                "label_raw": float(label_raw),
            }
        )
        if i % 2000 == 0 or len(rows) % 500 == 0:
            print(
                f"  [m5] {i}/{n_pairs} hit={n_hit} labeled={len(rows)} " f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    ev = pd.DataFrame(rows)
    print(
        f"\ntick400×日: {n_pairs:,}；有滾動實體5分: {n_hit:,}；"
        f"ATR5剔除: {n_atr_drop:,}；無標籤: {n_no_label:,}；標籤完整: {len(ev):,}",
        flush=True,
    )
    if ev.empty:
        return ev

    sample = ev.head(5)[["stock_id", "trade_date", "win_start", "win_end", "atr5", "body_ratio"]]
    print("\n樣例視窗:")
    print(sample.to_string(index=False), flush=True)

    atr_levels = (0.0, 0.006, 0.008, 0.010) if min_atr5 <= 0 else (min_atr5,)
    print("\n" + "=" * 64)
    print("ATR5 門檻對照（尚未加大單條件；母體＝每日第一根實體5分）")
    print("=" * 64)
    hdr = f"{'min_atr5':>10} {'進場總數':>8} {'勝':>6} {'敗':>6} {'持平':>6} {'勝率':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for thr in atr_levels:
        sub = ev if thr <= 0 else ev[ev["atr5"] >= thr]
        st = _label_stats(sub["target"].tolist() if not sub.empty else [])
        _print_outcome_row(f"{thr:.3f}", st)

    # 第二輪：只對方向表需要的 ATR 門檻讀 tick（最低 0.006 或 CLI min_atr5）
    sweep_atr = [t for t in atr_levels if t > 0] or [min_atr5]
    tick_floor = min(sweep_atr) if sweep_atr else 0.0
    need_tick = ev[ev["atr5"] >= tick_floor].copy() if tick_floor > 0 else ev.copy()
    print(
        f"\n讀取 tick（atr5>={tick_floor:.5f}）: {len(need_tick):,} 筆...",
        flush=True,
    )
    tick_rows: list[dict] = []
    for j, r in enumerate(need_tick.itertuples(index=False), start=1):
        try:
            ticks = load_tick_by_stock(str(r.stock_id), date=str(r.trade_date))
        except Exception:
            ticks = pd.DataFrame()
        feat = _tick_in_window(ticks, r.win_start, r.win_end)
        tick_rows.append(feat)
        if j % 200 == 0 or j == len(need_tick):
            print(f"  [tick] {j}/{len(need_tick)} elapsed={time.time()-t0:.0f}s", flush=True)
    tick_df = pd.DataFrame(tick_rows)
    need_tick = pd.concat([need_tick.reset_index(drop=True), tick_df], axis=1)

    for atr_thr in sweep_atr:
        base = need_tick[need_tick["atr5"] >= atr_thr].copy()
        dom = base[base["bar_large_buy_ratio"] > base["bar_large_sell_ratio"]]
        print("\n" + "=" * 64)
        print(f"大買>大賣 總覽（ATR5>={atr_thr:.5f}；" f"M5 n={len(base)} → 買>賣 n={len(dom)}）")
        print("=" * 64)
        hdr = f"{'條件':>10} {'進場總數':>8} {'勝':>6} {'敗':>6} {'持平':>6} {'勝率':>8}"
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)
        _print_outcome_row("M5only", _label_stats(base["target"].tolist()))
        _print_outcome_row("買>賣", _label_stats(dom["target"].tolist() if not dom.empty else []))

        _print_direction_table(
            base,
            f"ATR5>={atr_thr:.5f} n={len(base)} 單筆>{TICK_LARGE_LOT}張",
        )

    print(
        f"\n說明: 無支撐硬過濾；勝率=勝/進場總數；標籤 ±{TP_PCT:.0%}/{LABEL_HORIZON_MINUTES}分。"
        f"\n對照: 支撐版同季淨大單≥10%（ATR≥0.006）約 29 筆。",
        flush=True,
    )
    print(f"耗時 {time.time()-t0:.1f}s", flush=True)
    return ev


def main():
    p = argparse.ArgumentParser(description="tick400 無支撐：滾動5分 + ATR5 + 大買>大賣")
    p.add_argument("--start_date", default="2026-05-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument(
        "--min_atr5",
        type=float,
        default=0.0,
        help="收集時最低 atr5；0=全收再於表內掃門檻。預設 0",
    )
    args = p.parse_args()
    run(args.start_date, args.end_date, min_atr5=args.min_atr5)


if __name__ == "__main__":
    main()
