"""
Tick 大單「筆數頻率」＋方向一致 vs 突破真偽（騙線）。

母體：tick400、每日第一根滾動5分實體陽線、ATR≥0.01、±3%/30分。
對 lot∈{10,30,50} 算窗內大單筆數頻率與買側占比。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_tick_freq \\
        --start_date 2026-05-01 --end_date 2026-07-31 --min_atr5 0.01
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
    TP_PCT,
)
from strategy.breakout_retest_ml.features import _triple_barrier_label
from strategy.breakout_retest_ml.experiments.verify_m5_tick import (
    ROLL_MINUTES,
    SESSION_START,
    _add_atr5_day,
    _atr5_at,
    _first_rolling_solid_m5,
)

LOTS = (10, 30, 50)
BUY_SHARE_HI = 0.70
BUY_SHARE_LO = 0.30
ABS_FREQ_THRESHOLDS = (3, 5, 10)


def _freq_in_window(
    ticks: pd.DataFrame,
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
    lots: tuple[int, ...] = LOTS,
) -> dict:
    """窗內各大單張數門檻的筆數頻率與買側占比。"""
    out: dict = {}
    for lot in lots:
        out[f"freq_{lot}"] = 0
        out[f"n_buy_{lot}"] = 0
        out[f"n_sell_{lot}"] = 0
        out[f"buy_share_{lot}"] = np.nan

    if ticks is None or ticks.empty:
        return out

    t = ticks.copy()
    t["date"] = pd.to_datetime(t["date"], format="mixed")
    w = t[(t["date"] >= win_start) & (t["date"] < win_end)]
    if w.empty:
        return out

    vol = w["volume"].astype(float)
    is_buy = w["tick_type"] == 1
    for lot in lots:
        large = vol > lot
        n_buy = int((large & is_buy).sum())
        n_sell = int((large & ~is_buy).sum())
        freq = n_buy + n_sell
        out[f"freq_{lot}"] = freq
        out[f"n_buy_{lot}"] = n_buy
        out[f"n_sell_{lot}"] = n_sell
        out[f"buy_share_{lot}"] = (n_buy / freq) if freq > 0 else np.nan
    return out


def _row_stats(sub: pd.DataFrame) -> dict:
    if sub is None or sub.empty:
        return {
            "n": 0,
            "n_tp": 0,
            "n_sl": 0,
            "n_flat": 0,
            "win": 0.0,
            "fake_all": 0.0,
            "fake_dec": 0.0,
        }
    n = len(sub)
    n_tp = int((sub["target"] == 2).sum())
    n_sl = int((sub["target"] == 0).sum())
    n_flat = int((sub["target"] == 1).sum())
    dec = n_tp + n_sl
    return {
        "n": n,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_flat": n_flat,
        "win": 100.0 * n_tp / n,
        "fake_all": 100.0 * n_sl / n,  # 騙線率（含持平分母）
        "fake_dec": 100.0 * n_sl / dec if dec else 0.0,  # 決勝負中的騙線
    }


def _print_row(label: str, st: dict) -> None:
    print(
        f"{label:>18} {st['n']:>8} {st['n_tp']:>6} {st['n_sl']:>6} {st['n_flat']:>6} "
        f"{st['win']:>7.1f}% {st['fake_all']:>7.1f}% {st['fake_dec']:>7.1f}%",
        flush=True,
    )


def _print_lot_tables(ev: pd.DataFrame, lot: int) -> None:
    freq_col = f"freq_{lot}"
    share_col = f"buy_share_{lot}"
    print("\n" + "=" * 88)
    print(f"lot>{lot} 張（窗內大單筆數頻率）")
    print("=" * 88)

    # 分位數
    freq = ev[freq_col].astype(float)
    p75 = float(freq.quantile(0.75)) if len(ev) else 0.0
    p90 = float(freq.quantile(0.90)) if len(ev) else 0.0
    print(
        f"freq 分布: mean={freq.mean():.2f} p50={freq.median():.1f} "
        f"p75={p75:.1f} p90={p90:.1f} max={freq.max():.0f}",
        flush=True,
    )

    hdr = (
        f"{'條件':>18} {'進場總數':>8} {'勝':>6} {'敗':>6} {'持平':>6} "
        f"{'勝率':>8} {'騙線%':>8} {'決騙線%':>8}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    print("說明: 騙線%=敗/總數；決騙線%=敗/(勝+敗)。勝=+3%先觸，敗=-3%先觸。", flush=True)

    _print_row("僅M5+ATR", _row_stats(ev))

    for name, thr in (("高頻p75", p75), ("高頻p90", p90)):
        if thr <= 0:
            hi = ev[ev[freq_col] > 0] if thr == 0 else ev.iloc[0:0]
        else:
            hi = ev[ev[freq_col] >= thr]
        _print_row(f"{name}(≥{thr:.0f})", _row_stats(hi))
        if hi.empty:
            continue
        share = hi[share_col]
        buy_ok = hi[share >= BUY_SHARE_HI]
        inconsist = hi[(share > BUY_SHARE_LO) & (share < BUY_SHARE_HI)]
        _print_row(f"  +買側一致≥{BUY_SHARE_HI:.0%}", _row_stats(buy_ok))
        _print_row("  +方向不一致", _row_stats(inconsist))

    print("\n絕對頻率門檻:", flush=True)
    for thr in ABS_FREQ_THRESHOLDS:
        hi = ev[ev[freq_col] >= thr]
        _print_row(f"freq≥{thr}", _row_stats(hi))
        if hi.empty:
            continue
        share = hi[share_col]
        buy_ok = hi[share >= BUY_SHARE_HI]
        inconsist = hi[(share > BUY_SHARE_LO) & (share < BUY_SHARE_HI)]
        _print_row(f"  +買一致", _row_stats(buy_ok))
        _print_row("  +不一致", _row_stats(inconsist))


def run(start_date: str, end_date: str, min_atr5: float = 0.01) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("Tick 頻率／方向一致／騙線驗證", flush=True)
    print(f"母體: tick400 滾動5分實體陽線；ATR5>={min_atr5}", flush=True)
    print(
        f"決策窗: {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}；"
        f"body≥{MIN_BODY_RATIO:.0%} 上影≤{MAX_UPPER_SHADOW_RATIO:.0%}；"
        f"連續{ROLL_MINUTES}根",
        flush=True,
    )
    print(
        f"大單筆數: lot>{list(LOTS)}；買側一致 buy_share≥{BUY_SHARE_HI:.0%}；"
        f"標籤 ±{TP_PCT:.0%}/{LABEL_HORIZON_MINUTES}分",
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

    grouped = m1.groupby(["stock_id", "day_str"], sort=False)
    n_pairs = grouped.ngroups
    rows: list[dict] = []
    n_hit = 0
    for i, ((sid, trade_day), m1_day) in enumerate(grouped, start=1):
        trig = _first_rolling_solid_m5(m1_day)
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
                f"  [m5] {i}/{n_pairs} labeled={len(rows)} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    ev = pd.DataFrame(rows)
    print(f"\nstock×day={n_pairs:,} M5 hit={n_hit:,} ATR+標籤={len(ev):,}", flush=True)
    if ev.empty:
        return ev

    print("讀取 tick 並算頻率...", flush=True)
    feats: list[dict] = []
    for j, r in enumerate(ev.itertuples(index=False), start=1):
        try:
            ticks = load_tick_by_stock(str(r.stock_id), date=str(r.trade_date)[:10])
        except Exception:
            ticks = pd.DataFrame()
        feats.append(_freq_in_window(ticks, r.win_start, r.win_end, LOTS))
        if j % 200 == 0 or j == len(ev):
            print(f"  [tick] {j}/{len(ev)} elapsed={time.time()-t0:.0f}s", flush=True)
    ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(feats)], axis=1)

    for lot in LOTS:
        _print_lot_tables(ev, lot)

    print(
        f"\n解讀: 若「高頻+買側一致」勝率↑且騙線%↓，而「高頻+不一致」沒有，"
        f"則支持頻率＋方向假設。耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return ev


def main():
    p = argparse.ArgumentParser(description="Tick 大單頻率／方向一致／騙線驗證")
    p.add_argument("--start_date", default="2026-05-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument("--min_atr5", type=float, default=0.01)
    args = p.parse_args()
    run(args.start_date, args.end_date, min_atr5=args.min_atr5)


if __name__ == "__main__":
    main()
