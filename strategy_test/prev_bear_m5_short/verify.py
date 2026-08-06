"""
鎖定設計（小樣表現最佳、不用 atr5）：
昨陰實體≥5% → 今開高≥2% → 0050 開低 → 首 m5@09:05 跌
→ 做空 TB 對稱 ±3% / 最多持有 30 分。

用法：
    python -m strategy_test.prev_bear_m5_short.verify \\
        --start_date 2024-01-01 --end_date 2026-07-31
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import time as dtime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from data.adjustment_query import load_pattern_day, load_pattern_m5_std
from finmind.tick_universe import load_tick_universe
from strategy.mkt.config import IDX_SYMBOL

PREV_BODY_MIN = 0.05
OPEN_GAP_MIN = 0.02
FIRST_M5_TIME = dtime(9, 5)
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_M5_BARS = 6  # 30 分鐘


def _short_tb_m5(day_m5: pd.DataFrame, entry_ts: pd.Timestamp, entry: float) -> dict | None:
    """做空對稱 TB：+1 TP（價跌）/ -1 SL（價漲）/ 0 持滿。同根先 TP。"""
    if entry <= 0:
        return None
    fut = day_m5[day_m5["date"] > entry_ts].sort_values("date").head(HOLD_M5_BARS)
    if fut.empty:
        return None
    tp = entry * (1.0 - TP_PCT)
    sl = entry * (1.0 + SL_PCT)
    for j, (_, row) in enumerate(fut.iterrows(), start=1):
        ts = pd.Timestamp(row["date"])
        hi, lo = float(row["high"]), float(row["low"])
        if lo <= tp:
            return {
                "label": 1.0,
                "exit_ts": ts,
                "exit_price": tp,
                "exit_reason": "tp",
                "bars_held": j,
            }
        if hi >= sl:
            return {
                "label": -1.0,
                "exit_ts": ts,
                "exit_price": sl,
                "exit_reason": "sl",
                "bars_held": j,
            }
    last = fut.iloc[-1]
    if len(fut) < HOLD_M5_BARS:
        return None
    return {
        "label": 0.0,
        "exit_ts": pd.Timestamp(last["date"]),
        "exit_price": float(last["close"]),
        "exit_reason": "time",
        "bars_held": len(fut),
    }


def _summarize(df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print("  n=0", flush=True)
        return
    n_tp = int((df["label"] == 1.0).sum())
    n_flat = int((df["label"] == 0.0).sum())
    n_sl = int((df["label"] == -1.0).sum())
    print(f"  n={n:,}", flush=True)
    print(f"  止盈(+{TP_PCT:.0%}): {n_tp:,}  {100 * n_tp / n:.1f}%", flush=True)
    print(f"  震盪(30分): {n_flat:,}  {100 * n_flat / n:.1f}%", flush=True)
    print(f"  止損(-{SL_PCT:.0%}): {n_sl:,}  {100 * n_sl / n:.1f}%", flush=True)
    print(
        f"  做空 mean={100 * df['pnl_pct'].mean():.3f}%  "
        f"median={100 * df['pnl_pct'].median():.3f}%",
        flush=True,
    )


def run(start_date: str, end_date: str) -> pd.DataFrame:
    t0 = time.time()
    universe = {str(s) for s in load_tick_universe()}
    trade_universe = universe - {IDX_SYMBOL}

    print("prev_bear_m5_short（鎖定：無 atr5、對稱±3%）", flush=True)
    print(
        f"母體: tick_universe {len(universe)}（排除 {IDX_SYMBOL} → {len(trade_universe)}）",
        flush=True,
    )
    print(
        f"條件: 昨陰實體≥{PREV_BODY_MIN:.0%}  今開高≥{OPEN_GAP_MIN:.0%}  "
        f"{IDX_SYMBOL}開低  m5@{FIRST_M5_TIME.strftime('%H:%M')} 跌",
        flush=True,
    )
    print(
        f"標籤: 做空 TB ±{TP_PCT:.0%} / 持有 {HOLD_M5_BARS} 根 m5（30分）",
        flush=True,
    )
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    print(f"載入 pattern day（start={hist})...", flush=True)
    day = load_pattern_day(start_date=hist)
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].isin(trade_universe | {IDX_SYMBOL})].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)
    day["day_str"] = day["date"].dt.strftime("%Y-%m-%d")

    g = day.groupby("stock_id", sort=False)
    day["prev_open"] = g["open"].shift(1)
    day["prev_close"] = g["close"].shift(1)

    idx = day[day["stock_id"] == IDX_SYMBOL].copy()
    idx_down_days = set(
        idx.loc[
            (idx["date"] >= start_date)
            & (idx["date"] <= end_date)
            & idx["prev_close"].notna()
            & (idx["open"].astype(float) < idx["prev_close"].astype(float)),
            "day_str",
        ]
    )
    print(f"{IDX_SYMBOL} 開低日: {len(idx_down_days)}", flush=True)

    stocks = day[day["stock_id"].isin(trade_universe)].copy()
    po = stocks["prev_open"].astype(float)
    pc = stocks["prev_close"].astype(float)
    o = stocks["open"].astype(float)

    prev_bear_body = (pc < po) & ((po - pc) / po.replace(0, np.nan) >= PREV_BODY_MIN)
    gap_up = (pc > 0) & ((o / pc - 1.0) >= OPEN_GAP_MIN)

    cands = stocks[
        (stocks["date"] >= start_date)
        & (stocks["date"] <= end_date)
        & prev_bear_body
        & gap_up
        & stocks["day_str"].isin(idx_down_days)
    ].copy()
    print(f"條件1+2+{IDX_SYMBOL}開低 候選: {len(cands):,}", flush=True)
    if cands.empty:
        return cands

    need_sids = set(cands["stock_id"].unique())
    need_days = set(cands["day_str"].unique())
    print("載入 pattern m5_std...", flush=True)
    m5 = load_pattern_m5_std(start_date=start_date)
    m5["stock_id"] = m5["stock_id"].astype(str)
    m5["date"] = pd.to_datetime(m5["date"], format="mixed")
    m5["day_str"] = m5["date"].dt.strftime("%Y-%m-%d")
    m5 = m5[
        m5["stock_id"].isin(need_sids)
        & m5["day_str"].isin(need_days)
        & (m5["date"].dt.time >= FIRST_M5_TIME)
        & (m5["date"].dt.time <= dtime(10, 0))
    ].copy()
    m5 = m5.sort_values(["stock_id", "date"]).reset_index(drop=True)
    print(f"m5 bars（候補窗）: {len(m5):,}", flush=True)

    first = m5[m5["date"].dt.time == FIRST_M5_TIME].drop_duplicates(
        ["stock_id", "day_str"], keep="last"
    )
    first = first.rename(
        columns={
            "date": "entry_ts",
            "open": "m5_open",
            "close": "m5_close",
            "high": "m5_high",
            "low": "m5_low",
        }
    )[["stock_id", "day_str", "entry_ts", "m5_open", "m5_close", "m5_high", "m5_low"]]

    ev = cands.merge(first, on=["stock_id", "day_str"], how="inner")
    ev = ev[ev["m5_close"].astype(float) < ev["m5_open"].astype(float)].copy()
    print(f"首 m5 陰線觸發: {len(ev):,}", flush=True)
    if ev.empty:
        return ev

    m5_by = {k: g.reset_index(drop=True) for k, g in m5.groupby(["stock_id", "day_str"], sort=False)}
    rows = []
    for _, r in ev.iterrows():
        key = (r["stock_id"], r["day_str"])
        day_m5 = m5_by.get(key)
        if day_m5 is None:
            continue
        entry = float(r["m5_close"])
        entry_ts = pd.Timestamp(r["entry_ts"])
        detail = _short_tb_m5(day_m5, entry_ts, entry)
        if detail is None:
            continue
        rows.append(
            {
                "stock_id": r["stock_id"],
                "day_str": r["day_str"],
                "prev_close": float(r["prev_close"]),
                "open": float(r["open"]),
                "entry_ts": entry_ts,
                "entry": entry,
                "label": detail["label"],
                "exit_reason": detail["exit_reason"],
                "exit_price": detail["exit_price"],
                "bars_held": detail["bars_held"],
                "pnl_pct": (entry - detail["exit_price"]) / entry,
            }
        )

    out = pd.DataFrame(rows)
    print(f"可標籤事件: {len(out):,}", flush=True)
    if out.empty:
        print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
        return out

    print("\n" + "=" * 56)
    print(f"做空 TB 三分類（±{TP_PCT:.0%} / 震盪30分）— 無 atr5")
    print("=" * 56)
    _summarize(out)

    if len(out):
        out = out.copy()
        out["year"] = pd.to_datetime(out["day_str"]).dt.year
        print("\n分年:", flush=True)
        for y, g in out.groupby("year"):
            n = len(g)
            print(
                f"  {y}: n={n}  TP={100 * (g['label'] == 1).mean():.1f}%  "
                f"flat={100 * (g['label'] == 0).mean():.1f}%  "
                f"SL={100 * (g['label'] == -1).mean():.1f}%  "
                f"mean={100 * g['pnl_pct'].mean():.3f}%",
                flush=True,
            )

    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description="prev_bear_m5_short 鎖定設計（無 atr5）")
    p.add_argument("--start_date", default="2024-01-01")
    p.add_argument("--end_date", default="2026-07-31")
    args = p.parse_args()
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
