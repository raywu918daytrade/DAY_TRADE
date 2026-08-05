"""
開高 → 補缺口回昨收 → m5 在昨收「跌+漲」做多 → TB ±3% 三分類比例。

定義：
1. 母體：db/tickers/tick_universe.parquet（約 400 支）；今日 open > prev_close
2. 補缺口 / 反轉：當日第一組連續 m5 陰線+陽線，且兩根都觸及昨收
   （low ≤ prev_close ≤ high；陰=c<o，陽=c>o）
3. 進場濾網：陽線 m5 收盤那一分鐘，用 m1 ATR5 跨股同分位 PR≥75 才進場
   （atr5 = TR5 / day_open，同 strategy.mkt.features.add_atr5；PR 只在 universe 內排）
4. 進場：陽線 m5 收盤做多；進場時間須 < 10:00
5. 標籤（做多 TB，掃後續 m5 至 13:25）：
   +3% 止盈 / -3% 止損 / 都未觸 → 13:25 收（震盪）
   同根高低都觸：先判止盈

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_gapfill_m5_reversal \\
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
from data.raw_query import iter_m1_months
from finmind.tick_universe import load_tick_universe
from strategy.mkt.config import IDX_SYMBOL
from strategy.mkt.features import add_atr5

TP_PCT = 0.03
SL_PCT = 0.03
ENTRY_DEADLINE = dtime(10, 0)  # 進場須 < 10:00（不含 10:00）
FORCE_EXIT_TIME = dtime(13, 25)
ATR5_MIN_PR = 0.75


def _touches(low: float, high: float, level: float) -> bool:
    return low <= level <= high


def _find_reversal(
    grp: pd.DataFrame, prev_close: float
) -> tuple[int, float, pd.Timestamp] | None:
    """回傳 (bull_idx_in_grp, entry, entry_ts) 或 None。"""
    o = grp["open"].astype(float).values
    c = grp["close"].astype(float).values
    h = grp["high"].astype(float).values
    lo = grp["low"].astype(float).values
    ts = grp["date"].values
    n = len(grp)
    for i in range(n - 1):
        t_bull = pd.Timestamp(ts[i + 1])
        if t_bull.time() >= ENTRY_DEADLINE:
            break
        if not (c[i] < o[i] and c[i + 1] > o[i + 1]):
            continue
        if not (
            _touches(lo[i], h[i], prev_close)
            and _touches(lo[i + 1], h[i + 1], prev_close)
        ):
            continue
        return i + 1, float(c[i + 1]), t_bull
    return None


def _long_tb_m5(
    grp: pd.DataFrame, entry_idx: int, entry: float
) -> dict | None:
    """做多 TB：+1 TP / -1 SL / 0 time。"""
    if entry <= 0:
        return None
    day = pd.Timestamp(grp["date"].iloc[0]).normalize()
    exit_ts = day + pd.Timedelta(hours=FORCE_EXIT_TIME.hour, minutes=FORCE_EXIT_TIME.minute)
    fut = grp.iloc[entry_idx + 1 :].copy()
    fut = fut[pd.to_datetime(fut["date"]) <= exit_ts]
    if fut.empty:
        return None
    tp = entry * (1.0 + TP_PCT)
    sl = entry * (1.0 - SL_PCT)
    for j, (_, row) in enumerate(fut.iterrows(), start=1):
        ts = pd.Timestamp(row["date"])
        hi, lo = float(row["high"]), float(row["low"])
        if hi >= tp:
            return {
                "label": 1.0,
                "exit_ts": ts,
                "exit_price": tp,
                "exit_reason": "tp",
                "bars_held": j,
            }
        if lo <= sl:
            return {
                "label": -1.0,
                "exit_ts": ts,
                "exit_price": sl,
                "exit_reason": "sl",
                "bars_held": j,
            }
    last = fut.iloc[-1]
    last_ts = pd.Timestamp(last["date"])
    if last_ts.time() < FORCE_EXIT_TIME:
        return None
    return {
        "label": 0.0,
        "exit_ts": last_ts,
        "exit_price": float(last["close"]),
        "exit_reason": "time",
        "bars_held": len(fut),
    }


def _attach_atr5_pr(
    ev: pd.DataFrame,
    start_date: str,
    need_days: set[str],
    universe: set[str],
) -> pd.DataFrame:
    """在進場分鐘掛上 m1 atr5 與跨股同分 PR（只在 tick_universe 內排）。"""
    if ev.empty:
        return ev
    entry_times = set(pd.to_datetime(ev["entry_ts"]))
    parts = []
    months = list(iter_m1_months(start_date=start_date))
    print(f"載入 m1 算 atr5（{len(months)} 月檔，universe={len(universe)}）...", flush=True)
    for i, month in enumerate(months, start=1):
        month = month.copy()
        month["stock_id"] = month["stock_id"].astype(str)
        month["date"] = pd.to_datetime(month["date"], format="mixed")
        month["day_str"] = month["date"].dt.strftime("%Y-%m-%d")
        month = month[month["stock_id"].isin(universe) & month["day_str"].isin(need_days)]
        if month.empty:
            continue
        month = month.sort_values(["stock_id", "date"]).reset_index(drop=True)
        month["day_date"] = month["date"].dt.date
        month = add_atr5(month)
        month["atr5_pr"] = month.groupby("date")["atr5"].rank(pct=True)
        sub = month[month["date"].isin(entry_times)][
            ["stock_id", "date", "atr5", "atr5_pr"]
        ].dropna(subset=["atr5", "atr5_pr"])
        if not sub.empty:
            parts.append(sub)
        print(f"  m1 atr5 {i}/{len(months)}  rows={len(sub):,}", flush=True)

    if not parts:
        ev = ev.copy()
        ev["atr5"] = np.nan
        ev["atr5_pr"] = np.nan
        return ev

    atr = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["stock_id", "date"], keep="last"
    )
    atr = atr.rename(columns={"date": "entry_ts"})
    out = ev.merge(atr, on=["stock_id", "entry_ts"], how="left")
    return out


def _summarize_3class(label: str, df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print(f"  {label}: n=0", flush=True)
        return
    n_tp = int((df["label"] == 1.0).sum())
    n_flat = int((df["label"] == 0.0).sum())
    n_sl = int((df["label"] == -1.0).sum())
    print(f"  {label}: n={n:,}", flush=True)
    print(f"    止盈(+3%): {n_tp:,}  {100 * n_tp / n:.1f}%", flush=True)
    print(f"    震盪(時間牆): {n_flat:,}  {100 * n_flat / n:.1f}%", flush=True)
    print(f"    止損(-3%): {n_sl:,}  {100 * n_sl / n:.1f}%", flush=True)
    print(
        f"    進場→出場 mean={100 * df['pnl_pct'].mean():.3f}%  "
        f"median={100 * df['pnl_pct'].median():.3f}%",
        flush=True,
    )


def run(start_date: str, end_date: str) -> pd.DataFrame:
    t0 = time.time()
    universe = {str(s) for s in load_tick_universe()}
    # 不做 0050 本身（大盤代理，ret 特徵語意上也不當標的）
    trade_universe = universe - {IDX_SYMBOL}

    print("開高補缺口 m5 反轉（做多）三分類", flush=True)
    print(
        f"母體: tick_universe {len(universe)} 支（交易排除 {IDX_SYMBOL} → {len(trade_universe)}）"
        f" 且 open > prev_close",
        flush=True,
    )
    print(
        f"進場濾網: m1 atr5 跨股同分 PR≥{ATR5_MIN_PR:.0%}（universe 內排名）",
        flush=True,
    )
    print(
        f"觸發: 第一組 m5 陰+陽 皆觸昨收；進場=陽線收；進場 < {ENTRY_DEADLINE.strftime('%H:%M')}",
        flush=True,
    )
    print(
        f"標籤: 做多 TB ±{TP_PCT:.0%} / 時間牆 {FORCE_EXIT_TIME.strftime('%H:%M')}",
        flush=True,
    )
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    print(f"載入 pattern day（start={hist})...", flush=True)
    day = load_pattern_day(start_date=hist)
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].isin(trade_universe)].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)
    g = day.groupby("stock_id", sort=False)
    day["prev_close"] = g["close"].shift(1)
    day["day_str"] = day["date"].dt.strftime("%Y-%m-%d")
    gap_up = day["open"].astype(float) > day["prev_close"].astype(float)
    cands = day[
        (day["date"] >= start_date)
        & (day["date"] <= end_date)
        & day["prev_close"].notna()
        & gap_up
    ].copy()
    print(f"開高候選: {len(cands):,}", flush=True)
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
        & (m5["date"].dt.time >= dtime(9, 0))
        & (m5["date"].dt.time <= FORCE_EXIT_TIME)
    ].copy()
    m5 = m5.sort_values(["stock_id", "date"]).reset_index(drop=True)
    print(f"m5 bars: {len(m5):,}", flush=True)

    pc_map = cands.set_index(["stock_id", "day_str"])["prev_close"].astype(float).to_dict()
    rows = []
    n_groups = 0
    for (sid, day_str), grp in m5.groupby(["stock_id", "day_str"], sort=False):
        n_groups += 1
        pc = pc_map.get((sid, day_str))
        if pc is None or not np.isfinite(pc):
            continue
        grp = grp.reset_index(drop=True)
        found = _find_reversal(grp, float(pc))
        if found is None:
            continue
        entry_idx, entry, entry_ts = found
        detail = _long_tb_m5(grp, entry_idx, entry)
        if detail is None:
            continue
        rows.append(
            {
                "stock_id": sid,
                "day_str": day_str,
                "prev_close": float(pc),
                "entry_ts": entry_ts,
                "entry": entry,
                "label": detail["label"],
                "exit_reason": detail["exit_reason"],
                "exit_price": detail["exit_price"],
                "bars_held": detail["bars_held"],
                "pnl_pct": (detail["exit_price"] - entry) / entry,
            }
        )
        if n_groups % 5000 == 0:
            print(f"  掃過股日 {n_groups:,}  事件 {len(rows):,}", flush=True)

    ev = pd.DataFrame(rows)
    print(f"\n有 m5 的開高股日≈{n_groups:,}  觸發事件={len(ev):,}", flush=True)
    if ev.empty:
        print("無事件", flush=True)
        print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
        return ev

    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev = _attach_atr5_pr(
        ev, start_date=start_date, need_days=need_days, universe=universe
    )
    n_miss = int(ev["atr5_pr"].isna().sum())
    ev_ok = ev[ev["atr5_pr"].notna()].copy()
    ev_f = ev_ok[ev_ok["atr5_pr"] >= ATR5_MIN_PR].copy()
    print(
        f"atr5 對上={len(ev_ok):,}  缺={n_miss:,}  "
        f"PR≥{ATR5_MIN_PR:.0%}={len(ev_f):,}  "
        f"覆蓋={100 * len(ev_f) / max(len(ev), 1):.1f}%",
        flush=True,
    )

    print("\n" + "=" * 56)
    print("做多 TB 三分類（+3% / 震盪 / -3%）")
    print("=" * 56)
    _summarize_3class("A  僅 m5 反轉（無 atr5）", ev)
    _summarize_3class(f"B  + m1 atr5 PR≥{ATR5_MIN_PR:.0%}", ev_f)

    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return ev_f


def main():
    p = argparse.ArgumentParser(description="開高補缺口 m5 反轉做多三分類")
    p.add_argument("--start_date", default="2026-06-01")
    p.add_argument("--end_date", default="2026-07-31")
    args = p.parse_args()
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
