"""
小樣驗證：ORB 突破 ∩ 站上 VWAP（雙重突破）vs 僅 ORB / 仍在 VWAP 下。

定義：
- ORB：09:10~09:30，收盤自下方重新站上當日 or_high（同 orb/features.py）
- VWAP：當日累積近似 VWAP = cumsum(close*volume)/cumsum(volume)
- 站上 VWAP：close > vwap
- 標籤：ORB 做多 TB ±3%／最多 30 根（同 orb；1=勝 0=負）

用法：
    python -m strategy.orb.experiments.verify_orb_vwap_breakout
    python -m strategy.orb.experiments.verify_orb_vwap_breakout --start_date 2026-06-01 --end_date 2026-07-31
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

from data.raw_query import iter_m1_months
from strategy.orb.config import (
    BREAKOUT_SEARCH_MINUTES,
    HOLD_BARS,
    OPENING_RANGE_MINUTES,
    SL_PCT,
    TP_PCT,
)


def _is_stock_id(sid: str) -> bool:
    s = str(sid)
    return len(s) == 4 and s.isdigit() and not s.startswith("00")


def _barrier_label_group(closes: np.ndarray) -> np.ndarray:
    n = len(closes)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        entry = closes[i]
        tp_price = entry * (1 + TP_PCT)
        sl_price = entry * (1 - SL_PCT)
        future = closes[i + 1 : i + HOLD_BARS + 1]
        tp_idx = np.argmax(future >= tp_price) if (future >= tp_price).any() else len(future)
        sl_idx = np.argmax(future <= sl_price) if (future <= sl_price).any() else len(future)
        if tp_idx < sl_idx:
            labels[i] = 1
        elif sl_idx < tp_idx:
            labels[i] = 0
        elif len(future) == HOLD_BARS:
            labels[i] = 1 if future[-1] > entry else 0
    return labels


def _process_month(m1: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    m1 = m1.copy()
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1 = m1[m1["stock_id"].map(_is_stock_id)]
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[(m1["day_str"] >= start_date) & (m1["day_str"] <= end_date)]
    if m1.empty:
        return m1

    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date
    m1["minutes_since_open"] = (m1["date"].dt.hour - 9) * 60 + m1["date"].dt.minute

    g = m1.groupby(["stock_id", "day_date"], sort=False)
    in_or = m1["minutes_since_open"] < OPENING_RANGE_MINUTES
    m1["or_high"] = m1["high"].where(in_or).groupby([m1["stock_id"], m1["day_date"]]).transform("max")

    search = (m1["minutes_since_open"] >= OPENING_RANGE_MINUTES) & (
        m1["minutes_since_open"] < BREAKOUT_SEARCH_MINUTES
    )
    prev_close = g["close"].shift(1)
    is_breakout = search & (m1["close"] > m1["or_high"]) & (prev_close <= m1["or_high"])

    # 當日累積 VWAP（與專案近似一致：close*volume）
    m1["_pv"] = m1["close"].astype(float) * m1["volume"].astype(float)
    cum_pv = g["_pv"].cumsum()
    cum_vol = g["volume"].cumsum().replace(0, np.nan)
    m1["vwap"] = cum_pv / cum_vol
    m1["above_vwap"] = m1["close"].astype(float) > m1["vwap"].astype(float)

    # 標籤需全日路徑；先算再篩突破列
    m1["target"] = g["close"].transform(lambda s: pd.Series(_barrier_label_group(s.values), index=s.index))
    ev = m1[is_breakout & m1["target"].notna()].copy()
    if ev.empty:
        return ev
    ev["target"] = ev["target"].astype(int)
    ev["entry"] = ev["close"].astype(float)
    # 簡易 forward ret：用標籤無法還原實際出場價；另報 30 根後報酬近似
    return ev[
        ["stock_id", "date", "day_str", "entry", "vwap", "above_vwap", "target", "or_high"]
    ]


def _summarize(label: str, df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print(f"  {label}: n=0", flush=True)
        return
    win = df["target"].mean() * 100
    print(f"  {label}: n={n:,}  TB勝率={win:.1f}%  ({int(df['target'].sum())}/{n})", flush=True)


def run(start_date: str, end_date: str) -> pd.DataFrame:
    t0 = time.time()
    print("ORB 突破 ∩ 站上 VWAP（雙重突破）", flush=True)
    print(
        f"ORB: {OPENING_RANGE_MINUTES}分區間後~{BREAKOUT_SEARCH_MINUTES}分內站上 or_high",
        flush=True,
    )
    print(f"VWAP: 當日 cum(close*vol)/cum(vol)；站上 = close > vwap", flush=True)
    print(f"標籤: 做多 TB ±{TP_PCT:.0%}/{HOLD_BARS}根", flush=True)
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    parts = []
    for i, month in enumerate(iter_m1_months(start_date=start_date), start=1):
        ev = _process_month(month, start_date, end_date)
        if not ev.empty:
            parts.append(ev)
            print(f"  月檔 {i}: 事件 +{len(ev):,}", flush=True)
    if not parts:
        print("無事件", flush=True)
        return pd.DataFrame()

    ev = pd.concat(parts, ignore_index=True)
    print(f"\n總突破事件: {len(ev):,}", flush=True)
    print(f"其中站上 VWAP: {ev['above_vwap'].mean() * 100:.1f}%", flush=True)

    above = ev[ev["above_vwap"]]
    below = ev[~ev["above_vwap"]]

    print("\n" + "=" * 56)
    print("對照（做多 TB 勝率）")
    print("=" * 56)
    _summarize("A  ORB only", ev)
    _summarize("B  ORB ∩ 站上 VWAP（雙重突破）", above)
    _summarize("C  ORB ∩ 仍在 VWAP 下（假突破候選）", below)

    if len(above) and len(below):
        print(
            f"\nB−A 勝率差: {above['target'].mean() * 100 - ev['target'].mean() * 100:+.1f} pt  "
            f"B−C: {above['target'].mean() * 100 - below['target'].mean() * 100:+.1f} pt",
            flush=True,
        )
    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return ev


def main(
    start_date: str = "2026-06-01",
    end_date: str = "2026-07-31",
):
    if len(sys.argv) > 1:
        p = argparse.ArgumentParser()
        p.add_argument("--start_date", default=start_date)
        p.add_argument("--end_date", default=end_date)
        args = p.parse_args()
        start_date, end_date = args.start_date, args.end_date
    run(start_date, end_date)


if __name__ == "__main__":
    main()
