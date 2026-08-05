"""
兩層濾網驗證：D1 + M1 皆 VWAP expanding z |z|≥2。

1. D1（昨收可知，無前瞻）
   - day_vwap = 當日 M1 EOD 累積 VWAP
   - day_dev = close_eod - day_vwap
   - day_z = day_dev / expanding_std(day_dev)（per stock，min_periods=20）
   - 昨 |day_z|≥2 → shift 到今日：z>0 只准 M1 做空；z<0 只准 M1 做多
2. M1（09:00～09:30）
   - m1_vwap_z 同 vwap_ml（expanding std，min_periods=5）
   - |m1_vwap_z|≥2：上偏做空、下偏做多
3. 進場 = D1 通過且與 M1 回歸同向（兩層 AND）
4. 標籤：±3% / 30 根 TB

對照：
  A) 僅 M1
  B) D1 + M1 回歸（目標）
  C) D1 + M1 同向延續

用法：
    python -m strategy.vwap_ml.experiments.verify_d1_m1_vwap_z
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

from data.adjustment_query import load_pattern_day
from data.raw_query import iter_m1_months

D1_Z_MULT = 2.0
D1_Z_MIN_PERIODS = 20
M1_Z_MULT = 2.0
M1_Z_MIN_PERIODS = 5
SESSION_START_MIN = 0  # 09:00
SESSION_END_MIN = 30  # 09:30
HOLD_BARS = 30
TP_PCT = 0.03
SL_PCT = 0.03


def _is_stock_id(sid: str) -> bool:
    s = str(sid)
    return len(s) == 4 and s.isdigit() and not s.startswith("00")


def _barrier_long(closes: np.ndarray, i: int) -> float:
    entry = closes[i]
    fut = closes[i + 1 : i + HOLD_BARS + 1]
    if len(fut) == 0:
        return np.nan
    tp, sl = entry * (1 + TP_PCT), entry * (1 - SL_PCT)
    tp_i = np.argmax(fut >= tp) if (fut >= tp).any() else len(fut)
    sl_i = np.argmax(fut <= sl) if (fut <= sl).any() else len(fut)
    if tp_i < sl_i:
        return 1.0
    if sl_i < tp_i:
        return 0.0
    if len(fut) < HOLD_BARS:
        return np.nan
    return 1.0 if fut[-1] > entry else 0.0


def _barrier_short(closes: np.ndarray, i: int) -> float:
    entry = closes[i]
    fut = closes[i + 1 : i + HOLD_BARS + 1]
    if len(fut) == 0:
        return np.nan
    tp, sl = entry * (1 - TP_PCT), entry * (1 + SL_PCT)
    tp_i = np.argmax(fut <= tp) if (fut <= tp).any() else len(fut)
    sl_i = np.argmax(fut >= sl) if (fut >= sl).any() else len(fut)
    if tp_i < sl_i:
        return 1.0
    if sl_i < tp_i:
        return 0.0
    if len(fut) < HOLD_BARS:
        return np.nan
    return 1.0 if fut[-1] < entry else 0.0


def _eod_from_month(m1: pd.DataFrame) -> pd.DataFrame:
    m1 = m1.copy()
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1 = m1[m1["stock_id"].map(_is_stock_id)]
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1.sort_values(["stock_id", "date"])
    g = m1.groupby(["stock_id", "day_str"], sort=False)
    m1["_pv"] = m1["close"].astype(float) * m1["volume"].astype(float)
    m1["vwap"] = g["_pv"].cumsum() / g["volume"].cumsum().replace(0, np.nan)
    eod = m1.groupby(["stock_id", "day_str"], sort=False).tail(1)[
        ["stock_id", "day_str", "close", "vwap"]
    ]
    return eod.rename(columns={"close": "close_eod", "vwap": "day_vwap"})


def _build_d1_state(eod: pd.DataFrame) -> pd.DataFrame:
    """EOD VWAP z（expanding）→ shift(1) 到交易日。"""
    d = eod.sort_values(["stock_id", "day_str"]).copy()
    d["day_dev"] = d["close_eod"].astype(float) - d["day_vwap"].astype(float)
    g = d.groupby("stock_id", sort=False)
    d["day_z"] = g["day_dev"].transform(
        lambda s: s / s.expanding(min_periods=D1_Z_MIN_PERIODS).std().replace(0, np.nan)
    )
    # 當日 EOD 標記；給下一交易日用
    d["day_side_eod"] = np.where(
        d["day_z"] >= D1_Z_MULT,
        1,
        np.where(d["day_z"] <= -D1_Z_MULT, -1, 0),
    )
    d["day_side"] = g["day_side_eod"].shift(1)
    d["prev_day_z"] = g["day_z"].shift(1)
    out = d[["stock_id", "day_str", "day_side", "prev_day_z"]].dropna(subset=["day_side"])
    out["day_side"] = out["day_side"].astype(int)
    return out


def _process_trade_month(
    m1: pd.DataFrame,
    d1_state: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    m1 = m1.copy()
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1 = m1[m1["stock_id"].map(_is_stock_id)]
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[(m1["day_str"] >= start_date) & (m1["day_str"] <= end_date)]
    if m1.empty:
        return m1

    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["mins"] = (m1["date"].dt.hour - 9) * 60 + m1["date"].dt.minute
    g = m1.groupby(["stock_id", "day_str"], sort=False)
    m1["_pv"] = m1["close"].astype(float) * m1["volume"].astype(float)
    m1["m1_vwap"] = g["_pv"].cumsum() / g["volume"].cumsum().replace(0, np.nan)
    m1["_dev"] = m1["close"].astype(float) - m1["m1_vwap"]
    m1["m1_vwap_z"] = g["_dev"].transform(
        lambda s: s / s.expanding(min_periods=M1_Z_MIN_PERIODS).std().replace(0, np.nan)
    )
    m1 = m1.merge(d1_state, on=["stock_id", "day_str"], how="left")

    rows = []
    for (sid, day), grp in m1.groupby(["stock_id", "day_str"], sort=False):
        grp = grp.reset_index(drop=True)
        closes = grp["close"].astype(float).values
        day_side = grp["day_side"].iloc[0]
        if pd.isna(day_side):
            day_side = 0
        day_side = int(day_side)

        hit = (grp["mins"] >= SESSION_START_MIN) & (grp["mins"] < SESSION_END_MIN)
        hit &= grp["m1_vwap_z"].abs() >= M1_Z_MULT
        for i in np.where(hit.values)[0]:
            z = float(grp.iloc[i]["m1_vwap_z"])
            if not np.isfinite(z):
                continue
            trade_dir = "short" if z >= M1_Z_MULT else "long"
            win = _barrier_short(closes, i) if trade_dir == "short" else _barrier_long(closes, i)
            if not np.isfinite(win):
                continue
            day_ok = (day_side > 0 and trade_dir == "short") or (day_side < 0 and trade_dir == "long")
            day_cont = (day_side > 0 and trade_dir == "long") or (day_side < 0 and trade_dir == "short")
            rows.append(
                {
                    "stock_id": sid,
                    "day_str": day,
                    "date": grp.iloc[i]["date"],
                    "trade_dir": trade_dir,
                    "m1_vwap_z": z,
                    "day_side": day_side,
                    "day_ok_revert": bool(day_ok),
                    "day_cont": bool(day_cont),
                    "win": int(win),
                }
            )
    return pd.DataFrame(rows)


def _summarize(label: str, df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print(f"  {label}: n=0", flush=True)
        return
    print(
        f"  {label}: n={n:,}  勝率={100 * df['win'].mean():.1f}%  "
        f"多={int((df['trade_dir'] == 'long').sum())} 空={int((df['trade_dir'] == 'short').sum())}",
        flush=True,
    )


def run(start_date: str, end_date: str) -> pd.DataFrame:
    t0 = time.time()
    print("兩層濾網：D1 + M1 皆 VWAP z≥2（回歸）", flush=True)
    print(
        f"D1: 昨 |day_z|≥{D1_Z_MULT} (EOD VWAP expanding std, min_periods={D1_Z_MIN_PERIODS})",
        flush=True,
    )
    print(
        f"M1: |m1_vwap_z|≥{M1_Z_MULT} @ 09:00-09:30 (expanding std, min_periods={M1_Z_MIN_PERIODS})",
        flush=True,
    )
    print(f"標籤: ±{TP_PCT:.0%}/{HOLD_BARS}根", flush=True)
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    # 多載一段歷史給 D1 expanding z 暖機
    hist_start = (pd.Timestamp(start_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    print("逐月 M1：建 EOD VWAP...", flush=True)
    eod_parts = []
    months = list(iter_m1_months(start_date=hist_start))
    for i, month in enumerate(months, start=1):
        eod_parts.append(_eod_from_month(month))
        if i % 3 == 0 or i == len(months):
            print(f"  EOD {i}/{len(months)}", flush=True)
    eod = pd.concat(eod_parts, ignore_index=True).drop_duplicates(["stock_id", "day_str"], keep="last")

    # 日收用 pattern day 對齊（與 EOD vwap 同日）
    day = load_pattern_day(start_date=hist_start)
    day["stock_id"] = day["stock_id"].astype(str)
    day["day_str"] = pd.to_datetime(day["date"], format="mixed").dt.strftime("%Y-%m-%d")
    day = day[day["stock_id"].map(_is_stock_id)][["stock_id", "day_str", "close"]].rename(
        columns={"close": "close_eod"}
    )
    eod = eod.drop(columns=["close_eod"], errors="ignore").merge(day, on=["stock_id", "day_str"], how="inner")

    d1_state = _build_d1_state(eod)
    n_up = int((d1_state["day_side"] == 1).sum())
    n_dn = int((d1_state["day_side"] == -1).sum())
    print(f"  D1 極端（已 shift）: 上 z≥2 →{n_up:,}  下 z≤-2 →{n_dn:,}", flush=True)

    print("逐月 M1：觸發事件...", flush=True)
    trade_parts = []
    for i, month in enumerate(months, start=1):
        month = month.copy()
        month["date"] = pd.to_datetime(month["date"], format="mixed")
        if month["date"].max() < pd.Timestamp(start_date):
            continue
        if month["date"].min() > pd.Timestamp(end_date) + pd.Timedelta(days=1):
            continue
        ev = _process_trade_month(month, d1_state, start_date, end_date)
        if not ev.empty:
            trade_parts.append(ev)
            print(f"  事件月檔 {i}: +{len(ev):,}", flush=True)

    if not trade_parts:
        print("無事件", flush=True)
        return pd.DataFrame()

    ev = pd.concat(trade_parts, ignore_index=True)
    print(f"\nM1 觸發總數: {len(ev):,}", flush=True)

    a = ev
    b = ev[ev["day_ok_revert"]]
    c = ev[ev["day_cont"]]

    print("\n" + "=" * 56)
    print("對照（TB 勝率）")
    print("=" * 56)
    _summarize("A  僅 M1 z≥2", a)
    _summarize("B  D1 z≥2 + M1 回歸（兩層）", b)
    _summarize("C  D1 z≥2 + M1 延續（對照）", c)

    if len(b) and len(a):
        print(
            f"\nB−A 勝率差: {100 * (b['win'].mean() - a['win'].mean()):+.1f} pt  "
            f"覆蓋率={100 * len(b) / len(a):.1f}%",
            flush=True,
        )
    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start_date", default="2026-06-01")
    p.add_argument("--end_date", default="2026-07-31")
    args = p.parse_args()
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
