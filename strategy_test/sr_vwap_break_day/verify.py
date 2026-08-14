"""
當日 m1：突破支撐＋VWAP（多）、跌破壓力＋VWAP（空）的當日勢。

用法：
    python -m strategy_test.sr_vwap_break_day.verify \\
        --start_date 2026-07-01 --end_date 2026-08-14
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

from data.adjustment_query import load_pattern_day, load_pattern_m1
from finmind.tick_universe import load_tick_universe
from pattern.envelope import envelope_sr_prices

FRICTION = 0.0045  # 0.45%，跟其他 strategy_test 一致
SESSION_START = dtime(9, 0)
SESSION_END = dtime(13, 30)
TIME_BUCKETS = (
    ("0900_1000", dtime(9, 0), dtime(10, 0)),
    ("1000_1200", dtime(10, 0), dtime(12, 0)),
    ("1200_1330", dtime(12, 0), dtime(13, 30)),
)


def _first_cross_idx(closes: np.ndarray, levels: np.ndarray, direction: str) -> int | None:
    """收盤變號：above = close >= level。回傳穿越那根在 closes 裡的 index。"""
    if len(closes) < 2:
        return None
    prev_above = closes[:-1] >= levels[:-1]
    cur_above = closes[1:] >= levels[1:]
    valid = (
        np.isfinite(closes[:-1])
        & np.isfinite(closes[1:])
        & np.isfinite(levels[:-1])
        & np.isfinite(levels[1:])
    )
    if direction == "up":
        hits = valid & (~prev_above) & cur_above
    else:
        hits = valid & prev_above & (~cur_above)
    idx = np.flatnonzero(hits)
    if len(idx) == 0:
        return None
    return int(idx[0] + 1)


def _time_bucket(ts: pd.Timestamp) -> str:
    t = ts.time()
    if t < dtime(10, 0):
        return "0900_1000"
    if t < dtime(12, 0):
        return "1000_1200"
    return "1200_1330"


def _session_slice(g: pd.DataFrame) -> pd.DataFrame:
    tt = g["date"].dt.time
    return g.loc[(tt >= SESSION_START) & (tt <= SESSION_END)]


def _build_sr_map(
    day: pd.DataFrame, event_dates: list[pd.Timestamp], stocks: set[str]
) -> dict[tuple[str, str], tuple[float, float]]:
    """(stock_id, YYYY-MM-DD) → (resistance, support)；日K 只用該日之前。"""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    day = day[day["stock_id"].isin(stocks)].copy()
    day["stock_id"] = day["stock_id"].astype(str)
    n_stocks = day["stock_id"].nunique()
    done = 0
    for sid, g in day.groupby("stock_id", sort=False):
        g = g.sort_values("date")
        for d in event_dates:
            hist = g.loc[g["date"] < d]
            res, sup = envelope_sr_prices(hist)
            if res is None:
                continue
            out[(str(sid), pd.Timestamp(d).strftime("%Y-%m-%d"))] = (res, sup)
        done += 1
        if done % 50 == 0:
            print(f"  包絡 {done}/{n_stocks} 檔…", flush=True)
    return out


def _events_for_day(
    g: pd.DataFrame, res: float, sup: float
) -> list[dict]:
    g = _session_slice(g).sort_values("date")
    if len(g) < 3:
        return []
    closes = g["close"].to_numpy(dtype=float)
    vols = g["volume"].to_numpy(dtype=float)
    times = g["date"].to_numpy()
    cum_vol = np.cumsum(vols)
    cum_pv = np.cumsum(closes * vols)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol > 0)
    res_arr = np.full_like(closes, res)
    sup_arr = np.full_like(closes, sup)

    # 上：收盤由下往上穿越支撐（突破支撐）且穿越 VWAP
    # 下：收盤由上往下跌破壓力（跌破壓力）且跌破 VWAP
    i_sup_up = _first_cross_idx(closes, sup_arr, "up")
    i_vwap_up = _first_cross_idx(closes, vwap, "up")
    i_res_dn = _first_cross_idx(closes, res_arr, "down")
    i_vwap_dn = _first_cross_idx(closes, vwap, "down")

    day_close = float(closes[-1])
    day_vwap = float(vwap[-1]) if np.isfinite(vwap[-1]) else np.nan
    rows = []

    if i_sup_up is not None and i_vwap_up is not None:
        i = max(i_sup_up, i_vwap_up)
        px = float(closes[i])
        if px > 0 and np.isfinite(day_vwap):
            r = day_close / px - 1.0
            ts = pd.Timestamp(times[i])
            rows.append(
                {
                    "direction": "up",
                    "signal_time": ts,
                    "px": px,
                    "day_close": day_close,
                    "ret": r,
                    "hold": bool(day_close >= sup and day_close >= day_vwap),
                    "bucket": _time_bucket(ts),
                    "res": res,
                    "sup": sup,
                }
            )

    if i_res_dn is not None and i_vwap_dn is not None:
        i = max(i_res_dn, i_vwap_dn)
        px = float(closes[i])
        if px > 0 and np.isfinite(day_vwap):
            r = -(day_close / px - 1.0)  # 空方：進場後下跌為正
            ts = pd.Timestamp(times[i])
            rows.append(
                {
                    "direction": "down",
                    "signal_time": ts,
                    "px": px,
                    "day_close": day_close,
                    "ret": r,
                    "hold": bool(day_close < res and day_close < day_vwap),
                    "bucket": _time_bucket(ts),
                    "res": res,
                    "sup": sup,
                }
            )
    return rows


def _print_block(label: str, sub: pd.DataFrame) -> None:
    if sub is None or sub.empty:
        print(f"  {label}: n=0", flush=True)
        return
    r = sub["ret"].astype(float)
    n = len(r)
    mean = float(r.mean())
    med = float(r.median())
    win = float((r > 0).mean())
    hold = float(sub["hold"].mean())
    mean_net = mean - FRICTION
    print(
        f"  {label}: n={n:,}  "
        f"mean={100 * mean:+.3f}%  med={100 * med:+.3f}%  "
        f"win={100 * win:.1f}%  hold={100 * hold:.1f}%  "
        f"mean_net={100 * mean_net:+.3f}%",
        flush=True,
    )


def run(start_date: str, end_date: str) -> pd.DataFrame:
    t0 = time.time()
    stocks = set(str(s) for s in load_tick_universe())
    print(f"母體 tick_universe {len(stocks)} 檔  {start_date} ~ {end_date}", flush=True)

    hist_start = (pd.Timestamp(start_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    print(f"載入日K（start={hist_start})…", flush=True)
    day = load_pattern_day(start_date=hist_start, end_date=end_date)
    day["stock_id"] = day["stock_id"].astype(str)
    print(f"  day {len(day):,} rows", flush=True)

    print("載入 M1…", flush=True)
    m1 = load_pattern_m1(start_date=start_date, end_date=end_date)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(stocks)]
    m1 = m1[(m1["date"] >= start_date) & (m1["date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))]
    print(f"  m1 {len(m1):,} rows", flush=True)
    if m1.empty or day.empty:
        print("無資料", flush=True)
        return pd.DataFrame()

    m1 = m1.copy()
    m1["day"] = m1["date"].dt.normalize()
    event_dates = sorted(m1["day"].unique())
    print(f"交易日 {len(event_dates)} 天，算包絡壓力/支撐…", flush=True)
    sr_map = _build_sr_map(day, list(event_dates), stocks)
    print(f"  有包絡的 stock-day {len(sr_map):,}", flush=True)

    recs = []
    n_groups = 0
    for (sid, d), g in m1.groupby(["stock_id", "day"], sort=False):
        n_groups += 1
        key = (str(sid), pd.Timestamp(d).strftime("%Y-%m-%d"))
        sr = sr_map.get(key)
        if sr is None:
            continue
        res, sup = sr
        for ev in _events_for_day(g, res, sup):
            recs.append({"stock_id": str(sid), "date": key[1], **ev})
        if n_groups % 2000 == 0:
            print(f"  掃過 {n_groups:,} 個 stock-day，事件 {len(recs):,}…", flush=True)

    ev = pd.DataFrame(recs)
    print(f"\n事件 {len(ev):,} 筆（stock-day-方向）  耗時 {time.time() - t0:.1f}s", flush=True)
    if ev.empty:
        return ev

    print("\n" + "=" * 60, flush=True)
    print("當日勢：突破支撐＋VWAP上 / 跌破壓力＋VWAP下 → 收到收盤", flush=True)
    print("=" * 60, flush=True)
    print("[全部]", flush=True)
    _print_block("up  (突破支撐+VWAP上)", ev[ev["direction"] == "up"])
    _print_block("down(跌破壓力+VWAP下)", ev[ev["direction"] == "down"])
    _print_block("both", ev)

    for name, a, b in TIME_BUCKETS:
        print(f"\n[時段 {name}]", flush=True)
        sub = ev[ev["bucket"] == name]
        _print_block("up", sub[sub["direction"] == "up"])
        _print_block("down", sub[sub["direction"] == "down"])
        _print_block("both", sub)

    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return ev


def main() -> None:
    p = argparse.ArgumentParser(description="m1 突破支撐＋VWAP / 跌破壓力＋VWAP 的當日勢")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-08-14")
    args = p.parse_args()
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
