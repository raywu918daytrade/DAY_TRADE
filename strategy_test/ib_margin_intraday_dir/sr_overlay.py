"""把 D-1 法人／融資券接到「VWAP+壓力/支撐穿越」（規則同 live／dashboard）。

穿越用 pattern.vwap_sr_scan.events_for_group（1 分K 收盤變號，同一檔可重複）。
label 從訊號那根收盤走到 10:00（若訊號≤10:00）與走到當日收。

用法：
    python -m strategy_test.ib_margin_intraday_dir.sr_overlay \\
        --start_date 2025-01-01 --end_date 2026-08-14
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
from pattern.horizontal_sr import horizontal_sr_prices
from pattern.vwap_sr_scan import events_for_group
from strategy_test.ib_margin_intraday_dir.verify import (
    _print_mix,
    build_overnight_panel,
)

_T1000 = dtime(10, 0)
_SESSION_START = dtime(9, 0)
_SESSION_END = dtime(13, 30)
_PATH_COLS = (("→10:00", "ret_to_1000"), ("→收", "ret_to_close"))
_ALIGN_COLS = (("順向→收", "al_close"),)


def _build_sr_map(
    day: pd.DataFrame, event_dates: list[pd.Timestamp], stocks: set[str]
) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    out: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    day = day[day["stock_id"].isin(stocks)].copy()
    day["stock_id"] = day["stock_id"].astype(str)
    n_stocks = day["stock_id"].nunique()
    done = 0
    for sid, g in day.groupby("stock_id", sort=False):
        g = g.sort_values("date")
        for d in event_dates:
            hist = g.loc[g["date"] < d]
            res, sup = horizontal_sr_prices(hist)
            if res is None and sup is None:
                continue
            out[(str(sid), pd.Timestamp(d).strftime("%Y-%m-%d"))] = (res, sup)
        done += 1
        if done % 50 == 0:
            print(f"  橫向水位 {done}/{n_stocks} 檔…", flush=True)
    return out


def _session(g: pd.DataFrame) -> pd.DataFrame:
    tt = pd.to_datetime(g["date"]).dt.time
    return g.loc[(tt >= _SESSION_START) & (tt <= _SESSION_END)].sort_values("date")


def _scan_chunk(m1: pd.DataFrame, sr_map: dict) -> list[dict]:
    recs: list[dict] = []
    if m1 is None or m1.empty:
        return recs
    m1 = m1.copy()
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1["day"] = pd.to_datetime(m1["date"]).dt.strftime("%Y-%m-%d")
    for (sid, day), g in m1.groupby(["stock_id", "day"], sort=False):
        sr = sr_map.get((str(sid), str(day)))
        if sr is None:
            continue
        g = _session(g)
        if len(g) < 2:
            continue
        _, se = events_for_group(str(sid), str(sid), g, sr)
        if not se:
            continue
        times = pd.to_datetime(g["date"])
        closes = g["close"].astype(float).to_numpy()
        dclose = float(closes[-1])
        c1000 = np.nan
        hit_1000 = times.dt.time == _T1000
        if hit_1000.any():
            c1000 = float(g.loc[hit_1000, "close"].iloc[-1])
        for e in se:
            recs.append(
                {
                    "stock_id": str(sid),
                    "day": str(day),
                    "time": e["time"],
                    "sr_kind": e["sr_kind"],
                    "vwap_dir": e["vwap_dir"],
                    "px": float(e["price"]),
                    "dclose": dclose,
                    "c1000": c1000,
                }
            )
    return recs


def _month_starts(start_date: str, end_date: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    cur = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while cur <= end:
        nxt = (cur + pd.offsets.MonthEnd(0)).normalize()
        chunk_end = min(nxt, end)
        out.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + pd.Timedelta(days=1)
    return out


def _attach_paths(hit: pd.DataFrame) -> pd.DataFrame:
    hit = hit.copy()
    tod = hit["time"].map(lambda s: dtime(int(s[:2]), int(s[3:5])))
    hit["ret_to_close"] = hit["dclose"] / hit["px"].replace(0, np.nan) - 1.0
    hit["ret_to_1000"] = np.where(
        tod <= _T1000,
        hit["c1000"] / hit["px"].replace(0, np.nan) - 1.0,
        np.nan,
    )
    sign = np.where(hit["vwap_dir"] == "up", 1.0, -1.0)
    hit["al_close"] = sign * hit["ret_to_close"]
    return hit


def _overlay_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    short_up = df["short_dlt_pct"] > 0
    fa = df["feat_atr"]
    px_up = df["ret_1d"] >= 0.5 * fa
    px_dn = df["ret_1d"] <= -0.5 * fa
    return {
        "融券增 ∩ 真漲": short_up & px_up,
        "融券增 ∩ 真跌": short_up & px_dn,
        "僅真漲": px_up,
        "僅真跌": px_dn,
        "今ATR≥3%": df["day_atr"] >= 0.03,
        "融券增∩真漲 ∩ 今ATR≥3%": short_up & px_up & (df["day_atr"] >= 0.03),
        "僅真漲 ∩ 今ATR≥3%": px_up & (df["day_atr"] >= 0.03),
        "外資 PR≥80": df["foreign_pr"] >= 0.80,
        "投信 PR≥80": df["trust_pr"] >= 0.80,
    }


def run(
    start_date: str,
    end_date: str,
    lookback: int = 20,
    min_hist: int = 10,
    min_n: int = 100,
    tick_only: bool = False,
) -> pd.DataFrame:
    t0 = time.time()
    print("VWAP+壓力支撐穿越 × D-1 ib/margin", flush=True)
    print(f"窗 {start_date}～{end_date}  min_n={min_n}", flush=True)
    print(
        "事件=events_for_group（收盤變號，同 dashboard）；"
        "label=訊號收→10:00／收。母體再 ∩ ib∩margin。",
        flush=True,
    )
    print(flush=True)

    panel = build_overnight_panel(
        start_date, end_date, lookback=lookback, min_hist=min_hist, tick_only=tick_only
    )
    if panel.empty:
        return panel
    stocks = set(panel["stock_id"].astype(str))
    event_dates = sorted(pd.to_datetime(panel["day"].unique()))
    print(f"  隔夜面板 {len(panel):,}  stock-days；掃描 {len(event_dates)} 日", flush=True)

    hist_sr = (pd.Timestamp(start_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    print("載入 pattern 日K、算橫向壓力/支撐…", flush=True)
    day = load_pattern_day(start_date=hist_sr, end_date=end_date)
    day = day.copy()
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    sr_map = _build_sr_map(day, event_dates, stocks)
    print(f"  有水位 stock-day {len(sr_map):,}", flush=True)

    recs: list[dict] = []
    for a, b in _month_starts(start_date, end_date):
        print(f"  掃 m1 {a}～{b}…", flush=True)
        m1 = load_pattern_m1(start_date=a, end_date=b)
        if m1 is None or m1.empty:
            continue
        m1 = m1[m1["stock_id"].astype(str).isin(stocks)]
        recs.extend(_scan_chunk(m1, sr_map))
        print(f"    累計穿越 {len(recs):,}", flush=True)

    hit = pd.DataFrame(recs)
    print(f"  穿越列（重複計）{len(hit):,}", flush=True)
    if hit.empty:
        print("無穿越事件", flush=True)
        return hit

    feat_cols = [
        "stock_id",
        "day",
        "foreign_pr",
        "trust_pr",
        "dealer_pr",
        "margin_dlt_pr",
        "short_dlt_pr",
        "short_dlt_pct",
        "ret_1d",
        "ret_5d",
        "feat_atr",
        "day_atr",
    ]
    use = [c for c in feat_cols if c in panel.columns]
    hit = hit.merge(panel[use], on=["stock_id", "day"], how="inner")
    hit = _attach_paths(hit)
    print(f"  ∩ 隔夜特徵後 {len(hit):,}", flush=True)

    first = hit.sort_values(["stock_id", "day", "time"]).drop_duplicates(
        ["stock_id", "day"], keep="first"
    )
    n_days = panel["day"].nunique()
    print(
        f"  當日第一筆 {len(first):,}  "
        f"（約 {100 * len(first) / max(len(panel), 1):.1f}% 的 ib∩margin stock-day）",
        flush=True,
    )

    print("\n" + "=" * 64, flush=True)
    print("穿越本身（含同一檔當日重複）", flush=True)
    print("=" * 64, flush=True)
    all_h = pd.Series(True, index=hit.index)
    _print_mix("全部", hit, all_h, min_n, cols=_PATH_COLS)
    _print_mix("全部 順向（上=+、下=空）", hit, all_h, min_n, cols=_ALIGN_COLS)
    _print_mix("vwap_dir=up", hit, hit["vwap_dir"] == "up", min_n, cols=_PATH_COLS)
    _print_mix("vwap_dir=down", hit, hit["vwap_dir"] == "down", min_n, cols=_PATH_COLS)

    print("\n" + "=" * 64, flush=True)
    print("當日第一筆穿越（給 D-1 疊加，避免重複灌樣本）", flush=True)
    print("=" * 64, flush=True)
    all_f = pd.Series(True, index=first.index)
    _print_mix("第一筆 全部", first, all_f, min_n, cols=_PATH_COLS)
    _print_mix("第一筆 順向", first, all_f, min_n, cols=_ALIGN_COLS)
    _print_mix("第一筆 up", first, first["vwap_dir"] == "up", min_n, cols=_PATH_COLS)
    _print_mix("第一筆 down", first, first["vwap_dir"] == "down", min_n, cols=_PATH_COLS)

    print("\n" + "=" * 64, flush=True)
    print("第一筆 × D-1（融券真漲＝ret_1d≥0.5×feat_atr）", flush=True)
    print("=" * 64, flush=True)
    masks = _overlay_masks(first)
    for name, m in masks.items():
        _print_mix(name, first, m, min_n, cols=_PATH_COLS)
        _print_mix(f"{name} ∩ up", first, m & (first["vwap_dir"] == "up"), min_n, cols=_PATH_COLS)

    print(f"\n交易日 {n_days}  耗時 {time.time() - t0:.1f}s", flush=True)
    return hit


def main() -> None:
    p = argparse.ArgumentParser(description="VWAP+壓力支撐穿越 × D-1 ib/margin")
    p.add_argument("--start_date", default="2025-01-01")
    p.add_argument("--end_date", default="2026-08-14")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--min_hist", type=int, default=10)
    p.add_argument("--min_n", type=int, default=100)
    p.add_argument("--tick_only", action="store_true")
    args = p.parse_args()
    run(
        args.start_date,
        args.end_date,
        lookback=args.lookback,
        min_hist=args.min_hist,
        min_n=args.min_n,
        tick_only=args.tick_only,
    )


if __name__ == "__main__":
    main()
