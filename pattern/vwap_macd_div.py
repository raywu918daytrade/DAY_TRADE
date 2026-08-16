"""VWAP 表 MACD 柱體背離燈：盤中每根 1 分都重算，不是當天第一次就定終身。

規則與型態偵測器相同：柱體要左右各 2 根確認「縮小」才算極值，K 線也要
走出高低點（可與柱極值差 2 根）。不是當下柱對當下 K。綠紅綠／紅綠紅、
max_age=5。確認之後到 t2+max_age 才亮燈。

盤後／重現掃全日；盤中 on_minute 掃當下 m1_live。
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pattern.macd_hist_bull.detector import iter_macd_hist_div_pairs, macd_histogram
from pattern.vwap_sr_scan import _hhmm, load_day_m1, normalize_universe

_TW = timezone(timedelta(hours=8))

PIVOT_L = 2
MIN_DIST = 3
MAX_DIST = 30
MIN_CANDLES = 40
MAX_AGE = 5
# DIF 用 26 期 EMA、DEA 再 9；開頭這段柱體還在熱身，不當背離。
WARMUP = 26 + 9

_cache: dict[tuple[str, str], dict[str, dict]] = {}
_lock = threading.Lock()


def _epoch(ts) -> int:
    from api import tw_naive_to_epoch

    return tw_naive_to_epoch(pd.Timestamp(ts))


def _leg(kind: str, pair: dict, times, n: int) -> dict | None:
    h1, h2 = int(pair["h1"]), int(pair["h2"])
    p1, p2 = int(pair["p1"]), int(pair["p2"])
    confirmed = int(pair["confirmed"])
    if confirmed < MIN_CANDLES - 1 or confirmed >= n:
        return None
    until_i = min(n - 1, max(confirmed, h2 + MAX_AGE))
    if until_i < confirmed:
        return None
    return {
        "kind": kind,
        "t1": _hhmm(times[h1]),
        "t2": _hhmm(times[h2]),
        "t1_ts": _epoch(times[h1]),
        "t2_ts": _epoch(times[h2]),
        "hist1": round(float(pair["hist1"]), 6),
        "hist2": round(float(pair["hist2"]), 6),
        "p1": _hhmm(times[p1]),
        "p2": _hhmm(times[p2]),
        "p1_ts": _epoch(times[p1]),
        "p2_ts": _epoch(times[p2]),
        "price1": round(float(pair["price1"]), 4),
        "price2": round(float(pair["price2"]), 4),
        "time": _hhmm(times[confirmed]),
        "until": _hhmm(times[until_i]),
    }


def all_divs_for_group(g: pd.DataFrame) -> list[dict]:
    """同一檔當日 1 分 K：柱體縮小確認後的極值，搭配附近 K 線高低點。"""
    if g is None or g.empty or len(g) < MIN_CANDLES:
        return []
    need = ["date", "high", "low", "close"]
    if any(c not in g.columns for c in need):
        return []
    g = g.sort_values("date")
    close = g["close"].astype(float).to_numpy()
    if not np.isfinite(close).all():
        return []
    highs = g["high"].astype(float).to_numpy()
    lows = g["low"].astype(float).to_numpy()
    times = g["date"].to_numpy()
    hist = macd_histogram(close)
    n = len(hist)
    out: list[dict] = []
    kw = dict(pivot_l=PIVOT_L, min_dist=MIN_DIST, max_dist=MAX_DIST, warmup=WARMUP)
    for pair in iter_macd_hist_div_pairs(hist, highs, lows, side="bull", **kw):
        leg = _leg("bull", pair, times, n)
        if leg:
            out.append({**leg, "legs": [leg]})
    for pair in iter_macd_hist_div_pairs(hist, highs, lows, side="bear", **kw):
        leg = _leg("bear", pair, times, n)
        if leg:
            out.append({**leg, "legs": [leg]})
    out.sort(key=lambda e: (e["time"], e["kind"]))
    return out


def divs_map_from_m1(m1: pd.DataFrame) -> dict[str, dict]:
    """stock_id → {events: [...]}，沒背離的不進 map。"""
    out: dict[str, dict] = {}
    if m1 is None or m1.empty:
        return out
    for sid, g in m1.groupby("stock_id", sort=False):
        evs = all_divs_for_group(g)
        if evs:
            out[str(sid)] = {"events": evs}
    return out


def _compute(date_str: str, universe: str = "daytrade") -> dict[str, dict]:
    return divs_map_from_m1(load_day_m1(date_str, universe=universe))


def metrics_for_date(date_str: str, universe: str = "daytrade") -> dict[str, dict]:
    universe = normalize_universe(universe)
    key = (date_str, universe)
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    with _lock:
        if date_str != today and key in _cache:
            return _cache[key]
    result = _compute(date_str, universe=universe)
    if date_str != today:
        with _lock:
            _cache[key] = result
    return result
