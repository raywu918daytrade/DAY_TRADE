"""VWAP 表 MACD 柱體背離燈：當日 1 分 K 第一次有效底／頂背離。

規則與 pattern/macd_hist_bull、macd_hist_bear 相同（MACD 12/26/9、柱體左右
各 2 根極值、連續兩極值距離 3–30、價創新低／高且柱體抬高／降低、兩點同在
零軸一側）。差異只有：

- 掃完整當日 1 分（不截 120 根、不管 max_age=5），第一次成立就亮到收盤
- 成立時間用右極值確認根 t2+pivot_l（左右各 L 根齊了才算看到這個極值）
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pattern.macd_hist_bull.detector import macd_histogram
from pattern.vwap_sr_scan import _hhmm, load_day_m1

_TW = timezone(timedelta(hours=8))

PIVOT_L = 2
MIN_DIST = 3
MAX_DIST = 30
MIN_CANDLES = 40
# DIF 用 26 期 EMA、DEA 再 9；開頭這段柱體還在熱身，不當背離。
WARMUP = 26 + 9
TODAY_TTL_SEC = 30.0

_cache: dict[str, dict[str, dict]] = {}
_today_pack: tuple[str, float, dict[str, dict]] | None = None
_lock = threading.Lock()


def _extrema(hist: np.ndarray, kind: str) -> list[int]:
    n = len(hist)
    L = PIVOT_L
    out: list[int] = []
    for i in range(L, n - L):
        window = hist[i - L : i + L + 1]
        if kind == "trough":
            if hist[i] == np.min(window) and np.sum(window == hist[i]) == 1:
                out.append(i)
        elif hist[i] == np.max(window) and np.sum(window == hist[i]) == 1:
            out.append(i)
    return out


def first_div_for_group(g: pd.DataFrame) -> dict | None:
    """同一檔當日 1 分 K（已依時間排）。回 {time, kind} 或 None。"""
    if g is None or g.empty or len(g) < MIN_CANDLES:
        return None
    need = ["date", "high", "low", "close"]
    if any(c not in g.columns for c in need):
        return None
    g = g.sort_values("date")
    close = g["close"].astype(float).to_numpy()
    if not np.isfinite(close).all():
        return None
    highs = g["high"].astype(float).to_numpy()
    lows = g["low"].astype(float).to_numpy()
    times = g["date"].to_numpy()
    hist = macd_histogram(close)
    n = len(hist)

    best_i = n
    kinds: list[str] = []

    troughs = _extrema(hist, "trough")
    for j in range(1, len(troughs)):
        t1, t2 = troughs[j - 1], troughs[j]
        if t1 < WARMUP:
            continue
        if not (MIN_DIST <= t2 - t1 <= MAX_DIST):
            continue
        h1, h2 = float(hist[t1]), float(hist[t2])
        if h1 > 0 or h2 > 0:
            continue
        if not (float(lows[t2]) < float(lows[t1]) and h2 > h1):
            continue
        confirmed = t2 + PIVOT_L
        if confirmed < MIN_CANDLES - 1 or confirmed >= n or confirmed > best_i:
            continue
        if confirmed < best_i:
            best_i = confirmed
            kinds = ["bull"]
        elif confirmed == best_i and "bull" not in kinds:
            kinds.append("bull")

    peaks = _extrema(hist, "peak")
    for j in range(1, len(peaks)):
        p1, p2 = peaks[j - 1], peaks[j]
        if p1 < WARMUP:
            continue
        if not (MIN_DIST <= p2 - p1 <= MAX_DIST):
            continue
        h1, h2 = float(hist[p1]), float(hist[p2])
        if h1 < 0 or h2 < 0:
            continue
        if not (float(highs[p2]) > float(highs[p1]) and h2 < h1):
            continue
        confirmed = p2 + PIVOT_L
        if confirmed < MIN_CANDLES - 1 or confirmed >= n or confirmed > best_i:
            continue
        if confirmed < best_i:
            best_i = confirmed
            kinds = ["bear"]
        elif confirmed == best_i and "bear" not in kinds:
            kinds.append("bear")

    if best_i >= n:
        return None
    kind = "both" if len(kinds) > 1 else kinds[0]
    return {"time": _hhmm(times[best_i]), "kind": kind}


def _compute(date_str: str) -> dict[str, dict]:
    m1 = load_day_m1(date_str)
    if m1 is None or m1.empty:
        return {}
    out: dict[str, dict] = {}
    for sid, g in m1.groupby("stock_id", sort=False):
        hit = first_div_for_group(g)
        if hit:
            out[str(sid)] = hit
    return out


def metrics_for_date(date_str: str) -> dict[str, dict]:
    global _today_pack
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    now = time.monotonic()
    with _lock:
        if date_str != today and date_str in _cache:
            return _cache[date_str]
        if date_str == today and _today_pack is not None:
            cached_date, ts, result = _today_pack
            if cached_date == date_str and now - ts < TODAY_TTL_SEC:
                return result
    result = _compute(date_str)
    with _lock:
        if date_str != today:
            _cache[date_str] = result
        else:
            _today_pack = (date_str, time.monotonic(), result)
    return result
