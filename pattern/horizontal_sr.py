"""橫向壓力／支撐：雙轉折水平線（同突破壓力回測／跌破支撐反彈階段 1）。

不必型態過關——只要兩個峰／谷價近、中間有拉回／反彈，就定一條水平水位。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pattern.base import PivotPoint

_WINDOW = 3
_MAX_DIFF_PCT = 0.02
_MIN_MID_PULLBACK_PCT = 0.015
_MIN_DISTANCE = 4
_MIN_CANDLES = 25
_MAX_CANDLES = 120


def _extract_pivots(df: pd.DataFrame, window: int = _WINDOW) -> list[PivotPoint]:
    """與 breakout_retest.detector._extract_pivots 相同。"""
    highs = df["high"].values
    lows = df["low"].values
    dates = df["date"].astype(str).values
    n = len(df)
    w = window

    raw: list[PivotPoint] = []
    for i in range(w, n - w):
        is_peak = (highs[i] == np.max(highs[i - w : i + w + 1])) and (
            highs[i] > np.min(highs[i - w : i + w + 1])
        )
        is_trough = (lows[i] == np.min(lows[i - w : i + w + 1])) and (
            lows[i] < np.max(lows[i - w : i + w + 1])
        )
        if is_peak and not is_trough:
            raw.append(PivotPoint(index=i, date=dates[i], price=float(highs[i]), type="peak"))
        elif is_trough and not is_peak:
            raw.append(PivotPoint(index=i, date=dates[i], price=float(lows[i]), type="trough"))

    if not raw:
        return []

    pivots: list[PivotPoint] = [raw[0]]
    for p in raw[1:]:
        last = pivots[-1]
        if p.type == last.type:
            if p.type == "peak" and p.price > last.price:
                pivots[-1] = p
            elif p.type == "trough" and p.price < last.price:
                pivots[-1] = p
        else:
            pivots.append(p)
    return pivots


@dataclass(frozen=True)
class _LevelCand:
    level: float
    second_idx: int
    align: float  # 越小越好（相對價差）


def _best_resistance(peaks: list[PivotPoint], lows: np.ndarray) -> float | None:
    best: _LevelCand | None = None
    for i in range(len(peaks) - 1):
        for j in range(i + 1, len(peaks)):
            p1, p2 = peaks[i], peaks[j]
            if p2.index - p1.index < _MIN_DISTANCE:
                continue
            r_diff = abs(p1.price - p2.price) / min(p1.price, p2.price)
            if r_diff > _MAX_DIFF_PCT:
                continue
            r_level = (p1.price + p2.price) / 2.0
            mid_lows = lows[p1.index : p2.index + 1]
            if len(mid_lows) == 0:
                continue
            mid_pullback = (r_level - float(np.min(mid_lows))) / r_level
            if mid_pullback < _MIN_MID_PULLBACK_PCT:
                continue
            cand = _LevelCand(level=r_level, second_idx=p2.index, align=r_diff)
            if best is None or (cand.second_idx, -cand.align) > (best.second_idx, -best.align):
                best = cand
    return best.level if best else None


def _best_support(troughs: list[PivotPoint], highs: np.ndarray) -> float | None:
    best: _LevelCand | None = None
    for i in range(len(troughs) - 1):
        for j in range(i + 1, len(troughs)):
            t1, t2 = troughs[i], troughs[j]
            if t2.index - t1.index < _MIN_DISTANCE:
                continue
            s_diff = abs(t1.price - t2.price) / min(t1.price, t2.price)
            if s_diff > _MAX_DIFF_PCT:
                continue
            s_level = (t1.price + t2.price) / 2.0
            mid_highs = highs[t1.index : t2.index + 1]
            if len(mid_highs) == 0:
                continue
            mid_bounce = (float(np.max(mid_highs)) - s_level) / s_level
            if mid_bounce < _MIN_MID_PULLBACK_PCT:
                continue
            cand = _LevelCand(level=s_level, second_idx=t2.index, align=s_diff)
            if best is None or (cand.second_idx, -cand.align) > (best.second_idx, -best.align):
                best = cand
    return best.level if best else None


def horizontal_sr_prices(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """回傳 (壓力, 支撐)；單邊找不到為 None。df 應已是「D 之前」的日K。"""
    if df is None or df.empty or len(df) < _MIN_CANDLES:
        return None, None
    sub = df.iloc[-_MAX_CANDLES:].reset_index(drop=True)
    if len(sub) < _MIN_CANDLES:
        return None, None
    pivots = _extract_pivots(sub)
    if len(pivots) < 2:
        return None, None
    peaks = [p for p in pivots if p.type == "peak"]
    troughs = [p for p in pivots if p.type == "trough"]
    highs = sub["high"].astype(float).to_numpy()
    lows = sub["low"].astype(float).to_numpy()
    res = _best_resistance(peaks, lows) if len(peaks) >= 2 else None
    sup = _best_support(troughs, highs) if len(troughs) >= 2 else None
    if res is not None and (not np.isfinite(res) or res <= 0):
        res = None
    if sup is not None and (not np.isfinite(sup) or sup <= 0):
        sup = None
    return res, sup
