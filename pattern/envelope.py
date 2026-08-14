"""日K 包絡壓力／支撐價（與股票清單 m1 疊圖同一套，不要求三角過關）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pattern.triangle.detector import TriangleDetector

_DET = TriangleDetector()


def envelope_sr_prices(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """回傳最後一根日K 上的 (壓力, 支撐)。df 應已是「D 之前」的日K。"""
    if df is None or df.empty or len(df) < 20:
        return None, None
    sub = df.iloc[-120:].reset_index(drop=True)
    n = len(sub)
    pivots = _DET._filter_alternating_pivots(_DET._extract_raw_pivots(sub))
    peaks = [p for p in pivots if p.type == "peak"]
    troughs = [p for p in pivots if p.type == "trough"]
    if len(peaks) >= 3:
        peaks = peaks[-3:]
    if len(troughs) >= 3:
        troughs = troughs[-3:]
    if len(peaks) < 2 or len(troughs) < 2:
        return None, None

    p1, p_last = peaks[0], peaks[-1]
    dx_u = p_last.index - p1.index
    if dx_u <= 0:
        return None, None
    slope_u = (p_last.price - p1.price) / float(dx_u)
    intercept_u = p1.price - slope_u * p1.index
    diffs_u = np.array([p.price for p in peaks]) - (
        slope_u * np.array([p.index for p in peaks]) + intercept_u
    )
    max_diff_u = float(np.max(diffs_u))
    if max_diff_u > 0:
        intercept_u += max_diff_u

    t1, t_last = troughs[0], troughs[-1]
    dx_l = t_last.index - t1.index
    if dx_l <= 0:
        return None, None
    slope_l = (t_last.price - t1.price) / float(dx_l)
    intercept_l = t1.price - slope_l * t1.index
    diffs_l = (slope_l * np.array([p.index for p in troughs]) + intercept_l) - np.array(
        [p.price for p in troughs]
    )
    max_diff_l = float(np.max(diffs_l))
    if max_diff_l > 0:
        intercept_l -= max_diff_l

    end_idx = n - 1
    res = float(slope_u * end_idx + intercept_u)
    sup = float(slope_l * end_idx + intercept_l)
    if not np.isfinite(res) or not np.isfinite(sup) or res <= 0 or sup <= 0:
        return None, None
    return res, sup
