"""
MACD 柱體頂背離 (Bearish Histogram Divergence) 型態檢測器

硬性條件：
1. MACD(12,26,9) histogram = DIF − DEA
2. 柱體局部高點：左右各 L 根嚴格更高
3. 連續兩處柱體高點 p1 < p2：
   - price high[p2] > high[p1]（價創新高）
   - hist[p2] < hist[p1]（柱體降低）
   - hist[p1] ≥ 0 且 hist[p2] ≥ 0
4. 兩點間距 ∈ [min_dist, max_dist]
5. 右頂 p2 距最後一根 ≤ max_age
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine
from pattern.macd_hist_bull.detector import macd_histogram


class MacdHistBearDetector(BasePatternDetector):
    """MACD 柱體頂背離檢測器"""

    def __init__(
        self,
        pivot_l: int = 2,
        min_dist: int = 3,
        max_dist: int = 30,
        max_age: int = 5,
        min_candles: int = 40,
        max_candles: int = 120,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        super().__init__(name="macd_hist_bear", display_name="MACD柱頂背離")
        self.pivot_l = pivot_l
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_age = max_age
        self.min_candles = min_candles
        self.max_candles = max_candles
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def _hist_peaks(self, hist: np.ndarray) -> List[int]:
        n = len(hist)
        L = self.pivot_l
        peaks: List[int] = []
        for i in range(L, n - L):
            window = hist[i - L : i + L + 1]
            if hist[i] == np.max(window) and np.sum(window == hist[i]) == 1:
                peaks.append(i)
        return peaks

    def _score(
        self,
        p1: int,
        p2: int,
        hi1: float,
        hi2: float,
        h1: float,
        h2: float,
        latest_idx: int,
    ) -> float:
        age = latest_idx - p2
        freshness = max(0.0, 40.0 * (1.0 - age / max(self.max_age, 1)))
        drop = (h1 - h2) / (abs(h1) + 1e-12)
        hist_score = float(np.clip(drop / 0.5, 0.0, 1.0) * 35.0)
        price_up = (hi2 - hi1) / (abs(hi1) + 1e-12)
        price_score = float(np.clip(price_up / 0.03, 0.0, 1.0) * 25.0)
        return float(np.clip(freshness + hist_score + price_score, 0.0, 100.0))

    def detect(self, df: pd.DataFrame, stock_id: str, timeframe: str) -> Optional[PatternResult]:
        if df is None or df.empty or len(df) < self.min_candles:
            return None
        need = ["date", "open", "high", "low", "close"]
        if any(c not in df.columns for c in need):
            return None

        sub = df.iloc[-self.max_candles :].reset_index(drop=True)
        n = len(sub)
        if n < self.min_candles:
            return None

        close = sub["close"].astype(float).to_numpy()
        highs = sub["high"].astype(float).to_numpy()
        dates = sub["date"].astype(str).to_numpy()
        hist = macd_histogram(close, self.macd_fast, self.macd_slow, self.macd_signal)

        peaks = self._hist_peaks(hist)
        if len(peaks) < 2:
            return None

        latest_idx = n - 1
        best: Optional[Tuple[float, int, int]] = None

        for j in range(1, len(peaks)):
            p1, p2 = peaks[j - 1], peaks[j]
            dist = p2 - p1
            if not (self.min_dist <= dist <= self.max_dist):
                continue
            if latest_idx - p2 > self.max_age:
                continue
            h1, h2 = float(hist[p1]), float(hist[p2])
            if h1 < 0 or h2 < 0:
                continue
            hi1, hi2 = float(highs[p1]), float(highs[p2])
            if not (hi2 > hi1 and h2 < h1):
                continue
            score = self._score(p1, p2, hi1, hi2, h1, h2, latest_idx)
            if best is None or score > best[0]:
                best = (score, p1, p2)

        if best is None:
            return None

        score, p1, p2 = best
        hi1, hi2 = float(highs[p1]), float(highs[p2])
        h1, h2 = float(hist[p1]), float(hist[p2])

        pivots = [
            PivotPoint(index=p1, date=str(dates[p1]), price=hi1, type="peak"),
            PivotPoint(index=p2, date=str(dates[p2]), price=hi2, type="peak"),
        ]
        slope = (hi2 - hi1) / max(p2 - p1, 1)
        intercept = hi1 - slope * p1
        lines = [
            TrendLine(
                start_index=p1,
                end_index=p2,
                start_date=str(dates[p1]),
                end_date=str(dates[p2]),
                start_price=hi1,
                end_price=hi2,
                slope=float(slope),
                intercept=float(intercept),
                r_squared=1.0,
                line_type="resistance",
            )
        ]

        return PatternResult(
            stock_id=str(stock_id),
            pattern_type=self.name,
            sub_type="hist_bear_div",
            timeframe=str(timeframe),
            score=score,
            date=str(dates[latest_idx]),
            pivots=pivots,
            lines=lines,
            details={
                "hist1": round(h1, 6),
                "hist2": round(h2, 6),
                "dist_bars": int(p2 - p1),
                "p1_index": int(p1),
                "p2_index": int(p2),
                "price1": round(hi1, 4),
                "price2": round(hi2, 4),
            },
        )
