"""
MACD 柱體底背離 (Bullish Histogram Divergence) 型態檢測器

硬性條件：
1. MACD(12,26,9) histogram = DIF − DEA
2. 柱體局部低點：左右各 L 根嚴格更低
3. 連續兩處柱體低點 t1 < t2：
   - price low[t2] < low[t1]（價創新低）
   - hist[t2] > hist[t1]（柱體抬高）
   - hist[t1] ≤ 0 且 hist[t2] ≤ 0
4. 兩點間距 root ∈ [min_dist, max_dist]
5. 右底 t2 距最後一根 ≤ max_age（型態夠新）
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """與 pandas ewm(span=..., adjust=False) 對齊的 EMA。"""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def macd_histogram(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    dif = _ema(close, fast) - _ema(close, slow)
    dea = _ema(dif, signal)
    return dif - dea


class MacdHistBullDetector(BasePatternDetector):
    """MACD 柱體底背離檢測器"""

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
        super().__init__(name="macd_hist_bull", display_name="MACD柱底背離")
        self.pivot_l = pivot_l
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_age = max_age
        self.min_candles = min_candles
        self.max_candles = max_candles
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def _hist_troughs(self, hist: np.ndarray) -> List[int]:
        """柱體局部低點索引（左右各 pivot_l 根嚴格更低）。"""
        n = len(hist)
        L = self.pivot_l
        troughs: List[int] = []
        for i in range(L, n - L):
            window = hist[i - L : i + L + 1]
            if hist[i] == np.min(window) and np.sum(window == hist[i]) == 1:
                troughs.append(i)
        return troughs

    def _score(
        self,
        t1: int,
        t2: int,
        low1: float,
        low2: float,
        h1: float,
        h2: float,
        latest_idx: int,
    ) -> float:
        age = latest_idx - t2
        freshness = max(0.0, 40.0 * (1.0 - age / max(self.max_age, 1)))
        lift = (h2 - h1) / (abs(h1) + 1e-12)
        hist_score = float(np.clip(lift / 0.5, 0.0, 1.0) * 35.0)
        price_drop = (low1 - low2) / (abs(low1) + 1e-12)
        price_score = float(np.clip(price_drop / 0.03, 0.0, 1.0) * 25.0)
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
        lows = sub["low"].astype(float).to_numpy()
        dates = sub["date"].astype(str).to_numpy()
        hist = macd_histogram(close, self.macd_fast, self.macd_slow, self.macd_signal)

        troughs = self._hist_troughs(hist)
        if len(troughs) < 2:
            return None

        latest_idx = n - 1
        best: Optional[Tuple[float, int, int]] = None

        for j in range(1, len(troughs)):
            t1, t2 = troughs[j - 1], troughs[j]
            dist = t2 - t1
            if not (self.min_dist <= dist <= self.max_dist):
                continue
            if latest_idx - t2 > self.max_age:
                continue
            h1, h2 = float(hist[t1]), float(hist[t2])
            if h1 > 0 or h2 > 0:
                continue
            low1, low2 = float(lows[t1]), float(lows[t2])
            if not (low2 < low1 and h2 > h1):
                continue
            score = self._score(t1, t2, low1, low2, h1, h2, latest_idx)
            if best is None or score > best[0]:
                best = (score, t1, t2)

        if best is None:
            return None

        score, t1, t2 = best
        low1, low2 = float(lows[t1]), float(lows[t2])
        h1, h2 = float(hist[t1]), float(hist[t2])

        pivots = [
            PivotPoint(index=t1, date=str(dates[t1]), price=low1, type="trough"),
            PivotPoint(index=t2, date=str(dates[t2]), price=low2, type="trough"),
        ]
        slope = (low2 - low1) / max(t2 - t1, 1)
        intercept = low1 - slope * t1
        lines = [
            TrendLine(
                start_index=t1,
                end_index=t2,
                start_date=str(dates[t1]),
                end_date=str(dates[t2]),
                start_price=low1,
                end_price=low2,
                slope=float(slope),
                intercept=float(intercept),
                r_squared=1.0,
                line_type="support",
            )
        ]

        return PatternResult(
            stock_id=str(stock_id),
            pattern_type=self.name,
            sub_type="hist_bull_div",
            timeframe=str(timeframe),
            score=score,
            date=str(dates[latest_idx]),
            pivots=pivots,
            lines=lines,
            details={
                "hist1": round(h1, 6),
                "hist2": round(h2, 6),
                "dist_bars": int(t2 - t1),
                "t1_index": int(t1),
                "t2_index": int(t2),
                "price1": round(low1, 4),
                "price2": round(low2, 4),
            },
        )
