"""
MACD 柱體頂背離 (Bearish Histogram Divergence) 型態檢測器

硬性條件：
1. MACD(12,26,9) histogram = DIF − DEA
2. 柱體局部高點：左右各 L 根嚴格更高
3. 價格轉折高點：K 線 high 左右各 L 根更低；可與柱體峰差最多 L 根
   （不是當下收盤對當下柱）
4. 連續兩處確認後的紅柱峰 h1 < h2：
   - 對應價格高點創新高
   - hist[h2] < hist[h1]（柱體降低）
   - (h1, h2) 中間至少一根綠柱 → 紅綠紅
5. 兩柱峰間距 ∈ [min_dist, max_dist]；右峰距最後一根 ≤ max_age
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine
from pattern.macd_hist_bull.detector import iter_macd_hist_div_pairs, macd_histogram


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

        latest_idx = n - 1
        best: Optional[Tuple[float, dict]] = None

        for pair in iter_macd_hist_div_pairs(
            hist, highs, sub["low"].astype(float).to_numpy(),
            side="bear",
            pivot_l=self.pivot_l,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
        ):
            if latest_idx - pair["h2"] > self.max_age:
                continue
            score = self._score(
                pair["h1"], pair["h2"],
                pair["price1"], pair["price2"],
                pair["hist1"], pair["hist2"],
                latest_idx,
            )
            if best is None or score > best[0]:
                best = (score, pair)

        if best is None:
            return None

        score, pair = best
        p1, p2 = pair["p1"], pair["p2"]
        hi1, hi2 = pair["price1"], pair["price2"]
        h1, h2 = pair["hist1"], pair["hist2"]

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
                "dist_bars": int(pair["h2"] - pair["h1"]),
                "p1_index": int(pair["h1"]),
                "p2_index": int(pair["h2"]),
                "price1": round(hi1, 4),
                "price2": round(hi2, 4),
            },
        )
