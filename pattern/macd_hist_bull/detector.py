"""
MACD 柱體底背離 (Bullish Histogram Divergence) 型態檢測器

硬性條件：
1. MACD(12,26,9) histogram = DIF − DEA
2. 柱體局部低點：左右各 L 根柱體都更高＝極值之後柱體縮小才確認，
   最新未完成的那根不算
3. 價格轉折低點：K 線 low 同樣左右各 L 根更高；可與柱體谷差最多 L 根
   （不是拿「當下收盤」去對「當下柱」）
4. 連續兩處確認後的綠柱谷 h1 < h2：
   - 對應價格低點創新低
   - hist[h2] > hist[h1]（柱體抬高）
   - (h1, h2) 中間至少一根紅柱 → 綠紅綠
5. 兩柱谷間距 ∈ [min_dist, max_dist]；右谷距最後一根 ≤ max_age
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


def hist_has_sign_between(hist: np.ndarray, i1: int, i2: int, positive: bool) -> bool:
    """開區間 (i1, i2) 內是否出現指定正負的柱。底背離要正柱、頂背離要負柱。"""
    mid = hist[i1 + 1 : i2]
    if mid.size == 0:
        return False
    return bool(np.any(mid > 0) if positive else np.any(mid < 0))


def local_extrema(values: np.ndarray, kind: str, pivot_l: int = 2) -> List[int]:
    """左右各 pivot_l 根確認後的局部極值；序列尾端未確認的不算。"""
    n = len(values)
    L = pivot_l
    out: List[int] = []
    for i in range(L, n - L):
        window = values[i - L : i + L + 1]
        if kind == "trough":
            if values[i] == np.min(window) and np.sum(window == values[i]) == 1:
                out.append(i)
        elif values[i] == np.max(window) and np.sum(window == values[i]) == 1:
            out.append(i)
    return out


def nearest_pivot(pivots: List[int], i: int, max_off: int) -> Optional[int]:
    best: Optional[int] = None
    best_d = max_off + 1
    for p in pivots:
        d = abs(p - i)
        if d < best_d:
            best = p
            best_d = d
    return best if best is not None and best_d <= max_off else None


def iter_macd_hist_div_pairs(
    hist: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    side: str,
    pivot_l: int = 2,
    min_dist: int = 3,
    max_dist: int = 30,
    warmup: int = 0,
):
    """已確認的柱體極值 × 附近已確認的 K 線高低點。yield dict。

    side=bull：綠柱谷 + K 低點創新低、柱抬高、中間紅柱。
    side=bear：紅柱峰 + K 高點創新高、柱降低、中間綠柱。
    當下那根尚未走出「縮小／轉折」時不會進這裡。
    """
    L = pivot_l
    n = len(hist)
    if side == "bull":
        h_ext = [i for i in local_extrema(hist, "trough", L) if hist[i] <= 0]
        p_ext = local_extrema(lows, "trough", L)
        want_pos = True
    else:
        h_ext = [i for i in local_extrema(hist, "peak", L) if hist[i] >= 0]
        p_ext = local_extrema(highs, "peak", L)
        want_pos = False

    for j in range(1, len(h_ext)):
        h1, h2 = h_ext[j - 1], h_ext[j]
        if h1 < warmup:
            continue
        if not (min_dist <= h2 - h1 <= max_dist):
            continue
        if not hist_has_sign_between(hist, h1, h2, positive=want_pos):
            continue
        v1, v2 = float(hist[h1]), float(hist[h2])
        if side == "bull":
            if not (v2 > v1):
                continue
        elif not (v2 < v1):
            continue
        p1 = nearest_pivot(p_ext, h1, L)
        p2 = nearest_pivot(p_ext, h2, L)
        if p1 is None or p2 is None or p2 <= p1:
            continue
        if side == "bull":
            if not (float(lows[p2]) < float(lows[p1])):
                continue
            px1, px2 = float(lows[p1]), float(lows[p2])
        else:
            if not (float(highs[p2]) > float(highs[p1])):
                continue
            px1, px2 = float(highs[p1]), float(highs[p2])
        confirmed = max(h2, p2) + L
        if confirmed >= n:
            continue
        yield {
            "h1": h1,
            "h2": h2,
            "p1": p1,
            "p2": p2,
            "hist1": v1,
            "hist2": v2,
            "price1": px1,
            "price2": px2,
            "confirmed": confirmed,
        }


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

        latest_idx = n - 1
        best: Optional[Tuple[float, dict]] = None

        for pair in iter_macd_hist_div_pairs(
            hist, sub["high"].astype(float).to_numpy(), lows,
            side="bull",
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
        low1, low2 = pair["price1"], pair["price2"]
        h1, h2 = pair["hist1"], pair["hist2"]

        pivots = [
            PivotPoint(index=p1, date=str(dates[p1]), price=low1, type="trough"),
            PivotPoint(index=p2, date=str(dates[p2]), price=low2, type="trough"),
        ]
        slope = (low2 - low1) / max(p2 - p1, 1)
        intercept = low1 - slope * p1
        lines = [
            TrendLine(
                start_index=p1,
                end_index=p2,
                start_date=str(dates[p1]),
                end_date=str(dates[p2]),
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
                "dist_bars": int(pair["h2"] - pair["h1"]),
                "t1_index": int(pair["h1"]),
                "t2_index": int(pair["h2"]),
                "p1_index": int(p1),
                "p2_index": int(p2),
                "price1": round(low1, 4),
                "price2": round(low2, 4),
            },
        )
