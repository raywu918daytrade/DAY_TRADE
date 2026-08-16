"""
MACD 柱體底背離 (Bullish Histogram Divergence) 型態檢測器

規則（2026-08-17）：看過去 lookback 根（預設 5）即可鎖定
MACD 柱高低與 K 線高低，不必等右邊再縮小、不必綠紅綠。

1. MACD(12,26,9) histogram = DIF − DEA
2. 柱體谷：hist[j] 是 [j, j+lookback) 唯一最低，且比前一根低；確認在窗右端
3. 同一窗的 K 低點 = 該 5 根 low 的最低（可跟柱谷不同根）
4. 先配柱體：間距、綠紅綠、柱抬高；通過才取兩窗 K 低比價（創新低）
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
    """左右各 pivot_l 根確認後的局部極值；序列尾端未確認的不算。
    背離主流程已改走 lookback=5 的過去窗，這支留給舊呼叫。"""
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


def _lookback_hist_locks(hist: np.ndarray, kind: str, lookback: int) -> List[int]:
    """過去 lookback 根：左端是窗內唯一極值、且相對前一根轉折。確認在 i=j+lookback-1。"""
    n = len(hist)
    L = lookback
    out: List[int] = []
    for i in range(L - 1, n):
        j = i - L + 1
        if j > 0:
            if kind == "trough" and not (hist[j] < hist[j - 1]):
                continue
            if kind == "peak" and not (hist[j] > hist[j - 1]):
                continue
        w = hist[j : i + 1]
        if kind == "trough":
            if hist[j] != np.min(w) or np.sum(w == hist[j]) != 1:
                continue
            if hist[j] > 0:
                continue
        else:
            if hist[j] != np.max(w) or np.sum(w == hist[j]) != 1:
                continue
            if hist[j] < 0:
                continue
        out.append(j)
    return out


def iter_macd_hist_div_pairs(
    hist: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    side: str,
    pivot_l: int = 2,
    min_dist: int = 3,
    max_dist: int = 30,
    warmup: int = 0,
    lookback: int | None = None,
):
    """過去 lookback 根先鎖柱體極值；配對先通過間距、紅綠紅／綠紅綠、柱高低，
    才取兩窗 K 高低比價。lookback 預設 2*pivot_l+1（pivot_l=2 → 5 根）。
    """
    L = int(lookback) if lookback is not None else (2 * int(pivot_l) + 1)
    n = len(hist)
    want_pos = side == "bull"
    if side == "bull":
        h_ext = _lookback_hist_locks(hist, "trough", L)
        px = lows
    else:
        h_ext = _lookback_hist_locks(hist, "peak", L)
        px = highs

    hs: List[int] = []
    for h in h_ext:
        if h < warmup:
            continue
        if h + L - 1 >= n:
            continue
        hs.append(h)

    def _k_at(h: int) -> tuple[int, float]:
        right = h + L - 1
        w = px[h : right + 1]
        off = int(np.argmin(w) if side == "bull" else np.argmax(w))
        p = h + off
        return p, float(px[p])

    for b in range(1, len(hs)):
        h2 = hs[b]
        v2 = float(hist[h2])
        best_h1: int | None = None
        best_v1: float | None = None
        for a in range(b):
            h1 = hs[a]
            if not (min_dist <= h2 - h1 <= max_dist):
                continue
            if not hist_has_sign_between(hist, h1, h2, positive=want_pos):
                continue
            v1 = float(hist[h1])
            if side == "bull":
                if not (v2 > v1):
                    continue
                if best_v1 is None or v1 < best_v1:
                    best_h1, best_v1 = h1, v1
            else:
                if not (v2 < v1):
                    continue
                if best_v1 is None or v1 > best_v1:
                    best_h1, best_v1 = h1, v1
        if best_h1 is None or best_v1 is None:
            continue
        p1, price1 = _k_at(best_h1)
        p2, price2 = _k_at(h2)
        if side == "bull":
            if not (price2 < price1):
                continue
        elif not (price2 > price1):
            continue
        yield {
            "h1": best_h1,
            "h2": h2,
            "p1": p1,
            "p2": p2,
            "hist1": best_v1,
            "hist2": v2,
            "price1": price1,
            "price2": price2,
            "confirmed": h2 + L - 1,
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
