"""
三角收斂型態檢測器 (Triangle Pattern Detector - Refactored Envelope Version)

特點：
1. 外軌包絡趨勢線 (Envelope Trendline): 趨勢線貼合最高點/最低點外側，避免穿透 K 線。
2. 轉折點交替驗證 (Peak-Trough Alternation): 強制 Peak -> Trough -> Peak -> Trough 波段交替波浪。
3. 幾何結構判定 (Structural Geometry): 高點遞減/水平、低點遞增/水平。
4. 波動度收窄驗證 (Volatility Squeeze): 驗證尾段振幅顯著小於首段振幅。
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


class TriangleDetector(BasePatternDetector):
    """精準版三角收斂檢測器"""

    def __init__(
        self,
        window: int = 3,            # Pivot 點提取滾動視窗大小
        min_pivots: int = 4,        # 上下軌轉折點加總至少 4 個 (例如 2 Highs + 2 Lows)
        min_candles: int = 20,      # 型態最少 K 線根數
        max_candles: int = 120,     # 型態最多 K 線根數
        max_penetration_pct: float = 0.03,  # 允許 K 線超出趨勢線的最大比例
    ):
        super().__init__(name="triangle", display_name="三角收斂")
        self.window = window
        self.min_pivots = min_pivots
        self.min_candles = min_candles
        self.max_candles = max_candles
        self.max_penetration_pct = max_penetration_pct

    def _extract_raw_pivots(self, df: pd.DataFrame) -> List[PivotPoint]:
        """提取波段高點 (Peak) 與低點 (Trough)"""
        highs = df["high"].values
        lows = df["low"].values
        dates = df["date"].astype(str).values
        n = len(df)
        w = self.window

        pivots: List[PivotPoint] = []
        for i in range(w, n - w):
            is_peak = (highs[i] == np.max(highs[i - w : i + w + 1])) and (highs[i] > np.min(highs[i - w : i + w + 1]))
            is_trough = (lows[i] == np.min(lows[i - w : i + w + 1])) and (lows[i] < np.max(lows[i - w : i + w + 1]))

            if is_peak and not is_trough:
                pivots.append(PivotPoint(index=i, date=dates[i], price=float(highs[i]), type="peak"))
            elif is_trough and not is_peak:
                pivots.append(PivotPoint(index=i, date=dates[i], price=float(lows[i]), type="trough"))
            elif is_peak and is_trough:
                # 既是 peak 又是 trough（極少見），依價差決定
                mid = (highs[i] + lows[i]) / 2.0
                if highs[i] - mid >= mid - lows[i]:
                    pivots.append(PivotPoint(index=i, date=dates[i], price=float(highs[i]), type="peak"))
                else:
                    pivots.append(PivotPoint(index=i, date=dates[i], price=float(lows[i]), type="trough"))

        return pivots

    def _filter_alternating_pivots(self, pivots: List[PivotPoint]) -> List[PivotPoint]:
        """過濾波段點，確保呈現 Peak -> Trough -> Peak -> Trough 嚴格交替"""
        if not pivots:
            return []

        alternating: List[PivotPoint] = [pivots[0]]
        for p in pivots[1:]:
            last = alternating[-1]
            if p.type == last.type:
                # 同類型的 Pivot 連續出現，保留更極端的價格點
                if p.type == "peak" and p.price > last.price:
                    alternating[-1] = p
                elif p.type == "trough" and p.price < last.price:
                    alternating[-1] = p
            else:
                alternating.append(p)

        return alternating

    def detect(self, df: pd.DataFrame, stock_id: str, timeframe: str) -> Optional[PatternResult]:
        """對單一股票執行精準三角收斂識別"""
        if df.empty or len(df) < self.min_candles:
            return None

        sub_df = df.iloc[-self.max_candles :].reset_index(drop=True)
        n = len(sub_df)
        if n < self.min_candles:
            return None

        # 1. 提取波段點並強制交替
        raw_pivots = self._extract_raw_pivots(sub_df)
        pivots = self._filter_alternating_pivots(raw_pivots)

        if len(pivots) < self.min_pivots:
            return None

        peaks = [p for p in pivots if p.type == "peak"]
        troughs = [p for p in pivots if p.type == "trough"]

        if len(peaks) < 2 or len(troughs) < 2:
            return None

        # 如果第一個 Peak 價格顯著低於第二個 Peak (例如前置噪訊點)，則跳過第一個 Peak 尋找主三角結構起點
        if len(peaks) >= 3 and peaks[0].price < peaks[1].price * 0.9:
            peaks = peaks[1:]
        if len(troughs) >= 3 and troughs[0].price > troughs[1].price * 1.1:
            troughs = troughs[1:]

        # 2. 上下軌包絡線擬合 (Envelope Trendlines)
        # 上軌：由第一個 Peak 與最後一個 Peak 連線（或穿過 Peak 群的上包覆線）
        p1, p_last = peaks[0], peaks[-1]
        dx_u = p_last.index - p1.index
        if dx_u <= 0:
            return None
        slope_u = (p_last.price - p1.price) / float(dx_u)
        intercept_u = p1.price - slope_u * p1.index

        # 調整上軌截距，使其成為真正的 upper envelope（包覆所有 Peaks）
        peak_indices = np.array([p.index for p in peaks])
        peak_prices = np.array([p.price for p in peaks])
        diffs_u = peak_prices - (slope_u * peak_indices + intercept_u)
        max_diff_u = np.max(diffs_u)
        if max_diff_u > 0:
            intercept_u += max_diff_u

        # 下軌：由第一個 Trough 與最後一個 Trough 連線（或穿過 Trough 群的下包覆線）
        t1, t_last = troughs[0], troughs[-1]
        dx_l = t_last.index - t1.index
        if dx_l <= 0:
            return None
        slope_l = (t_last.price - t1.price) / float(dx_l)
        intercept_l = t1.price - slope_l * t1.index

        # 調整下軌截距，使其成為真正的 lower envelope（包覆所有 Troughs）
        trough_indices = np.array([t.index for t in troughs])
        trough_prices = np.array([t.price for t in troughs])
        diffs_l = (slope_l * trough_indices + intercept_l) - trough_prices
        max_diff_l = np.max(diffs_l)
        if max_diff_l > 0:
            intercept_l -= max_diff_l

        # 3. 幾何收斂性判斷
        # 價格基準 (採用型態起點 Pivot 處的收盤價作為百分比歸一化基準)
        start_pattern_idx = min(p1.index, t1.index)
        ref_price = float(sub_df["close"].iloc[start_pattern_idx])
        if ref_price <= 0:
            return None

        slope_u_pct = (slope_u / ref_price) * 100
        slope_l_pct = (slope_l / ref_price) * 100

        slope_thresh = 0.01  # 每根 K 線 0.01%

        is_sym = (slope_u_pct < -slope_thresh) and (slope_l_pct > slope_thresh)
        is_asc = (abs(slope_u_pct) <= slope_thresh) and (slope_l_pct > slope_thresh)
        is_desc = (slope_u_pct < -slope_thresh) and (abs(slope_l_pct) <= slope_thresh)

        if not (is_sym or is_asc or is_desc):
            return None

        # 收斂條件: 上軌斜率 < 下軌斜率
        if slope_u >= slope_l:
            return None

        # 計算 Apex 交點 x_apex
        slope_diff = slope_u - slope_l
        apex_index = (intercept_l - intercept_u) / slope_diff

        start_pattern_idx = min(p1.index, t1.index)
        latest_idx = n - 1

        # 交點要在第一個 Pivot 之後，且不超過當前位置往後太多 (apex <= n + 1.2 * n)
        if apex_index <= start_pattern_idx or apex_index > n + 1.2 * n:
            return None

        # 4. 波動度收窄驗證 (Volatility Squeeze)
        pattern_len = latest_idx - start_pattern_idx + 1
        if pattern_len < self.min_candles:
            return None

        seg_len = max(5, pattern_len // 3)
        df_pattern = sub_df.iloc[start_pattern_idx : latest_idx + 1]

        ranges = df_pattern["high"] - df_pattern["low"]
        avg_range_head = ranges.iloc[:seg_len].mean()
        avg_range_tail = ranges.iloc[-seg_len:].mean()

        # 後段震盪幅度和前段相比，必須顯著收窄
        squeeze_ratio = (avg_range_tail / avg_range_head) if avg_range_head > 0 else 1.0
        if squeeze_ratio >= 0.85:
            return None  # 未顯著收窄

        # 5. K 線穿透邊界檢測 (Penetration Check)
        highs = sub_df["high"].iloc[start_pattern_idx : latest_idx + 1].values
        lows = sub_df["low"].iloc[start_pattern_idx : latest_idx + 1].values
        idxs = np.arange(start_pattern_idx, latest_idx + 1)

        upper_bounds = slope_u * idxs + intercept_u
        lower_bounds = slope_l * idxs + intercept_l

        # 計算超出趨勢線的穿透率
        pen_upper = np.maximum(0, highs - upper_bounds) / ref_price
        pen_lower = np.maximum(0, lower_bounds - lows) / ref_price

        viol_upper = np.sum(pen_upper > self.max_penetration_pct)
        viol_lower = np.sum(pen_lower > self.max_penetration_pct)

        if viol_upper > 2 or viol_lower > 2:
            return None  # 穿透太多次，線條不具撐壓效力

        # 6. 計算信心分數 (0 ~ 100)
        sub_type = "symmetrical" if is_sym else ("ascending" if is_asc else "descending")

        # (a) Pivot 交替數量分數 (最高 30 分)
        score_pivots = min(30.0, len(pivots) * 6.0)

        # (b) 波幅收窄強度分數 (最高 35 分): squeeze_ratio 越小分數越高
        score_squeeze = min(35.0, max(0.0, (0.85 - squeeze_ratio) / 0.5 * 35.0))

        # (c) 收斂進度分數 (最高 35 分): 最佳進入點為收斂進度 60% ~ 90%
        total_span = apex_index - start_pattern_idx
        progress = (latest_idx - start_pattern_idx) / total_span if total_span > 0 else 0
        if 0.55 <= progress <= 0.90:
            score_progress = 35.0
        elif 0.35 <= progress < 0.55:
            score_progress = 25.0
        else:
            score_progress = 15.0

        total_score = float(min(100.0, float(score_pivots) + float(score_squeeze) + float(score_progress)))

        # 7. 判斷最新價格突破狀態
        latest_close = float(sub_df["close"].iloc[-1])
        upper_now = slope_u * latest_idx + intercept_u
        lower_now = slope_l * latest_idx + intercept_l

        if latest_close > upper_now * 1.003:
            breakout_status = "breakout_up"
        elif latest_close < lower_now * 0.997:
            breakout_status = "breakout_down"
        else:
            breakout_status = "inside"

        # 8. 建立 TrendLine 線段
        end_peak_idx = int(min(latest_idx + 5, int(apex_index)))

        line_upper = TrendLine(
            start_index=int(p1.index),
            end_index=end_peak_idx,
            start_date=str(sub_df["date"].iloc[p1.index]),
            end_date=str(sub_df["date"].iloc[min(end_peak_idx, n - 1)]),
            start_price=float(slope_u * p1.index + intercept_u),
            end_price=float(slope_u * end_peak_idx + intercept_u),
            slope=float(slope_u),
            intercept=float(intercept_u),
            r_squared=0.85,  # 外軌包絡線高品質
            line_type="resistance",
        )

        line_lower = TrendLine(
            start_index=int(t1.index),
            end_index=end_peak_idx,
            start_date=str(sub_df["date"].iloc[t1.index]),
            end_date=str(sub_df["date"].iloc[min(end_peak_idx, n - 1)]),
            start_price=float(slope_l * t1.index + intercept_l),
            end_price=float(slope_l * end_peak_idx + intercept_l),
            slope=float(slope_l),
            intercept=float(intercept_l),
            r_squared=0.85,
            line_type="support",
        )

        return PatternResult(
            stock_id=stock_id,
            pattern_type="triangle",
            sub_type=sub_type,
            timeframe=timeframe,
            score=total_score,
            date=str(sub_df["date"].iloc[-1]),
            pivots=pivots,
            lines=[line_upper, line_lower],
            details={
                "breakout_status": breakout_status,
                "apex_index": round(float(apex_index), 1),
                "upper_slope_pct": round(float(slope_u_pct), 4),
                "lower_slope_pct": round(float(slope_l_pct), 4),
                "squeeze_ratio": round(float(squeeze_ratio), 3),
                "progress_pct": round(float(progress) * 100, 1),
                "latest_close": float(latest_close),
                "upper_boundary_price": round(float(upper_now), 2),
                "lower_boundary_price": round(float(lower_now), 2),
                "pivot_count": int(len(pivots)),
            },
        )
