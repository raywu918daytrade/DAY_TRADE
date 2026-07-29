"""
頭肩頂 (Head and Shoulders Top) 型態檢測器

幾何結構與硬性條件：
1. L0 (Trough): 起漲低點
2. LS (Left Shoulder): 左肩高點
3. N1 (Neckline 1): 左肩與頭部之間的拉回低點
4. H (Head): 頭部高點 (必須是型態中最高點)
5. N2 (Neckline 2): 頭部與右肩之間的拉回低點
6. RS (Right Shoulder): 右肩高點
7. 頸線對齊: N1 與 N2 的連線斜率不宜過大 (用 support 支撐線表示)
8. 肩部對齊: LS 與 RS 的價格差距應在一定比例內
9. 時間對稱性: 左肩到頭部與頭部到右肩的時間應接近
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


class HeadShouldersTopDetector(BasePatternDetector):
    """頭肩頂 (Head and Shoulders Top) 型態檢測器"""

    def __init__(
        self,
        window: int = 3,
        max_shoulder_diff_pct: float = 0.10,  # 左右肩最大價格差距比例
        min_head_height_pct: float = 0.05,    # 頭部相對於肩部的最小高度比例
        max_neckline_slope_pct: float = 0.05, # 頸線最大斜率比例 (N1 vs N2)
        min_candles: int = 30,
        max_candles: int = 150,
    ):
        super().__init__(name="head_shoulders_top", display_name="頭肩頂")
        self.window = window
        self.max_shoulder_diff_pct = max_shoulder_diff_pct
        self.min_head_height_pct = min_head_height_pct
        self.max_neckline_slope_pct = max_neckline_slope_pct
        self.min_candles = min_candles
        self.max_candles = max_candles

    def _extract_pivots(self, df: pd.DataFrame) -> List[PivotPoint]:
        """提取波段點並保持 Peak/Trough 交替"""
        highs = df["high"].values
        lows = df["low"].values
        dates = df["date"].astype(str).values
        n = len(df)
        w = self.window

        raw_pivots: List[PivotPoint] = []
        for i in range(w, n - w):
            is_peak = (highs[i] == np.max(highs[i - w : i + w + 1])) and (highs[i] > np.min(highs[i - w : i + w + 1]))
            is_trough = (lows[i] == np.min(lows[i - w : i + w + 1])) and (lows[i] < np.max(lows[i - w : i + w + 1]))

            if is_peak and not is_trough:
                raw_pivots.append(PivotPoint(index=i, date=dates[i], price=float(highs[i]), type="peak"))
            elif is_trough and not is_peak:
                raw_pivots.append(PivotPoint(index=i, date=dates[i], price=float(lows[i]), type="trough"))

        if not raw_pivots:
            return []

        # 強制 Peak / Trough 嚴格交替
        pivots: List[PivotPoint] = [raw_pivots[0]]
        for p in raw_pivots[1:]:
            last = pivots[-1]
            if p.type == last.type:
                if p.type == "peak" and p.price > last.price:
                    pivots[-1] = p
                elif p.type == "trough" and p.price < last.price:
                    pivots[-1] = p
            else:
                pivots.append(p)

        return pivots

    def detect(self, df: pd.DataFrame, stock_id: str, timeframe: str) -> Optional[PatternResult]:
        """對單一股票執行頭肩頂型態檢測"""
        if df.empty or len(df) < self.min_candles:
            return None

        sub_df = df.iloc[-self.max_candles :].reset_index(drop=True)
        n = len(sub_df)
        pivots = self._extract_pivots(sub_df)

        if len(pivots) < 6: # 需要至少 L0, LS, N1, H, N2, RS
            return None

        latest_close = float(sub_df["close"].iloc[-1])
        latest_date = str(sub_df["date"].iloc[-1])

        best_match = None
        best_score = -1.0

        # 檢視組合：L0 (Trough) -> LS (Peak) -> N1 (Trough) -> H (Peak) -> N2 (Trough) -> RS (Peak)
        for i in range(1, len(pivots) - 4):
            p_l0 = pivots[i - 1]
            p_ls = pivots[i]
            p_n1 = pivots[i + 1]
            p_h  = pivots[i + 2]
            p_n2 = pivots[i + 3]
            p_rs = pivots[i + 4]

            if not (p_l0.type == "trough" and p_ls.type == "peak" and p_n1.type == "trough" and
                    p_h.type == "peak" and p_n2.type == "trough" and p_rs.type == "peak"):
                continue

            pls, pn1, ph, pn2, prs = p_ls.price, p_n1.price, p_h.price, p_n2.price, p_rs.price

            # 【硬性條件 1】：頭部必須是最高點
            if ph <= pls or ph <= prs:
                continue

            # 【硬性條件 2】：肩部價格對齊 (差距不超過 10%)
            shoulder_diff = abs(pls - prs) / max(pls, prs)
            if shoulder_diff > self.max_shoulder_diff_pct:
                continue

            # 【硬性條件 3】：頸線斜率比例 (N1 vs N2)
            neckline_diff = abs(pn1 - pn2) / min(pn1, pn2)
            if neckline_diff > self.max_neckline_slope_pct:
                continue

            # 【硬性條件 4】：頭部高度 (相對於肩部的高出比例)
            head_height = (ph - max(pls, prs)) / max(pls, prs)
            if head_height < self.min_head_height_pct:
                continue

            # ---- 決定右側繪圖結束點 D ----
            # 尋找 RS 之後第一個收盤價跌破頸線 (N1, N2 連線) 的位置
            # 頸線方程: y = slope * x + intercept
            slope = (pn2 - pn1) / max(1, p_n2.index - p_n1.index)
            intercept = pn1 - slope * p_n1.index
            
            breakdown_idx = None
            for idx in range(p_rs.index + 1, n):
                neckline_price = slope * idx + intercept
                if sub_df["close"].iloc[idx] <= neckline_price:
                    breakdown_idx = idx
                    break

            # 視覺對稱限制：右肩後段不超過左肩到頭部距離的 1.2 倍
            symmetry_dist = p_h.index - p_ls.index
            symmetry_limit_idx = p_rs.index + int(symmetry_dist * 1.2)
            
            if breakdown_idx is not None:
                d_index = min(n - 1, breakdown_idx + 2, symmetry_limit_idx)
            else:
                d_index = min(n - 1, symmetry_limit_idx)
            
            d_price = float(sub_df["close"].iloc[d_index])

            # ---- 計算時間與振幅對稱性 ----
            t1 = max(1, p_n1.index - p_ls.index) # LS -> N1
            t2 = max(1, p_h.index - p_n1.index)  # N1 -> H
            t3 = max(1, p_n2.index - p_h.index)  # H -> N2
            t4 = max(1, p_rs.index - p_n2.index) # N2 -> RS
            t5 = max(1, d_index - p_rs.index)    # RS -> D
            
            a1 = pls - pn1       # LS falling
            a2 = ph - pn1        # Head rising
            a3 = ph - pn2        # Head falling
            a4 = prs - pn2       # RS rising
            a5 = prs - d_price   # RS falling (breakdown)

            def calc_cv(data):
                mean = sum(data) / len(data)
                std = (sum([(x - mean)**2 for x in data]) / len(data))**0.5
                return std / mean if mean != 0 else 1.0

            times = [t1, t2, t3, t4, t5]
            amps = [a1, a2, a3, a4, a5]
            t_cv = calc_cv(times)
            a_cv = calc_cv(amps)

            # 【硬性條件 5】：時間比例差距
            if max(times) / min(times) > 4.5:
                continue

            # ---- 計算評分 (Score 0 ~ 100) ----
            # 1. 肩部對齊得分 (最高 20 分)
            score_shoulders = max(0.0, 20.0 * (1.0 - shoulder_diff / self.max_shoulder_diff_pct))
            
            # 2. 頸線平整度得分 (最高 10 分)
            score_neckline = max(0.0, 10.0 * (1.0 - neckline_diff / self.max_neckline_slope_pct))
            
            # 3. 頭部高度得分 (最高 15 分)
            score_head = min(15.0, head_height * 75.0)
            
            # 4. 時間與振幅對稱得分 (最高 15 分)
            score_symmetry = max(0.0, 15.0 * (1.0 - (t_cv + a_cv) / 1.6))
            
            # 5. 跌破力道得分 (最高 15 分)
            neckline_price_now = slope * (n - 1) + intercept
            if latest_close <= neckline_price_now * 1.01:
                score_breakdown = min(15.0, 7.5 + (neckline_price_now - latest_close) / latest_close * 150.0)
            else:
                score_breakdown = max(0.0, 7.5 * (ph - latest_close) / (ph - pn2))

            # 6. 近現性得分 (最高 25 分，右肩 RS 距離當前 K 線越近分數越高)
            bars_from_present = (n - 1) - p_rs.index
            if bars_from_present <= 10:
                score_recency = 25.0
            elif bars_from_present <= 20:
                score_recency = 20.0
            elif bars_from_present <= 35:
                score_recency = 15.0
            elif bars_from_present <= 60:
                score_recency = 10.0
            else:
                score_recency = 2.0

            total_score = float(min(100.0, score_shoulders + score_neckline + score_head + score_symmetry + score_breakdown + score_recency))

            if total_score > best_score:
                best_score = total_score
                best_match = (p_l0, p_ls, p_n1, p_h, p_n2, p_rs, d_index, d_price, shoulder_diff, head_height, times, amps, t_cv, a_cv, slope, intercept)

        if not best_match or best_score < 40.0:
            return None

        p_l0, p_ls, p_n1, p_h, p_n2, p_rs, d_idx, d_price, s_diff, h_height, times, amps, t_cv, a_cv, slope, intercept = best_match
        pls, pn1, ph, pn2, prs = p_ls.price, p_n1.price, p_h.price, p_n2.price, p_rs.price

        # 構建線段
        def create_line(p_start, p_end, price_start, price_end, l_type):
            dx = max(1, p_end.index - p_start.index)
            dy = price_end - price_start
            s = float(dy / dx)
            icpt = float(price_start - s * p_start.index)
            return TrendLine(
                start_index=int(p_start.index),
                end_index=int(p_end.index),
                start_date=str(p_start.date),
                end_date=str(p_end.date),
                start_price=float(price_start),
                end_price=float(price_end),
                slope=s,
                intercept=icpt,
                r_squared=0.95,
                line_type=l_type,
            )

        line1 = create_line(p_l0, p_ls, p_l0.price, pls, "support")
        line2 = create_line(p_ls, p_n1, pls, pn1, "resistance")
        line3 = create_line(p_n1, p_h, pn1, ph, "support")
        line4 = create_line(p_h, p_n2, ph, pn2, "resistance")
        line5 = create_line(p_n2, p_rs, pn2, prs, "support")
        
        # 結束點 D 點
        d_date = str(sub_df["date"].iloc[d_idx])
        line6 = create_line(p_rs, PivotPoint(d_idx, d_date, d_price, "trough"), prs, d_price, "resistance")
        
        # 計算頸線在 D 點處的預期價格
        slope_neck = (pn2 - pn1) / max(1, p_n2.index - p_n1.index)
        p_neck_d = float(pn1 + slope_neck * (d_idx - p_n1.index))
        
        # 頸線 (Neckline 用 support 表現支撐位跌破)
        line_neck = create_line(p_n1, PivotPoint(d_idx, d_date, p_neck_d, "trough"), pn1, p_neck_d, "support")

        neckline_price_now = slope * (n - 1) + intercept
        if latest_close <= neckline_price_now * 0.997:
            breakout_status = "breakout_down"
        else:
            breakout_status = "inside"

        return PatternResult(
            stock_id=stock_id,
            pattern_type="head_shoulders_top",
            sub_type="head_shoulders_top",
            timeframe=timeframe,
            score=best_score,
            date=latest_date,
            pivots=[p_l0, p_ls, p_n1, p_h, p_n2, p_rs],
            lines=[line1, line2, line3, line4, line5, line6, line_neck],
            details={
                "breakout_status": breakout_status,
                "shoulder_diff_pct": round(float(s_diff * 100.0), 2),
                "head_height_pct": round(float(h_height * 100.0), 2),
                "durations": [int(x) for x in times],
                "amplitudes": [round(float(x), 2) for x in amps],
                "time_cv": round(float(t_cv), 3),
                "amp_cv": round(float(a_cv), 3),
                "latest_close": latest_close,
            },
        )
