"""
W 底 (Double Bottom) 型態檢測器

幾何結構與硬性條件：
1. Trough 1 (左腳): 波段低點 L1
2. Peak (頸線高點): 兩底之間的回升高點 H
3. Trough 2 (右腳): 波段低點 L2
4. 雙底價格差距: |Price(L1) - Price(L2)| / Price(L1) <= 0.05 (兩底價格相差不超過 5%)
5. 頸線深度: 頸線高點 H 必須高於兩底，波幅 (Price(H) - Min(Price(L1), Price(L2))) / Min(...) 在 3% ~ 30% 之間
6. 時間間隔: L1 到 L2 之間的 K 線根數在 5 ~ 60 根之間
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


class WBottomDetector(BasePatternDetector):
    """W 底 (Double Bottom) 型態檢測器"""

    def __init__(
        self,
        window: int = 3,            # Pivot 點提取滾動視窗大小
        max_bottom_diff_pct: float = 0.05,  # 雙底最大價格差距比例 (預設 5%)
        min_depth_pct: float = 0.03,        # 頸線最小深度比例 (預設 3%)
        max_depth_pct: float = 0.30,        # 頸線最大深度比例 (預設 30%)
        min_bottom_dist: int = 5,           # 兩底之間最小 K 線根數
        max_bottom_dist: int = 60,          # 兩底之間最大 K 線根數
        min_candles: int = 20,              # 型態最少 K 線根數
        max_candles: int = 120,             # 型態最多 K 線根數
    ):
        super().__init__(name="w_bottom", display_name="W底")
        self.window = window
        self.max_bottom_diff_pct = max_bottom_diff_pct
        self.min_depth_pct = min_depth_pct
        self.max_depth_pct = max_depth_pct
        self.min_bottom_dist = min_bottom_dist
        self.max_bottom_dist = max_bottom_dist
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
        """對單一股票執行 W 底型態檢測"""
        if df.empty or len(df) < self.min_candles:
            return None

        sub_df = df.iloc[-self.max_candles :].reset_index(drop=True)
        n = len(sub_df)
        if n < self.min_candles:
            return None

        pivots = self._extract_pivots(sub_df)
        if len(pivots) < 4:
            return None

        latest_idx = n - 1
        latest_close = float(sub_df["close"].iloc[-1])
        latest_date = str(sub_df["date"].iloc[-1])

        best_match = None
        best_score = -1.0

        # 檢視組合：P0 (Peak) -> L1 (Trough) -> H (Peak) -> L2 (Trough)
        for i in range(1, len(pivots) - 2):
            p_p0_raw = pivots[i - 1]
            p_l1 = pivots[i]
            p_h = pivots[i + 1]
            p_l2 = pivots[i + 2]

            if not (p_p0_raw.type == "peak" and p_l1.type == "trough" and p_h.type == "peak" and p_l2.type == "trough"):
                continue

            pl1, ph, pl2 = p_l1.price, p_h.price, p_l2.price
            neckline_height = ph - min(pl1, pl2)
            
            # 【優化 P0 搜尋】：不一定要抓緊鄰的 Pivot，往回找價格與頸線 H 最接近的高點
            best_p0 = p_p0_raw
            min_p0_diff = abs(p_p0_raw.price - ph)
            
            for back_idx in range(i - 1, -1, -1):
                prev_p = pivots[back_idx]
                if prev_p.type == "peak":
                    # 幅度對稱性檢查：起跌浪幅度 vs 頸線幅度
                    a1 = prev_p.price - pl1
                    if 0.5 <= (a1 / neckline_height) <= 2.2:
                        diff = abs(prev_p.price - ph)
                        if diff < min_p0_diff:
                            min_p0_diff = diff
                            best_p0 = prev_p

            p_p0 = best_p0
            pp0 = p_p0.price
            a1 = pp0 - pl1

            # 【硬性條件 1】：左側起跌浪 P0 必須高於 L1，且漲跌幅對稱性要在 0.5 ~ 2.0 之間
            if pp0 <= pl1 or not (0.5 <= (a1 / neckline_height) <= 2.2):
                continue

            # 【硬性條件 2】：兩底之間 K 線距離
            bottom_dist = p_l2.index - p_l1.index
            if not (self.min_bottom_dist <= bottom_dist <= self.max_bottom_dist):
                continue

            # 【硬性條件 3】：兩底價格相差不超過 max_bottom_diff_pct (預設 5%)
            min_bottom = min(pl1, pl2)
            bottom_diff_pct = abs(pl1 - pl2) / min_bottom
            if bottom_diff_pct > self.max_bottom_diff_pct:
                continue

            # 【硬性條件 4】：頸線高度/深度在 min_depth_pct ~ max_depth_pct 之間
            depth_pct = (ph - min_bottom) / min_bottom
            if not (self.min_depth_pct <= depth_pct <= self.max_depth_pct):
                continue

            # ---- 決定右側繪圖結束點 D (為了四段全對稱) ----
            # 1. 尋找 L2 之後第一個收盤價突破頸線 H 的位置
            breakout_idx = None
            for idx in range(p_l2.index + 1, n):
                if sub_df["close"].iloc[idx] >= ph:
                    breakout_idx = idx
                    break

            # 2. 定義時間與振幅 S1, S2, S3
            t1 = max(1, p_l1.index - p_p0.index) # S1: P0 -> L1
            t2 = max(1, p_h.index - p_l1.index)  # S2: L1 -> H
            t3 = max(1, p_l2.index - p_h.index)  # S3: H -> L2
            
            a1 = pp0 - pl1      # S1 Amp
            a2 = ph - pl1       # S2 Amp (neckline height)
            a3 = ph - pl2       # S3 Amp

            # 3. 推算對稱的 S4 (L2 -> D)
            # 理想時間 T4 應接近 T2/T3
            target_t4 = int((t2 + t3) / 2)
            # 理想振幅 A4 應接近 A1
            target_p4 = pl2 + a1
            
            # 視覺對稱限制：右側時間不超過中間長度的 1.5 倍
            symmetry_limit_idx = p_l2.index + int(bottom_dist * 1.2)
            
            if breakout_idx is not None:
                # 已突破，結束點 D 設在突破後，且儘量靠近 target_p4
                d_index = min(latest_idx, breakout_idx + 2, symmetry_limit_idx)
                # 尋找 breakout 後最接近 target_p4 的點
                for idx in range(breakout_idx, min(latest_idx, symmetry_limit_idx) + 1):
                    if sub_df["close"].iloc[idx] >= target_p4 * 0.95:
                        d_index = idx
                        break
            else:
                # 未突破，延伸到對稱限制
                d_index = min(latest_idx, symmetry_limit_idx)
            
            d_price = float(sub_df["close"].iloc[d_index])
            t4 = max(1, d_index - p_l2.index)
            a4 = d_price - pl2

            # 【硬性條件 1】：四段時間對稱性
            times = [t1, t2, t3, t4]
            t_min, t_max = min(times), max(times)
            if t_max / t_min > 4.0:
                continue

            # 【硬性條件 2】：四段振幅對稱性
            amps = [a1, a2, a3, a4]
            a_min, a_max = min(amps), max(amps)
            if a_min <= 0 or a_max / a_min > 4.0:
                continue

            # 【硬性條件 3】：現價/結束點必須高於兩底最低點
            if d_price < min_bottom * 0.99:
                continue

            # ---- 計算評分 (Score 0 ~ 100) ----
            # 使用變異係數 (CV) 來衡量全對稱性
            def calc_cv(data):
                mean = sum(data) / len(data)
                std = (sum([(x - mean)**2 for x in data]) / len(data))**0.5
                return std / mean if mean != 0 else 1.0

            time_cv = calc_cv(times)
            amp_cv = calc_cv(amps)

            # 1. 時間全對稱得分 (最高 20 分)
            score_time_sym = max(0.0, 20.0 * (1.0 - time_cv / 0.8))

            # 2. 振幅全對稱得分 (最高 20 分)
            score_amp_sym = max(0.0, 20.0 * (1.0 - amp_cv / 0.8))

            # 3. 雙底平對齊得分 (最高 25 分)
            score_alignment = max(0.0, 25.0 * (1.0 - bottom_diff_pct / self.max_bottom_diff_pct))

            # 4. 頸線深度得分 (最高 15 分)
            depth_diff = abs(depth_pct - 0.12)
            score_depth = max(0.0, 15.0 * (1.0 - depth_diff / 0.15))

            # 5. 突破力道得分 (最高 20 分)
            if latest_close >= ph * 0.99:
                score_breakout = min(20.0, 10.0 + (latest_close - ph) / ph * 200.0)
            else:
                score_breakout = max(0.0, 10.0 * (latest_close - min_bottom) / (ph - min_bottom))

            total_score = float(min(100.0, score_time_sym + score_amp_sym + score_alignment + score_depth + score_breakout))

            if total_score > best_score:
                best_score = total_score
                best_match = (p_p0, p_l1, p_h, p_l2, d_index, d_price, bottom_diff_pct, depth_pct, times, amps, time_cv, amp_cv)

        if not best_match or best_score < 40.0:
            return None

        p_p0, p_l1, p_h, p_l2, d_idx, d_price, b_diff, depth, times, amps, t_cv, a_cv = best_match
        pl1, ph, pl2 = p_l1.price, p_h.price, p_l2.price
        t1, t2, t3, t4 = times
        a1, a2, a3, a4 = amps

        # 構建 W 型幾何 4 條連線 (P0->L1, L1->H, H->L2, L2->D) 與頸線
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

        line_start = create_line(p_p0, p_l1, p_p0.price, pl1, "resistance")
        line_left_leg = create_line(p_l1, p_h, pl1, ph, "support")
        line_right_leg = create_line(p_h, p_l2, ph, pl2, "resistance")
        
        d_date = str(sub_df["date"].iloc[d_idx])
        line_attack = create_line(p_l2, PivotPoint(d_idx, d_date, d_price, "peak"), pl2, d_price, "support")
        
        # 頸線水平線
        line_neckline = create_line(p_l1, PivotPoint(d_idx, d_date, ph, "peak"), ph, ph, "resistance")

        # 突破狀態判定 (以最新收盤價判定，非 D 點價格)
        if latest_close >= p_h.price * 1.003:
            breakout_status = "breakout_up"
        else:
            breakout_status = "inside"

        return PatternResult(
            stock_id=stock_id,
            pattern_type="w_bottom",
            sub_type="double_bottom",
            timeframe=timeframe,
            score=best_score,
            date=latest_date,
            pivots=[p_p0, p_l1, p_h, p_l2],
            lines=[line_start, line_left_leg, line_right_leg, line_attack, line_neckline],
            details={
                "breakout_status": breakout_status,
                "price_P0": round(float(p_p0.price), 2),
                "price_L1": round(float(p_l1.price), 2),
                "price_H": round(float(p_h.price), 2),
                "price_L2": round(float(p_l2.price), 2),
                "bottom_diff_pct": round(float(b_diff * 100.0), 2),
                "neckline_depth_pct": round(float(depth * 100.0), 2),
                "durations": [int(x) for x in times],
                "amplitudes": [round(float(x), 2) for x in amps],
                "time_cv": round(float(t_cv), 3),
                "amp_cv": round(float(a_cv), 3),
                "latest_close": latest_close,
            },
        )
