"""
M 頭 (Double Top) 型態檢測器

幾何結構與硬性條件：
1. Peak 1 (左頭): 波段高點 H1
2. Trough (頸線低點): 兩頭之間的修正低點 L
3. Peak 2 (右頭): 波段高點 H2
4. 雙頭價格差距: |Price(H1) - Price(H2)| / Price(H1) <= 0.05 (兩頭價格相差不超過 5%)
5. 頸線深度: 頸線低點 L 必須低於兩頭，幅度 (Max(Price(H1), Price(H2)) - Price(L)) / Max(...) 在 3% ~ 30% 之間
6. 時間間隔: H1 到 H2 之間的 K 線根數在 5 ~ 60 根之間
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


class MTopDetector(BasePatternDetector):
    """M 頭 (Double Top) 型態檢測器"""

    def __init__(
        self,
        window: int = 3,            # Pivot 點提取滾動視窗大小
        max_top_diff_pct: float = 0.05,     # 雙頭最大價格差距比例 (預設 5%)
        min_depth_pct: float = 0.03,        # 頸線最小深度比例 (預設 3%)
        max_depth_pct: float = 0.30,        # 頸線最大深度比例 (預設 30%)
        min_top_dist: int = 5,              # 兩頭之間最小 K 線根數
        max_top_dist: int = 60,             # 兩頭之間最大 K 線根數
        min_candles: int = 20,              # 型態最少 K 線根數
        max_candles: int = 120,             # 型態最多 K 線根數
    ):
        super().__init__(name="m_top", display_name="M頭")
        self.window = window
        self.max_top_diff_pct = max_top_diff_pct
        self.min_depth_pct = min_depth_pct
        self.max_depth_pct = max_depth_pct
        self.min_top_dist = min_top_dist
        self.max_top_dist = max_top_dist
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
        """對單一股票執行 M 頭型態檢測"""
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

        # 檢視組合：L0 (Trough) -> H1 (Peak) -> L (Trough) -> H2 (Peak)
        for i in range(1, len(pivots) - 2):
            p_l0_raw = pivots[i - 1]
            p_h1 = pivots[i]
            p_l = pivots[i + 1]
            p_h2 = pivots[i + 2]

            if not (p_l0_raw.type == "trough" and p_h1.type == "peak" and p_l.type == "trough" and p_h2.type == "peak"):
                continue

            ph1, pl, ph2 = p_h1.price, p_l.price, p_h2.price
            neckline_depth = max(ph1, ph2) - pl

            # 【優化 L0 搜尋】：往回找價格與頸線 L 最接近的低點
            best_l0 = p_l0_raw
            min_l0_diff = abs(p_l0_raw.price - pl)
            
            for back_idx in range(i - 1, -1, -1):
                prev_p = pivots[back_idx]
                if prev_p.type == "trough":
                    # 幅度對稱性檢查：起漲浪幅度 vs 頸線幅度
                    a1 = ph1 - prev_p.price
                    if 0.5 <= (a1 / neckline_depth) <= 2.2:
                        diff = abs(prev_p.price - pl)
                        if diff < min_l0_diff:
                            min_l0_diff = diff
                            best_l0 = prev_p

            p_l0 = best_l0
            pl0 = p_l0.price
            a1 = ph1 - pl0

            # 【硬性條件 1】：左側起漲浪 L0 必須低於 H1，且漲跌幅對稱性要在 0.5 ~ 2.2 之間
            if pl0 >= ph1 or not (0.5 <= (a1 / neckline_depth) <= 2.2):
                continue

            # 【硬性條件 2】：兩頭之間 K 線距離
            top_dist = p_h2.index - p_h1.index
            if not (self.min_top_dist <= top_dist <= self.max_top_dist):
                continue

            # 【硬性條件 3】：兩頭價格相差不超過 max_top_diff_pct (預設 5%)
            max_top = max(ph1, ph2)
            top_diff_pct = abs(ph1 - ph2) / max_top
            if top_diff_pct > self.max_top_diff_pct:
                continue

            # 【硬性條件 4】：頸線深度在 min_depth_pct ~ max_depth_pct 之間
            depth_pct = (max_top - pl) / max_top
            if not (self.min_depth_pct <= depth_pct <= self.max_depth_pct):
                continue

            # ---- 決定右側繪圖結束點 D (為了對稱性) ----
            # 尋找 H2 之後第一個收盤價跌破頸線 L 的位置
            breakdown_idx = None
            for idx in range(p_h2.index + 1, n):
                if sub_df["close"].iloc[idx] <= pl:
                    breakdown_idx = idx
                    break

            # ---- 決定右側繪圖結束點 D (為了四段全對稱) ----
            # 1. 尋找 H2 之後第一個收盤價跌破頸線 L 的位置
            breakdown_idx = None
            for idx in range(p_h2.index + 1, n):
                if sub_df["close"].iloc[idx] <= pl:
                    breakdown_idx = idx
                    break

            # 2. 定義時間與振幅 S1, S2, S3
            t1 = max(1, p_h1.index - p_l0.index) # S1: L0 -> H1
            t2 = max(1, p_l.index - p_h1.index)  # S2: H1 -> L
            t3 = max(1, p_h2.index - p_l.index)  # S3: L -> H2
            
            a1 = ph1 - pl0      # S1 Amp
            a2 = ph1 - pl       # S2 Amp (neckline depth)
            a3 = ph2 - pl       # S3 Amp

            # 3. 推算對稱的 S4 (H2 -> D)
            # 理想振幅 A4 應接近 A1
            target_p4 = ph2 - a1
            
            # 視覺對稱限制：右側時間不超過中間長度的 1.5 倍
            symmetry_limit_idx = p_h2.index + int(top_dist * 1.2)
            
            if breakdown_idx is not None:
                d_index = min(latest_idx, breakdown_idx + 2, symmetry_limit_idx)
                # 尋找 breakdown 後最接近 target_p4 的點
                for idx in range(breakdown_idx, min(latest_idx, symmetry_limit_idx) + 1):
                    if sub_df["close"].iloc[idx] <= target_p4 * 1.05:
                        d_index = idx
                        break
            else:
                d_index = min(latest_idx, symmetry_limit_idx)
            
            d_price = float(sub_df["close"].iloc[d_index])
            t4 = max(1, d_index - p_h2.index)
            a4 = ph2 - d_price

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

            # 【硬性條件 3】：結束點價格必須低於最高點
            if d_price > max_top * 1.01:
                continue

            # ---- 計算評分 (Score 0 ~ 100) ----
            def calc_cv(data):
                mean = sum(data) / len(data)
                std = (sum([(x - mean)**2 for x in data]) / len(data))**0.5
                return std / mean if mean != 0 else 1.0

            time_cv = calc_cv(times)
            amp_cv = calc_cv(amps)

            # 1. 時間全對稱得分 (最高 15 分)
            score_time_sym = max(0.0, 15.0 * (1.0 - time_cv / 0.8))

            # 2. 振幅全對稱得分 (最高 15 分)
            score_amp_sym = max(0.0, 15.0 * (1.0 - amp_cv / 0.8))

            # 3. 雙頭平對齊得分 (最高 20 分)
            score_alignment = max(0.0, 20.0 * (1.0 - top_diff_pct / self.max_top_diff_pct))

            # 4. 頸線深度得分 (最高 10 分)
            depth_diff = abs(depth_pct - 0.12)
            score_depth = max(0.0, 10.0 * (1.0 - depth_diff / 0.15))

            # 5. 跌破力道得分 (最高 15 分)
            if latest_close <= pl * 1.003:
                score_breakdown = min(15.0, 7.5 + (pl - latest_close) / pl * 150.0)
            else:
                score_breakdown = max(0.0, 7.5 * (max_top - latest_close) / (max_top - pl))

            # 6. 近現性得分 (最高 25 分，第二頭 H2 距離當前 K 線越近分數越高)
            bars_from_present = (n - 1) - p_h2.index
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

            total_score = float(min(100.0, score_time_sym + score_amp_sym + score_alignment + score_depth + score_breakdown + score_recency))

            if total_score > best_score:
                best_score = total_score
                best_match = (p_l0, p_h1, p_l, p_h2, d_index, d_price, top_diff_pct, depth_pct, times, amps, time_cv, amp_cv)

        if not best_match or best_score < 40.0:
            return None

        p_l0, p_h1, p_l, p_h2, d_idx, d_price, t_diff, depth, times, amps, t_cv, a_cv = best_match
        ph1, pl, ph2 = p_h1.price, p_l.price, p_h2.price
        t1, t2, t3, t4 = times
        a1, a2, a3, a4 = amps

        # 構建 M 型幾何 4 條連線 (L0->H1, H1->L, L->H2, H2->D) 與頸線
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

        line_start = create_line(p_l0, p_h1, p_l0.price, ph1, "support")
        line_left_leg = create_line(p_h1, p_l, ph1, pl, "resistance")
        line_right_leg = create_line(p_l, p_h2, pl, ph2, "support")
        
        d_date = str(sub_df["date"].iloc[d_idx])
        line_drop = create_line(p_h2, PivotPoint(d_idx, d_date, d_price, "trough"), ph2, d_price, "resistance")
        
        # 頸線水平線 (用 support 代表支撐位)
        line_neckline = create_line(p_h1, PivotPoint(d_idx, d_date, pl, "trough"), pl, pl, "support")

        # 突破/跌破狀態判定
        if latest_close <= p_l.price * 0.997:
            breakout_status = "breakout_down"
        else:
            breakout_status = "inside"

        return PatternResult(
            stock_id=stock_id,
            pattern_type="m_top",
            sub_type="double_top",
            timeframe=timeframe,
            score=best_score,
            date=latest_date,
            pivots=[p_l0, p_h1, p_l, p_h2],
            lines=[line_start, line_left_leg, line_right_leg, line_drop, line_neckline],
            details={
                "breakout_status": breakout_status,
                "price_L0": round(float(p_l0.price), 2),
                "price_H1": round(float(p_h1.price), 2),
                "price_L": round(float(p_l.price), 2),
                "price_H2": round(float(p_h2.price), 2),
                "top_diff_pct": round(float(t_diff * 100.0), 2),
                "neckline_depth_pct": round(float(depth * 100.0), 2),
                "durations": [int(x) for x in times],
                "amplitudes": [round(float(x), 2) for x in amps],
                "time_cv": round(float(t_cv), 3),
                "amp_cv": round(float(a_cv), 3),
                "latest_close": latest_close,
            },
        )
