"""
杯柄型態 (Cup and Handle) 型態檢測器

幾何結構與硬性條件：
1. Cup Start (P1): 杯身起始高點
2. Cup Bottom (T): 杯身底部低點 (應呈現 U 型而非 V 型)
3. Cup End / Handle Start (P2): 杯身結束高點 (杯緣)
4. Handle Bottom (T_handle): 柄部回測低點
5. U型底判定: 底部價格區間停留時間佔比
6. 柄部深度: 回測幅度不應超過杯身深度的 50%
7. 柄部長度: 柄部時間應顯著短於杯身時間
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


class CupHandleDetector(BasePatternDetector):
    """杯柄型態 (Cup and Handle) 型態檢測器"""

    def __init__(
        self,
        window: int = 3,
        min_u_shape_fullness: float = 0.20,  # 底部 20% 價格區間停留時間佔比 (區分 U/V)
        max_handle_retracement: float = 0.50, # 柄部最大回測比例 (相對於杯身深度)
        max_rim_diff_pct: float = 0.08,      # 杯緣兩端最大價差比例 (P1 vs P2)
        min_candles: int = 40,
        max_candles: int = 200,
    ):
        super().__init__(name="cup_handle", display_name="杯柄型態")
        self.window = window
        self.min_u_shape_fullness = min_u_shape_fullness
        self.max_handle_retracement = max_handle_retracement
        self.max_rim_diff_pct = max_rim_diff_pct
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
        """對單一股票執行杯柄型態檢測"""
        if df.empty or len(df) < self.min_candles:
            return None

        sub_df = df.iloc[-self.max_candles :].reset_index(drop=True)
        n = len(sub_df)
        pivots = self._extract_pivots(sub_df)

        if len(pivots) < 4:
            return None

        latest_close = float(sub_df["close"].iloc[-1])
        latest_date = str(sub_df["date"].iloc[-1])

        best_match = None
        best_score = -1.0

        # 檢視組合：P1 (Peak) -> T (Trough) -> P2 (Peak) -> T_handle (Trough)
        for i in range(len(pivots) - 3):
            p_p1 = pivots[i]
            p_t  = pivots[i + 1]
            p_p2 = pivots[i + 2]
            p_th = pivots[i + 3]

            if not (p_p1.type == "peak" and p_t.type == "trough" and p_p2.type == "peak" and p_th.type == "trough"):
                continue

            pp1, pt, pp2, pth = p_p1.price, p_t.price, p_p2.price, p_th.price

            # 【硬性條件 1】：杯緣兩端價格相近
            rim_diff = abs(pp1 - pp2) / min(pp1, pp2)
            if rim_diff > self.max_rim_diff_pct:
                continue

            # 【硬性條件 2】：杯身深度
            cup_depth = max(pp1, pp2) - pt
            if cup_depth <= 0:
                continue

            # 【硬性條件 3】：柄部回測深度 (不能超過杯身深度的 50%)
            handle_retracement = (pp2 - pth) / cup_depth
            if not (0.05 < handle_retracement < self.max_handle_retracement):
                continue

            # 【硬性條件 4】：柄部長度 (應短於杯身)
            cup_duration = p_p2.index - p_p1.index
            handle_duration = p_th.index - p_p2.index
            if handle_duration >= cup_duration * 0.8 or handle_duration < 3:
                continue

            # 【硬性條件 5】：U型底判定 (底部停留時間)
            # 統計杯身區間 [P1.idx, P2.idx] 內，價格在底部 20% 區間的 K 線比例
            cup_prices = sub_df["low"].iloc[p_p1.index : p_p2.index + 1].values
            bottom_threshold = pt + cup_depth * 0.2
            bottom_bars = np.sum(cup_prices <= bottom_threshold)
            u_fullness = bottom_bars / len(cup_prices)
            if u_fullness < self.min_u_shape_fullness:
                continue

            # ---- 決定右側繪圖結束點 D ----
            # 尋找 T_handle 之後第一個收盤價突破 P2 的位置
            breakout_idx = None
            for idx in range(p_th.index + 1, n):
                if sub_df["close"].iloc[idx] >= pp2:
                    breakout_idx = idx
                    break

            # 視覺對稱限制：柄部後段不超過柄部整理時間的 1.5 倍
            symmetry_limit_idx = p_th.index + int(handle_duration * 1.5)
            
            if breakout_idx is not None:
                d_index = min(n - 1, breakout_idx + 2, symmetry_limit_idx)
            else:
                d_index = min(n - 1, symmetry_limit_idx)
            
            d_price = float(sub_df["close"].iloc[d_index])

            # ---- 計算評分 (Score 0 ~ 100) ----
            # 1. U型底圓潤得分 (最高 30 分)
            score_u = min(30.0, u_fullness * 100.0)
            
            # 2. 柄部回測優質得分 (最高 25 分): 最佳回測為杯身深度的 1/3
            handle_diff = abs(handle_retracement - 0.33)
            score_handle = max(0.0, 25.0 * (1.0 - handle_diff / 0.4))
            
            # 3. 杯緣對稱得分 (最高 15 分)
            score_rim = max(0.0, 15.0 * (1.0 - rim_diff / self.max_rim_diff_pct))
            
            # 4. 突破力道得分 (最高 30 分)
            if latest_close >= pp2 * 0.99:
                score_breakout = min(30.0, 15.0 + (latest_close - pp2) / pp2 * 300.0)
            else:
                score_breakout = max(0.0, 15.0 * (latest_close - pth) / (pp2 - pth))

            total_score = float(min(100.0, score_u + score_handle + score_rim + score_breakout))

            if total_score > best_score:
                best_score = total_score
                best_match = (p_p1, p_t, p_p2, p_th, d_index, d_price, u_fullness, handle_retracement, rim_diff)

        if not best_match or best_score < 40.0:
            return None

        p_p1, p_t, p_p2, p_th, d_idx, d_price, u_f, h_r, r_d = best_match
        pp1, pt, pp2, pth = p_p1.price, p_t.price, p_p2.price, p_th.price

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

        line_cup_left = create_line(p_p1, p_t, pp1, pt, "resistance")
        line_cup_right = create_line(p_t, p_p2, pt, pp2, "support")
        line_handle_down = create_line(p_p2, p_th, pp2, pth, "resistance")
        
        # 結束點 D 點
        d_date = str(sub_df["date"].iloc[d_idx])
        line_handle_up = create_line(p_th, PivotPoint(d_idx, d_date, d_price, "peak"), pth, d_price, "support")
        
        # 杯緣水平線
        line_rim = create_line(p_p1, p_p2, pp1, pp2, "resistance")

        if latest_close >= pp2 * 1.003:
            breakout_status = "breakout_up"
        else:
            breakout_status = "inside"

        return PatternResult(
            stock_id=stock_id,
            pattern_type="cup_handle",
            sub_type="standard_cup_handle",
            timeframe=timeframe,
            score=best_score,
            date=latest_date,
            pivots=[p_p1, p_t, p_p2, p_th],
            lines=[line_cup_left, line_cup_right, line_handle_down, line_handle_up, line_rim],
            details={
                "breakout_status": breakout_status,
                "u_fullness": round(float(u_f), 3),
                "handle_retracement": round(float(h_r), 3),
                "rim_diff_pct": round(float(r_d * 100.0), 2),
                "cup_duration": int(p_p2.index - p_p1.index),
                "handle_duration": int(p_th.index - p_p2.index),
                "latest_close": latest_close,
            },
        )
