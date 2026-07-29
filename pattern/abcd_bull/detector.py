"""
ABCD 上漲型態檢測器 (Bullish ABCD Pattern Detector)

幾何結構與硬性條件：
1. A 點 (Trough 起漲低點): AB 浪開端
2. B 點 (Peak 第一波高點): PB > PA
3. C 點 (Trough 修正拉回點): PC > PA 且 PC < PB (硬性條件：拉回不能跌破起漲點 A)
4. D 點 (Peak 第二波目標/現價點): PD > PB 且 PD > PC (硬性條件：第二波必須突破前高 B)
5. 時間對稱性 (Time Symmetry): CD 浪 K 線根數與 AB 浪 K 線根數比例要在 0.4 ~ 2.5 之間

黃金比例 (Fibonacci Criteria):
- BC 拉回比例: (PB - PC) / (PB - PA) 介於 0.382 ~ 0.786 之間 (最佳為 0.618)
- CD 浪上漲幅程: (PD - PC) / (PB - PA) 介於 0.8 ~ 1.618 之間 (通常 AB ≈ CD)
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


class AbcdBullDetector(BasePatternDetector):
    """ABCD 上漲型態檢測器"""

    def __init__(
        self,
        window: int = 3,            # Pivot 點提取視窗大小
        min_candles: int = 20,      # 型態最少 K 線根數
        max_candles: int = 120,     # 型態最多 K 線根數
    ):
        super().__init__(name="abcd_bull", display_name="ABCD 上漲")
        self.window = window
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
        """對單一股票執行 ABCD 上漲型態檢測"""
        if df.empty or len(df) < self.min_candles:
            return None

        sub_df = df.iloc[-self.max_candles :].reset_index(drop=True)
        n = len(sub_df)
        if n < self.min_candles:
            return None

        pivots = self._extract_pivots(sub_df)
        if len(pivots) < 3:
            return None

        # 尋找最近一組符合 A(Trough) -> B(Peak) -> C(Trough) 的組合，再加上最新一根 K 線或 Pivot 作為 D
        best_match = None
        best_score = -1.0

        latest_idx = n - 1
        latest_close = float(sub_df["close"].iloc[-1])
        latest_high = float(sub_df["high"].iloc[-1])

        # 逐一檢視四點組合
        for i in range(len(pivots) - 2):
            p_a = pivots[i]
            p_b = pivots[i + 1]
            p_c = pivots[i + 2]

            # 檢查類型順序：A 必須為 trough, B 為 peak, C 為 trough
            if not (p_a.type == "trough" and p_b.type == "peak" and p_c.type == "trough"):
                continue

            pa, pb, pc = p_a.price, p_b.price, p_c.price

            # 【硬性條件 1】：B 高於 A，且 C 介於 A 與 B 之間 (低點 C 絕對不能跌破 A)
            if not (pb > pa and pc > pa and pc < pb):
                continue

            ab_height = pb - pa
            if ab_height <= 0:
                continue

            bc_retrace_ratio = (pb - pc) / ab_height

            # 【硬性條件 2】：BC 修正拉回比例要在 0.35 ~ 0.85 之間
            if not (0.35 <= bc_retrace_ratio <= 0.85):
                continue

            # 尋找 D 點：D 點可以是後續的某個 Peak，或者最新這根 K 線的最高價
            d_candidates = []
            if len(pivots) >= i + 4:
                p_d_pivot = pivots[i + 3]
                if p_d_pivot.type == "peak":
                    d_candidates.append(p_d_pivot)

            # 最新 K 線作為潛在的 D 點
            d_latest = PivotPoint(index=latest_idx, date=str(sub_df["date"].iloc[-1]), price=latest_high, type="peak")
            d_candidates.append(d_latest)

            for p_d in d_candidates:
                pd_price = p_d.price

                # 【硬性條件 3】：D 必須高於前高 B，且 D 必須高於 C
                if not (pd_price > pb and pd_price > pc):
                    continue

                cd_height = pd_price - pc
                cd_ratio = cd_height / ab_height

                # 【硬性條件 4】：CD 浪幅程比例要在 0.75 ~ 1.8 之間
                if not (0.75 <= cd_ratio <= 1.8):
                    continue

                # 【硬性條件 5】：時間對稱性 constraint - CD 浪與 AB 浪 K 線根數比例要在 0.4 ~ 2.5 之間
                time_ab_bars = max(1, int(p_b.index - p_a.index))
                time_cd_bars = max(1, int(p_d.index - p_c.index))
                time_ratio = float(time_cd_bars / time_ab_bars)

                if not (0.4 <= time_ratio <= 2.5):
                    continue

                # ---- 計算評分 (Score 0 ~ 100) ----
                # 1. BC 黃金比例貼合度 (最高 30 分，越接近 0.618 分數越高)
                bc_diff = abs(bc_retrace_ratio - 0.618)
                score_bc = max(0.0, 30.0 * (1.0 - bc_diff / 0.3))

                # 2. CD 展幅 AB=CD 貼合度 (最高 30 分，越接近 1.0 或 1.272 分數越高)
                cd_diff = min(abs(cd_ratio - 1.0), abs(cd_ratio - 1.272))
                score_cd = max(0.0, 30.0 * (1.0 - cd_diff / 0.5))

                # 3. 時間對稱性貼合度 (最高 20 分，越接近 1.0 或 0.618 / 1.272 分數越高)
                time_diff = min(abs(time_ratio - 1.0), abs(time_ratio - 1.272), abs(time_ratio - 0.618))
                score_time = max(0.0, 20.0 * (1.0 - time_diff / 0.8))

                # 4. 突破力道 (最高 20 分): 最新收盤價超過 B 點的比例
                breakout_margin = (latest_close - pb) / pb
                if breakout_margin > 0:
                    score_breakout = min(20.0, 10.0 + breakout_margin * 200.0)
                else:
                    score_breakout = 5.0

                total_score = float(min(100.0, score_bc + score_cd + score_time + score_breakout))

                if total_score > best_score:
                    best_score = total_score
                    best_match = (p_a, p_b, p_c, p_d, bc_retrace_ratio, cd_ratio, time_ab_bars, time_cd_bars, time_ratio)

        if not best_match or best_score < 40.0:
            return None

        p_a, p_b, p_c, p_d, bc_ratio, cd_ratio, ab_bars, cd_bars, t_ratio = best_match

        # 構建 3 條連線線段 (AB, BC, CD)
        dx_ab = max(1, p_b.index - p_a.index)
        s_ab = float((p_b.price - p_a.price) / dx_ab)
        icpt_ab = float(p_a.price - s_ab * p_a.index)

        dx_bc = max(1, p_c.index - p_b.index)
        s_bc = float((p_c.price - p_b.price) / dx_bc)
        icpt_bc = float(p_b.price - s_bc * p_b.index)

        dx_cd = max(1, p_d.index - p_c.index)
        s_cd = float((p_d.price - p_c.price) / dx_cd)
        icpt_cd = float(p_c.price - s_cd * p_c.index)

        line_ab = TrendLine(
            start_index=int(p_a.index),
            end_index=int(p_b.index),
            start_date=str(p_a.date),
            end_date=str(p_b.date),
            start_price=float(p_a.price),
            end_price=float(p_b.price),
            slope=s_ab,
            intercept=icpt_ab,
            r_squared=0.95,
            line_type="support",  # 上升段 AB
        )

        line_bc = TrendLine(
            start_index=int(p_b.index),
            end_index=int(p_c.index),
            start_date=str(p_b.date),
            end_date=str(p_c.date),
            start_price=float(p_b.price),
            end_price=float(p_c.price),
            slope=s_bc,
            intercept=icpt_bc,
            r_squared=0.95,
            line_type="resistance",  # 拉回段 BC
        )

        line_cd = TrendLine(
            start_index=int(p_c.index),
            end_index=int(p_d.index),
            start_date=str(p_c.date),
            end_date=str(p_d.date),
            start_price=float(p_c.price),
            end_price=float(p_d.price),
            slope=s_cd,
            intercept=icpt_cd,
            r_squared=0.95,
            line_type="support",  # 攻擊段 CD
        )

        # 突破狀態
        if latest_close > p_b.price * 1.003:
            breakout_status = "breakout_up"
        else:
            breakout_status = "inside"

        return PatternResult(
            stock_id=stock_id,
            pattern_type="abcd_bull",
            sub_type="bullish_abcd",
            timeframe=timeframe,
            score=best_score,
            date=str(sub_df["date"].iloc[-1]),
            pivots=[p_a, p_b, p_c, p_d],
            lines=[line_ab, line_bc, line_cd],
            details={
                "breakout_status": breakout_status,
                "price_A": round(float(p_a.price), 2),
                "price_B": round(float(p_b.price), 2),
                "price_C": round(float(p_c.price), 2),
                "price_D": round(float(p_d.price), 2),
                "bc_retrace_ratio": round(float(bc_ratio), 3),
                "cd_expansion_ratio": round(float(cd_ratio), 3),
                "time_ab_bars": int(ab_bars),
                "time_cd_bars": int(cd_bars),
                "time_ratio": round(float(t_ratio), 3),
                "latest_close": latest_close,
            },
        )
