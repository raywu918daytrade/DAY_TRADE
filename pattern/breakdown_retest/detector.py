"""
跌破支撐反彈 / 支撐轉壓力 (Breakdown & Retest / Role Reversal) 做空型態檢測器

價格行為學 (Price Action) 做空邏輯：
1. 階段 1 (橫向支撐形成): 在 K 線波段中尋求至少 2 個低點 (T1, T2)，其最低價極為接近 (差距在可接受誤差內)，形成水平關鍵支撐線 S。兩低點中間有適當反彈高點。
2. 階段 2 (關鍵支撐跌破): 價格隨後殺低跌破支撐線 S (跌破幅度 > min_breakdown_pct)。
3. 階段 3 (反彈原支撐變壓力/靜待方向): 價格弱勢反彈貼近原支撐線 S，且最高價被 S 壓制未向上實質突破，呈現原支撐轉換為壓力，最新價格處於壓力位下方整理靜待方向。
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


class BreakdownRetestDetector(BasePatternDetector):
    """跌破支撐反彈 / 支撐轉壓力 (Breakdown & Retest) 做空型態檢測器"""

    def __init__(
        self,
        window: int = 3,                       # Pivot 點提取滾動視窗大小
        max_support_diff_pct: float = 0.02,    # 兩支撐點最大允許價差比例 (預設 2%)
        min_trough_distance: int = 4,          # 兩支撐點之間最少 K 線根數
        min_mid_bounce_pct: float = 0.015,     # 兩支撐低點中間的最少反彈幅度 (預設 1.5%)
        min_breakdown_pct: float = 0.010,      # 跌破支撐線的最小下探幅度 (預設 1.0%)
        max_retest_dist_pct: float = 0.025,    # 反彈高點距離支撐線的最大下方差距 (預設 2.5%)
        max_breakout_pct: float = 0.015,       # 反彈允許突破支撐線的最大上方深度 (預設 1.5%)
        min_candles: int = 25,                 # 型態最少 K 線根數
        max_candles: int = 120,                # 型態最多 K 線根數
    ):
        super().__init__(name="breakdown_retest", display_name="跌破支撐反彈")
        self.window = window
        self.max_support_diff_pct = max_support_diff_pct
        self.min_trough_distance = min_trough_distance
        self.min_mid_bounce_pct = min_mid_bounce_pct
        self.min_breakdown_pct = min_breakdown_pct
        self.max_retest_dist_pct = max_retest_dist_pct
        self.max_breakout_pct = max_breakout_pct
        self.min_candles = min_candles
        self.max_candles = max_candles

    def _extract_pivots(self, df: pd.DataFrame) -> List[PivotPoint]:
        """提取波段轉折點 (Peaks 與 Troughs)"""
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

        # 保持 Peak / Trough 交替
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
        """對單一股票執行跌破支撐反彈做空型態檢測"""
        if df.empty or len(df) < self.min_candles:
            return None

        sub_df = df.iloc[-self.max_candles :].reset_index(drop=True)
        n = len(sub_df)
        if n < self.min_candles:
            return None

        pivots = self._extract_pivots(sub_df)
        if len(pivots) < 3:
            return None

        highs = sub_df["high"].values
        lows = sub_df["low"].values
        closes = sub_df["close"].values
        dates = sub_df["date"].astype(str).values

        latest_close = float(closes[-1])
        latest_date = dates[-1]

        best_match = None
        best_score = -1.0

        # 搜尋組合：尋找 Trough1 (T1) 與 Trough2 (T2)
        troughs = [p for p in pivots if p.type == "trough"]
        if len(troughs) < 2:
            return None

        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                t1 = troughs[i]
                t2 = troughs[j]

                # 【條件 1】：兩支撐點間距
                dist = t2.index - t1.index
                if dist < self.min_trough_distance:
                    continue

                # 【條件 2】：兩低點價格接近 (橫向支撐線)
                s_diff = abs(t1.price - t2.price) / min(t1.price, t2.price)
                if s_diff > self.max_support_diff_pct:
                    continue

                s_level = (t1.price + t2.price) / 2.0  # 橫向支撐價格

                # 【條件 3】：兩低點中間有適當反彈高點 (中間最高高點 P_mid)
                mid_highs = highs[t1.index : t2.index + 1]
                mid_peak_price = float(np.max(mid_highs))
                mid_bounce = (mid_peak_price - s_level) / s_level
                if mid_bounce < self.min_mid_bounce_pct:
                    continue

                mid_peak_idx = int(t1.index + np.argmax(mid_highs))
                p_mid_peak = PivotPoint(
                    index=mid_peak_idx,
                    date=dates[mid_peak_idx],
                    price=mid_peak_price,
                    type="peak",
                )

                # 【條件 4】：階段 2 - T2 之後發生跌破 (Breakdown)
                post_t2_lows = lows[t2.index + 1 :]
                if len(post_t2_lows) < 3:
                    continue  # T2 後面太少 K 線，尚未完成跌破與反彈

                breakdown_low = float(np.min(post_t2_lows))
                breakdown_rel = (s_level - breakdown_low) / s_level
                if breakdown_rel < self.min_breakdown_pct:
                    continue  # 殺低幅度不夠，未形成顯著跌破

                breakdown_idx = int(t2.index + 1 + np.argmin(post_t2_lows))
                p_breakdown = PivotPoint(
                    index=breakdown_idx,
                    date=dates[breakdown_idx],
                    price=breakdown_low,
                    type="trough",
                )

                # 【條件 5】：階段 3 - 殺低跌破後出現反彈 (Retest / Bounce)
                if breakdown_idx >= n - 1:
                    continue  # 跌破點就在最新一根，尚無反彈過程

                post_bd_highs = highs[breakdown_idx :]
                retest_high = float(np.max(post_bd_highs))
                retest_high_idx = int(breakdown_idx + np.argmax(post_bd_highs))

                p_retest_high = PivotPoint(
                    index=retest_high_idx,
                    date=dates[retest_high_idx],
                    price=retest_high,
                    type="peak",
                )

                # 反彈高點必須貼近支撐線 S (轉為壓力)：
                # - 不能過低 (沒有反彈回到支撐線附近)
                # - 不能過高 (突破支撐線太高，形成假跌破)
                retest_upper_bound = s_level * (1.0 + self.max_breakout_pct)
                retest_lower_bound = s_level * (1.0 - self.max_retest_dist_pct)

                if not (retest_lower_bound <= retest_high <= retest_upper_bound):
                    continue

                # 最新收盤價也要守在壓力位下方 (不高刷)
                if latest_close > retest_upper_bound or latest_close < s_level * 0.94:
                    continue

                # --- 品質與近現性評分 (Score 0~100) ---
                # 1. 支撐對齊精準度 (25分): 兩支撐點低點越相近越高分
                score_align = max(0.0, 1.0 - (s_diff / self.max_support_diff_pct)) * 25.0

                # 2. 跌破強度分數 (25分): 殺低力道越強越高分
                score_breakdown = min(1.0, breakdown_rel / (self.min_breakdown_pct * 3.0)) * 25.0

                # 3. 反彈壓力精準度 (25分): 反彈高點越精準觸及 S 線越好
                retest_touch_diff = abs(retest_high - s_level) / s_level
                score_retest = max(0.0, 1.0 - (retest_touch_diff / self.max_retest_dist_pct)) * 25.0

                # 4. 近現性分數 (25分): 反彈與當前 K 線越靠近最新日期越高分
                recency_ratio = (retest_high_idx + 1) / n
                score_recency = recency_ratio * 25.0

                total_score = round(score_align + score_breakdown + score_retest + score_recency, 2)

                if total_score > best_score:
                    best_score = total_score
                    best_match = {
                        "t1": t1,
                        "p_mid_peak": p_mid_peak,
                        "t2": t2,
                        "p_breakdown": p_breakdown,
                        "p_retest_high": p_retest_high,
                        "s_level": s_level,
                        "breakdown_low": breakdown_low,
                        "retest_high": retest_high,
                        "score": total_score,
                    }

        if best_match is None or best_score < 50.0:
            return None

        bm = best_match
        s_level = bm["s_level"]

        # 建立趨勢線 (水平支撐/壓力線 S)
        line = TrendLine(
            start_index=bm["t1"].index,
            end_index=n - 1,
            start_date=bm["t1"].date,
            end_date=latest_date,
            start_price=s_level,
            end_price=s_level,
            slope=0.0,
            intercept=s_level,
            r_squared=1.0,
            line_type="support",
        )

        pivots_list = [
            bm["t1"],
            bm["p_mid_peak"],
            bm["t2"],
            bm["p_breakdown"],
            bm["p_retest_high"],
        ]

        details = {
            "support_price": round(float(s_level), 2),
            "breakdown_low": round(float(bm["breakdown_low"]), 2),
            "retest_high": round(float(bm["retest_high"]), 2),
            "latest_close": round(latest_close, 2),
            "support_diff_pct": round(float(abs(bm["t1"].price - bm["t2"].price) / min(bm["t1"].price, bm["t2"].price) * 100.0), 2),
        }

        return PatternResult(
            stock_id=str(stock_id),
            pattern_type="breakdown_retest",
            sub_type="support_to_resistance",
            timeframe=str(timeframe),
            score=best_score,
            date=latest_date,
            pivots=pivots_list,
            lines=[line],
            details=details,
        )
