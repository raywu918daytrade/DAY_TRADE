"""
突破壓力回測 / 支撐壓力轉換 (Breakout & Retest / Role Reversal) 型態檢測器

價格行為學 (Price Action) 邏輯：
1. 階段 1 (橫向壓力形成): 在 K 線波段中尋求至少 2 個高點 (P1, P2)，其價格高點極為接近 (差距在可接受誤差內)，形成水平關鍵壓力線 R。兩高點中間有適當拉回谷底。
2. 階段 2 (關鍵壓力突破): 價格隨後衝高突破壓力線 R (突破幅度 > min_breakout_pct)。
3. 階段 3 (拉回原壓力變支撐/靜待方向): 價格回測拉回接近原壓力線 R，且最低價守在 R 附近不跌破，呈現原壓力轉換為支撐，最新價格處於支撐位附近整理靜待方向。
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from pattern.base import BasePatternDetector, PatternResult, PivotPoint, TrendLine


class BreakoutRetestDetector(BasePatternDetector):
    """突破壓力回測 / 支撐壓力轉換 (Breakout & Retest) 型態檢測器"""

    def __init__(
        self,
        window: int = 3,                       # Pivot 點提取滾動視窗大小
        max_resistance_diff_pct: float = 0.02, # 兩壓力點最大允許價差比例 (預設 2%)
        min_peak_distance: int = 4,            # 兩壓力點之間最少 K 線根數
        min_mid_pullback_pct: float = 0.015,   # 兩壓力高點中間的最少拉回幅度 (預設 1.5%)
        min_breakout_pct: float = 0.010,       # 突破壓力線的最小衝高幅度 (預設 1.0%)
        max_retest_dist_pct: float = 0.025,    # 回測低點距離壓力線的最大上方差距 (預設 2.5%)
        max_breakdown_pct: float = 0.015,      # 回測允許跌破壓力線的最大下方深度 (預設 1.5%)
        min_candles: int = 25,                 # 型態最少 K 線根數
        max_candles: int = 120,                # 型態最多 K 線根數
    ):
        super().__init__(name="breakout_retest", display_name="突破壓力回測")
        self.window = window
        self.max_resistance_diff_pct = max_resistance_diff_pct
        self.min_peak_distance = min_peak_distance
        self.min_mid_pullback_pct = min_mid_pullback_pct
        self.min_breakout_pct = min_breakout_pct
        self.max_retest_dist_pct = max_retest_dist_pct
        self.max_breakdown_pct = max_breakdown_pct
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

    def detect(
        self,
        df: pd.DataFrame,
        stock_id: str,
        timeframe: str,
        poc_df: Optional[pd.DataFrame] = None,
    ) -> Optional[PatternResult]:
        """對單一股票執行突破壓力回測型態檢測

        poc_df: 選填，外部預先批次載入的 POC 資料（避免每支股票各自查一次）。
        呼叫端要自己先用 data.query.load_poc_adjusted() 載入，不要傳未調整過的
        load_poc() 結果進來，否則會跟 df（K線，已還原權息）基準對不上。"""
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

        # 搜尋組合：尋找 Peak1 (P1) 與 Peak2 (P2)
        peaks = [p for p in pivots if p.type == "peak"]
        if len(peaks) < 2:
            return None

        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                p1 = peaks[i]
                p2 = peaks[j]

                # 【條件 1】：兩壓力點間距
                dist = p2.index - p1.index
                if dist < self.min_peak_distance:
                    continue

                # 【條件 2】：兩高點價格接近 (橫向壓力線)
                r_diff = abs(p1.price - p2.price) / min(p1.price, p2.price)
                if r_diff > self.max_resistance_diff_pct:
                    continue

                r_level = (p1.price + p2.price) / 2.0  # 橫向壓力價格

                # 【條件 3】：兩高點中間有適當拉回谷底 (中間最深低點 T_mid)
                mid_lows = lows[p1.index : p2.index + 1]
                mid_trough_price = float(np.min(mid_lows))
                mid_pullback = (r_level - mid_trough_price) / r_level
                if mid_pullback < self.min_mid_pullback_pct:
                    continue

                # Find mid trough pivot or index
                mid_trough_idx = int(p1.index + np.argmin(mid_lows))
                p_mid_trough = PivotPoint(
                    index=mid_trough_idx,
                    date=dates[mid_trough_idx],
                    price=mid_trough_price,
                    type="trough",
                )

                # 【條件 4】：階段 2 - P2 之後發生突破 (Breakout)
                post_p2_highs = highs[p2.index + 1 :]
                if len(post_p2_highs) < 3:
                    continue  # P2 後面太少 K 線，尚未完成突破與回測

                breakout_high = float(np.max(post_p2_highs))
                breakout_rel = (breakout_high - r_level) / r_level
                if breakout_rel < self.min_breakout_pct:
                    continue  # 衝高幅度不夠，未形成顯著突破

                breakout_idx = int(p2.index + 1 + np.argmax(post_p2_highs))
                p_breakout = PivotPoint(
                    index=breakout_idx,
                    date=dates[breakout_idx],
                    price=breakout_high,
                    type="peak",
                )

                # 【條件 5】：階段 3 - 衝高突破後出現拉回 (Retest)
                if breakout_idx >= n - 1:
                    continue  # 突破點就在最新一根，尚無拉回過程

                post_bo_lows = lows[breakout_idx :]
                retest_low = float(np.min(post_bo_lows))
                retest_low_idx = int(breakout_idx + np.argmin(post_bo_lows))

                p_retest_low = PivotPoint(
                    index=retest_low_idx,
                    date=dates[retest_low_idx],
                    price=retest_low,
                    type="trough",
                )

                # 回測低點必須貼近壓力線 R (轉為支撐)：
                # - 不能過高 (沒有回到壓力線附近)
                # - 不能過低 (跌破壓力線太深，形成假突破)
                retest_upper_bound = r_level * (1.0 + self.max_retest_dist_pct)
                retest_lower_bound = r_level * (1.0 - self.max_breakdown_pct)

                if not (retest_lower_bound <= retest_low <= retest_upper_bound):
                    continue

                # 最新收盤價也要守在支撐位附近 (不深跌)
                if latest_close < retest_lower_bound or latest_close > r_level * 1.06:
                    continue

                # --- 品質與近現性評分 (Score 0~100) ---
                # 1. 壓力對齊精準度 (25分): 兩壓力點高點越相近越高分
                score_align = max(0.0, 1.0 - (r_diff / self.max_resistance_diff_pct)) * 25.0

                # 2. 突破強度分數 (25分): 突破力道越強越高分
                score_breakout = min(1.0, breakout_rel / (self.min_breakout_pct * 3.0)) * 25.0

                # 3. 回測支撐精準度 (25分): 回測點越精準觸及 R 線越好
                retest_touch_diff = abs(retest_low - r_level) / r_level
                score_retest = max(0.0, 1.0 - (retest_touch_diff / self.max_retest_dist_pct)) * 25.0

                # 4. 近現性分數 (25分): 回測與當前 K 線越靠近最新日期越高分
                recency_ratio = (retest_low_idx + 1) / n
                score_recency = recency_ratio * 25.0

                total_score = round(score_align + score_breakout + score_retest + score_recency, 2)

                if total_score > best_score:
                    best_score = total_score
                    best_match = {
                        "p1": p1,
                        "p_mid_trough": p_mid_trough,
                        "p2": p2,
                        "p_breakout": p_breakout,
                        "p_retest_low": p_retest_low,
                        "r_level": r_level,
                        "breakout_high": breakout_high,
                        "retest_low": retest_low,
                        "score": total_score,
                    }

        if best_match is None or best_score < 50.0:
            return None

        bm = best_match
        r_level = bm["r_level"]

        # --- 檢測與 Volume Profile POC 的重疊共振 (Confluence) ---
        matched_poc = None
        poc_diff_pct = None
        poc_confluence = False

        try:
            if poc_df is None:
                from data.query import load_poc_adjusted
                stock_poc_df = load_poc_adjusted(stock_id=stock_id)
            else:
                stock_poc_df = poc_df[poc_df["stock_id"] == stock_id]

            if not stock_poc_df.empty:
                key_dates = {
                    str(bm["p1"].date)[:10],
                    str(bm["p2"].date)[:10],
                    str(bm["p_retest_low"].date)[:10],
                    str(latest_date)[:10],
                }
                sub_pocs = stock_poc_df[stock_poc_df["date"].astype(str).str[:10].isin(key_dates)]
                if not sub_pocs.empty:
                    candidate_pocs = []
                    for p_str in sub_pocs["pocs"]:
                        candidate_pocs.extend([float(p) for p in str(p_str).split(",") if p])
                    if candidate_pocs:
                        best_poc = min(candidate_pocs, key=lambda p: abs(r_level - p))
                        diff_pct = (abs(r_level - best_poc) / r_level) * 100.0
                        matched_poc = round(float(best_poc), 2)
                        poc_diff_pct = round(float(diff_pct), 2)
                        if diff_pct <= 2.0:
                            poc_confluence = True
                            # POC 雙重共振獎勵加分 (+5.0分，上限 100分)
                            best_score = min(100.0, round(best_score + 5.0, 2))
        except Exception:
            pass

        # 建立趨勢線 (水平壓力/支撐線 R)
        line = TrendLine(
            start_index=bm["p1"].index,
            end_index=n - 1,
            start_date=bm["p1"].date,
            end_date=latest_date,
            start_price=r_level,
            end_price=r_level,
            slope=0.0,
            intercept=r_level,
            r_squared=1.0,
            line_type="resistance",
        )

        pivots_list = [
            bm["p1"],
            bm["p_mid_trough"],
            bm["p2"],
            bm["p_breakout"],
            bm["p_retest_low"],
        ]

        details = {
            "resistance_price": round(float(r_level), 2),
            "breakout_high": round(float(bm["breakout_high"]), 2),
            "retest_low": round(float(bm["retest_low"]), 2),
            "latest_close": round(latest_close, 2),
            "resistance_diff_pct": round(float(abs(bm["p1"].price - bm["p2"].price) / min(bm["p1"].price, bm["p2"].price) * 100.0), 2),
            "matched_poc": matched_poc,
            "poc_diff_pct": poc_diff_pct,
            "poc_confluence": poc_confluence,
        }

        return PatternResult(
            stock_id=str(stock_id),
            pattern_type="breakout_retest",
            sub_type="resistance_to_support",
            timeframe=str(timeframe),
            score=best_score,
            date=latest_date,
            pivots=pivots_list,
            lines=[line],
            details=details,
        )
