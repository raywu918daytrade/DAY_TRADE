"""
價格行為型態 (突破/跌破回測) 與 Volume Profile POC (Point of Control) 重疊度全市場分析工具

功能：
1. 載入全市場日 K 線數據，執行 BreakoutRetestDetector (突破壓力回測) 與 BreakdownRetestDetector (跌破支撐反彈) 型態識別。
2. 對比 db/poc_day/ 籌碼密集成交 POC 價位，統計分析技術型態的關鍵壓力/支撐價格與 POC 的距離差距 (Min Difference %)。
3. 輸出全市場 POC 重疊率 (Confluence Rate) 統計數據與個股對照表。

用法：
    python -m pattern.experiments.verify_pattern_poc_overlap                       # 檢測全型態 (breakout + breakdown)
    python -m pattern.experiments.verify_pattern_poc_overlap --pattern breakout_retest  # 僅檢測突破壓力回測 (做多)
    python -m pattern.experiments.verify_pattern_poc_overlap --pattern breakdown_retest # 僅檢測跌破支撐反彈 (做空)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from data.query import load_poc
from pattern.breakdown_retest.detector import BreakdownRetestDetector
from pattern.breakout_retest.detector import BreakoutRetestDetector
from pattern.data_loader import get_all_stocks_candles


def analyze_overlap(pattern_type: str = "all", min_score: float = 60.0):
    t0 = time.time()
    print("正在載入全市場日 K 線與 db/poc_day/ 數據...", flush=True)

    candles = get_all_stocks_candles("day", limit=120)
    pocs_df = load_poc()

    if pocs_df.empty:
        print("錯誤: db/poc_day/ 中無資料，請先執行 python -m data.build_poc")
        return

    pocs_df["date_str"] = pocs_df["date"].astype(str)
    poc_stocks = set(pocs_df["stock_id"].unique())

    detectors = {}
    if pattern_type in ("breakout_retest", "all"):
        detectors["breakout_retest"] = BreakoutRetestDetector()
    if pattern_type in ("breakdown_retest", "all"):
        detectors["breakdown_retest"] = BreakdownRetestDetector()

    for p_name, detector in detectors.items():
        print(f"\n==================================================")
        print(f"  開始分析型態: {detector.display_name} ({p_name})")
        print(f"==================================================", flush=True)

        matches = []
        for stock_id, df in candles.items():
            if stock_id in poc_stocks:
                res = detector.detect(df, stock_id, "day")
                if res and res.score >= min_score:
                    matches.append(res)

        print(f"相符個股數量 (有 POC 籌碼記錄): {len(matches)} 支", flush=True)
        if not matches:
            continue

        records = []
        for res in matches:
            stock_id = res.stock_id
            latest_date = str(res.date)[:10]

            if p_name == "breakout_retest":
                level_price = res.details["resistance_price"]
                level_name = "壓力價"
            else:
                level_price = res.details["support_price"]
                level_name = "支撐價"

            p1_date = str(res.pivots[0].date)[:10]
            p2_date = str(res.pivots[2].date)[:10]
            retest_date = str(res.pivots[4].date)[:10]

            # 檢索型態關鍵日 (P1, P2, Retest, 最新日) 的 POC
            sub_pocs = pocs_df[
                (pocs_df["stock_id"] == stock_id)
                & (pocs_df["date_str"].isin([p1_date, p2_date, retest_date, latest_date]))
            ]

            if sub_pocs.empty:
                continue

            all_p_list = []
            for p_str in sub_pocs["pocs"]:
                all_p_list.extend([float(p) for p in str(p_str).split(",") if p])

            if not all_p_list:
                continue

            # 找出距離 level_price 最近的 POC 價位
            best_poc = min(all_p_list, key=lambda p: abs(level_price - p))
            diff_pct = (abs(level_price - best_poc) / level_price) * 100.0

            records.append(
                {
                    "stock_id": stock_id,
                    "date": latest_date,
                    "score": res.score,
                    "level_price": level_price,
                    "matched_poc": best_poc,
                    "diff_pct": round(diff_pct, 2),
                    "is_confluence_2pct": diff_pct <= 2.0,
                    "is_confluence_1pct": diff_pct <= 1.0,
                }
            )

        df_rec = pd.DataFrame(records)
        if df_rec.empty:
            print("無匹配數據。")
            continue

        df_rec.sort_values("score", ascending=False, inplace=True)

        n_total = len(df_rec)
        n_1pct = int((df_rec["diff_pct"] <= 1.0).sum())
        n_2pct = int((df_rec["diff_pct"] <= 2.0).sum())
        n_3pct = int((df_rec["diff_pct"] <= 3.0).sum())
        avg_diff = df_rec["diff_pct"].mean()

        print(f"\n--- {detector.display_name} POC 重疊度統計報告 ---")
        print(f"有效比對樣本數: {n_total} 筆")
        print(f"平均價位差距比: {avg_diff:.2f}%")
        print(f"POC 重疊率 (距離 <= 1%): {n_1pct} / {n_total} ({n_1pct / n_total * 100:.1f}%)")
        print(f"POC 重疊率 (距離 <= 2%): {n_2pct} / {n_total} ({n_2pct / n_total * 100:.1f}%) [雙重共振]")
        print(f"POC 重疊率 (距離 <= 3%): {n_3pct} / {n_total} ({n_3pct / n_total * 100:.1f}%)")

        print(f"\n前 15 支高分匹配個股與 POC 距離對照表:")
        print(
            f"{'股票代號':<8} {'日期':<12} {'型態分數':<8} {level_name:<10} {'最接近POC':<10} {'距離差距(%)':<10} {'2%內共振':<8}"
        )
        print("-" * 75)
        for _, row in df_rec.head(15).iterrows():
            conf_str = "✅ YES" if row["is_confluence_2pct"] else "  NO"
            print(
                f"{row['stock_id']:<8} {row['date']:<12} {row['score']:<8.1f} {row['level_price']:<10.2f} {row['matched_poc']:<10.2f} {row['diff_pct']:<10.2f}% {conf_str:<8}"
            )

    print(f"\n全市場驗證完畢，總耗時 {time.time()-t0:.2f}s ✅", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="驗證價格行為型態與 POC 重疊度")
    parser.add_argument(
        "--pattern",
        type=str,
        default="all",
        choices=["breakout_retest", "breakdown_retest", "all"],
        help="指定分析型態",
    )
    parser.add_argument("--min_score", type=float, default=60.0, help="最低型態分數門檻 (預設 60.0)")
    args = parser.parse_args()

    analyze_overlap(pattern_type=args.pattern, min_score=args.min_score)
