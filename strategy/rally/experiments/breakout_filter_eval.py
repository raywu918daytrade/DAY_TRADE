"""
實驗：use_breakout_filter 這道硬過濾規則，到底值不值得留？

背景
    predict.py 的 predict_live() 預設 use_breakout_filter=True：模型算完機率
    之後，強制只保留 breakout_signal=True（先跌後漲）的樣本才出訊號，其餘
    直接剔除。這支檔案驗證這道硬規則到底有沒有實際幫助。

2026-07-09 用全天訓練+時間特徵的新版 XGB 模型跑過一次 breakout_filter_report()：

    全體基準              : 387,639 筆  勝率 35.6%
    僅模型 proba≥0.55     :  67,734 筆  勝率 60.3%
    僅 breakout=True      :  58,228 筆  勝率 34.1%（幾乎沒比基準高，單獨看沒用）
    breakout + proba≥0.55:   7,971 筆  勝率 62.4%

結論是取捨，不是單純有用/沒用：加這道過濾勝率有提升（60.3%→62.4%，+2.1個百分點），
但訊號量砍掉 88%（67,734→7,971）。要不要留，看策略想要的是「少量精準」還是
「多量周轉」，不是能用數據一次性斷定的問題——所以先放在 experiments/ 觀察，
不是核心 validate.py 的固定報表。

跟 strategy/rally 核心檔案的關係
    breakout_minute_report()／breakout_filter_report() 原本在 validate.py，
    2026-07-09 搬來這裡獨立，理由跟 experiments/breakout_specialist.py 一樣：
    這是還沒有定論、需要持續觀察/實驗的問題，不是「已經決定要長期用」的
    核心驗證報表（confidence_report/coverage_report/model_hour_confidence_report
    才是，那些留在 validate.py）。

用法
    python strategy/rally/experiments/breakout_filter_eval.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from strategy.rally.config import BREAKOUT_TRADE_END, BREAKOUT_TRADE_START
from strategy.rally.features import FEATURES, load_features
from strategy.rally.train import load_model


def breakout_filter_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    強過濾評估：以 breakout_signal（先跌後漲破底翻）作為「硬過濾」，
    觀察勝率變化。

    在 9:11 時，breakout_signal 比較的是：
         前一根 M5（9:01→9:06）報酬率 < 0  → 先跌
         當前 M5（9:06→9:11）報酬率 > 0  → 後漲
     即「前 2 根 m5 先跌、後漲」的破底翻型態。

     本函式驗證：若只對 breakout_signal=True 的樣本下注，
     勝率是否高於全體基準（這就是「強過濾」的效果）。
    """
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    total = len(test_df)
    base_win = test_df["target"].mean()

    thr = 0.55
    # 僅模型門檻（不含破底翻）—— 用來對照「49.4% 主要是誰貢獻的」
    p_only = test_df[test_df["proba"] >= thr]
    p_only_win = p_only["target"].mean() if len(p_only) else float("nan")

    # 強過濾：只保留 breakout_signal=True 的樣本
    filt = test_df[test_df["breakout_signal"]]
    filt_win = filt["target"].mean() if len(filt) else float("nan")

    # 強過濾 + 模型門檻 0.55（雙重確認）
    both = filt[filt["proba"] >= thr]
    both_win = both["target"].mean() if len(both) else float("nan")

    print("\n── 強過濾（breakout_signal：先跌後漲）評估 ──")
    print(f"  全體基準              : {total:>7,} 筆  勝率 {base_win*100:>5.1f}%")
    print(f"  僅模型 proba≥{thr}     : {len(p_only):>7,} 筆  勝率 {p_only_win*100:>5.1f}%")
    print(
        f"  僅 breakout=True      : {len(filt):>7,} 筆  勝率 {filt_win*100:>5.1f}%  (佔全體 {len(filt)/total*100:.1f}%)"
    )
    print(f"  breakout + proba≥{thr}: {len(both):>7,} 筆  勝率 {both_win*100:>5.1f}%")


def breakout_minute_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    threshold: float = 0.0,
):
    """
    強過濾破底翻：只看 breakout_signal=True 的樣本，
    逐分鐘顯示：推論數、平均信心度、勝率。

    交易時段限制為黃金窗口 9:14~9:30（BREAKOUT_TRADE_START ~ BREAKOUT_TRADE_END），
    這段破底翻勝率明顯高於 9:30 之後。

    9:14 是 breakout_signal 第一個有效分鐘（需 9:01→9:06 與 9:06→9:11 兩根 M5）。

    threshold: 信心度門檻（預設 0.0 = 不過濾，看全部破底翻樣本）。
        設成例如 0.45 時，逐分鐘表與彙總只統計 proba≥threshold 的樣本。
    """
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    # ── 只留強過濾破底翻 ──────────────────────────────────────────
    test_df = test_df[test_df["breakout_signal"]]

    # ── 黃金窗口 9:14 ~ 9:30 ─────────────────────────────────────
    sh, sm = BREAKOUT_TRADE_START
    eh, em = BREAKOUT_TRADE_END
    mask = (test_df["hour"] == sh) & (test_df["minute"] >= sm) & (test_df["hour"] == eh) & (test_df["minute"] <= em)
    test_df = test_df[mask].copy()

    if test_df.empty:
        print(f"\n── 強過濾破底翻：{sh}:{sm:02d}~{eh}:{em:02d} 逐分鐘 ──")
        print("  （該區間無 breakout_signal=True 樣本）")
        return

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    # ── 信心度門檻掃描（在破底翻樣本內，看各門檻勝率 + 召回）──────
    total_pos = test_df["target"].sum()  # 破底翻樣本中實際漲（label=1）總數
    print(f"\n── 強過濾破底翻：信心度門檻掃描（{sh}:{sm:02d}~{eh}:{em:02d}）──")
    print(f"  破底翻樣本實際漲總數: {total_pos:,} 筆（程式要從中抓出多少）")
    print(f"  {'門檻':>6}  {'訊號數':>7}  {'抓到漲':>7}  {'勝率':>6}  {'召回率':>6}")
    print("  " + "-" * 40)
    for thr in [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sub = test_df[test_df["proba"] >= thr]
        if len(sub) == 0:
            print(f"  {thr:.2f}  {0:>7,}  {0:>7,}  {'--':>6}  {'--':>6}")
        else:
            tp = int((sub["target"] == 1).sum())
            recall = tp / total_pos if total_pos > 0 else 0
            print(f"  {thr:.2f}  {len(sub):>7,}  {tp:>7,}  {sub['target'].mean()*100:>5.1f}%  {recall*100:>5.1f}%")

    # ── 套用使用者指定門檻，做逐分鐘表 ───────────────────────────
    filt = test_df[test_df["proba"] >= threshold]

    if filt.empty:
        print(f"\n── 強過濾破底翻：{sh}:{sm:02d}~{eh}:{em:02d} 逐分鐘（proba≥{threshold:.2f}）──")
        print("  （該門檻下無樣本）")
        return

    grp = filt.groupby("minute").agg(
        推論數=("target", "count"),
        平均信心度=("proba", "mean"),
        勝率=("target", "mean"),
    )
    grp["平均信心度"] = (grp["平均信心度"] * 100).round(1)
    grp["勝率"] = (grp["勝率"] * 100).round(1)
    grp = grp.reset_index()
    grp.insert(0, "時間", grp["minute"].apply(lambda m: f"9:{m:02d}"))

    print(f"\n── 強過濾破底翻：{sh}:{sm:02d}~{eh}:{em:02d} 逐分鐘（proba≥{threshold:.2f}）──")
    print(grp[["時間", "推論數", "平均信心度", "勝率"]].to_string(index=False))

    total_n = len(filt)
    print(
        f"\n  {sh}:{sm:02d}~{eh}:{em:02d} 彙總（proba≥{threshold:.2f}）：推論 {total_n:,} 筆，"
        f"平均信心度 {filt['proba'].mean()*100:.1f}%，"
        f"勝率 {filt['target'].mean()*100:.1f}%"
    )


if __name__ == "__main__":
    breakout_filter_report(test_days=10)
    print()
    breakout_minute_report(test_days=10, threshold=0.45)
