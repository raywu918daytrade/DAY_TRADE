"""
ATR5 PR值過濾診斷（2026-07-23討論）：在正式接進pipeline之前，先看不同PR
門檻（0.5~0.9）下，會濾掉多少樣本、剩下的「跌/平/漲」比例分別是多少——
決定要用多少門檻之前，要先看過實際分布，不要憑感覺猜。

⚠️ 2026-07-23第二版：第一版用「跨股票、同一分鐘排名」，發現濾掉90%資料
後「平」還是有88.59%，效果比預期弱很多。診斷後發現是**用錯PR基準**，不是
bug——跨股票排名structurally沒辦法濾掉「整個市場都平靜」的情況（不管
大盤多平靜，永遠有個「排名前10%」），只能抓「相對其他股票誰比較活躍」。
改成「同一支股票、自己的歷史分布」（run_own_history()，這是原本
AskUserQuestion時使用者選的推薦選項）——這支才是真正抓「這支股票現在是
不是異常平靜（相對它自己平常）」，理論上濾除力道會強很多。

用法：
    python strategy/mkt/experiments/atr5_pr_check.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from data.query import load_m1
from strategy.mkt.config import IDX_SYMBOL, TOP_N
from strategy.mkt.features import add_atr5, add_ret_vs_idx, make_barrier_labels_3class, top_n_by_prev_day_volume
from strategy.mkt.train import _prepare_data


def _atr5_full_day(top_n: int = TOP_N, since: str | None = None) -> pd.DataFrame:
    """對完整一天（含9:00~9:10暖機期）、流動性過濾後的m1算atr5——保留
    day內的分鐘連續性，rolling(5)/shift(1)才不會算錯。只回傳
    stock_id/day_date/date/atr5，之後拿去merge。since可選，篩日期範圍
    加快測試（例如"2026-06-01"）。"""
    m1 = load_m1()
    m1["date"] = pd.to_datetime(m1["date"])
    if since is not None:
        m1 = m1[m1["date"] >= since]
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    df = add_ret_vs_idx(m1)
    df = df[df["stock_id"] != IDX_SYMBOL]
    df = top_n_by_prev_day_volume(df, n=top_n)
    df = add_atr5(df)
    return df[["stock_id", "day_date", "date", "close", "atr5"]]


def add_atr5_own_history_pr(m1: pd.DataFrame, n_days: int = 60, min_history: int = 20) -> pd.DataFrame:
    """
    每支股票、自己的ATR5歷史分布百分位（取代跨股票排名版，見檔頭2026-07-23
    第二版說明）。

    做法：
    1. 每支股票、每個交易日，算出當天9:11~9:30窗口atr5的平均值，當作
       「這一天」的代表值。
    2. 每支股票，對每一天，取「過去n_days個交易日（不含今天）」的代表值
       陣列當參考分布——避免lookahead，只用已經收盤、確定了的過去交易日，
       不會用到今天自己的資料。歷史不足min_history天的日子直接跳過（PR
       設NaN），避免用太少樣本算出來的百分位不可靠。
    3. 每一列（某分鐘）的即時atr5，去查當天對應的參考分布，算百分位
       （searchsorted，跟排序好的歷史陣列比對位置）。

    m1 需已有 stock_id/day_date/date/atr5 欄位（先呼叫 add_atr5()，且要
    先做過9:11~9:30時段過濾，因為daily summary只看這段窗口）。
    """
    daily = m1.groupby(["stock_id", "day_date"])["atr5"].mean().reset_index()
    daily = daily.sort_values(["stock_id", "day_date"]).reset_index(drop=True)

    ref_lookup = {}
    for stock_id, g in daily.groupby("stock_id"):
        vals = g["atr5"].to_numpy()
        days = g["day_date"].to_numpy()
        for i in range(len(vals)):
            start = max(0, i - n_days)
            ref = vals[start:i]  # 不含今天（index i）
            if len(ref) < min_history:
                continue
            ref_lookup[(stock_id, days[i])] = np.sort(ref)

    def _pr_row(stock_id, day_date, atr5_val):
        arr = ref_lookup.get((stock_id, day_date))
        if arr is None or pd.isna(atr5_val):
            return np.nan
        return np.searchsorted(arr, atr5_val, side="right") / len(arr)

    m1 = m1.copy()
    m1["atr5_own_pr"] = [_pr_row(sid, dd, v) for sid, dd, v in zip(m1["stock_id"], m1["day_date"], m1["atr5"])]
    return m1


def run_own_history(
    pr_values: list[float] | None = None,
    since: str | None = "2026-04-01",
    n_days: int = 60,
    hold_bars: int | None = None,
):
    """跟 run()（跨股票排名版）對照組——同一支股票、自己的歷史分布版本。
    since預設抓最近幾個月，加上前面n_days天當歷史暖機用，不用整套全歷史
    重算，跑起來快很多。

    hold_bars（2026-07-23討論）：None（預設）＝沿用 _prepare_data() cache
    裡現成的target（config.py正式設定HOLD_BARS=10，不用重跑慢的barrier
    標籤迴圈）。設數字（例如30）則不吃cache，改用這個hold_bars現算target
    ——只在這支診斷腳本用，不影響config.py或正式pipeline。config.py已經
    驗證過30分鐘窗口下ret_vs_idx訊號會反轉，這裡只是想單獨看「ATR5篩選」
    在更長窗口下效果會不會比較明顯，不是要改動正式的HOLD_BARS設定。
    """
    if pr_values is None:
        pr_values = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("對完整一天重算atr5（保留暖機期的分鐘連續性）...")
    atr5_df = _atr5_full_day(since=since)

    if hold_bars is None:
        print("讀取 _prepare_data() 的cache（已有target，不用重跑慢的barrier標籤迴圈）...")
        target_df = _prepare_data(use_cache=True)[["stock_id", "date", "target"]]
    else:
        print(f"用hold_bars={hold_bars}重新計算3分類標籤（跟正式HOLD_BARS=10不同，不用cache）...")
        labeled = atr5_df.copy()
        labeled["target"] = make_barrier_labels_3class(labeled, hold_bars=hold_bars)
        labeled = labeled.dropna(subset=["target"])
        labeled["target"] = labeled["target"].astype(int)
        target_df = labeled[["stock_id", "date", "target"]]

    print(f"算每支股票、自己過去{n_days}個交易日的ATR5歷史分布百分位...")
    # 時段過濾（daily summary只看9:11~9:30這段）
    window_df = atr5_df[
        (atr5_df["date"].dt.hour == 9) & (atr5_df["date"].dt.minute >= 11) & (atr5_df["date"].dt.minute < 30)
    ].copy()
    window_df = add_atr5_own_history_pr(window_df, n_days=n_days)

    df = target_df.merge(window_df[["stock_id", "date", "atr5", "atr5_own_pr"]], on=["stock_id", "date"], how="inner")
    df = df.dropna(subset=["atr5_own_pr"])
    print(f"\n時段過濾+歷史暖機後總樣本數: {len(df):,}")

    print(f"\n{'PR門檻':>8}  {'樣本數':>10}  {'保留%':>7}  {'跌%':>7}  {'平%':>7}  {'漲%':>7}")
    total = len(df)
    for pr in pr_values:
        sub = df[df["atr5_own_pr"] >= pr]
        n = len(sub)
        if n == 0:
            print(f"{pr:>8.1f}  {0:>10}  {'--':>7}  {'--':>7}  {'--':>7}  {'--':>7}")
            continue
        keep_pct = n / total * 100
        vc = sub["target"].value_counts(normalize=True) * 100
        print(
            f"{pr:>8.1f}  {n:>10,}  {keep_pct:>6.2f}%  {vc.get(0, 0):>6.2f}%  "
            f"{vc.get(1, 0):>6.2f}%  {vc.get(2, 0):>6.2f}%"
        )

    return df


def run_flat_only(percentiles: list[int] | None = None, use_cache: bool = True):
    """
    比照 strategy/cnn/experiments/atr5_flat_filter_check.py 的做法（2026-
    07-23參考）：**只篩「平」，跌/漲全部保留不動**，門檻用「平」這個類別
    自己的atr5全域百分位（不分股票、不分時間混在一起算一個固定門檻）。

    跟 run()/run_own_history() 的關鍵差異：那兩支對跌/平/漲三個類別
    「統一」套用同一個PR門檻，會連帶濾掉一些「前5分鐘看起來平靜、但突然
    噴出」的漲/跌樣本（稀少又珍貴），抵銷掉篩選帶來的效果。這支固定住
    跌/漲的筆數，只削減平的數量，比例改善才會直接反映篩選力道，不會被
    「漲跌也被誤殺」稀釋。

    use_cache：見 strategy/mkt/train.py::_prepare_data() 的說明，這裡預設
    True（沿用現成cache、HOLD_BARS=10）。
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95, 99]

    print("讀取 _prepare_data() 的cache（已有target）...")
    target_df = _prepare_data(use_cache=use_cache)[["stock_id", "date", "target"]]

    print("對完整一天重算atr5（保留暖機期的分鐘連續性）...")
    atr5_df = _atr5_full_day()

    df = target_df.merge(atr5_df[["stock_id", "date", "atr5"]], on=["stock_id", "date"], how="left")
    df = df.dropna(subset=["atr5"])
    print(f"\n總樣本數: {len(df):,}")

    label_names = {0: "跌", 1: "平", 2: "漲"}
    for target, name in label_names.items():
        sub = df.loc[df["target"] == target, "atr5"]
        print(f"\n── {name}（{len(sub):,} 筆） ──")
        pct = np.percentile(sub, percentiles)
        for p, v in zip(percentiles, pct):
            print(f"  p{p}: {v:.5f}")

    flat = df.loc[df["target"] == 1, "atr5"]
    non_flat_count = int((df["target"] != 1).sum())
    print(f"\n── 只篩「平」：不同門檻下平還會剩多少筆（跌/漲固定{non_flat_count:,}筆不變） ──")
    for p in percentiles:
        threshold = np.percentile(flat, p)
        kept = int((flat >= threshold).sum())
        total_after = kept + non_flat_count
        print(
            f"  門檻=p{p}分位數({threshold:.5f}): 平剩 {kept:,} 筆（保留{kept / len(flat):.2%}），"
            f"平:跌+漲 = {kept / non_flat_count:.2f} : 1，"
            f"篩選後跌%={df.loc[df['target']==0].shape[0]/total_after*100:.2f}%  "
            f"平%={kept/total_after*100:.2f}%  "
            f"漲%={df.loc[df['target']==2].shape[0]/total_after*100:.2f}%"
        )

    return df


def run_absolute_uniform(percentiles: list[int] | None = None, top_n: int = TOP_N):
    """
    「絕對門檻＋三類都篩」——可以上線用的版本（2026-07-23討論）。

    跟 run_flat_only() 不同：run_flat_only() 只篩「平」、跌/漲固定保留，
    這個規則依賴「事後才知道的label」，沒辦法在上線推論當下重現（那時候
    還不知道這支股票接下來是漲是跌是平）。這支改成：**不管類別，統一用
    同一個絕對atr5門檻篩**——上線時只看「這支股票現在的atr5」這個當下就
    知道的數字，跟訓練/測試用同一套規則，三邊一致，可以真的部署。

    代價：atr5剛好偏低的漲/跌樣本一樣會被濾掉（不像run_flat_only()那樣
    跌/漲完全不動），效果理論上會比run_flat_only()弱，但比
    run()/run_own_history()那兩個「相對排名」版本強（絕對門檻本身就比
    相對排名更能反映「客觀上到底活不活躍」，見兩邊差異的說明）。

    門檻用「全體」（不分類別）atr5的百分位，不是只看「平」的分布——因為
    這裡篩選不看類別，門檻自然也不該只看單一類別的分布。

    ⚠️ 2026-07-25修正：不能再用 _prepare_data() 的cache當target來源——
    自從2026-07-23把ATR5 p90過濾接進 _prepare_data() 之後，那份cache回傳
    的已經是「套用ATR5_FILTER_THRESHOLD過濾後」的population，如果拿它來
    測更高的百分位（例如p95/p99），等於在「已經被p90篩過一次」的資料上
    再篩一次，percentile的分母錯了，算出來的門檻/密度沒有意義。改成自己
    直接對「時段過濾後、ATR5過濾前」的population算target，才是公平的
    「全體」基準。
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    print("對完整一天重算atr5（保留暖機期的分鐘連續性）...")
    atr5_df = _atr5_full_day(top_n=top_n)
    atr5_df["target"] = make_barrier_labels_3class(atr5_df)

    print("時段過濾：9:11~9:30（跟_prepare_data()一致，ATR5過濾前）...")
    df = atr5_df[
        (atr5_df["date"].dt.hour == 9) & (atr5_df["date"].dt.minute >= 11) & (atr5_df["date"].dt.minute < 30)
    ].copy()
    df = df.dropna(subset=["target", "atr5"])
    df["target"] = df["target"].astype(int)
    print(f"\n總樣本數（ATR5過濾前）: {len(df):,}")

    print(f"\n{'門檻(全體pN)':>14}  {'atr5值':>10}  {'樣本數':>10}  {'保留%':>7}  {'跌%':>7}  {'平%':>7}  {'漲%':>7}")
    total = len(df)
    for p in percentiles:
        threshold = np.percentile(df["atr5"], p)
        sub = df[df["atr5"] >= threshold]
        n = len(sub)
        if n == 0:
            continue
        keep_pct = n / total * 100
        vc = sub["target"].value_counts(normalize=True) * 100
        print(
            f"{'p' + str(p):>14}  {threshold:>10.5f}  {n:>10,}  {keep_pct:>6.2f}%  "
            f"{vc.get(0, 0):>6.2f}%  {vc.get(1, 0):>6.2f}%  {vc.get(2, 0):>6.2f}%"
        )

    return df


if __name__ == "__main__":
    run_absolute_uniform()
