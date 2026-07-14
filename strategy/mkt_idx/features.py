"""
mkt_idx 策略 —— 核心特徵：個股跟大盤（0050）的累積報酬率差值。

第一版先只做這一個特徵（ret_vs_idx），驗證數字正確後再逐步加其他特徵，
不一次寫完整套 pipeline（見 2026-07-14 的討論：先精簡、慢慢加，方便debug）。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from strategy.mkt_idx.config import HOLD_BARS, IDX_SYMBOL, SL_PCT, TP_PCT


def add_ret_vs_idx(m1: pd.DataFrame, idx_symbol: str = IDX_SYMBOL) -> pd.DataFrame:
    """
    算「個股從今日開盤累積到現在的報酬率」減去「0050從今日開盤累積到現在
    的報酬率」，逐分鐘更新，回傳加了 ret_vs_idx 欄位的 m1（複本，不動傳
    進來的原始 df）。

    >0 代表這支股票從開盤到現在，漲得比大盤多（相對強勢）；
    <0 代表跑輸大盤（相對弱勢）。

    m1 需已有 stock_id/date/day_date/open/close 欄位，date 需為 datetime，
    day_date 為 date（不是 datetime）。
    """
    m1 = m1.copy()
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    m1["_ret_since_open"] = m1["close"] / day_open - 1

    idx = (
        m1[m1["stock_id"] == idx_symbol][["date", "_ret_since_open"]]
        .rename(columns={"_ret_since_open": "_idx_ret_since_open"})
        .drop_duplicates("date")
    )

    m1 = m1.merge(idx, on="date", how="left")
    m1["ret_vs_idx"] = m1["_ret_since_open"] - m1["_idx_ret_since_open"]
    m1 = m1.drop(columns=["_ret_since_open", "_idx_ret_since_open"])
    return m1


def top_n_by_prev_day_volume(m1: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """
    只留每天「依前一交易日全天成交量排序」前 n 名的股票（流動性過濾的簡易
    版本，2026-07-14 討論）。

    用「前一天」的量決定「今天」要不要納入候選池，不是用「今天」的量——
    避免用「今天發生的事」決定「今天要不要看這支股票」的循環問題（例如
    直接拿當天開盤第一分鐘的量排序，會混進「今天剛好爆量」這種跟要驗證
    的訊號本身有關聯的雜訊）。純粹用手上已有的 m1 全天量加總 + shift(1)，
    不用另外載入日K（db/fugle_day/）。

    m1 需已有 stock_id/day_date/volume 欄位。
    """
    daily_vol = m1.groupby(["stock_id", "day_date"])["volume"].sum().reset_index()
    daily_vol = daily_vol.sort_values(["stock_id", "day_date"])
    daily_vol["prev_day_volume"] = daily_vol.groupby("stock_id")["volume"].shift(1)
    daily_vol = daily_vol.dropna(subset=["prev_day_volume"])
    daily_vol["rank"] = daily_vol.groupby("day_date")["prev_day_volume"].rank(ascending=False, method="first")
    keep = daily_vol[daily_vol["rank"] <= n][["stock_id", "day_date"]]
    return m1.merge(keep, on=["stock_id", "day_date"], how="inner")


# ═══════════════════════════════════════════════════════════════════════════════
# Triple Barrier 標籤（3分類版）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 2026-07-14 討論：跟 rally/orb 的二分類 triple barrier 不同——這裡「時間到、
# 兩個barrier都沒碰到」的情況統一標成「平」，不是看當下比進場價高還低硬分
# 漲跌。理由：這支策略驗證出來的訊號優勢只有0.01~0.02個百分點，遠小於
# TP_PCT/SL_PCT=3%，如果沿用 rally/orb 那種「時間到就看sign」的規則，
# 幾乎所有「根本沒有真正波動」的樣本都會被塞進漲或跌的其中一類，稀釋掉
# 真正有意義的樣本。
#
# label： 2=漲（先碰到+TP_PCT）  1=平（HOLD_BARS內都沒碰到任一邊）
#         0=跌（先碰到-SL_PCT）


def _barrier_label_group_3class(
    closes: np.ndarray,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    hold_bars: int = HOLD_BARS,
) -> np.ndarray:
    """每日每股，逐bar往前看最多hold_bars根，判斷先碰到哪個barrier，
    或hold_bars內都沒碰到（平）。"""
    n = len(closes)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        entry = closes[i]
        tp_price = entry * (1 + tp_pct)
        sl_price = entry * (1 - sl_pct)
        future = closes[i + 1 : i + hold_bars + 1]
        tp_idx = np.argmax(future >= tp_price) if (future >= tp_price).any() else len(future)
        sl_idx = np.argmax(future <= sl_price) if (future <= sl_price).any() else len(future)
        if tp_idx < sl_idx:
            labels[i] = 2  # 漲
        elif sl_idx < tp_idx:
            labels[i] = 0  # 跌
        elif len(future) == hold_bars:
            labels[i] = 1  # 平：兩個都沒碰到，且有完整的 hold_bars 可看
        # else: 太靠近當日尾端，future 不足 hold_bars 根，保留 NaN
    return labels


def make_barrier_labels_3class(
    m1: pd.DataFrame,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    hold_bars: int = HOLD_BARS,
) -> pd.Series:
    """對 m1 逐股逐日呼叫 _barrier_label_group_3class()，回傳跟 m1 對齊的
    label Series（2=漲/1=平/0=跌，NaN=樣本不足無法判斷）。"""
    return m1.groupby(["stock_id", "day_date"], group_keys=False).apply(
        lambda g: pd.Series(
            _barrier_label_group_3class(g["close"].values, tp_pct, sl_pct, hold_bars),
            index=g.index,
        ),
        include_groups=False,
    )
