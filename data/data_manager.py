"""
DataManager：統一三個交易時段的資料載入介面。

Phase（時段）：
    PRE_MARKET  盤前  — 載入 D1 日K + 均量過濾（本機 db/fugle_day/）
    IN_MARKET   盤中  — M1 由 fubon/marketdata_ws.py 的富邦 WebSocket 推送，D1 已在盤前載好
    POST_MARKET 盤後  — 持續收 WebSocket 資料到收盤，D1 不變

上層規則：
    - 盤前啟動和每日 06:00 refresh 呼叫同一個函式 load_d1(stocks)，不再各自寫 if/else
"""

from __future__ import annotations

import os
from enum import Enum

import pandas as pd


class Phase(Enum):
    PRE_MARKET = "pre"    # 盤前：需要載入 D1 + tickers
    IN_MARKET = "in"      # 盤中：M1 由 Poller 推，D1 已備妥
    POST_MARKET = "post"  # 盤後：SL/TP only，backfill 由 Poller 處理


def load_d1(stocks: set) -> tuple[pd.DataFrame, set]:
    """
    載入日K並做均量過濾，回傳 (day_df, filtered_stocks)。

    目前固定讀本機 db/fugle_day/。

    盤前啟動和每日 06:00 refresh 都呼叫這一個函式。
    """
    if not stocks:
        return pd.DataFrame(), set()

    return _load_d1_local(stocks)


# ── 內部：本機 ────────────────────────────────────────────────────────────────

def _load_d1_local(stocks: set) -> tuple[pd.DataFrame, set]:
    from data.query import load_day

    # 均量過濾（_volume_filter()）只看每支股票最近20個交易日（.tail(20)），
    # 策略特徵最長也只到 rolling(20) 左右——不需要 load_day() 預設的全部
    # 歷史，60天留足夠緩衝應付連假（2026-07-25討論：這裡讀全部歷史是
    # live_trader.py 記憶體爆掉的主因之一，見 data/query.py::load_day()
    # 的說明）。
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    full = load_day(start_date=start_date)
    print(f"  [D1] 最近{len(full):,} 筆，開始均量過濾...", flush=True)
    top = set(_volume_filter(stocks, full[["stock_id", "date", "volume"]]))
    # 額外帶上 0050，理由同 _load_d1_from_hf；top（候選股集合）仍不含 0050
    df = full[full["stock_id"].isin(top | {"0050"})].copy()
    del full
    print(f"  [D1] {len(df):,} 筆，{df['stock_id'].nunique():,} 支（均量過濾後 + 0050）", flush=True)
    return df, top


# ── 均量過濾（從 live_trader.py 搬移，集中在此）─────────────────────────────

_MAX_SUBSCRIPTIONS = int(os.environ.get("MAX_SUBSCRIPTIONS", "500"))
_MIN_AVG_VOL_LOTS  = int(os.environ.get("MIN_AVG_VOL_LOTS", "1000"))


def _volume_filter(stocks: set, day: pd.DataFrame) -> list:
    """20日均量過濾，回傳按均量排序（高→低）的 stock_id list，最多 MAX_SUBSCRIPTIONS 支。"""
    if day.empty or not stocks:
        result = list(stocks)[:_MAX_SUBSCRIPTIONS]
        print(f"  均量過濾：無日K，取前 {len(result)} 支", flush=True)
        return result
    recent = day[day["stock_id"].isin(stocks)].copy()
    recent["date"] = pd.to_datetime(recent["date"])
    last20 = recent.sort_values("date").groupby("stock_id").tail(20)
    avg_vol = last20.groupby("stock_id")["volume"].mean()
    qualified = avg_vol[avg_vol >= _MIN_AVG_VOL_LOTS * 1000].sort_values(ascending=False)
    result = list(qualified.index[:_MAX_SUBSCRIPTIONS])
    print(
        f"  均量過濾（≥{_MIN_AVG_VOL_LOTS}張）: {len(stocks)} → {len(qualified)} 支"
        f" → 訂閱前 {len(result)} 支",
        flush=True,
    )
    return result
