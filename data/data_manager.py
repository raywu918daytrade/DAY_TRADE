"""
DataManager：統一三個交易時段的資料載入介面。

Phase（時段）：
    PRE_MARKET  盤前  — 載入 D1 日K + 均量過濾（HF Hub 或本機）
    IN_MARKET   盤中  — M1 由 M1RestPoller 推送，D1 已在盤前載好
    POST_MARKET 盤後  — 收盤後 backfill 由 M1RestPoller 處理，D1 不變

上層規則：
    - 不直接判斷 HF Hub vs 本機路徑，一律呼叫 load_d1(stocks)
    - 盤前啟動和每日 06:00 refresh 呼叫同一個函式，不再各自寫 if/else
"""

from __future__ import annotations

import os
from enum import Enum

import pandas as pd

_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")


class Phase(Enum):
    PRE_MARKET = "pre"    # 盤前：需要載入 D1 + tickers
    IN_MARKET = "in"      # 盤中：M1 由 Poller 推，D1 已備妥
    POST_MARKET = "post"  # 盤後：SL/TP only，backfill 由 Poller 處理


def load_d1(stocks: set) -> tuple[pd.DataFrame, set]:
    """
    載入日K並做均量過濾，回傳 (day_df, filtered_stocks)。

    資料來源自動判斷：
        有 HF_REPO_ID → HF Hub 最近 2 個月月份檔
        沒有          → 本機 db/fugle_day/

    盤前啟動和每日 06:00 refresh 都呼叫這一個函式。
    """
    if not stocks:
        return pd.DataFrame(), set()

    if _HF_REPO_ID:
        return _load_d1_from_hf(stocks)
    else:
        return _load_d1_local(stocks)


# ── 內部：HF Hub ──────────────────────────────────────────────────────────────

def _load_d1_from_hf(stocks: set) -> tuple[pd.DataFrame, set]:
    import re as _re
    import pyarrow.parquet as pq
    import pyarrow as pa
    from huggingface_hub import HfApi, hf_hub_download

    months_to_load = int(os.environ.get("DAY_MONTHS_TO_LOAD", "2"))
    token = os.environ.get("HF_TOKEN") or None

    day_files = sorted([
        f for f in HfApi().list_repo_files(
            repo_id=_HF_REPO_ID, repo_type="dataset", token=token)
        if _re.match(r"day_trade/day/\d{4}_\d+\.parquet$", f)
    ])
    if not day_files:
        print("  [D1] HF Hub 無月份日K檔", flush=True)
        return pd.DataFrame(), set()

    targets = day_files[-months_to_load:]
    paths = []
    for fname in targets:
        p = hf_hub_download(repo_id=_HF_REPO_ID, filename=fname,
                            repo_type="dataset", token=token)
        print(f"  [D1] → {p}", flush=True)
        paths.append(p)

    # 第 1 次：只讀 volume 欄位做均量過濾
    tables_vol = [pq.read_table(p, columns=["stock_id", "date", "volume"]) for p in paths]
    df_vol = pa.concat_tables(tables_vol).to_pandas()
    del tables_vol
    top = set(_volume_filter(stocks, df_vol))
    del df_vol
    if not top:
        return pd.DataFrame(), set()

    # 第 2 次：只讀過濾後股票的完整資料
    tables = [pq.read_table(p, filters=[("stock_id", "in", list(top))]) for p in paths]
    df = pa.concat_tables(tables).to_pandas()
    del tables
    df["date"] = pd.to_datetime(df["date"])
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    print(f"  [D1] {len(df):,} 筆，{df['stock_id'].nunique():,} 支（均量過濾後）", flush=True)
    return df, top


# ── 內部：本機 ────────────────────────────────────────────────────────────────

def _load_d1_local(stocks: set) -> tuple[pd.DataFrame, set]:
    from data.query import load_day

    full = load_day()
    print(f"  [D1] 本機全量：{len(full):,} 筆，開始均量過濾...", flush=True)
    top = set(_volume_filter(stocks, full[["stock_id", "date", "volume"]]))
    df = full[full["stock_id"].isin(top)].copy()
    del full
    print(f"  [D1] {len(df):,} 筆，{df['stock_id'].nunique():,} 支（均量過濾後）", flush=True)
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
