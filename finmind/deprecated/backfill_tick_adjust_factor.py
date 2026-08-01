"""
⚠️ 2026-08-01 已被 data/backfill_tick_adjust_factor.py 取代，不要再用這支！
原因：FinMind 終究是不同廠商，TaiwanStockPriceAdj 的還原基準跟 Fugle/富邦
（db/fugle_day 用的那套）有 ~0.5~1% 的小落差（0050同期實測比對過），不是
完全一致；而且 FinMind 的存取權限有時效（fubon/tick_api.py 提到只到
2026-08-18）。日K（D1）本身 Fugle/富邦就能查到很久以前（不像分K有近30日
限制），沒有必要為了反推係數繞去打 FinMind API、引入第三家廠商的落差。
這支保留著沒刪（0050/0056 當時用這支補過，資料還在，不用重補），純粹是
留著當參考，之後要延伸新股票的係數涵蓋範圍，一律改用
data/backfill_tick_adjust_factor.py。

以下是原本的說明：

用 FinMind 的 TaiwanStockPrice（原始）／TaiwanStockPriceAdj（已還原權息）日K，
把 db/tick_adjust_factor/ 的涵蓋範圍往前延伸到 db/tick 開始有資料（目前約
2025-08-01）之前——2026-08-01 發現，除權息事件如果發生在 db/tick 涵蓋範圍
之前（例如 0050 2025-06-18 的 1:4 拆股），data/build_tick_adjust_factor.py
（拿 tick 收盤 vs db/fugle_day 收盤反推）根本碰不到那個日期，算不出係數。

跟 data/build_tick_adjust_factor.py 是同一個目的（都是算 factor = 調整後收盤
÷ 原始收盤，供 data/query.py 的 load_volume_profile_adjusted()/
load_poc_adjusted() 在查詢時換算用），差別只在資料來源：那支是從本機既有的
db/tick + db/fugle_day 直接算（快、不用打API）；這支是打 FinMind API 換一組
原始/已還原的日K來算（給 db/tick 涵蓋不到的更早期間用）。

TaiwanStockPrice/TaiwanStockPriceAdj 是日K資料集，支援 start_date~end_date
一次回傳整段區間（不像 TaiwanStockKBar/TaiwanStockPriceTick 那種分K/逐筆
資料集只能查一天），一支股票只要2個request（原始+已還原各一次）就能拿到
多年資料，用量很小，不需要 backfill_m1_history.py/backfill_tick_history.py
那種月份迴圈+批次併發的重量級架構。

只會新增 db/tick_adjust_factor 裡目前沒有的 (stock_id, date) 組合，不會
覆蓋 data/build_tick_adjust_factor.py 算出來的既有資料（tick direct 算出來的
更準，維持既有邏輯不動）。

用法：
    python -m finmind.backfill_tick_adjust_factor 0050 0056
    python -m finmind.backfill_tick_adjust_factor 0050 0056 --start 2019-01-01
"""

import asyncio
import sys
from pathlib import Path

import aiohttp
import pandas as pd

from finmind.m1_api import _ROOT, _fetch_finmind_day

_FACTOR_DIR = _ROOT / "db/tick_adjust_factor"
_DEFAULT_START = "2019-01-01"


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    import os

    os.replace(tmp_path, file_path)


async def _fetch_day_range(session: aiohttp.ClientSession, dataset: str, stock_id: str, start: str, end: str) -> pd.DataFrame:
    data = await _fetch_finmind_day(session, dataset, stock_id, start, end_date=end)
    if not data:
        return pd.DataFrame(columns=["stock_id", "date", "close"])
    df = pd.DataFrame(data)
    return df[["stock_id", "date", "close"]]


async def backfill_stock(session: aiohttp.ClientSession, stock_id: str, start: str, end: str) -> pd.DataFrame:
    """回傳單一股票的 (stock_id, date, factor) DataFrame。"""
    print(f"{stock_id}：查詢 TaiwanStockPrice（原始）...")
    raw = await _fetch_day_range(session, "TaiwanStockPrice", stock_id, start, end)
    print(f"{stock_id}：查詢 TaiwanStockPriceAdj（已還原）...")
    adj = await _fetch_day_range(session, "TaiwanStockPriceAdj", stock_id, start, end)

    if raw.empty or adj.empty:
        print(f"{stock_id}：任一邊查無資料，略過（raw={len(raw)}, adj={len(adj)}）")
        return pd.DataFrame(columns=["stock_id", "date", "factor"])

    merged = raw.merge(adj, on=["stock_id", "date"], suffixes=("_raw", "_adj"))
    merged = merged[merged["close_raw"] > 0]
    merged["factor"] = (merged["close_adj"] / merged["close_raw"]).astype("float32")
    print(f"{stock_id}：算出 {len(merged)} 天的 factor")
    return merged[["stock_id", "date", "factor"]]


def _merge_into_factor_dir(new_factor: pd.DataFrame):
    """只補 db/tick_adjust_factor 目前缺的 (stock_id, date)，不覆蓋既有資料
    （data/build_tick_adjust_factor.py 從 tick 直接算出來的維持原樣、優先）。"""
    if new_factor.empty:
        return
    new_factor = new_factor.copy()
    new_factor["month"] = pd.to_datetime(new_factor["date"]).dt.strftime("%Y_%m")

    for ym, group in new_factor.groupby("month"):
        group = group.drop(columns=["month"])
        path = _FACTOR_DIR / f"{ym}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            existing_pairs = set(zip(old["stock_id"], old["date"]))
            group = group[~group.apply(lambda r: (r["stock_id"], r["date"]) in existing_pairs, axis=1)]
            if group.empty:
                print(f"  {ym}: 沒有新的 (stock_id, date) 要補")
                continue
            final = pd.concat([old, group], ignore_index=True)
        else:
            final = group
        final = final.drop_duplicates(subset=["stock_id", "date"], keep="first").sort_values(["stock_id", "date"])
        _atomic_to_parquet(final.reset_index(drop=True), path, index=False, compression="zstd")
        print(f"  {ym}: 新增 {len(group)} 筆，寫入 {path.name}（目前共 {len(final)} 筆）")


async def main(stock_ids: list[str], start: str, end: str | None = None):
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    async with aiohttp.ClientSession() as session:
        for stock_id in stock_ids:
            factor_df = await backfill_stock(session, stock_id, start, end)
            _merge_into_factor_dir(factor_df)
    print("\n全部完成 ✅")


if __name__ == "__main__":
    argv = sys.argv[1:]
    start = _DEFAULT_START
    stock_ids = []
    i = 0
    while i < len(argv):
        if argv[i] == "--start":
            start = argv[i + 1]
            i += 2
        else:
            stock_ids.append(argv[i])
            i += 1

    if not stock_ids:
        print("用法：python -m finmind.backfill_tick_adjust_factor <stock_id> [<stock_id> ...] [--start YYYY-MM-DD]")
        sys.exit(1)

    asyncio.run(main(stock_ids, start))
