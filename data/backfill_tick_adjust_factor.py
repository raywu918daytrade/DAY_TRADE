"""
用 Fugle／富邦日K API 反推每支股票每天的除權息調整係數，把
db/tick_adjust_factor/ 的涵蓋範圍往前延伸到 db/tick 開始有資料（目前約
2025-08-01）之前——2026-08-01 發現，除權息事件如果發生在 db/tick 涵蓋範圍
之前（例如 0050 2025-06-18 的1:4拆股），data/build_tick_adjust_factor.py
（拿本機既有的 db/tick 收盤 vs db/fugle_day 收盤反推）碰不到那個日期，算不
出係數。

一開始（2026-08-01）先用 finmind/backfill_tick_adjust_factor.py（FinMind的
TaiwanStockPrice/TaiwanStockPriceAdj）補過0050/0056，後來發現更好的做法：
    1. FinMind 終究是不同廠商，兩邊「已還原」的基準有 ~0.5~1% 的小落差
       （0050同期實測比對過），不是完全一致的同一套邏輯
    2. FinMind 的存取權限有時效（fubon/tick_api.py 提到只到2026-08-18）
改成這支：Fugle／富邦的 historical/candles 本身就能一次帶 adjusted="true"
（已還原）或不帶（原始），跟 db/fugle_day 已經抓的「已還原」版本同一套邏輯
（兩家都是），不會有跨廠商落差，也不受FinMind權限時效影響。只需要額外抓
「原始」版本——「已還原」版本 db/fugle_day 本來就有了，不用重抓。

先試 Fugle，查無資料/失敗才 fallback 富邦（比照 data/day_data_loader.py 兩邊
都能查日K的做法）——富邦要先登入 SDK，比較重，所以只在真的需要時才登入，
不是每支股票都固定用富邦查一次備援。

用法：
    python -m data.backfill_tick_adjust_factor 0050 0056
    python -m data.backfill_tick_adjust_factor 0050 0056 --start 2016-01-01
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from fugle import fugle_api  # noqa: E402
from data.query import load_day  # noqa: E402

_TW = timezone(timedelta(hours=8))
_FACTOR_DIR = _ROOT / "db/tick_adjust_factor"
_DEFAULT_START = "2016-01-01"


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, file_path)


def _fetch_raw_day_close(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """抓 Fugle 原始（未還原權息）日K收盤價，自動切成 <1年 一段（Fugle單次
    請求限制），回傳 stock_id/date/close。不帶 adjusted 參數，跟
    data/m1_data_loader.py 2026-08-01 之後的做法一致——省略就是原始價格。"""
    cur = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    chunks = []
    while cur <= end:
        chunk_end = min(cur.replace(year=cur.year + 1) - timedelta(days=1), end)
        r = fugle_api.historical_candles(
            stock_id,
            **{
                "from": cur.strftime("%Y-%m-%d"),
                "to": chunk_end.strftime("%Y-%m-%d"),
                "fields": "close",
                "sort": "asc",
            },
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            df = pd.DataFrame(data)
            df["stock_id"] = stock_id
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            chunks.append(df[["stock_id", "date", "close"]])
        cur = chunk_end + timedelta(days=1)
        if cur <= end:
            time.sleep(1.05)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["stock_id", "date", "close"])


def _fetch_raw_day_close_fubon(sdk, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """富邦版的 _fetch_raw_day_close()，不帶 adjusted 參數就是原始價格
    （同 data/day_data_loader.py::_fetch_year_fubon() 的呼叫方式，只是那支
    帶了 adjusted="true"，這裡故意不帶）。"""
    from fubon import fubon_api as trade_api

    cur = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    chunks = []
    while cur <= end:
        chunk_end = min(cur.replace(year=cur.year + 1) - timedelta(days=1), end)
        try:
            bars = trade_api.historical_candles(
                sdk,
                stock_id,
                **{"from": cur.strftime("%Y-%m-%d"), "to": chunk_end.strftime("%Y-%m-%d"), "fields": "close", "sort": "asc"},
            )
        except Exception as e:
            print(f"    {stock_id} {cur}~{chunk_end} 富邦查詢失敗（視為這段沒資料）: {e}")
            bars = []
        if bars:
            df = pd.DataFrame(bars)
            df["stock_id"] = stock_id
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            chunks.append(df[["stock_id", "date", "close"]])
        cur = chunk_end + timedelta(days=1)
        if cur <= end:
            time.sleep(1.05)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["stock_id", "date", "close"])


def backfill_stock(stock_id: str, start: str, end: str, fubon_sdk_holder: dict | None = None) -> pd.DataFrame:
    """回傳單一股票的 (stock_id, date, factor) DataFrame。

    fubon_sdk_holder：選填，{"sdk": None} 這種容器，Fugle查無資料時第一次
    需要富邦備援才會登入、寫進這個容器讓後續股票重用同一個session，不用
    每支都重新登入。傳 None 就完全不 fallback 富邦（只用 Fugle）。"""
    print(f"{stock_id}：查詢 Fugle 原始日K...")
    raw = _fetch_raw_day_close(stock_id, start, end)
    if raw.empty and fubon_sdk_holder is not None:
        print(f"{stock_id}：Fugle 查無資料，改查富邦...")
        if fubon_sdk_holder.get("sdk") is None:
            from fubon import fubon_api as trade_api

            sdk, _ = trade_api.login()
            trade_api.init_market_data(sdk)
            fubon_sdk_holder["sdk"] = sdk
        raw = _fetch_raw_day_close_fubon(fubon_sdk_holder["sdk"], stock_id, start, end)
    if raw.empty:
        print(f"{stock_id}：查無原始日K資料，略過")
        return pd.DataFrame(columns=["stock_id", "date", "factor"])

    adj = load_day(start_date=start)
    adj = adj[(adj["stock_id"] == stock_id) & (adj["date"] >= start) & (adj["date"] <= end)]
    adj = adj[["stock_id", "date"]].assign(date=adj["date"].dt.strftime("%Y-%m-%d"), adj_close=adj["close"].values, stock_id=stock_id)

    merged = raw.rename(columns={"close": "raw_close"}).merge(adj, on=["stock_id", "date"])
    merged = merged[merged["raw_close"] > 0]
    if merged.empty:
        print(f"{stock_id}：跟 db/fugle_day 對不到任何日期，略過")
        return pd.DataFrame(columns=["stock_id", "date", "factor"])
    merged["factor"] = (merged["adj_close"] / merged["raw_close"]).astype("float32")
    print(f"{stock_id}：算出 {len(merged)} 天的 factor")
    return merged[["stock_id", "date", "factor"]]


def _merge_into_factor_dir(new_factor: pd.DataFrame):
    """只補 db/tick_adjust_factor 目前缺的 (stock_id, date)，不覆蓋既有資料。"""
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


def main(stock_ids: list[str], start: str, end: str | None = None):
    end = end or datetime.now(_TW).strftime("%Y-%m-%d")
    fubon_sdk_holder: dict = {"sdk": None}
    try:
        for stock_id in stock_ids:
            factor_df = backfill_stock(stock_id, start, end, fubon_sdk_holder=fubon_sdk_holder)
            _merge_into_factor_dir(factor_df)
    finally:
        if fubon_sdk_holder.get("sdk") is not None:
            from fubon import fubon_api as trade_api

            trade_api.logout(fubon_sdk_holder["sdk"])
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
        print("用法：python -m data.backfill_tick_adjust_factor <stock_id> [<stock_id> ...] [--start YYYY-MM-DD]")
        sys.exit(1)

    main(stock_ids, start)
