"""
⚠️ 已淘汰（2026-08-03）：只抓 close，現在 db/d1（見 data/day_data_loader.py
的 update_day()）本身就會抓完整 OHLCV 原始日K，這支多餘了。要回補 db/d1
的歷史，改用 `python -m data.backfill_day_history`。保留這支只是留紀錄，
不刪除，不要再呼叫。

以下是原始說明：

一次性把 tick_universe（400支）的原始（未還原權息）日收盤價，從 2016-01-01
（比照 db/fugle_day 最早的涵蓋範圍）一路補到今天，存進 db/day_raw_close/。

這是 data/build_tick_adjust_factor.py 偵測拆股/合股唯一要看的訊號來源——
只看原始日收盤本身的逐日跳空幅度，跟 db/fugle_day（Fugle 已還原）完全脫鉤，
不用比對任何廠商的「已還原」資料，避免依賴 Fugle 的還原邏輯（2026-08-01
也發現過 FinMind vs Fugle 兩邊還原有 ~0.5~1% 落差，不同廠商本來就不保證
完全一致）。

取代 data/deprecated/backfill_tick_adjust_factor.py（那支只能一次補幾支
指定股票，且要跟 db/fugle_day 比對才能算 factor）。

先試 Fugle，查無資料才 fallback 富邦（比照 data/day_data_loader.py 兩邊都能
查日K的做法），只需要收盤價（fields="close"），比抓完整OHLC輕量。用兩組
Fugle 帳號（FUGLE / FUGLE_DAYTRADE，各自獨立 rate limit）各開一條 thread
平行下載，減少一次性回補的等待時間。

只需要跑一次（之後靠 scripts/update_daily.py 每天用當天 db/tick 的收盤
增量延伸，見 data/build_tick_adjust_factor.py::update_today()）；只有
tick_universe 換股，或想把涵蓋範圍往更早年份延伸時才需要重跑。

用法：
    python -m data.backfill_day_raw_close                     # 全部400支，2016-01-01起
    python -m data.backfill_day_raw_close --start 2020-01-01
    python -m data.backfill_day_raw_close 2330 2454            # 只補指定股票
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from fugle import fugle_api  # noqa: E402

_TW = timezone(timedelta(hours=8))
_RAW_CLOSE_DIR = _ROOT / "db/day_raw_close"
_DEFAULT_START = "2016-01-01"

_save_lock = threading.Lock()
_fubon_lock = threading.Lock()


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def _fetch_raw_day_close(stock_id: str, start_date: str, end_date: str, token: str = None) -> pd.DataFrame:
    """抓 Fugle 原始（未還原權息）日K收盤價，自動切成 <1年 一段（Fugle單次
    請求限制）。不帶 adjusted 參數就是原始價，跟 data/m1_data_loader.py
    2026-08-01 之後的做法一致——省略就是原始價格。404 代表這支股票這段
    區間還沒有資料（例如尚未上市），當空段處理，不中斷整體迴圈。"""
    cur = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    chunks = []
    while cur <= end:
        chunk_end = min(cur.replace(year=cur.year + 1) - timedelta(days=1), end)
        r = fugle_api.historical_candles(
            stock_id,
            token=token,
            **{"from": cur.strftime("%Y-%m-%d"), "to": chunk_end.strftime("%Y-%m-%d"), "fields": "close", "sort": "asc"},
        )
        if r.status_code != 404:
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
    """富邦版的 _fetch_raw_day_close()，不帶 adjusted 參數就是原始價（同
    data/day_data_loader.py::_fetch_year_fubon() 的呼叫方式，只是那支帶了
    adjusted="true"，這裡故意不帶）。"""
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


def _save_raw_close(df: pd.DataFrame):
    """按月分檔寫入 db/day_raw_close/，跟既有 (stock_id, date) 合併去重。"""
    if df.empty:
        return
    df = df.copy()
    df["close"] = df["close"].astype("float32")
    with _save_lock:
        for month, group in df.groupby(pd.to_datetime(df["date"]).dt.to_period("M")):
            ym = f"{month.year}_{month.month:02d}"
            path = _RAW_CLOSE_DIR / f"{ym}.parquet"
            group = group[["stock_id", "date", "close"]]
            if path.exists():
                old = pd.read_parquet(path)
                group = pd.concat([old, group], ignore_index=True)
            group.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
            group.sort_values(["stock_id", "date"], inplace=True)
            _atomic_to_parquet(group.reset_index(drop=True), path, index=False, compression="zstd")


def _raw_close_from_tick(tick_df: pd.DataFrame) -> pd.DataFrame:
    """用當天 tick 資料算出每支股票的原始收盤價：13:30:00（含）以前最後一筆
    成交——13:30:00 本身就是收盤定盤價，之後（例如14:30:00）是盤後逐筆交易，
    不算正式收盤（同 data/deprecated/backfill_tick_adjust_factor.py 舊版
    compute_adjust_factor() 的邏輯）。"""
    df = tick_df.copy()
    df["day"] = df["date"].astype(str).str[:10]
    df["time"] = df["date"].astype(str).str[11:19]
    df = df[df["time"] <= "13:30:00"]
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "date", "close"])
    df.sort_values(["stock_id", "day", "date"], inplace=True)
    close = df.groupby(["stock_id", "day"], as_index=False).last()[["stock_id", "day", "deal_price"]]
    close.rename(columns={"day": "date", "deal_price": "close"}, inplace=True)
    return close


def update_today():
    """從今天的 db/tick 算出今天的原始收盤，append 進 db/day_raw_close。完全
    不用打額外的 API——scripts/update_daily.py 本來就會在這之前呼叫
    update_tick_today() 抓當天 tick，這裡只是從已經下載好的 tick 資料算收盤。"""
    import pyarrow.dataset as ds

    ym = datetime.now(_TW).strftime("%Y_%m")
    tick_path = _ROOT / f"db/tick/{ym}.parquet"
    if not tick_path.exists():
        print(f"  db/tick/{ym}.parquet 不存在，略過今天的原始收盤更新")
        return
    date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    tick_df = ds.dataset(str(tick_path), format="parquet").to_table(columns=["stock_id", "date", "deal_price"]).to_pandas()
    tick_df = tick_df[tick_df["date"].astype(str).str[:10] == date_str]
    if tick_df.empty:
        print("  今天沒有 tick 資料，略過原始收盤更新")
        return
    close = _raw_close_from_tick(tick_df)
    _save_raw_close(close)
    print(f"  今天（{date_str}）原始收盤已更新，共 {len(close)} 支")


def backfill_stock(stock_id: str, start: str, end: str, token: str = None, fubon_sdk_holder: dict | None = None):
    """單一股票回補，查完直接落地寫入（不是全部股票查完才一次寫，避免長時間
    執行中途失敗整批白跑）。

    fubon_sdk_holder：選填，{"sdk": None} 這種容器，Fugle查無資料時第一次
    需要富邦備援才會登入、寫進這個容器讓後續股票（含另一條thread）重用同一個
    session，不用每支都重新登入。傳 None 就完全不 fallback 富邦（只用 Fugle）。"""
    raw = _fetch_raw_day_close(stock_id, start, end, token=token)
    if raw.empty and fubon_sdk_holder is not None:
        print(f"  {stock_id}：Fugle 查無資料，改查富邦...")
        with _fubon_lock:
            if fubon_sdk_holder.get("sdk") is None:
                from fubon import fubon_api as trade_api

                sdk, _ = trade_api.login()
                trade_api.init_market_data(sdk)
                fubon_sdk_holder["sdk"] = sdk
            sdk = fubon_sdk_holder["sdk"]
        raw = _fetch_raw_day_close_fubon(sdk, stock_id, start, end)
    if raw.empty:
        print(f"  {stock_id}：查無原始日K資料，略過")
        return
    _save_raw_close(raw)


def _backfill_group(stocks: list, start: str, end: str, token: str, label: str, fubon_sdk_holder: dict):
    for i, stock_id in enumerate(stocks, 1):
        try:
            backfill_stock(stock_id, start, end, token=token, fubon_sdk_holder=fubon_sdk_holder)
        except Exception as e:
            print(f"  [{label} {i}/{len(stocks)}] {stock_id} 失敗: {e}")
        if i % 20 == 0 or i == len(stocks):
            print(f"  [{label} {i}/{len(stocks)}] 進度更新")


def main(stock_ids: list[str], start: str, end: str | None = None):
    end = end or datetime.now(_TW).strftime("%Y-%m-%d")
    fubon_sdk_holder: dict = {"sdk": None}

    if not fugle_api.TOKEN_DAYTRADE:
        print("警告：沒有設定第二組 Fugle 帳號 (FUGLE_DAYTRADE)，改單執行緒跑")
        _backfill_group(stock_ids, start, end, None, "Fugle", fubon_sdk_holder)
    else:
        half = len(stock_ids) // 2
        group1, group2 = stock_ids[:half], stock_ids[half:]
        t1 = threading.Thread(target=_backfill_group, args=(group1, start, end, None, "Fugle", fubon_sdk_holder))
        t2 = threading.Thread(
            target=_backfill_group, args=(group2, start, end, fugle_api.TOKEN_DAYTRADE, "Fugle-DT", fubon_sdk_holder)
        )
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    if fubon_sdk_holder.get("sdk") is not None:
        from fubon import fubon_api as trade_api

        trade_api.logout(fubon_sdk_holder["sdk"])
    print("\n全部完成 ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一次性回補 db/day_raw_close（原始日收盤，拆股/合股偵測用）")
    parser.add_argument("stocks", nargs="*", help="指定股票代號（不指定則用 tick_universe 400支）")
    parser.add_argument("--start", type=str, default=_DEFAULT_START, help=f"起始日期 (預設 {_DEFAULT_START})")
    parser.add_argument("--end", type=str, default=None, help="結束日期（預設今天）")
    args = parser.parse_args()

    if args.stocks:
        target_stocks = args.stocks
    else:
        from finmind.tick_universe import load_tick_universe

        target_stocks = load_tick_universe()

    print(f"回補 {len(target_stocks)} 支股票，{args.start} ~ {args.end or '今天'}...")
    main(target_stocks, args.start, args.end)
