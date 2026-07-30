"""FinMind Tick（TaiwanStockPriceTick）API 封裝與存檔邏輯 — 跟
finmind/finmind_api.py（分K/TaiwanStockKBar）刻意分開成獨立檔案，不要把tick
專屬的東西塞進 finmind_api.py 跟分K的函式混在一起，改分K時才不用擔心不小心
影響到tick、反之亦然。

底層帳號/流量相關機制（rate limiter、FINMIND_TOKEN、check_quota()、
FatalAPIError 錯誤分類體系、_atomic_to_parquet()）繼續從 finmind_api.py
import 共用，不重寫一份——這些是同一個帳號/同一組FinMind API規則，重寫一份
等於同樣的bug要修兩次，也失去額度管理的唯一真相來源。

用法：
    from finmind.tick_api import fetch_tick_day, load_tick_day, ...
    （這支檔案本身不是可執行入口，回補流程見 finmind/backfill_tick_history.py）
"""

from pathlib import Path

import pandas as pd

from finmind.finmind_api import _ROOT, _atomic_to_parquet, _fetch_finmind_day

_TICK_DATASET = "TaiwanStockPriceTick"


async def fetch_tick_day(session, stock_id: str, date_str: str, _retry: int = 0) -> pd.DataFrame:
    """單一股票、單一交易日的逐筆成交明細（FinMind TaiwanStockPriceTick）。
    date_str 格式 YYYY-MM-DD。

    2026-07-29 實測（data_id=2330, start_date=2026-07-28）回傳欄位：date
    （純日期，不含時間）、stock_id、deal_price、volume（單筆成交量，不是
    當日總量）、Time（"HH:MM:SS.ffffff"，跟分K的"minute"欄位不同，分K的
    minute已經是HH:MM、這裡是獨立欄位要自己併回date）、TickType（字串
    "0"/"1"/"2"：0=無法判定、1=外盤/買方主動、2=內盤/賣方主動）。跟分K
    一樣「單日單股票一次request、不支援start~end區間」。

    回傳欄位比照 db/tick 的 schema：stock_id, date("YYYY-MM-DD
    HH:MM:SS.ffffff")、deal_price、volume、tick_type（把 FinMind 的
    TickType 欄名改成小寫底線風格，跟其他欄位一致）。當天沒有資料回傳空
    DataFrame，跟 finmind_api.fetch_kbar_day() 一樣的處理方式。
    request/timeout/retry/錯誤分類邏輯見 finmind_api._fetch_finmind_day()
    （共用，不是這支檔案自己的邏輯）。
    """
    rows = await _fetch_finmind_day(session, _TICK_DATASET, stock_id, date_str, _retry=_retry)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = df["date"] + " " + df["Time"]
    return df.drop(columns=["Time"]).rename(columns={"TickType": "tick_type"})


def tick_file_path(year: int, month: int) -> Path:
    """跟 finmind_api._m1_file_path() 同一套命名（月份補零），存
    TaiwanStockPriceTick 逐筆成交明細，跟 db/m1 分開存在 db/tick/，欄位
    schema不同（見 save_tick()）不能混。"""
    return _ROOT / f"db/tick/{year}_{month:02d}.parquet"


def save_tick(new_df: pd.DataFrame, year: int, month: int):
    """合併進 db/tick/{year}_{month}.parquet，跟現有資料 dedupe（keep="last"，
    邏輯跟 finmind_api._save_m1() 一致）。tick_type 轉成 int8（0/1/2）而不是
    保留 FinMind 原本的字串——這是無損轉換（值域固定就3種），存數字比存
    字串省空間，之後篩選買賣方向（tick_type==2）也比字串比對快。"""
    new_df = new_df[["stock_id", "date", "deal_price", "volume", "tick_type"]].copy()
    new_df["deal_price"] = new_df["deal_price"].astype("float32")
    new_df["volume"] = new_df["volume"].astype("int64")
    new_df["tick_type"] = new_df["tick_type"].astype("int8")

    file_path = tick_file_path(year, month)
    if file_path.exists():
        old_df = pd.read_parquet(file_path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    new_df.sort_values(["date", "stock_id"], inplace=True)
    _atomic_to_parquet(new_df, file_path, index=False, compression="zstd")


def tick_empty_file_path(year: int, month: int) -> Path:
    """記錄 TaiwanStockPriceTick 確認過『真的沒有資料』的 (股票,交易日) 組合，
    跟 finmind_api._m1_empty_file_path() 同樣的理由，分開存在
    db/tick_empty/，不跟 db/tick 混。"""
    return _ROOT / f"db/tick_empty/{year}_{month:02d}.parquet"


def save_empty_tick_pairs(pairs: list[tuple[str, str]], year: int, month: int):
    """合併進 db/tick_empty/{year}_{month}.parquet（見 tick_empty_file_path()
    說明），跟現有資料 dedupe，邏輯跟 finmind_api._save_empty_pairs() 一致。"""
    new_df = pd.DataFrame(pairs, columns=["stock_id", "date"])
    file_path = tick_empty_file_path(year, month)
    if file_path.exists():
        old_df = pd.read_parquet(file_path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.drop_duplicates(inplace=True)
    new_df.sort_values(["date", "stock_id"], inplace=True)
    _atomic_to_parquet(new_df, file_path, index=False, compression="zstd")


def existing_tick_pairs(year: int, month: int) -> set[tuple[str, str]]:
    """db/tick 該月檔案裡已經有的 (stock_id, 日期) 組合，加上 db/tick_empty
    裡確認過『真的沒有資料』的組合，兩者都算『已處理過』，邏輯跟
    finmind_api._existing_pairs() 一致——db/tick 的 date 欄位帶微秒
    ("YYYY-MM-DD HH:MM:SS.ffffff")，pd.to_datetime() 一樣能正確解析、
    .dt.strftime() 照樣只取日期部分，不用特殊處理。"""
    pairs: set[tuple[str, str]] = set()
    path = tick_file_path(year, month)
    if path.exists():
        df = pd.read_parquet(path, columns=["stock_id", "date"])
        days = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        pairs |= set(zip(df["stock_id"], days))
    empty_path = tick_empty_file_path(year, month)
    if empty_path.exists():
        edf = pd.read_parquet(empty_path, columns=["stock_id", "date"])
        pairs |= set(zip(edf["stock_id"], edf["date"]))
    return pairs
