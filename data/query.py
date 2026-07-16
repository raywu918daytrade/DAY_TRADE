"""
本地資料讀取工具（唯讀）

功能：
    從本地 parquet 檔載入三種資料，供訓練腳本與即時推論使用。
    不負責下載或更新資料，只做讀取。

三種資料對應：
    load_m1()      → db/m1/        歷史分K（訓練用，2787支，由 GHA 每日更新）
    load_day()     → db/fugle_day/ 日K（模型特徵用，GHA 每日更新）
    load_m1_live() → db/m1_live/   今日即時分K（交易用，500支，收盤後丟棄）

單支股票查詢（用 pyarrow filter pushdown，不用像 load_day() 整個資料集讀進記憶體）：
    load_day_by_stock(stock_id) → db/fugle_day/ 單一股票的全部日K
"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

_ROOT = Path(__file__).parent.parent


def load_m1() -> pd.DataFrame:
    """載入 db/m1/ 全部歷史分K（訓練資料，~2787 支，按月分檔）"""
    df = ds.dataset(str(_ROOT / "db/m1"), format="parquet").to_table().to_pandas()
    # 按月分檔的 parquet 中 date 欄位型別可能不一致（string / timestamp 混雜），
    # 用 format="mixed" 讓 pandas 逐筆判斷格式，避免鎖死單一格式時遇到例外格式就整批炸掉
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_day() -> pd.DataFrame:
    """載入 db/fugle_day/ 全部日K（模型特徵用，按月分檔）"""
    df = ds.dataset(str(_ROOT / "db/fugle_day"), format="parquet").to_table().to_pandas()
    # 同上：按月分檔可能混雜不同型別的 date 欄位，用 format="mixed" 容忍
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_day_by_stock(stock_id: str, date: str = None) -> pd.DataFrame:
    """載入單一股票在 db/fugle_day/ 的日K，只讀該股票的 row group，不用像
    load_day() 一樣把全市場都讀進記憶體，適合只需要單支股票時用（例如查前一
    交易日收盤價）。

    date: 選填，格式 "YYYY-MM-DD"，指定只回傳該日那一筆（同樣走 pyarrow filter
    pushdown，不用先讀全部再篩）；不填則回傳該股票全部日K（依日期排序）。
    查無資料一律回傳空 DataFrame（欄位跟 load_day() 一致）。"""
    dataset = ds.dataset(str(_ROOT / "db/fugle_day"), format="parquet")
    filt = ds.field("stock_id") == stock_id
    if date is not None:
        filt = filt & (ds.field("date") == date)
    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
    df = table.to_pandas()
    # 同 load_day()：按月分檔可能混雜不同型別的 date 欄位，用 format="mixed" 容忍
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    return df.sort_values("date").reset_index(drop=True)


def load_m3() -> pd.DataFrame:
    """載入 db/m3/ 全部 3 分鐘K（由 build_m3_m5.py 預先聚合）"""
    path = _ROOT / "db/m3"
    if not path.exists():
        return pd.DataFrame()
    df = ds.dataset(str(path), format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m5() -> pd.DataFrame:
    """載入 db/m5/ 全部 5 分鐘K（由 build_m3_m5.py 預先聚合）"""
    path = _ROOT / "db/m5"
    if not path.exists():
        return pd.DataFrame()
    df = ds.dataset(str(path), format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m1_live(date: str = None) -> pd.DataFrame:
    """載入今日即時分K（db/m1_live/YYYY-MM-DD.parquet），盤後自動 backfill 補齊"""
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = _ROOT / f"db/m1_live/{date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)
