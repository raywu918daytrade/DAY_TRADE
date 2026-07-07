"""
本地資料讀取工具（唯讀）

功能：
    從本地 parquet 檔載入三種資料，供訓練腳本與即時推論使用。
    不負責下載或更新資料，只做讀取。

三種資料對應：
    load_m1()      → db/m1/        歷史分K（訓練用，2787支，由 GHA 每日更新）
    load_day()     → db/fugle_day/ 日K（模型特徵用，GHA 每日更新）
    load_m1_live() → db/m1_live/   今日即時分K（交易用，500支，收盤後丟棄）
"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

_ROOT = Path(__file__).parent.parent


def load_m1() -> pd.DataFrame:
    """載入 db/m1/ 全部歷史分K（訓練資料，~2787 支，按月分檔）"""
    df = ds.dataset(str(_ROOT / "db/m1"), format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_day() -> pd.DataFrame:
    """載入 db/fugle_day/ 全部日K（模型特徵用，按月分檔）"""
    df = ds.dataset(str(_ROOT / "db/fugle_day"), format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


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
