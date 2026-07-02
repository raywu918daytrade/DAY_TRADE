from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

_ROOT = Path(__file__).parent.parent


def load_m1() -> pd.DataFrame:
    df = ds.dataset(str(_ROOT / "db/m1"), format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_day() -> pd.DataFrame:
    df = ds.dataset(str(_ROOT / "db/fugle_day"), format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m1_live(date: str = None) -> pd.DataFrame:
    """載入當日或指定日期的 WebSocket 即時分K"""
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = _ROOT / f"db/m1_live/{date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)
