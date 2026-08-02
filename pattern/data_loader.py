"""
K線資料載入器 (Data Loader for Pattern Detection)

支援時間週期：
- "1m"  : 1 分鐘 K 線 (讀 db/m1/ 或 db/m1_live/)
- "3m"  : 3 分鐘標準獨立 K 線 (讀 db/m3_std/ 或由 1m resample)
- "5m"  : 5 分鐘標準獨立 K 線 (讀 db/m5_std/ 或由 1m resample)
- "day" : 日 K 線 (讀 db/adjustment_day/)

價格基準（2026-08-01加，2026-08-03改）：pattern 系列是系統裡唯一需要「完整
還原（含一般除權息）」基準的地方——除息造成的真實跳空會讓型態偵測的轉折點
判斷誤判（實測數字見 data/adjustment_query.py 檔頭說明），跟系統預設的
「只還原拆股/合股」基準（data/query.py）不一樣。這支檔案不再直接戳
db/m1／db/fugle_day 路徑，統一透過 data/raw_query.py（讀原始資料）+
data/adjustment_query.py（套用 pattern 專用完整還原係數）取得資料，
不自己維護一套 factor 換算邏輯。db/m1_live（當天即時資料）不用調整——
「今天」相對於自己必然是同一個基準，raw=adjusted。
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import pyarrow.dataset as ds

from data import raw_query
from data.adjustment_query import load_pattern_day, load_pattern_day_by_stock, _load_adjustment_factor
from data.resample import compute_m3_std, compute_m5_std

_ROOT = Path(__file__).parent.parent


def _apply_pattern_adjust_factor(df: pd.DataFrame, stock_id: Optional[str] = None) -> pd.DataFrame:
    """把 df 的 open/high/low/close 換算成 pattern 專用完整還原後價格。df 需有
    stock_id、date（datetime）欄位；stock_id 參數選填，單一股票查詢時帶入可以
    讓 _load_adjustment_factor() 走 pyarrow filter pushdown，不用載入其他股票的
    factor。"""
    if df.empty:
        return df
    df = df.copy()
    df["_day"] = df["date"].dt.strftime("%Y-%m-%d")
    start = df["_day"].min()
    factor_df = _load_adjustment_factor(stock_id, None, start)
    if factor_df.empty:
        return df.drop(columns=["_day"])
    df = df.merge(
        factor_df[["stock_id", "date", "factor"]].rename(columns={"date": "_day"}),
        on=["stock_id", "_day"],
        how="left",
    )
    df["factor"] = df["factor"].fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = (df[col] * df["factor"]).round(2).astype("float32")
    return df.drop(columns=["_day", "factor"])


def get_stock_candles(
    stock_id: str,
    timeframe: str = "day",
    date: Optional[str] = None,
    limit: int = 120,
) -> pd.DataFrame:
    """載入單一股票指定時間週期的最近 N 根 K 線。

    Args:
        stock_id: 股票代號 (例如 "2330")
        timeframe: "1m", "3m", "5m", "day"
        date: 基準日期 "YYYY-MM-DD" (選填)。不填則抓最新可用的 K 線。
        limit: 最多抓取的 K 線根數，預設 120 根。

    Returns:
        包含 columns: ["stock_id", "date", "open", "high", "low", "close", "volume"]
        按 date 遞增排序。
    """
    if limit is not None and not isinstance(limit, int):
        try:
            limit = int(limit)
        except Exception:
            limit = 120
    if timeframe == "day":
        df = load_pattern_day_by_stock(stock_id, date=None)
        if df.empty:
            return df
        if date:
            df = df[df["date"] <= f"{date} 23:59:59"]
        if limit and len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        return df

    elif timeframe == "1m":
        # 先試 db/m1_live
        if date:
            live_path = _ROOT / f"db/m1_live/{date}.parquet"
            if live_path.exists():
                df_live = pd.read_parquet(live_path)
                df_live = df_live[df_live["stock_id"] == stock_id]
                if not df_live.empty:
                    df_live["date"] = pd.to_datetime(df_live["date"])
                    df_live = df_live.sort_values("date").reset_index(drop=True)
                    if limit and len(df_live) > limit:
                        df_live = df_live.iloc[-limit:].reset_index(drop=True)
                    return df_live

        dataset_dir = _ROOT / "db/m1"
        if not dataset_dir.exists():
            return pd.DataFrame()
        dataset = ds.dataset(str(dataset_dir), format="parquet")
        filt = ds.field("stock_id") == stock_id
        if date:
            filt = filt & (ds.field("date") <= f"{date} 23:59:59")
        table = dataset.to_table(filter=filt)
        if table.num_rows == 0:
            return pd.DataFrame()
        df = table.to_pandas()
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df.drop_duplicates(subset=["date"], keep="last", inplace=True)
        df = df.sort_values("date").reset_index(drop=True)
        df = _apply_pattern_adjust_factor(df, stock_id=stock_id)
        if limit and len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        return df

    elif timeframe in ("3m", "5m"):
        std_dir = _ROOT / f"db/{timeframe}_std"
        if std_dir.exists():
            dataset = ds.dataset(str(std_dir), format="parquet")
            filt = ds.field("stock_id") == stock_id
            if date:
                filt = filt & (ds.field("date") <= f"{date} 23:59:59")
            table = dataset.to_table(filter=filt)
            if table.num_rows > 0:
                df = table.to_pandas()
                df["date"] = pd.to_datetime(df["date"], format="mixed")
                df.drop_duplicates(subset=["date"], keep="last", inplace=True)
                df = df.sort_values("date").reset_index(drop=True)
                df = _apply_pattern_adjust_factor(df, stock_id=stock_id)
                if limit and len(df) > limit:
                    df = df.iloc[-limit:].reset_index(drop=True)
                return df

        # Fallback: 從 1m 現算 resample
        df_1m = get_stock_candles(stock_id, timeframe="1m", date=date, limit=limit * 5)
        if df_1m.empty:
            return pd.DataFrame()
        if timeframe == "3m":
            df_res = compute_m3_std(df_1m)
        else:
            df_res = compute_m5_std(df_1m)
        if limit and len(df_res) > limit:
            df_res = df_res.iloc[-limit:].reset_index(drop=True)
        return df_res

    return pd.DataFrame()


def get_all_stocks_candles(
    timeframe: str = "day",
    date: Optional[str] = None,
    limit: int = 120,
    start_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """批次載入全市場所有股票在指定時間週期的最近 N 根 K 線。

    Returns:
        Dict[stock_id, pd.DataFrame]
    """
    if limit is not None and not isinstance(limit, int):
        try:
            limit = int(limit)
        except Exception:
            limit = 120
    if timeframe == "day":
        ref_date = date if (date and isinstance(date, str)) else "today"
        cutoff = start_date or (pd.Timestamp(ref_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        df = load_pattern_day(start_date=cutoff)
        if df.empty:
            return {}
        if date and isinstance(date, str):
            df = df[df["date"] <= f"{date} 23:59:59"]

    elif timeframe == "1m":
        # 先看當日即時
        if date:
            live_path = _ROOT / f"db/m1_live/{date}.parquet"
            if live_path.exists():
                df = pd.read_parquet(live_path)
                df["date"] = pd.to_datetime(df["date"])
            else:
                dir_path = _ROOT / "db/m1"
                cutoff = date[:7].replace("-", "_")
                files = sorted(f for f in dir_path.iterdir() if f.suffix == ".parquet" and f.stem >= cutoff)
                if not files:
                    files = sorted(f for f in dir_path.iterdir() if f.suffix == ".parquet")[-1:]
                if not files:
                    return {}
                df = ds.dataset([str(f) for f in files], format="parquet").to_table().to_pandas()
                df["date"] = pd.to_datetime(df["date"], format="mixed")
                df = df[df["date"] <= f"{date} 23:59:59"]
                df = _apply_pattern_adjust_factor(df)
        else:
            dir_path = _ROOT / "db/m1"
            files = sorted(f for f in dir_path.iterdir() if f.suffix == ".parquet")[-1:]
            if not files:
                return {}
            df = ds.dataset([str(f) for f in files], format="parquet").to_table().to_pandas()
            df["date"] = pd.to_datetime(df["date"], format="mixed")
            df = _apply_pattern_adjust_factor(df)

    elif timeframe in ("3m", "5m"):
        std_dir = _ROOT / f"db/{timeframe}_std"
        if std_dir.exists():
            cutoff_file = date[:7].replace("-", "_") if date else ""
            files = sorted(f for f in std_dir.iterdir() if f.suffix == ".parquet" and (not cutoff_file or f.stem >= cutoff_file))
            if not files:
                files = sorted(f for f in std_dir.iterdir() if f.suffix == ".parquet")[-1:]
            if files:
                df = ds.dataset([str(f) for f in files], format="parquet").to_table().to_pandas()
                df["date"] = pd.to_datetime(df["date"], format="mixed")
                if date:
                    df = df[df["date"] <= f"{date} 23:59:59"]
                df = _apply_pattern_adjust_factor(df)
            else:
                m1_candles = get_all_stocks_candles("1m", date=date, limit=limit * 5)
                res = {}
                for sid, m1_df in m1_candles.items():
                    res_df = compute_m3_std(m1_df) if timeframe == "3m" else compute_m5_std(m1_df)
                    if not res_df.empty:
                        res[sid] = res_df.iloc[-limit:].reset_index(drop=True)
                return res
        else:
            m1_candles = get_all_stocks_candles("1m", date=date, limit=limit * 5)
            res = {}
            for sid, m1_df in m1_candles.items():
                res_df = compute_m3_std(m1_df) if timeframe == "3m" else compute_m5_std(m1_df)
                if not res_df.empty:
                    res[sid] = res_df.iloc[-limit:].reset_index(drop=True)
            return res
    else:
        return {}

    # 按 stock_id 分組裁切至 limit 根
    result = {}
    for sid, group in df.groupby("stock_id"):
        group = group.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        if limit and len(group) > limit:
            group = group.iloc[-limit:]
        result[str(sid)] = group.reset_index(drop=True)

    return result


def get_latest_candle_timestamp(timeframe: str = "day", date: Optional[str] = None) -> str:
    """取得特定 timeframe 及 date 設定下最新一根 K 線的時間戳記，作為 Cache Key 版本識別。"""
    try:
        sample = get_stock_candles("2330", timeframe=timeframe, date=date, limit=1)
        if not sample.empty:
            return str(sample["date"].iloc[-1])
    except Exception:
        pass
    return date or "latest"


def get_stocks_10d_avg_vol_lots(date: Optional[str] = None) -> Dict[str, float]:
    """從 db/adjustment_day/ 計算全市場各股票近 10 個交易日的平均成交量 (單位: 張 = 1000 股)。

    Returns:
        Dict[stock_id, float] (例如 {"2330": 28450.5})
    """
    ref_date = date if (date and isinstance(date, str)) else "today"
    cutoff = (pd.Timestamp(ref_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    df = load_pattern_day(start_date=cutoff)
    if df.empty:
        return {}
    if date and isinstance(date, str):
        df = df[df["date"] <= f"{date} 23:59:59"]

    avg_vols = {}
    for sid, group in df.groupby("stock_id"):
        recent10 = group.drop_duplicates(subset=["date"], keep="last").sort_values("date").tail(10)
        if not recent10.empty:
            avg_shares = float(recent10["volume"].mean())
            avg_vols[str(sid)] = round(avg_shares / 1000.0, 1)

    return avg_vols
