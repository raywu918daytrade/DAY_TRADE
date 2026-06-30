from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import threading
import time

import pandas as pd
import pyarrow.dataset as ds
import requests
from dotenv import load_dotenv

from tay_trade.fugle_tickers import fugle_stocks

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_TW = timezone(timedelta(hours=8))

token = os.environ.get("FUGLE", "")

_BASE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"
_FLAG_PATH = _ROOT / "db/fugle_day_flags/day_flag.parquet"


def _day_file_path(date: str) -> Path:
    """依資料日期決定存入哪個月份檔"""
    ts = pd.Timestamp(date)
    return _ROOT / f"db/fugle_day/{ts.year}_{ts.month}.parquet"


def _last_stored_date() -> str | None:
    """掃 db/fugle_day/ 全部檔案，找最大的已存日期（跨月正確）"""
    day_dir = _ROOT / "db/fugle_day"
    if not day_dir.exists():
        return None
    dataset = ds.dataset(str(day_dir), format="parquet")
    if dataset.count_rows() == 0:
        return None
    return dataset.to_table(columns=["date"]).column("date").to_pandas().max()


def _atomic_to_parquet(df: pd.DataFrame, file_path: str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    tmp_path = f"{file_path}.tmp"
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def _download_day(stock_id: str, start_date: str) -> pd.DataFrame:
    now = datetime.now(_TW)
    params = {
        "from": start_date,
        "to": now.strftime("%Y-%m-%d"),
        "fields": "open,high,low,close,volume",
        "sort": "asc",
        "adjusted": "true",
    }
    headers = {"X-API-KEY": token}
    r = requests.get(f"{_BASE_URL}/historical/candles/{stock_id}", params=params, headers=headers, timeout=10)
    if r.status_code == 429:
        time.sleep(float(r.headers.get("Retry-After", 60)) + 1)
        r = requests.get(f"{_BASE_URL}/historical/candles/{stock_id}", params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if "data" not in data or not data["data"]:
        return pd.DataFrame()

    df = pd.DataFrame(data["data"])
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _save_day(new_df: pd.DataFrame):
    new_df = new_df[["stock_id", "date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close"]:
        new_df[col] = new_df[col].astype("float32")
    new_df["volume"] = new_df["volume"].astype("int64")

    for month, group in new_df.groupby(pd.to_datetime(new_df["date"]).dt.to_period("M")):
        file_path = _day_file_path(str(month.to_timestamp().date()))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            for attempt in range(3):
                try:
                    old_df = pd.read_parquet(file_path)
                    group = pd.concat([old_df, group], ignore_index=True)
                    break
                except Exception:
                    if attempt == 2:
                        print(f"警告：{file_path} 讀取失敗3次（可能是 Dropbox 同步中），略過 merge")
                    else:
                        time.sleep(1)
        group.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
        group.sort_values(["date", "stock_id"], inplace=True)
        _atomic_to_parquet(group, file_path, index=False, compression="zstd")


def _update_flag(stock_id: str, date_str: str):
    new_row = pd.DataFrame([{"stock_id": stock_id, "date": date_str}])
    if os.path.exists(_FLAG_PATH):
        df = pd.read_parquet(_FLAG_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
        df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    else:
        df = new_row
    os.makedirs(os.path.dirname(_FLAG_PATH), exist_ok=True)
    _atomic_to_parquet(df, _FLAG_PATH, index=False)


def _all_stocks() -> list:
    """從 Fugle 當日清單取得可交易股票"""
    return fugle_stocks()


def _get_done_stocks(date_str: str) -> set:
    if not os.path.exists(_FLAG_PATH):
        return set()
    df = pd.read_parquet(_FLAG_PATH)
    return set(df[df["date"] == date_str]["stock_id"].tolist())


def update_day(start_date: str = None, stocks: list = None, workers: int = 5):
    """
    日線（Fugle），flag 避免同日重複下載。
    workers: 並發執行緒數（預設 5，約 5 分鐘下載 1500 支）
    """
    if not token:
        raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    now = datetime.now(_TW)
    date_str = now.strftime("%Y-%m-%d")
    os.makedirs(_ROOT / "db/fugle_day", exist_ok=True)

    if start_date is None:
        last = _last_stored_date()
        start_date = last if last else f"{now.year}-{now.month:02d}-01"

    candidates = _all_stocks() if stocks is None else stocks
    done = _get_done_stocks(date_str)
    wait_stocks = [s for s in candidates if s not in done]
    print(f"start_date={start_date}，{len(wait_stocks)} 支待下載（{workers} 並發）")

    _save_lock = threading.Lock()
    completed = 0

    def _fetch_one(stock_id: str):
        nonlocal completed
        time.sleep(0.2)  # 每執行緒小延遲，5 執行緒合計 ~1 req/s
        try:
            df = _download_day(stock_id, start_date)
            if not df.empty:
                with _save_lock:
                    _save_day(df)
            _update_flag(stock_id, date_str)
            return stock_id, len(df), None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                _update_flag(stock_id, date_str)
                return stock_id, 0, None
            return stock_id, 0, str(e)
        except Exception as e:
            return stock_id, 0, str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, sid): sid for sid in wait_stocks}
        for fut in as_completed(futures):
            sid, rows, err = fut.result()
            completed += 1
            if err:
                print(f"  [{completed}/{len(wait_stocks)}] {sid} 失敗: {err}")
            elif completed % 100 == 0 or completed == len(wait_stocks):
                print(f"  [{completed}/{len(wait_stocks)}] 進度更新")

    print("日K 下載完成")


if __name__ == "__main__":
    update_day()
