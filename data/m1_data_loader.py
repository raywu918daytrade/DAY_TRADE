"""
歷史分K 下載器（訓練資料用）

功能：
    從 Fugle REST API 下載 1 分鐘 K 線，存入 db/m1/（訓練資料庫）。
    flag 機制避免同一支股票在同一天重複下載。

與 m1_rest.py 的差異：
    m1_data_loader.py → 下載歷史分K（近30日），存 db/m1/，給訓練用，GHA 每日觸發
    m1_rest.py        → 盤中每分鐘 poll，存 db/m1_live/，給當天交易推論用

主要函式：
    update_m1(stocks)
        循序下載（2.1s/支），避免超過 API rate limit（60 req/min）。
        stocks 預設為 fugle_stocks() 所有 ~2787 支。
"""
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from data.fugle_tickers import fugle_stocks

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_TW = timezone(timedelta(hours=8))

token = os.environ.get("FUGLE", "")

_BASE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"
_FLAG_PATH = _ROOT / "db/m1_flags/m1_flag.parquet"


def _m1_file_path(date: pd.Timestamp) -> Path:
    return _ROOT / f"db/m1/{date.year}_{date.month}.parquet"


def _atomic_to_parquet(df: pd.DataFrame, file_path: str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    tmp_path = f"{file_path}.tmp"
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def _download_m1(stock_id: str) -> pd.DataFrame:
    """取得近30日1分鐘K線（Fugle分K無法指定 from/to，一律回傳近30日資料）"""
    params = {"timeframe": "1", "fields": "open,high,low,close,volume", "sort": "asc", "adjusted": "true"}
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
    date = pd.to_datetime(df["date"])
    if date.dt.tz is not None:
        date = date.dt.tz_convert(_TW).dt.tz_localize(None)
    df["date"] = date.dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _download_m1_today(stock_id: str) -> pd.DataFrame:
    """取得今日盤中1分鐘K線（/intraday/candles，即時，約1分鐘delay）"""
    params = {"timeframe": "1"}
    headers = {"X-API-KEY": token}
    r = requests.get(f"{_BASE_URL}/intraday/candles/{stock_id}", params=params, headers=headers, timeout=10)
    if r.status_code == 429:
        time.sleep(float(r.headers.get("Retry-After", 60)) + 1)
        r = requests.get(f"{_BASE_URL}/intraday/candles/{stock_id}", params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if "data" not in data or not data["data"]:
        return pd.DataFrame()

    df = pd.DataFrame(data["data"])
    df["stock_id"] = stock_id
    date = pd.to_datetime(df["date"])
    if date.dt.tz is not None:
        date = date.dt.tz_convert(_TW).dt.tz_localize(None)
    df["date"] = date.dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _save_m1(new_df: pd.DataFrame):
    new_df = new_df[["stock_id", "date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close"]:
        new_df[col] = new_df[col].astype("float32")
    new_df["volume"] = new_df["volume"].astype("int64")

    for month, group in new_df.groupby(pd.to_datetime(new_df["date"]).dt.to_period("M")):
        file_path = _m1_file_path(month.to_timestamp())
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


def update_m1(stocks: list = None):
    """1分鐘K線（Fugle，僅能取得近30日資料），flag避免同日重複下載"""
    if not token:
        raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")

    now = datetime.now(_TW)
    date_str = now.strftime("%Y-%m-%d")
    os.makedirs(_ROOT / "db/m1", exist_ok=True)

    candidates = _all_stocks() if stocks is None else stocks
    done = _get_done_stocks(date_str)
    wait_stocks = [s for s in candidates if s not in done]
    print("還有", len(wait_stocks), "個股票未更新（已排除今日已下載 flag）")
    for stock_id in wait_stocks:
        try:
            hist = _download_m1(stock_id)        # historical/candles：近30日（至昨日）
            today = _download_m1_today(stock_id) # intraday/candles：今日盤中
            df = pd.concat([hist, today], ignore_index=True) if not today.empty else hist
            if not df.empty:
                _save_m1(df)
                print(stock_id, "下載完成", len(df), "筆")
            else:
                print(stock_id, "無資料")
            _update_flag(stock_id, date_str)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(stock_id, "Fugle 無此股票資料，標記跳過")
                _update_flag(stock_id, date_str)
            else:
                print(f"{stock_id} 失敗: {e}")
        except Exception as e:
            print(f"{stock_id} 失敗: {e}")
        time.sleep(2.1)  # 每支股票 2 次 API，維持在 60 req/min 以內

    print("全部完成")


if __name__ == "__main__":
    update_m1()
