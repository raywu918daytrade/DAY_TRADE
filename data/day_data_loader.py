"""
日K 資料下載器（Fugle + 富邦 historical/candles）

功能：
    從 Fugle、富邦 REST API 下載股票日K，按月份分檔存入 db/fugle_day/。
    flag 機制避免同一支股票在同一天重複下載。

Fugle + 富邦同時下載：
    待下載清單拆一半，Fugle 那一半沿用原本的 ThreadPoolExecutor（多執行緒併發，
    429 交給 Retry-After 被動重試）；富邦那一半用單一執行緒依序下載（比照
    data/m1_data_loader.py 的作法，1.05秒/支留在 60次/分鐘以內）。富邦
    historical/candles 不帶 timeframe 就是日K，from/to 用法跟 Fugle 一致
    （同一套底層 fugle_marketdata 元件，2026-07-13 實測過），富邦 SDK 呼叫都透過
    fubon/fubon_api.py，Fugle REST 呼叫都透過 fugle/fugle_api.py，這裡不直接
    碰 fubon_neo 或組 Fugle 的 URL/header。

主要函式：
    update_day(start_date, stocks, workers)
        增量下載指定標的的日K，多執行緒並發，5 workers 約 5 分鐘下載 1500 支
        （這是 Fugle 那一半的預估，富邦那一半另外跑）。

資料格式（db/fugle_day/YYYY_M.parquet）：
    stock_id, date(str), open, high, low, close(float32), volume(int64)
"""

from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import threading
import time

import pandas as pd
import pyarrow.dataset as ds
import requests
from dotenv import load_dotenv

from fugle import fugle_api

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))

_FLAG_PATH = _ROOT / "db/fugle_day_flags/day_flag.parquet"

# _save_day()/_update_flag() 都是「讀舊檔→merge→寫回去」，Fugle 執行緒池、富邦
# 執行緒同時呼叫會搶同一個檔案（比照 data/m1_data_loader.py 的作法），這裡直接
# 用鎖序列化。
_save_lock = threading.Lock()
_flag_lock = threading.Lock()


def _day_file_path(date: str) -> Path:
    """依資料日期決定存入哪個月份檔（月份補零，需與 push_day_to_hf.py 的
    pd.Period.astype(str) 命名一致，否則同一個月會在本機/HF Hub 各自產生
    一個檔名不同的分檔，觸發 schema 衝突）"""
    ts = pd.Timestamp(date)
    return _ROOT / f"db/fugle_day/{ts.year}_{ts.month:02d}.parquet"


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
    import tempfile

    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    # 用系統 tmp 目錄避免 Dropbox 干擾 rename
    with tempfile.NamedTemporaryFile(dir=dir_path, suffix=".tmp", delete=False) as f:
        tmp_path = f.name
    try:
        df.to_parquet(tmp_path, **kwargs)
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _fetch_year(stock_id: str, from_date: str, to_date: str, token: str = None) -> pd.DataFrame:
    """單次請求（區間 < 1 年）。404 代表這支股票在這段區間還沒有資料（例如
    尚未上市／尚未開始交易），回傳空 DataFrame，不當例外處理——2026-07-15
    實測：對現在才被列入候選清單、但更早以前還沒上市的股票回補歷史時很常見
    （例如某支股票2016年404、2023年正常有資料），如果讓它往上拋例外，
    _download_day() 的年度迴圈會被中斷，之後年份（明明有資料）也不會再嘗試，
    整支股票這次直接算失敗。例外只保留給真正的請求失敗（限流、逾時、其他
    錯誤）。"""
    r = fugle_api.historical_candles(
        stock_id,
        token=token,
        **{"from": from_date, "to": to_date, "fields": "open,high,low,close,volume", "sort": "asc", "adjusted": "true"},
    )
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    data = r.json()
    if "data" not in data or not data["data"]:
        return pd.DataFrame()
    df = pd.DataFrame(data["data"])
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _download_day(stock_id: str, start_date: str, end_date: str | None = None, token: str = None) -> pd.DataFrame:
    """自動切割成年度區間（Fugle 限制每次 < 1 年）。end_date 選填，預設到今天；
    data/backfill_day_history.py 回補歷史缺口時會指定成「現有資料最早日期的
    前一天」，避免重複下載已經有的部分。

    token：不帶用預設 FUGLE 帳號，第二組 Fugle 執行緒池（_update_day_fugle2）
    會傳入 fugle_api.TOKEN_DAYTRADE 走另一組獨立 rate limit。"""
    now = datetime.now(_TW)
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else now.date()
    cur = datetime.strptime(start_date, "%Y-%m-%d").date()
    chunks = []
    while cur <= end:
        chunk_end = min(cur.replace(year=cur.year + 1) - timedelta(days=1), end)
        df = _fetch_year(stock_id, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), token=token)
        if not df.empty:
            chunks.append(df)
        cur = chunk_end + timedelta(days=1)
        if cur <= end:
            time.sleep(0.2)  # 2026-07-15：跨好幾年會切成好幾個請求，年度區間之間留點間隔，不要瞬間連發
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _fetch_year_fubon(sdk, stock_id: str, from_date: str, to_date: str) -> pd.DataFrame:
    """單次請求（區間 < 1 年），富邦 historical/candles 不帶 timeframe 就是日K，
    from/to 用法跟 Fugle 一致（見 fubon/fubon_api.py::historical_candles()）。

    這支股票在這段區間還沒有資料時（例如尚未上市），把 SDK 拋出的例外當成
    「這段沒資料」處理、回傳空 DataFrame，不往上拋——理由同 _fetch_year() 的
    說明，避免中斷 _download_day_fubon() 的年度迴圈，讓後面明明有資料的年份
    也抓不到。"""
    from fubon import fubon_api as trade_api

    try:
        bars = trade_api.historical_candles(
            sdk,
            stock_id,
            **{"from": from_date, "to": to_date, "fields": "open,high,low,close,volume", "sort": "asc", "adjusted": "true"},
        )
    except Exception as e:
        print(f"    {stock_id} {from_date}~{to_date} 富邦查詢失敗（視為這段沒資料）: {e}")
        return pd.DataFrame()
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _download_day_fubon(sdk, stock_id: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """自動切割成年度區間，比照 _download_day()。end_date 選填，用法同 _download_day()。"""
    now = datetime.now(_TW)
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else now.date()
    cur = datetime.strptime(start_date, "%Y-%m-%d").date()
    chunks = []
    while cur <= end:
        chunk_end = min(cur.replace(year=cur.year + 1) - timedelta(days=1), end)
        df = _fetch_year_fubon(sdk, stock_id, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        if not df.empty:
            chunks.append(df)
        cur = chunk_end + timedelta(days=1)
        if cur <= end:
            time.sleep(1.05)  # 2026-07-15：跟 _update_day_fubon() 既有的請求間隔一致，維持 60 req/min 以內
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _save_day(new_df: pd.DataFrame):
    new_df = new_df[["stock_id", "date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close"]:
        new_df[col] = new_df[col].astype("float32")
    new_df["volume"] = new_df["volume"].astype("int64")

    with _save_lock:
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
    with _flag_lock:
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
    """股票母體：讀 db/tickers/tickers.parquet 現有內容（見
    fubon/intraday_tickers.py::load_tickers()），不觸發即時富邦API重新查詢——
    這裡要的是「現在 db/tickers 裡有記錄的全部股票」，不是「這一刻盤中報的
    最新清單」，兩者通常一致，但即時查詢還要多一次富邦API往返、且非盤中會
    回傳空資料，沒必要。db/tickers 由 fubon/intraday_tickers.py::update_tickers()
    每天更新一次（見 main/premarket.py::refresh_tickers()）。"""
    from fubon.intraday_tickers import load_tickers

    return load_tickers()["stock_id"].tolist()


def _get_done_stocks(date_str: str) -> set:
    if not os.path.exists(_FLAG_PATH):
        return set()
    df = pd.read_parquet(_FLAG_PATH)
    return set(df[df["date"] == date_str]["stock_id"].tolist())


def _update_day_fugle(stocks: list, start_date: str, date_str: str, workers: int, token: str = None, label: str = "Fugle"):
    """Fugle 那一份：多執行緒併發，429 交給 Retry-After 被動重試
    （見 _fetch_year()），workers=5 約 5 分鐘下載 1500 支。

    token/label：讓 update_day() 可以開兩組 Fugle 執行緒池各用一組帳號
    （FUGLE / FUGLE_DAYTRADE），互不共用 rate limit。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = 0

    def _fetch_one(stock_id: str):
        nonlocal completed
        time.sleep(0.2)  # 每執行緒小延遲，5 執行緒合計 ~1 req/s
        try:
            df = _download_day(stock_id, start_date, token=token)
            if not df.empty:
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
        futures = {pool.submit(_fetch_one, sid): sid for sid in stocks}
        for fut in as_completed(futures):
            sid, rows, err = fut.result()
            completed += 1
            if err:
                print(f"  [{label} {completed}/{len(stocks)}] {sid} 失敗: {err}")
            elif completed % 100 == 0 or completed == len(stocks):
                print(f"  [{label} {completed}/{len(stocks)}] 進度更新")


def _update_day_fubon(stocks: list, start_date: str, date_str: str, sdk):
    """富邦那一半：單一執行緒依序下載，1.05秒/支，維持在 60 req/min 以內留緩衝
    （比照 data/m1_data_loader.py 的作法）。"""
    for i, stock_id in enumerate(stocks, 1):
        try:
            df = _download_day_fubon(sdk, stock_id, start_date)
            if not df.empty:
                _save_day(df)
            _update_flag(stock_id, date_str)
            if i % 100 == 0 or i == len(stocks):
                print(f"  [富邦 {i}/{len(stocks)}] 進度更新")
        except Exception as e:
            print(f"  [富邦 {i}/{len(stocks)}] {stock_id} 失敗: {e}")
        time.sleep(1.05)


def update_day(start_date: str = None, stocks: list = None, workers: int = 5):
    """
    日線，flag 避免同日重複下載。待下載清單拆三份，Fugle 兩組帳號
    （FUGLE、FUGLE_DAYTRADE，各自獨立 rate limit）+ 富邦一份同時下載。
    workers: 每組 Fugle 執行緒池的並發數（預設 5，約 5 分鐘下載 1500 支）
    """
    if not fugle_api.TOKEN:
        raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")
    if not fugle_api.TOKEN_DAYTRADE:
        raise RuntimeError("缺少第二組 Fugle API Key，請在 .env 設定 FUGLE_DAYTRADE")

    now = datetime.now(_TW)
    date_str = now.strftime("%Y-%m-%d")
    os.makedirs(_ROOT / "db/fugle_day", exist_ok=True)

    if start_date is None:
        last = _last_stored_date()
        start_date = last if last else f"{now.year}-{now.month:02d}-01"

    candidates = _all_stocks() if stocks is None else stocks
    done = _get_done_stocks(date_str)
    wait_stocks = [s for s in candidates if s not in done]

    third = len(wait_stocks) // 3
    fugle_half1 = wait_stocks[:third]
    fugle_half2 = wait_stocks[third : 2 * third]
    fubon_half = wait_stocks[2 * third :]
    print(
        f"start_date={start_date}，Fugle {len(fugle_half1)} 支、"
        f"Fugle(daytrade) {len(fugle_half2)} 支（各 {workers} 並發）、"
        f"富邦 {len(fubon_half)} 支，同時下載..."
    )

    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    trade_api.init_market_data(sdk)
    try:
        t_fugle1 = threading.Thread(target=_update_day_fugle, args=(fugle_half1, start_date, date_str, workers))
        t_fugle2 = threading.Thread(
            target=_update_day_fugle,
            args=(fugle_half2, start_date, date_str, workers, fugle_api.TOKEN_DAYTRADE, "Fugle-DT"),
        )
        t_fubon = threading.Thread(target=_update_day_fubon, args=(fubon_half, start_date, date_str, sdk))
        t_fugle1.start()
        t_fugle2.start()
        t_fubon.start()
        t_fugle1.join()
        t_fugle2.join()
        t_fubon.join()
    finally:
        trade_api.logout(sdk)

    print("日K 下載完成")


if __name__ == "__main__":
    update_day()
