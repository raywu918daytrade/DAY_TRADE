"""
歷史分K 下載器（訓練資料用）

功能：
    從 Fugle + 富邦 REST API 下載 1 分鐘 K 線，存入 db/m1/（訓練資料庫）。
    flag 機制避免同一支股票在同一天重複下載。

與 fubon/marketdata_ws.py 的差異：
    m1_data_loader.py       → 下載歷史分K（近30日），存 db/m1/，給訓練用，GHA 每日觸發
    fubon/marketdata_ws.py  → 盤中富邦 WebSocket 即時推送，存 db/m1_live/，給當天交易推論用

Fugle + 富邦同時下載：
    兩邊 historical/candles 各自獨立 rate limit（都是 60次/分鐘，互不影響），
    待下載股票清單拆一半分別跑，兩條執行緒同時進行，約可把時間壓到一半。
    兩邊的 historical/candles 都已實測確認本身就含「今天」的資料（2026-07-13
    20:30 實測，update_m1.yml 排程在台北18:00跑，遠晚於13:30收盤），一支
    股票各自只需要呼叫一次，不用再另外呼叫 intraday/candles 補今天。
    富邦 SDK 呼叫都透過 fubon/fubon_api.py，這裡不直接碰 fubon_neo。

主要函式：
    update_m1(stocks)
        stocks 預設為 fubon.subscribe_list.all_normal_stocks() 完整股票母體，
        拆一半給 Fugle、一半給富邦下載（母體來源跟下載來源是分開的兩件事）。
"""
import threading
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import time

import pandas as pd
import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))

token = os.environ.get("FUGLE", "")

_BASE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"
_FLAG_PATH = _ROOT / "db/m1_flags/m1_flag.parquet"

# _save_m1()/_update_flag() 都是「讀舊檔→merge→寫回去」，Fugle、富邦兩條
# 執行緒同時呼叫會搶同一個檔案（2026-07-13 在 fubon/marketdata_ws.py 實際
# 撞過一次同類型的 race condition，_atomic_to_parquet 的 .tmp 檔名也不是
# 唯一的），這裡直接用鎖序列化，兩條執行緒不會真的同時寫檔。
_save_lock = threading.Lock()
_flag_lock = threading.Lock()


def _m1_file_path(date: pd.Timestamp) -> Path:
    """月份補零，需與 push_m1_to_hf.py 的 pd.Period.astype(str) 命名一致，
    否則同一個月會在本機/HF Hub 各自產生一個檔名不同的分檔，觸發 schema 衝突
    （db/fugle_day/ 已經因此撞過一次，見 2026_7.parquet vs 2026_07.parquet）"""
    return _ROOT / f"db/m1/{date.year}_{date.month:02d}.parquet"


def _atomic_to_parquet(df: pd.DataFrame, file_path: str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    tmp_path = f"{file_path}.tmp"
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def _download_m1(stock_id: str) -> pd.DataFrame:
    """取得近30日1分鐘K線（Fugle分K無法指定 from/to，一律回傳近30日資料）。

    2026-07-13 實測（20:30）：這個端點本身就含「今天」的資料（最後一筆是
    當天13:30收盤），不是舊註解講的「僅到昨日」——update_m1.yml 排程在台北
    18:00 跑，永遠晚於收盤，所以不需要再另外呼叫 intraday/candles 補今天。
    """
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


def _save_m1(new_df: pd.DataFrame):
    new_df = new_df[["stock_id", "date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close"]:
        new_df[col] = new_df[col].astype("float32")
    new_df["volume"] = new_df["volume"].astype("int64")

    with _save_lock:
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


def _download_m1_fubon(sdk, stock_id: str) -> pd.DataFrame:
    """近30日+今日分K，富邦 historical/candles 一次就含今天，不用另外呼叫
    intraday/candles。"""
    from fubon import fubon_api as trade_api

    bars = trade_api.historical_candles(sdk, stock_id, timeframe=1)
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["stock_id"] = stock_id
    date = pd.to_datetime(df["date"])
    if date.dt.tz is not None:
        date = date.dt.tz_convert(_TW).dt.tz_localize(None)
    df["date"] = date.dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _all_stocks() -> list:
    """股票母體：讀 db/tickers/tickers.parquet 現有內容（見
    fubon/intraday_tickers.py::load_tickers()），不觸發即時富邦API重新查詢——
    這裡要的是「現在 db/tickers 裡有記錄的全部股票」，不是「這一刻盤中報的
    最新清單」，兩者通常一致，但即時查詢還要多一次富邦API往返、且非盤中會
    回傳空資料，沒必要。db/tickers 由 fubon/intraday_tickers.py::update_tickers()
    每天更新一次（見 main/premarket.py::refresh_tickers()）。母體來源改了不
    影響下載邏輯本身，Fugle/富邦兩邊還是各自下載一半。"""
    from fubon.intraday_tickers import load_tickers

    return load_tickers()["stock_id"].tolist()


def _get_done_stocks(date_str: str) -> set:
    if not os.path.exists(_FLAG_PATH):
        return set()
    df = pd.read_parquet(_FLAG_PATH)
    return set(df[df["date"] == date_str]["stock_id"].tolist())


def _update_m1_fugle(stocks: list, date_str: str):
    """Fugle 那一半：historical/candles 一次就含今天（見 _download_m1() 的
    2026-07-13 實測結果），1.05秒/支（1次API），維持在 60 req/min 以內留緩衝。"""
    for stock_id in stocks:
        try:
            df = _download_m1(stock_id)
            if not df.empty:
                _save_m1(df)
                print(f"[Fugle] {stock_id} 下載完成 {len(df)} 筆")
            else:
                print(f"[Fugle] {stock_id} 無資料")
            _update_flag(stock_id, date_str)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"[Fugle] {stock_id} 無此股票資料，標記跳過")
                _update_flag(stock_id, date_str)
            else:
                print(f"[Fugle] {stock_id} 失敗: {e}")
        except Exception as e:
            print(f"[Fugle] {stock_id} 失敗: {e}")
        time.sleep(1.05)


def _update_m1_fubon(stocks: list, date_str: str, sdk):
    """富邦那一半：historical/candles 一次就含今天，1.05秒/支（1次API），
    維持在 60 req/min 以內留緩衝。"""
    for stock_id in stocks:
        try:
            df = _download_m1_fubon(sdk, stock_id)
            if not df.empty:
                _save_m1(df)
                print(f"[富邦] {stock_id} 下載完成 {len(df)} 筆")
            else:
                print(f"[富邦] {stock_id} 無資料")
            _update_flag(stock_id, date_str)
        except Exception as e:
            print(f"[富邦] {stock_id} 失敗: {e}")
        time.sleep(1.05)


def update_m1(stocks: list = None):
    """1分鐘K線，flag避免同日重複下載。待下載清單拆一半，Fugle 跟富邦
    同時下載（各自獨立 rate limit），約可把時間壓到單一資料源的一半。"""
    if not token:
        raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")

    now = datetime.now(_TW)
    date_str = now.strftime("%Y-%m-%d")
    os.makedirs(_ROOT / "db/m1", exist_ok=True)

    candidates = _all_stocks() if stocks is None else stocks
    done = _get_done_stocks(date_str)
    wait_stocks = [s for s in candidates if s not in done]
    print("還有", len(wait_stocks), "個股票未更新（已排除今日已下載 flag）")

    half = len(wait_stocks) // 2
    fugle_half, fubon_half = wait_stocks[:half], wait_stocks[half:]
    print(f"Fugle {len(fugle_half)} 支、富邦 {len(fubon_half)} 支，同時下載...")

    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    trade_api.init_market_data(sdk)
    try:
        t_fugle = threading.Thread(target=_update_m1_fugle, args=(fugle_half, date_str))
        t_fubon = threading.Thread(target=_update_m1_fubon, args=(fubon_half, date_str, sdk))
        t_fugle.start()
        t_fubon.start()
        t_fugle.join()
        t_fubon.join()
    finally:
        trade_api.logout(sdk)

    print("全部完成")


if __name__ == "__main__":
    update_m1()
