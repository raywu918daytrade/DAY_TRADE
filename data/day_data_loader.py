"""
日K 資料下載器（Fugle + 富邦 historical/candles）

功能：
    從 Fugle、富邦 REST API 下載股票日K，按月份分檔存入 db/d1/（原始）或
    db/adjustment_day/（完整還原，pattern 專用）。flag 機制避免同一支股票在
    同一天重複下載。

⚠️ 2026-08-03 改：拆成兩個獨立目的的下載函式，共用同一套抓取/重試/多執行緒
邏輯（只差 adjusted 參數跟輸出目錄/flag檔）：
    update_day()            → db/d1/（原始，不帶 adjusted，系統預設資料源，
                               data/query.py::load_day() 查詢時再套拆股/合股
                               factor）
    update_adjustment_day() → db/adjustment_day/（完整還原，帶
                               adjusted="true"，只給 pattern 系列
                               [data/adjustment_query.py] 用，因為技術型態
                               偵測需要除息缺口也被抹平，見
                               data/adjustment_query.py 檔頭說明）
這支檔案原本叫 fugle_day 的下載器，只下載完整還原版本；改動之前 db/fugle_day
被兩種不同需求依賴（原始資料源 + pattern專用還原源），拆開之後 db/fugle_day
整個目錄改名成 db/adjustment_day，原始下載改成新的 db/d1。

⚠️ 2026-08-08 改：股票母體從 400支（依成交量排名的 tick_universe）擴大成
1877支（db/tickers/stock_universe_2000.parquet，見
finmind/stock_universe_2000.py——全市場4碼一般個股、不排名、只排除ETF），
理由：finmind 分K回補已經先擴大到這份清單，日K母體要跟著擴大，避免大量
股票「有分K沒日K」，跨資料源特徵（例如idx_gap_pct）算不出來。見 _all_stocks()。

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
        增量下載指定標的的原始日K，多執行緒並發，5 workers 約 5 分鐘下載 1500 支
        （這是 Fugle 那一半的預估，富邦那一半另外跑）。
    update_adjustment_day(start_date, stocks, workers)
        同上，但抓完整還原版本，寫入 db/adjustment_day/。

⚠️ update_day() 跟 update_adjustment_day() 要依序執行、不能同時跑——兩者都會
用到 Fugle 雙帳號 + 富邦三路併發下載，同時跑會互搶同一組 rate limit
（見 scripts/update_daily.py）。

資料格式（db/d1/YYYY_M.parquet、db/adjustment_day/YYYY_M.parquet）：
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

_D1_DIR = "db/d1"
_ADJUSTMENT_DAY_DIR = "db/adjustment_day"
_D1_FLAG_PATH = _ROOT / "db/d1_flags/day_flag.parquet"
_ADJUSTMENT_DAY_FLAG_PATH = _ROOT / "db/adjustment_day_flags/day_flag.parquet"

# _save_day()/_update_flag() 都是「讀舊檔→merge→寫回去」，Fugle 執行緒池、富邦
# 執行緒同時呼叫會搶同一個檔案（比照 data/m1_data_loader.py 的作法），這裡直接
# 用鎖序列化。
_save_lock = threading.Lock()
_flag_lock = threading.Lock()

# 2026-08-08：三路併發（Fugle/Fugle-DT/富邦）依實際節流速率分配股票數，見
# _split_by_rate()。
# ⚠️ 2026-08-11 修正：這裡的富邦指 fubon/fubon_api.py::historical_candles()
# （日K/分K歷史，跟 Fugle 同一套底層 fugle_marketdata 元件），該函式自己的
# docstring 明載官方限制是 60次/分鐘，跟 Fugle 一樣——不是 300次/分鐘。
# 300次/分鐘是 fubon/tick_api.py 用的 intraday/trades（抓當天tick）另一個
# 端點家族的限制，兩者不能混為一談，之前誤用同一個常數導致實際跑
# scripts/update_daily.py 時打出 429 Rate limit exceeded。三路現在都是
# 60次/分鐘，_split_by_rate() 在這個常數設定下會自動退化成均分三等份。
_FUGLE_INTERVAL = 1.05  # 秒/次，60次/分鐘留緩衝
_FUBON_INTERVAL = 1.05  # 秒/次，60次/分鐘留緩衝（historical_candles，不是intraday/trades）


def _split_by_rate(stocks: list, n_fugle_accounts: int = 2) -> tuple:
    """依照 Fugle（n_fugle_accounts 組帳號，各自獨立 rate limit）跟富邦
    實際節流速率的比例切股票清單，讓幾條平行執行緒理論上同時做完，不要
    再無腦均分（那樣會讓步調較快的富邦做完閒置、拖累整體時間的還是最慢
    的那條）。回傳 (fugle帳號1清單, fugle帳號2清單, ..., 富邦清單)，
    data/m1_data_loader.py::update_m1() 也共用這支，不要各自複製一份。"""
    fugle_rate = 1 / _FUGLE_INTERVAL
    fubon_rate = 1 / _FUBON_INTERVAL
    total_rate = fugle_rate * n_fugle_accounts + fubon_rate
    n = len(stocks)
    fugle_each = round(n * fugle_rate / total_rate)
    groups = []
    idx = 0
    for _ in range(n_fugle_accounts):
        groups.append(stocks[idx : idx + fugle_each])
        idx += fugle_each
    groups.append(stocks[idx:])  # 富邦拿剩下的（含四捨五入誤差，不用另外處理）
    return tuple(groups)


def _day_file_path(date: str, base_dir: str = _D1_DIR) -> Path:
    """依資料日期決定存入哪個月份檔（月份補零，維持跟 db/m1 等其他資料夾一致
    的命名慣例）"""
    ts = pd.Timestamp(date)
    return _ROOT / f"{base_dir}/{ts.year}_{ts.month:02d}.parquet"


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


def _fetch_year(stock_id: str, from_date: str, to_date: str, token: str = None, adjusted: bool = False) -> pd.DataFrame:
    """單次請求（區間 < 1 年）。404 代表這支股票在這段區間還沒有資料（例如
    尚未上市／尚未開始交易），回傳空 DataFrame，不當例外處理——2026-07-15
    實測：對現在才被列入候選清單、但更早以前還沒上市的股票回補歷史時很常見
    （例如某支股票2016年404、2023年正常有資料），如果讓它往上拋例外，
    _download_day() 的年度迴圈會被中斷，之後年份（明明有資料）也不會再嘗試，
    整支股票這次直接算失敗。例外只保留給真正的請求失敗（限流、逾時、其他
    錯誤）。

    adjusted：預設 False（原始價，給 db/d1 用）；update_adjustment_day() 呼叫
    時帶 True，跟 Fugle 要完整還原版本（給 db/adjustment_day / pattern 用）。"""
    params = {"from": from_date, "to": to_date, "fields": "open,high,low,close,volume", "sort": "asc"}
    if adjusted:
        params["adjusted"] = "true"
    r = fugle_api.historical_candles(stock_id, token=token, **params)
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


def _download_day(
    stock_id: str, start_date: str, end_date: str | None = None, token: str = None, adjusted: bool = False
) -> pd.DataFrame:
    """自動切割成年度區間（Fugle 限制每次 < 1 年）。end_date 選填，預設到今天；
    data/backfill_day_history.py 回補歷史缺口時會指定成「現有資料最早日期的
    前一天」，避免重複下載已經有的部分。

    token：不帶用預設 FUGLE 帳號，第二組 Fugle 執行緒池（_update_day_fugle2）
    會傳入 fugle_api.TOKEN_DAYTRADE 走另一組獨立 rate limit。
    adjusted：說明同 _fetch_year()。"""
    now = datetime.now(_TW)
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else now.date()
    cur = datetime.strptime(start_date, "%Y-%m-%d").date()
    chunks = []
    while cur <= end:
        chunk_end = min(cur.replace(year=cur.year + 1) - timedelta(days=1), end)
        df = _fetch_year(stock_id, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), token=token, adjusted=adjusted)
        if not df.empty:
            chunks.append(df)
        cur = chunk_end + timedelta(days=1)
        if cur <= end:
            time.sleep(0.2)  # 2026-07-15：跨好幾年會切成好幾個請求，年度區間之間留點間隔，不要瞬間連發
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _fetch_year_fubon(sdk, stock_id: str, from_date: str, to_date: str, adjusted: bool = False) -> pd.DataFrame:
    """單次請求（區間 < 1 年），富邦 historical/candles 不帶 timeframe 就是日K，
    from/to 用法跟 Fugle 一致（見 fubon/fubon_api.py::historical_candles()）。

    這支股票在這段區間還沒有資料時（例如尚未上市），把 SDK 拋出的例外當成
    「這段沒資料」處理、回傳空 DataFrame，不往上拋——理由同 _fetch_year() 的
    說明，避免中斷 _download_day_fubon() 的年度迴圈，讓後面明明有資料的年份
    也抓不到。adjusted：說明同 _fetch_year()。"""
    from fubon import fubon_api as trade_api

    params = {"from": from_date, "to": to_date, "fields": "open,high,low,close,volume", "sort": "asc"}
    if adjusted:
        params["adjusted"] = "true"
    try:
        bars = trade_api.historical_candles(sdk, stock_id, **params)
    except Exception as e:
        print(f"    {stock_id} {from_date}~{to_date} 富邦查詢失敗（視為這段沒資料）: {e}")
        return pd.DataFrame()
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _download_day_fubon(sdk, stock_id: str, start_date: str, end_date: str | None = None, adjusted: bool = False) -> pd.DataFrame:
    """自動切割成年度區間，比照 _download_day()。end_date 選填，用法同 _download_day()。"""
    now = datetime.now(_TW)
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else now.date()
    cur = datetime.strptime(start_date, "%Y-%m-%d").date()
    chunks = []
    while cur <= end:
        chunk_end = min(cur.replace(year=cur.year + 1) - timedelta(days=1), end)
        df = _fetch_year_fubon(sdk, stock_id, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), adjusted=adjusted)
        if not df.empty:
            chunks.append(df)
        cur = chunk_end + timedelta(days=1)
        if cur <= end:
            time.sleep(_FUBON_INTERVAL)  # 跟 _update_day_fubon() 一致，見該常數說明（60次/分鐘留緩衝）
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _save_day(new_df: pd.DataFrame, base_dir: str = _D1_DIR):
    new_df = new_df[["stock_id", "date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close"]:
        new_df[col] = new_df[col].astype("float32")
    new_df["volume"] = new_df["volume"].astype("int64")

    with _save_lock:
        for month, group in new_df.groupby(pd.to_datetime(new_df["date"]).dt.to_period("M")):
            file_path = _day_file_path(str(month.to_timestamp().date()), base_dir=base_dir)
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


def _update_flag(stock_id: str, date_str: str, flag_path: Path = _D1_FLAG_PATH):
    with _flag_lock:
        new_row = pd.DataFrame([{"stock_id": stock_id, "date": date_str}])
        if os.path.exists(flag_path):
            df = pd.read_parquet(flag_path)
            df = pd.concat([df, new_row], ignore_index=True)
            df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
        else:
            df = new_row
        os.makedirs(os.path.dirname(flag_path), exist_ok=True)
        _atomic_to_parquet(df, flag_path, index=False)


def _all_stocks() -> list:
    """股票母體（2026-08-08改，2026-08-08再改共用helper）：讀
    finmind.stock_universe_2000.load_stock_universe_2000_with_0050()
    （1877支全市場一般個股+強制併入0050，共1878支，見該函式說明），取代
    原本400支依成交量排名的tick_universe——理由見檔頭說明。

    注意：這是 data/day_data_loader.py 自己的 _all_stocks()，跟
    data/m1_data_loader.py::_all_stocks() 是兩個不同函式，但2026-08-08
    起兩者統一讀同一份 stock_universe_2000_with_0050 清單（理由對稱：
    日K/分K母體要一致，避免大量股票「有日K沒分K」或反過來）。"""
    from finmind.stock_universe_2000 import load_stock_universe_2000_with_0050

    return load_stock_universe_2000_with_0050()


def _get_done_stocks(date_str: str, flag_path: Path = _D1_FLAG_PATH) -> set:
    if not os.path.exists(flag_path):
        return set()
    df = pd.read_parquet(flag_path)
    return set(df[df["date"] == date_str]["stock_id"].tolist())


def _last_stored_dates(base_dir: str = _D1_DIR) -> dict[str, str]:
    """base_dir 裡每支股票目前最新的存檔日期，一次掃描全部檔案（比逐支
    股票各自查一次快很多）。比照 data/backfill_day_history.py::_earliest_dates()
    的做法，但這裡要的是每支股票的「最新」日期，不是「最早」。股票沒出現過
    就不在這個 dict 裡（呼叫端要自己 fallback 一個起始日期）。"""
    day_dir = _ROOT / base_dir
    if not day_dir.exists():
        return {}
    dataset = ds.dataset(str(day_dir), format="parquet")
    if dataset.count_rows() == 0:
        return {}
    df = dataset.to_table(columns=["stock_id", "date"]).to_pandas()
    return df.groupby("stock_id")["date"].max().to_dict()


def _has_today_data(df: pd.DataFrame, date_str: str) -> bool:
    """判斷下載回來的資料是否真的包含「今天」這一天（2026-08-04發現：day K
    也有跟 m1 一樣的問題——如果在今天日K還沒發布時執行，API 回傳非空
    （有更早的歷史），但最新一筆只到昨天，這時候不能標記 flag，否則收盤後
    重跑會被誤判成「今天已處理過」而跳過，today 這一天永遠抓不到。比照
    data/m1_data_loader.py::_has_today_data() 的同一種修法，day K 的 date
    欄位是純日期字串（YYYY-MM-DD，不含時分秒），用完全比對即可，不用像
    m1 那樣 str.startswith()。"""
    return bool((df["date"] == date_str).any())


def _update_day_fugle(
    stocks: list,
    stock_start_dates: dict[str, str],
    date_str: str,
    workers: int,
    token: str = None,
    label: str = "Fugle",
    adjusted: bool = False,
    base_dir: str = _D1_DIR,
    flag_path: Path = _D1_FLAG_PATH,
):
    """Fugle 那一份：多執行緒併發，429 交給 Retry-After 被動重試
    （見 _fetch_year()），workers=5 約 5 分鐘下載 1500 支。

    token/label：讓呼叫端可以開兩組 Fugle 執行緒池各用一組帳號
    （FUGLE / FUGLE_DAYTRADE），互不共用 rate limit。

    stock_start_dates：逐支股票各自的起始日期（見 update_day() 的說明，
    2026-07-26 改成不用單一全域 start_date——全域值會讓已經落後的股票
    永遠只查「今天附近」這一小段，問不到它自己真正缺的那一大段）。

    2026-07-26 改：查到空結果（含404，_fetch_year() 已經把404轉成空
    DataFrame，不會走到下面的HTTPError分支）不再標記 _update_flag()——
    空結果可能是暫時性問題（API異常、限流），標記完成的話當天重跑也不會
    再重試，之前 3055/4707/6174/2466 這幾支卡住好幾週就是類似情況。真的
    確認有資料才標記完成，避免同一天內對已確認有結果的股票重複打API。

    adjusted/base_dir/flag_path：2026-08-03加，讓 update_day()（原始，
    db/d1）跟 update_adjustment_day()（完整還原，db/adjustment_day）共用
    這套下載邏輯。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = 0

    def _fetch_one(stock_id: str):
        nonlocal completed
        time.sleep(0.2)  # 每執行緒小延遲，5 執行緒合計 ~1 req/s
        try:
            df = _download_day(stock_id, stock_start_dates[stock_id], token=token, adjusted=adjusted)
            if not df.empty:
                _save_day(df, base_dir=base_dir)
                if _has_today_data(df, date_str):
                    _update_flag(stock_id, date_str, flag_path=flag_path)
                else:
                    print(f"  [{label}] {stock_id} 尚無今日資料，不標記 flag（稍後重跑會再抓）")
            return stock_id, len(df), None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
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


def _update_day_fubon(
    stocks: list,
    stock_start_dates: dict[str, str],
    date_str: str,
    sdk,
    adjusted: bool = False,
    base_dir: str = _D1_DIR,
    flag_path: Path = _D1_FLAG_PATH,
):
    """富邦那一半：單一執行緒依序下載，1.05秒/支（`historical_candles()`
    官方限制60次/分鐘留緩衝，見 _FUBON_INTERVAL 說明——2026-08-11修正：
    不是300次/分鐘，那是另一個端點家族的限制）。

    stock_start_dates/空結果不標記完成：說明同 _update_day_fugle()。
    adjusted/base_dir/flag_path：說明同 _update_day_fugle()。"""
    for i, stock_id in enumerate(stocks, 1):
        try:
            df = _download_day_fubon(sdk, stock_id, stock_start_dates[stock_id], adjusted=adjusted)
            if not df.empty:
                _save_day(df, base_dir=base_dir)
                if _has_today_data(df, date_str):
                    _update_flag(stock_id, date_str, flag_path=flag_path)
                else:
                    print(f"  [富邦] {stock_id} 尚無今日資料，不標記 flag（稍後重跑會再抓）")
            if i % 100 == 0 or i == len(stocks):
                print(f"  [富邦 {i}/{len(stocks)}] 進度更新")
        except Exception as e:
            print(f"  [富邦 {i}/{len(stocks)}] {stock_id} 失敗: {e}")
        time.sleep(_FUBON_INTERVAL)


def _update_day_generic(
    start_date: str | None,
    stocks: list | None,
    workers: int,
    adjusted: bool,
    base_dir: str,
    flag_path: Path,
):
    """update_day()/update_adjustment_day() 共用的核心邏輯，只差 adjusted
    參數跟輸出目錄/flag檔——見這兩支函式各自的 docstring。"""
    if not fugle_api.TOKEN:
        raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")
    if not fugle_api.TOKEN_DAYTRADE:
        raise RuntimeError("缺少第二組 Fugle API Key，請在 .env 設定 FUGLE_DAYTRADE")

    now = datetime.now(_TW)
    date_str = now.strftime("%Y-%m-%d")
    os.makedirs(_ROOT / base_dir, exist_ok=True)

    candidates = _all_stocks() if stocks is None else stocks
    done = _get_done_stocks(date_str, flag_path=flag_path)
    wait_stocks = [s for s in candidates if s not in done]

    if start_date is not None:
        stock_start_dates = {sid: start_date for sid in wait_stocks}
        start_date_desc = start_date
    else:
        last_dates = _last_stored_dates(base_dir=base_dir)
        default_start = f"{now.year}-{now.month:02d}-01"
        stock_start_dates = {sid: last_dates.get(sid, default_start) for sid in wait_stocks}
        behind = sorted(set(stock_start_dates.values()))
        start_date_desc = f"逐支查詢（{len(behind)} 種不同起始日期，最舊 {behind[0] if behind else '無'}）"

    fugle_half1, fugle_half2, fubon_half = _split_by_rate(wait_stocks)
    print(
        f"[{base_dir}] start_date={start_date_desc}，Fugle {len(fugle_half1)} 支、"
        f"Fugle(daytrade) {len(fugle_half2)} 支（各 {workers} 並發）、"
        f"富邦 {len(fubon_half)} 支，同時下載..."
    )

    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    trade_api.init_market_data(sdk)
    try:
        t_fugle1 = threading.Thread(
            target=_update_day_fugle,
            args=(fugle_half1, stock_start_dates, date_str, workers),
            kwargs={"adjusted": adjusted, "base_dir": base_dir, "flag_path": flag_path},
        )
        t_fugle2 = threading.Thread(
            target=_update_day_fugle,
            args=(fugle_half2, stock_start_dates, date_str, workers, fugle_api.TOKEN_DAYTRADE, "Fugle-DT"),
            kwargs={"adjusted": adjusted, "base_dir": base_dir, "flag_path": flag_path},
        )
        t_fubon = threading.Thread(
            target=_update_day_fubon,
            args=(fubon_half, stock_start_dates, date_str, sdk),
            kwargs={"adjusted": adjusted, "base_dir": base_dir, "flag_path": flag_path},
        )
        t_fugle1.start()
        t_fugle2.start()
        t_fubon.start()
        t_fugle1.join()
        t_fugle2.join()
        t_fubon.join()
    finally:
        trade_api.logout(sdk)

    print(f"[{base_dir}] 下載完成")


def update_day(start_date: str = None, stocks: list = None, workers: int = 5):
    """
    原始日線（不還原權息），寫入 db/d1/，flag 避免同日重複下載。待下載清單拆
    三份，Fugle 兩組帳號（FUGLE、FUGLE_DAYTRADE，各自獨立 rate limit）+ 富邦
    一份同時下載。workers: 每組 Fugle 執行緒池的並發數（預設 5，約 5 分鐘
    下載 1500 支）

    start_date：選填，明確指定的話對這次待下載的股票全部套用同一個起始日期
    （例如手動重跑想強制指定範圍）。**不指定（預設，日常排程走這條）時，
    2026-07-26 改成逐支查詢各自的起始日期**（每支股票用「db/d1 裡這支自己
    最新存到哪」當起點，沒資料的股票才 fallback 到這個月1號）——改之前是
    全部股票共用同一個全域最新日期，一旦某支股票落後（例如某次 API暫時異常
    沒抓到），之後每天都只會查「今天附近」這一小段，永遠問不到這支自己真正
    缺的那一大段，實測 3055/4707/6174/2466 這4支就是卡在這個問題，最新資料
    停在好幾週前不會自動追上。比照 data/backfill_day_history.py 逐支查詢的
    做法（那支是查最早日期補「更早的歷史」，這裡查最新日期補「跟上今天」，
    方向不同但用同一套思路）。

    ⚠️ 跟 update_adjustment_day() 要依序執行、不能同時跑，見檔頭說明。
    """
    _update_day_generic(start_date, stocks, workers, adjusted=False, base_dir=_D1_DIR, flag_path=_D1_FLAG_PATH)


def update_adjustment_day(start_date: str = None, stocks: list = None, workers: int = 5):
    """
    完整還原日線（Fugle adjusted="true"，含拆股/合股+一般除權息），寫入
    db/adjustment_day/，只給 pattern 系列（data/adjustment_query.py）用，
    理由見 data/adjustment_query.py 檔頭說明（型態偵測需要除息缺口也被抹平，
    否則會誤判轉折點，2026-08-03 用 1101 除息實測過影響幅度足以跨過偵測器
    門檻）。用法/參數同 update_day()。

    ⚠️ 跟 update_day() 要依序執行、不能同時跑（兩者都用 Fugle雙帳號+富邦三路
    併發下載，同時跑會互搶同一組 rate limit），見 scripts/update_daily.py。
    """
    _update_day_generic(
        start_date, stocks, workers, adjusted=True, base_dir=_ADJUSTMENT_DAY_DIR, flag_path=_ADJUSTMENT_DAY_FLAG_PATH
    )


if __name__ == "__main__":
    update_day()
