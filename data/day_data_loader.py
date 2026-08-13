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

Fugle + 富邦同時下載（2026-08-13改成共用queue，取代原本事先切固定清單）：
    Fugle 兩組帳號（各自 ThreadPoolExecutor 併發，429 交給 Retry-After
    被動重試）+ 富邦，一起從同一個 `queue.Queue` 搶股票下載，誰快就自然
    多吃，不會出現「份內做完就閒置、拖累整體」的情況（理由見
    _update_day_fugle()/_update_day_fubon_fast() 的 docstring）。
    `update_day()`（adjusted=False、預設 start_date）額外拆
    Phase 1/Phase 2：Phase 1 排除掉「已知有歷史缺口」的股票（那些反正
    Phase 2 就會整段含今天重抓一次，Phase 1 抓等於白抓），用富邦
    intraday_candles（併發，只拿「今天」）+ Fugle 正常呼叫搶共用queue；
    Phase 2 只補 Phase 1 開始前就已經有歷史缺口的股票，富邦改用
    historical_candles（單執行緒序列，1.05秒/支留在 60次/分鐘以內）。
    `update_adjustment_day()` 跟手動指定 start_date 的情況沒有快路徑，
    維持單一階段。富邦
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
import queue
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

# ⚠️ 2026-08-11 修正：這裡的富邦指 fubon/fubon_api.py::historical_candles()
# （日K/分K歷史，跟 Fugle 同一套底層 fugle_marketdata 元件），該函式自己的
# docstring 明載官方限制是 60次/分鐘，跟 Fugle 一樣——不是 300次/分鐘。
# 300次/分鐘是 fubon/tick_api.py 用的 intraday/trades（抓當天tick）另一個
# 端點家族的限制，兩者不能混為一談，之前誤用同一個常數導致實際跑
# scripts/update_daily.py 時打出 429 Rate limit exceeded。
# 2026-08-13：三路併發（Fugle/Fugle-DT/富邦）改成共用queue動態搶股票
# （見 _update_day_generic()），不再事先按節流速率切固定份數。
_FUGLE_INTERVAL = 1.05  # 秒/次，60次/分鐘留緩衝
_FUBON_INTERVAL = 1.05  # 秒/次，60次/分鐘留緩衝（historical_candles，不是intraday/trades）
_INTRADAY_FAST_WORKERS = 10  # 富邦快路徑（intraday_candles）併發數，比照 marketdata_ws.py::_BACKFILL_WORKERS
_INTRADAY_FAST_INTERVAL = 0.25  # 秒/次，300次/分鐘留緩衝，靠共用rate_lock控制整體間隔（不是每執行緒各自sleep）


def _expected_prior_trading_day(today) -> str:
    """2026-08-11加：回傳「今天」往前推的上一個預期交易日（只處理週末，
    不含國定假日）。

    給快路徑（intraday_candles）判斷「這支股票是不是真的只缺今天」用——
    ⚠️ 不能拿股票池裡彼此的最新日期互相比較（例如「這支的最新日期是不是
    等於全部股票裡最新的那個」），那樣如果整個 pipeline 昨天忘記執行，
    全部股票會一起卡在同一天，彼此比對會誤判成「大家都跟上進度」，快路徑
    只抓當天、永遠不會發現、也補不回中間漏掉的那個缺口。必須跟這個「今天
    理論上該有的上一個交易日」的絕對值比較，對不上（不管是單一股票落後、
    還是整個pipeline昨天沒跑）就一律不給快路徑、落到慢路徑（本身有自動
    補近30天缺口的能力）。

    刻意不處理國定假日：假日隔天這個函式算出來的「預期上一交易日」會跟
    實際最後交易日對不上，導致那天所有股票都被判定「不能信任快路徑」、
    全部改走慢路徑——這是安全的失敗方式（頂多那天沒享受到快路徑的速度），
    比錯誤地信任一個沒考慮到假日的推算、导致真正的缺口被快路徑跳過安全
    得多。"""
    weekday = today.weekday()  # Monday=0 ... Sunday=6
    if weekday == 0:  # 星期一，理論上的上一交易日是上週五
        delta = 3
    elif weekday == 6:  # 星期日（理論上不會在這天跑，防呆用）
        delta = 2
    else:
        delta = 1
    return (today - timedelta(days=delta)).strftime("%Y-%m-%d")


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


def _download_day_fubon_intraday(sdk, stock_id: str) -> pd.DataFrame:
    """2026-08-11加：只抓「今天」的日K，借用 intraday_candles()（分K，官方
    限制300次/分鐘，比 historical_candles() 的60次/分鐘快5倍）抓到的今天
    分K，自己聚合成一根日K（open=第一根、high=最高、low=最低、close=最後
    一根、volume=加總）。回傳欄位對齊 _download_day_fubon()
    （stock_id/date/open/high/low/close/volume）。

    ⚠️ 只能給 db/d1（原始）用，不能用在 db/adjustment_day——intraday_candles()
    不支援 adjusted 參數，聚合出來的只會是原始價格。呼叫端（_update_day_fubon()）
    只在 adjusted=False 時才會用這支。

    只有富邦這個端點有這個優勢，Fugle 的 intraday 跟 historical 都是
    60次/分鐘、沒有差異，這支只給富邦這一路用。"""
    from fubon import fubon_api as trade_api

    bars = trade_api.intraday_candles(sdk, stock_id, timeframe=1)
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    date = pd.to_datetime(df["date"])
    if date.dt.tz is not None:
        date = date.dt.tz_convert(_TW).dt.tz_localize(None)
    day_str = date.dt.strftime("%Y-%m-%d").iloc[0]
    agg = pd.DataFrame(
        [
            {
                "stock_id": stock_id,
                "date": day_str,
                "open": float(df["open"].iloc[0]),
                "high": float(df["high"].max()),
                "low": float(df["low"].min()),
                "close": float(df["close"].iloc[-1]),
                "volume": float(df["volume"].sum()),
            }
        ]
    )
    return agg


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


def _flush_day_buffer(buffer: list, date_str: str, base_dir: str = _D1_DIR, flag_path: Path = _D1_FLAG_PATH):
    """把整批下載結果一次性存檔＋標記flag，取代逐支呼叫 _save_day()/
    _update_flag()——理由跟 data/m1_data_loader.py::_flush_m1_buffer()
    同一輪的教訓完全一樣：_save_day() 每次呼叫都要「讀整個月份的
    parquet檔→concat→去重複→排序→整檔重寫」，這個操作被全域 _save_lock
    保護，逐支呼叫的話不管開幾條併發執行緒都要排隊搶同一把鎖，是比
    API rate limit更嚴重的瓶頸（m1那邊實測單次存檔約0.87秒，d1檔案
    目前小很多但架構問題一樣，會隨資料量增加越來越慢）。

    改成：worker（_update_day_fugle()/_update_day_fubon_fast()/
    _update_day_fubon_slow()）下載完不立刻存檔，疊加進共用 buffer；等
    整個 Phase 的執行緒都 join() 完了，呼叫這支函式一次性合併成一個
    大DataFrame，只呼叫一次 _save_day()、一次批次標記flag。

    ⚠️ 取捨說明同 _flush_m1_buffer()：Phase 完成前中途當掉會遺失這個
    Phase 已下載但還沒flush的資料，用來換取整體大幅縮短的耗時。"""
    if not buffer:
        return
    dfs = [df for _, df, _ in buffer]
    combined = pd.concat(dfs, ignore_index=True)
    _save_day(combined, base_dir=base_dir)
    to_flag = [stock_id for stock_id, _, has_today in buffer if has_today]
    _update_flags_batch(to_flag, date_str, flag_path=flag_path)
    print(f"[{base_dir}] 批次存檔完成：{len(buffer)} 支（{len(combined)} 筆），標記 {len(to_flag)} 支flag")


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


def _update_flags_batch(stock_ids: list, date_str: str, flag_path: Path = _D1_FLAG_PATH):
    """一次幫多支股票標記 flag（一次讀寫），取代逐支呼叫 _update_flag()。
    2026-08-13加：搭配 _flush_day_buffer() 使用——見該函式說明。"""
    if not stock_ids:
        return
    with _flag_lock:
        new_rows = pd.DataFrame({"stock_id": stock_ids, "date": date_str})
        if os.path.exists(flag_path):
            df = pd.read_parquet(flag_path)
            df = pd.concat([df, new_rows], ignore_index=True)
            df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
        else:
            df = new_rows
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
    q: "queue.Queue",
    stock_start_dates: dict[str, str],
    date_str: str,
    workers: int,
    buffer: list,
    buffer_lock: threading.Lock,
    token: str = None,
    label: str = "Fugle",
    adjusted: bool = False,
):
    """Fugle 那一路：從共用queue搶股票下載，多執行緒併發（429 交給
    Retry-After 被動重試，見 _fetch_year()），workers=5 約 5 分鐘下載
    1500 支。

    2026-08-13改成queue模式（原本是 _update_day_generic() 事先切好的
    固定清單）：理由同 data/m1_data_loader.py::update_m1() 同一輪的
    修改——富邦快路徑併發化之後比較快，靜態切固定比例會讓富邦提早做完
    後閒置，改成共用queue動態搶，誰快就自然多吃。D1 本身跟 m1 不同，
    Fugle 這裡「已經」是逐支查詢最適範圍（不像 m1 的 historical_candles
    固定回傳近30天），Phase 1/Phase 2（見 update_day()）都用同一套邏輯
    呼叫這支，不用像 _update_day_fubon_fast()/_update_day_fubon_slow()
    那樣特別拆快慢。

    token/label：讓呼叫端可以開兩組 Fugle 執行緒池各用一組帳號
    （FUGLE / FUGLE_DAYTRADE），互不共用 rate limit。

    stock_start_dates：逐支股票各自的起始日期（見 update_day() 的說明，
    2026-07-26 改成不用單一全域 start_date——全域值會讓已經落後的股票
    永遠只查「今天附近」這一小段，問不到它自己真正缺的那一大段）。

    2026-07-26 改：查到空結果（含404，_fetch_year() 已經把404轉成空
    DataFrame，不會走到下面的HTTPError分支）不再標記完成——空結果可能
    是暫時性問題（API異常、限流），標記完成的話當天重跑也不會再重試，
    之前 3055/4707/6174/2466 這幾支卡住好幾週就是類似情況。真的確認有
    資料才標記完成，避免同一天內對已確認有結果的股票重複打API。

    adjusted：2026-08-03加，讓 update_day()（原始，db/d1）跟
    update_adjustment_day()（完整還原，db/adjustment_day）共用這套
    下載邏輯。

    ⚠️ 2026-08-13再改：不再逐支呼叫 _save_day()/_update_flag()，改成把
    (stock_id, df, has_today) 疊加進 buffer——理由見
    _flush_day_buffer() docstring：逐支存檔會被全域 _save_lock 卡住，
    每次呼叫都要整檔重讀重寫，是比API rate limit更嚴重的瓶頸。真正的
    存檔/標記flag，交給 _update_day_generic() 在這個 Phase 全部執行緒
    join() 完之後，一次呼叫 _flush_day_buffer()。
    """
    from concurrent.futures import ThreadPoolExecutor

    def _worker():
        while True:
            try:
                stock_id = q.get_nowait()
            except queue.Empty:
                return
            left = q.qsize()
            time.sleep(0.2)  # 每執行緒小延遲，5 執行緒合計 ~1 req/s
            try:
                df = _download_day(stock_id, stock_start_dates[stock_id], token=token, adjusted=adjusted)
                if not df.empty:
                    has_today = _has_today_data(df, date_str)
                    with buffer_lock:
                        buffer.append((stock_id, df, has_today))
                    if has_today:
                        print(f"  [{label}] {stock_id} 下載完成 {len(df)} 筆，等批次存檔（queue剩餘 {left} 支）")
                    else:
                        print(f"  [{label}] {stock_id} 尚無今日資料，不標記 flag（稍後重跑會再抓，queue剩餘 {left} 支）")
            except requests.exceptions.HTTPError as e:
                if e.response is None or e.response.status_code != 404:
                    print(f"  [{label}] {stock_id} 失敗: {e}（queue剩餘 {left} 支）")
            except Exception as e:
                print(f"  [{label}] {stock_id} 失敗: {e}（queue剩餘 {left} 支）")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker) for _ in range(workers)]
        for f in futures:
            f.result()


def _update_day_fubon_fast(q: "queue.Queue", date_str: str, sdk, buffer: list, buffer_lock: threading.Lock):
    """富邦 Phase 1（queue模式，僅 update_day()/adjusted=False 使用）：
    `ThreadPoolExecutor(max_workers=_INTRADAY_FAST_WORKERS)` 併發搶 queue
    裡的股票，呼叫 `_download_day_fubon_intraday()`（intraday_candles
    聚合成日K，官方限制300次/分鐘，只拿「今天」）。跟
    fubon/marketdata_ws.py::FubonM1Collector._backfill_intraday() 同樣
    用一個「函式內共用」的 rate_lock 控制整體請求間隔在
    `_INTRADAY_FAST_INTERVAL=0.25秒`（不是每條執行緒各自 sleep——那樣
    合計會超過300次/分鐘限制）。

    2026-08-13改：呼叫端（_update_day_generic()）只會把「還沒確認有
    歷史缺口」的股票放進這個 queue——已知有缺口的股票不需要在這裡多打
    一次「今天」，反正 Phase 2 的 _update_day_fubon_slow() 本身就會抓到
    今天（省流量，見 _update_day_generic() docstring）。adjusted=True
    （update_adjustment_day()）不支援這條路徑——intraday_candles() 不
    支援 adjusted 參數，_update_day_generic() 不會呼叫這支。

    空結果不標記完成：說明同 _update_day_fugle()。⚠️ 2026-08-13再改：
    不再逐支存檔，改成疊加進 buffer，見 _flush_day_buffer() docstring
    的詳細說明（實測逐支存檔本身就是比300次/分鐘更嚴重的瓶頸，跟這裡
    的併發下載無關）。"""
    from concurrent.futures import ThreadPoolExecutor

    rate_lock = threading.Lock()
    last_req = [0.0]

    def _worker():
        while True:
            try:
                stock_id = q.get_nowait()
            except queue.Empty:
                return
            left = q.qsize()
            with rate_lock:
                wait = _INTRADAY_FAST_INTERVAL - (time.time() - last_req[0])
                if wait > 0:
                    time.sleep(wait)
                last_req[0] = time.time()
            try:
                df = _download_day_fubon_intraday(sdk, stock_id)
                if not df.empty:
                    has_today = _has_today_data(df, date_str)
                    with buffer_lock:
                        buffer.append((stock_id, df, has_today))
                    if has_today:
                        print(f"  [富邦-intraday] {stock_id} 下載完成 {len(df)} 筆，等批次存檔（queue剩餘 {left} 支）")
                    else:
                        print(f"  [富邦-intraday] {stock_id} 尚無今日資料，不標記 flag（稍後重跑會再抓，queue剩餘 {left} 支）")
            except Exception as e:
                print(f"  [富邦-intraday] {stock_id} 失敗: {e}（queue剩餘 {left} 支）")

    with ThreadPoolExecutor(max_workers=_INTRADAY_FAST_WORKERS) as pool:
        futures = [pool.submit(_worker) for _ in range(_INTRADAY_FAST_WORKERS)]
        for f in futures:
            f.result()


def _update_day_fubon_slow(
    q: "queue.Queue",
    stock_start_dates: dict[str, str],
    date_str: str,
    sdk,
    buffer: list,
    buffer_lock: threading.Lock,
    adjusted: bool = False,
):
    """富邦慢路徑（queue模式）：單執行緒序列搶 queue 裡的股票，呼叫
    `_download_day_fubon()`（historical_candles，官方限制60次/分鐘留
    緩衝，見 _FUBON_INTERVAL 說明——2026-08-11修正：不是300次/分鐘，
    那是另一個端點家族的限制），節流1.05秒/支，本身有自動補到位的能力
    （用 stock_start_dates 裡的正確起始日期）。

    維持單執行緒不加併發——60次/分鐘本身就慢，併發能省的時間有限
    （理論下限是股數/60分鐘），不值得為了有限的改善再冒一次誤判 rate
    limit 導致429的風險。

    adjusted=True（update_adjustment_day()）沒有快路徑，全部股票都走
    這支，是 _update_day_generic() 裡唯一會用到的富邦下載邏輯；
    adjusted=False（update_day()）時只有 Phase 2（真的有歷史缺口的
    股票）才會用到。

    stock_start_dates/空結果不標記完成：說明同 _update_day_fugle()。
    ⚠️ 2026-08-13再改：不再逐支存檔，改成疊加進 buffer，見
    _flush_day_buffer() docstring 的詳細說明。"""
    while True:
        try:
            stock_id = q.get_nowait()
        except queue.Empty:
            return
        left = q.qsize()
        try:
            df = _download_day_fubon(sdk, stock_id, stock_start_dates[stock_id], adjusted=adjusted)
            if not df.empty:
                has_today = _has_today_data(df, date_str)
                with buffer_lock:
                    buffer.append((stock_id, df, has_today))
                if has_today:
                    print(f"  [富邦] {stock_id} 下載完成 {len(df)} 筆，等批次存檔（queue剩餘 {left} 支）")
                else:
                    print(f"  [富邦] {stock_id} 尚無今日資料，不標記 flag（稍後重跑會再抓，queue剩餘 {left} 支）")
        except Exception as e:
            print(f"  [富邦] {stock_id} 失敗: {e}（queue剩餘 {left} 支）")
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
    參數跟輸出目錄/flag檔——見這兩支函式各自的 docstring。

    2026-08-13大改（跟 data/m1_data_loader.py::update_m1() 同一輪）：
    從「事先切三份固定清單」改成「共用queue動態搶」，理由見
    _update_day_fugle()/_update_day_fubon_fast() 的 docstring。

    只有 `not adjusted and start_date is None`（也就是 update_day() 的
    日常排程路徑）才拆 Phase 1/Phase 2：Phase 1 只補「還沒確認有歷史
    缺口」的股票（`phase1_stocks = wait_stocks - gap_stocks`）的
    「今天」；已知有缺口的股票（`gap_stocks`）不放進 Phase 1，直接留給
    Phase 2 補——Phase 2 的慢路徑本身預設就會抓到今天（end_date 預設
    今天），Phase 1 幫這些股票多打一次「今天」等於白抓，省流量。其餘
    情況（update_adjustment_day()，或手動指定 start_date 強制範圍）
    沒有快路徑可用，維持單一階段的共用queue（一樣動態分配，但不分
    Phase）。"""
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

    gap_stocks: list = []
    phase1_stocks: list = []
    two_phase = False
    if start_date is not None:
        stock_start_dates = {sid: start_date for sid in wait_stocks}
        start_date_desc = start_date
    else:
        last_dates = _last_stored_dates(base_dir=base_dir)
        default_start = f"{now.year}-{now.month:02d}-01"
        stock_start_dates = {sid: last_dates.get(sid, default_start) for sid in wait_stocks}
        behind = sorted(set(stock_start_dates.values()))
        start_date_desc = f"逐支查詢（{len(behind)} 種不同起始日期，最舊 {behind[0] if behind else '無'}）"

        # 2026-08-11加：只有 adjusted=False（update_day()，db/d1原始）才拆
        # Phase 1/Phase 2——intraday_candles() 不支援 adjusted 參數，
        # update_adjustment_day() 用不到富邦快路徑。
        # ⚠️ 一定要在 Phase 1 開始前（任何下載發生之前）就算好
        # gap_stocks，不能等 Phase 1 跑完才查——Phase 1 會幫「全部」股票
        # （含真的有缺口的）都補上「今天」，寫進 base_dir 後那支股票的
        # 最新日期會直接變成今天，中間沒真正回補的缺口會被完全蓋掉、
        # 事後查就再也看不出來了（跟 data/m1_data_loader.py::update_m1()
        # 同一輪的教訓一樣）。跟「今天理論上該有的上一個交易日」比較，
        # 不是跟股票池裡彼此的最新日期比較——見
        # _expected_prior_trading_day() 的說明，避免整個 pipeline 昨天
        # 沒跑時，全部股票互相比對誤判成「已跟上進度」。
        if not adjusted and last_dates:
            two_phase = True
            expected_prior = _expected_prior_trading_day(now.date())
            gap_stocks = [sid for sid in wait_stocks if last_dates.get(sid, "") < expected_prior]
            gap_set = set(gap_stocks)
            # Phase 2 的慢路徑（_download_day_fubon()/_download_day()）本身
            # 預設就會抓到「今天」（end_date 預設今天），所以已經確認有
            # 缺口的股票不用在 Phase 1 再多打一次「今天」——反正 Phase 2
            # 會把它整段（含今天）重新抓一次，Phase 1 抓的等於白抓，
            # 省下這些流量/API額度（尤其富邦快路徑是併發搶，queue裡少放
            # 這些注定被 Phase 2 蓋過的股票，能把併發資源留給真正只缺
            # 今天的股票）。
            phase1_stocks = [sid for sid in wait_stocks if sid not in gap_set]
            print(
                f"[{base_dir}] 預期上一交易日={expected_prior}，{len(gap_stocks)} 支股票目前有歷史缺口"
                "（跳過Phase 1直接排進Phase 2補，省流量）"
            )

    print(
        f"[{base_dir}] start_date={start_date_desc}，{len(wait_stocks)} 支股票，"
        f"Fugle x2（各 {workers} 併發）+ 富邦 從共用queue搶下載..."
    )

    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    trade_api.init_market_data(sdk)
    try:
        if two_phase:
            # ── Phase 1：排除已知有缺口的股票，Fugle x2 + 富邦(intraday併發) 搶共用queue ──
            print(f"[{base_dir}] Phase 1：{len(phase1_stocks)} 支股票...")
            q1 = queue.Queue()
            for sid in phase1_stocks:
                q1.put(sid)
            buffer1: list = []
            buffer1_lock = threading.Lock()
            t_fugle1 = threading.Thread(
                target=_update_day_fugle,
                args=(q1, stock_start_dates, date_str, workers, buffer1, buffer1_lock),
                kwargs={"adjusted": adjusted},
            )
            t_fugle2 = threading.Thread(
                target=_update_day_fugle,
                args=(q1, stock_start_dates, date_str, workers, buffer1, buffer1_lock, fugle_api.TOKEN_DAYTRADE, "Fugle-DT"),
                kwargs={"adjusted": adjusted},
            )
            t_fubon = threading.Thread(
                target=_update_day_fubon_fast,
                args=(q1, date_str, sdk, buffer1, buffer1_lock),
            )
            for t in (t_fugle1, t_fugle2, t_fubon):
                t.start()
            for t in (t_fugle1, t_fugle2, t_fubon):
                t.join()
            _flush_day_buffer(buffer1, date_str, base_dir=base_dir, flag_path=flag_path)
            print(f"[{base_dir}] Phase 1 完成")

            # ── Phase 2：補 Phase 1 開始前就已經確認有缺口的股票 ──
            print(f"[{base_dir}] Phase 2：{len(gap_stocks)} 支股票有歷史缺口，補歷史資料...")
            if gap_stocks:
                q2 = queue.Queue()
                for sid in gap_stocks:
                    q2.put(sid)
                buffer2: list = []
                buffer2_lock = threading.Lock()
                t_fugle1 = threading.Thread(
                    target=_update_day_fugle,
                    args=(q2, stock_start_dates, date_str, workers, buffer2, buffer2_lock),
                    kwargs={"adjusted": adjusted},
                )
                t_fugle2 = threading.Thread(
                    target=_update_day_fugle,
                    args=(
                        q2, stock_start_dates, date_str, workers, buffer2, buffer2_lock,
                        fugle_api.TOKEN_DAYTRADE, "Fugle-DT",
                    ),
                    kwargs={"adjusted": adjusted},
                )
                t_fubon = threading.Thread(
                    target=_update_day_fubon_slow,
                    args=(q2, stock_start_dates, date_str, sdk, buffer2, buffer2_lock),
                    kwargs={"adjusted": adjusted},
                )
                for t in (t_fugle1, t_fugle2, t_fubon):
                    t.start()
                for t in (t_fugle1, t_fugle2, t_fubon):
                    t.join()
                _flush_day_buffer(buffer2, date_str, base_dir=base_dir, flag_path=flag_path)
        else:
            # 單一階段（update_adjustment_day()，或手動指定 start_date）：
            # 沒有快路徑可用，一樣用共用queue動態分配，但不分 Phase。
            q = queue.Queue()
            for sid in wait_stocks:
                q.put(sid)
            buffer: list = []
            buffer_lock = threading.Lock()
            t_fugle1 = threading.Thread(
                target=_update_day_fugle,
                args=(q, stock_start_dates, date_str, workers, buffer, buffer_lock),
                kwargs={"adjusted": adjusted},
            )
            t_fugle2 = threading.Thread(
                target=_update_day_fugle,
                args=(q, stock_start_dates, date_str, workers, buffer, buffer_lock, fugle_api.TOKEN_DAYTRADE, "Fugle-DT"),
                kwargs={"adjusted": adjusted},
            )
            t_fubon = threading.Thread(
                target=_update_day_fubon_slow,
                args=(q, stock_start_dates, date_str, sdk, buffer, buffer_lock),
                kwargs={"adjusted": adjusted},
            )
            for t in (t_fugle1, t_fugle2, t_fubon):
                t.start()
            for t in (t_fugle1, t_fugle2, t_fubon):
                t.join()
            _flush_day_buffer(buffer, date_str, base_dir=base_dir, flag_path=flag_path)
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
