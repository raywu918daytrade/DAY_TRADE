"""
歷史分K 下載器（訓練資料用）

功能：
    從 Fugle + 富邦 REST API 下載 1 分鐘 K 線，存入 db/m1/（訓練資料庫）。
    flag 機制避免同一支股票在同一天重複下載。

價格基準（2026-08-01改）：db/m1 存的是原始（未還原權息）價格，故意不帶
adjusted 參數——需要還原後價格的地方改用 data/query.py::load_m1_adjusted()，
查詢時 join db/tick_adjust_factor 換算，不用管這筆資料當初是哪個來源
（Fugle/富邦/finmind）抓的。db/fugle_day（日K）不受影響，仍是還原後價格，
當作反推係數的基準來源。

與 fubon/marketdata_ws.py 的差異：
    m1_data_loader.py       → 下載歷史分K（近30日），存 db/m1/，給訓練用，GHA 每日觸發
    fubon/marketdata_ws.py  → 盤中富邦 WebSocket 即時推送，存 db/m1_live/，給當天交易推論用

Fugle + 富邦同時下載（2026-08-13改成共用queue+兩階段，取代原本事先切固定
清單）：
    Fugle 兩組帳號（各自單執行緒序列，60次/分鐘，historical/candles 固定
    回傳近30天資料，一支只需呼叫一次，不用再另外呼叫 intraday/candles 補
    今天）+ 富邦，一起從同一個 `queue.Queue` 搶股票下載，誰快就自然多吃，
    不會出現「份內做完就閒置、拖累整體」的情況（理由見
    _update_m1_fugle()/_update_m1_fubon_fast() 的 docstring）。
    **Phase 1**：全部候選股票，富邦用 intraday_candles（併發，只拿
    「今天」）+ Fugle 正常呼叫（固定近30天）搶共用queue。**Phase 2**：
    只補 Phase 1 開始前就已經有歷史缺口的股票，富邦改用
    historical_candles（單執行緒序列，1.05秒/支留在 60次/分鐘以內）。
    見 update_m1() 的完整說明。富邦 SDK 呼叫都透過 fubon/fubon_api.py，
    Fugle REST 呼叫都透過 fugle/fugle_api.py，這裡不直接碰 fubon_neo 或
    組 Fugle 的 URL/header。

主要函式：
    update_m1(stocks)
        stocks 預設為 _all_stocks()（見該函式，1878支全市場一般個股+0050）
        完整股票母體，兩階段從共用queue動態分配給 Fugle x2 + 富邦。
"""
import queue
import threading
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import time

import pandas as pd
import pyarrow.dataset as ds
import requests
from dotenv import load_dotenv

from fugle import fugle_api

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))

_FLAG_PATH = _ROOT / "db/m1_flags/m1_flag.parquet"

# _save_m1()/_update_flag() 都是「讀舊檔→merge→寫回去」，Fugle、富邦兩條
# 執行緒同時呼叫會搶同一個檔案（2026-07-13 在 fubon/marketdata_ws.py 實際
# 撞過一次同類型的 race condition，_atomic_to_parquet 的 .tmp 檔名也不是
# 唯一的），這裡直接用鎖序列化，兩條執行緒不會真的同時寫檔。
_save_lock = threading.Lock()
_flag_lock = threading.Lock()


def _m1_file_path(date: pd.Timestamp) -> Path:
    """月份補零，避免同一個月產生檔名不同的分檔、觸發 schema 衝突（db/fugle_day/
    已經因此撞過一次，見 2026_7.parquet vs 2026_07.parquet）。scripts/push_db_to_hf.py
    現在是直接鏡像整個 db/ 資料夾上 HF Hub，本機檔名就是 HF 上的檔名，這裡的
    命名慣例統一與否直接反映到雲端，更要保持一致。"""
    return _ROOT / f"db/m1/{date.year}_{date.month:02d}.parquet"


def _atomic_to_parquet(df: pd.DataFrame, file_path: str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    tmp_path = f"{file_path}.tmp"
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def _download_m1(stock_id: str, token: str = None) -> pd.DataFrame:
    """取得近30日1分鐘K線（Fugle分K無法指定 from/to，一律回傳近30日資料）。

    2026-07-13 實測（20:30）：這個端點本身就含「今天」的資料（最後一筆是
    當天13:30收盤），不是舊註解講的「僅到昨日」——update_m1.yml 排程在台北
    18:00 跑，永遠晚於收盤，所以不需要再另外呼叫 intraday/candles 補今天。

    token：不帶用預設 FUGLE 帳號，第二條 Fugle 執行緒（_update_m1_fugle2）
    會傳入 fugle_api.TOKEN_DAYTRADE 走另一組獨立 rate limit。

    不帶 adjusted（2026-08-01 改，故意不還原權息）：db/m1 定位改成跟 db/tick
    一樣是「原始價格」的資料層，還原權息統一交給查詢層的 load_m1_adjusted()
    （見 data/query.py，join db/tick_adjust_factor 在讀取時換算）在需要的地方
    處理。這支曾經帶過 adjusted="true"，但 db/m1 併發下載時 Fugle/富邦兩邊
    各自一半、富邦那半邊一度漏帶這個參數（2026-08-01 發現並修過一次），造成
    db/m1 內部同一支股票不同天可能一邊還原一邊沒還原——與其繼續維護「兩邊
    都要記得帶」這件事，不如兩邊都不帶，統一在查詢層做，來源是富邦/Fugle/
    finmind 都一樣是原始價格，不用再擔心哪個來源有沒有做對。db/fugle_day
    （日K）不受影響，維持 adjusted="true"，繼續當作反推係數的基準來源。
    """
    r = fugle_api.historical_candles(
        stock_id, token=token, timeframe="1", fields="open,high,low,close,volume", sort="asc"
    )
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


def _flush_m1_buffer(buffer: list, date_str: str):
    """把整批下載結果一次性存檔＋標記flag，取代逐支呼叫 _save_m1()/
    _update_flag()。

    2026-08-13發現：_save_m1() 每次呼叫都要「讀整個月份的parquet檔
    （2026-08已經150萬列）→ concat → 去重複 → 排序 → 整檔重寫」，這個
    操作被全域 _save_lock 保護，逐支呼叫的話，不管開幾條併發執行緒
    （Fugle x2 + 富邦最多10條）全部都要排隊搶同一把鎖，實測單次存檔
    約0.87秒，即使富邦快路徑本身能逼近300次/分鐘的API吞吐量，全域存檔
    這個瓶頸會把整體壓到 1/0.87≈70次/分鐘，遠低於API限制，而且檔案只會
    越存越大、越存越慢。

    改成：worker（_update_m1_fugle()/_update_m1_fubon_fast()/
    _update_m1_fubon_slow()）下載完不立刻存檔，而是把 (stock_id, df,
    has_today) 疊加進共用 buffer（append是O(1)，不會卡）；等整個 Phase
    的執行緒都 join() 完了，呼叫這支函式一次性合併成一個大DataFrame，
    只呼叫一次 _save_m1()（內部本來就會依月份分組，一次處理完全部月份，
    不會因為資料量變大而變成多次呼叫）、一次呼叫 _update_flags_batch()。

    ⚠️ 取捨：这样一个 Phase 完成前如果整個程式中途當掉，這個 Phase 已經
    下載但還沒flush的資料會遺失（原本逐支存檔則是已存檔的部分不會遺失）
    ——但 Phase 本身耗時通常從幾十分鐘壓縮到幾分鐘，風險窗口大幅縮小，
    而且本來就有 flag 機制，重跑只是重新下載、不會產生錯誤資料，用這個
    交換整體速度是值得的。"""
    if not buffer:
        return
    dfs = [df for _, df, _ in buffer]
    combined = pd.concat(dfs, ignore_index=True)
    _save_m1(combined)
    to_flag = [stock_id for stock_id, _, has_today in buffer if has_today]
    _update_flags_batch(to_flag, date_str)
    print(f"批次存檔完成：{len(buffer)} 支（{len(combined)} 筆），標記 {len(to_flag)} 支flag")


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


def _update_flags_batch(stock_ids: list, date_str: str):
    """一次幫多支股票標記 flag（一次讀寫），取代逐支呼叫 _update_flag()。
    2026-08-13加：搭配 _flush_m1_buffer() 使用——見該函式說明。"""
    if not stock_ids:
        return
    with _flag_lock:
        new_rows = pd.DataFrame({"stock_id": stock_ids, "date": date_str})
        if os.path.exists(_FLAG_PATH):
            df = pd.read_parquet(_FLAG_PATH)
            df = pd.concat([df, new_rows], ignore_index=True)
            df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
        else:
            df = new_rows
        os.makedirs(os.path.dirname(_FLAG_PATH), exist_ok=True)
        _atomic_to_parquet(df, _FLAG_PATH, index=False)


def _download_m1_fubon(sdk, stock_id: str) -> pd.DataFrame:
    """近30日+今日分K，富邦 historical/candles 一次就含今天，不用另外呼叫
    intraday/candles。給「還沒跟上進度」（缺口>1天或全新股票）的股票用，
    已跟上進度的股票改用更快的 _download_m1_fubon_intraday()（見該函式
    說明），這支保留給需要補一段範圍的情況，本身就有自動補到近30天缺口
    的能力，不用額外寫回補邏輯。

    不帶 adjusted（2026-08-01 改，故意不還原權息）：說明同 _download_m1()——
    db/m1 現在統一存原始價格，還原交給查詢層的 load_m1_adjusted() 處理。"""
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


def _download_m1_fubon_intraday(sdk, stock_id: str) -> pd.DataFrame:
    """2026-08-11加：只抓「今天」的分K（intraday/candles，官方限制
    300次/分鐘，比 historical_candles() 的60次/分鐘快5倍），給已經跟上
    進度（db/m1最新資料剛好是「今天理論上該有的上一個交易日」，見
    data.day_data_loader._expected_prior_trading_day()）的股票用，不用像
    _download_m1_fubon() 那樣每次都抓一次近30天窗口——只差今天這一根，
    抓30天是浪費的。回傳欄位/格式對齊 _download_m1_fubon()。

    只有富邦這個端點有這個優勢，Fugle 的 intraday 跟 historical 都是
    60次/分鐘、沒有差異（見 bugs_and_todos.md 的 rate limit 確認表），
    所以這支只給 _update_m1_fubon() 用，Fugle 那兩路不變。"""
    from fubon import fubon_api as trade_api

    bars = trade_api.intraday_candles(sdk, stock_id, timeframe=1)
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
    """股票母體（2026-08-08改）：讀
    finmind.stock_universe_2000.load_stock_universe_2000_with_0050()
    （1877支全市場一般個股+強制併入0050，共1878支），取代原本400支
    tick_universe——跟 data/day_data_loader.py::_all_stocks() 統一成同一份
    清單（理由對稱：日K母體已經先擴大，分K母體要跟著擴大，避免大量股票
    「有日K沒分K」，跨資料源特徵算不出來）。母體來源改了不影響下載邏輯
    本身，Fugle/富邦還是各自下載三分之一。"""
    from finmind.stock_universe_2000 import load_stock_universe_2000_with_0050

    return load_stock_universe_2000_with_0050()


def _get_done_stocks(date_str: str) -> set:
    if not os.path.exists(_FLAG_PATH):
        return set()
    df = pd.read_parquet(_FLAG_PATH)
    return set(df[df["date"] == date_str]["stock_id"].tolist())


def _has_today_data(df: pd.DataFrame, date_str: str) -> bool:
    """判斷下載回來的近30日資料是否真的包含「今天」這一天。如果在開盤前
    或盤中很早執行，近30日回傳可能還沒有今天的K線（df 非空，但最新一筆
    只到昨天），此時不能標記 flag，否則收盤後重跑會被誤判成「今天已處理
    過」而跳過，導致當天真正的分鐘K永遠抓不到（2026-07-29 發現：早上跑過
    一次後，下午重跑 2000 多支股票被跳過，db/m1 裡當天資料只剩早上抓到的
    那一小批）。"""
    return bool(df["date"].str.startswith(date_str).any())


def _last_stored_dates_m1() -> dict:
    """db/m1 每支股票目前最新的存檔日期（只看最近2個月份檔案，1分K資料量
    大，不用像 data.day_data_loader._last_stored_dates() 那樣掃全部月份
    檔案——「最新日期」不可能藏在更早的月份裡）。回傳
    {stock_id: "YYYY-MM-DD"}（只取日期部分）。股票沒出現在這2個月裡就不在
    這個 dict 裡，呼叫端視為「落後很多」處理，走慢路徑。"""
    m1_dir = _ROOT / "db/m1"
    if not m1_dir.exists():
        return {}
    files = sorted(f for f in m1_dir.iterdir() if f.suffix == ".parquet")[-2:]
    if not files:
        return {}
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    if dataset.count_rows() == 0:
        return {}
    df = dataset.to_table(columns=["stock_id", "date"]).to_pandas()
    df["date"] = df["date"].astype(str).str[:10]
    return df.groupby("stock_id")["date"].max().to_dict()


_INTRADAY_FAST_WORKERS = 8  # 2026-08-14從10調降：GHA實跑撞到約7.5%(140/1864)的429，降併發數當第一道防線
_INTRADAY_FAST_INTERVAL = 0.25


def _update_m1_fugle(
    q: "queue.Queue",
    date_str: str,
    buffer: list,
    buffer_lock: threading.Lock,
    failed: list,
    token: str = None,
    label: str = "Fugle",
):
    """Fugle 那一路：從共用queue搶股票下載，直到queue空為止。

    2026-08-13改成queue模式（原本是 update_m1() 事先切好的固定清單）：
    富邦快路徑併發化之後比 Fugle 快很多，如果還是照事先算好的固定比例
    分配，富邦提早做完後只能閒置等 Fugle 兩路慢慢跑完，整體 wall time
    沒有真正縮短。改成 Fugle 兩組帳號＋富邦一起搶同一個 queue，誰快就
    自然多吃，不會出現「份內做完但整體還沒完成」的情況。

    historical/candles 一次就含今天（見 _download_m1() 的 2026-07-13
    實測結果），1.05秒/支（1次API），維持在 60 req/min 以內留緩衝——
    Fugle 分K無法指定 from/to，一律回傳近30日資料，速度沒辦法再優化，
    所以這支不管在 Phase 1（見 update_m1()）或 Phase 2 都是同一套邏輯，
    處理過的股票基本上順便就把近30天缺口也補了。

    token/label：讓 update_m1() 可以開兩條 Fugle 執行緒各用一組帳號
    （FUGLE / FUGLE_DAYTRADE），互不共用 rate limit。

    2026-07-26 改：查到空結果（含404）不再標記 flag——比照
    data/day_data_loader.py 同一次的修正，避免空結果（可能只是暫時性問題）
    被誤判成「今天已確認處理過」，當天重跑不會再重試。

    ⚠️ 2026-08-13再改：不再逐支呼叫 _save_m1()/_update_flag()，改成把
    (stock_id, df, has_today) 疊加進 buffer（append是O(1)，buffer_lock
    只是保護這個很輕量的操作，不會卡）——真正的存檔／標記flag，交給
    update_m1() 在這個 Phase 全部執行緒 join() 完之後，一次呼叫
    _flush_m1_buffer()。理由見該函式的 docstring：逐支存檔會被全域
    _save_lock 卡住，每次呼叫都要整檔重讀重寫，是比API rate limit更嚴重
    的瓶頸。

    ⚠️ 2026-08-14加 failed 參數：實際跑GHA時發現 Phase 1（不管Fugle還是
    富邦快路徑）真的請求失敗（不是404、是暫時性錯誤如rate limit——實測
    一次GHA執行就有~140支股票因為富邦intraday撞到429失敗）時，原本這批
    股票當天完全沒有第二次機會，要等隔天gap偵測才會被抓回來補。現在把
    這種真正的請求失敗記進 failed（線程安全，用同一把 buffer_lock 保護，
    append本身很輕量不會卡），update_m1() 收集完 Phase 1 全部 failed 後
    會併入 Phase 2 的 queue，同一天內就用慢路徑補回來，不用等到隔天。
    404（真的沒有這支股票）跟「無資料」不算這種失敗，不會被加進去——
    404是確定性的、重試也沒用；「無資料」保留給隔天的flag機制自然重試，
    避免genuinely沒資料的股票每天都被硬塞進Phase 2浪費一次慢路徑額度。"""
    while True:
        try:
            stock_id = q.get_nowait()
        except queue.Empty:
            return
        left = q.qsize()
        try:
            df = _download_m1(stock_id, token=token)
            if not df.empty:
                has_today = _has_today_data(df, date_str)
                with buffer_lock:
                    buffer.append((stock_id, df, has_today))
                if has_today:
                    print(f"[{label}] {stock_id} 下載完成 {len(df)} 筆，等批次存檔（queue剩餘 {left} 支）")
                else:
                    print(f"[{label}] {stock_id} 尚無今日資料，不標記 flag（稍後重跑會再抓，queue剩餘 {left} 支）")
            else:
                print(f"[{label}] {stock_id} 無資料（queue剩餘 {left} 支）")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"[{label}] {stock_id} 無此股票資料（queue剩餘 {left} 支）")
            else:
                with buffer_lock:
                    failed.append(stock_id)
                print(f"[{label}] {stock_id} 失敗: {e}（queue剩餘 {left} 支，追加進Phase 2重試）")
        except Exception as e:
            with buffer_lock:
                failed.append(stock_id)
            print(f"[{label}] {stock_id} 失敗: {e}（queue剩餘 {left} 支，追加進Phase 2重試）")
        time.sleep(1.05)


def _update_m1_fubon_fast(
    q: "queue.Queue", date_str: str, sdk, buffer: list, buffer_lock: threading.Lock, failed: list
):
    """富邦 Phase 1（queue模式）：`ThreadPoolExecutor(_INTRADAY_FAST_WORKERS)`
    併發搶 queue 裡的股票，呼叫 `_download_m1_fubon_intraday()`
    （intraday_candles，官方限制300次/分鐘，只拿「今天」）。

    2026-08-13改：呼叫端（update_m1()）只會把「還沒確認有歷史缺口」的
    股票放進這個 queue——已知有缺口的股票（`gap_stocks`）不需要在這裡
    多打一次「今天」，反正 Phase 2 的慢路徑 `_update_m1_fubon_slow()`
    本身就會帶到今天（historical_candles 一次含近30天+今天），省流量，
    見 update_m1() docstring。

    共用一個函式內 `rate_lock` 控制整體請求間隔在
    `_INTRADAY_FAST_INTERVAL=0.25秒`（不是每條執行緒各自 sleep 0.25秒
    ——那樣多條執行緒合計會超過300次/分鐘限制）。改之前是單執行緒序列
    迴圈，每筆請求的實際網路延遲遠大於0.25秒的節流下限，2026-08-13
    實測 log 間隔到 5~7秒/支，離300/分鐘的理論吞吐量差很多，改成併發
    才能真正隱藏單一請求的延遲、逼近理論上限。

    空結果不標記完成：說明同 _update_m1_fugle()。⚠️ 2026-08-13再改：跟
    _update_m1_fugle() 同一輪的修改，不再逐支存檔，改成疊加進 buffer，
    見該函式 docstring 的詳細說明（實測逐支存檔本身就是比300次/分鐘更
    嚴重的瓶頸，跟這裡的併發下載無關）。

    ⚠️ 2026-08-14加 failed 參數：理由/語意同 _update_m1_fugle() 的
    failed 參數說明——實測GHA一次執行約140支股票在這裡因為富邦intraday
    撞到429失敗，記進 failed 讓 update_m1() 併入 Phase 2 同一天內用
    慢路徑補回來，不用等隔天。"""
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
                df = _download_m1_fubon_intraday(sdk, stock_id)
                if not df.empty:
                    has_today = _has_today_data(df, date_str)
                    with buffer_lock:
                        buffer.append((stock_id, df, has_today))
                    if has_today:
                        print(f"[富邦-intraday] {stock_id} 下載完成 {len(df)} 筆，等批次存檔（queue剩餘 {left} 支）")
                    else:
                        print(f"[富邦-intraday] {stock_id} 尚無今日資料，不標記 flag（稍後重跑會再抓，queue剩餘 {left} 支）")
                else:
                    print(f"[富邦-intraday] {stock_id} 無資料（queue剩餘 {left} 支）")
            except Exception as e:
                with buffer_lock:
                    failed.append(stock_id)
                print(f"[富邦-intraday] {stock_id} 失敗: {e}（queue剩餘 {left} 支，追加進Phase 2重試）")

    with ThreadPoolExecutor(max_workers=_INTRADAY_FAST_WORKERS) as pool:
        futures = [pool.submit(_worker) for _ in range(_INTRADAY_FAST_WORKERS)]
        for f in futures:
            f.result()


def _update_m1_fubon_slow(q: "queue.Queue", date_str: str, sdk, buffer: list, buffer_lock: threading.Lock):
    """富邦 Phase 2（queue模式）：單執行緒序列搶 queue 裡的股票，呼叫
    `_download_m1_fubon()`（historical_candles，官方限制60次/分鐘留
    緩衝——2026-08-11修正：2026-08-08一度誤改成0.25秒/300次分鐘，是跟
    fubon/tick_api.py 的 intraday/trades 端點搞混，那是另一個限流更
    寬鬆的端點家族，historical_candles 實際上跟 Fugle 一樣是60次/分鐘，
    見 data.day_data_loader._FUBON_INTERVAL 說明），節流1.05秒/支，
    本身有自動補近30天缺口的能力。

    只有 Phase 2（真的有歷史缺口，見 update_m1()）會用到這支，維持
    單執行緒不加併發——60次/分鐘本身就慢，併發能省的時間有限（理論
    下限是股數/60分鐘），不值得為了有限的改善再冒一次誤判 rate limit
    導致429的風險。

    ⚠️ 2026-08-13再改：不再逐支存檔，改成疊加進 buffer，見
    _update_m1_fugle() docstring 的詳細說明。"""
    while True:
        try:
            stock_id = q.get_nowait()
        except queue.Empty:
            return
        left = q.qsize()
        try:
            df = _download_m1_fubon(sdk, stock_id)
            if not df.empty:
                has_today = _has_today_data(df, date_str)
                with buffer_lock:
                    buffer.append((stock_id, df, has_today))
                if has_today:
                    print(f"[富邦] {stock_id} 下載完成 {len(df)} 筆，等批次存檔（queue剩餘 {left} 支）")
                else:
                    print(f"[富邦] {stock_id} 尚無今日資料，不標記 flag（稍後重跑會再抓，queue剩餘 {left} 支）")
            else:
                print(f"[富邦] {stock_id} 無資料（queue剩餘 {left} 支）")
        except Exception as e:
            print(f"[富邦] {stock_id} 失敗: {e}（queue剩餘 {left} 支）")
        time.sleep(1.05)


def update_m1(stocks: list = None):
    """1分鐘K線，flag避免同日重複下載。

    2026-08-13大改：從「事先切三份固定清單」改成「兩階段＋共用queue」。
    動機：富邦快路徑（intraday_candles,300次/分鐘）併發化之後比 Fugle
    （60次/分鐘,固定近30天，見 _download_m1() docstring：分K無法指定
    from/to）快很多，如果還是照 _split_by_rate() 事先切好固定比例，
    富邦那份提早做完後只能閒置等 Fugle 兩路慢慢跑完，整體 wall time
    沒有真正縮短。改成 Fugle 兩組帳號＋富邦一起搶同一個 queue，誰快就
    自然多吃，不會出現「份內做完但整體還沒完成」的情況。

    **Phase 1**（`wait_stocks` 排除掉已知有缺口的股票）：建一個共用
    queue，Fugle兩組帳號（各自單執行緒序列 `_update_m1_fugle()`）＋
    富邦（`_update_m1_fubon_fast()`，ThreadPoolExecutor併發，
    intraday_candles只拿「今天」）一起搶。Fugle處理過的股票這階段就
    順便把近30天缺口也補了（沒辦法只拿今天）。

    ⚠️ 2026-08-13再加一版：已經確認有缺口的股票（`gap_stocks`，見下面
    Phase 2 說明）**不放進 Phase 1 的queue**——這些股票反正等一下
    Phase 2 的慢路徑（`historical_candles`）本身就會帶到「今天」（見
    `_download_m1_fubon()` docstring），Phase 1 再多打一次只拿今天的
    intraday等於白抓，省下這些流量/API額度（尤其富邦快路徑是併發搶，
    queue裡少放這些注定要被 Phase 2 蓋過的股票，能把併發資源留給真正
    只缺今天的股票）。

    **Phase 2**（事後檢查缺口，只有真的落後的才進來）：Phase 1 跑完後，
    重新讀一次 db/m1 目前每支股票的最新存檔日期，跟
    `data.day_data_loader._expected_prior_trading_day()` 算出的「今天
    理論上該有的上一個交易日」比較——**嚴格小於**才算真缺口（等於代表
    只是今天資料還沒公佈，不是缺口，不需要進 Phase 2，下次重跑會自然
    補上）。這些股票一樣建共用queue，Fugle兩組帳號＋富邦
    （`_update_m1_fubon_slow()`，單執行緒序列，historical_candles慢
    路徑）一起搶，補齊。

    ⚠️ 2026-08-13（同一輪）：判斷「有沒有缺口」原本是跟股票池自己的
    最大日期比較（相對值），這樣若整個排程當天忘記執行，隔天所有股票
    會「相對彼此」一樣落後、誤判成全部跟上進度。已改成跟
    `_expected_prior_trading_day()` 算出來的絕對日期比較。

    ⚠️⚠️ 這個比較**必須在 Phase 1 開始前**就做完、把結果記下來，不能等
    Phase 1 跑完才查——Phase 1 會幫「全部」股票（含真的有缺口的）都補上
    「今天」這一筆，寫進 db/m1 後，那支股票在檔案裡的最新日期會直接變成
    「今天」，這時候才查 `_last_stored_dates_m1()` 只會看到「今天」，
    中間那段沒真正回補的缺口會被完全蓋掉、永遠看不出來（實測過：一支
    股票原本卡在6天前，Phase 1 補了今天的intraday之後，事後查最新日期
    就直接變成今天，缺口從此消失、Phase 2 不會再抓它）。所以這裡先用
    Phase 1 開始前的快照決定 `gap_stocks`，Phase 1 造成的寫入不會影響
    這個判斷結果。這支函式刻意不處理台股實際的國定假日，判斷錯了頂多
    多走一次 Phase 2，不會漏資料。

    ⚠️ 2026-08-14加：Phase 2 除了原本的 `gap_stocks`，還會併入 **Phase 1
    這次真的請求失敗**（rate limit等暫時性錯誤，不是404、也不是單純空
    結果）的股票——實測GHA一次執行就有約140支股票在富邦intraday快路徑
    撞到429失敗，原本這批股票當天完全沒有第二次機會，要等隔天gap偵測
    才會被抓回來補，現在同一天內就用Phase 2的慢路徑補回來。用
    `set(gap_stocks) | set(failed1)` 去重，避免同一支股票被塞進queue
    兩次。"""
    if not fugle_api.TOKEN:
        raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")
    if not fugle_api.TOKEN_DAYTRADE:
        raise RuntimeError("缺少第二組 Fugle API Key，請在 .env 設定 FUGLE_DAYTRADE")

    now = datetime.now(_TW)
    date_str = now.strftime("%Y-%m-%d")
    os.makedirs(_ROOT / "db/m1", exist_ok=True)

    candidates = _all_stocks() if stocks is None else stocks
    done = _get_done_stocks(date_str)
    wait_stocks = [s for s in candidates if s not in done]
    print("還有", len(wait_stocks), "個股票未更新（已排除今日已下載 flag）")

    # ⚠️ 一定要在 Phase 1 寫入任何資料之前算好 gap_stocks，見上面 docstring
    # 的說明——Phase 1 補「今天」會讓每支股票的最新日期直接跳成今天，
    # 事後才查會把中間的缺口蓋掉、永遠看不出來。
    from data.day_data_loader import _expected_prior_trading_day

    last_dates_before = _last_stored_dates_m1()
    expected_prior = _expected_prior_trading_day(now.date())
    gap_stocks = [sid for sid in wait_stocks if last_dates_before.get(sid, "") < expected_prior]
    gap_set = set(gap_stocks)
    # Phase 2 的慢路徑（historical_candles）本身就會帶到「今天」（見
    # _download_m1_fubon() docstring），所以已經確認有缺口的股票不需要
    # 在 Phase 1 再多打一次「今天」——反正等一下 Phase 2 就會把它整段
    # （含今天）重新抓一次，Phase 1 抓的等於白抓，省下這些流量/API額度。
    phase1_stocks = [sid for sid in wait_stocks if sid not in gap_set]
    print(
        f"預期上一交易日={expected_prior}，{len(gap_stocks)} 支股票目前有歷史缺口"
        f"（跳過Phase 1直接排進Phase 2補，省流量）"
    )

    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    trade_api.init_market_data(sdk)
    try:
        # ── Phase 1：排除已知有缺口的股票，Fugle x2 + 富邦(intraday併發) 搶共用queue ──
        print(f"Phase 1：{len(phase1_stocks)} 支股票，Fugle x2 + 富邦(intraday併發) 從共用queue搶下載...")
        q1 = queue.Queue()
        for sid in phase1_stocks:
            q1.put(sid)
        buffer1: list = []
        buffer1_lock = threading.Lock()
        failed1: list = []
        t_fugle1 = threading.Thread(target=_update_m1_fugle, args=(q1, date_str, buffer1, buffer1_lock, failed1))
        t_fugle2 = threading.Thread(
            target=_update_m1_fugle,
            args=(q1, date_str, buffer1, buffer1_lock, failed1, fugle_api.TOKEN_DAYTRADE, "Fugle-DT"),
        )
        t_fubon = threading.Thread(
            target=_update_m1_fubon_fast, args=(q1, date_str, sdk, buffer1, buffer1_lock, failed1)
        )
        for t in (t_fugle1, t_fugle2, t_fubon):
            t.start()
        for t in (t_fugle1, t_fugle2, t_fubon):
            t.join()
        _flush_m1_buffer(buffer1, date_str)
        print("Phase 1 完成")

        # ── Phase 2：補 Phase 1 開始前就已經確認有缺口的股票，
        # 加上 Phase 1 這次真的請求失敗（rate limit等暫時性錯誤，不是
        # 404）的股票——2026-08-14加：實測GHA一次執行約140支股票在富邦
        # intraday快路徑撞到429失敗，原本要等隔天gap偵測才會被抓回來，
        # 現在同一天內就用慢路徑補，不用等隔天。用 set 去重，避免同一支
        # 股票同時在 gap_stocks 跟 failed1 裡導致 queue 重複塞兩次。
        gap_stocks_final = sorted(set(gap_stocks) | set(failed1))
        if failed1:
            print(f"Phase 1 有 {len(failed1)} 支股票請求失敗，併入 Phase 2 同一天內重試")
        print(f"Phase 2：{len(gap_stocks_final)} 支股票有歷史缺口，補歷史資料...")
        if gap_stocks_final:
            q2 = queue.Queue()
            for sid in gap_stocks_final:
                q2.put(sid)
            buffer2: list = []
            buffer2_lock = threading.Lock()
            failed2: list = []
            t_fugle1 = threading.Thread(
                target=_update_m1_fugle, args=(q2, date_str, buffer2, buffer2_lock, failed2)
            )
            t_fugle2 = threading.Thread(
                target=_update_m1_fugle,
                args=(q2, date_str, buffer2, buffer2_lock, failed2, fugle_api.TOKEN_DAYTRADE, "Fugle-DT"),
            )
            t_fubon = threading.Thread(target=_update_m1_fubon_slow, args=(q2, date_str, sdk, buffer2, buffer2_lock))
            for t in (t_fugle1, t_fugle2, t_fubon):
                t.start()
            for t in (t_fugle1, t_fugle2, t_fubon):
                t.join()
            _flush_m1_buffer(buffer2, date_str)
            if failed2:
                print(f"Phase 2 仍有 {len(failed2)} 支股票請求失敗，留給下次重跑（flag機制自然重試）")
    finally:
        trade_api.logout(sdk)

    print("全部完成")


if __name__ == "__main__":
    update_m1()
