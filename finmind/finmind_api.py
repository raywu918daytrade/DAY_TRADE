"""
FinMind API 封裝 — 下載歷史分K（TaiwanStockKBar），補齊 db/m1/ 更早的歷史。

動機：db/m1/ 目前主要靠 Fugle historical/candles 補，但那支 API 有「近30日」
硬限制，沒辦法一次拉更久以前的資料，db/m1 只能每天執行 update_m1() 慢慢累積
（見 data/m1_data_loader.py）。FinMind 的 TaiwanStockKBar 資料集有更長的分K
歷史（資料區間：2019-01-01 ~ now），可以用來一次性補齊缺的月份/日期。

限制（FinMind 官方文件 + 2026-07-13 實測）：
    - 單次請求只回傳「一天」資料（不支援 from/to 區間，不像 Fugle 那樣可以一次
      拉一年），要補一段期間必須逐日、逐股票各發一次請求。
    - 需要 FinMind Sponsor 付費會員權限（一般token會被拒絕，訊息是
      "Token is illegal"，跟token過期/格式錯無關，是帳號權限問題）。
    - Rate limit：6000 次/小時（.env 的 FINMIND_TOKEN 帳號實測值），這支檔案
      用滑動視窗限流（_RateLimiter）確保不超過，不是單純均勻節流。

volume 單位：實測跟 db/m1 現有資料（Fugle 來源）逐分鐘完全一致（同一天同一支
股票，兩邊每分鐘volume一模一樣），不需要轉換單位就能直接合併進 db/m1。
（附註：db/m1 這欄跟 db/fugle_day 的 volume 不是同一個單位，db/fugle_day
是「股」，db/m1 是「張」，這是既有資料的另一件事，跟這支檔案無關，不在這裡
處理。）

用法：
    python -m finmind.api                # 補齊 2026-05（預設，見 __main__）
    python -m finmind.api 2026 4          # 補齊指定年月
"""

import asyncio
import builtins as _builtins
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))

# 加時間戳記 + 強制 flush（比照 main/live_trader.py 同樣的 monkey-patch 做法，
# 見 CLAUDE.md/memory 的 feedback_coding：廣泛性修改用monkey-patch，不逐一
# 改每個print）。2026-07-14 教訓：背景執行時 nohup 重導向到檔案，Python
# 預設對非終端機輸出是block-buffered，log檔案會長時間看起來是空的，即使
# 程式其實有在正常運作、也真的在消耗API額度——一定要flush=True才能即時
# 看到進度，不然中途想確認狀況/決定要不要kill掉程序時完全是盲飛。
_orig_print = _builtins.print


def _ts_print(*args, **kwargs):
    ts = datetime.now(_TW).strftime("%H:%M:%S")
    kwargs.setdefault("flush", True)
    _orig_print(f"[{ts}]", *args, **kwargs)


_builtins.print = _ts_print
_TOKEN = os.environ.get("FINMIND_TOKEN", "")
_BASE_URL = "https://api.finmindtrade.com/api/v4/data"
_USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"
_DATASET = "TaiwanStockKBar"


def reload_token() -> None:
    """重新讀 .env 的 FINMIND_TOKEN，蓋掉目前記憶體裡的值。長時間背景執行
    （見 finmind/backfill_history.py 的 run_forever()）撞到 400 TokenError 時，
    使用者可以直接改 .env 換新token，不用重啟程式，下次自動重試前呼叫這個
    就能讀到新值。"""
    global _TOKEN
    load_dotenv(_ROOT / ".env", override=True)
    _TOKEN = os.environ.get("FINMIND_TOKEN", "")


async def check_quota() -> dict:
    """查 FinMind 官方帳號用量 API（v2/user_info），回傳 user_count（已用次數）/
    api_request_limit_hour（每小時上限）等資訊，用來確認額度有沒有恢復，比
    盲目重試更準——2026-07-14 實測過這支帳號 api_request_limit_hour=6000，
    api_request_limit_day='-'（沒有另外的日/月總量限制，純小時制滾動）。"""
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {_TOKEN}"}
        async with session.get(
            _USER_INFO_URL, headers=headers, params={"token": _TOKEN}, timeout=aiohttp.ClientTimeout(total=15)
        ) as r:
            return await r.json()

_RATE_LIMIT_PER_HOUR = 6000  # 帳號實測上限，改帳號方案要跟著調
# 我們自己實際用的上限，故意比 _RATE_LIMIT_PER_HOUR 低留緩衝（2026-07-14
# 加）：本地 _RateLimiter 只認得這個process自己送出的request，跟伺服器端
# 真實計數之間難免有些微時間差/誤差（例如剛好卡在滑動視窗邊界、或
# check_quota()本身也會佔用一點額度），卡在跟6000貼死的邊界跑，一有誤差
# 就會真的撞到402。主動早一點（5500）就自我節流，留500的安全空間。
_SAFE_LIMIT_PER_HOUR = 5500
# 同時併發請求數。2026-07-14 教訓：測試時用40併發、25.7秒內衝了7285組請求，
# 被FinMind判定異常流量、整個IP封鎖30分鐘。改成 _RateLimiter 強制間隔
# 3600/6000=0.6秒（見 _RateLimiter 說明）之後，併發數已經不影響吞吐量——
# 瓶頸從頭到尾都是額度（6000/小時），不是「我們發得不夠快」，併發只會讓
# 撞到fatal error時同時在飛行中受影響的request數變多，沒有任何好處，
# 改成1（完全序列化），流量形狀最平穩、最不像爬蟲。
_CONCURRENCY = 1


class FatalAPIError(RuntimeError):
    """任何 4xx 狀態碼的共同基底類別。2026-07-14 教訓：一開始只認得特定訊息
    （token illegal、upper limit、ip banned）才停下來，其他4xx會被當成普通
    失敗一筆一筆記錄、繼續對後面幾千組請求做註定失敗的嘗試，浪費時間。改成
    「只要是4xx，一律當成需要整批任務停止、等狀況恢復再重試」，不用一一列
    舉每種訊息（見 fetch_kbar_day() 的判斷邏輯、backfill_month()/run_forever()
    怎麼處理這個基底類別）。"""


class IPBannedError(FatalAPIError):
    """403 ip banned。跟流量總量的 QuotaError 是不同機制（疑似針對瞬間爆量/
    請求模式的異常偵測，不是單純算總數），2026-07-14 實測過封鎖約30分鐘
    後自動解除。"""


class TokenError(FatalAPIError):
    """token 無效或權限不足（400 TokenIllegal / TokenLevelTooLow）。跟流量無關，
    重試沒有意義。"""


class QuotaError(FatalAPIError):
    """402 超過會員等級的請求上限。FinMind 官方說明：「請升級會員或等待下個
    計費週期」——用詞聽起來像帳單週期（可能是月）的總額度，不一定是單純的
    「1小時內」滾動限流，退避重試幾次還是一直 402 的話繼續重試沒意義。"""


class OtherFatalError(FatalAPIError):
    """沒特別辨識出來的其他 4xx（不是400/402/403，或訊息格式跟預期不同）。
    保守起見一樣當成整批任務停止、等狀況恢復再重試，不要嘗試分析原因後
    決定「這個可以繼續」——4xx本質上代表請求本身有問題，重試同樣的請求
    不會突然成功，除非外部狀況（額度/token/封鎖）改變了。"""


class NetworkError(FatalAPIError):
    """連線層級失敗（斷線/DNS/timeout），不是 FinMind 回應的 4xx。2026-07-26
    發現：這種錯誤本來會被 backfill_month()::_one() 的 `except Exception` 當成
    單筆失敗記錄、繼續往下跑，斷線期間會整批（甚至整個歷史範圍）都被當成
    失敗跑完，還誤判成「全部月份跑完」正常結束、不會像其他 FatalAPIError
    一樣暫停等待。故意繼承 FatalAPIError，讓 backfill_month()/backfill_history()/
    run_forever() 既有的「整批停止→定期呼叫 check_quota() 確認狀況→恢復後
    自動接著補」邏輯直接適用，不用另外寫一套——網路真的恢復後，
    check_quota() 自然會成功，跟等 token/額度恢復是同一種等待模式。"""


class _RateLimiter:
    """2026-07-14 從「本地模擬伺服器滑動視窗」改成更簡單直接的做法：不猜
    伺服器狀態，每 poll_interval 秒直接呼叫 check_quota() 問FinMind官方
    真實用量，低於 safe_limit 才放行，超過就等下一次定期檢查——本地模擬
    版本（seed一批同時間戳記的假資料、算滑動視窗何時過期）曾經搞出一段
    無法分辨「正常在等安全視窗」還是「真的卡住」的狀況，改成定期問真實
    數字，狀態清楚、log天然就有意義（每次檢查都印出真實用量），不用自己
    猜測、也不用擔心估算跟伺服器對不起來。

    每筆之間仍強制至少間隔 _MIN_INTERVAL 秒（=3600/6000=0.6秒，按真實
    上限算，不是安全上限），把流量攤平成穩定間隔，不要瞬間爆衝（2026-07-14
    教訓：40併發25秒內衝7285筆，就算沒超過總量上限，瞬間爆量本身可能就
    被FinMind判定異常、IP封鎖30分鐘）。

    在兩次定期檢查之間，用本地簡單計數器樂觀追蹤這個process自己新發出的
    request數（不是完整模擬滑動視窗，只是「上次真實數字 + 之後自己發了
    幾筆」），下一次 poll_interval 到了就整個重新校正，drift 最多不會超過
    一個 poll_interval 內能發的請求數（30秒 ÷ 0.6秒 ≈ 50筆，遠小於安全
    上限留的500緩衝，不會有實際風險）。"""

    def __init__(self, safe_limit: int, poll_interval: int = 10, min_interval: float = 3600.0 / 5500):
        self._safe_limit = safe_limit
        self._min_interval = min_interval  # 2026-07-14：3600/5500≈0.655秒，這個節奏長期跑下來剛好讓用量持續貼齊安全上限5500，不衝不等、不忽快忽慢
        self._poll_interval = poll_interval
        self._usage = 0
        self._usage_at_last_sync = 0
        self._usage_checked_at = 0.0
        self._last_dispatch: float | None = None
        self._lock = asyncio.Lock()
        # 2026-07-14：這個process自己實際dispatch過幾次，跟FinMind回報的
        # user_count並排印出來，才能直接比對是不是本地間隔真的沒生效，
        # 不然光看user_count漲多少，猜不出來是本地問題還是伺服器端計數
        # 本身的行為（例如批次上報、延遲更新）。
        self._dispatch_count = 0
        self._dispatch_count_at_last_sync = 0

    async def _refresh_usage(self) -> bool:
        loop = asyncio.get_event_loop()
        try:
            info = await check_quota()
        except Exception as e:
            # 2026-07-26：斷線時 check_quota() 本身也會炸例外，這支函式在正常
            # 請求流程中每 poll_interval 秒都會被呼叫一次（見 acquire()），
            # 不能讓它把整個 process 弄炸掉——比照下面 msg!="success" 的處理
            # 方式，當成「這次查詢失敗，沿用舊值，下次再試」。真正讓整批任務
            # 停下來的是 fetch_kbar_day() 拋出的 NetworkError。
            print(f"  查詢用量失敗（{e}），沿用上次已知的 {self._usage}")
            return False
        if info.get("msg") == "success":
            used = int(info.get("user_count")) if isinstance(info.get("user_count"), (int, float)) else None
            if used is not None:
                # 2026-07-14：改成算「伺服器真實用量的實際變化」（新讀數-上次真實
                # 讀數），不是「新讀數 vs 本地樂觀累加過的數字」——後者只要本地
                # 有dispatch過就一定會出現一長串正負號黏在一起的負數（例如
                # "伺服器+-19"），看起來像伺服器用量在減少，其實只是本地
                # 樂觀計數被拉回真實值，容易誤導。
                server_change = used - self._usage_at_last_sync
                dispatched_since = self._dispatch_count - self._dispatch_count_at_last_sync
                self._usage = used
                self._usage_at_last_sync = used
                self._usage_checked_at = loop.time()
                self._dispatch_count_at_last_sync = self._dispatch_count
                sign = "+" if server_change >= 0 else ""
                print(
                    f"  用量同步：{self._usage}/{_RATE_LIMIT_PER_HOUR}（安全上限 {self._safe_limit}）"
                    f"  這段期間本地送出{dispatched_since}筆，伺服器實際變化{sign}{server_change}"
                    f"（過期/回補的比新增的{'多' if server_change < dispatched_since else '少'}）"
                )
                return True
        print(f"  查詢用量失敗（{info.get('msg')}），沿用上次已知的 {self._usage}")
        return False

    async def acquire(self):
        async with self._lock:
            loop = asyncio.get_event_loop()
            while True:
                now = loop.time()
                if now - self._usage_checked_at >= self._poll_interval:
                    await self._refresh_usage()
                    now = loop.time()
                if self._usage >= self._safe_limit:
                    print(f"  用量 {self._usage}/{self._safe_limit} 已達安全上限，{self._poll_interval}秒後重新檢查")
                    await asyncio.sleep(self._poll_interval)
                    continue
                since_last = None if self._last_dispatch is None else now - self._last_dispatch
                if since_last is not None and since_last < self._min_interval:
                    await asyncio.sleep(self._min_interval - since_last)
                    continue
                self._last_dispatch = now
                self._usage += 1  # 樂觀+1，下次定期檢查會用真實數字整個校正回來
                self._dispatch_count += 1
                return


_rate_limiter = _RateLimiter(_SAFE_LIMIT_PER_HOUR)


async def sync_rate_limiter() -> dict:
    """真正開始下載前一定要呼叫這個，先跟FinMind官方用量API同步一次，
    不要讓第一筆request用「本地還沒查過、預設用量0」這種過度樂觀的假設
    送出去（見 _RateLimiter 的說明）。回傳 check_quota() 的原始回應。"""
    async with _rate_limiter._lock:
        await _rate_limiter._refresh_usage()
    return {
        "msg": "success" if _rate_limiter._usage_checked_at else "failed",
        "user_count": _rate_limiter._usage,
        "api_request_limit_hour": _RATE_LIMIT_PER_HOUR,
    }


async def fetch_kbar_day(
    session: aiohttp.ClientSession, stock_id: str, date_str: str, _retry: int = 0
) -> pd.DataFrame:
    """單一股票、單一交易日的分K。date_str 格式 YYYY-MM-DD。

    回傳欄位跟 db/m1 一致：stock_id, date("YYYY-MM-DD HH:MM:SS"), open, high,
    low, close, volume。當天沒有資料（停牌/尚未上市等）回傳空 DataFrame，
    不當例外處理——例外只保留給真正的請求失敗（網路、限流、權限問題）。

    2026-07-14 實測：帳號真實上限就是 6000/小時（跟 _RATE_LIMIT_PER_HOUR 一致，
    衝高流量測過會撞到 402 "Requests reach the upper limit"）。_RateLimiter
    是純客戶端計數，如果同一個 token 還有別的程序在用、或時間邊界剛好卡到，
    還是可能撞到伺服器端已經算過的額度，這裡加一層退避重試（最多3次），
    不要整批任務因為單一次撞到邊界就失敗。

    但 400（TokenIllegal）或訊息帶 "token level too low" 代表 token 本身無效
    或權限不足，跟流量無關，重試也不會成功——這種要立刻停止、不重試，直接
    拋例外請人去重新登入拿新 token，不要浪費重試次數在一個註定失敗的請求上。
    """
    if not _TOKEN:
        raise RuntimeError("缺少 FINMIND_TOKEN，請在 .env 設定")
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    params = {"dataset": _DATASET, "data_id": stock_id, "start_date": date_str}
    await _rate_limiter.acquire()
    # 2026-07-14：外層再包一次 asyncio.wait_for()，雙重保險——aiohttp自己的
    # ClientTimeout理論上該夠了，但實測遇過整支程式長時間沒有任何進度、
    # 又查不到CPU活動或開啟中的網路連線，不確定是不是某種邊界情況讓內建
    # timeout沒生效，這裡強制25秒後一定會拋TimeoutError，不會無限卡住。
    async def _do_request():
        async with session.get(
            _BASE_URL, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=20)
        ) as r:
            return r.status, await r.json()

    try:
        status, payload = await asyncio.wait_for(_do_request(), timeout=25)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        # 2026-07-26：斷線/DNS失敗/連線逾時，跟 FinMind 有沒有回應無關，重試
        # 單一請求沒有意義（下一筆一樣會失敗）——當成 FatalAPIError 讓整批
        # 任務停止，run_forever() 會定期呼叫 check_quota() 確認狀況，網路真的
        # 恢復後 check_quota() 自然會成功，接著自動繼續（見 NetworkError 說明）。
        raise NetworkError(f"{stock_id} {date_str} 網路錯誤（{e}），整批任務停止，等網路恢復後自動接著補") from e
    if payload.get("msg") != "success":
        msg = payload.get("msg") or ""
        if status == 403 or "ip banned" in msg.lower():
            raise IPBannedError(f"{stock_id} {date_str} IP被封鎖（{msg}），整批任務停止，不會自動重試")
        if status == 400 or "token level too low" in msg.lower() or "token is illegal" in msg.lower():
            raise TokenError(
                f"{stock_id} {date_str} token 無效或權限不足（{msg}），請重新登入 FinMind "
                "取得新 token 並更新 .env 的 FINMIND_TOKEN，不會自動重試"
            )
        if "upper limit" in msg.lower():
            if _retry < 3:
                await asyncio.sleep(60 * (_retry + 1))
                return await fetch_kbar_day(session, stock_id, date_str, _retry=_retry + 1)
            raise QuotaError(
                f"{stock_id} {date_str} 重試3次還是402（{msg}），可能是額度真的用完了"
                "（FinMind官方說法：「請升級會員或等待下個計費週期」，不一定是1小時內會恢復），"
                "整批任務停止，不會繼續重試"
            )
        if 400 <= status < 500:
            # 沒特別辨識出來的其他4xx，保守起見一樣當成整批任務停止（見
            # OtherFatalError 的說明），不要當成單筆失敗繼續跑下一筆。
            raise OtherFatalError(f"{stock_id} {date_str} 4xx錯誤（status={status}, msg={msg}），整批任務停止")
        raise RuntimeError(f"{stock_id} {date_str} 失敗: {msg}")
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = df["date"] + " " + df["minute"]
    return df.drop(columns=["minute"])


def _m1_file_path(year: int, month: int) -> Path:
    """跟 data/m1_data_loader.py::_m1_file_path() 同一套命名（月份補零），
    確保寫進同一批檔案、不會因為命名不一致在 HF Hub 觸發 schema 衝突。"""
    return _ROOT / f"db/m1/{year}_{month:02d}.parquet"


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀（跟
    data/m1_data_loader.py::_atomic_to_parquet() 同做法）。"""
    dir_path = file_path.parent
    os.makedirs(dir_path, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dir_path, suffix=".tmp", delete=False) as f:
        tmp_path = f.name
    try:
        df.to_parquet(tmp_path, **kwargs)
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _save_m1(new_df: pd.DataFrame, year: int, month: int):
    """合併進 db/m1/{year}_{month}.parquet，跟現有資料 dedupe（keep="last"，
    FinMind 優先覆蓋掉理論上不該存在的重複列）。"""
    new_df = new_df[["stock_id", "date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close"]:
        new_df[col] = new_df[col].astype("float32")
    new_df["volume"] = new_df["volume"].astype("int64")

    file_path = _m1_file_path(year, month)
    if file_path.exists():
        old_df = pd.read_parquet(file_path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    new_df.sort_values(["date", "stock_id"], inplace=True)
    _atomic_to_parquet(new_df, file_path, index=False, compression="zstd")


def _month_universe(
    year: int, month: int, top_n_by_volume: int | None = None
) -> tuple[list[str], list[str]]:
    """從 db/fugle_day/{year}_{month}.parquet 取得該月「實際交易日」跟「實際
    有交易的股票清單」——比用 fubon.subscribe_list.all_normal_stocks()（今天
    的候選股名單）準，那份名單是現在的，跟過去某個月實際有交易的股票對不上
    （例如當時還沒上市、或現在已下市的股票）。

    top_n_by_volume: 選填，只取當月「平均日成交量」最高的前N支股票，用來縮小
    範圍、減少請求數（例如先補流動性夠的股票就好）。跟不帶這個參數的呼叫
    混用完全沒問題——_existing_pairs() 是直接查 db/m1 實際內容，不是看這次
    呼叫的範圍設定，之後想把剩下的股票也補齊，直接呼叫不帶這個參數的版本，
    已經有的會自動跳過，不會重複下載或衝突。
    """
    path = _ROOT / f"db/fugle_day/{year}_{month:02d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，無法確定 {year}-{month:02d} 的交易日/股票清單")
    cols = ["stock_id", "date"] + (["volume"] if top_n_by_volume else [])
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    days = sorted(d.strftime("%Y-%m-%d") for d in df["date"].dt.date.unique())
    if top_n_by_volume:
        avg_vol = df.groupby("stock_id")["volume"].mean().sort_values(ascending=False)
        stocks = avg_vol.head(top_n_by_volume).index.tolist()
    else:
        stocks = sorted(df["stock_id"].unique().tolist())
    return days, stocks


def _existing_pairs(year: int, month: int) -> set[tuple[str, str]]:
    """db/m1 該月檔案裡已經有的 (stock_id, 日期) 組合，用來跳過已經補過的，
    節省 rate limit 額度；重跑這支腳本時天然可以從中斷處續傳。"""
    path = _m1_file_path(year, month)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["stock_id", "date"])
    days = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return set(zip(df["stock_id"], days))


async def backfill_month(
    year: int,
    month: int,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n_by_volume: int | None = None,
    flush_every: int = 100,
):
    """補齊 db/m1/{year}_{month}.parquet 缺的 (股票, 交易日) 組合。

    start_date/end_date: 選填，格式 YYYY-MM-DD，只補這個範圍內的交易日（一定要
    落在 year-month 這個月份內，跨月請分開呼叫）。留空就補整個月。
    top_n_by_volume: 選填，只補當月平均日成交量最高的前N支（見
    _month_universe() 的說明）。之後想補剩下的股票，直接不帶這個參數再呼叫
    一次即可，已經有的會自動跳過，不會跟這次的結果衝突。
    flush_every: 累積多少組結果就寫一次檔。2026-07-14 從2000調降到100
    ——先前2000太大，測試時程式被中途kill掉，累積不到2000組、一筆都沒
    存到db/m1，浪費掉的request額度完全打了水漂；小批次頻繁寫檔，就算隨時
    被中斷，最多只損失接近100組的額度，不會整批不見。

    2026-07-14 也修掉一個安全漏洞：撞到 FatalAPIError（400/402/403/其他4xx）
    時，用 stop_event 立刻擋掉「還沒送出去」的請求，不會像以前那樣讓整批
    （最多flush_every組）全部發完才停下來——FinMind官方文件說明，短時間內
    累積大量4xx錯誤本身就是觸發IP封鎖的原因，「先讓batch跑完才檢查」等於
    自己主動製造這個模式，要在第一個fatal error出現的當下就盡快停止，
    受影響的頂多是當下已經在飛行中的 _CONCURRENCY(5) 筆請求。
    """
    days, stocks = _month_universe(year, month, top_n_by_volume=top_n_by_volume)
    if start_date:
        days = [d for d in days if d >= start_date]
    if end_date:
        days = [d for d in days if d <= end_date]
    existing = _existing_pairs(year, month)
    pairs = [(sid, d) for sid in stocks for d in days if (sid, d) not in existing]
    print(
        f"{year}-{month:02d}：目標 {len(stocks)} 支 x {len(days)} 天 = "
        f"{len(stocks) * len(days)} 組，已有 {len(existing)} 組，還要補 {len(pairs)} 組"
    )
    if not pairs:
        print("已經補齊，不用下載")
        return

    est_hours = len(pairs) / _RATE_LIMIT_PER_HOUR
    print(f"預估耗時：約 {est_hours:.1f} 小時（rate limit {_RATE_LIMIT_PER_HOUR}/小時）")

    sem = asyncio.Semaphore(_CONCURRENCY)
    done = 0
    failed: list[tuple[str, str, str]] = []
    buffer: list[pd.DataFrame] = []
    stop_event = asyncio.Event()
    fatal_error_holder: list[FatalAPIError] = []

    async def _one(session: aiohttp.ClientSession, sid: str, day: str):
        nonlocal done
        if stop_event.is_set():
            return  # 已經有 fatal error，不要再送新請求
        async with sem:
            if stop_event.is_set():
                return  # 排隊等併發額度時，可能中途已經被別的任務觸發停止
            try:
                df = await fetch_kbar_day(session, sid, day)
            except FatalAPIError as e:
                stop_event.set()
                fatal_error_holder.append(e)
                return
            except Exception as e:
                failed.append((sid, day, str(e)))
                return
            done += 1
            if done % 200 == 0 or done == len(pairs):
                print(f"  [{done}/{len(pairs)}] 進度...")
            if not df.empty:
                buffer.append(df)

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(pairs), flush_every):
            if stop_event.is_set():
                break
            batch = pairs[i : i + flush_every]
            await asyncio.gather(*(_one(session, sid, day) for sid, day in batch))
            if buffer:
                _save_m1(pd.concat(buffer, ignore_index=True), year, month)
                print(f"  已寫入 {len(buffer)} 組結果到 {_m1_file_path(year, month).name}")
                buffer.clear()
            if stop_event.is_set():
                break

    if fatal_error_holder:
        e = fatal_error_holder[0]
        print(f"  ⚠️ {e}")
        print("  整批任務停止，處理好問題後重跑這支腳本會自動從中斷處繼續")
        raise e

    print(f"{year}-{month:02d} 補齊完成，成功 {done} 組，失敗 {len(failed)} 組")
    if failed:
        print("失敗清單（沒有寫入db/m1，重跑這支腳本會自動重試，因為沒被記錄成『已有』）：")
        for sid, day, err in failed[:20]:
            print(f"  {sid} {day}: {err}")
        if len(failed) > 20:
            print(f"  ...還有 {len(failed) - 20} 筆")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        _year, _month = int(sys.argv[1]), int(sys.argv[2])
    else:
        _year, _month = 2026, 5
    asyncio.run(backfill_month(_year, _month))
