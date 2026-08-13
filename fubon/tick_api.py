"""富邦 intraday/trades/{symbol} 逐筆成交 — 每日「今天」tick更新。

背景動機：tick 資料原本靠 FinMind TaiwanStockPriceTick 回補（見
finmind/tick_api.py、finmind/backfill_tick_history.py），但FinMind的存取權限
只到2026-08-18，之後能不能繼續用還在評估，這支是找的備案。也順便解決
FinMind回補「今天」資料要佔用額度限流(6000/小時)的缺點——富邦是自己帳號的
即時行情REST API，不佔那個額度。

跟 finmind/ 那套抓取邏輯刻意分開成獨立檔案（tick跟分K/不同資料源的抓取邏輯
要分開，同一個原則），但**存檔直接重用 finmind.tick_api.save_tick()**——寫
進的是同一份 db/tick schema，不要為了「獨立」複製一份合併/去重/atomic write
邏輯，那樣兩邊之後容易資料品質不一致。

限制：intraday/trades/{symbol} 只能拿「今天」的資料，帶 date 參數會被忽略，
不支援指定過去日期，不能拿來回補歷史——那部分還是要靠FinMind（或找其他歷史
資料源）。

已驗證（2026-07-30，用2330實測）：
    - 分頁 offset 沒有深度上限，單日11,030筆全部連續拿到，涵蓋開盤09:00到
      收盤定盤14:30，沒有缺口。
    - tick_type 用 price 相對 bid/ask 的關係推斷（跟FinMind官方TickType不是
      同一套算法），跟FinMind官方判定比對：時間+價格+成交量都吻合的10,992筆
      裡一致率99.98%（10,990/10,992），可以放心採用。

Rate limit：intraday 家族端點官方文件是300次/分鐘，沿用
fubon/subscribe_list.py:158 同樣的節流方式（0.25秒/次，留緩衝抓~240次/分鐘）。

2026-08-13改成併發下載：`update_tick_today()` 原本是單執行緒逐檔序列處理，
每支股票內部還要分頁（`fetch_trades_today()`），熱門股單支就要好幾秒到
十幾秒（實測2317單日21,128筆tick、16.7秒），1878支序列跑可能要數小時。
改成 `ThreadPoolExecutor(_FETCH_WORKERS)` 併發抓不同股票（每支股票自己
內部分頁仍然序列，不同股票之間併發），共用一個 `rate_lock` 把「整體」
分頁請求間隔控制在 `_REQUEST_INTERVAL`（不是每條執行緒各自 sleep——理由
跟 data/m1_data_loader.py::_update_m1_fubon_fast() 同一輪的修改一樣）。

用法：
    python -m fubon.tick_api   # 更新今天固定清單（stock_universe_2000+0050，1878檔）的tick到db/tick
"""

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# finmind.tick_api 匯入時會連帶 import finmind.m1_api，那支檔案本身已經
# monkey-patch 過全域 print（加時間戳記+強制flush，見該檔說明），這裡不用
# 重複做一次——重複patch會把時間戳記包兩層，變成 "[HH:MM:SS] [HH:MM:SS] ..."。
from finmind.tick_api import save_tick
from finmind.stock_universe_2000 import load_stock_universe_2000_with_0050
from finmind.m1_api import _ROOT, _atomic_to_parquet

_TW = timezone(timedelta(hours=8))
_REQUEST_INTERVAL = 0.25  # 300次/分鐘上限，留緩衝（同 fubon/subscribe_list.py:158）
_PAGE_LIMIT = 500  # 富邦 intraday/trades 單次 limit 上限，實測帶更大的值(5000)還是只回500
_TICK_FLAG_PATH = _ROOT / "db/tick_flags/tick_flag.parquet"
_FLUSH_EVERY = 50  # 每50檔寫一次檔，避免單一次性存全部400檔中途被中斷就整批遺失


def _get_done_stocks(date_str: str) -> set:
    """2026-08-13加：比照 data/m1_data_loader.py／data/day_data_loader.py
    的flag機制——原本 update_tick_today() 沒有這個，每次執行都重新下載
    全部1878支，重跑（或中途中斷後再跑）完全沒辦法只補還沒抓過的部分，
    使用者實測發現「跑兩次都要重新下載1878支」才補上這個缺口。"""
    if not _TICK_FLAG_PATH.exists():
        return set()
    df = pd.read_parquet(_TICK_FLAG_PATH)
    return set(df[df["date"] == date_str]["stock_id"].tolist())


def _update_flags_batch(stock_ids: list, date_str: str):
    """一次幫多支股票標記 flag（一次讀寫），配合 update_tick_today() 的
    批次flush頻率呼叫，不是逐支呼叫——理由跟 data/m1_data_loader.py::
    _update_flags_batch() 同一輪的教訓一樣，避免逐支讀寫拖慢速度。

    這裡「有資料」跟「確認今天沒有成交（df為空）」都算完成、會被標記——
    跟 m1/d1 的 _has_today_data() 邏輯不同：那邊的「空」代表「今天資料
    還沒公佈，之後會有」，這裡的 intraday/trades 是即時端點，
    update_daily.py 一定是收盤後才跑，這時候查到空結果代表這支股票今天
    確實沒有成交（停牌/無人交易），是一個確定的最終狀態，不是還沒公佈，
    重跑也不會變成有資料，標記完成避免每次重跑都白白查詢。只有真正的
    請求失敗（例外）才不標記，讓下次重跑自動重試。"""
    if not stock_ids:
        return
    new_rows = pd.DataFrame({"stock_id": stock_ids, "date": date_str})
    if _TICK_FLAG_PATH.exists():
        df = pd.read_parquet(_TICK_FLAG_PATH)
        df = pd.concat([df, new_rows], ignore_index=True)
        df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    else:
        df = new_rows
    _TICK_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_to_parquet(df, _TICK_FLAG_PATH, index=False)


def _infer_tick_type(df: pd.DataFrame) -> np.ndarray:
    """用 price 相對 bid/ask 的關係推斷買賣方主動性，跟FinMind TickType語意
    對齊（0=無法判定、1=買方主動、2=賣方主動），但是推斷值不是官方判定——
    見檔頭說明的99.98%一致率驗證結果。開盤參考價/收盤定盤價這種沒有bid/ask
    的記錄（例如富邦serial=99999999那筆），bid/ask是NaN，比較會自然為False，
    落到0（無法判定），不用特別判斷。"""
    for col in ("bid", "ask"):
        if col not in df.columns:
            df[col] = float("nan")
    return np.where(df["price"] >= df["ask"], 1, np.where(df["price"] <= df["bid"], 2, 0))


def fetch_trades_today(sdk, symbol: str, rate_lock=None, last_req: list = None) -> pd.DataFrame:
    """分頁呼叫 reststock.intraday.trades(symbol=symbol, limit=500, offset=N)
    直到回傳空清單，組成當天全部tick。

    回傳欄位比照 db/tick schema：stock_id, date("YYYY-MM-DD
    HH:MM:SS.ffffff")、deal_price、volume、tick_type。當天沒有資料（停牌/
    尚未開盤/查詢的股票代號當天沒有成交）回傳空 DataFrame。

    rate_lock/last_req：選填（2026-08-13加，配合 update_tick_today() 併發
    下載）。多支股票的分頁請求要共用「同一個」節流器，把整體請求間隔控制
    在 _REQUEST_INTERVAL——不能每個執行緒各自 sleep(_REQUEST_INTERVAL)，
    那樣多條執行緒合計會超過300次/分鐘限制（跟 data/m1_data_loader.py::
    _update_m1_fubon_fast() 同一種節流方式）。不傳（單獨測試這支函式時）
    就退回原本「每次呼叫都自己 sleep」的簡單模式，行為不變。"""
    reststock = sdk.marketdata.rest_client.stock
    rows = []
    offset = 0
    while True:
        if rate_lock is not None:
            with rate_lock:
                wait = _REQUEST_INTERVAL - (time.time() - last_req[0])
                if wait > 0:
                    time.sleep(wait)
                last_req[0] = time.time()
        r = reststock.intraday.trades(symbol=symbol, limit=_PAGE_LIMIT, offset=offset)
        if rate_lock is None:
            time.sleep(_REQUEST_INTERVAL)
        data = r.get("data", [])
        if not data:
            break
        rows.extend(data)
        offset += _PAGE_LIMIT
        if len(data) < _PAGE_LIMIT:
            break
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["stock_id"] = symbol
    # time 欄位是微秒(us)精度的unix epoch，用 unit="us" 直接轉換，避免除以1e6
    # 的浮點數精度損失。
    dt = pd.to_datetime(df["time"], unit="us", utc=True).dt.tz_convert(_TW).dt.tz_localize(None)
    df["date"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    df["deal_price"] = df["price"].astype("float32")
    df["volume"] = df["size"].astype("int64")
    df["tick_type"] = _infer_tick_type(df).astype("int8")
    return df[["stock_id", "date", "deal_price", "volume", "tick_type"]]


_FETCH_WORKERS = 10  # 併發抓不同股票的數量，比照 data/m1_data_loader.py::_INTRADAY_FAST_WORKERS


def update_tick_today(stocks: list[str] | None = None):
    """登入一次，`ThreadPoolExecutor(_FETCH_WORKERS)` 併發抓不同股票的今天
    tick（每支股票自己內部的分頁請求仍然序列，但不同股票之間可以同時進行），
    累積緩衝、每 _FLUSH_EVERY 檔寫一次 db/tick/{year}_{month}.parquet（呼叫
    finmind.tick_api.save_tick()，同一個merge+dedupe+atomic write邏輯），
    同批次批次標記flag（`db/tick_flags/`）。stocks=None 時讀
    finmind.stock_universe_2000.load_stock_universe_2000_with_0050() 的固定
    清單（2026-08-08改，1877支全市場一般個股+強制併入0050，共1878支，跟
    data/day_data_loader.py／data/m1_data_loader.py 的股票母體統一），會先
    排除今天已經標記完成的股票。

    2026-08-13改成併發（原本是單執行緒序列，逐檔處理）：實測熱門股（例如
    2317單日21,128筆tick）光是分頁就要16秒以上，1878支若都序列跑，粗估
    可能要數小時。改成多支股票併發處理，共用一個 `rate_lock` 把「整體」
    分頁請求間隔控制在 `_REQUEST_INTERVAL=0.25秒`（不是每條執行緒各自
    sleep——那樣併發後會超過300次/分鐘限制，見 fetch_trades_today() 的
    rate_lock 說明），讓不同股票的網路等待時間互相重疊，逼近300次/分鐘的
    理論吞吐量，而不是被單一股票（尤其是分頁很多的熱門股）的序列延遲卡住。
    實測驗證過真的有效：5支股票序列29.71秒，併發只要5.79秒。

    2026-08-13再加：原本完全沒有flag機制，重跑（或中途中斷後再跑）永遠
    是完整1878支重新下載一次，使用者實測發現這個問題。現在比照 m1/d1 加
    `_get_done_stocks()`/`_update_flags_batch()`，「有資料」跟「確認今天
    沒有成交」都算完成（intraday/trades 是即時端點，收盤後執行查到空
    代表這支真的今天沒成交，不是資料還沒公佈，見 _update_flags_batch()
    說明），只有真正請求失敗的才不標記、留給下次重跑重試。
    """
    from concurrent.futures import ThreadPoolExecutor
    import queue as _queue
    import threading

    from fubon import fubon_api as trade_api

    if stocks is None:
        stocks = load_stock_universe_2000_with_0050()
    today = datetime.now(_TW)
    year, month = today.year, today.month
    date_str = today.strftime("%Y-%m-%d")

    done_before = _get_done_stocks(date_str)
    wait_stocks = [s for s in stocks if s not in done_before]
    print(f"還有 {len(wait_stocks)} 個股票未更新（已排除今日已完成 flag，候選共 {len(stocks)} 支）")
    if not wait_stocks:
        print("全部已完成，不用下載")
        return

    sdk, _ = trade_api.login()
    try:
        trade_api.init_market_data(sdk)
        got = 0
        empty = 0
        failed: list[tuple[str, str]] = []
        buffer: list[pd.DataFrame] = []
        done_ids: list[str] = []
        done = 0
        state_lock = threading.Lock()
        rate_lock = threading.Lock()
        last_req = [0.0]

        def _flush_locked():
            # 呼叫端已經持有 state_lock。
            if buffer:
                save_tick(pd.concat(buffer, ignore_index=True), year, month)
                buffer.clear()
            if done_ids:
                _update_flags_batch(done_ids, date_str)
                done_ids.clear()

        q = _queue.Queue()
        for sid in wait_stocks:
            q.put(sid)

        def _worker():
            nonlocal got, empty, done
            while True:
                try:
                    sid = q.get_nowait()
                except _queue.Empty:
                    return
                try:
                    df = fetch_trades_today(sdk, sid, rate_lock=rate_lock, last_req=last_req)
                except Exception as e:
                    df = None
                    with state_lock:
                        failed.append((sid, str(e)))
                with state_lock:
                    if df is not None:
                        if df.empty:
                            empty += 1
                        else:
                            got += 1
                            buffer.append(df)
                        done_ids.append(sid)  # 有結果（含確認無資料）才標記，失敗的留給下次重試
                    # flush檢查一定要在這裡（每支股票處理完就檢查），不能被
                    # 上面的例外處理跳過——2026-07-30 實測踩過的bug：原本
                    # 失敗時提早跳過檢查，如果失敗的剛好是最後一支，緩衝區
                    # 裡前面已經抓到的資料就會直接遺失，從來沒被寫進檔案。
                    done += 1
                    d = done
                    if d % _FLUSH_EVERY == 0 or d == len(wait_stocks):
                        _flush_locked()
                        print(f"  [{d}/{len(wait_stocks)}] 進度... 有資料{got}/無資料{empty}/失敗{len(failed)}")

        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
            futures = [pool.submit(_worker) for _ in range(_FETCH_WORKERS)]
            for f in futures:
                f.result()
        with state_lock:
            _flush_locked()  # 保底：理論上 done==len(wait_stocks) 那次已經flush過，這裡確保萬無一失
    finally:
        trade_api.logout(sdk)

    print(f"完成：{got} 檔有資料、{empty} 檔無資料、{len(failed)} 檔失敗")
    if failed:
        print("失敗清單：")
        for sid, err in failed[:20]:
            print(f"  {sid}: {err}")
        if len(failed) > 20:
            print(f"  ...還有 {len(failed) - 20} 檔")


if __name__ == "__main__":
    update_tick_today()
