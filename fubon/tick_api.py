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

用法：
    python -m fubon.tick_api   # 更新今天固定清單（tick_universe，400檔+0050）的tick到db/tick
"""

from datetime import datetime, timedelta, timezone
import time

import numpy as np
import pandas as pd

from finmind.tick_api import save_tick
from finmind.tick_universe import load_tick_universe

_TW = timezone(timedelta(hours=8))
_REQUEST_INTERVAL = 0.25  # 300次/分鐘上限，留緩衝（同 fubon/subscribe_list.py:158）
_PAGE_LIMIT = 500  # 富邦 intraday/trades 單次 limit 上限，實測帶更大的值(5000)還是只回500
_FLUSH_EVERY = 50  # 每50檔寫一次檔，避免單一次性存全部400檔中途被中斷就整批遺失


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


def fetch_trades_today(sdk, symbol: str) -> pd.DataFrame:
    """分頁呼叫 reststock.intraday.trades(symbol=symbol, limit=500, offset=N)
    直到回傳空清單，組成當天全部tick。

    回傳欄位比照 db/tick schema：stock_id, date("YYYY-MM-DD
    HH:MM:SS.ffffff")、deal_price、volume、tick_type。當天沒有資料（停牌/
    尚未開盤/查詢的股票代號當天沒有成交）回傳空 DataFrame。
    """
    reststock = sdk.marketdata.rest_client.stock
    rows = []
    offset = 0
    while True:
        r = reststock.intraday.trades(symbol=symbol, limit=_PAGE_LIMIT, offset=offset)
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


def update_tick_today(stocks: list[str] | None = None):
    """登入一次，逐檔抓今天的tick，累積緩衝、每 _FLUSH_EVERY 檔寫一次
    db/tick/{year}_{month}.parquet（呼叫 finmind.tick_api.save_tick()，
    同一個merge+dedupe+atomic write邏輯）。stocks=None 時讀
    finmind.tick_universe.load_tick_universe() 的固定清單（沿用同一份
    名單，不另外維護）。"""
    from fubon import fubon_api as trade_api

    if stocks is None:
        stocks = load_tick_universe()
    today = datetime.now(_TW)
    year, month = today.year, today.month

    sdk, _ = trade_api.login()
    try:
        trade_api.init_market_data(sdk)
        got = 0
        empty = 0
        failed: list[tuple[str, str]] = []
        buffer: list[pd.DataFrame] = []

        def _flush():
            if buffer:
                save_tick(pd.concat(buffer, ignore_index=True), year, month)
                buffer.clear()

        for i, sid in enumerate(stocks, 1):
            try:
                df = fetch_trades_today(sdk, sid)
            except Exception as e:
                df = None
                failed.append((sid, str(e)))
            if df is not None:
                if df.empty:
                    empty += 1
                else:
                    got += 1
                    buffer.append(df)
            # flush檢查一定要在迴圈本體最後執行，不能被上面的例外處理跳過
            # ——2026-07-30 實測踩過的bug：原本失敗時用 continue 提早跳過這裡，
            # 如果失敗的剛好是清單最後一支，緩衝區裡前面已經抓到的資料就會
            # 直接遺失、從來沒被寫進檔案。
            if i % _FLUSH_EVERY == 0 or i == len(stocks):
                _flush()
                print(f"  [{i}/{len(stocks)}] 進度... 有資料{got}/無資料{empty}/失敗{len(failed)}")
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
