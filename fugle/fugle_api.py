"""
Fugle marketdata REST API 統一包裝層。

職責：所有直接呼叫 Fugle REST API 的地方集中在這支檔案，data/day_data_loader.py、
data/m1_data_loader.py 等上層程式只呼叫這裡的函式，不直接組 URL/header。
好處：base URL、429 重試邏輯只要改一個地方。
"""
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env", override=True)

TOKEN = os.environ.get("FUGLE", "")
_BASE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"


def historical_candles(symbol: str, **params) -> requests.Response:
    """REST 行情 API：取得歷史K線（historical/candles/{symbol}），429 自動重試
    一次（依 Retry-After 等待秒數）。

    回傳原始 requests.Response，不在這裡處理狀態碼——404 要當「這段沒資料」
    還是例外，各呼叫端行為不一樣（day K 一次抓一年、404常見，股票在那年
    可能還沒上市；分K 抓近30日、404少見，當例外處理），交給呼叫端自己判斷，
    不在這支共用函式裡假設。

    params 直接透傳給 Fugle API：timeframe/from/to/fields/sort/adjusted 等
    （語意對齊 fubon/fubon_api.py::historical_candles()，2026-07-13 實測過
    from/to 抓日K兩邊行為一致，同一套底層 fugle_marketdata 元件）。
    """
    headers = {"X-API-KEY": TOKEN}
    r = requests.get(f"{_BASE_URL}/historical/candles/{symbol}", params=params, headers=headers, timeout=10)
    if r.status_code == 429:
        time.sleep(float(r.headers.get("Retry-After", 60)) + 1)
        r = requests.get(f"{_BASE_URL}/historical/candles/{symbol}", params=params, headers=headers, timeout=10)
    return r
