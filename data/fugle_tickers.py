"""
當沖標的清單管理（Fugle /intraday/tickers）

功能：
    從 Fugle API 取得每日可當沖股票清單（過濾條件 ②③），
    存到 db/fugle_tickers/tickers.parquet 供其他模組使用。

過濾條件（Fugle API 端）：
    ② isDayTrading=true  可當沖
    ③ isNormal=true      正常交易（排除全額交割等）
    → 約 2,787 支（TWSE + TPEx）

注意：不含 ① 20日均量過濾（那是 live_trader.py 的 _volume_filter 做的）

主要函式：
    update_tickers()  每日開盤前呼叫一次，更新並回傳清單
    fugle_stocks()    回傳 stock_id 列表（無清單時自動呼叫 update_tickers）
"""
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))
token = os.environ.get("FUGLE", "")
_BASE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"
_TICKERS_PATH = _ROOT / "db/fugle_tickers/tickers.parquet"


def update_tickers() -> pd.DataFrame:
    """
    從 Fugle /intraday/tickers 取得當日可當沖股票清單（TWSE + TPEx），
    存到 db/fugle_tickers/tickers.parquet。
    建議每日開盤前呼叫一次。
    """
    if not token:
        raise RuntimeError("缺少 FUGLE API Key，請在 .env 設定 FUGLE")

    date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    rows = []
    for exchange in ("TWSE", "TPEx"):
        r = requests.get(
            f"{_BASE_URL}/intraday/tickers",
            params={"type": "EQUITY", "exchange": exchange, "isDayTrading": "true", "isNormal": "true"},
            headers={"X-API-KEY": token},
            timeout=10,
        )
        if r.status_code == 429:
            print(f"  Fugle tickers API 429 限速，使用舊快取")
            return load_tickers() if os.path.exists(_TICKERS_PATH) else pd.DataFrame()
        r.raise_for_status()
        for item in r.json().get("data", []):
            rows.append({
                "stock_id": item["symbol"],
                "exchange": exchange,
                "name": item.get("name", ""),
                "industry": item.get("industry", ""),
                "date": date_str,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["stock_id"])
    if df.empty:
        print(f"update_tickers: API 回傳空資料（非盤中？），保留舊檔案")
        return load_tickers() if os.path.exists(_TICKERS_PATH) else df
    os.makedirs(os.path.dirname(_TICKERS_PATH), exist_ok=True)
    df.to_parquet(_TICKERS_PATH, index=False)
    print(f"儲存完成：{len(df)} 支股票（{date_str}）→ {_TICKERS_PATH}")
    return df


def load_tickers() -> pd.DataFrame:
    """讀取已存的股票清單，欄位：stock_id, exchange, name, industry, date"""
    if not os.path.exists(_TICKERS_PATH):
        raise FileNotFoundError(f"找不到 {_TICKERS_PATH}，請先執行 update_tickers()")
    return pd.read_parquet(_TICKERS_PATH)


def fugle_stocks() -> list[str]:
    """
    回傳已存清單的 stock_id 列表。
    若清單不存在，自動呼叫 update_tickers() 建立。
    """
    if not os.path.exists(_TICKERS_PATH):
        update_tickers()
    return load_tickers()["stock_id"].tolist()


if __name__ == "__main__":
    df = update_tickers()
    print(df.head(10).to_string(index=False))
    print(f"\nTWSE: {(df['exchange']=='TWSE').sum()} 支，TPEx: {(df['exchange']=='TPEx').sum()} 支")
