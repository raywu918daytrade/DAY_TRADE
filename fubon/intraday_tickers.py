"""
當沖標的清單管理（富邦 intraday.tickers()）

取代原本的 data/fugle_tickers.py（同一份用途，改用富邦的 API 當資料源）。

功能：
    從富邦 REST 行情 API 取得可交易股票清單（TWSE+TPEx，type=EQUITY），
    過濾條件：
        - industry 欄位非數字：富邦這支 API 混了一批非個股代號（例如 A00104、
          A01102），industry 是 A1/A2 這種產業分類代碼，不是真正的股票/ETF，
          name 欄位也直接等於代號本身（沒解析出公司名稱）。2026-07-14 實測：
          這批代號在日K historical/candles 一律 404（Fugle/富邦共用同一套
          底層資料源），濾掉可避免 update_day()/update_m1() 每天重複打
          注定失敗的請求。
        - 債券型ETF／固定收益基金（見 _is_bond()）：只留股票和一般權益類ETF。
        - 現金增資股權批次（見 _is_tranche()）：名稱結尾中文數字一~十、代號
          5碼以上，這批 historical/candles 一律404。
    存到 db/tickers/tickers.parquet 供其他模組使用。

這裡是唯一的股票清單過濾源頭：
    - fubon/subscribe_list.py 在這份清單上再做均量排序/WebSocket分組
    - data/day_data_loader.py、data/m1_data_loader.py 的訓練資料母體
      （經 fubon/subscribe_list.py::all_normal_stocks()）
    - scripts/push_day_to_hf.py、push_m1_to_hf.py

主要函式：
    update_tickers()  每日開盤前呼叫一次，更新並回傳清單
    load_tickers()     讀取已存的清單（不重算）
"""
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import re
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))
_TICKERS_PATH = _ROOT / "db/tickers/tickers.parquet"

# 台股 ETF 代號後綴慣例：00XXXB＝債券型ETF，00XXXD＝主動式債券/固定收益基金
# （實測這批代號的名稱都帶「非投」「債」「入息」）。只要股票跟一般/槓桿/反向/
# 主動股票型 ETF（無後綴、L、R、A），不要固定收益類。名稱含「債」字再補一層，
# 避免名稱被 API 截斷看不出後面有「債」的漏網之魚。
_BOND_ETF_PATTERN = re.compile(r"^00\d{3}[BD]$")

# 現金增資股權批次代號（例如「中砂一」15601、「和大四」15364、KY股的
# 「鮮活果汁一KY」12561、「廣華二KY」13382、永續特別股批次的「台泥一永」
# 11011、「遠東新E1永」140202），名稱結尾是中文數字一~十、中文數字接在
# "KY"前面、或結尾是"永"，代號都是5碼以上（2026-07-14 實測：
# securityType="02"/"03"，historical/candles 一律404，跟正常股票的
# securityType="01"/特別股"04"/ETF"24"不同）。只限代號5碼以上，避免誤殺
# 「統一」(1216)這種名稱剛好結尾是中文數字、但本身是正常4碼股票的公司。
_TRANCHE_SUFFIX = set("一二三四五六七八九十") | {"永"}
_TRANCHE_KY_PATTERN = re.compile(r"[一二三四五六七八九十]KY$")


def _is_bond(stock_id: str, name: str) -> bool:
    return bool(_BOND_ETF_PATTERN.match(stock_id)) or "債" in name


def _is_tranche(stock_id: str, name: str) -> bool:
    if len(stock_id) < 5 or not name:
        return False
    return name[-1] in _TRANCHE_SUFFIX or bool(_TRANCHE_KY_PATTERN.search(name))


def update_tickers() -> pd.DataFrame:
    """
    從富邦 intraday.tickers() 取得當日可交易股票清單（TWSE + TPEx），濾掉
    industry 非數字的垃圾代碼和債券型ETF，存到 db/tickers/tickers.parquet。
    建議每日開盤前呼叫一次。
    """
    from fubon import fubon_api as trade_api

    date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    sdk, _ = trade_api.login()
    try:
        trade_api.init_market_data(sdk)
        rows = []
        for exchange in ("TWSE", "TPEx"):
            for item in trade_api.intraday_tickers(sdk, exchange, type_="EQUITY"):
                sid = item["symbol"]
                name = item.get("name", "")
                industry = item.get("industry", "")
                if not str(industry).isdigit():
                    continue
                if _is_bond(sid, name):
                    continue
                if _is_tranche(sid, name):
                    continue
                rows.append({
                    "stock_id": sid,
                    "exchange": exchange,
                    "name": name,
                    "industry": industry,
                    "date": date_str,
                })
    finally:
        trade_api.logout(sdk)

    df = pd.DataFrame(rows).drop_duplicates(subset=["stock_id"])
    if df.empty:
        print("update_tickers: API 回傳空資料（非盤中？），保留舊檔案")
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


if __name__ == "__main__":
    df = update_tickers()
    print(df.head(10).to_string(index=False))
    print(f"\nTWSE: {(df['exchange']=='TWSE').sum()} 支，TPEx: {(df['exchange']=='TPEx').sum()} 支")
