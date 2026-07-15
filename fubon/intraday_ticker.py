"""
診斷用：呼叫 fubon/fubon_api.py::intraday_ticker()（單支股票明細，
intraday/ticker/{symbol}），檢查 canDayTrade/canBuyDayTrade 這兩個欄位，
確認能不能拿來取代 fubon/intraday_tickers.py 目前用的 isNormal 篩選。

跟 intraday_tickers.py（清單端點，一次拿全部股票但只有
symbol/industry/name）不同，這是單支股票查詢，欄位更完整，但要每支個股各
呼叫一次（Rate limit 300次/分鐘），拿全市場 ~2700 支會花不少時間，先用少量
股票測試。

使用方式：
    python -m fubon.intraday_ticker 2330 2317 00878
"""
import sys

import pandas as pd


def fetch_one(sdk, symbol: str) -> dict:
    from fubon import fubon_api as trade_api

    return trade_api.intraday_ticker(sdk, symbol)


if __name__ == "__main__":
    from fubon import fubon_api as trade_api

    symbols = sys.argv[1:] or ["2330"]
    sdk, _ = trade_api.login()
    try:
        trade_api.init_market_data(sdk)
        rows = [fetch_one(sdk, sid) for sid in symbols]
    finally:
        trade_api.logout(sdk)

    df = pd.DataFrame(rows)
    cols = ["symbol", "name", "canDayTrade", "canBuyDayTrade", "isAttention",
            "isDisposition", "securityStatus"]
    print(df[cols].to_string(index=False))
