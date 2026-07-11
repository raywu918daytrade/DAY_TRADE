"""
富邦 fubon_neo SDK 統一包裝層。

職責：所有直接呼叫 fubon_neo SDK 的地方集中在這支檔案，subscribe_list.py /
marketdata_ws.py 等上層程式只呼叫這裡的函式，不直接 import fubon_neo。
好處：SDK 版本升級或呼叫方式變動時，只要改這一支檔案。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from fubon_neo.sdk import FubonSDK, Mode, build_websocket_client

load_dotenv(Path(__file__).parents[1] / ".env")

_ROOT = Path(__file__).parents[1]


def login() -> tuple[FubonSDK, list]:
    """第一次連線測試須用身分證字號＋登入密碼＋憑證（不可用 API Key）。"""
    sdk = FubonSDK()
    cert_path = _ROOT / os.environ["FUBON_CERT_PATH"]
    result = sdk.login(
        os.environ["FUBON_ID"],
        os.environ["FUBON_PASSWORD"],
        str(cert_path),
        os.environ.get("FUBON_CERT_PASS") or None,
    )
    if not result.is_success:
        raise RuntimeError(f"富邦登入失敗: {result.message}")
    return sdk, result.data


def logout(sdk: FubonSDK):
    try:
        sdk.logout()
    except Exception:
        pass


def init_market_data(sdk: FubonSDK, mode: Mode = Mode.Normal):
    """行情初始化，REST／WebSocket 都要先呼叫這個。candles channel 只支援 Normal mode，
    所以預設用 Normal（REST 查詢不受 mode 影響，用同一個預設值即可）。"""
    sdk.init_realtime(mode)


def intraday_tickers(sdk: FubonSDK, exchange: str, type_: str = "EQUITY", is_normal: bool = True) -> list[dict]:
    """REST 行情 API：取得指定交易所的股票清單。呼叫前須先 init_market_data()。"""
    reststock = sdk.marketdata.rest_client.stock
    r = reststock.intraday.tickers(type=type_, exchange=exchange, isNormal=is_normal)
    return r.get("data", [])


def realtime_token(sdk: FubonSDK) -> str:
    return sdk.exchange_realtime_token()


def open_candles_connection(token: str, mode: Mode = Mode.Normal):
    """開一條 WebSocket 連線（stock client），尚未 connect()／subscribe()。
    每呼叫一次會建立一條獨立連線（富邦上限 5 條，見 fubon/config.py）。"""
    return build_websocket_client(mode, token).stock


def subscribe_candles(stock_client, symbol: str):
    stock_client.subscribe({"channel": "candles", "symbol": symbol})


if __name__ == "__main__":
    sdk, accounts = login()
    print("登入成功，帳戶：")
    for acc in accounts:
        print(f"  {acc.name}  {acc.branch_no}-{acc.account}  ({acc.account_type})")
