"""
富邦 fubon_neo SDK 統一包裝層。

職責：所有直接呼叫 fubon_neo SDK 的地方集中在這支檔案，subscribe_list.py /
marketdata_ws.py 等上層程式只呼叫這裡的函式，不直接 import fubon_neo。
好處：SDK 版本升級或呼叫方式變動時，只要改這一支檔案。
"""
import base64
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from fubon_neo.sdk import FubonSDK, Mode, build_websocket_client

load_dotenv(Path(__file__).parents[1] / ".env", override=True)


def _resolve_cert_path() -> str:
    """憑證路徑：.p12 檔案不進版控，一律用 FUBON_CERT_B64（憑證 base64 編碼）
    在執行期寫一個暫存檔，本機、雲端（Render）都走同一條路，比照
    sinopac/sinopac_api.py 對永豐 .pfx 的處理方式。
    產生 FUBON_CERT_B64：base64 -i 憑證.p12 | tr -d '\\n'
    """
    cert_data = base64.b64decode(os.environ["FUBON_CERT_B64"])
    with tempfile.NamedTemporaryFile(suffix=".p12", delete=False) as f:
        f.write(cert_data)
        return f.name


def login(retry_delays: tuple[float, ...] = (60, 300, 600)) -> tuple[FubonSDK, list]:
    """身分證字號＋API Key（.env 的 FUBON_API_KEY）＋憑證登入。第一次連線測試
    （身分證字號＋密碼＋憑證）已經在 2026-07 完成、帳號權限開通，之後一律用這個。

    2026-08-28加重試：2026-08-27 GHA排程整個失敗，原因是 FubonSDK() 建構
    子連線時噴 ValueError（Unable to connect to wss://neoapi.fbs.com.tw/...），
    這個例外沒接住，直接把整支 update_daily.py 炸掉（後面 m1/d1/
    adjustment_day/tick_universe全部沒機會跑，見對話紀錄的診斷）。查過去
    21次排程只有這一次失敗，屬於偶發的暫時性連線問題（不是帳密/憑證這種
    重試也沒用的錯誤），所以在這裡包一層重試，換掉單點失敗就讓一整天全部
    作廢的行為。FubonSDK() 建構子跟 apikey_login() 都可能是連線失敗的
    來源，一起包進 try。retry_delays 用遞增間隔（預設1/5/10分鐘）而不是
    固定間隔——如果失敗原因是跟本機live_trader搶同一組帳密的session
    （見對話紀錄的假設），本機那邊通常不會秒退，給更長的等待時間比較有
    機會等到session釋放。"""
    last_err: Exception | None = None
    attempts = len(retry_delays) + 1
    for i in range(attempts):
        try:
            sdk = FubonSDK()
            result = sdk.apikey_login(
                os.environ["FUBON_ID"],
                os.environ["FUBON_API_KEY"],
                _resolve_cert_path(),
                os.environ.get("FUBON_CERT_PASS") or None,
            )
            if not result.is_success:
                raise RuntimeError(f"富邦 API Key 登入失敗: {result.message}")
            return sdk, result.data
        except Exception as e:
            last_err = e
            if i < len(retry_delays):
                delay = retry_delays[i]
                print(
                    f"[fubon_api] 登入失敗（第{i + 1}/{attempts}次）：{e}，"
                    f"{delay}秒後重試...",
                    flush=True,
                )
                time.sleep(delay)
    raise last_err


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


def intraday_ticker(sdk: FubonSDK, symbol: str) -> dict:
    """REST 行情 API：取得單一股票的即時交易狀態（intraday/ticker/{symbol}），
    含 canDayTrade/canBuyDayTrade/isAttention/isDisposition/securityStatus 等欄位
    ——比 intraday_tickers() 清單端的 isNormal 更直接反映「能不能當沖」。呼叫前
    須先 init_market_data()。Rate limit 300次/分鐘（富邦官方文件），呼叫端要
    自行節流。"""
    reststock = sdk.marketdata.rest_client.stock
    return reststock.intraday.ticker(symbol=symbol)


def intraday_candles(sdk: FubonSDK, symbol: str, timeframe: int = 1) -> list[dict]:
    """REST 行情 API：取得單一股票當日分K（intraday/candles/{symbol}）。呼叫前須先
    init_market_data()。Rate limit 300次/分鐘（富邦官方文件），呼叫端要自行節流。"""
    reststock = sdk.marketdata.rest_client.stock
    r = reststock.intraday.candles(symbol=symbol, timeframe=timeframe)
    return r.get("data", [])


def historical_candles(sdk: FubonSDK, symbol: str, **params) -> list[dict]:
    """REST 行情 API：取得單一股票歷史K線（historical/candles/{symbol}），語意
    對齊 Fugle 的 /historical/candles（同一套底層 fugle_marketdata 元件，接受
    一樣的 query params：timeframe/from/to/fields/sort/adjusted，2026-07-13
    實測過 from/to 抓日K跟 Fugle 行為一致）。呼叫前須先 init_market_data()。
    Rate limit 60次/分鐘（富邦官方文件），呼叫端要自行節流。

    不帶 timeframe → 日K；timeframe=1 → 近30日分K（無法指定 from/to）。
    """
    reststock = sdk.marketdata.rest_client.stock
    r = reststock.historical.candles(symbol=symbol, **params)
    return r.get("data", [])


def realtime_token(sdk: FubonSDK) -> str:
    return sdk.exchange_realtime_token()


def open_candles_connection(token: str, mode: Mode = Mode.Normal):
    """開一條 WebSocket 連線（stock client），尚未 connect()／subscribe()。
    每呼叫一次會建立一條獨立連線（富邦上限 5 條，見 fubon/config.py）。"""
    return build_websocket_client(mode, token).stock


def subscribe_candles(stock_client, symbol: str):
    stock_client.subscribe({"channel": "candles", "symbol": symbol})


def subscribe_trades(stock_client, symbol: str):
    """訂閱成交明細 tick channel（跟 candles 不同，是逐筆成交，不是分K），
    見 fubon/tick_ws.py::FubonTickCollector。連線本身用 open_candles_connection()
    開就好（channel-agnostic，只是 build_websocket_client(mode, token).stock），
    不需要另外包一個 open_trades_connection()。"""
    stock_client.subscribe({"channel": "trades", "symbol": symbol})


if __name__ == "__main__":
    sdk, accounts = login()
    print("登入成功，帳戶：")
    for acc in accounts:
        print(f"  {acc.name}  {acc.branch_no}-{acc.account}  ({acc.account_type})")
