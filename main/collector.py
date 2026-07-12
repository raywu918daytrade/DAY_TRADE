"""
分K收集器背景執行緒：REST（Fugle，預設）或富邦 WebSocket，依 main/config.py
的 M1_COLLECTOR 切換。

on_minute callback 由呼叫端（main/live_trader.py）傳進來，這裡不 import
live_trader，避免循環匯入（live_trader.py 會 import 這支檔案）。
"""
from api import set_collector_status
from data.m1_rest import M1RestPoller
from main.config import M1_COLLECTOR


def _get_stocks(state):
    """每次重連都取最新當沖標的；非盤中無清單時回退到所有可交易股票

    固定加入 0050：它本身不是當沖候選股，但 rally 策略的 idx_* 特徵
    （大盤 1分K 相對強弱）需要它當天的即時分K，若當沖候選清單剛好沒選到
    0050，db/m1_live/ 就不會有它的資料，predict_live() 算 idx_* 特徵時
    會直接 KeyError。
    """
    from data.fugle_tickers import fugle_stocks

    stocks = list(state.day_trade_stocks) if state.day_trade_stocks else fugle_stocks()
    if "0050" not in stocks:
        stocks.append("0050")
    return stocks


def _on_rate_limited():
    """Fugle 429 時推送前端警示。"""
    try:
        from api import push_alert

        push_alert("Fugle REST API 限流，使用快取分K繼續監控（SL/TP 仍有效）", level="warning")
    except Exception:
        pass


def start_collector(state, on_minute) -> None:
    """分K收集器背景執行緒，異常時更新 collector 狀態供 /health 回傳。

    M1_COLLECTOR=fubon_ws：改用富邦 WebSocket（fubon/marketdata_ws.py），需要
    .env 設好 FUBON_ID/PASSWORD/CERT，且開盤前先跑過 `python -m fubon.subscribe_list`
    產生訂閱清單。2026-07-12 起 .env 預設已切到 fubon_ws；candles channel 實際
    payload 格式還沒在平日開盤時段驗證過，第一個開盤日務必看 log 確認資料正常，
    有異常先切回 rest（M1RestPoller，Fugle REST）。
    """
    if M1_COLLECTOR == "fubon_ws":
        from fubon.marketdata_ws import FubonM1Collector

        collector = FubonM1Collector(on_minute=on_minute)
    else:
        collector = M1RestPoller(
            on_minute=on_minute,
            stocks=lambda: _get_stocks(state),
            on_rate_limited=_on_rate_limited,
        )

    try:
        set_collector_status("running")
        collector.start()
    except Exception as e:
        set_collector_status("error")
        print(f"Collector 中斷: {e}")
    else:
        set_collector_status("stopped")
