"""
分K收集器背景執行緒：REST（Fugle，預設）或富邦 WebSocket，依 main/config.py
的 M1_COLLECTOR 切換。

on_minute callback 由呼叫端（main/live_trader.py）傳進來，這裡不 import
live_trader，避免循環匯入（live_trader.py 會 import 這支檔案）。
"""
import os
import time

from api import append_system_log as _log_sys, set_collector_status
from data.m1_rest import M1RestPoller
from main.config import M1_COLLECTOR

# collector.start() 崩潰後，等幾秒再自動重建一個全新的 collector 重試。
# 2026-07-13 實際發生過：db/m1_live/ 寫檔 race condition 讓 collector 整個
# 掛掉，start_collector() 原本只印一行「Collector 中斷」就徹底放棄，沒有
# 自動恢復機制，導致停擺到有人發現、手動重啟才恢復。
_COLLECTOR_RETRY_DELAY = int(os.environ.get("COLLECTOR_RETRY_DELAY", "10"))


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


def _build_collector(state, on_minute):
    if M1_COLLECTOR == "fubon_ws":
        from fubon.marketdata_ws import FubonM1Collector

        return FubonM1Collector(on_minute=on_minute)
    return M1RestPoller(
        on_minute=on_minute,
        stocks=lambda: _get_stocks(state),
        on_rate_limited=_on_rate_limited,
    )


def start_collector(state, on_minute) -> None:
    """分K收集器背景執行緒，異常時更新 collector 狀態供 /health 回傳，並自動重試。

    collector.start() 是長時間 block 的主迴圈（WebSocket 連線或 REST 輪詢），
    任何未預期的例外都會讓它整個結束。這裡外層包一層重試迴圈：崩潰後等
    _COLLECTOR_RETRY_DELAY 秒、重新建立一個全新的 collector 實例再試一次
    （不重用崩潰的舊實例，避免帶著壞掉的連線/session 狀態），不會無限次
    但也不設上限——只要還在交易時段，就應該持續嘗試恢復，不要停擺到
    需要人工發現、手動重啟。

    M1_COLLECTOR=fubon_ws：改用富邦 WebSocket（fubon/marketdata_ws.py），需要
    .env 設好 FUBON_ID/PASSWORD/CERT，且開盤前先跑過 `python -m fubon.subscribe_list`
    產生訂閱清單。2026-07-12 起 .env 預設已切到 fubon_ws；candles channel 實際
    payload 格式還沒在平日開盤時段驗證過，第一個開盤日務必看 log 確認資料正常，
    有異常先切回 rest（M1RestPoller，Fugle REST）。
    """
    attempt = 0
    while True:
        attempt += 1
        collector = _build_collector(state, on_minute)
        try:
            set_collector_status("running")
            collector.start()
        except Exception as e:
            set_collector_status("error")
            msg = f"Collector 中斷（第{attempt}次）: {e}"
            print(msg, flush=True)
            _log_sys(msg, "error")
            print(f"  {_COLLECTOR_RETRY_DELAY} 秒後自動重試...", flush=True)
            time.sleep(_COLLECTOR_RETRY_DELAY)
            continue
        else:
            set_collector_status("stopped")
            break
