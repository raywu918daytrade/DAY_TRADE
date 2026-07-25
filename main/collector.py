"""
分K收集器背景執行緒：富邦 WebSocket（fubon/marketdata_ws.py）。

on_minute callback 由呼叫端（main/live_trader.py）傳進來，這裡不 import
live_trader，避免循環匯入（live_trader.py 會 import 這支檔案）。

2026-07-13：拿掉了 Fugle REST 輪詢器（M1RestPoller）這個備援選項——
Fugle 日內行情 API 官方就是 60次/分鐘，本來就跟不上「每分鐘」的節奏
（500~1000支要8~17分鐘一輪，這正是當初改用富邦 WebSocket 的原因），留著
當備援意義不大，維護它本身也有成本（不同資料源、獨立的 bug 歷史）。
真正該防的是「我們自己的富邦 WebSocket 程式碼出包」（2026-07-13 實際發生
過一次），不是換一個更慢的資料源，而是用富邦自己的 REST intraday（
300次/分鐘，見 fubon/fubon_api.py::intraday_candles()）當備援——這個還沒
做，是已知的 TODO（見 memory）。
"""
import os
import time

from api import append_system_log as _log_sys, set_collector_status

# collector.start() 崩潰後，等幾秒再自動重建一個全新的 collector 重試。
# 2026-07-13 實際發生過：db/m1_live/ 寫檔 race condition 讓 collector 整個
# 掛掉，start_collector() 原本只印一行「Collector 中斷」就徹底放棄，沒有
# 自動恢復機制，導致停擺到有人發現、手動重啟才恢復。
_COLLECTOR_RETRY_DELAY = int(os.environ.get("COLLECTOR_RETRY_DELAY", "10"))


def start_collector(on_minute, backfill_done=None) -> None:
    """分K收集器背景執行緒，異常時更新 collector 狀態供 /health 回傳，並自動重試。

    collector.start() 是長時間 block 的主迴圈（WebSocket 連線），任何未預期
    的例外都會讓它整個結束。這裡外層包一層重試迴圈：崩潰後等
    _COLLECTOR_RETRY_DELAY 秒、重新建立一個全新的 collector 實例再試一次
    （不重用崩潰的舊實例，避免帶著壞掉的連線/session 狀態），不設重試上限
    ——只要還在交易時段，就應該持續嘗試恢復，不要停擺到需要人工發現。

    backfill_done：threading.Event（見 main/state.py::AppState.backfill_done
    的說明），原封不動傳給每一輪新建立的 FubonM1Collector，讓 on_minute()
    知道這一輪連線的資料缺口補完了沒有。
    """
    from fubon.marketdata_ws import FubonM1Collector

    attempt = 0
    while True:
        attempt += 1
        collector = FubonM1Collector(on_minute=on_minute, backfill_done=backfill_done)
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
