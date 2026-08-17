"""
開機補載：若今日 db/m1_live/ 已經有資料（例如重啟、或分K收集器開機前就已經
在收），立刻用現有資料跑一次 predict_live()，填 SSE monitoring，不用等下一個
分鐘 tick。只在開機呼叫一次，不是排程。

跟 fubon/marketdata_ws.py::_backfill_m1_live() 是不同層級的東西：那支是用
REST 補「WebSocket 連線前」缺的原始分K（寫進 db/m1_live/），這裡是用「已經
在 db/m1_live/ 的資料」跑一次推論（填前端監控畫面），兩者互不重疊。
"""
from datetime import datetime, timezone, timedelta

from api import get_setting, push_monitoring
from data.query import load_m1_live

_TW = timezone(timedelta(hours=8))


def run_startup_backfill(state) -> None:
    if state.day.empty:
        return
    try:
        today_str = datetime.now(_TW).strftime("%Y-%m-%d")
        print(f"[M1] 補載今日分K（{today_str}）...", flush=True)
        m1_now = load_m1_live(today_str)
        if m1_now.empty:
            print("  → 今日無分K資料（尚未開盤或非交易日）", flush=True)
            return
        last_min = str(m1_now["date"].max())
        print(
            f"  → {len(m1_now):,} 筆，{m1_now['stock_id'].nunique():,} 支，最新分鐘：{last_min}",
            flush=True,
        )
        last_dt = datetime.strptime(last_min, "%Y-%m-%d %H:%M:%S")
        h, m = last_dt.hour, last_dt.minute
        for s in state.strategies.values():
            # 2026-07-29發現的bug：這裡原本沒有比對 session_start/session_end，
            # 不管重啟發生在幾點，都會硬對每個策略跑一次 predict_live()——如果
            # 重啟發生在該策略的進場窗口之外（例如收盤後13:30重啟，cnn_up只在
            # 9:00~10:00有效），會餵給模型它從沒訓練過的時段資料，算出來的
            # 監控數字沒有意義，還會卡在 _monitoring 裡（push_monitoring() 的
            # 過期清理只在被呼叫時才觸發，這個策略當天不會再被呼叫，會一直
            # 卡著直到隔天重置）。跟 main/live_trader.py::on_minute() 用同一套
            # session時段判斷，超出範圍就跳過，不产生這筆監控資料。
            if (h, m) < s.session_start or (h, m) > s.session_end:
                print(f"  [{s.name}] 補載當下（{last_min[11:16]}）不在進場窗口內，跳過", flush=True)
                continue

            # 每個策略各自的門檻：先查前端 settings 的全域覆蓋值，沒設定才
            # fallback 該策略自己的預設值（見 main/state.py::StrategyState 的
            # 說明），跟 main/live_trader.py 的 on_minute() 用同一套邏輯。
            threshold = float(get_setting("threshold") or s.threshold)
            init_results = s.predict_live(
                last_min,
                state.day,
                model=s.model,
                day_trade_stocks=state.day_trade_stocks,
                m1_live=m1_now,
                **s.prewarm_cache,
            )
            push_monitoring(last_min, init_results, threshold, strategy=s.name)
            print(f"✓ [M1] [{s.name}] 補載監控完成：{len(init_results)} 支訊號", flush=True)
    except Exception as e:
        print(f"✗ [M1] 補載失敗: {e}", flush=True)
