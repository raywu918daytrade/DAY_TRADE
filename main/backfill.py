"""
開機補載：若今日 db/m1_live/ 已經有資料（例如重啟、或分K收集器開機前就已經
在收），立刻用現有資料跑一次 predict_live()，填 SSE monitoring，不用等下一個
分鐘 tick。只在開機呼叫一次，不是排程。

跟 fubon/marketdata_ws.py::_backfill_intraday() 是不同層級的東西：那支是用
REST 補「WebSocket 連線前」缺的原始分K（寫進 db/m1_live/），這裡是用「已經
在 db/m1_live/ 的資料」跑一次推論（填前端監控畫面），兩者互不重疊。
"""
from datetime import datetime, timezone, timedelta

from api import push_monitoring
from data.query import load_m1_live

_TW = timezone(timedelta(hours=8))


def run_startup_backfill(state, threshold: float) -> None:
    if state.day.empty:
        return
    try:
        today_str = datetime.now(_TW).strftime("%Y-%m-%d")
        print(f"[M1] 補載今日分K（{today_str}）...", flush=True)
        m1_now = load_m1_live(today_str)
        if not m1_now.empty:
            last_min = str(m1_now["date"].max())
            print(
                f"  → {len(m1_now):,} 筆，{m1_now['stock_id'].nunique():,} 支，最新分鐘：{last_min}",
                flush=True,
            )
            init_results = state.predict_live(
                last_min,
                state.day,
                day_trade_stocks=state.day_trade_stocks,
                m1_live=m1_now,
                **state.prewarm_cache,
            )
            push_monitoring(last_min, init_results, threshold)
            print(f"✓ [M1] 補載監控完成：{len(init_results)} 支訊號", flush=True)
        else:
            print("  → 今日無分K資料（尚未開盤或非交易日）", flush=True)
    except Exception as e:
        print(f"✗ [M1] 補載失敗: {e}", flush=True)
