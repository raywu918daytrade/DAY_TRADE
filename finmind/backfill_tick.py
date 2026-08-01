"""FinMind Tick 補齊 — TaiwanStockPriceTick，固定股票清單（見
finmind/tick_universe.py，399檔4碼股票依成交量排序+0050，共400檔，排除ETF代號）、
固定範圍 2025-08 ~ 2026-07（含當月至今）。

薄wrapper，核心邏輯在 finmind/backfill_tick_history.py，這支只固定範圍/名單。
用 m1_api.run_forever()（跟分K共用同一套額度恢復輪詢邏輯，見
backfill_m1_history.py::run_forever() 的說明）：撞到400/402不會整個程式結束，
每60秒查一次額度自動恢復。

⚠️ 正式跑一定要用終端機 nohup 啟動，不要用 VS Code 執行/偵錯按鈕（同
backfill_all.py/backfill_top1000.py 的既有警告，長時間連續執行的教訓同樣
適用）。

⚠️ 不要跟分K的 backfill_all.py / backfill_top1000.py 同時跑：兩支各自是
獨立process，各自的 _RateLimiter 都只認得自己本地送出的request，同時起會
有一段時間各自以為額度夠用而一起衝量，可能共同超過FinMind伺服器端真實
6000/小時上限。序列跑，先跑完一個再跑下一個。

正式啟動（複製貼到終端機，不是VS Code）：
    cd /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade
    nohup caffeinate -i python3 -m finmind.backfill_tick > finmind_tick.log 2>&1 &

電腦快關機、想把剩下的額度用完不浪費：帶 --max-requests=N，送滿N筆request
就安全停止、正常結束（不是錯誤），下次重跑（不帶這個參數）會自動接續：
    python3 -m finmind.backfill_tick --max-requests=3000

再加 --burst 可以跳過節流、接近同時把N筆全部發出去（見
finmind/backfill_tick_history.py 檔頭的風險說明，一定要精算過剩餘額度才用）：
    python3 -m finmind.backfill_tick --max-requests=3000 --burst

看即時進度：
    tail -f /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade/finmind_tick.log

確認還在跑：
    ps aux | grep backfill_tick

指定範圍（不指定就用預設的 2025-08 ~ 2026-07，股票清單一律固定400檔）：
    caffeinate -i python3 -m finmind.backfill_tick 2024-01 2025-12
"""

import asyncio

from finmind.backfill_m1_history import run_forever
from finmind.backfill_tick_history import backfill_tick_history
from finmind.m1_api import RequestBudgetExhausted, parse_max_requests, set_burst_mode, set_request_budget
from finmind.tick_universe import load_tick_universe

_DEFAULT_START = "2025-08"
_DEFAULT_END = "2026-07"

if __name__ == "__main__":
    import sys

    # 2026-07-31 修：原本這裡完全沒讀位置參數，不管使用者傳什麼日期範圍都
    # 悄悄忽略、直接用寫死的_DEFAULT_START/_DEFAULT_END，跟backfill_all.py/
    # backfill_top1000.py/backfill_tick_history.py的行為不一致，容易誤會
    # 「有帶參數就會生效」。股票清單還是固定400檔（這支腳本存在的意義），
    # 但日期範圍现在比照其他腳本可以覆蓋。
    _argv, _max_requests, _burst = parse_max_requests(sys.argv[1:])
    if len(_argv) >= 2:
        _start, _end = _argv[0], _argv[1]
    else:
        _start, _end = _DEFAULT_START, _DEFAULT_END
    if _max_requests:
        set_request_budget(_max_requests)
    if _burst:
        set_burst_mode(True)
    _stocks = load_tick_universe()
    try:
        asyncio.run(
            run_forever(
                history_fn=backfill_tick_history,
                history_kwargs={"start_ym": _start, "end_ym": _end, "stocks": _stocks},
            )
        )
    except RequestBudgetExhausted as e:
        print(f"\n⏸ {e}")
        print("已達到本次設定的 request 上限，安全停止（已完成的部分都存檔了）。"
              "之後重跑這支腳本（不用帶 --max-requests）會自動從中斷處繼續。")
