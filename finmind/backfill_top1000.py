"""
FinMind 分K 補齊 — 只補前1000支（依當月平均日成交量排序）。

薄wrapper，核心邏輯全部在 finmind/backfill_history.py，這支只是固定好範圍跟
top_n_by_volume=1000，不用每次執行都記得帶對參數。跟 finmind/backfill_all.py
是分開的兩支腳本，範圍/股票數不同，但用同一套續傳邏輯（見
finmind/finmind_api.py::_existing_pairs()），互不衝突：這支跑過的 (股票,日期) 組合，
backfill_all.py 之後執行會自動跳過，只補剩下的股票，不會重複下載。

用 run_forever()（不是 backfill_history()）：撞到400（token無效）或402
（額度用完）不會整個程式結束，會每60秒自動查一次FinMind官方用量API，
恢復後自動繼續，長時間背景執行不用人工盯著重啟（見
finmind/backfill_history.py::run_forever() 的說明）。

用法：
    python -m finmind.backfill_top1000                      # 2026-01 補到 2026-05（預設）
    python -m finmind.backfill_top1000 2026-01 2026-05      # 指定範圍

⚠️ 正式跑（要撐15.7小時，見預估耗時）一定要用終端機的 nohup 啟動，不要用
VS Code 的執行/偵錯按鈕跑（2026-07-14 發現：那樣程式綁在 VS Code 的
debugger session 上，關掉 VS Code/停止偵錯/電腦睡眠都會讓它中斷，而且
輸出只會進 VS Code 的偵錯主控台，不會寫進下面這個log檔）。

正式啟動（複製貼到終端機，不是VS Code）：
    cd /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade
    nohup caffeinate -i python3 -m finmind.backfill_top1000 > finmind_top1000.log 2>&1 &

電腦快關機、想把剩下的額度用完不浪費：帶 --max-requests=N，送滿N筆request
就安全停止、正常結束（不是錯誤），下次重跑（不帶這個參數）會自動接續：
    python3 -m finmind.backfill_top1000 --max-requests=3000

再加 --burst 可以跳過節流、接近同時把N筆全部發出去（見
finmind/backfill_history.py 檔頭的風險說明，一定要精算過剩餘額度才用）：
    python3 -m finmind.backfill_top1000 --max-requests=3000 --burst

看即時進度：
    tail -f /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade/finmind_top1000.log

確認還在跑：
    ps aux | grep backfill_top1000
"""

import asyncio

from finmind.backfill_history import run_forever
from finmind.finmind_api import RequestBudgetExhausted, parse_max_requests, set_burst_mode, set_request_budget

_DEFAULT_START = "2026-01"
_DEFAULT_END = "2026-05"
_TOP_N = 1000

if __name__ == "__main__":
    import sys

    _argv, _max_requests, _burst = parse_max_requests(sys.argv[1:])
    if len(_argv) >= 2:
        _start, _end = _argv[0], _argv[1]
    else:
        _start, _end = _DEFAULT_START, _DEFAULT_END
    if _max_requests:
        # 電腦快關機、想把剩下的額度用完不浪費：--max-requests=N，送滿N筆就
        # 安全停止、正常結束，下次重跑（不帶這個參數）自動接續（見
        # finmind/finmind_api.py::RequestBudgetExhausted）。
        set_request_budget(_max_requests)
    if _burst:
        set_burst_mode(True)
    try:
        asyncio.run(run_forever(history_kwargs={"start_ym": _start, "end_ym": _end, "top_n_by_volume": _TOP_N}))
    except RequestBudgetExhausted as e:
        print(f"\n⏸ {e}")
        print("已達到本次設定的 request 上限，安全停止（已完成的部分都存檔了）。"
              "之後重跑這支腳本（不用帶 --max-requests）會自動從中斷處繼續。")
