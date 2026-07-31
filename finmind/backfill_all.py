"""
FinMind 分K 補齊 — 全部股票，不限前N支。

薄wrapper，核心邏輯全部在 finmind/backfill_history.py，這支只是固定好預設
範圍（2019-01 ~ 2026-05，FinMind TaiwanStockKBar 資料起點到現在），不帶
top_n_by_volume，補完整股票母體。跟 finmind/backfill_top1000.py 是分開的
兩支腳本，用同一套續傳邏輯（見 finmind/finmind_api.py::_existing_pairs()），互不
衝突：backfill_top1000.py 先跑過的 (股票,日期) 組合，這支執行時會自動跳過，
只補剩下沒補到的股票，不會重複下載。

⚠️ 規模警告：2019-01 ~ 2026-05 全部股票、全部補完，大約需要18天連續執行
（見 finmind/backfill_history.py 檔頭說明），要用 nohup/caffeinate 背景跑，
不能在對話 session 裡背景執行完。

用 run_forever()（不是 backfill_history()）：撞到400（token無效）或402
（額度用完）不會整個程式結束，會每60秒自動查一次FinMind官方用量API，
恢復後自動繼續，18天不用人工盯著重啟（見
finmind/backfill_history.py::run_forever() 的說明）。

用法：
    python -m finmind.backfill_all                      # 2019-01 補到 2026-05（預設）
    python -m finmind.backfill_all 2019-01 2026-05       # 指定範圍

⚠️ 一定要用終端機的 nohup 啟動，不要用 VS Code 的執行/偵錯按鈕跑
（2026-07-14 發現：那樣程式綁在 VS Code 的 debugger session 上，關掉
VS Code/停止偵錯/電腦睡眠都會讓它中斷，輸出也只會進 VS Code 的偵錯主控台，
不會寫進下面這個log檔）。18天要連續執行，中途不能讓電腦睡眠。

正式啟動（複製貼到終端機，不是VS Code）：
    cd /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade
    nohup caffeinate -i python3 -m finmind.backfill_all > finmind_all.log 2>&1 &

電腦快關機、想把剩下的額度用完不浪費：帶 --max-requests=N，送滿N筆request
就安全停止、正常結束（不是錯誤），下次重跑（不帶這個參數）會自動接續：
    python3 -m finmind.backfill_all --max-requests=3000

看即時進度：
    tail -f /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade/finmind_all.log

確認還在跑：
    ps aux | grep backfill_all
"""

import asyncio

from finmind.backfill_history import run_forever
from finmind.finmind_api import RequestBudgetExhausted, parse_max_requests, set_request_budget

_DEFAULT_START = "2019-01"
_DEFAULT_END = "2026-05"

if __name__ == "__main__":
    import sys

    _argv, _max_requests = parse_max_requests(sys.argv[1:])
    if len(_argv) >= 2:
        _start, _end = _argv[0], _argv[1]
    else:
        _start, _end = _DEFAULT_START, _DEFAULT_END
    if _max_requests:
        # 電腦快關機、想把剩下的額度用完不浪費：--max-requests=N，送滿N筆就
        # 安全停止、正常結束，下次重跑（不帶這個參數）自動接續（見
        # finmind/finmind_api.py::RequestBudgetExhausted）。
        set_request_budget(_max_requests)
    try:
        asyncio.run(run_forever(history_kwargs={"start_ym": _start, "end_ym": _end, "top_n_by_volume": None}))
    except RequestBudgetExhausted as e:
        print(f"\n⏸ {e}")
        print("已達到本次設定的 request 上限，安全停止（已完成的部分都存檔了）。"
              "之後重跑這支腳本（不用帶 --max-requests）會自動從中斷處繼續。")
