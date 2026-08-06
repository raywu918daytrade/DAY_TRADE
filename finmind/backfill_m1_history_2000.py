"""FinMind 分K 補齊 — TaiwanStockKBar，擴大股票母體版（2026-08-06加）：固定
股票清單見 finmind/stock_universe_2000.py（db/tickers/tickers.parquet 過濾出
的4碼一般個股，排除ETF，共1877支，比 finmind/backfill_m1.py 那400支多了
1477支）、固定範圍 2020-01 ~ 2026-07（跟400支那邊已經補完的起點對齊）。

薄wrapper，核心邏輯在 finmind/backfill_m1_history.py，這支只固定範圍/名單，
跟 finmind/backfill_m1.py 是同一種「不用想、直接執行」的固定版本，差別只在
股票清單來源（400支依成交量排名 vs 這支1877支「全部留下」）跟預設範圍起點
（2025-08 vs 2020-01）。想自己指定任意股票清單/範圍的通用工具版本，見
finmind/backfill_m1_history.py。

⚠️ 規模警告：1877支 × 79個月（2020-01~2026-07）× 約21個交易日 ≈ 310萬組
(股票,交易日) 請求，扣掉跟400支重疊、已經補過的部分後粗估仍有 ~245萬組
待補，6000次/小時的rate limit（安全上限5500）算要跑約445+小時（約18.5天）
連續執行，量級接近 finmind/backfill_m1_history.py docstring 記載的「全市場
--all」規模。一定要在自己的機器/伺服器上用 nohup、caffeinate 跑成真正持續
執行的程序，不是能在對話 session 裡「背景跑一下」的任務。

⚠️ 正式跑一定要用終端機 nohup 啟動，不要用 VS Code 執行/偵錯按鈕。

⚠️ 不要跟 finmind/backfill_m1.py / finmind/backfill_tick.py 同時跑：都是
獨立process，各自的 _RateLimiter 只認得自己本地送出的request，同時起會
共同超過FinMind伺服器端真實6000/小時上限。序列跑，先跑完其他回補再跑這支
（這支規模最大，建議排最後）。

正式啟動（複製貼到終端機，不是VS Code）：
    cd /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade
    python -m finmind.stock_universe_2000   # 先產生股票清單（只需跑一次）
    nohup caffeinate -i python3 -m finmind.backfill_m1_history_2000 > finmind_m1_2000.log 2>&1 &

電腦快關機、想把剩下的額度用完不浪費：帶 --max-requests=N，送滿N筆request
就安全停止、正常結束（不是錯誤），下次重跑（不帶這個參數）會自動接續：
    python3 -m finmind.backfill_m1_history_2000 --max-requests=3000

再加 --burst 可以跳過節流、接近同時把N筆全部發出去（見
finmind/m1_api.py 檔頭的風險說明，一定要精算過剩餘額度才用）：
    python3 -m finmind.backfill_m1_history_2000 --max-requests=3000 --burst

看即時進度：
    tail -f /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade/finmind_m1_2000.log

確認還在跑：
    ps aux | grep backfill_m1_history_2000

指定範圍（不指定就用預設的 2020-01 ~ 2026-07，股票清單一律固定1877檔）：
    caffeinate -i python3 -m finmind.backfill_m1_history_2000 2024-01 2025-12
"""

import asyncio

from finmind.backfill_m1_history import backfill_history, run_forever
from finmind.m1_api import RequestBudgetExhausted, parse_max_requests, set_burst_mode, set_request_budget
from finmind.stock_universe_2000 import load_stock_universe_2000

_DEFAULT_START = "2020-01"
_DEFAULT_END = "2026-07"

if __name__ == "__main__":
    import sys

    _argv, _max_requests, _burst = parse_max_requests(sys.argv[1:])
    if len(_argv) >= 2:
        _start, _end = _argv[0], _argv[1]
    else:
        _start, _end = _DEFAULT_START, _DEFAULT_END
    if _max_requests:
        set_request_budget(_max_requests)
    if _burst:
        set_burst_mode(True)
    _stocks = load_stock_universe_2000()
    try:
        asyncio.run(
            run_forever(
                history_fn=backfill_history,
                history_kwargs={"start_ym": _start, "end_ym": _end, "stock_list": _stocks},
            )
        )
    except RequestBudgetExhausted as e:
        print(f"\n⏸ {e}")
        print(
            "已達到本次設定的 request 上限，安全停止（已完成的部分都存檔了）。"
            "之後重跑這支腳本（不用帶 --max-requests）會自動從中斷處繼續。"
        )
