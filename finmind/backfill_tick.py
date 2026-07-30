"""FinMind Tick 補齊 — TaiwanStockPriceTick，固定股票清單（見
finmind/tick_universe.py，399檔4碼股票依成交量排序+0050，共400檔，排除ETF代號）、
固定範圍 2025-08 ~ 2026-07（含當月至今）。

薄wrapper，核心邏輯在 finmind/backfill_tick_history.py，這支只固定範圍/名單。
用 finmind_api.run_forever()（跟分K共用同一套額度恢復輪詢邏輯，見
backfill_history.py::run_forever() 的說明）：撞到400/402不會整個程式結束，
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

看即時進度：
    tail -f /Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_day_trade/finmind_tick.log

確認還在跑：
    ps aux | grep backfill_tick
執行：
    caffeinate -i python3 -m finmind.backfill_tick 2025-08 2026-07
"""

import asyncio

from finmind.backfill_history import run_forever
from finmind.backfill_tick_history import backfill_tick_history
from finmind.tick_universe import load_tick_universe

_DEFAULT_START = "2025-08"
_DEFAULT_END = "2026-07"

if __name__ == "__main__":
    _stocks = load_tick_universe()
    asyncio.run(
        run_forever(
            history_fn=backfill_tick_history,
            history_kwargs={"start_ym": _DEFAULT_START, "end_ym": _DEFAULT_END, "stocks": _stocks},
        )
    )
