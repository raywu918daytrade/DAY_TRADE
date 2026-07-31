"""
FinMind 分K 一次性歷史補齊 — 從 2019-01-01（FinMind TaiwanStockKBar 資料起點）
補到指定月份為止，逐月呼叫 finmind/finmind_api.py 的 backfill_month()，核心邏輯只有
一份（見 finmind/finmind_api.py），這支只負責「跑哪些月份、依序跑、印跨月進度」。

跟 finmind/finmind_api.py 的 __main__（一次補一個月，像 Fugle 的 update_m1() 那種
「今天缺哪幾天就補哪幾天」用法）是分開的兩支腳本：
    finmind/finmind_api.py             日常用，一次補一個月（例如剛好發現某個月缺資料）
    finmind/backfill_history.py 一次性用，把很久以前的歷史全部補起來

⚠️ 規模警告（2026-07-14 實測 db/fugle_day 估算）：2019-01-01 ~ 2026-05 全部
補完，大約 258萬組 (股票,交易日) 請求，用 6000次/小時的rate limit 算，要跑
約 430小時（~18天）連續執行。這不是能在一次對話 session 裡「背景跑一下」
的任務，要在自己的機器/伺服器上用 nohup、screen、tmux 或系統服務跑成真正
持續執行的程序，能撐過電腦睡眠/斷線中斷（見下面用法）。

中斷續傳：每個月呼叫 backfill_month() 時，都會先讀那個月 db/m1 現有的
(股票,交易日) 組合、只補缺的部分（見 finmind/finmind_api.py::_existing_pairs()），
所以任何時候中斷、重新執行這支腳本，會自動跳過已經完成的月份/組合，不會
重新請求已經有的資料，不會浪費 rate limit 額度。

用法：
    python -m finmind.backfill_history                          # 2019-01 補到 2026-05（預設）
    python -m finmind.backfill_history 2019-01 2026-05           # 指定範圍
    python -m finmind.backfill_history 2026-01 2026-05 1000      # 指定範圍 + 每月只補前1000支（依成交量）
    nohup python -m finmind.backfill_history > finmind_backfill.log 2>&1 &   # 背景持續執行
    python -m finmind.backfill_history --max-requests=3000       # 送滿3000筆就安全停止（見下方說明）
    python -m finmind.backfill_history --max-requests=3000 --burst  # 上面那個+不節流，接近同時發出去

電腦快關機、想把剩下的額度用完不浪費：帶 --max-requests=N（可以跟其他參數
併用，順序不拘），送滿N筆request就存檔收工、正常結束（不是錯誤），不用等
FatalAPIError；下次重跑（不用帶這個參數）會用既有的中斷續傳機制自動接著補，
不會重複下載已經有的部分（見 finmind/finmind_api.py::RequestBudgetExhausted）。

再加 --burst：跳過原本每筆間隔約0.655秒的節流，併發數/flush批次放大成
min(待補組數, 剩餘budget)，接近同時把N筆全部送出去，不會乖乖排隊等。
⚠️ 只給「已經自己精算過FinMind帳號剩餘額度、只跑這一次」的場景用——
2026-07-14 曾經因為瞬間爆量被封鎖IP 30分鐘（見 finmind/finmind_api.py::
IPBannedError 的說明），這個模式沒辦法保證不會再發生，一定要搭配
--max-requests 一起用，不然直接拒絕執行。
"""

import asyncio

import pandas as pd

from finmind.finmind_api import (
    FatalAPIError,
    RequestBudgetExhausted,
    backfill_month,
    check_quota,
    parse_max_requests,
    reload_token,
    set_request_budget,
    sync_rate_limiter,
)

_DEFAULT_START = "2019-01"
_DEFAULT_END = "2026-05"
RESUME_BELOW_USAGE = 4000  # run_forever() 暫停後，用量要降到這個門檻以下才恢復（見下方說明）


def _month_range(start_ym: str, end_ym: str) -> list[tuple[int, int]]:
    """展開成 [(year, month), ...] 清單，start_ym/end_ym 格式 YYYY-MM，由新到舊
    （end_ym 先跑、start_ym 最後跑）——近期資料對模型比較有價值，中途要是
    停下來，先補到的會是比較有用的那部分，不要照時間先後順序從最舊補起。"""
    periods = pd.period_range(start=start_ym, end=end_ym, freq="M")
    return [(p.year, p.month) for p in periods][::-1]


async def backfill_history(
    start_ym: str = _DEFAULT_START,
    end_ym: str = _DEFAULT_END,
    top_n_by_volume: int | None = None,
):
    """top_n_by_volume: 選填，每個月只補平均日成交量最高的前N支（見
    finmind/finmind_api.py::backfill_month() 的說明）。之後想補剩下的股票，直接不帶
    這個參數對同一個範圍再呼叫一次即可，已經有的會自動跳過。"""
    months = _month_range(start_ym, end_ym)
    print(f"歷史補齊範圍：{start_ym} ~ {end_ym}，共 {len(months)} 個月")
    if top_n_by_volume:
        print(f"每月只補前 {top_n_by_volume} 支（依平均日成交量排序）")
    print("=" * 60)

    for i, (year, month) in enumerate(months, 1):
        print(f"\n[{i}/{len(months)}] {year}-{month:02d}")
        try:
            await backfill_month(year, month, top_n_by_volume=top_n_by_volume)
        except FileNotFoundError as e:
            # db/fugle_day 沒有那個月的資料，沒辦法確定交易日/股票清單，
            # 跳過但不中斷整個歷史補齊（例如 db/fugle_day 本身也有缺口）。
            print(f"  跳過：{e}")
        except RequestBudgetExhausted:
            # 使用者主動設的用量上限，不是「這個月失敗」——直接往上拋，
            # 不要落到下面的 except Exception 被當成單月失敗、繼續跑下個月。
            raise
        except FatalAPIError as e:
            # 任何4xx（token壞掉/額度用完/IP被封/其他），繼續跑後面88個月
            # 只會全部用同一個原因失敗，浪費時間——整支腳本直接停止，處理好
            # 問題後重跑會自動從這個月接著繼續（見 backfill_month 的續傳邏輯）。
            print(f"\n⚠️ {e}")
            print(f"在 {year}-{month:02d} 停止，處理好問題後重跑這支腳本會自動接著補")
            raise
        except Exception as e:
            print(f"  {year}-{month:02d} 失敗，跳到下一個月，稍後重跑這支腳本會重試: {e}")

    print("\n" + "=" * 60)
    print("全部月份跑完")


async def run_forever(
    history_fn=backfill_history,
    history_kwargs: dict | None = None,
    check_interval: int = 60,
):
    """backfill_history() 的外層自動重試包裝，給長時間背景執行用（18天的
    全歷史補齊、或15.7小時的前1000支，都不希望撞到任何4xx就整個停掉、需要
    人工重新執行）。

    history_fn/history_kwargs：2026-07-29 從固定呼叫 backfill_history() 改成
    接受任意 callable + kwargs——額度/token恢復輪詢這段等待迴圈是這整套系統
    教訓最多、最容易出錯的部分（下面的說明都是踩過的坑），值得只維護一份；
    finmind/backfill_tick_history.py::backfill_tick_history() 也是同構的
    「逐月回補、FatalAPIError整批停止」，直接重用這個迴圈，不用複製一份。
    預設值 history_fn=backfill_history（不帶kwargs）保留舊行為，但舊呼叫端
    如果是用位置參數帶 start_ym/end_ym/top_n_by_volume 的寫法就要改成
    history_kwargs={...}（見 backfill_all.py/backfill_top1000.py 的呼叫點）。

    撞到 FatalAPIError（TokenError/QuotaError/IPBannedError/OtherFatalError，
    見 finmind/finmind_api.py 的說明）時不會讓程式結束，而是每 check_interval 秒
    呼叫一次 finmind.finmind_api.check_quota()（FinMind官方帳號用量查詢API）：
        - TokenError：使用者要自己去 FinMind 官網重新登入拿新token、更新
          .env 的 FINMIND_TOKEN，這裡等待期間每次重試前都會 reload_token()
          撿到新值，不用手動重啟這支程式。
        - QuotaError：2026-07-14 實測帳號是 api_request_limit_hour=6000、
          api_request_limit_day='-'（純小時滾動，沒有另外的日/月總量限制），
          等額度用量（user_count）掉回門檻以下，代表這個小時視窗已經重置，
          自動繼續。
        - IPBannedError：check_quota() 本身也會被同一個IP封鎖擋掉（回應
          msg 不會是 "success"），下面的迴圈會自然持續等待，等封鎖解除
          （2026-07-14實測約30分鐘）check_quota() 才會成功，不用另外寫
          判斷邏輯。
    只要 check_quota() 回應 msg=="success" 且用量低於上限，就重新呼叫
    backfill_history()——續傳邏輯會自動跳過已完成的部分，不會重複下載。

    2026-07-14 教訓：真正開始下載前一定要先 sync_rate_limiter()——本地的
    _RateLimiter 只認得「這個process自己送出的request」，如果之前有別次
    執行（例如中途被強制中斷的測試）已經真的消耗掉部分額度，這次重新啟動
    完全不知道，會誤以為滿額可用，等真的送出去才會撞到402，等於「檢查的
    時機太晚」。第一次啟動、跟每次額度恢復要重新開始前，都會先同步一次。
    """
    kwargs = history_kwargs or {}
    while True:
        await sync_rate_limiter()
        try:
            await history_fn(**kwargs)
            return
        except FatalAPIError as e:
            print(f"\n⏸ 暫停：{e}")
            print(f"每 {check_interval} 秒自動查一次額度/token狀態，恢復後會自動繼續")
            while True:
                await asyncio.sleep(check_interval)
                reload_token()
                try:
                    info = await check_quota()
                except Exception as qe:
                    print(f"  查詢用量失敗（{qe}），{check_interval}秒後再查一次")
                    continue
                if info.get("msg") != "success":
                    print(f"  仍異常（{info.get('msg')}），{check_interval}秒後再查一次")
                    continue
                used = info.get("user_count")
                limit = info.get("api_request_limit_hour")
                print(f"  目前用量: {used}/{limit}")
                # 2026-07-14：不要用量一低於上限（例如5999/6000）就馬上恢復，
                # 那樣重新開始沒多久、額度視窗還沒真的空出多少，很快又會
                # 撞回去。等降到4000以下（明顯有餘裕）才恢復，比較不會
                # 剛恢復又立刻暫停、反覆橫跳。
                if isinstance(used, (int, float)) and used < RESUME_BELOW_USAGE:
                    print(f"  用量已降到 {RESUME_BELOW_USAGE} 以下，重新開始執行")
                    break


if __name__ == "__main__":
    import sys

    _argv, _max_requests = parse_max_requests(sys.argv[1:])
    if len(_argv) >= 2:
        _start, _end = _argv[0], _argv[1]
    else:
        _start, _end = _DEFAULT_START, _DEFAULT_END
    _top_n = int(_argv[2]) if len(_argv) >= 3 else None
    if _max_requests:
        set_request_budget(_max_requests)
    try:
        asyncio.run(run_forever(history_kwargs={"start_ym": _start, "end_ym": _end, "top_n_by_volume": _top_n}))
    except RequestBudgetExhausted as e:
        print(f"\n⏸ {e}")
        print("已達到本次設定的 request 上限，安全停止（已完成的部分都存檔了）。"
              "之後重跑這支腳本（不用帶 --max-requests）會自動從中斷處繼續。")
