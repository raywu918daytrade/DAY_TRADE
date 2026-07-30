"""FinMind Tick（TaiwanStockPriceTick）歷史回補 — 補進 db/tick/，跟
finmind/backfill_history.py（分K）同構的月份迴圈驅動，差別是股票清單用固定的
801檔名單（見 finmind/tick_universe.py），不像分K的 top_n_by_volume 是每月
動態重算。

跟 finmind/finmind_api.py 的核心 fetch/rate-limit/錯誤處理邏輯完全共用（見
fetch_tick_day()/_fetch_finmind_day()），這支只負責「跑哪些月份、依序跑、
股票清單哪裡來」。

⚠️ 規模警告：801支 x 12個月的交易日，request數量級是 股票數 x 交易日數，
跟分K同一套規則，但實際 request 數比全部2325支股票的分K回補省很多（約
801/2325 ≈ 三分之一）。不過 tick 資料量遠比分K大（單日單股可能上萬筆，
2330實測單日9424筆），寫檔案的I/O跟磁碟空間會比分K重，仍然不是能在一次
對話 session 裡「背景跑一下」的任務，要用 nohup/caffeinate 背景跑（見
finmind/backfill_tick.py 的啟動說明）。

中斷續傳：邏輯跟分K一致（見 finmind/tick_api.py::existing_tick_pairs()），任何
時候中斷、重新執行都會自動跳過已完成的組合。

volume 單位（2026-07-29 實測驗證）：tick 的 volume（單筆成交量）加總後，跟
db/m1 同一天同一支股票的分K volume加總幾乎完全一致（2330 2026-07-28：tick加總
36220 vs m1加總36184，差距0.1%，來自盤後定盤交易的兩筆tick 13:30/14:30 沒有
對應的分K），也就是 tick 的 volume 單位是「張」，跟 db/m1 相同、**不是**
db/fugle_day 的「股」（db/fugle_day 同一天同一支股票的volume會是tick加總的
約1000倍）——之後如果要拿 tick volume 跟 db/fugle_day 對比，要記得除以1000，
不需要轉換的是拿去跟 db/m1 比。

用法：
    python -m finmind.backfill_tick_history                      # 2025-08 補到 2026-07（預設）
    python -m finmind.backfill_tick_history 2025-08 2026-07      # 指定範圍
"""

import asyncio

import pandas as pd

import aiohttp

from finmind.finmind_api import FatalAPIError, _ROOT
from finmind.tick_api import existing_tick_pairs, fetch_tick_day, save_empty_tick_pairs, save_tick
from finmind.tick_universe import load_tick_universe

_DEFAULT_START = "2025-08"
_DEFAULT_END = "2026-07"
_CONCURRENCY = 4  # 2026-07-30 從1調高：分K回應很小（~270列），完全做完的時間
# 本來就接近節流器的0.655秒下限，併發=1沒有影響；但tick回應大很多、落差極大
# （冷門股幾十列、2330這種熱門股單日上萬列），實測遇到熱門股卡住時，10秒
# 輪詢窗口內只送出3筆（≈6.3秒/筆，遠慢於設計節奏），因為併發=1讓後面的請求
# 全部在等這一筆下載+解析完。調高併發不影響送出「頻率」——每筆還是要先過
# _rate_limiter.acquire()（同一個全域鎖，一樣是每0.655秒最多放行一筆），只是
# 讓「等大檔案回應跑完」這件事可以幾筆一起等，不再被單一熱門股卡住整條
# pipeline。2026-07-14 那次40併發被封鎖的教訓，是在還沒有 _rate_limiter 節流
# 的情況下發生的，跟這裡「送出頻率不變、只加大in-flight上限」是不同情況。
_FLUSH_EVERY = 25  # 比分K的100小很多：tick單筆request資料量比分K重約10-40倍
# （2330單日9424筆 vs 分K單日約270筆），維持「中途被砍掉最多損失多少額度/
# 記憶體緩衝多少列」跟分K同一個量級，不要因為換了dataset就放大風險。


def _month_range(start_ym: str, end_ym: str) -> list[tuple[int, int]]:
    """跟 backfill_history.py::_month_range() 一致：由新到舊，近期資料先補。"""
    periods = pd.period_range(start=start_ym, end=end_ym, freq="M")
    return [(p.year, p.month) for p in periods][::-1]


def _tick_month_days(year: int, month: int) -> list[str]:
    """只借 db/fugle_day 確認哪幾天有開盤（不取股票清單——tick backfill 的
    股票清單是外部傳入的固定名單，不是這裡動態算的，見
    finmind_api.py::_month_universe() 的股票清單版本，這裡只要交易日）。"""
    path = _ROOT / f"db/fugle_day/{year}_{month:02d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，無法確定 {year}-{month:02d} 的交易日")
    df = pd.read_parquet(path, columns=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return sorted(d.strftime("%Y-%m-%d") for d in df["date"].dt.date.unique())


async def backfill_tick_month(
    year: int,
    month: int,
    stocks: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    flush_every: int = _FLUSH_EVERY,
):
    """補齊 db/tick/{year}_{month}.parquet 缺的 (股票, 交易日) 組合。結構跟
    finmind_api.backfill_month() 幾乎一樣（stop_event/Semaphore/每flush_every筆
    存檔一次/fatal error立刻停止新請求），差別只在股票清單用傳入的固定名單、
    交易日用 _tick_month_days()、讀寫呼叫 tick 專用的函式。

    start_date/end_date: 選填，格式 YYYY-MM-DD，只補這個範圍內的交易日（一定要
    落在 year-month 這個月份內）。留空就補整個月。
    """
    days = _tick_month_days(year, month)
    if start_date:
        days = [d for d in days if d >= start_date]
    if end_date:
        days = [d for d in days if d <= end_date]
    existing = existing_tick_pairs(year, month)
    pairs = [(sid, d) for sid in stocks for d in days if (sid, d) not in existing]
    print(
        f"{year}-{month:02d}：目標 {len(stocks)} 支 x {len(days)} 天 = "
        f"{len(stocks) * len(days)} 組，已有 {len(existing)} 組，還要補 {len(pairs)} 組"
    )
    if not pairs:
        print("已經補齊，不用下載")
        return

    sem = asyncio.Semaphore(_CONCURRENCY)
    done = 0
    got_data = 0
    confirmed_empty = 0
    failed: list[tuple[str, str, str]] = []
    buffer: list[pd.DataFrame] = []
    empty_buffer: list[tuple[str, str]] = []
    stop_event = asyncio.Event()
    fatal_error_holder: list[FatalAPIError] = []

    async def _one(session: aiohttp.ClientSession, sid: str, day: str):
        nonlocal done, got_data, confirmed_empty
        if stop_event.is_set():
            return
        async with sem:
            if stop_event.is_set():
                return
            try:
                df = await fetch_tick_day(session, sid, day)
            except FatalAPIError as e:
                stop_event.set()
                fatal_error_holder.append(e)
                return
            except Exception as e:
                failed.append((sid, day, str(e)))
                return
            done += 1
            if done % 50 == 0 or done == len(pairs):
                print(f"  [{done}/{len(pairs)}] 進度...")
            if not df.empty:
                got_data += 1
                buffer.append(df)
            else:
                confirmed_empty += 1
                empty_buffer.append((sid, day))

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(pairs), flush_every):
            if stop_event.is_set():
                break
            batch = pairs[i : i + flush_every]
            await asyncio.gather(*(_one(session, sid, day) for sid, day in batch))
            if buffer:
                save_tick(pd.concat(buffer, ignore_index=True), year, month)
                print(f"  已寫入 {len(buffer)} 組結果")
                buffer.clear()
            if empty_buffer:
                save_empty_tick_pairs(empty_buffer, year, month)
                print(f"  記錄 {len(empty_buffer)} 組確認無資料，之後不會重複請求")
                empty_buffer.clear()
            if stop_event.is_set():
                break

    if fatal_error_holder:
        e = fatal_error_holder[0]
        print(f"  ⚠️ {e}")
        print("  整批任務停止，處理好問題後重跑這支腳本會自動從中斷處繼續")
        raise e

    print(
        f"{year}-{month:02d} 補齊完成：{got_data} 組有資料、"
        f"{confirmed_empty} 組確認無資料（已記錄跳過）、失敗 {len(failed)} 組"
    )
    if failed:
        print("失敗清單（沒有寫入db/tick，重跑這支腳本會自動重試）：")
        for sid, day, err in failed[:20]:
            print(f"  {sid} {day}: {err}")
        if len(failed) > 20:
            print(f"  ...還有 {len(failed) - 20} 筆")


async def backfill_tick_history(
    start_ym: str = _DEFAULT_START,
    end_ym: str = _DEFAULT_END,
    stocks: list[str] | None = None,
):
    """跟 backfill_history.backfill_history() 同構：逐月呼叫
    backfill_tick_month()。FileNotFoundError 跳過該月繼續，FatalAPIError
    整批停止並往上拋（run_forever() 會處理暫停/恢復），其他 Exception
    跳過該月繼續。stocks=None 時從 tick_universe.load_tick_universe() 讀
    固定的801檔清單。"""
    if stocks is None:
        stocks = load_tick_universe()
    months = _month_range(start_ym, end_ym)
    print(f"Tick歷史補齊範圍：{start_ym} ~ {end_ym}，共 {len(months)} 個月，{len(stocks)} 支股票")
    print("=" * 60)

    for i, (year, month) in enumerate(months, 1):
        print(f"\n[{i}/{len(months)}] {year}-{month:02d}")
        try:
            await backfill_tick_month(year, month, stocks)
        except FileNotFoundError as e:
            print(f"  跳過：{e}")
        except FatalAPIError as e:
            print(f"\n⚠️ {e}")
            print(f"在 {year}-{month:02d} 停止，處理好問題後重跑這支腳本會自動接著補")
            raise
        except Exception as e:
            print(f"  {year}-{month:02d} 失敗，跳到下一個月，稍後重跑這支腳本會重試: {e}")

    print("\n" + "=" * 60)
    print("全部月份跑完")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        _start, _end = sys.argv[1], sys.argv[2]
    else:
        _start, _end = _DEFAULT_START, _DEFAULT_END
    asyncio.run(backfill_tick_history(_start, _end))
