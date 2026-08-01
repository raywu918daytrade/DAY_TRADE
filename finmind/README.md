# finmind/ 說明

FinMind 是「歷史回補」用的資料源，解決 Fugle/富邦「近30日」硬限制碰不到的
更早期間。**日常自動排程（`scripts/update_daily.py`）完全不會呼叫這個資料夾
裡任何一支腳本**，全部都是手動、需要時才跑的一次性/背景工具。

股票母體固定是 `tick_universe.py` 算出來的400支，**所有CLI/函式現在預設就是
這400支，不用另外加旗標**——真的要補全市場才需要明確帶 `--all` 選擇退出。

## 結構（m1 跟 tick 對稱，各兩層）

| | 核心API層 | 通用工具（自己帶年月範圍） | 固定範圍薄wrapper（直接執行） |
|---|---|---|---|
| **分K（M1）** | `m1_api.py` | `backfill_m1_history.py` | `backfill_m1.py` |
| **Tick** | `tick_api.py` | `backfill_tick_history.py` | `backfill_tick.py` |

薄wrapper（`backfill_m1.py`/`backfill_tick.py`）固定用400支+固定範圍
（2025-08~2026-07），不用想任何參數，複製指令貼到終端機就能跑，給「不想
每次都要打年月」的日常場景用。通用工具（`backfill_m1_history.py`/
`backfill_tick_history.py`）自己指定年月範圍，給補特定缺口用。

## 常用指令

**補分K（db/m1），指定月份+日期範圍（母體預設固定400支）：**
```bash
python -m finmind.m1_api 2026 6 --start=2026-06-13 --end=2026-06-30
```

**補分K，跨月份範圍：**
```bash
python -m finmind.backfill_m1_history 2025-08 2026-07
```

**補分K，不用想參數、直接執行（固定2025-08~2026-07，背景長跑）：**
```bash
nohup caffeinate -i python3 -m finmind.backfill_m1 > finmind_m1.log 2>&1 &
```

**補tick（db/tick）：**
```bash
python -m finmind.backfill_tick_history 2025-08 2026-07      # 指定範圍
nohup caffeinate -i python3 -m finmind.backfill_tick > finmind_tick.log 2>&1 &   # 固定範圍，直接執行
```

**真的要補全市場（選擇退出400支固定母體）：**
```bash
python -m finmind.m1_api 2026 6 --all
python -m finmind.backfill_m1_history 2026-01 2026-05 --all 1000   # 全市場前1000支（依成交量）
```

額度不夠一次補完時，都支援 `--max-requests=N`：送滿N筆安全停止，下次重跑
（不用帶這個參數）會自動從中斷處接續，不會重複下載已有的部分。

## 核心限制（決定了每支腳本存在的理由）

- `TaiwanStockKBar`（分K）、`TaiwanStockPriceTick`（逐筆tick）：**沒有還原
  權息**，單次請求只能查一天，要補一段期間得逐日逐股票發request。
- `TaiwanStockPrice`（日K原始）、`TaiwanStockPriceAdj`（日K已還原）：也沒有
  跟 Fugle/富邦100%一致的還原基準（同期比對過，落差 ~0.5~1%），而且
  FinMind 存取權限有時效（見 `fubon/tick_api.py` 檔頭）。**日K已經有更好的
  選擇**（Fugle/富邦自己就能查到很久以前，見下面「除權息調整係數」），所以
  只有分K/tick真的需要靠 FinMind。
- 帳號額度：6000 次/小時（Sponsor方案），`m1_api.py` 的 `_RateLimiter`
  自動節流+額度恢復輪詢，見 `run_forever()`。

## 分K（db/m1）回補

| 檔案 | 用途 |
|---|---|
| `m1_api.py` | 核心邏輯（rate limiter、`_fetch_finmind_day()`、`backfill_month()`）。其他腳本都是它的薄wrapper，不要重複實作 |
| `backfill_m1_history.py` | 通用工具，自己帶年月範圍，日常單月/多月修補用 |
| `backfill_m1.py` | 固定範圍薄wrapper（2025-08~2026-07+固定400支），不用想參數直接執行 |

## Tick（db/tick）回補

| 檔案 | 用途 |
|---|---|
| `tick_api.py` | tick 專屬的 fetch/save（`fetch_tick_day()`/`save_tick()`），複用 `m1_api.py` 的核心請求邏輯 |
| `tick_universe.py` | **固定候選股名單來源**（399支排名+0050強制併入=400支，2026-01~06均量排序算出來，存 `db/tickers/tick_universe.parquet`，含 `name` 欄位）。2026-08-01起，這份名單是 `data/m1_data_loader.py`/`data/day_data_loader.py`/`fubon/subscribe_list.py`/`m1_api.py`/`backfill_m1_history.py` 的**唯一股票母體來源**，不是只給tick用了 |
| `backfill_tick_history.py` | 通用工具，自己帶年月範圍，固定用 `tick_universe.py` 這400支 |
| `backfill_tick.py` | `backfill_tick_history.py` 的薄wrapper，固定範圍 2025-08~2026-07 |

## 除權息調整係數（db/tick_adjust_factor）

日常自動的 `data/build_tick_adjust_factor.py`（在 `scripts/update_daily.py`
裡）拿本機既有的 `db/tick`+`db/fugle_day` 直接算，不打任何API，只能算
`db/tick` 涵蓋範圍內（目前約2025-08起）的係數。

`db/tick` 涵蓋範圍**之前**的除權息事件（例如某支股票的除息/拆股發生在
2025-08以前）要用 **`data/backfill_tick_adjust_factor.py`**（不在
`finmind/` 資料夾，手動執行）：Fugle/富邦自己的日K就能查到很久以前，直接
跟 `db/fugle_day`（已還原）比對原始版本反推係數，不用打FinMind。
```bash
python -m data.backfill_tick_adjust_factor 0050 0056 --start 2016-01-01
```

## `deprecated/`（已停用，保留當備援，不要用）

2026-08-01 從 `finmind/` 移過去，理由是股票母體已經固定成400支，這幾支
腳本設計時針對的範圍（全市場/前1000支/FinMind日K）都不再符合現在的需求：

| 檔案 | 舊用途 |
|---|---|
| `backfill_all.py` | 固定補全部股票~2700支的分K（規模最大，~18天） |
| `backfill_top1000.py` | 固定補前1000支（依月均量排序）的分K |
| `backfill_tick_adjust_factor.py` | 用 FinMind `TaiwanStockPrice`/`TaiwanStockPriceAdj` 反推除權息係數，已被 `data/backfill_tick_adjust_factor.py`（Fugle/富邦版）取代 |

## 目前已知缺口（2026-08-01）

`db/m1` 有一段「灰色地帶」（2026-06-13~07-01）新舊資料混雜，需要重新抓
（見上面「常用指令」的第一個範例，就是在補這段）。目前還缺約1,500組
(股票,日期)，卡在FinMind額度，等額度恢復繼續補即可。
