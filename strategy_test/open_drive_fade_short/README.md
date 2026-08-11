# open_drive_fade_short（gap-only）

0050／個股只看開盤 gap，**不管 ret5**；再加 atr5 p99 → m5 close 做空 TB。

## 鎖定

| 條件 | 規則 |
|--|--|
| 0050 | gap >1%（不看 ret5） |
| 個股 | gap >1%（不看 ret5） |
| ATR | 進場當根 m1 atr5 ≥ p99（0.01095） |
| 進場 | 預設僅 09:05；`--entry_until 13:00` 則 09:05～13:00 每根 m5 可進（同日同股可多次） |
| 出場 | TB ±3%／最多持有 11 根 m5（晚進場若當日棒不足則以最後一根 time exit） |

## TB 結果（tick_universe ~400）

| 版本 | 區間 | 0050 日 | n | 止盈 | 震盪 | 止損 | mean |
|--|--|--|--|--|--|--|--|
| 舊（gap+ret5） | 放大 | 6 | 112 | 50.0% | 25.9% | 24.1% | +1.02% |
| **gap-only** | 2026-06～07 | 10 | 403 | 27.5% | 46.9% | 25.6% | +0.12% |
| **gap-only** | 2024-01～2026-07 | **119** | **2,689** | 36.0% | 37.8% | 26.2% | +0.31% |

分年（gap-only 放大）：2024 n=410／38日；2025 n=509／36日；2026 n=1770／45日。

## TB 結果（stock_universe_2000，資料至 2022）

母體改 `db/tickers/stock_universe_2000.parquet`（~1877 股）；分K目前主要到 2022，故回測 **2020-01～2022-12**。

| 進場窗 | 交易日 | n | 止盈 | 震盪 | 止損 | mean |
|--|--|--|--|--|--|--|
| 僅 09:05 | 66 | 438 | 27.6% | 46.8% | 25.6% | **+0.03%** |
| 09:05～13:00 每根 m5 | 68 | 1,418 | 23.0% | 56.6% | 20.5% | **+0.04%** |

交易至 13:00：同日進場 mean≈21、max=141；分年 mean 2020 +0.00%／2021 −0.11%／2022 +0.16%。

（先前誤做成「09:05 進場後持有至 13:00」的數字已作廢，不採用。）

**結論**：
- tick_universe gap-only：頻率夠但 mean +0.31% ＜摩擦 ~0.45%，不宜主力。
- 擴大 2000 母體後僅 09:05 進場 mean≈0；拉長**交易窗**到 13:00 雖把 n 拉到 ~1.4k，mean 仍≈0。
- **擴大池／延長交易窗都救不起**；回到 `prev_bear_m5_short` 較務實。

## 跑法

```bash
# tick_universe（預設，僅 09:05）
python -m strategy_test.open_drive_fade_short.verify \
    --start_date 2024-01-01 --end_date 2026-07-31

# stock_universe_2000（資料區間建議 ≤2022）
python -m strategy_test.open_drive_fade_short.verify \
    --start_date 2020-01-01 --end_date 2022-12-31 --use_2000

# 交易窗拉到 13:00（每根 m5 可進）
python -m strategy_test.open_drive_fade_short.verify \
    --start_date 2020-01-01 --end_date 2022-12-31 --use_2000 --entry_until 13:00
```
