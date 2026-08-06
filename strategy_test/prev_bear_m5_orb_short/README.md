# prev_bear_m5_orb_short

日線同 [`prev_bear_m5_short`](../prev_bear_m5_short/)，但首 m5 **不管漲跌**，當 ORB；之後收盤跌破 OR.low 做空。

## 鎖定定義

| 條件 | 規則 |
|--|--|
| 日線 | 昨陰≥5% + 今開高≥2% + 0050 開低 |
| ORB | `m5_std@09:05` 的 high/low（不論陰陽） |
| 進場 | 之後第一根 m5 `close < OR.low`（≤09:30） |
| 出場 | 做空 TB ±3% / 30 分 |
| 勝率 | 止盈筆數 / n |

## 結果

| 區間 | n | 勝率 | 震盪 | 止損 | mean |
|--|--|--|--|--|--|
| 2026-06～07 | 18 | 38.9% | 38.9% | 22.2% | +0.42% |
| 2024-01～2026-07 | 49 | 16.3% | 63.3% | 20.4% | +0.14% |

對照 `prev_bear_m5_short`（同區間勝率 47.4% / 29.2%）：ORB 跌破**未優於**首 m5 陰線進場。

分年（放大）：2024 n=12 勝率 0%；2025 n=4 勝率 0%；2026 n=33 勝率 24.2%。

OR 棒陰陽拆分（放大）：OR 陰 n=32 勝率 21.9%；OR 非陰 n=17 勝率 5.9%。

## 跑法

```bash
python -m strategy_test.prev_bear_m5_orb_short.verify \
    --start_date 2026-06-01 --end_date 2026-07-31

python -m strategy_test.prev_bear_m5_orb_short.verify \
    --start_date 2024-01-01 --end_date 2026-07-31
```
