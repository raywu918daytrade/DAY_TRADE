# open_drive_0050_stats

0050 Open-Drive 敘述統計（Phase 0+1）。結論見 [`strategy/breakout_retest_ml/README.md`](../../strategy/breakout_retest_ml/README.md)。

## 定義摘要

| 項目 | 規則 |
|--|--|
| 母體 | `tick_universe` 399 + 0050 基準 |
| Gap | `(open/prev_close)-1`；強度 le1 / 1to2 / 2to5 / gt5；方向 up/down/flat |
| Ret5 | m5@09:05 `(close-open)/open` |
| 進場 | 09:05 close |
| 出場 | 09:15 / 09:30 / 09:45 / 10:00 |
| 多空 | long=`r`；short=`-r`；\|mean\|≥0.45% 標 opportunity |
| 勝率 | win0＝方向>0；win3＝方向且 \|r\|≥3% |
| Phase 1.5 | `0050 gap×ret5 × 個股 gap×ret5`（81 格），`--min_n` 過濾 |
| 0050 分組 | 固定% 與 σ（`|z|>1.5`，gap／ret5 各 σ20）並行；個股仍固定% |
| ATR | 個股主濾網：09:05 m1 `atr5 ≥ p99=0.01095`；另掃 0/0.006/0.008 對照 |

## 跑法

```bash
python -m strategy_test.open_drive_0050_stats.verify \
    --start_date 2026-06-01 --end_date 2026-07-31 --min_n 10

python -m strategy_test.open_drive_0050_stats.verify \
    --start_date 2024-01-01 --end_date 2026-07-31 --min_n 30
```
