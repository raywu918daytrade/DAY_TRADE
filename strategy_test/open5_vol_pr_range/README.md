# open5_vol_pr_range

09:00–09:05 成交量相對**自己過去同段**的分位（PR），當天高低幅 `(high−low)/open` 會不會比較大。

## 定義摘要

| 項目 | 規則 |
|--|--|
| 母體 | `tick_universe` ~400 |
| vol5 | `m5_std` 09:05 那根 volume（拆股還原，`data.query`） |
| PR | 當天 vol5 在過去 `lookback` 日（不含當天）的 empirical CDF；少於 `min_hist` 日不算 |
| rng | 當日 `(high−low)/open`（含前 5 分鐘已走出的高低） |
| 分桶 | lt50 / 50to80 / 80to90 / ge90（另列 ge95） |

## 跑法

```bash
python -m strategy_test.open5_vol_pr_range.verify \
    --start_date 2024-01-01 --end_date 2026-08-14
```
