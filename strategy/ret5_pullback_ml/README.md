# ret5_pullback_ml（做多）

早盤 ret5 強過濾當事件起點 → 純規則 TB 驗證，或 + AVWAP／結構特徵 → LightGBM。

原 `ret5_pullback_reversal` 已并入本目錄（`verify.py` 與 `dataset` 共用事件邏輯）。

## 事件（硬過濾）

| 條件 | 規則 |
|--|--|
| 母體 | 預設 `stock_universe_2000`；可 `--use_tick_universe` |
| 上漲 | ret5 vs 昨收 ≥ 3%，且 m5@09:05 紅K |
| 參考低點 | 首根 m5（09:05）low = `m5_1_low` |
| 拉回 | m5 陰線且 low &gt; m5_1_low，收盤 &lt; 09:30 |
| 進場 | 之後 **5 根 m1**：陽線、量&gt;前1、close &gt; 該 m5 high（&lt; 10:00） |
| 標籤 | 做多 TB ±3%；最多 30 分（6 根 m5）；同根先 TP |

- **純規則 verify**：另加 `atr5 ≥ p99` 硬刪
- **ML train**：`atr5` 改特徵、不當硬刪

## AVWAP（ML）

自 `m5_down_ts` **下一根** m1 累積至 `entry_ts`；特徵含 `close_vs_avwap`、`avwap_z`、session VWAP 對照。

## 結果摘要

### 純規則（tick_universe，2026-01～07，+atr5 p99）

| 條件 | 日 | n | 止盈 | 持平 | 止損 | mean |
|--|--|--|--|--|--|--|
| m1 帶量 + 突破 m5 high | 68 | 124 | 12.1% | 41.1% | 46.8% | **−0.87%** |

### ML（stock_universe_2000，2024-01～2026-07）

漏斗：ret5 32.4k → 事件 5.8k。Test（末 90 日 n=952）：全交易 mean −0.47%；p_tp≥0.4 mean −0.41%。

## 跑法

```bash
# 純規則（含 atr5≥p99）
python -m strategy.ret5_pullback_ml.verify \
    --start_date 2026-01-01 --end_date 2026-07-31 --use_tick_universe

# 建事件 + 訓練
python -m strategy.ret5_pullback_ml.train train \
    --start_date 2024-01-01 --end_date 2026-07-31 --test_days 90

python -m strategy.ret5_pullback_ml.train confidence --use_cache \
    --start_date 2024-01-01 --end_date 2026-07-31
python -m strategy.ret5_pullback_ml.run_backtest --use_cache \
    --start_date 2024-01-01 --end_date 2026-07-31 --threshold 0.4
```
