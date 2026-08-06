# breakout_retest_ml（突破壓力回測 + Tick / LightGBM）

日 K `breakout_retest`（前壓力轉支撐）候選，隔日 09:10–10:00 以 **M1 陽線實體 K + 有方向 Tick 大量買進** 硬觸發，再以 LightGBM 三分類過濾做多。持有可超過 10:00（30 根 M1）。POC 欄位可作特徵，**不再當硬門檻**。

硬過濾上游已物化到 `db/`，調門檻／加條件應讀表 filter，不要重掃日 K／重讀 tick。

## 硬過濾（5 階）

1. **breakout_retest** — 日 K 突破後拉回（score≥60）→ `db/breakout_retest_day`
2. **有支撐線** — 型態成立（`resistance_price`／前壓力轉支撐；不再要求 `poc_confluence`）
3. **隔日有 M1** — 下一交易日 09:10–10:00
4. **M1 陽線實體 K** — body≥50%、上影線≤35% → 與 Tick 特徵一併寫入 `db/breakout_retest_trigger`（M1 用 `load_pattern_m1`，與日K/POC 同基準）
5. **Tick 大單方向** — 大單買比≥10% 且 **大買 > 大賣**，並 CVD>0（對 trigger 表 filter，不重讀 tick）

標籤／出場：Triple Barrier **±3%／最多 30 分**（先觸先平；未觸為震盪）。

## 物化 / 日更

```bash
# Layer1 日K候選（約 1 分鐘）
python -m data.build_breakout_retest_day --force

# Layer2 盤中 M1實體K + Tick 特徵（首次慢；之後增量）
python -m data.build_breakout_retest_trigger

# 或整條日更（含上述兩步）
python scripts/update_daily.py
```

## 漏斗 / 訓練

```bash
# 讀 db 印硬過濾漏斗（秒級；缺表會自動建）
python -m strategy.breakout_retest_ml.experiments.verify_funnel
python -m strategy.breakout_retest_ml.experiments.verify_funnel --labels --save
python -m strategy.breakout_retest_ml.experiments.verify_funnel --compare-tick  # A無tick vs B有tick
python -m strategy.breakout_retest_ml.experiments.verify_funnel --compare-tick --start_date 2026-07-01 --end_date 2026-07-31
# 滾動5分實體 + M1 ATR5 + 窗內大單；決策窗 09:05～10:00
python -m strategy.breakout_retest_ml.experiments.verify_m5_tick --start_date 2026-05-01 --end_date 2026-07-31 --min_atr5 0

# 訓練（自動讀 trigger 物化表）
python -m strategy.breakout_retest_ml.train train --start_date 2025-07-01

python -m strategy.breakout_retest_ml.experiments.walk_forward --use_cache
python strategy/breakout_retest_ml/run_backtest.py
```

Live：`.env` 加 `strategy.breakout_retest_ml.up.live`（需即時 tick；尚未接入時不過 Tick 硬過濾）。

## 特徵（FEATURES）

| 特徵 | 說明 |
|---|---|
| `pattern_score` / `poc_diff_pct` | 日 K 型態與 POC 距離 |
| `dist_to_poc_pct` / `dist_to_support_pct` | 觸發價相對 POC／支撐 |
| `body_ratio` / 影線比例 / `volume_surge_ratio` | M1 K 線 |
| `tick_large_buy/sell/net_ratio` / `cvd_30s_delta` | 大單買賣對抗 + CVD |

標籤與 prepared cache 仍在 `cache/`（訓練產物，不進 `db/`）。

## 策略驗證紀錄

慣例：每個驗證一個資料夾 `strategy_test/<name>/`；此處只記 plan／小樣結論。

### 2026-08-06 — prev_bear_m5_short（做空，鎖定）

路徑：`strategy_test/prev_bear_m5_short/`

**鎖定**：昨陰≥5% + 今開高≥2% + 0050 開低 + 首 m5@09:05 跌；做空 TB ±3%/30 分；**不用 atr5**。

| 區間 | n | 止盈 | 震盪 | 止損 | mean |
|--|--|--|--|--|--|
| 2026-06～07（小樣最佳） | 19 | 47.4% | 42.1% | 10.5% | +1.21% |
| 2024-01～2026-07 | 48 | 29.2% | 47.9% | 22.9% | +0.41% |

```bash
python -m strategy_test.prev_bear_m5_short.verify \
    --start_date 2026-06-01 --end_date 2026-07-31
```

### 2026-08-06 — prev_bear_m5_long（同濾網做多對照）

路徑：`strategy_test/prev_bear_m5_long/`

濾網同 short，進場改做多。小樣 / 放大 mean 分別為 **−1.21% / −0.41%**（約為 short 的符號翻轉）→ 支持 short 方向性。

### 2026-08-06 — prev_bear_m5_orb_short（ORB 跌破，未優於 short）

路徑：`strategy_test/prev_bear_m5_orb_short/`

日線同 short；首 m5@09:05 當 OR（不管陰陽）；之後 ≤09:30 第一根 `close < OR.low` 做空；TB ±3%/30 分。主看勝率。

| 區間 | n | 勝率 | 震盪 | 止損 | mean |
|--|--|--|--|--|--|
| 2026-06～07 | 18 | 38.9% | 38.9% | 22.2% | +0.42% |
| 2024-01～2026-07 | 49 | 16.3% | 63.3% | 20.4% | +0.14% |

對照 short 勝率 47.4% / 29.2% → ORB 跌破較差，維持首 m5 陰線進場。

```bash
python -m strategy_test.prev_bear_m5_orb_short.verify \
    --start_date 2026-06-01 --end_date 2026-07-31
```
