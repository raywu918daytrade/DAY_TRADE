# breakout_retest_ml（突破壓力回測 + POC + Tick / LightGBM）

日 K `breakout_retest` 且支撐 ≈ POC 的候選，隔日 09:10–10:00 以 **M1 陽線實體 K + 有方向 Tick 大量買進** 硬觸發，再以 LightGBM 三分類過濾做多。持有可超過 10:00（30 根 M1）。

硬過濾上游已物化到 `db/`，調門檻／加條件應讀表 filter，不要重掃日 K／重讀 tick。

## 硬過濾（5 階）

1. **breakout_retest** — 日 K 突破後拉回（score≥60）→ `db/breakout_retest_day`
2. **POC ≈ 支撐** — `poc_confluence`（≤2%）
3. **隔日有 M1** — 下一交易日 09:10–10:00
4. **M1 陽線實體 K** — body≥50%、上影線≤35% → 與 Tick 特徵一併寫入 `db/breakout_retest_trigger`（M1 用 `load_m1_adjusted`，與日K/POC 同基準）
5. **Tick 大量買進** — 大單買比≥10% 且 CVD>0（對 trigger 表 filter，不重讀 tick）

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
| `tick_large_buy_ratio` / `cvd_30s_delta` | 有方向 Tick |

標籤與 prepared cache 仍在 `cache/`（訓練產物，不進 `db/`）。
