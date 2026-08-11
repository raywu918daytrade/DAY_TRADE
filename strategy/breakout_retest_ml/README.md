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

### 2026-08-07 — open_drive_0050_stats（敘述統計 Phase0+1）

路徑：`strategy_test/open_drive_0050_stats/`

**定義**：day gap + m5@09:05 ret5；進場 09:05 close；出場 09:15/30/45/10:00；每格同時報 long=`r` / short=`-r`；摩擦門檻 |mean|≥0.45%；σ20 僅輔標。母體 tick_universe 399。

勝率兩種：
- **win0**：方向為正（long `r>0`；short `r<0`）
- **win3**：方向且 |r|≥3%（long `r≥3%`；short `r≤-3%`）

#### Phase 0（0050 alone）

自身持有各出場 mean 皆 ≈0（過不了摩擦）。Gap 大多落在 |gap|≤1%（放大 424/616 日）。

#### Phase 1（個股）

| 指標 | 小樣 2026-06～07 | 放大 2024-01～2026-07 |
|--|--|--|
| n | 15,899 | 221,128 |
| corr(gap_stock, gap_0050) | 0.69 | 0.59 |
| gap 同向率 | 71.6% | 62.0% |
| corr(ret5) / 同向 | 0.13 / 48% | 0.11 / 44% |
| 無條件持有 | 略偏 short、mean≪0.45% | 同左 |

放大後過摩擦的 **穩定機會（皆做空）**：

| 條件 | 出場 | n | short mean | win0 | win3 |
|--|--|--|--|--|--|
| gap=up × ret5=up | 09:45 | 2,308 | +0.81% | 64.9% | 11.3% |
| gap=up × ret5=up | 10:00 | 2,308 | +1.08% | 67.8% | 16.5% |
| gap=down × ret5=down | 09:30 | 2,283 | +0.73% | 59.8% | 13.1% |
| gap=down × ret5=down | 09:45 | 2,283 | +0.61% | 61.5% | 10.9% |
| gap=down × ret5=down | 10:00 | 2,283 | +0.61% | 63.2% | 11.3% |
| gap=flat × ret5=down | 09:30 | 5,408 | +0.64% | 58.8% | 9.0% |

解讀（僅 0050 分組）：
- **開高且前5分續強 → 偏 fade 做空**（持有到 09:45～10:00 較明顯）。
- **開低且前5分續弱 → 偏續跌做空**（09:30 起）；小樣曾誤顯 long，放大後翻成 short。
- **平盤 + 前5分轉弱 → 做空**。
- 不做個股分組時，對稱做多幾乎不過摩擦。

#### Phase 1.5（0050 × 個股 四維交叉）

分組：`0050 gap × 0050 ret5 × 個股 gap × 個股 ret5`（各 up/down/flat，最多 81 格）× 四出場；只報過摩擦且 min_n 足夠者。

| 區間 | 有效格子 | 過摩擦(格×出場) | long | short |
|--|--|--|--|--|
| 小樣（min_n=10） | 57/81 | 123 | 68 | 55 |
| 放大（min_n=30） | 67/81 | 63 | 19 | 44 |

放大後仍以 **做空為主**，但加入個股維度後出現少數站得住的做多。

**Top short（放大，摘錄）**

| 0050 | 個股 | 出場 | n | mean | win0 | win3 |
|--|--|--|--|--|--|--|
| down/down | up/down | 09:30 | 33 | +2.22% | 72.7% | 36.4% |
| up/up | up/down | 10:00 | 505 | +1.68% | 77.2% | 26.3% |
| down/down | down/down | 10:00 | 710 | +1.44% | 78.0% | 22.0% |
| flat/down | down/up | 09:30 | 107 | +1.43% | 72.9% | 22.4% |
| flat/down | down/down | 09:45 | 226 | +1.30% | 71.2% | 21.7% |
| up/up | flat/down | 10:00 | 282 | +1.27% | 71.6% | 17.7% |

模式：大盤開高續強時，**個股開高但前5分轉弱** → fade 做空（n 大、穩）；大盤開低續弱時，**個股也開低續弱** → 續跌做空。win0 多在 70%+，但 **win3（≥3%）多只 18～26%**——方向對的多，打到 3% 的少。

**Top long（放大，摘錄）**

| 0050 | 個股 | 出場 | n | mean | win0 | win3 |
|--|--|--|--|--|--|--|
| up/down | up/flat | 10:00 | 89 | +1.16% | 69.7% | 15.7% |
| up/down | up/down | 09:45 | 78 | +0.98% | 73.1% | 14.1% |
| down/up | down/down | 09:30 | 113 | +0.76% | 66.4% | 6.2% |
| down/flat | down/down | 10:00 | 2,652 | +0.69% | 59.4% | 12.4% |

模式：大盤開高但前5分轉弱、個股仍開高 → 偏做多；long 的 win3 更低（多 <16%）。

結論：交叉後仍 **short 主導**；最值得跟的是 `0050 up/up × stock up/down → short@10:00`（n=505，win0 77%／win3 26%）與 `0050 down/down × stock down/down → short`（n=710，win0 78%／win3 22%）。若策略要對齊 ±3% TB，應以 **win3** 為主，不要被 win0 高估。

#### Phase 1.5b — 0050 改用 σ 分組（|z|>1.5）

並行：`z_gap = gap/σ20`、`z_ret5 = ret5/σ20`；`|z|>1.5` → up/down，其餘 flat。個股仍固定 %。

| 項目 | 固定% | σ\|z\|>1.5 |
|--|--|--|
| 0050 gap up/down/flat 日數 | 119 / 73 / 424 | 55 / 41 / 520 |
| 標籤一致率 gap / ret5 | — | 82.8% / 91.1% |
| 0050-only 過摩擦 | 8 格（含 up/up fade） | **僅 2 格**（down/down short） |
| 交叉過摩擦 long/short | 19 / 44 | 13 / 41 |

σ 下 0050-only 的 **up/up fade 消失**（極端開高日變少）；殘留 `down/down → short`（n≈2,198，mean≈+0.6～0.7%，win3≈9～10%）。

交叉 Top short（σ，摘錄）：

| 0050(σ) | 個股(%) | 出場 | n | mean | win0 | win3 |
|--|--|--|--|--|--|--|
| down/down | down/down | 09:45 | 600 | +1.62% | 75.7% | 26.8% |
| flat/down | down/down | 09:30 | 510 | +1.62% | 77.6% | 26.1% |
| down/down | down/down | 10:00 | 600 | +1.56% | 75.5% | 24.7% |
| up/flat | down/up | 09:45 | 93 | +2.62% | 69.9% | 35.5% |

解讀：σ 較嚴 → 大盤「極端開高續強」樣本變薄，固定% 的 fade short 不易再現；**續跌 short（兩邊 down/down）在兩種分組都穩**，且 σ 下 win3 略升（~25～27%）。Long 在 σ 下更弱。實務可先鎖固定% 的 fade + 兩邊 down/down；σ 當稳健對照，不是取代。

#### Phase ATR — 09:05 atr5 絕對門檻（放大）

進場可觀測：當日 09:00–09:05 m1 的 `atr5`（同 mkt：TR5／day_open）。掃 0 / 0.006 / 0.008 / **0.01095（mkt p99）**。

**Fade：`0050 up/up × stock up/down` @10:00（最受惠）**

| atr5 | n | short mean | win0 | win3 |
|--|--|--|--|--|
| ≥0 | 505 | +1.68% | 77.2% | 26.3% |
| ≥0.006 | 309 | +1.87% | 79.9% | 32.4% |
| ≥0.008 | 210 | +2.03% | 79.5% | 33.8% |
| ≥0.01095 | 112 | +2.09% | 75.9% | **40.2%** |

@09:45 同條件：win3 17.0% → **32.1%**（atr≥0.01095）。

**續跌：`0050 down/down × stock down/down` @10:00**

| atr5 | n | short mean | win0 | win3 |
|--|--|--|--|--|
| ≥0 | 710 | +1.44% | 78.0% | 22.0% |
| ≥0.01095 | 199 | +1.39% | 78.4% | 23.6% |

ATR 對續跌 **幾乎沒抬 win3**（跟 prev_bear 丟 atr5 的經驗一致）；對 **fade short 則明顯**（win3 +14pt）。

**0050-only up/up @10:00**：win3 16.5% → 33.8%（n 2308→281）。無個股維度時 ATR 也有用，但交叉再加個股 up/down 更好。

結論：open-drive 若要抬 **win3**，優先 `固定% fade + atr5≥0.01095 + 持有到 09:45/10:00`；續跌格不必硬加 ATR。

#### 個股濾網：m1 atr5 ≥ p99（0.01095）

放大區間、個股一律先過 **atr5 p99** 後，候選格（short）：

| 條件 | 出場 | n | mean | win0 | win3 | 備註 |
|--|--|--|--|--|--|--|
| 0050 up/up × stock up/down（fade） | 10:00 | 112 | +2.09% | 75.9% | **40.2%** | n 夠、首選 |
| 同上 | 09:45 | 112 | +1.58% | 72.3% | 32.1% | |
| 0050 σ up/flat × stock down/up | 09:45 | 42 | +2.38% | 64.3% | **40.5%** | win3 高、n 偏小 |
| 同上 | 10:00 | 42 | +2.24% | 69.0% | 38.1% | |
| 0050 down/down × stock up/down | 09:30 | **18** | +3.00% | 72.2% | 55.6% | 好看但 n 太小 |
| 0050 down/down × stock down/down | 10:00 | 199 | +1.39% | 78.4% | 23.6% | win0 高、win3 普通 |

排序建議：**fade + p99 @10:00**（穩）＞ σ up/flat×down/up + p99（可當第二候選）＞ down/down×up/down + p99（先不當鎖定，n=18）。

```bash
python -m strategy_test.open_drive_0050_stats.verify \
    --start_date 2026-06-01 --end_date 2026-07-31 --min_n 10
python -m strategy_test.open_drive_0050_stats.verify \
    --start_date 2024-01-01 --end_date 2026-07-31 --min_n 30
```

### 2026-08-07 — open_drive_fade_short（TB）

路徑：`strategy_test/open_drive_fade_short/`

#### 舊版（gap+ret5，太稀）

0050 gap>1% + ret5>0.5%；個股 gap>1% + ret5<−0.5%；atr5≥p99。放大 n=112、僅 **6** 個 0050 日；mean +1.02% 但不可日常出手。

#### 現行 gap-only（不管 ret5，tick_universe）

0050／個股只看 gap>1% + atr5≥p99；09:05 做空；TB ±3%／至 10:00。

| 區間 | 0050 日 | n | 止盈 | 震盪 | 止損 | mean |
|--|--|--|--|--|--|--|
| 2026-06～07 | 10 | 403 | 27.5% | 46.9% | 25.6% | +0.12% |
| 2024-01～2026-07 | **119** | **2,689** | 36.0% | 37.8% | 26.2% | +0.31% |

#### 2026-08-08 — stock_universe_2000（資料至 2022）

母體改 ~1877 股；區間 **2020-01～2022-12**（2000 分K目前主要到 2022）。
出場一律 TB ±3%／最多 11 根 m5；變的是**進場窗**（不是持有到 13:00）。

| 進場窗 | 交易日 | n | 止盈 | 震盪 | 止損 | mean |
|--|--|--|--|--|--|--|
| 僅 09:05 | 66 | 438 | 27.6% | 46.8% | 25.6% | +0.03% |
| 09:05～13:00 每根 m5 | 68 | 1,418 | 23.0% | 56.6% | 20.5% | +0.04% |

**結論**：擴大 2000 母體／把交易窗拉到 13:00 後 n 增到 ~1.4k，mean 仍≈0，**救不起**。回到 `prev_bear_m5_short` 較務實。

```bash
python -m strategy_test.open_drive_fade_short.verify \
    --start_date 2024-01-01 --end_date 2026-07-31
python -m strategy_test.open_drive_fade_short.verify \
    --start_date 2020-01-01 --end_date 2022-12-31 --use_2000 --entry_until 13:00
```

### 2026-08-10 — ret5_pullback_reversal（做多 TB）

路徑：`strategy/ret5_pullback_reversal/`

**規則**：ret5 vs 昨收 ≥3% 且 **m5@09:05 紅K** → **m5** 陰線 low &gt; 首根 m5 low → 下一根 m5 陽線實體（body≥50%）做多；訊號 &lt;09:30、進場 &lt;10:00；atr5≥p99；三分類 TB ±3%／最多 30 分。母體 `stock_universe_2000`。

| 區間 | 日 | n | 止盈 | 持平 | 止損 | mean |
|--|--|--|--|--|--|--|
| 2026-01～07（m5 + ret5 紅K） | 92 | 185 | 19.5% | 58.4% | 22.2% | **−0.10%** |
| 2026-01～07（m5，未要求紅K） | 96 | 205 | 17.6% | 61.5% | 21.0% | −0.16% |

**結論**：純規則期望仍接近 0／略負，不宜直接當主力。

```bash
python -m strategy.ret5_pullback_reversal.verify \
    --start_date 2026-01-01 --end_date 2026-07-31
```
