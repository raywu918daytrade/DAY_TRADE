---
title: 分K / 日K 特徵說明
updated_at: 2026-07-06
---

# 分K / 日K 特徵說明

`make_features()`（`strategy/date_trade_model.py`）產出的特徵欄位，供 [lightgbm_barrier_model](../model/lightgbm_barrier_model.md) 訓練與推論使用。所有分K特徵皆以「當日」為分組單位（`groupby(stock_id, day_date)`），不跨日計算。

訓練資料仍從 1 分鐘 OHLCV 載入，模型會在每一根 1 分鐘樣本上，同時計算 1/3/5 分鐘K特徵。3 分鐘K與 5 分鐘K是由當下與過去的 1 分鐘K滾動聚合產生，不使用未來資料。

若設定 `USE_KLINE_SEQUENCE_FEATURES=1`，每分鐘推論時模型也會看到近期 K 線序列。以 09:15 推論為例，`lag_0` 是截至 09:15 的最新一根，接著依時間週期往前跳：

- 1 分鐘K：`k1_*_lag_0`...`k1_*_lag_14`，約 09:01~09:15 的最近 15 根
- 3 分鐘K：`k3_*_lag_0`...`k3_*_lag_4`，約 09:13~09:15、09:10~09:12、... 的最近 5 根
- 5 分鐘K：`k5_*_lag_0`...`k5_*_lag_2`，約 09:11~09:15、09:06~09:10、09:01~09:05 的最近 3 根

## 分K 報酬率

- `ret_1` / `ret_3` / `ret_5` / `ret_10` / `ret_15`：過去 N 根分K 的報酬率，最長 15 根，因此 09:16 起才有效值。

## 多週期分K特徵

- `tf3_ret_1` / `tf3_ret_2` / `tf3_ret_3`：3 分鐘K的 1/2/3 根報酬率，也就是約 3/6/9 分鐘動能
- `tf5_ret_1` / `tf5_ret_2` / `tf5_ret_3`：5 分鐘K的 1/2/3 根報酬率，也就是約 5/10/15 分鐘動能
- `tf3_vol_ratio` / `tf5_vol_ratio`：滾動 3/5 分鐘成交量相對近期均量
- `tf3_close_pos` / `tf5_close_pos`：現價在滾動 3/5 分鐘 High-Low 區間的位置
- `tf3_range_pct` / `tf5_range_pct`：滾動 3/5 分鐘振幅相對該段開盤價
- `tf3_reversal` / `tf5_reversal`：現價相對滾動 3/5 分鐘低點的反彈幅度

## 過去 K 線序列特徵

- `k1_ret_lag_*` / `k3_ret_lag_*` / `k5_ret_lag_*`：該根 K 線相對前一根同週期 K 線的收盤報酬率
- `k1_body_lag_*` / `k3_body_lag_*` / `k5_body_lag_*`：該根 K 線收盤相對開盤的幅度
- `k1_range_lag_*` / `k3_range_lag_*` / `k5_range_lag_*`：該根 K 線 high-low 振幅相對開盤價
- `k1_close_pos_lag_*` / `k3_close_pos_lag_*` / `k5_close_pos_lag_*`：收盤在該根 K 線 high-low 區間的位置
- `k1_vol_ratio_lag_*` / `k3_vol_ratio_lag_*` / `k5_vol_ratio_lag_*`：該根 K 線成交量相對近期同週期量的比值

## 量能與 K 線形態

- `vol_ratio`：當根成交量 / 當日 15 分鐘移動平均量
- `close_pos`：收盤價在當根 High-Low 區間的相對位置

## 時間特徵

- `hour` / `minute`：分K 對應的時、分（讓模型學到盤中不同時段的行為差異）

## 當日盤中累積特徵

- `price_vs_open`：現價相對今日開盤的漲跌幅
- `vwap_dev`：現價偏離「當日累積 VWAP」的幅度
- `high_pos_today`：現價在「今日目前最高/最低」區間的相對位置
- `reversal_3` / `reversal_5` / `reversal_10`（破底翻）：現價相對「近 N 根最低點」的反彈幅度，0 代表仍在低點

## 日K 背景特徵（前一日收盤前，避免未來資料）

- `gap`：今日開盤相對昨日收盤的跳空幅度
- `prev_ret`：前一日日報酬率
- `prev_vol_ratio`：前一日成交量 / 20 日均量
- `pos_20d`：收盤價在近 20 日高低區間的相對位置
- `day_ret_1`...`day_ret_10`：過去 10 天逐日報酬率（lag 1~10）
- `day_vol_1`...`day_vol_10`：過去 10 天逐日量比（相對 5 日均量）

## 注意事項

- warm-up 限制：`vol_ratio` 需要 15 根、`reversal_10` 需要 10 根，開盤初期（09:01~09:15）部分特徵會是 NaN
- 預設新模型會使用 1/3/5 分鐘K特徵；設定 `USE_KLINE_SEQUENCE_FEATURES=1` 後才會額外使用過去 K 線序列特徵
- 尚未重訓前，舊模型仍可用原本訓練時的欄位推論
- 所有 rolling / cumsum 計算皆確保只用「當下與過去」資料，不會洩漏未來 bar
