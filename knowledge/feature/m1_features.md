---
title: 分K / 日K 特徵說明
updated_at: 2026-07-03
---

# 分K / 日K 特徵說明

`make_features()`（`strategy/date_trade_model.py`）產出的特徵欄位，供 [lightgbm_barrier_model](../model/lightgbm_barrier_model.md) 訓練與推論使用。所有分K特徵皆以「當日」為分組單位（`groupby(stock_id, day_date)`），不跨日計算。

## 分K 報酬率

- `ret_1` / `ret_3` / `ret_5` / `ret_10` / `ret_15`：過去 N 根分K 的報酬率，最長 15 根，因此 09:16 起才有效值。

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
- 所有 rolling / cumsum 計算皆確保只用「當下與過去」資料，不會洩漏未來 bar
