---
title: 當沖 ML 訊號策略總覽
updated_at: 2026-07-03
---

# 當沖 ML 訊號策略總覽

本系統是台股現股當沖策略，核心邏輯是：模型只負責判斷「進場時機」，出場完全交給固定的停損停利規則（SL/TP），不依賴模型預測出場點。

## 交易時段

- 訊號時段：09:01 起（第一根完整分K），可用 `SESSION_END_HOUR` / `SESSION_END_MIN` 調整收單時間，預設 10:00。
- 收盤前（EOD）會強制平倉所有未平倉部位（`exit_reason=force_close_eod`）。

## 標的篩選（每日更新）

1. ① `isDayTrading=true` → ② `isNormal=true` → 約 2787 支候選
2. ③ 20 日均量 ≥ `MIN_AVG_VOL_LOTS`（張）過濾流動性
3. ④ `MAX_SUBSCRIPTIONS` 上限截斷（預設 500 支），避免同時訂閱過多即時報價

## 進場邏輯

每分鐘（`on_minute`）對所有監控中股票跑一次模型推論（見 [lightgbm_barrier_model](../model/lightgbm_barrier_model.md)），`proba >= THRESHOLD` 的股票產生買進訊號 → 風控檢查（額度、部位數上限）→ 送出市價/限價買單。

## 出場邏輯

進場後不再由模型判斷，改用固定規則：

- 停損 / 停利百分比（見 [risk_rules](../risk/risk_rules.md)）
- 收盤前強制平倉（`force_close_eod`）
- 使用者手動觸發 `/close_now`

## 已知限制

- 現股當沖：買賣各佔總資金一半（`TOTAL_CAPITAL / 2`）
- 模型訓練與推論用的特徵完全限定在「當日盤中」不跨日累積（避免用到未來資料），詳見 [m1_features](../feature/m1_features.md)
