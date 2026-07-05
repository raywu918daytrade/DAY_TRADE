---
title: 風控規則
updated_at: 2026-07-03
---

# 風控規則

## 停損 / 停利

- `stop_loss_pct` / `take_profit_pct`（使用者可調整設定，預設各 3.0%）
- 停損停利以「進場均價」為基準，SL/TP 每次 `reconcile()` 用永豐即時報價判斷是否觸發
- 重啟後會立刻執行一次 `startup_sltp_check()`，不等下一分鐘，避免中斷期間錯過出場

## 收盤強制平倉

- 交易時段結束（預設 10:00，可用 `SESSION_END_HOUR` / `SESSION_END_MIN` 調整）後，所有未平倉部位一律市價強制平倉
- 對應 `exit_reason=force_close_eod`

## 資金與額度控管

- `TOTAL_CAPITAL`：當沖總資金（.env，預設 200 萬）
- 現股當沖買賣各佔一半資金：`_buy_capital = TOTAL_CAPITAL / 2`
- `used_quota`（今日已用額度）每次 `reconcile()` 一律從永豐 `get_positions()` 重新計算，不用本地累加 —— 因為當沖一買一賣算兩次成交，本地累加曾經造成 -3211 萬的計算錯誤（永豐才是唯一真實來源）

## 標的篩選（風控第一道防線）

1. `isDayTrading=true` 且 `isNormal=true`
2. 20 日均量 ≥ `MIN_AVG_VOL_LOTS`（張），過濾流動性不足的標的
3. `MAX_SUBSCRIPTIONS`（預設 500）：即時報價訂閱數上限，避免超過永豐/Fugle 限制

## 訊號門檻

- `THRESHOLD`（預設 0.55）：模型 `proba >= THRESHOLD` 才視為有效買進訊號
- 未達門檻但仍被記錄在 `/monitoring`，供人工檢視模型的即時信心分數

## 例外處理

- 委託失敗（`status=FAILED`）不計入 `open_trades`，會出現在 `/failed_orders`
- 使用者可透過 `/close_now` 手動立即市價平倉指定股票，不受 SL/TP 條件限制
