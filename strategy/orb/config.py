"""
交易相關設定 — macd 策略 LGBM 模型與 entry.py 共用的常數

只放常數，不放邏輯，比照 strategy/rally/config.py 的作法。
"""

# ── Triple Barrier 參數（標籤怎麼定義，沿用 rally 的定義） ──────────────────
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── MACD 參數 ────────────────────────────────────────────────────────────
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
