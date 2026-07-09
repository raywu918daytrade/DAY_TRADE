"""
交易相關設定 — orb 策略 LGBM 模型與 entry.py 共用的常數

只放常數，不放邏輯，比照 strategy/rally/config.py 的作法。
"""

# ── Triple Barrier 參數（標籤怎麼定義，沿用 rally 的定義） ──────────────────
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── ORB（開盤區間突破）參數 ──────────────────────────────────────────────
OPENING_RANGE_MINUTES = 15  # 開盤前N分鐘的高低點定義區間（9:00~9:15）
