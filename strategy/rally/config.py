"""
交易相關設定 — RFC/XGB/LGBM 三模型與 entry.py 共用的常數

只放常數，不放邏輯，避免其他模組互相依賴造成 circular import。
"""

import os

# ── Triple Barrier 參數（標籤怎麼定義） ──────────────────────────────────────
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── 交易時段（早盤過濾範圍） ──────────────────────────────────────────────────
SESSION_START = (9, 1)
_end_h = int(os.environ.get("SESSION_END_HOUR", "10"))
_end_m = int(os.environ.get("SESSION_END_MIN", "0"))
SESSION_END = (_end_h, _end_m)

# ── 強過濾破底翻交易時段（黃金窗口） ───────────────────────────────────────────
BREAKOUT_TRADE_START = (9, 14)
BREAKOUT_TRADE_END = (9, 30)
