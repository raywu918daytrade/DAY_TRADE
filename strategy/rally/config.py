"""
交易相關設定 — RFC/XGB/LGBM 三模型與 entry.py 共用的常數

只放常數，不放邏輯，避免其他模組互相依賴造成 circular import。
"""

import os

# ── Triple Barrier 參數（標籤怎麼定義） ──────────────────────────────────────
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── 要用哪個模型（rfc / xgb / lgbm） ──────────────────────────────────────
# run_backtest.py（回測）跟 live.py（即時交易）都讀這個，只改這裡一個地方
# 就能同時切換兩邊要用的模型，不用分別改兩個檔案。
MODEL_TYPE = os.environ.get("RALLY_MODEL_TYPE", "xgb")

# ── 交易時段（早盤過濾範圍） ──────────────────────────────────────────────────
SESSION_START = (9, 1)
_end_h = int(os.environ.get("SESSION_END_HOUR", "10"))
_end_m = int(os.environ.get("SESSION_END_MIN", "0"))
SESSION_END = (_end_h, _end_m)

# ── 強過濾破底翻交易時段（黃金窗口） ───────────────────────────────────────────
BREAKOUT_TRADE_START = (9, 14)
BREAKOUT_TRADE_END = (9, 30)
