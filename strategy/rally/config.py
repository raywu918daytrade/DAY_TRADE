"""
交易相關設定 — RFC/XGB/LGBM 三模型與 entry.py 共用的常數

只放常數，不放邏輯，避免其他模組互相依賴造成 circular import。
"""

import os

# ── Triple Barrier 參數（標籤怎麼定義） ──────────────────────────────────────
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── 即時交易信心度門檻（RFC/XGB/LGBM 各自一個，rally_xgb/rally_lgbm 兩個
# 獨立策略各自查自己的 key） ──────────────────────────────────────────────
# rally 自己的 RFC/XGB/LGBM 三個模型機率校準不一樣（見
# experiments/walk_forward_xgb.py、walk_forward_lgbm.py 的驗證結果，XGB/LGBM
# 在 threshold=0.60 才開始有穩定明顯的改善）。寫死（2026-07-22：改成跟
# strategy/orb/config.py 統一寫法，不走 .env 環境變數——這些值是 walk-forward
# 驗證出來的結論，不是隨時能調的操作旋鈕，.env 沒進版控，改了不會留下
# git 紀錄，寫死才能讓每次調整都對得起某次驗證結果）。
# RFC 的 walk-forward 驗證還沒跑完，先沿用舊預設 0.55，之後驗證結果出來
# 再調整。
THRESHOLD_BY_MODEL = {"rfc": 0.55, "xgb": 0.60, "lgbm": 0.60}

# ── 交易時段（早盤過濾範圍） ──────────────────────────────────────────────────
SESSION_START = (9, 1)
_end_h = int(os.environ.get("SESSION_END_HOUR", "10"))
_end_m = int(os.environ.get("SESSION_END_MIN", "0"))
SESSION_END = (_end_h, _end_m)

# ── 強過濾破底翻交易時段（黃金窗口） ───────────────────────────────────────────
BREAKOUT_TRADE_START = (9, 14)
BREAKOUT_TRADE_END = (9, 30)
