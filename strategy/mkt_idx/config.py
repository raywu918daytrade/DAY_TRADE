"""
交易相關設定 — mkt_idx 策略專用常數，跟 rally/orb 各自獨立，不共用。

只放常數，不放邏輯，避免其他模組互相依賴造成 circular import。
"""

import os

# ── Triple Barrier 參數（標籤怎麼定義） ──────────────────────────────────────
# HOLD_BARS=10：2026-07-14 用 strategy/mkt_idx/experiments/ret_vs_idx_signal_check.py
# 驗證過，ret_vs_idx 最落後大盤那前3個decile的優勢在5~10分鐘內最明顯，拉到
# 15分鐘優勢快沒了，30分鐘直接反轉，所以持有時間對齊到10根分K，不要照抄
# rally的30（那是給不同訊號類型用的參數，這裡訊號衰退得快很多）。
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 10

# ── 交易時段 ──────────────────────────────────────────────────────────────
SESSION_START = (9, 1)
_end_h = int(os.environ.get("MKT_IDX_SESSION_END_HOUR", "13"))
_end_m = int(os.environ.get("MKT_IDX_SESSION_END_MIN", "25"))
SESSION_END = (_end_h, _end_m)

# ── 大盤代理股票代號 ───────────────────────────────────────────────────────
IDX_SYMBOL = "0050"

# ── 流動性篩選 ────────────────────────────────────────────────────────────
# 2026-07-14 討論：台股本身有很多低成交量的股票，本來就不太會動，會拉低
# 「訊號密度」的統計。門檻沿用 orb 的 MIN_VOL_MA20=1,000,000股（=20日均量
# 1000張），只是起始值，還沒針對 mkt_idx 重新驗證過這個數字合不合適。
MIN_VOL_MA20 = 1_000_000
