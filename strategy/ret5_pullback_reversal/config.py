"""ret5_pullback_reversal 常數。"""

from datetime import time as dtime

from strategy.mkt.config import ATR5_FILTER_THRESHOLD  # noqa: F401 — p99 = 0.01095

RET5_MIN = 0.03
BODY_RATIO_MIN = 0.5
SIGNAL_DEADLINE = dtime(9, 30)  # 陰線／反轉陽線收盤皆 < 09:30
ENTRY_DEADLINE = dtime(10, 0)  # 進場 < 10:00
FIRST_M5_T = dtime(9, 5)
HOLD_MINUTES = 30
HOLD_M5_BARS = 6  # 30 分 / 5 分
TP_PCT = 0.03
SL_PCT = 0.03
# m5 載到進場最晚(~09:29) + 30 分
M5_LOAD_UNTIL = dtime(10, 30)
