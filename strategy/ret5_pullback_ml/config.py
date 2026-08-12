"""ret5_pullback_ml 常數。

強過濾事件（規則／ML 共用）；ML 將 atr5 當特徵、verify 可另套 p99 硬過濾。
AVWAP 自 m5_down_ts 下一根 m1 起累積至 entry_ts。
"""

from __future__ import annotations

import os
from datetime import time as dtime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "cache"
MODEL_DIR = _ROOT / "models"

RET5_MIN = 0.03
SIGNAL_DEADLINE = dtime(9, 30)
ENTRY_DEADLINE = dtime(10, 0)
FIRST_M5_T = dtime(9, 5)
HOLD_M5_BARS = 6
TP_PCT = 0.03
SL_PCT = 0.03
M1_REV_LOOKAHEAD = 5
M5_LOAD_UNTIL = dtime(10, 30)
M1_SIG_UNTIL = dtime(9, 35)
SESSION_OPEN = dtime(9, 0)

# expanding std for avwap_z：錨點後至少 N 根才給 z（不足則 NaN → drop）
AVWAP_Z_MIN_PERIODS = 2

MODEL_TYPE = os.environ.get("RET5_PULLBACK_ML_MODEL_TYPE", "lgbm")
THRESHOLD = float(os.environ.get("RET5_PULLBACK_ML_THRESHOLD", "0.5"))

FEE_RATE = 0.001425
TAX_RATE = 0.003


def events_cache_path(start_date: str | None, end_date: str | None = None) -> Path:
    s = start_date or "all"
    e = end_date or "latest"
    return CACHE_DIR / f"ret5_pullback_ml_events_{s}_{e}.parquet"
