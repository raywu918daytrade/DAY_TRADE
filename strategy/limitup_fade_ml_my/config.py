"""
limitup_fade_ml 策略常數。

硬過濾：前日實體漲停 → 今開高 → 首根 m3_std（09:03）下跌。
ML：LightGBM 三分類（止損/震盪/止盈），進場看 P(止盈)。
"""

from __future__ import annotations

import os
from datetime import time as dtime
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = _ROOT / "cache"
MODEL_DIR = _ROOT / "models"

# ── 硬過濾 ────────────────────────────────────────────────────────────────
LIMIT_UP_RET = 0.095
MIN_BODY = 0.50
MAX_UPPER = 0.20
FIRST_M3_TIME = dtime(9, 3)
IDX_SYMBOL = "0050"

# ── Triple Barrier（做空視角）────────────────────────────────────────────
# 價格下跌 TP_PCT → 止盈；上漲 SL_PCT → 止損；否則持有至 FORCE_EXIT → 震盪
TP_PCT = 0.03
SL_PCT = 0.03
FORCE_EXIT_TIME = dtime(13, 25)

# ── Live session：只在首 3 分收確認那一分鐘發訊 ───────────────────────────
SESSION_START = (9, 3)
SESSION_END = (9, 4)

# ── 模型 ──────────────────────────────────────────────────────────────────
MODEL_TYPE = os.environ.get("LIMITUP_FADE_ML_MODEL_TYPE", "lgbm")
THRESHOLD = float(os.environ.get("LIMITUP_FADE_ML_THRESHOLD", "0.6"))


def prepared_cache_path(start_date: str | None, end_date: str | None = None) -> Path:
    s = start_date or "all"
    e = end_date or "latest"
    return CACHE_DIR / f"limitup_fade_ml_prepared_{s}_{e}.parquet"


def events_cache_path(start_date: str | None, end_date: str | None = None) -> Path:
    s = start_date or "all"
    e = end_date or "latest"
    return CACHE_DIR / f"limitup_fade_ml_events_{s}_{e}.parquet"
