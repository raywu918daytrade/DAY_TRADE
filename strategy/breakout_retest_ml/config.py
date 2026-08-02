"""
breakout_retest_ml 策略常數。

日 K breakout_retest + POC 共振 → 盤中 M1 陽線實體 K + Tick 大量買進
→ LightGBM 三分類。物化表：db/breakout_retest_day、db/breakout_retest_trigger。
"""

import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = _ROOT / "cache"
MODEL_DIR = _ROOT / "models"

# ── 交易時段 ──────────────────────────────────────────────────────────────
SESSION_START = (9, 10)
_end_h = int(os.environ.get("BREAKOUT_RETEST_ML_SESSION_END_HOUR", "10"))
_end_m = int(os.environ.get("BREAKOUT_RETEST_ML_SESSION_END_MIN", "0"))
SESSION_END = (_end_h, _end_m)

# 持有最晚到 SESSION_END + LABEL_HORIZON（預設 10:30）
LABEL_HORIZON_MINUTES = 30

# ── 日 K 候選 ─────────────────────────────────────────────────────────────
MIN_PATTERN_SCORE = 60.0
POC_CONFLUENCE_MAX_PCT = 2.0  # 與 detector 內建一致；候選硬過濾用

# ── 盤中觸發（硬）：M1 陽線實體 K + 有方向的 Tick 大量買進 ───────────────
# 實體比例 = |close-open| / (high-low)；達門檻才算「實體 K」（非十字／影線主導）
MIN_BODY_RATIO = 0.50
# 上影線佔全長超過此比例 → 視為拋壓，排除
MAX_UPPER_SHADOW_RATIO = 0.35

# ── Triple Barrier（報酬率）───────────────────────────────────────────────
# 規格書 ±3% 偏嚴；第一版用 ±2%，之後可對照實驗
TP_PCT = 0.02
SL_PCT = 0.02

# ── Tick 硬過濾／特徵（tick_type=1 為外盤買進，有方向）──────────────────
TICK_CVD_SECONDS = 30
TICK_LARGE_BUY_SECONDS = 60
TICK_LARGE_LOT = 50  # 單筆 > 50 張視為大單
# 觸發前 60s：大單買量 / 總量 ≥ 此值，才算「大量買進」
MIN_TICK_LARGE_BUY_RATIO = 0.10
# 觸發前 30s CVD（買−賣）必須為正
REQUIRE_CVD_POSITIVE = True

# ── 模型 ──────────────────────────────────────────────────────────────────
MODEL_TYPE = os.environ.get("BREAKOUT_RETEST_ML_MODEL_TYPE", "lgbm")
THRESHOLD = float(os.environ.get("BREAKOUT_RETEST_ML_THRESHOLD", "0.6"))

# ── 回測出場（與 label 的 Triple Barrier 對齊第一版 ±2%/30 分）────────────
BACKTEST_TP_PCT = TP_PCT
BACKTEST_SL_PCT = SL_PCT
BACKTEST_HOLD_BARS = LABEL_HORIZON_MINUTES


def prepared_cache_path(start_date: str | None) -> Path:
    if start_date is None:
        return CACHE_DIR / "breakout_retest_ml_prepared.parquet"
    return CACHE_DIR / f"breakout_retest_ml_prepared_{start_date}.parquet"


def candidates_cache_path(start_date: str | None) -> Path:
    if start_date is None:
        return CACHE_DIR / "breakout_retest_ml_candidates.parquet"
    return CACHE_DIR / f"breakout_retest_ml_candidates_{start_date}.parquet"
