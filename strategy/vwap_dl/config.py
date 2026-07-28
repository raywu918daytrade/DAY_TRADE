"""
vwap_dl 策略專用常數 — 繼承 vwap_ml 的 VWAP z-score 邏輯，加上 DL 模型設定。
"""

import os
import torch

# ── VWAP band 設定（參照 vwap_ml/config.py） ─────────────────────────────────
STD_MULT = 2.0
LABEL_HORIZON_MINUTES = 30

# 即時交易信心度門檻
THRESHOLD = float(os.environ.get("VWAP_DL_THRESHOLD", "0.6"))

# 交易時段
SESSION_START = (9, 10)
_session_end_h = int(os.environ.get("VWAP_DL_SESSION_END_HOUR", "10"))
_session_end_m = int(os.environ.get("VWAP_DL_SESSION_END_MIN", "0"))
SESSION_END = (_session_end_h, _session_end_m)

# VWAP z-score 候選觸發來自 VWAP_ML 的 features — 指向同一個 STD_MULT
# 這邊不重複定義 ATR5_FILTER_THRESHOLD，直接在 dataset.py 裡引用 vwap_ml 的。
ATR5_FILTER_THRESHOLD = 0.01000

# ── DL 模型架構（2026-07-27 改：ResNet + GRU 混合） ────────────────────────
# ResNet 看近 10 分鐘原始 OHLCV（5 channels × 10 步）
# GRU 從 9:00 到當下逐分鐘看 18 維特徵（2026-07-28 加 4 個大盤 VWAP 特徵）
LOOKBACK_MINUTES = 10

# CNN embedding 維度 / GRU hidden 維度
CNN_EMBED_DIM = 32
# GRU 每步輸入維度：OHLCV(5) + atr5/ma10/ma5/ma3/ret_vs_idx/idx_ret_since_open(6)
#   + m1_vwap_z/m3_vwap_z/m5_vwap_z(3) + 大盤 VWAP 特徵(4) = 18
#   2026-07-28 新增 4 個大盤 VWAP 特徵（market_z_score_m5 /
#   market_vwap_alignment_score / market_vwap_spread_1_5 /
#   velocity_ratio_to_market），GRU_INPUT_DIM 從 14 改為 18。
GRU_INPUT_DIM = 18
GRU_HIDDEN_DIM = 64
GRU_N_LAYERS = 2
DROPOUT = 0.3

# 訓練裝置
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# 回測出場（跟 vwap_ml 一致）
BACKTEST_TP_PCT = 0.03
BACKTEST_SL_PCT = 0.03
BACKTEST_HOLD_BARS = 30

# 流動性篩選
MIN_VOL_MA20 = 1_000_000

# 大盤相對特徵用
IDX_SYMBOL = "0050"
