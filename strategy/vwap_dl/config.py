"""
vwap_dl 策略專用常數 — 繼承 vwap_ml 的 VWAP z-score 邏輯，加上 DL 模型設定。
"""

import os
import torch

# ── VWAP band 設定（參照 vwap_ml/config.py） ─────────────────────────────────
# 2026-08-04 從 2.0 調降到 1.5：候選觸發門檻放寬，換取更多候選事件；跟
# vwap_ml/config.py 的 STD_MULT 是各自獨立的常數，改這裡不影響 vwap_ml。
STD_MULT = 1.5
LABEL_HORIZON_MINUTES = 30

# 即時交易信心度門檻
THRESHOLD = float(os.environ.get("VWAP_DL_THRESHOLD", "0.6"))

# 交易時段
SESSION_START = (9, 10)
_session_end_h = int(os.environ.get("VWAP_DL_SESSION_END_HOUR", "10"))
_session_end_m = int(os.environ.get("VWAP_DL_SESSION_END_MIN", "0"))
SESSION_END = (_session_end_h, _session_end_m)

# VWAP z-score 候選觸發來自 VWAP_ML 的 features — 指向同一個 STD_MULT。
# ATR5_FILTER_THRESHOLD 是獨立的一份（跟 vwap_ml/config.py 那份數值上曾經
# 剛好一樣，但不是同一個常數，改這裡不影響 vwap_ml）。
#
# 2026-08-04 曾經調降到 0.0008（≈候選事件ATR5分布的p10，理由見git history/
# 對話紀錄：0.01在候選ATR5分布裡落在第97.4百分位，標籤分布分析顯示調低
# 門檻樣本量大增、標籤品質也沒變差），照這個門檻重建shard+重訓後離線指標
# （AUC/accuracy/recall全面提升）明顯變好，但同一批候選跑回測（threshold
# =0.6，最後30天）反而從80%勝率/+1.20%報酬掉到31%勝率/-2.62%（58筆交易
# 51筆卡在hold_bars強制出場，訊號方向性明顯變弱）——離線分類指標變好
# 不代表回測會變好，兩者出現背離，先改回0.01（模型備份見
# models/vwap_dl_cnngru.pt.bak_pre_atr_adj，之後如果要重新嘗試調低門檻，
# 記得同時看回測、不能只看離線指標）。
ATR5_FILTER_THRESHOLD = 0.01

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
#   2026-08-04 曾經試過加 m3_std/m5_std 獨立K棒形態（6維，GRU_INPUT_DIM
#   一度改成24，_add_std_bar_shape() 函式還留著沒刪，供之後想重試時直接
#   重用），但離線指標變好、實際回測（10筆/80%勝率/+1.20% → 4筆/50%勝率/
#   -0.13%）反而變差，改回不用，GRU_INPUT_DIM 改回 18。
#   2026-08-04 也曾加 m1_engulfing（M1吞噬型態，見
#   strategy/vwap_dl/dataset.py::_add_engulfing()，函式還在，之後想重試
#   可以直接重用）——搭配STD_MULT=1.5一起測，30天/90天回測都比基準差，
#   改回不用，GRU_INPUT_DIM 改回 18（STD_MULT保留1.5）。
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

# 大盤相對特徵用
IDX_SYMBOL = "0050"
