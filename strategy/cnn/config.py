"""
交易相關設定 — cnn 策略專用常數，跟 rally/orb/mkt 各自獨立，不共用。

只放常數，不放邏輯，避免其他模組互相依賴造成 circular import。
"""

import torch

# ── Triple Barrier 參數（標籤怎麼定義） ──────────────────────────────────────
# 沿用 rally/orb 的起始值，之後用 evaluate() 的結果再調整。
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── 五路多解析度分支的 lookback 長度 ───────────────────────────────────────
# 五路對齊到同一段「過去10分鐘」的時間範圍，只是取樣密度不同：
#   m1/m3/m5（rolling）是每分鐘一列，抓過去10點＝過去10分鐘
#   m3_std 是獨立3分K棒，抓過去3根＝9分鐘；m5_std 獨立5分K棒抓過去2根＝10分鐘
#
# 2026-07-24 從30分鐘再改成10分鐘：使用者要求最早9:10就能開始推論（比30分鐘
# 版本的9:30更早），才能完整涵蓋9:00~10:00這段他認為最活躍的窗口。連帶把
# model.py的_Branch感受野排程也從RF=31(4層dilated conv)降成RF=15(3層)，
# 配合縮小後的視窗（見model.py說明）。
LOOKBACK_MINUTES = 10
M1_LEN = 10
M3_LEN = 10
M5_LEN = 10
M3STD_LEN = 3
M5STD_LEN = 2

# ── 流動性篩選 ────────────────────────────────────────────────────────────
# 沿用 orb/mkt 的起始門檻（20日均量 ≥ 1000張）。
MIN_VOL_MA20 = 1_000_000

# ── 大盤相對特徵（2026-07-24新增，比照strategy/mkt的add_ret_vs_idx()） ──────
IDX_SYMBOL = "0050"

# ── 訓練裝置 ──────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── 即時交易設定（2026-07-25新增，比照strategy/mkt的慣例） ───────────────────
# train()目前只訓練9~10點候選（見train.py的hour_start/hour_end），
# SESSION_START/END跟這個範圍對齊，main/state.py::StrategyState讀這兩個常數
# 決定這個策略哪個時間區間該呼叫predict_live()。
SESSION_START = (9, 0)
SESSION_END = (10, 0)
THRESHOLD = 0.6  # 信心度門檻，2026-07-25驗證：0.6是精準度/涵蓋率較平衡的操作點

# 只篩「平」（訓練時，見train.py::_atr5_mask()）；即時推論不知道真正label，
# 改成不分類別、對所有候選一視同仁套用（見predict.py::predict_live()的說明）。
# 數字用strategy/cnn/experiments/atr5_flat_filter_check.py在2026-03~07資料上
# 算出來的9~10點p90分位數；目前訓練範圍已擴大到2024-01起，這個門檻沿用同一個
# 數字（2026-07-25討論：先直接沿用，還沒針對擴大後的範圍重新校準）。
ATR5_FILTER_THRESHOLD = 0.00541
