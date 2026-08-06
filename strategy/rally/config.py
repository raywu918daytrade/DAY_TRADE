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
# 2026-08-06：改成只在 9:00~9:30 開盤黃金窗口交易，寫死（不走 .env）——理由
# 跟 THRESHOLD_BY_MODEL 一樣，這是驗證過的結論，不是隨時能調的操作旋鈕。
# 用 2021年起+固定400支重訓後，逐分鐘信心度桶比對顯示 RFC 在這個窗口的
# 機率校準明顯最乾淨（precision 隨信心度單調遞增，0.50-0.55 桶 83.6%），
# XGB/LGBM 在這個窗口也是校準最好的時段，但精準率、單調性都不如 RFC；
# 9:30 之後（尤其10點以後）三個模型的高信心度桶都出現精準率不升反降的
# 校準劣化，不值得冒險交易。

# 字串格式（"H:MM"）給人類看比較清楚，不是 tuple——main/state.py::
# StrategyState.__init__ 是所有策略模組共用的唯一進入點，那裡會轉成
# (h, m) tuple，main/live_trader.py 後面的比較/索引邏輯不用改，orb/mkt/
# cnn/vwap_ml/vwap_dl 那些還是傳 tuple 的策略也不受影響（見 state.py 的
# _parse_hhmm() 說明）。run_backtest.py 直接把這幾個字串傳給
# backtest/intraday_platform.py::run_backtest() 的 first_entry_time/
# last_entry_time，格式本來就是 "HH:MM"，不用再轉換。
SESSION_START = "9:00"
SESSION_END = "9:30"

# ── ATR 平盤過濾 ─────────────────────────────────────────────────────────────
# 2026-08-06 討論：用 m1_atr（已經是 FEATURES 裡的欄位，1分鐘K ATR(14)相對
# 波動）當絕對門檻，篩掉「本來就沒什麼波動、幾乎注定不會觸發3%停利/停損」的
# 平盤樣本，比照 strategy/mkt/config.py::ATR5_FILTER_THRESHOLD 的做法（絕對
# 門檻、不分漲跌都篩、train/predict/predict_live 三邊用同一個門檻才能真的
# 部署，避免 training-serving skew）。
#
# 一開始用 p90（0.00422）試過一輪，2026-08-06 實測發現套用後 LGBM 高信心度桶
# 訊號裡還是有太多「持平」樣本沒被篩掉，改提高到 p99（0.00761，2021年起+固定
# 400支母體，25,918,051筆全體樣本算出來的分位數，篩完剩 1%＝約26萬筆）——比照
# mkt 最終驗證出來最好的門檻也是 p99。跟 mkt 不同的是，這裡還沒有像 mkt 當初
# 那樣 p90/p95/p97/p99 四個候選都各自跑一輪 walk-forward 才選，只是直接跳去
# 用 mkt 驗證過的同一個百分位，之後如果要重新嚴謹驗證，用
# experiments/walk_forward_*.py 加上不同 ATR_FILTER_THRESHOLD candidate 重跑。
ATR_FILTER_THRESHOLD = 0.00761
