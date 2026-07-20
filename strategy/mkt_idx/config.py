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

# ── 要用哪個模型（rfc / xgb / lgbm） ──────────────────────────────────────
# run_backtest.py（回測）跟 live.py（即時交易）都讀這個，只改這裡一個地方
# 就能同時切換兩邊要用的模型，比照 strategy/rally/config.py、strategy/orb/
# config.py 的做法。
MODEL_TYPE = os.environ.get("MKT_IDX_MODEL_TYPE", "lgbm")

# ── 即時交易信心度門檻預設值 ──────────────────────────────────────────────
# main/live_trader.py 每分鐘先查前端 settings 裡的全域 threshold，沒設定才
# fallback 這裡（見 main/state.py::StrategyState、main/live_trader.py 的
# 說明）——原本 main/config.py 有一個全域 THRESHOLD 給所有策略共用，但
# orb/rally/mkt_idx 三個模型的機率校準跟最佳門檻不一定一樣，2026-07-21 拆成
# 各策略自己一個。跟 predict.py::predict_live() 的 threshold 參數預設值一致。
THRESHOLD = float(os.environ.get("MKT_IDX_THRESHOLD", "0.6"))

# ── 交易時段 ──────────────────────────────────────────────────────────────
# 2026-07-20 討論：9:00~9:10排除不用——開盤集合競價剛結束，波動雖大
# （strategy/mkt_idx/experiments/opening_hour_10min_distribution.py 測過這
# 段訊號密度最高），但被認為是雜訊、沒有方向性，不是要進場的時機。實際
# 交易時段改成9:11~9:30，跟 train.py::_prepare_data() 的 minute_min/
# minute_max 要保持一致。
SESSION_START = (9, 11)
_end_h = int(os.environ.get("MKT_IDX_SESSION_END_HOUR", "9"))
_end_m = int(os.environ.get("MKT_IDX_SESSION_END_MIN", "30"))
SESSION_END = (_end_h, _end_m)

# ── 大盤代理股票代號 ───────────────────────────────────────────────────────
IDX_SYMBOL = "0050"

# ── 流動性篩選 ────────────────────────────────────────────────────────────
# 2026-07-14 討論：台股本身有很多低成交量的股票，本來就不太會動，會拉低
# 「訊號密度」的統計。門檻沿用 orb 的 MIN_VOL_MA20=1,000,000股（=20日均量
# 1000張），只是起始值，還沒針對 mkt_idx 重新驗證過這個數字合不合適。
MIN_VOL_MA20 = 1_000_000
