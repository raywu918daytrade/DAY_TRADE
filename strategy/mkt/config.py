"""
交易相關設定 — mkt 策略專用常數，跟 rally/orb 各自獨立，不共用。

只放常數，不放邏輯，避免其他模組互相依賴造成 circular import。
"""

import os

# ── Triple Barrier 參數（標籤怎麼定義） ──────────────────────────────────────
# HOLD_BARS=10：2026-07-14 用 strategy/mkt/experiments/ret_vs_idx_signal_check.py
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
MODEL_TYPE = os.environ.get("MKT_MODEL_TYPE", "lgbm")

# ── 即時交易信心度門檻預設值 ──────────────────────────────────────────────
# main/live_trader.py 每分鐘先查前端 settings 裡的全域 threshold，沒設定才
# fallback 這裡（見 main/state.py::StrategyState、main/live_trader.py 的
# 說明）——原本 main/config.py 有一個全域 THRESHOLD 給所有策略共用，但
# orb/rally/mkt 三個模型的機率校準跟最佳門檻不一定一樣，2026-07-21 拆成
# 各策略自己一個。跟 predict.py::predict_live() 的 threshold 參數預設值一致。
THRESHOLD = float(os.environ.get("MKT_THRESHOLD", "0.6"))

# ── 交易時段 ──────────────────────────────────────────────────────────────
# 2026-07-20 討論：9:00~9:10排除不用——開盤集合競價剛結束，波動雖大
# （strategy/mkt/experiments/opening_hour_10min_distribution.py 測過這
# 段訊號密度最高），但被認為是雜訊、沒有方向性，不是要進場的時機。實際
# 交易時段改成9:11~9:30，跟 train.py::_prepare_data() 的 minute_min/
# minute_max 要保持一致。
SESSION_START = (9, 11)
_end_h = int(os.environ.get("MKT_SESSION_END_HOUR", "9"))
_end_m = int(os.environ.get("MKT_SESSION_END_MIN", "30"))
SESSION_END = (_end_h, _end_m)

# ── 大盤代理股票代號 ───────────────────────────────────────────────────────
IDX_SYMBOL = "0050"

# ── 流動性篩選 ────────────────────────────────────────────────────────────
# 2026-07-14 討論：台股本身有很多低成交量的股票，本來就不太會動，會拉低
# 「訊號密度」的統計。門檻沿用 orb 的 MIN_VOL_MA20=1,000,000股（=20日均量
# 1000張），只是起始值，還沒針對 mkt 重新驗證過這個數字合不合適。
MIN_VOL_MA20 = 1_000_000

# ── 流動性過濾 top_n ─────────────────────────────────────────────────────
# 2026-07-25討論：重跑walk-forward比較top_n=100 vs 300（皆搭配ATR5 p90
# 過濾），300在高門檻(0.7/0.8)precision明顯較好且穩定（std從17~22%降到
# 6~10%），尤其解決了最早一個窗口（2025-12-10~2026-01-24）訓練樣本不足、
# precision直接掉到0%的問題（那個窗口的漲樣本數460→884筆）。⚠️ 這不是
# 「平衡了label比例」——跌/平/漲比例兩者幾乎一樣（top100: 4.76/88.93/6.31、
# top300: 4.52/89.22/6.26），是靠絕對樣本量變多在幫忙，不是密度改善。
# 低門檻(0.5/0.6)top300反而略輸top100（2/5窗口贏），整體高門檻更常用、
# 且穩定性更重要，權衡後改採top_n=300為正式設定，見README.md。
TOP_N = 300

# ── ATR5 平盤過濾（實驗性） ─────────────────────────────────────────────────
# 2026-07-23討論：「平」佔比過高（~98%），想用ATR5篩掉「本來就沒什麼波動、
# 幾乎注定不會動」的樣本。試過兩種相對排名（跨股票同一分鐘排名、同一支
# 股票自己的歷史分布）效果都弱，跨股票排名PR>=0.9時漲密度1.27%→5.41%，
# 逐股票歷史版更弱只到1.82%——相對排名structurally沒辦法區分「客觀上很
# 平靜」跟「只是相對平靜」。參考strategy/cnn/experiments/
# atr5_flat_filter_check.py的做法（只篩平、用絕對門檻），發現「只篩平」
# 效果最強（PR=0.9時漲密度到10.76%），但那個規則依賴事後才知道的label，
# 沒辦法在即時推論時重現（不知道會不會變成漲/跌，就沒辦法只篩平）。
# 改成「絕對門檻＋跌/平/漲三類都篩」——不看類別、只看當下atr5這個數字，
# train/test/上線推論三邊用同一套規則，可以真的部署。效果介於兩種相對
# 排名版本跟「只篩平」版本之間（p90時漲密度1.27%→6.47%）。
#
# 門檻是「全體樣本atr5的p90分位數」，用 strategy/mkt/experiments/
# atr5_pr_check.py::run_absolute_uniform() 對截至2026-07-23的資料算出來的
# 固定數字，不是每次訓練動態重算——之後資料分布如果有明顯變化，要記得
# 回來重新驗證這個數字合不合適，不要無限期沿用。
ATR5_FILTER_THRESHOLD = 0.00748
