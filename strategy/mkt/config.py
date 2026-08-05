"""
交易相關設定 — mkt 策略專用常數，跟 rally/orb 各自獨立，不共用。

只放常數，不放邏輯，避免其他模組互相依賴造成 circular import。
"""

import os

# ── Triple Barrier 參數（標籤怎麼定義） ──────────────────────────────────────
# HOLD_BARS=30（2026-07-25改，取代原本的10）：原本2026-07-14用
# strategy/mkt/experiments/ret_vs_idx_signal_check.py 在top_n=100母體上
# 驗證過，說30分鐘訊號會反轉。但2026-07-25發現top_n母體本身有嚴重問題
# （見上面「股票母體」那段說明），改用tick_universe固定400支、抓最近3個月
# 重新驗證同一支腳本（forward_minutes=5/10/15/30），結果完全不一樣：
# 相關係數從5分鐘的-0.0161一路增強到30分鐘的-0.0198，decile0（落後大盤最多）
# vs decile9（領先最多）的未來報酬率差距在30分鐘也沒有縮小（+8.13pp→
# +8.91pp）——完全沒有反轉，訊號反而更強。不確定是母體換了、還是時間段
# 剛好不同造成結論不一樣，但用新驗證結果改成30。
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

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

# ── 股票母體（2026-07-25起改用固定tick_universe，不再用top_n動態排名） ──────
# ⚠️ 曾經短暫用過「前一日量前N名」（TOP_N=100→300）動態排名當流動性過濾，
# 2026-07-25發現這個設計建立在一個錯誤假設上：db/m1的股票覆蓋範圍其實
# 一直在變（歷史從333支逐漸長到2026-06的2710支，2026-07驟降到1605支，
# 2026-08起固定變成400支——data/m1_data_loader.py 2026-08-01改成只收錄
# db/tickers/tick_universe.parquet固定的400支，不再收全市場）。這代表
# 「top_n=300」在不同訓練/回測期間實際上是從不同大小的母體裡選，越晚的
# 窗口母體越小，跟訊號好壞無關的母體結構變化會被誤判成precision進步，
# 汙染了整個top_n=100 vs 300、ATR5門檻p90~p99的walk-forward驗證結果
# （README.md裡2026-07-25那幾段記錄的數字，是在這個問題被發現前算出來
# 的，已知不可信，保留只是當作發現過程的記錄）。
#
# 改法：直接用 finmind.tick_universe.load_tick_universe() 讀固定400支
# （399支排名+0050強制併入），train/predict都用同一份、不再依日期/成交量
# 動態變動，train.py::_prepare_data() 直接呼叫，不需要另外開常數。

# ── ATR5 平盤過濾（實驗性） ─────────────────────────────────────────────────
# 2026-07-23討論：「平」佔比過高，想用ATR5篩掉「本來就沒什麼波動、幾乎注定
# 不會動」的樣本。試過兩種相對排名（跨股票同一分鐘排名、同一支股票自己的
# 歷史分布）效果都弱——相對排名structurally沒辦法區分「客觀上很平靜」跟
# 「只是相對平靜」。參考strategy/cnn/experiments/atr5_flat_filter_check.py
# 的做法（只篩平、用絕對門檻）發現「只篩平」效果最強，但那個規則依賴事後
# 才知道的label，沒辦法在即時推論時重現。改成「絕對門檻＋跌/平/漲三類都
# 篩」——不看類別、只看當下atr5這個數字，train/test/上線推論三邊用同一套
# 規則，可以真的部署。踩坑細節見`add_atr5()`/`atr5_pr_check.py` docstring。
#
# 2026-07-25第一版（top_n=300、HOLD_BARS=10）選過p97=0.01000，但那個母體
# 有嚴重bug（見上面「股票母體」那段說明），數字不可信。
#
# 2026-07-25第二版（改用tick_universe固定400支之後，同一天又把HOLD_BARS
# 從10改成30，見config.py的HOLD_BARS說明）：用
# strategy/mkt/experiments/retest_atr5_tick_universe.py 重新對p90(0.00552)/
# p95(0.00707)/p97(0.00827)/p99(0.01095)四個候選跑一次5窗口45天walk-
# forward，**p99在全部4個模型門檻(0.5/0.6/0.7/0.8)都是最好**（19.82%→
# 28.66%、24.32%→30.80%、29.15%→34.74%、32.47%→35.09%，數字是p90→p99），
# 這次沒有像第一版那樣p97在高門檻反而贏，p99全面勝出。代價一樣是樣本量少
# （過濾前648萬筆全體population → p99只剩64,834筆，約1%），但precision
# 優勢夠明確，選p99。
#
# 門檻是「全體樣本atr5（tick_universe固定400支population）的p99分位數」，
# 對截至2026-07-25的資料算出來的固定數字，不是每次訓練動態重算——之後
# 資料分布如果有明顯變化，要記得回來重新驗證這個數字合不合適。
ATR5_FILTER_THRESHOLD = 0.01095
