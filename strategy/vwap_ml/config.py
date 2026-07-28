"""
交易相關設定 — vwap_ml 策略專用常數，跟 orb/rally/mkt 各自獨立，不共用。

只放常數，不放邏輯，避免其他模組互相依賴造成 circular import（比照
strategy/mkt/config.py 的做法）。

2026-07-26 討論：ML 相關的門檻/參數比照 orb/rally/mkt 2026-07-22 之後的慣例，
直接寫死常數＋日期理由註解，不放 .env（.env 沒進版控，這些是要驗證過才能
定案的策略結論，不是隨手調的操作旋鈕）。只有 session 起訖時間比照 rally/mkt
用 .env 覆蓋。
"""

import os

# ── 大盤代理股票代號 ───────────────────────────────────────────────────────
# 2026-07-27 新增：加 ret_vs_idx（個股跟大盤的累積報酬率差值）當特徵，需要
# 0050 的盤中分K資料當基準，比照 strategy/mkt/config.py 的做法。
IDX_SYMBOL = "0050"

# ── VWAP band 設定 ───────────────────────────────────────────────────────
# 2026-07-26 討論：在 m1/m3/m5 三個時間框上，各自算「今日累積 VWAP」跟
# 「累積至今的偏離標準差（expanding std，只用當下已知資訊，不能用全天
# std，否則會有 lookahead）」，超過 STD_MULT 個標準差視為候選訊號。三個
# 時間框任一觸發就算候選（OR），不要求同時滿足——使用者顧慮「三個時間框
# 都要同時觸發，訊號會太少」，改成候選產生用OR、分類特徵留三個時間框的
# z-score讓模型自己學組合規律。
#
# ⚠️ 這個數字先寫 2.0 當起點，還沒驗證過是不是最佳門檻——使用者提醒不要
# 寫死，之後要比照 strategy/mkt 驗證 ATR5_FILTER_THRESHOLD（p90/p95/p97/
# p99 walk-forward比較）的方式，實際跑 1.0/1.5/2.0 等候選值比較樣本量、
# 標籤分布、precision 才能定案。features.py::add_vwap_features()/
# make_vwap_labels()、train.py::_prepare_data() 都開放 std_mult 參數覆蓋
# 這個預設值，不用改這裡就能做實驗（比照 mkt 的 atr5_threshold 參數化＋
# 非預設值跳過cache寫入的做法）。
STD_MULT = 2.0

# 「延續」判定門檻預設值＝STD_MULT*2（例如 std_mult=1.5 時延續門檻=3.0），
# 讓回歸/延續兩個barrier到觸發點的距離對稱（觸發點z=std_mult，回歸要走到
# 0、距離=std_mult；延續要走到std_mult*2、距離也是std_mult），不對稱
# 的話其中一類天生比較容易觸發，label比例會被barrier設計本身扭曲，不是
# 真實反映市場行為（2026-07-26討論，原本用std_mult+1.0，延續label比例
# 43.72%明顯比回歸25.36%高，一部分就是這個不對稱造成的假象）。這個
# relationship寫在 features.py::make_vwap_labels() 裡（continuation_mult
# 留 None 時動態算 std_mult*2），這裡不重複定義成獨立常數，避免兩邊
# 各自維護一份、std_mult 做實驗改了但這裡忘記跟著改。

# ── 標籤視窗（比照 orb/rally/mkt 的 triple barrier HOLD_BARS 概念） ───────
# 2026-07-26 討論：原本視窗大小是「觸發的那個時間框自己的後N根」（m1觸發
# 看10分鐘、m3看30分鐘、m5看50分鐘），後來改成三個時間框統一看未來
# LABEL_HORIZON_MINUTES 分鐘——m1/m3/m5 的 z-score 都是「每分鐘更新一次」
# 的序列（不是每3/5分鐘才有一筆），所以「30分鐘」對三者來說都直接等於
# 「未來30根」，不用再乘時間框的分鐘數。這個數字還沒針對 vwap_ml 重新
# 驗證過合不合適。
LABEL_HORIZON_MINUTES = 30

# ── 即時交易信心度門檻預設值 ──────────────────────────────────────────────
# 跟 predict.py::predict_live() 的 threshold 參數預設值一致，比照
# strategy/mkt/config.py::THRESHOLD 的說明。
THRESHOLD = float(os.environ.get("VWAP_ML_THRESHOLD", "0.6"))

# ── 交易時段 ──────────────────────────────────────────────────────────────
# 2026-07-26 討論：VWAP 從 9:00 開盤就開始累積，但 9:00~9:10 這段視為暖機期
# 不交易（開盤沒多久，累積樣本數太少，expanding std 還不穩定）；下午盤前
# （10:00）收工，避免抓到中盤盤整雜訊。跟 train.py 時段過濾要保持一致。
SESSION_START = (9, 10)
_end_h = int(os.environ.get("VWAP_ML_SESSION_END_HOUR", "10"))
_end_m = int(os.environ.get("VWAP_ML_SESSION_END_MIN", "0"))
SESSION_END = (_end_h, _end_m)

# ── 要用哪個模型（目前只有 lgbm） ───────────────────────────────────────────
# 2026-07-26 討論：v1 雖然只做了 LightGBM 一種，還是先比照
# strategy/mkt/config.py::MODEL_TYPE 的做法把切換機制建起來（train.py 的
# _LOAD_MODEL_BY_TYPE/_TRAIN_BY_TYPE 字典、up/down live.py 都依這個常數
# 決定要載入哪個模型），之後要加 rfc/xgb 只要在 train.py 補一組
# train_xxx()/load_model_xxx() 並登記進字典，不用動 config.py 或
# up/down/live.py。run_backtest.py（回測）跟 live.py（即時交易）都讀這個，
# 只改這裡一個地方就能同時切換兩邊要用的模型。
MODEL_TYPE = os.environ.get("VWAP_ML_MODEL_TYPE", "lgbm")

# ── ATR5 平盤過濾 ─────────────────────────────────────────────────────────
# 2026-07-26 討論：z-score 分母是「該股票累積至今的偏離標準差」，本來就
# 沒什麼波動的股票（分母趨近於0）連小幅價格雜訊都會被除出誇張的z值，產生
# 假的候選訊號。加 ATR5（True Range 5根滾動平均/當日開盤價）當絕對門檻，
# 濾掉這種假候選，比照 strategy/mkt/config.py::ATR5_FILTER_THRESHOLD 的
# 做法（那邊驗證過「絕對門檻＋三類都篩」效果最好，且train/test/上線推論
# 三邊能用同一套規則，見 strategy/mkt/README.md 的「ATR5平盤過濾」章節）。
#
# ⚠️ 這個數字直接沿用 mkt 當時驗證出來的 p97 門檻當起點，還沒針對 vwap_ml
# 自己的樣本分布重新驗證過——mkt 的母體（9:11~9:30時段、跟大盤比較的
# ret_vs_idx情境）跟 vwap_ml（9:10~10:00、VWAP偏離情境）不是同一種資料，
# 這個門檻對vwap_ml合不合適要之後用 walk-forward 重新驗證。
ATR5_FILTER_THRESHOLD = 0.01000

# ── 回測出場規則（跟 label 的 z-score barrier 是兩回事） ───────────────────
# 2026-07-26：回測用的共用引擎 backtest/intraday_platform.py 需要具體的
# 百分比停利/停損/持有根數才能模擬實際進出場，跟 make_vwap_labels() 拿來
# 定義「回歸/延續/無訊號」label 的 z-score barrier 是兩個獨立的概念（label
# 衡量「價格相對VWAP的位置」，這裡衡量「實際損益」）。先沿用 orb/rally/
# mkt 共用的 3%/3%/30根 當起點，還沒針對 vwap_ml 重新驗證過這組數字合不
# 合適。
BACKTEST_TP_PCT = 0.03
BACKTEST_SL_PCT = 0.03
BACKTEST_HOLD_BARS = 30
