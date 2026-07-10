"""
交易相關設定 — orb 策略 LGBM 模型與 entry.py 共用的常數

只放常數，不放邏輯，比照 strategy/rally/config.py 的作法。
"""

# ── Triple Barrier 參數（標籤怎麼定義，沿用 rally 的定義） ──────────────────
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── ORB（開盤區間突破）參數 ──────────────────────────────────────────────
# 三組獨立的開盤區間，窗口長度故意不同（不是同一段時間換分K方式算，那樣
# 高低點會幾乎一樣、沒有新資訊）：1分K版抓最早期的初始動能，3分K版窗口
# 更短、比1分K版更早收斂，5分K版窗口更長、更晚才收斂但雜訊較少。
OPENING_RANGE_MINUTES = 15  # 1分K版：開盤前15分鐘（9:00~9:15）
OPENING_RANGE_M3_MINUTES = 9  # 3分K版：開盤前9分鐘（前3根3分K，9:00~9:09）
OPENING_RANGE_M5_MINUTES = 20  # 5分K版：開盤前20分鐘（前4根5分K，9:00~9:20）

# ── 訓練/驗證共用的測試集天數 ─────────────────────────────────────────────
# train.py/validate.py/predict.py/entry.py 的 test_days 參數統一預設讀這裡，
# 不要各自寫死 10——train_lgbm() 跟 confidence_report() 這類驗證函式如果各自
# 預設不同的 test_days，訓練切點跟驗證切點會對不齊，測試集會偷看到訓練時
# 看過的資料，指標虛高卻不知道（2026-07-10 實測過這個 bug，AUC 從 0.65
# 假漲到 0.73）。validate.py 另外有 _warn_if_train_test_overlap() 會在真的
# 傳了不一致的 test_days 時印警告，這裡的統一預設是從源頭降低發生機率。
DEFAULT_TEST_DAYS = 10
