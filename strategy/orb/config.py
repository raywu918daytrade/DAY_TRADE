"""
交易相關設定 — orb 策略各模型與 train.py CLI 共用的常數

只放常數，不放邏輯。
"""

# ── 即時交易信心度門檻（RFC/LGBM/XGB 各自一個，orb_xgb/orb_lgbm 兩個獨立
# 策略各自查自己的 key） ───────────────────────────────────────────────────
# 同一個 orb 策略內 RFC/LGBM/XGB 三個模型的機率校準不一樣（見 validate.py
# compare_report() 的門檻表——同樣門檻，三個模型的精確率/召回率取捨完全
# 不同），共用同一個門檻沒有意義。寫死（目前沒有另外開 .env 變數覆蓋的
# 需求），取「精確率明顯優於基準線（測試集實際上漲比例約45%）且訊號量
# 還沒被壓到個位數」的保守門檻，見 2026-07-22 用19個月資料跑 validate()
# 的門檻表：
#   RFC  0.60 → 精確率 79.5%（n=44）
#   LGBM 0.65 → 精確率 64.0%（n=100）
#   XGB  0.65 → 精確率 68.1%（n=210）
# 2026-07-22：原本這裡還有個 MODEL_TYPE（讀 ORB_MODEL_TYPE）決定「單一」
# THRESHOLD 給 predict.py 當預設參數值，但 main/live_trader.py 呼叫
# predict_live() 一律明確傳 threshold=0（見該檔案），這個預設值從來沒被
# 用到過，run_backtest.py 也已經改成直接傳 model_type 參數（不再讀
# ORB_MODEL_TYPE），確認完全沒有消費者後拿掉。
THRESHOLD_BY_MODEL = {"rfc": 0.60, "lgbm": 0.7, "xgb": 0.65}

# ── Triple Barrier 參數（標籤怎麼定義，沿用 rally 的定義） ──────────────────
# 2026-07-11 測過改成 2%：test_days=10 AUC 幾乎沒變（0.4925→0.4916），
# test_days=5 有改善（0.5328→0.5732），但樣本量太小（430~1039筆測試）撐不住
# 明確結論，改回 3%。
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── ORB（開盤區間突破）時間窗口 ──────────────────────────────────────────
# 2026-08-06：train（features.py）/backtest（run_backtest.py）/live（本檔
# SESSION_START/SESSION_END）三邊原本各自用不同格式定義時間（分鐘數/tuple/
# "HH:MM"字串），run_backtest.py 甚至一度出現函式預設值(09:20)跟 __main__
# 實際呼叫值(09:30)兜不起來的情況。改成這裡統一用 "HH:MM" 字串定義一次，
# 其他格式（分鐘數、tuple）都用下面兩個 helper 從這裡派生，改一個地方
# 三邊都會跟著變。
#
# 2026-07-11 改版說明（窗口本身的設計理由，數值沿用不變）：不再是「每分鐘
# 都是一筆樣本」，改成「每次真正的突破事件才是一筆樣本」——09:00~09:10
# 建立開盤區間，09:10~09:30 內每一次收盤價從區間內/以下重新站上區間上緣，
# 都算一筆候選（見 features.py 的 make_features()，同一天可能有好幾筆）。
# 這解決了兩個問題：
#   1. 舊版每分鐘都當樣本，同一天200多根K棒高度相關，稀釋了「有沒有突破」
#      這個訊號（純1分K突破訊號特徵重要性只有3%）。
#   2. 一開始改成「一天只留第一次突破」又太嚴格——「一天只交易一次」是
#      交易執行層該決定的規則，不該在特徵/樣本產生階段就先幫模型篩掉後面
#      的突破事件；模型該做的只是替每一次真正的突破事件評分，之後回測/
#      實單要不要只挑一天裡最高信心度那筆，是另一層的決定。
# 搜尋窗口從 09:30~10:00 那次擴大測試後改回 09:10~09:30——09:30~10:00那段的
# 候選基準勝率只有33.3%，明顯比09:10~09:30那段的51.3%差，混進去反而稀釋。
# 舊版三組平行窗口（1分K版15分鐘/3分K版9分鐘/5分K版20分鐘）已經拿掉。
MARKET_OPEN = "09:00"  # 開盤時間
OPENING_RANGE_END = "09:10"  # 開盤區間結束時間（09:00~09:10 建立開盤區間）
SEARCH_WINDOW_END = "09:30"  # 突破搜尋窗口結束時間（09:10~09:30 內找真正的突破事件）


def _hhmm_to_minutes(hhmm: str, base: str = MARKET_OPEN) -> int:
    """ "HH:MM" 轉成距離 base（預設開盤時間）幾分鐘——features.py 的
    minutes_since_open 是逐分鐘算出來的整數，要用同單位才能比較。"""
    h, m = (int(x) for x in hhmm.split(":"))
    bh, bm = (int(x) for x in base.split(":"))
    return (h - bh) * 60 + (m - bm)


def _hhmm_to_tuple(hhmm: str) -> tuple[int, int]:
    """ "HH:MM" 轉成 (hour, minute) tuple——main/strategy_loader.py 規定
    SESSION_START/SESSION_END 這組跨策略共用介面吃的是 tuple（main/state.py
    存起來、main/live_trader.py 拿去跟 (h, m) 比大小），不是字串，這裡只
    負責轉型，不要把 main/ 那邊共用介面的型別也一起改掉。"""
    h, m = (int(x) for x in hhmm.split(":"))
    return (h, m)


OPENING_RANGE_MINUTES = _hhmm_to_minutes(OPENING_RANGE_END)  # 開盤區間：09:00~09:10
BREAKOUT_SEARCH_MINUTES = _hhmm_to_minutes(SEARCH_WINDOW_END)  # 找突破事件的搜尋窗口終點：09:10~09:30

# ── 訓練/驗證共用的測試集天數 ─────────────────────────────────────────────
# train.py/validate.py/predict.py 的 test_days 參數統一預設讀這裡，
# 不要各自寫死 10——train_lgbm() 跟 confidence_report() 這類驗證函式如果各自
# 預設不同的 test_days，訓練切點跟驗證切點會對不齊，測試集會偷看到訓練時
# 看過的資料，指標虛高卻不知道（2026-07-10 實測過這個 bug，AUC 從 0.65
# 假漲到 0.73）。validate.py 另外有 _warn_if_train_test_overlap() 會在真的
# 傳了不一致的 test_days 時印警告，這裡的統一預設是從源頭降低發生機率。
DEFAULT_TEST_DAYS = 30

# ── 流動性篩選 ────────────────────────────────────────────────────────────
# 20日均量（股數，用「昨天為止」算，見 features.py 的 vol_ma20）門檻，只留
# 成交量夠大的股票訓練/交易。1,000,000 股約等於台股慣用說法「20日均量1000張
# 以上」（1張=1000股）。2026-07-10 測過1000張跟5000張兩個門檻，AUC 都在
# 雜訊範圍內浮動（1000張: test_days=10→0.5400/test_days=5→0.6524；
# 5000張: 0.5477/0.6390），沒有證據支持「冷門股是1分K雜訊主因」——濾掉快
# 一半樣本（5000張門檻）都沒換到明顯改善，問題可能不在流動性，是1分K瞬時
# 事件訊號本身就弱（跟成交量高低關係不大，見 FEATURES 重要性分析：純1分K
# 突破訊號只佔3%重要性）。先改回1000張（濾得少、保留較多樣本）。
MIN_VOL_MA20 = 1_000_000

# ── 即時交易 session 邊界（live.py 對外介面用，live_trader.py 靠這兩個值判斷
# 何時該呼叫 predict_live()） ─────────────────────────────────────────────
# SESSION_START=MARKET_OPEN 要涵蓋建立開盤區間所需的K棒；SESSION_END 直接
# 等於 SEARCH_WINDOW_END——過了這個搜尋窗口，make_features() 內部本來就
# 不會再產生任何候選（見 features.py 的 _is_breakout 過濾），這裡提前停止
# 開新倉，避免每分鐘還在白跑一次完整特徵計算 pipeline。既有持倉的 SL/TP
# 監控不受這裡影響，live_trader.py 過了 SESSION_END 仍會持續呼叫
# reconcile([]) 監控到收盤。直接從 MARKET_OPEN/SEARCH_WINDOW_END 轉型，
# 不要另外寫死數字——避免跟上面的時間窗口設定各自改、兜不起來（同樣的
# 教訓見 DEFAULT_TEST_DAYS 的說明）。
SESSION_START = _hhmm_to_tuple(MARKET_OPEN)
SESSION_END = _hhmm_to_tuple(SEARCH_WINDOW_END)
