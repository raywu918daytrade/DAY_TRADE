"""
limitup_fade_ml 策略常數。

漲停隔日開高 → 首根 3 分K（09:00~09:03）下跌（Stage 1候選）→ 09:10 延續確認
（Stage 2，價格比 09:03 更低才進場）→ LightGBM 三分類過濾做空。
全新實作（2026-08-04），不依賴／不參考 strategy/limitup_fade_ml.zip 裡的舊版。

2026-08-04 二次改版：09:03 立即進場版本已驗證邊際不穩定（純規則扣成本五年不賺、
ML門檻2024/2025 walk-forward互相矛盾），改成兩階段觸發（09:03候選 + 09:10延續
確認才進場），理由/驗證結果見使用者 memory project_limitup_fade_ml.md。
"""

import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = _ROOT / "cache"
MODEL_DIR = _ROOT / "models"

# ── 前日（漲停日）硬過濾 ─────────────────────────────────────────────────
LIMIT_UP_RET = 0.095  # 日報酬 >= 9.5%（留一點空間給接近漲停但沒完全鎖死的情況）
MIN_BODY_RATIO = 0.05  # 實體比例 = |close-open| / (high-low)
MAX_UPPER_SHADOW_RATIO = 0.20  # 上影比例 = (high-max(open,close)) / (high-low)

# ── 今日觸發：跳空開高 + 首根 3 分K（09:00~09:03）下跌（Stage 1候選）───────
FIRST_M3_TIME = (9, 3)

# ── Stage 2：09:10 延續確認（db/m1 的 09:09 這根，涵蓋09:09:00~09:09:59，
# 標籤慣例是「起始時間」，等於「09:10 當下最新一根已收完的分K」，語意對齊
# FIRST_M3_TIME「@09:03代表涵蓋到09:02:59」）。收盤價要比 Stage1 的 m3 收盤價
# 更低才算延續下跌，才真的進場；沒有就整筆事件捨棄（視為假突破/已反彈）。
CONFIRM_TIME = (9, 9)

# live 端做 Stage2 判斷的實際「當下時刻」：db/m1 用起始時間標籤（09:09 那根要到
# 09:10:00 才會完整收完、出現在 m1_live 裡），所以 predict_live() 要在
# minute_str 對到 09:10 時才去讀 CONFIRM_TIME 那根、做確認判斷。
CONFIRM_CHECK_TIME = (9, 10)

# live 現在要橫跨 09:03（記錄Stage1候選）~ 09:10（Stage2確認+進場）這段，
# 不是單一分鐘；main/live_trader.py 的 on_minute() 每分鐘都會呼叫
# predict_live()，內部依 minute_str 分流處理 09:03/09:10，其餘分鐘不動作。
SESSION_START = (9, 3)
SESSION_END = (9, 11)

# ── Triple Barrier（空單，動態 ATR，2026-08-04 三次改版）──────────────────
# 固定 ±3% 已驗證邊際不穩定（見兩階段驗證結果），改成用個股自身波動（日K
# ATR14）當停利/停損距離，取代全市場統一的固定%。TP=SL=1×day_atr（對稱），
# 比照 strategy/orb/features.py 的 day_atr 寫法（日K True Range 滾動14日
# 平均、shift(1)避免偷看未來、除以開盤價正規化），repo 沒有共用ATR工具，
# 這是既有慣例（mkt/vwap_ml 也是各自抄一份）。TP/SL 的參考進場價固定用
# Stage1 的 m3_close（不是 Stage2 的 confirm_price），維持「延續確認只決定
# 要不要進場、不影響停利停損距離」的語意。
ATR_WINDOW = 14
FORCE_EXIT_TIME = "13:25"  # 時間牆：兩個barrier都沒觸發就在這根收盤價強制平倉

# ── 模型 ──────────────────────────────────────────────────────────────
MODEL_TYPE = os.environ.get("LIMITUP_FADE_ML_MODEL_TYPE", "lgbm")
THRESHOLD = float(os.environ.get("LIMITUP_FADE_ML_THRESHOLD", "0.6"))

# ── 回測成本假設（比照 backtest/intraday_platform.py::intraday_backtest() 的預設值，
# 維持跟其他策略一致的手續費/交易稅假設，不另外校正當沖稅減半，這是既有共用引擎
# 就有的簡化，這裡沿用不特別修正）──────────────────────────────────────
FEE_RATE = 0.001425
TAX_RATE = 0.003


def events_cache_path(start_date: str | None) -> Path:
    if start_date is None:
        return CACHE_DIR / "limitup_fade_ml_events.parquet"
    return CACHE_DIR / f"limitup_fade_ml_events_{start_date}.parquet"
