"""
live_trader.py 的執行期設定：集中管理從 .env 讀出來的常數，以及
settings.json 優先、.env 當 fallback 的動態設定（TOTAL_CAPITAL）。
live_trader.py 只 import 這裡的值，不直接讀 os.environ，改設定只要
改這支檔案或 .env，不用動流程邏輯。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

# 策略模組（可同時跑多個）：改 .env 的 STRATEGY_MODULES（逗號分隔）就能切換，
# 不用改 live_trader.py。每個策略模組（例如 strategy/rally/live.py）都要暴露
# 同一組介面：load_model() / predict_live(...) / SESSION_START / SESSION_END。
# 2026-07-13：交易執行先暫停（見 main/live_trader.py 的 TRADE_MODE），多策略
# 同時跑只做「各自推論、結果分開傳到前端，最終由使用者決定」，還沒有處理
# 多策略同時下單的資金分配/訊號衝突問題——等實盤穩定、真的要多策略同時
# 交易時，要先設計好這塊，不要假設現在的架構已經支援。
STRATEGY_MODULES = [
    s.strip() for s in os.environ.get("STRATEGY_MODULES", "strategy.orb.lgbm.live").split(",") if s.strip()
]

# 多策略「前N名重疊」比對：每個策略各自依 proba 排名取前N名（不管有沒有過
# 該策略自己的門檻），同一支股票同時出現在2個以上策略的前N名，視為多模型
# 共識訊號，額外推送給前端參考（見 main/live_trader.py 的 on_minute()）。
CONSENSUS_TOP_N = int(os.environ.get("CONSENSUS_TOP_N", "10"))
# 進共識前N名比對的信心度下限（2026-08-04要求）：proba 沒到這個門檻的，
# 就算排進前N名也不列入共識比對，避免信心度很低（例如10%、20%）的股票只
# 因為「排名還算前面」就被兩個策略同時湊到、誤判成共識。跟各策略自己的
# threshold（決定要不要送出訊號）是兩件事，這個只影響共識判斷。
CONSENSUS_MIN_PROBA = float(os.environ.get("CONSENSUS_MIN_PROBA", "0.5"))

# 多空衝突（反轉）訊號：同一支股票同一分鐘，如果「某策略」看多信心度 ≥ 這個
# 門檻、「某策略」（可以是同一個或不同策略）看空信心度也 ≥ 這個門檻，代表
# 模型之間對這支股票的方向嚴重分歧，標記成「衝突/反轉」給前端參考（見
# main/live_trader.py 的 on_minute()）。跟 CONSENSUS_TOP_N 的共識邏輯相反：
# 共識要求同方向、衝突要求異方向。
CONFLICT_THRESHOLD = float(os.environ.get("CONFLICT_THRESHOLD", "0.6"))

# 頁首固定追蹤股票（不經過策略候選篩選，例如 ETF 0050）：on_minute() 每分鐘
# 收到這些股票的 m1 就查前一交易日收盤價、算漲跌幅，push_quote() 推給前端。
WATCHLIST_QUOTES = [
    s.strip() for s in os.environ.get("WATCHLIST_QUOTES", "0050").split(",") if s.strip()
]

TRADE_MODE = os.environ.get("TRADE_MODE", "off")  # off | paper | sim | live
# 2026-07-21：原本這裡有一個全域 THRESHOLD 給所有策略共用，拆成各策略自己
# 一個（見 strategy/{orb,rally,mkt}/config.py 的 THRESHOLD、
# main/state.py::StrategyState 的說明），前端 settings 的 threshold 設定仍是
# 全域覆蓋值（未設定時才各自 fallback 各策略的預設值）。
FORCE_CLOSE_HOUR = int(os.environ.get("FORCE_CLOSE_HOUR", "13"))
FORCE_CLOSE_MIN = int(os.environ.get("FORCE_CLOSE_MIN", "25"))

_TOTAL_CAPITAL_ENV = float(os.environ.get("TOTAL_CAPITAL", "1000000"))


def _resolve_capital() -> float:
    """settings.json 存在且有 total_capital → 優先使用，否則回落 .env。"""
    try:
        from api import get_setting

        v = get_setting("total_capital")
        if v is not None:
            print(f"[設定] 總額度從 settings 載入：{float(v):,.0f}")
            return float(v)
    except Exception:
        pass
    return _TOTAL_CAPITAL_ENV


TOTAL_CAPITAL = _resolve_capital()
