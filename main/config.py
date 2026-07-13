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

# 策略模組切換：改 .env 的 STRATEGY_MODULE 就能換策略，不用改 live_trader.py。
# 每個策略模組（例如 strategy/rally/live.py）都要暴露同一組介面：
# load_model() / predict_live(...) / SESSION_START / SESSION_END
STRATEGY_MODULE = os.environ.get("STRATEGY_MODULE", "strategy.orb.live")

TRADE_MODE = os.environ.get("TRADE_MODE", "off")  # off | paper | sim | live
THRESHOLD = float(os.environ.get("THRESHOLD", "0.55"))
FORCE_CLOSE_HOUR = int(os.environ.get("FORCE_CLOSE_HOUR", "13"))
FORCE_CLOSE_MIN = int(os.environ.get("FORCE_CLOSE_MIN", "25"))

# 分K收集器：rest（Fugle REST）| fubon_ws（富邦 WebSocket）。實際值看 .env，
# candles channel payload 格式尚未在平日開盤驗證過，見 fubon/marketdata_ws.py 的說明。
M1_COLLECTOR = os.environ.get("M1_COLLECTOR", "rest")

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
