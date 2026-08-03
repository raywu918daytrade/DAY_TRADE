"""
breakout_retest_ml 做多 live 介面。

路徑 strategy.breakout_retest_ml.up.live → 策略名 breakout_retest_ml_up。
只固定 DIRECTIONS={"up"}；模型與 SESSION 沿用本體。
"""

from strategy.breakout_retest_ml.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.breakout_retest_ml.predict import predict_live
from strategy.breakout_retest_ml.train import load_model_by_type

DIRECTIONS = {"up"}
# 這個策略的 predict_live() 吃 ticks_by_stock 參數（Tick 硬過濾，見
# predict.py 的說明）。main/live_trader.py::on_minute() 依這個決定要不要組
# ticks_by_stock 傳進來（見 main/state.py::StrategyState.uses_ticks 的說明），
# 資料來源是 fubon/tick_ws.py 收集的 db/tick_live/。
USES_TICKS = True


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS", "USES_TICKS"]
