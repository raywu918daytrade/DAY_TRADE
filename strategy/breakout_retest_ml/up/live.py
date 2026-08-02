"""
breakout_retest_ml 做多 live 介面。

路徑 strategy.breakout_retest_ml.up.live → 策略名 breakout_retest_ml_up。
只固定 DIRECTIONS={"up"}；模型與 SESSION 沿用本體。
"""

from strategy.breakout_retest_ml.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.breakout_retest_ml.predict import predict_live
from strategy.breakout_retest_ml.train import load_model_by_type

DIRECTIONS = {"up"}


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
