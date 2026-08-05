"""
limitup_fade_ml 做空 live 介面。

路徑 strategy.limitup_fade_ml.down.live → 策略名 limitup_fade_ml_down。
"""

from strategy.limitup_fade_ml_my.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.limitup_fade_ml_my.predict import predict_live
from strategy.limitup_fade_ml_my.train import load_model_by_type

DIRECTIONS = {"down"}


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = [
    "load_model",
    "predict_live",
    "SESSION_START",
    "SESSION_END",
    "THRESHOLD",
    "DIRECTIONS",
]
