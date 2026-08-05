"""
limitup_fade_ml 即時交易介面（只做空單，DIRECTIONS={"down"}）。

現行 executor 只支援開多單，本策略訊號目前僅供監控（真下空單另議，見
strategy/limitup_fade_ml/README.md）。尚未加進 .env 的 STRATEGY_MODULES，
需要上線監控時再手動加入。
"""

from strategy.limitup_fade_ml.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.limitup_fade_ml.predict import predict_live
from strategy.limitup_fade_ml.train import load_model_by_type

DIRECTIONS = {"down"}


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
