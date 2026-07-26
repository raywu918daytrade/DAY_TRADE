"""
vwap_ml 只送做空訊號的即時交易介面 — 跟 strategy/vwap_ml/up 是同一策略、
不同方向各自掛成獨立策略的樣版，見 strategy/vwap_ml/up/live.py 檔頭說明。

只固定 DIRECTIONS={"down"}，模型、SESSION_START/SESSION_END/predict_live
全部沿用 strategy/vwap_ml 本體的實作。
"""

from strategy.vwap_ml.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.vwap_ml.predict import predict_live
from strategy.vwap_ml.train import load_model_by_type

DIRECTIONS = {"down"}


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
