"""
vwap_dl 只送做空訊號的即時交易介面 — 跟 strategy/vwap_dl/up 是同一策略、
不同方向各自掛成獨立策略的樣版。

只固定 DIRECTIONS={"down"}，模型、SESSION_START/SESSION_END/predict_live
全部沿用 strategy/vwap_dl 本體的實作。

方向邏輯（predict.py::_direction_probas）：
    上軌（z > +2σ，價格高於 VWAP）→ 回歸到 VWAP → 做空 (down)
"""

from strategy.vwap_dl.config import SESSION_END, SESSION_START, THRESHOLD
from strategy.vwap_dl.predict import predict_live
from strategy.vwap_dl.train import load_model as _load_model

DIRECTIONS = {"down"}


def load_model():
    return _load_model()


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
