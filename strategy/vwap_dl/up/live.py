"""
vwap_dl 只送做多訊號的即時交易介面 — 跟 strategy/vwap_dl/down 是同一策略、
不同方向各自掛成獨立策略的樣版。

只固定 DIRECTIONS={"up"}，模型、SESSION_START/SESSION_END/predict_live
全部沿用 strategy/vwap_dl 本體的實作。

方向邏輯（predict.py::_direction_probas）：
    下軌（z < -2σ，價格低於 VWAP）→ 回歸到 VWAP → 做多 (up)
"""

from strategy.vwap_dl.config import SESSION_END, SESSION_START, THRESHOLD
from strategy.vwap_dl.predict import predict_live
from strategy.vwap_dl.train import load_model as _load_model

DIRECTIONS = {"up"}


def load_model():
    return _load_model()


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
