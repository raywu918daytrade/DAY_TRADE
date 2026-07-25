"""
cnn 只送做空訊號的即時交易介面 — 跟 strategy/cnn/up 是同一策略、不同方向
各自掛成獨立策略的樣版，見 strategy/cnn/up/live.py 檔頭說明。

只固定 DIRECTIONS={"down"}，模型、SESSION_START/SESSION_END/predict_live
全部沿用 strategy/cnn 本體的實作。
"""

from strategy.cnn.config import SESSION_END, SESSION_START, THRESHOLD
from strategy.cnn.predict import predict_live
from strategy.cnn.train import load_model

DIRECTIONS = {"down"}

__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
