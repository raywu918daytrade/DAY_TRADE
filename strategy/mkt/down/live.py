"""
mkt 只送做空訊號的即時交易介面 — 跟 strategy/mkt/up 是同一策略、不同方向
各自掛成獨立策略的樣版，見 strategy/mkt/up/live.py 檔頭說明。

只固定 DIRECTIONS={"down"}，模型、SESSION_START/SESSION_END/predict_live/
build_prewarm_cache 全部沿用 strategy/mkt 本體的實作。
"""

from strategy.mkt.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.mkt.predict import build_prewarm_cache, predict_live
from strategy.mkt.train import load_model_by_type

DIRECTIONS = {"down"}


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "build_prewarm_cache", "THRESHOLD", "DIRECTIONS"]
