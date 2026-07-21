"""
rally 固定用 lgbm 模型的即時交易介面 — 跟 strategy/rally/xgb 是同一策略、不同
模型各自掛成獨立策略的樣版，見 strategy/rally/xgb/live.py 檔頭說明。

THRESHOLD 查 strategy/rally/config.py 的 THRESHOLD_BY_MODEL["lgbm"]，不要直接
import THRESHOLD（見 strategy/rally/xgb/live.py 的說明）。
"""

from strategy.rally.config import SESSION_END, SESSION_START, THRESHOLD_BY_MODEL
from strategy.rally.predict import predict_live
from strategy.rally.train import load_model_by_type

THRESHOLD = THRESHOLD_BY_MODEL["lgbm"]


def load_model():
    return load_model_by_type("lgbm")


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD"]
