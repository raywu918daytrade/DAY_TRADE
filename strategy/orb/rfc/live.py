"""
orb 固定用 rfc 模型的即時交易介面 — 跟 strategy/orb/xgb、strategy/orb/lgbm 是
同一策略、不同模型各自掛成獨立策略的樣版，比照 strategy/orb/xgb/live.py 的
做法。放在 strategy/orb/ 底下巢狀一層，路徑 strategy.orb.rfc.live 會被解析
成策略名 "orb_rfc"。

只固定 load_model() 要用哪個演算法，SESSION_START/SESSION_END/predict_live/
build_prewarm_cache 全部沿用 strategy/orb 本體的實作，不重複寫一份。

THRESHOLD 查 strategy/orb/config.py 的 THRESHOLD_BY_MODEL["rfc"]。

DIRECTIONS：只送做多訊號（2026-07-23討論：orb 的標籤跟 rally 一樣是二分類
Triple Barrier，「跌」代表突破失敗，不是放空訊號）。
"""

from strategy.orb.config import SESSION_END, SESSION_START, THRESHOLD_BY_MODEL
from strategy.orb.predict import build_prewarm_cache, predict_live
from strategy.orb.train import load_model_by_type

THRESHOLD = THRESHOLD_BY_MODEL["rfc"]
DIRECTIONS = {"up"}


def load_model():
    return load_model_by_type("rfc")


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "build_prewarm_cache", "THRESHOLD", "DIRECTIONS"]
