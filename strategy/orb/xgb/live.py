"""
orb 固定用 xgb 模型的即時交易介面 — 跟 strategy/orb/lgbm 是同一策略、不同模型
各自掛成獨立策略的樣版，比照 strategy/rally/xgb/live.py 的做法（2026-07-22
討論：不要平均/投票，2個模型各自獨立判斷、獨立進出場，直接當成
main/config.py::STRATEGY_MODULES裡2個不同的策略掛進去）。放在 strategy/orb/
底下巢狀一層，路徑 strategy.orb.xgb.live 會被解析成策略名 "orb_xgb"。

只固定 load_model() 要用哪個演算法，SESSION_START/SESSION_END/predict_live/
build_prewarm_cache 全部沿用 strategy/orb 本體的實作，不重複寫一份。

THRESHOLD 查 strategy/orb/config.py 的 THRESHOLD_BY_MODEL["xgb"]，不要直接
import該檔案的 THRESHOLD——那個是跟著 ORB_MODEL_TYPE 走的，可能對到別的模型
（2026-07-22發現的bug，見 config.py 的說明）。

DIRECTIONS：只送做多訊號（2026-07-23討論：orb 的標籤是二分類 Triple
Barrier，「跌」代表突破失敗，不是放空訊號）。
"""

from strategy.orb.config import SESSION_END, SESSION_START, THRESHOLD_BY_MODEL
from strategy.orb.predict import build_prewarm_cache, predict_live
from strategy.orb.train import load_model_by_type

THRESHOLD = THRESHOLD_BY_MODEL["xgb"]
DIRECTIONS = {"up"}


def load_model():
    return load_model_by_type("xgb")


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "build_prewarm_cache", "THRESHOLD", "DIRECTIONS"]
