"""
rally 固定用 xgb 模型的即時交易介面 — 跟 strategy/rally/lgbm 是同一策略、不同
模型各自掛成獨立策略的樣版（2026-07-22 討論：不要平均/投票，2個模型各自獨立
判斷、獨立進出場，直接當成main/config.py::STRATEGY_MODULES裡2個不同的策略
掛進去，比照main/strategy_loader.py「策略名取模組路徑中間段接起來」的設計，
不用改main/live_trader.py或main/state.py）。放在 strategy/rally/ 底下巢狀
一層，路徑 strategy.rally.xgb.live 會被解析成策略名 "rally_xgb"。

只固定 load_model() 要用哪個演算法，SESSION_START/SESSION_END/predict_live
全部沿用 strategy/rally 本體的實作，不重複寫一份特徵/推論邏輯。

THRESHOLD 查 strategy/rally/config.py 的 THRESHOLD_BY_MODEL["xgb"]，不要直接
import該檔案的 THRESHOLD——那個是跟著 RALLY_MODEL_TYPE 走的，可能對到別的
模型（2026-07-22發現的bug，見 config.py 的說明）。
"""

from strategy.rally.config import SESSION_END, SESSION_START, THRESHOLD_BY_MODEL
from strategy.rally.predict import predict_live
from strategy.rally.train import load_model_by_type

THRESHOLD = THRESHOLD_BY_MODEL["xgb"]


def load_model():
    return load_model_by_type("xgb")


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD"]
