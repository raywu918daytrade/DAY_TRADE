"""
vwap_ml 只送做多訊號的即時交易介面 — 跟 strategy/vwap_ml/down 是同一策略、
不同方向各自掛成獨立策略的樣版，比照 strategy/mkt/up/live.py 的做法。放在
strategy/vwap_ml/ 底下巢狀一層，路徑 strategy.vwap_ml.up.live 會被解析成
策略名 "vwap_ml_up"。

模型、SESSION_START/SESSION_END/predict_live 全部沿用 strategy/vwap_ml
本體的實作，只固定 DIRECTIONS={"up"}——main/live_trader.py 的 on_minute()
依這個過濾 predict_live() 回傳結果，做空訊號不會流進這個策略的
monitoring/signals，只會出現在 strategy/vwap_ml/down。

vwap_ml 沒有 build_prewarm_cache()（見 predict.py::predict_live() 的
說明——VWAP z-score 只需要當天資料，沒有跨分鐘要快取的東西），跟 rally
一樣不用實作，strategy/prewarm.py 對缺這個屬性的模組會自動當作 {}。
"""

from strategy.vwap_ml.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.vwap_ml.predict import predict_live
from strategy.vwap_ml.train import load_model_by_type

DIRECTIONS = {"up"}


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
