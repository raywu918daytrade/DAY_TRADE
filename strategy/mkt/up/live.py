"""
mkt 只送做多訊號的即時交易介面 — 跟 strategy/mkt/down 是同一策略、不同方向
各自掛成獨立策略的樣版（2026-07-25討論：mkt 本來是多空混在同一欄，使用者
覺得看不清楚，改成比照 strategy/orb/xgb、strategy/orb/lgbm 那種「同一策略
拆成獨立變體」的做法，拆成 mkt_up/mkt_down 兩個策略）。放在 strategy/mkt/
底下巢狀一層，路徑 strategy.mkt.up.live 會被解析成策略名 "mkt_up"。

模型、SESSION_START/SESSION_END/predict_live/build_prewarm_cache 全部沿用
strategy/mkt 本體的實作，只固定 DIRECTIONS={"up"}——main/live_trader.py 的
on_minute() 依這個過濾 predict_live() 回傳結果，跌訊號不會流進這個策略的
monitoring/signals/共識比對，只會出現在 strategy/mkt/down。
"""

from strategy.mkt.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.mkt.predict import build_prewarm_cache, predict_live
from strategy.mkt.train import load_model_by_type

DIRECTIONS = {"up"}


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "build_prewarm_cache", "THRESHOLD", "DIRECTIONS"]
