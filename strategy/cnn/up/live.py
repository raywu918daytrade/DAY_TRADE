"""
cnn 只送做多訊號的即時交易介面 — 跟 strategy/cnn/down 是同一策略、不同方向
各自掛成獨立策略的樣版，比照 strategy/mkt/up、strategy/mkt/down 的做法
（2026-07-25）。放在 strategy/cnn/ 底下巢狀一層，路徑 strategy.cnn.up.live
會被解析成策略名 "cnn_up"。

只固定 DIRECTIONS={"up"}，模型、SESSION_START/SESSION_END/predict_live
全部沿用 strategy/cnn 本體的實作——main/live_trader.py 的 on_minute() 依
DIRECTIONS 過濾 predict_live() 回傳結果，跌訊號不會流進這個策略的
monitoring/signals/共識比對，只會出現在 strategy/cnn/down。

不需要 build_prewarm_cache()：cnn 目前沒有像 orb/mkt 那樣需要盤前算好的
歷史彙總表（見 strategy/prewarm.py 的說明，沒實作會自動跳過、回傳空dict，
不影響 predict_live() 正常運作）。
"""

from strategy.cnn.config import SESSION_END, SESSION_START, THRESHOLD
from strategy.cnn.predict import predict_live
from strategy.cnn.train import load_model

DIRECTIONS = {"up"}

__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
