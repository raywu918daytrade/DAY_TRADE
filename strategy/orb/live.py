"""
orb 策略對外的即時交易介面 — live_trader.py 只透過這支檔案跟 orb 互動

暴露固定介面（load_model / predict_live / SESSION_START / SESSION_END），
比照 strategy/rally/live.py 的作法。live_trader.py 用 STRATEGY_MODULE 環境
變數切換要載入哪個模組（改成 "strategy.orb.live"），不用改 live_trader.py
本身的程式碼。

train.py/validate.py/features.py 這些訓練/驗證用的東西，live_trader.py 完全
不會碰到，依賴方向永遠是單向的：live_trader → live.py → predict.py/train.py。

要用哪個模型（rfc/lgbm/xgb）由 config.MODEL_TYPE 決定（讀 .env 的
ORB_MODEL_TYPE，預設 lgbm），跟 run_backtest.py 共用同一個參數，比照
strategy/rally/live.py 的做法，改 config.py/.env 一個地方兩邊就會一起換。
"""

from strategy.orb.config import MODEL_TYPE, SESSION_END, SESSION_START, THRESHOLD
from strategy.orb.predict import build_prewarm_cache, predict_live
from strategy.orb.train import load_model_by_type


def load_model():
    return load_model_by_type(MODEL_TYPE)


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "build_prewarm_cache", "THRESHOLD"]
