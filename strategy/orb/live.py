"""
orb 策略對外的即時交易介面 — live_trader.py 只透過這支檔案跟 orb 互動

暴露固定介面（load_model / predict_live / SESSION_START / SESSION_END），
比照 strategy/rally/live.py 的作法。live_trader.py 用 STRATEGY_MODULE 環境
變數切換要載入哪個模組（改成 "strategy.orb.live"），不用改 live_trader.py
本身的程式碼。

train.py/validate.py/features.py 這些訓練/驗證用的東西，live_trader.py 完全
不會碰到，依賴方向永遠是單向的：live_trader → live.py → predict.py/train.py。

目前 live 用的是 LightGBM（m1_orb_lgbm.pkl）——見 strategy/orb/validate.py
的 compare_report()，同一份測試集上表現比 XGB 好。
"""

from strategy.orb.config import SESSION_END, SESSION_START
from strategy.orb.predict import build_prewarm_cache, predict_live
from strategy.orb.train import load_model_xgb as load_model

__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "build_prewarm_cache"]
