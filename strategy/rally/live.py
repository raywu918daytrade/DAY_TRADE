"""
rally 策略對外的即時交易介面 — live_trader.py 只透過這支檔案跟 rally 互動

暴露固定介面（load_model / predict_live / SESSION_START / SESSION_END），
live_trader.py 用 STRATEGY_MODULE 環境變數切換要載入哪個模組，不用改
live_trader.py 本身的程式碼。以後如果又新增別的策略模組，一樣要暴露這組介面。

train.py/validate.py/features.py 這些訓練/驗證用的東西，live_trader.py 完全
不會碰到，依賴方向永遠是單向的：live_trader → live.py → predict.py/train.py。

目前 live 用的是 XGBoost（m1_xgb.pkl）——三個模型裡目前表現最好的一個，
之後想換 RFC/LGBM 就改這裡的 import。
"""

from strategy.rally.config import SESSION_END, SESSION_START
from strategy.rally.predict import predict_live
from strategy.rally.train import load_model_xgb as load_model

__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END"]
