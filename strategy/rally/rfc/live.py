"""
rally 固定用 rfc 模型的即時交易介面 — 跟 strategy/rally/xgb、strategy/rally/lgbm
是同一策略、不同模型各自掛成獨立策略的樣版，見 strategy/rally/xgb/live.py 檔頭
說明。放在 strategy/rally/ 底下巢狀一層，路徑 strategy.rally.rfc.live 會被解析
成策略名 "rally_rfc"。

只固定 load_model() 要用哪個演算法，SESSION_START/SESSION_END/predict_live
全部沿用 strategy/rally 本體的實作，不重複寫一份特徵/推論邏輯。

THRESHOLD 查 strategy/rally/config.py 的 THRESHOLD_BY_MODEL["rfc"]，不要直接
import該檔案的 THRESHOLD——那個是跟著 RALLY_MODEL_TYPE 走的，可能對到別的
模型（見 config.py 的說明）。

2026-08-06：三個模型（RFC/XGB/LGBM）用 2021年起+固定400支重訓後，逐分鐘
信心度桶比對顯示只有 9:00~9:30 這個開盤窗口的機率校準夠乾淨可信任（見
config.py 的 SESSION_START/SESSION_END 說明），RFC 又是三個裡校準最好、
precision 隨信心度單調遞增的一個（0.50-0.55 桶 83.6%，XGB/LGBM 同一桶
只有 59~63%），所以三個模型裡優先掛這個上線。

DIRECTIONS：這個策略只送做多訊號，理由同 xgb/live.py 的說明（rally 的標籤
是「先碰停利=漲/先碰停損=跌」，「跌」代表這次做多失敗，不是可以放空的訊號）。
"""

from strategy.rally.config import SESSION_END, SESSION_START, THRESHOLD_BY_MODEL
from strategy.rally.predict import predict_live
from strategy.rally.train import load_model_by_type

THRESHOLD = THRESHOLD_BY_MODEL["rfc"]
DIRECTIONS = {"up"}


def load_model():
    return load_model_by_type("rfc")


__all__ = ["load_model", "predict_live", "SESSION_START", "SESSION_END", "THRESHOLD", "DIRECTIONS"]
