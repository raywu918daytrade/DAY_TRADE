"""
切換策略／模型：用 .env 的 STRATEGY_MODULE（main/config.py 讀出來）動態載入
對應的策略模組。每個策略模組（例如 strategy/rally/live.py）都要暴露同一組
固定介面：SESSION_START / SESSION_END / load_model() / predict_live(...)。
"""
import importlib


def load_strategy(module_path: str):
    """動態載入策略模組，回傳模組本身。呼叫端自己從回傳值取
    SESSION_START/SESSION_END/load_model/predict_live，填進 AppState。"""
    return importlib.import_module(module_path)
