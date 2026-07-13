"""
切換策略／模型：用 .env 的 STRATEGY_MODULES（main/config.py 讀出來，逗號分隔）
動態載入對應的策略模組。每個策略模組（例如 strategy/rally/live.py）都要暴露
同一組固定介面：SESSION_START / SESSION_END / load_model() / predict_live(...)。

可以同時載入多個策略模組（見 main/state.py 的 StrategyState），策略名取模組
路徑倒數第二段（例如 "strategy.orb.live" → "orb"）當 key，用來在推論結果、
SSE 推送裡標記「這筆是哪個策略產生的」。
"""
import importlib


def load_strategies(module_paths: list[str]) -> dict:
    """載入多個策略模組，回傳 {策略名: 模組}。"""
    strategies = {}
    for path in module_paths:
        module = importlib.import_module(path)
        name = path.split(".")[-2]
        if name in strategies:
            raise ValueError(f"策略名稱重複：{name}（來自 {path}），策略名取模組路徑倒數第二段，不能撞名")
        strategies[name] = module
    return strategies
