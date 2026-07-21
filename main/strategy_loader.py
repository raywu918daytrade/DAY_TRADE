"""
切換策略／模型：用 .env 的 STRATEGY_MODULES（main/config.py 讀出來，逗號分隔）
動態載入對應的策略模組。每個策略模組（例如 strategy/rally/live.py）都要暴露
同一組固定介面：SESSION_START / SESSION_END / load_model() / predict_live(...)。

可以同時載入多個策略模組（見 main/state.py 的 StrategyState），策略名取模組
路徑「strategy.」跟「.live」中間那一段，多層用底線接起來當 key，用來在推論
結果、SSE 推送裡標記「這筆是哪個策略產生的」：
    "strategy.orb.live"      → "orb"
    "strategy.orb.xgb.live"  → "orb_xgb"（同一策略、固定模型的變體用巢狀資料
                                夾放，例如 strategy/orb/xgb/live.py，見
                                2026-07-22 討論：orb_xgb/orb_lgbm 原本是跟
                                strategy/orb 平行的資料夾，改成巢狀後路徑更
                                看得出從屬關係）
"""
import importlib


def _strategy_name(path: str) -> str:
    """"strategy.orb.xgb.live" → "orb_xgb"：取頭尾（"strategy"/"live"）中間
    的部分，多層路徑用底線接起來，深度1層或多層都適用（見檔頭說明）。"""
    return "_".join(path.split(".")[1:-1])


def load_strategies(module_paths: list[str]) -> dict:
    """載入多個策略模組，回傳 {策略名: 模組}。"""
    strategies = {}
    for path in module_paths:
        module = importlib.import_module(path)
        name = _strategy_name(path)
        if name in strategies:
            raise ValueError(f"策略名稱重複：{name}（來自 {path}），策略名取模組路徑中間段接起來，不能撞名")
        strategies[name] = module
    return strategies
