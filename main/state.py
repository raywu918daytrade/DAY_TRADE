"""
即時交易的執行期共用狀態。跨 _daily_refresh / on_minute / collector 執行緒
共用讀寫的東西全部包在這裡，main/live_trader.py 建立唯一一份 AppState，
其他模組（premarket.py/backfill.py/collector.py）都吃這個物件的參照，
不再用模組級 global 變數。
"""
import pandas as pd


class StrategyState:
    """單一策略模組的執行期狀態（可同時存在多個，見 AppState.strategies）。"""

    def __init__(self, name: str, module):
        self.name = name
        self.module = module
        self.session_start = module.SESSION_START
        self.session_end = module.SESSION_END
        self.load_model = module.load_model
        self.predict_live = module.predict_live
        self.model = None
        self.prewarm_cache: dict = {}


class AppState:
    def __init__(self):
        # 策略介面：main/strategy_loader.py 載入後，main/live_trader.py 逐個包成
        # StrategyState 填進這裡，key 是策略名（例如 "orb"、"rally"）。可以同時
        # 存在多個，on_minute() 會逐一呼叫每個策略自己的 predict_live()。
        self.strategies: dict[str, StrategyState] = {}

        # 盤前資料（main/premarket.py 填入，_daily_refresh 每天重算）：候選股
        # 清單/日K 是所有策略共用的候選池，不分策略；prewarm_cache 才是
        # 各策略自己的，存在對應的 StrategyState 裡。
        self.tickers: dict = {}
        self.day_trade_stocks: set | None = None
        self.day: pd.DataFrame = pd.DataFrame()

        # 交易引擎（main/live_trader.py 依 TRADE_MODE 建立一次，之後只讀）。
        # 2026-07-13：交易先暫停，多策略同時跑訊號時怎麼分資金/處理同股票
        # 衝突還沒設計，見 main/config.py 的 STRATEGY_MODULES 說明。
        self.executor = None
