"""
即時交易的執行期共用狀態。跨 _daily_refresh / on_minute / collector 執行緒
共用讀寫的東西全部包在這裡，main/live_trader.py 建立唯一一份 AppState，
其他模組（premarket.py/backfill.py/collector.py）都吃這個物件的參照，
不再用模組級 global 變數。
"""
import pandas as pd


class AppState:
    def __init__(self):
        # 策略介面（main/strategy_loader.py 載入後填入，之後只讀）
        self.strategy_module = None
        self.session_start = None
        self.session_end = None
        self.load_model = None
        self.predict_live = None
        self.model = None

        # 盤前資料（main/premarket.py 填入，_daily_refresh 每天重算）
        self.tickers: dict = {}
        self.day_trade_stocks: set | None = None
        self.day: pd.DataFrame = pd.DataFrame()
        self.prewarm_cache: dict = {}

        # 交易引擎（main/live_trader.py 依 TRADE_MODE 建立一次，之後只讀）
        self.executor = None
