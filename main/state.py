"""
即時交易的執行期共用狀態。跨 _daily_refresh / on_minute / collector 執行緒
共用讀寫的東西全部包在這裡，main/live_trader.py 建立唯一一份 AppState，
其他模組（premarket.py/backfill.py/collector.py）都吃這個物件的參照，
不再用模組級 global 變數。
"""
import threading

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
        # 各策略自己的信心度門檻預設值（module.THRESHOLD，見各策略 config.py 的
        # 說明）——main/live_trader.py 每分鐘先查前端 settings 的全域 threshold，
        # 沒設定才 fallback 這個，取代原本全部策略共用一個 main/config.py 全域
        # THRESHOLD 的做法（2026-07-21 討論：三個模型的機率校準不一定一樣，
        # 共用同一個門檻可能對某些策略太鬆或太嚴）。
        self.threshold = module.THRESHOLD
        # 這個策略允許送出的訊號方向（module.DIRECTIONS，見各策略 live.py 的
        # 說明）——2026-07-23討論：orb/rally 目前只有做多訊號堪用（{"up"}），
        # mkt 多空都送（{"up","down"}）。on_minute() 依這個過濾
        # predict_live() 回傳結果，不屬於這個集合的方向不會送到前端；也是
        # 共識訊號拆成「多方」「空方」兩欄分開比對時，判斷這個策略算不算
        # 進哪一欄的依據。
        self.directions = module.DIRECTIONS
        # 這個策略要不要吃即時 tick 資料（module.USES_TICKS，選填，見
        # strategy/breakout_retest_ml/up/live.py 的說明）——用 getattr 給
        # 預設值 False，因為大部分策略模組不會宣告這個屬性（跟強制要有的
        # DIRECTIONS 不同）。main/live_trader.py::on_minute() 依這個決定
        # 要不要把 ticks_by_stock 塞進 predict_live() 的呼叫參數——沒有
        # **kwargs 的策略（orb/mkt/cnn/vwap_ml/vwap_dl）硬塞這個參數會
        # TypeError，不能無條件全部策略都傳。
        self.uses_ticks = getattr(module, "USES_TICKS", False)
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

        # 盤中重啟時，db/m1_live/ 可能有缺口（連線建立前那段沒有即時資料），
        # fubon/marketdata_ws.py::FubonM1Collector._backfill_intraday() 會用
        # REST 補這個缺口，補完才 set() 這個 flag（2026-07-25討論）。
        # main/live_trader.py::on_minute() 用這個決定要不要先跳過推論——沒補完
        # 前，m1_live 資料不完整，貿然推論算出來的機率不可靠，寧可先不推論
        # （K線/報價還是照樣更新，不受影響），等補完再開始正常判斷訊號。
        # 開機前如果 db/m1_live/ 本來就沒有缺口（例如開盤前就啟動），這個
        # backfill 幾乎瞬間跑完，不會有感覺得到的延遲。
        self.backfill_done = threading.Event()

        # 富邦 WebSocket realtime_token（見 fubon/marketdata_ws.py::
        # FubonM1Collector.__init__ 的 on_token_ready callback）：
        # fubon/tick_ws.py::FubonTickCollector 重用這個 token 開 tick 訂閱
        # 連線，不用再登入一次富邦帳號（避免重複登入的 race condition，
        # 跟 backfill_done 一樣是給不同執行緒溝通用的旗標）。
        self.fubon_ws_token: str | None = None
        self.fubon_ws_token_ready = threading.Event()
