"""
策略共用的盤前預算快取入口 — main/live_trader.py 開盤前呼叫一次
build_prewarm_cache(strategy_module)，把結果快取起來，之後 on_minute() 每
分鐘呼叫 predict_live() 時直接 **cache 展開傳進去，避免策略內部需要「過去
歷史彙總表」的部分（例如 strategy/orb 的 open_vol_history/hourly_tr_history）
每分鐘重複計算同樣的結果。

策略模組（例如 strategy/orb/live.py）如果需要盤前快取，要暴露一個
build_prewarm_cache() -> dict，dict 的 key 必須跟該策略 predict_live()
接受的參數名一致（因為會直接 spread 進去）。不需要的策略（目前的 rally）
不用實作這個函式，這裡會自動跳過、回傳空 dict——predict_live() 收到空
dict 展開等於沒有額外傳參數，行為不變，不會 breaking rally。

以後任何策略需要盤前建快取，都比照 strategy/orb/predict.py 的
build_prewarm_cache() 做法：寫一個同名函式、在該策略的 live.py re-export，
不用改這支檔案或 main/live_trader.py。
"""


def build_prewarm_cache(strategy_module, day_trade_stocks: set | None = None) -> dict:
    """strategy_module 是 main/live_trader.py 用 STRATEGY_MODULE 動態載入好的
    那個策略模組（例如 strategy.orb.live），這裡不重複 import，直接吃現成的。

    day_trade_stocks：今天的當沖候選清單（main/premarket.py::refresh_tickers()
    算好的，見 refresh_prewarm() 的說明），2026-07-25 起統一傳給每個策略的
    build_prewarm_cache()——不是每個策略都用得到（例如 mkt 自己算流動性排名，
    不受這份清單限制），用不到的策略在自己的 build_prewarm_cache() 簽名裡
    收下、忽略即可，維持所有策略同一組介面。"""
    builder = getattr(strategy_module, "build_prewarm_cache", None)
    if builder is None:
        return {}
    return builder(day_trade_stocks=day_trade_stocks)
