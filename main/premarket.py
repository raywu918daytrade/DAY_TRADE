"""
盤前資料準備：當沖候選清單、日K、策略盤前快取。開機 bootstrap 跟
_daily_refresh() 的每日排程共用這裡的函式，邏輯只寫一份。

錯誤處理刻意留給呼叫端：這裡的函式不吞例外，失敗直接往外拋，因為
bootstrap 跟 _daily_refresh() 對同一種失敗要印的訊息、要做的 fallback
不一樣（見 main/live_trader.py）。
"""
from data.data_manager import load_d1
from fubon.subscribe_list import build_and_save_subscribe_list, load_candidates
from main.config import DAILY_REFRESH_TICKERS
from strategy.prewarm import build_prewarm_cache


def refresh_tickers(state) -> None:
    """更新當沖候選清單，寫入 state.tickers / state.day_trade_stocks。

    直接呼叫 fubon.subscribe_list.build_and_save_subscribe_list()：那是富邦
    WebSocket 訂閱清單唯一的來源，這裡重用同一份，避免候選股跟 WebSocket
    實際訂閱的股票不一致（先前這裡走 Fugle、WebSocket 那邊走富邦，兩邊
    各自過濾，理論上該一致但沒有保證）。API 回傳空值（例如非盤中）不算
    例外，視為「不過濾」。

    only_4digit=True：濾掉5碼以上的槓桿/反向/主動型ETF（例如00631L、
    00632R），這類ETF成交量常常很大，會擠掉均量排序前1000名裡真正的個股
    （2026-07-14 發現，見 fubon/subscribe_list.py 的說明）。0050 這種4碼ETF
    不受影響，因為 idx_* 特徵需要它。

    DAILY_REFRESH_TICKERS=false（.env）時不重新計算，改讀既有的
    db/fubon_subscribe/subscribe_list.parquet——這支函式在開機時跟每天
    06:00 都會被呼叫，兩個時機都會套用這個開關，維持候選股清單不隨每天
    均量排名變動（2026-07-14：要讓清單跟 FinMind 歷史資料補齊的範圍
    保持一致，暫時關閉）。
    """
    if not DAILY_REFRESH_TICKERS:
        print("  DAILY_REFRESH_TICKERS=false，讀取既有候選清單，不重新計算", flush=True)
        df = load_candidates()
    else:
        df = build_and_save_subscribe_list(only_4digit=True)
    if df.empty:
        print("  警告：無法取得候選股清單（非盤中或富邦 API 失敗），不過濾股票", flush=True)
        state.tickers = {}
        state.day_trade_stocks = None
        return
    state.tickers = df.set_index("stock_id")["name"].to_dict()
    state.day_trade_stocks = set(state.tickers.keys()) or None  # None = 不過濾


def refresh_day(state) -> None:
    """載入日K（均量過濾），需要先呼叫 refresh_tickers() 設好
    state.day_trade_stocks。回傳的候選股集合會覆寫 state.day_trade_stocks
    （均量過濾後可能變少）。"""
    state.day, state.day_trade_stocks = load_d1(state.day_trade_stocks)


def refresh_prewarm(state) -> None:
    """重算每個策略各自的盤前快取，寫入對應 StrategyState.prewarm_cache。
    策略不需要就回傳空 dict（見 strategy/prewarm.py），predict_live() 展開
    空 dict 等於沒有額外參數。"""
    for s in state.strategies.values():
        s.prewarm_cache = build_prewarm_cache(s.module)
