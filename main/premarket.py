"""
盤前資料準備：當沖候選清單、日K、策略盤前快取。開機 bootstrap 跟
_daily_refresh() 的每日排程共用這裡的函式，邏輯只寫一份。

錯誤處理刻意留給呼叫端：這裡的函式不吞例外，失敗直接往外拋，因為
bootstrap 跟 _daily_refresh() 對同一種失敗要印的訊息、要做的 fallback
不一樣（見 main/live_trader.py）。
"""
from data.data_manager import load_d1
from fubon.subscribe_list import build_and_save_subscribe_list
from strategy.prewarm import build_prewarm_cache


def check_and_refresh_from_hf() -> None:
    """開機檢查：用 0050 的 db/d1_flags/day_flag.parquet 有沒有該有的flag，
    判斷本機 db/ 是否跟上最新排程進度。0050 是候選池強制納入的股票（見
    finmind/tick_universe.py 的 _FORCE_INCLUDE），update_daily.py 每天都
    會處理它，用它當代理指標比逐一檢查每支股票便宜。

    2026-08-19修正：一開始誤用「今天」比對——d1是收盤後才會有的整天K線，
    早上開機（開盤前/盤中）當下，今天的d1本來就不可能存在，用「今天」當
    基準會導致每天早上開機都被誤判成「過期」，天天觸發不必要的全量下載。

    2026-08-19再修正（使用者發現）：改成單純用
    data.day_data_loader._expected_prior_trading_day() 還是不夠精確——
    13:30收盤後，今天的d1理論上已經有了（該用「今天」比對才抓得出「今天
    真的還沒更新」這件事），但只查前一交易日會永遠只看到早就存在的舊flag，
    對「今天該更新卻還沒更新」這種情況視而不見。改成依現在時間分兩段：
    13:30（收盤）前，今天的d1本來就不會有，比對「上一個預期交易日」
    （_expected_prior_trading_day()：週一回推到上週五，其他天回推1天，
    跟這個專案「快路徑判斷是否落後」既有的邏輯一致，刻意不處理國定假日，
    見該函式的說明）；13:30（含）之後，改比對「今天」。

    沒有該比對的那個flag（不管是本機真的落後、還是HF自己都還沒有——後者
    是上游排程的問題，這裡不負責分辨，下載了也可能沒用，但不影響後續
    開機流程）就從 HF Hub 全量下載補齊。

    2026-08-19討論：跟本檔案其他函式「不吞例外，留給呼叫端處理」的慣例
    刻意不同——這裡下載失敗只印log、不往外拋，讓 live_trader 照樣用現有
    本機資料繼續開機，不因為HF暫時連不上（例如已知的xet傳輸問題，見
    scripts/download_hf_for_local.py 檔頭說明）就卡死整個開機流程。"""
    from datetime import datetime, timezone, timedelta

    from data.day_data_loader import _expected_prior_trading_day, _get_done_stocks

    tw = timezone(timedelta(hours=8))
    now = datetime.now(tw)
    if (now.hour, now.minute) < (13, 30):
        check_date = _expected_prior_trading_day(now)
    else:
        check_date = now.strftime("%Y-%m-%d")

    done = _get_done_stocks(check_date)
    if "0050" in done:
        print(f"[HF同步檢查] 0050 {check_date} 的 d1 flag 已存在，本機資料新鮮，跳過下載", flush=True)
        return

    print(f"[HF同步檢查] 0050 {check_date} 沒有 d1 flag，本機資料可能落後，從 HF Hub 全量下載...", flush=True)
    try:
        from scripts.download_hf_for_local import main as download_from_hf

        download_from_hf()
        print("[HF同步檢查] 下載完成", flush=True)
    except Exception as e:
        print(f"[HF同步檢查] 下載失敗，改用現有本機資料繼續開機: {e}", flush=True)


def refresh_tickers(state) -> None:
    """更新當沖候選清單，寫入 state.tickers / state.day_trade_stocks。

    直接呼叫 fubon.subscribe_list.build_and_save_subscribe_list()：那是富邦
    WebSocket 訂閱清單唯一的來源，這裡重用同一份，避免候選股跟 WebSocket
    實際訂閱的股票不一致（先前這裡走 Fugle、WebSocket 那邊走富邦，兩邊
    各自過濾，理論上該一致但沒有保證）。API 回傳空值（例如非盤中）不算
    例外，視為「不過濾」。這支函式開機時跟每天 06:00 都會被呼叫，每次都
    重新計算（2026-07-22：拿掉 DAILY_REFRESH_TICKERS 開關——那是2026-07-14
    為了讓候選股清單跟 FinMind 歷史資料補齊的範圍保持一致才暫時凍結的，
    現在不需要了）。

    不額外濾槓桿/反向/主動型ETF（曾經用代號碼數濾過，2026-07-14 發現這是錯的：
    00878/00919 這類完全正常、成交量很大的高股息ETF也是5碼，用碼數過濾會連
    這些一起誤殺，見 fubon/subscribe_list.py 的說明）。是否進候選股交給
    isNormal=true + 均量排序決定。
    """
    df = build_and_save_subscribe_list()
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
    空 dict 等於沒有額外參數。

    帶入 state.day_trade_stocks（2026-07-25討論）：orb 的 build_prewarm_cache()
    要載入全歷史分K算 open_vol_history/hourly_tr_history，這份候選清單已經
    先算好了，讓它可以只算候選股那幾百支的歷史彙總表，不用連全市場~3000支
    都算一次、算完又因為 predict_live() 只吃候選股而白算（見
    strategy/orb/predict.py::build_prewarm_cache() 的說明）。"""
    for s in state.strategies.values():
        s.prewarm_cache = build_prewarm_cache(s.module, day_trade_stocks=state.day_trade_stocks)
