"""
富邦當沖候選股清單，三步驟篩選（見 subscription_batches()）：
    1. isNormal + industry數字過濾 + 排除債券ETF（fubon/intraday_tickers.py::update_tickers()）
    2. 20日均量排序，取前 N 支（ranked_candidates()，最多 MAX_SUBSCRIPTIONS 支）
    3. canDayTrade/canBuyDayTrade 逐支確認（_filter_day_tradable()）
   最後切成每組 ≤200 支、最多 5 組（對應富邦 WebSocket rate limit：
200 檔/連線、5 條連線 → 上限 1000 檔）。

這份清單是「唯一」的當沖候選股來源：
    - fubon/marketdata_ws.py 用分組後的 batches 決定 WebSocket 訂閱誰
    - main/premarket.py::refresh_tickers() 直接呼叫 build_and_save_subscribe_list()
      當作 state.day_trade_stocks / state.tickers 的來源
避免這兩處各自獨立算一次候選股（先前 main/premarket.py 走 Fugle、這裡走富邦，
兩邊資料源不同，理論上該是同一份清單卻沒有保證真的一致）。

均量排序邏輯直接沿用 data/data_manager.py 的 _volume_filter，避免另外
寫一套排序規則。

使用方式：
    main/premarket.py::refresh_tickers() 開機、每天 06:00 都會自動呼叫
    build_and_save_subscribe_list()，一般不用手動跑。要單獨測試/預覽：
        python -m fubon.subscribe_list

    其他地方要讀已存好的清單（不重算）：
        from fubon.subscribe_list import load_candidates, load_subscribe_batches
        df = load_candidates()          # 完整 DataFrame（含 name）
        batches = load_subscribe_batches()  # 分組後的 stock_id list，給 WebSocket 用

    訓練資料批次下載器（不受均量排序/WebSocket上限限制，只要完整母體）：
        from fubon.subscribe_list import all_normal_stocks
        stocks = all_normal_stocks()    # 每次呼叫都現抓，不經過存檔的候選清單
"""
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from fubon.config import MAX_CONNECTIONS, MAX_PER_CONNECTION

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))
_SUBSCRIBE_PATH = _ROOT / "db/fubon_subscribe/subscribe_list.parquet"


def _fubon_normal_tickers() -> dict[str, str]:
    """股票清單（垃圾代號、債券ETF已經在 fubon/intraday_tickers.py::update_tickers()
    這個唯一過濾源頭濾掉了），回傳 {stock_id: name}。

    不額外用代號碼數濾槓桿/反向/主動型ETF（曾經這樣做過，2026-07-14 發現這是錯的：
    台股ETF代號是「00+3碼」＝5碼才是現行規則，00878/00919這類完全正常、成交量
    很大的高股息ETF也是5碼，用碼數過濾會連這些一起誤殺）。是否進候選股完全交給
    isNormal=true（intraday_tickers.py 呼叫 API 時已經濾過）+ ranked_candidates()
    的成交量排序決定，槓桿/反向ETF成交量高就會排前面，不特別排除。"""
    from fubon.intraday_tickers import update_tickers

    df = update_tickers()
    if df.empty:
        return {}
    return dict(zip(df["stock_id"], df["name"]))


def all_normal_stocks() -> list[str]:
    """完整股票母體（isNormal=true、排除債券ETF，不做均量排序/上限），給訓練資料
    批次下載器用（data/day_data_loader.py、data/m1_data_loader.py）——那兩支不受
    WebSocket 訂閱數限制，不需要 ranked_candidates() 的排序/截斷，只要「今天有哪些
    股票可以交易」這個母體即可。"""
    return list(_fubon_normal_tickers().keys())


def ranked_candidates(names: dict[str, str]) -> list[str]:
    """依 20日均量排序（高→低）的候選股，最多 MAX_SUBSCRIPTIONS 支（.env，
    目前設為 1000，剛好對應 5 條連線 × 200 檔）。"""
    from data.data_manager import _volume_filter
    from data.query import load_day

    day = load_day()
    return _volume_filter(set(names.keys()), day[["stock_id", "date", "volume"]])


def _filter_day_tradable(stock_ids: list[str]) -> list[str]:
    """用 intraday/ticker/{symbol}（單支查詢，見 fubon/intraday_ticker.py 的
    診斷測試）逐支確認 canDayTrade/canBuyDayTrade 皆為 true，比 isNormal 更直接
    反映「能不能當沖」，2026-07-14 實測連垃圾代碼（industry非數字那批）也會被
    這兩個欄位抓到 canDayTrade=false。

    放在 ranked_candidates() 之後（均量排序+截斷到 MAX_SUBSCRIPTIONS 之後）才做，
    只查最後入選的候選股，不用對全市場 ~2700 支都查一次（300次/分鐘的話要
    9分鐘，候選股通常只有 1000 支內，約 4 分鐘內）。"""
    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    tradable = []
    try:
        trade_api.init_market_data(sdk)
        for sid in stock_ids:
            try:
                info = trade_api.intraday_ticker(sdk, sid)
                # canDayTrade 先註解掉，不要求（2026-07-22：0050 這種正常標的
                # 也會 canDayTrade=False/canBuyDayTrade=True，兩個都要求會把
                # 它濾掉，先只看 canBuyDayTrade；不要刪，之後確認 canDayTrade
                # 的正確用法再決定要不要恢復）。
                if info.get("canBuyDayTrade"):  # and info.get("canDayTrade")
                    tradable.append(sid)
            except Exception as e:
                print(f"  警告：{sid} intraday_ticker 查詢失敗，略過: {e}", flush=True)
            time.sleep(0.25)  # 300次/分鐘上限，留緩衝
    finally:
        trade_api.logout(sdk)
    return tradable


# 大盤代理（0050）— strategy/rally/features.py 的市場代理特徵（idx_ret_1/idx_atr/
# idx_up 等）用這一檔的即時分K廣播給所有候選股，是必要依賴，不是要拿來當沖
# 交易，只是要抓報價。2026-07-22 發現：0050 canDayTrade=False 時會被
# _filter_day_tradable() 濾掉，導致 rally 全部候選股的大盤特徵在 merge 時對不到
# 這一分鐘的 0050 row、整批變 NaN，被 dropna 篩光，rally 當下完全沒有任何推論
# 結果（而且不會報錯，只會安靜地一直印 0 支）。所以這裡無條件強制加回去，不受
# 當沖資格檢查限制。
_FORCE_INCLUDE = ["0050"]


def subscription_batches(names: dict[str, str]) -> list[list[str]]:
    """把候選股切成 ≤MAX_PER_CONNECTION 支一組，最多 MAX_CONNECTIONS 組
    （富邦 WebSocket 連線本身的 rate limit，定義在 fubon/config.py）。

    順序：均量排序＋截斷（ranked_candidates） → canDayTrade/canBuyDayTrade
    逐支確認（_filter_day_tradable） → 強制加回 _FORCE_INCLUDE → 分連線。"""
    candidates = ranked_candidates(names)
    candidates = _filter_day_tradable(candidates)
    for sid in _FORCE_INCLUDE:
        if sid not in candidates:
            candidates.insert(0, sid)
    batches = [
        candidates[i : i + MAX_PER_CONNECTION]
        for i in range(0, len(candidates), MAX_PER_CONNECTION)
    ]
    return batches[:MAX_CONNECTIONS]


def build_and_save_subscribe_list() -> pd.DataFrame:
    """算好分連線的候選股清單（含名稱），存到 db/fubon_subscribe/。
    main/premarket.py::refresh_tickers() 開機、每天 06:00 都會呼叫這個，
    不用另外排程。"""
    names = _fubon_normal_tickers()
    if not names:
        print("  警告：富邦 API 沒回傳任何股票（非盤中？），清單維持空白", flush=True)
        return pd.DataFrame()

    batches = subscription_batches(names)
    date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    rows = [
        {"stock_id": sid, "name": names.get(sid, ""), "connection_id": conn_id, "rank": rank, "date": date_str}
        for conn_id, batch in enumerate(batches)
        for rank, sid in enumerate(batch)
    ]
    df = pd.DataFrame(rows)
    os.makedirs(_SUBSCRIBE_PATH.parent, exist_ok=True)
    df.to_parquet(_SUBSCRIBE_PATH, index=False)
    print(
        f"儲存完成：{len(df)} 支，{len(batches)} 條連線 → {_SUBSCRIBE_PATH}（{date_str}）",
        flush=True,
    )
    return df


def load_candidates() -> pd.DataFrame:
    """讀取已存的候選股清單（不重算），欄位：stock_id/name/connection_id/rank/date。

    找不到檔案或不是今天存的，仍照樣回傳（可能是舊清單），並印警告讓呼叫端自行判斷。
    """
    if not _SUBSCRIBE_PATH.exists():
        print(f"找不到 {_SUBSCRIBE_PATH}，請先執行 python -m fubon.subscribe_list", flush=True)
        return pd.DataFrame()
    df = pd.read_parquet(_SUBSCRIBE_PATH)
    if df.empty:
        return df
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    saved_date = str(df["date"].iloc[0])
    if saved_date != today:
        print(f"警告：候選股清單是 {saved_date} 存的，非今日（{today}），可能尚未更新", flush=True)
    return df


def load_subscribe_batches() -> list[list[str]]:
    """開 WebSocket 連線時用：回傳依 connection_id 分組、依 rank 排序的 batches。"""
    df = load_candidates()
    if df.empty:
        return []
    return [
        g.sort_values("rank")["stock_id"].tolist()
        for _, g in df.sort_values("connection_id").groupby("connection_id", sort=False)
    ]


if __name__ == "__main__":
    build_and_save_subscribe_list()
    for i, b in enumerate(load_subscribe_batches(), 1):
        preview = "、".join(b[:5])
        print(f"  連線 {i}：{len(b)} 支  前5：{preview}")
