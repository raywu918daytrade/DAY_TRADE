"""
富邦當沖候選股清單。

2026-08-01改：候選股母體固定成 db/tickers/tick_universe.parquet（399支排名
+0050強制併入共400支，見 finmind/tick_universe.py），不再即時打富邦API抓
全市場~2700支、也不再用20日均量動態排序截斷（400支遠低於WebSocket上限
1000支，不需要截斷）。訓練（data/m1_data_loader.py、data/day_data_loader.py）
跟推論/交易統一用同一批固定母體，減少訓練雜訊，也大幅減少每天開盤前的API
用量/等待時間。

固定母體流程（見 subscription_batches()）：
    1. 讀 db/tickers/tick_universe.parquet 固定400支 + db/tickers/tickers.parquet
       本地既有清單查名稱（_fixed_universe_names()，都不觸發即時富邦API）
    2. canDayTrade/canBuyDayTrade 逐支確認（_filter_day_tradable()）——這個
       還是每天做，確認這400支裡有沒有股票今天被停資/注意處置、不能當沖
   最後切成每組 ≤200 支、最多 5 組（對應富邦 WebSocket rate limit：
200 檔/連線、5 條連線 → 上限 1000 檔，400支只會用到2組）。

舊的動態選股邏輯（_fubon_normal_tickers()/ranked_candidates()/
all_normal_stocks()）保留在檔案裡沒有刪除，只是目前的日常流程不再呼叫，
之後如果要重新擴大母體/改回動態選股可以直接復用。

這份清單是「唯一」的當沖候選股來源：
    - fubon/marketdata_ws.py 用分組後的 batches 決定 WebSocket 訂閱誰
    - main/premarket.py::refresh_tickers() 直接呼叫 build_and_save_subscribe_list()
      當作 state.day_trade_stocks / state.tickers 的來源

使用方式：
    main/premarket.py::refresh_tickers() 開機、每天 06:00 都會自動呼叫
    build_and_save_subscribe_list()，一般不用手動跑。要單獨測試/預覽：
        python -m fubon.subscribe_list

    其他地方要讀已存好的清單（不重算）：
        from fubon.subscribe_list import load_candidates, load_subscribe_batches
        df = load_candidates()          # 完整 DataFrame（含 name）
        batches = load_subscribe_batches()  # 分組後的 stock_id list，給 WebSocket 用
"""
import os
import threading
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

# fubon_neo SDK 底層沒有設定任何 timeout（翻過套件原始碼確認過），網路/伺服器
# 卡住的話 REST 呼叫可能無限期不回應，2026-07-25 討論加這層保護。
_TICKER_CHECK_TIMEOUT = float(os.environ.get("FUBON_TICKER_CHECK_TIMEOUT", "8"))


def _call_with_timeout(fn, timeout: float, *args):
    """幫沒有內建 timeout 的呼叫包一個逾時保護。用 daemon thread（不是
    concurrent.futures.ThreadPoolExecutor）——ThreadPoolExecutor 預設會在
    process 結束時等所有丟進去的任務 join 完，真的卡住不回應的呼叫會讓
    程式沒辦法乾淨結束；daemon thread 不會擋 process 退出，逾時就直接放棄
    等待，底下那條線程若之後真的回應了，結果被丟棄，不影響呼叫端。
    每次呼叫都開一支新的 thread（不是共用 pool），確保某支股票卡住不會
    連帶拖慢後面要查的股票（各自獨立逾時，不會排隊等前一支放棄）。
    """
    result: dict = {}

    def _target():
        try:
            result["value"] = fn(*args)
        except Exception as e:  # noqa: BLE001
            result["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"逾時 {timeout} 秒未回應")
    if "error" in result:
        raise result["error"]
    return result.get("value")


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
    """完整股票母體（isNormal=true、排除債券ETF，不做均量排序/上限）。
    2026-08-01改：日常流程（data/m1_data_loader.py、data/day_data_loader.py、
    fubon/subscribe_list.py 自己）都已經改成固定讀 tick_universe（見
    _fixed_universe_names()），不再呼叫這支——保留函式本身，之後如果要
    重新擴大母體/改回動態選股可以直接復用，不用重寫。"""
    return list(_fubon_normal_tickers().keys())


def ranked_candidates(names: dict[str, str]) -> list[str]:
    """依 20日均量排序（高→低）的候選股，最多 MAX_SUBSCRIPTIONS 支（.env，
    目前設為 1000，剛好對應 5 條連線 × 200 檔）。2026-08-01改：日常流程不再
    呼叫這支，說明同 all_normal_stocks()。"""
    from data.data_manager import _volume_filter
    from data.query import load_day

    day = load_day()
    return _volume_filter(set(names.keys()), day[["stock_id", "date", "volume"]])


def _fixed_universe_names() -> dict[str, str]:
    """固定候選股母體＋名稱查詢（2026-08-01加）：db/tickers/tick_universe.parquet
    （399支排名+0050強制併入共400支，見 finmind/tick_universe.py，name欄位
    2026-08-01一併加進去，不用再另外讀 db/tickers/tickers.parquet）決定「有
    哪些股票」跟名稱——讀本機檔案，不觸發即時富邦API。取代原本
    _fubon_normal_tickers()（全市場~2700支）+ ranked_candidates()（20日均量
    動態排序）這兩步，訓練/推論/交易統一用同一批固定母體。"""
    from finmind.tick_universe import _universe_file_path

    df = pd.read_parquet(_universe_file_path(), columns=["stock_id", "name"])
    return dict(zip(df["stock_id"], df["name"].fillna("")))


def _filter_day_tradable(stock_ids: list[str]) -> list[str]:
    """用 intraday/ticker/{symbol}（單支查詢，見 fubon/intraday_ticker.py 的
    診斷測試）逐支確認 canDayTrade/canBuyDayTrade 皆為 true，比 isNormal 更直接
    反映「能不能當沖」，2026-07-14 實測連垃圾代碼（industry非數字那批）也會被
    這兩個欄位抓到 canDayTrade=false。

    放在 ranked_candidates() 之後（均量排序+截斷到 MAX_SUBSCRIPTIONS 之後）才做，
    只查最後入選的候選股，不用對全市場 ~2700 支都查一次（300次/分鐘的話要
    9分鐘，候選股通常只有 1000 支內，約 4 分鐘內）。

    每支呼叫都包 _call_with_timeout()（預設8秒，.env 的 FUBON_TICKER_CHECK_TIMEOUT
    可調）——fubon_neo SDK 本身沒有 timeout，網路/伺服器卡住的話單一支股票可能
    無限期不回應，2026-07-25 討論加這層保護，逾時當作查詢失敗、直接跳過那支，
    不會拖住後面的股票（見 _call_with_timeout() 的說明）。

    _FORCE_INCLUDE（例如0050）的股票直接跳過查詢、無條件視為可以當沖
    （2026-07-25討論）：這些股票不管 canBuyDayTrade/canDayTrade 回傳什麼都
    一定要留著（見 _FORCE_INCLUDE 的說明），與其查完API結果又忽略，不如
    根本不查，省一次API呼叫跟0.25秒等待，語意也更明確——不是「查了但結果
    被覆蓋」，是「這幾支從一開始就不受這個檢查限制」。"""
    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    tradable = list(_FORCE_INCLUDE)
    try:
        trade_api.init_market_data(sdk)
        for sid in stock_ids:
            if sid in _FORCE_INCLUDE:
                continue
            try:
                info = _call_with_timeout(trade_api.intraday_ticker, _TICKER_CHECK_TIMEOUT, sdk, sid)
                # canDayTrade 先註解掉，不要求（2026-07-22：0050 這種正常標的
                # 也會 canDayTrade=False/canBuyDayTrade=True，兩個都要求會把
                # 它濾掉，先只看 canBuyDayTrade；不要刪，之後確認 canDayTrade
                # 的正確用法再決定要不要恢復）。
                if info.get("canBuyDayTrade"):  # and info.get("canDayTrade")
                    tradable.append(sid)
            except TimeoutError as e:
                print(f"  警告：{sid} intraday_ticker {e}，略過", flush=True)
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

    2026-08-01改：names 現在固定是 _fixed_universe_names() 的400支（不再是
    ranked_candidates() 動態均量排序後的結果），這裡直接用 dict 的 key 當
    候選清單。順序：固定母體 → canDayTrade/canBuyDayTrade 逐支確認，
    _FORCE_INCLUDE 的股票不查、無條件視為可以當沖（_filter_day_tradable）
    → 分連線。400支遠低於 MAX_CONNECTIONS×MAX_PER_CONNECTION 上限，實務上
    只會用到2組連線。"""
    candidates = list(names.keys())
    candidates = _filter_day_tradable(candidates)
    batches = [
        candidates[i : i + MAX_PER_CONNECTION]
        for i in range(0, len(candidates), MAX_PER_CONNECTION)
    ]
    return batches[:MAX_CONNECTIONS]


def build_and_save_subscribe_list(force: bool = False) -> pd.DataFrame:
    """算好分連線的候選股清單（含名稱），存到 db/fubon_subscribe/。
    main/premarket.py::refresh_tickers() 開機、每天 06:00 都會呼叫這個，
    不用另外排程。

    force=False（預設，2026-07-25新增）：如果 db/fubon_subscribe/subscribe_list.parquet
    已經是「今天」存的，直接讀取回傳，不重新跑一次 _filter_day_tradable() 那段
    3~4分鐘、逐支查富邦API的流程——同一天內 main/live_trader.py 重啟多次（開機時
    的 _startup()、_daily_refresh() 的立即補載都會呼叫這支函式），沒必要每次都
    重新查一次同一天的候選資格。force=True 才強制整個重跑（例如手動用
    `python -m fubon.subscribe_list` 想確認今天最新結果、或懷疑舊資料有問題時）。
    """
    if not force and _SUBSCRIBE_PATH.exists():
        existing = pd.read_parquet(_SUBSCRIBE_PATH)
        if not existing.empty:
            today = datetime.now(_TW).strftime("%Y-%m-%d")
            saved_date = str(existing["date"].iloc[0])
            if saved_date == today:
                print(f"候選清單已是今天（{today}）存的，直接沿用，不重新查詢 → {_SUBSCRIBE_PATH}", flush=True)
                return existing

    names = _fixed_universe_names()
    if not names:
        print("  警告：找不到 tick_universe/tickers 本地清單，清單維持空白", flush=True)
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
    # 手動跑（單獨測試/預覽）故意強制重跑，不要沿用今天已存的舊結果。
    build_and_save_subscribe_list(force=True)
    for i, b in enumerate(load_subscribe_batches(), 1):
        preview = "、".join(b[:5])
        print(f"  連線 {i}：{len(b)} 支  前5：{preview}")
