"""
富邦當沖候選股清單：從富邦 REST 行情 API 抓「正常交易」股票，依 20日均量
排序取前 N 支，切成每組 ≤200 支、最多 5 組（對應富邦 WebSocket rate limit：
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
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from fubon.config import MAX_CONNECTIONS, MAX_PER_CONNECTION

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))
_SUBSCRIBE_PATH = _ROOT / "db/fubon_subscribe/subscribe_list.parquet"

# 台股 ETF 代號後綴慣例：00XXXB＝債券型ETF，00XXXD＝主動式債券/固定收益基金
# （實測這批代號的名稱都帶「非投」「債」「入息」）。只要股票跟一般/槓桿/反向/
# 主動股票型 ETF（無後綴、L、R、A），不要固定收益類。名稱含「債」字再補一層，
# 避免名稱被 API 截斷看不出後面有「債」的漏網之魚。
_BOND_ETF_PATTERN = re.compile(r"^00\d{3}[BD]$")

# 一般股票代號固定4碼數字；5碼以上（例如00631L槓桿ETF、00632R反向ETF）都是
# ETF，不是個股。只取4碼的話連 0050 這種4碼ETF也留著沒濾掉（main/live_trader.py
# 的 idx_* 特徵需要 0050，見 _get_stocks() 固定補進候選清單那段），這裡的
# 4碼過濾只是額外把5碼以上的槓桿/反向/主動型ETF擋掉，不是要把ETF全部濾乾淨。
_STOCK_CODE_PATTERN = re.compile(r"^\d{4}$")


def _is_bond(stock_id: str, name: str) -> bool:
    return bool(_BOND_ETF_PATTERN.match(stock_id)) or "債" in name


def _is_junk(stock_id: str) -> bool:
    """富邦 intraday.tickers() 回傳的清單裡混了一批非個股代號（例如 A00104、
    A01102，industry 欄位是 A1/A2 這種產業分類代碼，不是真正的股票/ETF）。
    台股股票/ETF代號一律數字開頭（4碼股票、00開頭ETF、含字母尾碼的特別股如
    2887Z1 也是數字開頭），這批垃圾代號則是字母開頭，用這個規則排除。
    2026-07-14 實測：這批代號在日K historical/candles 一律 404（Fugle/富邦
    共用同一套底層資料源，兩邊都查不到），會讓 update_day() 每天重複打一樣
    的失敗請求；也實測過跟「name==symbol」這個較脆弱的判斷法比對，兩者篩出
    的集合完全一致，改用代號開頭是不是數字判斷。"""
    return not stock_id[0].isdigit()


def _is_4digit_stock(stock_id: str) -> bool:
    return bool(_STOCK_CODE_PATTERN.match(stock_id))


def _fubon_normal_tickers(only_4digit: bool = False) -> dict[str, str]:
    """用富邦自己的 REST 行情 API 抓「正常交易」股票清單（isNormal=true，排除注意/處置股），
    回傳 {stock_id: name}。排除債券型ETF／固定收益基金（見 _is_bond()），只留股票和
    一般權益類 ETF。

    only_4digit: 選填，只留4碼代號（見 _is_4digit_stock()），把5碼以上的槓桿/反向/
    主動型ETF（例如00631L、00632R）也濾掉，不只濾債券ETF。預設 False，不改變既有
    行為（例如即時交易候選股清單、其他呼叫端不會突然少了這些ETF）。

    改用富邦而不是 Fugle 的 /intraday/tickers：Fugle 那邊原本多帶的 isDayTrading=true
    查無官方文件依據（見 memory 的 TODO）。TWSE / TPEx 各查一次再合併。實際 SDK 呼叫
    都包在 fubon/trade_api.py，這裡不直接碰 fubon_neo。
    """
    from fubon import trade_api

    sdk, _ = trade_api.login()
    try:
        trade_api.init_market_data(sdk)
        stocks: dict[str, str] = {}
        for exchange in ("TWSE", "TPEx"):
            for item in trade_api.intraday_tickers(sdk, exchange):
                sid = item["symbol"]
                name = item.get("name", "")
                if _is_bond(sid, name) or _is_junk(sid):
                    continue
                if only_4digit and not _is_4digit_stock(sid):
                    continue
                stocks[sid] = name
        return stocks
    finally:
        trade_api.logout(sdk)


def all_normal_stocks(only_4digit: bool = False) -> list[str]:
    """完整股票母體（isNormal=true、排除債券ETF，不做均量排序/上限），給訓練資料
    批次下載器用（data/day_data_loader.py、data/m1_data_loader.py）——那兩支不受
    WebSocket 訂閱數限制，不需要 ranked_candidates() 的排序/截斷，只要「今天有哪些
    股票可以交易」這個母體即可。

    only_4digit: 選填，見 _fubon_normal_tickers() 的說明，只留4碼個股代號，濾掉
    5碼以上的槓桿/反向/主動型ETF。"""
    return list(_fubon_normal_tickers(only_4digit=only_4digit).keys())


def ranked_candidates(names: dict[str, str]) -> list[str]:
    """依 20日均量排序（高→低）的候選股，最多 MAX_SUBSCRIPTIONS 支（.env，
    目前設為 1000，剛好對應 5 條連線 × 200 檔）。"""
    from data.data_manager import _volume_filter
    from data.query import load_day

    day = load_day()
    return _volume_filter(set(names.keys()), day[["stock_id", "date", "volume"]])


def subscription_batches(names: dict[str, str]) -> list[list[str]]:
    """把候選股切成 ≤MAX_PER_CONNECTION 支一組，最多 MAX_CONNECTIONS 組
    （富邦 WebSocket 連線本身的 rate limit，定義在 fubon/config.py）。"""
    candidates = ranked_candidates(names)
    batches = [
        candidates[i : i + MAX_PER_CONNECTION]
        for i in range(0, len(candidates), MAX_PER_CONNECTION)
    ]
    return batches[:MAX_CONNECTIONS]


def build_and_save_subscribe_list(only_4digit: bool = False) -> pd.DataFrame:
    """算好分連線的候選股清單（含名稱），存到 db/fubon_subscribe/。
    main/premarket.py::refresh_tickers() 開機、每天 06:00 都會呼叫這個，
    不用另外排程。

    only_4digit: 選填，見 _fubon_normal_tickers() 的說明，濾掉5碼以上的槓桿/
    反向/主動型ETF——這類ETF成交量常常很大，會擠掉均量排序裡真正的個股
    （2026-07-14 實測：現有清單前幾名就有00685L/00631L/00403A這幾支ETF）。
    """
    names = _fubon_normal_tickers(only_4digit=only_4digit)
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
