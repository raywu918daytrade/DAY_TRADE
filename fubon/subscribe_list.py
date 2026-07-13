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
"""
import os
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
    """用富邦自己的 REST 行情 API 抓「正常交易」股票清單（isNormal=true，排除注意/處置股），
    回傳 {stock_id: name}。

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
                stocks[item["symbol"]] = item.get("name", "")
        return stocks
    finally:
        trade_api.logout(sdk)


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
