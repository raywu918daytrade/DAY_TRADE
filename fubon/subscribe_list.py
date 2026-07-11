"""
富邦 WebSocket 訂閱清單：從當沖候選股中，依 20日均量排序取前 N 支，
切成每組 ≤200 支，對應最多 5 條 WebSocket 連線（富邦行情 rate limit：
200 檔/連線、5 條連線 → 上限 1000 檔）。

均量排序邏輯直接沿用 data/data_manager.py 的 _volume_filter，避免另外
寫一套排序規則。

使用方式（每日開盤前跑一次 + 開連線時只讀檔）：
    開盤前（例如排程在 06:00，跟 live_trader._daily_refresh 同時段）：
        python -m fubon.subscribe_list   # 算好清單存到 db/fubon_subscribe/

    開 WebSocket 連線時：
        from fubon.subscribe_list import load_subscribe_batches
        batches = load_subscribe_batches()   # 直接讀檔，不重算
"""
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from fubon.config import MAX_CONNECTIONS, MAX_PER_CONNECTION

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_TW = timezone(timedelta(hours=8))
_SUBSCRIBE_PATH = _ROOT / "db/fubon_subscribe/subscribe_list.parquet"


def _fubon_normal_tickers() -> set[str]:
    """用富邦自己的 REST 行情 API 抓「正常交易」股票清單（isNormal=true，排除注意/處置股）。

    改用富邦而不是 Fugle 的 /intraday/tickers：Fugle 那邊原本多帶的 isDayTrading=true
    查無官方文件依據（見 memory 的 TODO），且既然分K都走富邦，篩選母體也一併改成
    同一個資料源，比較一致。TWSE / TPEx 各查一次再合併。實際 SDK 呼叫都包在
    fubon/trade_api.py，這裡不直接碰 fubon_neo。
    """
    from fubon import trade_api

    sdk, _ = trade_api.login()
    try:
        trade_api.init_market_data(sdk)
        stocks: set[str] = set()
        for exchange in ("TWSE", "TPEx"):
            stocks.update(item["symbol"] for item in trade_api.intraday_tickers(sdk, exchange))
        return stocks
    finally:
        trade_api.logout(sdk)


def ranked_candidates() -> list[str]:
    """依 20日均量排序（高→低）的候選股，最多 MAX_SUBSCRIPTIONS 支（.env，
    目前設為 1000，剛好對應 5 條連線 × 200 檔）。"""
    from data.data_manager import _volume_filter
    from data.query import load_day

    stocks = _fubon_normal_tickers()
    day = load_day()
    return _volume_filter(stocks, day[["stock_id", "date", "volume"]])


def subscription_batches() -> list[list[str]]:
    """把候選股切成 ≤MAX_PER_CONNECTION 支一組，最多 MAX_CONNECTIONS 組
    （富邦 WebSocket 連線本身的 rate limit，定義在 fubon/config.py）。"""
    candidates = ranked_candidates()
    batches = [
        candidates[i : i + MAX_PER_CONNECTION]
        for i in range(0, len(candidates), MAX_PER_CONNECTION)
    ]
    return batches[:MAX_CONNECTIONS]


def build_and_save_subscribe_list() -> pd.DataFrame:
    """開盤前跑一次：算好分連線的訂閱清單，存到 db/fubon_subscribe/。"""
    batches = subscription_batches()
    date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    rows = [
        {"stock_id": sid, "connection_id": conn_id, "rank": rank, "date": date_str}
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


def load_subscribe_batches() -> list[list[str]]:
    """開連線時只讀檔，不重算。回傳依 connection_id 分組、依 rank 排序的 batches。

    找不到檔案或不是今天存的，仍照樣回傳（可能是舊清單），並印警告讓呼叫端自行判斷。
    """
    if not _SUBSCRIBE_PATH.exists():
        print(f"找不到 {_SUBSCRIBE_PATH}，請先執行 python -m fubon.subscribe_list", flush=True)
        return []
    df = pd.read_parquet(_SUBSCRIBE_PATH)
    if df.empty:
        return []
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    saved_date = str(df["date"].iloc[0])
    if saved_date != today:
        print(f"警告：訂閱清單是 {saved_date} 存的，非今日（{today}），可能尚未更新", flush=True)
    return [
        g.sort_values("rank")["stock_id"].tolist()
        for _, g in df.sort_values("connection_id").groupby("connection_id", sort=False)
    ]


if __name__ == "__main__":
    build_and_save_subscribe_list()
    for i, b in enumerate(load_subscribe_batches(), 1):
        preview = "、".join(b[:5])
        print(f"  連線 {i}：{len(b)} 支  前5：{preview}")
