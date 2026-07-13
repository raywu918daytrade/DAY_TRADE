"""
分K REST 資料的共用小工具：解析 REST bars、寫進 db/m1_live/。

從 data/m1_rest.py（Fugle REST 輪詢收集器，2026-07-13 已移除——富邦
WebSocket 穩定後不再需要 Fugle 當備援，理由見 memory）搬出來，因為
fubon/marketdata_ws.py 也需要同一套解析/存檔邏輯，不想為了兩個函式
留著整支已經沒在用的 REST 輪詢器。
"""
import os
from pathlib import Path

import pandas as pd


def _parse_rest_bars(symbol: str, bars: list, date_str: str) -> pd.DataFrame:
    """將 REST API（Fugle/富邦，格式相同）回傳的 bars 轉換為標準分K格式"""
    rows = []
    for b in bars:
        # date format: '2026-07-02T09:00:00.000+08:00'
        dt = pd.to_datetime(b["date"]).tz_convert("Asia/Taipei").tz_localize(None)
        if dt.strftime("%Y-%m-%d") != date_str:
            continue
        rows.append({
            "stock_id": symbol,
            "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(b["open"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "close": float(b["close"]),
            "volume": int(b["volume"]),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float32")
    df["volume"] = df["volume"].astype("int64")
    return df[["stock_id", "date", "open", "high", "low", "close", "volume"]]


def _atomic_save(df: pd.DataFrame, file_path: Path):
    """讀舊檔 merge 新資料、dedup（stock_id, date）keep last，再原子性寫回去。
    呼叫端如果會有多執行緒同時對同一個 file_path 呼叫，要自己加鎖——這裡
    不處理併發（2026-07-13 曾經因為兩條執行緒同時呼叫沒加鎖，撞到
    os.replace() 的 .tmp 檔名衝突，見 fubon/marketdata_ws.py 的 _save_lock）。
    """
    os.makedirs(file_path.parent, exist_ok=True)
    if file_path.exists():
        old = pd.read_parquet(file_path)
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values(["date", "stock_id"], inplace=True)
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float32")
    df["volume"] = df["volume"].astype("int64")
    tmp = str(file_path) + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, file_path)
