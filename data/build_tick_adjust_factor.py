"""
從 db/d1（原始日K，見 data/day_data_loader.py::update_day()）自己的逐日跳空幅度，
反推每支股票每天的拆股/合股調整係數，讓 db/m1／db/volume_profile／db/poc_day
（都是原始價格，維持不動）能在查詢時（見 data/query.py 的 load_m1()/load_m3()/
load_day()/load_volume_profile()/load_poc() 等）換算成還原後基準，不用改動這些
既有資料本身。

⚠️ 2026-08-02 語意：這支的 factor **只還原拆股/合股**，不還原一般現金/股票股利
除權息。原因：一般除權息造成的價格下跌通常會「填息」，是真實發生過的價格波動，沒有
必要特別還原，直接用原始K線即可；只有拆股/合股是永久性的價位重定基準（例如 0050
2025-06-18 的 1:4 拆股，單日原始價格 -75%），不會填息，才需要調整。

⚠️ 2026-08-02 再改：偵測方法不再跟任何「已還原」資料比對（原本拿 db/tick 原始收盤
vs db/fugle_day 已還原收盤反推，仍依賴 Fugle 自己的還原邏輯，且不同廠商的還原基準
彼此有 ~0.5~1% 落差，不保證一致）。改成直接看原始日收盤序列本身：逐日檢查漲跌幅，
見 compute_split_factor()。這在台股特別可靠——台股單日漲跌幅限制 ±10%，正常交易日
收盤對收盤的變動理論上不可能超過 10%，只有除權息參考價重設（含拆股/合股）才會讓
收盤價跳動超過 10%；threshold（預設 20%）進一步只挑「大到不像一般除權息」的事件，
濾掉現金/股票股利雜訊（拆股/合股的跳空天生很大：1:2=50%、1:4=75%…最小常見比例也
有 33% 以上；一般除權息很少超過15~20%）。

⚠️ 2026-08-03 再改：訊號來源從專門的 db/day_raw_close（close-only）改成直接讀
db/d1（完整OHLCV，取代原本的 db/fugle_day 原始資料角色）——db/d1 現在每天都會
透過 day_data_loader.py 正常下載更新，不需要再另外維護一份只抓 close 的資料。

儲存格式：按月分檔 parquet (db/tick_adjust_factor/{YYYY_MM}.parquet)，月份命名比照
db/tick／db/d1 補零格式，欄位為：
    stock_id (str), date (str, YYYY-MM-DD), factor (float32)

因為判斷拆股/合股需要看每支股票完整、連續的歷史（新偵測到一次事件要把該事件之前的
全部歷史 factor 往回乘上比例），這裡每次都是讀 db/d1 全部歷史重新算一遍
（資料量小，全部股票*全部歷史約幾十萬列，幾秒鐘跑得完，不需要像抓 API 那樣做增量
判斷），不是逐月增量合併。

用法：
    python -m data.build_tick_adjust_factor                # 全部股票重新計算
    python -m data.build_tick_adjust_factor --stock 2330     # 只算/更新指定股票
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_D1_DIR = _ROOT / "db/d1"
_FACTOR_DIR = _ROOT / "db/tick_adjust_factor"

# 拆股/合股 vs 一般除權息的判斷門檻：單日跳空幅度達到這個比例才視為拆股/合股
# （台股常見拆股/合股比例最小也有 33% 以上的跳空；一般除權息很少超過 15~20%）
_SPLIT_THRESHOLD = 0.20


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path | str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def compute_split_factor(raw_df: pd.DataFrame, threshold: float = _SPLIT_THRESHOLD) -> pd.DataFrame:
    """從原始日收盤序列，逐股票由新到舊掃描，偵測拆股/合股造成的單日跳空
    （>= threshold）並往回推算累積調整係數（同一支股票歷史上如果不只一次拆股/
    合股，比例會複合相乘，不是只反映最近一次）。

    raw_df 欄位需求：stock_id, date (YYYY-MM-DD), close（原始，未還原權息）
    回傳欄位：stock_id, date, factor——最新一天固定是 1.0，往回遇到拆股/合股
    事件才會變動；一般除權息造成的小跳空視為雜訊，不影響 factor（等同沿用
    前一個水位，那段日期查詢時直接用原始K線）。
    """
    empty = pd.DataFrame(columns=["stock_id", "date", "factor"])
    if raw_df.empty:
        return empty

    df = raw_df[raw_df["close"] > 0].drop_duplicates(subset=["stock_id", "date"], keep="last")
    if df.empty:
        return empty
    # 每支股票內部由新到舊排序，才能「今天=1.0，往回遇到事件才複合相乘」
    df = df.sort_values(["stock_id", "date"], ascending=[True, False]).reset_index(drop=True)

    stock_ids = df["stock_id"].to_numpy()
    closes = df["close"].to_numpy(dtype="float64")
    n = len(df)
    factor = np.empty(n, dtype="float64")

    cum = 1.0
    prev_stock = None
    prev_close = None
    for i in range(n):
        sid = stock_ids[i]
        if sid != prev_stock:
            cum = 1.0
            prev_stock = sid
        else:
            ratio = prev_close / closes[i]  # 較新一天收盤 / 較舊一天收盤
            if abs(ratio - 1.0) >= threshold:
                cum *= ratio
        factor[i] = cum
        prev_close = closes[i]

    result = df.copy()
    result["factor"] = factor.astype("float32")
    return result[["stock_id", "date", "factor"]].sort_values(["stock_id", "date"]).reset_index(drop=True)


def build(stock_id: str | None = None):
    t0 = time.time()
    _FACTOR_DIR.mkdir(parents=True, exist_ok=True)

    if not _D1_DIR.exists() or not any(_D1_DIR.glob("*.parquet")):
        print("錯誤: db/d1/ 中沒有資料，請先執行 python -m data.backfill_day_history")
        return

    print(f"讀取 db/d1/ 全部歷史{'（股票 ' + stock_id + '）' if stock_id else ''}...", flush=True)
    raw_ds = ds.dataset(str(_D1_DIR), format="parquet")
    filt = ds.field("stock_id") == stock_id if stock_id else None
    raw_df = raw_ds.to_table(filter=filt, columns=["stock_id", "date", "close"]).to_pandas()
    if raw_df.empty:
        print("  db/d1/ 無相符資料")
        return
    print(f"  共 {len(raw_df):,} 列 ({time.time()-t0:.2f}s)", flush=True)

    t1 = time.time()
    new_factor = compute_split_factor(raw_df)
    print(f"計算拆股/合股 factor：{len(new_factor):,} 列 ({time.time()-t1:.2f}s)", flush=True)
    if new_factor.empty:
        return

    new_factor = new_factor.copy()
    new_factor["_month"] = new_factor["date"].str[:7].str.replace("-", "_")
    for ym, group in new_factor.groupby("_month"):
        path = _FACTOR_DIR / f"{ym}.parquet"
        group = group.drop(columns=["_month"]).reset_index(drop=True)
        if stock_id and path.exists():
            # 只更新指定股票時，同月份檔案裡其他股票的既有資料要保留
            old = pd.read_parquet(path)
            old = old[old["stock_id"] != stock_id]
            group = pd.concat([old, group], ignore_index=True).sort_values(["stock_id", "date"]).reset_index(drop=True)
        _atomic_to_parquet(group, path, index=False, compression="zstd")

    print(f"\n除權息調整係數建置完成，共 {len(new_factor):,} 列，總耗時 {time.time()-t0:.2f}s ✅", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="反推拆股/合股限定的調整係數")
    parser.add_argument("--stock", type=str, default=None, help="只計算/更新指定股票代號 (例: 2330)")
    args = parser.parse_args()

    build(stock_id=args.stock)
