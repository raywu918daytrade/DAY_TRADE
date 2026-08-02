"""
從 db/adjustment_day（Fugle 完整還原，含一般除權息）與 db/d1（原始）逐日比對，
算出 pattern 系列（型態偵測、POC疊圖，見 data/adjustment_query.py）專用的
「完整還原」調整係數，跟系統預設的 db/tick_adjust_factor（只還原拆股/合股）
分開維護——理由見 data/adjustment_query.py 檔頭說明：pattern 型態偵測需要
除息缺口也被抹平，否則會誤判轉折點（2026-08-02 用 1101 除息實測過影響幅度
足以跨過偵測器門檻）。

反推邏輯：兩邊現在都是日K，直接逐日比對收盤價：
    factor(stock, date) = db/adjustment_day 當天 close（完整還原）
                         / db/d1 當天 close（原始）
不需要像舊版 compute_adjust_factor() 那樣從 db/tick 拉13:30收盤——db/d1 本身
現在就有原始日收盤可以直接比。這張表刻意**不做拆股/合股簡化**（不像
db/tick_adjust_factor 那樣篩選跳空幅度），完整反映所有除權息事件。

儲存格式：按月分檔 parquet (db/adjustment_factor/{YYYY_MM}.parquet)，欄位：
    stock_id (str), date (str, YYYY-MM-DD), factor (float32)

因為兩邊資料量小（400支*全部歷史約幾十萬列），每次都是全部重新算一遍，不做
逐月增量判斷（比照 data/build_tick_adjust_factor.py 的做法）。

用法：
    python -m data.build_adjustment_factor                # 全部股票重新計算
    python -m data.build_adjustment_factor --stock 2330     # 只算/更新指定股票
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_D1_DIR = _ROOT / "db/d1"
_ADJUSTMENT_DAY_DIR = _ROOT / "db/adjustment_day"
_FACTOR_DIR = _ROOT / "db/adjustment_factor"


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path | str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def compute_full_adjust_factor(raw_df: pd.DataFrame, adjusted_df: pd.DataFrame) -> pd.DataFrame:
    """逐日比對 db/d1（原始）跟 db/adjustment_day（完整還原）的收盤價，算出
    完整還原係數（含一般除權息，不做拆股/合股門檻篩選）。

    raw_df 欄位需求：stock_id, date (YYYY-MM-DD), close（原始）
    adjusted_df 欄位需求：stock_id, date (YYYY-MM-DD), close（完整還原）
    回傳欄位：stock_id, date, factor
    """
    empty = pd.DataFrame(columns=["stock_id", "date", "factor"])
    if raw_df.empty or adjusted_df.empty:
        return empty

    raw = raw_df[["stock_id", "date", "close"]].rename(columns={"close": "raw_close"})
    raw = raw[raw["raw_close"] > 0]
    adj = adjusted_df[["stock_id", "date", "close"]].rename(columns={"close": "adjusted_close"})

    merged = raw.merge(adj, on=["stock_id", "date"], how="inner")
    if merged.empty:
        return empty
    merged["factor"] = (merged["adjusted_close"] / merged["raw_close"]).astype("float32")
    return merged[["stock_id", "date", "factor"]].sort_values(["stock_id", "date"]).reset_index(drop=True)


def build(stock_id: str | None = None):
    t0 = time.time()
    _FACTOR_DIR.mkdir(parents=True, exist_ok=True)

    if not _D1_DIR.exists() or not any(_D1_DIR.glob("*.parquet")):
        print("錯誤: db/d1/ 中沒有資料，請先執行 python -m data.backfill_day_history")
        return
    if not _ADJUSTMENT_DAY_DIR.exists() or not any(_ADJUSTMENT_DAY_DIR.glob("*.parquet")):
        print("錯誤: db/adjustment_day/ 中沒有資料")
        return

    filt = ds.field("stock_id") == stock_id if stock_id else None
    print(f"讀取 db/d1/ 與 db/adjustment_day/{'（股票 ' + stock_id + '）' if stock_id else ''}...", flush=True)
    raw_df = (
        ds.dataset(str(_D1_DIR), format="parquet").to_table(filter=filt, columns=["stock_id", "date", "close"]).to_pandas()
    )
    adj_df = (
        ds.dataset(str(_ADJUSTMENT_DAY_DIR), format="parquet")
        .to_table(filter=filt, columns=["stock_id", "date", "close"])
        .to_pandas()
    )
    print(f"  db/d1: {len(raw_df):,} 列，db/adjustment_day: {len(adj_df):,} 列 ({time.time()-t0:.2f}s)", flush=True)

    t1 = time.time()
    new_factor = compute_full_adjust_factor(raw_df, adj_df)
    print(f"計算完整還原 factor：{len(new_factor):,} 列 ({time.time()-t1:.2f}s)", flush=True)
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

    print(f"\n完整還原係數建置完成，共 {len(new_factor):,} 列，總耗時 {time.time()-t0:.2f}s ✅", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="反推 pattern 專用完整還原調整係數")
    parser.add_argument("--stock", type=str, default=None, help="只計算/更新指定股票代號 (例: 2330)")
    args = parser.parse_args()

    build(stock_id=args.stock)
