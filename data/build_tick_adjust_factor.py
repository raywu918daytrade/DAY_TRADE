"""
從 db/tick（原始成交價）與 db/fugle_day（已還原權息）反推每支股票每天的除權息調整
係數，讓 db/volume_profile／db/poc_day（都是直接從 db/tick 算出來的，價格維持原始
不動）能在查詢時（見 data/query.py 的 load_volume_profile_adjusted()/load_poc_adjusted()）
換算成跟K線一致的「調整後」基準，不用改動 db/tick／db/volume_profile／db/poc_day
這些既有資料本身。

背景：db/m1、db/fugle_day 下載時都帶 adjusted="true"（見 CLAUDE.md 以外的另一個
基準問題，2026-08-01 除錯 pattern 圖表 volume profile 疊加時發現），除權息日之前的
歷史K線價格會被 Fugle 往下（或整體乘上一個比例）調整，讓價格序列連續；但 db/tick
是逐筆原始成交價，沒有做這個調整。兩邊直接比對（例如 pattern/breakout_retest 用
K線壓力位比對 POC 共振）會因為基準不同系統性偏移——實測 1101 除息（2026-07-01）前
偏移約 0.85 元（~3.5%），超過偵測器裡 2% 的共振門檻；0050 更誇張，2025-06-18
1:4 拆股當天原始價格單日 -75%，用固定金額的加法沒辦法處理，一定要用乘法係數。

反推邏輯：不需要外部除權息公告資料，直接拿「同一天」兩邊的收盤價互相比對：
    factor = db/fugle_day 當天 close（已調整） / db/tick 當天收盤（原始）
db/tick 的「當天收盤」取當天 13:30:00（含）以前最後一筆成交——13:30:00 本身就是
收盤定盤價，之後（例如 14:30:00）是盤後逐筆交易，不算正式收盤。

儲存格式：按月分檔 parquet (db/tick_adjust_factor/{YYYY_MM}.parquet)，月份命名比照
db/tick／db/fugle_day 補零格式，欄位為：
    stock_id (str), date (str, YYYY-MM-DD), factor (float32)

機制支援增量更新：邏輯比照 build_volume_profile.py／build_poc.py，當 db/tick 或
db/fugle_day 有新交易日/新月份時自動檢測增量並合併寫入。

用法：
    python -m data.build_tick_adjust_factor                # 增量建置（預設）
    python -m data.build_tick_adjust_factor --force         # 強制全部重新計算
    python -m data.build_tick_adjust_factor --month 2026_05 # 指定月份
    python -m data.build_tick_adjust_factor --stock 2330     # 指定股票
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

_TICK_DIR = _ROOT / "db/tick"
_DAY_DIR = _ROOT / "db/fugle_day"
_FACTOR_DIR = _ROOT / "db/tick_adjust_factor"


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path | str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def compute_adjust_factor(tick_df: pd.DataFrame, day_df: pd.DataFrame) -> pd.DataFrame:
    """從單一月份的原始 tick 與已調整日K反推每天的調整係數。

    tick_df 欄位需求: stock_id, date（完整時間戳字串，含時分秒）, deal_price
    day_df 欄位需求: stock_id, date（YYYY-MM-DD 或可轉字串的日期), close（已還原權息）
    回傳欄位: stock_id, date (YYYY-MM-DD), factor
    """
    empty = pd.DataFrame(columns=["stock_id", "date", "factor"])
    if tick_df.empty or day_df.empty:
        return empty

    df = tick_df.copy()
    df["day"] = df["date"].astype(str).str[:10]
    df["time"] = df["date"].astype(str).str[11:19]
    df = df[df["time"] <= "13:30:00"]
    if df.empty:
        return empty
    df.sort_values(["stock_id", "day", "date"], inplace=True)
    raw_close = df.groupby(["stock_id", "day"], as_index=False).last()[["stock_id", "day", "deal_price"]]
    raw_close.rename(columns={"day": "date", "deal_price": "raw_close"}, inplace=True)

    day = day_df[["stock_id", "date", "close"]].copy()
    day["date"] = day["date"].astype(str).str[:10]
    day.rename(columns={"close": "adjusted_close"}, inplace=True)

    merged = raw_close.merge(day, on=["stock_id", "date"], how="inner")
    merged = merged[merged["raw_close"] > 0]
    if merged.empty:
        return empty
    merged["factor"] = (merged["adjusted_close"] / merged["raw_close"]).astype("float32")
    return merged[["stock_id", "date", "factor"]].sort_values(["stock_id", "date"]).reset_index(drop=True)


def build(incremental: bool = True, force: bool = False, target_month: str | None = None, stock_id: str | None = None):
    t0 = time.time()
    _FACTOR_DIR.mkdir(parents=True, exist_ok=True)

    tick_paths = sorted(_TICK_DIR.glob("*.parquet"))
    if not tick_paths:
        print("錯誤: db/tick/ 中沒有資料")
        return

    if target_month:
        norm_month = target_month.replace("-", "_")
        tick_paths = [p for p in tick_paths if p.stem == norm_month or p.stem == target_month]
        if not tick_paths:
            print(f"錯誤: 找不到月份 {target_month} 的 tick parquet 檔案")
            return

    mode_str = "完整重建" if force else ("增量建置" if incremental else "全量處理")
    print(f"開始 {mode_str} 除權息調整係數，共有 {len(tick_paths)} 個月份檔案...", flush=True)

    for tick_path in tick_paths:
        ym = tick_path.stem
        day_path = _DAY_DIR / f"{ym}.parquet"
        factor_path = _FACTOR_DIR / f"{ym}.parquet"
        print(f"\n處理 {ym} ({tick_path.name})...", flush=True)
        t1 = time.time()

        if not day_path.exists():
            print(f"  {ym}: 找不到對應的 db/fugle_day/{day_path.name}，跳過", flush=True)
            continue

        # 增量檢查邏輯：tick 或 day K 任一比 factor 檔新，就要重算
        if incremental and not force and factor_path.exists():
            tick_mtime = tick_path.stat().st_mtime
            day_mtime = day_path.stat().st_mtime
            factor_mtime = factor_path.stat().st_mtime
            if factor_mtime >= tick_mtime and factor_mtime >= day_mtime and stock_id is None:
                print(f"  {ym}: 已是最新，無須更新 ✅ ({time.time()-t1:.2f}s)", flush=True)
                continue

            tick_ds = ds.dataset(str(tick_path), format="parquet")
            filt = ds.field("stock_id") == stock_id if stock_id else None
            tick_table = tick_ds.to_table(filter=filt, columns=["stock_id", "date"])
            if tick_table.num_rows == 0:
                print(f"  {ym}: 無相符 tick 資料，跳過 ✅", flush=True)
                continue
            tick_df_meta = tick_table.to_pandas()
            tick_pairs = set(zip(tick_df_meta["stock_id"], tick_df_meta["date"].astype(str).str[:10]))

            factor_df_old = pd.read_parquet(factor_path, columns=["stock_id", "date"])
            if stock_id:
                factor_df_old = factor_df_old[factor_df_old["stock_id"] == stock_id]
            factor_pairs = set(zip(factor_df_old["stock_id"], factor_df_old["date"]))

            missing_pairs = tick_pairs - factor_pairs
            if not missing_pairs:
                print(f"  {ym}: 內容已有全部 (stock_id, date) 資料，無須更新 ✅ ({time.time()-t1:.2f}s)", flush=True)
                factor_path.touch()
                continue
            print(f"  {ym}: 發現 {len(missing_pairs)} 組新的 (stock_id, date) 待增量更新...", flush=True)

        tick_ds = ds.dataset(str(tick_path), format="parquet")
        filt = ds.field("stock_id") == stock_id if stock_id else None
        tick_table = tick_ds.to_table(filter=filt, columns=["stock_id", "date", "deal_price"])
        if tick_table.num_rows == 0:
            print(f"  {ym}: 無相符 tick 資料", flush=True)
            continue
        tick_df = tick_table.to_pandas()
        print(f"  載入 tick: {len(tick_df):,} 筆 ({time.time()-t1:.2f}s)", flush=True)

        day_ds = ds.dataset(str(day_path), format="parquet")
        day_filt = ds.field("stock_id") == stock_id if stock_id else None
        day_table = day_ds.to_table(filter=day_filt, columns=["stock_id", "date", "close"])
        day_df = day_table.to_pandas()

        t2 = time.time()
        new_factor = compute_adjust_factor(tick_df, day_df)
        print(f"  計算 factor: {len(new_factor):,} 列 ({time.time()-t2:.2f}s)", flush=True)

        if factor_path.exists() and not force:
            old_factor = pd.read_parquet(factor_path)
            if stock_id:
                old_factor = old_factor[old_factor["stock_id"] != stock_id]
            final_factor = pd.concat([old_factor, new_factor], ignore_index=True)
            final_factor.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
            final_factor.sort_values(["stock_id", "date"], inplace=True)
        else:
            final_factor = new_factor

        final_factor.reset_index(drop=True, inplace=True)
        _atomic_to_parquet(final_factor, factor_path, index=False, compression="zstd")
        print(f"  寫入 {factor_path.name}: {len(final_factor):,} 列 ({time.time()-t1:.2f}s)", flush=True)

    print(f"\n除權息調整係數建置完成，總耗時 {time.time()-t0:.2f}s ✅", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="反推每日除權息調整係數")
    parser.add_argument("--incremental", action="store_true", default=True, help="增量更新 (預設 True)")
    parser.add_argument("--force", action="store_true", help="強制重新計算並覆蓋")
    parser.add_argument("--month", type=str, default=None, help="指定月份 (例: 2026_05)")
    parser.add_argument("--stock", type=str, default=None, help="指定股票代號 (例: 2330)")
    args = parser.parse_args()

    build(incremental=args.incremental, force=args.force, target_month=args.month, stock_id=args.stock)
