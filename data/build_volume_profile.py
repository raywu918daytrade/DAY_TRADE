"""
將 db/tick/ 逐筆成交資料，預先聚合出 db/volume_profile/ (Volume Profile 價位成交量分布)。

儲存格式：按月分檔 parquet (db/volume_profile/{YYYY_MM}.parquet)，欄位為：
    stock_id (str), date (str, YYYY-MM-DD), price (float32), volume (int64),
    buy_volume (int64), sell_volume (int64), neutral_volume (int64)

機制支援增量更新：當 db/tick/ 有新交易日或新月份時，自動檢測增量並合併寫入。

用法：
    python -m data.build_volume_profile               # 增量建置（預設）
    python -m data.build_volume_profile --force        # 強制全部重新計算
    python -m data.build_volume_profile --month 2026_05 # 指定月份
    python -m data.build_volume_profile --stock 2330    # 指定股票
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
_VP_DIR = _ROOT / "db/volume_profile"


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path | str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def compute_volume_profile(tick_df: pd.DataFrame) -> pd.DataFrame:
    """從逐筆成交明細 (tick) 計算 Volume Profile。

    欄位需求 (tick_df): stock_id, date, deal_price, volume, tick_type
    回傳欄位 (vp_df): stock_id, date, price, volume, buy_volume, sell_volume, neutral_volume
    """
    if tick_df.empty:
        return pd.DataFrame(
            columns=["stock_id", "date", "price", "volume", "buy_volume", "sell_volume", "neutral_volume"]
        )

    df = tick_df.copy()
    # 取日期前 10 碼 (YYYY-MM-DD)
    df["date"] = df["date"].astype(str).str[:10]
    df["price"] = df["deal_price"].round(2).astype("float32")

    vol = df["volume"].astype("int64")
    tick_type = df["tick_type"]

    df["buy_volume"] = (tick_type == 1).astype("int64") * vol
    df["sell_volume"] = (tick_type == 2).astype("int64") * vol
    df["neutral_volume"] = (tick_type == 0).astype("int64") * vol

    vp = df.groupby(["stock_id", "date", "price"], as_index=False).agg(
        volume=("volume", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        neutral_volume=("neutral_volume", "sum"),
    )

    vp["volume"] = vp["volume"].astype("int64")
    vp["buy_volume"] = vp["buy_volume"].astype("int64")
    vp["sell_volume"] = vp["sell_volume"].astype("int64")
    vp["neutral_volume"] = vp["neutral_volume"].astype("int64")

    return vp.sort_values(["stock_id", "date", "price"]).reset_index(drop=True)


def build(incremental: bool = True, force: bool = False, target_month: str | None = None, stock_id: str | None = None):
    t0 = time.time()
    _VP_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"開始 {mode_str} Volume Profile，共有 {len(tick_paths)} 個月份檔案...", flush=True)

    for tick_path in tick_paths:
        ym = tick_path.stem
        vp_path = _VP_DIR / f"{ym}.parquet"
        print(f"\n處理 {ym} ({tick_path.name})...", flush=True)
        t1 = time.time()

        # 增量檢查邏輯
        if incremental and not force and vp_path.exists():
            tick_mtime = tick_path.stat().st_mtime
            vp_mtime = vp_path.stat().st_mtime
            if vp_mtime >= tick_mtime and stock_id is None:
                print(f"  {ym}: 已是最新，無須更新 ✅ ({time.time()-t1:.2f}s)", flush=True)
                continue

            # 當 tick mtime 較新，或指定單一 stock_id 時，進行 (stock_id, date) 差集檢測
            tick_ds = ds.dataset(str(tick_path), format="parquet")
            filt = ds.field("stock_id") == stock_id if stock_id else None

            tick_table = tick_ds.to_table(filter=filt, columns=["stock_id", "date"])
            if tick_table.num_rows == 0:
                print(f"  {ym}: 無相符 tick 資料，跳過 ✅", flush=True)
                continue

            tick_df_meta = tick_table.to_pandas()
            tick_pairs = set(zip(tick_df_meta["stock_id"], tick_df_meta["date"].astype(str).str[:10]))

            vp_df_old = pd.read_parquet(vp_path, columns=["stock_id", "date"])
            if stock_id:
                vp_df_old = vp_df_old[vp_df_old["stock_id"] == stock_id]
            vp_pairs = set(zip(vp_df_old["stock_id"], vp_df_old["date"]))

            missing_pairs = tick_pairs - vp_pairs
            if not missing_pairs:
                print(f"  {ym}: 內容已有全部 (stock_id, date) 資料，無須更新 ✅ ({time.time()-t1:.2f}s)", flush=True)
                # 更新 vp 檔案 mtime，避免下次重複檢查
                vp_path.touch()
                continue
            print(f"  {ym}: 發現 {len(missing_pairs)} 組新的 (stock_id, date) 待增量更新...", flush=True)

        # 載入 tick 資料進行計算
        tick_ds = ds.dataset(str(tick_path), format="parquet")
        filt = ds.field("stock_id") == stock_id if stock_id else None

        tick_table = tick_ds.to_table(filter=filt)
        if tick_table.num_rows == 0:
            print(f"  {ym}: 無相符 tick 資料", flush=True)
            continue

        tick_df = tick_table.to_pandas()
        print(f"  載入 tick: {len(tick_df):,} 筆 ({time.time()-t1:.2f}s)", flush=True)

        t2 = time.time()
        new_vp = compute_volume_profile(tick_df)
        print(f"  計算 VP: {len(new_vp):,} 列 ({time.time()-t2:.2f}s)", flush=True)

        # 合併舊資料（如果存在且非 force）
        if vp_path.exists() and not force:
            old_vp = pd.read_parquet(vp_path)
            if stock_id:
                old_vp = old_vp[old_vp["stock_id"] != stock_id]
            final_vp = pd.concat([old_vp, new_vp], ignore_index=True)
            final_vp.drop_duplicates(subset=["stock_id", "date", "price"], keep="last", inplace=True)
            final_vp.sort_values(["stock_id", "date", "price"], inplace=True)
        else:
            final_vp = new_vp

        final_vp.reset_index(drop=True, inplace=True)
        _atomic_to_parquet(final_vp, vp_path, index=False, compression="zstd")
        print(f"  寫入 {vp_path.name}: {len(final_vp):,} 列 ({time.time()-t1:.2f}s)", flush=True)

    print(f"\nVolume Profile 建置完成，總耗時 {time.time()-t0:.2f}s ✅", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="預先聚合 Volume Profile")
    parser.add_argument("--incremental", action="store_true", default=True, help="增量更新 (預設 True)")
    parser.add_argument("--force", action="store_true", help="強制重新計算並覆蓋")
    parser.add_argument("--month", type=str, default=None, help="指定月份 (例: 2026_05)")
    parser.add_argument("--stock", type=str, default=None, help="指定股票代號 (例: 2330)")
    args = parser.parse_args()

    build(incremental=args.incremental, force=args.force, target_month=args.month, stock_id=args.stock)
