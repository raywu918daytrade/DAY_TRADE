"""
將 db/volume_profile/ 價位成交量分布資料，計算並預聚合出每日 POC (Point of Control) 與 Value Area 關鍵價位。

存入 db/poc_day/{YYYY_MM}.parquet，欄位包含：
    stock_id (str), date (str, YYYY-MM-DD), poc (float32), poc_volume (int64),
    pocs (str, 逗號分隔的顯著POC價位), poc_count (int8), profile_type (str, 'single'/'multi'),
    vah (float32), val (float32), total_volume (int64)

機制支援增量更新：當 db/volume_profile/ 有新交易日或新月份時，自動檢測增量並合併寫入。

用法：
    python -m data.build_poc               # 增量建置（預設）
    python -m data.build_poc --force        # 強制全部重新計算
    python -m data.build_poc --month 2026_05 # 指定月份
    python -m data.build_poc --stock 2330    # 指定股票
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

_VP_DIR = _ROOT / "db/volume_profile"
_POC_DIR = _ROOT / "db/poc_day"


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path | str, **kwargs):
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀"""
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def compute_poc_dataframe(
    vp_df: pd.DataFrame, threshold_ratio: float = 0.5, min_price_dist_pct: float = 0.005
) -> pd.DataFrame:
    """從 Volume Profile 計算每日 POC、多重 POCs、VAH (Value Area High)、VAL (Value Area Low)。

    傳入 vp_df 欄位需求: stock_id, date, price, volume
    回傳 df 欄位: stock_id, date, poc, poc_volume, pocs, poc_count, profile_type, vah, val, total_volume
    """
    if vp_df.empty:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "date",
                "poc",
                "poc_volume",
                "pocs",
                "poc_count",
                "profile_type",
                "vah",
                "val",
                "total_volume",
            ]
        )

    records = []
    groups = vp_df.groupby(["stock_id", "date"], sort=False)

    for (stock_id, date_str), g in groups:
        g = g.sort_values("price")
        prices = g["price"].values
        vols = g["volume"].values
        total_vol = int(vols.sum())

        if total_vol == 0:
            continue

        max_idx = int(np.argmax(vols))
        max_vol = int(vols[max_idx])
        primary_poc = float(prices[max_idx])

        # Value Area (70% total volume)
        target_vol = 0.7 * total_vol
        cum_vol = max_vol
        va_indices = {max_idx}
        L = max_idx - 1
        R = max_idx + 1
        n = len(vols)

        while cum_vol < target_vol and (L >= 0 or R < n):
            if L >= 0 and R < n:
                vl = vols[L]
                vr = vols[R]
                if vl > vr:
                    va_indices.add(L)
                    cum_vol += vl
                    L -= 1
                elif vr > vl:
                    va_indices.add(R)
                    cum_vol += vr
                    R += 1
                else:
                    va_indices.add(L)
                    va_indices.add(R)
                    cum_vol += vl + vr
                    L -= 1
                    R += 1
            elif L >= 0:
                vl = vols[L]
                va_indices.add(L)
                cum_vol += vl
                L -= 1
            else:
                vr = vols[R]
                va_indices.add(R)
                cum_vol += vr
                R += 1

        va_prices = prices[list(va_indices)]
        vah = float(va_prices.max())
        val = float(va_prices.min())

        # 檢測波峰 (Local Maxima)
        peaks = []
        if n == 1:
            peaks.append(0)
        else:
            for i in range(n):
                if i == 0 and vols[0] > vols[1]:
                    peaks.append(0)
                elif i == n - 1 and vols[n - 1] > vols[n - 2]:
                    peaks.append(n - 1)
                elif 0 < i < n - 1 and vols[i] > vols[i - 1] and vols[i] > vols[i + 1]:
                    peaks.append(i)

        # 篩選成交量達到門檻之波峰 (volume >= max_vol * threshold_ratio)
        cand_peaks = [i for i in peaks if vols[i] >= threshold_ratio * max_vol]
        cand_peaks.sort(key=lambda i: vols[i], reverse=True)

        accepted = []
        for i in cand_peaks:
            p = prices[i]
            if all(abs(p - prices[a]) / prices[a] >= min_price_dist_pct for a in accepted):
                accepted.append(i)

        poc_prices = [float(prices[a]) for a in accepted]
        pocs_str = ",".join([f"{p:.2f}" for p in poc_prices])
        poc_count = len(poc_prices)
        profile_type = "single" if poc_count == 1 else "multi"

        records.append(
            {
                "stock_id": str(stock_id),
                "date": str(date_str),
                "poc": primary_poc,
                "poc_volume": max_vol,
                "pocs": pocs_str,
                "poc_count": poc_count,
                "profile_type": profile_type,
                "vah": vah,
                "val": val,
                "total_volume": total_vol,
            }
        )

    res = pd.DataFrame(records)
    if res.empty:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "date",
                "poc",
                "poc_volume",
                "pocs",
                "poc_count",
                "profile_type",
                "vah",
                "val",
                "total_volume",
            ]
        )

    res["poc"] = res["poc"].astype("float32")
    res["poc_volume"] = res["poc_volume"].astype("int64")
    res["poc_count"] = res["poc_count"].astype("int8")
    res["vah"] = res["vah"].astype("float32")
    res["val"] = res["val"].astype("float32")
    res["total_volume"] = res["total_volume"].astype("int64")

    return res.sort_values(["stock_id", "date"]).reset_index(drop=True)


def build(incremental: bool = True, force: bool = False, target_month: str | None = None, stock_id: str | None = None):
    t0 = time.time()
    _POC_DIR.mkdir(parents=True, exist_ok=True)

    vp_paths = sorted(_VP_DIR.glob("*.parquet"))
    if not vp_paths:
        print("錯誤: db/volume_profile/ 中沒有資料，請先執行 data.build_volume_profile")
        return

    if target_month:
        norm_month = target_month.replace("-", "_")
        vp_paths = [p for p in vp_paths if p.stem == norm_month or p.stem == target_month]
        if not vp_paths:
            print(f"錯誤: 找不到月份 {target_month} 的 volume profile parquet 檔案")
            return

    mode_str = "完整重建" if force else ("增量建置" if incremental else "全量處理")
    print(f"開始 {mode_str} 每日 POC，共有 {len(vp_paths)} 個月份檔案...", flush=True)

    for vp_path in vp_paths:
        ym = vp_path.stem
        poc_path = _POC_DIR / f"{ym}.parquet"
        print(f"\n處理 {ym} ({vp_path.name})...", flush=True)
        t1 = time.time()

        # 增量檢查邏輯
        if incremental and not force and poc_path.exists():
            vp_mtime = vp_path.stat().st_mtime
            poc_mtime = poc_path.stat().st_mtime
            if poc_mtime >= vp_mtime and stock_id is None:
                print(f"  {ym}: 已是最新，無須更新 ✅ ({time.time()-t1:.2f}s)", flush=True)
                continue

            vp_ds = ds.dataset(str(vp_path), format="parquet")
            filt = ds.field("stock_id") == stock_id if stock_id else None

            vp_table = vp_ds.to_table(filter=filt, columns=["stock_id", "date"])
            if vp_table.num_rows == 0:
                print(f"  {ym}: 無相符 volume profile 資料，跳過 ✅", flush=True)
                continue

            vp_df_meta = vp_table.to_pandas()
            vp_pairs = set(zip(vp_df_meta["stock_id"], vp_df_meta["date"]))

            poc_df_old = pd.read_parquet(poc_path, columns=["stock_id", "date"])
            if stock_id:
                poc_df_old = poc_df_old[poc_df_old["stock_id"] == stock_id]
            poc_pairs = set(zip(poc_df_old["stock_id"], poc_df_old["date"]))

            missing_pairs = vp_pairs - poc_pairs
            if not missing_pairs:
                print(f"  {ym}: 內容已有全部 (stock_id, date) 資料，無須更新 ✅ ({time.time()-t1:.2f}s)", flush=True)
                poc_path.touch()
                continue
            print(f"  {ym}: 發現 {len(missing_pairs)} 組新的 (stock_id, date) 待增量更新...", flush=True)

        # 載入 volume profile 資料進行計算
        vp_ds = ds.dataset(str(vp_path), format="parquet")
        filt = ds.field("stock_id") == stock_id if stock_id else None

        vp_table = vp_ds.to_table(filter=filt)
        if vp_table.num_rows == 0:
            print(f"  {ym}: 無相符 volume profile 資料", flush=True)
            continue

        vp_df = vp_table.to_pandas()
        print(f"  載入 VP: {len(vp_df):,} 筆 ({time.time()-t1:.2f}s)", flush=True)

        t2 = time.time()
        new_poc = compute_poc_dataframe(vp_df)
        print(f"  計算 POC: {len(new_poc):,} 列 ({time.time()-t2:.2f}s)", flush=True)

        # 合併舊資料
        if poc_path.exists() and not force:
            old_poc = pd.read_parquet(poc_path)
            if stock_id:
                old_poc = old_poc[old_poc["stock_id"] != stock_id]
            final_poc = pd.concat([old_poc, new_poc], ignore_index=True)
            final_poc.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
            final_poc.sort_values(["stock_id", "date"], inplace=True)
        else:
            final_poc = new_poc

        final_poc.reset_index(drop=True, inplace=True)
        _atomic_to_parquet(final_poc, poc_path, index=False, compression="zstd")
        print(f"  寫入 {poc_path.name}: {len(final_poc):,} 列 ({time.time()-t1:.2f}s)", flush=True)

    print(f"\n每日 POC 建置完成，總耗時 {time.time()-t0:.2f}s ✅", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="預先聚合每日 POC 與 Value Area")
    parser.add_argument("--incremental", action="store_true", default=True, help="增量更新 (預設 True)")
    parser.add_argument("--force", action="store_true", help="強制重新計算並覆蓋")
    parser.add_argument("--month", type=str, default=None, help="指定月份 (例: 2026_05)")
    parser.add_argument("--stock", type=str, default=None, help="指定股票代號 (例: 2330)")
    args = parser.parse_args()

    build(incremental=args.incremental, force=args.force, target_month=args.month, stock_id=args.stock)
