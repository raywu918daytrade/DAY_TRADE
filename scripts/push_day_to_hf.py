"""
GH Actions 執行：增量下載當沖日K → 按月份合併 HF Hub 舊資料 → 推回 HF Hub

流程：
    1. 取當沖標的清單（盤中直接拿；盤後從 HF Hub 快取）
    2. 從 HF Hub 現有月份檔取最後日期，決定增量起始日
    3. 只下載 last_date+1 以後的新資料
    4. 按月份與 HF Hub 舊資料合併後推回（只更新有變動的月份）

HF Hub 路徑：day_trade/day/YYYY_M.parquet（月份分檔，與 m1 一致）

需要的環境變數：
    FUGLE      : Fugle API Key
    HF_TOKEN   : Hugging Face write token
    HF_REPO_ID : HF dataset repo
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.day_data_loader import update_day
from fubon.intraday_tickers import update_tickers

_TW   = timezone(timedelta(hours=8))
_ROOT = Path(__file__).parent.parent

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

if not HF_REPO_ID or not HF_TOKEN:
    raise RuntimeError("請設定 HF_REPO_ID 和 HF_TOKEN 環境變數")

api = HfApi()

# ── 1. 取當沖標的清單 ─────────────────────────────────────────────────────────
print("更新當沖標的清單...")
tickers_df = update_tickers()

if tickers_df.empty:
    print("  非盤中，從 HF Hub 取快取標的清單...")
    try:
        cached = hf_hub_download(
            repo_id=HF_REPO_ID, filename="day_trade/tickers/tickers.parquet",
            repo_type="dataset", token=None,
        )
        tickers_df = pd.read_parquet(cached)
        print(f"  快取標的：{len(tickers_df)} 支")
    except Exception as e:
        raise RuntimeError(
            f"無法取得當沖標的清單（非盤中且 HF Hub 無快取）: {e}\n"
            "請在盤中手動執行一次 push_day_to_hf.py 以建立快取"
        )
else:
    print(f"  {len(tickers_df)} 支，更新 HF Hub 快取...")
    tickers_out = _ROOT / "tickers.parquet"
    tickers_df.to_parquet(tickers_out, index=False)
    api.upload_file(
        path_or_fileobj=str(tickers_out),
        path_in_repo="day_trade/tickers/tickers.parquet",
        repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN,
        commit_message=f"update tickers {datetime.now(_TW).strftime('%Y-%m-%d')}",
    )
    tickers_out.unlink(missing_ok=True)

tickers_local = _ROOT / "db/tickers"
tickers_local.mkdir(parents=True, exist_ok=True)
tickers_df.to_parquet(tickers_local / "tickers.parquet", index=False)
stocks = tickers_df["stock_id"].tolist()
print(f"  使用 {len(stocks)} 支標的")

# ── 2. 從 HF Hub 取現有月份清單，決定增量起始日 ───────────────────────────────
print("掃描 HF Hub 現有日K月份...")
import re as _re
day_files = [
    f for f in api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
    if _re.match(r"day_trade/day/\d{4}_\d+\.parquet$", f)
]

def _month_key(f):
    m = _re.search(r"(\d{4})_(\d+)\.parquet$", f)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

last_date = None
if day_files:
    latest_file = max(day_files, key=_month_key)  # 按年月數字排，避免 2026_7 > 2026_10
    try:
        existing_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=latest_file,
            repo_type="dataset", token=HF_TOKEN,
        )
        df_latest = pd.read_parquet(existing_path)
        last_date = pd.to_datetime(df_latest["date"]).max()
        print(f"  最新月份 {Path(latest_file).stem}，資料至 {last_date.date()}")
    except Exception:
        pass

today = datetime.now(_TW).strftime("%Y-%m-%d")
if last_date is not None:
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
else:
    start_date = (datetime.now(_TW) - timedelta(days=60)).strftime("%Y-%m-%d")
    print(f"  HF Hub 無現有資料，下載最近 60 天（{start_date}）")

if start_date > today:
    print("資料已是最新，無需更新")
    sys.exit(0)

print(f"  增量起始：{start_date}")

# ── 3. 下載增量資料 ───────────────────────────────────────────────────────────
print(f"下載日K：{start_date} 至 {today}...")
update_day(start_date=start_date, stocks=stocks)

# ── 4. 讀本機新資料，按月份與 HF Hub 合併後推回 ───────────────────────────────
day_dir = _ROOT / "db/fugle_day"
df_new = ds.dataset(str(day_dir), format="parquet").to_table().to_pandas()
df_new = df_new[df_new["date"] >= start_date]
df_new["_month"] = pd.to_datetime(df_new["date"]).dt.to_period("M").astype(str)

for month_str, group in df_new.groupby("_month"):
    group = group.drop(columns=["_month"])
    hf_filename = f"day_trade/day/{month_str.replace('-', '_')}.parquet"

    # 下載該月 HF Hub 舊資料（若有）
    try:
        existing_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=hf_filename,
            repo_type="dataset", token=HF_TOKEN,
        )
        df_existing = pd.read_parquet(existing_path)
        print(f"  [{month_str}] HF Hub 現有 {len(df_existing):,} 筆，本次新增 {len(group):,} 筆")
        df_merged = pd.concat([df_existing, group], ignore_index=True)
    except Exception:
        print(f"  [{month_str}] 首次建立（{len(group):,} 筆）")
        df_merged = group

    df_merged.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    df_merged.sort_values(["date", "stock_id"], inplace=True)

    out_path = _ROOT / f"day_{month_str.replace('-', '_')}.parquet"
    df_merged.to_parquet(out_path, index=False, compression="zstd")
    size_kb = out_path.stat().st_size / 1024

    print(f"  [{month_str}] 推送 {hf_filename}（{len(df_merged):,} 筆，{size_kb:.0f} KB）...")
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=hf_filename,
        repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN,
        commit_message=f"day update {today}",
    )
    out_path.unlink(missing_ok=True)
    print(f"  [{month_str}] 完成")

print("完成")
