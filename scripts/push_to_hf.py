"""
GH Actions 執行：增量下載當沖日K → 合併 HF Hub 舊資料 → 推回 HF Hub

流程：
    1. 取當沖標的清單（盤中直接拿；盤後從 HF Hub 快取）
    2. 從 HF Hub 下載現有 fugle_day.parquet，取最後日期
    3. 只下載 last_date+1 以後的新資料
    4. 新舊合併後推回 HF Hub

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
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tay_trade.day_data_loader import update_day
from tay_trade.fugle_tickers import update_tickers

_TW  = timezone(timedelta(hours=8))
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
            repo_id=HF_REPO_ID, filename="fugle_tickers.parquet",
            repo_type="dataset", token=None,
        )
        tickers_df = pd.read_parquet(cached)
        print(f"  快取標的：{len(tickers_df)} 支")
    except Exception as e:
        raise RuntimeError(
            f"無法取得當沖標的清單（非盤中且 HF Hub 無快取）: {e}\n"
            "請在盤中手動執行一次 push_to_hf.py 以建立快取"
        )
else:
    print(f"  {len(tickers_df)} 支，更新 HF Hub 快取...")
    tickers_out = _ROOT / "fugle_tickers.parquet"
    tickers_df.to_parquet(tickers_out, index=False)
    api.upload_file(
        path_or_fileobj=str(tickers_out),
        path_in_repo="fugle_tickers.parquet",
        repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN,
        commit_message=f"update tickers {datetime.now(_TW).strftime('%Y-%m-%d')}",
    )
    tickers_out.unlink(missing_ok=True)

tickers_local = _ROOT / "db/fugle_tickers"
tickers_local.mkdir(parents=True, exist_ok=True)
tickers_df.to_parquet(tickers_local / "tickers.parquet", index=False)
stocks = tickers_df["stock_id"].tolist()
print(f"  使用 {len(stocks)} 支標的")

# ── 2. 從 HF Hub 取現有資料，決定增量起始日 ──────────────────────────────────
print("從 HF Hub 下載現有日K...")
try:
    existing_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="fugle_day.parquet",
        repo_type="dataset", token=HF_TOKEN,
    )
    df_existing = pd.read_parquet(existing_path)
    last_date = pd.to_datetime(df_existing["date"]).max()
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  現有資料至 {last_date.date()}，增量起始：{start_date}")
except Exception:
    df_existing = pd.DataFrame()
    start_date = (datetime.now(_TW) - timedelta(days=60)).strftime("%Y-%m-%d")
    print(f"  HF Hub 無現有資料，下載最近 60 天（{start_date}）")

today = datetime.now(_TW).strftime("%Y-%m-%d")
if start_date > today:
    print("資料已是最新，無需更新")
    sys.exit(0)

# ── 3. 下載增量資料 ──────────────────────────────────────────────────────────
print(f"下載日K：{start_date} 至 {today}...")
update_day(start_date=start_date, stocks=stocks)

# ── 4. 合併本地新資料 + HF Hub 舊資料 ────────────────────────────────────────
import pyarrow.dataset as ds

day_dir = _ROOT / "db/fugle_day"
print("合併資料...")
df_new = ds.dataset(str(day_dir), format="parquet").to_table().to_pandas()
df_new = df_new[df_new["date"] >= start_date]   # 只取新資料

if not df_existing.empty:
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    df_all.sort_values(["date", "stock_id"], inplace=True)
else:
    df_all = df_new

print(f"  合併後共 {len(df_all):,} 筆，{df_all['stock_id'].nunique():,} 支，"
      f"日期 {df_all['date'].min()} ~ {df_all['date'].max()}")

# ── 5. 推回 HF Hub ────────────────────────────────────────────────────────────
out_path = _ROOT / "fugle_day.parquet"
df_all.to_parquet(out_path, index=False, compression="zstd")
print(f"  檔案大小：{out_path.stat().st_size / 1024:.0f} KB")

print(f"推送至 HF Hub: {HF_REPO_ID}...")
api.upload_file(
    path_or_fileobj=str(out_path),
    path_in_repo="fugle_day.parquet",
    repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN,
    commit_message=f"incremental update {datetime.now(_TW).strftime('%Y-%m-%d')}",
)
out_path.unlink(missing_ok=True)
print("完成")
