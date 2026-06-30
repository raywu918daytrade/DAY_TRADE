"""
GH Actions 執行：下載當沖日K → 推到 HF Hub Dataset

需要的環境變數（GitHub Secrets）：
    FUGLE      : Fugle API Key
    HF_TOKEN   : Hugging Face write token
    HF_REPO_ID : HF dataset repo，例如 your-name/just1stock-data
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

_TW = timezone(timedelta(hours=8))
_ROOT = Path(__file__).parent.parent

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

if not HF_REPO_ID or not HF_TOKEN:
    raise RuntimeError("請設定 HF_REPO_ID 和 HF_TOKEN 環境變數")

api = HfApi()

# ── 1. 取得當沖標的清單 ────────────────────────────────────────────────────────
print("更新當沖標的清單...")
tickers_df = update_tickers()

if tickers_df.empty:
    # 盤後：從 HF Hub 取快取（上次盤中存的）
    print("  非盤中，從 HF Hub 取快取標的清單...")
    try:
        cached = hf_hub_download(
            repo_id=HF_REPO_ID, filename="fugle_tickers.parquet",
            repo_type="dataset", token=HF_TOKEN,
        )
        tickers_df = pd.read_parquet(cached)
        print(f"  快取標的：{len(tickers_df)} 支")
    except Exception as e:
        raise RuntimeError(
            f"無法取得當沖標的清單（非盤中且 HF Hub 無快取）: {e}\n"
            "請在盤中手動執行一次 push_to_hf.py 以建立快取"
        )
else:
    # 盤中拿到資料 → 順便更新 HF Hub 快取
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

# 寫入本機供 update_day 使用
tickers_local = _ROOT / "db/fugle_tickers"
tickers_local.mkdir(parents=True, exist_ok=True)
tickers_df.to_parquet(tickers_local / "tickers.parquet", index=False)
stocks = tickers_df["stock_id"].tolist()
print(f"  使用 {len(stocks)} 支標的")

# ── 2. 下載最近 60 天日K ────────────────────────────────────────────────────────
start_date = (datetime.now(_TW) - timedelta(days=60)).strftime("%Y-%m-%d")
print(f"下載日K：{start_date} 至今（當沖標的）...")
update_day(start_date=start_date, stocks=stocks)

# ── 3. 合併所有月份 parquet 成單一檔案 ─────────────────────────────────────────
import pyarrow.dataset as ds

day_dir = _ROOT / "db/fugle_day"
print("合併 parquet...")
df = ds.dataset(str(day_dir), format="parquet").to_table().to_pandas()
print(f"  共 {len(df):,} 筆，{df['stock_id'].nunique():,} 支股票")

out_path = _ROOT / "fugle_day.parquet"
df.to_parquet(out_path, index=False, compression="zstd")
print(f"  已存 {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

# ── 4. 推到 HF Hub ────────────────────────────────────────────────────────────
print(f"推送至 HF Hub: {HF_REPO_ID}...")
api.upload_file(
    path_or_fileobj=str(out_path),
    path_in_repo="fugle_day.parquet",
    repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN,
    commit_message=f"auto update {datetime.now(_TW).strftime('%Y-%m-%d')}",
)
out_path.unlink(missing_ok=True)
print("完成")
