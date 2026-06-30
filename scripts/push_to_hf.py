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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tay_trade.day_data_loader import update_day
from tay_trade.fugle_tickers import update_tickers

_TW = timezone(timedelta(hours=8))

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

if not HF_REPO_ID or not HF_TOKEN:
    raise RuntimeError("請設定 HF_REPO_ID 和 HF_TOKEN 環境變數")

# ── 1. 更新當沖清單 ────────────────────────────────────────────────────────────
print("更新當沖標的清單...")
update_tickers()

# ── 2. 下載最近 60 天日K（含足夠的 lag 特徵 warmup）─────────────────────────
start_date = (datetime.now(_TW) - timedelta(days=60)).strftime("%Y-%m-%d")
print(f"下載日K：{start_date} 至今（當沖標的）...")
update_day(start_date=start_date)

# ── 3. 合併所有月份 parquet 成單一檔案 ─────────────────────────────────────────
import pyarrow.dataset as ds
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).parent.parent
day_dir = _ROOT / "db/fugle_day"
print("合併 parquet...")
df = ds.dataset(str(day_dir), format="parquet").to_table().to_pandas()
print(f"  共 {len(df):,} 筆，{df['stock_id'].nunique():,} 支股票")

out_path = _ROOT / "fugle_day.parquet"
df.to_parquet(out_path, index=False, compression="zstd")
print(f"  已存 {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

# ── 4. 推到 HF Hub ────────────────────────────────────────────────────────────
from huggingface_hub import HfApi

print(f"推送至 HF Hub: {HF_REPO_ID}...")
api = HfApi()
api.upload_file(
    path_or_fileobj=str(out_path),
    path_in_repo="fugle_day.parquet",
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message=f"auto update {datetime.now(_TW).strftime('%Y-%m-%d')}",
)
print("完成")
