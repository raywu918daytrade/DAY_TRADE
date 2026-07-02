"""
從 HF Hub 下載訓練資料到本機 db/

用途：本機訓練前執行一次，把雲端最新資料拉下來。

下載內容：
    db/fugle_day/fugle_day.parquet  ← day_trade/day/fugle_day.parquet
    db/m1/YYYY_M.parquet            ← day_trade/m1/YYYY_M.parquet（所有月份）

需要的環境變數（.env）：
    HF_REPO_ID : HF dataset repo
    HF_TOKEN   : Hugging Face read token（可省略，若 repo 為公開）
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

if not HF_REPO_ID:
    raise RuntimeError("請在 .env 設定 HF_REPO_ID")

api = HfApi()

# ── 1. 日K ────────────────────────────────────────────────────────────────────
print("下載日K（fugle_day.parquet）...")
local_day = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="day_trade/day/fugle_day.parquet",
    repo_type="dataset",
    token=HF_TOKEN,
)
dest_day = _ROOT / "db/fugle_day/fugle_day.parquet"
dest_day.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(local_day, dest_day)
print(f"  → {dest_day}")

# ── 2. 分K（所有月份檔）───────────────────────────────────────────────────────
print("下載分K（db/m1/）...")
m1_files = [
    f for f in api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
    if f.startswith("day_trade/m1/") and f.endswith(".parquet")
]

if not m1_files:
    print("  HF Hub 尚無分K資料（先跑 GHA update_m1 或 push_m1_to_hf.py）")
else:
    dest_m1 = _ROOT / "db/m1"
    dest_m1.mkdir(parents=True, exist_ok=True)
    for hf_path in sorted(m1_files):
        filename = Path(hf_path).name   # e.g. 2026_6.parquet
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        dest = dest_m1 / filename
        shutil.copy2(local_path, dest)
        print(f"  → {dest}")

print("完成")
