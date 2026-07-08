"""
從 HF Hub 下載訓練資料到本機 db/

用途：本機訓練前執行一次，把雲端最新資料拉下來。

下載內容：
    db/fugle_day/YYYY_M.parquet  ← day_trade/day/YYYY_M.parquet（增量：跳過本機已有的月份）
    db/m1/YYYY_M.parquet         ← day_trade/m1/YYYY_M.parquet（增量：跳過本機已有的月份）

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


def _normalize_month_stem(stem: str) -> str:
    """"2026_7" / "2026_07" 一律歸一成補零格式 "2026_07"。

    HF Hub 上早期上傳的月檔沒補零，近期（push_day_to_hf.py／push_m1_to_hf.py
    用 pd.Period.astype(str)）一律補零；下載時若直接沿用 HF 檔名，同一個月
    可能因為補零與否落地成本機兩個不同檔案，讀取時 schema 衝突。一律用補零
    檔名存本機，並用它做「是否已下載過」的判斷，避免重複／衝突。
    """
    year, month = stem.split("_")
    return f"{year}_{int(month):02d}"


# ── 1. 日K（增量：跳過本機已有的月份）────────────────────────────────────────
print("下載日K（db/fugle_day/）...")
import re as _re
day_files = [
    f for f in api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
    if _re.match(r"day_trade/day/\d{4}_\d+\.parquet$", f)  # 只取 YYYY_M.parquet，排除舊 fugle_day.parquet
]

if not day_files:
    print("  HF Hub 尚無日K資料")
else:
    dest_day_dir = _ROOT / "db/fugle_day"
    dest_day_dir.mkdir(parents=True, exist_ok=True)
    latest = max(_normalize_month_stem(Path(f).stem) for f in day_files)
    for hf_path in sorted(day_files):
        stem = _normalize_month_stem(Path(hf_path).stem)
        filename = f"{stem}.parquet"   # 本機一律用補零檔名，例如 2026_06.parquet
        dest = dest_day_dir / filename
        if dest.exists() and stem != latest:
            print(f"  跳過 {filename}（已有）")
            continue
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=hf_path,
            repo_type="dataset", token=HF_TOKEN,
        )
        shutil.copy2(local_path, dest)
        print(f"  → {dest}")

# ── 2. 分K（增量：跳過本機已有的月份）────────────────────────────────────────
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
    # 找出 HF Hub 上最新的月份（當月資料每天都在更新，需重新下載）
    latest = max(_normalize_month_stem(Path(f).stem) for f in m1_files)  # e.g. "2026_07"
    for hf_path in sorted(m1_files):
        stem = _normalize_month_stem(Path(hf_path).stem)
        filename = f"{stem}.parquet"   # 本機一律用補零檔名，例如 2026_06.parquet
        dest = dest_m1 / filename
        # 已有且不是當月 → 跳過
        if dest.exists() and stem != latest:
            print(f"  跳過 {filename}（已有）")
            continue
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        shutil.copy2(local_path, dest)
        print(f"  → {dest}")

print("完成")
