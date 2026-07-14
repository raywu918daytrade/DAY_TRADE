"""
GH Actions 執行：下載當沖分K歷史資料（~2787支，近30日）→ 合併 HF Hub 舊資料 → 推回 HF Hub

訓練資料用；盤中資料（500支/每分鐘）存 db/m1_live/，這份不在此處理。

流程：
    1. 取當沖標的清單（盤中直接拿；盤後從 HF Hub 快取）
    2. 循序下載所有股票近30日1分鐘K（Fugle /historical/candles，2.1s/支，約 100 分鐘）
    3. 按月份讀取本機 db/m1/，逐月與 HF Hub 舊資料合併後推回

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

from data.m1_data_loader import update_m1
from fubon.intraday_tickers import update_tickers

_TW = timezone(timedelta(hours=8))
_ROOT = Path(__file__).parent.parent

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

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
            repo_type="dataset", token=HF_TOKEN,
        )
        tickers_df = pd.read_parquet(cached)
        print(f"  快取標的：{len(tickers_df)} 支")
    except Exception as e:
        raise RuntimeError(
            f"無法取得當沖標的清單（非盤中且 HF Hub 無快取）: {e}\n"
            "請先執行 push_to_hf.py 建立標的快取"
        )
else:
    print(f"  {len(tickers_df)} 支")

tickers_local = _ROOT / "db/tickers"
tickers_local.mkdir(parents=True, exist_ok=True)
tickers_df.to_parquet(tickers_local / "tickers.parquet", index=False)
stocks = tickers_df["stock_id"].tolist()
print(f"  使用 {len(stocks)} 支標的")

# ── 2. 下載所有股票近30日分K ──────────────────────────────────────────────────
print(f"下載 1 分鐘K（{len(stocks)} 支，預計 {len(stocks) * 2.1 / 60:.0f} 分鐘）...")
update_m1(stocks=stocks)

# ── 3. 讀取本機 db/m1/，找出下載到的月份 ──────────────────────────────────────
m1_dir = _ROOT / "db/m1"
print("讀取本機分K資料...")
df_local = ds.dataset(str(m1_dir), format="parquet").to_table().to_pandas()
if df_local.empty:
    print("  無資料，結束")
    sys.exit(0)

df_local["_month"] = pd.to_datetime(df_local["date"]).dt.to_period("M").astype(str)
months = sorted(df_local["_month"].unique())
today = datetime.now(_TW).strftime("%Y-%m-%d")
print(f"  共 {len(df_local):,} 筆，月份：{months}")

# ── 4. 逐月與 HF Hub 合併後推回 ───────────────────────────────────────────────
for month_str in months:
    hf_filename = f"day_trade/m1/{month_str.replace('-', '_')}.parquet"
    df_new = df_local[df_local["_month"] == month_str].drop(columns=["_month"])

    # 嘗試下載 HF Hub 現有該月資料
    try:
        existing_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=hf_filename,
            repo_type="dataset", token=HF_TOKEN,
        )
        df_existing = pd.read_parquet(existing_path)
        print(f"  [{month_str}] HF Hub 現有 {len(df_existing):,} 筆，本次新增 {len(df_new):,} 筆")
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    except Exception:
        print(f"  [{month_str}] HF Hub 無現有資料，首次建立（{len(df_new):,} 筆）")
        df_merged = df_new

    df_merged.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    df_merged.sort_values(["date", "stock_id"], inplace=True)
    for col in ["open", "high", "low", "close"]:
        df_merged[col] = df_merged[col].astype("float32")
    df_merged["volume"] = df_merged["volume"].astype("int64")

    out_path = _ROOT / f"m1_{month_str.replace('-', '_')}.parquet"
    df_merged.to_parquet(out_path, index=False, compression="zstd")
    size_kb = out_path.stat().st_size / 1024

    print(f"  [{month_str}] 推送 {hf_filename}（{len(df_merged):,} 筆，{size_kb:.0f} KB）...")
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=hf_filename,
        repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN,
        commit_message=f"m1 update {today}",
    )
    out_path.unlink(missing_ok=True)
    print(f"  [{month_str}] 完成")

print("全部完成")
