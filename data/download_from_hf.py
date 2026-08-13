"""
從 HF Hub 下載最新資料到本機 db/——手動執行，不進 GHA 排程。

用途：本機訓練模型前手動跑一次，把雲端（GHA 排程執行 update_daily.py 後
用 scripts/push_db_to_hf.py 推上去的）最新資料拉下來，補齊本機缺的部分。

2026-08-13 重寫：舊版（已刪除，見 git log 5acdb84）對應的是舊的 HF Hub 佈局
（`day_trade/day/`、`day_trade/m1/`，且只涵蓋日K/分K兩種），現在
push_db_to_hf.py 改成把整個本機 `db/` 資料夾原樣鏡像到 HF Hub 的 `db/`
路徑下，這支下載端也對應改成直接鏡像下載（`snapshot_download()`），不用
再維護一份跟舊佈局綁死的路徑/檔名正規化邏輯。

需要的環境變數（.env）：
    HF_REPO_ID : HF dataset repo
    HF_TOKEN   : Hugging Face read token（repo 若為 private 一定要帶）

用法：
    python -m data.download_from_hf                # 下載整個 db/
    python -m data.download_from_hf --only m1 d1    # 只下載指定子資料夾
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def main(only: list[str] | None = None):
    if not HF_REPO_ID:
        raise RuntimeError("請在 .env 設定 HF_REPO_ID")

    if only:
        allow_patterns = [f"db/{name}/*" for name in only]
        print(f"從 HF Hub 下載 db/ 的子集：{only} ...")
    else:
        allow_patterns = ["db/*"]
        print("從 HF Hub 下載整個 db/ ...")

    # snapshot_download 本身就有本地快取比對（依檔案 etag/hash），已經下載過
    # 且雲端沒變動的檔案不會重複下載，適合每次都直接呼叫、不用自己維護
    # 「跳過已有月份」這種邏輯。local_dir=_ROOT 讓 repo 裡的 db/... 路徑直接
    # 對應到本機的 db/...（push_db_to_hf.py 用 path_in_repo="db" 上傳，
    # 兩邊路徑結構對稱）。
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns=allow_patterns,
        local_dir=str(_ROOT),
    )
    print("完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None, help="只下載指定的 db/ 子資料夾，例如 --only m1 d1")
    args = parser.parse_args()
    main(only=args.only)
