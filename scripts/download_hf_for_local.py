"""
從 HF Hub 下載資料到本機 db/——給人手動執行用（訓練模型前），不進 GHA 排程。

用途：本機訓練模型前手動跑一次，把雲端（GHA 排程執行 update_daily.py 後
用 scripts/push_db_to_hf.py 推上去的）最新資料拉下來，補齊本機缺的部分。
預設整包 `db/` 全拉（訓練通常需要完整歷史，不是只要「最近」），也可以
用 `--only` 只拉需要的子資料夾，節省時間。

⚠️ 跟 scripts/download_hf_for_gha.py 不同：那支是給 GHA workflow
（update_daily.yml）在跑 update_daily.py **之前**自動呼叫的，只拉「最近
幾個月＋小資料夾」這個精簡子集（讓增量/快路徑判斷能正確運作就好，不需要
完整歷史），刻意跟這支分開成兩個檔案、兩個不同的預設行為——這支（本機
手動、預設全拉）跟那支（GHA自動、預設精簡）用途/預設值本來就不一樣，
混在同一支腳本裡容易搞混哪個情境該用什麼參數。2026-08-13使用者要求
用 `download_hf_for_local.py`／`download_hf_for_gha.py` 這組檔名，比
原本的 `download_from_hf.py`／`pull_recent_from_hf.py` 更直接看得出
「哪支給哪個情境用」。

2026-08-13 重寫：舊版（已刪除，見 git log 5acdb84）對應的是舊的 HF Hub 佈局
（`day_trade/day/`、`day_trade/m1/`，且只涵蓋日K/分K兩種），現在
push_db_to_hf.py 改成把整個本機 `db/` 資料夾原樣鏡像到 HF Hub 的 `db/`
路徑下，這支下載端也對應改成直接鏡像下載（`snapshot_download()`）。

需要的環境變數（.env）：
    HF_REPO_ID : HF dataset repo
    HF_TOKEN   : Hugging Face read token（repo 若為 private 一定要帶）

用法：
    python -m scripts.download_hf_for_local                # 下載整個 db/
    python -m scripts.download_hf_for_local --only m1 d1    # 只下載指定子資料夾
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
        print("從 HF Hub 下載整個 db/（全量，檔案數多時可能較久，甚至撞到HF rate limit——GHA自動化用的是 scripts/download_hf_for_gha.py，不是這支）...")

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
