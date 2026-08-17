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

2026-08-17遇過：全量下載卡在最後一兩個檔案，報
`RuntimeError: File size mismatch: expected X bytes but downloaded Y bytes`
（xet 傳輸協定）或對應的 `OSError: Consistency check failed`（一般 HTTP），
且同一個檔案重試多次報的 expected/實際 bytes 數字都一樣——不是網路不穩,
換一次性關掉 xet 改走一般 HTTP 下載通常能繞過：
    HF_HUB_DISABLE_XET=1 python -m scripts.download_hf_for_local
如果還是同一個檔案報同樣的 mismatch，改用 `--only` 分批下載繞過那個資料夾，
問題出在 HF Hub 上那個特定 LFS 物件本身（不是本機端），要用
scripts/push_db_to_hf.py 重新推一次正確版本才會根治。

--repo-id：選填，覆蓋 .env 的 HF_REPO_ID，改從指定的其他 HF dataset repo
下載。用途：一次性從別的repo補資料，不影響 .env 設定的預設repo（不帶這個
參數時行為完全不變）。

2026-08-16例：db/margin（融資券）、db/ib（法人買賣，FinMind
TaiwanStockInstitutionalInvestorsBuySell，欄位 stock_id/date/name/buy/sell）
這兩份資料舊帳號的 repo 上已經有（本地 db/margin 只到 2026-08-03，
db/ib 本地還不存在），用這個指令一次補齊：
    python -m scripts.download_hf_for_local --only margin ib --repo-id raywu918python/j1s-data
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


def main(only: list[str] | None = None, repo_id: str | None = None):
    repo_id = repo_id or HF_REPO_ID
    if not repo_id:
        raise RuntimeError("請在 .env 設定 HF_REPO_ID，或用 --repo-id 指定要下載的 repo")

    if only:
        allow_patterns = [f"db/{name}/*" for name in only]
        print(f"從 HF Hub（{repo_id}）下載 db/ 的子集：{only} ...")
    else:
        allow_patterns = ["db/*"]
        print(f"從 HF Hub（{repo_id}）下載整個 db/（全量，檔案數多時可能較久，甚至撞到HF rate limit——GHA自動化用的是 scripts/download_hf_for_gha.py，不是這支）...")

    # snapshot_download 本身就有本地快取比對（依檔案 etag/hash），已經下載過
    # 且雲端沒變動的檔案不會重複下載，適合每次都直接呼叫、不用自己維護
    # 「跳過已有月份」這種邏輯。local_dir=_ROOT 讓 repo 裡的 db/... 路徑直接
    # 對應到本機的 db/...（push_db_to_hf.py 用 path_in_repo="db" 上傳，
    # 兩邊路徑結構對稱）。
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns=allow_patterns,
        local_dir=str(_ROOT),
    )
    print("完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None, help="只下載指定的 db/ 子資料夾，例如 --only m1 d1")
    parser.add_argument("--repo-id", default=None, help="覆蓋 .env 的 HF_REPO_ID，改從指定的其他 HF dataset repo 下載")
    args = parser.parse_args()
    main(only=args.only, repo_id=args.repo_id)
