"""
GH Actions / 手動執行：把本機整個 db/ 資料夾鏡像同步推到 HF Hub dataset repo。

跟舊版 push_m1_to_hf.py／push_day_to_hf.py（已刪除）不同：這支不會自己重新
下載資料、也不逐月跟 HF Hub 現有資料 merge——db/ 底下每支下載器
（data/m1_data_loader.py、data/day_data_loader.py、fubon/tick_api.py 等）
寫入本機時本身就是「累積式」（新的一天資料 merge 進既有月份檔案，不是只留
近30天），本機 db/ 已經是完整歷史的 authoritative 來源，直接整個資料夾
鏡像同步上去即可，不用再繞一次「下載→跟HF合併→推回」。

用 huggingface_hub 的 upload_folder()：內部會比對雲端現有內容的 hash，只
上傳有變動/新增的檔案，一次 commit，適合每天在 update_daily.py 跑完之後
執行（增量同步，不是每次都整包重傳）。

需要的環境變數：
    HF_TOKEN   : Hugging Face write token（要有 write 權限）
    HF_REPO_ID : HF dataset repo，格式 "<user>/<repo>"

用法：
    python -m scripts.push_db_to_hf
    python -m scripts.push_db_to_hf --only tickers 更新 hf 清單
    python -m scripts.push_db_to_hf --only m1        # 只推指定的 db/ 子資料夾（可多個）
    python -m scripts.push_db_to_hf --path-in-repo db  # 預設值，通常不用改
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# _atomic_to_parquet()（各下載器共用的原子寫入 helper）寫入中會先產生 .tmp
# 暫存檔再 rename，正常情況下不該遺留在 db/ 底下，但防呆排除，避免不小心把
# 寫入中的半成品推上去。
#
# db/m1_live 是盤中即時分K，本機自己有就好、不需要也不該同步到 HF Hub：
# 2026-08-17發現全量下載會卡在這個資料夾裡的檔案報 size mismatch
# （HF Hub上該LFS物件本身壞掉），乾脆從源頭排除不上傳。
#
# 2026-08-19：db/fubon_subscribe/subscribe_list.parquet 已經合併進
# db/tickers/tick_universe.parquet（見 fubon/subscribe_list.py 檔頭說明），
# 不再是獨立檔案，這條 ignore pattern 拿掉——tick_universe.parquet 本身
# 要同步（avg_volume/atr_pct/day_trade_tier 這些訓練會用到的欄位），裡面
# 多帶的 daytrade_ok/connection_id/verify_date 幾欄是本機當天驗證結果，
# 同步過去不會造成問題（下次任一環境的 premarket 流程重新驗證就會覆蓋）。
_IGNORE_PATTERNS = ["*.tmp", "**/*.tmp", "m1_live/*"]


def main(path_in_repo: str = "db", only: list[str] | None = None):
    if not HF_REPO_ID or not HF_TOKEN:
        raise RuntimeError("請設定 HF_REPO_ID 和 HF_TOKEN 環境變數")

    db_dir = _ROOT / "db"
    if not db_dir.exists():
        raise RuntimeError(f"{db_dir} 不存在")

    allow_patterns = [f"{name}/*" for name in only] if only else None

    api = HfApi()
    # dataset repo 若還不存在，第一次呼叫直接建立（exist_ok=True 避免重跑報錯）。
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN, private=True, exist_ok=True)

    if only:
        print(f"開始同步 {db_dir} 的子集 {only} → HF Hub {HF_REPO_ID}/{path_in_repo}/ ...")
    else:
        print(f"開始同步 {db_dir} → HF Hub {HF_REPO_ID}/{path_in_repo}/ ...")
    api.upload_folder(
        folder_path=str(db_dir),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns=allow_patterns,
        ignore_patterns=_IGNORE_PATTERNS,
        commit_message="db/ sync",
    )
    print("同步完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-in-repo", default="db")
    parser.add_argument("--only", nargs="+", default=None, help="只推指定的 db/ 子資料夾，例如 --only m1 d1")
    args = parser.parse_args()
    main(path_in_repo=args.path_in_repo, only=args.only)
