"""
GHA workflow 專用：跑 scripts/update_daily.py 之前，先把 HF Hub 上的 db/
「精簡子集」拉下來墊成本機起始狀態——不是整包下載，見下方「抓取範圍」。

⚠️ 跟 scripts/download_hf_for_local.py 不同：那支是給人手動執行（訓練
模型前），預設整包 db/ 全拉。這支只給 .github/workflows/update_daily.yml
呼叫，刻意分成兩個檔案——GHA runner 每次都是全新機器（見下方動機說明），
只需要「剛好夠讓增量/快路徑判斷正確運作」的最小子集，不需要、也不該像
本機訓練那樣拉完整歷史。

動機：GitHub Actions 的 runner 每次執行都是全新環境，db/ 不進版控，
update_daily.py 裡大部分增量/快路徑邏輯（`_last_stored_dates()`、
`build_tick_adjust_factor.py`/`build_adjustment_factor.py` 的全量重建、
`build_volume_profile.py`/`build_poc.py` 的「輸出檔案是否已存在」判斷）
都需要本機已經有資料才能正確運作。一開始直接整包拉 `db/`（1199個檔案）
實測在 GHA 上直接撞到 HF Hub 的 429 Too Many Requests（xet-read-token
端點請求太密集），逐一查證 update_daily.py 每個步驟的原始碼後確認：

    - `data/day_data_loader.py::_last_stored_dates()`／
      `data/m1_data_loader.py::_last_stored_dates_m1()`：只需要「最近的」
      存檔日期，不用全部歷史。
    - `data/build_tick_adjust_factor.py`／`data/build_adjustment_factor.py`：
      無條件全量重建（沒有incremental選項），呼叫時不帶 stock_id 就是
      直接覆蓋每個月份檔案（不會跟舊資料merge），等於每天執行都會用
      db/d1＋db/adjustment_day（本身很小）從頭產生一份新的
      db/adjustment_factor／db/tick_adjust_factor，**完全不需要預先
      下載這兩個輸出資料夾**。
    - `data/build_m3_m5_rolling.py`／`data/build_m3_m5_std.py`：
      逐月獨立計算（`_compute_m3()`/`_compute_m5()` 只吃單一個月的
      db/m1，沒有跨月依賴），**完全不需要預先下載 db/m3／db/m5／
      db/m3_std／db/m5_std**，每天會用（已拉到本機的近幾個月）db/m1
      重新算出對應月份、push 時只覆蓋這幾個月，舊月份不受影響。
    - `data/build_volume_profile.py`／`data/build_poc.py`：incremental
      判斷靠「輸出檔案（db/volume_profile／db/poc_day，都很小）存不
      存在」，來源 db/tick 只逐月讀取本機實際有的月份檔案，缺舊月份
      就直接跳過（不會報錯、也不會誤判成要重建），所以 db/tick 只要
      最近幾個月即可。

抓取範圍：
    - 全拉（小，合計數十MB量級）：d1、adjustment_day、volume_profile、
      poc_day、breakout_retest_day、breakout_retest_trigger、tickers、
      fubon_subscribe、margin、info，以及所有 *_flags。
    - 只拉最近 _RECENT_MONTHS_COUNT 個月：m1、tick。
    - 完全不拉：adjustment_factor、tick_adjust_factor、m3、m5、m3_std、
      m5_std（理由見上方）。
    本機比對過實際檔案數：這個範圍約362個檔案，比整包1199個少約70%。

需要的環境變數：
    HF_REPO_ID : HF dataset repo
    HF_TOKEN   : Hugging Face read token（repo 若為 private 一定要帶）

用法：
    python -m scripts.download_hf_for_gha
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_FULL_PULL_FOLDERS = [
    "d1", "adjustment_day",
    "volume_profile", "poc_day",
    "breakout_retest_day", "breakout_retest_trigger",
    "tickers", "fubon_subscribe", "margin", "info",
    "d1_flags", "adjustment_day_flags", "m1_flags", "tick_flags",
]
_RECENT_MONTHS_FOLDERS = ["m1", "tick"]
_RECENT_MONTHS_COUNT = 2


def _recent_year_months(n: int) -> list[str]:
    """回傳最近 n 個月的 "YYYY_MM" 字串（含當月），對齊各下載器的月份分檔
    命名慣例（例如 db/m1/2026_08.parquet）。"""
    months = []
    y, m = datetime.now().year, datetime.now().month
    for i in range(n):
        mm = m - i
        yy = y
        while mm <= 0:
            mm += 12
            yy -= 1
        months.append(f"{yy}_{mm:02d}")
    return months


def _allow_patterns() -> list[str]:
    patterns = [f"db/{name}/*" for name in _FULL_PULL_FOLDERS]
    recent = _recent_year_months(_RECENT_MONTHS_COUNT)
    for name in _RECENT_MONTHS_FOLDERS:
        for ym in recent:
            patterns.append(f"db/{name}/{ym}.parquet")
    return patterns


def main():
    if not HF_REPO_ID:
        raise RuntimeError("請設定 HF_REPO_ID 環境變數")

    allow_patterns = _allow_patterns()
    print(
        f"從 HF Hub 拉取 db/ 精簡子集（{_FULL_PULL_FOLDERS} 全拉，"
        f"{_RECENT_MONTHS_FOLDERS} 只拉最近{_RECENT_MONTHS_COUNT}個月）..."
    )
    # local_dir=_ROOT 讓 repo 裡的 db/... 路徑直接對應到本機的 db/...
    # （push_db_to_hf.py 用 path_in_repo="db" 上傳，兩邊路徑結構對稱）。
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns=allow_patterns,
        local_dir=str(_ROOT),
    )
    print("完成")


if __name__ == "__main__":
    main()
