"""
將 db/m1/ 的 1 分鐘 K 線，預先聚合出 db/m3/（3分鐘K）和 db/m5/（5分鐘K）。

儲存格式與 db/m1/ 一致：按月分檔 parquet，欄位為
  stock_id, date, open, high, low, close, volume

每分鐘一筆（rolling 聚合），不是獨立 K 線週期。

用法：
  python -m data.build_m3_m5              # 完整重建（全部月份）
  python -m data.build_m3_m5 --incremental # 只重建比 m1 舊的月份
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from data.query import load_m1
from strategy.rally.features import compute_m3 as _compute_m3
from strategy.rally.features import compute_m5 as _compute_m5

_M1_DIR = _ROOT / "db/m1"
_M3_DIR = _ROOT / "db/m3"
_M5_DIR = _ROOT / "db/m5"


def _month_key(path: Path) -> str:
    """從 m1 檔名抽出 '2026_6' 這樣的 key。"""
    return path.stem  # "2026_6.parquet" → "2026_6"


def _needs_update(target_dir: Path, month_key: str, m1_mtime: float) -> bool:
    """檢查 target_dir/{month_key}.parquet 是否比 m1_mtime 新。"""
    target = target_dir / f"{month_key}.parquet"
    if not target.exists():
        return True
    return target.stat().st_mtime < m1_mtime


def _save_monthly(df: pd.DataFrame, target_dir: Path, prefix: str):
    """按年月分檔存 parquet。月份補零（"2026_07" 不是 "2026_7"），
    要跟 db/m1/ 既有檔名一致——不補零會跟 db/m1/ 對不上，load_m3()/load_m5()
    用 ds.dataset() 整個資料夾一起讀，新舊檔名並存會把同一個月的資料算兩次
    （2026-07-08 修過同一種 bug，這支腳本當時漏改）。"""
    df["date"] = pd.to_datetime(df["date"])
    df["_ym"] = df["date"].dt.strftime("%Y_%m")
    target_dir.mkdir(parents=True, exist_ok=True)
    for ym, g in df.groupby("_ym"):
        path = target_dir / f"{ym}.parquet"
        g.drop(columns=["_ym"]).to_parquet(path)
        print(f"  {prefix} {path.name}: {len(g):,} 筆")


def build(incremental: bool = False):
    t0 = time.time()

    # 收集 m1 所有月份
    m1_paths = sorted(_M1_DIR.glob("*.parquet"))
    if not m1_paths:
        print("錯誤: db/m1/ 中沒有資料")
        return

    if incremental:
        # 只重建 m1 更新過的月份
        todo = []
        for p in m1_paths:
            mk = _month_key(p)
            m1_mtime = p.stat().st_mtime
            if _needs_update(_M3_DIR, mk, m1_mtime) or _needs_update(_M5_DIR, mk, m1_mtime):
                todo.append(p)
        if not todo:
            print("所有月份都已是最新 ✅")
            return
        print(f"需要更新的月份: {[p.stem for p in todo]}")
    else:
        todo = m1_paths
        print(f"完整重建，共 {len(todo)} 個月")

    # 逐月載入、計算、存檔
    for m1_path in todo:
        mk = _month_key(m1_path)
        print(f"\n處理 {mk}...")
        t1 = time.time()

        df = pd.read_parquet(m1_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
        df["day_date"] = df["date"].dt.date
        print(f"  載入 m1: {len(df):,} 筆 ({time.time()-t1:.1f}s)")

        t2 = time.time()
        m3 = _compute_m3(df)
        _save_monthly(m3, _M3_DIR, "m3")
        print(f"  m3 耗時: {time.time()-t2:.1f}s")

        t3 = time.time()
        m5 = _compute_m5(df)
        _save_monthly(m5, _M5_DIR, "m5")
        print(f"  m5 耗時: {time.time()-t3:.1f}s")

    print(f"\n{'增量' if incremental else '完整'}重建完成，總耗時 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="預先聚合 m3/m5 K 線")
    parser.add_argument("--incremental", action="store_true", help="只更新比 m1 舊的月份")
    args = parser.parse_args()
    build(incremental=args.incremental)
