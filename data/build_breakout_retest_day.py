"""
物化日 K breakout_retest 候選 → db/breakout_retest_day/{YYYY_MM}.parquet

欄位：
    stock_id, candidate_date, pattern_score, resistance_price,
    matched_poc, poc_diff_pct, poc_confluence

含／不含 POC 共振都存；策略端預設再 filter poc_confluence==True。
偵測需要日 K 回看視窗，因此以「全宇宙重掃後按 candidate_date 月份寫檔」為主
（400 檔約 1 分鐘）。incremental：來源日 K／POC 沒比較新就跳過。

用法：
    python -m data.build_breakout_retest_day
    python -m data.build_breakout_retest_day --force
    python -m data.build_breakout_retest_day --start_date 2025-07-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_OUT_DIR = _ROOT / "db/breakout_retest_day"
_DAY_DIR = _ROOT / "db/adjustment_day"  # pattern專用完整還原日K，2026-08-03 從 db/fugle_day 改名
_POC_DIR = _ROOT / "db/poc_day"
_LOOKBACK_DAYS = 400
_DEFAULT_START = "2025-07-01"


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path, **kwargs) -> None:
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def _source_mtime() -> float:
    latest = []
    for d in (_DAY_DIR, _POC_DIR):
        if not d.exists():
            continue
        files = sorted(f for f in d.iterdir() if f.suffix == ".parquet")
        if files:
            latest.append(files[-1].stat().st_mtime)
    return max(latest) if latest else 0.0


def _out_mtime() -> float:
    if not _OUT_DIR.exists():
        return 0.0
    files = list(_OUT_DIR.glob("*.parquet"))
    if not files:
        return 0.0
    return max(f.stat().st_mtime for f in files)


def build(force: bool = False, start_date: str = _DEFAULT_START) -> pd.DataFrame:
    from data.adjustment_query import load_pattern_day, load_pattern_poc
    from finmind.tick_universe import load_tick_universe
    from strategy.breakout_retest_ml.features import find_day_candidates

    t0 = time.time()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _out_mtime() >= _source_mtime() and _out_mtime() > 0:
        print("db/breakout_retest_day 已比日K/POC 新，跳過重建 ✅", flush=True)
        from data.query import load_breakout_retest_day

        return load_breakout_retest_day(start_date=start_date)

    stock_ids = load_tick_universe()
    hist_start = (pd.Timestamp(start_date) - pd.Timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    print(
        f"掃描 breakout_retest_day：universe={len(stock_ids)} "
        f"hist_start={hist_start} keep_from={start_date}",
        flush=True,
    )

    day = load_pattern_day(start_date=hist_start)
    day = day[day["stock_id"].isin(stock_ids)].copy()
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    poc = load_pattern_poc(start_date=hist_start)
    if not poc.empty:
        poc = poc[poc["stock_id"].isin(stock_ids)]

    cands = find_day_candidates(day, poc, stock_ids=stock_ids, day_step=1, require_poc=False)
    if cands.empty:
        print("無候選", flush=True)
        return cands

    cands = cands[cands["candidate_date"] >= start_date].reset_index(drop=True)
    cands["ym"] = pd.to_datetime(cands["candidate_date"]).dt.strftime("%Y_%m")

    cols = [
        "stock_id",
        "candidate_date",
        "pattern_score",
        "resistance_price",
        "matched_poc",
        "poc_diff_pct",
        "poc_confluence",
    ]
    n_written = 0
    for ym, g in cands.groupby("ym"):
        out = g[cols].sort_values(["stock_id", "candidate_date"]).reset_index(drop=True)
        path = _OUT_DIR / f"{ym}.parquet"
        _atomic_to_parquet(out, path, index=False, compression="zstd")
        n_written += len(out)
        print(f"  寫入 {path.name}: {len(out):,} 列", flush=True)

    print(
        f"完成 breakout_retest_day：{n_written:,} 列 "
        f"(poc={int(cands['poc_confluence'].sum()):,}) 耗時 {time.time() - t0:.1f}s ✅",
        flush=True,
    )
    return cands[cols]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="物化日 K breakout_retest 候選")
    p.add_argument("--force", action="store_true")
    p.add_argument("--start_date", default=_DEFAULT_START)
    args = p.parse_args()
    build(force=args.force, start_date=args.start_date)
