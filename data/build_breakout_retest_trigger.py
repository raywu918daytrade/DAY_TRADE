"""
物化盤中觸發快照 → db/breakout_retest_trigger/{YYYY_MM}.parquet

對 db/breakout_retest_day（poc_confluence=True）的下一交易日，
在 09:10～10:00 寫入每一根「M1 陽線實體 K」及其 Tick 特徵（不套大單門檻）。
M1 價格用 load_pattern_m1（與 Layer1 day/POC 同一還原基準：pattern 專用完整還原）。

用法：
    python -m data.build_breakout_retest_trigger
    python -m data.build_breakout_retest_trigger --force
    python -m data.build_breakout_retest_trigger --start_date 2025-07-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_OUT_DIR = _ROOT / "db/breakout_retest_trigger"
_DEFAULT_START = "2025-07-01"

_COLS = [
    "stock_id",
    "candidate_date",
    "trade_date",
    "trigger_ts",
    "entry_price",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "volume_surge_ratio",
    "tick_large_buy_ratio",
    "cvd_30s_delta",
    "dist_to_poc_pct",
    "dist_to_support_pct",
    "pattern_score",
    "poc_diff_pct",
    "resistance_price",
    "matched_poc",
]


def _atomic_to_parquet(df: pd.DataFrame, file_path: Path, **kwargs) -> None:
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, file_path)


def _merge_month(ym: str, new_rows: pd.DataFrame, force: bool) -> None:
    path = _OUT_DIR / f"{ym}.parquet"
    if path.exists() and not force:
        old = pd.read_parquet(path)
        final = pd.concat([old, new_rows], ignore_index=True)
        final.drop_duplicates(subset=["stock_id", "candidate_date", "trigger_ts"], keep="last", inplace=True)
    else:
        final = new_rows
    final = final.sort_values(["stock_id", "trigger_ts"]).reset_index(drop=True)
    _atomic_to_parquet(final[_COLS], path, index=False, compression="zstd")
    print(f"  寫入 {path.name}: {len(final):,} 列", flush=True)


def build(
    force: bool = False,
    start_date: str = _DEFAULT_START,
    progress_every: int = 100,
) -> pd.DataFrame:
    from data.query import (
        load_breakout_retest_day,
        load_breakout_retest_trigger,
        load_tick_by_stock,
    )
    from data.adjustment_query import load_pattern_m1
    from strategy.breakout_retest_ml.features import _next_trade_day, collect_m1_body_triggers

    t0 = time.time()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    cands = load_breakout_retest_day(start_date=start_date, only_poc=True)
    if cands.empty:
        print("無 Layer1 候選，請先 python -m data.build_breakout_retest_day", flush=True)
        return cands

    if not force:
        existing = load_breakout_retest_trigger(start_date=start_date)
        if not existing.empty:
            done = set(zip(existing["stock_id"].astype(str), existing["candidate_date"].astype(str)))
            before = len(cands)
            mask = [
                (str(r.stock_id), str(r.candidate_date)[:10]) not in done
                for r in cands.itertuples(index=False)
            ]
            cands = cands.loc[mask].reset_index(drop=True)
            print(f"增量：略過已存在 {before - len(cands)}，待處理 {len(cands)}", flush=True)
            if cands.empty:
                print("已全部建完 ✅", flush=True)
                return existing

    # 與 Layer1 day/POC（還原權息）同一基準，避免 entry_price / dist_to_* 混用
    print(f"載入 M1 adjusted（start_date={start_date})...", flush=True)
    m1 = load_pattern_m1(start_date=start_date)
    stock_ids = set(cands["stock_id"].astype(str))
    m1 = m1[m1["stock_id"].isin(stock_ids)].copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    trading_days = np.array(sorted(m1["day_str"].unique()))
    print(f"M1 {len(m1):,} rows / trading_days={len(trading_days)}", flush=True)

    buf: list[dict] = []
    t1 = time.time()
    for i, cand in enumerate(cands.itertuples(index=False)):
        if (i + 1) % progress_every == 0 or i == 0:
            print(
                f"  [{i + 1}/{len(cands)}] rows_buf={len(buf)} elapsed={time.time() - t1:.0f}s",
                flush=True,
            )
        sid = str(cand.stock_id)
        cand_day = str(cand.candidate_date)[:10]
        trade_day = _next_trade_day(cand_day, trading_days)
        if trade_day is None or trade_day < start_date:
            continue

        m1_day = m1[(m1["stock_id"] == sid) & (m1["day_str"] == trade_day)].sort_values("date")
        if m1_day.empty or len(m1_day) < 15:
            continue

        try:
            ticks = load_tick_by_stock(sid, date=trade_day)
        except Exception:
            ticks = pd.DataFrame()

        hits = collect_m1_body_triggers(
            m1_day.reset_index(drop=True),
            resistance=float(cand.resistance_price),
            matched_poc=float(cand.matched_poc) if pd.notna(cand.matched_poc) else np.nan,
            ticks=ticks,
        )
        for h in hits:
            buf.append(
                {
                    "stock_id": sid,
                    "candidate_date": cand_day,
                    "trade_date": trade_day,
                    "trigger_ts": h["trigger_ts"],
                    "entry_price": h["entry_price"],
                    "body_ratio": h["body_ratio"],
                    "upper_shadow_ratio": h["upper_shadow_ratio"],
                    "lower_shadow_ratio": h["lower_shadow_ratio"],
                    "volume_surge_ratio": h["volume_surge_ratio"],
                    "tick_large_buy_ratio": h["tick_large_buy_ratio"],
                    "cvd_30s_delta": h["cvd_30s_delta"],
                    "dist_to_poc_pct": h["dist_to_poc_pct"],
                    "dist_to_support_pct": h["dist_to_support_pct"],
                    "pattern_score": float(cand.pattern_score),
                    "poc_diff_pct": float(cand.poc_diff_pct) if pd.notna(cand.poc_diff_pct) else 0.0,
                    "resistance_price": float(cand.resistance_price),
                    "matched_poc": float(cand.matched_poc) if pd.notna(cand.matched_poc) else np.nan,
                }
            )

    if not buf:
        print("本輪無新 trigger 列", flush=True)
        return load_breakout_retest_trigger(start_date=start_date)

    new_df = pd.DataFrame(buf)
    new_df["ym"] = pd.to_datetime(new_df["trade_date"]).dt.strftime("%Y_%m")
    for ym, g in new_df.groupby("ym"):
        _merge_month(ym, g.drop(columns=["ym"]), force=False)

    out = load_breakout_retest_trigger(start_date=start_date)
    print(f"完成 breakout_retest_trigger：{len(out):,} 列 耗時 {time.time() - t0:.1f}s ✅", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="物化盤中 M1 實體K + Tick 觸發快照")
    p.add_argument("--force", action="store_true", help="忽略已存在列，全部重算（仍按月 merge keep=last）")
    p.add_argument("--start_date", default=_DEFAULT_START)
    p.add_argument("--progress_every", type=int, default=100)
    args = p.parse_args()
    # force: 清空 done 集合＝不略過；merge 仍 keep=last
    if args.force:
        # 刪除既有輸出再重建，避免舊定義殘留
        if _OUT_DIR.exists():
            for f in _OUT_DIR.glob("*.parquet"):
                f.unlink()
            print(f"已清空 {_OUT_DIR}", flush=True)
    build(force=False, start_date=args.start_date, progress_every=args.progress_every)
