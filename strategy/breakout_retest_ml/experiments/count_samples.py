"""
樣本／進場漏斗（精簡版）。

完整「條件 1～6 分開計數」請用：
    python -m strategy.breakout_retest_ml.experiments.verify_funnel

本檔從「已過 POC 的候選」起算，硬觸發 = M1 陽線實體 K + Tick 大量買進。

用法：
    python -m strategy.breakout_retest_ml.experiments.count_samples
    python -m strategy.breakout_retest_ml.experiments.count_samples --save
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from data.query import load_tick_by_stock
from data.adjustment_query import load_pattern_day, load_pattern_m1, load_pattern_poc
from finmind.tick_universe import load_tick_universe
from strategy.breakout_retest_ml.config import (
    MAX_UPPER_SHADOW_RATIO,
    MIN_BODY_RATIO,
    MIN_PATTERN_SCORE,
    MIN_TICK_LARGE_BUY_RATIO,
    REQUIRE_CVD_POSITIVE,
    SESSION_END,
    SESSION_START,
    SL_PCT,
    TP_PCT,
    candidates_cache_path,
    prepared_cache_path,
)
from strategy.breakout_retest_ml.features import (
    FEATURES,
    _next_trade_day,
    _triple_barrier_label,
    find_day_candidates,
    find_intraday_trigger,
)


def _load_or_build_candidates(
    start_date: str,
    stock_ids: list[str],
    rebuild: bool,
) -> pd.DataFrame:
    path = candidates_cache_path(start_date)
    if path.exists() and not rebuild:
        cands = pd.read_parquet(path)
        print(f"[candidates] 讀取 cache {path.name} → {len(cands)} 筆", flush=True)
        return cands

    print(f"[candidates] 掃描日 K（min_score={MIN_PATTERN_SCORE}）...", flush=True)
    t0 = time.time()
    day = load_pattern_day(start_date=start_date)
    day = day[day["stock_id"].isin(stock_ids)].copy()
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    poc = load_pattern_poc(start_date=start_date)
    if not poc.empty:
        poc = poc[poc["stock_id"].isin(stock_ids)]
    cands = find_day_candidates(day, poc, stock_ids=stock_ids, day_step=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cands.to_parquet(path)
    print(f"[candidates] {len(cands)} 筆 → {path.name}（{time.time() - t0:.1f}s）", flush=True)
    return cands


def _build_funnel(
    candidates: pd.DataFrame,
    m1: pd.DataFrame,
    start_date: str,
    progress_every: int = 200,
) -> tuple[pd.DataFrame, dict]:
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    trading_days = np.array(sorted(m1["day_str"].unique()))

    stats = {
        "n_candidates": len(candidates),
        "n_no_trade_day": 0,
        "n_no_m1": 0,
        "n_no_trigger": 0,
        "n_triggered": 0,
        "n_no_label": 0,
        "n_labeled": 0,
    }
    events: list[dict] = []
    t0 = time.time()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for i, (_, cand) in enumerate(candidates.iterrows()):
            if (i + 1) % progress_every == 0 or i == 0:
                print(
                    f"  [{i + 1}/{len(candidates)}] triggered={stats['n_triggered']} "
                    f"labeled={stats['n_labeled']} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            sid = str(cand["stock_id"])
            cand_day = str(cand["candidate_date"])[:10]
            trade_day = _next_trade_day(cand_day, trading_days)
            if trade_day is None:
                stats["n_no_trade_day"] += 1
                continue
            if trade_day < start_date:
                continue

            m1_day = m1[(m1["stock_id"] == sid) & (m1["day_str"] == trade_day)].sort_values("date")
            if m1_day.empty or len(m1_day) < 15:
                stats["n_no_m1"] += 1
                continue

            try:
                ticks = load_tick_by_stock(sid, date=trade_day)
            except Exception:
                ticks = pd.DataFrame()

            trigger = find_intraday_trigger(
                m1_day.reset_index(drop=True),
                resistance=float(cand["resistance_price"]),
                matched_poc=float(cand["matched_poc"]) if pd.notna(cand["matched_poc"]) else np.nan,
                ticks=ticks,
            )
            if trigger is None:
                stats["n_no_trigger"] += 1
                continue

            stats["n_triggered"] += 1

            label_raw = _triple_barrier_label(m1_day, trigger["trigger_ts"], trigger["entry_price"])
            if not np.isfinite(label_raw):
                stats["n_no_label"] += 1
                continue

            stats["n_labeled"] += 1
            target = {-1.0: 0, 0.0: 1, 1.0: 2}[float(label_raw)]
            events.append(
                {
                    "stock_id": sid,
                    "date": trigger["trigger_ts"],
                    "candidate_date": cand_day,
                    "trade_date": trade_day,
                    "pattern_score": float(cand["pattern_score"]),
                    "poc_diff_pct": float(cand["poc_diff_pct"]) if pd.notna(cand["poc_diff_pct"]) else 0.0,
                    "resistance_price": float(cand["resistance_price"]),
                    "matched_poc": float(cand["matched_poc"]) if pd.notna(cand["matched_poc"]) else np.nan,
                    "entry_price": trigger["entry_price"],
                    "close": trigger["entry_price"],
                    "trigger_tf": trigger["trigger_tf"],
                    "body_ratio": trigger["body_ratio"],
                    "lower_shadow_ratio": trigger["lower_shadow_ratio"],
                    "upper_shadow_ratio": trigger["upper_shadow_ratio"],
                    "volume_surge_ratio": trigger["volume_surge_ratio"],
                    "dist_to_poc_pct": trigger["dist_to_poc_pct"],
                    "dist_to_support_pct": trigger["dist_to_support_pct"],
                    "tick_large_buy_ratio": trigger["tick_large_buy_ratio"],
                    "cvd_30s_delta": trigger["cvd_30s_delta"],
                    "target": target,
                    "label_raw": float(label_raw),
                }
            )

    return pd.DataFrame(events), stats


def _print_report(cands: pd.DataFrame, events: pd.DataFrame, stats: dict) -> None:
    print("\n" + "=" * 60)
    print("POC 共振 + M1實體K + Tick大量買進 — 樣本／進場漏斗")
    print("=" * 60)
    print(
        f"session          {SESSION_START[0]:02d}:{SESSION_START[1]:02d} ~ "
        f"{SESSION_END[0]:02d}:{SESSION_END[1]:02d}"
    )
    print(f"M1 body          >= {MIN_BODY_RATIO:.0%}  upper_shadow <= {MAX_UPPER_SHADOW_RATIO:.0%}")
    print(f"tick large buy   >= {MIN_TICK_LARGE_BUY_RATIO:.0%}  CVD>0={REQUIRE_CVD_POSITIVE}")
    print(f"barrier          TP={TP_PCT:.0%} / SL={SL_PCT:.0%}")
    print()
    print(f"1) POC 候選（streak 壓扁）     {stats['n_candidates']:>6,}")
    if len(cands):
        print(
            f"   stocks / date range         {cands['stock_id'].nunique()} 檔  "
            f"{cands['candidate_date'].min()} ~ {cands['candidate_date'].max()}"
        )
    print(f"   └ 無下一交易日              {stats['n_no_trade_day']:>6,}")
    print(f"   └ 當日無/過短 M1            {stats['n_no_m1']:>6,}")
    print(f"   └ 未通過 M1實體+Tick大單    {stats['n_no_trigger']:>6,}")
    print(f"2) 規則進場（有觸發）           {stats['n_triggered']:>6,}")
    print(f"   └ 標籤視窗不足（丟棄）       {stats['n_no_label']:>6,}")
    print(f"3) 訓練樣本（標籤完整）         {stats['n_labeled']:>6,}")

    if events.empty:
        print("\n（無標籤完整事件）")
        return

    print()
    print(f"訓練區間  {events['date'].min()} ~ {events['date'].max()}")
    print(f"涵蓋股票  {events['stock_id'].nunique()} 檔")
    dist = events["target"].value_counts().sort_index()
    pct = (events["target"].value_counts(normalize=True).sort_index() * 100).round(1)
    names = {0: "止損", 1: "震盪", 2: "止盈"}
    print("標籤分布:")
    for k in (0, 1, 2):
        print(f"  {k} {names[k]}: {int(dist.get(k, 0)):>5,}  ({float(pct.get(k, 0)):5.1f}%)")
    print(f"body_ratio mean: {events['body_ratio'].mean():.3f}")
    print(f"tick_large_buy_ratio mean: {events['tick_large_buy_ratio'].mean():.3f}")
    events = events.copy()
    events["ym"] = pd.to_datetime(events["trade_date"]).dt.to_period("M").astype(str)
    by_m = events.groupby("ym").size()
    print("月進場（有標籤）:")
    for ym, n in by_m.items():
        print(f"  {ym}: {n}")
    print(f"月均: {by_m.mean():.1f}")


def run(
    start_date: str = "2025-07-01",
    rebuild_candidates: bool = False,
    save: bool = False,
    progress_every: int = 200,
) -> pd.DataFrame:
    stock_ids = load_tick_universe()
    print(f"universe={len(stock_ids)}  start_date={start_date}", flush=True)

    cands = _load_or_build_candidates(start_date, stock_ids, rebuild=rebuild_candidates)
    if cands.empty:
        print("無 POC 候選")
        return cands

    t0 = time.time()
    m1 = load_pattern_m1(start_date=start_date)
    m1 = m1[m1["stock_id"].isin(stock_ids)].copy()
    print(f"[m1] {len(m1):,} rows / {m1['stock_id'].nunique()} stocks（{time.time() - t0:.1f}s）", flush=True)

    print("[events] M1實體K + Tick大量買進 + 標籤...", flush=True)
    events, stats = _build_funnel(cands, m1, start_date=start_date, progress_every=progress_every)
    _print_report(cands, events, stats)

    if save and not events.empty:
        missing = [c for c in FEATURES if c not in events.columns]
        if missing:
            print(f"缺少 FEATURES {missing}，略過存檔")
        else:
            path = prepared_cache_path(start_date)
            path.parent.mkdir(parents=True, exist_ok=True)
            events.to_parquet(path)
            print(f"\n已存 prepared cache → {path}")
            print("（舊 cache 特徵欄位已變，訓練請重跑本指令 --save 或 train 不帶舊 cache）")

    return events


def main():
    p = argparse.ArgumentParser(description="breakout_retest_ml 樣本／進場漏斗驗證")
    p.add_argument("--start_date", default="2025-07-01")
    p.add_argument("--rebuild_candidates", action="store_true", help="忽略候選 cache，重掃日 K")
    p.add_argument("--save", action="store_true", help="寫入 prepared parquet")
    p.add_argument("--progress_every", type=int, default=200)
    args = p.parse_args()
    run(
        start_date=args.start_date,
        rebuild_candidates=args.rebuild_candidates,
        save=args.save,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
