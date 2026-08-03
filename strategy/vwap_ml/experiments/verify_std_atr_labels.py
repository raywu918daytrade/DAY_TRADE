"""
掃 ATR5 門檻下，VWAP |z|>=std_mult 後是回歸還是延續。

只 load_m1（m3/m5 現算），避開 load_m3 還原 bug。

用法：
    python -m strategy.vwap_ml.experiments.verify_std_atr_labels \\
        --start_date 2026-05-01 --end_date 2026-07-31 --std_mult 1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from data.query import load_m1
from finmind.tick_universe import load_tick_universe
from strategy.vwap_ml.features import make_features

LABEL_NAME = {0: "回歸", 1: "持平", 2: "延續"}
ATR_LEVELS = (0.010, 0.012, 0.015, 0.020)


def _print_block(sub: pd.DataFrame, title: str) -> None:
    n = len(sub)
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)
    print(f"{'結果':>8} {'進場總數':>8} {'占比':>8}", flush=True)
    print("-" * 32, flush=True)
    if n == 0:
        print("(無樣本)", flush=True)
        return
    for k in (0, 2, 1):
        c = int((sub["target"] == k).sum())
        print(f"{LABEL_NAME[k]:>8} {c:>8} {100.0 * c / n:>7.1f}%", flush=True)
    print(f"{'合計':>8} {n:>8}", flush=True)
    dec = sub[sub["target"] != 1]
    nd = len(dec)
    if nd:
        r = int((dec["target"] == 0).sum())
        c = int((dec["target"] == 2).sum())
        print(f"  決勝負 n={nd}: 回歸 {100.0 * r / nd:.1f}% / 延續 {100.0 * c / nd:.1f}%", flush=True)


def _summary_line(sub: pd.DataFrame) -> str:
    n = len(sub)
    if n == 0:
        return "n=0"
    r = int((sub["target"] == 0).sum())
    c = int((sub["target"] == 2).sum())
    f = int((sub["target"] == 1).sum())
    dec = r + c
    rev_dec = 100.0 * r / dec if dec else 0.0
    return (
        f"n={n} 回歸={r} 延續={c} 持平={f} | "
        f"決勝負回歸%={rev_dec:.1f}"
    )


def run(
    start_date: str,
    end_date: str,
    std_mult: float = 1.5,
    atr_levels: tuple[float, ...] = ATR_LEVELS,
) -> pd.DataFrame:
    t0 = time.time()
    load_start = (pd.Timestamp(start_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    univ = set(str(s) for s in load_tick_universe())
    atr_floor = min(atr_levels)

    print(
        f"VWAP 標籤掃 ATR | tick400 | std_mult={std_mult} | "
        f"atr 門檻={list(atr_levels)} | {start_date}~{end_date}",
        flush=True,
    )
    print(f"載入 m1（start={load_start}；m3/m5 現算）...", flush=True)
    m1 = load_m1(start_date=load_start)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(univ)].copy()
    m1["stock_id"] = m1["stock_id"].astype("category")
    print(f"m1={len(m1):,}", flush=True)

    print(f"make_features（先收 atr5>={atr_floor:.5f}）...", flush=True)
    ev = make_features(m1, std_mult=std_mult, atr5_threshold=atr_floor, m3=None, m5=None)
    ev = ev.dropna(subset=["target"]).copy()
    ev["target"] = ev["target"].astype(int)
    ev["trade_date"] = pd.to_datetime(ev["date"]).dt.strftime("%Y-%m-%d")
    ev = ev[(ev["trade_date"] >= start_date) & (ev["trade_date"] <= end_date)].copy()
    print(f"基礎候選（atr>={atr_floor}）: {len(ev):,}\n", flush=True)

    print("=" * 72)
    print("ATR 掃總表（全體觸發）")
    print("=" * 72)
    print(
        f"{'atr5≥':>8} {'進場總數':>8} {'回歸':>7} {'延續':>7} {'持平':>7} "
        f"{'決回歸%':>8} {'上軌決回歸%':>10} {'下軌決回歸%':>10} {'首觸決回歸%':>10}",
        flush=True,
    )
    print("-" * 72, flush=True)

    for atr in atr_levels:
        sub = ev[ev["atr5"] >= atr]
        n = len(sub)
        r = int((sub["target"] == 0).sum()) if n else 0
        c = int((sub["target"] == 2).sum()) if n else 0
        f = int((sub["target"] == 1).sum()) if n else 0
        dec = r + c
        rev = 100.0 * r / dec if dec else 0.0

        up = sub[sub["trigger_side"] == "upper"]
        lo = sub[sub["trigger_side"] == "lower"]
        up_dec = up[up["target"] != 1]
        lo_dec = lo[lo["target"] != 1]
        up_rev = (
            100.0 * int((up_dec["target"] == 0).sum()) / len(up_dec) if len(up_dec) else 0.0
        )
        lo_rev = (
            100.0 * int((lo_dec["target"] == 0).sum()) / len(lo_dec) if len(lo_dec) else 0.0
        )

        first = (
            sub.sort_values(["stock_id", "trade_date", "date"])
            .drop_duplicates(["stock_id", "trade_date"], keep="first")
        )
        first_dec = first[first["target"] != 1]
        first_rev = (
            100.0 * int((first_dec["target"] == 0).sum()) / len(first_dec)
            if len(first_dec)
            else 0.0
        )

        print(
            f"{atr:>8.3f} {n:>8} {r:>7} {c:>7} {f:>7} "
            f"{rev:>7.1f}% {up_rev:>9.1f}% {lo_rev:>9.1f}% {first_rev:>9.1f}%",
            flush=True,
        )

    for atr in atr_levels:
        sub = ev[ev["atr5"] >= atr]
        _print_block(sub, f"全體 |z|>={std_mult} ATR>={atr:.5f}")
        _print_block(sub[sub["trigger_side"] == "upper"], f"上軌 ATR>={atr:.5f}")
        _print_block(sub[sub["trigger_side"] == "lower"], f"下軌 ATR>={atr:.5f}")
        first = (
            sub.sort_values(["stock_id", "trade_date", "date"])
            .drop_duplicates(["stock_id", "trade_date"], keep="first")
        )
        print(f"  每日首觸發: {_summary_line(first)}", flush=True)

    print(
        f"\n說明: 標籤=vwap_ml（30分內 z→0=回歸、z→{std_mult*2}=延續）；"
        f"決回歸%=回歸/(回歸+延續)。耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return ev


def main():
    p = argparse.ArgumentParser(description="VWAP std_mult × ATR 回歸/延續掃表")
    p.add_argument("--start_date", default="2026-05-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument("--std_mult", type=float, default=1.5)
    args = p.parse_args()
    run(args.start_date, args.end_date, std_mult=args.std_mult)


if __name__ == "__main__":
    main()
