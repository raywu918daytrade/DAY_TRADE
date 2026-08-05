"""
VWAP |z|>=std_mult + 高 ATR 母體上，疊 tick 大單，看回歸 vs 延續是否拉開。

只驗證硬規則，不寫入 FEATURES。只 load_m1（m3/m5 現算）。

用法：
    python -m strategy.vwap_ml.experiments.verify_vwap_tick_direction \\
        --start_date 2026-05-01 --end_date 2026-07-31 --std_mult 1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from data.query import load_m1, load_tick_by_stock
from finmind.tick_universe import load_tick_universe
from strategy.breakout_retest_ml.config import TICK_LARGE_BUY_SECONDS, TICK_LARGE_LOT
from strategy.vwap_ml.features import make_features

LABEL_NAME = {0: "回歸", 1: "持平", 2: "延續"}
ATR_LEVELS = (0.020, 0.025, 0.030)


def _tick_before(ticks: pd.DataFrame, trigger_ts: pd.Timestamp, seconds: int = TICK_LARGE_BUY_SECONDS) -> dict:
    empty = {
        "large_buy_ratio": 0.0,
        "large_sell_ratio": 0.0,
        "large_net_ratio": 0.0,
    }
    if ticks is None or ticks.empty:
        return empty
    t = ticks.copy()
    t["date"] = pd.to_datetime(t["date"], format="mixed")
    w = t[(t["date"] > trigger_ts - pd.Timedelta(seconds=seconds)) & (t["date"] <= trigger_ts)]
    if w.empty:
        return empty
    vol = w["volume"].astype(float)
    total = float(vol.sum())
    if total <= 0:
        return empty
    lb = float(vol[(w["tick_type"] == 1) & (vol > TICK_LARGE_LOT)].sum()) / total
    ls = float(vol[(w["tick_type"] != 1) & (vol > TICK_LARGE_LOT)].sum()) / total
    return {
        "large_buy_ratio": round(lb, 4),
        "large_sell_ratio": round(ls, 4),
        "large_net_ratio": round(lb - ls, 4),
    }


def _stats(sub: pd.DataFrame) -> dict:
    if sub is None or sub.empty:
        return {"n": 0, "n_rev": 0, "n_cont": 0, "n_flat": 0, "rev_dec": 0.0}
    n = len(sub)
    n_rev = int((sub["target"] == 0).sum())
    n_cont = int((sub["target"] == 2).sum())
    n_flat = int((sub["target"] == 1).sum())
    dec = n_rev + n_cont
    return {
        "n": n,
        "n_rev": n_rev,
        "n_cont": n_cont,
        "n_flat": n_flat,
        "rev_dec": 100.0 * n_rev / dec if dec else 0.0,
    }


def _print_stats(label: str, st: dict) -> None:
    print(
        f"{label:>16} {st['n']:>8} {st['n_rev']:>7} {st['n_cont']:>7} {st['n_flat']:>7} " f"{st['rev_dec']:>7.1f}%",
        flush=True,
    )


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
    st = _stats(sub)
    print(f"{'合計':>8} {n:>8}", flush=True)
    if st["n_rev"] + st["n_cont"]:
        print(
            f"  決勝負: 回歸 {st['rev_dec']:.1f}% / " f"延續 {100.0 - st['rev_dec']:.1f}%",
            flush=True,
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
        f"VWAP+大單驗證 | tick400 | std_mult={std_mult} | " f"ATR={list(atr_levels)} | {start_date}~{end_date}",
        flush=True,
    )
    print(f"載入 m1（start={load_start}）...", flush=True)
    m1 = load_m1(start_date=load_start)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(univ)].copy()
    m1["stock_id"] = m1["stock_id"].astype("category")
    print(f"m1={len(m1):,}", flush=True)

    print(f"make_features atr5>={atr_floor:.5f}...", flush=True)
    ev = make_features(m1, std_mult=std_mult, atr5_threshold=atr_floor, m3=None, m5=None)
    ev = ev.dropna(subset=["target"]).copy()
    ev["target"] = ev["target"].astype(int)
    ev["trade_date"] = pd.to_datetime(ev["date"]).dt.strftime("%Y-%m-%d")
    ev = ev[(ev["trade_date"] >= start_date) & (ev["trade_date"] <= end_date)].copy()
    print(f"候選 atr>={atr_floor}: {len(ev):,}", flush=True)

    print(f"讀取 tick（觸發前 {TICK_LARGE_BUY_SECONDS}s，單筆>{TICK_LARGE_LOT}張）...", flush=True)
    feats: list[dict] = []
    for i, r in enumerate(ev.itertuples(index=False), start=1):
        try:
            ticks = load_tick_by_stock(str(r.stock_id), date=str(r.trade_date)[:10])
        except Exception:
            ticks = pd.DataFrame()
        feats.append(_tick_before(ticks, pd.Timestamp(r.date)))
        if i % 100 == 0 or i == len(ev):
            print(f"  [tick] {i}/{len(ev)} elapsed={time.time()-t0:.0f}s", flush=True)
    ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(feats)], axis=1)

    # 順回歸大單：上軌要賣壓（sell>buy），下軌要買壓（buy>sell）
    ev["tick_with_reversion"] = np.where(
        ev["trigger_side"] == "upper",
        ev["large_sell_ratio"] > ev["large_buy_ratio"],
        ev["large_buy_ratio"] > ev["large_sell_ratio"],
    )
    ev["reversion_net"] = np.where(
        ev["trigger_side"] == "upper",
        ev["large_sell_ratio"] - ev["large_buy_ratio"],
        ev["large_buy_ratio"] - ev["large_sell_ratio"],
    )

    for atr in atr_levels:
        base = ev[ev["atr5"] >= atr].copy()
        print("\n" + "=" * 72)
        print(f"ATR5>={atr:.5f}  n={len(base)}  （單筆>{TICK_LARGE_LOT}張）")
        print("=" * 72)
        hdr = f"{'條件':>16} {'進場總數':>8} {'回歸':>7} {'延續':>7} {'持平':>7} {'決回歸%':>8}"
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)

        _print_stats("僅VWAP+ATR", _stats(base))
        buy_gt = base[base["large_buy_ratio"] > base["large_sell_ratio"]]
        _print_stats("大買>大賣", _stats(buy_gt))
        with_rev = base[base["tick_with_reversion"]]
        _print_stats("順回歸大單", _stats(with_rev))
        for thr in (0.10, 0.20):
            net = base[base["reversion_net"] >= thr]
            _print_stats(f"順回歸淨≥{thr:.0%}", _stats(net))

        _print_block(base, f"基線 ATR>={atr:.5f}")
        _print_block(with_rev, f"順回歸大單 ATR>={atr:.5f}")
        up = base[base["trigger_side"] == "upper"]
        lo = base[base["trigger_side"] == "lower"]
        _print_block(up[up["tick_with_reversion"]], f"上軌+賣>買 ATR>={atr:.5f}")
        _print_block(lo[lo["tick_with_reversion"]], f"下軌+買>賣 ATR>={atr:.5f}")

    print(
        f"\n說明: 順回歸大單＝上軌大賣>大買、下軌大買>大賣；"
        f"決回歸%=回歸/(回歸+延續)。標籤 vwap_ml 30分 z-barrier。"
        f"耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return ev


def main():
    p = argparse.ArgumentParser(description="VWAP+高ATR+大單 回歸/延續驗證")
    p.add_argument("--start_date", default="2026-05-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument("--std_mult", type=float, default=1.5)
    args = p.parse_args()
    run(args.start_date, args.end_date, std_mult=args.std_mult)


if __name__ == "__main__":
    main()
