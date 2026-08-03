"""
M1 與淨大單「反向」後，價格是否往回（相對 M1 方向回歸）？

小樣本：同向／反向／net=0 三組，看 5／15／30 分後報酬。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_m1_net_reverse_revert \\
        --start_date 2026-07-01 --end_date 2026-07-07 --max_rows 4000
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

from data.adjustment_query import load_pattern_m1
from data.query import load_tick_by_stock
from finmind.tick_universe import load_tick_universe
from strategy.breakout_retest_ml.config import SESSION_END, TICK_LARGE_LOT
from strategy.breakout_retest_ml.experiments.verify_m5_tick import (
    SESSION_START,
    _tick_in_window,
)
from strategy.breakout_retest_ml.experiments.verify_m1_net_concordance import (
    PER_DAY_MAX,
    _collect_candidates,
)

HORIZONS = (5, 15, 30)


def _fwd_ret(m1_day: pd.DataFrame, win_start: pd.Timestamp, entry: float, minutes: int) -> float:
    """棒收後 minutes 分鐘那根 M1 close 相對 entry 報酬；缺棒 → nan。"""
    if entry <= 0 or m1_day is None or m1_day.empty:
        return float("nan")
    t = pd.Timestamp(win_start) + pd.Timedelta(minutes=minutes)
    hit = m1_day.loc[m1_day["date"] == t, "close"]
    if hit.empty:
        return float("nan")
    c = float(hit.iloc[0])
    if not np.isfinite(c):
        return float("nan")
    return (c - entry) / entry


def _print_group(name: str, sub: pd.DataFrame) -> None:
    n = len(sub)
    print(f"\n[{name}] n={n}", flush=True)
    if n == 0:
        return
    hdr = f"{'H':>4} {'n有效':>6} {'回歸%':>8} {'延續%':>8} {'mean(signed)':>14} {'mean(ret)':>10}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for h in HORIZONS:
        col = f"ret_{h}"
        ok = sub[np.isfinite(sub[col])]
        if ok.empty:
            print(f"{h:>4} {0:>6}", flush=True)
            continue
        # signed = ret * m1_dir：>0 延續 M1，<0 回歸
        signed = ok[col] * ok["m1_dir"]
        n_ok = len(ok)
        revert = 100.0 * (signed < 0).sum() / n_ok
        cont = 100.0 * (signed > 0).sum() / n_ok
        print(
            f"{h:>4} {n_ok:>6} {revert:>7.1f}% {cont:>7.1f}% "
            f"{100*signed.mean():>13.3f}% {100*ok[col].mean():>9.3f}%",
            flush=True,
        )


def run(start_date: str, end_date: str, max_rows: int = 4000) -> pd.DataFrame:
    t0 = time.time()
    stock_ids = [str(s) for s in load_tick_universe()]
    print("反向後會回歸嗎？（相對 M1 方向）", flush=True)
    print(
        f"母體 tick400；窗 {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}；"
        f"淨大單 >{TICK_LARGE_LOT}張；每 stock×day≤{PER_DAY_MAX}；max_rows={max_rows}",
        flush=True,
    )
    print(f"回歸 = 後續報酬與 M1 方向相反（signed=ret×m1_dir < 0）", flush=True)
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    print(f"載入 pattern M1（start={start_date})...", flush=True)
    m1 = load_pattern_m1(start_date=start_date)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1 = m1[m1["stock_id"].isin(stock_ids)].copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[(m1["day_str"] >= start_date) & (m1["day_str"] <= end_date)]
    print(f"M1 {len(m1):,}", flush=True)

    # 全日 M1 索引，方便算 forward（含 10:00 後）
    m1_by_day = {
        (str(sid), str(day)): g.sort_values("date").reset_index(drop=True)
        for (sid, day), g in m1.groupby(["stock_id", "day_str"], sort=False)
    }

    ev = _collect_candidates(m1, max_rows=max_rows)
    print(
        f"抽樣列={len(ev):,} up={(ev['m1_dir']==1).sum()} down={(ev['m1_dir']==-1).sum()}",
        flush=True,
    )
    if ev.empty:
        return ev

    print("讀取 tick + 算 forward...", flush=True)
    nets: list[float] = []
    fwds: dict[int, list[float]] = {h: [] for h in HORIZONS}
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for j, r in enumerate(ev.itertuples(index=False), start=1):
        key = (str(r.stock_id), str(r.trade_date)[:10])
        if key not in cache:
            try:
                cache[key] = load_tick_by_stock(key[0], date=key[1])
            except Exception:
                cache[key] = pd.DataFrame()
        feat = _tick_in_window(cache[key], r.win_start, r.win_end)
        nets.append(float(feat["bar_large_net_ratio"]))
        day_df = m1_by_day.get(key)
        for h in HORIZONS:
            fwds[h].append(_fwd_ret(day_df, r.win_start, float(r.close), h))
        if j % 500 == 0 or j == len(ev):
            print(f"  {j}/{len(ev)} elapsed={time.time()-t0:.0f}s", flush=True)

    ev = ev.copy()
    ev["net"] = nets
    for h in HORIZONS:
        ev[f"ret_{h}"] = fwds[h]

    same = ((ev["m1_dir"] == 1) & (ev["net"] > 0)) | ((ev["m1_dir"] == -1) & (ev["net"] < 0))
    opp = ((ev["m1_dir"] == 1) & (ev["net"] < 0)) | ((ev["m1_dir"] == -1) & (ev["net"] > 0))
    zero = ev["net"] == 0

    print("\n" + "=" * 64)
    print("分組：同向 / 反向 / net=0")
    print("=" * 64)
    print(
        f"同向 n={int(same.sum())}  反向 n={int(opp.sum())}  net=0 n={int(zero.sum())}",
        flush=True,
    )
    _print_group("反向（M1⊥淨大單）", ev[opp])
    _print_group("同向", ev[same])
    _print_group("net=0", ev[zero])

    print(
        f"\n解讀: 若反向組「回歸%」明顯高於同向／net=0，且 mean(signed)<0，"
        f"才像「反向後回歸」。耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return ev


def main():
    p = argparse.ArgumentParser(description="M1×淨大單反向後是否回歸")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-07-07")
    p.add_argument("--max_rows", type=int, default=4000)
    args = p.parse_args()
    run(args.start_date, args.end_date, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
