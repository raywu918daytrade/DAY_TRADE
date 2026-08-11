"""
0050 Open-Drive 敘述統計（Phase 0+1）

- 0050 / 個股 gap（固定 % 分桶；σ20 僅輔標）
- 09:05 ret5 同向
- 09:05 進場 → 09:15 / 09:30 / 09:45 / 10:00 做多與做空機會

用法：
    python -m strategy_test.open_drive_0050_stats.verify \\
        --start_date 2026-06-01 --end_date 2026-07-31
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import time as dtime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from data.adjustment_query import load_pattern_day, load_pattern_m1, load_pattern_m5_std
from finmind.tick_universe import load_tick_universe
from strategy.mkt.config import ATR5_FILTER_THRESHOLD, IDX_SYMBOL
from strategy.mkt.features import add_atr5

ENTRY_T = dtime(9, 5)
EXIT_TIMES = {
    "0915": dtime(9, 15),
    "0930": dtime(9, 30),
    "0945": dtime(9, 45),
    "1000": dtime(10, 0),
}
FRICTION = 0.0045  # 0.45%
WIN3 = 0.03  # 勝率門檻：|r| ≥ 3%
Z_THRESH = 1.5  # σ 分組：|z| > 1.5 → up/down
SIGNS = ("up", "down", "flat")
# 絕對門檻（沿用 mkt／breakout；0=不過濾）
ATR5_GATES = (0.0, 0.006, 0.008, ATR5_FILTER_THRESHOLD)


def _gap_mag_bucket(abs_gap: float) -> str:
    if abs_gap <= 0.01:
        return "le1"
    if abs_gap <= 0.02:
        return "1to2"
    if abs_gap <= 0.05:
        return "2to5"
    return "gt5"


def _gap_sign(gap: float) -> str:
    if gap > 0.01:
        return "up"
    if gap < -0.01:
        return "down"
    return "flat"


def _ret5_mag_bucket(abs_r: float) -> str:
    if abs_r <= 0.005:
        return "le0.5"
    if abs_r <= 0.01:
        return "0.5to1"
    return "gt1"


def _ret5_sign(r: float) -> str:
    if r > 0.005:
        return "up"
    if r < -0.005:
        return "down"
    return "flat"


def _z_sign(z: float) -> str:
    """標準差分組：|z| > Z_THRESH → up/down。"""
    if pd.isna(z):
        return "flat"
    if z > Z_THRESH:
        return "up"
    if z < -Z_THRESH:
        return "down"
    return "flat"


def _side_stats(r: pd.Series) -> dict:
    """r = long return; short = -r。
    win0 = 方向為正（long: r>0；short: r<0）
    win3 = 方向且 |r|≥3%
    """
    r = r.dropna()
    n = len(r)
    empty = {
        "n": 0,
        "long_mean": np.nan,
        "long_win0": np.nan,
        "long_win3": np.nan,
        "short_mean": np.nan,
        "short_win0": np.nan,
        "short_win3": np.nan,
        "better": "none",
    }
    if n == 0:
        return empty
    long_mean = float(r.mean())
    short_mean = float((-r).mean())
    long_win0 = float((r > 0).mean())
    short_win0 = float((r < 0).mean())
    long_win3 = float((r >= WIN3).mean())
    short_win3 = float((r <= -WIN3).mean())
    better = "none"
    if long_mean >= FRICTION and long_mean > short_mean:
        better = "long"
    elif short_mean >= FRICTION and short_mean > long_mean:
        better = "short"
    elif abs(long_mean) >= abs(short_mean) and abs(long_mean) >= 0.002:
        better = "long?" if long_mean > 0 else "short?"
    elif abs(short_mean) >= 0.002:
        better = "short?" if short_mean > 0 else "long?"
    return {
        "n": n,
        "long_mean": long_mean,
        "long_win0": long_win0,
        "long_win3": long_win3,
        "short_mean": short_mean,
        "short_win0": short_win0,
        "short_win3": short_win3,
        "better": better,
    }


def _print_side(label: str, st: dict) -> None:
    if st["n"] == 0:
        print(f"  {label}: n=0", flush=True)
        return
    print(
        f"  {label}: n={st['n']:,}  "
        f"long mean={100 * st['long_mean']:+.3f}% "
        f"win0={100 * st['long_win0']:.1f}% win3={100 * st['long_win3']:.1f}%  "
        f"short mean={100 * st['short_mean']:+.3f}% "
        f"win0={100 * st['short_win0']:.1f}% win3={100 * st['short_win3']:.1f}%  "
        f"→ {st['better']}",
        flush=True,
    )


def _scan_0050_only(
    ev: pd.DataFrame, gap_col: str, ret5_col: str, min_n: int, label: str
) -> None:
    print(f"\n[機會掃瞄 0050-only {label}] {gap_col}×{ret5_col}×exit×side", flush=True)
    hits = []
    for gsgn in SIGNS:
        for rsgn in SIGNS:
            sub = ev[(ev[gap_col] == gsgn) & (ev[ret5_col] == rsgn)]
            if len(sub) < min_n:
                continue
            for name in EXIT_TIMES:
                st = _side_stats(sub[f"r_{name}"])
                if st["better"] in ("long", "short"):
                    hits.append((gsgn, rsgn, name, st))
    if not hits:
        print("  （無）", flush=True)
        return
    for gsgn, rsgn, name, st in sorted(
        hits, key=lambda x: -max(x[3]["long_mean"], x[3]["short_mean"])
    ):
        side = st["better"]
        m = st["long_mean"] if side == "long" else st["short_mean"]
        w0 = st["long_win0"] if side == "long" else st["short_win0"]
        w3 = st["long_win3"] if side == "long" else st["short_win3"]
        print(
            f"  gap={gsgn} ret5={rsgn} @{name} → {side}  "
            f"n={st['n']:,} mean={100 * m:+.3f}% "
            f"win0={100 * w0:.1f}% win3={100 * w3:.1f}%",
            flush=True,
        )


def _scan_cross4(
    ev: pd.DataFrame,
    gap_col: str,
    ret5_col: str,
    min_n: int,
    label: str,
) -> None:
    print("\n" + "=" * 60)
    print(
        f"Phase 1.5 [{label}] — 0050({gap_col}×{ret5_col}) × 個股(gap×ret5) "
        f"（min_n={min_n}，只印過摩擦）"
    )
    print("=" * 60)
    n_cells_ok = 0
    n_cells_total = 0
    hits4: list[tuple] = []
    grouped = {
        key: g
        for key, g in ev.groupby(
            [gap_col, ret5_col, "stock_gap_sign", "stock_ret5_sign"],
            sort=False,
        )
    }
    for g50 in SIGNS:
        for r50 in SIGNS:
            for gs in SIGNS:
                for rs in SIGNS:
                    n_cells_total += 1
                    sub = grouped.get((g50, r50, gs, rs))
                    if sub is None or len(sub) < min_n:
                        continue
                    n_cells_ok += 1
                    for name in EXIT_TIMES:
                        st = _side_stats(sub[f"r_{name}"])
                        if st["better"] in ("long", "short"):
                            side = st["better"]
                            m = st["long_mean"] if side == "long" else st["short_mean"]
                            w0 = st["long_win0"] if side == "long" else st["short_win0"]
                            w3 = st["long_win3"] if side == "long" else st["short_win3"]
                            hits4.append(
                                (g50, r50, gs, rs, name, side, st["n"], m, w0, w3)
                            )

    print(
        f"有足夠 n 的格子: {n_cells_ok}/{n_cells_total}；"
        f"過摩擦 (格子×出場): {len(hits4)}",
        flush=True,
    )
    print("  勝率: win0=方向>0；win3=方向且|r|≥3%", flush=True)
    if not hits4:
        print("  （無過摩擦組合）", flush=True)
        return
    hits4.sort(key=lambda x: -x[7])
    n_long = sum(1 for h in hits4 if h[5] == "long")
    n_short = sum(1 for h in hits4 if h[5] == "short")
    print(f"  其中 long={n_long}  short={n_short}", flush=True)
    print("\n  Top short:", flush=True)
    shorts = [h for h in hits4 if h[5] == "short"][:15]
    if not shorts:
        print("    （無）", flush=True)
    for g50, r50, gs, rs, name, side, n, m, w0, w3 in shorts:
        print(
            f"    0050[{g50}/{r50}] stock[{gs}/{rs}] @{name}  "
            f"n={n:,} mean={100 * m:+.3f}% "
            f"win0={100 * w0:.1f}% win3={100 * w3:.1f}%",
            flush=True,
        )
    print("\n  Top long:", flush=True)
    longs = [h for h in hits4 if h[5] == "long"][:15]
    if not longs:
        print("    （無）", flush=True)
    for g50, r50, gs, rs, name, side, n, m, w0, w3 in longs:
        print(
            f"    0050[{g50}/{r50}] stock[{gs}/{rs}] @{name}  "
            f"n={n:,} mean={100 * m:+.3f}% "
            f"win0={100 * w0:.1f}% win3={100 * w3:.1f}%",
            flush=True,
        )


def _atr5_at_entry(
    start_date: str, need_sids: set[str], need_days: set[str]
) -> pd.DataFrame:
    """09:00–09:05 m1 算 atr5，取 09:05 當下值（進場可觀測）。"""
    print("載入 pattern m1（09:00–09:05 算 atr5）...", flush=True)
    m1 = load_pattern_m1(start_date=start_date)
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    m1 = m1[
        m1["stock_id"].isin(need_sids)
        & m1["day_str"].isin(need_days)
        & (m1["date"].dt.time >= dtime(9, 0))
        & (m1["date"].dt.time <= ENTRY_T)
    ].copy()
    print(f"m1 bars（開盤5分）: {len(m1):,}", flush=True)
    if m1.empty:
        return pd.DataFrame(columns=["stock_id", "day_str", "atr5"])
    m1["day_date"] = m1["day_str"]
    m1 = add_atr5(m1)
    at = m1[m1["date"].dt.time == ENTRY_T].drop_duplicates(
        ["stock_id", "day_str"], keep="last"
    )
    return at[["stock_id", "day_str", "atr5"]].reset_index(drop=True)


def _print_atr_gates(sub: pd.DataFrame, label: str) -> None:
    """同一條件下掃 atr5 絕對門檻 → short 視角 win0/win3。"""
    print(f"\n  [{label}]", flush=True)
    if sub.empty or "atr5" not in sub.columns:
        print("    （無資料）", flush=True)
        return
    for name in EXIT_TIMES:
        print(f"    @{name}:", flush=True)
        for thr in ATR5_GATES:
            s = sub if thr <= 0 else sub[sub["atr5"] >= thr]
            st = _side_stats(s[f"r_{name}"])
            if st["n"] == 0:
                print(f"      atr5>={thr:.5f}: n=0", flush=True)
                continue
            print(
                f"      atr5>={thr:.5f}: n={st['n']:,}  "
                f"short mean={100 * st['short_mean']:+.3f}%  "
                f"win0={100 * st['short_win0']:.1f}%  "
                f"win3={100 * st['short_win3']:.1f}%",
                flush=True,
            )


def _pivot_exits(m5: pd.DataFrame) -> pd.DataFrame:
    """每 stock_id×day_str 抽出 09:05 open/close 與各出場 close。"""
    m5 = m5.copy()
    m5["t"] = m5["date"].dt.time
    want = {ENTRY_T, *EXIT_TIMES.values()}
    m5 = m5[m5["t"].isin(want)].sort_values(["stock_id", "day_str", "date"])
    m5 = m5.drop_duplicates(["stock_id", "day_str", "t"], keep="last")

    entry = m5[m5["t"] == ENTRY_T][["stock_id", "day_str", "open", "close"]].rename(
        columns={"open": "o905", "close": "p905"}
    )
    out = entry.copy()
    for name, tt in EXIT_TIMES.items():
        ex = m5[m5["t"] == tt][["stock_id", "day_str", "close"]].rename(
            columns={"close": f"p_{name}"}
        )
        out = out.merge(ex, on=["stock_id", "day_str"], how="inner")
    out = out[(out["p905"] > 0) & (out["o905"] > 0)].copy()
    out["ret5"] = (out["p905"] - out["o905"]) / out["o905"]
    for name in EXIT_TIMES:
        out[f"r_{name}"] = (out[f"p_{name}"] - out["p905"]) / out["p905"]
    return out.reset_index(drop=True)


def run(start_date: str, end_date: str, min_n: int = 30) -> None:
    t0 = time.time()
    universe = {str(s) for s in load_tick_universe()}
    trade_universe = universe - {IDX_SYMBOL}

    print("open_drive_0050_stats Phase0+1+1.5", flush=True)
    print(f"母體: {len(trade_universe)} 股 + {IDX_SYMBOL}", flush=True)
    print(
        f"進場 09:05 close；出場 {', '.join(EXIT_TIMES)}；多空皆報",
        flush=True,
    )
    print(
        f"摩擦門檻 |mean|≥{100 * FRICTION:.2f}%；交叉格子 min_n={min_n}",
        flush=True,
    )
    print(
        f"0050 分組: 固定% 與 σ(|z|>{Z_THRESH}) 並行；個股仍用固定%",
        flush=True,
    )
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    print(f"載入 pattern day（start={hist})...", flush=True)
    day = load_pattern_day(start_date=hist)
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].isin(trade_universe | {IDX_SYMBOL})].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)
    day["day_str"] = day["date"].dt.strftime("%Y-%m-%d")
    g = day.groupby("stock_id", sort=False)
    day["prev_close"] = g["close"].shift(1)
    day["gap"] = (day["open"].astype(float) / day["prev_close"].astype(float)) - 1.0

    # 0050 σ20（輔標）
    idx_hist = day[day["stock_id"] == IDX_SYMBOL].copy()
    idx_hist["gap_sigma20"] = idx_hist["gap"].rolling(20, min_periods=10).std()
    idx_hist["z_gap"] = idx_hist["gap"] / idx_hist["gap_sigma20"].replace(0, np.nan)

    idx = idx_hist[
        (idx_hist["date"] >= start_date) & (idx_hist["date"] <= end_date) & idx_hist["gap"].notna()
    ].copy()
    idx["gap_mag"] = idx["gap"].abs().map(_gap_mag_bucket)
    idx["gap_sign"] = idx["gap"].map(_gap_sign)
    print(f"{IDX_SYMBOL} 交易日: {len(idx)}", flush=True)

    need_days = set(idx["day_str"])
    print("載入 pattern m5_std（09:05–10:00）...", flush=True)
    m5 = load_pattern_m5_std(start_date=start_date)
    m5["stock_id"] = m5["stock_id"].astype(str)
    m5["date"] = pd.to_datetime(m5["date"], format="mixed")
    m5["day_str"] = m5["date"].dt.strftime("%Y-%m-%d")
    m5 = m5[
        m5["stock_id"].isin(trade_universe | {IDX_SYMBOL})
        & m5["day_str"].isin(need_days)
        & (m5["date"].dt.time >= ENTRY_T)
        & (m5["date"].dt.time <= dtime(10, 0))
    ].copy()
    print(f"m5 bars: {len(m5):,}", flush=True)

    print("樞紐出場價...", flush=True)
    px = _pivot_exits(m5)
    print(f"完整 09:05+四出場事件: {len(px):,}", flush=True)

    # ---------- Phase 0: 0050 alone ----------
    print("\n" + "=" * 60)
    print("Phase 0 — 0050 alone")
    print("=" * 60)
    idx_px = idx.merge(px[px["stock_id"] == IDX_SYMBOL], on=["stock_id", "day_str"], how="inner")
    print(f"0050 可標籤日: {len(idx_px)}", flush=True)

    print("\n[0050 gap 強度分布]", flush=True)
    for mag, gmag in idx_px.groupby("gap_mag"):
        print(f"  {mag}: n={len(gmag)}  mean_gap={100 * gmag['gap'].mean():+.3f}%", flush=True)

    print("\n[0050 gap 方向分布]", flush=True)
    for sgn, gs in idx_px.groupby("gap_sign"):
        zmean = gs["z_gap"].mean()
        ztxt = f"  mean_z={zmean:+.2f}" if pd.notna(zmean) else ""
        print(
            f"  {sgn}: n={len(gs)}  mean_gap={100 * gs['gap'].mean():+.3f}%{ztxt}",
            flush=True,
        )

    idx_px = idx_px.copy()
    idx_px["ret5_mag"] = idx_px["ret5"].abs().map(_ret5_mag_bucket)
    idx_px["ret5_sign"] = idx_px["ret5"].map(_ret5_sign)

    # 0050 ret5 的 σ20 / z（用完整歷史日序列，再併回區間）
    idx_ret_hist = idx_hist[["day_str", "date"]].merge(
        px[px["stock_id"] == IDX_SYMBOL][["day_str", "ret5"]],
        on="day_str",
        how="left",
    )
    idx_ret_hist = idx_ret_hist.sort_values("date")
    idx_ret_hist["ret5_sigma20"] = idx_ret_hist["ret5"].rolling(20, min_periods=10).std()
    idx_ret_hist["z_ret5"] = idx_ret_hist["ret5"] / idx_ret_hist["ret5_sigma20"].replace(
        0, np.nan
    )
    idx_px = idx_px.merge(
        idx_ret_hist[["day_str", "z_ret5"]], on="day_str", how="left"
    )
    idx_px["gap_sign_z"] = idx_px["z_gap"].map(_z_sign)
    idx_px["ret5_sign_z"] = idx_px["z_ret5"].map(_z_sign)

    print("\n[0050 ret5@09:05 固定%]", flush=True)
    for sgn, gs in idx_px.groupby("ret5_sign"):
        print(f"  {sgn}: n={len(gs)}  mean_ret5={100 * gs['ret5'].mean():+.3f}%", flush=True)

    print(f"\n[0050 gap σ分組 |z|>{Z_THRESH}]", flush=True)
    for sgn, gs in idx_px.groupby("gap_sign_z"):
        print(
            f"  {sgn}: n={len(gs)}  mean_gap={100 * gs['gap'].mean():+.3f}%  "
            f"mean_z={gs['z_gap'].mean():+.2f}",
            flush=True,
        )
    print(f"\n[0050 ret5 σ分組 |z|>{Z_THRESH}]", flush=True)
    for sgn, gs in idx_px.groupby("ret5_sign_z"):
        zm = gs["z_ret5"].mean()
        print(
            f"  {sgn}: n={len(gs)}  mean_ret5={100 * gs['ret5'].mean():+.3f}%  "
            f"mean_z={zm:+.2f}",
            flush=True,
        )
    # 固定% vs σ 一致性
    same_gap = (idx_px["gap_sign"] == idx_px["gap_sign_z"]).mean()
    same_r5 = (idx_px["ret5_sign"] == idx_px["ret5_sign_z"]).mean()
    print(
        f"\n[固定% vs σ 標籤一致率] gap={100 * same_gap:.1f}%  ret5={100 * same_r5:.1f}%",
        flush=True,
    )

    print("\n[0050 自身持有：long / short]", flush=True)
    for name in EXIT_TIMES:
        _print_side(name, _side_stats(idx_px[f"r_{name}"]))

    print("\n[0050 gap_sign × 持有]", flush=True)
    for sgn in ["up", "down", "flat"]:
        sub = idx_px[idx_px["gap_sign"] == sgn]
        if sub.empty:
            continue
        print(f"\n gap_sign={sgn} n={len(sub)}", flush=True)
        for name in EXIT_TIMES:
            _print_side(name, _side_stats(sub[f"r_{name}"]))

    # ---------- Phase 1: stocks ----------
    print("\n" + "=" * 60)
    print("Phase 1 — 個股 join")
    print("=" * 60)

    stocks_day = day[
        day["stock_id"].isin(trade_universe)
        & (day["date"] >= start_date)
        & (day["date"] <= end_date)
        & day["gap"].notna()
    ][["stock_id", "day_str", "gap", "open", "prev_close"]].copy()
    stocks_day = stocks_day.rename(columns={"gap": "gap_stock"})

    idx_f = idx_px[
        [
            "day_str",
            "gap",
            "gap_mag",
            "gap_sign",
            "gap_sign_z",
            "z_gap",
            "ret5",
            "ret5_mag",
            "ret5_sign",
            "ret5_sign_z",
            "z_ret5",
        ]
    ].rename(
        columns={
            "gap": "gap_0050",
            "ret5": "ret5_0050",
            "ret5_mag": "ret5_0050_mag",
            "ret5_sign": "ret5_0050_sign",
        }
    )

    stock_px = px[px["stock_id"] != IDX_SYMBOL].rename(
        columns={"ret5": "ret5_stock", **{f"r_{k}": f"r_{k}" for k in EXIT_TIMES}}
    )
    ev = stocks_day.merge(stock_px, on=["stock_id", "day_str"], how="inner")
    ev = ev.merge(idx_f, on="day_str", how="inner")
    ev["delta_gap"] = ev["gap_stock"] - ev["gap_0050"]
    ev["same_gap_dir"] = np.sign(ev["gap_stock"]) == np.sign(ev["gap_0050"])
    ev["same_ret5_dir"] = np.sign(ev["ret5_stock"]) == np.sign(ev["ret5_0050"])
    ev["stock_gap_sign"] = ev["gap_stock"].map(_gap_sign)
    ev["stock_ret5_sign"] = ev["ret5_stock"].map(_ret5_sign)
    print(f"個股事件: {len(ev):,}", flush=True)

    print("\n[1] 個股 vs 0050 gap", flush=True)
    print(
        f"  corr(gap_stock, gap_0050)={ev['gap_stock'].corr(ev['gap_0050']):.3f}",
        flush=True,
    )
    print(
        f"  同向率={100 * ev['same_gap_dir'].mean():.1f}%  "
        f"delta_gap mean={100 * ev['delta_gap'].mean():+.3f}%  "
        f"p50={100 * ev['delta_gap'].median():+.3f}%",
        flush=True,
    )
    print("  條件於 0050 gap_sign:", flush=True)
    for sgn, gs in ev.groupby("gap_sign"):
        print(
            f"    {sgn}: n={len(gs):,}  gap_stock mean={100 * gs['gap_stock'].mean():+.3f}%  "
            f"delta mean={100 * gs['delta_gap'].mean():+.3f}%  "
            f"stock_up%={100 * (gs['gap_stock'] > 0).mean():.1f}%",
            flush=True,
        )
    print("  條件於 0050 gap_mag:", flush=True)
    for mag, gs in ev.groupby("gap_mag"):
        print(
            f"    {mag}: n={len(gs):,}  gap_stock mean={100 * gs['gap_stock'].mean():+.3f}%  "
            f"delta mean={100 * gs['delta_gap'].mean():+.3f}%",
            flush=True,
        )

    print("\n[2] 0050 ret5 → 個股 ret5", flush=True)
    print(
        f"  corr={ev['ret5_stock'].corr(ev['ret5_0050']):.3f}  "
        f"同向率={100 * ev['same_ret5_dir'].mean():.1f}%",
        flush=True,
    )
    for sgn, gs in ev.groupby("ret5_0050_sign"):
        print(
            f"    0050_ret5={sgn}: n={len(gs):,}  "
            f"stock_ret5 mean={100 * gs['ret5_stock'].mean():+.3f}%  "
            f"同向={100 * gs['same_ret5_dir'].mean():.1f}%",
            flush=True,
        )

    print("\n[3] 0050 gap_sign × ret5_sign → 個股持有多空", flush=True)
    for gsgn in ["up", "down", "flat"]:
        for rsgn in ["up", "down", "flat"]:
            sub = ev[(ev["gap_sign"] == gsgn) & (ev["ret5_0050_sign"] == rsgn)]
            if len(sub) < 10:
                continue
            print(f"\n  gap={gsgn} × ret5={rsgn}  n={len(sub):,}", flush=True)
            for name in EXIT_TIMES:
                _print_side(name, _side_stats(sub[f"r_{name}"]))

    print("\n[4a] 對照 short 候選：0050 gap<0 且 gap_stock≥2%", flush=True)
    short_c = ev[(ev["gap_0050"] < 0) & (ev["gap_stock"] >= 0.02)]
    print(f"  n={len(short_c):,}", flush=True)
    for name in EXIT_TIMES:
        _print_side(name, _side_stats(short_c[f"r_{name}"]))

    print("\n[4b] 對照 long 候選：0050 gap>0 且 delta_gap≥1%", flush=True)
    long_c = ev[(ev["gap_0050"] > 0) & (ev["delta_gap"] >= 0.01)]
    print(f"  n={len(long_c):,}", flush=True)
    for name in EXIT_TIMES:
        _print_side(name, _side_stats(long_c[f"r_{name}"]))

    print("\n[4c] 全樣本基準（無條件）", flush=True)
    for name in EXIT_TIMES:
        _print_side(name, _side_stats(ev[f"r_{name}"]))

    # 0050-only：固定% vs σ
    _scan_0050_only(ev, "gap_sign", "ret5_0050_sign", min_n, "固定%")
    _scan_0050_only(ev, "gap_sign_z", "ret5_sign_z", min_n, f"σ|z|>{Z_THRESH}")

    # Phase 1.5：固定% vs σ（個股維度仍固定%）
    _scan_cross4(ev, "gap_sign", "ret5_0050_sign", min_n, "固定%")
    _scan_cross4(ev, "gap_sign_z", "ret5_sign_z", min_n, f"σ|z|>{Z_THRESH}")

    # ---------- Phase ATR：進場 atr5 絕對門檻 ----------
    print("\n" + "=" * 60)
    print(
        "Phase ATR — 09:05 atr5 絕對門檻（short 視角；"
        f"門檻 {', '.join(f'{t:.5f}' for t in ATR5_GATES)}）"
    )
    print("=" * 60)
    atr = _atr5_at_entry(
        start_date, set(ev["stock_id"].unique()), set(ev["day_str"].unique())
    )
    ev = ev.merge(atr, on=["stock_id", "day_str"], how="left")
    n_atr = int(ev["atr5"].notna().sum())
    print(
        f"有 atr5: {n_atr:,}/{len(ev):,}  "
        f"p50={ev['atr5'].median():.5f}  p90={ev['atr5'].quantile(0.9):.5f}  "
        f"p99={ev['atr5'].quantile(0.99):.5f}",
        flush=True,
    )

    # 關鍵 short 格子；個股以 atr5 p99 為主濾網（並掃其他門檻對照）
    key_masks = [
        (
            "0050 up/up × stock up/down（fade）",
            (ev["gap_sign"] == "up")
            & (ev["ret5_0050_sign"] == "up")
            & (ev["stock_gap_sign"] == "up")
            & (ev["stock_ret5_sign"] == "down"),
        ),
        (
            "0050 down/down × stock up/down（固定%）",
            (ev["gap_sign"] == "down")
            & (ev["ret5_0050_sign"] == "down")
            & (ev["stock_gap_sign"] == "up")
            & (ev["stock_ret5_sign"] == "down"),
        ),
        (
            "0050 σ up/flat × stock down/up",
            (ev["gap_sign_z"] == "up")
            & (ev["ret5_sign_z"] == "flat")
            & (ev["stock_gap_sign"] == "down")
            & (ev["stock_ret5_sign"] == "up"),
        ),
        (
            "0050 down/down × stock down/down（續跌）",
            (ev["gap_sign"] == "down")
            & (ev["ret5_0050_sign"] == "down")
            & (ev["stock_gap_sign"] == "down")
            & (ev["stock_ret5_sign"] == "down"),
        ),
        (
            "0050-only up/up",
            (ev["gap_sign"] == "up") & (ev["ret5_0050_sign"] == "up"),
        ),
        (
            "0050-only down/down",
            (ev["gap_sign"] == "down") & (ev["ret5_0050_sign"] == "down"),
        ),
        (
            "0050 σ down/down × stock down/down",
            (ev["gap_sign_z"] == "down")
            & (ev["ret5_sign_z"] == "down")
            & (ev["stock_gap_sign"] == "down")
            & (ev["stock_ret5_sign"] == "down"),
        ),
    ]
    print(
        f"\n個股濾網主軸: m1 atr5 ≥ p99 = {ATR5_FILTER_THRESHOLD:.5f} "
        f"（下表仍列出各門檻對照）",
        flush=True,
    )
    for label, mask in key_masks:
        _print_atr_gates(ev.loc[mask], label)
        # 主軸 p99 摘要（各出場）
        sub_p99 = ev.loc[mask & (ev["atr5"] >= ATR5_FILTER_THRESHOLD)]
        print(f"    >> p99 摘要 [{label}] n={len(sub_p99):,}", flush=True)
        for name in EXIT_TIMES:
            st = _side_stats(sub_p99[f"r_{name}"])
            if st["n"] == 0:
                print(f"       @{name}: n=0", flush=True)
                continue
            print(
                f"       @{name}: n={st['n']:,}  "
                f"short mean={100 * st['short_mean']:+.3f}%  "
                f"win0={100 * st['short_win0']:.1f}%  "
                f"win3={100 * st['short_win3']:.1f}%",
                flush=True,
            )

    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description="0050 open-drive 敘述統計 Phase0+1+1.5")
    p.add_argument("--start_date", default="2026-06-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument("--min_n", type=int, default=30, help="交叉格子最少樣本數")
    args = p.parse_args()
    run(args.start_date, args.end_date, min_n=args.min_n)


if __name__ == "__main__":
    main()
