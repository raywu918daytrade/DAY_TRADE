"""
ret5 拉回反轉做多 — 純規則三分類 TB 驗證。

拉回／反轉 K 只看 m5_std。規則見 strategy/ret5_pullback_reversal/README.md。

用法：
    python -m strategy.ret5_pullback_reversal.verify \\
        --start_date 2026-01-01 --end_date 2026-07-31
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import time as dtime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from data import raw_query
from data.adjustment_query import _adjust_ohlc, load_pattern_day
from finmind.stock_universe_2000 import load_stock_universe_2000
from finmind.tick_universe import load_tick_universe
from strategy.mkt.config import ATR5_FILTER_THRESHOLD, IDX_SYMBOL
from strategy.mkt.features import add_atr5
from strategy.ret5_pullback_reversal.config import (
    BODY_RATIO_MIN,
    ENTRY_DEADLINE,
    FIRST_M5_T,
    HOLD_M5_BARS,
    M5_LOAD_UNTIL,
    RET5_MIN,
    SIGNAL_DEADLINE,
    SL_PCT,
    TP_PCT,
)


def _ym_from_month_path(path: str) -> str:
    return Path(path).stem.replace("_", "-")


def _body_ratio(o: float, h: float, lo: float, c: float) -> float:
    rng = h - lo
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng


def _is_solid_bull(o: float, h: float, lo: float, c: float) -> bool:
    return c > o and _body_ratio(o, h, lo, c) >= BODY_RATIO_MIN


def _find_pullback_reversal(
    grp: pd.DataFrame, m5_1_low: float
) -> tuple[pd.Timestamp, float] | None:
    """掃連續陰→陽實體，陰線 low > m5_1_low；兩根時間皆 < SIGNAL_DEADLINE。
    回傳 (entry_ts, entry_price) 或 None（取第一組）。"""
    if grp.empty or not np.isfinite(m5_1_low):
        return None
    g = grp.sort_values("date").reset_index(drop=True)
    o = g["open"].astype(float).values
    h = g["high"].astype(float).values
    lo = g["low"].astype(float).values
    c = g["close"].astype(float).values
    ts = g["date"].values
    n = len(g)
    for i in range(n - 1):
        t0 = pd.Timestamp(ts[i])
        t1 = pd.Timestamp(ts[i + 1])
        if t0.time() >= SIGNAL_DEADLINE:
            break
        if t1.time() >= SIGNAL_DEADLINE:
            break
        if t1.time() >= ENTRY_DEADLINE:
            break
        if not (c[i] < o[i]):
            continue
        if not (lo[i] > m5_1_low):
            continue
        if not _is_solid_bull(o[i + 1], h[i + 1], lo[i + 1], c[i + 1]):
            continue
        return t1, float(c[i + 1])
    return None


def _long_tb_m5(
    day_m5: pd.DataFrame, entry_ts: pd.Timestamp, entry: float
) -> dict | None:
    """做多三分類 TB：+1 TP / -1 SL / 0 持滿或當日無更多棒。同根先 TP。"""
    if entry <= 0:
        return None
    fut = day_m5[day_m5["date"] > entry_ts].sort_values("date").head(HOLD_M5_BARS)
    if fut.empty:
        return None
    tp = entry * (1.0 + TP_PCT)
    sl = entry * (1.0 - SL_PCT)
    for j, (_, row) in enumerate(fut.iterrows(), start=1):
        ts = pd.Timestamp(row["date"])
        hi, lo = float(row["high"]), float(row["low"])
        if hi >= tp:
            return {
                "label": 1.0,
                "exit_ts": ts,
                "exit_price": tp,
                "exit_reason": "tp",
                "bars_held": j,
            }
        if lo <= sl:
            return {
                "label": -1.0,
                "exit_ts": ts,
                "exit_price": sl,
                "exit_reason": "sl",
                "bars_held": j,
            }
    last = fut.iloc[-1]
    return {
        "label": 0.0,
        "exit_ts": pd.Timestamp(last["date"]),
        "exit_price": float(last["close"]),
        "exit_reason": "time",
        "bars_held": len(fut),
    }


def _load_bars_months(
    db_subdir: str,
    start_date: str,
    end_date: str,
    keys: pd.DataFrame,
    t_lo: dtime,
    t_hi: dtime,
    label: str,
) -> pd.DataFrame:
    """逐月讀 db/{subdir}，只留 keys 的 (stock, day) 與時段。"""
    paths = raw_query._month_file_list(_ROOT / db_subdir, start_date, end_date)
    if not paths:
        return pd.DataFrame()
    keys = keys[["stock_id", "day_str"]].drop_duplicates().copy()
    keys["stock_id"] = keys["stock_id"].astype(str)
    keys = keys.assign(ym=keys["day_str"].str[:7])
    chunks: list[pd.DataFrame] = []
    for path in paths:
        ym = _ym_from_month_path(path)
        sub = keys[keys["ym"] == ym]
        if sub.empty:
            continue
        month_days = set(sub["day_str"])
        sid_list = sub["stock_id"].unique().tolist()
        filt = ds.field("stock_id").isin(sid_list)
        df = ds.dataset(path, format="parquet").to_table(filter=filt).to_pandas()
        if df.empty:
            continue
        df["stock_id"] = df["stock_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df["day_str"] = df["date"].dt.strftime("%Y-%m-%d")
        # t_hi 用 <：SIGNAL_DEADLINE 當 exclusive 時由呼叫端傳 t_hi=09:29 不方便，
        # 這裡用 <= t_hi，呼叫端傳 09:25/09:29 或之後再濾 < SIGNAL_DEADLINE
        df = df[
            df["day_str"].isin(month_days)
            & (df["date"].dt.time >= t_lo)
            & (df["date"].dt.time <= t_hi)
        ]
        if df.empty:
            continue
        df = df.merge(sub[["stock_id", "day_str"]], on=["stock_id", "day_str"], how="inner")
        if df.empty:
            continue
        chunks.append(df)
        print(
            f"  {label} {ym}: sids={len(sid_list)} days={len(month_days)} rows={len(df):,}",
            flush=True,
        )
        del df
        gc.collect()
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    out = _adjust_ohlc(out, start_date, end_date)
    return out.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _summarize(df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print("  n=0", flush=True)
        return
    n_tp = int((df["label"] == 1.0).sum())
    n_flat = int((df["label"] == 0.0).sum())
    n_sl = int((df["label"] == -1.0).sum())
    print(f"  n={n:,}", flush=True)
    print(f"  止盈(+{TP_PCT:.0%}): {n_tp:,}  {100 * n_tp / n:.1f}%", flush=True)
    print(f"  持平: {n_flat:,}  {100 * n_flat / n:.1f}%", flush=True)
    print(f"  止損(-{SL_PCT:.0%}): {n_sl:,}  {100 * n_sl / n:.1f}%", flush=True)
    print(
        f"  做多 mean={100 * df['pnl_pct'].mean():.3f}%  "
        f"median={100 * df['pnl_pct'].median():.3f}%",
        flush=True,
    )


def run(start_date: str, end_date: str, use_tick_universe: bool = False) -> pd.DataFrame:
    t0 = time.time()
    if use_tick_universe:
        raw = load_tick_universe()
        univ_label = "tick_universe"
    else:
        raw = load_stock_universe_2000()
        univ_label = "stock_universe_2000"
    universe = {str(s) for s in raw} | {IDX_SYMBOL}
    trade_universe = universe - {IDX_SYMBOL}

    print("ret5_pullback_reversal 做多三分類 TB", flush=True)
    print(
        f"母體: {univ_label}（交易 {len(trade_universe)} 支）",
        flush=True,
    )
    print(
        f"濾網: ret5 紅K 且 vs_prev≥{RET5_MIN:.0%}；m5 陰線 low>m5_1_low；"
        f"下一根 m5 陽線實體 body≥{BODY_RATIO_MIN:.0%}；"
        f"訊號<{SIGNAL_DEADLINE.strftime('%H:%M')}；進場<{ENTRY_DEADLINE.strftime('%H:%M')}；"
        f"atr5≥{ATR5_FILTER_THRESHOLD:.5f}(p99)",
        flush=True,
    )
    print(
        f"標籤: 做多 TB ±{TP_PCT:.0%} / 最多 {HOLD_M5_BARS} 根 m5（30 分）",
        flush=True,
    )
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    print(f"載入 pattern day（{hist} ~ {end_date})...", flush=True)
    day = load_pattern_day(start_date=hist, end_date=end_date)
    if day.empty:
        print("無 day 資料", flush=True)
        return pd.DataFrame()
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].isin(trade_universe)].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)
    day["day_str"] = day["date"].dt.strftime("%Y-%m-%d")
    g = day.groupby("stock_id", sort=False)
    day["prev_close"] = g["close"].shift(1)
    day_cands = day[
        (day["date"] >= start_date)
        & (day["date"] <= end_date)
        & day["prev_close"].notna()
    ][["stock_id", "day_str", "prev_close"]].copy()
    del day
    gc.collect()
    print(f"日K 股日: {len(day_cands):,}", flush=True)
    if day_cands.empty:
        return pd.DataFrame()

    print("載入 m5_std（逐月）...", flush=True)
    m5 = _load_bars_months(
        "db/m5_std",
        start_date,
        end_date,
        day_cands,
        FIRST_M5_T,
        M5_LOAD_UNTIL,
        "m5",
    )
    if m5.empty:
        print("無 m5 資料", flush=True)
        return pd.DataFrame()
    print(f"m5 bars: {len(m5):,}", flush=True)

    bar905 = m5[m5["date"].dt.time == FIRST_M5_T].drop_duplicates(
        ["stock_id", "day_str"], keep="last"
    )[["stock_id", "day_str", "open", "low", "close"]].rename(
        columns={"low": "m5_1_low", "close": "p905", "open": "o905"}
    )
    # ret5 那根須為紅K（收 > 開）
    bar905 = bar905[bar905["p905"].astype(float) > bar905["o905"].astype(float)].copy()
    cands = day_cands.merge(bar905, on=["stock_id", "day_str"], how="inner")
    cands["ret5_vs_prev"] = (cands["p905"].astype(float) / cands["prev_close"].astype(float)) - 1.0
    cands = cands[cands["ret5_vs_prev"] >= RET5_MIN].copy()
    print(
        f"ret5 紅K 且 vs_prev≥{RET5_MIN:.0%} 候選: {len(cands):,}",
        flush=True,
    )
    if cands.empty:
        return cands

    keys = cands[["stock_id", "day_str"]]
    # 拉回／反轉只看 m5（訊號窗 < 09:30）
    m5_sig = m5[
        (m5["date"].dt.time >= FIRST_M5_T) & (m5["date"].dt.time < SIGNAL_DEADLINE)
    ].copy()
    m5_sig = m5_sig.merge(keys, on=["stock_id", "day_str"], how="inner")

    low_map = cands.set_index(["stock_id", "day_str"])["m5_1_low"].astype(float).to_dict()
    ret_map = cands.set_index(["stock_id", "day_str"])["ret5_vs_prev"].astype(float).to_dict()
    pc_map = cands.set_index(["stock_id", "day_str"])["prev_close"].astype(float).to_dict()

    signal_rows: list[dict] = []
    for (sid, day_str), grp in m5_sig.groupby(["stock_id", "day_str"], sort=False):
        m5_1_low = low_map.get((sid, day_str))
        if m5_1_low is None:
            continue
        found = _find_pullback_reversal(grp, float(m5_1_low))
        if found is None:
            continue
        entry_ts, entry = found
        if entry_ts.time() >= ENTRY_DEADLINE:
            continue
        signal_rows.append(
            {
                "stock_id": sid,
                "day_str": day_str,
                "tf": "m5",
                "entry_ts": entry_ts,
                "entry": entry,
                "m5_1_low": float(m5_1_low),
                "ret5_vs_prev": float(ret_map[(sid, day_str)]),
                "prev_close": float(pc_map[(sid, day_str)]),
            }
        )
    del m5_sig
    gc.collect()

    if not signal_rows:
        print("無 m5 拉回反轉訊號", flush=True)
        print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
        return pd.DataFrame()

    sig = pd.DataFrame(signal_rows)
    print(f"m5 訊號事件: {len(sig):,}", flush=True)

    # atr5 @ 進場分鐘
    print("載入 m1 算 atr5@進場...", flush=True)
    atr_keys = sig[["stock_id", "day_str"]].drop_duplicates()
    m1_atr = _load_bars_months(
        "db/m1",
        start_date,
        end_date,
        atr_keys,
        dtime(9, 0),
        ENTRY_DEADLINE,  # 含到 10:00 前最後一分
        "m1_atr",
    )
    if m1_atr.empty:
        print("無 m1 atr 資料", flush=True)
        return pd.DataFrame()
    m1_atr["day_date"] = m1_atr["day_str"]
    m1_atr = add_atr5(m1_atr)
    atr = m1_atr[["stock_id", "day_str", "date", "atr5"]].dropna(subset=["atr5"])
    del m1_atr
    gc.collect()
    atr = atr.rename(columns={"date": "entry_ts"})
    sig = sig.merge(atr, on=["stock_id", "day_str", "entry_ts"], how="left")
    n_before = len(sig)
    sig = sig[sig["atr5"].notna() & (sig["atr5"] >= ATR5_FILTER_THRESHOLD)].copy()
    print(
        f"atr5≥p99: {len(sig):,} / {n_before:,}",
        flush=True,
    )
    if sig.empty:
        print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
        return sig

    m5_by = {
        k: g.reset_index(drop=True)
        for k, g in m5.groupby(["stock_id", "day_str"], sort=False)
    }
    del m5
    gc.collect()

    rows = []
    for _, r in sig.iterrows():
        key = (r["stock_id"], r["day_str"])
        day_m5 = m5_by.get(key)
        if day_m5 is None:
            continue
        entry = float(r["entry"])
        entry_ts = pd.Timestamp(r["entry_ts"])
        detail = _long_tb_m5(day_m5, entry_ts, entry)
        if detail is None:
            continue
        rows.append(
            {
                "stock_id": r["stock_id"],
                "day_str": r["day_str"],
                "tf": r["tf"],
                "ret5_vs_prev": float(r["ret5_vs_prev"]),
                "m5_1_low": float(r["m5_1_low"]),
                "atr5": float(r["atr5"]),
                "entry_ts": entry_ts,
                "entry": entry,
                "label": detail["label"],
                "exit_reason": detail["exit_reason"],
                "exit_price": detail["exit_price"],
                "bars_held": detail["bars_held"],
                "pnl_pct": (detail["exit_price"] - entry) / entry,
            }
        )

    out = pd.DataFrame(rows)
    print(f"可標籤事件: {len(out):,}", flush=True)
    if out.empty:
        print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
        return out

    print(f"觸發交易日數: {out['day_str'].nunique()}", flush=True)
    print(f"TF 分布: {out['tf'].value_counts().to_dict()}", flush=True)

    print("\n" + "=" * 56)
    print("做多 TB 三分類（±3% / 最多 30 分）")
    print("=" * 56)
    _summarize(out)

    out = out.copy()
    out["year"] = pd.to_datetime(out["day_str"]).dt.year
    print("\n分年:", flush=True)
    for y, g in out.groupby("year"):
        n = len(g)
        print(
            f"  {y}: n={n}  days={g['day_str'].nunique()}  "
            f"TP={100 * (g['label'] == 1).mean():.1f}%  "
            f"flat={100 * (g['label'] == 0).mean():.1f}%  "
            f"SL={100 * (g['label'] == -1).mean():.1f}%  "
            f"mean={100 * g['pnl_pct'].mean():.3f}%",
            flush=True,
        )

    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description="ret5 拉回反轉做多三分類驗證（拉回/反轉僅 m5）")
    p.add_argument("--start_date", default="2026-01-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument(
        "--use_tick_universe",
        action="store_true",
        help="改用 db/tickers/tick_universe.parquet（約 400 支）",
    )
    args = p.parse_args()
    run(args.start_date, args.end_date, use_tick_universe=args.use_tick_universe)


if __name__ == "__main__":
    main()
