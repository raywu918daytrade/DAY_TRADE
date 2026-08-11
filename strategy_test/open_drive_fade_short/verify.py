"""
鎖定（gap-only）：0050 開高>1% × 個股開高>1% × atr5≥p99
→ m5 close 做空；TB ±3% / 持有 11 根 m5。不管 ret5。

進場窗口：
  預設只在 09:05 進場
  --entry_until 13:00 → 09:05～13:00 每根 m5 都可進場（同日同股可多次）

記憶體：m5／m1 逐月讀 + 只留 gap 候選 (stock, day)。

用法：
    python -m strategy_test.open_drive_fade_short.verify \\
        --start_date 2020-01-01 --end_date 2022-12-31 --use_2000
    python -m strategy_test.open_drive_fade_short.verify \\
        --start_date 2020-01-01 --end_date 2022-12-31 --use_2000 --entry_until 13:00
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

ENTRY_T = dtime(9, 5)
SESSION_END = dtime(13, 30)  # TB 後續 m5 用到收盤
GAP_UP = 0.01  # >1%
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_M5_BARS = 11  # 等同原「持有至 10:00」的 11 根


def _short_tb_m5(
    day_m5: pd.DataFrame, entry_ts: pd.Timestamp, entry: float, hold_bars: int
) -> dict | None:
    """做空對稱 TB：+1 TP / -1 SL / 0 持滿或當日無更多棒。同根先 TP。"""
    if entry <= 0:
        return None
    fut = day_m5[day_m5["date"] > entry_ts].sort_values("date").head(hold_bars)
    if fut.empty:
        return None
    tp = entry * (1.0 - TP_PCT)
    sl = entry * (1.0 + SL_PCT)
    for j, (_, row) in enumerate(fut.iterrows(), start=1):
        ts = pd.Timestamp(row["date"])
        hi, lo = float(row["high"]), float(row["low"])
        if lo <= tp:
            return {
                "label": 1.0,
                "exit_ts": ts,
                "exit_price": tp,
                "exit_reason": "tp",
                "bars_held": j,
            }
        if hi >= sl:
            return {
                "label": -1.0,
                "exit_ts": ts,
                "exit_price": sl,
                "exit_reason": "sl",
                "bars_held": j,
            }
    last = fut.iloc[-1]
    # 盤中晚進場可能湊不滿 hold_bars → 以當日最後一根 time exit
    return {
        "label": 0.0,
        "exit_ts": pd.Timestamp(last["date"]),
        "exit_price": float(last["close"]),
        "exit_reason": "time",
        "bars_held": len(fut),
    }


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
    print(f"  震盪(time): {n_flat:,}  {100 * n_flat / n:.1f}%", flush=True)
    print(f"  止損(-{SL_PCT:.0%}): {n_sl:,}  {100 * n_sl / n:.1f}%", flush=True)
    print(
        f"  做空 mean={100 * df['pnl_pct'].mean():.3f}%  "
        f"median={100 * df['pnl_pct'].median():.3f}%",
        flush=True,
    )


def _ym_from_month_path(path: str) -> str:
    """db/.../2020_01.parquet -> '2020-01'。"""
    return Path(path).stem.replace("_", "-")


def _load_m5_months(
    start_date: str,
    end_date: str,
    keys: pd.DataFrame,
    t_lo: dtime,
    t_hi: dtime,
) -> pd.DataFrame:
    """逐月讀 m5_std：keys 為 (stock_id, day_str)，另強制含 0050 同日。"""
    paths = raw_query._month_file_list(_ROOT / "db/m5_std", start_date, end_date)
    if not paths:
        return pd.DataFrame()
    keys = keys[["stock_id", "day_str"]].drop_duplicates().copy()
    keys["stock_id"] = keys["stock_id"].astype(str)
    idx_keys = pd.DataFrame(
        {"stock_id": IDX_SYMBOL, "day_str": sorted(keys["day_str"].unique())}
    )
    keys = pd.concat([keys, idx_keys], ignore_index=True).drop_duplicates()
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
            f"  m5 {ym}: sids={len(sid_list)} days={len(month_days)} rows={len(df):,}",
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


def _load_m1_atr5_bars(
    start_date: str,
    end_date: str,
    keys: pd.DataFrame,
    t_hi: dtime,
) -> pd.DataFrame:
    """逐月讀 m1：候選 (stock, day) 的 09:00～t_hi，回傳每分鐘 atr5。"""
    paths = raw_query._month_file_list(_ROOT / "db/m1", start_date, end_date)
    if not paths:
        return pd.DataFrame()
    t_lo = dtime(9, 0)
    ev_keys = keys[["stock_id", "day_str"]].drop_duplicates().copy()
    ev_keys["stock_id"] = ev_keys["stock_id"].astype(str)
    ev_keys = ev_keys.assign(ym=ev_keys["day_str"].str[:7])
    chunks: list[pd.DataFrame] = []
    for path in paths:
        ym = _ym_from_month_path(path)
        sub = ev_keys[ev_keys["ym"] == ym]
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
            f"  m1 {ym}: sids={len(sid_list)} days={len(month_days)} rows={len(df):,}",
            flush=True,
        )
        del df
        gc.collect()
    if not chunks:
        return pd.DataFrame(columns=["stock_id", "day_str", "date", "atr5"])
    m1 = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    m1 = _adjust_ohlc(m1, start_date, end_date)
    m1["day_date"] = m1["day_str"]
    m1 = add_atr5(m1)
    atr = m1[["stock_id", "day_str", "date", "atr5"]].dropna(subset=["atr5"])
    del m1
    gc.collect()
    return atr


def run(
    start_date: str,
    end_date: str,
    use_2000: bool = False,
    entry_until_str: str = "09:05",
    filter_mode: str = "gap_only",
) -> pd.DataFrame:
    t0 = time.time()
    if entry_until_str == "13:00":
        entry_until = dtime(13, 0)
        multi_entry = True
    else:
        entry_until = ENTRY_T
        multi_entry = False
    hold_bars = HOLD_M5_BARS

    if use_2000:
        raw_univ = load_stock_universe_2000()
        universe = {str(s) for s in raw_univ} | {IDX_SYMBOL}
        univ_label = f"stock_universe_2000 ({len(universe)-1} 支個股)"
    else:
        raw_univ = load_tick_universe()
        universe = {str(s) for s in raw_univ} | {IDX_SYMBOL}
        univ_label = f"tick_universe ({len(universe)-1} 支個股)"

    trade_universe = universe - {IDX_SYMBOL}

    print(f"open_drive_fade_short TB 驗證 ({univ_label})", flush=True)
    print(
        f"濾網模式: {filter_mode} (0050 gap>{GAP_UP:.0%}, atr5≥{ATR5_FILTER_THRESHOLD:.5f}(p99))",
        flush=True,
    )
    if multi_entry:
        print(
            f"進場：09:05～{entry_until_str} 每根 m5 close 做空（同日同股可多次）；"
            f"TB ±{TP_PCT:.0%} / 持有最多 {hold_bars} 根 m5",
            flush=True,
        )
    else:
        print(
            f"進場：僅 09:05 close 做空；TB ±{TP_PCT:.0%} / 持有最多 {hold_bars} 根 m5",
            flush=True,
        )
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    print(f"載入 pattern day（{hist} ~ {end_date})...", flush=True)
    day = load_pattern_day(start_date=hist, end_date=end_date)
    if day.empty:
        print("未載入到 day 資料", flush=True)
        return pd.DataFrame()

    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].isin(trade_universe | {IDX_SYMBOL})].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)
    day["day_str"] = day["date"].dt.strftime("%Y-%m-%d")
    g = day.groupby("stock_id", sort=False)
    day["prev_close"] = g["close"].shift(1)
    day["gap"] = (day["open"].astype(float) / day["prev_close"].astype(float)) - 1.0

    idx = day[
        (day["stock_id"] == IDX_SYMBOL)
        & (day["date"] >= start_date)
        & (day["date"] <= end_date)
        & day["gap"].notna()
        & (day["gap"] > GAP_UP)
    ][["day_str", "gap"]].rename(columns={"gap": "gap_0050"})
    print(f"0050 gap>{GAP_UP:.0%} 日: {len(idx)}", flush=True)
    if idx.empty:
        return pd.DataFrame()

    need_days = set(idx["day_str"])
    stocks_day = day[
        day["stock_id"].isin(trade_universe)
        & (day["date"] >= start_date)
        & (day["date"] <= end_date)
        & day["gap"].notna()
        & (day["gap"] > GAP_UP)
        & day["day_str"].isin(need_days)
    ][["stock_id", "day_str", "gap"]].rename(columns={"gap": "gap_stock"})
    del day
    gc.collect()
    print(f"日K gap 候選: {len(stocks_day):,}（{stocks_day['stock_id'].nunique()} 股）", flush=True)
    if stocks_day.empty:
        return pd.DataFrame()

    # m5：進場窗 + 之後 TB（至收盤）
    print(f"載入 pattern m5_std（逐月，{len(need_days)} 日）...", flush=True)
    m5 = _load_m5_months(
        start_date,
        end_date,
        stocks_day[["stock_id", "day_str"]],
        ENTRY_T,
        SESSION_END,
    )
    if m5.empty:
        print("未載入到 m5_std 資料", flush=True)
        return pd.DataFrame()
    print(f"m5 bars: {len(m5):,}", flush=True)

    # 0050／fade：仍看 09:05 ret5（日級濾網）
    bar905 = m5[m5["date"].dt.time == ENTRY_T].drop_duplicates(
        ["stock_id", "day_str"], keep="last"
    )
    bar905 = bar905.rename(columns={"date": "ts905", "open": "o905", "close": "p905"})
    bar905["ret5"] = (bar905["p905"] - bar905["o905"]) / bar905["o905"].replace(0, np.nan)

    idx905 = bar905[bar905["stock_id"] == IDX_SYMBOL][["day_str", "ret5"]].rename(
        columns={"ret5": "ret5_0050"}
    )
    idx905 = idx905.merge(idx, on="day_str", how="inner")
    if filter_mode == "fade":
        idx905 = idx905[idx905["ret5_0050"] > 0.005].copy()
    print(f"符合 0050 條件日（有 09:05）: {len(idx905)}", flush=True)
    if idx905.empty:
        return pd.DataFrame()

    stocks_day = stocks_day[stocks_day["day_str"].isin(set(idx905["day_str"]))].copy()
    if filter_mode == "fade":
        fade_ok = bar905[
            (bar905["stock_id"] != IDX_SYMBOL) & (bar905["ret5"] < -0.005)
        ][["stock_id", "day_str"]]
        stocks_day = stocks_day.merge(fade_ok, on=["stock_id", "day_str"], how="inner")

    stocks_day = stocks_day.merge(idx905, on="day_str", how="inner")
    print(f"日級候選: {len(stocks_day):,}", flush=True)
    if stocks_day.empty:
        return stocks_day

    # 進場棒：單點 09:05 或整段至 entry_until
    entry_bars = m5[
        (m5["stock_id"] != IDX_SYMBOL)
        & (m5["date"].dt.time >= ENTRY_T)
        & (m5["date"].dt.time <= entry_until)
    ][["stock_id", "day_str", "date", "open", "close"]].copy()
    entry_bars = entry_bars.merge(
        stocks_day[["stock_id", "day_str", "gap_stock", "gap_0050", "ret5_0050"]],
        on=["stock_id", "day_str"],
        how="inner",
    )
    entry_bars["ret5_bar"] = (entry_bars["close"] - entry_bars["open"]) / entry_bars[
        "open"
    ].replace(0, np.nan)
    print(f"進場棒候選: {len(entry_bars):,}", flush=True)
    if entry_bars.empty:
        return pd.DataFrame()

    print(
        f"載入 m1 算 atr5（逐月，至 {entry_until_str}）...",
        flush=True,
    )
    atr = _load_m1_atr5_bars(
        start_date,
        end_date,
        entry_bars[["stock_id", "day_str"]],
        entry_until,
    )
    if atr.empty:
        print("未載入到 m1 atr5 資料", flush=True)
        return pd.DataFrame()
    ev = entry_bars.merge(
        atr, left_on=["stock_id", "day_str", "date"], right_on=["stock_id", "day_str", "date"], how="inner"
    )
    ev = ev[ev["atr5"] >= ATR5_FILTER_THRESHOLD].copy()
    print(f"atr5≥p99 後進場: {len(ev):,}", flush=True)
    if ev.empty:
        return ev

    keys = ev[["stock_id", "day_str"]].drop_duplicates()
    m5_tb = m5[m5["stock_id"] != IDX_SYMBOL].merge(keys, on=["stock_id", "day_str"], how="inner")
    m5_by = {
        k: g.reset_index(drop=True)
        for k, g in m5_tb.groupby(["stock_id", "day_str"], sort=False)
    }
    del m5, m5_tb, atr, entry_bars
    gc.collect()

    rows = []
    for _, r in ev.iterrows():
        key = (r["stock_id"], r["day_str"])
        day_m5 = m5_by.get(key)
        if day_m5 is None:
            continue
        entry = float(r["close"])
        entry_ts = pd.Timestamp(r["date"])
        detail = _short_tb_m5(day_m5, entry_ts, entry, hold_bars)
        if detail is None:
            continue
        rows.append(
            {
                "stock_id": r["stock_id"],
                "day_str": r["day_str"],
                "gap_0050": float(r["gap_0050"]),
                "ret5_0050": float(r["ret5_0050"]),
                "gap_stock": float(r["gap_stock"]),
                "ret5_bar": float(r["ret5_bar"]) if pd.notna(r["ret5_bar"]) else np.nan,
                "atr5": float(r["atr5"]),
                "entry_ts": entry_ts,
                "entry": entry,
                "label": detail["label"],
                "exit_reason": detail["exit_reason"],
                "exit_price": detail["exit_price"],
                "bars_held": detail["bars_held"],
                "pnl_pct": (entry - detail["exit_price"]) / entry,
            }
        )

    out = pd.DataFrame(rows)
    print(f"可標籤事件: {len(out):,}", flush=True)
    if out.empty:
        print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
        return out

    n_days = out["day_str"].nunique()
    print(f"觸發交易日數: {n_days}", flush=True)
    if multi_entry and len(out):
        per_day = out.groupby("day_str").size()
        print(
            f"同日進場數: mean={per_day.mean():.1f}  max={per_day.max()}  "
            f"總進場棒時段={out['entry_ts'].dt.time.nunique()}",
            flush=True,
        )

    print("\n" + "=" * 56)
    mode = f"entry≤{entry_until_str}" if multi_entry else "entry@09:05"
    print(f"做空 TB（±3% / 最多 {hold_bars} 根 m5）— {filter_mode} / {mode}")
    print("=" * 56)
    _summarize(out)

    if len(out):
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
    p = argparse.ArgumentParser(description="open_drive short TB 驗證")
    p.add_argument("--start_date", default="2024-01-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument("--use_2000", action="store_true", help="使用 stock_universe_2000 清單")
    p.add_argument(
        "--entry_until",
        default="09:05",
        choices=["09:05", "13:00"],
        help="最晚進場時間：09:05=只開盤那根；13:00=09:05～13:00 每根 m5 可進",
    )
    p.add_argument(
        "--filter_mode",
        default="gap_only",
        choices=["gap_only", "fade"],
        help="濾網條件模式 (gap_only 或 fade)",
    )
    args = p.parse_args()
    run(
        args.start_date,
        args.end_date,
        use_2000=args.use_2000,
        entry_until_str=args.entry_until,
        filter_mode=args.filter_mode,
    )


if __name__ == "__main__":
    main()
