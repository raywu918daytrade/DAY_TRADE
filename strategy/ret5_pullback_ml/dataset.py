"""
ret5_pullback_ml 事件挖掘 + 做多三分類 TB 標籤。

硬過濾（規則 verify／ML 共用）：
1. ret5@09:05 紅K 且 vs 昨收 ≥ RET5_MIN
2. m5 陰線、low > m5_1_low、收盤 < SIGNAL_DEADLINE
3. 之後 M1_REV_LOOKAHEAD 根 m1：陽線、量>前1、close > 該 m5 high、進場 < ENTRY_DEADLINE

標籤：做多 TB ±TP_PCT／最多 HOLD_M5_BARS 根 m5 → target 0=止損 / 1=震盪 / 2=止盈。
atr5 在本函式不當硬過濾（進 FEATURES）；純規則見 verify.py。
"""

from __future__ import annotations

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

from data.adjustment_query import _adjust_ohlc, load_pattern_day
from data import raw_query
from finmind.stock_universe_2000 import load_stock_universe_2000
from finmind.tick_universe import load_tick_universe
from strategy.mkt.config import IDX_SYMBOL
from strategy.mkt.features import add_atr5
from strategy.ret5_pullback_ml.config import (
    ENTRY_DEADLINE,
    FIRST_M5_T,
    HOLD_M5_BARS,
    M1_REV_LOOKAHEAD,
    M1_SIG_UNTIL,
    M5_LOAD_UNTIL,
    RET5_MIN,
    SESSION_OPEN,
    SIGNAL_DEADLINE,
    SL_PCT,
    TP_PCT,
)
from strategy.ret5_pullback_ml.features import attach_entry_features, make_features


def _ym_from_month_path(path: str) -> str:
    return Path(path).stem.replace("_", "-")


def find_m5_down_m1_vol_brk(
    m5_day: pd.DataFrame,
    m1_day: pd.DataFrame,
    m5_1_low: float,
) -> dict | None:
    """回傳進場事件 dict，或 None。"""
    if m5_day.empty or m1_day.empty or not np.isfinite(m5_1_low):
        return None
    m5g = m5_day.sort_values("date").reset_index(drop=True)
    m1g = m1_day.sort_values("date").reset_index(drop=True)
    m1_o = m1g["open"].astype(float).values
    m1_c = m1g["close"].astype(float).values
    m1_v = m1g["volume"].astype(float).values
    m1_ts = m1g["date"].values

    for _, row in m5g.iterrows():
        ts_dn = pd.Timestamp(row["date"])
        if ts_dn.time() >= SIGNAL_DEADLINE:
            break
        o, c = float(row["open"]), float(row["close"])
        lo = float(row["low"])
        hi_dn = float(row["high"])
        if not (c < o and lo > m5_1_low):
            continue
        after = m1g["date"] > ts_dn
        idxs = np.flatnonzero(after.to_numpy())
        if len(idxs) == 0:
            continue
        idxs = idxs[:M1_REV_LOOKAHEAD]
        for j in idxs:
            t_ent = pd.Timestamp(m1_ts[j])
            if t_ent.time() >= ENTRY_DEADLINE:
                break
            if j == 0:
                continue
            if not (m1_c[j] > m1_o[j]):
                continue
            if not (m1_v[j] > m1_v[j - 1] and m1_v[j - 1] > 0):
                continue
            if not (m1_c[j] > hi_dn):
                continue
            entry = float(m1_c[j])
            return {
                "entry_ts": t_ent,
                "entry": entry,
                "m5_down_ts": ts_dn,
                "m5_dn_open": o,
                "m5_dn_high": hi_dn,
                "m5_dn_low": lo,
                "m5_dn_close": c,
                "m1_open": float(m1_o[j]),
                "m1_vol": float(m1_v[j]),
                "m1_prev_vol": float(m1_v[j - 1]),
                "m1_bars_after_dn": int(j - idxs[0] + 1),
            }
    return None


def long_tb_m5(
    day_m5: pd.DataFrame, entry_ts: pd.Timestamp, entry: float
) -> dict | None:
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


def build_events(
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    use_tick_universe: bool = False,
    attach_features: bool = True,
) -> pd.DataFrame:
    """建事件 + TB 標籤（可選進場特徵）。"""
    t0 = time.time()
    if use_tick_universe:
        raw = load_tick_universe()
        univ_label = "tick_universe"
    else:
        raw = load_stock_universe_2000()
        univ_label = "stock_universe_2000"
    universe = {str(s) for s in raw} | {IDX_SYMBOL}
    trade_universe = universe - {IDX_SYMBOL}

    print("ret5_pullback_ml build_events", flush=True)
    print(f"母體: {univ_label}（交易 {len(trade_universe)} 支）", flush=True)
    print(
        f"硬過濾: ret5紅K≥{RET5_MIN:.0%} → m5陰線 low>m5_1_low → "
        f"m1帶量陽破 m5 high；atr5 不當硬刪",
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
    bar905 = bar905[bar905["p905"].astype(float) > bar905["o905"].astype(float)].copy()
    cands = day_cands.merge(bar905, on=["stock_id", "day_str"], how="inner")
    cands["ret5_vs_prev"] = (cands["p905"].astype(float) / cands["prev_close"].astype(float)) - 1.0
    cands = cands[cands["ret5_vs_prev"] >= RET5_MIN].copy()
    n_ret5 = len(cands)
    print(f"漏斗 ret5 紅K≥{RET5_MIN:.0%}: {n_ret5:,}", flush=True)
    if cands.empty:
        return cands

    keys = cands[["stock_id", "day_str"]]
    m5_sig = m5[
        (m5["date"].dt.time >= FIRST_M5_T) & (m5["date"].dt.time < SIGNAL_DEADLINE)
    ].copy()
    m5_sig = m5_sig.merge(keys, on=["stock_id", "day_str"], how="inner")

    print(f"載入 m1（{SESSION_OPEN.strftime('%H:%M')}～{M1_SIG_UNTIL.strftime('%H:%M')}）...", flush=True)
    m1 = _load_bars_months(
        "db/m1",
        start_date,
        end_date,
        keys,
        SESSION_OPEN,
        M1_SIG_UNTIL,
        "m1",
    )
    if m1.empty or "volume" not in m1.columns:
        print("無 m1／缺 volume", flush=True)
        return pd.DataFrame()

    low_map = cands.set_index(["stock_id", "day_str"])["m5_1_low"].astype(float).to_dict()
    ret_map = cands.set_index(["stock_id", "day_str"])["ret5_vs_prev"].astype(float).to_dict()

    m5_by_sig = {
        k: g.reset_index(drop=True) for k, g in m5_sig.groupby(["stock_id", "day_str"], sort=False)
    }
    m1_by = {k: g.reset_index(drop=True) for k, g in m1.groupby(["stock_id", "day_str"], sort=False)}
    del m5_sig
    gc.collect()

    signal_rows: list[dict] = []
    for key, m5_day in m5_by_sig.items():
        sid, day_str = key
        m5_1_low = low_map.get(key)
        if m5_1_low is None:
            continue
        m1_day = m1_by.get(key)
        if m1_day is None:
            continue
        found = find_m5_down_m1_vol_brk(m5_day, m1_day, float(m5_1_low))
        if found is None:
            continue
        signal_rows.append(
            {
                "stock_id": sid,
                "day_str": day_str,
                "m5_1_low": float(m5_1_low),
                "ret5_vs_prev": float(ret_map[key]),
                **found,
            }
        )
    del m5_by_sig, m1_by
    gc.collect()
    print(f"漏斗 m5↓→m1帶量破高: {len(signal_rows):,}", flush=True)
    if not signal_rows:
        return pd.DataFrame()

    sig = pd.DataFrame(signal_rows)

    # atr5 + session/avwap 特徵（需完整 m1）
    print("算 atr5 / VWAP 特徵@進場...", flush=True)
    m1 = m1.copy()
    m1["day_date"] = m1["day_str"]
    m1 = add_atr5(m1)
    if attach_features:
        sig = attach_entry_features(sig, m1)
    else:
        atr = m1[["stock_id", "day_str", "date", "atr5"]].rename(columns={"date": "entry_ts"})
        sig = sig.merge(atr, on=["stock_id", "day_str", "entry_ts"], how="left")
    del m1
    gc.collect()

    m5_by = {
        k: g.reset_index(drop=True) for k, g in m5.groupby(["stock_id", "day_str"], sort=False)
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
        detail = long_tb_m5(day_m5, entry_ts, entry)
        if detail is None:
            continue
        label = float(detail["label"])
        # target: 0=止損 / 1=震盪 / 2=止盈
        if label > 0:
            target = 2
        elif label < 0:
            target = 0
        else:
            target = 1
        row = r.to_dict()
        row.update(
            {
                "label": label,
                "target": target,
                "exit_ts": detail["exit_ts"],
                "exit_price": detail["exit_price"],
                "exit_reason": detail["exit_reason"],
                "bars_held": detail["bars_held"],
                "pnl_pct": (detail["exit_price"] - entry) / entry,
                "trigger_ts": entry_ts,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    print(f"漏斗 可標籤: {len(out):,}", flush=True)
    if out.empty:
        print(f"耗時 {time.time() - t0:.1f}s", flush=True)
        return out

    n = len(out)
    print(
        f"基線全事件: TP={100 * (out['label'] == 1).mean():.1f}%  "
        f"flat={100 * (out['label'] == 0).mean():.1f}%  "
        f"SL={100 * (out['label'] == -1).mean():.1f}%  "
        f"mean={100 * out['pnl_pct'].mean():.3f}%",
        flush=True,
    )
    if attach_features:
        out = make_features(out)
        print(f"特徵完整樣本: {len(out):,} / {n:,}", flush=True)

    print(f"耗時 {time.time() - t0:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="ret5_pullback_ml build_events")
    p.add_argument("--start_date", default="2024-01-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument("--use_tick_universe", action="store_true")
    args = p.parse_args()
    df = build_events(
        start_date=args.start_date,
        end_date=args.end_date,
        use_tick_universe=args.use_tick_universe,
    )
    if not df.empty:
        print(df[["day_str", "stock_id", "entry_ts", "target", "pnl_pct"]].tail(10).to_string(index=False))
