"""
事件母體：前日實體漲停 → 今開高 → 首根 m3_std 下跌。
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.adjustment_query import _adjust_ohlc, load_pattern_day
from data.raw_query import iter_m1_months, iter_m3_std_months
from strategy.limitup_fade_ml_my.config import (
    FIRST_M3_TIME,
    FORCE_EXIT_TIME,
    IDX_SYMBOL,
    LIMIT_UP_RET,
    MAX_UPPER,
    MIN_BODY,
    SL_PCT,
    TP_PCT,
)


def _is_stock_id(sid: str) -> bool:
    s = str(sid)
    return len(s) == 4 and s.isdigit() and not s.startswith("00")


def short_triple_barrier_detail(
    m1_day: pd.DataFrame,
    trigger_ts: pd.Timestamp,
    entry: float,
) -> dict | None:
    """做空 TB 明細：label (+1/0/-1)、exit_ts、exit_price、exit_reason、bars_held。"""
    if m1_day is None or m1_day.empty or entry <= 0:
        return None
    m1 = m1_day.copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    day = pd.Timestamp(trigger_ts).normalize()
    exit_ts = day + pd.Timedelta(hours=FORCE_EXIT_TIME.hour, minutes=FORCE_EXIT_TIME.minute)
    fut = m1[(m1["date"] > trigger_ts) & (m1["date"] <= exit_ts)].sort_values("date")
    if fut.empty:
        return None
    tp = entry * (1.0 - TP_PCT)
    sl = entry * (1.0 + SL_PCT)
    for i, (_, row) in enumerate(fut.iterrows(), start=1):
        ts = pd.Timestamp(row["date"])
        if float(row["low"]) <= tp:
            return {
                "label": 1.0,
                "exit_ts": ts,
                "exit_price": tp,
                "exit_reason": "tp",
                "bars_held": i,
            }
        if float(row["high"]) >= sl:
            return {
                "label": -1.0,
                "exit_ts": ts,
                "exit_price": sl,
                "exit_reason": "sl",
                "bars_held": i,
            }
    if fut["date"].iloc[-1].time() < FORCE_EXIT_TIME:
        return None
    last = fut.iloc[-1]
    return {
        "label": 0.0,
        "exit_ts": pd.Timestamp(last["date"]),
        "exit_price": float(last["close"]),
        "exit_reason": "time",
        "bars_held": len(fut),
    }


def short_triple_barrier_label(
    m1_day: pd.DataFrame,
    trigger_ts: pd.Timestamp,
    entry: float,
) -> float:
    """做空 TB：+1 先觸 TP（價跌）、-1 先觸 SL（價漲）、0 時間牆；不足 → NaN。"""
    detail = short_triple_barrier_detail(m1_day, trigger_ts, entry)
    return float("nan") if detail is None else detail["label"]


def build_gap_candidates(start_date: str, end_date: str) -> pd.DataFrame:
    """前日實體漲停且今日開高的日 K 候選（尚未套首 3 分條件）。"""
    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    day_all = load_pattern_day(start_date=hist)
    day_all["stock_id"] = day_all["stock_id"].astype(str)
    day_all["date"] = pd.to_datetime(day_all["date"], format="mixed")
    day = day_all[day_all["stock_id"].map(_is_stock_id)].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)

    g = day.groupby("stock_id", sort=False)
    day["prev_close"] = g["close"].shift(1)
    day["prev_open"] = g["open"].shift(1)
    day["prev_high"] = g["high"].shift(1)
    day["prev_low"] = g["low"].shift(1)
    day["prev_volume"] = g["volume"].shift(1) if "volume" in day.columns else np.nan
    vol20 = (
        g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
        if "volume" in day.columns
        else np.nan
    )
    day["prev_volume_avg20"] = vol20

    po = day["prev_open"].astype(float)
    ph = day["prev_high"].astype(float)
    pl = day["prev_low"].astype(float)
    pc = day["prev_close"].astype(float)
    ppc = g["close"].shift(2)
    prev_ret = pc / ppc - 1.0
    rng = (ph - pl).replace(0, np.nan)
    body = (pc - po) / rng
    upper = (ph - pc) / rng
    prev_limit_solid = (prev_ret >= LIMIT_UP_RET) & (pc > po) & (body >= MIN_BODY) & (upper <= MAX_UPPER)
    gap_up = day["open"].astype(float) > pc

    cands = day[(day["date"] >= start_date) & (day["date"] <= end_date) & prev_limit_solid & gap_up].copy()
    cands["day_str"] = cands["date"].dt.strftime("%Y-%m-%d")
    cands["prev_ret"] = prev_ret.loc[cands.index]
    cands["prev_body_ratio"] = body.loc[cands.index]
    cands["prev_upper_ratio"] = upper.loc[cands.index]
    cands["gap_pct"] = cands["open"].astype(float) / pc.loc[cands.index] - 1.0
    cands["open_vs_prev_high"] = cands["open"].astype(float) / ph.loc[cands.index] - 1.0
    pv = cands["prev_volume"].astype(float)
    pva = cands["prev_volume_avg20"].astype(float).replace(0, np.nan)
    cands["prev_volume_z"] = (pv - pva) / pva.replace(0, np.nan)

    # 0050 缺口（代號 00xx，不在個股過濾內）
    day_idx = day_all[day_all["stock_id"] == IDX_SYMBOL].sort_values("date").copy()
    if not day_idx.empty:
        day_idx["prev_close"] = day_idx["close"].shift(1)
        day_idx["day_str"] = day_idx["date"].dt.strftime("%Y-%m-%d")
        day_idx["idx_gap_pct"] = day_idx["open"].astype(float) / day_idx["prev_close"].astype(float) - 1.0
        cands = cands.merge(day_idx[["day_str", "idx_gap_pct"]], on="day_str", how="left")
        cands["gap_vs_0050"] = cands["gap_pct"] - cands["idx_gap_pct"].fillna(0.0)
    else:
        cands["gap_vs_0050"] = cands["gap_pct"]

    return cands.reset_index(drop=True)


def attach_first_m3(cands: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """合併首根 m3_std（逐月載入），只保留 close < open。"""
    if cands.empty:
        return cands
    need_sids = set(cands["stock_id"].astype(str))
    need_days = set(cands["day_str"])
    parts: list[pd.DataFrame] = []
    for i, m3_raw in enumerate(iter_m3_std_months(start_date=start_date), start=1):
        m3_raw = m3_raw[m3_raw["stock_id"].astype(str).isin(need_sids)].copy()
        if m3_raw.empty:
            continue
        m3 = _adjust_ohlc(m3_raw, start_date)
        m3["stock_id"] = m3["stock_id"].astype(str)
        m3["date"] = pd.to_datetime(m3["date"], format="mixed")
        m3 = m3[(m3["date"].dt.time == FIRST_M3_TIME) & (m3["date"].dt.strftime("%Y-%m-%d").isin(need_days))]
        if m3.empty:
            continue
        m3 = m3[["stock_id", "date", "open", "close", "high", "low"]].copy()
        parts.append(m3)
        if i % 6 == 0:
            print(f"    m3_std 已處理 {i} 個月…", flush=True)

    if not parts:
        return cands.iloc[0:0].copy()
    m3 = pd.concat(parts, ignore_index=True)
    m3["day_str"] = m3["date"].dt.strftime("%Y-%m-%d")
    m3 = m3.drop_duplicates(subset=["stock_id", "day_str"], keep="last")
    first = m3.set_index(["stock_id", "day_str"])[["open", "close", "high", "low", "date"]]
    first.columns = ["m3_open", "m3_close", "m3_high", "m3_low", "trigger_ts"]

    ev = cands.merge(first, left_on=["stock_id", "day_str"], right_index=True, how="inner")
    ev = ev[ev["m3_close"].astype(float) < ev["m3_open"].astype(float)].copy()
    return ev.reset_index(drop=True)


def _label_events_monthly(ev: pd.DataFrame, start_date: str) -> pd.Series:
    """逐月載入 pattern M1 打做空 TB 標籤，峰值≈單月 M1。"""
    n = len(ev)
    labels = np.full(n, np.nan, dtype=float)
    key_to_idx = {(str(r.stock_id), r.day_str): i for i, r in enumerate(ev.itertuples(index=False))}
    need_sids = {k[0] for k in key_to_idx}
    need_days = {k[1] for k in key_to_idx}
    pending = set(key_to_idx)

    for i, m1_raw in enumerate(iter_m1_months(start_date=start_date), start=1):
        if not pending:
            break
        m1_raw = m1_raw[m1_raw["stock_id"].astype(str).isin(need_sids)].copy()
        if m1_raw.empty:
            continue
        m1 = _adjust_ohlc(m1_raw, start_date)
        m1["stock_id"] = m1["stock_id"].astype(str)
        m1["date"] = pd.to_datetime(m1["date"], format="mixed")
        m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
        m1 = m1[m1["day_str"].isin(need_days)]
        if m1.empty:
            continue

        month_keys = {(sid, day) for sid, day in zip(m1["stock_id"], m1["day_str"]) if (sid, day) in pending}
        if not month_keys:
            continue
        grouped = {k: v for k, v in m1.groupby(["stock_id", "day_str"], sort=False) if k in month_keys}
        for key in month_keys:
            idx = key_to_idx[key]
            r = ev.iloc[idx]
            labels[idx] = short_triple_barrier_label(
                grouped.get(key),
                pd.Timestamp(r["trigger_ts"]),
                float(r["entry_price"]),
            )
            pending.discard(key)
        if i % 3 == 0 or not pending:
            print(
                f"    M1 標籤 {i} 個月… 已標 {n - len(pending)}/{n}",
                flush=True,
            )

    if pending:
        print(f"    警告：{len(pending)} 筆找不到 M1，標籤留 NaN", flush=True)
    return pd.Series(labels, index=ev.index)


def build_events(
    start_date: str,
    end_date: str,
    *,
    with_labels: bool = True,
) -> pd.DataFrame:
    """完整事件：開高候選 ∩ 首 3 分跌（+ 可選 TB 標籤）。"""
    print(f"建事件母體 {start_date} ~ {end_date} ...", flush=True)
    cands = build_gap_candidates(start_date, end_date)
    print(f"  開高候選: {len(cands):,}", flush=True)
    ev = attach_first_m3(cands, start_date=start_date)
    print(f"  首3分跌: {len(ev):,}", flush=True)
    if ev.empty:
        return ev

    o = ev["m3_open"].astype(float)
    c = ev["m3_close"].astype(float)
    h = ev["m3_high"].astype(float)
    l = ev["m3_low"].astype(float)
    rng = (h - l).replace(0, np.nan)
    ev["m3_ret"] = c / o - 1.0
    ev["m3_body_ratio"] = (o - c) / rng  # 陰線實體
    ev["m3_upper_ratio"] = (h - o) / rng
    ev["m3_lower_ratio"] = (c - l) / rng
    ev["m3_range_pct"] = rng / o
    day_open = ev["open"].astype(float)
    ev["m3_close_vs_open"] = c / day_open - 1.0
    ev["entry_price"] = c
    ev["day_close"] = ev["close"].astype(float)
    ev["short_ret_to_close"] = (ev["entry_price"] - ev["day_close"]) / ev["entry_price"]
    ev["short_win_close"] = ev["day_close"] < ev["entry_price"]

    if not with_labels:
        return ev

    print("  標籤 TB（做空，逐月 M1）...", flush=True)
    ev["label_raw"] = _label_events_monthly(ev, start_date=start_date)
    ev = ev[ev["label_raw"].notna()].copy()
    ev["target"] = ev["label_raw"].map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    print(
        f"  有標籤: {len(ev):,}  " f"分布%={(ev['target'].value_counts(normalize=True) * 100).round(1).to_dict()}",
        flush=True,
    )
    return ev.reset_index(drop=True)
