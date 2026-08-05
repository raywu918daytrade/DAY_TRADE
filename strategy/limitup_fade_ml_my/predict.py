"""
即時／批次推論。

predict_live：09:03 分鐘，對「昨漲停＋今開高」候選檢查首 3 分陰線後輸出做空訊號。
"""

from __future__ import annotations

import sys
from datetime import time as dtime
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.adjustment_query import load_pattern_day
from data.query import load_m1_live
from strategy.limitup_fade_ml_my.config import (
    FIRST_M3_TIME,
    IDX_SYMBOL,
    LIMIT_UP_RET,
    MAX_UPPER,
    MIN_BODY,
    MODEL_TYPE,
    THRESHOLD,
)
from strategy.limitup_fade_ml_my.features import FEATURES
from strategy.limitup_fade_ml_my.train import load_model_by_type

_live_cand_cache: dict[str, pd.DataFrame] = {}


def _is_stock_id(sid: str) -> bool:
    s = str(sid)
    return len(s) == 4 and s.isdigit() and not s.startswith("00")


def _agg_first_m3(m1_day: pd.DataFrame) -> dict | None:
    """09:00～09:02 三根 M1 → 首 3 分（對齊 m3_std@09:03）。"""
    if m1_day is None or m1_day.empty:
        return None
    m1 = m1_day.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    w = m1[(m1["date"].dt.time >= dtime(9, 0)) & (m1["date"].dt.time <= dtime(9, 2))]
    if len(w) < 3:
        return None
    w = w.iloc[:3]
    deltas = w["date"].diff().dt.total_seconds().iloc[1:]
    if not (deltas == 60).all():
        return None
    o = float(w.iloc[0]["open"])
    c = float(w.iloc[-1]["close"])
    h = float(w["high"].astype(float).max())
    l = float(w["low"].astype(float).min())
    if not all(np.isfinite([o, h, l, c])) or o <= 0:
        return None
    rng = h - l
    return {
        "m3_open": o,
        "m3_close": c,
        "m3_high": h,
        "m3_low": l,
        "m3_ret": c / o - 1.0,
        "m3_body_ratio": (o - c) / rng if rng > 0 else 0.0,
        "m3_upper_ratio": (h - o) / rng if rng > 0 else 0.0,
        "m3_lower_ratio": (c - l) / rng if rng > 0 else 0.0,
        "m3_range_pct": rng / o,
        "bear": c < o,
    }


def _load_live_gap_candidates(trade_date: str) -> pd.DataFrame:
    if trade_date in _live_cand_cache:
        return _live_cand_cache[trade_date]

    hist = (pd.Timestamp(trade_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    day = load_pattern_day(start_date=hist)
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    stocks = day[day["stock_id"].map(_is_stock_id)].sort_values(["stock_id", "date"])
    g = stocks.groupby("stock_id", sort=False)
    stocks = stocks.copy()
    stocks["prev_close"] = g["close"].shift(1)
    stocks["prev_open"] = g["open"].shift(1)
    stocks["prev_high"] = g["high"].shift(1)
    stocks["prev_low"] = g["low"].shift(1)
    stocks["prev_volume"] = g["volume"].shift(1) if "volume" in stocks.columns else np.nan
    stocks["prev_volume_avg20"] = (
        g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
        if "volume" in stocks.columns
        else np.nan
    )
    stocks["prev2_close"] = g["close"].shift(2)
    stocks["prev_ret"] = stocks["prev_close"].astype(float) / stocks["prev2_close"].astype(float) - 1.0
    po = stocks["prev_open"].astype(float)
    ph = stocks["prev_high"].astype(float)
    pl = stocks["prev_low"].astype(float)
    pc = stocks["prev_close"].astype(float)
    rng = (ph - pl).replace(0, np.nan)
    stocks["prev_body_ratio"] = (pc - po) / rng
    stocks["prev_upper_ratio"] = (ph - pc) / rng

    stocks["_day"] = stocks["date"].dt.strftime("%Y-%m-%d")
    today = stocks[stocks["_day"] == trade_date].copy()
    if today.empty:
        _live_cand_cache[trade_date] = today
        return today

    pc_t = today["prev_close"].astype(float)
    prev_limit = (
        (today["prev_ret"] >= LIMIT_UP_RET)
        & (pc_t > today["prev_open"].astype(float))
        & (today["prev_body_ratio"] >= MIN_BODY)
        & (today["prev_upper_ratio"] <= MAX_UPPER)
    )
    gap_up = today["open"].astype(float) > pc_t
    cands = today[prev_limit & gap_up].copy()
    cands["gap_pct"] = cands["open"].astype(float) / pc_t.loc[cands.index] - 1.0
    cands["open_vs_prev_high"] = cands["open"].astype(float) / cands["prev_high"].astype(float) - 1.0
    pv = cands["prev_volume"].astype(float)
    pva = cands["prev_volume_avg20"].astype(float).replace(0, np.nan)
    cands["prev_volume_z"] = (pv - pva) / pva

    idx = day[day["stock_id"] == IDX_SYMBOL].sort_values("date")
    idx_gap = 0.0
    if not idx.empty:
        idx = idx.copy()
        idx["prev_close"] = idx["close"].shift(1)
        row = idx[idx["date"].dt.strftime("%Y-%m-%d") == trade_date]
        if len(row) and pd.notna(row.iloc[0]["prev_close"]):
            idx_gap = float(row.iloc[0]["open"]) / float(row.iloc[0]["prev_close"]) - 1.0
    cands["gap_vs_0050"] = cands["gap_pct"] - idx_gap

    print(f"[limitup_fade_ml live] {trade_date} 開高候選 {len(cands)}", flush=True)
    _live_cand_cache[trade_date] = cands
    return cands


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    model=None,
    threshold: float = THRESHOLD,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
    **kwargs,
) -> list:
    """回傳 [{"stock_id", "proba", "price", "direction": "down"}, ...]。"""
    _ = day, kwargs
    ts = pd.Timestamp(minute_str)
    if ts.time() != FIRST_M3_TIME:
        return []

    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    cands = _load_live_gap_candidates(date_str)
    if cands.empty:
        return []
    if day_trade_stocks:
        cands = cands[cands["stock_id"].isin(set(map(str, day_trade_stocks)))]

    m1_live = m1_live.copy()
    m1_live["stock_id"] = m1_live["stock_id"].astype(str)
    m1_live["date"] = pd.to_datetime(m1_live["date"], format="mixed")

    rows = []
    feat_rows = []
    meta = []
    for r in cands.itertuples(index=False):
        sid = str(r.stock_id)
        m3 = _agg_first_m3(m1_live[m1_live["stock_id"] == sid])
        if m3 is None or not m3["bear"]:
            continue
        day_open = float(r.open)
        feat = {
            "gap_pct": float(r.gap_pct),
            "m3_ret": m3["m3_ret"],
            "m3_body_ratio": m3["m3_body_ratio"],
            "m3_upper_ratio": m3["m3_upper_ratio"],
            "m3_lower_ratio": m3["m3_lower_ratio"],
            "m3_range_pct": m3["m3_range_pct"],
            "prev_ret": float(r.prev_ret),
            "prev_body_ratio": float(r.prev_body_ratio),
            "prev_upper_ratio": float(r.prev_upper_ratio),
            "prev_volume_z": float(r.prev_volume_z) if pd.notna(r.prev_volume_z) else 0.0,
            "open_vs_prev_high": float(r.open_vs_prev_high),
            "m3_close_vs_open": m3["m3_close"] / day_open - 1.0 if day_open > 0 else 0.0,
            "gap_vs_0050": float(r.gap_vs_0050),
        }
        feat_rows.append(feat)
        meta.append((sid, m3["m3_close"]))

    if not feat_rows:
        return []

    X = pd.DataFrame(feat_rows)[FEATURES]
    proba = model.predict_proba(X)
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    p_tp = proba[:, class_idx[2]] if 2 in class_idx else np.zeros(len(X))

    for (sid, price), p in zip(meta, p_tp):
        if p < threshold:
            continue
        rows.append(
            {
                "stock_id": sid,
                "proba": float(p),
                "price": float(price),
                "direction": "down",
            }
        )
    return rows
