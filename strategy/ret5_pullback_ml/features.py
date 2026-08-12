"""
ret5_pullback_ml 進場特徵：結構欄位 + session VWAP + 自 m5_down_ts 錨定的 AVWAP。

AVWAP：從 m5_down_ts **下一根** m1 起（date > m5_down_ts）累積至 entry_ts（含），
typical price 用 close（與 vwap_ml 一致）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.ret5_pullback_ml.config import AVWAP_Z_MIN_PERIODS

FEATURES = [
    "ret5_vs_prev",
    "dist_to_m5_1_low",
    "m5_dn_ret",
    "m5_dn_range_pct",
    "m1_vol_ratio",
    "breakout_pct",
    "m1_bars_after_dn",
    "entry_minute",
    "atr5",
    "close_vs_avwap",
    "avwap_z",
    "close_vs_session_vwap",
    "avwap_vs_session_vwap",
]


def _session_vwap_at_bars(m1: pd.DataFrame) -> pd.DataFrame:
    """在整段 m1 上算當日累積 session VWAP，回傳 stock_id/day_str/date/session_vwap。"""
    df = m1[["stock_id", "day_str", "date", "close", "volume"]].copy()
    df = df.sort_values(["stock_id", "day_str", "date"])
    df["_pv"] = df["close"].astype(float) * df["volume"].astype(float)
    g = df.groupby(["stock_id", "day_str"], sort=False)
    cum_pv = g["_pv"].cumsum()
    cum_v = g["volume"].cumsum().replace(0, np.nan)
    df["session_vwap"] = cum_pv / cum_v
    return df[["stock_id", "day_str", "date", "session_vwap"]]


def _avwap_at_entry(events: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    """對每筆事件算 entry 當下的 avwap / close_vs_avwap / avwap_z。"""
    if events.empty:
        return events

    m1s = m1[["stock_id", "day_str", "date", "close", "volume"]].copy()
    m1s["date"] = pd.to_datetime(m1s["date"])
    m1s = m1s.sort_values(["stock_id", "day_str", "date"]).reset_index(drop=True)

    avwap_vals = []
    close_vs = []
    avwap_zs = []

    # 依 (stock, day) group 加速
    m1_by = {k: g for k, g in m1s.groupby(["stock_id", "day_str"], sort=False)}

    for _, r in events.iterrows():
        key = (r["stock_id"], r["day_str"])
        day = m1_by.get(key)
        if day is None or len(day) == 0:
            avwap_vals.append(np.nan)
            close_vs.append(np.nan)
            avwap_zs.append(np.nan)
            continue
        m5_dn = pd.Timestamp(r["m5_down_ts"])
        entry_ts = pd.Timestamp(r["entry_ts"])
        entry = float(r["entry"])
        seg = day[(day["date"] > m5_dn) & (day["date"] <= entry_ts)].copy()
        if seg.empty:
            avwap_vals.append(np.nan)
            close_vs.append(np.nan)
            avwap_zs.append(np.nan)
            continue
        vol = seg["volume"].astype(float).values
        close = seg["close"].astype(float).values
        if vol.sum() <= 0:
            avwap_vals.append(np.nan)
            close_vs.append(np.nan)
            avwap_zs.append(np.nan)
            continue
        # expanding AVWAP along segment; take last (= at entry)
        cum_pv = np.cumsum(close * vol)
        cum_v = np.cumsum(vol)
        with np.errstate(divide="ignore", invalid="ignore"):
            avwap_path = cum_pv / np.where(cum_v > 0, cum_v, np.nan)
        avwap = float(avwap_path[-1])
        avwap_vals.append(avwap)
        close_vs.append((entry - avwap) / avwap if avwap > 0 else np.nan)
        # expanding std of (close - avwap_t)；不足 bar 或 std=0 時填 0（保留樣本）
        if len(avwap_path) < AVWAP_Z_MIN_PERIODS:
            avwap_zs.append(0.0)
        else:
            dev = close - avwap_path
            std = float(np.nanstd(dev, ddof=1))
            if std == 0 or not np.isfinite(std):
                avwap_zs.append(0.0)
            else:
                avwap_zs.append(float(dev[-1] / std))

    out = events.copy()
    out["avwap"] = avwap_vals
    out["close_vs_avwap"] = close_vs
    out["avwap_z"] = avwap_zs
    return out


def attach_entry_features(events: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    """在事件表上組齊結構 + atr5 + session/AVWAP 特徵。"""
    if events.empty:
        return events
    ev = events.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["m5_down_ts"] = pd.to_datetime(ev["m5_down_ts"])

    # 結構
    entry = ev["entry"].astype(float)
    m5_1_low = ev["m5_1_low"].astype(float)
    m5_hi = ev["m5_dn_high"].astype(float)
    ev["dist_to_m5_1_low"] = (entry - m5_1_low) / m5_1_low.replace(0, np.nan)
    m5_o = ev["m5_dn_open"].astype(float)
    ev["m5_dn_ret"] = (ev["m5_dn_close"].astype(float) - m5_o) / m5_o.replace(0, np.nan)
    ev["m5_dn_range_pct"] = (m5_hi - ev["m5_dn_low"].astype(float)) / m5_o.replace(0, np.nan)
    prev_v = ev["m1_prev_vol"].astype(float).replace(0, np.nan)
    ev["m1_vol_ratio"] = ev["m1_vol"].astype(float) / prev_v
    ev["breakout_pct"] = (entry - m5_hi) / m5_hi.replace(0, np.nan)
    ev["entry_minute"] = (
        ev["entry_ts"].dt.hour * 60 + ev["entry_ts"].dt.minute - (9 * 60)
    ).astype(float)

    # atr5 @ entry
    atr = m1[["stock_id", "day_str", "date", "atr5"]].copy()
    atr["date"] = pd.to_datetime(atr["date"])
    atr = atr.rename(columns={"date": "entry_ts"})
    ev = ev.merge(atr, on=["stock_id", "day_str", "entry_ts"], how="left")

    # session VWAP @ entry
    sv = _session_vwap_at_bars(m1)
    sv = sv.rename(columns={"date": "entry_ts"})
    sv["entry_ts"] = pd.to_datetime(sv["entry_ts"])
    ev = ev.merge(sv, on=["stock_id", "day_str", "entry_ts"], how="left")
    ev["close_vs_session_vwap"] = (entry - ev["session_vwap"]) / ev["session_vwap"].replace(
        0, np.nan
    )

    # AVWAP from m5_down
    ev = _avwap_at_entry(ev, m1)
    ev["avwap_vs_session_vwap"] = (ev["avwap"] - ev["session_vwap"]) / ev["session_vwap"].replace(
        0, np.nan
    )
    return ev


def make_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    missing = [c for c in FEATURES if c not in events.columns]
    if missing:
        raise KeyError(f"事件資料缺少特徵欄位: {missing}")
    return events.dropna(subset=FEATURES).reset_index(drop=True)
