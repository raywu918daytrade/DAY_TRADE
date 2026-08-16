"""VWAP 兩欄前端過濾用的每股活動度：日ATR%、開盤 5 分幅%、09:05 量 PR。

三個數都是相對量，高低價股同一套門檻。日 ATR = 過去 14 個交易日 TR 平均
（算到昨天）/ 今天開盤，算法同 strategy.orb.features.day_atr。
量 PR = 今日 09:00–09:05 volume vs 自己過去 20 個交易日同一段。
"""

from __future__ import annotations

import threading
from datetime import datetime, time as dtime, timedelta, timezone

import numpy as np
import pandas as pd

_TW = timezone(timedelta(hours=8))
_ENTRY = dtime(9, 5)
_LOOKBACK = 20
_MIN_HIST = 10
_ATR_N = 14

_cache: dict[str, dict[str, dict]] = {}
_lock = threading.Lock()


def _round_or_none(x, n: int):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), n)


def _day_atr14(day: pd.DataFrame) -> pd.DataFrame:
    """在日K上算 _atr14（含當列那天的 TR）；相對今開還要再 shift 到 D。"""
    day = day.sort_values(["stock_id", "date"]).copy()
    prev = day.groupby("stock_id", sort=False)["close"].shift(1)
    tr = np.maximum(
        np.maximum((day["high"] - day["low"]).abs(), (day["high"] - prev).abs()),
        (day["low"] - prev).abs(),
    )
    day["_atr14"] = tr.groupby(day["stock_id"]).transform(
        lambda s: s.rolling(_ATR_N, min_periods=_ATR_N).mean()
    )
    return day


def _open5_from_m1(date_str: str) -> pd.DataFrame:
    """盤中 m5_std 還沒今天 09:05 時，用 m1_live 09:01–09:05 合成一根。"""
    from data.query import load_m1_live

    m1 = load_m1_live(date_str)
    empty = pd.DataFrame(columns=["stock_id", "open", "high", "low", "volume"])
    if m1 is None or m1.empty:
        return empty
    m1 = m1.copy()
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    t = m1["date"].dt.time
    m1 = m1[(t >= dtime(9, 1)) & (t <= _ENTRY)]
    if m1.empty:
        return empty
    g = m1.sort_values("date").groupby("stock_id", sort=False)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "volume": g["volume"].sum(),
        }
    ).reset_index()
    out["stock_id"] = out["stock_id"].astype(str)
    return out


def _vol5_pr_map(
    m5_905: pd.DataFrame, date_str: str, extra_today: pd.DataFrame | None
) -> dict[str, float]:
    hist = m5_905[["stock_id", "day", "volume"]].copy() if not m5_905.empty else pd.DataFrame(
        columns=["stock_id", "day", "volume"]
    )
    if extra_today is not None and not extra_today.empty:
        extra = extra_today[["stock_id", "volume"]].copy()
        extra["day"] = date_str
        extra["stock_id"] = extra["stock_id"].astype(str)
        have = set()
        if not hist.empty:
            have = set(hist.loc[hist["day"] == date_str, "stock_id"].astype(str))
        extra = extra[~extra["stock_id"].isin(have)]
        hist = pd.concat([hist, extra[["stock_id", "day", "volume"]]], ignore_index=True)
    out: dict[str, float] = {}
    if hist.empty:
        return out
    hist["stock_id"] = hist["stock_id"].astype(str)
    for sid, g in hist.groupby("stock_id", sort=False):
        g = g.sort_values("day")
        today = g.loc[g["day"] == date_str, "volume"]
        if today.empty:
            continue
        past = g.loc[g["day"] < date_str, "volume"].tail(_LOOKBACK).to_numpy(dtype=float)
        cur = float(today.iloc[-1])
        if len(past) < _MIN_HIST or not np.isfinite(cur):
            continue
        out[str(sid)] = float(np.mean(past < cur))
    return out


def _as_open_map(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    d = df.drop_duplicates("stock_id").copy()
    d["stock_id"] = d["stock_id"].astype(str)
    return d.set_index("stock_id")["open"].astype(float)


def _compute(date_str: str) -> dict[str, dict]:
    from data.adjustment_query import load_pattern_day
    from data.query import load_m5_std

    hist_start = (pd.Timestamp(date_str) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    end_excl = pd.Timestamp(date_str) + pd.Timedelta(days=1)

    atr_asof = pd.Series(dtype=float)
    open_d = pd.Series(dtype=float)
    day = load_pattern_day(start_date=hist_start, end_date=date_str)
    if not day.empty:
        day = day.copy()
        day["stock_id"] = day["stock_id"].astype(str)
        day["date"] = pd.to_datetime(day["date"], format="mixed")
        day["day"] = day["date"].dt.strftime("%Y-%m-%d")
        day = day[(day["date"] < end_excl) & (day["open"] > 0)]
        if not day.empty:
            day = _day_atr14(day)
            before = day[day["day"] < date_str]
            if not before.empty:
                atr_asof = before.sort_values("date").groupby("stock_id")["_atr14"].last()
                atr_asof.index = atr_asof.index.astype(str)
            today_day = day[day["day"] == date_str]
            if not today_day.empty:
                open_d = _as_open_map(today_day)

    m5 = load_m5_std(start_date=hist_start)
    m5_905 = pd.DataFrame(columns=["stock_id", "date", "day", "open", "high", "low", "volume"])
    if m5 is not None and not m5.empty:
        m5 = m5.copy()
        m5["stock_id"] = m5["stock_id"].astype(str)
        m5["date"] = pd.to_datetime(m5["date"], format="mixed")
        m5 = m5[
            (m5["date"] < end_excl)
            & (m5["date"].dt.hour == 9)
            & (m5["date"].dt.minute == 5)
        ]
        if not m5.empty:
            m5["day"] = m5["date"].dt.strftime("%Y-%m-%d")
            m5_905 = m5.drop_duplicates(["stock_id", "day"], keep="last")

    extra = _open5_from_m1(date_str)
    today_m5 = m5_905[m5_905["day"] == date_str] if not m5_905.empty else m5_905
    if today_m5 is not None and not today_m5.empty:
        today_bar = today_m5[["stock_id", "open", "high", "low", "volume"]].copy()
        extra_for_pr = None
        if extra is not None and not extra.empty:
            have = set(today_bar["stock_id"].astype(str))
            add = extra[~extra["stock_id"].astype(str).isin(have)]
            if not add.empty:
                today_bar = pd.concat(
                    [today_bar, add[["stock_id", "open", "high", "low", "volume"]]],
                    ignore_index=True,
                )
                extra_for_pr = add
    else:
        today_bar = extra
        extra_for_pr = extra

    m5_open = _as_open_map(today_bar)
    if open_d.empty:
        open_d = m5_open
    else:
        fill = m5_open.loc[~m5_open.index.isin(open_d.index)]
        if not fill.empty:
            open_d = pd.concat([open_d, fill])

    open5: dict[str, float] = {}
    if today_bar is not None and not today_bar.empty:
        tb = today_bar.copy()
        tb["stock_id"] = tb["stock_id"].astype(str)
        rng = (tb["high"].astype(float) - tb["low"].astype(float)) / tb["open"].replace(0, np.nan)
        for sid, val in zip(tb["stock_id"], rng):
            if np.isfinite(val):
                open5[str(sid)] = float(val)

    vol_pr = _vol5_pr_map(m5_905, date_str, extra_for_pr)

    day_atr: dict[str, float] = {}
    if not atr_asof.empty and not open_d.empty:
        a, o = atr_asof.align(open_d, join="inner")
        pct = a / o.replace(0, np.nan)
        for sid, val in pct.items():
            if np.isfinite(val):
                day_atr[str(sid)] = float(val)

    out: dict[str, dict] = {}
    for sid in set(day_atr) | set(open5) | set(vol_pr):
        out[sid] = {
            "day_atr": _round_or_none(day_atr.get(sid), 5),
            "open5_rng": _round_or_none(open5.get(sid), 5),
            "vol5_pr": _round_or_none(vol_pr.get(sid), 4),
        }
    return out


def metrics_for_date(date_str: str) -> dict[str, dict]:
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    with _lock:
        if date_str != today and date_str in _cache:
            return _cache[date_str]
    result = _compute(date_str)
    if date_str != today:
        with _lock:
            _cache[date_str] = result
    return result
