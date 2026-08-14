"""盤後一次掃全日 m1，還原 VWAP突破 / VWAP+壓力支撐（規則同 live_trader.on_minute）。

不模擬時鐘；db/m1_live/{date}.parquet 本來就留檔。盤中仍走 on_minute 增量推
SSE，這裡只給 GET ?date= 用。
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from pattern.envelope import envelope_sr_prices

_scan_cache: dict[str, tuple[list, list]] = {}
_sr_levels_cache: dict[str, dict[str, tuple[float, float]]] = {}
_names_cache: dict[str, str] | None = None
_scan_lock = threading.Lock()


def _hhmm(ts) -> str:
    t = pd.Timestamp(ts)
    return f"{t.hour:02d}:{t.minute:02d}"


def _name_map() -> dict[str, str]:
    global _names_cache
    if _names_cache is not None:
        return _names_cache
    try:
        from fubon.intraday_tickers import load_tickers

        df = load_tickers()
        _names_cache = dict(
            zip(df["stock_id"].astype(str), df["name"].astype(str))
        )
    except FileNotFoundError:
        _names_cache = {}
    return _names_cache


def load_day_m1(date_str: str) -> pd.DataFrame:
    """優先 db/m1_live/{date}；沒檔再 load_pattern_m1 濾那一天。"""
    from data.query import load_m1_live

    df = load_m1_live(date_str)
    if not df.empty:
        df = df.copy()
        df["stock_id"] = df["stock_id"].astype(str)
        return df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    from data.adjustment_query import load_pattern_m1

    m1 = load_pattern_m1(start_date=date_str, end_date=date_str)
    if m1.empty:
        return m1
    m1 = m1.copy()
    m1["stock_id"] = m1["stock_id"].astype(str)
    start = pd.Timestamp(date_str)
    end = start + pd.Timedelta(days=1)
    m1 = m1[(m1["date"] >= start) & (m1["date"] < end)]
    return m1.sort_values(["stock_id", "date"]).reset_index(drop=True)


def sr_levels_for_date(
    date_str: str, stocks: set[str]
) -> dict[str, tuple[float, float]]:
    """D 之前日K 包絡壓力／支撐，與 live_trader._refresh_sr_levels 相同。"""
    if date_str in _sr_levels_cache:
        cached = _sr_levels_cache[date_str]
        return {sid: cached[sid] for sid in stocks if sid in cached}

    from data.adjustment_query import load_pattern_day

    if not stocks:
        _sr_levels_cache[date_str] = {}
        return {}
    hist_start = (pd.Timestamp(date_str) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    day = load_pattern_day(start_date=hist_start, end_date=date_str)
    if day.empty:
        _sr_levels_cache[date_str] = {}
        return {}
    day = day.copy()
    day["stock_id"] = day["stock_id"].astype(str)
    cutoff = pd.Timestamp(date_str)
    levels: dict[str, tuple[float, float]] = {}
    for sid, g in day.groupby("stock_id", sort=False):
        sid = str(sid)
        hist = g.loc[g["date"] < cutoff]
        res, sup = envelope_sr_prices(hist)
        if res is not None:
            levels[sid] = (res, sup)
    _sr_levels_cache[date_str] = levels
    return {sid: levels[sid] for sid in stocks if sid in levels}


def scan_day_events(
    m1: pd.DataFrame,
    sr_levels: dict[str, tuple[float, float]],
    names: dict[str, str] | None = None,
) -> tuple[list, list]:
    """回傳 (vwap_breakouts, sr_vwap_hits)，欄位與盤中 push payload + time 相同。"""
    names = names or {}
    vwap_events: list = []
    sr_events: list = []
    if m1 is None or m1.empty:
        return vwap_events, sr_events

    for sid, g in m1.groupby("stock_id", sort=False):
        sid = str(sid)
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        closes = g["close"].astype(float).to_numpy()
        vols = g["volume"].astype(float).to_numpy()
        times = g["date"].to_numpy()
        cum_vol = vols.cumsum()
        cum_pv = (closes * vols).cumsum()
        vwap = np.divide(
            cum_pv,
            cum_vol,
            out=np.full_like(cum_pv, np.nan),
            where=cum_vol > 0,
        )
        valid = np.isfinite(vwap)
        above = closes >= vwap
        flip = np.zeros(len(closes), dtype=bool)
        flip[1:] = (above[1:] != above[:-1]) & valid[1:] & valid[:-1]
        if not flip.any():
            continue

        name = names.get(sid, sid)
        vwap_idxs = np.flatnonzero(flip)
        for i in vwap_idxs:
            vwap_events.append(
                {
                    "stock_id": sid,
                    "name": name,
                    "direction": "up" if above[i] else "down",
                    "price": round(float(closes[i]), 2),
                    "vwap": round(float(vwap[i]), 2),
                    "time": _hhmm(times[i]),
                }
            )

        sr = sr_levels.get(sid)
        if not sr:
            continue
        res, sup = sr
        res_flip = np.zeros(len(closes), dtype=bool)
        res_flip[1:] = (closes[1:] >= res) != (closes[:-1] >= res)
        sup_flip = np.zeros(len(closes), dtype=bool)
        sup_flip[1:] = (closes[1:] >= sup) != (closes[:-1] >= sup)
        sr_flip = res_flip | sup_flip
        if not sr_flip.any():
            continue
        first_vwap = int(vwap_idxs[0])
        first_sr = int(np.flatnonzero(sr_flip)[0])
        fire = max(first_vwap, first_sr)
        res_x = bool(res_flip[: fire + 1].any())
        sup_x = bool(sup_flip[: fire + 1].any())
        if res_x and sup_x:
            sr_kind = "both"
        elif sup_x:
            sr_kind = "support"
        else:
            sr_kind = "resistance"
        sr_events.append(
            {
                "stock_id": sid,
                "name": name,
                "sr_kind": sr_kind,
                "vwap_dir": "up" if closes[fire] >= vwap[fire] else "down",
                "price": round(float(closes[fire]), 2),
                "vwap": round(float(vwap[fire]), 2),
                "resistance": round(float(res), 2),
                "support": round(float(sup), 2),
                "time": _hhmm(times[fire]),
            }
        )

    return vwap_events, sr_events


def scan_date(date_str: str) -> tuple[list, list]:
    """掃某一曆日，回傳 (vwap_breakouts, sr_vwap_hits)。過去日期 cache 事件；
    今日每次重算（m1_live 可能還在寫），SR 水位仍 cache。"""
    today = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
    with _scan_lock:
        if date_str != today and date_str in _scan_cache:
            return _scan_cache[date_str]

        m1 = load_day_m1(date_str)
        stocks = set(m1["stock_id"].astype(str)) if not m1.empty else set()
        levels = sr_levels_for_date(date_str, stocks)
        result = scan_day_events(m1, levels, _name_map())
        if date_str != today:
            _scan_cache[date_str] = result
        return result
