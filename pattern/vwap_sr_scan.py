"""盤後一次掃全日 m1，還原 VWAP突破 / VWAP+壓力支撐（規則同 live_trader.on_minute）。

不模擬時鐘；db/m1_live/{date}.parquet 本來就留檔。盤中仍走 on_minute 增量推
SSE，這裡只給 GET ?date= 用。
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from pattern.horizontal_sr import horizontal_sr_prices

_scan_cache: dict[tuple[str, str], tuple[list, list]] = {}
_sr_levels_cache: dict[str, dict[str, tuple[float | None, float | None]]] = {}
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


def normalize_universe(universe: str | None) -> str:
    return "full" if str(universe or "").strip().lower() == "full" else "tick"


def _from_m1_live(date_str: str) -> pd.DataFrame:
    from data.query import load_m1_live

    df = load_m1_live(date_str)
    if df.empty:
        return df
    df = df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _from_pattern_m1(date_str: str) -> pd.DataFrame:
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


def load_day_m1(date_str: str, universe: str = "tick") -> pd.DataFrame:
    """tick：優先 db/m1_live（盤中訂閱 ~400 檔）。
    full：走 db/m1 全市場；沒檔才退回 m1_live（重現才有多出來的列）。"""
    universe = normalize_universe(universe)
    if universe == "full":
        m1 = _from_pattern_m1(date_str)
        if not m1.empty:
            return m1
        return _from_m1_live(date_str)

    live = _from_m1_live(date_str)
    if not live.empty:
        return live
    return _from_pattern_m1(date_str)


def sr_levels_for_date(
    date_str: str, stocks: set[str]
) -> dict[str, tuple[float | None, float | None]]:
    """D 之前日K 橫向壓力／支撐，與 live_trader._refresh_sr_levels 相同。
    cache 以日期為鍵；缺的股票再補算（先掃當沖再掃全市場時會用到）。"""
    stocks = {str(s) for s in stocks}
    cached = _sr_levels_cache.get(date_str, {})
    missing = stocks - set(cached)
    if not missing:
        return {sid: cached[sid] for sid in stocks if sid in cached}

    from data.adjustment_query import load_pattern_day

    if not stocks:
        _sr_levels_cache[date_str] = cached
        return {}
    hist_start = (pd.Timestamp(date_str) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    day = load_pattern_day(start_date=hist_start, end_date=date_str)
    if day.empty:
        _sr_levels_cache[date_str] = cached
        return {sid: cached[sid] for sid in stocks if sid in cached}
    day = day.copy()
    day["stock_id"] = day["stock_id"].astype(str)
    cutoff = pd.Timestamp(date_str)
    levels = dict(cached)
    for sid, g in day.groupby("stock_id", sort=False):
        sid = str(sid)
        if sid not in missing:
            continue
        hist = g.loc[g["date"] < cutoff]
        res, sup = horizontal_sr_prices(hist)
        if res is not None or sup is not None:
            levels[sid] = (res, sup)
    _sr_levels_cache[date_str] = levels
    return {sid: levels[sid] for sid in stocks if sid in levels}


def sr_record_indices(vwap_flip: np.ndarray, sr_flip: np.ndarray) -> list[int]:
    """同一支可以重複：壓力/支撐這一根變號且當天已有 VWAP 變號；
    或這一根才第一次跨 VWAP、但稍早已穿越過壓力/支撐。"""
    out: list[int] = []
    vwap_seen = False
    sr_seen = False
    for i in range(1, len(vwap_flip)):
        v_now = bool(vwap_flip[i])
        r_now = bool(sr_flip[i])
        if r_now and (vwap_seen or v_now):
            out.append(i)
        elif v_now and (not vwap_seen) and sr_seen:
            out.append(i)
        if v_now:
            vwap_seen = True
        if r_now:
            sr_seen = True
    return out


def _sr_kind_at(i: int, res_flip: np.ndarray, sup_flip: np.ndarray) -> str:
    res_now = bool(res_flip[i]) if i < len(res_flip) else False
    sup_now = bool(sup_flip[i]) if i < len(sup_flip) else False
    if res_now and sup_now:
        return "both"
    if res_now:
        return "resistance"
    if sup_now:
        return "support"
    res_h = bool(res_flip[: i + 1].any())
    sup_h = bool(sup_flip[: i + 1].any())
    if res_h and sup_h:
        return "both"
    if sup_h:
        return "support"
    return "resistance"


def events_for_group(
    sid: str,
    name: str,
    g: pd.DataFrame,
    sr: tuple[float | None, float | None] | None,
) -> tuple[list, list]:
    """單檔當日 m1 → (vwap 變號列, 壓力/支撐記錄列)。live 與盤後掃描共用。"""
    vwap_events: list = []
    sr_events: list = []
    if g is None or len(g) < 2:
        return vwap_events, sr_events
    g = g.sort_values("date")
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
        return vwap_events, sr_events

    for i in np.flatnonzero(flip):
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

    if not sr:
        return vwap_events, sr_events
    res, sup = sr
    res_flip = np.zeros(len(closes), dtype=bool)
    if res is not None:
        res_flip[1:] = (closes[1:] >= res) != (closes[:-1] >= res)
    sup_flip = np.zeros(len(closes), dtype=bool)
    if sup is not None:
        sup_flip[1:] = (closes[1:] >= sup) != (closes[:-1] >= sup)
    sr_flip = res_flip | sup_flip
    if not sr_flip.any():
        return vwap_events, sr_events
    for fire in sr_record_indices(flip, sr_flip):
        sr_events.append(
            {
                "stock_id": sid,
                "name": name,
                "sr_kind": _sr_kind_at(fire, res_flip, sup_flip),
                "vwap_dir": "up" if closes[fire] >= vwap[fire] else "down",
                "price": round(float(closes[fire]), 2),
                "vwap": round(float(vwap[fire]), 2),
                "resistance": round(float(res), 2) if res is not None else None,
                "support": round(float(sup), 2) if sup is not None else None,
                "time": _hhmm(times[fire]),
            }
        )
    return vwap_events, sr_events


def scan_day_events(
    m1: pd.DataFrame,
    sr_levels: dict[str, tuple[float, float]],
    names: dict[str, str] | None = None,
) -> tuple[list, list]:
    """回傳 (vwap_breakouts, sr_vwap_hits)，欄位與盤中 push payload + time 相同。
    VWAP 每次變號一筆；壓力/支撐穿越（且當天已有 VWAP 變號）每次一筆，同一支可重複。
    """
    names = names or {}
    vwap_events: list = []
    sr_events: list = []
    if m1 is None or m1.empty:
        return vwap_events, sr_events

    for sid, g in m1.groupby("stock_id", sort=False):
        sid = str(sid)
        ve, se = events_for_group(sid, names.get(sid, sid), g, sr_levels.get(sid))
        vwap_events.extend(ve)
        sr_events.extend(se)

    return vwap_events, sr_events


def scan_date(date_str: str, universe: str = "tick") -> tuple[list, list]:
    """掃某一曆日，回傳 (vwap_breakouts, sr_vwap_hits)。過去日期 cache 事件；
    今日每次重算（m1_live 可能還在寫），SR 水位仍 cache。"""
    universe = normalize_universe(universe)
    key = (date_str, universe)
    today = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
    with _scan_lock:
        if date_str != today and key in _scan_cache:
            return _scan_cache[key]

        m1 = load_day_m1(date_str, universe=universe)
        stocks = set(m1["stock_id"].astype(str)) if not m1.empty else set()
        levels = sr_levels_for_date(date_str, stocks)
        result = scan_day_events(m1, levels, _name_map())
        if date_str != today:
            _scan_cache[key] = result
        return result
