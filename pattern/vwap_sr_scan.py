"""盤後一次掃全日 m1，還原 VWAP突破 / VWAP+壓力支撐（規則同 live_trader.on_minute）。

不模擬時鐘。當天讀 db/m1_live/{date}.parquet；非當天只讀 db/m1，
不退回 m1_live。盤中仍走 on_minute 增量推 SSE。GET ?date= 的
universe=daytrade|full 對齊 GET /api/pattern/stocks/daytrade、/stocks/full。
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pattern.horizontal_sr import horizontal_sr_prices

_ROOT = Path(__file__).resolve().parent.parent

_scan_cache: dict[tuple[str, str], tuple[list, list]] = {}
_m1_cache: dict[tuple[str, str], pd.DataFrame] = {}
_m1_bars: dict[tuple[str, str], int] = {}
_sr_levels_cache: dict[str, dict[str, tuple[float | None, float | None]]] = {}
_names_cache: dict[str, str] | None = None
# RLock：scan_date 持鎖時還會再進 load_day_m1。MACD 共用這把鎖，同一日 m1 只讀一次。
_scan_lock = threading.RLock()
# 掃描放背景執行緒：Django／瀏覽器超時斷線後仍繼續，下一點重現才能接到結果。
_scan_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vwap-replay")
_scan_jobs: dict[tuple[str, str], Future] = {}


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
    """正式值 daytrade / full；舊 tick 當 daytrade 別名。"""
    v = str(universe or "").strip().lower()
    return "full" if v == "full" else "daytrade"


def _parquet_stock_ids(path: Path, only_col: str | None = None) -> set[str]:
    """only_col 給的話，只留該欄位=True的列（見 stock_ids_for_universe()）。"""
    try:
        cols = ["stock_id"] + ([only_col] if only_col else [])
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        return set()
    if df is None or df.empty:
        return set()
    if only_col and only_col in df.columns:
        df = df[df[only_col] == True]  # noqa: E712
    return set(df["stock_id"].astype(str))


def stock_ids_for_universe(universe: str | None) -> set[str]:
    """daytrade＝tick_universe.parquet 裡 daytrade_ok=True 的列；讀不到（或
    這欄還沒驗證過）才退整份 tick_universe。full＝2000 ∪ daytrade。

    2026-08-19改：原本 daytrade 讀的是另外存的
    db/fubon_subscribe/subscribe_list.parquet，跟 tick_universe.parquet
    內容高度重疊，使用者要求合併成一份，daytrade_ok 現在是
    tick_universe.parquet 自己的欄位（見 fubon/subscribe_list.py::
    build_and_save_subscribe_list() 的說明），不用再讀第二個檔案。"""
    universe = normalize_universe(universe)
    tick_path = _ROOT / "db/tickers/tick_universe.parquet"
    daytrade = _parquet_stock_ids(tick_path, only_col="daytrade_ok")
    if not daytrade:
        daytrade = _parquet_stock_ids(tick_path)
    if universe == "full":
        full = _parquet_stock_ids(_ROOT / "db/tickers/stock_universe_2000.parquet")
        return full | daytrade
    return daytrade


def _from_m1_live(date_str: str) -> pd.DataFrame:
    from data.query import load_m1_live

    df = load_m1_live(date_str)
    if df.empty:
        return df
    df = df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _from_pattern_m1(date_str: str) -> pd.DataFrame:
    """只還原那一天。load_pattern_m1(start=end=當日) 仍會把整個月 parquet
    讀進記憶體再調整，第一次從 Dropbox 拉月檔很容易超時、前端就當沒資料。"""
    from data import raw_query
    from data.adjustment_query import _adjust_ohlc
    import pyarrow.dataset as ds

    start = pd.Timestamp(date_str)
    end = start + pd.Timedelta(days=1)
    paths = raw_query._dataset_paths(_ROOT / "db/m1", date_str, date_str)
    if not paths:
        return pd.DataFrame()
    dataset = ds.dataset(paths, format="parquet")
    df = pd.DataFrame()
    try:
        import pyarrow as pa

        df = dataset.to_table(
            filter=(ds.field("date") >= pa.scalar(start.to_pydatetime()))
            & (ds.field("date") < pa.scalar(end.to_pydatetime()))
        ).to_pandas()
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        df = dataset.to_table().to_pandas()
    df = df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[(df["date"] >= start) & (df["date"] < end)]
    if df.empty:
        return df
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return _adjust_ohlc(df, date_str, date_str)


def _filter_m1(m1: pd.DataFrame, universe: str) -> pd.DataFrame:
    ids = stock_ids_for_universe(universe)
    if not ids:
        if m1 is None or m1.empty:
            return m1 if m1 is not None else pd.DataFrame()
        return m1.iloc[0:0].copy()
    if m1 is None or m1.empty:
        return m1 if m1 is not None else pd.DataFrame()
    return m1[m1["stock_id"].astype(str).isin(ids)].reset_index(drop=True)


def load_day_m1(date_str: str, universe: str = "daytrade") -> pd.DataFrame:
    """當天：db/m1_live。非當天：只讀 db/m1（不退回 m1_live，避免第一次掃到
    盤中暫存、第二次才換成完整歷史）。過去日期 cache 非空的 db/m1。"""
    universe = normalize_universe(universe)
    today = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
    key = (date_str, universe)
    with _scan_lock:
        if date_str != today and key in _m1_cache:
            return _m1_cache[key]
        if date_str == today:
            m1 = _from_m1_live(date_str)
            if m1.empty:
                m1 = _from_pattern_m1(date_str)
        else:
            m1 = _from_pattern_m1(date_str)
        m1 = _filter_m1(m1, universe)
        _m1_bars[key] = 0 if m1 is None or m1.empty else int(len(m1))
        if date_str != today and not m1.empty:
            _m1_cache[key] = m1
        return m1


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
    highs = g["high"].astype(float).to_numpy()
    lows = g["low"].astype(float).to_numpy()
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
    # 2026-08-17改：原本用「收盤價 vs 前一根收盤價」判斷有沒有穿越壓力/支撐，
    # 抓不到「這根K棒的高低價曾經觸及/突破，但收盤又拉回」的情況——例如
    # 6770 2026-08-17開盤09:00 high=79.9 直接站上壓力79.2，但那根收盤78.7已經
    # 拉回，全天每根收盤價都沒超過79.2，res_flip 永遠是 False，SR燈不會亮，
    # 但盤中明明已經測過這個壓力（使用者回報）。改用「這根高/低價有沒有
    # 觸及水位」當狀態，觸及狀態前後改變才算一次「flip」（跟原本收盤價的
    # flip 邏輯同一種寫法，只是判斷觸及與否的欄位從 close 換成 high/low），
    # 壓力用 high、支撐用 low，不再要求收盤價真的站上/跌破。
    res_flip = np.zeros(len(closes), dtype=bool)
    if res is not None:
        res_touch = highs >= res
        res_flip[1:] = res_touch[1:] != res_touch[:-1]
    sup_flip = np.zeros(len(closes), dtype=bool)
    if sup is not None:
        sup_touch = lows <= sup
        sup_flip[1:] = sup_touch[1:] != sup_touch[:-1]
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


def _scan_date_body(date_str: str, universe: str) -> tuple[list, list]:
    m1 = load_day_m1(date_str, universe=universe)
    stocks = set(m1["stock_id"].astype(str)) if not m1.empty else set()
    levels = sr_levels_for_date(date_str, stocks)
    result = scan_day_events(m1, levels, _name_map())
    today = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
    if date_str != today and not m1.empty:
        with _scan_lock:
            _scan_cache[(date_str, universe)] = result
    return result


def last_m1_bars(date_str: str, universe: str = "daytrade") -> int:
    return _m1_bars.get((date_str, normalize_universe(universe)), 0)


def last_chg_map(date_str: str, universe: str = "daytrade") -> dict[str, float]:
    """每股最新 1 分收相對昨收漲跌幅（%）。重現／catchup 用。"""
    m1 = load_day_m1(date_str, universe=universe)
    if m1 is None or m1.empty:
        return {}
    m1 = m1.copy()
    m1["stock_id"] = m1["stock_id"].astype(str)
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    last = m1.sort_values("date").groupby("stock_id", sort=False)["close"].last()

    from data.adjustment_query import load_pattern_day

    hist_start = (pd.Timestamp(date_str) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    day = load_pattern_day(start_date=hist_start, end_date=date_str)
    prev: dict[str, float] = {}
    if day is not None and not day.empty:
        day = day.copy()
        day["stock_id"] = day["stock_id"].astype(str)
        day["date"] = pd.to_datetime(day["date"], format="mixed")
        before = day[day["date"] < pd.Timestamp(date_str)]
        if not before.empty:
            prev = (
                before.sort_values("date")
                .groupby("stock_id", sort=False)["close"]
                .last()
                .astype(float)
                .to_dict()
            )

    out: dict[str, float] = {}
    for sid, close in last.items():
        pc = prev.get(sid)
        if pc is None or not np.isfinite(pc) or pc <= 0:
            continue
        if not np.isfinite(close):
            continue
        out[str(sid)] = round((float(close) / float(pc) - 1.0) * 100, 2)
    return out


def scan_date(date_str: str, universe: str = "daytrade") -> tuple[list, list]:
    """掃某一曆日，回傳 (vwap_breakouts, sr_vwap_hits)。過去日期 cache 事件；
    今日每次重算（m1_live 可能還在寫），SR 水位仍 cache。
    過去日期丟背景執行緒：HTTP 超時斷線後掃描繼續，下一次請求等同一個 job。"""
    universe = normalize_universe(universe)
    key = (date_str, universe)
    today = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
    if date_str == today:
        with _scan_lock:
            return _scan_date_body(date_str, universe)
    with _scan_lock:
        if key in _scan_cache:
            return _scan_cache[key]
        fut = _scan_jobs.get(key)
        if fut is None or fut.done():
            fut = _scan_pool.submit(_scan_date_body, date_str, universe)
            _scan_jobs[key] = fut
    return fut.result()
