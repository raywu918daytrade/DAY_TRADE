"""VWAP 表 OBV 背離燈：盤中每根 1 分都重算，不是當天第一次就定終身。

跟 pattern/vwap_macd_div.py 是同一套「先鎖過去 lookback 根的極值、再配對、
再比價格」架構，差別在抓極值的來源——這裡直接用原始累積 OBV（能量潮）自己
的高低點，不是先把 OBV 套進 MACD 公式再算。OBV 是當天從 0 開始累加的量能
指標（跟 VWAP 一樣每天重新起算，不跨日），不會像 MACD 柱體那樣自然繞著 0
震盪，所以極值鎖定不能直接沿用 pattern/macd_hist_bull/detector.py 的
_lookback_hist_locks()（那支函式要求谷必須在0以下、峰必須在0以上），這裡
另外寫一支不要求正負號的版本；配對邏輯裡「兩個極值之間要出現異號」的檢查
（hist_has_sign_between，背離要看到柱體確實縮小過）也是 MACD 柱體特有的，
OBV 版本沒有這個概念，一併拿掉。

2026-08-17討論：原始 OBV 版本訊號會比先去趨勢化（例如 OBV 減自己的 EMA，
或直接套 macd_histogram(obv)）少很多——去趨勢化後 OBV 會變成震盪指標，
局部高低點天生密集，還能直接沿用 iter_macd_hist_div_pairs()。使用者決定
先用原始 OBV 版本，訊號太少的話之後再改；把 OBV 專屬的鎖定/配對邏輯獨立
寫在這支檔案，不動 macd_hist_bull/detector.py，是為了讓那個切換之後好做
——到時候只要把餵進 all_divs_for_group() 的 obv 陣列換成去趨勢化後的序列，
把下面 iter_obv_div_pairs() 的呼叫換成 iter_macd_hist_div_pairs()，_leg()
之後（含 API／SSE／前端 render）完全不用動。
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pattern.vwap_sr_scan import _hhmm, load_day_m1, normalize_universe

_TW = timezone(timedelta(hours=8))

PIVOT_L = 2
LOOKBACK = 5
MIN_DIST = 3
MAX_DIST = 30
MIN_CANDLES = 40
MAX_AGE = 5
WARMUP = 0

_cache: dict[tuple[str, str], dict[str, dict]] = {}
_lock = threading.Lock()


def _epoch(ts) -> int:
    from api import tw_naive_to_epoch

    return tw_naive_to_epoch(pd.Timestamp(ts))


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """標準能量潮：obv[0]=0，之後收盤價漲一根加量、跌一根減量、平盤不變，
    逐根累加。輸入已經是單一股票當天的資料，天然每天從0重新起算，不用
    另外按日期分組。"""
    direction = np.sign(np.diff(close, prepend=close[0]))
    direction[0] = 0.0
    return np.cumsum(direction * volume)


def _lookback_extrema_locks(values: np.ndarray, kind: str, lookback: int) -> list[int]:
    """過去 lookback 根：左端是窗內唯一極值、且相對前一根轉折。確認在
    i=j+lookback-1。跟 macd_hist_bull/detector.py::_lookback_hist_locks()
    幾乎一樣，差別是不要求正負號——OBV 是累積量，不會自然繞著0震盪，
    可能整天都在正值或負值區，強制要求正負號會找不到任何極值。"""
    n = len(values)
    L = lookback
    out: list[int] = []
    for i in range(L - 1, n):
        j = i - L + 1
        if j > 0:
            if kind == "trough" and not (values[j] < values[j - 1]):
                continue
            if kind == "peak" and not (values[j] > values[j - 1]):
                continue
        w = values[j : i + 1]
        if kind == "trough":
            if values[j] != np.min(w) or np.sum(w == values[j]) != 1:
                continue
        else:
            if values[j] != np.max(w) or np.sum(w == values[j]) != 1:
                continue
        out.append(j)
    return out


def iter_obv_div_pairs(
    obv: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    side: str,
    pivot_l: int = 2,
    min_dist: int = 3,
    max_dist: int = 30,
    warmup: int = 0,
    lookback: int | None = None,
):
    """跟 iter_macd_hist_div_pairs() 同一種配對邏輯（間距限制、比對兩次
    極值的指標值高低、比對對應價格窗的高低點），拿掉「中間要有異號」的
    檢查——OBV 是累積量，沒有柱體縮小這個概念。

    bull：OBV 兩個谷比較（第二谷比第一谷高＝量能墊高），配對的價格窗
    取最低點，要求第二窗價格比第一窗低（價格破底、OBV沒破底＝底背離）。
    bear：OBV 兩個峰比較（第二峰比第一峰低＝量能走弱），配對的價格窗
    取最高點，要求第二窗價格比第一窗高（價格創高、OBV沒創高＝頂背離）。
    """
    L = int(lookback) if lookback is not None else (2 * int(pivot_l) + 1)
    n = len(obv)
    if side == "bull":
        o_ext = _lookback_extrema_locks(obv, "trough", L)
        px = lows
    else:
        o_ext = _lookback_extrema_locks(obv, "peak", L)
        px = highs

    os_: list[int] = []
    for o in o_ext:
        if o < warmup:
            continue
        if o + L - 1 >= n:
            continue
        os_.append(o)

    def _k_at(o: int) -> tuple[int, float]:
        right = o + L - 1
        w = px[o : right + 1]
        off = int(np.argmin(w) if side == "bull" else np.argmax(w))
        p = o + off
        return p, float(px[p])

    for b in range(1, len(os_)):
        o2 = os_[b]
        v2 = float(obv[o2])
        best_o1: int | None = None
        best_v1: float | None = None
        for a in range(b):
            o1 = os_[a]
            if not (min_dist <= o2 - o1 <= max_dist):
                continue
            v1 = float(obv[o1])
            if side == "bull":
                if not (v2 > v1):
                    continue
                if best_v1 is None or v1 < best_v1:
                    best_o1, best_v1 = o1, v1
            else:
                if not (v2 < v1):
                    continue
                if best_v1 is None or v1 > best_v1:
                    best_o1, best_v1 = o1, v1
        if best_o1 is None or best_v1 is None:
            continue
        p1, price1 = _k_at(best_o1)
        p2, price2 = _k_at(o2)
        if side == "bull":
            if not (price2 < price1):
                continue
        elif not (price2 > price1):
            continue
        yield {
            "h1": best_o1,
            "h2": o2,
            "p1": p1,
            "p2": p2,
            "obv1": best_v1,
            "obv2": v2,
            "price1": price1,
            "price2": price2,
            "confirmed": o2 + L - 1,
        }


def _leg(kind: str, pair: dict, times, n: int) -> dict | None:
    h1, h2 = int(pair["h1"]), int(pair["h2"])
    p1, p2 = int(pair["p1"]), int(pair["p2"])
    confirmed = int(pair["confirmed"])
    if confirmed < MIN_CANDLES - 1 or confirmed >= n:
        return None
    until_i = min(n - 1, max(confirmed, h2 + MAX_AGE))
    if until_i < confirmed:
        return None
    return {
        "kind": kind,
        "t1": _hhmm(times[h1]),
        "t2": _hhmm(times[h2]),
        "t1_ts": _epoch(times[h1]),
        "t2_ts": _epoch(times[h2]),
        "obv1": round(float(pair["obv1"]), 2),
        "obv2": round(float(pair["obv2"]), 2),
        "p1": _hhmm(times[p1]),
        "p2": _hhmm(times[p2]),
        "p1_ts": _epoch(times[p1]),
        "p2_ts": _epoch(times[p2]),
        "price1": round(float(pair["price1"]), 4),
        "price2": round(float(pair["price2"]), 4),
        "time": _hhmm(times[confirmed]),
        "until": _hhmm(times[until_i]),
    }


def all_divs_for_group(g: pd.DataFrame) -> list[dict]:
    """同一檔當日 1 分 K：OBV 縮小確認後的極值，搭配附近 K 線高低點。"""
    if g is None or g.empty or len(g) < MIN_CANDLES:
        return []
    need = ["date", "high", "low", "close", "volume"]
    if any(c not in g.columns for c in need):
        return []
    g = g.sort_values("date")
    close = g["close"].astype(float).to_numpy()
    if not np.isfinite(close).all():
        return []
    highs = g["high"].astype(float).to_numpy()
    lows = g["low"].astype(float).to_numpy()
    volume = g["volume"].astype(float).to_numpy()
    times = g["date"].to_numpy()
    obv = _obv(close, volume)
    n = len(obv)
    out: list[dict] = []
    kw = dict(
        pivot_l=PIVOT_L,
        lookback=LOOKBACK,
        min_dist=MIN_DIST,
        max_dist=MAX_DIST,
        warmup=WARMUP,
    )
    for pair in iter_obv_div_pairs(obv, highs, lows, side="bull", **kw):
        leg = _leg("bull", pair, times, n)
        if leg:
            out.append({**leg, "legs": [leg]})
    for pair in iter_obv_div_pairs(obv, highs, lows, side="bear", **kw):
        leg = _leg("bear", pair, times, n)
        if leg:
            out.append({**leg, "legs": [leg]})
    out.sort(key=lambda e: (e["time"], e["kind"]))
    return out


def divs_map_from_m1(m1: pd.DataFrame) -> dict[str, dict]:
    """stock_id → {events: [...]}，沒背離的不進 map。"""
    out: dict[str, dict] = {}
    if m1 is None or m1.empty:
        return out
    for sid, g in m1.groupby("stock_id", sort=False):
        evs = all_divs_for_group(g)
        if evs:
            out[str(sid)] = {"events": evs}
    return out


def _compute(date_str: str, universe: str = "daytrade") -> dict[str, dict]:
    return divs_map_from_m1(load_day_m1(date_str, universe=universe))


def metrics_for_date(date_str: str, universe: str = "daytrade") -> dict[str, dict]:
    universe = normalize_universe(universe)
    key = (date_str, universe)
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    with _lock:
        if date_str != today and key in _cache:
            return _cache[key]
        result = _compute(date_str, universe=universe)
        # 空 map 也可能是 m1 還沒讀到，不 cache，讓下一點重現再試。
        if date_str != today and result:
            _cache[key] = result
        return result
