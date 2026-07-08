"""
即時推論 — predict()（批次機率矩陣）、predict_live()（正式即時推論入口，含強過濾）
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from data.query import load_day, load_m1_live
from strategy.rally.config import BREAKOUT_TRADE_END, BREAKOUT_TRADE_START, SESSION_END, SESSION_START
from strategy.rally.features import FEATURES, load_features, make_features
from strategy.rally.train import load_model


def predict(
    model=None,
    breakout_filter: bool = False,
    test_days: int = 10,
) -> pd.DataFrame:
    """
    對早盤時段產生預測機率矩陣（index=datetime, columns=stock_id）。

    breakout_filter=True 時，只保留 breakout_signal=True 的樣本
    （強過濾：先跌後漲破底翻才納入預測）。
    """
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES)

    if breakout_filter:
        df = df[df["breakout_signal"]]

    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()

    df["proba"] = model.predict_proba(df[FEATURES])[:, 1]
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def predict_live(
    minute_str: str,
    model=None,
    threshold: float = 0.55,
    use_breakout_filter: bool = True,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論 + 強過濾（breakout_signal 硬規則）。

    若 use_breakout_filter=True（預設，即「強做強過濾」），只有
    breakout_signal=True 的樣本才會進入模型預測——先跌後漲破底翻
    才允許下單，其餘直接剔除，不給任何訊號。

    回傳格式：[{"stock_id": ..., "proba": ..., "price": ..., "breakout": ...}, ...]
    """
    if model is None:
        model = load_model()

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    # ── 黃金窗口限制：只在 9:14~9:30 交易 ──────────────────────────
    _dt = pd.Timestamp(minute_str)
    _sh, _sm = BREAKOUT_TRADE_START
    _eh, _em = BREAKOUT_TRADE_END
    if not ((_dt.hour == _sh and _dt.minute >= _sm) and (_dt.hour == _eh and _dt.minute <= _em)):
        return []

    # make_features 以 day_date 做 merge；今日 day 資料尚不存在，補一行今日摘要
    day = load_day()
    day = day.copy()
    day["date"] = pd.to_datetime(day["date"])
    today_ts = pd.Timestamp(date_str)
    if not (day["date"] == today_ts).any():
        rows = []
        for sid, g in m1_live.groupby("stock_id"):
            g_s = g.sort_values("date")
            rows.append(
                {
                    "stock_id": sid,
                    "date": today_ts,
                    "open": float(g_s.iloc[0]["open"]),
                    "high": float(g["high"].max()),
                    "low": float(g["low"].min()),
                    "close": float(g_s.iloc[-1]["close"]),
                    "volume": int(g["volume"].sum()),
                }
            )
        if rows:
            day = pd.concat([day, pd.DataFrame(rows)], ignore_index=True)
            day["date"] = pd.to_datetime(day["date"])
            day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)

    df = make_features(m1_live, day=day, compute_labels=False)
    current = df[df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []

    valid = current.dropna(subset=FEATURES)
    if valid.empty:
        return []

    # ── 強過濾：硬規則（預設強制開啟）─────────────────────────────
    if use_breakout_filter:
        valid = valid[valid["breakout_signal"]]
        if valid.empty:
            return []

    proba = model.predict_proba(valid[FEATURES])[:, 1]
    signals = [
        {
            "stock_id": row["stock_id"],
            "proba": float(p),
            "price": float(row["close"]),
            "breakout": bool(row["breakout_signal"]),
        }
        for (_, row), p in zip(valid.iterrows(), proba)
        if p >= threshold
    ]
    return sorted(signals, key=lambda x: -x["proba"])
