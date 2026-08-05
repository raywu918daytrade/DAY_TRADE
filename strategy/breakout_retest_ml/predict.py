"""
即時／回測推論。

- predict(): 回傳做多機率矩陣（止盈 class=2 的機率），給 run_backtest 用
- predict_live(): 盤中每分鐘呼叫；以「昨日收盤」日 K 判定 POC 共振候選，
  當分鐘檢查 M1 陽線實體 K + Tick 大量買進後輸出做多訊號
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.query import load_m1_live
from data.adjustment_query import load_pattern_day, load_pattern_poc
from finmind.tick_universe import load_tick_universe
from strategy.breakout_retest_ml.config import MODEL_TYPE, THRESHOLD
from strategy.breakout_retest_ml.features import (
    FEATURES,
    detect_candidate_asof,
    find_intraday_trigger,
)
from strategy.breakout_retest_ml.train import _prepare_data, load_model_by_type

# 盤中同一天重複用的候選快取：{trade_date: {stock_id: cand_dict}}
_live_cand_cache: dict[str, dict[str, dict]] = {}


def predict(
    model=None,
    test_days: int = 30,
    test_only: bool = True,
    use_cache: bool = False,
    start_date: str | None = "2025-07-01",
) -> pd.DataFrame:
    """對事件樣本產生「做多＝止盈」機率矩陣。"""
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    df = _prepare_data(use_cache=use_cache, start_date=start_date)
    if df.empty:
        return pd.DataFrame()
    if test_only:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        df = df[df["date"] > cutoff]

    df = df.copy()
    proba = model.predict_proba(df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    df["proba"] = proba[:, class_idx[2]] if 2 in class_idx else 0.0
    return df.pivot(index="date", columns="stock_id", values="proba")


def _prev_calendar_day(date_str: str) -> str:
    return (pd.Timestamp(date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _load_live_candidates(trade_date: str, stock_ids: list[str]) -> dict[str, dict]:
    """用 trade_date 前一自然日以前的日 K（asof）掃描 POC 共振候選。"""
    if trade_date in _live_cand_cache:
        return _live_cand_cache[trade_date]

    asof = _prev_calendar_day(trade_date)
    hist_start = (pd.Timestamp(trade_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    day_df = load_pattern_day(start_date=hist_start)
    day_df = day_df[day_df["stock_id"].isin(stock_ids)]
    poc_df = load_pattern_poc(start_date=hist_start)
    if not poc_df.empty:
        poc_df = poc_df[poc_df["stock_id"].isin(stock_ids)]

    out: dict[str, dict] = {}
    for sid in stock_ids:
        cand = detect_candidate_asof(day_df, poc_df, sid, asof_date=asof)
        if cand is not None:
            out[sid] = cand
    print(f"[breakout_retest_ml live] {trade_date} 候選 {len(out)} 檔（asof={asof}）", flush=True)
    _live_cand_cache[trade_date] = out
    return out


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    model=None,
    threshold: float = THRESHOLD,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
    ticks_by_stock: dict | None = None,
) -> list:
    """即時推論。參數順序與 vwap_ml/orb 一致（day 佔第 2 位）。

    ticks_by_stock: 可選 {stock_id: DataFrame}；未提供時 Tick 硬過濾會失敗、無訊號。
    （live tick 接入後由此傳入；訓練／回測走 db/tick。）

    回傳: [{"stock_id", "proba", "price", "direction": "up"}, ...]
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    universe = set(load_tick_universe())
    if day_trade_stocks:
        universe &= set(day_trade_stocks)
    stock_ids = sorted(universe)
    if not stock_ids:
        return []

    candidates = _load_live_candidates(date_str, stock_ids)
    if not candidates:
        return []

    m1_live = m1_live.copy()
    m1_live["date"] = pd.to_datetime(m1_live["date"], format="mixed")
    trigger_ts = pd.Timestamp(minute_str)

    rows = []
    for sid, cand in candidates.items():
        m1_day = m1_live[m1_live["stock_id"] == sid].sort_values("date")
        if m1_day.empty:
            continue
        ticks = None
        if ticks_by_stock is not None:
            ticks = ticks_by_stock.get(sid)
        trigger = find_intraday_trigger(
            m1_day.reset_index(drop=True),
            resistance=float(cand["resistance_price"]),
            matched_poc=float(cand["matched_poc"]) if pd.notna(cand.get("matched_poc")) else np.nan,
            ticks=ticks,
            only_at=trigger_ts,
        )
        if trigger is None:
            continue
        rows.append(
            {
                "stock_id": sid,
                "pattern_score": float(cand["pattern_score"]),
                "poc_diff_pct": float(cand["poc_diff_pct"]) if pd.notna(cand.get("poc_diff_pct")) else 0.0,
                "dist_to_poc_pct": trigger["dist_to_poc_pct"],
                "dist_to_support_pct": trigger["dist_to_support_pct"],
                "body_ratio": trigger["body_ratio"],
                "lower_shadow_ratio": trigger["lower_shadow_ratio"],
                "upper_shadow_ratio": trigger["upper_shadow_ratio"],
                "volume_surge_ratio": trigger["volume_surge_ratio"],
                "tick_large_buy_ratio": trigger["tick_large_buy_ratio"],
                "tick_large_sell_ratio": trigger.get("tick_large_sell_ratio", 0.0),
                "tick_large_net_ratio": trigger.get(
                    "tick_large_net_ratio",
                    float(trigger["tick_large_buy_ratio"]) - float(trigger.get("tick_large_sell_ratio", 0.0) or 0.0),
                ),
                "cvd_30s_delta": trigger["cvd_30s_delta"],
                "close": trigger["entry_price"],
            }
        )

    if not rows:
        return []

    feat_df = pd.DataFrame(rows)
    valid = feat_df.dropna(subset=FEATURES)
    if valid.empty:
        return []

    proba = model.predict_proba(valid[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    p_tp = proba[:, class_idx[2]] if 2 in class_idx else np.zeros(len(valid))

    signals = []
    for (_, row), p in zip(valid.iterrows(), p_tp):
        if float(p) >= threshold:
            signals.append(
                {
                    "stock_id": row["stock_id"],
                    "proba": float(p),
                    "price": float(row["close"]),
                    "direction": "up",
                }
            )
    return sorted(signals, key=lambda x: -x["proba"])
