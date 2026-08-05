"""
即時／回測推論。

- predict(): 對事件資料集算「止盈（class=2）」機率，附加在原始事件 DataFrame 上，
  供 run_backtest.py 用（保留 entry_price/exit_price/exit_reason，不用重算一次 Triple Barrier）。
- predict_live(): 兩階段（2026-08-04 改版，理由見 config.py 檔頭）。09:03 那分鐘
  只記錄 Stage1 候選（跳空+首根3分K下跌）到當日待確認快取，不出訊號；09:10 那分鐘
  讀待確認快取，比對 09:09 那根 M1 收盤價是否比 Stage1 的 m3 收盤價更低（延續下跌），
  確認才組完整特徵、餵模型、出訊號。其餘分鐘直接回傳空 list。
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
from data.resample import compute_m3_std
from finmind.tick_universe import load_tick_universe
from strategy.limitup_fade_ml.config import (
    ATR_WINDOW,
    CONFIRM_CHECK_TIME,
    CONFIRM_TIME,
    FIRST_M3_TIME,
    LIMIT_UP_RET,
    MAX_UPPER_SHADOW_RATIO,
    MIN_BODY_RATIO,
    MODEL_TYPE,
    THRESHOLD,
)
from strategy.limitup_fade_ml.dataset import IDX_SYMBOL, _shadow_ratios
from strategy.limitup_fade_ml.features import FEATURES
from strategy.limitup_fade_ml.train import _split_data, load_model_by_type


def predict(
    model=None,
    test_days: int = 60,
    test_only: bool = True,
    use_cache: bool = False,
    start_date: str | None = "2022-01-01",
) -> pd.DataFrame:
    """對事件樣本算「止盈」機率，回傳原始事件欄位 + proba（給 run_backtest 用）。"""
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    if test_only:
        _, df = _split_data(test_days, use_cache=use_cache, start_date=start_date)
    else:
        from strategy.limitup_fade_ml.train import _prepare_data

        df = _prepare_data(use_cache=use_cache, start_date=start_date)
    if df.empty:
        return df

    df = df.copy()
    proba = model.predict_proba(df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    df["proba"] = proba[:, class_idx[2]] if 2 in class_idx else 0.0
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Live
# ═══════════════════════════════════════════════════════════════════════════

# 盤中同一天重複用的候選快取：{trade_date: {stock_id: cand_dict}}
_live_cand_cache: dict[str, dict[str, dict]] = {}


def _day_atr_raw(g: pd.DataFrame) -> float:
    """算「最新一筆」(=候選日/漲停日) 為止的日K ATR14（未正規化，價格單位），
    比照 dataset.py::_add_day_atr() 的公式，取最後 ATR_WINDOW+1 列（含最新一筆）
    算 True Range 再平均。live 端沒有 today_open 可以正規化（要等 Stage1 從
    m1_live 讀到才知道），所以回傳原始 ATR 值，正規化留給呼叫端。"""
    if len(g) < ATR_WINDOW + 1:
        return float("nan")
    tail = g.iloc[-(ATR_WINDOW + 1) :]
    prev_close_tail = tail["close"].shift(1)
    tr_tail = np.maximum(
        np.maximum((tail["high"] - tail["low"]).abs(), (tail["high"] - prev_close_tail).abs()),
        (tail["low"] - prev_close_tail).abs(),
    )
    return float(tr_tail.iloc[1:].mean())


def _scan_candidate_asof(day_df: pd.DataFrame) -> dict[str, dict]:
    """檢查每支股票「最新一筆日K」（asof=昨日）是否符合前日漲停硬過濾條件。

    跟 dataset.py::build_gap_candidates() 不同：這裡不找「下一交易日」（因為
    call 這支的當下，今天就是下一交易日，不需要再查表），只針對最後一筆asof
    日K 直接判斷，避免 build_gap_candidates() 的 next-trade-day 邏輯在 day_df
    還沒有「今天」這筆資料時失效。"""
    out: dict[str, dict] = {}
    for sid, g in day_df.groupby("stock_id"):
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        last = g.iloc[-1]
        prev_close = float(g["close"].iloc[-2])
        if prev_close <= 0:
            continue

        day_ret = float(last["close"]) / prev_close - 1.0
        if day_ret < LIMIT_UP_RET or last["close"] <= last["open"]:
            continue
        upper, _lower, body = _shadow_ratios(
            float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        )
        if body < MIN_BODY_RATIO or upper > MAX_UPPER_SHADOW_RATIO:
            continue

        day_atr_raw = _day_atr_raw(g)
        if not np.isfinite(day_atr_raw):
            continue  # ATR14不足14天歷史（新掛牌股），比照dataset.py捨棄

        prev_vol_5d_avg = g["volume"].iloc[-6:-1].mean() if len(g) >= 6 else np.nan
        prev_vol_20d_avg = g["volume"].iloc[-21:-1].mean() if len(g) >= 6 else np.nan
        prev5 = float(g["close"].iloc[-7]) if len(g) >= 7 else np.nan

        out[sid] = {
            "stock_id": sid,
            "prev_close": float(last["close"]),
            "prev_high": float(last["high"]),
            "prev_day_ret": day_ret,
            "prev_body_ratio": round(body, 4),
            "prev_upper_shadow_ratio": round(upper, 4),
            "prev_volume_ratio": (
                float(last["volume"]) / prev_vol_5d_avg if prev_vol_5d_avg and prev_vol_5d_avg > 0 else np.nan
            ),
            "prev_volume_z": (
                (float(last["volume"]) - prev_vol_20d_avg) / prev_vol_20d_avg
                if prev_vol_20d_avg and prev_vol_20d_avg > 0
                else np.nan
            ),
            "prev5d_ret": float(last["close"]) / prev5 - 1.0 if prev5 and prev5 > 0 else np.nan,
            "day_atr_raw": day_atr_raw,
        }
    return out


def _load_live_candidates(trade_date: str, stock_ids: list[str]) -> tuple[dict[str, dict], float | None]:
    """用 trade_date 前一自然日以前的日K（asof）掃描前日漲停候選，每個交易日快取一次。

    回傳 (candidates, idx_prev_close)：idx_prev_close 是 0050 asof 前一日收盤價，
    給 predict_live() 用當下 m1_live 的 0050 開盤價算 gap_vs_0050（0050 本身不會通過
    漲停篩選，不會出現在 candidates 裡，要另外算）。"""
    if trade_date in _live_cand_cache:
        return _live_cand_cache[trade_date]

    hist_start = (pd.Timestamp(trade_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    stock_ids_with_idx = list(set(stock_ids) | {IDX_SYMBOL})
    day_df = load_pattern_day(start_date=hist_start)
    day_df = day_df[(day_df["stock_id"].isin(stock_ids_with_idx)) & (day_df["date"] < pd.Timestamp(trade_date))]

    out = _scan_candidate_asof(day_df[day_df["stock_id"].isin(stock_ids)]) if not day_df.empty else {}

    idx_df = day_df[day_df["stock_id"] == IDX_SYMBOL].sort_values("date")
    idx_prev_close = float(idx_df["close"].iloc[-1]) if not idx_df.empty else None

    print(f"[limitup_fade_ml live] {trade_date} 候選 {len(out)} 檔", flush=True)
    _live_cand_cache[trade_date] = (out, idx_prev_close)
    return out, idx_prev_close


# Stage2 待確認快取：{trade_date: {stock_id: stage1_feature_dict}}——09:03那分鐘存進去，
# 09:10那分鐘讀出來判斷延續確認，兩個時間點分開呼叫 predict_live()，中間要跨分鐘保存。
_pending_confirm_cache: dict[str, dict[str, dict]] = {}


def _run_stage1(date_str: str, m1_live: pd.DataFrame, day: pd.DataFrame | None, day_trade_stocks: set | None) -> None:
    """09:03 那分鐘呼叫：跳空開高+首根3分K下跌都成立，把 Stage1 特徵存進待確認
    快取，不出訊號（要等09:10 Stage2確認才知道要不要出訊號）。"""
    if day_trade_stocks:
        stock_ids = sorted(day_trade_stocks)
    elif day is not None and not day.empty:
        stock_ids = sorted(day["stock_id"].unique())
    else:
        stock_ids = load_tick_universe()
    if not stock_ids:
        return

    candidates, idx_prev_close = _load_live_candidates(date_str, stock_ids)
    if not candidates:
        return

    idx_gap_pct = 0.0
    if idx_prev_close and idx_prev_close > 0:
        idx_m1 = m1_live[m1_live["stock_id"] == IDX_SYMBOL].sort_values("date")
        if not idx_m1.empty:
            idx_gap_pct = float(idx_m1["open"].iloc[0]) / idx_prev_close - 1.0

    pending: dict[str, dict] = {}
    for sid, cand in candidates.items():
        m1_day = m1_live[m1_live["stock_id"] == sid].sort_values("date")
        if m1_day.empty:
            continue

        today_open = float(m1_day["open"].iloc[0])
        if today_open <= cand["prev_close"]:
            continue  # 沒跳空開高

        m3 = compute_m3_std(m1_day)
        trig = m3[m3["date"].dt.time == dtime(*FIRST_M3_TIME)]
        if trig.empty:
            continue
        tr = trig.iloc[0]
        m3_open, m3_high, m3_low, m3_close = (
            float(tr["open"]),
            float(tr["high"]),
            float(tr["low"]),
            float(tr["close"]),
        )
        if m3_close >= m3_open:
            continue  # 首根3分K沒下跌

        upper, lower, body = _shadow_ratios(m3_open, m3_high, m3_low, m3_close)
        gap_pct = (today_open - cand["prev_close"]) / cand["prev_close"]
        day_atr = cand["day_atr_raw"] / today_open if today_open > 0 else float("nan")

        pending[sid] = {
            "m3_close": m3_close,
            "day_atr": day_atr,
            "tp_price": m3_close * (1.0 - day_atr),
            "sl_price": m3_close * (1.0 + day_atr),
            "gap_pct": gap_pct,
            "gap_vs_0050": gap_pct - idx_gap_pct,
            "open_vs_prev_high": today_open / cand["prev_high"] - 1.0 if cand["prev_high"] > 0 else np.nan,
            "prev_day_ret": cand["prev_day_ret"],
            "prev_body_ratio": cand["prev_body_ratio"],
            "prev_upper_shadow_ratio": cand["prev_upper_shadow_ratio"],
            "prev_volume_ratio": cand["prev_volume_ratio"],
            "prev_volume_z": cand["prev_volume_z"],
            "prev5d_ret": cand["prev5d_ret"],
            "m3_ret": (m3_close - m3_open) / m3_open,
            "m3_body_ratio": round(body, 4),
            "m3_upper_shadow_ratio": round(upper, 4),
            "m3_lower_shadow_ratio": round(lower, 4),
            "m3_range_pct": (m3_high - m3_low) / m3_open if m3_open > 0 else 0.0,
            "dist_from_open_pct": (m3_close - today_open) / today_open,
        }

    print(f"[limitup_fade_ml live] {date_str} Stage1觸發 {len(pending)} 檔，等09:10延續確認", flush=True)
    _pending_confirm_cache[date_str] = pending


def _run_stage2(date_str: str, m1_live: pd.DataFrame, model, threshold: float) -> list:
    """09:10 那分鐘呼叫：讀 Stage1 待確認快取，比對 09:09 那根 M1 收盤價是否比
    m3_close 更低（延續下跌），確認才組完整特徵、餵模型、出訊號。"""
    pending = _pending_confirm_cache.pop(date_str, None)
    if not pending:
        return []

    confirm_time = dtime(*CONFIRM_TIME)
    rows = []
    for sid, feat in pending.items():
        m1_day = m1_live[m1_live["stock_id"] == sid]
        confirm_bar = m1_day[m1_day["date"].dt.time == confirm_time]
        if confirm_bar.empty:
            continue
        confirm_price = float(confirm_bar["close"].iloc[0])
        if confirm_price >= feat["m3_close"]:
            continue  # 沒延續下跌，視為假突破/已反彈，不進場

        row = dict(feat)
        row["stock_id"] = sid
        row["confirm_ret"] = (confirm_price - feat["m3_close"]) / feat["m3_close"]
        row["close"] = confirm_price
        rows.append(row)

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
                    "direction": "down",
                }
            )
    return sorted(signals, key=lambda x: -x["proba"])


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    model=None,
    threshold: float = THRESHOLD,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """即時推論。參數順序與 breakout_retest_ml/mkt 一致（day 佔第 2 位，目前用不到，
    只保留給 main/live_trader.py 統一呼叫介面用）。

    兩階段（見檔頭說明）：minute_str 對到 09:03 只記錄 Stage1 候選，不出訊號；
    對到 09:10 才讀待確認快取做 Stage2 確認、出訊號；其餘分鐘直接回傳空 list。

    回傳: [{"stock_id", "proba", "price", "direction": "down"}, ...]
    """
    date_str = minute_str[:10]
    now_time = pd.Timestamp(minute_str).time()

    if now_time not in (dtime(*FIRST_M3_TIME), dtime(*CONFIRM_CHECK_TIME)):
        return []

    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []
    m1_live = m1_live.copy()
    m1_live["date"] = pd.to_datetime(m1_live["date"], format="mixed")

    if now_time == dtime(*FIRST_M3_TIME):
        _run_stage1(date_str, m1_live, day, day_trade_stocks)
        return []

    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    return _run_stage2(date_str, m1_live, model, threshold)
