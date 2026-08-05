"""
驗證「滾動 5 分鐘實體陽線 + 窗內 tick」各種用法（±3%/30 分）。

重點：
- 只看 09:10～10:00（窗結束時刻落在此區間）
- 用連續 5 根 M1 合成任意起點視窗（例如 09:06～09:11）
- 可選 ATR5（/當日開盤）過濾低波動，壓震盪樣本（對齊 mkt 思路）
- 主比較：同一批進場上掃窗內大單買比門檻

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_m5_tick \\
        --start_date 2026-05-01 --end_date 2026-07-31 --min_atr5 0.01
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import time as dtime
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from data.adjustment_query import load_pattern_m1
from data.query import load_breakout_retest_day, load_tick_by_stock
from strategy.breakout_retest_ml.config import (
    LABEL_HORIZON_MINUTES,
    MAX_UPPER_SHADOW_RATIO,
    MIN_BODY_RATIO,
    SESSION_END,
    SL_PCT,
    TICK_LARGE_LOT,
    TP_PCT,
)
from strategy.breakout_retest_ml.features import (
    _next_trade_day,
    _shadow_ratios,
    _triple_barrier_label,
)

ROLL_MINUTES = 5
# 與 strategy/mkt/config.ATR5_FILTER_THRESHOLD 對齊
DEFAULT_MIN_ATR5 = 0.010
# 決策窗起點 09:05：滾動5分 + ATR5 到此刻已有足夠 M1（不沿用 config 的 09:10）
SESSION_START = (9, 5)


def _add_atr5_day(m1_day: pd.DataFrame) -> pd.DataFrame:
    """單日 M1：atr5 = ATR(5) / 當日開盤（公式同 mkt.features.add_atr5）。"""
    m1 = m1_day.sort_values("date").reset_index(drop=True).copy()
    if m1.empty:
        m1["atr5"] = pd.Series(dtype=float)
        return m1
    day_open = float(m1["open"].iloc[0])
    if not np.isfinite(day_open) or day_open <= 0:
        m1["atr5"] = np.nan
        return m1
    prev_close = m1["close"].shift(1).fillna(m1["open"])
    tr = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - prev_close).abs()),
        (m1["low"] - prev_close).abs(),
    )
    m1["atr5"] = tr.rolling(5, min_periods=5).mean() / day_open
    return m1


def _atr5_at(m1_day_atr: pd.DataFrame, last_min: pd.Timestamp) -> float:
    """取窗最後一根 M1（date==last_min）的 atr5。"""
    if m1_day_atr is None or m1_day_atr.empty:
        return float("nan")
    hit = m1_day_atr[m1_day_atr["date"] == last_min]
    if hit.empty:
        # 退而求次：<= last_min 最後一根
        hit = m1_day_atr[m1_day_atr["date"] <= last_min].tail(1)
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0]["atr5"])


def _tick_in_window(ticks: pd.DataFrame, win_start: pd.Timestamp, win_end: pd.Timestamp) -> dict:
    """窗內 [win_start, win_end) 的外盤比／CVD／大單買／賣比。"""
    empty = {
        "bar_buy_ratio": 0.0,
        "bar_cvd": 0.0,
        "bar_large_buy_ratio": 0.0,
        "bar_large_sell_ratio": 0.0,
        "bar_large_net_ratio": 0.0,
        "bar_tick_volume": 0.0,
    }
    if ticks is None or ticks.empty:
        return empty
    t = ticks.copy()
    t["date"] = pd.to_datetime(t["date"], format="mixed")
    w = t[(t["date"] >= win_start) & (t["date"] < win_end)]
    if w.empty:
        return empty
    vol = w["volume"].astype(float)
    total = float(vol.sum())
    if total <= 0:
        return empty
    buy = float(vol[w["tick_type"] == 1].sum())
    sell = float(vol[w["tick_type"] != 1].sum())
    large_buy = float(vol[(w["tick_type"] == 1) & (vol > TICK_LARGE_LOT)].sum())
    large_sell = float(vol[(w["tick_type"] != 1) & (vol > TICK_LARGE_LOT)].sum())
    lb = large_buy / total
    ls = large_sell / total
    return {
        "bar_buy_ratio": round(buy / total, 4),
        "bar_cvd": round(buy - sell, 2),
        "bar_large_buy_ratio": round(lb, 4),
        "bar_large_sell_ratio": round(ls, 4),
        "bar_large_net_ratio": round(lb - ls, 4),
        "bar_tick_volume": total,
    }


def _agg_roll_ohlc(window: pd.DataFrame) -> dict | None:
    """連續 N 根 M1 → 合成 OHLC。"""
    o = float(window.iloc[0]["open"])
    c = float(window.iloc[-1]["close"])
    h = float(window["high"].astype(float).max())
    l = float(window["low"].astype(float).min())
    if not all(np.isfinite([o, h, l, c])):
        return None
    if c <= o:
        return None
    upper, lower, body = _shadow_ratios(o, h, l, c)
    if body < MIN_BODY_RATIO or upper > MAX_UPPER_SHADOW_RATIO:
        return None
    win_start = pd.Timestamp(window.iloc[0]["date"])
    last_min = pd.Timestamp(window.iloc[-1]["date"])
    win_end = last_min + pd.Timedelta(minutes=1)
    return {
        "win_start": win_start,
        "win_end": win_end,
        "last_min": last_min,
        "trigger_ts": win_end,
        "entry_price": c,
        "body_ratio": round(body, 4),
        "lower_shadow_ratio": round(lower, 4),
        "upper_shadow_ratio": round(upper, 4),
    }


def _first_rolling_solid_m5(m1_day: pd.DataFrame) -> dict | None:
    """09:10～10:00 內，第一個「連續 5 根 M1 合成實體陽線」視窗。"""
    if m1_day is None or m1_day.empty:
        return None
    m1 = m1_day.dropna(subset=["open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if len(m1) < ROLL_MINUTES:
        return None

    t0 = dtime(*SESSION_START)
    t1 = dtime(*SESSION_END)

    for i in range(ROLL_MINUTES - 1, len(m1)):
        window = m1.iloc[i - (ROLL_MINUTES - 1) : i + 1]
        dates = pd.to_datetime(window["date"], format="mixed")
        deltas = dates.diff().dt.total_seconds().iloc[1:]
        if not (deltas == 60).all():
            continue
        agg = _agg_roll_ohlc(window)
        if agg is None:
            continue
        end_tm = agg["win_end"].time()
        if end_tm < t0 or end_tm > t1:
            continue
        return agg
    return None


def _label_stats(targets: list[int]) -> dict:
    if not targets:
        return {
            "n": 0,
            "n_sl": 0,
            "n_flat": 0,
            "n_tp": 0,
            "pct_sl": 0.0,
            "pct_flat": 0.0,
            "pct_tp": 0.0,
            "win_all": 0.0,
            "win_dec": 0.0,
            "decisive": 0,
            "er": 0.0,
        }
    arr = np.asarray(targets, dtype=int)
    n = len(arr)
    n_sl = int((arr == 0).sum())
    n_flat = int((arr == 1).sum())
    n_tp = int((arr == 2).sum())
    decisive = n_tp + n_sl
    return {
        "n": n,
        "n_sl": n_sl,
        "n_flat": n_flat,
        "n_tp": n_tp,
        "pct_sl": 100.0 * n_sl / n,
        "pct_flat": 100.0 * n_flat / n,
        "pct_tp": 100.0 * n_tp / n,
        "win_all": 100.0 * n_tp / n,
        "win_dec": 100.0 * n_tp / decisive if decisive else 0.0,
        "decisive": decisive,
        "er": (n_tp / n) * TP_PCT + (n_sl / n) * (-SL_PCT),
    }


def _print_outcome_row(label: str, st: dict) -> None:
    """進場總數 / 勝 / 敗 / 持平 / 勝率（勝÷進場總數）。"""
    print(
        f"{label:>10} {st['n']:>8} {st['n_tp']:>6} {st['n_sl']:>6} {st['n_flat']:>6} " f"{st['win_all']:>7.1f}%",
        flush=True,
    )


def _print_simple_table(
    ev: pd.DataFrame,
    title: str,
    col: str,
    thresholds: list[float],
    thr0_label: str,
) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    hdr = f"{'門檻':>10} {'進場總數':>8} {'勝':>6} {'敗':>6} {'持平':>6} {'勝率':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for thr in thresholds:
        sub = ev if thr <= 0 else ev[ev[col] >= thr]
        st = _label_stats(sub["target"].tolist() if not sub.empty else [])
        label = thr0_label if thr <= 0 else f"≥{thr:.0%}"
        _print_outcome_row(label, st)


def _print_direction_table(ev: pd.DataFrame, atr_label: str) -> None:
    """大買／大買>大賣／淨大單 三張簡表。"""
    thresholds = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    _print_simple_table(
        ev,
        f"大買門檻（{atr_label}；大買量/總量）",
        "bar_large_buy_ratio",
        thresholds,
        "M5only",
    )

    print("\n" + "=" * 64)
    print(f"大買門檻 且 大買>大賣（{atr_label}）")
    print("=" * 64)
    hdr = f"{'門檻':>10} {'進場總數':>8} {'勝':>6} {'敗':>6} {'持平':>6} {'勝率':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for thr in thresholds:
        sub = ev if thr <= 0 else ev[ev["bar_large_buy_ratio"] >= thr]
        if not sub.empty:
            sub = sub[sub["bar_large_buy_ratio"] > sub["bar_large_sell_ratio"]]
        st = _label_stats(sub["target"].tolist() if not sub.empty else [])
        label = "M5only" if thr <= 0 else f"≥{thr:.0%}"
        _print_outcome_row(label, st)

    _print_simple_table(
        ev,
        f"淨大單門檻（{atr_label}；大買占比−大賣占比）",
        "bar_large_net_ratio",
        thresholds,
        "any",
    )


def run(start_date: str, end_date: str, min_atr5: float = DEFAULT_MIN_ATR5) -> pd.DataFrame:
    t0 = time.time()
    print("滾動 5 分鐘實體陽線 + 窗內 tick 方向性驗證", flush=True)
    print(
        f"合成門檻: 連續 {ROLL_MINUTES} 根 M1、body≥{MIN_BODY_RATIO:.0%}、" f"上影≤{MAX_UPPER_SHADOW_RATIO:.0%}",
        flush=True,
    )
    print(
        f"決策窗: 窗結束落在 {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f"～{SESSION_END[0]:02d}:{SESSION_END[1]:02d}（起點可更早）",
        flush=True,
    )
    print(
        f"ATR5 過濾: atr5>= {min_atr5:.5f}（ATR(5)/當日開盤；0=不過濾）",
        flush=True,
    )
    print(f"標籤 ±{TP_PCT:.0%}/{LABEL_HORIZON_MINUTES}分；區間 {start_date} ~ {end_date}\n", flush=True)

    cand_start = (pd.Timestamp(start_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    cands = load_breakout_retest_day(start_date=cand_start, only_poc=False)
    if cands.empty:
        print("無 breakout_retest 候選")
        return pd.DataFrame()

    print(f"載入 pattern M1（start={cand_start})...", flush=True)
    m1 = load_pattern_m1(start_date=cand_start)
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    trading_days = np.array(sorted(m1["day_str"].unique()))
    print(f"M1 {len(m1):,} / days={len(trading_days)}", flush=True)

    rows: list[dict] = []
    n_cand = 0
    n_hit = 0
    n_atr_drop = 0
    for cand in cands.itertuples(index=False):
        sid = str(cand.stock_id)
        cdate = str(cand.candidate_date)[:10]
        trade_day = _next_trade_day(cdate, trading_days)
        if trade_day is None or trade_day < start_date or trade_day > end_date:
            continue
        n_cand += 1
        m1_day = m1[(m1["stock_id"].astype(str) == sid) & (m1["day_str"] == trade_day)]
        trig = _first_rolling_solid_m5(m1_day)
        if trig is None:
            continue
        n_hit += 1
        m1_atr = _add_atr5_day(m1_day)
        atr5 = _atr5_at(m1_atr, trig["last_min"])
        if min_atr5 > 0 and (not np.isfinite(atr5) or atr5 < min_atr5):
            n_atr_drop += 1
            continue
        try:
            ticks = load_tick_by_stock(sid, date=trade_day)
        except Exception:
            ticks = pd.DataFrame()
        tick_feat = _tick_in_window(ticks, trig["win_start"], trig["win_end"])
        label_raw = _triple_barrier_label(m1_day, trig["trigger_ts"], trig["entry_price"])
        if not np.isfinite(label_raw):
            continue
        rows.append(
            {
                "stock_id": sid,
                "candidate_date": cdate,
                "trade_date": trade_day,
                "win_start": trig["win_start"],
                "win_end": trig["win_end"],
                "trigger_ts": trig["trigger_ts"],
                "entry_price": trig["entry_price"],
                "body_ratio": trig["body_ratio"],
                "upper_shadow_ratio": trig["upper_shadow_ratio"],
                "atr5": atr5,
                **tick_feat,
                "target": {-1.0: 0, 0.0: 1, 1.0: 2}[float(label_raw)],
                "label_raw": float(label_raw),
            }
        )
        if (len(rows) % 50 == 0) and rows:
            print(f"  labeled={len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)

    ev = pd.DataFrame(rows)
    print(
        f"\n支撐線候選→區間進場日: {n_cand}；有滾動實體5分: {n_hit}；" f"ATR5剔除: {n_atr_drop}；標籤完整: {len(ev)}",
        flush=True,
    )
    if ev.empty:
        return ev

    sample = ev.head(5)[["stock_id", "trade_date", "win_start", "win_end", "atr5", "body_ratio"]]
    print("\n樣例視窗:")
    print(sample.to_string(index=False), flush=True)

    # ATR 門檻對照 + 各 ATR 下的大單買比表
    atr_levels = (0.0, 0.006, 0.008, 0.010) if min_atr5 <= 0 else (min_atr5,)
    print("\n" + "=" * 64)
    print("ATR5 門檻對照（尚未加大單條件）")
    print("=" * 64)
    hdr = f"{'min_atr5':>10} {'進場總數':>8} {'勝':>6} {'敗':>6} {'持平':>6} {'勝率':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for thr in atr_levels:
        sub = ev if thr <= 0 else ev[ev["atr5"] >= thr]
        st = _label_stats(sub["target"].tolist() if not sub.empty else [])
        _print_outcome_row(f"{thr:.3f}", st)

    # 各 ATR 下：大買／買>賣／淨大單
    sweep_atr = [t for t in atr_levels if t > 0] or [min_atr5]
    for atr_thr in sweep_atr:
        base = ev[ev["atr5"] >= atr_thr].copy()
        _print_direction_table(
            base,
            f"ATR5>={atr_thr:.5f} n={len(base)} 單筆>{TICK_LARGE_LOT}張",
        )

    print(
        f"\n說明: 勝率=勝/進場總數；勝=TP、敗=SL、持平=時間牆；" f"標籤 ±{TP_PCT:.0%}/{LABEL_HORIZON_MINUTES}分。",
        flush=True,
    )
    print(f"耗時 {time.time()-t0:.1f}s", flush=True)
    return ev


def main():
    p = argparse.ArgumentParser(description="滾動5分實體 + ATR5 + 窗內大單買賣對抗掃描")
    p.add_argument("--start_date", default="2026-07-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument(
        "--min_atr5",
        type=float,
        default=DEFAULT_MIN_ATR5,
        help="進場最低 atr5（ATR5/開盤）；0=不過濾。預設 0.01（同 mkt）",
    )
    args = p.parse_args()
    run(args.start_date, args.end_date, min_atr5=args.min_atr5)


if __name__ == "__main__":
    main()
