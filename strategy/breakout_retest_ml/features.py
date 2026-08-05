"""
breakout_retest_ml 特徵工程。

流程：
1. 日 K breakout_retest 硬過濾（支撐線＝前壓力轉支撐；不再要求 POC）→ 候選（以收盤日 D 為準）
2. 下一交易日 09:10~10:00，硬觸發：
   - M1 陽線實體 K（body_ratio 達門檻、非上影線主導）
   - Tick 方向：大單買比達門檻且 > 大單賣比（買賣對抗），可選 CVD>0
3. Triple Barrier 標籤（+1 止盈 / -1 止損 / 0 時間牆）→ target 0/1/2
"""

from __future__ import annotations

import sys
from datetime import time as dtime
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.query import (
    load_breakout_retest_day,
    load_breakout_retest_trigger,
    load_tick_by_stock,
)
from data.adjustment_query import load_pattern_day, load_pattern_m1, load_pattern_poc
from finmind.tick_universe import load_tick_universe
from pattern.breakout_retest.detector import BreakoutRetestDetector
from strategy.breakout_retest_ml.config import (
    LABEL_HORIZON_MINUTES,
    MAX_UPPER_SHADOW_RATIO,
    MIN_BODY_RATIO,
    MIN_PATTERN_SCORE,
    MIN_TICK_LARGE_BUY_RATIO,
    REQUIRE_CVD_POSITIVE,
    REQUIRE_LARGE_BUY_GT_SELL,
    SESSION_END,
    SESSION_START,
    SL_PCT,
    TICK_CVD_SECONDS,
    TICK_LARGE_BUY_SECONDS,
    TICK_LARGE_LOT,
    TP_PCT,
)

FEATURES = [
    "pattern_score",
    "poc_diff_pct",
    "dist_to_poc_pct",
    "dist_to_support_pct",
    "body_ratio",
    "lower_shadow_ratio",
    "upper_shadow_ratio",
    "volume_surge_ratio",
    "tick_large_buy_ratio",
    "tick_large_sell_ratio",
    "tick_large_net_ratio",
    "cvd_30s_delta",
]

# ═══════════════════════════════════════════════════════════════════════════
# K 線型態輔助
# ═══════════════════════════════════════════════════════════════════════════


def _shadow_ratios(o: float, h: float, l: float, c: float) -> tuple[float, float, float]:
    """回傳 (upper_shadow_ratio, lower_shadow_ratio, body_ratio)。全長為 0 時回 (0,0,0)。"""
    full = h - l
    if full <= 0:
        return 0.0, 0.0, 0.0
    upper = h - max(o, c)
    lower = min(o, c) - l
    body = abs(c - o)
    return float(upper / full), float(lower / full), float(body / full)


def _passes_m1_body_trigger(cur: pd.Series) -> tuple[bool, dict]:
    """M1 陽線實體 K：收 > 開、實體比例達門檻、上影線未過長。"""
    o = float(cur["open"])
    h = float(cur["high"])
    l = float(cur["low"])
    c = float(cur["close"])
    if not all(np.isfinite([o, h, l, c])):
        return False, {}
    if c <= o:
        return False, {}
    upper, lower, body = _shadow_ratios(o, h, l, c)
    ok = body >= MIN_BODY_RATIO and upper <= MAX_UPPER_SHADOW_RATIO
    return ok, {
        "body_ratio": round(body, 4),
        "lower_shadow_ratio": round(lower, 4),
        "upper_shadow_ratio": round(upper, 4),
    }


def _passes_tick_direction(tick_feat: dict) -> bool:
    """大單買賣對抗後定方向：買比達門檻、買>賣，且（可選）CVD>0。"""
    buy = float(tick_feat.get("tick_large_buy_ratio", 0.0) or 0.0)
    sell = float(tick_feat.get("tick_large_sell_ratio", 0.0) or 0.0)
    if buy < MIN_TICK_LARGE_BUY_RATIO:
        return False
    if REQUIRE_LARGE_BUY_GT_SELL and buy <= sell:
        return False
    if REQUIRE_CVD_POSITIVE and float(tick_feat.get("cvd_30s_delta", 0.0) or 0.0) <= 0:
        return False
    return True


def _volume_surge(cur_vol: float, hist_vols: np.ndarray) -> float:
    if hist_vols.size == 0:
        return 1.0
    avg = float(np.nanmean(hist_vols))
    if not np.isfinite(avg) or avg <= 0:
        return 1.0
    return float(cur_vol / avg)


# ═══════════════════════════════════════════════════════════════════════════
# 日 K 候選（支撐線＝前壓力；POC 非硬門檻）
# ═══════════════════════════════════════════════════════════════════════════


def _collapse_candidate_streaks(out: pd.DataFrame) -> pd.DataFrame:
    """滑窗連續命中只留每個區塊第一天，避免進場樣本灌水。"""
    if out.empty:
        return out
    out = out.sort_values(["stock_id", "candidate_date"]).reset_index(drop=True)
    keep = []
    prev_sid, prev_dt = None, None
    for _, r in out.iterrows():
        sid = r["stock_id"]
        dt = pd.Timestamp(r["candidate_date"])
        if sid != prev_sid or prev_dt is None or (dt - prev_dt).days > 5:
            keep.append(True)
        else:
            keep.append(False)
        prev_sid, prev_dt = sid, dt
    return out.loc[keep].reset_index(drop=True)


def find_day_candidates(
    day_df: pd.DataFrame,
    poc_df: pd.DataFrame,
    stock_ids: list[str] | None = None,
    min_score: float = MIN_PATTERN_SCORE,
    day_step: int = 1,
    require_poc: bool = False,
) -> pd.DataFrame:
    """對 tick_universe 掃描日 K breakout_retest。

    require_poc=False（預設）：保留所有達分的 breakout（支撐線＝前壓力）；
    True 時才只留 poc_confluence。
    回傳的 candidate_date 是型態成立的日 K 收盤日；實際盤中觸發在下一交易日。
    """
    detector = BreakoutRetestDetector()
    if stock_ids is None:
        stock_ids = load_tick_universe()

    rows: list[dict] = []
    n_stocks = len(stock_ids)
    for si, sid in enumerate(stock_ids):
        if (si + 1) % 50 == 0 or si == 0:
            print(f"  [candidates] {si + 1}/{n_stocks} stocks, hits={len(rows)}", flush=True)
        sub = day_df[day_df["stock_id"] == sid].sort_values("date").reset_index(drop=True)
        if len(sub) < detector.min_candles:
            continue
        stock_poc = poc_df[poc_df["stock_id"] == sid] if not poc_df.empty else poc_df
        for i in range(detector.min_candles - 1, len(sub), day_step):
            window = sub.iloc[max(0, i - detector.max_candles + 1) : i + 1].reset_index(drop=True)
            try:
                res = detector.detect(window, sid, "day", poc_df=stock_poc)
            except Exception:
                continue
            if res is None or res.score < min_score:
                continue
            poc_ok = bool(res.details.get("poc_confluence"))
            if require_poc and not poc_ok:
                continue
            rows.append(
                {
                    "stock_id": sid,
                    "candidate_date": str(res.date)[:10],
                    "pattern_score": float(res.score),
                    "resistance_price": float(res.details["resistance_price"]),
                    "matched_poc": (
                        float(res.details["matched_poc"]) if res.details.get("matched_poc") is not None else np.nan
                    ),
                    "poc_diff_pct": (
                        float(res.details["poc_diff_pct"]) if res.details.get("poc_diff_pct") is not None else np.nan
                    ),
                    "poc_confluence": poc_ok,
                }
            )

    cols = [
        "stock_id",
        "candidate_date",
        "pattern_score",
        "resistance_price",
        "matched_poc",
        "poc_diff_pct",
        "poc_confluence",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).drop_duplicates(subset=["stock_id", "candidate_date"], keep="last")
    return _collapse_candidate_streaks(out)


def detect_candidate_asof(
    day_df: pd.DataFrame,
    poc_df: pd.DataFrame,
    stock_id: str,
    asof_date: str,
    min_score: float = MIN_PATTERN_SCORE,
) -> dict | None:
    """以 asof_date（含）以前的日 K 偵測是否為 POC 共振候選。"""
    detector = BreakoutRetestDetector()
    sub = day_df[day_df["stock_id"] == stock_id].copy()
    if sub.empty:
        return None
    sub["date"] = pd.to_datetime(sub["date"], format="mixed")
    asof = pd.Timestamp(asof_date)
    sub = sub[sub["date"] <= asof].sort_values("date").reset_index(drop=True)
    if len(sub) < detector.min_candles:
        return None
    window = sub.iloc[-detector.max_candles :].reset_index(drop=True)
    stock_poc = poc_df[poc_df["stock_id"] == stock_id] if not poc_df.empty else poc_df
    try:
        res = detector.detect(window, stock_id, "day", poc_df=stock_poc)
    except Exception:
        return None
    if res is None or res.score < min_score:
        return None
    return {
        "stock_id": stock_id,
        "candidate_date": str(res.date)[:10],
        "pattern_score": float(res.score),
        "resistance_price": float(res.details["resistance_price"]),
        "matched_poc": (float(res.details["matched_poc"]) if res.details.get("matched_poc") is not None else np.nan),
        "poc_diff_pct": (float(res.details["poc_diff_pct"]) if res.details.get("poc_diff_pct") is not None else np.nan),
        "poc_confluence": bool(res.details.get("poc_confluence")),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tick 特徵
# ═══════════════════════════════════════════════════════════════════════════


def _tick_features_at(ticks: pd.DataFrame, trigger_ts: pd.Timestamp) -> dict:
    """觸發時刻前 CVD(30s) 與大單買／賣比(60s)。"""
    empty = {
        "tick_large_buy_ratio": 0.0,
        "tick_large_sell_ratio": 0.0,
        "tick_large_net_ratio": 0.0,
        "cvd_30s_delta": 0.0,
    }
    if ticks is None or ticks.empty:
        return empty

    t = ticks.copy()
    t["date"] = pd.to_datetime(t["date"], format="mixed")
    t = t[t["date"] <= trigger_ts]
    if t.empty:
        return empty

    cvd_start = trigger_ts - pd.Timedelta(seconds=TICK_CVD_SECONDS)
    cvd_win = t[t["date"] >= cvd_start]
    if cvd_win.empty:
        cvd_delta = 0.0
    else:
        buy = float(cvd_win.loc[cvd_win["tick_type"] == 1, "volume"].sum())
        sell = float(cvd_win.loc[cvd_win["tick_type"] != 1, "volume"].sum())
        cvd_delta = buy - sell

    lb_start = trigger_ts - pd.Timedelta(seconds=TICK_LARGE_BUY_SECONDS)
    lb_win = t[t["date"] >= lb_start]
    if lb_win.empty:
        large_buy_ratio = 0.0
        large_sell_ratio = 0.0
    else:
        total = float(lb_win["volume"].sum())
        vol = lb_win["volume"].astype(float)
        large_buy = float(vol[(lb_win["tick_type"] == 1) & (vol > TICK_LARGE_LOT)].sum())
        large_sell = float(vol[(lb_win["tick_type"] != 1) & (vol > TICK_LARGE_LOT)].sum())
        if total > 0:
            large_buy_ratio = large_buy / total
            large_sell_ratio = large_sell / total
        else:
            large_buy_ratio = 0.0
            large_sell_ratio = 0.0

    return {
        "tick_large_buy_ratio": round(large_buy_ratio, 4),
        "tick_large_sell_ratio": round(large_sell_ratio, 4),
        "tick_large_net_ratio": round(large_buy_ratio - large_sell_ratio, 4),
        "cvd_30s_delta": round(cvd_delta, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 盤中觸發 + 標籤
# ═══════════════════════════════════════════════════════════════════════════


def _triple_barrier_label(m1_day: pd.DataFrame, trigger_ts: pd.Timestamp, entry: float) -> float:
    """+1 先觸 TP、-1 先觸 SL、0 時間牆內都沒碰到。視窗不足 → NaN。"""
    fut = m1_day[
        (m1_day["date"] > trigger_ts) & (m1_day["date"] <= trigger_ts + pd.Timedelta(minutes=LABEL_HORIZON_MINUTES))
    ]
    if fut.empty or entry <= 0:
        return np.nan
    tp = entry * (1.0 + TP_PCT)
    sl = entry * (1.0 - SL_PCT)
    for _, row in fut.iterrows():
        if float(row["high"]) >= tp:
            return 1.0
        if float(row["low"]) <= sl:
            return -1.0
    if len(fut) < LABEL_HORIZON_MINUTES:
        return np.nan
    return 0.0


def collect_m1_body_triggers(
    m1_day: pd.DataFrame,
    resistance: float,
    matched_poc: float,
    ticks: pd.DataFrame | None = None,
    only_at: pd.Timestamp | None = None,
) -> list[dict]:
    """SESSION 內所有通過 M1 陽線實體 K 的分鐘，附帶 Tick 特徵（不套大單門檻）。

    給 db/breakout_retest_trigger 物化用；實驗端再 filter tick 門檻。
    """
    if m1_day is None or m1_day.empty:
        return []

    m1 = m1_day.sort_values("date").reset_index(drop=True).copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")

    t_start = dtime(*SESSION_START)
    t_end = dtime(*SESSION_END)

    indices = range(len(m1))
    if only_at is not None:
        only_at = pd.Timestamp(only_at)
        match = m1.index[m1["date"] == only_at]
        if len(match) == 0:
            return []
        indices = [int(match[0])]

    out: list[dict] = []
    for i in indices:
        ts = pd.Timestamp(m1.loc[i, "date"])
        tm = ts.time()
        if tm < t_start or tm >= t_end:
            continue

        cur = m1.iloc[i]
        ok, candle_feat = _passes_m1_body_trigger(cur)
        if not ok:
            continue

        tick_feat = _tick_features_at(ticks, ts)
        hist = m1["volume"].iloc[max(0, i - 5) : i].to_numpy(dtype=float)
        surge = _volume_surge(float(cur["volume"]), hist)
        close = float(cur["close"])
        poc = float(matched_poc) if np.isfinite(matched_poc) and matched_poc > 0 else close
        support = float(resistance) if resistance > 0 else close

        out.append(
            {
                "trigger_ts": ts,
                "trigger_tf": "m1",
                "entry_price": close,
                "body_ratio": candle_feat["body_ratio"],
                "lower_shadow_ratio": candle_feat["lower_shadow_ratio"],
                "upper_shadow_ratio": candle_feat["upper_shadow_ratio"],
                "volume_surge_ratio": round(surge, 4),
                "dist_to_poc_pct": round((close - poc) / poc * 100.0, 4) if poc else 0.0,
                "dist_to_support_pct": round((close - support) / support * 100.0, 4) if support else 0.0,
                "tick_large_buy_ratio": tick_feat["tick_large_buy_ratio"],
                "tick_large_sell_ratio": tick_feat["tick_large_sell_ratio"],
                "tick_large_net_ratio": tick_feat["tick_large_net_ratio"],
                "cvd_30s_delta": tick_feat["cvd_30s_delta"],
            }
        )
    return out


def find_intraday_trigger(
    m1_day: pd.DataFrame,
    resistance: float,
    matched_poc: float,
    ticks: pd.DataFrame | None = None,
    only_at: pd.Timestamp | None = None,
    require_tick: bool = True,
) -> dict | None:
    """在 SESSION 內找第一個硬觸發：M1 陽線實體 K（+ 可選 Tick 大單方向）。"""
    rows = collect_m1_body_triggers(
        m1_day, resistance=resistance, matched_poc=matched_poc, ticks=ticks, only_at=only_at
    )
    for row in rows:
        if require_tick and not _passes_tick_direction(row):
            continue
        return row
    return None


def filter_trigger_tick_hard(df: pd.DataFrame) -> pd.DataFrame:
    """對物化 trigger 表套用 Tick 大單買賣對抗硬過濾。"""
    if df is None or df.empty:
        return df
    out = df[df["tick_large_buy_ratio"] >= MIN_TICK_LARGE_BUY_RATIO]
    if REQUIRE_LARGE_BUY_GT_SELL:
        sell = out["tick_large_sell_ratio"] if "tick_large_sell_ratio" in out.columns else 0.0
        out = out[out["tick_large_buy_ratio"] > sell]
    if REQUIRE_CVD_POSITIVE:
        out = out[out["cvd_30s_delta"] > 0]
    return out.reset_index(drop=True)


def first_trigger_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """每個 (stock_id, trade_date) 取最早一根觸發。"""
    if df is None or df.empty:
        return df
    return (
        df.sort_values(["stock_id", "trade_date", "trigger_ts"])
        .drop_duplicates(subset=["stock_id", "trade_date"], keep="first")
        .reset_index(drop=True)
    )


def _next_trade_day(day_str: str, trading_days: np.ndarray) -> str | None:
    """回傳 day_str 之後的第一個交易日（字串 YYYY-MM-DD）。"""
    ds = np.asarray(trading_days)
    # trading_days 已排序的字串陣列
    idx = np.searchsorted(ds, day_str, side="right")
    if idx >= len(ds):
        return None
    return str(ds[idx])


def build_event_dataset(
    candidates: pd.DataFrame,
    m1_df: pd.DataFrame,
    start_date: str | None = None,
) -> pd.DataFrame:
    """由日 K 候選 → 下一交易日盤中觸發事件 + Tick 特徵 + Triple Barrier 標籤。"""
    if candidates.empty:
        return pd.DataFrame()

    m1_df = m1_df.copy()
    m1_df["date"] = pd.to_datetime(m1_df["date"], format="mixed")
    m1_df["day_str"] = m1_df["date"].dt.strftime("%Y-%m-%d")
    trading_days = np.array(sorted(m1_df["day_str"].unique()))

    events: list[dict] = []
    for _, cand in candidates.iterrows():
        sid = str(cand["stock_id"])
        cand_day = str(cand["candidate_date"])[:10]
        trade_day = _next_trade_day(cand_day, trading_days)
        if trade_day is None:
            continue
        if start_date and trade_day < start_date:
            continue

        m1_day = m1_df[(m1_df["stock_id"] == sid) & (m1_df["day_str"] == trade_day)].sort_values("date")
        if m1_day.empty or len(m1_day) < 15:
            continue

        try:
            ticks = load_tick_by_stock(sid, date=trade_day)
        except Exception:
            ticks = pd.DataFrame()

        trigger = find_intraday_trigger(
            m1_day.reset_index(drop=True),
            resistance=float(cand["resistance_price"]),
            matched_poc=float(cand["matched_poc"]) if pd.notna(cand["matched_poc"]) else np.nan,
            ticks=ticks,
        )
        if trigger is None:
            continue

        label_raw = _triple_barrier_label(m1_day, trigger["trigger_ts"], trigger["entry_price"])
        if not np.isfinite(label_raw):
            continue
        # -1/0/+1 → 0/1/2（止損 / 震盪 / 止盈）
        target = {-1.0: 0, 0.0: 1, 1.0: 2}[float(label_raw)]

        events.append(
            {
                "stock_id": sid,
                "date": trigger["trigger_ts"],
                "candidate_date": cand_day,
                "trade_date": trade_day,
                "pattern_score": float(cand["pattern_score"]),
                "poc_diff_pct": float(cand["poc_diff_pct"]) if pd.notna(cand["poc_diff_pct"]) else 0.0,
                "resistance_price": float(cand["resistance_price"]),
                "matched_poc": float(cand["matched_poc"]) if pd.notna(cand["matched_poc"]) else np.nan,
                "entry_price": trigger["entry_price"],
                "close": trigger["entry_price"],
                "trigger_tf": trigger["trigger_tf"],
                "body_ratio": trigger["body_ratio"],
                "lower_shadow_ratio": trigger["lower_shadow_ratio"],
                "upper_shadow_ratio": trigger["upper_shadow_ratio"],
                "volume_surge_ratio": trigger["volume_surge_ratio"],
                "dist_to_poc_pct": trigger["dist_to_poc_pct"],
                "dist_to_support_pct": trigger["dist_to_support_pct"],
                "tick_large_buy_ratio": trigger["tick_large_buy_ratio"],
                "tick_large_sell_ratio": trigger.get("tick_large_sell_ratio", 0.0),
                "tick_large_net_ratio": trigger.get(
                    "tick_large_net_ratio",
                    float(trigger["tick_large_buy_ratio"]) - float(trigger.get("tick_large_sell_ratio", 0.0) or 0.0),
                ),
                "cvd_30s_delta": trigger["cvd_30s_delta"],
                "target": target,
                "label_raw": float(label_raw),
            }
        )

    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values(["date", "stock_id"]).reset_index(drop=True)


def build_event_dataset_from_triggers(
    triggers: pd.DataFrame,
    m1_df: pd.DataFrame,
    start_date: str | None = None,
) -> pd.DataFrame:
    """從物化 trigger 表（已套 Tick 硬過濾、每日第一根）+ M1 打 Triple Barrier 標籤。"""
    if triggers is None or triggers.empty:
        return pd.DataFrame()

    m1_df = m1_df.copy()
    m1_df["date"] = pd.to_datetime(m1_df["date"], format="mixed")
    m1_df["day_str"] = m1_df["date"].dt.strftime("%Y-%m-%d")

    events: list[dict] = []
    for _, tr in triggers.iterrows():
        sid = str(tr["stock_id"])
        trade_day = str(tr["trade_date"])[:10]
        if start_date and trade_day < start_date:
            continue
        trigger_ts = pd.Timestamp(tr["trigger_ts"])
        entry = float(tr["entry_price"])
        m1_day = m1_df[(m1_df["stock_id"] == sid) & (m1_df["day_str"] == trade_day)]
        label_raw = _triple_barrier_label(m1_day, trigger_ts, entry)
        if not np.isfinite(label_raw):
            continue
        target = {-1.0: 0, 0.0: 1, 1.0: 2}[float(label_raw)]
        events.append(
            {
                "stock_id": sid,
                "date": trigger_ts,
                "candidate_date": str(tr["candidate_date"])[:10],
                "trade_date": trade_day,
                "pattern_score": float(tr["pattern_score"]),
                "poc_diff_pct": float(tr["poc_diff_pct"]) if pd.notna(tr["poc_diff_pct"]) else 0.0,
                "resistance_price": float(tr["resistance_price"]),
                "matched_poc": float(tr["matched_poc"]) if pd.notna(tr["matched_poc"]) else np.nan,
                "entry_price": entry,
                "close": entry,
                "trigger_tf": "m1",
                "body_ratio": float(tr["body_ratio"]),
                "lower_shadow_ratio": float(tr["lower_shadow_ratio"]),
                "upper_shadow_ratio": float(tr["upper_shadow_ratio"]),
                "volume_surge_ratio": float(tr["volume_surge_ratio"]),
                "dist_to_poc_pct": float(tr["dist_to_poc_pct"]),
                "dist_to_support_pct": float(tr["dist_to_support_pct"]),
                "tick_large_buy_ratio": float(tr["tick_large_buy_ratio"]),
                "tick_large_sell_ratio": (
                    float(tr["tick_large_sell_ratio"])
                    if "tick_large_sell_ratio" in tr.index and pd.notna(tr["tick_large_sell_ratio"])
                    else 0.0
                ),
                "tick_large_net_ratio": (
                    float(tr["tick_large_net_ratio"])
                    if "tick_large_net_ratio" in tr.index and pd.notna(tr["tick_large_net_ratio"])
                    else float(tr["tick_large_buy_ratio"])
                    - (
                        float(tr["tick_large_sell_ratio"])
                        if "tick_large_sell_ratio" in tr.index and pd.notna(tr["tick_large_sell_ratio"])
                        else 0.0
                    )
                ),
                "cvd_30s_delta": float(tr["cvd_30s_delta"]),
                "target": target,
                "label_raw": float(label_raw),
            }
        )
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values(["date", "stock_id"]).reset_index(drop=True)


def make_features(
    start_date: str | None = "2025-07-01",
    day_step: int = 1,
    stock_ids: list[str] | None = None,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """端到端：優先讀 db 物化表 → Tick 硬過濾 → 標籤；缺表時 fallback 現場掃描。"""
    if stock_ids is None:
        stock_ids = load_tick_universe()

    # ── 快路徑：db/breakout_retest_trigger ────────────────────────────────
    triggers = load_breakout_retest_trigger(start_date=start_date)
    if not triggers.empty:
        print(
            f"[breakout_retest_ml] 讀 db/breakout_retest_trigger：{len(triggers):,} 列",
            flush=True,
        )
        entries = first_trigger_per_day(filter_trigger_tick_hard(triggers))
        print(f"[breakout_retest_ml] Tick 硬過濾後每日首觸發：{len(entries):,}", flush=True)
        if entries.empty:
            return pd.DataFrame()
        print("[breakout_retest_ml] 載入 M1 打標籤...", flush=True)
        m1_df = load_pattern_m1(start_date=start_date)
        m1_df = m1_df[m1_df["stock_id"].isin(entries["stock_id"].unique())].copy()
        events = build_event_dataset_from_triggers(entries, m1_df, start_date=start_date)
        print(f"[breakout_retest_ml] 事件數: {len(events)}", flush=True)
        return events

    # ── fallback：現場掃描（慢）──────────────────────────────────────────
    print(
        "[breakout_retest_ml] 無 trigger 物化表，fallback 現場掃描 "
        "（建議先 python -m data.build_breakout_retest_day / trigger）...",
        flush=True,
    )
    if candidates is None:
        day_cands = load_breakout_retest_day(start_date=start_date, only_poc=False)
        if not day_cands.empty:
            print(f"[breakout_retest_ml] 讀 db/breakout_retest_day：{len(day_cands):,}", flush=True)
            candidates = day_cands
        else:
            print("[breakout_retest_ml] 現場掃日 K...", flush=True)
            day_df = load_pattern_day(start_date=start_date)
            day_df = day_df[day_df["stock_id"].isin(stock_ids)].copy()
            day_df["date"] = pd.to_datetime(day_df["date"], format="mixed")
            poc_df = load_pattern_poc(start_date=start_date)
            if not poc_df.empty:
                poc_df = poc_df[poc_df["stock_id"].isin(stock_ids)]
            candidates = find_day_candidates(day_df, poc_df, stock_ids=stock_ids, day_step=day_step, require_poc=False)

    print(f"[breakout_retest_ml] 候選日數: {len(candidates)}", flush=True)
    if candidates.empty:
        return pd.DataFrame()

    m1_df = load_pattern_m1(start_date=start_date)
    m1_df = m1_df[m1_df["stock_id"].isin(stock_ids)].copy()
    events = build_event_dataset(candidates, m1_df, start_date=start_date)
    print(f"[breakout_retest_ml] 事件數: {len(events)}", flush=True)
    return events
