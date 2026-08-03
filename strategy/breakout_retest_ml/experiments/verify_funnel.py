"""
驗證 breakout_retest_ml 硬過濾漏斗（不含訓練標籤）。

硬過濾（由寬到窄；通過第 5 階 = 規則可進場）：
  1. breakout_retest：日 K「突破後拉回」
  2. 有支撐線（型態成立＝前壓力轉支撐）
  3. 隔日 09:10～10:00 有可用 M1（trigger 物化時已隱含）
  4. M1 陽線實體 K
  5. Tick 大單買＞大單賣（買賣對抗後定方向）

預設讀 db/breakout_retest_day + db/breakout_retest_trigger（秒級）。
缺表或加 --rebuild 才打原始資料重建。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_funnel
    python -m strategy.breakout_retest_ml.experiments.verify_funnel --labels --save
    python -m strategy.breakout_retest_ml.experiments.verify_funnel --compare-tick
    python -m strategy.breakout_retest_ml.experiments.verify_funnel --compare-tick --start_date 2026-07-01 --end_date 2026-07-31
    python -m strategy.breakout_retest_ml.experiments.verify_funnel --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from data.query import (
    load_breakout_retest_day,
    load_breakout_retest_trigger,
)
from data.adjustment_query import load_pattern_m1
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
    TP_PCT,
    prepared_cache_path,
)
from strategy.breakout_retest_ml.features import (
    FEATURES,
    _triple_barrier_label,
    filter_trigger_tick_hard,
    first_trigger_per_day,
)

FUNNEL_STEPS = """
硬過濾（5 階；第 5 階 = 規則進場）
  1. breakout_retest     日K 突破後拉回（score≥{min_score}）
  2. 有支撐線            型態成立（前壓力轉支撐；不再要求 POC）
  3. 隔日 session M1     下一交易日有足夠分K（進物化流程）
  4. M1 陽線實體K        body≥{body:.0%} 且上影線≤{upper:.0%}
  5. Tick 大單方向       大買≥{tick:.0%} 且大買>大賣={dom} 且 CVD>0={cvd}
""".strip()


def _print_conditions() -> None:
    print(
        FUNNEL_STEPS.format(
            min_score=MIN_PATTERN_SCORE,
            body=MIN_BODY_RATIO,
            upper=MAX_UPPER_SHADOW_RATIO,
            tick=MIN_TICK_LARGE_BUY_RATIO,
            dom=REQUIRE_LARGE_BUY_GT_SELL,
            cvd=REQUIRE_CVD_POSITIVE,
        )
    )
    print(
        f"session = {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f" ~ {SESSION_END[0]:02d}:{SESSION_END[1]:02d}（進場）；持有可過 10:00"
    )
    print(
        f"標籤 Triple Barrier = ±{TP_PCT:.0%}/{LABEL_HORIZON_MINUTES}分"
        f"（先觸先平；加 --labels / --compare-tick 才印）\n"
    )


def _ensure_db(start_date: str, rebuild: bool) -> None:
    day = load_breakout_retest_day(start_date=start_date)
    if rebuild or day.empty:
        print("[rebuild] build_breakout_retest_day...", flush=True)
        from data.build_breakout_retest_day import build as build_day

        build_day(force=True, start_date=start_date)

    trig = load_breakout_retest_trigger(start_date=start_date)
    if rebuild or trig.empty:
        print("[rebuild] build_breakout_retest_trigger...", flush=True)
        from data.build_breakout_retest_trigger import build as build_trig

        # force 清空由 CLI --force 處理；這裡增量補齊
        if rebuild:
            out = Path(__file__).parent.parent.parent.parent / "db/breakout_retest_trigger"
            if out.exists():
                for f in out.glob("*.parquet"):
                    f.unlink()
        build_trig(force=False, start_date=start_date)


def _filter_day_candidates(entry: pd.DataFrame, cand_keys: set) -> pd.DataFrame:
    if entry is None or entry.empty:
        return entry if entry is not None else pd.DataFrame()
    return entry[
        entry.apply(
            lambda r: (str(r["stock_id"]), str(r["candidate_date"])[:10]) in cand_keys,
            axis=1,
        )
    ].reset_index(drop=True)


def _clip_date_col(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    col: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df
    s = out[col].astype(str).str[:10]
    if start_date:
        out = out[s >= start_date]
        s = out[col].astype(str).str[:10]
    if end_date:
        out = out[s <= end_date]
    return out.reset_index(drop=True)


def _label_entries(entry: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    """對進場列打 Triple Barrier 標籤；視窗不足則丟棄。"""
    if entry is None or entry.empty:
        return pd.DataFrame()

    rows = []
    for _, tr in entry.iterrows():
        sid = str(tr["stock_id"])
        trade_day = str(tr["trade_date"])[:10]
        ts = pd.Timestamp(tr["trigger_ts"])
        m1_day = m1[(m1["stock_id"] == sid) & (m1["day_str"] == trade_day)]
        label_raw = _triple_barrier_label(m1_day, ts, float(tr["entry_price"]))
        if not np.isfinite(label_raw):
            continue
        row = tr.to_dict()
        row["date"] = ts
        row["close"] = float(tr["entry_price"])
        row["target"] = {-1.0: 0, 0.0: 1, 1.0: 2}[float(label_raw)]
        row["label_raw"] = float(label_raw)
        rows.append(row)
    return pd.DataFrame(rows)


def _label_stats(ev: pd.DataFrame) -> dict:
    if ev is None or ev.empty or "target" not in ev.columns:
        return {
            "n": 0,
            "n_sl": 0,
            "n_flat": 0,
            "n_tp": 0,
            "win_all": 0.0,
            "win_dec": 0.0,
            "decisive": 0,
            "er": 0.0,
            "pct_sl": 0.0,
            "pct_flat": 0.0,
            "pct_tp": 0.0,
        }
    dist = ev["target"].value_counts()
    n = len(ev)
    n_sl = int(dist.get(0, 0))
    n_flat = int(dist.get(1, 0))
    n_tp = int(dist.get(2, 0))
    decisive = n_tp + n_sl
    return {
        "n": n,
        "n_sl": n_sl,
        "n_flat": n_flat,
        "n_tp": n_tp,
        "win_all": 100.0 * n_tp / n if n else 0.0,
        "win_dec": 100.0 * n_tp / decisive if decisive else 0.0,
        "decisive": decisive,
        "er": (n_tp / n) * TP_PCT + (n_sl / n) * (-SL_PCT) if n else 0.0,
        "pct_sl": 100.0 * n_sl / n if n else 0.0,
        "pct_flat": 100.0 * n_flat / n if n else 0.0,
        "pct_tp": 100.0 * n_tp / n if n else 0.0,
    }


def _print_label_stats(title: str, st: dict) -> None:
    print(f"\n{title}")
    print(f"  標籤完整: {st['n']}  （±{TP_PCT:.0%}/{LABEL_HORIZON_MINUTES}分）")
    print(f"  止損: {st['n_sl']:>5,}  ({st['pct_sl']:5.1f}%)")
    print(f"  震盪: {st['n_flat']:>5,}  ({st['pct_flat']:5.1f}%)")
    print(f"  止盈: {st['n_tp']:>5,}  ({st['pct_tp']:5.1f}%)")
    print(f"  止盈占比 TP/全部:           {st['win_all']:5.1f}%")
    print(f"  決勝負勝率 TP/(TP+SL):      {st['win_dec']:5.1f}%  （n={st['decisive']}）")
    print(f"  簡易期望 E[r]（flat=0）:    {st['er'] * 100:+.3f}% / 筆")


def _load_m1_for_entries(
    entry: pd.DataFrame,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    rng = f"{start_date}" + (f" ~ {end_date}" if end_date else "")
    print(f"載入 pattern M1（打標用；{rng}）...", flush=True)
    m1 = load_pattern_m1(start_date=start_date)
    if m1.empty or entry is None or entry.empty:
        return m1
    stock_ids = set(entry["stock_id"].astype(str))
    m1 = m1[m1["stock_id"].astype(str).isin(stock_ids)].copy()
    m1["date"] = pd.to_datetime(m1["date"], format="mixed")
    m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
    if end_date:
        m1 = m1[m1["day_str"] <= end_date]
    print(f"M1 篩選後 {len(m1):,} rows / stocks={len(stock_ids)}", flush=True)
    return m1


def run(
    start_date: str = "2025-07-01",
    end_date: str | None = None,
    rebuild: bool = False,
    save: bool = False,
    labels: bool = False,
    compare_tick: bool = False,
) -> pd.DataFrame:
    _print_conditions()
    if end_date:
        print(f"區間: {start_date} ~ {end_date}\n", flush=True)
    _ensure_db(start_date, rebuild=rebuild)

    day_all = load_breakout_retest_day(start_date=start_date, only_poc=False)
    day_all = _clip_date_col(day_all, start_date, end_date, col="candidate_date")
    if day_all.empty:
        day_support = day_all
        n_poc_ref = 0
    else:
        day_support = day_all[day_all["resistance_price"].notna()]
        n_poc_ref = int((day_all["poc_confluence"] == True).sum())  # noqa: E712
    triggers = load_breakout_retest_trigger(start_date=start_date)
    triggers = _clip_date_col(triggers, start_date, end_date, col="trade_date")

    n1 = len(day_all)
    n2 = len(day_support)
    cand_keys: set = set()
    entry_a = pd.DataFrame()
    entry_b = pd.DataFrame()

    if triggers.empty:
        n3 = n4 = n5 = 0
        entry = triggers
    else:
        cand_keys = set(
            zip(day_support["stock_id"].astype(str), day_support["candidate_date"].astype(str).str[:10])
        )
        trig_keys = set(zip(triggers["stock_id"].astype(str), triggers["candidate_date"].astype(str).str[:10]))
        body_keys = trig_keys & cand_keys
        n3 = n2
        n4 = len(body_keys)

        # A：不過 tick；B：現行 tick 硬過濾（皆每日首觸發）
        entry_a = _filter_day_candidates(first_trigger_per_day(triggers), cand_keys)
        entry_b = _filter_day_candidates(
            first_trigger_per_day(filter_trigger_tick_hard(triggers)),
            cand_keys,
        )
        entry_a = _clip_date_col(entry_a, start_date, end_date, col="trade_date")
        entry_b = _clip_date_col(entry_b, start_date, end_date, col="trade_date")
        entry = entry_b
        n5 = len(entry_b)
        print(f"（參考）有支撐線候選中無任何 M1 實體K 觸發列: {n2 - n4}", flush=True)
        print(f"（參考）其中 poc_confluence=True: {n_poc_ref:,}（已非硬門檻）", flush=True)

    print("\n" + "=" * 60)
    print("硬過濾漏斗計數（來源：db/）")
    print("=" * 60)
    print(f"  1. breakout_retest（突破後拉回）     {n1:>6,}")
    print(f"  2. + 有支撐線（型態成立）             {n2:>6,}")
    print(f"  3. + 隔日有可用 M1                   {n3:>6,}   （Layer2 母體＝全部達分候選）")
    print(f"  4. + M1 陽線實體 K（候選日數）       {n4:>6,}")
    print(f"  5. + Tick 大買>大賣  ← 規則進場       {n5:>6,}")
    if not triggers.empty:
        print(f"     （trigger 快照列數，含同日多根實體K: {len(triggers):,}）")
        if compare_tick:
            print(f"     A 無 tick 每日首觸發: {len(entry_a):,}")
            print(f"     B 有 tick 每日首觸發: {len(entry_b):,}")

    if compare_tick:
        if entry_a.empty:
            print("\n無進場列，略過 tick 對照")
            return entry_a
        m1 = _load_m1_for_entries(entry_a, start_date, end_date=end_date)
        print("\n打標籤中（A 無 tick / B 有 tick）...", flush=True)
        ev_a = _label_entries(entry_a, m1)
        ev_b = _label_entries(entry_b, m1)
        st_a = _label_stats(ev_a)
        st_b = _label_stats(ev_b)
        print("\n" + "=" * 60)
        print("Tick 方向性對照（母體＝硬過濾 1–4 每日首根實體K）")
        print("=" * 60)
        _print_label_stats("A. 無 tick 硬過濾", st_a)
        _print_label_stats(
            f"B. 有 tick（大買≥{MIN_TICK_LARGE_BUY_RATIO:.0%} 且大買>大賣 且 CVD>0={REQUIRE_CVD_POSITIVE}）",
            st_b,
        )
        print("\nB − A")
        print(f"  決勝負勝率差:  {st_b['win_dec'] - st_a['win_dec']:+.1f} pp")
        print(f"  止盈占比差:    {st_b['win_all'] - st_a['win_all']:+.1f} pp")
        print(f"  E[r] 差:       {(st_b['er'] - st_a['er']) * 100:+.3f} pp / 筆")
        if st_b["win_dec"] <= st_a["win_dec"] + 2.0 and st_b["er"] <= st_a["er"] + 1e-6:
            print("  結論: 現行 tick 硬過濾未明顯優於無 tick（看不出方向性）")
        elif st_b["win_dec"] > st_a["win_dec"] + 2.0 or st_b["er"] > st_a["er"]:
            print("  結論: B 優於 A，現行 tick 定義可能帶有方向性")
        return ev_b

    if labels or save:
        if entry is None or entry.empty:
            print("\n無進場列，略過標籤")
            return entry if entry is not None else pd.DataFrame()
        m1 = _load_m1_for_entries(entry, start_date, end_date=end_date)
        print("\n打標籤中...", flush=True)
        ev = _label_entries(entry, m1)
        if not ev.empty:
            _print_label_stats("規則進場（含 tick）", _label_stats(ev))
        if save and not ev.empty:
            missing = [c for c in FEATURES if c not in ev.columns]
            if missing:
                print(f"缺少 FEATURES {missing}")
            else:
                path = prepared_cache_path(start_date)
                path.parent.mkdir(parents=True, exist_ok=True)
                ev.to_parquet(path)
                print(f"已存 prepared → {path}")
        return ev

    return entry if entry is not None else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description="breakout_retest_ml 硬過濾漏斗（讀 db）")
    p.add_argument("--start_date", default="2025-07-01")
    p.add_argument("--end_date", default=None, help="含當日；短區間摸底用")
    p.add_argument("--rebuild", action="store_true", help="強制重建 db 物化表")
    p.add_argument("--labels", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument(
        "--compare-tick",
        action="store_true",
        help="A=無 tick / B=有 tick 硬過濾，±3% 標籤對照方向性",
    )
    args = p.parse_args()
    run(
        start_date=args.start_date,
        end_date=args.end_date,
        rebuild=args.rebuild,
        save=args.save,
        labels=args.labels or args.save,
        compare_tick=args.compare_tick,
    )


if __name__ == "__main__":
    main()
