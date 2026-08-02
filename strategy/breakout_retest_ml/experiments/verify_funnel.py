"""
驗證 breakout_retest_ml 硬過濾漏斗（不含訓練標籤）。

硬過濾（由寬到窄；通過第 5 階 = 規則可進場）：
  1. breakout_retest：日 K「突破後拉回」
  2. POC ≈ 支撐
  3. 隔日 09:10～10:00 有可用 M1（trigger 物化時已隱含）
  4. M1 陽線實體 K
  5. Tick 有方向大量買進

預設讀 db/breakout_retest_day + db/breakout_retest_trigger（秒級）。
缺表或加 --rebuild 才打原始資料重建。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_funnel
    python -m strategy.breakout_retest_ml.experiments.verify_funnel --labels --save
    python -m strategy.breakout_retest_ml.experiments.verify_funnel --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from data.query import (
    load_breakout_retest_day,
    load_breakout_retest_trigger,
)
from data.adjustment_query import load_pattern_m1
from strategy.breakout_retest_ml.config import (
    MAX_UPPER_SHADOW_RATIO,
    MIN_BODY_RATIO,
    MIN_PATTERN_SCORE,
    MIN_TICK_LARGE_BUY_RATIO,
    POC_CONFLUENCE_MAX_PCT,
    REQUIRE_CVD_POSITIVE,
    SESSION_END,
    SESSION_START,
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
  2. POC ≈ 支撐          poc_confluence（距離≤{poc_pct}%）
  3. 隔日 session M1     下一交易日有足夠分K（進物化流程）
  4. M1 陽線實體K        body≥{body:.0%} 且上影線≤{upper:.0%}
  5. Tick 大量買進       大單買比≥{tick:.0%} 且 CVD>0={cvd}
""".strip()


def _print_conditions() -> None:
    print(
        FUNNEL_STEPS.format(
            min_score=MIN_PATTERN_SCORE,
            poc_pct=POC_CONFLUENCE_MAX_PCT,
            body=MIN_BODY_RATIO,
            upper=MAX_UPPER_SHADOW_RATIO,
            tick=MIN_TICK_LARGE_BUY_RATIO,
            cvd=REQUIRE_CVD_POSITIVE,
        )
    )
    print(
        f"session = {SESSION_START[0]:02d}:{SESSION_START[1]:02d}"
        f" ~ {SESSION_END[0]:02d}:{SESSION_END[1]:02d}（進場）；持有可過 10:00"
    )
    print("（Triple Barrier 標籤 ≠ 硬過濾，加 --labels 才印）\n")


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


def run(
    start_date: str = "2025-07-01",
    rebuild: bool = False,
    save: bool = False,
    labels: bool = False,
) -> pd.DataFrame:
    _print_conditions()
    _ensure_db(start_date, rebuild=rebuild)

    day_all = load_breakout_retest_day(start_date=start_date, only_poc=False)
    day_poc = day_all[day_all["poc_confluence"] == True] if not day_all.empty else day_all  # noqa: E712
    triggers = load_breakout_retest_trigger(start_date=start_date)

    n1 = len(day_all)
    n2 = len(day_poc)
    # 條件3：有進 trigger 建置且至少有 trade_date（用 candidate 對得上的 trade 覆蓋）
    if triggers.empty:
        n3 = n4 = n5 = 0
        entry = triggers
    else:
        cand_keys = set(zip(day_poc["stock_id"].astype(str), day_poc["candidate_date"].astype(str).str[:10]))
        trig_keys = set(zip(triggers["stock_id"].astype(str), triggers["candidate_date"].astype(str).str[:10]))
        # 有 M1 實體K 的候選日
        body_keys = trig_keys & cand_keys
        n4_cands = len(body_keys)
        # 粗估：有進物化且有任一 trigger 列的候選 ≈ 條件3∩4；無 body 的候選可能是無 M1 或無實體K
        # 條件3：day_poc 裡 trade 當天有出現在 trigger 的 stock+candidate，或我們用「有嘗試」不好追
        # 簡化：n3 = 至少能對到 trigger 表或（無任何 trigger 時）無法區分 → 用 n4 的母體
        # 更好：n3 = day_poc 中存在同 stock 在 trade_date 有 m1 — 需讀 m1，慢。
        # 物化表語意：有 trigger 列 ⇒ 過了 3+4；n3 用「有 candidate 對應到至少一列 trigger」的數量作為「過了3且4」
        # 另報：day_poc 中完全沒有 trigger 列的數量 = 無 M1 或無實體K
        n3 = n2  # Layer2 建置時已要求有足夠 M1；缺 M1 的不會進 trigger，下面用缺口說明
        n4 = n4_cands
        tick_df = filter_trigger_tick_hard(triggers)
        entry = first_trigger_per_day(tick_df)
        # 只計 day_poc 對應
        if not entry.empty:
            entry = entry[
                entry.apply(
                    lambda r: (str(r["stock_id"]), str(r["candidate_date"])[:10]) in cand_keys,
                    axis=1,
                )
            ]
        n5 = len(entry)
        n_no_body = n2 - n4
        print(f"（參考）POC 候選中無任何 M1 實體K 觸發列: {n_no_body}", flush=True)

    print("\n" + "=" * 60)
    print("硬過濾漏斗計數（來源：db/）")
    print("=" * 60)
    print(f"  1. breakout_retest（突破後拉回）     {n1:>6,}")
    print(f"  2. + POC ≈ 支撐                      {n2:>6,}")
    print(f"  3. + 隔日有可用 M1                   {n3:>6,}   （Layer2 母體＝POC 候選）")
    print(f"  4. + M1 陽線實體 K（候選日數）       {n4:>6,}")
    print(f"  5. + Tick 大量買進  ← 規則進場       {n5:>6,}")
    if not triggers.empty:
        print(f"     （trigger 快照列數，含同日多根實體K: {len(triggers):,}）")

    if labels or save:
        if entry is None or entry.empty:
            print("\n無進場列，略過標籤")
            return entry if entry is not None else pd.DataFrame()
        print("\n打標籤中...", flush=True)
        m1 = load_pattern_m1(start_date=start_date)
        m1["date"] = pd.to_datetime(m1["date"], format="mixed")
        m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
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
        ev = pd.DataFrame(rows)
        if not ev.empty and "target" in ev.columns:
            dist = ev["target"].value_counts().sort_index()
            pct = (ev["target"].value_counts(normalize=True).sort_index() * 100).round(1)
            print(f"標籤完整: {len(ev)}")
            for k, name in [(0, "止損"), (1, "震盪"), (2, "止盈")]:
                print(f"  {name}: {int(dist.get(k, 0)):>5,}  ({float(pct.get(k, 0)):5.1f}%)")
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
    p.add_argument("--rebuild", action="store_true", help="強制重建 db 物化表")
    p.add_argument("--labels", action="store_true")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    run(
        start_date=args.start_date,
        rebuild=args.rebuild,
        save=args.save,
        labels=args.labels or args.save,
    )


if __name__ == "__main__":
    main()
