"""
ret5_pullback_ml 純規則三分類 TB 驗證（含 atr5≥p99 硬過濾）。

事件偵測與 dataset.build_events 共用；此腳本多套 atr5 門檻後印統計。

用法：
    python -m strategy.ret5_pullback_ml.verify \\
        --start_date 2026-01-01 --end_date 2026-07-31
    python -m strategy.ret5_pullback_ml.verify \\
        --start_date 2026-01-01 --end_date 2026-07-31 --use_tick_universe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from strategy.mkt.config import ATR5_FILTER_THRESHOLD
from strategy.ret5_pullback_ml.config import (
    ENTRY_DEADLINE,
    HOLD_M5_BARS,
    M1_REV_LOOKAHEAD,
    RET5_MIN,
    SIGNAL_DEADLINE,
    SL_PCT,
    TP_PCT,
)
from strategy.ret5_pullback_ml.dataset import build_events


def _summarize(df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print("  n=0", flush=True)
        return
    n_tp = int((df["label"] == 1.0).sum())
    n_flat = int((df["label"] == 0.0).sum())
    n_sl = int((df["label"] == -1.0).sum())
    print(f"  n={n:,}", flush=True)
    print(f"  止盈(+{TP_PCT:.0%}): {n_tp:,}  {100 * n_tp / n:.1f}%", flush=True)
    print(f"  持平: {n_flat:,}  {100 * n_flat / n:.1f}%", flush=True)
    print(f"  止損(-{SL_PCT:.0%}): {n_sl:,}  {100 * n_sl / n:.1f}%", flush=True)
    print(
        f"  做多 mean={100 * df['pnl_pct'].mean():.3f}%  "
        f"median={100 * df['pnl_pct'].median():.3f}%",
        flush=True,
    )


def run(start_date: str, end_date: str, use_tick_universe: bool = False) -> pd.DataFrame:
    print("ret5_pullback_ml.verify 純規則 TB（atr5≥p99）", flush=True)
    print(
        f"濾網: ret5紅K≥{RET5_MIN:.0%}；m5陰線 low>m5_1_low；"
        f"之後 {M1_REV_LOOKAHEAD} 根 m1 陽線量>前1且 close>該 m5 high；"
        f"m5下跌<{SIGNAL_DEADLINE.strftime('%H:%M')}；進場<{ENTRY_DEADLINE.strftime('%H:%M')}；"
        f"atr5≥{ATR5_FILTER_THRESHOLD:.5f}(p99)",
        flush=True,
    )
    print(
        f"標籤: 做多 TB ±{TP_PCT:.0%} / 最多 {HOLD_M5_BARS} 根 m5（30 分）\n",
        flush=True,
    )

    out = build_events(
        start_date=start_date,
        end_date=end_date,
        use_tick_universe=use_tick_universe,
        attach_features=False,
    )
    if out.empty:
        return out

    n_before = len(out)
    out = out[out["atr5"].notna() & (out["atr5"] >= ATR5_FILTER_THRESHOLD)].copy()
    print(f"atr5≥p99: {len(out):,} / {n_before:,}", flush=True)
    if out.empty:
        return out

    print(f"觸發交易日數: {out['day_str'].nunique()}", flush=True)
    print("\n" + "=" * 56)
    print("做多 TB 三分類（±3% / 最多 30 分）")
    print("=" * 56)
    _summarize(out)

    out = out.copy()
    out["year"] = pd.to_datetime(out["day_str"]).dt.year
    print("\n分年:", flush=True)
    for y, g in out.groupby("year"):
        n = len(g)
        print(
            f"  {y}: n={n}  days={g['day_str'].nunique()}  "
            f"TP={100 * (g['label'] == 1).mean():.1f}%  "
            f"flat={100 * (g['label'] == 0).mean():.1f}%  "
            f"SL={100 * (g['label'] == -1).mean():.1f}%  "
            f"mean={100 * g['pnl_pct'].mean():.3f}%",
            flush=True,
        )
    return out


def main():
    p = argparse.ArgumentParser(description="ret5_pullback_ml 純規則三分類驗證")
    p.add_argument("--start_date", default="2026-01-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument(
        "--use_tick_universe",
        action="store_true",
        help="改用 db/tickers/tick_universe.parquet（約 400 支）",
    )
    args = p.parse_args()
    run(args.start_date, args.end_date, use_tick_universe=args.use_tick_universe)


if __name__ == "__main__":
    main()
