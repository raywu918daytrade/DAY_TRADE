"""
規則基線：前日實體漲停 → 今開高 → 首 3 分跌 → 09:03 做空。

用法：
    python -m strategy.limitup_fade_ml.experiments.verify_rule_baseline \\
        --start_date 2026-06-01 --end_date 2026-07-31
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from strategy.limitup_fade_ml_my.dataset import build_events


def run(start_date: str, end_date: str) -> None:
    t0 = time.time()
    print("規則基線（m3 @ 09:03）", flush=True)
    # 先不建 TB，只看收盤勝率（對齊早期驗證）
    ev = build_events(start_date, end_date, with_labels=False)
    if ev.empty:
        print("n=0", flush=True)
        return
    n = len(ev)
    win = int(ev["short_win_close"].sum())
    print(
        f"n={n:,}  做空勝率(→日收)={100 * ev['short_win_close'].mean():.1f}%  "
        f"({win}/{n})  short mean={100 * ev['short_ret_to_close'].mean():.3f}%  "
        f"median={100 * ev['short_ret_to_close'].median():.3f}%",
        flush=True,
    )
    print(f"耗時 {time.time() - t0:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start_date", default="2026-06-01")
    p.add_argument("--end_date", default="2026-07-31")
    args = p.parse_args()
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
