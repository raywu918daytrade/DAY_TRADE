"""
一次性假設驗證：limitup_fade_ml 純規則（不套 ML）基準表現。

在投入完整 LightGBM pipeline 之前，先確認「漲停隔日開高 + 首根3分K下跌 → 做空」
這個純規則觸發後的樣本數、三類別（止盈/震盪/止損）比例、平均報酬夠不夠支撐往下
做模型訓練。

用法：
    python -m strategy.limitup_fade_ml.experiments.verify_rule_baseline --start_date 2022-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from strategy.limitup_fade_ml.dataset import build_events

_TARGET_NAMES = {0: "止損", 1: "震盪", 2: "止盈"}


def run(start_date: str | None = "2022-01-01") -> None:
    events = build_events(start_date=start_date)
    if events.empty:
        print("無符合條件的事件，規則可能太嚴或資料範圍不夠")
        return

    n = len(events)
    n_stocks = events["stock_id"].nunique()
    date_min = events["trigger_ts"].min()
    date_max = events["trigger_ts"].max()

    print(f"\n{'=' * 60}")
    print(f"limitup_fade_ml 規則基準（{start_date} ~ 今）")
    print(f"{'=' * 60}")
    print(f"事件數        : {n:,}")
    print(f"涉及股票數    : {n_stocks:,}")
    print(f"日期範圍      : {date_min} ~ {date_max}")

    print("\n三類別分佈:")
    dist = events["target"].value_counts(normalize=True).sort_index() * 100
    for cls, pct in dist.items():
        cnt = int((events["target"] == cls).sum())
        print(f"  {_TARGET_NAMES.get(cls, cls):6s}: {cnt:>5,} ({pct:.2f}%)")

    # 空單報酬率：entry 賣、exit 買回，價格下跌才賺
    pnl_pct = (events["entry_price"] - events["exit_price"]) / events["entry_price"] * 100
    win_rate = (events["target"] == 2).mean() * 100
    loss_rate = (events["target"] == 0).mean() * 100

    print("\n純規則（無ML過濾）表現:")
    print(f"  止盈率（>=+3%）: {win_rate:.2f}%")
    print(f"  止損率（<=-3%）: {loss_rate:.2f}%")
    print(f"  平均報酬        : {pnl_pct.mean():.3f}%")
    print(f"  中位數報酬      : {pnl_pct.median():.3f}%")
    print(f"  報酬標準差      : {pnl_pct.std():.3f}%")

    print("\n出場原因分佈:")
    print(events["exit_reason"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="limitup_fade_ml 規則基準驗證")
    parser.add_argument("--start_date", type=str, default="2022-01-01")
    args = parser.parse_args()
    run(start_date=args.start_date)
