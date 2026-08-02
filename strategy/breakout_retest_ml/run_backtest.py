"""
breakout_retest_ml 回測入口 — 串接 predict() 與 backtest/intraday_platform。

用法：
    python strategy/breakout_retest_ml/run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.intraday_platform import print_trades, run_backtest
from strategy.breakout_retest_ml.config import (
    BACKTEST_HOLD_BARS,
    BACKTEST_SL_PCT,
    BACKTEST_TP_PCT,
    SESSION_END,
    SESSION_START,
)
from strategy.breakout_retest_ml.predict import predict
from strategy.breakout_retest_ml.train import load_model_by_type


def run(
    test_days: int = 30,
    threshold: float = 0.6,
    top_n: int = 5,
    max_positions: int = 10,
    model_type: str = "lgbm",
    start_date: str | None = "2025-07-01",
    use_cache: bool = True,
):
    model = load_model_by_type(model_type)
    df_proba = predict(model=model, test_days=test_days, start_date=start_date, use_cache=use_cache)
    print(f"機率矩陣: {df_proba.shape}，非空值 {df_proba.notna().sum().sum()} 筆")

    portfolio_df, trades_df = run_backtest(
        df_proba,
        threshold=threshold,
        top_n=top_n,
        max_positions=max_positions,
        sl_pct=BACKTEST_SL_PCT,
        tp_pct=BACKTEST_TP_PCT,
        hold_bars=BACKTEST_HOLD_BARS,
        first_entry_time=f"{SESSION_START[0]:02d}:{SESSION_START[1]:02d}",
        last_entry_time=f"{SESSION_END[0]:02d}:{SESSION_END[1]:02d}",
    )
    print()
    print_trades(trades_df)
    return portfolio_df, trades_df


if __name__ == "__main__":
    run(test_days=30, threshold=0.6, start_date="2025-07-01")
