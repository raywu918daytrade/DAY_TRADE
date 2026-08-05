"""
vwap_dl 策略回測入口 — 串接 predict.py 跟共用回測引擎 backtest/intraday_platform.py。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.intraday_platform import print_trades, run_backtest
from finmind.tick_universe import load_tick_universe
from strategy.vwap_dl.config import BACKTEST_HOLD_BARS, BACKTEST_SL_PCT, BACKTEST_TP_PCT, SESSION_END, SESSION_START
from strategy.vwap_dl.predict import predict
from strategy.vwap_dl.train import load_model


def run(
    test_days: int = 30,
    threshold: float = 0.6,
    top_n: int = 5,
    max_positions: int = 10,
    start_date: str = "2024-01-01",
    batch_size: int = 256,
):
    """
    跑一次 vwap_dl 回測。

    共用引擎只吃單一（做多）機率矩陣，這裡回測的是 predict() 算出來的
    「做多」訊號（見 predict.py 的 _direction_probas() 說明）。
    """
    model = load_model()

    df_proba = predict(model=model, test_days=test_days, start_date=start_date, batch_size=batch_size)
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
        stock_ids=load_tick_universe(),
    )
    print()
    print_trades(trades_df)
    return portfolio_df, trades_df


if __name__ == "__main__":
    run(test_days=30, threshold=0.6, start_date="2024-01-01")
