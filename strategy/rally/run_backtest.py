"""
rally 策略回測入口 — 串接 strategy/rally/predict.py 跟共用回測引擎
backtest/intraday_platform.py（不改共用引擎本身，orb 也在用同一份）。

用法：
    python strategy/rally/run_backtest.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.intraday_platform import print_trades, run_backtest
from strategy.rally.config import HOLD_BARS, MODEL_TYPE, SL_PCT, TP_PCT
from strategy.rally.predict import predict
from strategy.rally.train import load_model_by_type


def run(
    test_days: int = 10,
    threshold: float = 0.70,
):
    """跑一次 rally 回測（模型由 config.MODEL_TYPE 決定）。"""
    model = load_model_by_type(MODEL_TYPE)

    df_proba = predict(model=model, test_days=test_days)
    print(f"機率矩陣: {df_proba.shape}，非空值 {df_proba.notna().sum().sum()} 筆")

    portfolio_df, trades_df = run_backtest(
        df_proba,
        threshold=threshold,
        sl_pct=SL_PCT,
        tp_pct=TP_PCT,
        hold_bars=HOLD_BARS,
    )
    print()
    print_trades(trades_df)
    return portfolio_df, trades_df


if __name__ == "__main__":
    run(test_days=30, threshold=0.70)
