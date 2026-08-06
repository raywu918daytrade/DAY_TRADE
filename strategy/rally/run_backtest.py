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
from strategy.rally.config import HOLD_BARS, SESSION_END, SESSION_START, SL_PCT, TP_PCT
from strategy.rally.predict import predict
from strategy.rally.train import load_model_by_type


def run(
    test_days: int = 10,
    threshold: float = 0.70,
    model_type: str = "xgb",
):
    """跑一次 rally 回測。model_type：要用哪個模型（rfc/xgb/lgbm），直接傳參數
    指定（2026-07-22 討論：即時交易現在是 rally_xgb/rally_lgbm 兩個獨立策略
    各自寫死模型，不再共用 config.MODEL_TYPE/RALLY_MODEL_TYPE 切換，回測這裡
    也直接改成參數輸入，不用再繞去改 .env）。

    2026-08-06：first_entry_time/last_entry_time 改傳 config.py 的
    SESSION_START/SESSION_END（跟即時交易同一份設定，"H:MM" 字串，
    backtest/intraday_backtest.py 用 pd.Timestamp(...).time() 解析，不用
    零填補也行），不然回測預設是 09:01~10:00，跟即時交易實際限定的 9:00~9:30
    對不上，回測結果會涵蓋實盤根本不會出手的時間，數字不可信。"""
    model = load_model_by_type(model_type)

    df_proba = predict(model=model, test_days=test_days)
    print(f"機率矩陣: {df_proba.shape}，非空值 {df_proba.notna().sum().sum()} 筆")

    portfolio_df, trades_df = run_backtest(
        df_proba,
        threshold=threshold,
        sl_pct=SL_PCT,
        tp_pct=TP_PCT,
        hold_bars=HOLD_BARS,
        first_entry_time=SESSION_START,
        last_entry_time=SESSION_END,
    )
    print()
    print_trades(trades_df)
    return portfolio_df, trades_df


if __name__ == "__main__":
    run(
        test_days=30,
        threshold=0.6,
        model_type="lgbm",
    )
