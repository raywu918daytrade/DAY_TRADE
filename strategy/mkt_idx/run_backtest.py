"""
mkt_idx 策略回測入口 — 串接 strategy/mkt_idx/predict.py 跟共用回測引擎
backtest/intraday_platform.py（不改共用引擎本身，rally/orb 也在用同一份）。

用法：
    python strategy/mkt_idx/run_backtest.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.intraday_platform import print_trades, run_backtest
from strategy.mkt_idx.config import HOLD_BARS, MODEL_TYPE, SL_PCT, TP_PCT
from strategy.mkt_idx.predict import predict
from strategy.mkt_idx.train import load_model_by_type


def run(
    test_days: int = 30,
    threshold: float = 0.6,
    top_n: int = 5,
    max_positions: int = 10,
):
    """
    跑一次 mkt_idx 回測。

    要用哪個模型（rfc/xgb/lgbm）由 config.MODEL_TYPE 決定（讀 .env 的
    MKT_IDX_MODEL_TYPE，預設lgbm），跟 live.py 共用同一個參數，比照
    strategy/rally/run_backtest.py、strategy/orb/run_backtest.py 的做法——
    只改 config.py/.env 一個地方，回測跟即時交易就會一起換模型。

    threshold: 信心度門檻，predict() 產生的機率矩陣是「漲」（class=2）的
        機率，只有 proba >= threshold 的候選才會進場（純做多，跟共用引擎
        gen_entries() 的語意一致）。
    first_entry_time/last_entry_time 固定用 config.SESSION_START/END
        （9:11~9:30，訓練/推論時段），不像 orb 那樣額外開放參數調整——mkt_idx
        沒有「候選池範圍」跟「實際交易窗口」分開的設計，訓練時的時段過濾
        跟這裡的交易窗口本來就要保持一致（見 train.py::_prepare_data() 的
        說明），不應該讓呼叫端自己調鬆調緊。
    """
    model = load_model_by_type(MODEL_TYPE)

    df_proba = predict(model=model, test_days=test_days)
    print(f"機率矩陣: {df_proba.shape}，非空值 {df_proba.notna().sum().sum()} 筆")

    portfolio_df, trades_df = run_backtest(
        df_proba,
        threshold=threshold,
        top_n=top_n,
        max_positions=max_positions,
        sl_pct=SL_PCT,
        tp_pct=TP_PCT,
        hold_bars=HOLD_BARS,
        first_entry_time="09:11",
        last_entry_time="09:30",
    )
    print()
    print_trades(trades_df)
    return portfolio_df, trades_df


if __name__ == "__main__":
    run(test_days=30, threshold=0.6)
