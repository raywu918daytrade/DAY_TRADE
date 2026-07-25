"""
cnn 策略回測入口 — 串接 strategy/cnn/predict.py 跟共用回測引擎
backtest/intraday_platform.py（不改共用引擎本身，rally/orb/mkt也在用同一份）。

只回測「漲」（做多）訊號——共用引擎只吃單一機率矩陣，沒有方向概念，回測
放空是另一個規模的工程，目前不做（見 strategy/mkt/run_backtest.py 同樣的
說明）。「跌」訊號目前只在 predict_live()/strategy/cnn/down 出現，給即時
監控用，還沒有對應的回測驗證方式。

用法：
    python strategy/cnn/run_backtest.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.intraday_platform import print_trades, run_backtest
from strategy.cnn.config import HOLD_BARS, SESSION_END, SESSION_START, SL_PCT, TP_PCT
from strategy.cnn.predict import predict
from strategy.cnn.train import load_model


def run(
    test_days: int = 30,
    start_date: str = "2024-01-01",
    threshold: float = 0.6,
    top_n: int = 5,
    max_positions: int = 10,
):
    """
    跑一次 cnn 回測。

    start_date 要跟目前這個模型實際訓練時用的範圍一致，見
    strategy/cnn/predict.py::predict() 的說明——留空會觸發db/m1整包掃描，
    跑得極慢。

    threshold: 信心度門檻，predict() 產生的機率矩陣是「漲」（class=2）的
        機率，只有 proba >= threshold 的候選才會進場（純做多，跟共用引擎
        gen_entries() 的語意一致）。2026-07-25驗證：0.6是精準度/涵蓋率
        較平衡的操作點，見 strategy/cnn/config.py::THRESHOLD 的說明。
    first_entry_time/last_entry_time 固定用 config.SESSION_START/END
        （9:00~10:00，訓練/推論時段），理由同 strategy/mkt/run_backtest.py：
        訓練時的時段過濾跟回測交易窗口本來就要保持一致，不開放呼叫端自己
        調鬆調緊。
    """
    model = load_model()

    df_proba = predict(model=model, test_days=test_days, start_date=start_date)
    print(f"機率矩陣: {df_proba.shape}，非空值 {df_proba.notna().sum().sum()} 筆")

    portfolio_df, trades_df = run_backtest(
        df_proba,
        threshold=threshold,
        top_n=top_n,
        max_positions=max_positions,
        sl_pct=SL_PCT,
        tp_pct=TP_PCT,
        hold_bars=HOLD_BARS,
        first_entry_time=f"{SESSION_START[0]:02d}:{SESSION_START[1]:02d}",
        last_entry_time=f"{SESSION_END[0]:02d}:{SESSION_END[1]:02d}",
    )
    print()
    print_trades(trades_df)
    return portfolio_df, trades_df


if __name__ == "__main__":
    run(test_days=30, threshold=0.6)
