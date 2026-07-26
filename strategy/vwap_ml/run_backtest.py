"""
vwap_ml 策略回測入口 — 串接 strategy/vwap_ml/predict.py 跟共用回測引擎
backtest/intraday_platform.py（不改共用引擎本身，orb/rally/mkt 也在用
同一份）。

⚠️ 共用引擎只吃單一（做多）機率矩陣，沒有方向概念，這裡回測的是
predict.py::predict() 算出來的「做多」訊號（見該函式檔頭說明）——真正
的多空雙向訊號要看 predict_live()，回測放空是完全另一個規模的工程，先
不做（比照 strategy/mkt/run_backtest.py 的同樣考量）。

用法：
    python strategy/vwap_ml/run_backtest.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.intraday_platform import print_trades, run_backtest
from strategy.vwap_ml.config import BACKTEST_HOLD_BARS, BACKTEST_SL_PCT, BACKTEST_TP_PCT, SESSION_END, SESSION_START
from strategy.vwap_ml.predict import predict
from strategy.vwap_ml.train import load_model_by_type


def run(
    test_days: int = 30,
    threshold: float = 0.6,
    top_n: int = 5,
    max_positions: int = 10,
    model_type: str = "lgbm",
    start_date: str | None = None,
    use_cache: bool = True,
):
    """
    跑一次 vwap_ml 回測。

    model_type: 要用哪個模型（lgbm/xgb），直接傳參數指定，比照
        strategy/mkt/run_backtest.py 的做法——回測獨立於 config.MODEL_TYPE/
        up/down/live.py 讀的設定，不用繞去改 .env。
    threshold: 信心度門檻，predict() 產生的機率矩陣是「做多」機率（上軌
        延續／下軌回歸，見 predict.py::_direction_probas() 的說明），只有
        proba >= threshold 的候選才會進場。
    start_date: 要跟訓練那個模型用的 start_date 一致，才會讀到同一份cache
        （見 train.py::_prepare_data()::_cache_path_for() 的說明）。
    use_cache: 預設 True（跟 train.py 的預設相反）——回測通常緊接在訓練
        後面跑，直接沿用剛建好的cache最合理；如果改過 features.py 的
        計算邏輯，記得手動傳 False 強制重算。
    first_entry_time/last_entry_time 固定用 config.SESSION_START/END
        （9:10~10:00），跟 train.py 的時段過濾保持一致，比照
        strategy/mkt/run_backtest.py 不額外開放參數調整的做法。
    sl_pct/tp_pct/hold_bars 用 config.BACKTEST_*（跟 label 用的 z-score
        barrier是兩回事，見 config.py 的說明）。
    """
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
    run(test_days=30, threshold=0.6, start_date="2024-01-01")
