"""
ORB 策略回測入口 — 串接 strategy/orb/predict.py 跟共用回測引擎
backtest/intraday_platform.py（不改共用引擎本身，rally 也在用同一份）。

用法：
    python strategy/orb/run_backtest.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.intraday_platform import print_trades, run_backtest
from strategy.orb.config import DEFAULT_TEST_DAYS, HOLD_BARS, OPENING_RANGE_END, SEARCH_WINDOW_END, SL_PCT, TP_PCT
from strategy.orb.predict import predict
from strategy.orb.train import load_model_by_type


def run(
    test_days: int = DEFAULT_TEST_DAYS,
    threshold: float = 0.70,
    top_n: int = 999,
    max_positions: int = 99,
    first_entry_time: str = OPENING_RANGE_END,
    last_entry_time: str = SEARCH_WINDOW_END,
    model_type: str = "lgbm",
):
    """
    跑一次 ORB 回測。

    model_type: 要用哪個模型（rfc/lgbm/xgb），直接傳參數指定（2026-07-22
    討論：即時交易現在是 orb_xgb/orb_lgbm 兩個獨立策略各自寫死模型，不再
    共用 config.MODEL_TYPE/ORB_MODEL_TYPE 切換，回測這裡也直接改成參數輸入，
    不用再繞去改 .env）。
    threshold: 信心度門檻，只有 proba >= threshold 的候選才會進場。
    top_n: 同一分鐘最多取幾支（999 等於不限制，因為候選本來就稀疏，見
        strategy/orb/features.py 的候選篩選邏輯）。
    first_entry_time/last_entry_time: 實際交易時段——預設直接等於 config.py
        的 OPENING_RANGE_END/SEARCH_WINDOW_END（訓練/推論用的突破搜尋窗口，
        09:10~09:30），train/backtest/live 三邊統一讀同一份設定，不用再各自
        寫死時間、也不會再兜不起來（2026-08-06 之前這裡曾經預設09:20、
        __main__ 卻寫死09:30，兩邊對不上）。如果之後真的想測試「候選裡
        只交易某個子區間」，還是可以在呼叫端明確傳這兩個參數覆蓋，不影響
        訓練/推論那邊的候選產生範圍。
    max_positions: 同時最多持倉幾檔。回測發現候選會集中在特定日子（例如
        2026-07-03 單日9:10~9:20內就有54筆proba>=0.7的候選），max_positions
        太小（原本共用引擎預設10）會讓大部分候選連進場機會都沒有就被卡住，
        這裡拉高，讓回測結果比較能反映模型真正篩出的候選品質，不是被倉位
        上限意外限制住。
    vol_window/min_vol_threshold: gen_entries() 內建的量能過濾器（用當天
        1分K滾動均量，不是我們自己的20日均量篩選 MIN_VOL_MA20），預設
        rolling(20) 在開盤前20分鐘天生是NaN，會把我們的候選（都在9:10~9:30
        內）誤濾掉。改用 rolling(10)——跟 OPENING_RANGE_MINUTES=10 對齊，
        搜尋窗口一開始（第10分鐘）就有滿10根K棒，不會有暖機期問題。
    """
    model = load_model_by_type(model_type)

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
        first_entry_time=first_entry_time,
        last_entry_time=last_entry_time,
        min_vol_threshold=0,
        vol_window=10,
    )
    print()
    print_trades(trades_df)
    return portfolio_df, trades_df


if __name__ == "__main__":
    run(
        test_days=DEFAULT_TEST_DAYS,
        threshold=0.85,
        model_type="lgbm",
    )
