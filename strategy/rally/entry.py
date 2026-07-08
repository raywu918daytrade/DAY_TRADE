"""
當沖策略 — RandomForest 簡單模型（CLI 進入點）

實際邏輯拆到同資料夾底下：
    config.py     交易相關設定（TP/SL/HOLD_BARS、SESSION、BREAKOUT_TRADE 時段）
    features.py   特徵工程、triple barrier 標籤、FEATURES 清單、load_features() cache
    train.py      RandomForest / XGBoost / LightGBM 訓練與模型載入
    validate.py   信心度/召回率/強過濾驗證報表、特徵重要性
    predict.py    批次與即時推論（predict_live 是正式對外入口）

本檔只留 main()/CLI，組裝上面幾支模組。要調整交易參數（停利停損、早盤時段、
破底翻黃金窗口）直接改 config.py，不用動這裡。

== Main 模式 ==

train / train_xgb / train_lgbm   訓練模型
compare                          三模型破底翻黃金窗口門檻掃描對照
validate                         信心度分析 + 召回率分析 + 強過濾評估
importance                       特徵重要性
breakout                         強過濾破底翻逐分鐘報表
"""

import argparse
import os
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.rally.config import (  # noqa: F401  (re-export：交易參數一眼看到全部)
    BREAKOUT_TRADE_END,
    BREAKOUT_TRADE_START,
    HOLD_BARS,
    SESSION_END,
    SESSION_START,
    SL_PCT,
    TP_PCT,
)
from strategy.rally.train import compare_breakout, train, train_lgbm, train_xgb
from strategy.rally.validate import (
    breakout_filter_report,
    breakout_minute_report,
    confidence_report,
    coverage_report,
    feature_importance,
)


def main(
    mode: str = "",
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    當沖策略 RandomForest 主程式。

    支援兩種用法：
      1. 直接傳參數：main(mode="train")
      2. CLI 執行：python strategy/rally/date_trade_rfc_model.py train

    Parameters
    ----------
    mode : str
        執行模式。留空則從 CLI 讀取。
        train / validate / importance / breakout
    test_days : int
        測試集天數（預設 10）
    start_date : str
        資料起日，格式 "YYYY-MM-DD"。留空不限制。
    end_date : str
        資料迄日，格式 "YYYY-MM-DD"。留空不限制。
    """
    if not mode:
        parser = argparse.ArgumentParser(
            description="當沖策略 — RandomForest（close + volume + 過去5天日K）",
        )
        parser.add_argument(
            "mode",
            choices=[
                "train",
                "train_xgb",
                "train_lgbm",
                "compare",
                "validate",
                "importance",
                "breakout",
            ],
            help="執行模式",
        )
        parser.add_argument("--test_days", type=int, default=10, help="測試集天數")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        parser.add_argument(
            "--threshold", type=float, default=0.0, help="強過濾破底翻信心度門檻（breakout mode 用，預設 0.0）"
        )
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        start_date = args.start_date
        end_date = args.end_date
        threshold = args.threshold

    if mode == "train":
        train(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "train_xgb":
        train_xgb(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "train_lgbm":
        train_lgbm(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "compare":
        compare_breakout(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "validate":
        confidence_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        coverage_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        breakout_filter_report(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "importance":
        feature_importance()

    elif mode == "breakout":
        breakout_minute_report(
            test_days=test_days,
            start_date=start_date,
            end_date=end_date,
            threshold=float(os.environ.get("BREAKOUT_THR", "0.0")),
        )

    else:
        print(f"未知模式: {mode}，可用: train / validate / importance")


if __name__ == "__main__":
    """
    直接執行本檔的進入點。

    在這裡直接改 mode 變數即可切換模式，不用每次打 CLI 參數。
    也可從 terminal 用 argparse 呼叫，例如：
        python strategy/rally/date_trade_rfc_model.py train
        python strategy/rally/date_trade_rfc_model.py validate
        python strategy/rally/date_trade_rfc_model.py importance
        python strategy/rally/date_trade_rfc_model.py breakout

    可用 mode:
        "train"       訓練 RandomForest 模型（存至 models/m1_rfc.pkl）
        "validate"    信心度分析 + 召回率分析 + 強過濾評估（breakout_filter_report）
        "importance"  顯示特徵重要性
        "breakout"    強過濾破底翻：9:14~9:30 黃金窗口逐分鐘推論數 / 平均信心度 / 勝率
        "train_xgb"   訓練 XGBoost 模型（存至 models/m1_xgb.pkl）
        "train_lgbm"  訓練 LightGBM 模型（存至 models/m1_lgbm_breakout.pkl）
        "compare"     三模型（RFC/XGB/LGBM）破底翻黃金窗口門檻掃描對照
        ""            走 CLI argparse（terminal 下帶參數）

    可選參數（CLI 或下方變數）：
        test_days     測試集天數（預設 10，取最後 N 天）
        start_date    資料起日 YYYY-MM-DD（留空 = 不限制）
        end_date      資料迄日 YYYY-MM-DD（留空 = 不限制）
        threshold     強過濾破底翻信心度門檻（breakout mode 用，預設 0.0）

    備註：
        - breakout_signal 為「前一根 M5 跌 + 當前 M5 漲」的破底翻硬過濾特徵
        - 交易時段限制為黃金窗口 9:14~9:30（features.py 的 BREAKOUT_TRADE_START/END）
        - predict.py 的 predict_live(use_breakout_filter=True) 為即時推論強過濾入口
          （預設開啟），且只在 9:14~9:30 內產生訊號，其餘時間回傳 []
        - 直接執行本檔不會觸發即時下單，僅做訓練 / 驗證 / 報表
    """
    # ══════════════════════════════════════════════════════════════════════
    #  在這裡直接改 mode，不用每次打 CLI
    # ══════════════════════════════════════════════════════════════════════
    mode = "compare"
    test_days = 10
    start_date = ""
    end_date = ""

    main(mode=mode, test_days=test_days, start_date=start_date, end_date=end_date)
