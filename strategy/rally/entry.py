"""
當沖策略 — RandomForest 簡單模型（CLI 進入點）

實際邏輯拆到同資料夾底下：
    config.py     交易相關設定（TP/SL/HOLD_BARS、SESSION、BREAKOUT_TRADE 時段）
    features.py   特徵工程、triple barrier 標籤、FEATURES 清單、load_features() cache
    train.py      RandomForest / XGBoost / LightGBM 訓練與模型載入
    validate.py   信心度/召回率/模型×時段×信心度交叉報表、特徵重要性
    predict.py    批次與即時推論（predict_live 是正式對外入口）
    experiments/  一次性假設驗證（例如破底翻要不要拆專門模型/硬過濾值不值得留），
                  還沒有定論、不是核心流程，獨立於上面幾支之外

本檔只留 main()/CLI，組裝上面幾支模組。要調整交易參數（停利停損、早盤時段、
破底翻黃金窗口）直接改 config.py，不用動這裡。

== Main 模式 ==

train / train_xgb / train_lgbm   訓練模型
validate                         信心度分析 + 召回率分析 + 模型×時段×信心度交叉報表
signals                          每小時訊號數 vs 抓到筆數（固定門檻，原始筆數，見 validate.hour_signal_report）
importance                       特徵重要性
build_m3_m5                      補 db/m3/db/m5（增量，只重算比 db/m1 舊的月份）——訓練前務必先跑，
                                  不然 load_features() fallback 讀到的 m3/m5 可能比 db/m1 舊，訓練資料
                                  會悄悄漏掉最新幾天（2026-07-13 實際發生過：db/m1 到 07-09、
                                  db/m3/db/m5 卡在 07-07，少算了2個交易日）
"""

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.build_m3_m5_rolling import build as build_m3_m5
from strategy.rally.config import (  # noqa: F401  (re-export：交易參數一眼看到全部)
    BREAKOUT_TRADE_END,
    BREAKOUT_TRADE_START,
    HOLD_BARS,
    SESSION_END,
    SESSION_START,
    SL_PCT,
    TP_PCT,
)
from strategy.rally.train import train, train_lgbm, train_xgb
from strategy.rally.validate import (
    confidence_report,
    coverage_report,
    feature_importance,
    hour_signal_report,
    model_hour_confidence_report,
)


def main(
    mode: str = "",
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    force_cache: bool = False,
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
        train / validate / importance
    test_days : int
        測試集天數（預設 10）
    start_date : str
        資料起日，格式 "YYYY-MM-DD"。留空不限制。
    end_date : str
        資料迄日，格式 "YYYY-MM-DD"。留空不限制。
    force_cache : bool
        True 時略過特徵 cache 新鮮度檢查，只要 cache 存在就直接讀（例如背景資料
        下載動到 db/m1 檔案時間戳、但資料內容沒變時，跳過不必要的重算）。
        只影響 train / train_xgb / train_lgbm。
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
                "validate",
                "signals",
                "importance",
                "build_m3_m5",
            ],
            help="執行模式",
        )
        parser.add_argument("--test_days", type=int, default=10, help="測試集天數")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        parser.add_argument(
            "--force_cache",
            action="store_true",
            help="略過特徵 cache 新鮮度檢查，只要 cache 存在就直接讀（只影響 train 系列 mode）",
        )
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        start_date = args.start_date
        end_date = args.end_date
        force_cache = args.force_cache

    if mode == "train":
        train(test_days=test_days, start_date=start_date, end_date=end_date, force_cache=force_cache)

    elif mode == "train_xgb":
        train_xgb(test_days=test_days, start_date=start_date, end_date=end_date, force_cache=force_cache)

    elif mode == "train_lgbm":
        train_lgbm(test_days=test_days, start_date=start_date, end_date=end_date, force_cache=force_cache)

    elif mode == "validate":
        confidence_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        coverage_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        model_hour_confidence_report(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "signals":
        hour_signal_report(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "importance":
        feature_importance()

    elif mode == "build_m3_m5":
        # 增量重算：只補比 db/m1 舊的月份，訓練前跑這個確保 load_features()
        # fallback 讀到的 db/m3/db/m5 沒有漏掉最新幾天。
        build_m3_m5(incremental=True)

    else:
        print(f"未知模式: {mode}，可用: train / validate / signals / importance / build_m3_m5")


if __name__ == "__main__":
    """
    直接執行本檔的進入點。

    在這裡直接改 mode 變數即可切換模式，不用每次打 CLI 參數。
    也可從 terminal 用 argparse 呼叫，例如：
        python strategy/rally/date_trade_rfc_model.py train
        python strategy/rally/date_trade_rfc_model.py validate
        python strategy/rally/date_trade_rfc_model.py importance

    可用 mode:
        "train"       訓練 RandomForest 模型（存至 models/m1_rfc.pkl）
        "validate"    信心度分析 + 召回率分析 + 模型×時段×信心度交叉報表
        "signals"     每小時訊號數 vs 抓到筆數（固定門檻，原始筆數）
        "importance"  顯示特徵重要性
        "train_xgb"   訓練 XGBoost 模型（存至 models/m1_xgb.pkl）
        "train_lgbm"  訓練 LightGBM 模型（存至 models/m1_lgbm_breakout.pkl）
        "build_m3_m5" 增量補 db/m3/db/m5（只重算比 db/m1 舊的月份），train 前先跑這個
        ""            走 CLI argparse（terminal 下帶參數）

    可選參數（CLI 或下方變數）：
        test_days     測試集天數（預設 10，取最後 N 天）
        start_date    資料起日 YYYY-MM-DD（留空 = 不限制）
        end_date      資料迄日 YYYY-MM-DD（留空 = 不限制）
        force_cache   True 時略過特徵 cache 新鮮度檢查，只要 cache 存在就直接讀
                      （只影響 train/train_xgb/train_lgbm；db/m1 被背景下載動到
                      檔案時間戳但資料沒變時用得到，見 features.load_features()）

    備註：
        - breakout_signal（先跌後漲的破底翻型態）是模型的普通輸入特徵之一，
          不是訓練目標；它的相關驗證報表跟「要不要拆專門模型」的實驗都在
          strategy/rally/experiments/ 底下，還沒有定論，不是這裡的核心模式
        - 模型是全天訓練的（features.py 的 minutes_since_open 讓模型自己判斷時段），
          進場時段限制交給 backtest/intraday_backtest.py 的
          first_entry_time/last_entry_time 控制，不再鎖死在 make_features()/predict()
        - predict.py 的 predict_live(use_breakout_filter=True) 為即時推論強過濾入口
          （預設開啟），交易時段限制交給呼叫端（例如 live_trader.py 的
          SESSION_START/END），這裡不再額外鎖 9:14~9:30 黃金窗口
        - 直接執行本檔不會觸發即時下單，僅做訓練 / 驗證 / 報表
    """
    # ══════════════════════════════════════════════════════════════════════
    #  在這裡直接改 mode，不用每次打 CLI
    # ══════════════════════════════════════════════════════════════════════
    mode = "train_lgbm"  # build_m3_m5,train_xgb,validate
    test_days = 30
    start_date = ""
    end_date = ""
    force_cache = False  # 用現有 cache，略過 db/m1 新鮮度檢查（不重算特徵）

    main(
        mode=mode,
        test_days=test_days,
        start_date=start_date,
        end_date=end_date,
        force_cache=force_cache,
    )
