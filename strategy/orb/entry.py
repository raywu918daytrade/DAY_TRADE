"""
當沖策略 — ORB（開盤區間突破）特徵 + LightGBM / XGBoost 模型（CLI 進入點）

實際邏輯拆到同資料夾底下：
    config.py     交易相關設定（TP/SL/HOLD_BARS、開盤區間分鐘數）
    features.py   ORB 特徵工程、triple barrier 標籤、FEATURES 清單、load_features() cache
    train.py      LightGBM / XGBoost 訓練與模型載入
    validate.py   信心度分析、召回率分析、分鐘區間交叉報表、特徵重要性、LGBM vs XGB 比較
    predict.py    predict()（批次機率矩陣，回測用）、predict_live()（即時推論入口）

== Main 模式 ==

train_lgbm   訓練 LightGBM 模型
train_xgb    訓練 XGBoost 模型
validate     信心度分析 + 召回率分析 + 突破後分鐘區間交叉報表 + 特徵重要性（LGBM、XGB 已訓練的都跑）
compare      LGBM vs XGB 同一份測試集 Accuracy/AUC 對照（兩個模型都要先訓練過）
predict      印批次機率矩陣（predict()）的形狀跟最後一個時間點的排行榜，用來肉眼檢查
"""

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.orb.config import DEFAULT_TEST_DAYS, HOLD_BARS, OPENING_RANGE_MINUTES, SL_PCT, TP_PCT  # noqa: F401
from strategy.orb.predict import predict as predict_batch
from strategy.orb.train import train_lgbm, train_xgb
from strategy.orb.validate import (
    available_models,
    compare_report,
    confidence_report,
    confusion_matrix_report,
    coverage_report,
    feature_importance,
    minute_confidence_report,
)


def main(
    mode: str = "",
    test_days: int = DEFAULT_TEST_DAYS,
    start_date: str = "",
    end_date: str = "",
):
    """
    當沖策略 ORB + LightGBM 主程式。

    支援兩種用法：
      1. 直接傳參數：main(mode="train_lgbm")
      2. CLI 執行：python strategy/orb/entry.py train_lgbm
    """
    if not mode:
        parser = argparse.ArgumentParser(
            description="當沖策略 — ORB 特徵 + LightGBM",
        )
        parser.add_argument(
            "mode", choices=["train_lgbm", "train_xgb", "validate", "compare", "predict"], help="執行模式"
        )
        parser.add_argument("--test_days", type=int, default=DEFAULT_TEST_DAYS, help="測試集天數")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        start_date = args.start_date
        end_date = args.end_date

    if mode == "train_lgbm":
        train_lgbm(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "train_xgb":
        train_xgb(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "validate":
        for name, model in available_models().items():
            print(f"\n══════════ {name} ══════════")
            confidence_report(model=model, test_days=test_days, start_date=start_date, end_date=end_date)
            print()
            coverage_report(model=model, test_days=test_days, start_date=start_date, end_date=end_date)
            print()
            confusion_matrix_report(model=model, test_days=test_days, start_date=start_date, end_date=end_date)
            print()
            minute_confidence_report(model=model, test_days=test_days, start_date=start_date, end_date=end_date)
            print()
            feature_importance(model=model)

    elif mode == "compare":
        compare_report(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "predict":
        proba = predict_batch(test_days=test_days)
        print(f"  機率矩陣形狀: {proba.shape}（{proba.shape[0]} 個時間點 × {proba.shape[1]} 支股票）")
        print(f"  時間區間: {proba.index.min()} ~ {proba.index.max()}")
        last_ts = proba.index.max()
        top = proba.loc[last_ts].dropna().sort_values(ascending=False).head(10)
        print(f"\n  -- {last_ts} 機率排行榜（前10） --")
        for stock_id, p in top.items():
            print(f"    {stock_id:>8s}  {p:.4f}")

    else:
        print(f"未知模式: {mode}，可用: train_lgbm / train_xgb / validate / compare / predict")


if __name__ == "__main__":
    mode = "validate"  # validate,train_lgbm,train_xgb
    test_days = DEFAULT_TEST_DAYS  # 統一用 config.py 的預設值，不要在這裡另外寫死數字
    start_date = ""
    end_date = ""

    main(mode=mode, test_days=test_days, start_date=start_date, end_date=end_date)
