"""
當沖策略 — MACD 特徵 + LightGBM 模型（CLI 進入點）

實際邏輯拆到同資料夾底下：
    config.py     交易相關設定（TP/SL/HOLD_BARS、MACD 參數）
    features.py   MACD 特徵工程、triple barrier 標籤、FEATURES 清單、load_features() cache
    train.py      LightGBM 訓練與模型載入

== Main 模式 ==

train_lgbm   訓練模型
"""

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.macd.config import HOLD_BARS, MACD_FAST, MACD_SIGNAL, MACD_SLOW, SL_PCT, TP_PCT  # noqa: F401
from strategy.macd.train import train_lgbm


def main(
    mode: str = "",
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    當沖策略 MACD + LightGBM 主程式。

    支援兩種用法：
      1. 直接傳參數：main(mode="train_lgbm")
      2. CLI 執行：python strategy/macd/entry.py train_lgbm
    """
    if not mode:
        parser = argparse.ArgumentParser(
            description="當沖策略 — MACD 特徵 + LightGBM",
        )
        parser.add_argument("mode", choices=["train_lgbm"], help="執行模式")
        parser.add_argument("--test_days", type=int, default=10, help="測試集天數")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        start_date = args.start_date
        end_date = args.end_date

    if mode == "train_lgbm":
        train_lgbm(test_days=test_days, start_date=start_date, end_date=end_date)
    else:
        print(f"未知模式: {mode}，可用: train_lgbm")


if __name__ == "__main__":
    mode = "train_lgbm"
    test_days = 10
    start_date = ""
    end_date = ""

    main(mode=mode, test_days=test_days, start_date=start_date, end_date=end_date)
