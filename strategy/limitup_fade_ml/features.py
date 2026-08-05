"""
limitup_fade_ml 特徵清單。

所有特徵欄位都已經在 strategy/limitup_fade_ml/dataset.py::build_gap_candidates()/
attach_m3_trigger() 算好（訓練／回測走事件資料集；live 走 predict.py 用同一套邏輯
現算單筆），這裡只定義 FEATURES 清單 + 一個輕量的欄位檢查/篩選函式，不重算特徵。

初版特徵，訓練後可依 train.py::feature_importance() 的結果調整增減。
"""

from __future__ import annotations

import pandas as pd

FEATURES = [
    "gap_pct",  # 跳空幅度：(今日開盤 - 前日收盤) / 前日收盤
    "gap_vs_0050",  # 個股跳空幅度扣掉0050大盤同日跳空幅度（相對大盤強弱）
    "open_vs_prev_high",  # 今日開盤相對前日最高：(今日開盤 / 前日最高) - 1
    "prev_day_ret",  # 前日（漲停日）日報酬率
    "prev_body_ratio",  # 前日實體比例
    "prev_upper_shadow_ratio",  # 前日上影比例
    "prev_volume_ratio",  # 前日成交量 / 前5日均量
    "prev_volume_z",  # 前日成交量相對前20日均量的偏離幅度
    "prev5d_ret",  # 前5日累積報酬（動能）
    "m3_ret",  # 首根3分K跌幅：(m3收盤 - m3開盤) / m3開盤
    "m3_body_ratio",  # 首根3分K實體比例
    "m3_upper_shadow_ratio",  # 首根3分K上影比例
    "m3_lower_shadow_ratio",  # 首根3分K下影比例
    "m3_range_pct",  # 首根3分K振幅相對開盤：(m3最高 - m3最低) / m3開盤
    "dist_from_open_pct",  # 距今日開盤跌幅：(m3收盤 - 今日開盤) / 今日開盤
    "confirm_ret",  # Stage2延續確認段跌幅：(09:10確認價 - m3收盤) / m3收盤
    "day_atr",  # 日K ATR14（正規化成比例），也是這筆交易TP/SL的距離
]


def make_features(events: pd.DataFrame) -> pd.DataFrame:
    """檢查 FEATURES 欄位都存在，回傳去除特徵缺值的事件子集。"""
    if events.empty:
        return events
    missing = [c for c in FEATURES if c not in events.columns]
    if missing:
        raise KeyError(f"事件資料缺少特徵欄位: {missing}")
    return events.dropna(subset=FEATURES).reset_index(drop=True)
