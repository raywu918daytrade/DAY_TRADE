"""limitup_fade_ml 特徵清單與組裝。

所有特徵都以決策時點 09:03（首根 m3 收完）為準，不用當天後續資訊。
"""

from __future__ import annotations

import pandas as pd

FEATURES = [
    "gap_pct",  # 開盤相對昨收：(open / prev_close) - 1
    "m3_ret",  # 首 3 分報酬：(m3_close / m3_open) - 1（觸發時為負）
    "m3_body_ratio",  # 陰線實體占比：(m3_open - m3_close) / (high - low)
    "m3_upper_ratio",  # 上影占比：(m3_high - m3_open) / (high - low)
    "m3_lower_ratio",  # 下影占比：(m3_close - m3_low) / (high - low)
    "m3_range_pct",  # 振幅相對開：(m3_high - m3_low) / m3_open
    "prev_ret",  # 前日日報酬（相對再前一日收）
    "prev_body_ratio",  # 前日陽線實體占比：(close - open) / (high - low)
    "prev_upper_ratio",  # 前日上影占比：(high - close) / (high - low)
    "prev_volume_z",  # 前日量相對近 20 日均量：(vol - avg20) / avg20
    "open_vs_prev_high",  # 開盤相對昨高：(open / prev_high) - 1
    "m3_close_vs_open",  # 首 3 分收相對今日開盤：(m3_close / day_open) - 1
    "gap_vs_0050",  # 個股缺口 − 0050 開盤缺口（相對大盤強弱）
]


def make_features(events: pd.DataFrame) -> pd.DataFrame:
    """從 build_events() 結果取出特徵欄；缺欄補 NaN。"""
    df = events.copy()
    for col in FEATURES:
        if col not in df.columns:
            df[col] = float("nan")
    return df
