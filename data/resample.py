"""
1 分K → 3 分K / 5 分K 的 rolling 聚合邏輯。

data/build_m3_m5.py（批次預算 db/m3/db/m5）、strategy/rally、strategy/orb
即時推論都共用同一份計算邏輯，統一放在 data/ 這個中立層，讓 data/ 不必反過來
依賴任何策略資料夾，策略也不用各自複製一份維護。
"""

import pandas as pd


def _degroup(s: pd.Series, index: pd.Index) -> pd.Series:
    """把 groupby(...).rolling(...) 產生的多層 index 結果攤平回與原始 df 對齊的 Series。

    用原生 GroupBy.rolling/shift/pct_change 取代 transform(lambda ...)：後者對每個分組
    各呼叫一次 python function，分組數一多（本專案動輒數萬個 stock×day 分組）就非常慢；
    前者走 pandas 內建向量化路徑，同樣的操作快上一到兩個數量級。
    """
    n_key_levels = s.index.nlevels - 1
    if n_key_levels:
        s = s.reset_index(level=list(range(n_key_levels)), drop=True)
    return s.reindex(index)


def compute_m3(m1: pd.DataFrame) -> pd.DataFrame:
    """
    從 1 分K 現算 rolling-3 OHLCV（量用 sum）。

    輸入需先有 stock_id / date / day_date 欄位並依 stock_id、date 排序。
    train/live 共用同一份邏輯：data/build_m3_m5.py 批次預算 db/m3/ 給訓練用
    （load_features() 走 cache，走的是這支函式批次算好存檔的結果）；
    predict_live() 則是直接對當天的 m1_live 呼叫這支函式現算——db/m3/ 是批次
    產物，不包含「今天」的資料，即時推論不能拿它 merge，否則 m3_* 全部變 NaN。
    """
    g = m1.groupby(["stock_id", "day_date"], group_keys=False)
    out = m1[["stock_id", "date"]].copy()
    out["open"] = g["open"].shift(2)
    out["high"] = _degroup(g["high"].rolling(3).max(), m1.index)
    out["low"] = _degroup(g["low"].rolling(3).min(), m1.index)
    out["close"] = m1["close"].values
    out["volume"] = _degroup(g["volume"].rolling(3).sum(), m1.index)
    return out


def compute_m5(m1: pd.DataFrame) -> pd.DataFrame:
    """從 1 分K 現算 rolling-5 OHLCV（量用 sum）。用法同 compute_m3()。"""
    g = m1.groupby(["stock_id", "day_date"], group_keys=False)
    out = m1[["stock_id", "date"]].copy()
    out["open"] = g["open"].shift(4)
    out["high"] = _degroup(g["high"].rolling(5).max(), m1.index)
    out["low"] = _degroup(g["low"].rolling(5).min(), m1.index)
    out["close"] = m1["close"].values
    out["volume"] = _degroup(g["volume"].rolling(5).sum(), m1.index)
    return out
