"""
1 分K → 3 分K / 5 分K 的聚合邏輯，兩種版本：

  compute_m3()/compute_m5()          rolling（每分鐘一列，過去 N 根滑動視窗）
  compute_m3_std()/compute_m5_std()  標準獨立K棒（一根 N 分鐘一列，非 rolling）

data/build_m3_m5_rolling.py 用前者批次預算 db/m3/db/m5，data/build_m3_m5_std.py
用後者批次預算 db/m3_std/db/m5_std。strategy/rally、strategy/orb 即時推論也
共用 rolling 版本。統一放在 data/ 這個中立層，讓 data/ 不必反過來依賴任何
策略資料夾，策略也不用各自複製一份維護。
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
    train/live 共用同一份邏輯：data/build_m3_m5_rolling.py 批次預算 db/m3/ 給訓練用
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


def _fixed_bars(m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """把 1 分K 打包成標準獨立 K 棒（非 rolling），一根 N 分鐘，一列一根。

    窗口是半開區間 [bar_start, bar_start+minutes)，標記時間點取窗口結束時刻
    （例如 3 分K 第一根覆蓋 9:00:00~9:02:59，標記為 9:03:00，銜接下一根
    9:03:00~9:05:59 標記 9:06:00——跟 K 線軟體「這根收了才畫下一根」的慣例
    一致）。開盤 9:00 剛好可以被 3、5 整除，用 pandas 內建 epoch 對齊的
    pd.Grouper 就會自然切在整點分界上，不用逐日各自算 origin。

    最後一根若 db/m1 資料不夠 minutes 根（例如當天收盤前剛好不滿、或該股票
    當天有缺分鐘），會是一根不滿 minutes 分鐘的殘餘 K 棒，不會補齊或丟棄。
    """
    g = m1.groupby(["stock_id", pd.Grouper(key="date", freq=f"{minutes}min", label="right", closed="left")])
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return out.dropna(subset=["open"]).reset_index()


def compute_m3_std(m1: pd.DataFrame) -> pd.DataFrame:
    """把 1 分K 打包成標準 3 分K 獨立K棒（非 rolling）。用法見 _fixed_bars()。"""
    return _fixed_bars(m1, 3)


def compute_m5_std(m1: pd.DataFrame) -> pd.DataFrame:
    """把 1 分K 打包成標準 5 分K 獨立K棒（非 rolling）。用法見 _fixed_bars()。"""
    return _fixed_bars(m1, 5)
