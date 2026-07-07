"""
Profile: 用 10 支股票測試 date_trade_rfc_model.py 的特徵建立速度
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 確保能從根目錄導入
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from data.query import load_m1

# 只載入 10 支股票
print("載入分K（全部）...")
t0 = time.time()
m1_all = load_m1()
print(f"  {len(m1_all):,} 筆, {m1_all['stock_id'].nunique()} 支股票, 耗時 {time.time()-t0:.1f}s")

# 取 10 支股票
stocks = m1_all["stock_id"].unique()[:10]
m1 = m1_all[m1_all["stock_id"].isin(stocks)].copy()
print(f"\n取 10 支股票: {len(m1):,} 筆")
print(f"  日期範圍: {m1['date'].min()} ~ {m1['date'].max()}")

# ── 準備（sort + day_date） ──────────────────────────────────────────────
t0 = time.time()
m1 = m1.copy()
m1["date"] = pd.to_datetime(m1["date"])
m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
m1["day_date"] = m1["date"].dt.date
g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
print(f"\n準備（sort + groupby）: {time.time()-t0:.3f}s")

# ── 當日開盤 ─────────────────────────────────────────────────────────────
t0 = time.time()
day_open = g_day["open"].transform("first").replace(0, np.nan)
print(f"day_open: {time.time()-t0:.3f}s")

# ── 1分鐘K OHLCV ─────────────────────────────────────────────────────────
t0 = time.time()
m1["m1_open"] = m1["open"] / day_open
m1["m1_high"] = m1["high"] / day_open
m1["m1_low"] = m1["low"] / day_open
m1["m1_close"] = m1["close"] / day_open
m1["_cum_vol"] = g_day["volume"].transform("cumsum")
m1["_bar_count"] = g_day["volume"].transform("cumcount") + 1
m1["m1_volume"] = m1["volume"] / (m1["_cum_vol"] / m1["_bar_count"]).replace(0, np.nan)
print(f"1分鐘K OHLCV: {time.time()-t0:.3f}s")

# ── 3分鐘K OHLCV ─────────────────────────────────────────────────────────
t0 = time.time()
m1["m3_open"] = g_day["open"].transform(lambda x: x.shift(2)) / day_open
m1["m3_high"] = g_day["high"].transform(lambda x: x.rolling(3).max()) / day_open
m1["m3_low"] = g_day["low"].transform(lambda x: x.rolling(3).min()) / day_open
m1["m3_close"] = m1["close"] / day_open
m1["m3_volume"] = g_day["volume"].transform(lambda x: x.rolling(3).sum().pct_change(3))
m1["m3_ret"] = g_day["close"].transform(lambda x: x.pct_change(3))
m1["m3_open_lag1"] = g_day["m3_open"].transform(lambda x: x.shift(3))
m1["m3_high_lag1"] = g_day["m3_high"].transform(lambda x: x.shift(3))
m1["m3_low_lag1"] = g_day["m3_low"].transform(lambda x: x.shift(3))
m1["m3_close_lag1"] = g_day["m3_close"].transform(lambda x: x.shift(3))
m1["m3_volume_lag1"] = g_day["m3_volume"].transform(lambda x: x.shift(3))
m1["m3_open_lag2"] = g_day["m3_open"].transform(lambda x: x.shift(6))
m1["m3_high_lag2"] = g_day["m3_high"].transform(lambda x: x.shift(6))
m1["m3_low_lag2"] = g_day["m3_low"].transform(lambda x: x.shift(6))
m1["m3_close_lag2"] = g_day["m3_close"].transform(lambda x: x.shift(6))
m1["m3_volume_lag2"] = g_day["m3_volume"].transform(lambda x: x.shift(6))
print(f"3分鐘K OHLCV（含 lag1/lag2）: {time.time()-t0:.3f}s")

# ── 5分鐘K OHLCV ─────────────────────────────────────────────────────────
t0 = time.time()
m1["m5_open"] = g_day["open"].transform(lambda x: x.shift(4)) / day_open
m1["m5_high"] = g_day["high"].transform(lambda x: x.rolling(5).max()) / day_open
m1["m5_low"] = g_day["low"].transform(lambda x: x.rolling(5).min()) / day_open
m1["m5_close"] = m1["close"] / day_open
m1["m5_volume"] = g_day["volume"].transform(lambda x: x.rolling(5).sum().pct_change(5))
m1["m5_ret"] = g_day["close"].transform(lambda x: x.pct_change(5))
m1["m5_open_lag1"] = g_day["m5_open"].transform(lambda x: x.shift(5))
m1["m5_high_lag1"] = g_day["m5_high"].transform(lambda x: x.shift(5))
m1["m5_low_lag1"] = g_day["m5_low"].transform(lambda x: x.shift(5))
m1["m5_close_lag1"] = g_day["m5_close"].transform(lambda x: x.shift(5))
m1["m5_volume_lag1"] = g_day["m5_volume"].transform(lambda x: x.shift(5))
m1["m5_open_lag2"] = g_day["m5_open"].transform(lambda x: x.shift(10))
m1["m5_high_lag2"] = g_day["m5_high"].transform(lambda x: x.shift(10))
m1["m5_low_lag2"] = g_day["m5_low"].transform(lambda x: x.shift(10))
m1["m5_close_lag2"] = g_day["m5_close"].transform(lambda x: x.shift(10))
m1["m5_volume_lag2"] = g_day["m5_volume"].transform(lambda x: x.shift(10))
print(f"5分鐘K OHLCV（含 lag1/lag2）: {time.time()-t0:.3f}s")

# ── 報酬率與量比 ─────────────────────────────────────────────────────────
t0 = time.time()
m1["ret_1"] = g_day["close"].transform(lambda x: x.pct_change(1))
m1["vol_ratio"] = g_day["volume"].transform(lambda x: x / x.shift(1).replace(0, np.nan))
m1["tf3_ret"] = g_day["close"].transform(lambda x: x.pct_change(3))
m1["tf3_vol_ratio"] = g_day["volume"].transform(
    lambda x: x.rolling(3).sum() / x.rolling(3).sum().shift(3).replace(0, np.nan)
)
m1["tf5_ret"] = g_day["close"].transform(lambda x: x.pct_change(5))
m1["tf5_vol_ratio"] = g_day["volume"].transform(
    lambda x: x.rolling(5).sum() / x.rolling(5).sum().shift(5).replace(0, np.nan)
)
print(f"報酬率與量比: {time.time()-t0:.3f}s")

print(f"\n總計 10 支股票, {len(m1):,} 筆, 特徵數: 53")
print("全部特徵建立完成 ✅")
