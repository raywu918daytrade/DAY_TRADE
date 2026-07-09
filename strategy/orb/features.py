"""
特徵工程與資料標籤 — 開盤區間突破（ORB, Opening Range Breakout）+ 3分K/5分K 中期動能確認

用 db/m1/ 歷史分K，每日每股取開盤前 OPENING_RANGE_MINUTES 分鐘的最高/最低點
當作當天的關鍵區間（不跨日）。區間形成期間本身（前 OPENING_RANGE_MINUTES
分鐘）的樣本會整批排除——那幾分鐘區間還沒收斂，此時去看「距上緣距離」
這類特徵等於用當天最終的區間高低點回頭算，會把「區間形成期間之後才知道」
的值洩漏回形成期間本身；且真實交易時，區間沒收斂也不可能有突破訊號。
標籤沿用 rally 的 triple barrier 定義：未來 HOLD_BARS 根分K內
  +TP_PCT 停利先碰到 → target = 1（漲）
  -SL_PCT 停損先碰到 → target = 0（跌）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.query import load_m1, load_m3, load_m5
from strategy.orb.config import HOLD_BARS, OPENING_RANGE_MINUTES, SL_PCT, TP_PCT

_ROOT = Path(__file__).parent.parent.parent
_M1_DIR = _ROOT / "db/m1"
_CACHE_PATH = _ROOT / "cache/m1_orb_features.parquet"


# ── Cache 管理 ───────────────────────────────────────────────────────────────


def _m1_mtime() -> float:
    """db/m1/ 中最新檔案的修改時間戳。"""
    mtimes = [f.stat().st_mtime for f in _M1_DIR.iterdir() if f.suffix == ".parquet"]
    return max(mtimes) if mtimes else 0


def _cache_is_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    return _CACHE_PATH.stat().st_mtime >= _m1_mtime()


def load_features(force_rebuild: bool = False) -> pd.DataFrame:
    """載入特徵（自動使用 / 重建 cache）。"""
    if not force_rebuild and _cache_is_fresh():
        cached = pd.read_parquet(_CACHE_PATH)
        missing = [c for c in FEATURES + ["hour"] if c not in cached.columns]
        if not missing:
            print("  讀取 cache 特徵...")
            return cached
        print(f"  cache 缺少欄位 {missing}，重新計算...")

    print("  重新計算特徵（db/m1 有更新）...")
    m1 = load_m1()
    df = make_features(m1, compute_labels=True)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = ["stock_id", "date", "hour", "target"]
    cache_cols = [c for c in df.columns if c in set(FEATURES) | set(meta_cols)]
    df[cache_cols].to_parquet(_CACHE_PATH)
    print(f"  cache 已存至 {_CACHE_PATH}（{len(df):,} 筆）")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Triple Barrier Label（沿用 rally 的定義，見 strategy/rally/features.py）
# ═══════════════════════════════════════════════════════════════════════════════


def _barrier_label_group(closes: np.ndarray) -> np.ndarray:
    """每日每股，逐 bar 往前看最多 HOLD_BARS 根，判斷先碰到 tp 或 sl。"""
    n = len(closes)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        entry = closes[i]
        tp_price = entry * (1 + TP_PCT)
        sl_price = entry * (1 - SL_PCT)
        future = closes[i + 1 : i + HOLD_BARS + 1]
        tp_idx = np.argmax(future >= tp_price) if (future >= tp_price).any() else len(future)
        sl_idx = np.argmax(future <= sl_price) if (future <= sl_price).any() else len(future)
        if tp_idx < sl_idx:
            labels[i] = 1
        elif sl_idx < tp_idx:
            labels[i] = 0
        elif len(future) == HOLD_BARS:
            labels[i] = 1 if future[-1] > entry else 0
    return labels


def _make_barrier_labels(m1: pd.DataFrame) -> pd.Series:
    return m1.groupby(["stock_id", "day_date"], group_keys=False).apply(
        lambda g: pd.Series(_barrier_label_group(g["close"].values), index=g.index),
        include_groups=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ORB（開盤區間突破）特徵工程
# ═══════════════════════════════════════════════════════════════════════════════


FEATURES = [
    "or_range_pct",  # 開盤區間高度 / 區間下緣（區間本身寬不寬）
    "close_pos_in_range",  # close 在區間內相對位置（<0跌破下緣，0~1區間內，>1突破上緣）
    "dist_to_or_high",  # close 距區間上緣的相對距離
    "dist_to_or_low",  # close 距區間下緣的相對距離
    "broke_above",  # close > 區間上緣（0/1）
    "broke_below",  # close < 區間下緣（0/1）
    "breakout_up_signal",  #    當根K棒首次向上突破（0/1，開倉訊號本體）
    "breakout_down_signal",  # 當根K棒首次向下突破（0/1）
    "bars_since_breakout_up",  # 距首次向上突破幾根K棒（今日未發生過=999）
    "bars_since_breakout_down",  # 距首次向下突破幾根K棒
    "vol_ratio_vs_or",  # 當根量 / 開盤區間期間均量（量能確認突破）
    "minutes_since_or_end",  # 距開盤區間結束幾分鐘
    # ── 3分K（過去3根）/ 5分K（過去2根）—— 中期動能當多空判斷（比照 rally 做法）──
    "m3_high",
    "m3_low",
    "m3_close",
    "m3_ret",  # 3分鐘K報酬率（K棒間變化%）
    "m3_high_lag1",
    "m3_low_lag1",
    "m3_close_lag1",
    "m3_open_lag2",
    "m3_high_lag2",
    "m3_low_lag2",
    "m5_open",
    "m5_high",
    "m5_low",
    "m5_ret",  # 5分鐘K報酬率（K棒間變化%）
    "m5_open_lag1",
    "m5_high_lag1",
    "m5_low_lag1",
    "m5_close_lag1",
]


def _bars_since(m1: pd.DataFrame, flag_col: str) -> pd.Series:
    """距最近一次 flag_col==1 幾根K棒（同一天內），今日尚未發生過則回傳 999。"""
    group_cols = [m1["stock_id"], m1["day_date"]]
    pos = m1.groupby(["stock_id", "day_date"], group_keys=False).cumcount()
    last_true_pos = pos.where(m1[flag_col] == 1)
    last_true_pos = last_true_pos.groupby(group_cols, group_keys=False).ffill()
    bars = (pos - last_true_pos).fillna(999)
    return bars


def make_features(m1: pd.DataFrame, compute_labels: bool = True) -> pd.DataFrame:
    """
    只算 ORB（開盤區間突破）衍生特徵 + triple barrier 標籤。

    回傳欄位：FEATURES 全部 + target（若 compute_labels=True）。
    區間形成期間（前 OPENING_RANGE_MINUTES 分鐘）本身的樣本會被整批排除
    （FEATURES 全部設為 NaN，靠呼叫端 dropna(subset=FEATURES) 篩掉）。
    """
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    m1["hour"] = m1["date"].dt.hour
    m1["minute"] = m1["date"].dt.minute
    m1["minutes_since_open"] = (m1["hour"] - 9) * 60 + m1["minute"]

    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)

    # 當日開盤價（第一根 open），用於 3分K/5分K 正規化
    day_open = g_day["open"].transform("first").replace(0, np.nan)

    # ── 3分K（過去3根）/ 5分K（過去2根）—— 中期動能當多空判斷 ─────────────
    # db/m3、db/m5 是批次預算（scripts/build_m3_m5.py 從 db/m1/ 算好存檔，
    # 用的是 strategy/rally/features.py 的 compute_m3()/compute_m5()），
    # 只有訓練用的完整歷史 m1 才會跟它對得上。
    m3 = load_m3()
    m5 = load_m5()
    m3["date"] = pd.to_datetime(m3["date"])
    m5["date"] = pd.to_datetime(m5["date"])

    m3_feat = m3[["stock_id", "date"]].copy()
    m3_feat["m3_open"] = m3["open"] / day_open
    m3_feat["m3_high"] = m3["high"] / day_open
    m3_feat["m3_low"] = m3["low"] / day_open
    m3_feat["m3_close"] = m1["close"] / day_open

    m5_feat = m5[["stock_id", "date"]].copy()
    m5_feat["m5_open"] = m5["open"] / day_open
    m5_feat["m5_high"] = m5["high"] / day_open
    m5_feat["m5_low"] = m5["low"] / day_open
    m5_feat["m5_close"] = m1["close"] / day_open

    m1 = m1.merge(m3_feat, on=["stock_id", "date"], how="left", suffixes=("", "_y3"))
    m1 = m1.merge(m5_feat, on=["stock_id", "date"], how="left", suffixes=("", "_y5"))

    g_m3 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["m3_high_lag1"] = g_m3["m3_high"].shift(3)
    m1["m3_low_lag1"] = g_m3["m3_low"].shift(3)
    m1["m3_close_lag1"] = g_m3["m3_close"].shift(3)
    m1["m3_open_lag2"] = g_m3["m3_open"].shift(6)
    m1["m3_high_lag2"] = g_m3["m3_high"].shift(6)
    m1["m3_low_lag2"] = g_m3["m3_low"].shift(6)
    m1["m3_ret"] = g_m3["m3_close"].pct_change(3)

    g_m5 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["m5_open_lag1"] = g_m5["m5_open"].shift(5)
    m1["m5_high_lag1"] = g_m5["m5_high"].shift(5)
    m1["m5_low_lag1"] = g_m5["m5_low"].shift(5)
    m1["m5_close_lag1"] = g_m5["m5_close"].shift(5)
    m1["m5_ret"] = g_m5["m5_close"].pct_change(5)

    m1 = m1.drop(columns=["m3_open", "m5_close"])

    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)

    # ── 開盤區間高低點（用前 OPENING_RANGE_MINUTES 分鐘的 high/low） ─────────
    in_or = m1["minutes_since_open"] < OPENING_RANGE_MINUTES
    m1["_or_high_raw"] = m1["high"].where(in_or)
    m1["_or_low_raw"] = m1["low"].where(in_or)
    m1["_or_vol_raw"] = m1["volume"].where(in_or)
    or_high = g_day["_or_high_raw"].transform("max")
    or_low = g_day["_or_low_raw"].transform("min")
    or_vol_mean = g_day["_or_vol_raw"].transform("mean")

    or_range = (or_high - or_low).replace(0, np.nan)
    m1["or_range_pct"] = or_range / or_low.replace(0, np.nan)
    m1["close_pos_in_range"] = (m1["close"] - or_low) / or_range
    m1["dist_to_or_high"] = (m1["close"] - or_high) / or_high.replace(0, np.nan)
    m1["dist_to_or_low"] = (m1["close"] - or_low) / or_low.replace(0, np.nan)
    m1["broke_above"] = (m1["close"] > or_high).astype(int)
    m1["broke_below"] = (m1["close"] < or_low).astype(int)

    prev_close = g_day["close"].shift(1)
    m1["breakout_up_signal"] = ((prev_close <= or_high) & (m1["close"] > or_high)).astype(int)
    m1["breakout_down_signal"] = ((prev_close >= or_low) & (m1["close"] < or_low)).astype(int)

    m1["bars_since_breakout_up"] = _bars_since(m1, "breakout_up_signal")
    m1["bars_since_breakout_down"] = _bars_since(m1, "breakout_down_signal")

    m1["vol_ratio_vs_or"] = m1["volume"] / or_vol_mean.replace(0, np.nan)
    m1["minutes_since_or_end"] = m1["minutes_since_open"] - OPENING_RANGE_MINUTES

    # 區間形成期間本身不是可交易的樣本，整批排除（見檔頭說明）
    for col in FEATURES:
        m1.loc[in_or, col] = np.nan

    m1 = m1.drop(columns=["_or_high_raw", "_or_low_raw", "_or_vol_raw"])

    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    return m1
