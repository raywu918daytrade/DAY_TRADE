"""
特徵工程與資料標籤 — 只用 MACD(12,26,9) 衍生特徵

用 db/m1/ 歷史分K，每日每股逐日重算 MACD（不跨日，理由同 rally 的 m1_atr：
日內動能指標，隔天開盤重新累積，不該延續昨天收盤前的動能狀態）。
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

from data.query import load_m1
from strategy.macd.config import HOLD_BARS, MACD_FAST, MACD_SIGNAL, MACD_SLOW, SL_PCT, TP_PCT

_ROOT = Path(__file__).parent.parent.parent
_M1_DIR = _ROOT / "db/m1"
_CACHE_PATH = _ROOT / "cache/m1_macd_features.parquet"


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
        missing = [c for c in FEATURES if c not in cached.columns]
        if not missing:
            print("  讀取 cache 特徵...")
            return cached
        print(f"  cache 缺少欄位 {missing}，重新計算...")

    print("  重新計算特徵（db/m1 有更新）...")
    m1 = load_m1()
    df = make_features(m1, compute_labels=True)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = ["stock_id", "date", "target"]
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
# 2. MACD 特徵工程
# ═══════════════════════════════════════════════════════════════════════════════


FEATURES = [
    "macd_line",
    "macd_signal",
    "macd_hist",
    "macd_hist_slope",  # histogram 一階變化（1根前）
    "macd_hist_slope_3",  # histogram 3根前變化（動能持續性）
    "macd_above_signal",  # macd_line > macd_signal（0/1）
    "macd_above_zero",  # macd_line > 0（0/1，是否站上零軸）
    "golden_cross",  # 當根K棒 macd_line 上穿 macd_signal（0/1）
    "dead_cross",  # 當根K棒 macd_line 下穿 macd_signal（0/1）
    "bars_since_golden_cross",  # 距最近一次金叉幾根K棒（今日未發生過=999）
    "bars_since_dead_cross",  # 距最近一次死叉幾根K棒（今日未發生過=999）
    "macd_divergence",  # 背離分數：過去10根動能變化 - 過去10根價格變化
]


def _degroup(s: pd.Series, index: pd.Index) -> pd.Series:
    """把 groupby(...).rolling(...)/ewm(...) 產生的多層 index 結果攤平回原始 df 對齊。"""
    n_key_levels = s.index.nlevels - 1
    if n_key_levels:
        s = s.reset_index(level=list(range(n_key_levels)), drop=True)
    return s.reindex(index)


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
    只算 MACD(12,26,9) 衍生特徵 + triple barrier 標籤。

    回傳欄位：FEATURES 全部 + target（若 compute_labels=True）。
    """
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)

    # 用當日開盤正規化收盤價，MACD line/histogram 天生就是「相對 day_open」
    # 的比例值，跨股票可比（比照 rally 的做法）。
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    m1["_m1_close"] = m1["close"] / day_open

    ema_fast = _degroup(g_day["_m1_close"].ewm(span=MACD_FAST, adjust=False).mean(), m1.index)
    ema_slow = _degroup(g_day["_m1_close"].ewm(span=MACD_SLOW, adjust=False).mean(), m1.index)
    m1["macd_line"] = ema_fast - ema_slow

    g_macd = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["macd_signal"] = _degroup(g_macd["macd_line"].ewm(span=MACD_SIGNAL, adjust=False).mean(), m1.index)
    m1["macd_hist"] = m1["macd_line"] - m1["macd_signal"]

    g_macd2 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["macd_hist_slope"] = m1["macd_hist"] - g_macd2["macd_hist"].shift(1)
    m1["macd_hist_slope_3"] = m1["macd_hist"] - g_macd2["macd_hist"].shift(3)

    m1["macd_above_signal"] = (m1["macd_line"] > m1["macd_signal"]).astype(int)
    m1["macd_above_zero"] = (m1["macd_line"] > 0).astype(int)

    prev_line = g_macd2["macd_line"].shift(1)
    prev_signal = g_macd2["macd_signal"].shift(1)
    m1["golden_cross"] = ((prev_line <= prev_signal) & (m1["macd_line"] > m1["macd_signal"])).astype(int)
    m1["dead_cross"] = ((prev_line >= prev_signal) & (m1["macd_line"] < m1["macd_signal"])).astype(int)

    m1["bars_since_golden_cross"] = _bars_since(m1, "golden_cross")
    m1["bars_since_dead_cross"] = _bars_since(m1, "dead_cross")

    g_macd3 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    macd_chg_10 = m1["macd_hist"] - g_macd3["macd_hist"].shift(10)
    price_chg_10 = m1["_m1_close"] - g_macd3["_m1_close"].shift(10)
    m1["macd_divergence"] = macd_chg_10 - price_chg_10

    m1 = m1.drop(columns=["_m1_close"])

    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    return m1
