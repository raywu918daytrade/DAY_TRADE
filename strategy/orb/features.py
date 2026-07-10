"""
特徵工程與資料標籤 — 開盤區間突破（ORB, Opening Range Breakout）+ 3分K/5分K 中期動能確認

三組獨立的開盤區間突破特徵（見 config.py 的 OPENING_RANGE_MINUTES /
OPENING_RANGE_M3_MINUTES / OPENING_RANGE_M5_MINUTES），窗口長度故意不同
（9/15/20分鐘）——高低點本質是「這段時間內所有價格的極值」，同一段時間窗口
不管用幾分鐘一根去分組算出來的極值都一樣，窗口長度不同才是真的在比較不同
時間尺度的區間。每組區間形成期間本身（例如 15分K版的前15分鐘）的樣本會
整批排除——那幾分鐘區間還沒收斂，此時去看「距上緣距離」這類特徵等於用
當天最終的區間高低點回頭算，會把「區間形成期間之後才知道」的值洩漏回
形成期間本身；且真實交易時，區間沒收斂也不可能有突破訊號。
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
from strategy.orb.config import (
    HOLD_BARS,
    OPENING_RANGE_M3_MINUTES,
    OPENING_RANGE_M5_MINUTES,
    OPENING_RANGE_MINUTES,
    SL_PCT,
    TP_PCT,
)

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


def _orb_cols(suffix: str) -> list:
    """一組開盤區間突破特徵的欄位名稱清單（見 _orb_block()）。"""
    return [
        f"or_range_pct{suffix}",
        f"close_pos_in_range{suffix}",
        f"dist_to_or_high{suffix}",
        f"dist_to_or_low{suffix}",
        f"broke_above{suffix}",
        f"broke_below{suffix}",
        f"breakout_up_signal{suffix}",
        f"breakout_down_signal{suffix}",
        f"bars_since_breakout_up{suffix}",
        f"bars_since_breakout_down{suffix}",
        f"vol_ratio_vs_or{suffix}",
    ]


FEATURES = [
    *_orb_cols(""),  # 1分K版開盤區間（15分鐘），見 config.OPENING_RANGE_MINUTES
    "minutes_since_or_end",  # 距1分K版開盤區間結束幾分鐘
    *_orb_cols("_m3"),  # 3分K版開盤區間（9分鐘），見 config.OPENING_RANGE_M3_MINUTES
    *_orb_cols("_m5"),  # 5分K版開盤區間（20分鐘），見 config.OPENING_RANGE_M5_MINUTES
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
    # ── 量能/波動/趨勢強度（1分K，逐日重算不跨日）──────────────────────
    "vol_surge",  # 當根量 / 過去20根（不含當根）均量，跟自己近期比，不是跟開盤區間比
    "m1_atr",  # 1分鐘K ATR(14) 相對波動（/ 當日開盤，比照 rally 的 m1_atr）
    "adx",  # ADX(14) 趨勢強度（用 talib 算，Wilder平滑手刻容易出錯）
    "vwap_dev",  # close 相對當日累積VWAP的偏離（比照 rally 的 vwap_dev）
]


def _degroup(s: pd.Series, index: pd.Index) -> pd.Series:
    """把 groupby(...).rolling(...) 產生的多層 index 結果攤平回原始 df 對齊。"""
    n_key_levels = s.index.nlevels - 1
    if n_key_levels:
        s = s.reset_index(level=list(range(n_key_levels)), drop=True)
    return s.reindex(index)


_ADX_PERIOD = 14


def _adx_group(g: pd.DataFrame) -> pd.Series:
    """單一 stock_id×day_date 分組的 ADX(14)，逐日重算不跨日（理由同 m1_atr）。

    Wilder 平滑遞迴公式手刻容易出錯，用已裝好的 talib 算，比較可靠。
    """
    import talib

    high = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    close = g["close"].to_numpy(dtype=float)
    if len(high) < _ADX_PERIOD * 2:
        return pd.Series(np.full(len(high), np.nan), index=g.index)
    adx = talib.ADX(high, low, close, timeperiod=_ADX_PERIOD)
    return pd.Series(adx, index=g.index)


def _bars_since(m1: pd.DataFrame, flag: pd.Series) -> pd.Series:
    """距最近一次 flag==1 幾根K棒（同一天內），今日尚未發生過則回傳 999。"""
    group_cols = [m1["stock_id"], m1["day_date"]]
    pos = m1.groupby(["stock_id", "day_date"], group_keys=False).cumcount()
    last_true_pos = pos.where(flag == 1)
    last_true_pos = last_true_pos.groupby(group_cols, group_keys=False).ffill()
    bars = (pos - last_true_pos).fillna(999)
    return bars


def _orb_block(m1: pd.DataFrame, window_minutes: int, suffix: str) -> tuple:
    """
    算一組開盤區間突破特徵，區間長度 window_minutes 分鐘，欄位名稱加 suffix。

    回傳 ({欄位名: Series}, in_or遮罩)——in_or 用來把這組欄位在區間形成期間
    本身設為 NaN（見檔頭「區間形成期間」說明），呼叫端負責 assign 回 m1
    並套用遮罩。
    """
    group_keys = [m1["stock_id"], m1["day_date"]]
    in_or = m1["minutes_since_open"] < window_minutes

    or_high = m1["high"].where(in_or).groupby(group_keys, group_keys=False).transform("max")
    or_low = m1["low"].where(in_or).groupby(group_keys, group_keys=False).transform("min")
    or_vol_mean = m1["volume"].where(in_or).groupby(group_keys, group_keys=False).transform("mean")
    or_range = (or_high - or_low).replace(0, np.nan)

    feats = {}
    feats[f"or_range_pct{suffix}"] = or_range / or_low.replace(0, np.nan)
    feats[f"close_pos_in_range{suffix}"] = (m1["close"] - or_low) / or_range
    feats[f"dist_to_or_high{suffix}"] = (m1["close"] - or_high) / or_high.replace(0, np.nan)
    feats[f"dist_to_or_low{suffix}"] = (m1["close"] - or_low) / or_low.replace(0, np.nan)
    feats[f"broke_above{suffix}"] = (m1["close"] > or_high).astype(int)
    feats[f"broke_below{suffix}"] = (m1["close"] < or_low).astype(int)

    prev_close = m1.groupby(["stock_id", "day_date"], group_keys=False)["close"].shift(1)
    up_signal = ((prev_close <= or_high) & (m1["close"] > or_high)).astype(int)
    down_signal = ((prev_close >= or_low) & (m1["close"] < or_low)).astype(int)
    feats[f"breakout_up_signal{suffix}"] = up_signal
    feats[f"breakout_down_signal{suffix}"] = down_signal
    feats[f"bars_since_breakout_up{suffix}"] = _bars_since(m1, up_signal)
    feats[f"bars_since_breakout_down{suffix}"] = _bars_since(m1, down_signal)
    feats[f"vol_ratio_vs_or{suffix}"] = m1["volume"] / or_vol_mean.replace(0, np.nan)

    return feats, in_or


def make_features(m1: pd.DataFrame, compute_labels: bool = True) -> pd.DataFrame:
    """
    算三組獨立開盤區間突破特徵（1分K/3分K/5分K版，窗口長度不同）
    + 3分K/5分K 中期動能確認 + triple barrier 標籤。

    回傳欄位：FEATURES 全部 + target（若 compute_labels=True）。
    每組區間形成期間本身的樣本，該組欄位會被設為 NaN；三組窗口長度不同
    （9/15/20分鐘），所以最終能交易的起點是三組裡最長的那個（20分鐘），
    靠呼叫端 dropna(subset=FEATURES) 篩掉尚未全部備齊的樣本。
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

    # ── 量能突增 / 1分鐘ATR / ADX / VWAP偏離（逐日重算，不跨日）────────────
    # 量能突增：當根量 / 過去20根（不含當根）均量，跟自己近期比，不是跟開盤區間比
    # （vol_ratio_vs_or 系列是跟開盤區間期間比，這裡是另一個獨立的角度）。
    _vol_shift1 = g_day["volume"].shift(1)
    vol_ma_recent = _degroup(
        _vol_shift1.groupby([m1["stock_id"], m1["day_date"]], group_keys=False).rolling(20, min_periods=20).mean(),
        m1.index,
    )
    m1["vol_surge"] = m1["volume"] / vol_ma_recent.replace(0, np.nan)

    # 1分鐘K ATR(14)：True Range 逐日重算，rolling mean 近似（跟 rally 的 m1_atr 同做法）
    _prev_close = g_day["close"].shift(1).fillna(m1["open"])
    m1["_tr"] = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - _prev_close).abs()),
        (m1["low"] - _prev_close).abs(),
    )
    g_tr = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["m1_atr"] = _degroup(g_tr["_tr"].rolling(14, min_periods=14).mean(), m1.index) / day_open

    # ADX(14)：逐日逐股用 talib 算（見 _adx_group）
    m1["adx"] = m1.groupby(["stock_id", "day_date"], group_keys=False).apply(_adx_group, include_groups=False)

    # VWAP 偏離：close 相對當日累積VWAP的偏離（跟 rally 的 vwap_dev 同做法）
    m1["_cum_vol"] = g_day["volume"].transform("cumsum")
    m1["_pv"] = m1["close"] * m1["volume"]
    m1["_cum_pv"] = g_day["_pv"].transform("cumsum")
    _vwap = m1["_cum_pv"] / m1["_cum_vol"].replace(0, np.nan)
    m1["vwap_dev"] = (m1["close"] - _vwap) / _vwap.replace(0, np.nan)

    m1 = m1.drop(columns=["_tr", "_cum_vol", "_pv", "_cum_pv"])

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

    # ── 三組獨立的開盤區間突破特徵（窗口長度不同，見 config.py 說明）───────
    m1_orb_feats, in_or_1m = _orb_block(m1, OPENING_RANGE_MINUTES, "")
    for col, series in m1_orb_feats.items():
        m1[col] = series
    m1["minutes_since_or_end"] = m1["minutes_since_open"] - OPENING_RANGE_MINUTES
    m1.loc[in_or_1m, list(m1_orb_feats.keys()) + ["minutes_since_or_end"]] = np.nan

    m3_orb_feats, in_or_m3 = _orb_block(m1, OPENING_RANGE_M3_MINUTES, "_m3")
    for col, series in m3_orb_feats.items():
        m1[col] = series
    m1.loc[in_or_m3, list(m3_orb_feats.keys())] = np.nan

    m5_orb_feats, in_or_m5 = _orb_block(m1, OPENING_RANGE_M5_MINUTES, "_m5")
    for col, series in m5_orb_feats.items():
        m1[col] = series
    m1.loc[in_or_m5, list(m5_orb_feats.keys())] = np.nan

    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    return m1
