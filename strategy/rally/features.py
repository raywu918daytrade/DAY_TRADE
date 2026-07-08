"""
特徵工程與資料標籤 — RFC / XGB / LGBM 三模型共用

只用 1 分鐘線收盤價與成交量等衍生特徵，搭配 3分K/5分K/日K/大盤(0050) 背景特徵，
用 triple barrier 標籤未來 30 根分K內：
  +3% 停利先碰到 → target = 1（漲）
  -3% 停損先碰到 → target = 0（跌）

載入 db/m1/ 歷史分K（與 date_trade_model.py 共用），另載入 db/fugle_day/ 日K
提供過去 5 天日K 背景特徵（day_ret_1~5、day_vol_1~5）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 確保能從根目錄導入
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.query import load_day, load_m1, load_m3, load_m5
from strategy.rally.config import HOLD_BARS, SL_PCT, TP_PCT

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

_M1_DIR = _ROOT / "db/m1"
_DAY_DIR = _ROOT / "db/fugle_day"
_CACHE_PATH = _ROOT / "cache/m1_rfc_features.parquet"


# ── Cache 管理 ───────────────────────────────────────────────────────────────


def _m1_mtime() -> float:
    """db/m1/ 中最新檔案的修改時間戳。"""
    mtimes = [f.stat().st_mtime for f in _M1_DIR.iterdir() if f.suffix == ".parquet"]
    return max(mtimes) if mtimes else 0


def _day_mtime() -> float:
    """db/fugle_day/ 中最新檔案的修改時間戳。"""
    if not _DAY_DIR.exists():
        return 0
    mtimes = [f.stat().st_mtime for f in _DAY_DIR.iterdir() if f.suffix == ".parquet"]
    return max(mtimes) if mtimes else 0


def _cache_is_fresh() -> bool:
    """檢查 cache 是否比 db/m1/ 與 db/fugle_day/ 所有檔案都新。"""
    if not _CACHE_PATH.exists():
        return False
    cache_mtime = _CACHE_PATH.stat().st_mtime
    return cache_mtime >= _m1_mtime() and cache_mtime >= _day_mtime()


def load_features(force_rebuild: bool = False) -> pd.DataFrame:
    """
    載入特徵（自動使用 / 重建 cache）。

    若 cache 存在且 db/m1/ 與 db/fugle_day/ 都沒有新檔案，直接讀取 cache parquet。
    否則重新執行 make_features() 並更新 cache。
    """
    if not force_rebuild and _cache_is_fresh():
        cached = pd.read_parquet(_CACHE_PATH)
        missing = [c for c in FEATURES if c not in cached.columns]
        if not missing:
            print("  讀取 cache 特徵...")
            return cached
        print(f"  cache 缺少欄位 {missing}，重新計算...")

    print("  重新計算特徵（db/m1 有更新）...")
    m1 = load_m1()
    day = load_day()
    df = make_features(m1, day=day, compute_labels=True)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 只存特徵欄位 + time/target，去掉原始 m1 的 raw ohlcv 避免體積過大
    meta_cols = ["stock_id", "date", "day_date", "hour", "minute", "target"]
    cache_cols = [c for c in df.columns if c in set(FEATURES) | set(meta_cols)]
    df[cache_cols].to_parquet(_CACHE_PATH)
    print(f"  cache 已存至 {_CACHE_PATH}（{len(df):,} 筆）")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Triple Barrier Label
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
    result = m1.groupby(["stock_id", "day_date"], group_keys=False).apply(
        lambda g: pd.Series(
            _barrier_label_group(g["close"].values),
            index=g.index,
        ),
        include_groups=False,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 特徵工程（close、volume 與過去 5 天日K）
# ═══════════════════════════════════════════════════════════════════════════════


FEATURES = [
    # 1分鐘K OHLCV（價格除以當日開盤正規化，量除以當日累積均量）
    "m1_open",
    "m1_high",
    "m1_low",
    "m1_close",
    "m1_volume",
    "m1_atr",  # 1分鐘K ATR(14) 相對波動（/ 當日開盤）
    # 3分鐘K OHLCV（當前 + 前2根，每根間隔3分鐘）
    "m3_open",
    "m3_high",
    "m3_low",
    "m3_close",
    "m3_volume",
    "m3_ret",  # 3分鐘K報酬率（K棒間變化%）
    "m3_open_lag1",
    "m3_high_lag1",
    "m3_low_lag1",
    "m3_close_lag1",
    "m3_volume_lag1",
    "m3_open_lag2",
    "m3_high_lag2",
    "m3_low_lag2",
    "m3_close_lag2",
    "m3_volume_lag2",
    # 5分鐘K OHLCV（當前 + 前1根，每根間隔5分鐘）
    "m5_open",
    "m5_high",
    "m5_low",
    "m5_close",
    "m5_volume",
    "m5_ret",  # 5分鐘K報酬率（K棒間變化%）
    "m5_open_lag1",
    "m5_high_lag1",
    "m5_low_lag1",
    "m5_close_lag1",
    "m5_volume_lag1",
    # 報酬率與量比
    "ret_1",
    "vol_ratio",
    "tf3_ret",
    "tf3_vol_ratio",
    "tf5_ret",
    "tf5_vol_ratio",
    # 強過濾：破底翻（第1根5分鐘跌，第2根漲）
    "breakout_signal",
    # 日K 特徵（過去 5 天，無未來洩漏）
    "day_ret_1",
    "day_ret_2",
    "day_ret_3",
    "day_ret_4",
    "day_ret_5",
    "day_vol_1",
    "day_vol_2",
    "day_vol_3",
    "day_vol_4",
    "day_vol_5",
    "day_atr",  # 日K ATR(14) 相對波動（前一日，/ 當日開盤）
    # 大盤代理（0050）日K 特徵（前 5 天，無未來洩漏，廣播至所有個股）
    "idx_day_ret_1",
    "idx_day_ret_2",
    "idx_day_ret_3",
    "idx_day_ret_4",
    "idx_day_ret_5",
    "idx_day_atr",  # 0050 日K ATR(14) 相對波動（前一日，/ 當日開盤）
    # 大盤代理（0050）1分K 特徵（廣播至所有個股）
    "idx_ret_1",  # 0050 前1分報酬率
    "idx_vs_open",  # 0050 收盤 / 0050 當日開盤（大盤相對開盤漲跌幅）
    "idx_atr",  # 0050 1分K ATR(14) 相對波動（/ 0050 當日開盤）
    "idx_up",  # 0050 收盤 > 開盤（0/1，大盤是否站上開盤）
    "idx_breakout",  # 0050 破底翻（前1分跌 + 當分漲，0/1）
]


def make_features(
    m1: pd.DataFrame,
    m3: pd.DataFrame | None = None,
    m5: pd.DataFrame | None = None,
    day: pd.DataFrame | None = None,
    compute_labels: bool = True,
) -> pd.DataFrame:
    """
    簡單特徵工程：close、volume 衍生特徵 + 過去 5 天日K 背景特徵。

    回傳欄位：
      ret_1, vol_ratio, tf3_ret, tf3_vol_ratio, tf5_ret, tf5_vol_ratio,
      day_ret_1~5, day_vol_1~5（過去 5 天日K 報酬率與量比）,
      target（若 compute_labels=True）
    """
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)

    # ── 當日開盤價（第一根 open）用於正規化 ──────────────────────────
    day_open = g_day["open"].transform("first").replace(0, np.nan)

    # ── 1分鐘K OHLCV（價格 / 當日開盤，量用當日累積均量正規化） ─────
    m1["m1_open"] = m1["open"] / day_open
    m1["m1_high"] = m1["high"] / day_open
    m1["m1_low"] = m1["low"] / day_open
    m1["m1_close"] = m1["close"] / day_open
    # 量比（當前量 / 當日累積平均量）
    m1["_cum_vol"] = g_day["volume"].transform("cumsum")
    m1["_bar_count"] = g_day["volume"].transform("cumcount") + 1
    m1["m1_volume"] = m1["volume"] / (m1["_cum_vol"] / m1["_bar_count"]).replace(0, np.nan)

    # ── 1分鐘K ATR（日內，不跨日，ATR(14) 正規化為相對波動）──────
    # True Range = max(H-L, |H-前收|, |L-前收|)，首日第一根前收以開盤替代
    _prev_close = g_day["close"].shift(1).fillna(m1["open"])
    m1["_tr"] = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - _prev_close).abs()),
        (m1["low"] - _prev_close).abs(),
    )
    m1["m1_atr"] = g_day["_tr"].transform(lambda x: x.rolling(14, min_periods=14).mean()) / day_open

    # ── 載入預先聚合的 db/m3、db/m5 ──────────────────────────────────
    if m3 is None:
        m3 = load_m3()
    if m5 is None:
        m5 = load_m5()
    m3["date"] = pd.to_datetime(m3["date"])
    m5["date"] = pd.to_datetime(m5["date"])

    # 正規化 m3/m5 價格，量先保留 raw sum 等 merge 後再算 pct_change
    m3_feat = m3[["stock_id", "date"]].copy()
    m3_feat["m3_open"] = m3["open"] / day_open
    m3_feat["m3_high"] = m3["high"] / day_open
    m3_feat["m3_low"] = m3["low"] / day_open
    m3_feat["m3_close"] = m1["close"] / day_open
    m3_feat["m3_vol_raw"] = m3["volume"].values  # raw sum

    m5_feat = m5[["stock_id", "date"]].copy()
    m5_feat["m5_open"] = m5["open"] / day_open
    m5_feat["m5_high"] = m5["high"] / day_open
    m5_feat["m5_low"] = m5["low"] / day_open
    m5_feat["m5_close"] = m1["close"] / day_open
    m5_feat["m5_vol_raw"] = m5["volume"].values  # raw sum

    # merge 到 m1
    m1 = m1.merge(m3_feat, on=["stock_id", "date"], how="left", suffixes=("", "_y3"))
    m1 = m1.merge(m5_feat, on=["stock_id", "date"], how="left", suffixes=("", "_y5"))

    # ── 3分鐘K volume pct_change + lag1/lag2 ────────────────────────
    g_m3 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["m3_volume"] = g_m3["m3_vol_raw"].transform(lambda x: x.pct_change(3))
    for col in ["m3_open", "m3_high", "m3_low", "m3_close", "m3_volume"]:
        m1[f"{col}_lag1"] = g_m3[col].transform(lambda x: x.shift(3))
        m1[f"{col}_lag2"] = g_m3[col].transform(lambda x: x.shift(6))
    m1["m3_ret"] = g_m3["m3_close"].transform(lambda x: x.pct_change(3))

    # ── 5分鐘K volume pct_change + lag1 ─────────────────────────────
    g_m5 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["m5_volume"] = g_m5["m5_vol_raw"].transform(lambda x: x.pct_change(5))
    for col in ["m5_open", "m5_high", "m5_low", "m5_close", "m5_volume"]:
        m1[f"{col}_lag1"] = g_m5[col].transform(lambda x: x.shift(5))
    m1["m5_ret"] = g_m5["m5_close"].transform(lambda x: x.pct_change(5))

    # ── 強過濾：破底翻訊號 ───────────────────────────────────────────
    # 第1根5分鐘K跌（lag1的ret < 0），第2根5分鐘K漲（當前ret > 0）
    m1["breakout_signal"] = (g_m5["m5_ret"].transform(lambda x: x.shift(5)) < 0) & (m1["m5_ret"] > 0)

    # ── 報酬率與量比 ────────────────────────────────────────────────
    # 前1分鐘報酬率
    m1["ret_1"] = g_day["close"].transform(lambda x: x.pct_change(1))

    # 量比（當前量 / 前1分鐘量）
    m1["vol_ratio"] = g_day["volume"].transform(lambda x: x / x.shift(1).replace(0, np.nan))

    # 3分鐘K報酬率
    m1["tf3_ret"] = g_day["close"].transform(lambda x: x.pct_change(3))

    # 3分鐘K量比
    m1["tf3_vol_ratio"] = g_day["volume"].transform(
        lambda x: x.rolling(3).sum() / x.rolling(3).sum().shift(3).replace(0, np.nan)
    )

    # 5分鐘K報酬率
    m1["tf5_ret"] = g_day["close"].transform(lambda x: x.pct_change(5))

    # 5分鐘K量比
    m1["tf5_vol_ratio"] = g_day["volume"].transform(
        lambda x: x.rolling(5).sum() / x.rolling(5).sum().shift(5).replace(0, np.nan)
    )

    # ── 日K 特徵（過去 5 天，無未來洩漏）─────────────────────────────
    if day is None:
        day = load_day()
    day = day.copy()
    day["date"] = pd.to_datetime(day["date"])
    dg = day.groupby("stock_id")
    # 短線均量（5日），避免 rolling(20) 吃掉過多歷史資料
    vol_ma5 = dg["volume"].transform(lambda x: x.rolling(5).mean()).replace(0, np.nan)
    day_ret_cols, day_vol_cols = [], []
    for lag in range(1, 6):
        cr, cv = f"day_ret_{lag}", f"day_vol_{lag}"
        # day_ret_1 = 前1日報酬率，day_ret_2 = 前2日，以此類推
        day[cr] = dg["close"].transform(lambda x, l=lag: x.pct_change(1).shift(l - 1))
        # day_vol_1 = 前1日量 / 5日均量，以此類推
        day[cv] = dg["volume"].transform(lambda x, l=lag: (x / vol_ma5.loc[x.index]).shift(l - 1))
        day_ret_cols.append(cr)
        day_vol_cols.append(cv)
    # ── 日K ATR（ATR(14)，用前一日避免當日洩漏，以當日開盤正規化）──
    day["_prev_close"] = dg["close"].shift(1)
    day["_day_tr"] = np.maximum(
        np.maximum((day["high"] - day["low"]).abs(), (day["high"] - day["_prev_close"]).abs()),
        (day["low"] - day["_prev_close"]).abs(),
    )
    day["_atr14"] = dg["_day_tr"].transform(lambda x: x.rolling(14, min_periods=14).mean())
    day["day_atr"] = day["_atr14"].shift(1) / day["open"].replace(0, np.nan)

    day["day_date"] = day["date"].dt.date
    day_feat_cols = ["stock_id", "day_date"] + day_ret_cols + day_vol_cols + ["day_atr"]
    m1 = m1.merge(day[day_feat_cols], on=["stock_id", "day_date"], how="left")

    # ── 大盤（0050）日K 特徵（前 5 天，廣播至所有個股）────────────────
    idx_day = day[day["stock_id"] == "0050"].copy()
    if not idx_day.empty:
        idx_dg = idx_day.groupby("stock_id")
        idx_vol_ma5 = idx_dg["volume"].transform(lambda x: x.rolling(5).mean()).replace(0, np.nan)
        idx_day_ret_cols = []
        for lag in range(1, 6):
            cr = f"idx_day_ret_{lag}"
            idx_day[cr] = idx_dg["close"].transform(lambda x, l=lag: x.pct_change(1).shift(l - 1))
            idx_day_ret_cols.append(cr)
        # 0050 日K ATR
        idx_day["_idx_prev_close"] = idx_dg["close"].shift(1)
        idx_day["_idx_day_tr"] = np.maximum(
            np.maximum((idx_day["high"] - idx_day["low"]).abs(), (idx_day["high"] - idx_day["_idx_prev_close"]).abs()),
            (idx_day["low"] - idx_day["_idx_prev_close"]).abs(),
        )
        idx_day["_idx_atr14"] = idx_dg["_idx_day_tr"].transform(lambda x: x.rolling(14, min_periods=14).mean())
        idx_day["idx_day_atr"] = idx_day["_idx_atr14"].shift(1) / idx_day["open"].replace(0, np.nan)
        idx_day["day_date"] = idx_day["date"].dt.date
        idx_day_feat_cols = ["day_date"] + idx_day_ret_cols + ["idx_day_atr"]
        m1 = m1.merge(idx_day[idx_day_feat_cols], on=["day_date"], how="left")

    # ── 大盤（0050）1分K 特徵（廣播至所有個股）──────────────────────────
    # 欄位一律先建立（預設 NaN）：predict_live() 用的 m1_live 是當沖候選清單
    # （依成交量篩出，0050 本身不是候選股，只當特徵用），理論上 live_trader.py
    # 的 _get_stocks() 已固定把 0050 加進即時輪詢清單，但若某天剛好抓不到
    # 0050（歷史回放、API 失敗等），這裡也不該讓 FEATURES 缺欄位、
    # 讓 dropna(subset=FEATURES) 直接 KeyError——缺資料時該幾欄就是 NaN，
    # 之後 dropna 自然會把當天樣本篩掉，行為才符合預期。
    for _col in ["idx_ret_1", "idx_vs_open", "idx_atr", "idx_up", "idx_breakout"]:
        m1[_col] = np.nan

    idx_m1 = m1[m1["stock_id"] == "0050"].copy()
    if not idx_m1.empty:
        idx_g_day = idx_m1.groupby("day_date", group_keys=False)
        # 0050 當日開盤價
        idx_day_open = idx_g_day["open"].transform("first").replace(0, np.nan)
        # 0050 前1分報酬率
        m1.loc[idx_m1.index, "idx_ret_1"] = idx_g_day["close"].transform(lambda x: x.pct_change(1)).values
        # 0050 收盤 / 當日開盤（大盤相對開盤漲跌幅）
        m1.loc[idx_m1.index, "idx_vs_open"] = (idx_m1["close"] / idx_day_open).values
        # 0050 1分K ATR(14) 相對波動
        idx_prev_close = idx_g_day["close"].shift(1).fillna(idx_m1["open"])
        idx_m1["_idx_tr"] = np.maximum(
            np.maximum((idx_m1["high"] - idx_m1["low"]).abs(), (idx_m1["high"] - idx_prev_close).abs()),
            (idx_m1["low"] - idx_prev_close).abs(),
        )
        m1.loc[idx_m1.index, "idx_atr"] = (
            idx_g_day["_idx_tr"].transform(lambda x: x.rolling(14, min_periods=14).mean()) / idx_day_open
        ).values
        # 0050 收盤 > 開盤（0/1）
        m1.loc[idx_m1.index, "idx_up"] = (idx_m1["close"] > idx_day_open).astype(int).values
        # 0050 破底翻（前1分跌 + 當分漲）
        m1.loc[idx_m1.index, "idx_breakout"] = (
            (
                (idx_g_day["close"].transform(lambda x: x.pct_change(1)).shift(1) < 0)
                & (idx_g_day["close"].transform(lambda x: x.pct_change(1)) > 0)
            )
            .astype(int)
            .values
        )
        # 廣播 0050 特徵至所有個股（以完整時間戳 date 為 key，逐分鐘對齊，
        # 不能只用 day_date——那樣 drop_duplicates 只會留下每天第一根，
        # 而第一根的 idx_ret_1/idx_atr 必為 NaN，等於整欄報廢）
        idx_feat_cols = ["date", "idx_ret_1", "idx_vs_open", "idx_atr", "idx_up", "idx_breakout"]
        idx_feat = m1.loc[idx_m1.index, idx_feat_cols].drop_duplicates("date")
        m1 = m1.drop(columns=["idx_ret_1", "idx_vs_open", "idx_atr", "idx_up", "idx_breakout"], errors="ignore")
        m1 = m1.merge(idx_feat, on=["date"], how="left")

    # 清除暫存欄位
    m1 = m1.drop(
        columns=[
            "_cum_vol",
            "_bar_count",
            "m3_vol_raw",
            "m5_vol_raw",
            "_tr",
            "_idx_prev_close",
            "_idx_day_tr",
            "_idx_atr14",
            "_idx_tr",
        ],
        errors="ignore",
    )

    # 時間特徵（用於過濾時段，不作為模型輸入）
    m1["hour"] = m1["date"].dt.hour
    m1["minute"] = m1["date"].dt.minute

    # 目標標籤
    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    return m1
