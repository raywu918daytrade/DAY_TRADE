"""
當沖策略 — RandomForest 簡單模型

== 策略邏輯 ==

只用 1 分鐘線收盤價與成交量 2 個特徵，預測未來 30 根分K內：
  +3% 停利先碰到 → label = 1（漲）
  -3% 停損先碰到 → label = 0（跌）

== 資料 ==

載入 db/m1/ 歷史分K（與 date_trade_model.py 共用）
只用 close 與 volume 兩個欄位做特徵。
另載入 db/fugle_day/ 日K，提供過去 5 天日K 背景特徵（day_ret_1~5、day_vol_1~5）。

== Main 模式 ==

train      訓練 RandomForest 模型
validate   驗證模型（信心度分析 + 召回率分析）
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 確保能從根目錄導入
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from data.query import load_day, load_m1, load_m3, load_m5, load_m1_live

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_MODEL_PATH = _ROOT / "models/m1_rfc.pkl"
_MODEL_PATH_XGB = _ROOT / "models/m1_xgb.pkl"
_MODEL_PATH_LGBM = _ROOT / "models/m1_lgbm_breakout.pkl"
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


# ── Triple Barrier 參數 ──────────────────────────────────────────────────────
TP_PCT = 0.03
SL_PCT = 0.03
HOLD_BARS = 30

# ── 交易時段 ──────────────────────────────────────────────────────────────────
SESSION_START = (9, 1)
_end_h = int(os.environ.get("SESSION_END_HOUR", "10"))
_end_m = int(os.environ.get("SESSION_END_MIN", "0"))
SESSION_END = (_end_h, _end_m)


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

    # 清除暫存欄位
    m1 = m1.drop(columns=["_cum_vol", "_bar_count", "m3_vol_raw", "m5_vol_raw", "_tr"], errors="ignore")

    # 時間特徵（用於過濾時段，不作為模型輸入）
    m1["hour"] = m1["date"].dt.hour
    m1["minute"] = m1["date"].dt.minute

    # 目標標籤
    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    return m1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 訓練
# ═══════════════════════════════════════════════════════════════════════════════


def train(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    訓練 RandomForest 模型。

    Parameters
    ----------
    test_days : int
        以最後 N 天為測試集（預設 10）。若指定 start_date/end_date 則忽略此參數。
    start_date : str
        訓練資料起日，格式 "YYYY-MM-DD"。留空表示不限制。
    end_date : str
        訓練資料迄日，格式 "YYYY-MM-DD"。留空表示不限制。
    """
    print("特徵工程...")
    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵: {FEATURES}")

    # 只保留早盤時段
    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()
    print(f"  早盤有效樣本: {len(df):,} 筆")

    # 日期過濾
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    print(f"  日期區間: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

    # 時間切割（若有指定明確日期範圍則以日期切割，否則用 test_days）
    if start_date and end_date:
        cutoff = df["date"].quantile(0.8)  # 前 80% 訓練、後 20% 測試
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    else:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    print(
        f"  訓練: {len(train_df):,} ({train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"  測試: {len(test_df):,} ({test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')})"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}")

    return model


def load_model():
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型，請先執行 train()")
    return joblib.load(_MODEL_PATH)


def load_model_xgb():
    if not _MODEL_PATH_XGB.exists():
        raise FileNotFoundError(f"找不到 XGB 模型，請先執行 train_xgb()")
    return joblib.load(_MODEL_PATH_XGB)


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError(f"找不到 LGBM 模型，請先執行 train_lgbm()")
    return joblib.load(_MODEL_PATH_LGBM)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.5 XGBoost / LightGBM 訓練（與 RFC 共用同一份 FEATURES 與標籤）
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare_train_test(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """載入特徵並切分早盤訓練/測試集（三模型共用）。"""
    print("特徵工程...")
    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵數: {len(FEATURES)}")

    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()
    print(f"  早盤有效樣本: {len(df):,} 筆")

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    print(f"  日期區間: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

    if start_date and end_date:
        cutoff = df["date"].quantile(0.8)
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    else:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    print(
        f"  訓練: {len(train_df):,} ({train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"  測試: {len(test_df):,} ({test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')})"
    )
    return train_df, test_df


def train_xgb(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """訓練 XGBoost 模型（與 RFC 共用 FEATURES）。"""
    from xgboost import XGBClassifier

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_XGB.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_XGB)
    print(f"模型已存至 {_MODEL_PATH_XGB}")
    return model


def train_lgbm(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """訓練 LightGBM 模型（與 RFC 共用 FEATURES）。"""
    import lightgbm as lgb

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_LGBM.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_LGBM)
    print(f"模型已存至 {_MODEL_PATH_LGBM}")
    return model


def _breakout_scan(model, test_df: pd.DataFrame) -> dict:
    """對單一模型在破底翻黃金窗口做門檻掃描，回傳各門檻指標。"""
    sh, sm = BREAKOUT_TRADE_START
    eh, em = BREAKOUT_TRADE_END
    sub = test_df[test_df["breakout_signal"]].copy()
    mask = (sub["hour"] == sh) & (sub["minute"] >= sm) & (sub["hour"] == eh) & (sub["minute"] <= em)
    sub = sub[mask].copy()
    if sub.empty:
        return {}
    sub["proba"] = model.predict_proba(sub[FEATURES])[:, 1]
    total_pos = int(sub["target"].sum())
    out = {"total_pos": total_pos}
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        s = sub[sub["proba"] >= thr]
        tp = int((s["target"] == 1).sum()) if len(s) else 0
        out[thr] = {
            "n": len(s),
            "tp": tp,
            "win": s["target"].mean() if len(s) else float("nan"),
            "recall": tp / total_pos if total_pos else 0,
        }
    return out


def compare_breakout(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    三模型（RFC / XGB / LGBM）在破底翻黃金窗口的門檻掃描對照。
    使用同一份測試集與同一組 FEATURES，公平比較。
    """
    models = {
        "RFC ": (load_model, _MODEL_PATH),
        "XGB ": (load_model_xgb, _MODEL_PATH_XGB),
        "LGBM": (load_model_lgbm, _MODEL_PATH_LGBM),
    }
    loaded = {}
    for name, (loader, path) in models.items():
        if not path.exists():
            print(f"  （跳過 {name.strip()}：模型不存在，請先 train_{name.strip().lower()}）")
            continue
        loaded[name] = loader()

    if not loaded:
        print("  無可用模型，請先訓練 RFC / XGB / LGBM。")
        return

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    scans = {name: _breakout_scan(m, test_df) for name, m in loaded.items()}

    print("\n── 破底翻黃金窗口 門檻掃描對照（9:14~9:30）──")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        print(f"\n  門檻 {thr:.2f}")
        print(f"    {'模型':<5} {'訊號數':>7} {'抓到漲':>7} {'勝率':>7} {'召回率':>7}")
        print("    " + "-" * 36)
        for name, sc in scans.items():
            if not sc or thr not in sc:
                continue
            r = sc[thr]
            print(f"    {name:<5} {r['n']:>7,} {r['tp']:>7,} {r['win']*100:>6.1f}% {r['recall']*100:>6.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 驗證
# ═══════════════════════════════════════════════════════════════════════════════


def confidence_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """依信心度區間顯示樣本數與勝率。"""
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])

    # 日期過濾
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    bins = [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.01]
    labels = [
        "<0.30",
        "0.30-0.35",
        "0.35-0.40",
        "0.40-0.45",
        "0.45-0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.70",
        "≥0.70",
    ]
    test_df["bucket"] = pd.cut(test_df["proba"], bins=bins, labels=labels, right=False)

    report = (
        test_df.groupby("bucket", observed=True)
        .agg(樣本數=("target", "count"), 勝率=("target", "mean"))
        .assign(勝率=lambda x: (x["勝率"] * 100).round(1).astype(str) + "%")
    )
    print("\n── 信心度分析（測試集）──")
    print(report.to_string())
    return report


def coverage_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """召回率分析：在不同門檻下的精確率與召回率。"""
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])

    # 日期過濾
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    total_pos = test_df["target"].sum()
    total_neg = (test_df["target"] == 0).sum()

    print(f"\n── 召回率分析（測試集）──")
    print(f"  實際漲（label=1）: {total_pos:,} 筆")
    print(f"  實際跌（label=0）: {total_neg:,} 筆")
    print()
    print(f"  {'門檻':>6}  {'訊號數':>7}  {'精確率':>7}  {'召回率':>7}  {'F1':>6}")
    print("  " + "-" * 45)

    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        flagged = test_df["proba"] >= thr
        tp = (flagged & (test_df["target"] == 1)).sum()
        precision = tp / flagged.sum() if flagged.sum() > 0 else 0
        recall = tp / total_pos if total_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {thr:.2f}  {flagged.sum():>7,}  {precision*100:>6.1f}%  {recall*100:>6.1f}%  {f1:.3f}")


def breakout_filter_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    強過濾評估：以 breakout_signal（先跌後漲破底翻）作為「硬過濾」，
    觀察勝率變化。

    在 9:11 時，breakout_signal 比較的是：
         前一根 M5（9:01→9:06）報酬率 < 0  → 先跌
         當前 M5（9:06→9:11）報酬率 > 0  → 後漲
     即「前 2 根 m5 先跌、後漲」的破底翻型態。

     本函式驗證：若只對 breakout_signal=True 的樣本下注，
     勝率是否高於全體基準（這就是「強過濾」的效果）。
    """
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    total = len(test_df)
    base_win = test_df["target"].mean()

    thr = 0.55
    # 僅模型門檻（不含破底翻）—— 用來對照「49.4% 主要是誰貢獻的」
    p_only = test_df[test_df["proba"] >= thr]
    p_only_win = p_only["target"].mean() if len(p_only) else float("nan")

    # 強過濾：只保留 breakout_signal=True 的樣本
    filt = test_df[test_df["breakout_signal"]]
    filt_win = filt["target"].mean() if len(filt) else float("nan")

    # 強過濾 + 模型門檻 0.55（雙重確認）
    both = filt[filt["proba"] >= thr]
    both_win = both["target"].mean() if len(both) else float("nan")

    print("\n── 強過濾（breakout_signal：先跌後漲）評估 ──")
    print(f"  全體基準              : {total:>7,} 筆  勝率 {base_win*100:>5.1f}%")
    print(f"  僅模型 proba≥{thr}     : {len(p_only):>7,} 筆  勝率 {p_only_win*100:>5.1f}%")
    print(
        f"  僅 breakout=True      : {len(filt):>7,} 筆  勝率 {filt_win*100:>5.1f}%  (佔全體 {len(filt)/total*100:.1f}%)"
    )
    print(f"  breakout + proba≥{thr}: {len(both):>7,} 筆  勝率 {both_win*100:>5.1f}%")


# 強過濾破底翻交易時段（黃金窗口）
BREAKOUT_TRADE_START = (9, 14)
BREAKOUT_TRADE_END = (9, 30)


def breakout_minute_report(
    model=None,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    threshold: float = 0.0,
):
    """
    強過濾破底翻：只看 breakout_signal=True 的樣本，
    逐分鐘顯示：推論數、平均信心度、勝率。

    交易時段限制為黃金窗口 9:14~9:30（BREAKOUT_TRADE_START ~ BREAKOUT_TRADE_END），
    這段破底翻勝率明顯高於 9:30 之後。

    9:14 是 breakout_signal 第一個有效分鐘（需 9:01→9:06 與 9:06→9:11 兩根 M5）。

    threshold: 信心度門檻（預設 0.0 = 不過濾，看全部破底翻樣本）。
        設成例如 0.45 時，逐分鐘表與彙總只統計 proba≥threshold 的樣本。
    """
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    # ── 只留強過濾破底翻 ──────────────────────────────────────────
    test_df = test_df[test_df["breakout_signal"]]

    # ── 黃金窗口 9:14 ~ 9:30 ─────────────────────────────────────
    sh, sm = BREAKOUT_TRADE_START
    eh, em = BREAKOUT_TRADE_END
    mask = (test_df["hour"] == sh) & (test_df["minute"] >= sm) & (test_df["hour"] == eh) & (test_df["minute"] <= em)
    test_df = test_df[mask].copy()

    if test_df.empty:
        print(f"\n── 強過濾破底翻：{sh}:{sm:02d}~{eh}:{em:02d} 逐分鐘 ──")
        print("  （該區間無 breakout_signal=True 樣本）")
        return

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    # ── 信心度門檻掃描（在破底翻樣本內，看各門檻勝率 + 召回）──────
    total_pos = test_df["target"].sum()  # 破底翻樣本中實際漲（label=1）總數
    print(f"\n── 強過濾破底翻：信心度門檻掃描（{sh}:{sm:02d}~{eh}:{em:02d}）──")
    print(f"  破底翻樣本實際漲總數: {total_pos:,} 筆（程式要從中抓出多少）")
    print(f"  {'門檻':>6}  {'訊號數':>7}  {'抓到漲':>7}  {'勝率':>6}  {'召回率':>6}")
    print("  " + "-" * 40)
    for thr in [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sub = test_df[test_df["proba"] >= thr]
        if len(sub) == 0:
            print(f"  {thr:.2f}  {0:>7,}  {0:>7,}  {'--':>6}  {'--':>6}")
        else:
            tp = int((sub["target"] == 1).sum())
            recall = tp / total_pos if total_pos > 0 else 0
            print(f"  {thr:.2f}  {len(sub):>7,}  {tp:>7,}  {sub['target'].mean()*100:>5.1f}%  {recall*100:>5.1f}%")

    # ── 套用使用者指定門檻，做逐分鐘表 ───────────────────────────
    filt = test_df[test_df["proba"] >= threshold]

    if filt.empty:
        print(f"\n── 強過濾破底翻：{sh}:{sm:02d}~{eh}:{em:02d} 逐分鐘（proba≥{threshold:.2f}）──")
        print("  （該門檻下無樣本）")
        return

    grp = filt.groupby("minute").agg(
        推論數=("target", "count"),
        平均信心度=("proba", "mean"),
        勝率=("target", "mean"),
    )
    grp["平均信心度"] = (grp["平均信心度"] * 100).round(1)
    grp["勝率"] = (grp["勝率"] * 100).round(1)
    grp = grp.reset_index()
    grp.insert(0, "時間", grp["minute"].apply(lambda m: f"9:{m:02d}"))

    print(f"\n── 強過濾破底翻：{sh}:{sm:02d}~{eh}:{em:02d} 逐分鐘（proba≥{threshold:.2f}）──")
    print(grp[["時間", "推論數", "平均信心度", "勝率"]].to_string(index=False))

    total_n = len(filt)
    print(
        f"\n  {sh}:{sm:02d}~{eh}:{em:02d} 彙總（proba≥{threshold:.2f}）：推論 {total_n:,} 筆，"
        f"平均信心度 {filt['proba'].mean()*100:.1f}%，"
        f"勝率 {filt['target'].mean()*100:.1f}%"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 特徵重要性
# ═══════════════════════════════════════════════════════════════════════════════


def feature_importance(model=None, top_n: int = 10):
    """顯示 RandomForest 特徵重要性。"""
    if model is None:
        model = load_model()

    print("\n── 特徵重要性 ──")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s}  {imp:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.5 即時推論（含強過濾）
# ═══════════════════════════════════════════════════════════════════════════════


def predict(
    model=None,
    breakout_filter: bool = False,
    test_days: int = 10,
) -> pd.DataFrame:
    """
    對早盤時段產生預測機率矩陣（index=datetime, columns=stock_id）。

    breakout_filter=True 時，只保留 breakout_signal=True 的樣本
    （強過濾：先跌後漲破底翻才納入預測）。
    """
    if model is None:
        model = load_model()

    df = load_features()
    df = df.dropna(subset=FEATURES)

    if breakout_filter:
        df = df[df["breakout_signal"]]

    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()

    df["proba"] = model.predict_proba(df[FEATURES])[:, 1]
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def predict_live(
    minute_str: str,
    model=None,
    threshold: float = 0.55,
    use_breakout_filter: bool = True,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論 + 強過濾（breakout_signal 硬規則）。

    若 use_breakout_filter=True（預設，即「強做強過濾」），只有
    breakout_signal=True 的樣本才會進入模型預測——先跌後漲破底翻
    才允許下單，其餘直接剔除，不給任何訊號。

    回傳格式：[{"stock_id": ..., "proba": ..., "price": ..., "breakout": ...}, ...]
    """
    if model is None:
        model = load_model()

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    # ── 黃金窗口限制：只在 9:14~9:30 交易 ──────────────────────────
    _dt = pd.Timestamp(minute_str)
    _sh, _sm = BREAKOUT_TRADE_START
    _eh, _em = BREAKOUT_TRADE_END
    if not ((_dt.hour == _sh and _dt.minute >= _sm) and (_dt.hour == _eh and _dt.minute <= _em)):
        return []

    # make_features 以 day_date 做 merge；今日 day 資料尚不存在，補一行今日摘要
    day = load_day()
    day = day.copy()
    day["date"] = pd.to_datetime(day["date"])
    today_ts = pd.Timestamp(date_str)
    if not (day["date"] == today_ts).any():
        rows = []
        for sid, g in m1_live.groupby("stock_id"):
            g_s = g.sort_values("date")
            rows.append(
                {
                    "stock_id": sid,
                    "date": today_ts,
                    "open": float(g_s.iloc[0]["open"]),
                    "high": float(g["high"].max()),
                    "low": float(g["low"].min()),
                    "close": float(g_s.iloc[-1]["close"]),
                    "volume": int(g["volume"].sum()),
                }
            )
        if rows:
            day = pd.concat([day, pd.DataFrame(rows)], ignore_index=True)
            day["date"] = pd.to_datetime(day["date"])
            day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)

    df = make_features(m1_live, day=day, compute_labels=False)
    current = df[df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []

    valid = current.dropna(subset=FEATURES)
    if valid.empty:
        return []

    # ── 強過濾：硬規則（預設強制開啟）─────────────────────────────
    if use_breakout_filter:
        valid = valid[valid["breakout_signal"]]
        if valid.empty:
            return []

    proba = model.predict_proba(valid[FEATURES])[:, 1]
    signals = [
        {
            "stock_id": row["stock_id"],
            "proba": float(p),
            "price": float(row["close"]),
            "breakout": bool(row["breakout_signal"]),
        }
        for (_, row), p in zip(valid.iterrows(), proba)
        if p >= threshold
    ]
    return sorted(signals, key=lambda x: -x["proba"])


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 主程式
# ═══════════════════════════════════════════════════════════════════════════════


def main(
    mode: str = "",
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    當沖策略 RandomForest 主程式。

    支援兩種用法：
      1. 直接傳參數：main(mode="train")
      2. CLI 執行：python strategy/date_trade_rfc_model.py train

    Parameters
    ----------
    mode : str
        執行模式。留空則從 CLI 讀取。
        train / validate / importance / breakout
    test_days : int
        測試集天數（預設 10）
    start_date : str
        資料起日，格式 "YYYY-MM-DD"。留空不限制。
    end_date : str
        資料迄日，格式 "YYYY-MM-DD"。留空不限制。
    """
    if not mode:
        parser = argparse.ArgumentParser(
            description="當沖策略 — RandomForest（close + volume + 過去5天日K）",
        )
        parser.add_argument(
            "mode",
            choices=[
                "train",
                "train_xgb",
                "train_lgbm",
                "compare",
                "validate",
                "importance",
                "breakout",
            ],
            help="執行模式",
        )
        parser.add_argument("--test_days", type=int, default=10, help="測試集天數")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        parser.add_argument(
            "--threshold", type=float, default=0.0, help="強過濾破底翻信心度門檻（breakout mode 用，預設 0.0）"
        )
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        start_date = args.start_date
        end_date = args.end_date
        threshold = args.threshold

    if mode == "train":
        train(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "train_xgb":
        train_xgb(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "train_lgbm":
        train_lgbm(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "compare":
        compare_breakout(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "validate":
        confidence_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        coverage_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        breakout_filter_report(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "importance":
        feature_importance()

    elif mode == "breakout":
        breakout_minute_report(
            test_days=test_days,
            start_date=start_date,
            end_date=end_date,
            threshold=float(os.environ.get("BREAKOUT_THR", "0.0")),
        )

    else:
        print(f"未知模式: {mode}，可用: train / validate / importance")


if __name__ == "__main__":
    """
    直接執行本檔的進入點。

    在這裡直接改 mode 變數即可切換模式，不用每次打 CLI 參數。
    也可從 terminal 用 argparse 呼叫，例如：
        python strategy/date_trade_rfc_model.py train
        python strategy/date_trade_rfc_model.py validate
        python strategy/date_trade_rfc_model.py importance
        python strategy/date_trade_rfc_model.py breakout

    可用 mode:
        "train"       訓練 RandomForest 模型（存至 models/m1_rfc.pkl）
        "validate"    信心度分析 + 召回率分析 + 強過濾評估（breakout_filter_report）
        "importance"  顯示特徵重要性
        "breakout"    強過濾破底翻：9:14~9:30 黃金窗口逐分鐘推論數 / 平均信心度 / 勝率
        "train_xgb"   訓練 XGBoost 模型（存至 models/m1_xgb.pkl）
        "train_lgbm"  訓練 LightGBM 模型（存至 models/m1_lgbm_breakout.pkl）
        "compare"     三模型（RFC/XGB/LGBM）破底翻黃金窗口門檻掃描對照
        ""            走 CLI argparse（terminal 下帶參數）

    可選參數（CLI 或下方變數）：
        test_days     測試集天數（預設 10，取最後 N 天）
        start_date    資料起日 YYYY-MM-DD（留空 = 不限制）
        end_date      資料迄日 YYYY-MM-DD（留空 = 不限制）
        threshold     強過濾破底翻信心度門檻（breakout mode 用，預設 0.0）

    備註：
        - breakout_signal 為「前一根 M5 跌 + 當前 M5 漲」的破底翻硬過濾特徵
        - 交易時段限制為黃金窗口 9:14~9:30（BREAKOUT_TRADE_START/END）
        - predict_live(use_breakout_filter=True) 為即時推論強過濾入口（預設開啟），
          且只在 9:14~9:30 內產生訊號，其餘時間回傳 []
        - 直接執行本檔不會觸發即時下單，僅做訓練 / 驗證 / 報表
    """
    # ══════════════════════════════════════════════════════════════════════
    #  在這裡直接改 mode，不用每次打 CLI
    # ══════════════════════════════════════════════════════════════════════
    mode = "breakout"
    test_days = 10
    start_date = ""
    end_date = ""

    main(mode=mode, test_days=test_days, start_date=start_date, end_date=end_date)
