"""
特徵工程與資料標籤 — RFC / XGB / LGBM 三模型共用

只用 1 分鐘線收盤價與成交量等衍生特徵，搭配 3分K/5分K/日K/大盤(0050) 背景特徵，
用 triple barrier 標籤未來 30 根分K內：
  +3% 停利先碰到 → target = 1（漲）
  -3% 停損先碰到 → target = 0（跌）

載入 db/m1/ 歷史分K，另載入 db/fugle_day/ 日K 提供過去 10 天日K 背景特徵
（day_ret_1~10、day_vol_1~10）。特徵集合含從 strategy/base/date_trade_model.py
整合過來的部分（2026-07-09，base 已刪除）。
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
load_dotenv(_ROOT / ".env", override=True)

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
    # 只存特徵欄位 + time/target，去掉原始 m1 的 raw ohlcv 避免體積過大。
    # breakout_signal 不在 FEATURES（2026-07-09 因為重要性太低被拿掉了），
    # 但 predict_live() 的 use_breakout_filter 硬過濾、experiments/ 底下的
    # breakout_filter_eval.py/breakout_specialist.py 都還要用這個欄位本身
    # （不是拿去當模型輸入），所以要跟 meta_cols 一樣保留在 cache 裡。
    meta_cols = ["stock_id", "date", "day_date", "hour", "minute", "target", "breakout_signal"]
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
    # 5分鐘K OHLCV（當前 + 前1根，每根間隔5分鐘）
    "m5_open",
    "m5_high",
    "m5_low",
    "m5_ret",  # 5分鐘K報酬率（K棒間變化%）
    "m5_open_lag1",
    "m5_high_lag1",
    "m5_low_lag1",
    "m5_close_lag1",
    # 報酬率與量比
    "ret_1",
    "tf3_ret",
    "tf5_ret",
    # 時間（距離開盤幾分鐘），讓模型能分辨開盤動能期 vs 中午盤整期
    "minutes_since_open",
    # ── 以下從 strategy/base/date_trade_model.py 整合（2026-07-09）─────
    "ret_10",  # 前10分鐘報酬率
    "ret_15",  # 前15分鐘報酬率
    "close_pos",  # close 在當根K棒(high-low)的相對位置 [0,1]
    "tf3_ret_2",  # 3分鐘K 前2根報酬率（lag6）
    "tf3_ret_3",  # 3分鐘K 前3根報酬率（lag9）
    "tf3_close_pos",
    "tf3_range_pct",
    "tf3_reversal",
    "tf5_close_pos",
    "tf5_range_pct",
    "tf5_reversal",
    "vwap_dev",  # close vs 當日累積VWAP 偏離
    "high_pos_today",  # close 在當日目前最高/最低的位置 [0,1]
    "reversal_10",  # 距近10根最低點反彈幅度
    "macd_hist",  # MACD(12,26,9) histogram（日內重算，/day_open 比例值）
    "macd_divergence",  # MACD 背離分數：過去10根動能變化 - 過去10根價格變化
    # 日K 特徵（過去 10 天，無未來洩漏）
    "day_ret_1",
    "day_ret_2",
    "day_ret_3",
    "day_ret_4",
    "day_ret_5",
    "day_ret_6",
    "day_ret_7",
    "day_ret_8",
    "day_ret_9",
    "day_ret_10",
    "day_vol_1",
    "day_vol_2",
    "day_vol_3",
    "day_vol_4",
    "day_vol_5",
    "day_vol_6",
    "day_vol_7",
    "day_vol_8",
    "day_vol_9",
    "day_vol_10",
    "day_atr",  # 日K ATR(14) 相對波動（前一日，/ 當日開盤）
    "gap",  # 今日跳空幅度（今日開盤 vs 昨日收盤）
    "prev_vol_ratio",  # 前1日量 / 20日均量
    "pos_20d",  # 前1日收盤在近20日高低點的相對位置 [0,1]
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
]
# 2026-07-09 整理：拿掉 feature_importance 排名吊車尾（3模型正規化後平均 <0.3%）
# 的 17 個特徵——m3_open/m3_volume(+lag1/2)/m5_close/m5_volume(+lag1)/vol_ratio/
# vol_ratio_15ma/tf3_vol_ratio/tf5_vol_ratio/reversal_3/reversal_5/idx_breakout，
# 全部量能比類的特徵目前看起來是雜訊。
# breakout_signal 也拿掉了（重要性 0.12%，幾乎沒用），但這欄位還是保留計算——
# predict_live() 的 use_breakout_filter 硬過濾規則、experiments/ 底下兩個
# 實驗都還在用這個欄位本身（不是拿它當模型輸入），只是不再餵給模型當特徵。


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
    m1["m1_atr"] = _degroup(g_day["_tr"].rolling(14, min_periods=14).mean(), m1.index) / day_open

    # ── 載入預先聚合的 db/m3、db/m5（訓練走 cache，效能考量）───────────
    # db/m3、db/m5 是批次預算（data/build_m3_m5.py 從 db/m1/ 算好存檔），
    # 只有訓練用的完整歷史 m1 才會跟它對得上。即時推論的 m1（當天 m1_live）
    # 不在這個 cache 裡，呼叫端必須自己用 compute_m3()/compute_m5() 現算後
    # 明確傳進來，不能依賴這裡的 fallback，否則 merge 完全對不上、m3_*/m5_*
    # 會整批變 NaN（連帶 breakout_signal 也壞掉）。
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
    m1["m3_volume"] = g_m3["m3_vol_raw"].pct_change(3, fill_method=None)
    g_m3 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    for col in ["m3_open", "m3_high", "m3_low", "m3_close", "m3_volume"]:
        m1[f"{col}_lag1"] = g_m3[col].shift(3)
        m1[f"{col}_lag2"] = g_m3[col].shift(6)
    m1["m3_ret"] = g_m3["m3_close"].pct_change(3, fill_method=None)

    # ── 5分鐘K volume pct_change + lag1 ─────────────────────────────
    g_m5 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["m5_volume"] = g_m5["m5_vol_raw"].pct_change(5, fill_method=None)
    g_m5 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    for col in ["m5_open", "m5_high", "m5_low", "m5_close", "m5_volume"]:
        m1[f"{col}_lag1"] = g_m5[col].shift(5)
    m1["m5_ret"] = g_m5["m5_close"].pct_change(5, fill_method=None)

    # ── 強過濾：破底翻訊號 ───────────────────────────────────────────
    # 第1根5分鐘K跌（lag1的ret < 0），第2根5分鐘K漲（當前ret > 0）
    g_m5 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["breakout_signal"] = (g_m5["m5_ret"].shift(5) < 0) & (m1["m5_ret"] > 0)

    # ── 報酬率與量比 ────────────────────────────────────────────────
    # 前1分鐘報酬率
    m1["ret_1"] = g_day["close"].pct_change(1, fill_method=None)

    # 量比（當前量 / 前1分鐘量）
    vol_shift1 = g_day["volume"].shift(1)
    m1["vol_ratio"] = m1["volume"] / vol_shift1.replace(0, np.nan)

    # 3分鐘K報酬率
    m1["tf3_ret"] = g_day["close"].pct_change(3, fill_method=None)

    # 3分鐘K量比
    m1["_vol_roll3"] = _degroup(g_day["volume"].rolling(3).sum(), m1.index)
    g_day2 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["tf3_vol_ratio"] = m1["_vol_roll3"] / g_day2["_vol_roll3"].shift(3).replace(0, np.nan)

    # 5分鐘K報酬率
    m1["tf5_ret"] = g_day["close"].pct_change(5, fill_method=None)

    # 5分鐘K量比
    m1["_vol_roll5"] = _degroup(g_day["volume"].rolling(5).sum(), m1.index)
    g_day2 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["tf5_vol_ratio"] = m1["_vol_roll5"] / g_day2["_vol_roll5"].shift(5).replace(0, np.nan)

    # ── 從 strategy/base/date_trade_model.py 整合過來的特徵（2026-07-09）──
    # base 即將刪除。以下只合併「跟現有欄位不重複」的部分：base 的
    # ret_1/ret_3/ret_5、tf3_ret_1/tf5_ret_1、hour/minute、price_vs_open、
    # prev_ret 分別跟這裡已有的 ret_1/tf3_ret/tf5_ret、minutes_since_open、
    # m1_close、day_ret_1 是同一份資訊（同公式或線性變換），對樹模型不會有
    # 新增資訊，不重複加。base 用預設關閉的過去K線序列特徵（k1/k3/k5_*_lag_N，
    # 約115個）也不合併，維持特徵數量可控。

    # 更長天期報酬率（ret_10 約9:16後、ret_15 約9:21後才有值）
    m1["ret_10"] = g_day["close"].pct_change(10, fill_method=None)
    m1["ret_15"] = g_day["close"].pct_change(15, fill_method=None)

    # close 在當根K棒 (high-low) 的相對位置 [0,1]
    _bar_range = (m1["high"] - m1["low"]).replace(0, np.nan)
    m1["close_pos"] = (m1["close"] - m1["low"]) / _bar_range

    # 量比（15分鐘均量版本，跟現有 vol_ratio「/前一分鐘量」是不同角度的量能訊號）
    _vol_ma15 = _degroup(g_day["volume"].rolling(15).mean(), m1.index)
    m1["vol_ratio_15ma"] = m1["volume"] / _vol_ma15.replace(0, np.nan)

    # tf3 第2、3根報酬率（lag6/lag9；lag3已有 tf3_ret，不重複）+ K棒形態。
    # close_pos/range_pct/reversal 是「比值」，用已正規化的 m3_open/high/low/
    # close（除以 day_open）算，day_open 在分子分母會自動消掉，結果等於用
    # 原始未正規化價格算，不用另外拿一份原始高低價。
    m1["tf3_ret_2"] = g_day["close"].pct_change(6, fill_method=None)
    m1["tf3_ret_3"] = g_day["close"].pct_change(9, fill_method=None)
    _tf3_range = (m1["m3_high"] - m1["m3_low"]).replace(0, np.nan)
    m1["tf3_close_pos"] = (m1["m3_close"] - m1["m3_low"]) / _tf3_range
    m1["tf3_range_pct"] = _tf3_range / m1["m3_open"].replace(0, np.nan)
    m1["tf3_reversal"] = (m1["m3_close"] - m1["m3_low"]) / m1["m3_low"].replace(0, np.nan)

    # tf5 K棒形態（tf5_ret_2=lag10 跟 ret_10 重複、tf5_ret_3=lag15 跟 ret_15
    # 重複，不重複加，只加 K棒形態）
    _tf5_range = (m1["m5_high"] - m1["m5_low"]).replace(0, np.nan)
    m1["tf5_close_pos"] = (m1["m5_close"] - m1["m5_low"]) / _tf5_range
    m1["tf5_range_pct"] = _tf5_range / m1["m5_open"].replace(0, np.nan)
    m1["tf5_reversal"] = (m1["m5_close"] - m1["m5_low"]) / m1["m5_low"].replace(0, np.nan)

    # ── 當日盤中特徵（cumsum/cummax/cummin，不含未來資料）─────────────
    # g_day 是合併 m3/m5 前建立的，綁定舊的 m1 物件看不到新欄位，這裡重建一份
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    # VWAP 偏離：_cum_vol 前面 m1_volume 已經算過，這裡重用
    m1["_pv"] = m1["close"] * m1["volume"]
    m1["_cum_pv"] = g_day["_pv"].transform("cumsum")
    _vwap = m1["_cum_pv"] / m1["_cum_vol"].replace(0, np.nan)
    m1["vwap_dev"] = (m1["close"] - _vwap) / _vwap.replace(0, np.nan)

    # close 在今日目前最高/最低的位置（rolling 只含今天已經發生的部分）
    _high_max = g_day["high"].transform("cummax")
    _low_min = g_day["low"].transform("cummin")
    _today_range = (_high_max - _low_min).replace(0, np.nan)
    m1["high_pos_today"] = (m1["close"] - _low_min) / _today_range

    # 破底翻反彈幅度：距近 N 根最低點的反彈幅度（=0 代表仍在低點）
    for _n in [3, 5, 10]:
        _roll_min = _degroup(g_day["close"].rolling(_n, min_periods=_n).min(), m1.index)
        m1[f"reversal_{_n}"] = (m1["close"] - _roll_min) / _roll_min.replace(0, np.nan)

    # ── MACD(12,26,9) 背離 ──────────────────────────────────────────────
    # 逐日重算（不跨日，理由同 m1_atr：日內動能指標，隔天開盤重新累積，
    # 不該延續昨天收盤前的動能狀態）。用已正規化的 m1_close（close/當日開盤）
    # 算 EMA，MACD line/histogram 天生就是「相對 day_open」的比例值，跨股票可比。
    _ema12 = _degroup(g_day["m1_close"].ewm(span=12, adjust=False).mean(), m1.index)
    _ema26 = _degroup(g_day["m1_close"].ewm(span=26, adjust=False).mean(), m1.index)
    m1["_macd_line"] = _ema12 - _ema26
    g_macd = m1.groupby(["stock_id", "day_date"], group_keys=False)
    _macd_signal = _degroup(g_macd["_macd_line"].ewm(span=9, adjust=False).mean(), m1.index)
    m1["macd_hist"] = m1["_macd_line"] - _macd_signal

    # 背離分數 = 過去10根 MACD 動能的變化量 − 過去10根價格本身的變化量（同單位，
    # 都是「相對 day_open」的比例值，可以直接相減）。>0 代表動能轉強的幅度比
    # 價格漲幅還多（價格弱、動能強 = 偏多背離，常見於「先跌但跌勢趨緩」）；
    # <0 代表價格漲得比動能快（偏空背離，漲勢可能後繼無力）。
    # 用差值而不是「AND 兩個布林條件」（像 breakout_signal 那樣），是因為這是
    # 兩條連續曲線的斜率差——樹模型光靠 price_change_10、macd_change_10 兩個
    # 各自獨立的特徵，用軸對齊分割很難逼近這種斜線邊界，跟 breakout_signal 那種
    # 「兩個布林條件 AND」（樹本來就能用兩次分割輕鬆逼近，所以額外做的旗標
    # 幾乎沒有新資訊、事後驗證重要性只有0.12%）情況不同，這裡才值得額外算
    # 一個組合特徵。
    g_macd2 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    _macd_chg_10 = m1["macd_hist"] - g_macd2["macd_hist"].shift(10)
    _price_chg_10 = m1["m1_close"] - g_macd2["m1_close"].shift(10)
    m1["macd_divergence"] = _macd_chg_10 - _price_chg_10

    # ── 日K 特徵（過去 10 天，無未來洩漏；5天/10天/20天視需要用不同均量窗口）──
    if day is None:
        day = load_day()
    day = day.copy()
    day["date"] = pd.to_datetime(day["date"])
    dg = day.groupby("stock_id")
    # 短線均量（5日），避免 rolling(20) 吃掉過多歷史資料
    vol_ma5 = _degroup(dg["volume"].rolling(5).mean(), day.index).replace(0, np.nan)
    day["_ret1"] = dg["close"].pct_change(1, fill_method=None)
    day["_volr"] = day["volume"] / vol_ma5
    dg2 = day.groupby("stock_id", group_keys=False)
    day_ret_cols, day_vol_cols = [], []
    for lag in range(1, 11):
        cr, cv = f"day_ret_{lag}", f"day_vol_{lag}"
        # day_ret_1 = 前1日報酬率，day_ret_2 = 前2日，以此類推。
        # 注意是 shift(lag) 不是 shift(lag-1)：_ret1 在 day_date=D 那一列，
        # 存的是「D 自己收盤 vs D-1 收盤」的報酬率——D 自己全天的報酬率要收盤
        # 才知道，intraday 當下不该看得到。shift(lag-1)（lag=1 時等於不平移）
        # 會把這個「未來才知道」的值直接留在 D 這一列，merge 回 m1 後，D 當天
        # 每一分鐘的樣本都能看到「今天自己會漲會跌」，是未來資料洩漏。
        # shift(lag) 才會把 D-1（甚至更早）已經收盤、確定知道的報酬率帶到 D 這一列。
        day[cr] = dg2["_ret1"].shift(lag)
        # day_vol_1 = 前1日量 / 5日均量，以此類推（同樣道理，避免看到當天自己的量）
        day[cv] = dg2["_volr"].shift(lag)
        day_ret_cols.append(cr)
        day_vol_cols.append(cv)
    day = day.drop(columns=["_ret1", "_volr"])
    dg = day.groupby("stock_id")

    # 20日均量比、20日高低位置（跟 base/date_trade_model.py 整合，2026-07-09）。
    # 兩者都要 shift(1)：day["volume"]/day["close"] 是 D 自己全天收完才知道的
    # 值，D 當天 rolling(20) 的窗口如果含 D 自己，就是同一種未來洩漏，理由
    # 跟上面 day_ret_N 一樣，所以先算出「當日版」再整個 shift(1) 到下一列。
    vol_ma20 = _degroup(dg["volume"].rolling(20).mean(), day.index).replace(0, np.nan)
    _prev_vol_ratio_raw = day["volume"] / vol_ma20
    day["prev_vol_ratio"] = _prev_vol_ratio_raw.groupby(day["stock_id"], group_keys=False).shift(1)

    roll_min20 = _degroup(dg["close"].rolling(20).min(), day.index)
    roll_max20 = _degroup(dg["close"].rolling(20).max(), day.index)
    _pos_20d_raw = (day["close"] - roll_min20) / (roll_max20 - roll_min20).replace(0, np.nan)
    day["pos_20d"] = _pos_20d_raw.groupby(day["stock_id"], group_keys=False).shift(1)
    # 今日跳空：今日開盤（開盤當下就知道）vs 昨日收盤（shift(1)，已知），不用再 shift
    _prev_close_gap = dg["close"].shift(1)
    day["gap"] = (day["open"] - _prev_close_gap) / _prev_close_gap.replace(0, np.nan)
    dg = day.groupby("stock_id")

    # ── 日K ATR（ATR(14)，用前一日避免當日洩漏，以當日開盤正規化）──
    day["_prev_close"] = dg["close"].shift(1)
    day["_day_tr"] = np.maximum(
        np.maximum((day["high"] - day["low"]).abs(), (day["high"] - day["_prev_close"]).abs()),
        (day["low"] - day["_prev_close"]).abs(),
    )
    dg = day.groupby("stock_id")
    day["_atr14"] = _degroup(dg["_day_tr"].rolling(14, min_periods=14).mean(), day.index)
    day["day_atr"] = day["_atr14"].shift(1) / day["open"].replace(0, np.nan)

    day["day_date"] = day["date"].dt.date
    day_feat_cols = (
        ["stock_id", "day_date"] + day_ret_cols + day_vol_cols + ["day_atr", "gap", "prev_vol_ratio", "pos_20d"]
    )
    m1 = m1.merge(day[day_feat_cols], on=["stock_id", "day_date"], how="left")

    # ── 大盤（0050）日K 特徵（前 5 天，廣播至所有個股）────────────────
    idx_day = day[day["stock_id"] == "0050"].copy()
    if not idx_day.empty:
        idx_dg = idx_day.groupby("stock_id")
        idx_day["_idx_ret1"] = idx_dg["close"].pct_change(1, fill_method=None)
        idx_dg2 = idx_day.groupby("stock_id", group_keys=False)
        idx_day_ret_cols = []
        for lag in range(1, 6):
            cr = f"idx_day_ret_{lag}"
            # shift(lag) 不是 shift(lag-1)，理由同 day_ret_N 的修正
            idx_day[cr] = idx_dg2["_idx_ret1"].shift(lag)
            idx_day_ret_cols.append(cr)
        idx_day = idx_day.drop(columns=["_idx_ret1"])
        idx_dg = idx_day.groupby("stock_id")
        # 0050 日K ATR
        idx_day["_idx_prev_close"] = idx_dg["close"].shift(1)
        idx_day["_idx_day_tr"] = np.maximum(
            np.maximum((idx_day["high"] - idx_day["low"]).abs(), (idx_day["high"] - idx_day["_idx_prev_close"]).abs()),
            (idx_day["low"] - idx_day["_idx_prev_close"]).abs(),
        )
        idx_dg = idx_day.groupby("stock_id")
        idx_day["_idx_atr14"] = _degroup(idx_dg["_idx_day_tr"].rolling(14, min_periods=14).mean(), idx_day.index)
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
        idx_ret_1 = idx_g_day["close"].pct_change(1, fill_method=None)
        m1.loc[idx_m1.index, "idx_ret_1"] = idx_ret_1.values
        # 0050 收盤 / 當日開盤（大盤相對開盤漲跌幅）
        m1.loc[idx_m1.index, "idx_vs_open"] = (idx_m1["close"] / idx_day_open).values
        # 0050 1分K ATR(14) 相對波動
        idx_prev_close = idx_g_day["close"].shift(1).fillna(idx_m1["open"])
        idx_m1["_idx_tr"] = np.maximum(
            np.maximum((idx_m1["high"] - idx_m1["low"]).abs(), (idx_m1["high"] - idx_prev_close).abs()),
            (idx_m1["low"] - idx_prev_close).abs(),
        )
        idx_g_day2 = idx_m1.groupby("day_date", group_keys=False)
        m1.loc[idx_m1.index, "idx_atr"] = (
            _degroup(idx_g_day2["_idx_tr"].rolling(14, min_periods=14).mean(), idx_m1.index) / idx_day_open
        ).values
        # 0050 收盤 > 開盤（0/1）
        m1.loc[idx_m1.index, "idx_up"] = (idx_m1["close"] > idx_day_open).astype(int).values
        # 0050 破底翻（前1分跌 + 當分漲）
        m1.loc[idx_m1.index, "idx_breakout"] = ((idx_ret_1.shift(1) < 0) & (idx_ret_1 > 0)).astype(int).values
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
            "_vol_roll3",
            "_vol_roll5",
            "_pv",
            "_cum_pv",
            "_idx_prev_close",
            "_idx_day_tr",
            "_idx_atr14",
            "_idx_tr",
            "_macd_line",
        ],
        errors="ignore",
    )

    # 時間特徵
    # hour/minute 只用於過濾時段（validate.py 的分時報表、回測進出場時間），不作模型輸入。
    # minutes_since_open 是模型輸入：train.py/predict.py 已經不再把訓練/推論限制在固定時段
    # （早盤 9:01~10:00），改成用全天資料——但 FEATURES 裡沒有任何欄位讓模型知道「現在
    # 是開盤動能期還是中午盤整期」，全天訓練等於把不同時段的行為模式混在一起當雜訊。
    # 給模型這個欄位，讓樹模型自己學會依時段分岔判斷。
    m1["hour"] = m1["date"].dt.hour
    m1["minute"] = m1["date"].dt.minute
    m1["minutes_since_open"] = (m1["hour"] - 9) * 60 + m1["minute"]

    # 目標標籤
    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    return m1
