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

import numba
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 確保能從根目錄導入
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.query import load_day, load_m3, load_m5
from data.resample import compute_m3, compute_m5
from finmind.tick_universe import load_tick_universe
from strategy.rally.config import HOLD_BARS, SL_PCT, TP_PCT

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env", override=True)

_M1_DIR = _ROOT / "db/m1"
_M3_DIR = _ROOT / "db/m3"
_M5_DIR = _ROOT / "db/m5"
_DAY_DIR = _ROOT / "db/d1"  # 2026-08-03 從 db/fugle_day 改名而來，這裡只拿來檢查新鮮度用
# （實際讀取日K走 data.query.load_day()，見 _compute_month_features() 的說明）
_ADJUST_FACTOR_DIR = _ROOT / "db/tick_adjust_factor"  # 還原拆股/合股用的係數表
# （data/build_tick_adjust_factor.py），_month_is_fresh() 也要比對這個目錄的
# mtime——係數之後被回頭修正的話，現有 cache 分區要能被判定過期重算，不然會
# 一直沿用還原前算出來的舊特徵值（比照 strategy/orb/features.py 的做法）
_CACHE_DIR = _ROOT / "cache/m1_rally_features"

# 日K特徵最長回看窗口（pos_20d 用 rolling(20)），算某個月份的特徵時 db/d1
# 要多抓這麼多天當緩衝，不然那個月前面幾天會因為算不出 day_ret_10/pos_20d/
# day_atr 而被 dropna 丟掉，等於變相把可用起日往後推。
_DAY_LOOKBACK_CALENDAR_DAYS = 45

_META_COLS = ["stock_id", "date", "day_date", "hour", "minute", "target", "breakout_signal"]


# ── Cache 管理（按月分區，跟 db/m1/db/m3/db/m5/db/fugle_day 同樣的按月分檔慣例）──
#
# 2026-07-21 從單一大檔案改成按月分區：舊版一次讀全部歷史算完存一份 cache，
# 資料長到 5800萬筆時實測記憶體峰值到 86GB（機器只有 25.8GB RAM）。現在一次
# 只處理一個月（現在規模大概幾百萬筆），且 db/m1 幾乎每天只會動到「當月」
# 那個檔案，所以平常只有當月分區需要重算，其他月份的分區維持不動、不用重讀
# 也不用重算，兩個問題（記憶體峰值、每天都要整個重算）一起解決。


def _month_str(year: int, month: int) -> str:
    return f"{year}_{month:02d}"


def _month_file(dir_path: Path, year: int, month: int) -> Path:
    return dir_path / f"{_month_str(year, month)}.parquet"


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """回傳 start~end（含頭尾月份）涵蓋的所有 (year, month)，用月初對齊比較。"""
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)
    end_marker = pd.Timestamp(year=end.year, month=end.month, day=1)
    months = []
    while cur <= end_marker:
        months.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return months


def _day_lookback_months(year: int, month: int) -> list[tuple[int, int]]:
    """算某個月份的特徵時，day K 需要往前抓的月份範圍（含自己），
    新鮮度檢查跟實際讀取資料共用同一份邏輯，確保兩邊範圍一致。"""
    month_start = pd.Timestamp(year=year, month=month, day=1)
    day_cutoff = month_start - pd.Timedelta(days=_DAY_LOOKBACK_CALENDAR_DAYS)
    return _months_between(day_cutoff, month_start)


def _m1_available_months() -> list[tuple[int, int]]:
    """db/m1/ 現有的所有月份（依檔名判斷），由早到晚排序。"""
    months = []
    for f in _M1_DIR.iterdir():
        if f.suffix == ".parquet":
            year_str, month_str = f.stem.split("_")
            months.append((int(year_str), int(month_str)))
    return sorted(months)


def _month_is_fresh(year: int, month: int) -> bool:
    """檢查某個月份的 cache 分區是否比對應的 db/m1、db/d1、db/tick_adjust_factor
    （含回看範圍內的月份）都新。"""
    partition = _month_file(_CACHE_DIR, year, month)
    if not partition.exists():
        return False
    partition_mtime = partition.stat().st_mtime
    if partition_mtime < _file_mtime(_month_file(_M1_DIR, year, month)):
        return False
    for dy, dm in _day_lookback_months(year, month):
        if partition_mtime < _file_mtime(_month_file(_DAY_DIR, dy, dm)):
            return False
        if partition_mtime < _file_mtime(_month_file(_ADJUST_FACTOR_DIR, dy, dm)):
            return False
    return True


def _read_month_parquet(dir_path: Path, year: int, month: int) -> pd.DataFrame:
    """直接讀單一月份檔案（不像 data.query.load_m1() 等函式會掃整個資料夾），
    避免逐月處理時每個月都要重讀一次全部歷史。"""
    path = _month_file(dir_path, year, month)
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    if {"stock_id", "date"} <= set(df.columns):
        df = df.drop_duplicates(subset=["stock_id", "date"], keep="last")
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _compute_month_features(year: int, month: int, universe: set) -> pd.DataFrame:
    """對單一月份重算特徵，回傳只含 FEATURES + meta_cols 的 DataFrame。

    universe: 固定 400 支 tick_universe（見 finmind/tick_universe.py），
    load_features() 只算一次往下傳，不要在這裡每個月各自呼叫一次
    load_tick_universe()。2026-08-01 前 db/m1 的舊月份分檔還是全市場~2700支
    （data/m1_data_loader.py 寫入端限制不會回頭清理既有檔案），這裡要在算
    任何特徵之前先篩，比照 strategy/mkt/train.py 的教訓（篩太晚在算完跨股票
    的特徵之後才篩，2026-08-05撞過55GB記憶體爆炸）跟 strategy/orb/features.py
    的做法。0050 已經在 tick_universe 裡強制包含（見 finmind/tick_universe.py
    的 _FORCE_INCLUDE），不用另外處理。"""
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.DateOffset(months=1)

    m1 = _read_month_parquet(_M1_DIR, year, month)
    m1 = m1[m1["stock_id"].isin(universe)]

    m3_frames = [_read_month_parquet(_M3_DIR, year, month)]
    m5_frames = [_read_month_parquet(_M5_DIR, year, month)]
    m3 = m3_frames[0]
    m5 = m5_frames[0]

    # day K 改用 data.query.load_day()（還原拆股/合股版），不要用
    # _read_month_parquet(_DAY_DIR, ...) 讀原始 db/d1——predict_live()
    # （strategy/rally/predict.py）算日K背景特徵時用的就是這支還原版，兩邊
    # 不一致會在遇到拆股/合股事件時產生 training-serving skew（比照
    # strategy/orb/features.py 的做法）。日K一個月才幾百~幾千列，
    # load_day(start_date=...) 就算讀到比這裡的回看窗口更寬的範圍（底層依
    # 月份篩檔案，不是整個資料夾全讀），成本也可忽略。
    day_cutoff = (month_start - pd.Timedelta(days=_DAY_LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    day = load_day(start_date=day_cutoff)
    if not day.empty:
        day = day[day["date"] < month_end].reset_index(drop=True)

    df = make_features(m1, m3=m3, m5=m5, day=day, compute_labels=True)
    cache_cols = [c for c in df.columns if c in set(FEATURES) | set(_META_COLS)]
    return df[cache_cols]


def load_features(use_cache: bool = True, start_date: str = "") -> pd.DataFrame:
    """
    載入特徵（按月分區 cache，自動使用 / 增量重建）。

    start_date：決定要涵蓋哪些月份，從 start_date 所在月份到 db/m1/ 現有最新
    月份（含頭尾）；留空表示從 db/m1/ 最早的月份開始（等同全歷史，但仍然是
    按月分批算，不會一次把全部歷史塞進記憶體）。

    use_cache=True（預設）：逐月檢查該月分區是否比對應的 db/m1、day K 回看
    範圍內的月份檔都新，新鮮就沿用、不新鮮才重算「那一個月」——不會因為某個
    月有更新就牽動其他月份重算。db/m1 幾乎每天只會動到「當月」那個檔案，
    所以平常只有當月分區需要重算，其他月份直接沿用。
    use_cache=False：不管每個月分區現在是什麼狀態，全部月份都重算。
    """
    available = _m1_available_months()
    if not available:
        raise RuntimeError("db/m1/ 沒有任何資料")
    latest_year, latest_month = available[-1]
    if start_date:
        start_ts = pd.Timestamp(start_date)
        start_year, start_month = start_ts.year, start_ts.month
    else:
        start_year, start_month = available[0]

    target_months = _months_between(
        pd.Timestamp(year=start_year, month=start_month, day=1),
        pd.Timestamp(year=latest_year, month=latest_month, day=1),
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 欄位齊不齊檢查：只要有任何一個目標月份的分區缺欄位（例如 FEATURES 加了
    # 新特徵，舊分區沒算過），就強制全部目標月份重算一次，之後才會又回到
    # 逐月增量的狀態。
    force_all = False
    if use_cache:
        for year, month in target_months:
            partition = _month_file(_CACHE_DIR, year, month)
            if partition.exists():
                existing_cols = set(pd.read_parquet(partition, columns=None).columns)
                if not set(FEATURES) <= existing_cols:
                    print(f"  {_month_str(year, month)} 分區缺欄位，全部月份重算一次...")
                    force_all = True
                break

    universe = set(load_tick_universe())  # 固定400支，見 _compute_month_features() 的說明
    for year, month in target_months:
        if use_cache and not force_all and _month_is_fresh(year, month):
            continue
        print(f"  重新計算特徵：{_month_str(year, month)}...")
        month_df = _compute_month_features(year, month, universe)
        month_df.to_parquet(_month_file(_CACHE_DIR, year, month))
        print(f"    {_month_str(year, month)} 分區已存（{len(month_df):,} 筆）")

    partition_paths = [
        _month_file(_CACHE_DIR, year, month)
        for year, month in target_months
        if _month_file(_CACHE_DIR, year, month).exists()
    ]
    print(f"  讀取 {len(partition_paths)} 個月份分區...")
    df = pd.concat([pd.read_parquet(p) for p in partition_paths], ignore_index=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Triple Barrier Label
# ═══════════════════════════════════════════════════════════════════════════════


@numba.njit(cache=True)
def _barrier_label_numba(
    closes: np.ndarray,
    group_id: np.ndarray,
    hold_bars: int,
    tp_pct: float,
    sl_pct: float,
) -> np.ndarray:
    """單一 pass 掃過整個（已依 stock_id/day_date 排序）陣列，逐 bar 往前看最多
    hold_bars 根，判斷先碰到 tp 或 sl；group_id 不同代表跨股票或跨日，視同沒有
    future bar 可看。取代原本 groupby(...).apply() 逐組呼叫 Python function 的寫法
    ——40~50萬個 stock×day 分組，.apply() 本身的 per-group overhead 加上內層
    純 Python 迴圈，是 make_features() 裡唯一沒有走 pandas 向量化路徑的地方，
    也是特徵計算最慢的部分。JIT 編譯成單一迴圈後可以省掉這兩層 overhead。

    掃描時只要碰到第一個滿足 tp 或 sl 的 bar 就能立刻判定先後（因為是依時間
    順序往前掃，先出現的就是先碰到的那個），不需要像原本那樣分別對整段
    future 陣列各做一次 argmax。
    """
    n = len(closes)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        gid = group_id[i]
        entry = closes[i]
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 - sl_pct)
        max_j = i + hold_bars
        if max_j >= n:
            max_j = n - 1
        hit = -1  # 1=tp 先到, 0=sl 先到, -1=都沒碰到
        last_j = i
        for j in range(i + 1, max_j + 1):
            if group_id[j] != gid:
                break
            last_j = j
            c = closes[j]
            if c >= tp_price:
                hit = 1
                break
            if c <= sl_price:
                hit = 0
                break
        if hit == 1:
            labels[i] = 1.0
        elif hit == 0:
            labels[i] = 0.0
        elif last_j - i == hold_bars:
            labels[i] = 1.0 if closes[last_j] > entry else 0.0
        # else：當日剩餘 bar 數不足 hold_bars 且都沒碰到，維持 NaN
    return labels


def _make_barrier_labels(m1: pd.DataFrame) -> pd.Series:
    """m1 需已依 ["stock_id", "date"] 排序（make_features() 一開始就排好）。"""
    keys = m1[["stock_id", "day_date"]]
    group_id = (keys != keys.shift()).any(axis=1).cumsum().to_numpy(dtype=np.int64)
    closes = m1["close"].to_numpy(dtype=np.float64)
    labels = _barrier_label_numba(closes, group_id, HOLD_BARS, TP_PCT, SL_PCT)
    return pd.Series(labels, index=m1.index)


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
    # db/m3、db/m5 是批次預算（data/build_m3_m5_rolling.py 從 db/m1/ 算好存檔），
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

    # FEATURES 欄位統一降成 float32：原始 db/m1 的 ohlcv 雖然存的就是 float32，
    # 但上面一路算下來很多欄位是靠 pandas 的 .rolling()/.ewm()（m1_atr、
    # macd_hist、macd_divergence、reversal_N、day_atr 等）算出來的，這類操作
    # 內部一律用 float64 累加，輸出不管輸入是什麼 dtype 都會變 float64。這些
    # 特徵本質上是比例值/報酬率，float32 的精度（~7位有效數字）綽綽有餘，
    # 樹模型也不需要 float64，這裡統一轉一次能省接近一半記憶體。
    _feature_cols_present = [c for c in FEATURES if c in m1.columns]
    m1[_feature_cols_present] = m1[_feature_cols_present].astype("float32")

    return m1
