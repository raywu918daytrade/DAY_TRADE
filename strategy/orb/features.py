"""
特徵工程與資料標籤 — 開盤區間突破（ORB, Opening Range Breakout）+ 3分K/5分K 中期動能確認

2026-07-11 改版：不再是「每分鐘都是一筆樣本」，改成「每次真正的突破事件才是
一筆樣本」——9:00~OPENING_RANGE_MINUTES（預設9:10）建立開盤區間，
OPENING_RANGE_MINUTES~BREAKOUT_SEARCH_MINUTES（預設9:10~9:30）內，收盤價
從「區間內/以下」重新站上區間上緣的那一刻算一筆候選，同一天可能觸發好幾次
（突破、拉回、再突破），當天完全沒有突破就沒有這支股票的樣本（不當負樣本）。
改版理由分兩階段：
  1. 舊版每分鐘都是獨立樣本，同一天200多根K棒彼此高度相關（幾乎同樣的
     區間距離），把「有沒有突破」這個訊號稀釋掉了（純1分K突破訊號特徵
     重要性只有3%）。
  2. 一開始改成「一天只留第一次突破」又太嚴格——「一天只交易一次」是
     交易執行層該決定的規則，不該在特徵/樣本產生階段就先幫模型篩掉後面
     的突破事件；模型只該負責替每一次真正的突破事件評分，之後回測/實單
     要不要只挑一天裡最高信心度那筆，是另一層的決定。
背景特徵（3/5分K動能、量能/波動/趨勢、開盤量歷史、個股日K、大盤背景）仍然
逐分鐘算，只是最後只在候選那幾列被選出來。
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

from data.query import load_day, load_m1, load_m3, load_m5
from data.resample import compute_m3, compute_m5
from strategy.orb.config import (
    BREAKOUT_SEARCH_MINUTES,
    HOLD_BARS,
    MIN_VOL_MA20,
    OPENING_RANGE_MINUTES,
    SL_PCT,
    TP_PCT,
)

_ROOT = Path(__file__).parent.parent.parent
_M1_DIR = _ROOT / "db/m1"
_DAY_DIR = _ROOT / "db/fugle_day"
_CACHE_PATH = _ROOT / "cache/m1_orb_features.parquet"


# ── Cache 管理 ───────────────────────────────────────────────────────────────


def _relevant_m1_files(_dir: Path, since) -> list:
    """篩出檔名（YYYY_MM.parquet）月份 >= since 所在月份的檔案。

    since=None 時回傳全部檔案（跟以前行為一樣，全部月份都要比對 mtime）。
    db/m1/、db/fugle_day/ 都是按月分檔，只要月份早於 since，那個檔案不可能
    包含 since 之後的資料，即使那個檔案的 mtime 變動（例如背景
    finmind.backfill_history 在補很久以前的歷史），也跟「since 之後這段
    測試窗口的特徵新不新鮮」無關，不該讓它觸發整批重算。檔名不符
    YYYY_MM 格式的話保守起見還是列進來比對。
    """
    files = [f for f in _dir.iterdir() if f.suffix == ".parquet"]
    if since is None:
        return files
    cutoff_ym = (since.year, since.month)
    out = []
    for f in files:
        try:
            y, m = f.stem.split("_")
            if (int(y), int(m)) >= cutoff_ym:
                out.append(f)
        except ValueError:
            out.append(f)
    return out


def _m1_mtime(since=None) -> float:
    """db/m1/ 中（since 之後月份的）最新檔案修改時間戳。"""
    mtimes = [f.stat().st_mtime for f in _relevant_m1_files(_M1_DIR, since)]
    return max(mtimes) if mtimes else 0


def _day_mtime(since=None) -> float:
    """db/fugle_day/ 中（since 之後月份的）最新檔案修改時間戳。"""
    if not _DAY_DIR.exists():
        return 0
    mtimes = [f.stat().st_mtime for f in _relevant_m1_files(_DAY_DIR, since)]
    return max(mtimes) if mtimes else 0


def _cache_is_fresh(since=None) -> bool:
    if not _CACHE_PATH.exists():
        return False
    cache_mtime = _CACHE_PATH.stat().st_mtime
    return cache_mtime >= _m1_mtime(since) and cache_mtime >= _day_mtime(since)


def load_features(force_rebuild: bool = False, since=None, skip_staleness_check: bool = False) -> pd.DataFrame:
    """載入特徵（自動使用 / 重建 cache）。

    since: 選填，只比對「since 所在月份之後」的 db/m1、db/fugle_day 檔案
    mtime——呼叫端知道實際只會用到最近 N 天資料時（例如 predict() 的
    test_days 窗口）可以傳進來，避免背景程式（例如 finmind.backfill_history）
    補很久以前的歷史資料時，動到舊月份檔案的 mtime 就讓整批特徵被判定
    「過期」而重算一次（很慢，且跟真正要用的窗口無關）。留空則比對全部
    月份檔案（跟以前行為一樣，最保守）。

    skip_staleness_check: 預設 False。True 時完全不比對 mtime，只要 cache
    檔案存在、欄位齊全就直接用，即使 db/m1/db/fugle_day 有更新也不重算。
    train.py 用的是全歷史當訓練集（不是只有最近幾天），資料被背景程式
    補進/修正到任何舊月份，理論上都該讓模型重新訓練，所以不能像
    predict() 一樣用 since 篩掉「無關月份」；這個參數是給你明確知道
    「我現在要犧牲正確性換速度」時的手動開關（例如背景 backfill 還在跑、
    你只是想先跑一版模型測試程式碼邏輯），不是預設行為，訓練出的模型
    可能沒吃到最新/修正過的資料，正式要用的模型還是要關掉這個參數重跑一次。
    """
    if not force_rebuild and _CACHE_PATH.exists() and (skip_staleness_check or _cache_is_fresh(since)):
        cached = pd.read_parquet(_CACHE_PATH)
        missing = [c for c in FEATURES + ["hour", "vol_ma20"] if c not in cached.columns]
        if not missing:
            print("  讀取 cache 特徵...")
            return cached
        print(f"  cache 缺少欄位 {missing}，重新計算...")

    print("  重新計算特徵（db/m1 有更新）...")
    m1 = load_m1()
    df = make_features(m1, compute_labels=True)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = ["stock_id", "date", "hour", "target", "vol_ma20"]
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
    # ── 開盤區間突破（單一候選：見檔頭說明）──────────────────────────────
    "or_range_pct",  # 開盤區間高度 / 區間下緣（區間本身寬不寬）
    "close_pos_in_range",  # 突破當下 close 在區間內相對位置（>1，數值越大代表衝越多）
    "dist_to_or_high",  # 突破當下 close 距區間上緣的相對距離（用區間上緣當基準）
    "dist_to_or_low",  # 突破當下 close 距區間下緣的相對距離
    "breakout_depth",  # 突破幅度，用「區間寬度」當基準（不是價位），(close-or_high)/(or_high-or_low)
    "vol_ratio_vs_or",  # 突破當根量 / 開盤區間期間均量（量能確認）
    "breakout_volume_ratio",  # 突破當根量 / 突破前5分鐘均量（跟自己近期比，不是跟開盤區間比）
    "breakout_close_location",  # 突破那根K棒收在高點附近還是留長上影，(close-low)/(high-low)
    "minutes_to_breakout",  # 距開盤區間結束多久才突破，越快突破可能動能越強
    "gap_pct",  # 今日開盤相對昨收的跳空幅度
    "or_range_atr_ratio",  # 開盤區間寬度 / 個股日ATR（區間相對這支股票平常波動算寬還窄）
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
    # 2026-07-11：vol_surge(需20根)/m1_atr(需14根)/adx(需28根，雙重平滑)
    # 已拿掉——改成單一候選（9:10~9:30搜尋窗口，只有20根K棒）之後，這幾個
    # 特徵天生的暖機期跟搜尋窗口互斥，dropna 幾乎把樣本濾光（adx缺值98.5%、
    # vol_surge缺值89.1%、m1_atr缺值63.6%），跟窗口設計衝突，不是bug。
    "vwap_dev",  # close 相對當日累積VWAP的偏離（比照 rally 的 vwap_dev，累積型不受暖機期影響）
    "atr_hour_surprise",  # 當根TR / 這支股票在這個小時過去10天的歷史平均TR（只需2根當根TR，不受影響）
    "atr7",  # 1分鐘K ATR(7) 相對波動（/ 當日開盤，只需7根K棒，不撞暖機期）
    "efficiency_ratio_7",  # Kaufman效率係數(7根K棒)，[0,1]，越接近1代表這段上漲越乾淨、雜訊越少
    # ── 個股過去5日「開盤1/3/5分鐘成交量佔當日總量比例」lag特徵，無未來洩漏 ──
    "open_vol_m1_1",
    "open_vol_m1_2",
    "open_vol_m1_3",
    "open_vol_m1_4",
    "open_vol_m1_5",
    "open_vol_m3_1",
    "open_vol_m3_2",
    "open_vol_m3_3",
    "open_vol_m3_4",
    "open_vol_m3_5",
    "open_vol_m5_1",
    "open_vol_m5_2",
    "open_vol_m5_3",
    "open_vol_m5_4",
    "open_vol_m5_5",
    # ── 個股日K 背景特徵（過去5天報酬率、ATR、5/10/20日均線乖離，無未來洩漏）──
    "day_ret_1",
    "day_ret_2",
    "day_ret_3",
    "day_ret_4",
    "day_ret_5",
    "day_atr",  # 個股日K ATR(14) 相對波動（前一日，/ 當日開盤）
    "ma_dev_5",  # 昨收 vs 昨5日均線乖離
    "ma_dev_10",  # 昨收 vs 昨10日均線乖離
    "ma_dev_20",  # 昨收 vs 昨20日均線乖離
    "close_vs_close_lag1",  # 目前價格 vs 過去1~5天各自收盤價的漲跌幅（逐分鐘更新）
    "close_vs_close_lag2",
    "close_vs_close_lag3",
    "close_vs_close_lag4",
    "close_vs_close_lag5",
    # ── 大盤（0050）背景特徵（比照 rally 做法，廣播至所有個股）──────────────
    "idx_day_ret_1",
    "idx_day_ret_2",
    "idx_day_ret_3",
    "idx_day_ret_4",
    "idx_day_ret_5",
    "idx_day_atr",  # 0050 日K ATR(14) 相對波動（前一日，/ 當日開盤）
    "idx_ret_1",  # 0050 前1分報酬率
    "idx_vs_open",  # 0050 收盤 / 0050 當日開盤（大盤相對開盤漲跌幅）
    # idx_atr（0050 1分K ATR(14)）2026-07-11 拿掉，理由同上面 m1_atr（需14根，跟搜尋窗口互斥）
    "idx_up",  # 0050 收盤 > 開盤（0/1，大盤是否站上開盤）
]
# 2026-07-10 試過加 group（類股分類，db/info/info.parquet，49類）當類別特徵，
# LGBM/XGB 的測試集 AUC 都變差（LGBM 0.6460→0.6246、XGB 0.5977→0.5815），
# 儘管 LGBM 特徵重要性排名第1——高基數類別特徵過擬合的典型徵兆（不少類股
# 訓練樣本數很少，樹學到訓練期間的巧合，測試期不成立），已經拿掉，不要
# 再加回來，除非之後有辦法解決過擬合（例如調 min_data_per_group/cat_smooth）。

# hour（時段）2026-07-11 從 FEATURES 拿掉：候選現在都集中在9:10~9:30
# 搜尋窗口內，hour 是常數（永遠是9），完全沒有資訊量（實測特徵重要性=0）。
# hour 欄位本身還留著（meta_cols 保留，hour_confidence_report() 分時段
# 報表用得到，雖然現在只會有9點那一組），只是不再餵給模型。
CATEGORICAL_FEATURES: list = []


def to_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """把 FEATURES 欄位轉成餵給模型用的格式（CATEGORICAL_FEATURES 轉 category dtype）。

    train.py 的 fit()/predict() 跟 validate.py 的 predict_proba() 都要用這支，
    確保訓練跟推論看到的 dtype 一致——LightGBM/XGBoost 都是靠欄位 dtype 是不是
    pandas 'category' 來判斷要不要當類別特徵處理，train/predict 兩邊沒對齊
    的話，同一個模型可能一邊當類別、一邊當數值，結果不一致或直接報錯。
    """
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def apply_liquidity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    只留 20日均量（見 vol_ma20）≥ MIN_VOL_MA20 的樣本。

    冷門股1分K容易被單筆大單推動，雜訊比例偏高，2026-07-10 測過1000張/5000張
    兩個門檻，AUC都在雜訊範圍內浮動，沒有明確證據支持這是主因（見 config.py
    的 MIN_VOL_MA20 說明），但保留這個篩選作為額外的流動性把關。train.py
    （訓練）、validate.py（驗證）、predict.py（批次/即時推論）都要用這支，
    確保訓練/驗證/推論看到同一個股票池。

    vol_ma20 是用「昨天為止」算的（見 make_features()），股票上市不到20天
    或當天缺日K資料時會是 NaN，一併篩掉（沒有足夠歷史判斷流動性，保守起見
    不留）。
    """
    return df[df["vol_ma20"] >= MIN_VOL_MA20].copy()


def _degroup(s: pd.Series, index: pd.Index) -> pd.Series:
    """把 groupby(...).rolling(...) 產生的多層 index 結果攤平回原始 df 對齊。"""
    n_key_levels = s.index.nlevels - 1
    if n_key_levels:
        s = s.reset_index(level=list(range(n_key_levels)), drop=True)
    return s.reindex(index)


def _hourly_tr_summary(m1: pd.DataFrame) -> pd.DataFrame:
    """
    每 (stock_id, day_date, hour) 的平均 True Range（未做lag/rolling）。
    m1 需已有 day_date/hour/_tr 欄位。給 make_features() 的 hourly_tr_history
    參數、build_history_tables() 用。
    """
    out = m1.groupby(["stock_id", "day_date", "hour"], as_index=False)["_tr"].mean()
    return out.rename(columns={"_tr": "_hourly_tr"})


def _open_vol_ratios(m1: pd.DataFrame) -> pd.DataFrame:
    """
    每 (stock_id, day_date) 的「開盤1/3/5分鐘成交量佔當日總量比例」原始值
    （未做lag位移）。m1 需已有 day_date 欄位、依 stock_id/date 排序。
    給 make_features() 的 open_vol_history 參數、build_history_tables() 用。
    """
    group_keys = [m1["stock_id"], m1["day_date"]]
    bar_pos = m1.groupby(["stock_id", "day_date"], group_keys=False).cumcount()
    cum_vol_today = m1.groupby(["stock_id", "day_date"], group_keys=False)["volume"].transform("cumsum")
    day_total_vol = m1.groupby(["stock_id", "day_date"], group_keys=False)["volume"].transform("sum")

    summary = pd.DataFrame(
        {
            "stock_id": m1["stock_id"],
            "day_date": m1["day_date"],
            "_open_m1": m1["volume"].where(bar_pos == 0).groupby(group_keys, group_keys=False).transform("max"),
            "_open_m3": cum_vol_today.where(bar_pos == 2).groupby(group_keys, group_keys=False).transform("max"),
            "_open_m5": cum_vol_today.where(bar_pos == 4).groupby(group_keys, group_keys=False).transform("max"),
            "_day_total_vol": day_total_vol,
        }
    )
    summary = summary.dropna(subset=["_open_m1", "_open_m3", "_open_m5"]).drop_duplicates(["stock_id", "day_date"])
    total = summary["_day_total_vol"].replace(0, np.nan)
    summary["_r_m1"] = summary["_open_m1"] / total
    summary["_r_m3"] = summary["_open_m3"] / total
    summary["_r_m5"] = summary["_open_m5"] / total
    return summary[["stock_id", "day_date", "_r_m1", "_r_m3", "_r_m5"]]


def build_history_tables(m1: pd.DataFrame) -> tuple:
    """
    給 predict_live() 用：從歷史 db/m1/（load_m1() 的完整結果）算出
    open_vol_history、hourly_tr_history 這兩張表，讓 atr_hour_surprise、
    open_vol_m1~m5_1~5 這幾個「需要過去N天歷史」的特徵在即時推論時也能算——
    m1_live 只有當天一天的資料，自己算不出「過去」。

    呼叫端應該快取這兩張表（例如開盤前算一次），不要每分鐘重算一次——這裡面
    的 groupby 是對全歷史 db/m1/ 跑的，效能考量跟 day 的快取方式一樣。
    """
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date
    m1["hour"] = m1["date"].dt.hour

    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    prev_close = g_day["close"].shift(1).fillna(m1["open"])
    m1["_tr"] = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - prev_close).abs()),
        (m1["low"] - prev_close).abs(),
    )

    return _open_vol_ratios(m1), _hourly_tr_summary(m1)


def make_features(
    m1: pd.DataFrame,
    m3: pd.DataFrame | None = None,
    m5: pd.DataFrame | None = None,
    day: pd.DataFrame | None = None,
    open_vol_history: pd.DataFrame | None = None,
    hourly_tr_history: pd.DataFrame | None = None,
    compute_labels: bool = True,
) -> pd.DataFrame:
    """
    開盤區間突破（每次突破事件一筆候選，見檔頭說明）+ 3分K/5分K 中期動能確認 +
    背景特徵 + triple barrier 標籤。

    m3/m5/day 留空時分別 fallback 去讀 db/m3、db/m5、db/fugle_day 批次資料
    （訓練走這條路徑）；predict_live() 對當天 m1_live 現算 m3_live/m5_live
    後要明確傳進來，不能依賴 fallback——批次資料不含「今天」，會讓 m3_*/m5_*
    整批變 NaN（見 compute_m3()/compute_m5() 的說明）。

    open_vol_history/hourly_tr_history：atr_hour_surprise、open_vol_m1~m5_1~5
    這兩組特徵需要「過去N天」的歷史（不是只有今天），訓練時 m1 本來就是完整
    歷史，這兩個參數留空即可（fallback 用 m1 自己算）；predict_live() 傳進來
    的 m1 是 m1_live（只有今天一天），這兩個參數必須用 build_history_tables()
    對歷史 db/m1/ 算好、明確傳進來，否則這兩組特徵會整批變 NaN。

    回傳欄位：FEATURES 全部 + target（若 compute_labels=True）。每個
    (stock_id, day_date) 可能有 0~多列——OPENING_RANGE_MINUTES~
    BREAKOUT_SEARCH_MINUTES 內，每次收盤價從區間內/以下重新站上開盤區間
    上緣都算一筆候選，當天完全沒有突破就沒有這支股票的樣本（不當負樣本，
    見檔頭說明）。一天只交易一次是執行層的規則，不在這裡處理。
    predict_live() 用當天累積到目前為止的 m1_live 呼叫這支函式，會自然只在
    「這一分鐘剛好觸發一次突破事件」時回傳這支股票的候選列。
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

    # ── 開盤區間突破（每次突破事件一筆候選，見檔頭說明）───────────────────
    # 9:00~OPENING_RANGE_MINUTES 建區間，OPENING_RANGE_MINUTES~
    # BREAKOUT_SEARCH_MINUTES 內每次收盤價重新站上區間上緣都算一次突破事件。
    # 這裡先對全部列算好特徵，實際「只留突破事件那幾列」的篩選放在函式
    # 最後（要等 target 也算完才篩，篩選邏輯本身不需要 target）。
    _od_keys = [m1["stock_id"], m1["day_date"]]
    _in_or = m1["minutes_since_open"] < OPENING_RANGE_MINUTES
    _or_high = m1["high"].where(_in_or).groupby(_od_keys, group_keys=False).transform("max")
    _or_low = m1["low"].where(_in_or).groupby(_od_keys, group_keys=False).transform("min")
    _or_vol_mean = m1["volume"].where(_in_or).groupby(_od_keys, group_keys=False).transform("mean")
    _or_range = (_or_high - _or_low).replace(0, np.nan)

    m1["or_range_pct"] = _or_range / _or_low.replace(0, np.nan)
    m1["close_pos_in_range"] = (m1["close"] - _or_low) / _or_range
    m1["dist_to_or_high"] = (m1["close"] - _or_high) / _or_high.replace(0, np.nan)
    m1["dist_to_or_low"] = (m1["close"] - _or_low) / _or_low.replace(0, np.nan)
    m1["vol_ratio_vs_or"] = m1["volume"] / _or_vol_mean.replace(0, np.nan)
    m1["minutes_to_breakout"] = m1["minutes_since_open"] - OPENING_RANGE_MINUTES

    # 突破幅度，用「區間寬度」當基準（不是價位本身）：同樣是突破0.5%，
    # 窄區間（例如寬度只有0.3%）代表衝出去1.6倍區間寬，比寬區間（例如寬度2%）
    # 衝出0.25倍區間寬更有意義，dist_to_or_high 只看價位百分比看不出這個差異。
    m1["breakout_depth"] = (m1["close"] - _or_high) / _or_range

    # 突破那根K棒收在高點附近還是留長上影：收在高點附近代表買盤持續進場，
    # 留長上影代表上面有賣壓、隨即被打回，是常見的假突破過濾條件。
    m1["breakout_close_location"] = (m1["close"] - m1["low"]) / (m1["high"] - m1["low"]).replace(0, np.nan)

    # 突破當根量 / 突破前5分鐘均量：跟自己「剛剛」的量比，不是跟開盤區間
    # （vol_ratio_vs_or）比——開盤區間期間往往量本來就大，這裡看的是「已經
    # 過了開盤區間、量能是不是又額外放大」。OPENING_RANGE_MINUTES=10 保證
    # 搜尋窗口一開始（第10分鐘）前面就有至少10根K棒可用，不會有暖機期問題。
    _vol_shift1 = g_day["volume"].shift(1)
    _vol_recent5 = _degroup(
        _vol_shift1.groupby([m1["stock_id"], m1["day_date"]], group_keys=False).rolling(5, min_periods=3).mean(),
        m1.index,
    )
    m1["breakout_volume_ratio"] = m1["volume"] / _vol_recent5.replace(0, np.nan)

    # 突破事件：收盤價從「區間內/以下」重新站上區間上緣的那一刻（不是「目前
    # 是不是站在上緣」）——用 prev_close <= or_high AND close > or_high 判斷，
    # 同一天可能觸發好幾次（突破、拉回、再突破），每次都是獨立的候選，不
    # 限制一天只留第一次（見 config.py 開頭的說明：一天只交易一次是執行層
    # 的決定，不該在這裡先幫模型篩掉後面的突破事件）。
    _search_window = (m1["minutes_since_open"] >= OPENING_RANGE_MINUTES) & (
        m1["minutes_since_open"] < BREAKOUT_SEARCH_MINUTES
    )
    _prev_close_for_breakout = g_day["close"].shift(1)
    m1["_is_breakout"] = _search_window & (m1["close"] > _or_high) & (_prev_close_for_breakout <= _or_high)

    # ── True Range / ATR(7) / VWAP偏離（逐日重算，不跨日）───────────────────
    # 2026-07-11：vol_surge(需20根)/m1_atr(需14根)/adx(需28根) 拿掉了，因為跟
    # 9:10~9:30 只有20根K棒的搜尋窗口互斥（見 FEATURES 清單說明）。ATR(7) 只
    # 需要7根K棒，OPENING_RANGE_MINUTES=10 保證搜尋窗口一開始（第10分鐘）
    # 前面就有至少10根K棒，不會撞到暖機期問題，改用7天期。
    _prev_close = g_day["close"].shift(1).fillna(m1["open"])
    m1["_tr"] = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - _prev_close).abs()),
        (m1["low"] - _prev_close).abs(),
    )
    g_tr = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["atr7"] = _degroup(g_tr["_tr"].rolling(7, min_periods=7).mean(), m1.index) / day_open

    # 效率係數（Kaufman Efficiency Ratio，7根K棒窗口，跟 atr7 用同樣的窗口
    # 長度，一樣不會撞暖機期）：分子是「淨移動距離」，分母是「這7根裡逐根
    # 走的路程總和」，[0,1]有界。ER接近1代表這段上漲一路噴上去、沒什麼雜訊
    # （像一根長紅棒吃掉前面好幾根K的效果）；接近0代表來回震盪、原地磨。
    # 比單純比較「這根range/前幾根平均range」更能抓到「乾不乾淨」這件事，
    # 跟只看淨位移的 m3_ret/m5_ret 是互補而非重複的角度（2026-07-13 討論）。
    _close_diff_abs = g_day["close"].diff().abs()
    _path_len_7 = _degroup(
        _close_diff_abs.groupby([m1["stock_id"], m1["day_date"]], group_keys=False).rolling(7, min_periods=7).sum(),
        m1.index,
    )
    _net_move_7 = (m1["close"] - g_day["close"].shift(7)).abs()
    m1["efficiency_ratio_7"] = _net_move_7 / _path_len_7.replace(0, np.nan)

    # VWAP 偏離：close 相對當日累積VWAP的偏離（跟 rally 的 vwap_dev 同做法）
    m1["_cum_vol"] = g_day["volume"].transform("cumsum")
    m1["_pv"] = m1["close"] * m1["volume"]
    m1["_cum_pv"] = g_day["_pv"].transform("cumsum")
    _vwap = m1["_cum_pv"] / m1["_cum_vol"].replace(0, np.nan)
    m1["vwap_dev"] = (m1["close"] - _vwap) / _vwap.replace(0, np.nan)

    # 波動意外程度：當根TR / 這支股票在「這個小時」過去幾天的歷史平均TR。
    # 不是跟自己全天平均比（那是 m1_atr），是跟同一個股票、同一個小時的
    # 歷史水準比——同樣是1分K的TR，9點天生就比12點大，直接比較會把
    # 「這個時段本來就該動」跟「這個時段不該動卻在動」混在一起，後者才是
    # 比較有意義的訊號（例如平常清淡的午盤突然放量，比開盤的波動更值得注意）。
    # 用過去10天同一小時的TR平均當基準，shift(1)避免看到今天自己這小時的值。
    # hourly_tr_history 有傳進來的話（predict_live()）要跟「今天自己」的
    # hourly summary 合併，不然 shift(1) 在只有今天一天的資料裡永遠是 NaN。
    hourly_summary = _hourly_tr_summary(m1)
    if hourly_tr_history is not None:
        hourly_summary = pd.concat([hourly_tr_history, hourly_summary], ignore_index=True)
        hourly_summary = hourly_summary.drop_duplicates(["stock_id", "day_date", "hour"], keep="last")
    hourly_summary = hourly_summary.sort_values(["stock_id", "hour", "day_date"])
    _hourly_group_keys = [hourly_summary["stock_id"], hourly_summary["hour"]]
    _hourly_tr_shift1 = hourly_summary.groupby(["stock_id", "hour"], group_keys=False)["_hourly_tr"].shift(1)
    hourly_summary["_hourly_tr_baseline"] = _degroup(
        _hourly_tr_shift1.groupby(_hourly_group_keys, group_keys=False).rolling(10, min_periods=5).mean(),
        hourly_summary.index,
    )
    m1 = m1.merge(
        hourly_summary[["stock_id", "day_date", "hour", "_hourly_tr_baseline"]],
        on=["stock_id", "day_date", "hour"],
        how="left",
    )
    m1["atr_hour_surprise"] = m1["_tr"] / m1["_hourly_tr_baseline"].replace(0, np.nan)

    m1 = m1.drop(columns=["_tr", "_cum_vol", "_pv", "_cum_pv", "_hourly_tr_baseline"])

    # ── 3分K（過去3根）/ 5分K（過去2根）—— 中期動能當多空判斷 ─────────────
    # db/m3、db/m5 是批次預算（data/build_m3_m5_rolling.py 從 db/m1/ 算好存檔，用的是
    # data/resample.py 的 compute_m3()/compute_m5()），只有訓練用的完整歷史
    # m1 才會跟它對得上，即時推論要呼叫端傳 m3/m5 進來。
    if m3 is None:
        m3 = load_m3()
    if m5 is None:
        m5 = load_m5()
    m3 = m3.copy()
    m5 = m5.copy()
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

    # ── 個股過去5日「開盤1/3/5分鐘成交量佔當日總量比例」lag特徵 ─────────────
    # 用比例（佔當日總量%），不是原始量，才能跨股票比較。逐日算出比例後，
    # 用跟 day_ret_1~5 一樣的 shift(lag) 做法把過去5天的值帶到今天這一列，
    # 不會看到「今天自己」的開盤量佔比。open_vol_history 有傳進來的話
    # （predict_live()）要跟「今天自己」的比例合併，不然 shift(1) 在只有
    # 今天一天的資料裡永遠是 NaN。
    open_vol_summary = _open_vol_ratios(m1)
    if open_vol_history is not None:
        open_vol_summary = pd.concat([open_vol_history, open_vol_summary], ignore_index=True)
        open_vol_summary = open_vol_summary.drop_duplicates(["stock_id", "day_date"], keep="last")
    open_vol_summary = open_vol_summary.sort_values(["stock_id", "day_date"])

    osg = open_vol_summary.groupby("stock_id", group_keys=False)
    open_vol_cols = []
    for metric, raw_col in [("open_vol_m1", "_r_m1"), ("open_vol_m3", "_r_m3"), ("open_vol_m5", "_r_m5")]:
        for lag in range(1, 6):
            col = f"{metric}_{lag}"
            open_vol_summary[col] = osg[raw_col].shift(lag)
            open_vol_cols.append(col)

    m1 = m1.merge(open_vol_summary[["stock_id", "day_date"] + open_vol_cols], on=["stock_id", "day_date"], how="left")

    # ── 個股日K 背景特徵（過去5天報酬率、ATR、5/10/20日均線乖離）─────────────
    if day is None:
        day = load_day()
    day = day.copy()
    day["date"] = pd.to_datetime(day["date"])
    day["day_date"] = day["date"].dt.date

    dg = day.groupby("stock_id")
    day["_ret1"] = dg["close"].pct_change(1)
    dg2 = day.groupby("stock_id", group_keys=False)
    day_ret_cols = []
    for lag in range(1, 6):
        cr = f"day_ret_{lag}"
        # shift(lag) 不是 shift(lag-1)：理由同 idx_day_ret_N（CLAUDE.md 提過的
        # day_ret_N 未來資料洩漏 bug，D這一列要存「D-1（或更早）已經收盤、確定
        # 知道」的報酬率，不能讓D自己全天才知道的報酬率露出來）。
        day[cr] = dg2["_ret1"].shift(lag)
        day_ret_cols.append(cr)
    day = day.drop(columns=["_ret1"])

    # ATR(14)：True Range 要前一日收盤才算得出來（day2起才有第一個TR），
    # rolling(14, min_periods=14) 再等14個有效TR值才有第一個ATR14（day15起），
    # 「昨天的ATR14」用在今天要再 shift(1)——這支股票至少要16天歷史日K，
    # day_atr 才不是 NaN（同做法見 rally 的 day_atr）。
    dg = day.groupby("stock_id")
    day["_prev_close"] = dg["close"].shift(1)
    day["_day_tr"] = np.maximum(
        np.maximum((day["high"] - day["low"]).abs(), (day["high"] - day["_prev_close"]).abs()),
        (day["low"] - day["_prev_close"]).abs(),
    )
    dg = day.groupby("stock_id")
    day["_atr14"] = _degroup(dg["_day_tr"].rolling(14, min_periods=14).mean(), day.index)
    day["day_atr"] = day["_atr14"].shift(1) / day["open"].replace(0, np.nan)
    day = day.drop(columns=["_prev_close", "_day_tr", "_atr14"])

    # 5/10/20日均線乖離：昨收 vs（算到昨天為止的）N日均線，> 0 代表昨天站上均線。
    # 跟 rally 的 pos_20d 同做法：先算「當日版」乖離，再整個 shift(1) 到下一天，
    # 避免用到「今天自己」的收盤價。
    dg = day.groupby("stock_id")
    ma_cols = []
    for n in [5, 10, 20]:
        ma_n = _degroup(dg["close"].rolling(n, min_periods=n).mean(), day.index)
        col = f"ma_dev_{n}"
        _dev_raw = day["close"] / ma_n.replace(0, np.nan) - 1
        day[col] = _dev_raw.groupby(day["stock_id"], group_keys=False).shift(1)
        ma_cols.append(col)

    # 20日均量（股數，昨天為止）：給 train.py 篩選流動性用（例如只留20日均量
    # ≥1000張的股票），過濾冷門股的1分K雜訊。跟 ma_dev 同做法：先算「當日版」
    # （包含今天自己成交量，收盤才知道），再整個 shift(1) 到下一天。
    dg = day.groupby("stock_id")
    _vol_ma20_raw = _degroup(dg["volume"].rolling(20, min_periods=20).mean(), day.index)
    day["vol_ma20"] = _vol_ma20_raw.groupby(day["stock_id"], group_keys=False).shift(1)

    # 過去1~5天各自的收盤價（原始價格，只當合併用的暫存基準，merge完就丟掉）
    dg = day.groupby("stock_id")
    close_lag_ref_cols = []
    for lag in range(1, 6):
        col = f"_close_lag{lag}_ref"
        day[col] = dg["close"].shift(lag)
        close_lag_ref_cols.append(col)

    day_feat_cols = ["stock_id", "day_date"] + day_ret_cols + ["day_atr", "vol_ma20"] + ma_cols + close_lag_ref_cols
    m1 = m1.merge(day[day_feat_cols], on=["stock_id", "day_date"], how="left")

    # 目前價格 vs 過去1~5天各自收盤價的漲跌幅：逐分鐘更新，代表「今天到目前
    # 為止相對過去N天收盤的累積漲跌幅」——台股漲跌停、當日整體表現都是以
    # 昨收為基準，今天若突破昨收（由負轉正）是重要的技術位階；再往前比對
    # D-2~D-5，可以看出這波漲跌是只對昨天有意義、還是對過去幾天都成立
    # （例如 lag1~lag3 都轉正代表站上近3天的價格，比只看lag1更強的訊號）。
    close_vs_close_cols = []
    for lag in range(1, 6):
        col = f"close_vs_close_lag{lag}"
        m1[col] = m1["close"] / m1[f"_close_lag{lag}_ref"].replace(0, np.nan) - 1
        close_vs_close_cols.append(col)

    # 今日開盤相對昨收的跳空幅度：跟 close_vs_close_lag1（逐分鐘更新的目前
    # 價格vs昨收）不同，這個只看「開盤那一刻」的跳空，是固定值，跟 rally
    # 既有的 gap 特徵同做法。
    m1["gap_pct"] = day_open / m1["_close_lag1_ref"].replace(0, np.nan) - 1
    m1 = m1.drop(columns=close_lag_ref_cols)

    # 開盤區間寬度 / 個股日ATR：兩者都先除以 day_open 正規化成同單位再相除，
    # 代表「今天的開盤區間，相對這支股票平常的日內波動算寬還是窄」——同樣
    # 是1%的區間寬度，波動大的股票（日ATR本來就大）代表區間收得很窄
    # （盤整訊號強），波動小的股票可能只是正常波動，兩者意義不同，
    # or_range_pct（除以or_low）看不出這個差異。
    m1["or_range_atr_ratio"] = (_or_range / day_open) / m1["day_atr"].replace(0, np.nan)

    # ── 大盤（0050）背景特徵（比照 rally 做法，廣播至所有個股）─────────────
    idx_day = day[day["stock_id"] == "0050"].copy()
    if not idx_day.empty:
        idx_dg = idx_day.groupby("stock_id")
        idx_day["_idx_ret1"] = idx_dg["close"].pct_change(1)
        idx_dg2 = idx_day.groupby("stock_id", group_keys=False)
        idx_day_ret_cols = []
        for lag in range(1, 6):
            cr = f"idx_day_ret_{lag}"
            # shift(lag) 不是 shift(lag-1)：D這一列存的是「D自己收盤vs D-1收盤」的報酬率，
            # D自己全天報酬率要收盤才知道，intraday當下不該看得到（同 CLAUDE.md 提過的
            # day_ret_N 未來資料洩漏 bug，這裡直接用修正後的寫法）。
            idx_day[cr] = idx_dg2["_idx_ret1"].shift(lag)
            idx_day_ret_cols.append(cr)
        idx_day = idx_day.drop(columns=["_idx_ret1"])
        idx_dg = idx_day.groupby("stock_id")
        idx_day["_idx_prev_close"] = idx_dg["close"].shift(1)
        idx_day["_idx_day_tr"] = np.maximum(
            np.maximum((idx_day["high"] - idx_day["low"]).abs(), (idx_day["high"] - idx_day["_idx_prev_close"]).abs()),
            (idx_day["low"] - idx_day["_idx_prev_close"]).abs(),
        )
        idx_dg = idx_day.groupby("stock_id")
        idx_day["_idx_atr14"] = _degroup(idx_dg["_idx_day_tr"].rolling(14, min_periods=14).mean(), idx_day.index)
        idx_day["idx_day_atr"] = idx_day["_idx_atr14"].shift(1) / idx_day["open"].replace(0, np.nan)
        idx_day_feat_cols = ["day_date"] + idx_day_ret_cols + ["idx_day_atr"]
        m1 = m1.merge(idx_day[idx_day_feat_cols], on=["day_date"], how="left")

    # 欄位一律先建立（預設 NaN）：0050 若某天缺資料，這裡不該讓 FEATURES 缺欄位
    # idx_atr（0050 1分K ATR(14)）2026-07-11 拿掉了，理由同 m1_atr（需14根）
    for _col in ["idx_ret_1", "idx_vs_open", "idx_up"]:
        m1[_col] = np.nan

    idx_m1 = m1[m1["stock_id"] == "0050"].copy()
    if not idx_m1.empty:
        idx_g_day = idx_m1.groupby("day_date", group_keys=False)
        idx_day_open = idx_g_day["open"].transform("first").replace(0, np.nan)
        idx_ret_1 = idx_g_day["close"].pct_change(1)
        m1.loc[idx_m1.index, "idx_ret_1"] = idx_ret_1.values
        m1.loc[idx_m1.index, "idx_vs_open"] = (idx_m1["close"] / idx_day_open).values
        m1.loc[idx_m1.index, "idx_up"] = (idx_m1["close"] > idx_day_open).astype(int).values

        # 廣播 0050 特徵至所有個股（以完整時間戳 date 為 key，逐分鐘對齊）
        idx_feat_cols = ["date", "idx_ret_1", "idx_vs_open", "idx_up"]
        idx_feat = m1.loc[idx_m1.index, idx_feat_cols].drop_duplicates("date")
        m1 = m1.drop(columns=["idx_ret_1", "idx_vs_open", "idx_up"], errors="ignore")
        m1 = m1.merge(idx_feat, on=["date"], how="left")

    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    # ── 篩選成候選：只留真正的突破事件，同一天可能有好幾筆（見檔頭說明）──
    m1 = m1[m1["_is_breakout"]].copy()
    m1 = m1.sort_values(["stock_id", "day_date", "date"])
    m1 = m1.drop(columns=["_is_breakout"])

    return m1
