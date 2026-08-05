"""
將 vwap_ml 的 VWAP z-score 候選觸發邏輯，與 ResNet + GRU 視窗組裝結合。

ResNet 看近 10 分鐘原始 OHLCV，GRU 從 9:00 到當下逐分鐘累積 VWAP 軌跡。

流程：
  1. 用 vwap_ml/features.py::make_features() 算出 VWAP z-score、候選觸發、
     三分類標籤（回歸/無訊號/延續）。
  2. 對每個候選，從 wide frame 中取出：
     a. resnet_x: 近 10 分鐘的原始 OHLCV（5 channels × 10 步）
     b. gru_seq: 從 9:00 到當下的逐分鐘 14 維特徵（可變長度）
     c. gru_lengths: 實際長度
  3. 存成 cache，供 train.py 讀取。
"""

import sys
from collections import defaultdict
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.raw_query import iter_m1_months, iter_m3_months, iter_m3_std_months, iter_m5_months, iter_m5_std_months
from strategy.vwap_dl.config import ATR5_FILTER_THRESHOLD, IDX_SYMBOL, STD_MULT
from strategy.vwap_ml.features import make_features

_ROOT = Path(__file__).parent.parent.parent
_CACHE_DIR = _ROOT / "cache/vwap_dl"
_SOURCE_DIRS = ["db/m1", "db/m3", "db/m5", "db/m3_std", "db/m5_std"]

# GRU 每步 18 維特徵：OHLCV(5) + 技術指標(6) + VWAP z-score(3) + 大盤 VWAP 特徵(4)
# 2026-07-28 新增 4 個大盤 VWAP 特徵（market_z_score_m5 /
#   market_vwap_alignment_score / market_vwap_spread_1_5 /
#   velocity_ratio_to_market），GRU_INPUT_DIM 同步從 14 改為 18。
# 2026-08-04 曾經試過加 m3_std/m5_std 獨立K棒形態（body_ratio/
#   upper_shadow_ratio/lower_shadow_ratio 各時間框各3個，_add_std_bar_shape()
#   函式還在，供之後想重試時直接重用）——離線指標（accuracy/AUC/延續類
#   precision）全面變好，但實際回測從 10筆/80%勝率/+1.20% 掉到
#   4筆/50%勝率/-0.13%，判斷是這個改動讓結果變差，改回不用，
#   GRU_INPUT_DIM 改回 18。
# 2026-08-04 也曾試過加 m1_engulfing（M1吞噬型態，_add_engulfing() 函式
#   還在，供之後想重試時直接重用）——搭配STD_MULT=1.5一起測，30天/90天
#   回測都比基準差（10筆/80%/+1.20%→5筆/40%/+0.05%；21筆/71.4%/+1.60%→
#   13筆/61.5%/+0.86%），改回不用，GRU_INPUT_DIM 改回 18（STD_MULT保留
#   1.5，只有吞噬特徵這個改動被判斷是變差的原因才拿掉）。
_GRU_FEATURE_COLS = [
    "m1_open",
    "m1_high",
    "m1_low",
    "m1_close",
    "m1_volume",
    "atr5",
    "ma10",
    "ma5",
    "ma3",
    "ret_vs_idx",
    "idx_ret_since_open",
    "m1_vwap_z",
    "m3_vwap_z",
    "m5_vwap_z",
    "market_z_score_m5",
    "market_vwap_alignment_score",
    "market_vwap_spread_1_5",
    "velocity_ratio_to_market",
]
# ResNet 只看原始 OHLCV（5 channels）
_RESNET_COLS = ["m1_open", "m1_high", "m1_low", "m1_close", "m1_volume"]


# ═══════════════════════════════════════════════════════════════════════════════
# 正規化（參照 cnn/dataset.py 的同一套做法，只保留需要的）
# ═══════════════════════════════════════════════════════════════════════════════


def _degroup(s: pd.Series, index: pd.Index) -> pd.Series:
    n_key_levels = s.index.nlevels - 1
    if n_key_levels:
        s = s.reset_index(level=list(range(n_key_levels)), drop=True)
    return s.reindex(index)


def _normalize_ohlcv(df: pd.DataFrame, prefix: str, day_open: pd.Series, g_day) -> None:
    for col in ["open", "high", "low", "close"]:
        df[f"{prefix}_{col}"] = df[col] / day_open
    cum_vol = g_day["volume"].transform("cumsum")
    bar_count = g_day["volume"].transform("cumcount") + 1
    df[f"{prefix}_volume"] = df["volume"] / (cum_vol / bar_count).replace(0, np.nan)


def _add_atr_ma(df: pd.DataFrame, g_day, day_open: pd.Series) -> None:
    prev_close = g_day["close"].shift(1).fillna(df["open"])
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["_tr"] = tr
    atr5 = _degroup(
        df.groupby(["stock_id", "day_date"], group_keys=False)["_tr"].rolling(5, min_periods=5).mean(),
        df.index,
    )
    df["atr5"] = (atr5 / day_open).fillna(0)
    df.drop(columns="_tr", inplace=True)
    for name, window in [("ma10", 10), ("ma5", 5), ("ma3", 3)]:
        ma = _degroup(g_day["close"].rolling(window, min_periods=window).mean(), df.index)
        df[name] = (ma / day_open).fillna(0)


def _add_engulfing(df: pd.DataFrame, g_day) -> None:
    """M1吞噬型態（經典反轉訊號，2026-08-04新增）：
    bullish_engulfing = 當前陽線的實體完全包住前一根陰線的實體（看漲反轉）
    bearish_engulfing = 當前陰線的實體完全包住前一根陽線的實體（看跌反轉）

    輸出單一signed特徵 m1_engulfing ∈ {-1,0,+1}（bearish/都不是/bullish），
    比照 add_market_vwap_features()::market_vwap_alignment_score 的
    {-1,0,1}慣例，不拆成兩個獨立的0/1旗標——bullish/bearish互斥（同一根
    K棒不可能同時是陽線又是陰線），沒有必要各自佔一個維度。

    每天第一根K棒沒有「前一根」，shift(1)後prev_open/prev_close是NaN，
    NaN參與的比較（<,>,<=,>=）在pandas/numpy裡結果是False，不會誤判成
    任何一種吞噬，不用額外處理邊界。"""
    prev_open = g_day["open"].shift(1)
    prev_close = g_day["close"].shift(1)
    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open
    curr_bullish = df["close"] > df["open"]
    curr_bearish = df["close"] < df["open"]
    bullish_engulf = prev_bearish & curr_bullish & (df["open"] <= prev_close) & (df["close"] >= prev_open)
    bearish_engulf = prev_bullish & curr_bearish & (df["open"] >= prev_close) & (df["close"] <= prev_open)
    df["m1_engulfing"] = np.select([bullish_engulf, bearish_engulf], [1, -1], default=0)


def _add_std_bar_shape(df: pd.DataFrame, prefix: str, std_df: pd.DataFrame) -> pd.DataFrame:
    """把獨立K棒（m3_std/m5_std，一根一列，`date`=收盤時刻）的形態特徵——
    body_ratio/upper_shadow_ratio/lower_shadow_ratio（公式同
    strategy/breakout_retest_ml/features.py::_shadow_ratios()，都用該K棒
    自己的全長 high-low 正規化，全長<=0時三個比例都是0）——用
    `pd.merge_asof(direction="backward")` 併回逐分鐘寬表 df，而不是普通
    merge：獨立K棒不是每分鐘都有新值（3分K要等3分鐘才收一根），要取「這
    一分鐘為止，最後一根已經收盤的K棒」。`std_df["date"]` 標記的是K棒收盤
    時刻，backward+預設 allow_exact_matches=True 剛好符合「K棒收盤那一
    分鐘起，資訊才算已知」的語意，不會有未來函數。收盤前（例如當天前
    3/5分鐘）沒有值，fillna(0)，比照 atr5/market_* 欄位的做法。

    merge_asof 要求兩邊都按 `on`（這裡是 date）全域排序，不能只排
    [stock_id, date]（by分組不保證跨組date單調遞增）——df/std_df 進來時
    可能是 [stock_id, date] 排序，這裡各自重排一次。呼叫端不用管排序，
    回傳的 df 沒有保證維持任何特定排序（下游 _build_candidate_data() 是
    用 groupby 處理，不依賴frame本身的排序）。
    """
    std_df = std_df.sort_values("date").copy()
    full = std_df["high"] - std_df["low"]
    safe_full = full.where(full > 0, np.nan)
    std_df[f"{prefix}_body_ratio"] = ((std_df["close"] - std_df["open"]).abs() / safe_full).fillna(0)
    std_df[f"{prefix}_upper_shadow_ratio"] = (
        (std_df["high"] - std_df[["open", "close"]].max(axis=1)) / safe_full
    ).fillna(0)
    std_df[f"{prefix}_lower_shadow_ratio"] = (
        (std_df[["open", "close"]].min(axis=1) - std_df["low"]) / safe_full
    ).fillna(0)

    shape_cols = [f"{prefix}_body_ratio", f"{prefix}_upper_shadow_ratio", f"{prefix}_lower_shadow_ratio"]
    merged = pd.merge_asof(
        df.sort_values("date"),
        std_df[["stock_id", "date"] + shape_cols],
        on="date",
        by="stock_id",
        direction="backward",
    )
    for col in shape_cols:
        merged[col] = merged[col].fillna(0)
    return merged


def _add_market_relative(df: pd.DataFrame, day_open: pd.Series, idx_symbol: str) -> pd.DataFrame:
    ret_since_open = df["close"] / day_open - 1
    idx = (
        pd.DataFrame({"date": df["date"], "stock_id": df["stock_id"], "_idx_ret": ret_since_open})
        .loc[lambda x: x["stock_id"] == idx_symbol, ["date", "_idx_ret"]]
        .drop_duplicates("date")
        .rename(columns={"_idx_ret": "idx_ret_since_open"})
    )
    df = df.merge(idx, on="date", how="left")
    df["ret_vs_idx"] = ret_since_open.to_numpy() - df["idx_ret_since_open"]
    return df


def _build_wide_frame(
    m1: pd.DataFrame, m3: pd.DataFrame, m5: pd.DataFrame, m3_std: pd.DataFrame, m5_std: pd.DataFrame
) -> pd.DataFrame:
    """m1 → 寬表：正規化 OHLCV + atr/ma/大盤相對 + VWAP z-score merge +
    獨立K棒形態（m3_std/m5_std，2026-08-04新增，見 _add_std_bar_shape()）。

    m3/m5（2026-08-03改）：改成呼叫端傳入——原本這裡會自己再呼叫一次
    load_m3()/load_m5()，跟 build_dataset() 裡已經為了 make_features() 載入
    的 m3/m5 完全重複（多載一次全市場全歷史），而且呼叫端對 m3/m5 做的
    start_date/stock_ids 過濾也會在這裡被忽略掉。改成直接吃呼叫端已經篩過
    的版本，不重新載入。m3_std/m5_std 比照同一個原則，也是呼叫端傳入已經
    篩過範圍/股票池的版本。"""
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    m3 = m3.rename(
        columns={"open": "m3_open", "high": "m3_high", "low": "m3_low", "close": "m3_close", "volume": "m3_volume_raw"}
    )
    m5 = m5.rename(
        columns={"open": "m5_open", "high": "m5_high", "low": "m5_low", "close": "m5_close", "volume": "m5_volume_raw"}
    )

    wide = m1.merge(
        m3[["stock_id", "date", "m3_open", "m3_high", "m3_low", "m3_close", "m3_volume_raw"]],
        on=["stock_id", "date"],
        how="left",
    )
    wide = wide.merge(
        m5[["stock_id", "date", "m5_open", "m5_high", "m5_low", "m5_close", "m5_volume_raw"]],
        on=["stock_id", "date"],
        how="left",
    )

    wide = _add_std_bar_shape(wide, "m3s", m3_std)
    wide = _add_std_bar_shape(wide, "m5s", m5_std)

    day_open_pre = (
        wide.groupby(["stock_id", "day_date"], group_keys=False)["open"].transform("first").replace(0, np.nan)
    )
    wide = _add_market_relative(wide, day_open_pre, IDX_SYMBOL)

    g_day = wide.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    _add_atr_ma(wide, g_day, day_open)
    _normalize_ohlcv(wide, "m1", day_open, g_day)

    return wide


# ═══════════════════════════════════════════════════════════════════════════════
# 候選視窗組裝
# ═══════════════════════════════════════════════════════════════════════════════


def _build_candidate_data(
    wide: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """對所有候選，組裝 ResNet 與 GRU 的 tensor。

    回傳 ({key: np.ndarray}, meta df)，key 包含：
      - resnet_x: (N, 5, 10) 近 10 分鐘 OHLCV
      - gru_seq: list of np.ndarray, 每個 (seq_len, 14)
      - gru_lengths: (N,) 每筆的 seq_len
    """
    resnet_list, gru_seq_list, gru_len_list = [], [], []
    meta_stock, meta_date, meta_target, meta_atr5, meta_trigger_side = [], [], [], [], []

    wide_groups = wide.groupby(["stock_id", "day_date"], sort=False)
    cand_groups = candidates.groupby(["stock_id", "day_date"], sort=False)

    for key, cand_grp in cand_groups:
        stock_id, day_date = key
        if key not in wide_groups.groups:
            continue
        day_df = wide_groups.get_group(key).sort_values("date")
        day_len = len(day_df)
        if day_len < 10:
            continue

        # 當天第一分鐘 ~ 最後一分鐘的陣列
        day_arr = day_df[_GRU_FEATURE_COLS].to_numpy(dtype=np.float32)
        resnet_arr = day_df[_RESNET_COLS].to_numpy(dtype=np.float32)
        day_dates = day_df["date"].to_numpy()

        cand_dates = cand_grp["date"].to_numpy()

        for cand_date in cand_dates:
            idx = np.searchsorted(day_dates, cand_date, side="right") - 1
            if idx < 0 or idx >= day_len:
                continue

            # ResNet: 近 10 分鐘（候選往前 9 步到當下，共 10 步）
            r_start = idx - 9
            if r_start < 0:
                continue
            resnet_win = resnet_arr[r_start : idx + 1]  # (10, 5)
            if not np.isfinite(resnet_win).all():
                continue

            # GRU: 從 9:00（第 0 步）到候選當下（idx）
            gru_seq = day_arr[: idx + 1]  # (seq_len, 14)
            if not np.isfinite(gru_seq).all():
                continue

            resnet_list.append(resnet_win.T)  # (5, 10)
            gru_seq_list.append(gru_seq)  # (seq_len, 14)
            gru_len_list.append(len(gru_seq))

            cand_row = cand_grp[cand_grp["date"] == cand_date].iloc[0]
            meta_stock.append(stock_id)
            meta_date.append(cand_date)
            meta_target.append(int(cand_row["target"]))
            meta_atr5.append(float(cand_row["atr5"]))
            meta_trigger_side.append(cand_row["trigger_side"])

    if not meta_target:
        return {}, pd.DataFrame(columns=["stock_id", "date", "target", "atr5", "trigger_side"])

    data = {
        "resnet_x": np.stack(resnet_list, axis=0),  # (N, 5, 10)
        "gru_seq": gru_seq_list,  # list of (seq_len, 14)
        "gru_lengths": np.array(gru_len_list, dtype=np.int64),  # (N,)
    }
    meta = pd.DataFrame(
        {
            "stock_id": meta_stock,
            "date": meta_date,
            "target": np.array(meta_target, dtype=np.int64),
            "atr5": np.array(meta_atr5, dtype=np.float32),
            "trigger_side": meta_trigger_side,
        }
    )
    return data, meta


# ═══════════════════════════════════════════════════════════════════════════════
# 快取新鮮度檢查
# ═══════════════════════════════════════════════════════════════════════════════


def _month_bound(date_str: str) -> str:
    return f"{date_str[:4]}_{date_str[5:7]}"


def _source_months(start_date: str, end_date: str) -> list[str]:
    m1_dir = _ROOT / "db/m1"
    months = sorted(p.stem for p in m1_dir.glob("*.parquet"))
    if start_date:
        months = [m for m in months if m >= _month_bound(start_date)]
    if end_date:
        months = [m for m in months if m <= _month_bound(end_date)]
    return months


def _month_source_mtime(month: str) -> float:
    mtimes = []
    for d in _SOURCE_DIRS:
        p = _ROOT / d / f"{month}.parquet"
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def _shard_is_fresh(month: str) -> bool:
    meta_path = _CACHE_DIR / f"meta_{month}.parquet"
    if not meta_path.exists():
        return False
    return meta_path.stat().st_mtime >= _month_source_mtime(month)


# ═══════════════════════════════════════════════════════════════════════════════
# 對外入口 — 按月分片 build
# ═══════════════════════════════════════════════════════════════════════════════


def build_dataset(
    start_date: str = "",
    end_date: str = "",
    std_mult: float = STD_MULT,
    force_rebuild: bool = False,
    stock_ids: list[str] | None = "__default__",
) -> list[str]:
    """
    stock_ids（2026-08-03新增）：限制只用這份清單裡的股票建shard。預設
    （不傳，走 "__default__" sentinel）＝ finmind.tick_universe.
    load_tick_universe() 的400支固定候選股母體——vwap_dl 現在訓練/回測/
    即時交易統一用同一批母體，不用每個呼叫端自己記得傳。傳 None 才是
    「不過濾，吃全市場」，只給明確需要全市場實驗時用。

    2026-08-03發現：db/m1/db/m3/db/m5 按月分檔，2026-08-01之前backfill的
    月份檔案還是舊的全市場~2700支（含大量權證/冷門股，量能經常個位數），
    寫入端（data/m1_data_loader.py、finmind/backfill_m1_history.py）雖然
    已經改成固定抓400支，但既有檔案不會因此自動變乾淨——讀取端要自己
    過濾，才能保證訓練/回測用的股票池，跟即時交易的候選股（400支）一致。
    """
    if stock_ids == "__default__":
        from finmind.tick_universe import load_tick_universe

        stock_ids = load_tick_universe()

    months = _source_months(start_date, end_date)
    if not months:
        return []

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stale_months = {m for m in months if force_rebuild or not _shard_is_fresh(m)}

    if not stale_months:
        for month in months:
            print(f"  {month} shard 已是最新，略過重算")
        return months

    # 逐月讀取＋逐月算特徵＋逐月組裝（2026-08-04改，比照
    # strategy/vwap_ml/train.py::_prepare_data() 的做法）：原本這裡一次讀完
    # start_date到現在全部月份的m1/m3/m5/m3_std/m5_std，一次算完
    # make_features()/_build_wide_frame()，只有最後組裝shard才逐月處理——
    # 真正花記憶體的兩步（算VWAP z-score候選觸發/三分類標籤、建寬表）完全
    # 沒有分批，STD_MULT/ATR5_FILTER_THRESHOLD調鬆一點候選變多時就會把
    # 記憶體衝爆（2026-08-04 STD_MULT從2.0調到1.5時實測發現，free記憶體一度
    # 只剩~60MB、swap被逼到快用完）。改成每個月讀完立刻算完特徵、組裝完
    # shard，讀下一個月前這個月的資料就能被回收，尖峰記憶體壓到單月等級。
    print("逐月載入原始資料並算特徵...")
    m1_iter = iter_m1_months(start_date or None)
    m3_iter = iter_m3_months(start_date or None)
    m5_iter = iter_m5_months(start_date or None)
    m3_std_iter = iter_m3_std_months(start_date or None)
    m5_std_iter = iter_m5_std_months(start_date or None)

    for month, m1, m3, m5, m3_std, m5_std in zip(months, m1_iter, m3_iter, m5_iter, m3_std_iter, m5_std_iter):
        if month not in stale_months:
            print(f"  {month} shard 已是最新，略過重算")
            continue

        if end_date:
            m1 = m1[m1["date"] <= end_date]
            m3 = m3[m3["date"] <= end_date]
            m5 = m5[m5["date"] <= end_date]
            m3_std = m3_std[m3_std["date"] <= end_date]
            m5_std = m5_std[m5_std["date"] <= end_date]
        if stock_ids is not None:
            m1 = m1[m1["stock_id"].isin(stock_ids)]
            m3 = m3[m3["stock_id"].isin(stock_ids)]
            m5 = m5[m5["stock_id"].isin(stock_ids)]
            m3_std = m3_std[m3_std["stock_id"].isin(stock_ids)]
            m5_std = m5_std[m5_std["stock_id"].isin(stock_ids)]

        # 2026-08-04修：之前這裡沒傳 atr5_threshold，make_features() 會用它
        # 自己在 strategy/vwap_ml/features.py 裡的預設值（綁定的是
        # vwap_ml.config.ATR5_FILTER_THRESHOLD，不是這裡 import 進來的
        # vwap_dl.config.ATR5_FILTER_THRESHOLD）——等於 vwap_dl 自己的
        # ATR5_FILTER_THRESHOLD 從頭到尾沒真的生效過，一直在用 vwap_ml 那份
        # 的值（剛好兩邊都設0.01，才没被發現）。這裡明確傳進去，兩個常數
        # 才會分開運作。
        candidates = make_features(m1, std_mult=std_mult, atr5_threshold=ATR5_FILTER_THRESHOLD, m3=m3, m5=m5)
        candidates = candidates.dropna(subset=["target"])
        candidates["target"] = candidates["target"].astype(int)
        print(f"  {month} 有效候選樣本: {len(candidates):,} 筆")

        if len(candidates) == 0:
            print(f"  {month} 沒有候選樣本，略過存檔")
            continue

        wide = _build_wide_frame(m1, m3, m5, m3_std, m5_std)

        # VWAP z-score merge 回 wide
        vwap_z_cols = candidates[["stock_id", "date", "m1_vwap_z", "m3_vwap_z", "m5_vwap_z"]].drop_duplicates(
            subset=["stock_id", "date"]
        )
        wide = wide.merge(vwap_z_cols, on=["stock_id", "date"], how="left")
        wide["m1_vwap_z"] = wide["m1_vwap_z"].fillna(0)
        wide["m3_vwap_z"] = wide["m3_vwap_z"].fillna(0)
        wide["m5_vwap_z"] = wide["m5_vwap_z"].fillna(0)

        # 2026-07-28 新增：大盤（0050）VWAP 特徵 merge 回 wide（同一分鐘所有
        # 股票共用同一組大盤值，所以按 date 去重後 merge）。
        market_cols = candidates[
            [
                "date",
                "market_z_score_m5",
                "market_vwap_alignment_score",
                "market_vwap_spread_1_5",
                "velocity_ratio_to_market",
            ]
        ].drop_duplicates("date")
        wide = wide.merge(market_cols, on="date", how="left")
        wide["market_z_score_m5"] = wide["market_z_score_m5"].fillna(0)
        wide["market_vwap_alignment_score"] = wide["market_vwap_alignment_score"].fillna(0)
        wide["market_vwap_spread_1_5"] = wide["market_vwap_spread_1_5"].fillna(0)
        wide["velocity_ratio_to_market"] = wide["velocity_ratio_to_market"].fillna(0)

        print(f"  組裝 {month}（{len(candidates):,} 候選）...")
        data, meta = _build_candidate_data(wide, candidates)
        if len(meta) == 0:
            print(f"  {month} 視窗組裝後無有效樣本，略過存檔")
            continue

        # 存檔：resnet_x / gru_seq（list 用 np.save + allow_pickle）/ gru_lengths
        np.save(_CACHE_DIR / f"resnet_x_{month}.npy", data["resnet_x"])
        np.save(_CACHE_DIR / f"gru_lengths_{month}.npy", data["gru_lengths"])
        # gru_seq 是 list of arrays，需要用 pickle 模式存
        np.save(_CACHE_DIR / f"gru_seq_{month}.npy", np.array(data["gru_seq"], dtype=object), allow_pickle=True)
        meta.to_parquet(_CACHE_DIR / f"meta_{month}.parquet")
        print(f"  {month} 完成，{len(meta):,} 筆")

    return months


def available_months() -> list[str]:
    if not _CACHE_DIR.exists():
        return []
    return sorted(p.stem.replace("meta_", "") for p in _CACHE_DIR.glob("meta_*.parquet"))


def load_shard_meta(month: str) -> pd.DataFrame:
    return pd.read_parquet(_CACHE_DIR / f"meta_{month}.parquet")


def load_shard_data(month: str, mmap: bool = True) -> dict:
    """載入單月份 shard 的三個 numpy 陣列。gru_seq 用 allow_pickle=True 載入。"""
    return {
        "resnet_x": np.load(_CACHE_DIR / f"resnet_x_{month}.npy", mmap_mode="r" if mmap else None),
        "gru_seq": np.load(_CACHE_DIR / f"gru_seq_{month}.npy", allow_pickle=True),
        "gru_lengths": np.load(_CACHE_DIR / f"gru_lengths_{month}.npy"),
    }
