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

from data.raw_query import load_m1, load_m3, load_m3_std, load_m5, load_m5_std
from strategy.vwap_dl.config import ATR5_FILTER_THRESHOLD, IDX_SYMBOL, STD_MULT
from strategy.vwap_ml.features import make_features

_ROOT = Path(__file__).parent.parent.parent
_CACHE_DIR = _ROOT / "cache/vwap_dl"
_SOURCE_DIRS = ["db/m1", "db/m3", "db/m5", "db/m3_std", "db/m5_std"]

# GRU 每步 18 維特徵：OHLCV(5) + 技術指標(6) + VWAP z-score(3) + 大盤 VWAP 特徵(4)
# 2026-07-28 新增 4 個大盤 VWAP 特徵（market_z_score_m5 /
#   market_vwap_alignment_score / market_vwap_spread_1_5 /
#   velocity_ratio_to_market），GRU_INPUT_DIM 同步從 14 改為 18。
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


def _build_wide_frame(m1: pd.DataFrame) -> pd.DataFrame:
    """m1 → 寬表：正規化 OHLCV + atr/ma/大盤相對 + VWAP z-score merge。"""
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    m3 = load_m3().rename(
        columns={"open": "m3_open", "high": "m3_high", "low": "m3_low", "close": "m3_close", "volume": "m3_volume_raw"}
    )
    m5 = load_m5().rename(
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


def _month_key(day_date) -> str:
    return f"{day_date.year}_{day_date.month:02d}"


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
) -> list[str]:
    months = _source_months(start_date, end_date)
    if not months:
        return []

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stale_months = {m for m in months if force_rebuild or not _shard_is_fresh(m)}

    if not stale_months:
        for month in months:
            print(f"  {month} shard 已是最新，略過重算")
        return months

    # 載入原始資料
    print("載入原始資料...")
    raw_m1 = load_m1()
    if start_date:
        raw_m1 = raw_m1[raw_m1["date"] >= start_date]
    if end_date:
        raw_m1 = raw_m1[raw_m1["date"] <= end_date]
    raw_m1["day_date"] = raw_m1["date"].dt.date

    print("算 VWAP z-score / 候選觸發 / 三分類標籤（引用 vwap_ml/features.py，ATR5 過濾由內部處理）...")
    m1 = raw_m1
    m3 = load_m3()
    m5 = load_m5()
    if start_date:
        m3 = m3[m3["date"] >= start_date]
        m5 = m5[m5["date"] >= start_date]
    if end_date:
        m3 = m3[m3["date"] <= end_date]
        m5 = m5[m5["date"] <= end_date]

    candidates = make_features(m1, std_mult=std_mult, m3=m3, m5=m5)
    candidates = candidates.dropna(subset=["target"])
    candidates["target"] = candidates["target"].astype(int)
    print(f"  有效候選樣本: {len(candidates):,} 筆")

    if len(candidates) == 0:
        print("沒有候選樣本，跳過存檔。")
        return []

    # 建寬表
    print("建寬表 + 正規化 + 技術指標...")
    wide = _build_wide_frame(m1)

    # VWAP z-score merge 回 wide
    vwap_z_cols = candidates[["stock_id", "date", "m1_vwap_z", "m3_vwap_z", "m5_vwap_z"]].drop_duplicates(
        subset=["stock_id", "date"]
    )
    wide = wide.merge(vwap_z_cols, on=["stock_id", "date"], how="left")
    wide["m1_vwap_z"] = wide["m1_vwap_z"].fillna(0)
    wide["m3_vwap_z"] = wide["m3_vwap_z"].fillna(0)
    wide["m5_vwap_z"] = wide["m5_vwap_z"].fillna(0)

    # 2026-07-28 新增：大盤（0050）VWAP 特徵 merge 回 wide（同一分鐘所有股票共用
    # 同一組大盤值，所以按 date 去重後 merge）。
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

    # 依月份分組候選，逐月建視窗
    candidates["_month"] = candidates["date"].apply(_month_key)
    cand_by_month = {m: g.drop(columns="_month") for m, g in candidates.groupby("_month")}

    for month in months:
        if month not in stale_months:
            print(f"  {month} shard 已是最新，略過重算")
            continue

        cand_this = cand_by_month.get(month)
        if cand_this is None or len(cand_this) == 0:
            print(f"  {month} 沒有候選樣本，略過存檔")
            continue

        print(f"  組裝 {month}（{len(cand_this):,} 候選）...")
        data, meta = _build_candidate_data(wide, cand_this)
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
