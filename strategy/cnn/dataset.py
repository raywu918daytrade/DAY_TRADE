"""
把 db/m1、db/m3、db/m5（rolling）、db/m3_std、db/m5_std 五路報價組裝成
PyTorch 多分支 Conv1D 要的 tensor，外加 triple barrier label。

跟 rally/orb 的 features.py 不同：那邊回傳的是「表格特徵欄位」給樹模型，這裡
回傳的是「多分支原始序列 tensor」給 CNN，所以用 dataset.py 這個檔名，不叫
features.py，避免混淆。

五路對齊到同一段「過去 LOOKBACK_MINUTES 分鐘」，只是取樣密度不同：
  m1/m3/m5（rolling）：每分鐘一列，抓過去 M1_LEN/M3_LEN/M5_LEN 點
  m3_std/m5_std（獨立K棒）：抓過去 M3STD_LEN/M5STD_LEN 根獨立K棒

跟 rally 一樣不跨日——lookback 窗口只在同一個 (stock_id, day_date) 內取，
避免隔夜跳空汙染窗口（見 rally features.py 的 g_day 分組慣例）。
"""

from pathlib import Path

import numba
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data.query import load_m1, load_m3, load_m3_std, load_m5, load_m5_std
from strategy.cnn.config import (
    HOLD_BARS,
    M1_LEN,
    M3_LEN,
    M3STD_LEN,
    M5_LEN,
    M5STD_LEN,
    SL_PCT,
    TP_PCT,
)

_ROOT = Path(__file__).parent.parent.parent
_CACHE_DIR = _ROOT / "cache/cnn"
_SOURCE_DIRS = ["db/m1", "db/m3", "db/m5", "db/m3_std", "db/m5_std"]

_WIDE_FEATURE_COLS = [
    "m1_open", "m1_high", "m1_low", "m1_close", "m1_volume",
    "m3_open", "m3_high", "m3_low", "m3_close", "m3_volume",
    "m5_open", "m5_high", "m5_low", "m5_close", "m5_volume",
]
_STD_FEATURE_COLS = ["open", "high", "low", "close", "volume"]

BRANCH_NAMES = ["m1", "m3", "m5", "m3_std", "m5_std"]
BRANCH_LENGTHS = {"m1": M1_LEN, "m3": M3_LEN, "m5": M5_LEN, "m3_std": M3STD_LEN, "m5_std": M5STD_LEN}
N_CHANNELS = 5  # open/high/low/close/volume


# ═══════════════════════════════════════════════════════════════════════════════
# Triple Barrier Label（複製自 strategy/rally/features.py，各策略各自維護一份，
# 不額外抽共用模組——現有慣例就是策略之間互不依賴）
# ═══════════════════════════════════════════════════════════════════════════════


@numba.njit(cache=True)
def _barrier_label_numba(
    closes: np.ndarray,
    group_id: np.ndarray,
    hold_bars: int,
    tp_pct: float,
    sl_pct: float,
) -> np.ndarray:
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
        hit = -1
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
    return labels


def _make_barrier_labels(stock_id: pd.Series, day_date: pd.Series, close: pd.Series) -> np.ndarray:
    """close 可以是正規化過的值（/ day_open）——barrier 只看相對百分比變化，
    entry 跟未來 close 同乘/除一個當日常數不影響先漲TP%還是先跌SL%的判定，
    所以不用另外保留一份原始 close。"""
    keys = pd.DataFrame({"stock_id": stock_id, "day_date": day_date})
    group_id = (keys != keys.shift()).any(axis=1).cumsum().to_numpy(dtype=np.int64)
    closes_arr = close.to_numpy(dtype=np.float64)
    return _barrier_label_numba(closes_arr, group_id, HOLD_BARS, TP_PCT, SL_PCT)


# ═══════════════════════════════════════════════════════════════════════════════
# 快取新鮮度檢查（比照 rally/mkt 的 load_features() 慣例）
# ═══════════════════════════════════════════════════════════════════════════════


def _source_mtime() -> float:
    mtimes = []
    for d in _SOURCE_DIRS:
        p = _ROOT / d
        if p.exists():
            mtimes.extend(f.stat().st_mtime for f in p.glob("*.parquet"))
    return max(mtimes) if mtimes else 0.0


def _cache_is_fresh() -> bool:
    meta_path = _CACHE_DIR / "meta.parquet"
    if not meta_path.exists():
        return False
    return meta_path.stat().st_mtime >= _source_mtime()


# ═══════════════════════════════════════════════════════════════════════════════
# 正規化 + label
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_ohlcv(df: pd.DataFrame, prefix: str, day_open: pd.Series, g_day) -> None:
    """OHLC 除以當日開盤價正規化，volume 除以當日累積均量正規化
    （跟 rally features.py 的 m1_open/m1_volume 公式一致），就地改欄位。"""
    for col in ["open", "high", "low", "close"]:
        df[f"{prefix}_{col}"] = df[col] / day_open
    cum_vol = g_day["volume"].transform("cumsum")
    bar_count = g_day["volume"].transform("cumcount") + 1
    df[f"{prefix}_volume"] = df["volume"] / (cum_vol / bar_count).replace(0, np.nan)


def _load_wide_frame(start_date: str, end_date: str) -> pd.DataFrame:
    """把 m1/m3/m5（rolling，每分鐘一列）merge 成一張寬表，正規化 + 算 label。"""
    m1 = load_m1()
    if start_date:
        m1 = m1[m1["date"] >= start_date]
    if end_date:
        m1 = m1[m1["date"] <= end_date]
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    m3 = load_m3().rename(columns={"open": "m3_open", "high": "m3_high", "low": "m3_low", "close": "m3_close", "volume": "m3_volume_raw"})
    m5 = load_m5().rename(columns={"open": "m5_open", "high": "m5_high", "low": "m5_low", "close": "m5_close", "volume": "m5_volume_raw"})

    wide = m1.merge(m3[["stock_id", "date", "m3_open", "m3_high", "m3_low", "m3_close", "m3_volume_raw"]], on=["stock_id", "date"], how="left")
    wide = wide.merge(m5[["stock_id", "date", "m5_open", "m5_high", "m5_low", "m5_close", "m5_volume_raw"]], on=["stock_id", "date"], how="left")

    g_day = wide.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)

    _normalize_ohlcv(wide, "m1", day_open, g_day)
    for prefix in ["m3", "m5"]:
        for col in ["open", "high", "low", "close"]:
            wide[f"{prefix}_{col}"] = wide[f"{prefix}_{col}"] / day_open
        cum_vol = g_day[f"{prefix}_volume_raw"].transform("cumsum")
        bar_count = g_day[f"{prefix}_volume_raw"].transform("cumcount") + 1
        wide[f"{prefix}_volume"] = wide[f"{prefix}_volume_raw"] / (cum_vol / bar_count).replace(0, np.nan)

    wide["target"] = _make_barrier_labels(wide["stock_id"], wide["day_date"], wide["m1_close"])
    return wide


def _load_std_frame(loader, start_date: str, end_date: str) -> pd.DataFrame:
    """m3_std/m5_std（獨立K棒）正規化。day_open 用自己這根K棒所屬交易日的
    當日開盤價（跟 wide frame 用同一支股票同一天的 open 第一筆，數值上會對上）。"""
    df = loader()
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df["day_date"] = df["date"].dt.date

    g_day = df.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] / day_open
    cum_vol = g_day["volume"].transform("cumsum")
    bar_count = g_day["volume"].transform("cumcount") + 1
    df["volume"] = df["volume"] / (cum_vol / bar_count).replace(0, np.nan)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 視窗組裝
# ═══════════════════════════════════════════════════════════════════════════════


def _build_windows(start_date: str = "", end_date: str = "") -> tuple[dict, pd.DataFrame]:
    """回傳 ({branch_name: np.ndarray shape (N, 5, branch_len)}, meta df[stock_id/date/target])。

    效能備註：外層對每個 (stock_id, day_date) 分組跑一次 Python 迴圈（組數上看數十萬），
    組內全部用 numpy 向量化操作（sliding_window_view / fancy-index gather），沒有逐候選
    的 inner python 迴圈。這是「最基本架構」版本，第一次跑建議用 --start_date/--end_date
    限縮月份範圍，避免全量22個月資料第一次就跑很久；之後有需要再優化成更少 Python 迴圈
    開銷的寫法（例如整批 groupby.indices 一次算完）。
    """
    wide = _load_wide_frame(start_date, end_date)
    m3_std = _load_std_frame(load_m3_std, start_date, end_date)
    m5_std = _load_std_frame(load_m5_std, start_date, end_date)

    m3_std_groups = {k: v for k, v in m3_std.groupby(["stock_id", "day_date"], sort=False)}
    m5_std_groups = {k: v for k, v in m5_std.groupby(["stock_id", "day_date"], sort=False)}

    offsets3 = np.arange(-(M3STD_LEN - 1), 1)
    offsets5 = np.arange(-(M5STD_LEN - 1), 1)

    m1m3m5_list, m3std_list, m5std_list = [], [], []
    meta_stock, meta_date, meta_target = [], [], []

    n_groups = 0
    for (stock_id, day_date), day_df in wide.groupby(["stock_id", "day_date"], sort=False):
        n_groups += 1
        day_len = len(day_df)
        if day_len < M1_LEN:
            continue

        key = (stock_id, day_date)
        m3std_day = m3_std_groups.get(key)
        m5std_day = m5_std_groups.get(key)
        if m3std_day is None or m5std_day is None:
            continue

        arr = day_df[_WIDE_FEATURE_COLS].to_numpy(dtype=np.float32)
        win = sliding_window_view(arr, M1_LEN, axis=0)  # (day_len-M1_LEN+1, 15, M1_LEN)
        cand_local_idx = np.arange(M1_LEN - 1, day_len)
        cand_dates = day_df["date"].to_numpy()[cand_local_idx]
        cand_targets = day_df["target"].to_numpy()[cand_local_idx]

        finite = np.isfinite(win).all(axis=(1, 2)) & ~np.isnan(cand_targets)
        if not finite.any():
            continue

        m3std_arr = m3std_day[_STD_FEATURE_COLS].to_numpy(dtype=np.float32)
        m3std_dates = m3std_day["date"].to_numpy()
        m5std_arr = m5std_day[_STD_FEATURE_COLS].to_numpy(dtype=np.float32)
        m5std_dates = m5std_day["date"].to_numpy()

        # asof backward：候選當下最近一根已收盤的獨立K棒 index
        j3 = np.searchsorted(m3std_dates, cand_dates, side="right") - 1
        j5 = np.searchsorted(m5std_dates, cand_dates, side="right") - 1

        valid = finite & (j3 >= M3STD_LEN - 1) & (j5 >= M5STD_LEN - 1)
        if not valid.any():
            continue

        idx = np.nonzero(valid)[0]
        j3v, j5v = j3[idx], j5[idx]

        rows3 = j3v[:, None] + offsets3[None, :]  # (n_valid, M3STD_LEN)
        rows5 = j5v[:, None] + offsets5[None, :]
        w3 = m3std_arr[rows3].transpose(0, 2, 1)  # (n_valid, 5, M3STD_LEN)
        w5 = m5std_arr[rows5].transpose(0, 2, 1)

        finite_std = np.isfinite(w3).all(axis=(1, 2)) & np.isfinite(w5).all(axis=(1, 2))
        if not finite_std.any():
            continue

        final_idx = idx[finite_std]
        m1m3m5_list.append(win[final_idx])
        m3std_list.append(w3[finite_std])
        m5std_list.append(w5[finite_std])
        meta_stock.append(np.full(final_idx.shape[0], stock_id))
        meta_date.append(cand_dates[final_idx])
        meta_target.append(cand_targets[final_idx])

        if n_groups % 2000 == 0:
            print(f"  ...processed {n_groups} (stock_id, day) groups, {sum(len(m) for m in meta_target):,} samples so far")

    combined = np.concatenate(m1m3m5_list, axis=0)  # (N, 15, M1_LEN)
    branches = {
        "m1": combined[:, 0:5, :],
        "m3": combined[:, 5:10, :],
        "m5": combined[:, 10:15, :],
        "m3_std": np.concatenate(m3std_list, axis=0),
        "m5_std": np.concatenate(m5std_list, axis=0),
    }
    meta = pd.DataFrame({
        "stock_id": np.concatenate(meta_stock),
        "date": np.concatenate(meta_date),
        "target": np.concatenate(meta_target).astype(np.int64),
    })
    return branches, meta


# ═══════════════════════════════════════════════════════════════════════════════
# 對外入口
# ═══════════════════════════════════════════════════════════════════════════════


def build_dataset(start_date: str = "", end_date: str = "", force_rebuild: bool = False) -> None:
    """組裝五路 tensor + label，寫入 cache/cnn/ 底下。"""
    if not force_rebuild and _cache_is_fresh():
        print("cache/cnn/ 已是最新，略過重算（force_rebuild=True 可強制重算）")
        return

    print("組裝五路多解析度 tensor（第一次跑或資料有更新時才會執行，會需要一段時間）...")
    branches, meta = _build_windows(start_date, end_date)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in branches.items():
        np.save(_CACHE_DIR / f"{name}.npy", arr)
    meta.to_parquet(_CACHE_DIR / "meta.parquet")
    print(f"完成，共 {len(meta):,} 筆樣本，寫入 {_CACHE_DIR}")


def load_dataset(start_date: str = "", end_date: str = "", force_rebuild: bool = False) -> tuple[dict, pd.DataFrame]:
    """讀取（必要時先重建）cache/cnn/ 底下的五路 tensor + meta。"""
    build_dataset(start_date=start_date, end_date=end_date, force_rebuild=force_rebuild)
    branches = {name: np.load(_CACHE_DIR / f"{name}.npy") for name in BRANCH_NAMES}
    meta = pd.read_parquet(_CACHE_DIR / "meta.parquet")
    return branches, meta
