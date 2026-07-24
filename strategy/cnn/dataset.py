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

== 為什麼按月分片(shard)存，而不是一次存成一個大檔案 ==

2026-07-21 討論：試跑半年資料（~140個交易日）時整個process被系統OOM
killed（exit 137）——這台機器只有24GB RAM，140天粗估視窗tensor本身就要
~50GB，一次性在記憶體裡用python list累積全部視窗、最後才concatenate存檔
的寫法，記憶體峰值會隨著請求的日期範圍線性長大，超過24GB就會被砍。

改成「先把整個範圍的原始資料載入一次（這步驟本身不是問題——5天測試時
就已經是載入全部24個月的m1/m3/m5/m3_std/m5_std再篩選，這個固定成本
本來就沒有OOM過），但視窗組裝完之後不要全部累積在記憶體，改成每處理完
一個月份就立刻把該月的tensor寫進磁碟（cache/cnn/{branch}_{yyyy_mm}.npy
+ meta_{yyyy_mm}.parquet），釋放掉那個月的暫存陣列再處理下一個月」——
這樣記憶體峰值只跟「一個月的視窗量」有關，不會隨請求範圍變大而跟著長大，
可以放心跑到半年甚至全量22個月。

train.py 那邊對應也要用 np.load(..., mmap_mode='r') 跨shard讀取，不能整包
讀進RAM，否則train階段一樣會在大範圍資料上OOM（見 train.py 的
ShardedMultiScaleDataset）。
"""

from collections import defaultdict
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data.query import load_m1, load_m3, load_m3_std, load_m5, load_m5_std
from strategy.cnn.config import (
    HOLD_BARS,
    IDX_SYMBOL,
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

# m1多帶atr5/ma10/ma5/ma3（1分鐘技術指標）+ret_vs_idx/idx_ret_since_open
# （大盤相對強弱，2026-07-24新增，見_add_atr_ma()/_add_market_relative()）。
# 只做ATR沒做ADX——orb已踩過這個坑：ADX雙重平滑需要~28根K棒才穩定，縮小視窗
# 後（M1_LEN=10）候選視窗本身就只有10步，缺值率會非常高，見project_orb_strategy
# 記憶裡的教訓，直接沿用「拿掉adx，只留單次平滑的atr」的結論。
_M1_COLS = [
    "m1_open", "m1_high", "m1_low", "m1_close", "m1_volume",
    "atr5", "ma10", "ma5", "ma3", "ret_vs_idx", "idx_ret_since_open",
]
_M3_COLS = ["m3_open", "m3_high", "m3_low", "m3_close", "m3_volume"]
_M5_COLS = ["m5_open", "m5_high", "m5_low", "m5_close", "m5_volume"]
_WIDE_FEATURE_COLS = _M1_COLS + _M3_COLS + _M5_COLS

# m3_std/m5_std 共用同一份欄位清單（_load_std_frame()統一處理）。
_STD_FEATURE_COLS = ["open", "high", "low", "close", "volume"]

BRANCH_NAMES = ["m1", "m3", "m5", "m3_std", "m5_std"]
BRANCH_LENGTHS = {"m1": M1_LEN, "m3": M3_LEN, "m5": M5_LEN, "m3_std": M3STD_LEN, "m5_std": M5STD_LEN}
# 每個分支的channel數不均一（m1多了4個channel：atr5/ma10/ret_vs_idx/
# idx_ret_since_open；m3/m5/m3_std/m5_std都是原始5個OHLCV），model.py的
# _Branch建構時靠這個決定每個分支的Conv1d輸入channel數。
#
# 2026-07-24：原本m3/m5/m3_std/m5_std都多帶vol_ratio/direction兩個爆量方向
# channel，後來拿掉了——這個特徵(a)從沒證據證明真的有幫助（加到m3之後
# AUC/precision反而略降），(b)其5根獨立K棒基準線的暖機期需求，疊加這次
# 縮小M3STD_LEN/M5STD_LEN之後，把候選硬拖到9:35才出現（而不是視窗長度本該
# 決定的9:10），是一個跟這個特徵綁死的bug。「平」的過濾改用atr5（已證明
# 有用，見experiments/atr5_flat_filter_check.py），跟爆量方向的用途本來就
# 重疊，直接拿掉爆量特徵，不用另外修fillna/baseline_bars。
BRANCH_CHANNELS = {"m1": len(_M1_COLS), "m3": 5, "m5": 5, "m3_std": 5, "m5_std": 5}
N_CLASSES = 3  # 跌/平/漲（2026-07-24從2類triple barrier改成3類，見 _barrier_label_numba）

# 候選當下這一步的atr5值，額外存進meta（不只留在m1分支tensor裡）——
# train.py要用這個做「只篩平」的atr門檻過濾，見_build_windows_for_keys()。
_ATR5_META_IDX = _M1_COLS.index("atr5")


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
    """3類 triple barrier（2026-07-24從2類改成3類）：
      0 = 跌（SL先到）
      1 = 平（完整 hold_bars 根都沒碰到 +tp_pct/-sl_pct 任一邊界）
      2 = 漲（TP先到）
    「平」不是門檻式定義（例如±0.5%），而是「兩個barrier都沒被觸發」這個
    單純的事件——只要完整掃完hold_bars根都沒碰到任一邊，就算平，不看最後
    一根的相對位置（舊版2類的做法）。"""
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
            labels[i] = 2.0  # 漲
        elif hit == 0:
            labels[i] = 0.0  # 跌
        elif last_j - i == hold_bars:
            labels[i] = 1.0  # 平：完整掃完都沒碰到任一邊界
        # else：當日剩餘bar數不足hold_bars且都沒碰到，維持NaN（樣本不完整，不算數）
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


def _month_key(day_date) -> str:
    return f"{day_date.year}_{day_date.month:02d}"


def _month_source_mtime(month: str) -> float:
    """單一月份對應的來源檔案（db/m1、db/m3、db/m5、db/m3_std、db/m5_std 各自的
    {month}.parquet）裡最新的mtime，找不到就跳過。"""
    mtimes = []
    for d in _SOURCE_DIRS:
        p = _ROOT / d / f"{month}.parquet"
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def _shard_is_fresh(month: str) -> bool:
    """單一月份shard的新鮮度檢查——只比對「這個月份自己對應」的來源檔案
    mtime，不是整包 db/m1 等資料夾裡最新的檔案。

    2026-07-22 修過的 bug：原本用「整包資料夾裡最新的一個檔案」當基準，
    導致只更新了某一個月的來源資料（例如 update_m1 只重抓最近一兩個月），
    卻連其他完全沒變的月份 shard 也被誤判過期、全部重建，完全沒享受到
    按月分片的好處——使用者實測半年資料時親自碰到這個問題（改了db/m1
    的06/07月資料，01~05月shard也被牽連重建）。"""
    meta_path = _CACHE_DIR / f"meta_{month}.parquet"
    if not meta_path.exists():
        return False
    return meta_path.stat().st_mtime >= _month_source_mtime(month)


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


def _degroup(s: pd.Series, index: pd.Index) -> pd.Series:
    """把 groupby(...).rolling(...) 產生的多層 index 結果攤平回跟原始 df 對齊的
    Series（比照 rally features.py 的同名 helper）：用原生 GroupBy.rolling 取代
    transform(lambda ...)，分組數一多（本專案動輒數萬個 stock×day 分組）效能
    差非常多，前者走 pandas 內建向量化路徑。"""
    n_key_levels = s.index.nlevels - 1
    if n_key_levels:
        s = s.reset_index(level=list(range(n_key_levels)), drop=True)
    return s.reindex(index)


def _add_atr_ma(df: pd.DataFrame, g_day, day_open: pd.Series) -> None:
    """1分鐘ATR(5)/MA(10)/MA(5)/MA(3)，2026-07-24新增，只加m1分支。用原始
    （未正規化）high/low/close算，除以day_open正規化，跟其他OHLC欄位的正規化
    方式一致。MA5/MA3（2026-07-24加）比MA10短，想給CNN更接近即時的價格參考
    點——使用者觀察到「會不會動」（atr5過濾後有改善）跟「往哪動」（precision
    看似變好但方向對錯機率接近或輸給丟銅板）是不同難度的問題，短天期MA
    可能有助於後者（價格相對短均線的位置，是常見的動能/趨勢判斷依據）。

    ATR/MA本身需要暖機期（分別5根/10根K棒）：視窗縮小成M1_LEN=10之後，當天
    最早幾筆候選的視窗前段會落在暖機期內、算出NaN——一律fillna(0)，不能留
    NaN，否則_build_windows_for_keys()的finite檢查會把整個候選直接丟掉，
    等於白白排除掉這次縮小視窗最想多涵蓋到的早盤候選。

    只做ATR、沒做ADX——orb已經踩過這個坑：ADX雙重平滑需要~28根K棒才穩定，
    在這麼短的候選視窗裡缺值率會非常高（見project_orb_strategy記憶），
    改短週期(ADX5)也解決不了雙重平滑本身要的天數，直接沿用「拿掉」的結論。
    """
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
    """大盤相對強弱特徵（2026-07-24新增，比照strategy/mkt/features.py的
    add_ret_vs_idx()，同樣的定義搬過來）：

      ret_vs_idx = 個股從今日開盤累積到現在的報酬率 - 0050同一分鐘的累積報酬率
      idx_ret_since_open = 0050自己從今日開盤累積到現在的報酬率（廣播給每一列）

    >0 代表這支股票從開盤到現在漲得比大盤多（相對強勢）；<0代表跑輸大盤。
    mkt實測這是目前四個策略裡唯一有實質判斷力的特徵，cnn這五路分支目前都是
    單一股票自己的量價序列、完全沒有大盤相對資訊，這是最可能補上的訊號。

    用merge（不是groupby/rolling），所以跟_normalize_ohlcv不同，回傳新的df
    而不是就地mutate——on="date"（不含stock_id）讓0050
    當下這一分鐘的值廣播給同一分鐘的所有股票列，how="left"保留df原本的
    列順序（idx這邊每個date只會有唯一一列，不會產生重複配對）。"""
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

    # _add_market_relative() 用 merge（不是 groupby/rolling），會回傳新的
    # dataframe——放在 g_day/day_open 算出來之前，避免舊的 g_day/day_open
    # 綁在merge前的wide物件上、事後跟merge後的新wide做index對齊時產生風險。
    day_open_pre = wide.groupby(["stock_id", "day_date"], group_keys=False)["open"].transform("first").replace(0, np.nan)
    wide = _add_market_relative(wide, day_open_pre, IDX_SYMBOL)

    g_day = wide.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)

    _add_atr_ma(wide, g_day, day_open)

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


def _build_windows_for_keys(wide_groups, keys, m3_std_groups, m5_std_groups) -> tuple[dict, pd.DataFrame]:
    """回傳 ({branch_name: np.ndarray shape (N, channels, branch_len)}, meta df[stock_id/date/target])，
    channels 依分支不同（見 BRANCH_CHANNELS），
    只處理 keys 這個子集的 (stock_id, day_date) 分組——呼叫端一次只傳一個月份的
    keys，藉此把記憶體峰值限制在「一個月的視窗量」（見檔頭「為什麼按月分片存」
    的說明），不是這支函式自己知道月份的概念。

    效能備註：外層對 keys 跑一次 Python 迴圈，組內全部用 numpy 向量化操作
    （sliding_window_view / fancy-index gather），沒有逐候選的 inner python 迴圈。
    """
    offsets3 = np.arange(-(M3STD_LEN - 1), 1)
    offsets5 = np.arange(-(M5STD_LEN - 1), 1)

    m1m3m5_list, m3std_list, m5std_list = [], [], []
    meta_stock, meta_date, meta_target, meta_atr5 = [], [], [], []

    for key in keys:
        stock_id, day_date = key
        day_df = wide_groups.get_group(key)
        day_len = len(day_df)
        if day_len < M1_LEN:
            continue

        if key not in m3_std_groups.groups or key not in m5_std_groups.groups:
            continue
        m3std_day = m3_std_groups.get_group(key)
        m5std_day = m5_std_groups.get_group(key)

        arr = day_df[_WIDE_FEATURE_COLS].to_numpy(dtype=np.float32)
        win = sliding_window_view(arr, M1_LEN, axis=0)  # (day_len-M1_LEN+1, len(_WIDE_FEATURE_COLS), M1_LEN)
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
        w3 = m3std_arr[rows3].transpose(0, 2, 1)  # (n_valid, len(_STD_FEATURE_COLS), M3STD_LEN)
        w5 = m5std_arr[rows5].transpose(0, 2, 1)  # (n_valid, len(_STD_FEATURE_COLS), M5STD_LEN)

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
        meta_atr5.append(win[final_idx][:, _ATR5_META_IDX, -1])

    if not meta_target:
        return {}, pd.DataFrame(columns=["stock_id", "date", "target", "atr5"])

    combined = np.concatenate(m1m3m5_list, axis=0)  # (N, len(_WIDE_FEATURE_COLS), M1_LEN)
    # 用len()動態切分，不要寫死數字——m1/m3/m5各自的channel數以後還可能再變
    # （已經改過一次：m1從5加到9），寫死的slice會悄悄切錯、不會報錯。
    n1, n3 = len(_M1_COLS), len(_M3_COLS)
    branches = {
        "m1": combined[:, 0:n1, :],
        "m3": combined[:, n1 : n1 + n3, :],
        "m5": combined[:, n1 + n3 :, :],
        "m3_std": np.concatenate(m3std_list, axis=0),
        "m5_std": np.concatenate(m5std_list, axis=0),
    }
    meta = pd.DataFrame({
        "stock_id": np.concatenate(meta_stock),
        "date": np.concatenate(meta_date),
        "target": np.concatenate(meta_target).astype(np.int64),
        "atr5": np.concatenate(meta_atr5).astype(np.float32),
    })
    return branches, meta


# ═══════════════════════════════════════════════════════════════════════════════
# 對外入口（按月分片 build + 讀取）
# ═══════════════════════════════════════════════════════════════════════════════


def _month_bound(date_str: str) -> str:
    """ "2026-07-15" → "2026_07"，跟 train.py 同名 helper 一致（各自維護一份，
    這裡不 import train.py，避免策略內部模組互相依賴）。"""
    return f"{date_str[:4]}_{date_str[5:7]}"


def _source_months(start_date: str, end_date: str) -> list[str]:
    """從 db/m1 的檔名（依月分檔，{yyyy_mm}.parquet）列出這次請求範圍內有哪些
    月份——只讀檔名/mtime，不載入任何資料內容。2026-07-24新增，_shard_is_fresh()
    要用的月份清單原本是先把全部原始資料整包load_m1()/load_m3()/...載進來、
    merge、算完wide frame之後才反推出來的，導致「shard已是最新，略過重算」
    這個檢查完全沒省到「讀全部原始資料」這筆固定成本（讀2787支股票×N個月的
    1分K全部資料+merge+排序），這才是evaluate/evaluate_hours/confidence每次
    呼叫都很慢的真正原因，跟評估的天數/時段大小無關——shard新鮮度用檔名+mtime
    就能查清楚，不需要先把資料內容都讀進來才能知道「有哪些月份」。"""
    m1_dir = _ROOT / "db/m1"
    months = sorted(p.stem for p in m1_dir.glob("*.parquet"))
    if start_date:
        months = [m for m in months if m >= _month_bound(start_date)]
    if end_date:
        months = [m for m in months if m <= _month_bound(end_date)]
    return months


def build_dataset(start_date: str = "", end_date: str = "", force_rebuild: bool = False) -> list[str]:
    """組裝五路 tensor + label，依月份分片(shard)寫入 cache/cnn/ 底下
    （{branch}_{yyyy_mm}.npy + meta_{yyyy_mm}.parquet）。

    2026-07-24：先用 _source_months()/_shard_is_fresh()（只讀檔名+mtime）判斷
    這次範圍內「哪些月份真的需要重建」，如果全部都新鮮，直接跳過、完全不觸碰
    _load_wide_frame()/_load_std_frame()——只有真的有月份過期/缺失時，才付
    「載入全部原始資料」這筆固定成本（見_source_months()說明，這是之前
    evaluate/evaluate_hours每次呼叫都很慢的根因）。

    原始資料（m1/m3/m5/m3_std/m5_std）一旦要載入就是整包load（這步驟本身的
    記憶體成本是固定的，不隨 start_date/end_date 縮放——data/query.py 的
    load_*() 本來就是整包載入再篩選），但視窗組裝完之後逐月立刻存檔、釋放
    暫存陣列，讓記憶體峰值只跟「一個月的視窗量」有關，可以放心跑到半年甚至
    全量。

    ⚠️ groupby 之後務必維持 lazy 的 GroupBy 物件、用 .get_group(key) 隨用隨取，
    不要用 {k: v for k, v in df.groupby(...)} 這種寫法把每個 (stock_id, day_date)
    分組都具現化成獨立 DataFrame 存進 dict——半年資料分組數上看28萬組，每個小
    DataFrame本身的pandas物件開銷（跟資料量無關，是index/block manager等固定
    overhead）疊起來就能吃到幾十GB，這是2026-07-22實測半年資料時把系統記憶體
    衝到36GB（機器只有24GB）的真正原因，不是視窗tensor本身太大。

    回傳這次涵蓋到、且成功建好（或已是最新略過）的月份 key 清單（"yyyy_mm"）。
    """
    months = _source_months(start_date, end_date)
    if not months:
        return []

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stale_months = {m for m in months if force_rebuild or not _shard_is_fresh(m)}

    if not stale_months:
        for month in months:
            print(f"  {month} shard 已是最新，略過重算")
        return months

    wide = _load_wide_frame(start_date, end_date)
    m3_std = _load_std_frame(load_m3_std, start_date, end_date)
    m5_std = _load_std_frame(load_m5_std, start_date, end_date)

    wide_groups = wide.groupby(["stock_id", "day_date"], sort=False)
    m3_std_groups = m3_std.groupby(["stock_id", "day_date"], sort=False)
    m5_std_groups = m5_std.groupby(["stock_id", "day_date"], sort=False)

    keys_by_month = defaultdict(list)
    for key in wide_groups.groups.keys():
        _, day_date = key
        keys_by_month[_month_key(day_date)].append(key)

    for month in months:
        if month not in stale_months:
            print(f"  {month} shard 已是最新，略過重算")
            continue

        keys = keys_by_month.get(month, [])
        if not keys:
            print(f"  {month} 沒有通過篩選的有效樣本，略過存檔")
            continue
        print(f"  組裝 {month}（{len(keys)} 個 stock×day 分組）...")
        branches, meta = _build_windows_for_keys(wide_groups, keys, m3_std_groups, m5_std_groups)
        if len(meta) == 0:
            print(f"  {month} 沒有通過篩選的有效樣本，略過存檔")
            continue

        for name, arr in branches.items():
            np.save(_CACHE_DIR / f"{name}_{month}.npy", arr)
        meta.to_parquet(_CACHE_DIR / f"meta_{month}.parquet")
        print(f"  {month} 完成，{len(meta):,} 筆")

    return months


def available_months() -> list[str]:
    """回傳 cache/cnn/ 底下已經建好（存在 meta_{yyyy_mm}.parquet）的月份清單，由小到大排序。"""
    if not _CACHE_DIR.exists():
        return []
    return sorted(p.stem.replace("meta_", "") for p in _CACHE_DIR.glob("meta_*.parquet"))


def load_shard_meta(month: str) -> pd.DataFrame:
    """讀單一月份shard的meta（stock_id/date/target，檔案很小，直接整包讀沒關係）。"""
    return pd.read_parquet(_CACHE_DIR / f"meta_{month}.parquet")


def load_shard_branch(month: str, branch: str, mmap: bool = True) -> np.ndarray:
    """讀單一月份shard的單一分支tensor。mmap=True（預設）用
    np.load(mmap_mode='r')，不整包讀進RAM，訓練時只有實際用到的那幾筆才會被
    作業系統從磁碟page進記憶體——大範圍資料train階段才不會又OOM一次
    （見 train.py 的 ShardedMultiScaleDataset）。"""
    path = _CACHE_DIR / f"{branch}_{month}.npy"
    return np.load(path, mmap_mode="r" if mmap else None)
