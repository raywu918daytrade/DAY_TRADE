"""
vwap_ml 策略 —— VWAP±2σ 通道（m1/m3/m5 三個時間框，各自獨立計算）。

核心想法：在 m1/m3/m5 三個時間框上分別計算「今日累積 VWAP」與「累積至今
的偏離標準差（expanding std）」，超過 config.STD_MULT 個標準差視為候選
訊號——三個時間框任一觸發即算候選（OR，不是要求三者同時滿足），2026-07-26
討論：要求同時滿足會讓訊號太少，改成候選產生用OR、分類特徵留三個時間框
的z-score讓 LightGBM 自己學組合規律。

觸發後由 make_vwap_labels() 產生三分類標籤（0=回歸VWAP/1=無訊號（盤整）/
2=延續突破），配合觸發時價格在 VWAP 之上還是之下（trigger_side），在
predict.py 轉成多/空方向：
    上軌觸發(z>+STD_MULT)＋回歸 → 空單　　上軌觸發＋延續 → 多單
    下軌觸發(z<-STD_MULT)＋回歸 → 多單　　下軌觸發＋延續 → 空單

⚠️ db/m1、db/m1_live 只有 open/high/low/close/volume，沒有成交金額欄位，
這裡的 VWAP 是 close×volume 累積近似（跟 strategy/orb/features.py、
strategy/rally/features.py 的 vwap_dev 用同一種近似方式，是專案裡一貫的
做法，不是 vwap_ml 獨有的妥協）。

第一版先做 VWAP z-score + 少數量能/報酬率特徵，不一次寫完整套 pipeline
（比照 strategy/mkt/features.py 開頭「先精簡、慢慢加」的哲學）。
"""

import sys
from datetime import time as dtime
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.resample import compute_m3, compute_m5
from strategy.vwap_ml.config import (
    ATR5_FILTER_THRESHOLD,
    HOLD_BARS,
    SESSION_END,
    SESSION_START,
    STD_MULT,
    TIMEFRAME_MINUTES,
)


def _degroup(s: pd.Series, index: pd.Index) -> pd.Series:
    """把 groupby(...).expanding()/rolling() 產生的多層 index 結果攤平回
    與原始 df 對齊的 Series（複製自 data/resample.py 的同名函式——沒有
    共用的 helper 模組，orb/rally/mkt 各自也都各自複製一份，見
    data/resample.py::_degroup() 的說明）。"""
    n_key_levels = s.index.nlevels - 1
    if n_key_levels:
        s = s.reset_index(level=list(range(n_key_levels)), drop=True)
    return s.reindex(index)


def _add_vwap_z(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """算單一時間框（m1/m3/m5）的累積 VWAP 偏離 z-score，加欄位
    {prefix}_vwap_z（累積 VWAP 本身不留在輸出裡，呼叫端只需要 z-score）。

    標準差用「累積至今」的 expanding std，不是全天 std——全天 std 會用到
    未來才知道的資訊（lookahead），expanding 只用當下已經發生的 bar，跟
    即時推論當下能拿到的資訊一致。

    df 需已有 stock_id/day_date/date/close/volume 欄位，且已依
    stock_id/date 排序（m3/m5 呼叫前需先用 data/resample.py 的
    compute_m3()/compute_m5() 轉成對應時間框的 rolling OHLCV 再傳進來，
    用法跟 m1 原生欄位一致）。
    """
    df = df.copy()
    g_day = df.groupby(["stock_id", "day_date"], group_keys=False)
    cum_vol = g_day["volume"].transform("cumsum")
    df["_pv"] = df["close"] * df["volume"]
    cum_pv = df.groupby(["stock_id", "day_date"])["_pv"].transform("cumsum")
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    df["_dev"] = df["close"] - vwap
    g_dev = df.groupby(["stock_id", "day_date"], group_keys=False)
    dev_std = _degroup(g_dev["_dev"].expanding(min_periods=5).std(), df.index)
    df[f"{prefix}_vwap_z"] = df["_dev"] / dev_std.replace(0, np.nan)
    return df.drop(columns=["_pv", "_dev"])


def add_vwap_features(m1: pd.DataFrame, std_mult: float = STD_MULT) -> pd.DataFrame:
    """
    加 m1_vwap_z/m3_vwap_z/m5_vwap_z 三個時間框的 VWAP 偏離 z-score、各自
    的斜率（跟3根前比，diff(3)——正代表偏離擴大中，負代表正在往VWAP收斂），
    以及候選觸發相關欄位：
        is_candidate   任一時間框 |z| >= std_mult
        trigger_tf     觸發時間框（"m1"/"m3"/"m5"），多個同時觸發時取
                       |z| 最大的那個（訊號最強烈的時間框），非候選列為 None
        trigger_z      觸發時間框當下的 z 值（帶正負號），非候選列為 NaN
        trigger_side   "upper"（z>0，價格在VWAP之上）/"lower"（z<0，價格
                       在VWAP之下），非候選列為 None

    std_mult（2026-07-26 討論）：候選觸發門檻不寫死在函式內部，開放參數
    ——2σ只是起始猜測，之後要比照 strategy/mkt 驗證 ATR5_FILTER_THRESHOLD
    （p90/p95/p97/p99 walk-forward比較）的方式，實際跑 1.0/1.5/2.0 等不同
    候選門檻比較樣本量、標籤分布、precision，才能定案；預設值仍是
    config.STD_MULT，正式 train/predict 都不傳這個參數就會用預設值。

    m1 需已有 stock_id/date/day_date/open/high/low/close/volume 欄位，
    date 為 datetime，已依 stock_id/date 排序。
    """
    df = _add_vwap_z(m1, "m1")

    m3 = compute_m3(m1)
    m3["day_date"] = m3["date"].dt.date
    m3 = _add_vwap_z(m3, "m3")[["stock_id", "date", "m3_vwap_z"]]
    df = df.merge(m3, on=["stock_id", "date"], how="left")

    m5 = compute_m5(m1)
    m5["day_date"] = m5["date"].dt.date
    m5 = _add_vwap_z(m5, "m5")[["stock_id", "date", "m5_vwap_z"]]
    df = df.merge(m5, on=["stock_id", "date"], how="left")

    g_day = df.groupby(["stock_id", "day_date"], group_keys=False)
    df["m1_vwap_z_slope"] = g_day["m1_vwap_z"].diff(3)
    df["m3_vwap_z_slope"] = g_day["m3_vwap_z"].diff(3)
    df["m5_vwap_z_slope"] = g_day["m5_vwap_z"].diff(3)

    z_cols = ["m1_vwap_z", "m3_vwap_z", "m5_vwap_z"]
    z_vals = df[z_cols].to_numpy()
    abs_vals = np.abs(z_vals)
    filled = np.where(np.isnan(abs_vals), -np.inf, abs_vals)
    max_abs = np.max(filled, axis=1)
    trigger_idx = np.argmax(filled, axis=1)
    has_any = np.isfinite(max_abs)

    is_candidate = has_any & (max_abs >= std_mult)
    tf_names = np.array(["m1", "m3", "m5"], dtype=object)
    trigger_tf = np.where(is_candidate, tf_names[trigger_idx], None)
    trigger_z = np.where(is_candidate, z_vals[np.arange(len(df)), trigger_idx], np.nan)
    trigger_side = np.where(trigger_z > 0, "upper", np.where(trigger_z < 0, "lower", None))

    df["is_candidate"] = is_candidate
    df["trigger_tf"] = trigger_tf
    df["trigger_z"] = trigger_z
    df["trigger_side"] = trigger_side
    return df


def add_bar_features(m1: pd.DataFrame) -> pd.DataFrame:
    """
    當根K棒的量能/報酬率基礎特徵。第一版先只加這幾個，之後依驗證結果
    逐步加（比照 strategy/mkt/features.py 開頭「先精簡、慢慢加」的哲學）。

    m1 需已有 stock_id/day_date/open/close/volume 欄位。
    """
    m1 = m1.copy()
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    m1["vol_ratio_prev"] = m1["volume"] / g_day["volume"].shift(1).replace(0, np.nan)
    m1["ret_1m"] = g_day["close"].pct_change(1, fill_method=None)
    m1["body_pct"] = (m1["close"] - m1["open"]) / day_open
    return m1


def add_atr5(m1: pd.DataFrame) -> pd.DataFrame:
    """
    1分鐘K ATR(5)相對波動（True Range 5根滾動平均 / 當日開盤價），算法跟
    strategy/mkt/features.py::add_atr5() 完全一樣。

    2026-07-26 討論：加這個當「候選過濾」用——vwap_ml 的 z-score 是用該
    股票自己「累積至今的偏離標準差」當分母，本來就沒什麼波動的股票（分母
    趨近於0）連小幅價格雜訊都會被除出誇張的z值，產生假的候選訊號。跟
    ATR5_FILTER_THRESHOLD 搭配，篩掉這種「其實沒什麼波動、z只是分母太小
    造成」的假候選，比照 mkt 篩掉ATR5平盤樣本的做法。

    ⚠️ 只拿來當過濾用，**不進FEATURES**——這種不帶方向的純波動幅度指標
    當模型輸入時會被樹拿去投機分裂，稀釋掉真正有方向性的訊號（見
    strategy/mkt/features.py::add_atr5()/FEATURES 段落的說明，同樣的教訓
    照搬過來，不用重新踩一次）。

    m1 需已有 stock_id/day_date/open/high/low/close 欄位。
    """
    m1 = m1.copy()
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    prev_close = g_day["close"].shift(1).fillna(m1["open"])
    m1["_tr5"] = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - prev_close).abs()),
        (m1["low"] - prev_close).abs(),
    )
    g_tr = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["atr5"] = g_tr["_tr5"].transform(lambda x: x.rolling(5, min_periods=5).mean()) / day_open
    return m1.drop(columns=["_tr5"])


# ═══════════════════════════════════════════════════════════════════════════════
# 三分類標籤（0=回歸VWAP／1=無訊號（盤整）／2=延續突破）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 2026-07-26 討論：視窗大小是「觸發的那個時間框自己的後 N 根」，不是跨
# 時間框統一秒數——m1 觸發看後 HOLD_BARS 根 m1（= HOLD_BARS 分鐘），m5
# 觸發看後 HOLD_BARS 根 m5（= HOLD_BARS*5 分鐘），用該時間框自己的 z-score
# 序列（例如 m5_vwap_z）判斷未來走勢，而不是統一都看 m1 的 z。
#
# 兩個 barrier 分別是「z 回到 0（回歸完成）」跟「z 進一步擴大超過
# CONTINUATION_STD_MULT（延續）」，不是像 orb/rally/mkt 那樣固定百分比
# 報酬——這裡衡量的是「價格相對 VWAP 的位置」，不是絕對報酬率。


def _label_group_vwap(z: np.ndarray, horizon_bars: int, continuation_mult: float, std_mult: float) -> np.ndarray:
    """單一 stock/day 的 z-score 序列，逐 bar 往前看最多 horizon_bars 根，
    判斷先碰到哪個barrier，或 horizon_bars 內都沒碰到（無訊號）。非候選
    的列（|z| < std_mult）直接跳過不計算，省時間（後面會被篩掉）。"""
    n = len(z)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isfinite(z[i]) or abs(z[i]) < std_mult:
            continue
        side = 1.0 if z[i] > 0 else -1.0
        future = z[i + 1 : i + 1 + horizon_bars] * side
        future = future[np.isfinite(future)]
        if len(future) == 0:
            continue
        revert_hit = np.argmax(future <= 0) if (future <= 0).any() else len(future)
        cont_hit = np.argmax(future >= continuation_mult) if (future >= continuation_mult).any() else len(future)
        if revert_hit < cont_hit:
            labels[i] = 0  # 回歸
        elif cont_hit < revert_hit:
            labels[i] = 2  # 延續
        elif len(future) == horizon_bars:
            labels[i] = 1  # 無訊號：兩個barrier都沒碰到，且視窗完整
        # else: 太靠近資料尾端，視窗不足，保留 NaN
    return labels


def _make_labels_for_tf(df: pd.DataFrame, tf: str, continuation_mult: float, std_mult: float) -> pd.Series:
    horizon_bars = HOLD_BARS * TIMEFRAME_MINUTES[tf]
    zcol = f"{tf}_vwap_z"

    def _apply(g):
        return pd.Series(
            _label_group_vwap(g[zcol].to_numpy(), horizon_bars, continuation_mult, std_mult), index=g.index
        )

    return df.groupby(["stock_id", "day_date"], group_keys=False).apply(_apply, include_groups=False)


def make_vwap_labels(df: pd.DataFrame, std_mult: float = STD_MULT, continuation_mult: float | None = None) -> pd.Series:
    """
    對每個候選列（is_candidate=True，用 std_mult 判定），依 trigger_tf 用
    對應時間框自己的 z-score 序列算 label（0=回歸/1=無訊號/2=延續），非
    候選列回傳 NaN。

    std_mult/continuation_mult：見 add_vwap_features() 的說明，兩者要跟
    產生 is_candidate/trigger_tf 時用的 std_mult 一致，否則「候選」跟
    「label計算時判定candidate」的門檻會對不起來。continuation_mult 留
    None 時＝ std_mult + 1.0（config.CONTINUATION_STD_MULT 預設值的相對
    關係），實驗不同 std_mult 時延續門檻跟著等比例調整，不用另外手動算。

    df 需已有 stock_id/day_date/is_candidate/trigger_tf/m1_vwap_z/
    m3_vwap_z/m5_vwap_z 欄位（先呼叫 add_vwap_features(std_mult=std_mult)）。
    """
    if continuation_mult is None:
        continuation_mult = std_mult + 1.0
    label = pd.Series(np.nan, index=df.index)
    for tf in ("m1", "m3", "m5"):
        s = _make_labels_for_tf(df, tf, continuation_mult, std_mult)
        mask = df["trigger_tf"] == tf
        label[mask] = s[mask]
    return label


def _filter_session(df: pd.DataFrame, start=SESSION_START, end=SESSION_END) -> pd.DataFrame:
    """只保留 SESSION_START~SESSION_END 這段時間的列（entry 限定在這個
    窗口內，但 label 已經在這之前算完，計算時可以看到窗口外的未來資料，
    比照 strategy/mkt/train.py::_prepare_data() 的時段過濾順序——先算完
    特徵/標籤，最後才篩時段）。"""
    t = df["date"].dt.time
    return df[(t >= dtime(*start)) & (t < dtime(*end))]


def add_trigger_tf_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """觸發時間框 one-hot（trigger_is_m1/m3/m5）——model 要知道是哪個時間框
    觸發，因為三個時間框的 label 視窗長度不同，隱含的意義也不同。"""
    df = df.copy()
    df["trigger_is_m1"] = (df["trigger_tf"] == "m1").astype(int)
    df["trigger_is_m3"] = (df["trigger_tf"] == "m3").astype(int)
    df["trigger_is_m5"] = (df["trigger_tf"] == "m5").astype(int)
    return df


def make_features(
    m1: pd.DataFrame,
    std_mult: float = STD_MULT,
    atr5_threshold: float = ATR5_FILTER_THRESHOLD,
) -> pd.DataFrame:
    """
    給定全市場的 1 分K（load_m1() 或 load_m1_live() 的輸出），跑完整的
    vwap_ml pipeline：VWAP z-score → 候選觸發 → label → ATR5平盤過濾 →
    只保留候選列 → 時段過濾。

    std_mult：候選觸發門檻，見 add_vwap_features()/make_vwap_labels() 的
    說明，實驗不同門檻時傳這裡就好，train.py::_prepare_data() 會轉傳。
    atr5_threshold：ATR5平盤過濾的絕對門檻，見 add_atr5() 的說明。

    m1 需已有 stock_id/date/open/high/low/close/volume 欄位。
    """
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    df = add_vwap_features(m1, std_mult=std_mult)
    df = add_bar_features(df)
    df = add_atr5(df)
    df["target"] = make_vwap_labels(df, std_mult=std_mult)
    df = add_trigger_tf_dummies(df)

    # 只留真正觸發過 band 的候選事件當訓練樣本，不是每分鐘都當一筆（比照
    # strategy/orb/features.py 2026-07-11 之後「只留真正突破事件」的做法）。
    df = df[df["is_candidate"]]

    # ATR5平盤過濾（絕對門檻，三個class都篩，2026-07-26討論）：濾掉
    #「本來就沒什麼波動、z只是分母（累積偏離標準差）太小才被撐大」的假
    # 候選，見 add_atr5() 的說明。跟 train/predict_live 三邊用同一個門檻。
    df = df[df["atr5"] >= atr5_threshold]

    df = _filter_session(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 模型實際用哪些特徵（train.py 從這裡匯入，不要在 train.py 裡另外定義一份）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 2026-07-26：第一版先放 VWAP z-score/斜率（核心訊號）＋觸發時間框one-hot
# （model要知道是哪個時間框觸發，因為三個時間框的label視窗長度不同，隱含
# 的意義也不同）＋少數量能/報酬率特徵，之後依 feature_importance() 驗證
# 結果逐步加減，不一次補滿。
FEATURES = [
    "m1_vwap_z",
    "m3_vwap_z",
    "m5_vwap_z",
    "m1_vwap_z_slope",
    "m3_vwap_z_slope",
    "m5_vwap_z_slope",
    "vol_ratio_prev",
    "ret_1m",
    "body_pct",
    "trigger_is_m1",
    "trigger_is_m3",
    "trigger_is_m5",
]
