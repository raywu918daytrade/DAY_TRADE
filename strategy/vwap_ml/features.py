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
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.resample import compute_m3, compute_m5
from strategy.vwap_ml.config import (
    ATR5_FILTER_THRESHOLD,
    IDX_SYMBOL,
    LABEL_HORIZON_MINUTES,
    SESSION_END,
    SESSION_START,
    STD_MULT,
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
    g_day = df.groupby(["stock_id", "day_date"], group_keys=False, observed=True)
    cum_vol = g_day["volume"].transform("cumsum")
    df["_pv"] = df["close"] * df["volume"]
    cum_pv = df.groupby(["stock_id", "day_date"], observed=True)["_pv"].transform("cumsum")
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    df["_dev"] = df["close"] - vwap
    g_dev = df.groupby(["stock_id", "day_date"], group_keys=False, observed=True)
    dev_std = _degroup(g_dev["_dev"].expanding(min_periods=5).std(), df.index)
    df[f"{prefix}_vwap_z"] = df["_dev"] / dev_std.replace(0, np.nan)
    return df.drop(columns=["_pv", "_dev"])


def _trim_to_needed_window(df: pd.DataFrame) -> pd.DataFrame:
    """只保留到 SESSION_END + LABEL_HORIZON_MINUTES（例如 SESSION_END=
    10:00、視窗30分鐘 → 保留到10:30）這段時間的列，不動起點（VWAP要從
    市場開盤就開始累積，起點不能砍）。

    2026-07-26 討論：發現這是全市場全歷史訓練時記憶體爆掉的主因之一——
    候選只會落在 SESSION_START~SESSION_END（9:10~10:00）這段，且 label
    最多只需要看到 SESSION_END 之後再 LABEL_HORIZON_MINUTES 分鐘的未來
    資料，中午/下午盤的資料整個 pipeline 完全用不到，但先前是等最後
    _filter_session() 才裁掉，前面 VWAP z-score/m3/m5合併/量能特徵/ATR5/
    label 全部都對整天（9:00~13:30，約270分鐘）跑過一次，多算了好幾倍
    用不到的資料。改成載入後立刻裁掉尾端，後面所有計算的資料量直接減少。
    裁掉尾端不影響任何數值——VWAP/z-score都是只看「當下與更早」的累積/
    expanding計算，跟後面還有沒有資料無關。

    df 需已有 date 欄位（datetime）。
    """
    cutoff = (datetime.combine(datetime.min, dtime(*SESSION_END)) + timedelta(minutes=LABEL_HORIZON_MINUTES)).time()
    return df[df["date"].dt.time <= cutoff]


def add_ret_vs_idx(m1: pd.DataFrame, idx_symbol: str = IDX_SYMBOL) -> pd.DataFrame:
    """
    2026-07-27 新增：算「個股從今日開盤累積到現在的報酬率」減去「0050
    從今日開盤累積到現在的報酬率」，逐分鐘更新，回傳加了 ret_vs_idx/idx_ret_since_open
    欄位的 m1（複本，不動傳進來的原始 df）。

    比照 strategy/mkt/features.py::add_ret_vs_idx() 的實作——個股跑輸大盤
    （ret_vs_idx<0）時，回歸VWAP的機率是否更高？訓練時把這個資訊交給模型
    自己判斷。

    >0 代表這支股票從開盤到現在，漲得比大盤多（相對強勢）；
    <0 代表跑輸大盤（相對弱勢）。

    idx_ret_since_open：0050 自己從開盤累積到現在的報酬率，提供大盤自身
    的方向資訊（同樣 ret_vs_idx=-2%，大盤漲2%個股沒漲 vs 大盤跌1%個股跌3%
    意義完全不同）。

    ⚠️ 一定要在 user 篩選 day_trade_stocks / 排除 0050 之前呼叫（0050 被
    篩掉後就沒有基準可以算了），predict_live() 比照 mkt 的做法——先對完整
    m1_live 算特徵，才排除 0050 跟套用 day_trade_stocks 篩選。

    m1 需已有 stock_id/date/day_date/open/close 欄位，date 需為 datetime。
    """
    m1 = m1.copy()
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False, observed=True)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    ret_since_open = m1["close"] / day_open - 1

    idx = m1[m1["stock_id"] == idx_symbol][["date", "close", "open"]].drop_duplicates("date").copy()
    idx_g_day = idx.groupby(idx["date"].dt.date, group_keys=False)
    idx_day_open = idx_g_day["open"].transform("first").replace(0, np.nan)
    idx["idx_ret_since_open"] = idx["close"] / idx_day_open - 1
    idx = idx[["date", "idx_ret_since_open"]]

    m1 = m1.merge(idx, on="date", how="left")
    m1["ret_vs_idx"] = ret_since_open - m1["idx_ret_since_open"]
    return m1


def add_market_vwap_features(
    df: pd.DataFrame,
    m3: pd.DataFrame | None = None,
    m5: pd.DataFrame | None = None,
    idx_symbol: str = IDX_SYMBOL,
) -> pd.DataFrame:
    """
    2026-07-28 新增：大盤（0050）的 VWAP 相關特徵，共 4 個欄位：

    1. market_z_score_m5
       大盤 M5 VWAP 的偏離 z-score。
       當個股觸及 +2σ 且大盤自己也嚴重過熱時，是個股+大盤共振回歸空單點。
       直接從 df 裡的 0050 m5_vwap_z 廣播，不額外算。

    2. market_vwap_alignment_score
       大盤 M1/M3/M5 三個時間框的 VWAP 排列方向：
         +1 = M1 VWAP > M3 VWAP > M5 VWAP（多頭強勢排列，短線最強）
          0 = 盤整或纏繞（不符合上述兩種情況）
         -1 = M1 VWAP < M3 VWAP < M5 VWAP（空頭強勢排列，短線最弱）
       當個股突破 +2σ 且 alignment=+1 → 大盤帶動的真突破，做延續勝率高
       當個股突破 +2σ 但 alignment≤0  → 個股自己暴衝，大盤沒跟，做回歸

    3. market_vwap_spread_1_5
       大盤 M1 VWAP 與 M5 VWAP 的距離： (M1_VWAP - M5_VWAP) / M5_VWAP
       高 Expansion（絕對值大）→ 大盤處在趨勢擴張期，做延續勝率高
       低 Compression（絕對值接近 0）→ 大盤無量盤整，做回歸勝率高

    4. velocity_ratio_to_market
       個股價格斜率 vs 大盤 M1 VWAP 斜率的比值：
         slope_stock = close - close_shift(3)  （3分鐘價格變動）
         slope_market = market_m1_vwap - market_m1_vwap_shift(3)
         速度比過高（個股暴衝，大盤沒跟）→ 情緒性過熱，做回歸
         速度比適中（大盤與個股同步加速）→ 全面性突破，做延續

    df 需已有 stock_id/day_date/date/close/volume 欄位，且已執行過
    add_vwap_features()（確保 m1_vwap_z/m3_vwap_z/m5_vwap_z 存在）。
    m3/m5（選填）：3分鐘/5分鐘 rolling OHLCV，跟 make_features() 的慣例
    一致——訓練/回測傳 data/query.py::load_m3()/load_m5()，predict_live()
    留空（會對 0050 的 m1 現算）。
    """
    df = df.copy()

    # ── 提取 0050 的資料 ──
    idx_mask = df["stock_id"] == idx_symbol
    idx_rows = df[idx_mask]
    if idx_rows.empty:
        # 沒有 0050 就無法算大盤特徵，安靜地回傳原 df
        return df

    # ════════════════════════════════════════════════════════════════
    # 1. market_z_score_m5
    # ════════════════════════════════════════════════════════════════
    # 0050 的 m5_vwap_z 已經在 df 裡（add_vwap_features() 算的），
    # 直接廣播到同一分鐘的所有股票。
    idx_m5_z = idx_rows[["date", "m5_vwap_z"]].drop_duplicates("date")
    z_score = df.merge(
        idx_m5_z.rename(columns={"m5_vwap_z": "market_z_score_m5"}),
        on="date",
        how="left",
    )["market_z_score_m5"]
    df["market_z_score_m5"] = z_score

    # ════════════════════════════════════════════════════════════════
    # 2. market_vwap_alignment_score & 3. market_vwap_spread_1_5
    # ════════════════════════════════════════════════════════════════
    # 需要 0050 的 M1/M3/M5 VWAP 值。M1 直接從 df 算，M3/M5 從傳入的
    # m3/m5 DataFrame 或現算 compute_m3()/compute_m5() 取得。

    # -- M1 VWAP（從 df 裡的 0050 資料算累積 VWAP）--
    idx_m1 = idx_rows[["date", "close", "volume", "day_date"]].drop_duplicates("date").copy()
    idx_m1["_pv"] = idx_m1["close"] * idx_m1["volume"]
    g_m1 = idx_m1.groupby("day_date", group_keys=False, observed=True)
    idx_m1["_cum_pv"] = g_m1["_pv"].cumsum()
    idx_m1["_cum_vol"] = g_m1["volume"].cumsum().replace(0, np.nan)
    idx_m1["_m1_vwap"] = idx_m1["_cum_pv"] / idx_m1["_cum_vol"]

    # -- M3 VWAP --
    if m3 is None:
        # 從 df（含所有股票的 M1）現算 M3
        m3_df = compute_m3(df)
    else:
        m3_df = m3.copy()
    idx_m3 = m3_df[m3_df["stock_id"] == idx_symbol][["date", "close", "volume"]].drop_duplicates("date").copy()
    if not idx_m3.empty:
        idx_m3["day_date"] = idx_m3["date"].dt.date
        idx_m3["_pv"] = idx_m3["close"] * idx_m3["volume"]
        g_m3 = idx_m3.groupby("day_date", group_keys=False, observed=True)
        idx_m3["_cum_pv"] = g_m3["_pv"].cumsum()
        idx_m3["_cum_vol"] = g_m3["volume"].cumsum().replace(0, np.nan)
        idx_m3["_m3_vwap"] = idx_m3["_cum_pv"] / idx_m3["_cum_vol"]

    # -- M5 VWAP --
    if m5 is None:
        m5_df = compute_m5(df)
    else:
        m5_df = m5.copy()
    idx_m5 = m5_df[m5_df["stock_id"] == idx_symbol][["date", "close", "volume"]].drop_duplicates("date").copy()
    if not idx_m5.empty:
        idx_m5["day_date"] = idx_m5["date"].dt.date
        idx_m5["_pv"] = idx_m5["close"] * idx_m5["volume"]
        g_m5 = idx_m5.groupby("day_date", group_keys=False, observed=True)
        idx_m5["_cum_pv"] = g_m5["_pv"].cumsum()
        idx_m5["_cum_vol"] = g_m5["volume"].cumsum().replace(0, np.nan)
        idx_m5["_m5_vwap"] = idx_m5["_cum_pv"] / idx_m5["_cum_vol"]

    # merge VWAP 回 df
    vwap_m1 = idx_m1[["date", "_m1_vwap"]].drop_duplicates("date")
    df = df.merge(vwap_m1, on="date", how="left")
    if not idx_m3.empty:
        vwap_m3 = idx_m3[["date", "_m3_vwap"]].drop_duplicates("date")
        df = df.merge(vwap_m3, on="date", how="left")
    else:
        df["_m3_vwap"] = np.nan
    if not idx_m5.empty:
        vwap_m5 = idx_m5[["date", "_m5_vwap"]].drop_duplicates("date")
        df = df.merge(vwap_m5, on="date", how="left")
    else:
        df["_m5_vwap"] = np.nan

    # alignment score
    cond_bullish = (df["_m1_vwap"] > df["_m3_vwap"]) & (df["_m3_vwap"] > df["_m5_vwap"])
    cond_bearish = (df["_m1_vwap"] < df["_m3_vwap"]) & (df["_m3_vwap"] < df["_m5_vwap"])
    df["market_vwap_alignment_score"] = np.select(
        [cond_bullish, cond_bearish],
        [1, -1],
        default=0,
    )

    # spread 1-5
    m5_vwap_safe = df["_m5_vwap"].replace(0, np.nan)
    df["market_vwap_spread_1_5"] = (df["_m1_vwap"] - df["_m5_vwap"]) / m5_vwap_safe

    # ════════════════════════════════════════════════════════════════
    # 4. velocity_ratio_to_market
    # ════════════════════════════════════════════════════════════════
    # 個股價格斜率：diff(3) of close
    # 大盤 M1 VWAP 斜率：diff(3) of market_m1_vwap
    g_day = df.groupby(["stock_id", "day_date"], group_keys=False, observed=True)
    stock_slope = g_day["close"].diff(3)
    # 大盤 VWAP 斜率是逐日、同一支（0050）的 diff(3)，但廣播到所有股票後
    # 需要跨 stock_id 同一分鐘的值一樣，不能用 groupby 算（會逐股逐日 diff）
    # → 先算 0050 自己的 slope，再 merge 回 df
    idx_slope = df[idx_mask][["date", "_m1_vwap"]].drop_duplicates("date").copy()
    idx_slope["day_date"] = idx_slope["date"].dt.date
    g_idx = idx_slope.groupby("day_date", group_keys=False, observed=True)
    idx_slope["_market_slope"] = g_idx["_m1_vwap"].diff(3)
    market_slope = idx_slope[["date", "_market_slope"]].drop_duplicates("date")
    df = df.merge(market_slope, on="date", how="left")
    market_slope_safe = df["_market_slope"].replace(0, np.nan)
    df["velocity_ratio_to_market"] = stock_slope / market_slope_safe

    # 清理暫存欄位
    df = df.drop(columns=["_m1_vwap", "_m3_vwap", "_m5_vwap", "_market_slope"])
    return df


def add_vwap_features(
    m1: pd.DataFrame,
    std_mult: float = STD_MULT,
    m3: pd.DataFrame | None = None,
    m5: pd.DataFrame | None = None,
) -> pd.DataFrame:
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

    m3/m5（2026-07-26 新增）：3分鐘/5分鐘 rolling OHLCV，留空時退回內部
    用 data/resample.py::compute_m3()/compute_m5() 現算（predict_live() 的
    「今天」資料要這樣，因為批次快取 db/m3/db/m5 不含「今天」）。訓練/
    回測（全歷史批次）不應該傳空值——那樣會對全市場全歷史重新現算一次
    m3/m5，跟 db/m3/db/m5 早就存好的批次快取重複計算，也是造成全歷史訓練
    記憶體爆掉的另一個主因，train.py::_prepare_data() 改成直接讀
    data/query.py::load_m3()/load_m5() 傳進來，比照 orb/rally/mkt 訓練時
    的做法。

    m1 需已有 stock_id/date/day_date/open/high/low/close/volume 欄位，
    date 為 datetime，已依 stock_id/date 排序。
    """
    df = _add_vwap_z(m1, "m1")

    if m3 is None:
        m3 = compute_m3(m1)
    m3 = m3.copy()
    m3["day_date"] = m3["date"].dt.date
    m3 = _add_vwap_z(m3, "m3")[["stock_id", "date", "m3_vwap_z"]]
    df = df.merge(m3, on=["stock_id", "date"], how="left")

    if m5 is None:
        m5 = compute_m5(m1)
    m5 = m5.copy()
    m5["day_date"] = m5["date"].dt.date
    m5 = _add_vwap_z(m5, "m5")[["stock_id", "date", "m5_vwap_z"]]
    df = df.merge(m5, on=["stock_id", "date"], how="left")

    g_day = df.groupby(["stock_id", "day_date"], group_keys=False, observed=True)
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
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False, observed=True)
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
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False, observed=True)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    prev_close = g_day["close"].shift(1).fillna(m1["open"])
    m1["_tr5"] = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - prev_close).abs()),
        (m1["low"] - prev_close).abs(),
    )
    g_tr = m1.groupby(["stock_id", "day_date"], group_keys=False, observed=True)
    m1["atr5"] = g_tr["_tr5"].transform(lambda x: x.rolling(5, min_periods=5).mean()) / day_open
    return m1.drop(columns=["_tr5"])


# ═══════════════════════════════════════════════════════════════════════════════
# 三分類標籤（0=回歸VWAP／1=無訊號（盤整）／2=延續突破）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 2026-07-26 討論：視窗大小原本是「觸發的那個時間框自己的後 N 根」（m1
# 觸發看10分鐘、m3看30分鐘、m5看50分鐘），後來改成三個時間框統一看未來
# LABEL_HORIZON_MINUTES 分鐘——m1/m3/m5 的 z-score 都是每分鐘更新一次的
# 序列，「30分鐘」對三者直接就是「未來30根」，不用再乘時間框的分鐘數。
# 仍然是用觸發的那個時間框自己的 z-score 序列（例如 m5 觸發就看
# m5_vwap_z）判斷未來走勢，而不是統一都看 m1 的 z。
#
# 兩個 barrier 分別是「z 回到 0（回歸完成）」跟「z 進一步擴大超過
# CONTINUATION_STD_MULT（延續）」，不是像 orb/rally/mkt 那樣固定百分比
# 報酬——這裡衡量的是「價格相對 VWAP 的位置」，不是絕對報酬率。


def _shift_within_group(arr: np.ndarray, group_id: np.ndarray, k: int) -> np.ndarray:
    """把 arr 往未來位移 k 格（取得「未來第k筆」的值），但只在同一組
    （group_id相同，即同一支股票同一天）內位移——跨組（跨股票/跨日）的
    位置回傳 NaN，等同 groupby(...).shift(-k)，但用純 numpy 陣列操作
    實作，資料本來就已依 stock_id/date 排序、同組的列本來就是連續的，
    不需要真的呼叫 groupby。

    2026-07-26 討論：原本用 groupby(...).apply() 逐列 Python 迴圈算
    label，3個月全市場資料要等半天；改成外層只跑 horizon_bars 次
    （≤50，不是跑列數次，列數是千萬等級）、每次都是整欄向量化運算，
    去掉了逐列的 Python 直譯器開銷，這是實際的效能瓶頸所在。"""
    n = len(arr)
    out = np.full(n, np.nan)
    if k >= n:
        return out
    same_group = group_id[:-k] == group_id[k:]
    out[:-k][same_group] = arr[k:][same_group]
    return out


def _label_array_vwap(
    z: np.ndarray, group_id: np.ndarray, horizon_bars: int, continuation_mult: float, std_mult: float
) -> np.ndarray:
    """對整欄 z-score（不分股票/日，用 group_id 標記邊界）向量化算三分類
    label（0=回歸/1=無訊號/2=延續），語意跟舊版逐列迴圈完全一致，只是
    外層迴圈從「跑列數次」改成「跑 horizon_bars 次」。

    每次迴圈 k（1..horizon_bars）都是整欄一次算完「未來第k根」的訊號，
    用 revert_idx/cont_idx 記錄「目前為止看過的k裡，最早碰到該barrier的
    k值」——因為k是遞增順序跑的，第一次寫入的k就是最小的k，之後更大的k
    不會再覆蓋（k < revert_idx 這個條件只有第一次命中時成立）。
    """
    n = len(z)
    side = np.sign(z)
    revert_idx = np.full(n, horizon_bars + 1)  # 尚未命中的哨兵值
    cont_idx = np.full(n, horizon_bars + 1)
    valid_count = np.zeros(n, dtype=int)

    for k in range(1, horizon_bars + 1):
        z_fut = _shift_within_group(z, group_id, k)
        signed = side * z_fut
        valid = np.isfinite(signed)
        valid_count += valid

        hit_revert = valid & (signed <= 0) & (k < revert_idx)
        revert_idx = np.where(hit_revert, k, revert_idx)

        hit_cont = valid & (signed >= continuation_mult) & (k < cont_idx)
        cont_idx = np.where(hit_cont, k, cont_idx)

    labels = np.full(n, np.nan)
    labels[revert_idx < cont_idx] = 0  # 回歸
    labels[cont_idx < revert_idx] = 2  # 延續
    # 無訊號：兩個barrier在horizon_bars內都沒碰到，且視窗完整（horizon_bars
    # 根未來資料都有效，不是因為太靠近資料尾端/日底才沒碰到）
    neither_hit = (revert_idx > horizon_bars) & (cont_idx > horizon_bars) & (valid_count == horizon_bars)
    labels[neither_hit] = 1

    not_candidate = ~np.isfinite(z) | (np.abs(z) < std_mult)
    labels[not_candidate] = np.nan
    return labels


def make_vwap_labels(df: pd.DataFrame, std_mult: float = STD_MULT, continuation_mult: float | None = None) -> pd.Series:
    """
    對每個候選列（is_candidate=True，用 std_mult 判定），依 trigger_tf 用
    對應時間框自己的 z-score 序列算 label（0=回歸/1=無訊號/2=延續），非
    候選列回傳 NaN。

    std_mult/continuation_mult：見 add_vwap_features() 的說明，兩者要跟
    產生 is_candidate/trigger_tf 時用的 std_mult 一致，否則「候選」跟
    「label計算時判定candidate」的門檻會對不起來。

    continuation_mult 留 None 時＝ std_mult * 2（2026-07-26 討論：兩個
    barrier 到觸發點的「距離」要對稱，觸發點才不會天生偏向某一類——觸發時
    z 剛好等於 std_mult，回歸 barrier(0) 距離是 std_mult，延續 barrier
    要距離同樣是 std_mult 才公平，也就是 std_mult + std_mult = std_mult*2
    （例如 std_mult=2.0 時，延續門檻=4.0，兩邊都要走2.0）。原本用
    std_mult+1.0（std_mult=2.0時延續門檻=3.0，延續只需走1.0、回歸卻要走
    2.0），延續距離只有回歸的一半，天生比較容易觸發，訓練出來「延續」
    label比例(43.72%)明顯比回歸(25.36%)高，這個差距有多少是barrier不對稱
    造成的假象、有多少是市場真的比較常延續，原本設計沒辦法分開看，改成
    對稱門檻才能讓label比例真正反映市場行為。實驗不同std_mult時延續門檻
    跟著等比例調整，不用另外手動算。
    """
    if continuation_mult is None:
        continuation_mult = std_mult * 2

    # group_id：同一支股票同一天標同一個整數，給 _shift_within_group() 判斷
    # 「未來第k筆」是不是還在同一組內，比逐次呼叫 groupby(...).shift(-k)
    # 快（只需要算一次，跟 tf 無關）。
    group_id = df.groupby(["stock_id", "day_date"], sort=False, observed=True).ngroup().to_numpy()
    trigger_tf = df["trigger_tf"].to_numpy()

    label = np.full(len(df), np.nan)
    for tf in ("m1", "m3", "m5"):
        z = df[f"{tf}_vwap_z"].to_numpy()
        labels_arr = _label_array_vwap(z, group_id, LABEL_HORIZON_MINUTES, continuation_mult, std_mult)
        mask = trigger_tf == tf
        label[mask] = labels_arr[mask]
    return pd.Series(label, index=df.index)


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
    m3: pd.DataFrame | None = None,
    m5: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    給定全市場的 1 分K（load_m1() 或 load_m1_live() 的輸出），跑完整的
    vwap_ml pipeline：時段裁切 → VWAP z-score → 候選觸發 → ret_vs_idx →
    大盤 VWAP 特徵 → label → ATR5平盤過濾 → 只保留候選列 → 時段過濾。

    std_mult：候選觸發門檻，見 add_vwap_features()/make_vwap_labels() 的
    說明，實驗不同門檻時傳這裡就好，train.py::_prepare_data() 會轉傳。
    atr5_threshold：ATR5平盤過濾的絕對門檻，見 add_atr5() 的說明。
    m3/m5：3分鐘/5分鐘 rolling OHLCV，訓練/回測應該傳
    data/query.py::load_m3()/load_m5() 讀現成批次快取（不要留空自己現算，
    見 add_vwap_features() 的說明），predict_live() 因為要處理批次快取
    沒有的「今天」資料，留空即可。

    m1 需已有 stock_id/date/open/high/low/close/volume 欄位。
    """
    m1 = m1.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    # 裁到 SESSION_END + 最長label視窗，中午/下午盤的資料整個pipeline
    # 都用不到，提前砍掉能讓後面所有計算的資料量降到原本的4成左右，見
    # _trim_to_needed_window() 的說明（2026-07-26 討論，全歷史訓練記憶體
    # 爆掉的主因之一）。
    m1 = _trim_to_needed_window(m1)
    if m3 is not None:
        m3 = _trim_to_needed_window(m3)
    if m5 is not None:
        m5 = _trim_to_needed_window(m5)

    df = add_vwap_features(m1, std_mult=std_mult, m3=m3, m5=m5)

    # 2026-07-27 新增：個股 vs 大盤（0050）的累積報酬率差值。
    # ⚠️ 一定要在篩選 day_trade_stocks / 排除 0050 之前呼叫（0050 被篩掉
    # 後就沒有基準可以算了），predict_live() 比照 mkt 的做法——先對完整
    # m1_live 算特徵，才排除 0050 跟套用 day_trade_stocks 篩選。
    df = add_ret_vs_idx(df)

    # 2026-07-28 新增：大盤（0050）的 VWAP 特徵（alignment / spread / z-score /
    # velocity ratio），跟 ret_vs_idx 一樣需要在排除 0050 前先算。
    df = add_market_vwap_features(df, m3=m3, m5=m5)

    df = add_bar_features(df)
    df = add_atr5(df)
    df["target"] = make_vwap_labels(df, std_mult=std_mult)
    df = add_trigger_tf_dummies(df)

    # 只留真正觸發過 band 的候選事件當訓練樣本，不是每分鐘都當一筆（比照
    # strategy/orb/features.py 2026-07-11 之後「只留真正突破事件」的做法）。
    df = df[df["is_candidate"]]

    # ATR5平盤過濾（絕對門檻，三個class都篩，2026-07-26討論）：濾掉
    # 「本來就沒什麼波動、z只是分母（累積偏離標準差）太小才被撐大」的假
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
# 2026-07-27：加 ret_vs_idx（個股累積報酬率 vs 0050累積報酬率差值）、
# idx_ret_since_open（0050自己的累積報酬率）。
# 2026-07-28：加 market_z_score_m5（大盤 M5 VWAP 偏離程度）、
# market_vwap_alignment_score（大盤 M1/M3/M5 VWAP 排列方向）、
# market_vwap_spread_1_5（大盤 M1 vs M5 VWAP 擴張度）、
# velocity_ratio_to_market（個股 vs 大盤 VWAP 的速度比）。
FEATURES = [
    "m1_vwap_z",
    "m3_vwap_z",
    "m5_vwap_z",
    "m1_vwap_z_slope",
    "m3_vwap_z_slope",
    "m5_vwap_z_slope",
    "ret_vs_idx",
    "idx_ret_since_open",
    "market_z_score_m5",
    "market_vwap_alignment_score",
    "market_vwap_spread_1_5",
    "velocity_ratio_to_market",
    "vol_ratio_prev",
    "ret_1m",
    "body_pct",
    "trigger_is_m1",
    "trigger_is_m3",
    "trigger_is_m5",
]
