"""
即時推論 — predict()（批次機率矩陣，回測用）、predict_live()（正式即時推論入口）

mkt_idx 是3分類（跌=0/平=1/漲=2），但策略本身只做多、只在乎「漲」這個訊號的
機率，所以這裡跟 orb/rally 的二分類 predict_proba()[:, 1] 概念一致，只是改抓
class=2 那一欄，套進同一份 backtest/intraday_platform.py 共用回測引擎（那支
引擎只吃單一機率矩陣，不知道背後模型是幾分類）。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from data.query import load_m1, load_m1_live
from data.resample import compute_m3, compute_m3_std, compute_m5, compute_m5_std
from strategy.mkt_idx.config import IDX_SYMBOL, MODEL_TYPE
from strategy.mkt_idx.features import (
    FEATURES,
    add_bar_features,
    add_m3_m5_features,
    add_m3_m5_std_features,
    add_ret_vs_idx,
    top_n_stock_ids_by_latest_volume,
)
from strategy.mkt_idx.train import _prepare_data, load_model_by_type


def _up_proba(model, df: pd.DataFrame):
    """取「漲」（class=2）那一欄機率，不假設 classes_ 順序（雖然目前sklearn
    對整數標籤預設會照數值排序，仍明確用 class_idx 對應，避免哪天標籤改
    成非數值/亂序時默默取錯欄）。"""
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    return model.predict_proba(df[FEATURES])[:, class_idx[2]]


def predict(model=None, test_days: int = 30, test_only: bool = True, use_cache: bool = False) -> pd.DataFrame:
    """
    對全天資料產生「漲」機率矩陣（index=datetime, columns=stock_id），回測用。

    直接沿用 train.py::_prepare_data() 整條 pipeline（流動性過濾、時段過濾、
    特徵、cache機制都跟訓練時完全一致，不用另外重寫一份），target 欄位這裡
    用不到，但反正 _prepare_data() 本來就會算，直接忽略即可。

    test_only=True（預設）時，只回傳最後 test_days 天——這幾天是模型訓練時
    沒看過的樣本外資料，跟 train.py 切分訓練/測試集用同一套 cutoff 邏輯
    （見 train.py::_split_data() 的說明）。test_only=False 才會回傳全部（含
    訓練集），訓練集算出來的績效會被模型「背過答案」灌水，沒有參考價值，
    只用來除錯。
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    df = _prepare_data(use_cache=use_cache)
    if test_only:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        df = df[df["date"] > cutoff]

    df = df.copy()
    df["proba"] = _up_proba(model, df)
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def build_prewarm_cache(top_n: int = 100) -> dict:
    """
    盤前預算快取 — 給 strategy/prewarm.py 統一呼叫的介面，比照
    strategy/orb/predict.py、strategy/rally/predict.py 的 build_prewarm_cache()。

    top_n_stock_ids：今天的流動性名單（前一交易日全天量前 top_n 名），開盤
    前算一次、整天沿用，不要讓 predict_live() 每分鐘都重算一次
    top_n_stock_ids_by_latest_volume(load_m1())（load_m1() 讀全部歷史分K，
    很慢）。

    回傳的 dict key 要跟 predict_live() 接受的參數名一致，因為
    main/live_trader.py 會直接 **cache 展開傳進 predict_live()。
    """
    top_n_stock_ids = top_n_stock_ids_by_latest_volume(load_m1(), n=top_n)
    return {"top_n_stock_ids": top_n_stock_ids}


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    top_n_stock_ids: set | None = None,
    model=None,
    threshold: float = 0.6,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論。

    ⚠️ 參數順序（尤其 day 是第2個位置參數）要跟 orb/rally 的 predict_live()
    完全一致——main/live_trader.py 對所有策略模組一視同仁，統一用
    `s.predict_live(minute_str, state.day, model=..., threshold=0,
    day_trade_stocks=..., m1_live=..., **s.prewarm_cache)` 這種寫法呼叫
    （state.day 是位置參數），不會知道也不會檢查個別策略實際用不用得到
    day。mkt_idx 目前用不到日K背景資料（ret_vs_idx/m3/m5/m3_std/m5_std全部
    是分K現算），day 收下來直接忽略，但仍要保留在同一個位置，不然
    state.day 會被誤傳進下一個參數的位置（例如錯的話會被當成 m1_live，
    整條 pipeline 全部算錯，2026-07-21發現這個bug）。

    top_n_stock_ids: 今天的流動性名單，見 build_prewarm_cache()。留空則內部
        自己現算 top_n_stock_ids_by_latest_volume(load_m1())（對全歷史 db/m1/
        算，比較慢；production 環境應該開盤前算一次傳進來，不要每分鐘重算）。
    day_trade_stocks: 當沖標的 set，若提供則只推論這些股票（跟
        top_n_stock_ids 是「且」的關係，兩邊都要通過才會納入候選）。
    m1_live: 已載入的今日即時分K，傳入可避免每次呼叫都重讀 db/m1_live/
        （live_trader.py 應該自己維護一份、每分鐘更新後傳進來）。留空則沿用
        舊行為，內部自己 load_m1_live()。

    ⚠️ 跟 orb/rally 的 predict_live() 不同：那兩支一開始就用 day_trade_stocks
    篩掉 m1_live，這裡不能這麼做——ret_vs_idx 要用 0050 自己的分K資料當基準
    （見 features.py::add_ret_vs_idx()），如果 0050 不在當沖名單裡就會被
    先篩掉，之後就沒有基準可以算。所以這裡先對完整 m1_live 算完 ret_vs_idx，
    才排除 0050 本身、套用 top_n_stock_ids/day_trade_stocks 篩選。

    回傳格式：[{"stock_id": ..., "proba": ..., "price": ...}, ...]
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    m1_live = m1_live.copy()
    m1_live["date"] = pd.to_datetime(m1_live["date"])
    m1_live = m1_live.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1_live["day_date"] = m1_live["date"].dt.date

    # ret_vs_idx 要靠 0050 自己的資料算基準，一定要在排除/篩選之前先算完
    # （見上面 docstring 的說明），add_ret_vs_idx() 內部用 merge(on="date")
    # 且已對0050去重，會保留 m1_live 原本依 stock_id/date 排序的列順序，
    # 後面 compute_m3/compute_m5 的 rolling 計算才不會因為列序被打亂而算錯。
    df = add_ret_vs_idx(m1_live)
    df = df[df["stock_id"] != IDX_SYMBOL]

    if top_n_stock_ids is None:
        top_n_stock_ids = top_n_stock_ids_by_latest_volume(load_m1())
    df = df[df["stock_id"].isin(top_n_stock_ids)]
    if day_trade_stocks:
        df = df[df["stock_id"].isin(day_trade_stocks)]
    if df.empty:
        return []

    df = add_bar_features(df)

    # db/m3、db/m5、db/m3_std、db/m5_std 都是批次預算、不含「今天」的資料，
    # 即時推論要對 m1_live 現算，用 data/resample.py 共用的同一套函式（跟
    # 批次預算腳本用的是同一份邏輯），保證跟訓練用的公式一致，理由同
    # strategy/rally/predict.py、strategy/orb/predict.py 的 predict_live()。
    m3_live = compute_m3(df)
    m5_live = compute_m5(df)
    df = add_m3_m5_features(df, m3_live, m5_live)

    m3_std_live = compute_m3_std(df)
    m5_std_live = compute_m5_std(df)
    df = add_m3_m5_std_features(df, m3_std_live, m5_std_live)

    current = df[df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []

    valid = current.dropna(subset=FEATURES)
    if valid.empty:
        return []

    proba = _up_proba(model, valid)
    signals = [
        {"stock_id": row["stock_id"], "proba": float(p), "price": float(row["close"])}
        for (_, row), p in zip(valid.iterrows(), proba)
        if p >= threshold
    ]
    return sorted(signals, key=lambda x: -x["proba"])
