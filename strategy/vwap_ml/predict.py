"""
即時推論 — predict()（批次機率矩陣，回測用）、predict_live()（正式即時
推論入口）。

vwap_ml 是3分類（回歸=0/無訊號=1/延續=2），但策略本身要看的是「這個候選
事件該做多還是做空」，不是單純哪個class機率最高——要配合觸發時價格在
VWAP之上還是之下（trigger_side）才能決定方向（見 features.py 檔頭的
方向對應表，_direction_probas() 就是做這個轉換）。

⚠️ 2026-07-26 決定：只交易「回歸」訊號，不交易「延續」——walk-forward
驗證顯示延續(class=2)的precision只有18~23%，不夠可靠；回歸(class=0)在
門檻0.6~0.8時precision穩定在67~93%（5個窗口驗證過，見
strategy/vwap_ml/experiments/walk_forward.py）。_direction_probas() 因此
只把回歸機率換算成做多/做空，延續機率完全不使用、不會產生任何訊號。
之後如果延續的模型/特徵有改善、精確度夠高，才考慮加回來。

跟 mkt 一樣的限制：共用回測引擎 backtest/intraday_platform.py 只吃單一
（做多）機率矩陣，不知道背後模型是幾分類、也沒有方向概念，所以 predict()
（回測用）只回傳「做多」機率；predict_live()（即時訊號，給前端顯示用）
才會同時算多空兩個方向。支援回測/實際下單放空是完全另一個規模的工程，
先不做（2026-07-23 mkt 遇到同樣考量時的做法）。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.query import load_m1_live
from strategy.vwap_ml.config import MODEL_TYPE, THRESHOLD
from strategy.vwap_ml.features import FEATURES, make_features
from strategy.vwap_ml.train import _prepare_data, load_model_by_type


def _direction_probas(model, df: pd.DataFrame):
    """把「回歸」(class=0) 機率轉成「做多」「做空」機率：

        trigger_side="upper"（z>+STD_MULT，價格在VWAP之上）：
            回歸(class=0) → 做空（預期從高點拉回VWAP）
        trigger_side="lower"（z<-STD_MULT，價格在VWAP之下）：
            回歸(class=0) → 做多（預期從低點反彈回VWAP）

    只用回歸機率，不使用延續（class=2）——見本檔案檔頭 2026-07-26 的
    決定說明。上軌時回歸對應做空，所以 p_up=0；下軌時回歸對應做多，所以
    p_down=0（兩邊都各自只有一個方向有非零機率，不會同時做多又做空）。

    見 features.py 檔頭的方向對應表。回傳 (p_up, p_down) 兩個跟 df 等長
    的 numpy array。"""
    proba = model.predict_proba(df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    p_revert = proba[:, class_idx[0]]
    is_upper = (df["trigger_side"] == "upper").to_numpy()
    p_up = np.where(is_upper, 0.0, p_revert)
    p_down = np.where(is_upper, p_revert, 0.0)
    return p_up, p_down


def predict(
    model=None,
    test_days: int = 30,
    test_only: bool = True,
    use_cache: bool = False,
    start_date: str | None = None,
) -> pd.DataFrame:
    """
    對候選事件產生「做多」機率矩陣（index=datetime, columns=stock_id），
    回測用。直接沿用 train.py::_prepare_data() 整條 pipeline（VWAP z-score、
    候選觸發、標籤、cache機制都跟訓練時完全一致），比照
    strategy/mkt/predict.py::predict() 的做法。

    test_only=True（預設）時，只回傳最後 test_days 天——這幾天是模型訓練
    時沒看過的樣本外資料，跟 train.py 切分訓練/測試集用同一套 cutoff 邏輯。

    start_date 要跟訓練那個模型用的 start_date 一致，才會讀到同一份cache
    （見 train.py::_prepare_data()::_cache_path_for() 的說明）。
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    df = _prepare_data(use_cache=use_cache, start_date=start_date)
    if test_only:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        df = df[df["date"] > cutoff]

    df = df.copy()
    p_up, _ = _direction_probas(model, df)
    df["proba"] = p_up
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    model=None,
    threshold: float = THRESHOLD,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論。

    ⚠️ 參數順序（day 是第2個位置參數）要跟 orb/rally/mkt 的
    predict_live() 完全一致——main/live_trader.py 對所有策略模組一視同仁，
    統一用 `s.predict_live(minute_str, state.day, model=..., threshold=0,
    day_trade_stocks=..., m1_live=..., **s.prewarm_cache)` 這種寫法呼叫
    （state.day 是位置參數），錯位的話會讓後面的參數全部對錯位置
    （2026-07-21 mkt 發現過這個 bug）。vwap_ml 目前用不到 day，但一樣要
    佔住第2個位置。

    vwap_ml 的 VWAP z-score 只需要「今天」的資料（expanding，不看跨日
    歷史），不像 orb 需要跨日歷史查表，所以沒有 build_prewarm_cache()
    （比照 rally 的做法——沒有跨分鐘要快取的東西，strategy/prewarm.py 對
    沒實作這個函式的策略模組會自動當作 {}，不影響 main/premarket.py 的
    refresh_prewarm() 流程）。

    day_trade_stocks: 當沖標的 set，若提供則只推論這些股票——vwap_ml 沒有
    像 mkt 的 ret_vs_idx 需要靠大盤代理股票（0050）當基準的問題，可以在
    算特徵之前就先篩，不用像 mkt 那樣延後篩選（見 strategy/mkt/predict.py
    的說明）。

    回傳格式：[{"stock_id", "proba", "price", "direction": "up"|"down"}, ...]
    ——同一支股票在同一分鐘理論上只會出現一次（三分類機率加總=1，「同時
    看漲又看跌」機率上幾乎不會發生，除非門檻設很低）。純訊號，不是下單
    指令。
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    if day_trade_stocks:
        m1_live = m1_live[m1_live["stock_id"].isin(day_trade_stocks)]
        if m1_live.empty:
            return []

    df = make_features(m1_live)
    if df.empty:
        return []

    current = df[df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []

    valid = current.dropna(subset=FEATURES)
    if valid.empty:
        return []

    # trigger_side 決定唯一方向，p_up/p_down 只有一邊非零（_direction_probas
    # 的遮罩設計）。之前分開判斷 up_p>=threshold / down_p>=threshold，
    # threshold=0 時（on_minute 監控用途傳 threshold=0）0.0>=0 恆真，導致
    # 被遮罩成0的那個方向也會生一筆假訊號，讓 vwap_ml_up/down 同時把同一支
    # 股票推進各自監控清單（2026-07-27 vwap_dl 發現同樣的 bug，這裡一起修）。
    # 改成用 trigger_side 直接決定唯一方向，每支候選股最多只出現一筆。
    p_up, p_down = _direction_probas(model, valid)
    signals = []
    for (_, row), up_p, down_p in zip(valid.iterrows(), p_up, p_down):
        direction = "down" if row["trigger_side"] == "upper" else "up"
        proba = float(down_p) if direction == "down" else float(up_p)
        if proba >= threshold:
            signals.append(
                {"stock_id": row["stock_id"], "proba": proba, "price": float(row["close"]), "direction": direction}
            )
    return sorted(signals, key=lambda x: -x["proba"])
