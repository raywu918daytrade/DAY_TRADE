"""
即時推論 — predict()（批次機率矩陣，回測用）、predict_live()（正式即時推論入口）

cnn是3分類（跌=0/平=1/漲=2），跟strategy/mkt同樣的做法：predict()只取「漲」
（class=2）那一欄機率，套進backtest/intraday_platform.py共用回測引擎（那支
引擎只吃單一做多機率矩陣，沒有方向概念，回測放空是另一個規模的工程，目前
不做——見strategy/mkt/predict.py的同樣說明）；predict_live()兩個方向都算，
回傳訊號各自標記direction="up"/"down"，給strategy/cnn/up、strategy/cnn/down
兩個薄live.py介面依direction各自過濾。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.query import load_m1_live
from strategy.cnn.config import ATR5_FILTER_THRESHOLD, DEVICE, SESSION_END, SESSION_START
from strategy.cnn.dataset import BRANCH_NAMES, build_live_candidates
from strategy.cnn.train import _load_test_ds_hours, load_model


def predict(model=None, test_days: int = 30, start_date: str = "2024-01-01", batch_size: int = 256) -> pd.DataFrame:
    """
    對測試集產生「漲」機率矩陣（index=datetime, columns=stock_id），回測用。

    直接沿用 train.py::_load_test_ds_hours()（跟train()訓練時用同一套
    hour_start=9/hour_end=10、ATR5_FILTER_THRESHOLD過濾邏輯，母體跟訓練時
    看到的分布一致），只回傳固定測試集（樣本外資料，跟train.py的
    test_days切分同一套cutoff）。

    start_date 要跟目前這個模型實際訓練時用的範圍一致（2026-07-25發現的
    坑）：_load_test_ds_hours()留空start_date會去掃db/m1整段歷史（回溯到
    2023-11），發現db/m1有、但cache/cnn/還沒建過shard的月份（例如
    2023-11/2023-12，從沒被train()用過），就會觸發整包重建，跑得極慢。
    """
    if model is None:
        model = load_model()

    test_ds = _load_test_ds_hours(
        test_days,
        hour_start=SESSION_START[0],
        hour_end=SESSION_END[0],
        atr5_min=ATR5_FILTER_THRESHOLD,
        start_date=start_date,
    )
    if len(test_ds) == 0:
        return pd.DataFrame()

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for x, _y in loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)  # (N, 3)

    df = test_ds.meta[["stock_id", "date"]].copy()
    df["proba"] = probs[:, 2]  # 漲
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    model=None,
    threshold: float = 0.6,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論。

    ⚠️ 參數順序（尤其 day 是第2個位置參數）要跟 rally/orb/mkt 的
    predict_live() 完全一致——main/live_trader.py 對所有策略模組一視同仁，
    統一用位置參數呼叫 s.predict_live(minute_str, state.day, ...)，就算cnn
    這裡完全用不到day（cnn沒有用到db/fugle_day日K特徵），也要保留在同一個
    位置，不然state.day會被誤傳進下一個參數位置（見rally/mkt同名函式的
    docstring說明過的同一個bug）。

    day_trade_stocks: 當沖標的 set，若提供則只推論這些股票（在
        build_live_candidates() 算完所有候選之後才篩，理由同mkt：cnn的
        大盤相對特徵(ret_vs_idx)要用0050自己的m1資料當基準，太早篩掉
        m1_live會連0050自己的資料都被濾掉，後面沒有基準可以算）。
    m1_live: 已載入的今日即時分K，傳入可避免每次呼叫都重讀db/m1_live/。

    回傳格式：[{"stock_id":..., "proba":..., "price":..., "direction":"up"|"down"}, ...]
    ——同一支股票同一分鐘理論上最多出現2筆（up/down機率各自比threshold，
    兩邊都過門檻的話兩筆都會回傳，但3分類機率加總=1，兩邊同時很高機率的
    情況極少見，除非threshold設很低）。純訊號，不是下單指令。
    """
    if model is None:
        model = load_model()

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    branches, stock_ids, atr5 = build_live_candidates(m1_live, minute_str)
    if not stock_ids:
        return []

    # ATR5平盤過濾（絕對門檻，跌/平/漲一視同仁，理由見config.py::
    # ATR5_FILTER_THRESHOLD說明）：訓練時只篩「平」，但即時推論不知道真正
    # label，改成對所有候選一視同仁套用同一個門檻，跟train()/confidence_
    # report()看到的候選分布保持一致。
    keep = atr5 >= ATR5_FILTER_THRESHOLD
    if not keep.any():
        return []
    stock_ids = [sid for sid, k in zip(stock_ids, keep) if k]
    branches = {name: arr[keep] for name, arr in branches.items()}

    if day_trade_stocks:
        mask = np.array([sid in day_trade_stocks for sid in stock_ids])
        if not mask.any():
            return []
        stock_ids = [sid for sid, m in zip(stock_ids, mask) if m]
        branches = {name: arr[mask] for name, arr in branches.items()}

    x = {name: torch.from_numpy(branches[name]).float().to(DEVICE) for name in BRANCH_NAMES}
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()  # (N, 3)

    # m1分支的close是正規化過的（/day_open），不能拿來當price，改從m1_live
    # 當下這一分鐘的原始close取。
    latest_close = (
        m1_live[pd.to_datetime(m1_live["date"]) == pd.Timestamp(minute_str)]
        .set_index("stock_id")["close"]
        .to_dict()
    )

    signals = []
    for i, sid in enumerate(stock_ids):
        price = latest_close.get(sid)
        if price is None:
            continue
        up_p, down_p = float(probs[i, 2]), float(probs[i, 0])
        if up_p >= threshold:
            signals.append({"stock_id": sid, "proba": up_p, "price": price, "direction": "up"})
        if down_p >= threshold:
            signals.append({"stock_id": sid, "proba": down_p, "price": price, "direction": "down"})
    return sorted(signals, key=lambda x: -x["proba"])
