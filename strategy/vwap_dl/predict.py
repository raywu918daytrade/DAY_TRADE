"""
即時推論 — predict()（批次機率矩陣，回測用）、predict_live()（即時推論入口）。

3 分類（回歸=0/無訊號=1/延續=2），配合 trigger_side 轉成做多/做空方向。
只交易「回歸」訊號（同 vwap_ml 的決定，延續 precision 不夠可靠）。

方向對應（vwap_dl 是做回歸模型，預測價格回到 VWAP）：
    -2 標準差（trigger_side="lower"，價格低於 VWAP）→ 回歸到 VWAP → 做多 (up)
    +2 標準差（trigger_side="upper"，價格高於 VWAP）→ 回歸到 VWAP → 做空 (down)
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
from strategy.vwap_dl.config import DEVICE, THRESHOLD
from strategy.vwap_dl.dataset import _GRU_FEATURE_COLS, _RESNET_COLS
from strategy.vwap_dl.train import (
    ShardedDataset,
    available_months,
    collate_fn,
    load_model,
    load_shard_data,
    load_shard_meta,
    _month_bound,
)
from strategy.vwap_ml.features import make_features


def _direction_probas(probs: np.ndarray, trigger_sides: np.ndarray):
    """把「回歸」(class=0) 機率轉成做多/做空機率。

    VWAP z-score 觸發點與方向的對應（基於回歸到 VWAP 的假設）：

        trigger_side="upper"（z > +2σ，價格在 VWAP 之上）：
            VWAP = 下方吸引子，回歸到 VWAP → 價格下跌 → 做空 (down)
        trigger_side="lower"（z < -2σ，價格在 VWAP 之下）：
            VWAP = 上方吸引子，回歸到 VWAP → 價格上漲 → 做多 (up)

    只用回歸機率（class=0），延續（class=2）不使用——同 vwap_ml 的決定。

    回傳 (p_up, p_down) 兩個跟 probs 等長的 numpy array，
    同一個 trigger_side 只有對應方向有非零機率。
    """
    p_revert = probs[:, 0]
    is_upper = trigger_sides == "upper"
    p_up = np.where(is_upper, 0.0, p_revert)
    p_down = np.where(is_upper, p_revert, 0.0)
    return p_up, p_down


def predict(
    model=None,
    test_days: int = 30,
    start_date: str = "2024-01-01",
    batch_size: int = 256,
) -> pd.DataFrame:
    """對測試集產生「做多」機率矩陣（index=datetime, columns=stock_id），回測用。"""
    if model is None:
        model = load_model()

    months = available_months()
    if start_date:
        months = [m for m in months if m >= _month_bound(start_date)]
    months = sorted(months)
    if not months:
        return pd.DataFrame()

    metas = {m: load_shard_meta(m) for m in months}
    global_max_date = max(meta["date"].max() for meta in metas.values())
    cutoff = global_max_date - pd.Timedelta(days=test_days)
    test_filters = {m: np.nonzero(meta["date"].to_numpy() > cutoff)[0] for m, meta in metas.items()}
    test_ds = ShardedDataset(months, test_filters)
    if len(test_ds) == 0:
        return pd.DataFrame()

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for resnet_x, gru_seq, gru_len, _y in loader:
            resnet_x = resnet_x.to(DEVICE)
            gru_seq = gru_seq.to(DEVICE)
            gru_len = gru_len.to(DEVICE)
            probs = torch.softmax(model(resnet_x, gru_seq, gru_len), dim=1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)

    trigger_sides = test_ds.meta["trigger_side"].to_numpy()
    p_up, _ = _direction_probas(probs, trigger_sides)

    df = test_ds.meta[["stock_id", "date"]].copy()
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

    參數順序跟 vwap_ml/predict.py::predict_live() 一致，main/live_trader.py
    對所有策略模組一視同仁用位置參數呼叫。

    回傳格式：[{"stock_id", "proba", "price", "direction": "up"|"down"}, ...]
    """
    if model is None:
        model = load_model()

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    if day_trade_stocks:
        m1_live = m1_live[m1_live["stock_id"].isin(day_trade_stocks)]
        if m1_live.empty:
            return []

    # 先跑 VWAP features 找出候選
    from data.resample import compute_m3, compute_m5

    m1 = m1_live.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    m3 = compute_m3(m1)
    m5 = compute_m5(m1)
    cand_df = make_features(m1, m3=m3, m5=m5)

    # 只保留當下這一分鐘的候選
    current = cand_df[cand_df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []
    current = current[current["is_candidate"]]
    if current.empty:
        return []

    # 建寬表 + 正規化（比照 dataset.py::_build_wide_frame）
    from strategy.vwap_dl.dataset import _add_atr_ma, _add_market_relative, _normalize_ohlcv

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

    day_open_pre = (
        wide.groupby(["stock_id", "day_date"], group_keys=False)["open"].transform("first").replace(0, np.nan)
    )
    wide = _add_market_relative(wide, day_open_pre, "0050")
    g_day = wide.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    _add_atr_ma(wide, g_day, day_open)
    _normalize_ohlcv(wide, "m1", day_open, g_day)

    # VWAP z-score merge
    vwap_z = current[["stock_id", "date", "m1_vwap_z", "m3_vwap_z", "m5_vwap_z"]].drop_duplicates(
        subset=["stock_id", "date"]
    )
    wide = wide.merge(vwap_z, on=["stock_id", "date"], how="left")
    wide["m1_vwap_z"] = wide["m1_vwap_z"].fillna(0)
    wide["m3_vwap_z"] = wide["m3_vwap_z"].fillna(0)
    wide["m5_vwap_z"] = wide["m5_vwap_z"].fillna(0)

    # 對每個候選股票組裝 tensor
    candidate_stocks = set(current["stock_id"].unique())
    minute_ts = pd.Timestamp(minute_str)

    resnet_list, gru_seqs, gru_lens, stock_ids, atr5_vals, trigger_sides = [], [], [], [], [], []

    for sid in candidate_stocks:
        day_df = wide[(wide["stock_id"] == sid) & (wide["date"] <= minute_ts)].sort_values("date")
        if len(day_df) < 10 or day_df["date"].iloc[-1] != minute_ts:
            continue

        # ResNet: 近 10 分鐘 OHLCV
        resnet_win = day_df.iloc[-10:][_RESNET_COLS].to_numpy(dtype=np.float32).T  # (5, 10)
        if not np.isfinite(resnet_win).all():
            continue

        # GRU: 從 9:00 到當下
        gru_seq = day_df[_GRU_FEATURE_COLS].to_numpy(dtype=np.float32)  # (seq_len, 14)
        if not np.isfinite(gru_seq).all():
            continue

        resnet_list.append(resnet_win)
        gru_seqs.append(gru_seq)
        gru_lens.append(len(gru_seq))
        stock_ids.append(sid)
        atr5_vals.append(day_df["atr5"].iloc[-1])
        side = current.loc[current["stock_id"] == sid, "trigger_side"].values
        trigger_sides.append(side[0] if len(side) > 0 else "upper")

    if not stock_ids:
        return []

    # ATR5 過濾
    from strategy.vwap_dl.config import ATR5_FILTER_THRESHOLD

    keep = np.array(atr5_vals) >= ATR5_FILTER_THRESHOLD
    if not keep.any():
        return []
    stock_ids = [sid for sid, k in zip(stock_ids, keep) if k]
    trigger_sides = np.array(trigger_sides)[keep]
    resnet_x = np.stack([r for r, k in zip(resnet_list, keep) if k], axis=0)
    gru_seqs = [s for s, k in zip(gru_seqs, keep) if k]
    gru_lens = np.array([l for l, k in zip(gru_lens, keep) if k])

    # Padding + 推論
    max_len = gru_lens.max()
    gru_padded = []
    for seq in gru_seqs:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode="constant")
        gru_padded.append(seq)
    gru_seq = np.stack(gru_padded, axis=0)

    model.eval()
    with torch.no_grad():
        rx = torch.from_numpy(resnet_x).float().to(DEVICE)
        gs = torch.from_numpy(gru_seq).float().to(DEVICE)
        gl = torch.from_numpy(gru_lens).long().to(DEVICE)
        probs = torch.softmax(model(rx, gs, gl), dim=1).cpu().numpy()

    p_up, p_down = _direction_probas(probs, trigger_sides)

    latest_close = (
        m1_live[pd.to_datetime(m1_live["date"]) == pd.Timestamp(minute_str)].set_index("stock_id")["close"].to_dict()
    )

    # trigger_side 決定唯一方向，p_up/p_down 只有一邊非零（_direction_probas
    # 的遮罩設計）。之前分開判斷 p_up>=threshold / p_down>=threshold，threshold=0
    # 時（on_minute 傳 threshold=0 取全量做監控）0.0>=0 恆真，導致每支候選股
    # 兩個方向都會各生一筆（一筆真機率、一筆假的 0.0），讓 vwap_dl_up/down
    # 同時把同一支股票推進各自的監控清單（2026-07-27 發現）。改成用
    # trigger_side 直接決定唯一方向，每支候選股最多只出現一筆。
    signals = []
    for i, sid in enumerate(stock_ids):
        price = latest_close.get(sid)
        if price is None:
            continue
        direction = "down" if trigger_sides[i] == "upper" else "up"
        proba = float(p_down[i]) if direction == "down" else float(p_up[i])
        if proba >= threshold:
            signals.append({"stock_id": sid, "proba": proba, "price": price, "direction": direction})
    return sorted(signals, key=lambda x: -x["proba"])
