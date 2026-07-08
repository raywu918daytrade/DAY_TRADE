"""
事後驗證：對指定日期已存在 HF Hub 的推論訊號（day_trade/inference/{date}.parquet 裡
is_signal=True 的列），套用模型訓練時同一套 Triple Barrier 規則（TP_PCT/SL_PCT/HOLD_BARS，
定義在 strategy/date_trade_model.py，跟 _barrier_label_group() 完全一致），
往後看分K算出「如果那時候真的進場，最後是贏是輸」，補進 HF：
    day_trade/inference_eval/{date}.parquet

跟訓練 label 的差異：
    訓練時「太靠近當日尾端、不足 HOLD_BARS 根且都沒碰到 TP/SL」的樣本會直接捨棄（NaN）；
    這裡改成保留一筆 outcome="unresolved" 的紀錄，讓使用者知道「這幾筆訊號當天收盤前
    還沒有分出勝負」，而不是悄悄消失。

用法：
    python scripts/evaluate_inference_log.py --date 2026-07-03
"""

import argparse
import os
import sys
import time
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_TW = timezone(timedelta(hours=8))
_HF_INFERENCE_PREFIX = "day_trade/inference"
_HF_EVAL_PREFIX = "day_trade/inference_eval"


def parse_args():
    p = argparse.ArgumentParser(description="對指定日期的推論訊號做事後驗證（Triple Barrier），補進 HF Hub")
    p.add_argument("--date", required=True, help="YYYY-MM-DD，必須是已經有 day_trade/inference/{date}.parquet 的日期")
    return p.parse_args()


def _load_signals(date_str: str) -> pd.DataFrame:
    """從 HF Hub 讀該日推論記錄，只取 is_signal=True 的列（要驗證的對象）"""
    repo_id = os.environ.get("HF_REPO_ID", "")
    token = os.environ.get("HF_TOKEN") or None
    if not repo_id:
        raise RuntimeError("請設定 HF_REPO_ID")
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=repo_id,
        filename=f"{_HF_INFERENCE_PREFIX}/{date_str}.parquet",
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(local)
    signals = df[df["is_signal"]].copy()
    if signals.empty:
        raise RuntimeError(f"{date_str} 沒有任何 is_signal=True 的推論記錄")
    return signals


def _download_m1_for_stocks(date_str: str, stock_ids: set) -> pd.DataFrame:
    """向 Fugle 歷史 API 下載指定股票在該日的完整分K（含收盤後資料，供事後驗證用）"""
    from data.m1_data_loader import _download_m1

    day_start = pd.Timestamp(date_str)
    day_end = day_start + pd.Timedelta(days=1)
    n = len(stock_ids)
    print(f"下載 {n} 支股票的歷史分K（約 {n * 2.1 / 60:.0f} 分鐘）...")
    frames = []
    for i, stock_id in enumerate(sorted(stock_ids), 1):
        try:
            df = _download_m1(stock_id)
        except Exception as e:
            print(f"  {stock_id} 下載失敗，略過: {e}")
            time.sleep(2.1)
            continue
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            day_df = df[(df["date"] >= day_start) & (df["date"] < day_end)]
            if not day_df.empty:
                frames.append(day_df)
        if i % 50 == 0:
            print(f"  進度 {i}/{n}")
        time.sleep(2.1)  # 維持在 Fugle 60 req/min 以內
    if not frames:
        raise RuntimeError(f"{date_str} 無任何分K資料")
    return pd.concat(frames, ignore_index=True).sort_values(["stock_id", "date"]).reset_index(drop=True)


def _evaluate_one(entry_price: float, future_closes: np.ndarray, tp_pct: float, sl_pct: float, hold_bars: int):
    """跟 strategy/date_trade_model._barrier_label_group 同一套規則。
    回傳 (outcome, exit_price, pnl_pct, bars_to_exit)；outcome="unresolved" 時後三者為 None。
    """
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)
    tp_idx = np.argmax(future_closes >= tp_price) if (future_closes >= tp_price).any() else len(future_closes)
    sl_idx = np.argmax(future_closes <= sl_price) if (future_closes <= sl_price).any() else len(future_closes)

    if tp_idx < sl_idx:
        exit_idx = int(tp_idx)
        outcome = "win"
    elif sl_idx < tp_idx:
        exit_idx = int(sl_idx)
        outcome = "loss"
    elif len(future_closes) == hold_bars:
        # 都沒碰到，且有完整 HOLD_BARS 根可看 → 用最後一根決定（跟訓練 label 邏輯一致）
        exit_idx = hold_bars - 1
        outcome = "timeout_win" if future_closes[-1] > entry_price else "timeout_loss"
    else:
        # 收盤前不足 HOLD_BARS 根、且都沒碰到 TP/SL —— 跟訓練時一樣視為無法判斷，不硬湊答案
        return "unresolved", None, None, None

    exit_price = float(future_closes[exit_idx])
    pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4)
    return outcome, exit_price, pnl_pct, exit_idx + 1


def main():
    args = parse_args()
    date_str = args.date
    print(f"驗證日期：{date_str}")

    from strategy.base.date_trade_model import HOLD_BARS, SL_PCT, TP_PCT

    signals = _load_signals(date_str)
    print(f"共 {len(signals)} 筆訊號要驗證（{signals['stock_id'].nunique()} 支股票）")

    m1 = _download_m1_for_stocks(date_str, set(signals["stock_id"].unique()))
    m1["minute"] = m1["date"].dt.strftime("%H:%M")

    rows = []
    for stock_id, group in m1.groupby("stock_id"):
        group = group.sort_values("date").reset_index(drop=True)
        closes = group["close"].to_numpy()
        minute_index = {m: i for i, m in enumerate(group["minute"])}
        stock_signals = signals[signals["stock_id"] == stock_id]
        for _, sig in stock_signals.iterrows():
            i = minute_index.get(sig["time"])
            if i is None:
                continue  # 該分鐘沒有對應K線（理論上不會發生，訊號本身就是從這批資料算出來的）
            future = closes[i + 1 : i + HOLD_BARS + 1]
            if len(future) == 0:
                outcome, exit_price, pnl_pct, bars = "unresolved", None, None, None
            else:
                outcome, exit_price, pnl_pct, bars = _evaluate_one(sig["price"], future, TP_PCT, SL_PCT, HOLD_BARS)
            rows.append(
                {
                    "date": date_str,
                    "time": sig["time"],
                    "stock_id": stock_id,
                    "name": sig["name"],
                    "proba": sig["proba"],
                    "entry_price": sig["price"],
                    "outcome": outcome,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "bars_to_exit": bars,
                }
            )

    if not rows:
        raise RuntimeError(f"{date_str} 沒有任何訊號能對應到分K資料")

    result_df = pd.DataFrame(rows)
    resolved = result_df[result_df["outcome"] != "unresolved"]
    win = resolved["outcome"].isin(["win", "timeout_win"]).sum()
    win_rate = win / len(resolved) * 100 if len(resolved) else 0.0
    print(
        f"完成驗證 {len(result_df)} 筆（{len(resolved)} 筆有結果，"
        f"{len(result_df) - len(resolved)} 筆 unresolved），勝率 {win_rate:.1f}%"
    )

    local_dir = _ROOT / "db" / "inference_eval"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{date_str}.parquet"
    result_df.to_parquet(local_path, index=False, compression="zstd")
    print(f"本地已存：{local_path}")

    repo_id = os.environ.get("HF_REPO_ID", "")
    token = os.environ.get("HF_TOKEN") or None
    if not repo_id:
        print("未設定 HF_REPO_ID，跳過上傳（只存本地）")
        return

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=f"{_HF_EVAL_PREFIX}/{date_str}.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"evaluate inference log {date_str}",
    )
    print(f"完成，已上傳至 HF: {_HF_EVAL_PREFIX}/{date_str}.parquet")


if __name__ == "__main__":
    main()
