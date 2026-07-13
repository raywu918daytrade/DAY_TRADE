"""
回補指定交易日的每分鐘推論記錄，補進 HF Hub（跟 live_trader.py 即時記錄的路徑一致）。

用途：push_inference_log() 上線之前的交易日（或中途中斷漏記的交易日），
收盤後重新跑一次模型，把結果補進 HF：
    day_trade/inference/{date}.parquet

跟 on_minute() 的差異：
    on_minute() 每分鐘拿到的 m1_live 只累積到當下那一分鐘；
    這裡收盤後才跑，拿到的是當日完整分K。但 make_features() 的 rolling/cumsum
    都只往回看、不用未來資料，所以逐分鐘用同一份完整資料算出來的結果，
    跟當時即時算的完全一樣，不會有 look-ahead 問題。

當沖名單的來源：
    一律用「執行當下」的當沖候選清單 + 均量過濾（吃 .env 的 MAX_SUBSCRIPTIONS / MIN_AVG_VOL_LOTS）
    重建名單，再向 Fugle 歷史 API 下載這批股票的分K。
    不信任本機 db/m1_live/{date}.parquet 既有的快取——那可能是舊 .env 設定下的本機測試殘留
    （例如 MAX_SUBSCRIPTIONS 曾經被改成別的值），跟正式環境當天實際使用的名單不一定一致。

只回補 SESSION_START ~ SESSION_END 訊號時段（跟 on_minute() 一致，之後只做 SL/TP，
不產生新推論）。

用法：
    python scripts/backfill_inference_log.py --date 2026-07-03
    python scripts/backfill_inference_log.py            # 預設回補最近一個交易日（跳過六日）
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))
_HF_INFERENCE_PREFIX = "day_trade/inference"  # 跟 api.py 的 _HF_INFERENCE_PREFIX 保持一致


def _default_date() -> str:
    d = datetime.now(_TW).date() - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def parse_args():
    p = argparse.ArgumentParser(description="回補指定日期的每分鐘推論記錄到 HF Hub")
    p.add_argument("--date", default=None, help="YYYY-MM-DD，預設為最近一個交易日（跳過六日）")
    p.add_argument("--threshold", type=float, default=None, help="訊號門檻，預設讀 .env 的 THRESHOLD")
    return p.parse_args()


def _build_universe_and_download(date_str: str, tickers_df: pd.DataFrame) -> tuple[set, pd.DataFrame]:
    """用「執行當下」的當沖候選清單 + 均量過濾（吃現在的 .env）重建名單，
    再向 Fugle 歷史 API 下載這批股票該日的分K。
    """
    from data.data_manager import load_d1
    from data.m1_data_loader import _download_m1

    candidate_stocks = set(tickers_df["stock_id"])
    _, day_trade_stocks = load_d1(candidate_stocks)
    if not day_trade_stocks:
        raise RuntimeError("均量過濾後無標的")
    print(f"  → 均量過濾後 {len(day_trade_stocks)} 支")

    n = len(day_trade_stocks)
    print(f"下載 {n} 支股票的歷史分K（約 {n * 2.1 / 60:.0f} 分鐘）...")
    day_start = pd.Timestamp(date_str)
    day_end = day_start + pd.Timedelta(days=1)
    frames = []
    for i, stock_id in enumerate(sorted(day_trade_stocks), 1):
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
        raise RuntimeError(f"{date_str} 無任何分K資料，可能非交易日或超過 Fugle 近30日範圍")
    m1_live = pd.concat(frames, ignore_index=True).sort_values(["stock_id", "date"]).reset_index(drop=True)
    return day_trade_stocks, m1_live


def main():
    args = parse_args()
    date_str = args.date or _default_date()
    threshold = args.threshold if args.threshold is not None else float(os.environ.get("THRESHOLD", "0.55"))
    print(f"回補日期：{date_str}（訊號門檻僅影響 is_signal 標記，推論記錄本身門檻恆為 0）")

    from data.data_manager import load_d1
    from data.fugle_tickers import update_tickers
    from strategy.rally.live import SESSION_END, SESSION_START, load_model, predict_live

    # 1. 當沖候選清單（名稱查詢也用這份）
    print("取得當沖候選清單...")
    tickers_df = update_tickers()
    if tickers_df.empty:
        raise RuntimeError("無法取得當沖候選清單（tickers 為空，Fugle 可能非開放時段）")
    tickers = tickers_df.set_index("stock_id")["name"].to_dict()

    # 2. 均量過濾出訂閱名單 + 向 Fugle 下載該日分K
    print("載入日K並做均量過濾...")
    day_trade_stocks, m1_live = _build_universe_and_download(date_str, tickers_df)
    print(f"共取得 {len(m1_live):,} 筆分K，{m1_live['stock_id'].nunique()} 支股票")

    # 3. D1 日K（模型特徵用）
    day, _ = load_d1(day_trade_stocks)

    # 4. 逐分鐘重跑推論（跟 on_minute() 共用 predict_live，threshold=0 取得所有股票機率）
    model = load_model()
    start_min = SESSION_START[0] * 60 + SESSION_START[1]
    end_min = SESSION_END[0] * 60 + SESSION_END[1]

    rows = []
    for m in range(start_min, end_min + 1):
        h, mm = divmod(m, 60)
        minute_str = f"{date_str} {h:02d}:{mm:02d}:00"
        all_results = predict_live(
            minute_str,
            day,
            model=model,
            threshold=0,
            day_trade_stocks=day_trade_stocks,
            m1_live=m1_live,
        )
        if not all_results:
            continue
        for r in all_results:
            rows.append(
                {
                    "date": date_str,
                    "time": minute_str[11:16],
                    "stock_id": r["stock_id"],
                    "name": tickers.get(r["stock_id"], r["stock_id"]),
                    "proba": round(float(r["proba"]), 4),
                    "price": r.get("price"),
                    "direction": r.get("direction", "buy"),
                    "is_signal": bool(r["proba"] >= threshold),
                }
            )
        print(f"  {minute_str[11:16]}  {len(all_results)} 支")

    if not rows:
        raise RuntimeError(f"{date_str} 09:01~{SESSION_END[0]:02d}:{SESSION_END[1]:02d} 沒有算出任何推論結果")

    # 5. 寫本地 parquet + 上傳 HF（路徑跟 api.py 的即時記錄一致：day_trade/inference/{date}.parquet）
    result_df = pd.DataFrame(rows)
    local_dir = _ROOT / "db" / "inference_live"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{date_str}.parquet"
    result_df.to_parquet(local_path, index=False, compression="zstd")
    print(f"本地已存：{local_path}（{len(result_df):,} 筆）")

    repo_id = os.environ.get("HF_REPO_ID", "")
    token = os.environ.get("HF_TOKEN") or None
    if not repo_id:
        print("未設定 HF_REPO_ID，跳過上傳（只存本地）")
        return

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=f"{_HF_INFERENCE_PREFIX}/{date_str}.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"backfill inference log {date_str}",
    )
    print(f"完成，已上傳至 HF: {_HF_INFERENCE_PREFIX}/{date_str}.parquet")


if __name__ == "__main__":
    main()
