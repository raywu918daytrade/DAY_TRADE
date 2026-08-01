"""
日K 一次性歷史回補 — 補 db/fugle_day 缺的「更早以前」歷史，跟
data/day_data_loader.py 是分開的兩支：
    data/day_data_loader.py           日常用，每天增量補「今天」缺的那一小段
    data/backfill_day_history.py      一次性用，把候選股票缺的更早歷史一次補齊

動機（2026-07-15 發現）：update_day() 沒帶 start_date 時，是用全域「最後一筆
存檔日期」當起點做增量——新加入候選清單的股票（例如某支 ETF 原本不在富邦
isNormal 清單裡，後來才開始出現）永遠不會自動往前回補，只會從被發現的那天
開始累積。實測 0050 這種大型ETF，db/fugle_day 只有 2026-05 之後的資料，
2016~2026-04 完全沒有，要用這支腳本手動回補。

候選清單預設用 data/day_data_loader.py::_all_stocks()（今天的富邦可交易清單），
所以只會抓「現在還在候選清單裡、但缺更早歷史」的股票——已經下市、不在今天
清單裡的股票不在處理範圍內（這批通常在最早的一次性歷史匯入時就已經補過，
現在只是找「新進榜但沒補到位」的漏網之魚，不是要重建整個歷史母體）。

核心下載/存檔邏輯沿用 data/day_data_loader.py 的 _download_day()/
_download_day_fubon()/_save_day()，不重寫一份。

續傳：每次執行前，先掃一次 db/fugle_day 現有資料，算出每支股票「目前最早的
存檔日期」，只對「最早日期比 start_date 晚（或完全沒有資料）」的股票，補
[start_date, 現有最早日期 - 1天]（完全沒資料時補到現在）這一段，已經補到
start_date 或更早的股票會自動跳過——中途中斷、重新執行這支腳本，會自動只
處理還沒補完的部分，不會重複下載。

用法：
    python -m data.backfill_day_history                      # 補到 2016-01-01（預設起點），今天候選清單裡缺的股票
    python -m data.backfill_day_history 2016-01-01           # 指定起點
"""

import builtins as _builtins
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pyarrow.dataset as ds
import requests

from data.day_data_loader import _ROOT, _all_stocks, _download_day, _download_day_fubon, _save_day

_TW = timezone(timedelta(hours=8))

# 加時間戳記 + 強制 flush（比照 finmind/m1_api.py 同樣的 monkey-patch 做法）。
# 這支腳本要跑很久（背景 nohup），Python 對非終端機輸出預設 block-buffered，
# 不這樣做的話 log 檔案會長時間看起來是空的，即使其實在正常運作。
_orig_print = _builtins.print


def _ts_print(*args, **kwargs):
    ts = datetime.now(_TW).strftime("%H:%M:%S")
    kwargs.setdefault("flush", True)
    _orig_print(f"[{ts}]", *args, **kwargs)


_builtins.print = _ts_print

_DEFAULT_START = "2016-01-01"  # db/fugle_day 現有歷史最早的月份附近


def _earliest_dates() -> dict[str, str]:
    """db/fugle_day 裡每支股票目前最早的存檔日期，一次掃描全部檔案（比逐支股票
    各自查一次快很多）。股票沒出現過就不在這個 dict 裡。"""
    day_dir = _ROOT / "db/fugle_day"
    if not day_dir.exists():
        return {}
    dataset = ds.dataset(str(day_dir), format="parquet")
    if dataset.count_rows() == 0:
        return {}
    df = dataset.to_table(columns=["stock_id", "date"]).to_pandas()
    return df.groupby("stock_id")["date"].min().to_dict()


def _plan_targets(start_date: str, stocks: list) -> list:
    """回傳 (stock_id, start_date, end_date) 清單。end_date=None 代表補到現在
    （股票完全沒有資料時）；有既有資料的股票，end_date 是「現有最早日期的前一天」，
    不重複下載已經有的部分。已經補到 start_date（或更早）的股票不會出現在清單裡。"""
    earliest = _earliest_dates()
    targets = []
    for sid in stocks:
        existing_min = earliest.get(sid)
        if existing_min is None:
            targets.append((sid, start_date, None))
        elif existing_min > start_date:
            end = (pd.Timestamp(existing_min) - timedelta(days=1)).strftime("%Y-%m-%d")
            targets.append((sid, start_date, end))
    return targets


def _backfill_fugle(targets: list, workers: int):
    """比照 data/day_data_loader.py::_update_day_fugle()，只是每支股票的
    start/end 各自不同（來自 _plan_targets()），不是全部套同一個 start_date。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = 0

    def _fetch_one(item):
        nonlocal completed
        sid, start, end = item
        time.sleep(0.2)  # 每執行緒小延遲，5 執行緒合計 ~1 req/s
        try:
            df = _download_day(sid, start, end_date=end)
            if not df.empty:
                _save_day(df)
            return sid, len(df), None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return sid, 0, None
            return sid, 0, str(e)
        except Exception as e:
            return sid, 0, str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, item): item[0] for item in targets}
        for fut in as_completed(futures):
            sid, rows, err = fut.result()
            completed += 1
            if err:
                print(f"  [Fugle {completed}/{len(targets)}] {sid} 失敗: {err}")
            elif completed % 20 == 0 or completed == len(targets):
                print(f"  [Fugle {completed}/{len(targets)}] 進度更新（{sid}：新增 {rows} 筆）")


def _backfill_fubon(targets: list, sdk):
    """比照 data/day_data_loader.py::_update_day_fubon()。"""
    for i, (sid, start, end) in enumerate(targets, 1):
        try:
            df = _download_day_fubon(sdk, sid, start, end_date=end)
            if not df.empty:
                _save_day(df)
            if i % 20 == 0 or i == len(targets):
                print(f"  [富邦 {i}/{len(targets)}] 進度更新（{sid}：新增 {len(df)} 筆）")
        except Exception as e:
            print(f"  [富邦 {i}/{len(targets)}] {sid} 失敗: {e}")
        time.sleep(1.05)  # 維持 60 req/min 以內留緩衝，比照 _update_day_fubon()


def backfill_day_history(
    start_date: str = _DEFAULT_START, stocks: list = None, workers: int = 5, fugle_share: float = 0.75
):
    """把 candidates（預設今天的候選清單，見 _all_stocks()）裡還沒回補到
    start_date 的股票，缺的那一段一次補齊。

    fugle_share: 選填，分給 Fugle（併發、較快）那一半的比例，其餘給富邦
    （序列化、較慢）。2026-07-15 實測：對「一支股票要切成多個年度區間」的
    歷史回補場景，Fugle（5併發+年度間隔0.2秒）實測吞吐量約是富邦（序列化+
    年度間隔1.05秒）的3倍，50/50對分會讓富邦那一半拖累總時間到近5小時；
    改成 Fugle:富邦 ≈ 75:25，兩邊完成時間打平，總時間可以縮短到約一半。
    update_day() 的日常增量（每支股票通常只需要1個區間）不受影響，還是
    50/50——那邊瓶頸不一樣，這個參數只在這支腳本用。"""
    candidates = _all_stocks() if stocks is None else stocks
    targets = _plan_targets(start_date, candidates)
    print(f"候選 {len(candidates)} 支，其中 {len(targets)} 支缺 {start_date} 之前的歷史")
    if not targets:
        print("全部都已經補到位，不用下載")
        return

    split = round(len(targets) * fugle_share)
    fugle_targets, fubon_targets = targets[:split], targets[split:]
    print(f"Fugle {len(fugle_targets)} 支（{workers} 並發）、富邦 {len(fubon_targets)} 支，同時下載...")

    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    trade_api.init_market_data(sdk)
    try:
        t_fugle = threading.Thread(target=_backfill_fugle, args=(fugle_targets, workers))
        t_fubon = threading.Thread(target=_backfill_fubon, args=(fubon_targets, sdk))
        t_fugle.start()
        t_fubon.start()
        t_fugle.join()
        t_fubon.join()
    finally:
        trade_api.logout(sdk)

    print("歷史回補完成（重跑這支腳本會自動只處理還沒補完的部分）")


if __name__ == "__main__":
    _start = sys.argv[1] if len(sys.argv) >= 2 else _DEFAULT_START
    backfill_day_history(_start)
