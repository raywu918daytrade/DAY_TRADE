"""FinMind 分K 擴大回補用的固定股票清單（2026-08-06加）— 跟
finmind/tick_universe.py 那400支「依成交量排名」的清單不同，這份是
db/tickers/tickers.parquet（全市場代號參考表）篩選出的**所有**4碼一般個股，
不用排名、不看成交量，過濾規則只有：

    1. stock_id 是4碼數字（regex ^\\d{4}$）
    2. 不是"00"開頭的ETF代號（過濾邏輯跟 tick_universe.py 一致，見該檔
       docstring 的說明：db/tickers/tickers.parquet 裡"00"開頭的代號
       industry欄位都是"00"，沒有一般個股用這個開頭）

2026-08-06 實測：db/tickers/tickers.parquet 共2170筆，篩選後剩1877支
（TWSE + TPEx 4碼一般個股），供 finmind/backfill_m1_history_2000.py 使用，
補完400支之外的其餘一般個股歷史分K。

固定存檔的原因跟 tick_universe.py 一樣：避免每次執行 backfill 都重新讀
tickers.parquet 篩選一次，也避免 tickers.parquet 之後更新（例如新股上市/
下市）導致清單中途變動、資料補一半基準跑掉。

用法：
    python -m finmind.stock_universe_2000   # 算一次、存到 db/tickers/stock_universe_2000.parquet

之後其他程式要讀這份固定清單，呼叫 load_stock_universe_2000()，不要重新呼叫
build_stock_universe_2000()。
"""

import re
from pathlib import Path

import pandas as pd

from finmind.m1_api import _atomic_to_parquet, _ROOT

_FOUR_DIGIT = re.compile(r"^\d{4}$")
_ETF_PREFIX = "00"  # db/tickers/tickers.parquet 驗證過：00開頭的代號industry都是"00"（ETF），沒有一般個股用這個開頭


def _universe_file_path() -> Path:
    return _ROOT / "db/tickers/stock_universe_2000.parquet"


def build_stock_universe_2000() -> pd.DataFrame:
    """讀 db/tickers/tickers.parquet，篩4碼數字股票、排除"00"開頭的ETF代號，
    輸出 stock_id, exchange, name, industry 幾欄（沒有排名/成交量，因為這份
    清單不是top-N，是「全部留下」）。"""
    df = pd.read_parquet(_ROOT / "db/tickers/tickers.parquet")
    mask = df["stock_id"].str.match(_FOUR_DIGIT) & ~df["stock_id"].str.startswith(_ETF_PREFIX)
    return df.loc[mask, ["stock_id", "exchange", "name", "industry"]].reset_index(drop=True)


def load_stock_universe_2000() -> list[str]:
    """讀 db/tickers/stock_universe_2000.parquet，回傳 stock_id list，給
    finmind/backfill_m1_history_2000.py 用。"""
    path = _universe_file_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，請先執行 `python -m finmind.stock_universe_2000` 產生清單")
    return pd.read_parquet(path, columns=["stock_id"])["stock_id"].tolist()


def load_stock_universe_2000_with_0050() -> list[str]:
    """load_stock_universe_2000() 額外強制併入 0050——那份清單的篩選規則排除
    所有 ETF（含0050），但0050是系統裡的大盤參考指標（idx_gap_pct等特徵、
    live_trader.py/data_manager.py都依賴它），需要持續下載/更新。比照
    finmind/tick_universe.py::_FORCE_INCLUDE 的做法（2026-08-08加），統一供
    data/day_data_loader.py、data/m1_data_loader.py、fubon/tick_api.py
    共用，不要各自複製這段邏輯。"""
    stocks = load_stock_universe_2000()
    if "0050" not in stocks:
        stocks = stocks + ["0050"]
    return stocks


if __name__ == "__main__":
    universe = build_stock_universe_2000()
    path = _universe_file_path()
    _atomic_to_parquet(universe, path, index=False, compression="zstd")
    print(f"已寫入 {len(universe)} 支股票到 {path}")
    print(universe.head(10))
    print("...")
    print(universe.tail(5))
