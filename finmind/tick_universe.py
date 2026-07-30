"""FinMind Tick 回補用的固定股票清單 — 跟分K的 top_n_by_volume（每月動態重算）
不同，tick 回補整段 2025-08~2026-07 都要用「同一批」股票，所以清單只算一次、
存檔，之後 backfill_tick_history.py 每個月都讀同一份，不會因為某個月成交量
排名變動就換掉補的股票。

清單定義：db/fugle_day/2026_01.parquet ~ 2026_06.parquet 這6個月，4碼數字股票
（regex ^\\d{4}$，排除ETF/權證等非4碼代號）依平均日成交量排序，取前800名，
再強制併入 0050（即使 0050 沒過4碼regex——ETF代號規則是「00+3碼」共5碼，
0050 剛好是4碼的例外，見 fubon/subscribe_list.py 的既有註解）。

用法：
    python -m finmind.tick_universe   # 算一次、存到 db/tickers/tick_universe.parquet

之後其他程式要讀這份固定清單，呼叫 load_tick_universe()，不要重新呼叫
build_tick_universe()（那樣清單會因為之後又多了幾個月的 db/fugle_day 資料
而變動，違背「固定清單」的原意）。
"""

import re
from pathlib import Path

import pandas as pd

from finmind.finmind_api import _atomic_to_parquet, _ROOT

_TOP_N = 800
_FORCE_INCLUDE = ["0050"]
_DEFAULT_YEAR_MONTHS = [(2026, m) for m in range(1, 7)]
_FOUR_DIGIT = re.compile(r"^\d{4}$")


def _universe_file_path() -> Path:
    return _ROOT / "db/tickers/tick_universe.parquet"


def build_tick_universe(
    year_months: list[tuple[int, int]] = _DEFAULT_YEAR_MONTHS,
    top_n: int = _TOP_N,
    force_include: list[str] = _FORCE_INCLUDE,
) -> pd.DataFrame:
    """讀 year_months 每個月的 db/fugle_day/{year}_{month}.parquet（只取
    stock_id/date/volume 三欄，比照 finmind_api._month_universe() 的做法），
    篩 4碼數字股票，算跨月平均日成交量排序。force_include 的股票（例如
    0050）先從排名池排除，取「純排名」前 top_n 名，再把 force_include 額外
    併入——確保輸出永遠是 top_n + len(force_include) 檔（預設800+1=801），
    不會因為 force_include 剛好本來就排得進前top_n名而被「吃掉一個名額」。

    輸出欄位：stock_id, avg_volume, rank, forced_include。前 top_n 列
    rank=1~top_n、forced_include=False；force_include 的列 rank=NaN、
    forced_include=True，avg_volume 仍然照實填（純粹沒被排進top_n名額，
    不代表沒有成交量資料）——保留這些欄位是為了之後回頭稽核「這份清單是
    怎麼算出來的」，不是直接存一份沒有脈絡的股票代號清單。
    """
    frames = []
    for year, month in year_months:
        path = _ROOT / f"db/fugle_day/{year}_{month:02d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在，無法計算 tick universe")
        frames.append(pd.read_parquet(path, columns=["stock_id", "date", "volume"]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["stock_id"].str.match(_FOUR_DIGIT)]

    avg_vol = df.groupby("stock_id")["volume"].mean().sort_values(ascending=False)
    # force_include 的股票（例如0050）先從排名池排除再取前top_n名，這樣
    # 「前800名 + 強制併入」永遠是 top_n + len(force_include) 檔，不會因為
    # force_include剛好本來就排得進前top_n名而變成「800檔裡面含0050」，
    # 使用者要的是800純排名 + 0050額外併入，共801檔。
    ranked_pool = avg_vol.drop(index=force_include, errors="ignore")
    ranked = ranked_pool.head(top_n)
    result = pd.DataFrame(
        {
            "stock_id": ranked.index,
            "avg_volume": ranked.values,
            "rank": range(1, len(ranked) + 1),
            "forced_include": False,
        }
    )

    extra_df = pd.DataFrame(
        {
            "stock_id": force_include,
            "avg_volume": avg_vol.reindex(force_include).values,
            "rank": pd.NA,
            "forced_include": True,
        }
    )
    return pd.concat([result, extra_df], ignore_index=True)


def load_tick_universe() -> list[str]:
    """讀 db/tickers/tick_universe.parquet，回傳 stock_id list，給
    backfill_tick.py / backfill_tick_history.py 用。"""
    path = _universe_file_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，請先執行 `python -m finmind.tick_universe` 產生清單")
    return pd.read_parquet(path, columns=["stock_id"])["stock_id"].tolist()


if __name__ == "__main__":
    universe = build_tick_universe()
    path = _universe_file_path()
    _atomic_to_parquet(universe, path, index=False, compression="zstd")
    print(f"已寫入 {len(universe)} 支股票到 {path}")
    print(universe.head(10))
    print("...")
    print(universe.tail(5))
