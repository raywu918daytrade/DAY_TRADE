"""FinMind Tick 回補用的固定股票清單 — 跟分K的 top_n_by_volume（每月動態重算）
不同，tick 回補整段 2025-08~2026-07 都要用「同一批」股票，所以清單只算一次、
存檔，之後 backfill_tick_history.py 每個月都讀同一份，不會因為某個月成交量
排名變動就換掉補的股票。

清單定義：db/fugle_day/2026_01.parquet ~ 2026_06.parquet 這6個月，4碼數字股票
依平均日成交量排序，取前399名，再強制併入 0050，共400檔。

2026-07-30 發現：光是「4碼數字」（regex ^\\d{4}$）不夠，早期上市的ETF代號
（0050/0052/0056等）剛好也是4碼、會混進純股票的排名池——查過
db/tickers/tickers.parquet，凡是 stock_id 開頭"00"的，industry欄位都是"00"
（沒有任何一般個股用這個開頭），所以額外排除所有"00"開頭的代號，只留
0050 這一個明確要的例外（強制併入，不受排名池排除影響）。

2026-07-30：從800調降到400——800檔涵蓋2026-01~06累積成交量的100%，但87%的
量前300名就拿到了、92%前400名就拿到了，超過400名之後邊際貢獻明顯遞減（每多
50名只增加不到5個百分點），但回補時間是跟股票數線性成長，縮到400檔可以把
全部12個月的回補時間從約35小時砍到約17.5小時，CP值划算很多。

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

_TOP_N = 400  # 輸出總檔數（含force_include），不是「純排名檔數 + force_include」
_FORCE_INCLUDE = ["0050"]
_DEFAULT_YEAR_MONTHS = [(2026, m) for m in range(1, 7)]
_FOUR_DIGIT = re.compile(r"^\d{4}$")
_ETF_PREFIX = "00"  # db/tickers/tickers.parquet 驗證過：00開頭的代號industry都是"00"（ETF），沒有一般個股用這個開頭


def _universe_file_path() -> Path:
    return _ROOT / "db/tickers/tick_universe.parquet"


def build_tick_universe(
    year_months: list[tuple[int, int]] = _DEFAULT_YEAR_MONTHS,
    top_n: int = _TOP_N,
    force_include: list[str] = _FORCE_INCLUDE,
) -> pd.DataFrame:
    """讀 year_months 每個月的 db/fugle_day/{year}_{month}.parquet（只取
    stock_id/date/volume 三欄，比照 finmind_api._month_universe() 的做法），
    篩 4碼數字股票、排除"00"開頭的ETF代號（見上面模組docstring的說明，
    0050/0052/0056這種早期ETF代號剛好也是4碼，不能只靠regex ^\\d{4}$濾掉），
    算跨月平均日成交量排序，取前 (top_n - len(force_include)) 名，
    再把 force_include（例如0050）併入——確保輸出**總數固定是 top_n**
    （預設400，399純排名+1個強制併入），不是 top_n 之外再加碼。

    輸出欄位：stock_id, avg_volume, rank, forced_include。純排名列
    rank=1~(top_n-len(force_include))、forced_include=False；
    force_include 的列 rank=NaN、forced_include=True，avg_volume 仍然
    照實填（純粹沒被排進排名名額，不代表沒有成交量資料）——保留這些欄位
    是為了之後回頭稽核「這份清單是怎麼算出來的」，不是直接存一份沒有
    脈絡的股票代號清單。
    """
    frames = []
    for year, month in year_months:
        path = _ROOT / f"db/fugle_day/{year}_{month:02d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在，無法計算 tick universe")
        frames.append(pd.read_parquet(path, columns=["stock_id", "date", "volume"]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["stock_id"].str.match(_FOUR_DIGIT)]
    # avg_vol_all 包含ETF（給 force_include 查詢實際成交量用，例如0050要
    # 顯示真實avg_volume，不能因為它被排名池排除就查不到），avg_vol 是排除
    # 掉"00"開頭ETF代號之後、真正拿來排名選股的池子。
    avg_vol_all = df.groupby("stock_id")["volume"].mean()
    avg_vol = (
        df[~df["stock_id"].str.startswith(_ETF_PREFIX)]
        .groupby("stock_id")["volume"]
        .mean()
        .sort_values(ascending=False)
    )
    ranked = avg_vol.head(top_n - len(force_include))
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
            "avg_volume": avg_vol_all.reindex(force_include).values,
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
