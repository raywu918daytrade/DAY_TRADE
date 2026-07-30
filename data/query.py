"""
本地資料讀取工具（唯讀）

功能：
    從本地 parquet 檔載入三種資料，供訓練腳本與即時推論使用。
    不負責下載或更新資料，只做讀取。

三種資料對應：
    load_m1()      → db/m1/        歷史分K（訓練用，2787支，由 GHA 每日更新）
    load_day()     → db/fugle_day/ 日K（模型特徵用，GHA 每日更新）
    load_m1_live() → db/m1_live/   今日即時分K（交易用，500支，收盤後丟棄）

單支股票查詢（用 pyarrow filter pushdown，不用像 load_day() 整個資料集讀進記憶體）：
    load_day_by_stock(stock_id)  → db/fugle_day/ 單一股票的全部日K
    load_tick_by_stock(stock_id) → db/tick/      單一股票的逐筆成交明細（見下方
                                    load_tick_by_stock() 說明，tick資料量太大
                                    不提供整個資料集一次讀進記憶體的版本）
    load_volume_profile()        → db/volume_profile/ 價位成交量分布 (Volume Profile)
    load_poc()                   → db/poc_day/        每日 POC 關鍵價位 (Point of Control)
"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

_ROOT = Path(__file__).parent.parent


def _dataset_paths(dir_path: Path, start_date: str | None) -> str | list[str]:
    """依 start_date 決定要讀哪些月份分檔的 parquet（2026-07-25討論）：這些
    db/m1、db/fugle_day 等資料夾都是按月分檔（檔名格式 YYYY_MM.parquet），
    即時推論通常只需要最近幾十天的資料算 rolling 特徵，不需要每次都把
    資料庫存在以來的全部歷史（可能好幾年）都讀進記憶體——2026-07-25 實測
    live_trader.py 因為這樣吃了快 6GB RSS。

    start_date=None（預設）：回傳整個資料夾路徑，pyarrow 讀全部檔案，行為
    跟改之前完全一樣，訓練/回測腳本不用改就不受影響。
    start_date="YYYY-MM-DD"：只回傳「start_date 所在月份」到「最新月份」
    這段範圍的檔案路徑（按月份粒度篩選，不逐筆篩到剛好那一天——多讀到
    月初那幾天不影響 rolling 特徵計算，換取不用逐檔讀取後再篩選的複雜度）。
    """
    if start_date is None:
        return str(dir_path)
    cutoff = pd.Timestamp(start_date).strftime("%Y_%m")
    files = sorted(f for f in dir_path.iterdir() if f.suffix == ".parquet")
    return [str(f) for f in files if f.stem >= cutoff]


def load_m1(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/m1/ 分K（訓練資料，~2787 支，按月分檔）。

    start_date：選填 "YYYY-MM-DD"，只讀該月到最新月份的檔案，不重複載入
    整個歷史（見 _dataset_paths() 的說明）；預設 None = 讀全部（訓練/回測
    既有行為不變）。"""
    paths = _dataset_paths(_ROOT / "db/m1", start_date)
    if not paths:
        return pd.DataFrame()
    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    # 按月分檔的 parquet 中 date 欄位型別可能不一致（string / timestamp 混雜），
    # 用 format="mixed" 讓 pandas 逐筆判斷格式，避免鎖死單一格式時遇到例外格式就整批炸掉
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_day(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/fugle_day/ 日K（模型特徵用，按月分檔）。

    start_date：同 load_m1() 的說明，預設 None = 讀全部。"""
    paths = _dataset_paths(_ROOT / "db/fugle_day", start_date)
    if not paths:
        return pd.DataFrame()
    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    # 同上：按月分檔可能混雜不同型別的 date 欄位，用 format="mixed" 容忍
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_day_by_stock(stock_id: str, date: str = None) -> pd.DataFrame:
    """載入單一股票在 db/fugle_day/ 的日K，只讀該股票的 row group，不用像
    load_day() 一樣把全市場都讀進記憶體，適合只需要單支股票時用（例如查前一
    交易日收盤價）。

    date: 選填，格式 "YYYY-MM-DD"，指定只回傳該日那一筆（同樣走 pyarrow filter
    pushdown，不用先讀全部再篩）；不填則回傳該股票全部日K（依日期排序）。
    查無資料一律回傳空 DataFrame（欄位跟 load_day() 一致）。"""
    dataset = ds.dataset(str(_ROOT / "db/fugle_day"), format="parquet")
    filt = ds.field("stock_id") == stock_id
    if date is not None:
        filt = filt & (ds.field("date") == date)
    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
    df = table.to_pandas()
    # 同 load_day()：按月分檔可能混雜不同型別的 date 欄位，用 format="mixed" 容忍
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    return df.sort_values("date").reset_index(drop=True)


def load_tick_by_stock(stock_id: str, date: str = None) -> pd.DataFrame:
    """載入單一股票在 db/tick/ 的逐筆成交明細（FinMind TaiwanStockPriceTick，
    見 finmind/tick_api.py），只讀該股票的 row group，不提供整個資料集一次
    讀進記憶體的版本——tick資料量級跟分K差太多（一個月400檔股票就有2000多萬
    列），像 load_m1() 那樣整個資料夾讀進記憶體會是幾十GB，不現實。

    date: 選填，格式 "YYYY-MM-DD"，只回傳該日的tick；不填則回傳該股票在
    db/tick/ 裡全部月份的tick（小心：熱門股全部12個月的tick可能是數十萬列，
    沒有 date 就近一步限制的話記憶體用量會偏高）。查無資料回傳空 DataFrame
    （欄位跟 db/tick 一致：stock_id, date, deal_price, volume, tick_type）。

    date 欄位是完整時間字串（"YYYY-MM-DD HH:MM:SS.ffffff"，見
    finmind/tick_api.py::save_tick() 的說明），不是純日期，不能像
    load_day_by_stock() 那樣用 == 比對，改用當天 00:00:00~23:59:59.999999
    的字串範圍（欄位是固定寬度的ISO格式字串，字典序比較等同時間先後順序，
    pyarrow 一樣能靠 row group 的 min/max 統計做 filter pushdown）。
    """
    dataset = ds.dataset(str(_ROOT / "db/tick"), format="parquet")
    filt = ds.field("stock_id") == stock_id
    if date is not None:
        filt = filt & (ds.field("date") >= f"{date} 00:00:00") & (ds.field("date") <= f"{date} 23:59:59.999999")
    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["stock_id", "date", "deal_price", "volume", "tick_type"])
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    return df.sort_values("date").reset_index(drop=True)


def load_m3(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/m3/ 3 分鐘K，rolling 版本，每分鐘一列（由
    build_m3_m5_rolling.py 預先聚合）。

    start_date：同 load_m1() 的說明，預設 None = 讀全部。"""
    path = _ROOT / "db/m3"
    if not path.exists():
        return pd.DataFrame()
    paths = _dataset_paths(path, start_date)
    if not paths:
        return pd.DataFrame()
    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m5(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/m5/ 5 分鐘K，rolling 版本，每分鐘一列（由
    build_m3_m5_rolling.py 預先聚合）。

    start_date：同 load_m1() 的說明，預設 None = 讀全部。"""
    path = _ROOT / "db/m5"
    if not path.exists():
        return pd.DataFrame()
    paths = _dataset_paths(path, start_date)
    if not paths:
        return pd.DataFrame()
    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m3_std(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/m3_std/ 標準獨立 3 分K棒，一根K棒一列（由
    build_m3_m5_std.py 預先聚合）。

    start_date：同 load_m1() 的說明，預設 None = 讀全部。"""
    path = _ROOT / "db/m3_std"
    if not path.exists():
        return pd.DataFrame()
    paths = _dataset_paths(path, start_date)
    if not paths:
        return pd.DataFrame()
    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m5_std(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/m5_std/ 標準獨立 5 分K棒，一根K棒一列（由
    build_m3_m5_std.py 預先聚合）。

    start_date：同 load_m1() 的說明，預設 None = 讀全部。"""
    path = _ROOT / "db/m5_std"
    if not path.exists():
        return pd.DataFrame()
    paths = _dataset_paths(path, start_date)
    if not paths:
        return pd.DataFrame()
    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m1_live(date: str = None) -> pd.DataFrame:
    """載入今日即時分K（db/m1_live/YYYY-MM-DD.parquet），盤後自動 backfill 補齊"""
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = _ROOT / f"db/m1_live/{date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_volume_profile(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """載入 db/volume_profile/ 價位成交量分布 (Volume Profile)。

    stock_id: 選填，指定股票代號（走 pyarrow filter pushdown）
    date: 選填，格式 "YYYY-MM-DD"，指定交易日（走 pyarrow filter pushdown）
    start_date: 選填，格式 "YYYY-MM-DD"，依月份載入 start_date 之後的檔案

    回傳欄位：stock_id, date, price, volume, buy_volume, sell_volume, neutral_volume
    """
    path = _ROOT / "db/volume_profile"
    if not path.exists():
        return pd.DataFrame(
            columns=["stock_id", "date", "price", "volume", "buy_volume", "sell_volume", "neutral_volume"]
        )

    # 依 start_date 決定要讀哪些月份分檔
    # 若指定了 date，以 date 所在月份縮小讀取範圍
    eff_start = start_date
    if date is not None and (eff_start is None or date < eff_start):
        eff_start = date

    paths = _dataset_paths(path, eff_start)
    if not paths:
        return pd.DataFrame(
            columns=["stock_id", "date", "price", "volume", "buy_volume", "sell_volume", "neutral_volume"]
        )

    dataset = ds.dataset(paths, format="parquet")
    filt = None
    if stock_id is not None:
        filt = ds.field("stock_id") == stock_id
    if date is not None:
        filt = (filt & (ds.field("date") == date)) if filt is not None else (ds.field("date") == date)

    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(
            columns=["stock_id", "date", "price", "volume", "buy_volume", "sell_volume", "neutral_volume"]
        )

    df = table.to_pandas()
    df.drop_duplicates(subset=["stock_id", "date", "price"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date", "price"]).reset_index(drop=True)


def load_poc(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """載入 db/poc_day/ 每日 POC 關鍵價位 (Point of Control 與 Value Area)。

    stock_id: 選填，指定股票代號（走 pyarrow filter pushdown）
    date: 選填，格式 "YYYY-MM-DD"，指定交易日（走 pyarrow filter pushdown）
    start_date: 選填，格式 "YYYY-MM-DD"，依月份載入 start_date 之後的檔案

    回傳欄位：stock_id, date, poc, poc_volume, pocs, poc_count, profile_type, vah, val, total_volume
    """
    path = _ROOT / "db/poc_day"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "stock_id",
                "date",
                "poc",
                "poc_volume",
                "pocs",
                "poc_count",
                "profile_type",
                "vah",
                "val",
                "total_volume",
            ]
        )

    eff_start = start_date
    if date is not None and (eff_start is None or date < eff_start):
        eff_start = date

    paths = _dataset_paths(path, eff_start)
    if not paths:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "date",
                "poc",
                "poc_volume",
                "pocs",
                "poc_count",
                "profile_type",
                "vah",
                "val",
                "total_volume",
            ]
        )

    dataset = ds.dataset(paths, format="parquet")
    filt = None
    if stock_id is not None:
        filt = ds.field("stock_id") == stock_id
    if date is not None:
        filt = (filt & (ds.field("date") == date)) if filt is not None else (ds.field("date") == date)

    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "date",
                "poc",
                "poc_volume",
                "pocs",
                "poc_count",
                "profile_type",
                "vah",
                "val",
                "total_volume",
            ]
        )

    df = table.to_pandas()
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


