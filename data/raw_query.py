"""
底層原始資料讀取工具（唯讀，2026-08-01 從 data/query.py 拆出來，2026-08-03
加 load_day()/load_day_by_stock()）

這裡的函式回傳的都是**未還原權息**的原始價格（db/m1、db/m3、db/m5、
db/m3_std、db/m5_std、db/d1 這些K線本身就是原始價格；db/volume_profile、
db/poc_day 是從 db/tick 原始成交價算出來的，也是原始價格）。

一般情況不應該直接呼叫這裡的函式——絕大多數需要價格資料的地方（訓練特徵、
pattern 圖表、任何要跟日K比較/一起用的場合）都應該用 data/query.py 對應的
同名函式（那邊回傳的是還原權息後、可以直接跟 db/fugle_day 一起用的版本，
內部就是呼叫這裡的函式再 join db/tick_adjust_factor 換算）。

只有明確知道自己需要原始價格的地方才該直接 import 這支檔案。

不負責下載或更新資料，只做讀取。
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


def _month_file_list(dir_path: Path, start_date: str | None) -> list[str]:
    """跟 _dataset_paths() 一樣依 start_date 篩月份檔案，但一律回傳**檔案
    路徑清單**（不會像 _dataset_paths() 在 start_date=None 時偷懶回傳整個
    資料夾字串），iter_m1_months() 等逐月yield的函式需要真的逐檔迭代，不能
    把一個目錄字串直接丟給 ds.dataset() 當「單一檔案」處理。"""
    if not dir_path.exists():
        return []
    if start_date is None:
        return sorted(str(f) for f in dir_path.iterdir() if f.suffix == ".parquet")
    cutoff = pd.Timestamp(start_date).strftime("%Y_%m")
    files = sorted(f for f in dir_path.iterdir() if f.suffix == ".parquet")
    return [str(f) for f in files if f.stem >= cutoff]


def load_m1(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/m1/ 分K（原始價格，未還原權息，訓練資料，~2787 支，按月分檔）。

    一般情況請用 data/query.py::load_m1()（還原權息後版本），不要直接呼叫
    這支，除非明確需要原始價格。

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


def iter_m1_months(start_date: str | None = None):
    """逐月yield db/m1/分K（原始價格），跟 load_m1() 用同一套dtype正規化/
    去重邏輯，但**不會把所有月份一次讀進記憶體**——呼叫端（例如
    strategy/vwap_ml/train.py::_prepare_data()）需要對「當月裁到session
    window後的資料量」跑pipeline時，一次讀全部月份（load_m1()）會讓峰值
    記憶體卡在裁切之前，逐月處理才能把峰值壓到單月等級（2026-08-03
    vwap_ml回測記憶體爆炸時發現，見 strategy/vwap_ml/train.py::
    _prepare_data() 的說明）。

    start_date：同 load_m1()，None代表讀全部月份。"""
    paths = _month_file_list(_ROOT / "db/m1", start_date)
    for path in paths:
        df = ds.dataset(path, format="parquet").to_table().to_pandas()
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
        yield df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def iter_m3_months(start_date: str | None = None):
    """逐月yield db/m3/ rolling 3分鐘K（原始價格），同 iter_m1_months()
    的動機，去重行為維持跟 load_m3() 一致（不去重）。"""
    path = _ROOT / "db/m3"
    for p in _month_file_list(path, start_date):
        df = ds.dataset(p, format="parquet").to_table().to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        yield df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def iter_m5_months(start_date: str | None = None):
    """逐月yield db/m5/ rolling 5分鐘K（原始價格），同 iter_m1_months()
    的動機，去重行為維持跟 load_m5() 一致（不去重）。"""
    path = _ROOT / "db/m5"
    for p in _month_file_list(path, start_date):
        df = ds.dataset(p, format="parquet").to_table().to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        yield df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m3(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/m3/ 3 分鐘K（原始價格），rolling 版本，每分鐘一列（由
    build_m3_m5_rolling.py 預先聚合）。一般情況請用
    data/query.py::load_m3()（還原權息後版本）。

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
    """載入 db/m5/ 5 分鐘K（原始價格），rolling 版本，每分鐘一列（由
    build_m3_m5_rolling.py 預先聚合）。一般情況請用
    data/query.py::load_m5()（還原權息後版本）。

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
    """載入 db/m3_std/ 標準獨立 3 分K棒（原始價格），一根K棒一列（由
    build_m3_m5_std.py 預先聚合）。一般情況請用
    data/query.py::load_m3_std()（還原權息後版本）。

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
    """載入 db/m5_std/ 標準獨立 5 分K棒（原始價格），一根K棒一列（由
    build_m3_m5_std.py 預先聚合）。一般情況請用
    data/query.py::load_m5_std()（還原權息後版本）。

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


def load_day(start_date: str | None = None) -> pd.DataFrame:
    """載入 db/d1/ 日K（原始價格，未還原權息，按月分檔）。一般情況請用
    data/query.py::load_day()（還原權息後版本，只還原拆股/合股）；如果需要
    pattern型態偵測用的完整還原版本（含一般除權息），請用
    data/adjustment_query.py::load_pattern_day()。

    start_date：同 load_m1() 的說明，預設 None = 讀全部。"""
    paths = _dataset_paths(_ROOT / "db/d1", start_date)
    if not paths:
        return pd.DataFrame()

    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_day_by_stock(stock_id: str, date: str = None) -> pd.DataFrame:
    """載入單一股票在 db/d1/ 的日K（原始價格），只讀該股票的 row group，
    不用像 load_day() 一樣把全市場都讀進記憶體。

    date: 選填，格式 "YYYY-MM-DD"，指定只回傳該日那一筆；不填則回傳該股票
    全部日K（依日期排序）。查無資料一律回傳空 DataFrame。"""
    dataset = ds.dataset(str(_ROOT / "db/d1"), format="parquet")
    filt = ds.field("stock_id") == stock_id
    if date is not None:
        filt = filt & (ds.field("date") == date)
    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    return df.sort_values("date").reset_index(drop=True)


def load_volume_profile(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """載入 db/volume_profile/ 價位成交量分布 (Volume Profile)，原始價格
    （從 db/tick 原始成交價算出來的，見檔頭說明）。一般情況請用
    data/query.py::load_volume_profile()（還原權息後版本）。

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
    """載入 db/poc_day/ 每日 POC 關鍵價位 (Point of Control 與 Value Area)，
    原始價格。一般情況請用 data/query.py::load_poc()（還原權息後版本）。

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
