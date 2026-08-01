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

還原權息版本（2026-08-01加）：db/m1／db/tick／db/volume_profile／db/poc_day
存的都是原始（未還原權息）價格，只有 db/fugle_day（日K，下載時帶
adjusted="true"）是還原後的，當作反推係數的基準來源。
load_m1_adjusted()／load_volume_profile_adjusted()／load_poc_adjusted() 在讀取
時 join db/tick_adjust_factor（見 data/build_tick_adjust_factor.py／
finmind/backfill_tick_adjust_factor.py）換算成跟 db/fugle_day 一致的調整後
價格再回傳，不動底層 parquet。要跟日K價格一起比較/使用、或訓練特徵需要
連續價格序列時，應該用這三支，不要直接用 load_m1()/load_volume_profile()/
load_poc()。
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


def load_m1_adjusted(start_date: str | None = None) -> pd.DataFrame:
    """load_m1() 的「還原權息後」版本。

    db/m1 存的是原始價格（2026-08-01 改，見 data/m1_data_loader.py 頂部說明：
    故意不帶 adjusted，統一交給查詢層處理，不用管這筆資料當初是 Fugle／
    富邦／finmind 哪個來源抓的）。這支函式在讀取當下 join
    db/tick_adjust_factor 反推出的每日調整係數，把 open/high/low/close 換算
    成跟 db/fugle_day 一致的還原後基準再回傳，不動 db/m1 本身。缺 factor 的
    (stock_id, date)（該股票沒有tick資料/還沒建 factor）維持原始價格
    （factor=1.0），不會整筆丟掉。volume 不受影響。

    跟 load_volume_profile_adjusted() 不同：這裡不需要 round 後重新聚合——
    m1 每一列本來就用完整時間戳（精確到分鐘）識別，不是像 volume_profile
    那樣要用 price 當 groupby key，係數的微小日間差異不會造成假重複。

    參數/回傳欄位同 load_m1()。"""
    m1_df = load_m1(start_date=start_date)
    if m1_df.empty:
        return m1_df

    m1_df["day"] = m1_df["date"].dt.strftime("%Y-%m-%d")
    factor_df = _load_adjust_factor(None, None, start_date)
    m1_df = m1_df.merge(
        factor_df[["stock_id", "date", "factor"]].rename(columns={"date": "day"}),
        on=["stock_id", "day"],
        how="left",
    )
    m1_df["factor"] = m1_df["factor"].fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        m1_df[col] = (m1_df[col] * m1_df["factor"]).round(2).astype("float32")
    return m1_df.drop(columns=["day", "factor"]).sort_values(["stock_id", "date"]).reset_index(drop=True)


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


def _load_adjust_factor(stock_id: str | None, date: str | None, start_date: str | None) -> pd.DataFrame:
    """載入 db/tick_adjust_factor/（見 data/build_tick_adjust_factor.py），內部
    helper，給 load_volume_profile_adjusted()/load_poc_adjusted() 共用。"""
    path = _ROOT / "db/tick_adjust_factor"
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "date", "factor"])

    eff_start = start_date
    if date is not None and (eff_start is None or date < eff_start):
        eff_start = date

    paths = _dataset_paths(path, eff_start)
    if not paths:
        return pd.DataFrame(columns=["stock_id", "date", "factor"])

    dataset = ds.dataset(paths, format="parquet")
    filt = None
    if stock_id is not None:
        filt = ds.field("stock_id") == stock_id
    if date is not None:
        filt = (filt & (ds.field("date") == date)) if filt is not None else (ds.field("date") == date)

    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["stock_id", "date", "factor"])
    return table.to_pandas()


def load_volume_profile_adjusted(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """load_volume_profile() 的「還原權息後」版本。

    db/volume_profile 本身是從 db/tick 原始成交價算出來的（見 CLAUDE.md 之外的另一個
    基準問題：db/m1／db/fugle_day 下載時都帶 adjusted="true"，db/tick 沒有，兩邊價格
    基準對不上，2026-08-01 除錯 pattern 圖表時發現）。這支函式在讀取當下 join
    db/tick_adjust_factor 反推出的每日調整係數，把 price 換算成跟 K 線一致的調整後
    基準再回傳，不動 db/volume_profile 本身。缺 factor 的 (stock_id, date)（例如
    db/tick_adjust_factor 還沒建或還沒增量到那天）維持原始價格（factor=1.0），不會
    整筆丟掉。

    參數/回傳欄位同 load_volume_profile()。
    """
    vp_df = load_volume_profile(stock_id=stock_id, date=date, start_date=start_date)
    if vp_df.empty:
        return vp_df

    factor_df = _load_adjust_factor(stock_id, date, start_date)
    vp_df = vp_df.merge(factor_df[["stock_id", "date", "factor"]], on=["stock_id", "date"], how="left")
    vp_df["factor"] = vp_df["factor"].fillna(1.0)
    # round(2)：factor 是逐日反推出來的，除息日之前每天的 factor 都有微小差異（同一個
    # 原始tick檔位，不同天乘出來的 adjusted price 會差在小數點後幾位），不 round 直接
    # 用浮點數當 groupby key，同一個名目價位會被拆成一堆幾乎重複的假價位，2026-08-01
    # 實測 1101 半年份資料因此從約195個乾淨價位灌水成760+個——round 完再依
    # (stock_id, date, price) 重新加總，把這些假重複收斂回同一檔
    vp_df["price"] = (vp_df["price"] * vp_df["factor"]).round(2).astype("float32")
    vp_df = vp_df.groupby(["stock_id", "date", "price"], as_index=False)[
        ["volume", "buy_volume", "sell_volume", "neutral_volume"]
    ].sum()
    return vp_df.sort_values(["stock_id", "date", "price"]).reset_index(drop=True)


def load_poc_adjusted(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """load_poc() 的「還原權息後」版本，說明同 load_volume_profile_adjusted()。

    poc/vah/val/pocs 都是價位，乘上當天調整係數換算成調整後基準；poc_volume/
    total_volume/poc_count/profile_type 是量或分類欄位，不受價格調整影響，原樣回傳。
    """
    poc_df = load_poc(stock_id=stock_id, date=date, start_date=start_date)
    if poc_df.empty:
        return poc_df

    factor_df = _load_adjust_factor(stock_id, date, start_date)
    poc_df = poc_df.merge(factor_df[["stock_id", "date", "factor"]], on=["stock_id", "date"], how="left")
    poc_df["factor"] = poc_df["factor"].fillna(1.0)
    poc_df["poc"] = (poc_df["poc"] * poc_df["factor"]).round(2).astype("float32")
    poc_df["vah"] = (poc_df["vah"] * poc_df["factor"]).round(2).astype("float32")
    poc_df["val"] = (poc_df["val"] * poc_df["factor"]).round(2).astype("float32")
    poc_df["pocs"] = poc_df.apply(
        lambda r: ",".join(f"{float(p) * r['factor']:.2f}" for p in str(r["pocs"]).split(",") if p),
        axis=1,
    )
    return poc_df.drop(columns=["factor"]).sort_values(["stock_id", "date"]).reset_index(drop=True)


