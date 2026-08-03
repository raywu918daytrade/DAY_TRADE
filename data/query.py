"""
本地資料讀取工具（唯讀，還原權息後版本，2026-08-01 重整，2026-08-03 再改）

這裡的公開函式**預設回傳的就是還原權息後的價格**（只還原拆股/合股，不還原
一般除權息），內部是呼叫 data/raw_query.py 的原始版本、再 join
db/tick_adjust_factor 換算。不需要另外處理「這筆資料原始還是還原後」——
直接用這裡的函式就對了。

只有明確知道自己需要**原始（未還原權息）**價格的地方（例如
data/build_tick_adjust_factor.py 這種本來就是在反推調整係數的 pipeline
腳本），才需要改用 data/raw_query.py 對應的同名函式。如果需要 pattern
型態偵測用的「完整還原（含一般除權息）」版本，請用
data/adjustment_query.py，不要用這支——理由見那支檔頭說明。

三種資料對應：
    load_m1()      → db/m1/        歷史分K（訓練用，按月分檔，已還原拆股/合股）
    load_day()     → db/d1/        日K（模型特徵用，按月分檔，已還原拆股/合股）
    load_m1_live() → db/m1_live/   今日即時分K（交易用，收盤後丟棄，當天raw==已還原）
    load_tick_live() → db/tick_live/ 今日即時成交明細（由 fubon/tick_ws.py 收集，
                                    交易用，收盤後丟棄，schema 跟 db/tick 一致）

單支股票查詢（用 pyarrow filter pushdown，不用像 load_day() 整個資料集讀進記憶體）：
    load_day_by_stock(stock_id)  → db/d1/       單一股票的全部日K
    load_tick_by_stock(stock_id) → db/tick/      單一股票的逐筆成交明細（原始價格，
                                    tick沒有「還原版」這個概念，見下方
                                    load_tick_by_stock() 說明，tick資料量太大
                                    不提供整個資料集一次讀進記憶體的版本）
    load_volume_profile()        → db/volume_profile/ 價位成交量分布 (Volume Profile)，已還原
    load_poc()                   → db/poc_day/        每日 POC 關鍵價位 (Point of Control)，已還原
    load_breakout_retest_day()   → db/breakout_retest_day/  日K突破回測候選
    load_breakout_retest_trigger() → db/breakout_retest_trigger/ 盤中M1實體K+Tick快照

還原權息機制（2026-08-01加，2026-08-02/03改）：db/m1／db/d1／db/tick／
db/volume_profile／db/poc_day 存的都是原始（未還原權息）價格。這裡的
load_m1()/load_m3()/load_m5()/load_m3_std()/load_m5_std()/load_day()/
load_volume_profile()/load_poc() 在讀取時 join db/tick_adjust_factor（見
data/build_tick_adjust_factor.py，factor 只反映拆股/合股，一般除權息不
還原、直接用原始K線——理由見那支檔頭說明）換算後再回傳，不動底層 parquet／
不動 data/raw_query.py 讀到的原始資料。factor 是從 db/d1 自己的逐日跳空
幅度反推的，不依賴任何廠商的「已還原」資料當比較基準。

pattern 系列（型態偵測、POC疊圖）是系統裡唯一的例外，需要「完整還原（含
一般除權息）」的基準，因為除息造成的真實跳空會讓型態偵測的轉折點判斷誤判
（2026-08-02 用 1101 除息實測過），這部分改用 db/adjustment_day（沿用
Fugle adjusted="true" 完整還原下載）+ data/adjustment_query.py，跟這支
檔案完全分開維護，不要混用。
"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from data import raw_query

_ROOT = Path(__file__).parent.parent


def load_m1(start_date: str | None = None) -> pd.DataFrame:
    """load_m1() 的「還原權息後」版本（data.raw_query.load_m1() 是原始版本）。

    db/m1 存的是原始價格（2026-08-01 改，見 data/m1_data_loader.py 頂部說明：
    故意不帶 adjusted，統一交給查詢層處理，不用管這筆資料當初是 Fugle／
    富邦／finmind 哪個來源抓的）。這支函式在讀取當下 join
    db/tick_adjust_factor 反推出的每日調整係數，把 open/high/low/close 換算
    成跟 db/fugle_day 一致的還原後基準再回傳，不動 db/m1 本身。缺 factor 的
    (stock_id, date)（該股票沒有tick資料/還沒建 factor）維持原始價格
    （factor=1.0），不會整筆丟掉。

    volume 也會跟著除以 factor（2026-08-03 加）：這裡的 factor 只反映拆股/
    合股，拆股會直接改變股數（例如1:4拆股股數變4倍），拆股前後的原始股數
    不是同一個基準，volume 不調整的話，跨過拆股日算 rolling 均量/量能指標
    會看到一個假的成交量斷崖。volume 除以 factor（factor<1時等於乘上股數
    變多的倍數）才能讓「價格×成交量」這個成交金額維持不變，是正確的還原
    方向（跟price乘上factor互為倒數關係）。除息的 factor 固定是1.0（這次
    系統預設不還原除息），所以除息不會觸發volume調整，符合預期。

    跟 load_volume_profile() 不同：這裡不需要 round 後重新聚合——m1 每一列
    本來就用完整時間戳（精確到分鐘）識別，不是像 volume_profile 那樣要用
    price 當 groupby key，係數的微小日間差異不會造成假重複。

    參數/回傳欄位同 data.raw_query.load_m1()。"""
    m1_df = raw_query.load_m1(start_date=start_date)
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
    m1_df["volume"] = (m1_df["volume"] / m1_df["factor"]).round().astype("int64")
    return m1_df.drop(columns=["day", "factor"]).sort_values(["stock_id", "date"]).reset_index(drop=True)


def _adjust_ohlc(df: pd.DataFrame, start_date: str | None) -> pd.DataFrame:
    """共用邏輯：load_m3()/load_m5()/load_m3_std()/load_m5_std()/load_day() 都是
    跟 load_m1() 一樣的「完整時間戳識別、OHLC乘上當日係數、volume除以當日係數」
    模式，抽成共用函式避免好幾份幾乎一樣的程式碼（load_m1() 沒有直接用這支，
    是因為它比這幾支早寫、當時還沒抽出來，行為完全等價，不用特別去改）。
    volume 為什麼要除以 factor：見 load_m1() 的說明。"""
    if df.empty:
        return df
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    factor_df = _load_adjust_factor(None, None, start_date)
    df = df.merge(
        factor_df[["stock_id", "date", "factor"]].rename(columns={"date": "day"}),
        on=["stock_id", "day"],
        how="left",
    )
    df["factor"] = df["factor"].fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        df[col] = (df[col] * df["factor"]).round(2).astype("float32")
    if "volume" in df.columns:
        df["volume"] = (df["volume"] / df["factor"]).round().astype("int64")
    return df.drop(columns=["day", "factor"]).sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m3(start_date: str | None = None) -> pd.DataFrame:
    """load_m3() 的「還原權息後」版本（data.raw_query.load_m3() 是原始版本）。
    3 分鐘K，rolling 版本，每分鐘一列（由 build_m3_m5_rolling.py 預先聚合）。
    說明同 load_m1()。"""
    return _adjust_ohlc(raw_query.load_m3(start_date=start_date), start_date)


def load_m5(start_date: str | None = None) -> pd.DataFrame:
    """load_m5() 的「還原權息後」版本（data.raw_query.load_m5() 是原始版本）。
    5 分鐘K，rolling 版本，每分鐘一列（由 build_m3_m5_rolling.py 預先聚合）。
    說明同 load_m1()。"""
    return _adjust_ohlc(raw_query.load_m5(start_date=start_date), start_date)


def load_m3_std(start_date: str | None = None) -> pd.DataFrame:
    """load_m3_std() 的「還原權息後」版本（data.raw_query.load_m3_std() 是
    原始版本）。標準獨立 3 分K棒，一根K棒一列（由 build_m3_m5_std.py 預先
    聚合）。說明同 load_m1()。"""
    return _adjust_ohlc(raw_query.load_m3_std(start_date=start_date), start_date)


def load_m5_std(start_date: str | None = None) -> pd.DataFrame:
    """load_m5_std() 的「還原權息後」版本（data.raw_query.load_m5_std() 是
    原始版本）。標準獨立 5 分K棒，一根K棒一列（由 build_m3_m5_std.py 預先
    聚合）。說明同 load_m1()。"""
    return _adjust_ohlc(raw_query.load_m5_std(start_date=start_date), start_date)


def load_day(start_date: str | None = None) -> pd.DataFrame:
    """load_day() 的「還原權息後（只還原拆股/合股）」版本
    （data.raw_query.load_day() 是原始版本，讀 db/d1）。

    db/d1 存的是原始價格（2026-08-03 改，取代原本的 db/fugle_day）。這支
    函式在讀取當下 join db/tick_adjust_factor 反推出的每日拆股/合股調整
    係數，換算後回傳，不動 db/d1 本身。缺 factor 的 (stock_id, date) 維持
    原始價格（factor=1.0）。volume 不受影響。

    如果需要 pattern 型態偵測用的完整還原版本（含一般除權息），請用
    data/adjustment_query.py::load_pattern_day()，不要用這支——說明見
    data/adjustment_query.py 檔頭。

    start_date：同 load_m1() 的說明，預設 None = 讀全部。"""
    return _adjust_ohlc(raw_query.load_day(start_date=start_date), start_date)


def load_day_by_stock(stock_id: str, date: str = None) -> pd.DataFrame:
    """load_day_by_stock() 的「還原權息後（只還原拆股/合股）」版本
    （data.raw_query.load_day_by_stock() 是原始版本）。

    只讀該股票的 row group，不用像 load_day() 一樣把全市場都讀進記憶體，
    適合只需要單支股票時用（例如查前一交易日收盤價）。

    date: 選填，格式 "YYYY-MM-DD"，指定只回傳該日那一筆；不填則回傳該股票
    全部日K（依日期排序）。查無資料一律回傳空 DataFrame。"""
    day_df = raw_query.load_day_by_stock(stock_id, date=date)
    if day_df.empty:
        return day_df

    day_df["day"] = day_df["date"].dt.strftime("%Y-%m-%d")
    factor_df = _load_adjust_factor(stock_id, date, None)
    day_df = day_df.merge(
        factor_df[["stock_id", "date", "factor"]].rename(columns={"date": "day"}),
        on=["stock_id", "day"],
        how="left",
    )
    day_df["factor"] = day_df["factor"].fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        day_df[col] = (day_df[col] * day_df["factor"]).round(2).astype("float32")
    day_df["volume"] = (day_df["volume"] / day_df["factor"]).round().astype("int64")
    return day_df.drop(columns=["day", "factor"]).sort_values("date").reset_index(drop=True)


def load_tick_by_stock(stock_id: str, date: str = None) -> pd.DataFrame:
    """載入單一股票在 db/tick/ 的逐筆成交明細（FinMind TaiwanStockPriceTick，
    見 finmind/tick_api.py），原始價格——tick 沒有「還原版」這個概念，
    tick 的還原是透過 load_volume_profile()/load_poc() 間接做的，只讀該
    股票的 row group，不提供整個資料集一次讀進記憶體的版本（tick資料量級
    跟分K差太多，一個月400檔股票就有2000多萬列，像 load_m1() 那樣整個
    資料夾讀進記憶體會是幾十GB，不現實）。

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


def load_tick_live(date: str = None) -> pd.DataFrame:
    """載入今日即時成交明細（db/tick_live/YYYY-MM-DD.parquet），由
    fubon/tick_ws.py::FubonTickCollector 收集，schema 跟 db/tick 一致
    （stock_id, date, deal_price, volume, tick_type）。跟 load_m1_live()
    同樣的「今日單檔」讀法，不用 pyarrow dataset filter pushdown（單日
    檔案不大，直接整個讀進來，供 main/live_trader.py::on_minute() 每分鐘
    組 ticks_by_stock 用）。"""
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = _ROOT / f"db/tick_live/{date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_m1_live(date: str = None) -> pd.DataFrame:
    """載入今日即時分K（db/m1_live/YYYY-MM-DD.parquet），盤後自動 backfill
    補齊。「今天」相對於自己必然是同一個基準，raw==adjusted，不用換算。"""
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = _ROOT / f"db/m1_live/{date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _load_adjust_factor(stock_id: str | None, date: str | None, start_date: str | None) -> pd.DataFrame:
    """載入 db/tick_adjust_factor/（見 data/build_tick_adjust_factor.py），內部
    helper，給這支檔案裡的還原版函式共用。"""
    path = _ROOT / "db/tick_adjust_factor"
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "date", "factor"])

    eff_start = start_date
    if date is not None and (eff_start is None or date < eff_start):
        eff_start = date

    paths = raw_query._dataset_paths(path, eff_start)
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


def load_volume_profile(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """load_volume_profile() 的「還原權息後」版本
    （data.raw_query.load_volume_profile() 是原始版本）。

    db/volume_profile 本身是從 db/tick 原始成交價算出來的（見 CLAUDE.md 之外的另一個
    基準問題：db/m1／db/fugle_day 下載時都帶 adjusted="true"，db/tick 沒有，兩邊價格
    基準對不上，2026-08-01 除錯 pattern 圖表時發現）。這支函式在讀取當下 join
    db/tick_adjust_factor 反推出的每日調整係數，把 price 換算成跟 K 線一致的調整後
    基準再回傳，不動 db/volume_profile 本身。缺 factor 的 (stock_id, date)（例如
    db/tick_adjust_factor 還沒建或還沒增量到那天）維持原始價格（factor=1.0），不會
    整筆丟掉。

    參數/回傳欄位同 data.raw_query.load_volume_profile()。
    """
    vp_df = raw_query.load_volume_profile(stock_id=stock_id, date=date, start_date=start_date)
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


def load_poc(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """load_poc() 的「還原權息後」版本（data.raw_query.load_poc() 是原始
    版本），說明同 load_volume_profile()。

    poc/vah/val/pocs 都是價位，乘上當天調整係數換算成調整後基準；poc_volume/
    total_volume/poc_count/profile_type 是量或分類欄位，不受價格調整影響，原樣回傳。
    """
    poc_df = raw_query.load_poc(stock_id=stock_id, date=date, start_date=start_date)
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


_BREAKOUT_DAY_COLS = [
    "stock_id",
    "candidate_date",
    "pattern_score",
    "resistance_price",
    "matched_poc",
    "poc_diff_pct",
    "poc_confluence",
]

_BREAKOUT_TRIGGER_COLS = [
    "stock_id",
    "candidate_date",
    "trade_date",
    "trigger_ts",
    "entry_price",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "volume_surge_ratio",
    "tick_large_buy_ratio",
    "cvd_30s_delta",
    "dist_to_poc_pct",
    "dist_to_support_pct",
    "pattern_score",
    "poc_diff_pct",
    "resistance_price",
    "matched_poc",
]


def load_breakout_retest_day(
    start_date: str | None = None,
    only_poc: bool = False,
) -> pd.DataFrame:
    """載入 db/breakout_retest_day/ 日 K 突破回測候選。

    start_date: 依 candidate_date 所在月份篩檔（同 _dataset_paths）。
    only_poc: True 時只回傳 poc_confluence==True。
    """

    path = _ROOT / "db/breakout_retest_day"
    if not path.exists():
        return pd.DataFrame(columns=_BREAKOUT_DAY_COLS)
    paths = raw_query._dataset_paths(path, start_date)
    if not paths:
        return pd.DataFrame(columns=_BREAKOUT_DAY_COLS)
    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    if start_date is not None and "candidate_date" in df.columns:
        df = df[df["candidate_date"].astype(str) >= start_date]
    if only_poc and "poc_confluence" in df.columns:
        df = df[df["poc_confluence"] == True]  # noqa: E712
    return df.sort_values(["stock_id", "candidate_date"]).reset_index(drop=True)


def load_breakout_retest_trigger(
    start_date: str | None = None,
) -> pd.DataFrame:
    """載入 db/breakout_retest_trigger/ 盤中 M1 實體 K + Tick 特徵快照。

    start_date: 依 trade_date 所在月份篩檔。尚未套用大單買比門檻（實驗時再 filter）。
    """

    path = _ROOT / "db/breakout_retest_trigger"
    if not path.exists():
        return pd.DataFrame(columns=_BREAKOUT_TRIGGER_COLS)
    paths = raw_query._dataset_paths(path, start_date)
    if not paths:
        return pd.DataFrame(columns=_BREAKOUT_TRIGGER_COLS)
    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    if "trigger_ts" in df.columns:
        df["trigger_ts"] = pd.to_datetime(df["trigger_ts"], format="mixed")
    if start_date is not None and "trade_date" in df.columns:
        df = df[df["trade_date"].astype(str) >= start_date]
    return df.sort_values(["stock_id", "trigger_ts"]).reset_index(drop=True)
