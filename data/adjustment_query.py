"""
pattern 系列（型態偵測、POC疊圖，含 strategy/breakout_retest_ml/*）專用的
「完整還原（含一般除權息）」查詢層，2026-08-03 新增。

系統預設的還原基準（data/query.py）只還原拆股/合股，一般除權息刻意不還原
（除息會填息，是真實發生過的價格波動）。但實測發現：
`pattern/breakout_retest/detector.py` 這類型態偵測演算法，如果拿到的K線裡
還留著除息造成的真實跳空（例如 1101 2026-07-01 除息，跳空幅度 3.3~4.4%），
跟偵測器本身 1~2.5% 等級的門檻比較，確實大到會造成誤判（實測數字：偵測器的
`min_breakout_pct=1%`／`max_resistance_diff_pct=2%`／`max_retest_dist_pct=2.5%`
等門檻都在這個量級內）。所以 pattern 系列是系統裡**唯一的例外**，需要「完整
還原」基準（除息缺口也被抹平），其他所有地方請用 data/query.py（只還原拆股/
合股）或 data/raw_query.py（原始）。

這裡的函式刻意跟 data/query.py 不同名（`load_pattern_*` 前綴），避免兩套
「還原後」定義被不小心用混——這是這幾輪重構一路在避免的事，不要為了圖方便
改成同名。

資料來源：
    db/adjustment_day/    完整還原日K（沿用 Fugle adjusted="true" 下載，
                           2026-08-03 從 db/fugle_day 改名而來）
    db/adjustment_factor/ 完整還原係數（見 data/build_adjustment_factor.py，
                           = db/adjustment_day close / db/d1 close，逐日比對，
                           不做拆股/合股門檻篩選，完整反映所有除權息事件）
    db/m1／db/m3／db/m5／db/m3_std／db/m5_std／db/tick 衍生的
    volume_profile／poc_day：都跟 data/query.py 一樣讀原始版本（data/raw_query.py），
    只是套用這裡的完整還原係數，不是系統預設的拆股/合股限定係數。

不負責下載或更新資料，只做讀取。
"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from data import raw_query

_ROOT = Path(__file__).parent.parent


def _adjust_volume_only(df: pd.DataFrame, stock_id: str | None, date: str | None, start_date: str | None) -> pd.DataFrame:
    """把 df 的 volume 除以「系統預設、只還原拆股/合股」的 db/tick_adjust_factor
    （不是這支檔案自己的 db/adjustment_factor），open/high/low/close 不動。

    背景（2026-08-03）：db/adjustment_day 是 Fugle 下載時帶 adjusted="true" 的
    完整還原日K，實測發現 Fugle 只調整價格、volume 完全沒動（0050拆股前後
    volume 值跟原始資料一模一樣）——所以 load_pattern_day() 的
    open/high/low/close 可以直接信任 Fugle 已經處理好，但 volume 一樣有
    「拆股前後基準不一致」的問題，需要自己修。這裡刻意借用
    db/tick_adjust_factor（只反映拆股/合股，乾淨無除息雜訊）而不是
    db/adjustment_factor（混合拆股+除息）——用混合的那張會在每次除息時把
    volume 算錯，理由同 _adjust_ohlc() 的說明。"""
    if df.empty:
        return df
    from data.query import _load_adjust_factor

    df = df.copy()
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    factor_df = _load_adjust_factor(stock_id, date, start_date)
    df = df.merge(
        factor_df[["stock_id", "date", "factor"]].rename(columns={"date": "day"}),
        on=["stock_id", "day"],
        how="left",
    )
    df["factor"] = df["factor"].fillna(1.0)
    df["volume"] = (df["volume"] / df["factor"]).round().astype("int64")
    return df.drop(columns=["day", "factor"])


def load_pattern_day(
    start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """載入 db/adjustment_day/ 日K（完整還原，pattern 專用，按月分檔）。
    下載時已經帶 adjusted="true"，open/high/low/close 不用另外換算；volume
    另外處理，見 _adjust_volume_only()。

    start_date／end_date：同 data.raw_query.load_m1() 的說明，預設 None = 讀全部。"""
    paths = raw_query._dataset_paths(_ROOT / "db/adjustment_day", start_date, end_date)
    if not paths:
        return pd.DataFrame()

    df = ds.dataset(paths, format="parquet").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    df = _adjust_volume_only(df, None, None, start_date)
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_pattern_day_by_stock(stock_id: str, date: str = None) -> pd.DataFrame:
    """載入單一股票在 db/adjustment_day/ 的日K（完整還原），只讀該股票的
    row group，不用像 load_pattern_day() 一樣把全市場都讀進記憶體。open/high/
    low/close 不用另外換算；volume 另外處理，見 _adjust_volume_only()。

    date: 選填，格式 "YYYY-MM-DD"，指定只回傳該日那一筆；不填則回傳該股票
    全部日K（依日期排序）。查無資料一律回傳空 DataFrame。"""
    dataset = ds.dataset(str(_ROOT / "db/adjustment_day"), format="parquet")
    filt = ds.field("stock_id") == stock_id
    if date is not None:
        filt = filt & (ds.field("date") == date)
    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    df = _adjust_volume_only(df, stock_id, date, None)
    return df.sort_values("date").reset_index(drop=True)


def _load_adjustment_factor(
    stock_id: str | None,
    date: str | None,
    start_date: str | None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """載入 db/adjustment_factor/（見 data/build_adjustment_factor.py），內部
    helper，給這支檔案裡的還原版函式共用。跟 data/query.py::_load_adjust_factor()
    是兩張不同的表，不要混用。"""
    path = _ROOT / "db/adjustment_factor"
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "date", "factor"])

    eff_start = start_date
    if date is not None and (eff_start is None or date < eff_start):
        eff_start = date

    paths = raw_query._dataset_paths(path, eff_start, end_date)
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


def _adjust_ohlc(
    df: pd.DataFrame, start_date: str | None, end_date: str | None = None
) -> pd.DataFrame:
    """共用邏輯：load_pattern_m1()/m3()/m5()/m3_std()/m5_std() 都是「完整時間戳
    識別、OHLC乘上當日完整還原係數」模式，抽成共用函式。照抄
    data/query.py::_adjust_ohlc() 的 pattern，只是換成呼叫
    _load_adjustment_factor()。

    ⚠️ 2026-08-03 刻意跟 data/query.py::_adjust_ohlc() 不一樣的地方：這裡**不**
    把 volume 除以 factor。data/query.py 那張 db/tick_adjust_factor 只反映拆股/
    合股（真的會改變股數），volume除以factor是對的；但這裡的
    db/adjustment_factor 混合了拆股跟一般除權息（factor在除息日也會變動），
    除息不影響股數，如果對這張混合係數也做volume調整，會在每次除息時把volume
    錯誤放大/縮小——這張表沒有拆股/除息的區分標記，沒辦法只挑拆股的部分處理，
    所以維持volume不調整，代價是遇到lookback視窗剛好跨過拆股的股票，volume
    可能會有斷崖（比每次除息volume都算錯的風險小很多）。"""
    if df.empty:
        return df
    df = df.copy()
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    factor_df = _load_adjustment_factor(None, None, start_date, end_date)
    df = df.merge(
        factor_df[["stock_id", "date", "factor"]].rename(columns={"date": "day"}),
        on=["stock_id", "day"],
        how="left",
    )
    df["factor"] = df["factor"].fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = (df[col] * df["factor"]).round(2).astype("float32")
    return df.drop(columns=["day", "factor"]).sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_pattern_m1(
    start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """load_m1() 的「pattern 專用完整還原」版本（data.raw_query.load_m1()
    是原始版本）。說明同檔頭。"""
    return _adjust_ohlc(
        raw_query.load_m1(start_date=start_date, end_date=end_date),
        start_date,
        end_date,
    )


def load_pattern_m3(start_date: str | None = None) -> pd.DataFrame:
    """load_m3() 的「pattern 專用完整還原」版本。"""
    return _adjust_ohlc(raw_query.load_m3(start_date=start_date), start_date)


def load_pattern_m5(start_date: str | None = None) -> pd.DataFrame:
    """load_m5() 的「pattern 專用完整還原」版本。"""
    return _adjust_ohlc(raw_query.load_m5(start_date=start_date), start_date)


def load_pattern_m3_std(start_date: str | None = None) -> pd.DataFrame:
    """load_m3_std() 的「pattern 專用完整還原」版本。"""
    return _adjust_ohlc(raw_query.load_m3_std(start_date=start_date), start_date)


def load_pattern_m5_std(
    start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """load_m5_std() 的「pattern 專用完整還原」版本。"""
    return _adjust_ohlc(
        raw_query.load_m5_std(start_date=start_date, end_date=end_date),
        start_date,
        end_date,
    )


def load_pattern_volume_profile(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """load_volume_profile() 的「pattern 專用完整還原」版本
    （data.raw_query.load_volume_profile() 是原始版本）。套用
    db/adjustment_factor（完整還原，含一般除權息），不是系統預設的
    db/tick_adjust_factor（只還原拆股/合股）。

    round(2) 後重新 groupby 避免價位假重複：理由同
    data/query.py::load_volume_profile() 的說明。"""
    vp_df = raw_query.load_volume_profile(stock_id=stock_id, date=date, start_date=start_date)
    if vp_df.empty:
        return vp_df

    factor_df = _load_adjustment_factor(stock_id, date, start_date)
    vp_df = vp_df.merge(factor_df[["stock_id", "date", "factor"]], on=["stock_id", "date"], how="left")
    vp_df["factor"] = vp_df["factor"].fillna(1.0)
    vp_df["price"] = (vp_df["price"] * vp_df["factor"]).round(2).astype("float32")
    vp_df = vp_df.groupby(["stock_id", "date", "price"], as_index=False)[
        ["volume", "buy_volume", "sell_volume", "neutral_volume"]
    ].sum()
    return vp_df.sort_values(["stock_id", "date", "price"]).reset_index(drop=True)


def load_pattern_poc(
    stock_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """load_poc() 的「pattern 專用完整還原」版本（data.raw_query.load_poc()
    是原始版本），說明同 load_pattern_volume_profile()。

    poc/vah/val/pocs 都是價位，乘上當天完整還原係數換算；poc_volume/
    total_volume/poc_count/profile_type 是量或分類欄位，不受價格調整影響，
    原樣回傳。"""
    poc_df = raw_query.load_poc(stock_id=stock_id, date=date, start_date=start_date)
    if poc_df.empty:
        return poc_df

    factor_df = _load_adjustment_factor(stock_id, date, start_date)
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
