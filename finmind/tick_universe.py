"""固定股票候選母體（db/tickers/tick_universe.parquet）——供 fubon/subscribe_list.py
（WebSocket即時收集/當沖候選股）與 finmind tick 回補共用的唯一股票清單來源。

2026-08-17改版（取代原本「db/adjustment_day 近6個月平均成交量排前399名+0050」
的算法）：
    1. 母體改成 db/tickers/stock_universe_2000.parquet（全市場~1877支4碼個股，
       已經不含00開頭ETF，不用再另外排除）。
    2. 篩選：均量（近 _AVG_VOL_WINDOW 個「交易日」，預設20日，沿用專案裡常見
       的「20日均量」慣例）> _MIN_AVG_VOL_LOTS 張，且 ATR(14)% > _MIN_ATR_PCT%
       （用 db/adjustment_day 的 high/low/close 算，公式跟一般日K ATR一致：
       TR=max(高-低,|高-昨收|,|低-昨收|)，ATR14=14日TR滾動平均，
       ATR%=ATR14/收盤價）。均量/ATR的資料來源都是抓 _FETCH_CALENDAR_DAYS
       個日曆天（預設75天，涵蓋20個交易日均量+14個交易日ATR暖機的緩衝），
       事後再依實際交易日數截斷，不是日曆天數本身就是均量的計算範圍。
    3. 對通過2的股票逐支查富邦 intraday_ticker() 的 canBuyDayTrade/canDayTrade，
       分兩級：
           tier_both（canBuyDayTrade=canDayTrade=true，能當沖多空）
           tier_long_only（只有canBuyDayTrade=true，只能當沖多）
       兩級都要求先通過第2步的均量/ATR篩選。
    4. tier_both 全部先依均量排序放前面；不夠 _TOP_N 才用 tier_long_only（也
       依均量排序）補滿，總數上限 _TOP_N（含強制併入的0050）。
    5. 0050 一定併入（白名單，不受任何篩選/排名限制），確保 strategy/rally
       的大盤代理特徵永遠拿得到（見 fubon/subscribe_list.py 的說明）。

跟 fubon/subscribe_list.py::_filter_day_tradable() 的分工：這裡的 canDayTrade/
canBuyDayTrade 查詢只在建置這份「候選母體」快照時做一次（多久重建一次由使用
者手動決定，見檔尾用法），subscribe_list.py 每天早上 6 點還是會對這份母體重新
查一次 canBuyDayTrade（只驗證「今天還能不能當沖」，例如臨時被處置/注意，不重
算 tier 分級）——兩層檢查目的不同，不能互相取代。

用法：
    python -m finmind.tick_universe   # 算一次、存到 db/tickers/tick_universe.parquet

之後其他程式要讀這份固定清單，呼叫 load_tick_universe()，不要重新呼叫
build_tick_universe()（那樣清單會因為之後市場資料更新而變動，違背「固定清單」
的原意，且會觸發一輪富邦API逐支查詢）。
"""

import time
from pathlib import Path

import pandas as pd

from finmind.m1_api import _atomic_to_parquet, _ROOT

_TOP_N = 800  # 輸出總檔數（含force_include），不是「純排名檔數 + force_include」
_MIN_AVG_VOL_LOTS = 1000.0  # 均量門檻（張，即1000股=1張）
_MIN_ATR_PCT = 1.0  # ATR(14) 佔收盤價百分比門檻
_ATR_WINDOW = 14
_AVG_VOL_WINDOW = 20  # 均量取最近幾個「交易日」（不是日曆天），沿用專案裡
# 常見的「20日均量」慣例（2026-08-19使用者要求，原本是90個日曆天≈3個月）。
_FETCH_CALENDAR_DAYS = 75  # 抓資料時往回抓的日曆天數，只是為了確保撈到的
# db/adjustment_day 月檔涵蓋足夠的「交易日」——20日均量+14日ATR暖機，扣掉
# 週末/假日大概需要35個交易日左右，75個日曆天留了充分緩衝，實際計算範圍
# 還是依交易日數截斷（見 _compute_metrics()），不是這個日曆天數字本身。
_FORCE_INCLUDE = ["0050"]


def _universe_file_path() -> Path:
    return _ROOT / "db/tickers/tick_universe.parquet"


def _load_recent_day_k(lookback_days: int = _FETCH_CALENDAR_DAYS) -> pd.DataFrame:
    """讀最近 lookback_days 天涵蓋到的月份 db/adjustment_day/{year}_{month}.parquet，
    只取算均量/ATR用得到的欄位。用「今天往回推」動態決定月份，不是寫死某幾個月
    ——上一版（build_tick_universe 舊實作）寫死 2026-01~06，之後每次要重新計算
    都要手動改程式碼裡的年月，這次改成自動跟著執行當下的日期走。"""
    today = pd.Timestamp.now(tz="Asia/Taipei").normalize().tz_localize(None)
    start = today - pd.Timedelta(days=lookback_days)
    months = pd.period_range(start, today, freq="M")
    frames = []
    for p in months:
        path = _ROOT / f"db/adjustment_day/{p.year}_{p.month:02d}.parquet"
        if path.exists():
            frames.append(
                pd.read_parquet(path, columns=["stock_id", "date", "high", "low", "close", "volume"])
            )
    if not frames:
        raise FileNotFoundError(f"db/adjustment_day 找不到最近 {lookback_days} 天涵蓋的月份資料")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= start]


def _compute_atr_pct(day_df: pd.DataFrame) -> pd.Series:
    """回傳每支股票最新一筆 ATR(14) 佔收盤價百分比（index 是 stock_id）。缺
    ATR（例如新上市不到14個交易日）的股票直接濾掉（dropna），不是門檻判斷，
    門檻比較留給呼叫端。ATR 用抓進來的「全部」歷史算 rolling（比
    _AVG_VOL_WINDOW 更早的資料只是給 ATR14 暖機用）。"""
    day_df = day_df.sort_values(["stock_id", "date"])
    g = day_df.groupby("stock_id")
    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [
            day_df["high"] - day_df["low"],
            (day_df["high"] - prev_close).abs(),
            (day_df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    day_df = day_df.assign(_tr=tr)
    day_df["_atr14"] = day_df.groupby("stock_id")["_tr"].transform(
        lambda s: s.rolling(_ATR_WINDOW, min_periods=_ATR_WINDOW).mean()
    )
    day_df["_atr_pct"] = day_df["_atr14"] / day_df["close"] * 100
    return day_df.dropna(subset=["_atr_pct"]).groupby("stock_id")["_atr_pct"].last()


def _compute_metrics(day_df: pd.DataFrame) -> pd.DataFrame:
    """回傳每支股票的 avg_volume_lots（最近 _AVG_VOL_WINDOW 個交易日均量，張）、
    atr_pct（最新一筆 ATR(14) 佔收盤價百分比）。index 是 stock_id。

    均量改呼叫 data.adjustment_query.avg_volume_lots()（2026-08-19合併，見
    該函式的說明）——原本這裡自己用 day_df 現算，沒有走
    _adjust_volume_only() 的拆股volume校正，跟 pattern/data_loader.py 那份
    10日均量邏輯不一致，同一支股票兩邊算出來的均量可能對不上（使用者
    2026-08-19實測發現 5488 用這裡算超過1000張門檻、用另一份10日邏輯只有
    731張）。ATR 不受這個問題影響（用的是 high/low/close，不是volume），
    繼續用這支自己抓的 day_df 現算，不用額外呼叫。"""
    from data.adjustment_query import avg_volume_lots as _avg_volume_lots_shared

    latest_atr_pct = _compute_atr_pct(day_df)
    avg_vol_lots = pd.Series(_avg_volume_lots_shared(window=_AVG_VOL_WINDOW), dtype="float64")

    return pd.DataFrame({"avg_volume_lots": avg_vol_lots, "atr_pct": latest_atr_pct}).dropna()


def _check_day_trade_tiers(stock_ids: list[str]) -> tuple[set[str], set[str]]:
    """逐支查富邦 intraday_ticker() 的 canBuyDayTrade/canDayTrade，分兩級：
    tier_both（能當沖多空）、tier_long_only（只能當沖多，canBuyDayTrade=true
    但canDayTrade=false）。跟 fubon/subscribe_list.py::_filter_day_tradable()
    共用同一套 timeout/節流機制（_call_with_timeout），但那支只回傳單一布林值
    （能不能當沖），這裡要同時知道 canDayTrade 才能分級，不能直接沿用那支
    函式本身，改成重用它底層的 timeout helper。"""
    from fubon import fubon_api as trade_api
    from fubon.subscribe_list import _call_with_timeout, _TICKER_CHECK_TIMEOUT

    sdk, _ = trade_api.login()
    tier_both: set[str] = set()
    tier_long_only: set[str] = set()
    try:
        trade_api.init_market_data(sdk)
        for i, sid in enumerate(stock_ids, 1):
            try:
                info = _call_with_timeout(trade_api.intraday_ticker, _TICKER_CHECK_TIMEOUT, sdk, sid)
                if info.get("canBuyDayTrade"):
                    if info.get("canDayTrade"):
                        tier_both.add(sid)
                    else:
                        tier_long_only.add(sid)
            except TimeoutError as e:
                print(f"  警告：{sid} intraday_ticker {e}，略過", flush=True)
            except Exception as e:
                print(f"  警告：{sid} intraday_ticker 查詢失敗，略過: {e}", flush=True)
            if i % 100 == 0:
                print(f"  當沖資格查詢進度：{i}/{len(stock_ids)}", flush=True)
            time.sleep(0.25)  # 300次/分鐘上限，留緩衝
    finally:
        trade_api.logout(sdk)
    return tier_both, tier_long_only


def build_tick_universe(
    top_n: int = _TOP_N,
    min_avg_vol_lots: float = _MIN_AVG_VOL_LOTS,
    min_atr_pct: float = _MIN_ATR_PCT,
    force_include: list[str] = _FORCE_INCLUDE,
) -> pd.DataFrame:
    """依 db/tickers/stock_universe_2000.parquet 母體，篩「均量>min_avg_vol_lots
    張 且 ATR(14)%>min_atr_pct%」，通過的股票再逐支查富邦當沖資格分兩級（能
    多空優先、只能做多的補位），依均量排序取前 top_n（含 force_include）。

    輸出欄位：stock_id, name, avg_volume（股數，跟舊版欄位單位一致）, atr_pct,
    day_trade_tier（"both"/"long_only"/"forced"）, rank, forced_include。
    force_include 的列 rank=NaN、forced_include=True，avg_volume/atr_pct 仍照實
    填（查得到的話），不代表沒有資料，只是不受排名/篩選限制。
    """
    base_ids = set(
        pd.read_parquet(_ROOT / "db/tickers/stock_universe_2000.parquet", columns=["stock_id"])[
            "stock_id"
        ].astype(str)
    )

    day_df = _load_recent_day_k()
    day_df = day_df[day_df["stock_id"].isin(base_ids)]
    metrics = _compute_metrics(day_df)

    pool = metrics[(metrics["avg_volume_lots"] > min_avg_vol_lots) & (metrics["atr_pct"] > min_atr_pct)]
    pool_ids = [sid for sid in pool.index if sid not in force_include]
    print(
        f"量能+ATR篩選：{len(base_ids)} 支母體 → {len(pool_ids)} 支符合"
        f"（均量>{min_avg_vol_lots}張、ATR%>{min_atr_pct}%）",
        flush=True,
    )

    tier_both, tier_long_only = _check_day_trade_tiers(pool_ids)
    print(f"當沖資格：{len(tier_both)} 支能多空、{len(tier_long_only)} 支只能做多", flush=True)

    def _sorted_tier(ids: set[str]) -> pd.DataFrame:
        return pool.loc[pool.index.isin(ids)].sort_values("avg_volume_lots", ascending=False)

    both_df = _sorted_tier(tier_both)
    long_df = _sorted_tier(tier_long_only)

    need = top_n - len(force_include)
    combined = pd.concat([both_df, long_df]).head(need)
    tier_label = {sid: "both" for sid in both_df.index}
    tier_label.update({sid: "long_only" for sid in long_df.index})

    result = pd.DataFrame(
        {
            "stock_id": combined.index,
            "avg_volume": (combined["avg_volume_lots"] * 1000).values,  # 存回股數，跟舊版欄位單位一致
            "atr_pct": combined["atr_pct"].values,
            "day_trade_tier": [tier_label[sid] for sid in combined.index],
            "rank": range(1, len(combined) + 1),
            "forced_include": False,
        }
    )

    extra_df = pd.DataFrame(
        {
            "stock_id": force_include,
            "avg_volume": [
                metrics["avg_volume_lots"].get(sid, float("nan")) * 1000 for sid in force_include
            ],
            "atr_pct": [metrics["atr_pct"].get(sid, float("nan")) for sid in force_include],
            "day_trade_tier": "forced",
            "rank": pd.NA,
            "forced_include": True,
        }
    )

    out = pd.concat([result, extra_df], ignore_index=True)
    return _attach_names(out)


def _attach_names(df: pd.DataFrame) -> pd.DataFrame:
    """加上 name 欄位：查 db/tickers/tickers.parquet 本地既有清單（不觸發即時
    富邦API），讓 tick_universe.parquet 自己就有股票名稱，fubon/subscribe_list.py
    不用再額外讀一份全市場清單來查名稱——tick_universe 是唯一的股票母體來源，
    名稱也一併帶著走。查不到名稱的（tickers.parquet 沒有這支，例如剛好還沒
    更新到）留空字串，不擋清單本身。"""
    from fubon.intraday_tickers import load_tickers

    tickers_df = load_tickers()
    name_map = dict(zip(tickers_df["stock_id"], tickers_df["name"])) if not tickers_df.empty else {}
    df = df.copy()
    df["name"] = df["stock_id"].map(name_map).fillna("")
    return df


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
