"""
即時推論 — predict()（批次機率矩陣，回測用）、predict_live()（正式即時推論入口）

mkt 是3分類（跌=0/平=1/漲=2），但策略本身只做多、只在乎「漲」這個訊號的
機率，所以這裡跟 orb/rally 的二分類 predict_proba()[:, 1] 概念一致，只是改抓
class=2 那一欄，套進同一份 backtest/intraday_platform.py 共用回測引擎（那支
引擎只吃單一機率矩陣，不知道背後模型是幾分類）。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from data.query import load_day, load_m1_live
from data.resample import compute_m3, compute_m3_std, compute_m5, compute_m5_std
from finmind.tick_universe import load_tick_universe
from strategy.mkt.config import ATR5_FILTER_THRESHOLD, IDX_SYMBOL, MODEL_TYPE
from strategy.mkt.features import (
    FEATURES,
    add_atr5,
    add_bar_features,
    add_idx_gap_pct,
    add_m3_m5_features,
    add_m3_m5_std_features,
    add_ret_vs_idx,
)
from strategy.mkt.train import _prepare_data, load_model_by_type

# 即時推論只需要最近一段 rolling window，不需要 load_m1()/load_day() 預設的
# 全部歷史（2026-07-25討論：load_m1() 讀全部歷史一次要40秒+上百萬列，是
# process 記憶體爆掉的主因之一）。60天遠超過這裡實際用到的「最後一個交易日
# 均量排名」「昨收算跳空缺口」需求，留足夠緩衝應付連假。
_LOOKBACK_DAYS = 60


def _recent_start_date() -> str:
    return (pd.Timestamp.now() - pd.Timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")


def _class_proba(model, df: pd.DataFrame, cls: int):
    """取指定類別（0=跌/1=平/2=漲）那一欄機率，不假設 classes_ 順序（雖然
    目前sklearn對整數標籤預設會照數值排序，仍明確用 class_idx 對應，避免
    哪天標籤改成非數值/亂序時默默取錯欄）。"""
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    return model.predict_proba(df[FEATURES])[:, class_idx[cls]]


def _up_proba(model, df: pd.DataFrame):
    """取「漲」（class=2）那一欄機率——回測（predict()）跟共用回測引擎
    backtest/intraday_platform.py 只吃單一（做多）機率矩陣，這裡維持原樣
    不動；即時訊號（predict_live()）另外用 _down_proba() 補「跌」訊號，
    見該函式說明。"""
    return _class_proba(model, df, 2)


def _down_proba(model, df: pd.DataFrame):
    """取「跌」（class=0）那一欄機率——2026-07-23討論：這欄之前算出來後
    直接被丟掉，即時訊號只看得到做多訊號。只加在 predict_live()（訊號層，
    給前端顯示用），不動 predict()/共用回測引擎——那邊只認單一機率矩陣，
    沒有方向概念，要支援回測/實際下單放空是完全另一個規模的工程。"""
    return _class_proba(model, df, 0)


def predict(model=None, test_days: int = 30, test_only: bool = True, use_cache: bool = False) -> pd.DataFrame:
    """
    對全天資料產生「漲」機率矩陣（index=datetime, columns=stock_id），回測用。

    直接沿用 train.py::_prepare_data() 整條 pipeline（流動性過濾、時段過濾、
    特徵、cache機制都跟訓練時完全一致，不用另外重寫一份），target 欄位這裡
    用不到，但反正 _prepare_data() 本來就會算，直接忽略即可。

    test_only=True（預設）時，只回傳最後 test_days 天——這幾天是模型訓練時
    沒看過的樣本外資料，跟 train.py 切分訓練/測試集用同一套 cutoff 邏輯
    （見 train.py::_split_data() 的說明）。test_only=False 才會回傳全部（含
    訓練集），訓練集算出來的績效會被模型「背過答案」灌水，沒有參考價值，
    只用來除錯。
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    df = _prepare_data(use_cache=use_cache)
    if test_only:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        df = df[df["date"] > cutoff]

    df = df.copy()
    df["proba"] = _up_proba(model, df)
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def build_prewarm_cache(day_trade_stocks: set | None = None) -> dict:
    """
    盤前預算快取 — 給 strategy/prewarm.py 統一呼叫的介面，比照
    strategy/orb/predict.py、strategy/rally/predict.py 的 build_prewarm_cache()。

    2026-07-25討論：不再用「前一日量前N名」動態排名（TOP_N，見
    strategy/mkt/config.py股票母體那段說明）——改成固定讀
    db/tickers/tick_universe.parquet的400支，是靜態清單，不用像以前那樣
    現算load_m1()均量排名（load_m1()讀全部歷史很慢），這裡也不再需要。

    day_trade_stocks：strategy/prewarm.py 統一傳入的當沖候選清單，這裡不用
    ——mkt 自己的候選池就是tick_universe這400支，用途跟day_trade_stocks
    那份候選清單不同，收下只是為了跟其他策略維持同一組介面，不要拿去篩
    資料（會變成拿候選清單去篩「用來決定候選清單的原始資料」，邏輯本末
    倒置）。

    回傳的 dict key 要跟 predict_live() 接受的參數名一致，因為
    main/live_trader.py 會直接 **cache 展開傳進 predict_live()。
    """
    return {"universe_stock_ids": set(load_tick_universe())}


def predict_live(
    minute_str: str,
    day: pd.DataFrame | None = None,
    universe_stock_ids: set | None = None,
    model=None,
    threshold: float = 0.6,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論。

    ⚠️ 參數順序（尤其 day 是第2個位置參數）要跟 orb/rally 的 predict_live()
    完全一致——main/live_trader.py 對所有策略模組一視同仁，統一用
    `s.predict_live(minute_str, state.day, model=..., threshold=0,
    day_trade_stocks=..., m1_live=..., **s.prewarm_cache)` 這種寫法呼叫
    （state.day 是位置參數），不會知道也不會檢查個別策略實際用不用得到
    day。錯位的話 state.day 會被誤傳進下一個參數的位置（例如當成
    m1_live，整條 pipeline 全部算錯，2026-07-21發現過這個bug），所以就算
    用不到也要保留在同一個位置。2026-07-23起 idx_gap_pct（0050開盤跳空
    缺口）需要用到它，不再是純粹為了介面相容才收下、不使用。

    universe_stock_ids: 股票母體（tick_universe固定400支），見
        build_prewarm_cache()。留空則內部自己呼叫load_tick_universe()（很快，
        是靜態清單，不用像以前top_n設計那樣載入m1算均量排名）。
    day_trade_stocks: 當沖標的 set，若提供則只推論這些股票（跟
        universe_stock_ids 是「且」的關係，兩邊都要通過才會納入候選）。
    m1_live: 已載入的今日即時分K，傳入可避免每次呼叫都重讀 db/m1_live/
        （live_trader.py 應該自己維護一份、每分鐘更新後傳進來）。留空則沿用
        舊行為，內部自己 load_m1_live()。

    ⚠️ 跟 orb/rally 的 predict_live() 不同：那兩支一開始就用 day_trade_stocks
    篩掉 m1_live，這裡不能這麼做——ret_vs_idx 要用 0050 自己的分K資料當基準
    （見 features.py::add_ret_vs_idx()），如果 0050 不在當沖名單裡就會被
    先篩掉，之後就沒有基準可以算。所以這裡先對完整 m1_live 算完 ret_vs_idx，
    才排除 0050 本身、套用 universe_stock_ids/day_trade_stocks 篩選。

    回傳格式：[{"stock_id": ..., "proba": ..., "price": ..., "direction": "up"|"down"}, ...]
    ——同一支股票在同一分鐘理論上只會出現一次（做多機率、做空機率各自比
    threshold，兩邊都沒過就不會出現；兩邊都過門檻的話兩筆都會回傳，但
    「同時看漲又看跌」機率上幾乎不會發生，3分類機率加總=1，除非門檻設
    很低）。純訊號，不是下單指令——direction 只是標記model判斷的方向，
    見 2026-07-23 討論。
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    date_str = minute_str[:10]
    if m1_live is None:
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    m1_live = m1_live.copy()
    m1_live["date"] = pd.to_datetime(m1_live["date"])
    m1_live = m1_live.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1_live["day_date"] = m1_live["date"].dt.date

    # ret_vs_idx 要靠 0050 自己的資料算基準，一定要在排除/篩選之前先算完
    # （見上面 docstring 的說明），add_ret_vs_idx() 內部用 merge(on="date")
    # 且已對0050去重，會保留 m1_live 原本依 stock_id/date 排序的列順序，
    # 後面 compute_m3/compute_m5 的 rolling 計算才不會因為列序被打亂而算錯。
    df = add_ret_vs_idx(m1_live)

    # 0050開盤跳空缺口（idx_gap_pct，2026-07-23討論）：db/fugle_day 是GHA
    # 批次工作、收盤後才更新，「今天」這個day_date在load_day()裡完全沒有
    # 0050的列——merge(on="day_date")找不到today的列會讓idx_gap_pct整天
    # 都是NaN。修法：補一列0050「今天」的佔位列到day裡（值不重要，只需要
    # day_date存在，讓add_idx_gap_pct()內部的shift(1)能對到正確位置），
    # 理由同舊版day_ret_vs_idx實驗（已刪除）遇到的問題。也是要在排除0050
    # 之前算，理由同add_ret_vs_idx()。
    if day is None:
        day = load_day(start_date=_recent_start_date())
        if day_trade_stocks:
            # 0050 一定要留著，不管有沒有在 day_trade_stocks 裡——下面
            # add_idx_gap_pct() 要靠 day 裡的 0050 列算跳空缺口，被濾掉
            # 的話這個特徵會整批變 NaN（同樣的坑見 strategy/rally/predict.py
            # 的說明）。
            day = day[day["stock_id"].isin(day_trade_stocks | {IDX_SYMBOL})]
    day = day.copy()
    day["date"] = pd.to_datetime(day["date"])
    today_ts = pd.Timestamp(date_str)
    if not ((day["stock_id"] == IDX_SYMBOL) & (day["date"] == today_ts)).any():
        day = pd.concat(
            [day, pd.DataFrame([{"stock_id": IDX_SYMBOL, "date": today_ts, "close": None}])],
            ignore_index=True,
        )
    df = add_idx_gap_pct(df, day)
    df = df[df["stock_id"] != IDX_SYMBOL]

    if universe_stock_ids is None:
        universe_stock_ids = set(load_tick_universe())
    df = df[df["stock_id"].isin(universe_stock_ids)]
    if day_trade_stocks:
        df = df[df["stock_id"].isin(day_trade_stocks)]
    if df.empty:
        return []

    df = add_bar_features(df)

    # db/m3、db/m5、db/m3_std、db/m5_std 都是批次預算、不含「今天」的資料，
    # 即時推論要對 m1_live 現算，用 data/resample.py 共用的同一套函式（跟
    # 批次預算腳本用的是同一份邏輯），保證跟訓練用的公式一致，理由同
    # strategy/rally/predict.py、strategy/orb/predict.py 的 predict_live()。
    m3_live = compute_m3(df)
    m5_live = compute_m5(df)
    df = add_m3_m5_features(df, m3_live, m5_live)

    m3_std_live = compute_m3_std(df)
    m5_std_live = compute_m5_std(df)
    df = add_m3_m5_std_features(df, m3_std_live, m5_std_live)

    # atr5只拿來過濾用，不進FEATURES，見 features.py::add_atr5() 的說明。
    df = add_atr5(df)

    current = df[df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []

    # ATR5平盤過濾（絕對門檻，跌/平/漲一視同仁，2026-07-23討論）：跟
    # train.py::_prepare_data() 用同一個門檻，train/test/上線推論三邊一致，
    # 見 strategy/mkt/config.py::ATR5_FILTER_THRESHOLD 的說明。
    current = current[current["atr5"] >= ATR5_FILTER_THRESHOLD]
    if current.empty:
        return []

    valid = current.dropna(subset=FEATURES)
    if valid.empty:
        return []

    up_proba = _up_proba(model, valid)
    down_proba = _down_proba(model, valid)
    signals = []
    for (_, row), up_p, down_p in zip(valid.iterrows(), up_proba, down_proba):
        if up_p >= threshold:
            signals.append(
                {"stock_id": row["stock_id"], "proba": float(up_p), "price": float(row["close"]), "direction": "up"}
            )
        if down_p >= threshold:
            signals.append(
                {"stock_id": row["stock_id"], "proba": float(down_p), "price": float(row["close"]), "direction": "down"}
            )
    return sorted(signals, key=lambda x: -x["proba"])
