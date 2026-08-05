"""
mkt 策略 —— 核心特徵：個股跟大盤（0050）的累積報酬率差值。

第一版先只做這一個特徵（ret_vs_idx），驗證數字正確後再逐步加其他特徵，
不一次寫完整套 pipeline（見 2026-07-14 的討論：先精簡、慢慢加，方便debug）。
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from strategy.mkt.config import HOLD_BARS, IDX_SYMBOL, SL_PCT, TP_PCT


def add_ret_vs_idx(m1: pd.DataFrame, idx_symbol: str = IDX_SYMBOL) -> pd.DataFrame:
    """
    算「個股從今日開盤累積到現在的報酬率」減去「0050從今日開盤累積到現在
    的報酬率」，逐分鐘更新，回傳加了 ret_vs_idx 欄位的 m1（複本，不動傳
    進來的原始 df）。順便廣播 0050 自己「這一分鐘」的報酬率（idx_ret_1m，
    不是累積型），給 add_bar_features() 算「0050漲個股跌」這種分鐘級的
    背離旗標用（2026-07-20 討論）；也留下 0050 自己「從開盤累積到現在」的
    報酬率（idx_ret_since_open，2026-07-22討論加回FEATURES）——同樣是
    ret_vs_idx=-2%，「大盤漲2%、個股沒漲」（補漲情境，策略假設成立）跟
    「大盤跌1%、個股跌3%」（兩個都在跌，不是補漲邏輯）意義完全不同，之前
    只算了差值、沒把大盤自己的方向留給模型看，是個漏洞。

    >0 代表這支股票從開盤到現在，漲得比大盤多（相對強勢）；
    <0 代表跑輸大盤（相對弱勢）。

    這支函式一定要在 top_n_by_prev_day_volume()（流動性過濾，會把0050濾掉）
    之前呼叫，理由見下面 idx_symbol 那段——濾掉之後就沒有0050的列可以算了。

    m1 需已有 stock_id/date/day_date/open/close 欄位，date 需為 datetime，
    day_date 為 date（不是 datetime）。
    """
    m1 = m1.copy()
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    m1["_ret_since_open"] = m1["close"] / day_open - 1
    m1["_ret_1m"] = g_day["close"].pct_change(1, fill_method=None)

    idx = (
        m1[m1["stock_id"] == idx_symbol][["date", "_ret_since_open", "_ret_1m"]]
        .rename(columns={"_ret_since_open": "idx_ret_since_open", "_ret_1m": "idx_ret_1m"})
        .drop_duplicates("date")
    )

    m1 = m1.merge(idx, on="date", how="left")
    m1["ret_vs_idx"] = m1["_ret_since_open"] - m1["idx_ret_since_open"]
    m1 = m1.drop(columns=["_ret_since_open", "_ret_1m"])
    return m1


def add_idx_gap_pct(m1: pd.DataFrame, day: pd.DataFrame, idx_symbol: str = IDX_SYMBOL) -> pd.DataFrame:
    """
    0050開盤跳空缺口：今天開盤價 vs 昨天收盤價的漲跌幅（2026-07-23討論）。
    整天固定不變，不是像 ret_vs_idx/idx_ret_since_open 那樣逐分鐘更新——
    反映的是隔夜/開盤前的市場情緒，跟盤中才發生的事是不同資訊，不是
    idx_ret_since_open 的重複。

    今天開盤價：從 m1 裡 0050 自己當日第一根K棒抓（即時可用，不用等
    db/fugle_day 收盤後才更新）。昨天收盤價：從 day（歷史日K）抓，shift(1)
    往前推一天。

    ⚠️ 這支函式假設 day 裡每個 m1 出現過的 day_date，都能在 day 裡找到
    「前一個交易日」的列（訓練用全部歷史批次資料本來就滿足）。即時推論
    （db/fugle_day 收盤後才更新，今天這個day_date在day裡完全沒有列）要先
    在 day 補一列「今天」的佔位列（值不重要，只需要day_date存在，讓
    shift(1)排到正確位置），理由與做法跟舊版 day_ret_vs_idx 實驗
    （已刪除）遇到的問題一樣，見 predict.py::predict_live() 的處理。

    m1 需已有 stock_id/day_date/date/open 欄位。day 需為 load_day() 原始
    輸出（全市場、含0050）。
    """
    idx_m1 = m1[m1["stock_id"] == idx_symbol][["day_date", "date", "open"]].sort_values("date")
    idx_today_open = idx_m1.groupby("day_date")["open"].first().reset_index()
    idx_today_open = idx_today_open.rename(columns={"open": "_idx_today_open"})

    idx_day = day[day["stock_id"] == idx_symbol][["date", "close"]].copy()
    idx_day["date"] = pd.to_datetime(idx_day["date"])
    idx_day = idx_day.sort_values("date")
    idx_day["day_date"] = idx_day["date"].dt.date
    idx_day["_idx_prev_close"] = idx_day["close"].shift(1)
    idx_day = idx_day[["day_date", "_idx_prev_close"]]

    gap = idx_today_open.merge(idx_day, on="day_date", how="left")
    gap["idx_gap_pct"] = gap["_idx_today_open"] / gap["_idx_prev_close"] - 1
    gap = gap[["day_date", "idx_gap_pct"]]

    m1 = m1.merge(gap, on="day_date", how="left")
    return m1


def add_bar_features(m1: pd.DataFrame) -> pd.DataFrame:
    """
    當根1分鐘K棒自己的量能/波動特徵（2026-07-19 討論，先精簡加一批基本
    盤中特徵）：
        vol_ratio_cum   這一分鐘成交量 / 今日累積到目前為止的平均每分鐘量
                        - 1（pct_change形式，0=沒變、正=比今天平常放量、
                        負=縮量，反映「這分鐘是不是比今天平常還爆量」）——
                        原始股數（volume）不同股票量級差很多沒辦法跨股票
                        比較，這裡直接用比例，不另外存一份原始股數的欄位
        vol_ratio_prev  這一分鐘成交量 / 前一分鐘量 - 1（同樣是pct_change
                        形式，對短期急拉/急縮量比較敏感）
        atr7            1分鐘K ATR(7) 相對波動（True Range 7根滾動平均 / 當日
                        開盤價，比照 strategy/orb/features.py 的 atr7 算法）
        ret_1m          這一分鐘收盤 vs 前一分鐘收盤的變化率
        range_pct       這一分鐘 high-low 寬度 / 當日開盤價
        body_pct        這一分鐘 close-open 變化率 / 當日開盤價
        bullish_volume_surge 這一分鐘收紅K + 量至少是前一分鐘1.5倍的旗標
                        （0/1，2026-07-19 討論，「單一K棒翻轉」試過沒訊號後
                        改成這個更簡單的版本）
        bullish_reversal 前一分鐘收黑K、這一分鐘收紅K的翻轉旗標（0/1，
                        2026-07-20 在新時段9:11~9:30重新測，之前在
                        9:00~9:10測過沒訊號，但那個時段已經棄用）
        bearish_divergence 這一分鐘0050漲、但個股跌的背離旗標（0/1，
                        2026-07-20討論。需要先呼叫過 add_ret_vs_idx()
                        算出 idx_ret_1m 這個欄位才能用這支函式）

    m1 若沒有先呼叫過 add_ret_vs_idx()、缺少 idx_ret_1m 欄位，
    bearish_divergence 那一行會直接 KeyError，不會安靜地產生錯的結果。

    range_pct/body_pct 特意不是「除以 high-low」的比值（不是 close_pos 那種
    寫法）——分子就是 high-low / close-open 本身，平盤（high=low）時這兩個
    自然算出 0，不會產生 0/0 的 NaN，不會被 dropna(subset=FEATURES) 濾掉
    （見 2026-07-14 close_pos 那次討論的教訓）。

    m1 需已有 stock_id/date/day_date/open/high/low/close/volume 欄位，
    date 需為 datetime，day_date 為 date（不是 datetime）。
    """
    m1 = m1.copy()
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)

    # pct_change形式（0=沒變、正負=放量/縮量%），跟 ret_1m 的「0點」意義一致，
    # 方便對照著看（對RandomForest本身沒差——樹模型對整體平移不敏感，比值
    # 形式跟pct_change形式切出來的結果一樣，純粹是可讀性考量，2026-07-19討論）。
    cum_vol = g_day["volume"].transform("cumsum")
    bar_count = g_day["volume"].transform("cumcount") + 1
    m1["vol_ratio_cum"] = m1["volume"] / (cum_vol / bar_count).replace(0, np.nan) - 1

    vol_shift1 = g_day["volume"].shift(1)
    # 防止，為 0
    m1["vol_ratio_prev"] = m1["volume"] / vol_shift1.replace(0, np.nan) - 1

    # True Range（首根用開盤價當前收，理由同 rally/orb 的 m1_atr 寫法）
    prev_close = g_day["close"].shift(1).fillna(m1["open"])
    m1["_tr"] = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - prev_close).abs()),
        (m1["low"] - prev_close).abs(),
    )
    g_tr = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["atr7"] = g_tr["_tr"].transform(lambda x: x.rolling(7, min_periods=7).mean()) / day_open
    m1 = m1.drop(columns=["_tr"])

    m1["ret_1m"] = g_day["close"].pct_change(1, fill_method=None)
    m1["range_pct"] = (m1["high"] - m1["low"]) / day_open
    m1["body_pct"] = (m1["close"] - m1["open"]) / day_open

    # 2026-07-19 討論：「前跌+這漲+range放大+量放大」四條件AND試過兩次
    # （單條件版importance 0.47%、四條件版 0.22%，條件越嚴格越沒用，代表
    # 「單一K棒翻轉」這個角度本身沒訊號，不是門檻不夠嚴格的問題），已經
    # 放棄「翻轉」這個框架。改成更簡單的「這一分鐘收紅K + 量放大1.5倍以上」
    # ——不看前一分鐘是不是跌，單純看「這一分鐘是不是有量能明顯佐證的
    # 上漲」。量的條件直接重用上面已經算好的 vol_ratio_prev（這一分鐘量/
    # 前一分鐘量-1），>=0.5 代表量至少是前一分鐘的1.5倍。第一分鐘沒有
    # 「前一分鐘」，vol_ratio_prev 是 NaN，比較結果會是 False（0），不會是
    # NaN。
    m1["bullish_volume_surge"] = ((m1["body_pct"] > 0) & (m1["vol_ratio_prev"] >= 0.5)).astype(int)

    # 翻轉訊號（最單純版本：只有「前一分鐘跌、這一分鐘漲」兩個條件，不加
    # range/量的限制）：2026-07-19 在9:00~9:10這個時段測過importance只有
    # 0.47%，但那時候測的時段後來被判定是雜訊時段、已經棄用（改成
    # 9:11~9:30，見 config.py 的說明）。2026-07-20 在新時段下重新測一次，
    # 不能直接沿用舊時段的結論。
    g_day2 = m1.groupby(["stock_id", "day_date"], group_keys=False)
    body_pct_lag1 = g_day2["body_pct"].shift(1)
    m1["bullish_reversal"] = ((body_pct_lag1 < 0) & (m1["body_pct"] > 0)).astype(int)

    # 大盤漲、個股跌的分鐘級背離旗標（0/1，2026-07-20討論）：跟 ret_vs_idx
    # 不同——ret_vs_idx 是「從開盤累積到現在」的差距，這裡是「這一分鐘」
    # 兩者方向是不是相反，是更即時、不受累積效應影響的版本。idx_ret_1m 是
    # add_ret_vs_idx() 廣播過來的0050這一分鐘報酬率，一定要先呼叫過
    # add_ret_vs_idx() 才會有這個欄位。
    m1["bearish_divergence"] = ((m1["idx_ret_1m"] > 0) & (m1["ret_1m"] < 0)).astype(int)

    return m1


def add_atr5(m1: pd.DataFrame) -> pd.DataFrame:
    """
    1分鐘K ATR(5)相對波動（True Range 5根滾動平均 / 當日開盤價），算法
    跟 add_bar_features() 裡的 atr7 完全一樣，只是window改5（2026-07-23
    討論）。除以day_open是為了讓不同股價水準的股票可以互相比較——同樣
    5元的True Range，對NT$500的股票只是1%波動，對NT$20的股票卻是25%，
    不除以股價沒辦法跨股票比較，這裡沿用atr7/range_pct/body_pct同樣的
    慣例。

    只拿來當 filter_by_atr5_percentile() 過濾「平盤」樣本的依據，**不進
    FEATURES當模型輸入**——理由跟atr7/range_pct被拿掉一樣，這種純波動
    幅度指標（不帶方向）當模型輸入時會被樹拿去做投機分裂、稀釋掉
    ret_vs_idx等真正有方向性的訊號（見 FEATURES 那段 discussion note）。
    當「篩選條件」用就沒有這個問題，因為它根本不會進到模型的輸入裡。

    m1 需已有 stock_id/day_date/open/high/low/close 欄位。
    """
    m1 = m1.copy()
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    day_open = g_day["open"].transform("first").replace(0, np.nan)
    prev_close = g_day["close"].shift(1).fillna(m1["open"])
    m1["_tr5"] = np.maximum(
        np.maximum((m1["high"] - m1["low"]).abs(), (m1["high"] - prev_close).abs()),
        (m1["low"] - prev_close).abs(),
    )
    g_tr = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["atr5"] = g_tr["_tr5"].transform(lambda x: x.rolling(5, min_periods=5).mean()) / day_open
    m1 = m1.drop(columns=["_tr5"])
    return m1


def filter_by_atr5_percentile(m1: pd.DataFrame, min_pr: float = 0.3) -> pd.DataFrame:
    """
    跨股票、同一分鐘的ATR5排名過濾（2026-07-23討論）：同一分鐘，候選股票
    互相比波動度，濾掉排名最後（最平盤）的一部分，只留下相對比較活躍的
    股票。用「跨股票、同一時刻」排名而不是「同一支股票、自己的歷史分布」
    ——沒有lookahead問題（不用回顧過去N天的歷史，只跟同一分鐘的其他候選
    股票比），即時推論可以直接套用同一套邏輯，不用像day-K那些特徵一樣
    處理「今天還沒收盤」的問題。

    min_pr：百分位門檻（0~1），例如0.3代表濾掉每分鐘裡ATR5排名最後30%的
    股票。⚠️ 目前是隨便先設的預設值，還沒看過實際分布/label密度驗證過，
    正式使用前要先分析不同門檻對樣本量、「漲」樣本密度的影響再決定
    （2026-07-23討論）。

    ⚠️ 一定要放在 add_bar_features()/add_m3_m5_features()/
    add_m3_m5_std_features() 等所有lag/rolling特徵都算完之後才呼叫——這裡
    會刪列，如果放在算特徵之前，shift(1)這類lag運算會把不相鄰的兩根K棒
    誤判成相鄰，特徵全部悄悄算錯（2026-07-23討論）。

    m1 需已有 atr5 欄位（先呼叫 add_atr5()）、date 欄位。
    """
    pr = m1.groupby("date")["atr5"].rank(pct=True)
    return m1[pr >= min_pr]


def add_m3_m5_features(m1: pd.DataFrame, m3: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """
    加 m3（3分鐘K）、m5（5分鐘K）的收盤價、成交量特徵（2026-07-20討論），
    比照 strategy/rally/features.py 的 m3_ret/m3_volume 算法：
        m3_ret          3分鐘K收盤價，跟3分鐘前比的報酬率
        m3_vol_ratio    3分鐘K成交量，跟3分鐘前比的變化率（pct_change形式）
        m5_ret          5分鐘K收盤價，跟5分鐘前比的報酬率
        m5_vol_ratio    5分鐘K成交量，跟5分鐘前比的變化率

    db/m3、db/m5（build_m3_m5_rolling.py 預先聚合，見 data/query.py 的
    load_m3()/load_m5()）是「每分鐘一列」的滾動視窗（例如db/m3的09:05那列，
    是09:03~09:05這3根1分K滾動聚合的結果，close就是當下這一分鐘的收盤價，
    volume是這3根的量加總），不是每3分鐘存一列的固定區間——所以能直接用
    stock_id+date跟m1逐列對齊merge，不用另外做時間戳對齊/resample。

    m3/m5需要3/5根K棒才有值，09:00開盤後m3從第3分鐘（約09:02）、m5從第5
    分鐘（約09:04）才開始有值，都在現在的訓練時段9:11~9:30之前，不會
    像atr7那樣犧牲掉時段內的樣本（2026-07-19 atr7暖機期的教訓）。

    m1 需已有 stock_id/day_date/date 欄位，date 需為 datetime。
    """
    m1 = m1.copy()

    m3 = m3[["stock_id", "date", "close", "volume"]].rename(columns={"close": "_m3_close", "volume": "_m3_volume"})
    m5 = m5[["stock_id", "date", "close", "volume"]].rename(columns={"close": "_m5_close", "volume": "_m5_volume"})
    m1 = m1.merge(m3, on=["stock_id", "date"], how="left")
    m1 = m1.merge(m5, on=["stock_id", "date"], how="left")

    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["m3_ret"] = g_day["_m3_close"].pct_change(3, fill_method=None)
    m1["m3_vol_ratio"] = g_day["_m3_volume"].pct_change(3, fill_method=None)
    m1["m5_ret"] = g_day["_m5_close"].pct_change(5, fill_method=None)
    m1["m5_vol_ratio"] = g_day["_m5_volume"].pct_change(5, fill_method=None)

    m1 = m1.drop(columns=["_m3_close", "_m3_volume", "_m5_close", "_m5_volume"])
    return m1


def add_m3_m5_std_features(m1: pd.DataFrame, m3_std: pd.DataFrame, m5_std: pd.DataFrame) -> pd.DataFrame:
    """
    加 m3_std（3分鐘標準獨立K棒）、m5_std（5分鐘標準獨立K棒）的收盤價、
    成交量特徵（2026-07-21討論）。

    跟 add_m3_m5_features() 用的 db/m3、db/m5（rolling版本）不同：db/m3_std、
    db/m5_std（build_m3_m5_std.py 預先聚合，見 data/query.py 的
    load_m3_std()/load_m5_std()）是「標準獨立K棒」——一根K棒對應3/5根1分K，
    只在K棒收的那一分鐘有值（例如m5_std一根覆蓋09:10:00~09:14:59，標記在
    09:15:00），中間分鐘是空的，所以要先算完K棒對K棒的漲跌幅/量變化，再
    merge回m1後逐股逐日ffill，讓每一分鐘都能拿到「最近一根已收盤的獨立
    K棒」的值。

    m3_std_ret/m5_std_ret：目前這根已收盤獨立K棒的收盤價，跟上一根已收盤
    獨立K棒收盤價的報酬率（pct_change(1)——不是像rolling版那樣pct_change
    (3)/(5)，因為std版本本來就是不重疊的獨立K棒，差一根就是差一個完整
    週期）。m3_std_vol_ratio/m5_std_vol_ratio：同上，換成成交量。

    m1 需已有 stock_id/day_date/date 欄位，date 需為 datetime。
    """
    m1 = m1.copy()

    def _bar_changes(bar: pd.DataFrame, prefix: str) -> pd.DataFrame:
        bar = bar.sort_values(["stock_id", "date"]).copy()
        bar["_day"] = bar["date"].dt.date
        g = bar.groupby(["stock_id", "_day"], group_keys=False)
        bar[f"{prefix}_ret"] = g["close"].pct_change(1, fill_method=None)
        bar[f"{prefix}_vol_ratio"] = g["volume"].pct_change(1, fill_method=None)
        return bar[["stock_id", "date", f"{prefix}_ret", f"{prefix}_vol_ratio"]]

    m3_chg = _bar_changes(m3_std, "m3_std")
    m5_chg = _bar_changes(m5_std, "m5_std")

    m1 = m1.merge(m3_chg, on=["stock_id", "date"], how="left")
    m1 = m1.merge(m5_chg, on=["stock_id", "date"], how="left")

    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)
    m1["m3_std_ret"] = g_day["m3_std_ret"].ffill()
    m1["m3_std_vol_ratio"] = g_day["m3_std_vol_ratio"].ffill()
    m1["m5_std_ret"] = g_day["m5_std_ret"].ffill()
    m1["m5_std_vol_ratio"] = g_day["m5_std_vol_ratio"].ffill()

    return m1


def top_n_by_prev_day_volume(m1: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """
    只留每天「依前一交易日全天成交量排序」前 n 名的股票（流動性過濾的簡易
    版本，2026-07-14 討論）。

    用「前一天」的量決定「今天」要不要納入候選池，不是用「今天」的量——
    避免用「今天發生的事」決定「今天要不要看這支股票」的循環問題（例如
    直接拿當天開盤第一分鐘的量排序，會混進「今天剛好爆量」這種跟要驗證
    的訊號本身有關聯的雜訊）。純粹用手上已有的 m1 全天量加總 + shift(1)，
    不用另外載入日K（db/fugle_day/）。

    m1 需已有 stock_id/day_date/volume 欄位。
    """
    daily_vol = m1.groupby(["stock_id", "day_date"])["volume"].sum().reset_index()
    daily_vol = daily_vol.sort_values(["stock_id", "day_date"])
    daily_vol["prev_day_volume"] = daily_vol.groupby("stock_id")["volume"].shift(1)
    daily_vol = daily_vol.dropna(subset=["prev_day_volume"])
    daily_vol["rank"] = daily_vol.groupby("day_date")["prev_day_volume"].rank(ascending=False, method="first")
    keep = daily_vol[daily_vol["rank"] <= n][["stock_id", "day_date"]]
    return m1.merge(keep, on=["stock_id", "day_date"], how="inner")


# ═══════════════════════════════════════════════════════════════════════════════
# Triple Barrier 標籤（3分類版）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 2026-07-14 討論：跟 rally/orb 的二分類 triple barrier 不同——這裡「時間到、
# 兩個barrier都沒碰到」的情況統一標成「平」，不是看當下比進場價高還低硬分
# 漲跌。理由：這支策略驗證出來的訊號優勢只有0.01~0.02個百分點，遠小於
# TP_PCT/SL_PCT=3%，如果沿用 rally/orb 那種「時間到就看sign」的規則，
# 幾乎所有「根本沒有真正波動」的樣本都會被塞進漲或跌的其中一類，稀釋掉
# 真正有意義的樣本。
#
# label： 2=漲（先碰到+TP_PCT）  1=平（HOLD_BARS內都沒碰到任一邊）
#         0=跌（先碰到-SL_PCT）


def _barrier_label_group_3class(
    closes: np.ndarray,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    hold_bars: int = HOLD_BARS,
) -> np.ndarray:
    """每日每股，逐bar往前看最多hold_bars根，判斷先碰到哪個barrier，
    或hold_bars內都沒碰到（平）。"""
    n = len(closes)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        entry = closes[i]
        tp_price = entry * (1 + tp_pct)
        sl_price = entry * (1 - sl_pct)
        future = closes[i + 1 : i + hold_bars + 1]
        tp_idx = np.argmax(future >= tp_price) if (future >= tp_price).any() else len(future)
        sl_idx = np.argmax(future <= sl_price) if (future <= sl_price).any() else len(future)
        if tp_idx < sl_idx:
            labels[i] = 2  # 漲
        elif sl_idx < tp_idx:
            labels[i] = 0  # 跌
        elif len(future) == hold_bars:
            labels[i] = 1  # 平：兩個都沒碰到，且有完整的 hold_bars 可看
        # else: 太靠近當日尾端，future 不足 hold_bars 根，保留 NaN
    return labels


def make_barrier_labels_3class(
    m1: pd.DataFrame,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    hold_bars: int = HOLD_BARS,
) -> pd.Series:
    """對 m1 逐股逐日呼叫 _barrier_label_group_3class()，回傳跟 m1 對齊的
    label Series（2=漲/1=平/0=跌，NaN=樣本不足無法判斷）。"""
    return m1.groupby(["stock_id", "day_date"], group_keys=False).apply(
        lambda g: pd.Series(
            _barrier_label_group_3class(g["close"].values, tp_pct, sl_pct, hold_bars),
            index=g.index,
        ),
        include_groups=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 模型實際用哪些特徵（train.py 從這裡匯入，不要在 train.py 裡另外定義一份）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 2026-07-19：加了 add_bar_features() 這6個當根K棒特徵後，
# feature_importance() 顯示 atr7(45.69%)主導了樹的分割，把驗證過真的有
# 方向性訊號的 ret_vs_idx 重要性稀釋到只剩10.31%，precision/recall全面
# 變差——atr7本質上是「波動幅度」指標，不帶方向資訊。拿掉atr7後（順便拿掉
# 它6根K棒的暖機期，9:00~9:05樣本回來了）發現 range_pct 頂上去變成主導
# （51.12%），同樣的問題重演，尤其「漲」這個類別recall掉到只剩0.03——
# range_pct跟atr7是同一種東西（K棒振幅大小，不帶方向），所以也拿掉了。
# 拿掉atr7/range_pct後重新驗證：feature_importance() 顯示 ret_vs_idx(31.16%)
# 拿回主導地位，precision回升到接近原始1特徵版本（漲23%、跌26%，threshold
# =0.5），確認方向正確。目前「漲」還是比「跌」弱（threshold=0.6時漲recall
# 掉到0，跌還有38%precision），想針對性提升漲的precision試了「單一K棒
# 翻轉」（bullish_reversal：前跌+這漲+range放大+量放大），單條件版
# importance 0.47%、疊到四條件版反而更低（0.22%）——條件越嚴格越沒用，
# 代表「翻轉」這個框架本身沒訊號，不是門檻不夠嚴格。已改成更簡單的
# bullish_volume_surge（這分鐘收紅K+量放大1.5倍以上，不看前一分鐘方向），
# 還沒驗證有沒有效。
# 2026-07-22：加回 idx_ret_since_open（0050自己從開盤累積到現在的報酬率）
# ——之前只有 ret_vs_idx 這個差值，模型看不出「大盤本身是漲是跌」，同樣
# ret_vs_idx=-2%，大盤漲2%個股沒漲（補漲情境）跟大盤跌1%個股跌3%（兩個
# 都在跌）意義完全不同，這是個漏洞，見 add_ret_vs_idx() 的說明。
# 2026-07-23：試過加 idx_gap_pct（0050開盤跳空缺口，今天開盤vs昨天收盤），
# XGB加入前後（同一個test窗口 2026-06-22~07-21）「漲」precision幾乎持平、
# 甚至微降（threshold=0.6: 21%→20%、0.7: 26%→26%、0.8: 34%→33%），沒有幫助，
# 拿掉。函式 add_idx_gap_pct() 本身留著（train.py/predict.py 也還是會呼叫、
# 算出這欄，只是不進FEATURES），之後如果有新想法要用到跳空缺口可以直接
# 重新加回FEATURES這一行，不用重寫。
FEATURES = [
    "ret_vs_idx",
    "idx_ret_since_open",
    # "idx_gap_pct",
    "vol_ratio_cum",
    "vol_ratio_prev",
    "ret_1m",
    "body_pct",
    "bullish_volume_surge",
    "bullish_reversal",
    "bearish_divergence",
    "m3_ret",
    "m3_vol_ratio",
    "m5_ret",
    "m5_vol_ratio",
    "m3_std_ret",
    "m3_std_vol_ratio",
    "m5_std_ret",
    "m5_std_vol_ratio",
    # "range_pct",
    # "atr7",
]
