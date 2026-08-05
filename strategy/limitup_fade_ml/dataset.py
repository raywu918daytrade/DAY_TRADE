"""
limitup_fade_ml 事件挖掘 + Triple Barrier 標籤（空單）。

流程（2026-08-04 三次改版：Stage2延續確認 + 動態ATR停利/停損，理由見
config.py 檔頭說明）：
1. 全市場日K硬過濾候選（前日漲停）：日報酬 >= 9.5%、陽線、body_ratio >= 5%、上影比例 <= 20%
2. 下一交易日跳空開高：今日 open > 前日 close
3. Stage1 觸發：首根 3 分K（db/m3_std @ 09:03，覆蓋 09:00:00~09:02:59）close < open，
   同時算日K ATR14（day_atr），算出 tp_price/sl_price（= m3_close ∓ 1×day_atr）
4. Stage2 延續確認：09:09 那根 M1（涵蓋09:09:00~09:09:59，即「09:10當下」）收盤價
   要比 Stage1 的 m3 收盤價更低，才算延續下跌、真的進場（entry_price=確認價），
   沒有就整筆事件捨棄（視為假突破/已反彈）——只決定要不要進場，不影響 tp_price/
   sl_price（那是從 m3_close 算的，維持不變）
5. Triple Barrier（空單，動態 TP=SL=1×day_atr，時間牆 13:25，從 Stage2 進場
   時間算起）→ target 0=止損 / 1=震盪 / 2=止盈

全新實作（2026-08-04），不依賴／不參考 strategy/limitup_fade_ml.zip 裡的舊版。
"""

from __future__ import annotations

import sys
from datetime import time as dtime
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.adjustment_query import load_pattern_day, load_pattern_m1, load_pattern_m3_std
from finmind.tick_universe import load_tick_universe
from strategy.limitup_fade_ml.config import (
    ATR_WINDOW,
    CONFIRM_TIME,
    FIRST_M3_TIME,
    FORCE_EXIT_TIME,
    LIMIT_UP_RET,
    MAX_UPPER_SHADOW_RATIO,
    MIN_BODY_RATIO,
)

EVENT_COLS = [
    "stock_id",
    "candidate_date",
    "trade_date",
    "prev_close",
    "prev_high",
    "prev_day_ret",
    "prev_body_ratio",
    "prev_upper_shadow_ratio",
    "prev_volume",
    "prev_volume_ratio",
    "prev_volume_z",
    "prev5d_ret",
]

IDX_SYMBOL = "0050"


def _shadow_ratios(o: float, h: float, l: float, c: float) -> tuple[float, float, float]:
    """回傳 (upper_shadow_ratio, lower_shadow_ratio, body_ratio)。全長為 0 時回 (0,0,0)。
    給單筆／即時場景用（predict.py 的 live 路徑也重用這支）；批次向量化計算
    （build_gap_candidates/attach_m3_trigger）走 numpy 向量運算，不逐筆呼叫這支。"""
    full = h - l
    if full <= 0:
        return 0.0, 0.0, 0.0
    upper = h - max(o, c)
    lower = min(o, c) - l
    body = abs(c - o)
    return float(upper / full), float(lower / full), float(body / full)


# ═══════════════════════════════════════════════════════════════════════════
# 前日漲停候選（全市場，向量化）
# ═══════════════════════════════════════════════════════════════════════════


def _next_trade_day(day_str: str, trading_days: np.ndarray) -> str | None:
    idx = np.searchsorted(trading_days, day_str, side="right")
    if idx >= len(trading_days):
        return None
    return str(trading_days[idx])


def build_gap_candidates(day_df: pd.DataFrame) -> pd.DataFrame:
    """全市場掃描前日漲停候選 + 對應下一交易日，回傳每個 (stock_id, trade_date) 一列。

    day_df: data.adjustment_query.load_pattern_day() 完整還原日K（除息造成的跳空已被
    抹平，避免誤判成漲停/開高——理由同 strategy/breakout_retest_ml 的說明）。
    """
    if day_df.empty:
        return pd.DataFrame(columns=EVENT_COLS)

    df = day_df.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    df["day_str"] = df["date"].dt.strftime("%Y-%m-%d")

    g = df.groupby("stock_id")
    prev_close = g["close"].shift(1)
    prev_high = g["high"].shift(1)
    prev_volume_5d_avg = g["volume"].transform(lambda s: s.shift(1).rolling(5).mean())
    prev_volume_20d_avg = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    # 前5日累積報酬（動能）：候選日往前推6個交易日的收盤價 vs 前1日收盤價
    prev5d_ret = g["close"].transform(lambda s: s.shift(1) / s.shift(6) - 1.0)

    day_ret = df["close"] / prev_close - 1.0
    full = df["high"] - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    body = (df["close"] - df["open"]).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        upper_ratio = np.where(full > 0, upper / full, 0.0)
        body_ratio = np.where(full > 0, body / full, 0.0)

    is_bull = df["close"] > df["open"]
    mask = (
        day_ret.notna()
        & (day_ret >= LIMIT_UP_RET)
        & is_bull
        & (body_ratio >= MIN_BODY_RATIO)
        & (upper_ratio <= MAX_UPPER_SHADOW_RATIO)
    )

    cand = df.loc[mask, ["stock_id", "day_str", "close", "volume"]].copy()
    cand["prev_high"] = prev_high[mask].to_numpy()
    cand["prev_day_ret"] = day_ret[mask].to_numpy()
    cand["prev_body_ratio"] = np.round(body_ratio[mask], 4)
    cand["prev_upper_shadow_ratio"] = np.round(upper_ratio[mask], 4)
    cand["prev_volume_5d_avg"] = prev_volume_5d_avg[mask].to_numpy()
    cand["prev_volume_20d_avg"] = prev_volume_20d_avg[mask].to_numpy()
    cand["prev5d_ret"] = prev5d_ret[mask].to_numpy()
    cand = cand.rename(columns={"day_str": "candidate_date", "close": "prev_close", "volume": "prev_volume"})
    cand["prev_volume_ratio"] = cand["prev_volume"] / cand["prev_volume_5d_avg"].replace(0, np.nan)
    cand["prev_volume_z"] = (cand["prev_volume"] - cand["prev_volume_20d_avg"]) / cand["prev_volume_20d_avg"].replace(
        0, np.nan
    )

    trading_days = np.array(sorted(df["day_str"].unique()))
    cand["trade_date"] = cand["candidate_date"].apply(lambda d: _next_trade_day(d, trading_days))
    cand = cand.dropna(subset=["trade_date"])

    cand = cand[EVENT_COLS].drop_duplicates(subset=["stock_id", "trade_date"], keep="last")
    return cand.sort_values(["trade_date", "stock_id"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 日K ATR14（比照 strategy/orb/features.py 的 day_atr 寫法，沒有共用工具，
# 這是既有慣例——mkt/vwap_ml 也是各自抄一份）
# ═══════════════════════════════════════════════════════════════════════════


def _add_day_atr(day: pd.DataFrame) -> pd.DataFrame:
    """對逐股票排序好的日K，算 day_atr（True Range 滾動 ATR_WINDOW 日平均，
    shift(1) 避免用到當天才知道的資訊，除以當日開盤價正規化成比例）。

    在 trade_date 這一列取值時，因為 trade_date 是候選日（漲停日）的下一交易日，
    shift(1) 後的滾動視窗會自然含入漲停當天本身的暴衝波幅，即時反映事件後的
    波動升高，不需要額外處理。"""
    day = day.copy()
    g = day.groupby("stock_id")
    prev_close_atr = g["close"].shift(1)
    day_tr = np.maximum(
        np.maximum((day["high"] - day["low"]).abs(), (day["high"] - prev_close_atr).abs()),
        (day["low"] - prev_close_atr).abs(),
    )
    day["_day_tr"] = day_tr
    atr = day.groupby("stock_id")["_day_tr"].transform(lambda s: s.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean().shift(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        day["day_atr"] = np.where(day["open"] > 0, atr / day["open"], np.nan)
    return day.drop(columns=["_day_tr"])


# ═══════════════════════════════════════════════════════════════════════════
# 隔日跳空開高 + 首根 3 分K 觸發
# ═══════════════════════════════════════════════════════════════════════════


def attach_m3_trigger(candidates: pd.DataFrame, day_df: pd.DataFrame, m3_std_df: pd.DataFrame) -> pd.DataFrame:
    """比對隔日跳空開高（今日 open > 前日 close）+ 首根 3 分K（09:00~09:03）是否下跌，
    通過才附上進場資訊、日ATR14（用於 Triple Barrier 的動態停利/停損距離）。"""
    if candidates.empty or day_df.empty or m3_std_df.empty:
        return pd.DataFrame()

    day = day_df.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    day["day_str"] = day["date"].dt.strftime("%Y-%m-%d")
    day = _add_day_atr(day)
    today = day[["stock_id", "day_str", "open", "day_atr"]].rename(
        columns={"day_str": "trade_date", "open": "today_open"}
    )

    # 0050 大盤缺口（相對強弱用）：候選股跳空幅度扣掉大盤同一天自己的跳空幅度，
    # 分離「整體市場一起跳空」跟「這支股票自己特別跳空」的部分。
    idx = day[day["stock_id"] == IDX_SYMBOL].sort_values("date").copy()
    idx["idx_prev_close"] = idx["close"].shift(1)
    idx["idx_gap_pct"] = idx["open"] / idx["idx_prev_close"] - 1.0
    idx_gap = idx[["day_str", "idx_gap_pct"]].rename(columns={"day_str": "trade_date"})

    m3 = m3_std_df[m3_std_df["date"].dt.time == dtime(*FIRST_M3_TIME)].copy()
    m3["trade_date"] = m3["date"].dt.strftime("%Y-%m-%d")
    m3 = m3.rename(
        columns={
            "date": "trigger_ts",
            "open": "m3_open",
            "high": "m3_high",
            "low": "m3_low",
            "close": "m3_close",
            "volume": "m3_volume",
        }
    )[["stock_id", "trade_date", "trigger_ts", "m3_open", "m3_high", "m3_low", "m3_close", "m3_volume"]]

    out = candidates.merge(today, on=["stock_id", "trade_date"], how="inner")
    out = out.merge(m3, on=["stock_id", "trade_date"], how="inner")
    out = out.merge(idx_gap, on="trade_date", how="left")

    out = out[out["today_open"] > out["prev_close"]]  # 跳空開高
    out = out[out["m3_close"] < out["m3_open"]]  # 首根3分K下跌
    out = out[out["day_atr"].notna()]  # ATR14不足14天歷史（新掛牌股）無法算，捨棄
    if out.empty:
        return out

    out = out.copy()
    out["gap_pct"] = (out["today_open"] - out["prev_close"]) / out["prev_close"]
    out["gap_vs_0050"] = out["gap_pct"] - out["idx_gap_pct"].fillna(0.0)
    out["open_vs_prev_high"] = out["today_open"] / out["prev_high"] - 1.0

    full = out["m3_high"] - out["m3_low"]
    upper = out["m3_high"] - out[["m3_open", "m3_close"]].max(axis=1)
    lower = out[["m3_open", "m3_close"]].min(axis=1) - out["m3_low"]
    body = (out["m3_close"] - out["m3_open"]).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["m3_body_ratio"] = np.where(full > 0, body / full, 0.0)
        out["m3_upper_shadow_ratio"] = np.where(full > 0, upper / full, 0.0)
        out["m3_lower_shadow_ratio"] = np.where(full > 0, lower / full, 0.0)
        out["m3_range_pct"] = np.where(out["m3_open"] > 0, full / out["m3_open"], 0.0)
    out["m3_ret"] = (out["m3_close"] - out["m3_open"]) / out["m3_open"]
    out["dist_from_open_pct"] = (out["m3_close"] - out["today_open"]) / out["today_open"]

    # Triple Barrier 的動態停利/停損（TP=SL=1×day_atr，對稱），參考進場價固定用
    # m3_close（Stage1），不是 Stage2 的 confirm_price——延續確認只決定要不要
    # 進場，不影響停利停損的絕對價位，見 config.py 檔頭說明。
    out["tp_price"] = out["m3_close"] * (1.0 - out["day_atr"])
    out["sl_price"] = out["m3_close"] * (1.0 + out["day_atr"])

    return out.sort_values(["trigger_ts", "stock_id"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2：09:10 延續確認
# ═══════════════════════════════════════════════════════════════════════════


def attach_confirm(triggered: pd.DataFrame, m1_df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 延續確認：用 09:09 那根 M1（涵蓋 09:09:00~09:09:59，即「09:10 當下
    最新一根已收完的分K」）收盤價跟 Stage 1 的 m3_close 比，更低才算延續下跌、
    真的進場；沒有就整筆事件捨棄（視為假突破/已反彈）。

    進場價（entry_price）改成這裡的 confirm_price，取代 Stage1 的 m3_close；
    後續 Triple Barrier 標籤/回測都從這裡的 confirm_ts/confirm_price 算起，
    Stage1 的 trigger_ts/m3_close 繼續保留在事件裡當「初始觸發K棒」特徵用。"""
    if triggered.empty or m1_df.empty:
        return pd.DataFrame()

    confirm_time = dtime(*CONFIRM_TIME)
    confirm = m1_df[m1_df["date"].dt.time == confirm_time].copy()
    confirm["trade_date"] = confirm["date"].dt.strftime("%Y-%m-%d")
    confirm = confirm.rename(columns={"date": "confirm_ts", "close": "confirm_price"})[
        ["stock_id", "trade_date", "confirm_ts", "confirm_price"]
    ]

    out = triggered.merge(confirm, on=["stock_id", "trade_date"], how="inner")
    out = out[out["confirm_price"] < out["m3_close"]]  # 09:03~09:10延續下跌才進場
    if out.empty:
        return out

    out = out.copy()
    out["confirm_ret"] = (out["confirm_price"] - out["m3_close"]) / out["m3_close"]
    out["entry_price"] = out["confirm_price"]

    return out.sort_values(["confirm_ts", "stock_id"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Triple Barrier 標籤（空單）
# ═══════════════════════════════════════════════════════════════════════════


def short_triple_barrier_label(
    m1_day: pd.DataFrame,
    trigger_ts: pd.Timestamp,
    entry: float,
    tp_price: float,
    sl_price: float,
) -> dict:
    """空單 Triple Barrier：進場後逐根 M1 檢查，先觸價位者定輸贏；都沒觸發則在
    FORCE_EXIT_TIME（時間牆）那根的收盤價強制平倉。

    tp_price/sl_price 是呼叫端已經算好的絕對出場價位（2026-08-04 三次改版：
    動態 ATR 停利/停損，不再是 entry 的固定 ±3%——tp_price/sl_price 可能跟
    entry 是不同基準價算出來的，見 attach_m3_trigger() 的說明，這裡只負責
    比價，不管價位怎麼來的）。entry 只用來檢查 <=0 的無效輸入，不參與計算。

    做空「止盈」= 價格下跌到 tp_price；「止損」= 價格上漲到 sl_price。同一根
    K棒內高低都觸及兩個 barrier 時，跟 repo 既有慣例一致，優先判定止盈
    （見 strategy/breakout_retest_ml/features.py::_triple_barrier_label() 的順序）。

    回傳 dict：target(0/1/2)/exit_dt/exit_price/exit_reason/bars_held；資料不足時
    target=None（呼叫端應丟棄該筆事件，避免用不完整的未來資料硬湊標籤）。
    """
    if m1_day is None or m1_day.empty or entry <= 0:
        return {"target": None}

    force_t = pd.Timestamp(f"{trigger_ts.strftime('%Y-%m-%d')} {FORCE_EXIT_TIME}")
    fut = m1_day[(m1_day["date"] > trigger_ts) & (m1_day["date"] <= force_t)].sort_values("date")
    if fut.empty:
        return {"target": None}

    bars_held = 0
    for _, row in fut.iterrows():
        bars_held += 1
        if float(row["low"]) <= tp_price:
            return {
                "target": 2,
                "exit_dt": row["date"],
                "exit_price": tp_price,
                "exit_reason": "tp",
                "bars_held": bars_held,
            }
        if float(row["high"]) >= sl_price:
            return {
                "target": 0,
                "exit_dt": row["date"],
                "exit_price": sl_price,
                "exit_reason": "sl",
                "bars_held": bars_held,
            }

    last_ts = fut["date"].iloc[-1]
    if last_ts < force_t - pd.Timedelta(minutes=5):
        # M1 資料在時間牆前就中斷（例如當天缺分K），標籤不可信
        return {"target": None}

    last_close = float(fut["close"].iloc[-1])
    return {
        "target": 1,
        "exit_dt": last_ts,
        "exit_price": last_close,
        "exit_reason": "force_exit",
        "bars_held": bars_held,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 端到端建構事件資料集
# ═══════════════════════════════════════════════════════════════════════════


def build_events(start_date: str | None = "2022-01-01") -> pd.DataFrame:
    """端到端：日K候選 → 隔日跳空+首根3分K觸發(Stage1@09:03) → 09:10延續確認
    (Stage2) → M1 打 Triple Barrier 標籤（從 Stage2 的 confirm_ts/entry_price 算起）。

    候選股票母體收斂到 finmind.tick_universe.load_tick_universe()（固定
    400檔+0050，2026-08-04 決定，見 README「候選股票母體」的說明）：這是
    唯一每天被 update_daily.py 主動更新/校正的股票池，池外股票的
    db/adjustment_day 資料不再被日常流程碰過，新鮮度/正確性都沒有保障
    （查過 2832/6720/1294/6794 等池外股票，發現資料有缺天數/異常值，
    2832 甚至完全不在 db/d1 裡）。"""
    day_df = load_pattern_day(start_date=start_date)
    if day_df.empty:
        print("[limitup_fade_ml] 無日K資料", flush=True)
        return pd.DataFrame()

    universe = set(load_tick_universe())
    day_df = day_df[day_df["stock_id"].isin(universe)]
    if day_df.empty:
        print("[limitup_fade_ml] tick_universe 篩選後無日K資料", flush=True)
        return pd.DataFrame()

    candidates = build_gap_candidates(day_df)
    print(f"[limitup_fade_ml] 前日漲停候選: {len(candidates):,}", flush=True)
    if candidates.empty:
        return pd.DataFrame()

    m3_std_df = load_pattern_m3_std(start_date=start_date)
    triggered = attach_m3_trigger(candidates, day_df, m3_std_df)
    print(f"[limitup_fade_ml] 跳空開高+首根3分K下跌觸發(Stage1@09:03): {len(triggered):,}", flush=True)
    if triggered.empty:
        return pd.DataFrame()

    m1_df = load_pattern_m1(start_date=start_date)
    m1_df = m1_df[m1_df["stock_id"].isin(triggered["stock_id"].unique())].copy()

    confirmed = attach_confirm(triggered, m1_df)
    print(f"[limitup_fade_ml] 09:10延續確認(Stage2): {len(confirmed):,}", flush=True)
    if confirmed.empty:
        return pd.DataFrame()

    m1_df["day_str"] = m1_df["date"].dt.strftime("%Y-%m-%d")
    m1_groups = {k: v for k, v in m1_df.groupby(["stock_id", "day_str"])}

    events: list[dict] = []
    for _, ev in confirmed.iterrows():
        key = (ev["stock_id"], ev["trade_date"])
        m1_day = m1_groups.get(key)
        label = short_triple_barrier_label(
            m1_day,
            pd.Timestamp(ev["confirm_ts"]),
            float(ev["entry_price"]),
            float(ev["tp_price"]),
            float(ev["sl_price"]),
        )
        if label.get("target") is None:
            continue
        row = ev.to_dict()
        row.update(label)
        events.append(row)

    print(f"[limitup_fade_ml] 有效標籤事件: {len(events):,}", flush=True)
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values(["confirm_ts", "stock_id"]).reset_index(drop=True)
