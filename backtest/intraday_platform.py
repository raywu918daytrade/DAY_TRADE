"""
當沖回測平台

流程：
    1. 載入分K資料 + 模型預測機率
    2. gen_entries：依機率篩選進場訊號
    3. intraday_backtest：執行回測
    4. 輸出績效摘要
"""

import sys
from pathlib import Path
from time import time

import pandas as pd

# 確保能從根目錄導入
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.intraday_backtest import intraday_backtest
from data.query import load_m1

_ROOT = Path(__file__).parent


# ── 資料準備 ──────────────────────────────────────────────────────────────────


def prepare_price(m1: pd.DataFrame) -> pd.DataFrame:
    """分K 收盤價 pivot：index=datetime，columns=stock_id"""
    return m1.pivot(index="date", columns="stock_id", values="close")


def gen_entries(
    df_proba: pd.DataFrame,
    threshold: float = 0.6,
    top_n: int = 5,
    min_volume: pd.DataFrame = None,
    min_vol_threshold: int = 500,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    每分鐘從預測機率 >= threshold 的股票中，取機率最高的 top_n 支作為進場候選。
    若提供 min_volume，額外過濾 vol_window 根均量低於 min_vol_threshold 的股票。

    vol_window 預設 20（原本的行為，rally 用這個），策略如果集中在開盤前
    vol_window 分鐘內進場（例如 orb），rolling(vol_window) 在暖機期間會是
    NaN、把候選整批濾掉，要把 vol_window 調小（例如 orb 傳 10，見
    strategy/orb/run_backtest.py），不要直接改這裡的預設值影響到其他策略。
    """
    entries = df_proba >= threshold

    if min_volume is not None:
        avg_vol = min_volume.rolling(vol_window).mean()
        entries = entries & (avg_vol >= min_vol_threshold)

    # 超過 top_n 時，只保留機率最高的幾支
    rank = df_proba.where(entries).rank(axis=1, ascending=False)
    entries = entries & (rank <= top_n)
    return entries.fillna(False)


# ── 主回測流程 ────────────────────────────────────────────────────────────────


def run_backtest(
    df_proba: pd.DataFrame,
    threshold: float = 0.55,
    top_n: int = 5,
    max_positions: int = 10,
    sl_pct: float = 0.03,
    tp_pct: float = 0.03,
    hold_bars: int = 30,
    force_exit_time: str = "13:25",
    first_entry_time: str = "09:01",
    last_entry_time: str = "10:00",
    init_cash: float = 1_000_000,
    min_vol_threshold: int = 500,
    vol_window: int = 20,
):
    t0 = time()

    print("載入分K...")
    m1 = load_m1()

    # 只保留 df_proba 涵蓋的日期範圍（+ 幾天緩衝給 rolling volume 暖機用），
    # 不要整段歷史都拿去 pivot/回測——df_proba 通常只有最後 test_days 天
    # （見 strategy/*/predict.py 的 test_only 篩選），只交易得到這個窗口，
    # 但 gen_entries() 的 `entries & (avg_vol >= ...)` 是兩個布林 DataFrame
    # 做 & 運算，index 不同時 pandas 會取聯集而不是交集，若這裡不先篩，
    # entries 的 index 會被 avg_vol（全歷史）撐大，intraday_backtest() 的
    # 主迴圈就會跑遍全部歷史分K，而不是只跑實際可能交易的這幾十天
    # （2026-07-17 發現：ORB 回測因此多跑 10 個月分K，慢了10倍以上）。
    start = df_proba.index.min() - pd.Timedelta(days=5)
    end = df_proba.index.max()
    m1 = m1[(m1["date"] >= start) & (m1["date"] <= end)]

    price = prepare_price(m1)
    volume = m1.pivot(index="date", columns="stock_id", values="volume")

    print("產生進場訊號...")
    entries = gen_entries(
        df_proba,
        threshold=threshold,
        top_n=top_n,
        min_volume=volume,
        min_vol_threshold=min_vol_threshold,
        vol_window=vol_window,
    )

    # 對齊欄位
    common_stocks = price.columns.intersection(entries.columns)
    price = price[common_stocks]
    entries = entries[common_stocks]
    df_proba_aligned = df_proba.reindex(columns=common_stocks)

    print("執行回測...")
    portfolio_df, trades_df = intraday_backtest(
        price=price,
        entries=entries,
        df_proba=df_proba_aligned,
        max_positions=max_positions,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        hold_bars=hold_bars,
        force_exit_time=force_exit_time,
        first_entry_time=first_entry_time,
        last_entry_time=last_entry_time,
        init_cash=init_cash,
    )

    print(f"回測時間: {time() - t0:.1f} 秒")
    _print_summary(portfolio_df, trades_df, init_cash)
    return portfolio_df, trades_df


def _print_summary(portfolio_df, trades_df, init_cash):
    final = portfolio_df["total"].iloc[-1]
    ret = (final / init_cash - 1) * 100
    print(f"\n最終資產  : {final:,.0f}")
    print(f"總報酬率  : {ret:.2f}%")
    if not trades_df.empty:
        print(f"總交易次數: {len(trades_df)}")
        print(f"勝率      : {(trades_df['pnl'] > 0).mean() * 100:.1f}%")
        print(f"平均報酬  : {trades_df['pnl_pct'].mean():.2f}%")
        by_reason = trades_df.groupby("exit_reason").size()
        print(f"出場原因  :\n{by_reason.to_string()}")


def print_trades(trades_df: pd.DataFrame):
    """印出完整交易記錄，供人工驗證。"""
    if trades_df.empty:
        print("無交易記錄")
        return
    cols = [
        "stock_id",
        "entry_dt",
        "entry_price",
        "entry_proba",
        "exit_dt",
        "exit_price",
        "exit_reason",
        "bars_held",
        "pnl",
        "pnl_pct",
    ]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 130)
    print(trades_df[cols].to_string(index=False))


# ── Optuna 參數搜尋 ───────────────────────────────────────────────────────────


def optimize_optuna(df_proba: pd.DataFrame, n_trials: int = 50):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    m1 = load_m1()
    start = df_proba.index.min() - pd.Timedelta(days=5)
    end = df_proba.index.max()
    m1 = m1[(m1["date"] >= start) & (m1["date"] <= end)]
    price = prepare_price(m1)
    volume = m1.pivot(index="date", columns="stock_id", values="volume")

    def objective(trial):
        threshold = trial.suggest_float("threshold", 0.5, 0.9)
        top_n = trial.suggest_int("top_n", 3, 10)
        sl_pct = trial.suggest_float("sl_pct", 0.002, 0.02)
        tp_pct = trial.suggest_float("tp_pct", 0.005, 0.03)
        hold_bars = trial.suggest_int("hold_bars", 10, 60)

        entries = gen_entries(df_proba, threshold=threshold, top_n=top_n, min_volume=volume)
        common = price.columns.intersection(entries.columns)
        pv, _ = intraday_backtest(
            price=price[common],
            entries=entries[common],
            df_proba=df_proba.reindex(columns=common),
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            hold_bars=hold_bars,
        )
        return pv["total"].iloc[-1] / pv["total"].iloc[0] - 1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n最佳報酬: {study.best_value:.2%}")
    print("最佳參數:")
    for k, v in best.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return study


# ── 執行入口 ──────────────────────────────────────────────────────────────────
#
# 這支檔案是共用回測引擎（gen_entries/run_backtest/print_trades/
# optimize_optuna），不綁特定策略。各策略自己的回測入口：
#     python strategy/rally/run_backtest.py
#     python strategy/orb/run_backtest.py
