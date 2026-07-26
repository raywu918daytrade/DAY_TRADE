"""
vwap_ml walk-forward 回測 — 跟 experiments/walk_forward.py 用同一組5個
expanding window，但這次連實際交易報酬率也一起看（不只是分類
precision/recall），確認 run_backtest.py 單次測到的正報酬（+0.60%、
勝率71.4%）是不是在多個獨立窗口下都穩定，不是單一窗口運氣好（同樣的
教訓見 strategy/mkt/README.md 的說明）。

只回測「下軌回歸→做多」這一側——backtest/intraday_platform.py 共用引擎
只能模擬做多，跟 strategy/vwap_ml/predict.py::_direction_probas()
（2026-07-26起只交易回歸、不交易延續）的邏輯一致，不是另外一套規則。

用法：
    python -m strategy.vwap_ml.experiments.walk_forward_backtest
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from backtest.intraday_platform import run_backtest
from strategy.vwap_ml.config import BACKTEST_HOLD_BARS, BACKTEST_SL_PCT, BACKTEST_TP_PCT, SESSION_END, SESSION_START
from strategy.vwap_ml.experiments.walk_forward import _fit
from strategy.vwap_ml.features import FEATURES
from strategy.vwap_ml.train import _prepare_data

_WINDOW_DAYS = 45
_N_WINDOWS = 5
_INIT_CASH = 1_000_000


def run(
    start_date: str = "2024-01-01",
    threshold: float = 0.6,
    top_n: int = 5,
    max_positions: int = 10,
    min_train_rows: int = 5000,
    use_cache: bool = True,
):
    """
    window_days/n_windows 固定跟 experiments/walk_forward.py 一致（45天
    ×5個窗口），才能跟那次的precision/recall結果對照著看同一段期間。

    threshold/top_n/max_positions 意義跟 run_backtest.py::run() 一致。
    """
    df = _prepare_data(use_cache=use_cache, start_date=start_date)
    df = df.sort_values("date").reset_index(drop=True)
    max_date = df["date"].max()

    summaries = []
    for i in range(_N_WINDOWS):
        test_end = max_date - pd.Timedelta(days=_WINDOW_DAYS * i)
        test_start = test_end - pd.Timedelta(days=_WINDOW_DAYS)
        train_df = df[df["date"] < test_start]
        test_df = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()
        window_label = f"{test_start.date()}~{test_end.date()}"

        if len(train_df) < min_train_rows or test_df.empty:
            print(f"窗口 {i + 1}（{window_label}）資料量不足，跳過（train={len(train_df):,}, test={len(test_df):,}）")
            continue

        model = _fit(train_df)
        proba = model.predict_proba(test_df[FEATURES])
        class_idx = {c: idx for idx, c in enumerate(model.classes_)}
        p_revert = proba[:, class_idx[0]]
        is_upper = (test_df["trigger_side"] == "upper").to_numpy()
        # 只留下軌回歸→做多這一側，上軌回歸(做空)這裡先算0，共用回測引擎
        # 只能模擬做多，見 predict.py::_direction_probas() 的同樣做法。
        test_df["proba"] = np.where(is_upper, 0.0, p_revert)
        df_proba = test_df.pivot(index="date", columns="stock_id", values="proba")

        print(f"\n{'=' * 60}\n窗口 {i + 1}: {window_label}（train={len(train_df):,}, test={len(test_df):,}）\n{'=' * 60}")
        portfolio_df, trades_df = run_backtest(
            df_proba,
            threshold=threshold,
            top_n=top_n,
            max_positions=max_positions,
            sl_pct=BACKTEST_SL_PCT,
            tp_pct=BACKTEST_TP_PCT,
            hold_bars=BACKTEST_HOLD_BARS,
            first_entry_time=f"{SESSION_START[0]:02d}:{SESSION_START[1]:02d}",
            last_entry_time=f"{SESSION_END[0]:02d}:{SESSION_END[1]:02d}",
            init_cash=_INIT_CASH,
        )

        final = portfolio_df["total"].iloc[-1]
        ret = (final / _INIT_CASH - 1) * 100
        n_trades = len(trades_df)
        win_rate = (trades_df["pnl"] > 0).mean() * 100 if n_trades else float("nan")
        avg_pnl_pct = trades_df["pnl_pct"].mean() if n_trades else float("nan")
        summaries.append(
            {
                "window": window_label,
                "return_pct": ret,
                "n_trades": n_trades,
                "win_rate": win_rate,
                "avg_pnl_pct": avg_pnl_pct,
            }
        )

    summary_df = pd.DataFrame(summaries)
    print(f"\n{'=' * 60}\n各窗口回測摘要\n{'=' * 60}")
    if summary_df.empty:
        print("沒有任何窗口跑出結果")
        return summary_df
    print(summary_df.to_string(index=False))
    print(
        f"\n平均報酬率: {summary_df['return_pct'].mean():.2f}%（std={summary_df['return_pct'].std():.2f}%）\n"
        f"平均勝率  : {summary_df['win_rate'].mean():.1f}%\n"
        f"平均交易數: {summary_df['n_trades'].mean():.1f}"
    )
    return summary_df


if __name__ == "__main__":
    run()
