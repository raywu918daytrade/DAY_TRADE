"""
ret5_pullback_ml 回測 — 做多事件級：模型止盈機率門檻篩選 vs 純規則全交易。

用法：
    python -m strategy.ret5_pullback_ml.run_backtest --use_cache \\
        --start_date 2024-01-01 --end_date 2026-07-31 --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from strategy.ret5_pullback_ml.config import FEE_RATE, TAX_RATE
from strategy.ret5_pullback_ml.features import FEATURES
from strategy.ret5_pullback_ml.train import _split_data, load_model_by_type

_TRADE_COLS = [
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


def _simulate_long_trades(
    df: pd.DataFrame,
    threshold: float | None,
    max_positions: int,
    init_cash: float,
    fee_rate: float,
    tax_rate: float,
):
    """逐日做多：threshold=None 則全事件；否則 proba_tp>=threshold。"""
    sig = df.copy()
    if threshold is not None:
        sig = sig[sig["proba_tp"] >= threshold]
    if sig.empty:
        return pd.DataFrame(columns=["total"]), pd.DataFrame(columns=_TRADE_COLS)

    sig["trade_day"] = pd.to_datetime(sig["trigger_ts"]).dt.strftime("%Y-%m-%d")
    cash = float(init_cash)
    trades: list[dict] = []
    portfolio_records: list[dict] = []

    for day, g in sig.sort_values("trigger_ts").groupby("trade_day", sort=True):
        g = g.sort_values("proba_tp", ascending=False).head(max_positions)
        n = len(g)
        if n == 0 or cash <= 0:
            portfolio_records.append({"date": day, "total": cash})
            continue

        notional = cash / n
        day_pnl = 0.0
        for _, row in g.iterrows():
            entry_price = float(row["entry"])
            exit_price = float(row["exit_price"])
            shares = notional / entry_price
            cost = entry_price * shares * (1 + fee_rate)
            proceeds = exit_price * shares * (1 - fee_rate - tax_rate)
            pnl = proceeds - cost
            day_pnl += pnl
            trades.append(
                {
                    "stock_id": row["stock_id"],
                    "entry_dt": row["trigger_ts"],
                    "entry_price": entry_price,
                    "entry_proba": float(row["proba_tp"]),
                    "exit_dt": row["exit_ts"],
                    "exit_price": exit_price,
                    "exit_reason": row["exit_reason"],
                    "bars_held": row["bars_held"],
                    "pnl": pnl,
                    "pnl_pct": (exit_price - entry_price) / entry_price * 100,
                }
            )
        cash += day_pnl
        portfolio_records.append({"date": day, "total": cash})

    trades_df = pd.DataFrame(trades, columns=_TRADE_COLS) if trades else pd.DataFrame(columns=_TRADE_COLS)
    portfolio_df = (
        pd.DataFrame(portfolio_records).set_index("date")
        if portfolio_records
        else pd.DataFrame(columns=["total"])
    )
    return portfolio_df, trades_df


def _print_summary(label: str, portfolio_df: pd.DataFrame, trades_df: pd.DataFrame, init_cash: float):
    print(f"\n=== {label} ===")
    if portfolio_df.empty or trades_df.empty:
        print("無交易")
        return
    final = portfolio_df["total"].iloc[-1]
    ret = (final / init_cash - 1) * 100
    print(f"最終資產  : {final:,.0f}")
    print(f"總報酬率  : {ret:.2f}%")
    print(f"總交易次數: {len(trades_df)}")
    print(f"勝率      : {(trades_df['pnl'] > 0).mean() * 100:.1f}%")
    print(f"平均報酬  : {trades_df['pnl_pct'].mean():.2f}%")
    print(f"出場原因  :\n{trades_df.groupby('exit_reason').size().to_string()}")


def run(
    test_days: int = 90,
    threshold: float = 0.5,
    max_positions: int = 10,
    model_type: str = "lgbm",
    init_cash: float = 1_000_000,
    use_cache: bool = True,
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    use_tick_universe: bool = False,
):
    model = load_model_by_type(model_type)
    _, test_df = _split_data(
        test_days=test_days,
        use_cache=use_cache,
        start_date=start_date,
        end_date=end_date,
        use_tick_universe=use_tick_universe,
    )
    print(f"測試事件: {len(test_df):,} 筆")
    if test_df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=_TRADE_COLS)

    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    test_df = test_df.copy()
    test_df["proba_tp"] = proba[:, class_idx[2]] if 2 in class_idx else 0.0

    # baseline: 全交易
    port0, tr0 = _simulate_long_trades(
        test_df, threshold=None, max_positions=max_positions, init_cash=init_cash,
        fee_rate=FEE_RATE, tax_rate=TAX_RATE,
    )
    _print_summary("純規則全交易（test）", port0, tr0, init_cash)

    port1, tr1 = _simulate_long_trades(
        test_df, threshold=threshold, max_positions=max_positions, init_cash=init_cash,
        fee_rate=FEE_RATE, tax_rate=TAX_RATE,
    )
    _print_summary(f"模型 p_tp≥{threshold:.2f}", port1, tr1, init_cash)

    # no-fee mean for apples-to-apples with verify
    print("\n── 未扣成本 mean pnl（對齊 verify）──")
    print(f"  全交易: n={len(test_df)} mean={100 * test_df['pnl_pct'].mean():.3f}%")
    sub = test_df[test_df["proba_tp"] >= threshold]
    if len(sub):
        print(
            f"  p_tp≥{threshold:.2f}: n={len(sub)} mean={100 * sub['pnl_pct'].mean():.3f}%  "
            f"TP={100 * (sub['target'] == 2).mean():.1f}% SL={100 * (sub['target'] == 0).mean():.1f}%"
        )
    return port1, tr1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ret5_pullback_ml 做多事件級回測")
    parser.add_argument("--test_days", type=int, default=90)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_positions", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="lgbm", choices=["lgbm"])
    parser.add_argument("--init_cash", type=float, default=1_000_000)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--no_cache", action="store_true", help="強制重建事件")
    parser.add_argument("--start_date", type=str, default="2024-01-01")
    parser.add_argument("--end_date", type=str, default="2026-07-31")
    parser.add_argument("--use_tick_universe", action="store_true")
    args = parser.parse_args()
    run(
        test_days=args.test_days,
        threshold=args.threshold,
        max_positions=args.max_positions,
        model_type=args.model_type,
        init_cash=args.init_cash,
        use_cache=not args.no_cache,
        start_date=args.start_date,
        end_date=args.end_date,
        use_tick_universe=args.use_tick_universe,
    )
