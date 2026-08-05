"""
limitup_fade_ml 回測入口 — 空單事件級回測，自寫出場模擬（不套用
backtest/intraday_platform.py 的共用引擎，因為那支引擎是純多單設計：
sl=entry*(1-sl_pct)/tp=entry*(1+sl_pct)、買進持股、pnl=賣出值-成本，無法直接
用在空單上）。只重用該檔案的 print_trades() 欄位格式，方便跟其他策略的輸出
對照。

事件本身的出場（TP/SL/時間牆）已經在 strategy/limitup_fade_ml/dataset.py::
short_triple_barrier_label() 算好（entry_price/exit_price/exit_reason/bars_held），
這裡只需要：套模型 proba 門檻篩選要不要進場、模擬同一天多檔觸發時的資金分配。

用法：
    python strategy/limitup_fade_ml/run_backtest.py
    python strategy/limitup_fade_ml/run_backtest.py --threshold 0.5 --max_threshold 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from backtest.intraday_platform import print_trades
from strategy.limitup_fade_ml.config import FEE_RATE, TAX_RATE
from strategy.limitup_fade_ml.predict import predict
from strategy.limitup_fade_ml.train import load_model_by_type

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


def _simulate_short_trades(
    df: pd.DataFrame,
    threshold: float,
    max_positions: int,
    init_cash: float,
    fee_rate: float,
    tax_rate: float,
    max_threshold: float | None = None,
):
    """逐日模擬：每天 09:03 觸發的候選中，proba>=threshold（可選再用
    max_threshold 卡上限，只驗證某個信心度區間，例如 --threshold 0.5
    --max_threshold 0.6）且取信心度最高的 max_positions 檔，等權分配當下
    資金做空。同一天所有部位都在 13:25 前平倉（Triple Barrier 的時間牆），
    天與天之間沒有跨夜部位，可以逐日獨立結算資金再往下一天複利。"""
    sig = df[df["proba"] >= threshold]
    if max_threshold is not None:
        sig = sig[sig["proba"] < max_threshold]
    sig = sig.copy()
    if sig.empty:
        return pd.DataFrame(columns=["total"]), pd.DataFrame(columns=_TRADE_COLS)

    sig["trade_day"] = sig["trigger_ts"].dt.strftime("%Y-%m-%d")

    cash = float(init_cash)
    trades: list[dict] = []
    portfolio_records: list[dict] = []

    for day, g in sig.sort_values("trigger_ts").groupby("trade_day", sort=True):
        g = g.sort_values("proba", ascending=False).head(max_positions)
        n = len(g)
        if n == 0 or cash <= 0:
            portfolio_records.append({"date": day, "total": cash})
            continue

        notional = cash / n
        day_pnl = 0.0
        for _, row in g.iterrows():
            entry_price = float(row["entry_price"])
            exit_price = float(row["exit_price"])
            shares = notional / entry_price

            # 空單：進場賣出收現金（扣手續費），出場買回付現金（扣手續費+交易稅）
            proceeds = entry_price * shares * (1 - fee_rate)
            cost = exit_price * shares * (1 + fee_rate + tax_rate)
            pnl = proceeds - cost
            day_pnl += pnl

            trades.append(
                {
                    "stock_id": row["stock_id"],
                    "entry_dt": row["trigger_ts"],
                    "entry_price": entry_price,
                    "entry_proba": float(row["proba"]),
                    "exit_dt": row["exit_dt"],
                    "exit_price": exit_price,
                    "exit_reason": row["exit_reason"],
                    "bars_held": row["bars_held"],
                    "pnl": pnl,
                    "pnl_pct": (entry_price - exit_price) / entry_price * 100,
                }
            )

        cash += day_pnl
        portfolio_records.append({"date": day, "total": cash})

    trades_df = pd.DataFrame(trades, columns=_TRADE_COLS) if trades else pd.DataFrame(columns=_TRADE_COLS)
    portfolio_df = (
        pd.DataFrame(portfolio_records).set_index("date") if portfolio_records else pd.DataFrame(columns=["total"])
    )
    return portfolio_df, trades_df


def _print_summary(portfolio_df: pd.DataFrame, trades_df: pd.DataFrame, init_cash: float):
    if portfolio_df.empty:
        print("無交易日")
        return
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


def run(
    test_days: int = 60,
    threshold: float = 0.6,
    max_threshold: float | None = None,
    max_positions: int = 10,
    model_type: str = "lgbm",
    init_cash: float = 1_000_000,
    use_cache: bool = True,
    start_date: str | None = "2022-01-01",
):
    model = load_model_by_type(model_type)
    df = predict(model=model, test_days=test_days, use_cache=use_cache, start_date=start_date)
    print(f"測試事件: {len(df):,} 筆")
    if df.empty:
        print("無測試事件")
        return pd.DataFrame(), pd.DataFrame(columns=_TRADE_COLS)

    portfolio_df, trades_df = _simulate_short_trades(
        df,
        threshold=threshold,
        max_threshold=max_threshold,
        max_positions=max_positions,
        init_cash=init_cash,
        fee_rate=FEE_RATE,
        tax_rate=TAX_RATE,
    )
    print()
    print_trades(trades_df)
    _print_summary(portfolio_df, trades_df, init_cash)
    return portfolio_df, trades_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="limitup_fade_ml 空單事件級回測")
    parser.add_argument("--test_days", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--max_threshold", type=float, default=None, help="只驗證 [threshold, max_threshold) 這個信心度區間"
    )
    parser.add_argument("--max_positions", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="lgbm", choices=["lgbm"])
    parser.add_argument("--init_cash", type=float, default=1_000_000)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--start_date", type=str, default="2022-01-01")
    args = parser.parse_args()
    run(
        test_days=args.test_days,
        threshold=args.threshold,
        max_threshold=args.max_threshold,
        max_positions=args.max_positions,
        model_type=args.model_type,
        init_cash=args.init_cash,
        use_cache=args.use_cache,
        start_date=args.start_date,
    )
