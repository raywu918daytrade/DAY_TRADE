"""
當沖回測引擎（分K）

輸入：
    price       分K 收盤價，index=datetime，columns=stock_id
    entries     進場訊號 bool DataFrame，shape 同 price
    df_proba    模型預測機率（可選），用於同一分鐘多支進場時排序

邏輯：
    每根分K 先處理出場，再處理進場
    出場條件：停損 / 停利 / 持有超過 hold_bars 根 / 強制收盤出場
    當天未出場的部位在 force_exit_bar 強制平倉
"""

from pathlib import Path

import pandas as pd


def intraday_backtest(
    price: pd.DataFrame,
    entries: pd.DataFrame,
    df_proba: pd.DataFrame = None,
    max_positions: int = 3,
    sl_pct: float = 0.03,       # 固定停損 3%
    tp_pct: float = 0.03,       # 固定停利 3%
    hold_bars: int = 30,        # 最多持有幾根分K
    force_exit_time: str = "13:25",  # 強制出場時間（收盤前）
    last_entry_time: str = "10:00",  # 最晚進場時間（早盤動能窗口）
    init_cash: float = 1_000_000,
    fee_rate: float = 0.001425,  # 手續費（買賣各）
    tax_rate: float = 0.003,     # 交易稅（賣出）
):
    # 每天分開 ffill，避免跨日填補；解決某些股票特定分K無成交的 NaN 問題
    price = (
        price.groupby(price.index.date, group_keys=False)
             .apply(lambda g: g.ffill())
    )

    cash = float(init_cash)
    positions: dict = {}
    trades: list = []
    portfolio_records: list = []

    force_exit_t  = pd.Timestamp(force_exit_time).time()
    last_entry_t  = pd.Timestamp(last_entry_time).time()

    for i, dt in enumerate(price.index):
        is_force_exit = dt.time() >= force_exit_t

        # ── 出場 ─────────────────────────────────────────────────────────────
        for sid in list(positions.keys()):
            if sid not in price.columns:
                continue
            p = price.loc[dt, sid]
            pos = positions[sid]
            bars_held = i - pos["entry_bar"]

            hit_sl = p <= pos["sl"]
            hit_tp = p >= pos["tp"]
            hit_hold = bars_held >= hold_bars

            should_exit = is_force_exit or hit_sl or hit_tp or hit_hold

            if should_exit:
                exit_price = pos["sl"] if hit_sl else (pos["tp"] if hit_tp else p)
                sell_value = exit_price * pos["shares"] * (1 - fee_rate - tax_rate)
                cash += sell_value

                reason = (
                    "force_exit" if is_force_exit
                    else "sl" if hit_sl
                    else "tp" if hit_tp
                    else "hold_bars"
                )
                trades.append({
                    "stock_id": sid,
                    "entry_dt": pos["entry_dt"],
                    "entry_price": pos["entry_price"],
                    "exit_dt": dt,
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "bars_held": bars_held,
                    "pnl": sell_value - pos["cost"],
                    "pnl_pct": (exit_price / pos["entry_price"] - 1) * 100,
                })
                del positions[sid]

        # ── 進場（超過最晚進場時間不開新倉）─────────────────────────────────
        if not is_force_exit and dt.time() <= last_entry_t:
            slots = max_positions - len(positions)
            if slots > 0:
                row = entries.loc[dt]
                candidates = row[row].index.tolist()
                candidates = [s for s in candidates if s not in positions and s in price.columns]

                if df_proba is not None and dt in df_proba.index:
                    candidates.sort(key=lambda s: df_proba.loc[dt, s] if s in df_proba.columns else 0, reverse=True)

                for sid in candidates[:slots]:
                    p = price.loc[dt, sid]
                    if p <= 0:
                        continue
                    cost = cash / slots
                    if cost <= 0:
                        break
                    shares = cost * (1 - fee_rate) / p
                    cash -= cost
                    positions[sid] = {
                        "shares": shares,
                        "entry_price": p,
                        "entry_dt": dt,
                        "entry_bar": i,
                        "cost": cost,
                        "sl": p * (1 - sl_pct),
                        "tp": p * (1 + tp_pct),
                    }

        # ── 每根資產快照 ──────────────────────────────────────────────────────
        pos_value = sum(
            price.loc[dt, sid] * pos["shares"]
            for sid, pos in positions.items()
            if sid in price.columns
        )
        portfolio_records.append({
            "dt": dt,
            "cash": round(cash, 2),
            "market_value": round(pos_value, 2),
            "total": round(cash + pos_value, 2),
        })

    portfolio_df = pd.DataFrame(portfolio_records).set_index("dt")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
        "stock_id", "entry_dt", "entry_price", "exit_dt", "exit_price",
        "exit_reason", "bars_held", "pnl", "pnl_pct",
    ])
    return portfolio_df, trades_df
