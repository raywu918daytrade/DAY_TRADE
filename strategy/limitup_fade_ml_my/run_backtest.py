"""
limitup_fade_ml 事件做空回測 — 對照規則基線 vs ML 過濾，並印交易明細
（格式對齊 backtest.intraday_platform.print_trades）。

共用引擎 intraday_platform 只支援做多，這裡用事件表自行結算做空
（進場=09:03 m3_close，出場=做空 TB ±3% / 13:25）。

用法：
    # IDE：改 __main__ 變數後直接跑
    python -m strategy.limitup_fade_ml.run_backtest

    # CLI
    python -m strategy.limitup_fade_ml.run_backtest --start_date 2024-01-01 --end_date 2026-07-31 --use_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from backtest.intraday_platform import print_trades
from data.adjustment_query import _adjust_ohlc
from data.raw_query import iter_m1_months
from strategy.limitup_fade_ml_my.config import MODEL_TYPE, THRESHOLD, TP_PCT, SL_PCT
from strategy.limitup_fade_ml_my.dataset import short_triple_barrier_detail
from strategy.limitup_fade_ml_my.features import FEATURES
from strategy.limitup_fade_ml_my.train import _prepare_data, load_model_by_type


def _summarize(label: str, df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print(f"  {label}: n=0", flush=True)
        return
    n_tp = int((df["target"] == 2).sum())
    n_sl = int((df["target"] == 0).sum())
    n_flat = int((df["target"] == 1).sum())
    win_close = 100.0 * df["short_win_close"].mean() if "short_win_close" in df else float("nan")
    mean_close = 100.0 * df["short_ret_to_close"].mean() if "short_ret_to_close" in df else float("nan")
    print(
        f"  {label}: n={n:,}  TB止盈={n_tp / n * 100:.1f}%  止損={n_sl / n * 100:.1f}%  "
        f"震盪={n_flat / n * 100:.1f}%  | 收盤勝={win_close:.1f}%  "
        f"short→收 mean={mean_close:.3f}%",
        flush=True,
    )


def _resolve_exits(ev: pd.DataFrame) -> pd.DataFrame:
    """逐月 M1 還原出場價／時間（只對傳入的事件列）。"""
    if ev.empty:
        return ev
    ev = ev.copy().reset_index(drop=True)
    for col in ("exit_ts", "exit_price", "exit_reason", "bars_held"):
        ev[col] = np.nan if col != "exit_reason" else None

    need_sids = set(ev["stock_id"].astype(str))
    need_days = set(ev["day_str"])
    key_to_idx = {(str(r.stock_id), r.day_str): i for i, r in enumerate(ev.itertuples(index=False))}
    pending = set(key_to_idx)
    start_date = min(need_days)

    print(f"  解析出場（{len(ev)} 筆，逐月 M1）...", flush=True)
    for m1_raw in iter_m1_months(start_date=start_date):
        if not pending:
            break
        m1_raw = m1_raw[m1_raw["stock_id"].astype(str).isin(need_sids)].copy()
        if m1_raw.empty:
            continue
        m1 = _adjust_ohlc(m1_raw, start_date)
        m1["stock_id"] = m1["stock_id"].astype(str)
        m1["date"] = pd.to_datetime(m1["date"], format="mixed")
        m1["day_str"] = m1["date"].dt.strftime("%Y-%m-%d")
        m1 = m1[m1["day_str"].isin(need_days)]
        if m1.empty:
            continue
        grouped = dict(tuple(m1.groupby(["stock_id", "day_str"], sort=False)))
        for key in list(pending):
            if key not in grouped:
                continue
            i = key_to_idx[key]
            row = ev.iloc[i]
            detail = short_triple_barrier_detail(
                grouped[key],
                pd.Timestamp(row["trigger_ts"]),
                float(row["entry_price"]),
            )
            if detail is None:
                continue
            ev.at[i, "exit_ts"] = detail["exit_ts"]
            ev.at[i, "exit_price"] = detail["exit_price"]
            ev.at[i, "exit_reason"] = detail["exit_reason"]
            ev.at[i, "bars_held"] = detail["bars_held"]
            pending.discard(key)

    if pending:
        print(f"  警告：{len(pending)} 筆無法解析出場，改用日收近似", flush=True)
        for key in pending:
            i = key_to_idx[key]
            entry = float(ev.at[i, "entry_price"])
            target = int(ev.at[i, "target"])
            if target == 2:
                ev.at[i, "exit_price"] = entry * (1.0 - TP_PCT)
                ev.at[i, "exit_reason"] = "tp"
            elif target == 0:
                ev.at[i, "exit_price"] = entry * (1.0 + SL_PCT)
                ev.at[i, "exit_reason"] = "sl"
            else:
                ev.at[i, "exit_price"] = float(ev.at[i, "day_close"])
                ev.at[i, "exit_reason"] = "time"
            ev.at[i, "exit_ts"] = pd.Timestamp(ev.at[i, "day_str"])
            ev.at[i, "bars_held"] = 0
    return ev


def events_to_trades(ev: pd.DataFrame) -> pd.DataFrame:
    """事件表 → print_trades 相容的做空 trades_df（pnl 為空單損益）。"""
    if ev.empty:
        return pd.DataFrame()
    rows = []
    lot_value = 100_000.0  # 名義本金，僅方便看 pnl 金額
    for r in ev.itertuples(index=False):
        entry = float(r.entry_price)
        exit_p = float(r.exit_price)
        if entry <= 0 or not np.isfinite(exit_p):
            continue
        pnl_pct = (entry - exit_p) / entry * 100.0  # short
        pnl = lot_value * pnl_pct / 100.0
        rows.append(
            {
                "stock_id": str(r.stock_id),
                "entry_dt": pd.Timestamp(r.trigger_ts),
                "entry_price": entry,
                "entry_proba": float(getattr(r, "proba_tp", np.nan)),
                "exit_dt": pd.Timestamp(r.exit_ts),
                "exit_price": exit_p,
                "exit_reason": str(r.exit_reason),
                "bars_held": int(r.bars_held) if pd.notna(r.bars_held) else 0,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )
    return pd.DataFrame(rows)


def _print_trade_summary(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        print("無交易記錄", flush=True)
        return
    print(f"\n總交易次數: {len(trades_df)}", flush=True)
    print(f"勝率      : {(trades_df['pnl'] > 0).mean() * 100:.1f}%", flush=True)
    print(f"平均報酬  : {trades_df['pnl_pct'].mean():.2f}%", flush=True)
    print(f"出場原因  :\n{trades_df.groupby('exit_reason').size().to_string()}", flush=True)


def run(
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    threshold: float = THRESHOLD,
    use_cache: bool = True,
    test_frac: float = 0.2,
    model_type: str = MODEL_TYPE,
    show_trades: bool = True,
) -> pd.DataFrame:
    """跑事件做空回測；回傳 ML 進場的 trades_df。"""
    print("limitup_fade_ml 事件回測（做空）", flush=True)
    print(f"區間 {start_date} ~ {end_date}  threshold={threshold}", flush=True)
    df = _prepare_data(use_cache=use_cache, start_date=start_date, end_date=end_date)
    if df.empty:
        print("無事件", flush=True)
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1.0 - test_frac))
    cut = max(1, min(cut, len(df) - 1)) if len(df) > 1 else len(df)
    test_df = df.iloc[cut:].copy()

    print("\n" + "=" * 56)
    print("規則基線（硬過濾全量）")
    print("=" * 56)
    _summarize("全部觸發", df)
    print("\n規則基線（時間切分測試集）", flush=True)
    _summarize("test", test_df)

    try:
        model = load_model_by_type(model_type)
    except FileNotFoundError as e:
        print(f"\n無模型，跳過 ML 欄：{e}", flush=True)
        return pd.DataFrame()

    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    test_df["proba_tp"] = proba[:, class_idx[2]] if 2 in class_idx else 0.0
    ml = test_df[test_df["proba_tp"] >= threshold].copy()

    print("\n" + "=" * 56)
    print(f"ML 過濾（僅 test）P(止盈)>={threshold}")
    print("=" * 56)
    _summarize("ML 進場", ml)
    if len(ml) and len(test_df):
        print(
            f"  test 覆蓋率={len(ml) / len(test_df) * 100:.1f}%  " f"mean proba={ml['proba_tp'].mean():.3f}",
            flush=True,
        )

    if ml.empty:
        print("\n無 ML 進場，無交易記錄", flush=True)
        return pd.DataFrame()

    ml = _resolve_exits(ml)
    trades_df = events_to_trades(ml)
    print("\n" + "=" * 56)
    print("交易摘要（ML 進場／做空）")
    print("=" * 56)
    _print_trade_summary(trades_df)
    if show_trades:
        print("\n── 交易記錄 ──", flush=True)
        print_trades(trades_df)
    return trades_df


def main(
    start_date: str = "2024-01-01",
    end_date: str = "2026-07-31",
    threshold: float = THRESHOLD,
    use_cache: bool = True,
    test_frac: float = 0.2,
    model_type: str = MODEL_TYPE,
    show_trades: bool = True,
):
    if len(sys.argv) > 1:
        p = argparse.ArgumentParser()
        p.add_argument("--start_date", default=start_date)
        p.add_argument("--end_date", default=end_date)
        p.add_argument("--threshold", type=float, default=threshold)
        p.add_argument("--use_cache", action="store_true", default=use_cache)
        p.add_argument("--no_cache", action="store_true")
        p.add_argument("--test_frac", type=float, default=test_frac)
        p.add_argument("--model_type", default=model_type)
        p.add_argument("--no_trades", action="store_true")
        args = p.parse_args()
        start_date = args.start_date
        end_date = args.end_date
        threshold = args.threshold
        use_cache = False if args.no_cache else True
        test_frac = args.test_frac
        model_type = args.model_type
        show_trades = not args.no_trades

    return run(
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        use_cache=use_cache,
        test_frac=test_frac,
        model_type=model_type,
        show_trades=show_trades,
    )


if __name__ == "__main__":
    # IDE 直接跑：改這裡即可
    start_date = "2024-01-01"
    end_date = "2026-07-31"
    threshold = 0.6
    use_cache = True
    test_frac = 0.2
    model_type = "lgbm"
    show_trades = True
    main(
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        use_cache=use_cache,
        test_frac=test_frac,
        model_type=model_type,
        show_trades=show_trades,
    )
