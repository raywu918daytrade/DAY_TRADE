"""
券資比 + 近幾日漲幅 → 後續日報酬方向。

券資比 = short_sale_today_balance / margin_purchase_today_balance
訊號日 t（收盤後可知）→ 進場 t+1 open；forward 自進場起算。

用法：
    python -m strategy.breakout_retest_ml.experiments.verify_margin_short_ratio \\
        --start_date 2026-06-01 --end_date 2026-07-31
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from data.adjustment_query import load_pattern_day, load_pattern_day_by_stock

_ROOT = Path(__file__).resolve().parents[3]
_MARGIN_DIR = _ROOT / "db" / "margin"


def _load_margin(start_date: str, end_date: str) -> pd.DataFrame:
    paths = sorted(_MARGIN_DIR.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["stock_id"] = df["stock_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    m = df["margin_purchase_today_balance"].astype(float)
    s = df["short_sale_today_balance"].astype(float)
    df["short_margin_ratio"] = np.where(m > 0, s / m, np.nan)  # 券資比
    df["margin_short_ratio"] = np.where(s > 0, m / s, np.nan)  # 資券比
    return df


def _is_stock_id(sid: str) -> bool:
    """4 碼數字個股；排除 00xx ETF。"""
    s = str(sid)
    if len(s) != 4 or not s.isdigit():
        return False
    if s.startswith("00"):
        return False
    return True


def _prep_day(start_date: str, end_date: str) -> pd.DataFrame:
    # 多載一段歷史算 lookback；多載一段未來算 forward
    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    day = load_pattern_day(start_date=hist)
    if day.empty:
        return day
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].map(_is_stock_id)].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)

    g = day.groupby("stock_id", sort=False)
    day["close_prev3"] = g["close"].shift(3)
    day["close_prev5"] = g["close"].shift(5)
    day["ret_3d"] = day["close"] / day["close_prev3"] - 1.0
    day["ret_5d"] = day["close"] / day["close_prev5"] - 1.0

    # 進場日 t+1：open_next；ret_Hd = close_{t+H} / entry（H=1/3/5）
    day["open_next"] = g["open"].shift(-1)
    day["close_next"] = g["close"].shift(-1)
    day["fwd_close_1"] = g["close"].shift(-1)
    day["fwd_close_3"] = g["close"].shift(-3)
    day["fwd_close_5"] = g["close"].shift(-5)
    return day


def _signal_dates_0050_entry_open_down(start_date: str, end_date: str) -> set[pd.Timestamp]:
    """訊號日 t：進場日 t+1 的 0050 開盤 < t 收盤（開盤跌）。"""
    etf = load_pattern_day_by_stock("0050")
    if etf is None or etf.empty:
        return set()
    etf = etf.copy()
    etf["date"] = pd.to_datetime(etf["date"], format="mixed")
    etf = etf.sort_values("date").reset_index(drop=True)
    etf["open_next"] = etf["open"].astype(float).shift(-1)
    etf["close"] = etf["close"].astype(float)
    ok = (
        (etf["date"] >= start_date)
        & (etf["date"] <= end_date)
        & etf["open_next"].notna()
        & (etf["open_next"] < etf["close"])
    )
    return set(etf.loc[ok, "date"])


def _entry_price(row: pd.Series) -> float:
    o = float(row["open_next"]) if pd.notna(row["open_next"]) else float("nan")
    if np.isfinite(o) and o > 0:
        return o
    c = float(row["close_next"]) if pd.notna(row["close_next"]) else float("nan")
    return c if np.isfinite(c) and c > 0 else float("nan")


def _build_events(margin: pd.DataFrame, day: pd.DataFrame) -> pd.DataFrame:
    m = margin[margin["stock_id"].map(_is_stock_id)].copy()
    cols = [
        "stock_id",
        "date",
        "ret_3d",
        "ret_5d",
        "open_next",
        "close_next",
        "fwd_close_1",
        "fwd_close_3",
        "fwd_close_5",
        "close",
    ]
    d = day[cols].copy()
    ev = m.merge(d, on=["stock_id", "date"], how="inner")
    if ev.empty:
        return ev
    entries = ev.apply(_entry_price, axis=1)
    ev["entry"] = entries
    for h, col in ((1, "fwd_close_1"), (3, "fwd_close_3"), (5, "fwd_close_5")):
        # 進場日=t+1：ret_1d 用 t+1 收、ret_3d 用 t+3 收、ret_5d 用 t+5 收
        # 已用 shift(-h) from t → 對 ret_1d 是 shift(-1)=t+1 close 正確
        # ret_3d: 持有到進場後第3日收 = t+1+2 = t+3 → shift(-3) OK
        ev[f"fwd_{h}d"] = ev[col] / ev["entry"] - 1.0
    ev = ev[np.isfinite(ev["entry"]) & (ev["entry"] > 0)].copy()
    return ev


def _stats(sub: pd.DataFrame, col: str) -> dict:
    x = sub[col].astype(float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "up": 0.0, "dn": 0.0, "mean": 0.0, "med": 0.0}
    return {
        "n": n,
        "up": 100.0 * (x > 0).mean(),
        "dn": 100.0 * (x < 0).mean(),
        "mean": 100.0 * x.mean(),
        "med": 100.0 * x.median(),
    }


def _print_group(name: str, sub: pd.DataFrame) -> None:
    print(f"\n[{name}] 訊號數={len(sub)}", flush=True)
    if sub.empty:
        return
    hdr = f"{'H':>4} {'n':>6} {'上漲%':>8} {'下跌%':>8} {'mean':>9} {'median':>9}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for h in (1, 3, 5):
        st = _stats(sub, f"fwd_{h}d")
        print(
            f"{h:>4} {st['n']:>6} {st['up']:>7.1f}% {st['dn']:>7.1f}% "
            f"{st['mean']:>8.3f}% {st['med']:>8.3f}%",
            flush=True,
        )


def _print_short_group(name: str, sub: pd.DataFrame) -> None:
    """做空報酬 = -fwd；勝=價格下跌。未含券息／平盤下禁空。"""
    print(f"\n[{name}] 訊號數={len(sub)}", flush=True)
    if sub.empty:
        return
    hdr = f"{'H':>4} {'n':>6} {'空勝%':>8} {'空敗%':>8} {'mean':>9} {'median':>9}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for h in (1, 3, 5):
        st = _stats(sub, f"short_{h}d")
        print(
            f"{h:>4} {st['n']:>6} {st['up']:>7.1f}% {st['dn']:>7.1f}% "
            f"{st['mean']:>8.3f}% {st['med']:>8.3f}%",
            flush=True,
        )


def _print_yearly_short(ev: pd.DataFrame, r: pd.Series) -> None:
    """按年印 A/B/C 的 5 日做空摘要。"""
    tmp = ev.copy()
    tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
    masks = {
        "A": (r > 0.30) & (tmp["ret_5d"] > 0.03),
        "B": tmp["ret_5d"] > 0.03,
        "C": r > 0.30,
    }
    print("\n" + "=" * 72)
    print("按年做空摘要（5日；空勝%=short_5d>0）")
    print("=" * 72)
    hdr = (
        f"{'年':>6} "
        f"{'A_n':>6} {'A勝%':>7} {'Amean':>8} "
        f"{'B_n':>7} {'B勝%':>7} {'Bmean':>8} "
        f"{'C_n':>6} {'C勝%':>7} {'Cmean':>8} "
        f"{'A-B':>8}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    years = sorted(tmp["year"].dropna().unique())
    for y in years:
        yd = tmp["year"] == y
        cells = []
        means = {}
        for key in ("A", "B", "C"):
            sub = tmp[yd & masks[key]]
            x = sub["short_5d"].astype(float)
            x = x[np.isfinite(x)]
            n = len(x)
            if n == 0:
                cells.extend([f"{0:>6}", f"{'—':>7}", f"{'—':>8}"])
                means[key] = float("nan")
            else:
                cells.extend(
                    [
                        f"{n:>6}",
                        f"{100*(x>0).mean():>6.1f}%",
                        f"{100*x.mean():>7.3f}%",
                    ]
                )
                means[key] = float(x.mean())
        ab = (
            f"{100*(means['A']-means['B']):>7.3f}%"
            if np.isfinite(means["A"]) and np.isfinite(means["B"])
            else f"{'—':>8}"
        )
        print(f"{int(y):>6} " + " ".join(cells) + f" {ab}", flush=True)


def run(
    start_date: str = "2026-06-01",
    end_date: str = "2026-07-31",
    focus: str = "all",
    filter_0050_open_down: bool = False,
) -> pd.DataFrame:
    t0 = time.time()
    print("券資比 + 近幾日漲幅 → 後續方向", flush=True)
    print(
        "券資比=融券餘額/融資餘額；訊號日t收盤後→進場t+1 open；"
        f"區間 {start_date} ~ {end_date}；focus={focus}",
        flush=True,
    )
    print("個股：4碼數字、排除00xx ETF", flush=True)
    if filter_0050_open_down:
        print("過濾：進場日 0050 開盤 < 前收（開盤跌）", flush=True)
    print(flush=True)

    print("載入 margin...", flush=True)
    margin = _load_margin(start_date, end_date)
    print(
        f"margin rows={len(margin):,} stocks={margin['stock_id'].nunique()} "
        f"dates={margin['date'].nunique()}",
        flush=True,
    )
    if margin.empty:
        return margin

    print("載入 pattern day...", flush=True)
    day = _prep_day(start_date, end_date)
    print(f"day rows={len(day):,}", flush=True)

    ev = _build_events(margin, day)
    print(f"join 後事件={len(ev):,}", flush=True)
    if ev.empty:
        return ev

    if filter_0050_open_down:
        down_dates = _signal_dates_0050_entry_open_down(start_date, end_date)
        before = len(ev)
        ev = ev[ev["date"].isin(down_dates)].copy()
        print(
            f"0050開盤跌過濾: 訊號日={len(down_dates)} 事件 {before:,}→{len(ev):,}",
            flush=True,
        )
        if ev.empty:
            return ev

    r = ev["short_margin_ratio"]
    print(
        f"券資比>30% n={int((r>0.30).sum())}  >10% n={int((r>0.10).sum())}",
        flush=True,
    )

    for h in (1, 3, 5):
        ev[f"short_{h}d"] = -ev[f"fwd_{h}d"]

    if focus == "all":
        groups = {
            "A 主條件 比>30%且5日漲>3%": ev[(r > 0.30) & (ev["ret_5d"] > 0.03)],
            "A' 比>30%且3日漲>3%": ev[(r > 0.30) & (ev["ret_3d"] > 0.03)],
            "B 僅5日漲>3%": ev[ev["ret_5d"] > 0.03],
            "C 僅券資比>30%": ev[r > 0.30],
            "D 放寬 比>10%且5日漲>3%": ev[(r > 0.10) & (ev["ret_5d"] > 0.03)],
        }
        print("\n" + "=" * 64)
        print("各組後續報酬（相對 t+1 進場價）")
        print("=" * 64)
        for name, sub in groups.items():
            _print_group(name, sub)

    short_groups = {
        "A 做空 比>30%且5日漲>3%": ev[(r > 0.30) & (ev["ret_5d"] > 0.03)],
        "A' 做空 比>30%且3日漲>3%": ev[(r > 0.30) & (ev["ret_3d"] > 0.03)],
        "B 做空 僅5日漲>3%": ev[ev["ret_5d"] > 0.03],
        "C 做空 僅券資比>30%": ev[r > 0.30],
        "D 做空 比>10%且5日漲>3%": ev[(r > 0.10) & (ev["ret_5d"] > 0.03)],
    }
    print("\n" + "=" * 64)
    print("做空報酬（t+1 open 進場；未含券息／平盤下禁空）")
    print("=" * 64)
    for name, sub in short_groups.items():
        _print_short_group(name, sub)

    a_s = short_groups["A 做空 比>30%且5日漲>3%"]
    b_s = short_groups["B 做空 僅5日漲>3%"]
    c_s = short_groups["C 做空 僅券資比>30%"]
    if len(a_s) and len(b_s):
        print("\nA vs B（5日做空 mean）:", flush=True)
        print(
            f"  A short mean={100*a_s['short_5d'].mean():.3f}% 空勝%"
            f"={100*(a_s['short_5d']>0).mean():.1f}% n={len(a_s)}",
            flush=True,
        )
        print(
            f"  B short mean={100*b_s['short_5d'].mean():.3f}% 空勝%"
            f"={100*(b_s['short_5d']>0).mean():.1f}% n={len(b_s)}",
            flush=True,
        )
        if len(c_s):
            print(
                f"  C short mean={100*c_s['short_5d'].mean():.3f}% 空勝%"
                f"={100*(c_s['short_5d']>0).mean():.1f}% n={len(c_s)}",
                flush=True,
            )
        else:
            print("  C n=0", flush=True)
        print(
            f"  券資比加成 (A−B mean)={100*(a_s['short_5d'].mean()-b_s['short_5d'].mean()):.3f}%",
            flush=True,
        )

    _print_yearly_short(ev, r)

    if focus == "all":
        rev_long = {
            "A↔ 做多 比>30%且5日跌>3%": ev[(r > 0.30) & (ev["ret_5d"] < -0.03)],
            "B↔ 做多 僅5日跌>3%": ev[ev["ret_5d"] < -0.03],
            "C↔ 做多 僅券資比>30%": ev[r > 0.30],
            "D↔ 做多 比>10%且5日跌>3%": ev[(r > 0.10) & (ev["ret_5d"] < -0.03)],
        }
        print("\n" + "=" * 64)
        print("反向：大跌後做多（ret_5d < -3%）")
        print("=" * 64)
        for name, sub in rev_long.items():
            _print_group(name, sub)

        msr = ev["margin_short_ratio"]
        p90 = float(msr.dropna().quantile(0.90)) if msr.notna().any() else float("nan")
        hi_msr = msr >= p90
        print("\n" + "=" * 64)
        print(f"資券比高 + 大跌 → 做多（資券比≥p90={p90:.1f}）")
        print("=" * 64)
        long_groups = {
            "L1 資券比≥p90且5日跌>3%": ev[hi_msr & (ev["ret_5d"] < -0.03)],
            "L2 僅5日跌>3%": ev[ev["ret_5d"] < -0.03],
            "L3 僅資券比≥p90": ev[hi_msr],
        }
        for name, sub in long_groups.items():
            _print_group(f"{name} 做多", sub)

    print(
        f"\n解讀: 長年 A 空勝>50%且多數年份 mean>0→假說站得住；"
        f"僅空頭年好→行情相依；A不穩贏B→核心可能只是大漲回吐。"
        f" 耗時 {time.time()-t0:.1f}s",
        flush=True,
    )
    return ev


def main():
    p = argparse.ArgumentParser(description="券資比×漲幅→後續方向")
    p.add_argument("--start_date", default="2026-06-01")
    p.add_argument("--end_date", default="2026-07-31")
    p.add_argument(
        "--focus",
        choices=("all", "short"),
        default="all",
        help="short=只做空總表+按年；all=含做多/反向/資券比",
    )
    p.add_argument(
        "--filter_0050_open_down",
        action="store_true",
        help="只保留進場日 0050 開盤 < 前收 的訊號日",
    )
    args = p.parse_args()
    run(
        args.start_date,
        args.end_date,
        focus=args.focus,
        filter_0050_open_down=args.filter_0050_open_down,
    )


if __name__ == "__main__":
    main()
