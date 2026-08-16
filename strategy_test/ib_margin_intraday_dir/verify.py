"""D-1 法人／融資券（相對化）vs 當日 09:05→10:00、09:05→收。小量統計，無模型。

特徵用 D-1（收盤後才公布）；label 用 D 盤中。事件 = ib ∩ margin ∩ m5(09:05+10:00)。

用法：
    python -m strategy_test.ib_margin_intraday_dir.verify \\
        --start_date 2025-01-01 --end_date 2026-08-14
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import time as dtime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from data.query import load_day, load_m5_std
from finmind.tick_universe import load_tick_universe

_ATR_N = 14
_T905 = dtime(9, 5)
_T1000 = dtime(10, 0)
_MARGIN_DIR = _ROOT / "db" / "margin"
_IB_DIR = _ROOT / "db" / "ib"

BUCKETS = (
    ("lt20", lambda p: p < 0.20),
    ("20to50", lambda p: (p >= 0.20) & (p < 0.50)),
    ("50to80", lambda p: (p >= 0.50) & (p < 0.80)),
    ("ge80", lambda p: p >= 0.80),
)

PR_FEATS = (
    "foreign_pr",
    "trust_pr",
    "dealer_pr",
    "margin_dlt_pr",
    "short_dlt_pr",
    "sm_ratio_pr",
)
RET_COLS = ("ret_905_1000", "ret_905_close")


def _load_dated(folder: Path, start: str, end_excl: pd.Timestamp) -> pd.DataFrame:
    paths = sorted(folder.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start)
    parts = []
    for p in paths:
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df = df[(df["date"] >= start_ts) & (df["date"] < end_excl)]
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _pr_series(s: pd.Series, lookback: int, min_hist: int) -> pd.Series:
    def _at_end(x: np.ndarray) -> float:
        hist = x[:-1]
        today = x[-1]
        if len(hist) < min_hist or not np.isfinite(today):
            return np.nan
        return float(np.mean(hist < today))

    return s.rolling(lookback + 1, min_periods=min_hist + 1).apply(_at_end, raw=True)


def _attach_prs(df: pd.DataFrame, lookback: int, min_hist: int) -> pd.DataFrame:
    df = df.sort_values(["stock_id", "day"]).copy()
    parts = []
    for _, g in df.groupby("stock_id", sort=False):
        g = g.copy()
        g["foreign_pr"] = _pr_series(g["foreign_turn"], lookback, min_hist)
        g["trust_pr"] = _pr_series(g["trust_turn"], lookback, min_hist)
        g["dealer_pr"] = _pr_series(g["dealer_turn"], lookback, min_hist)
        g["margin_dlt_pr"] = _pr_series(g["margin_dlt_pct"], lookback, min_hist)
        g["short_dlt_pr"] = _pr_series(g["short_dlt_pct"], lookback, min_hist)
        g["sm_ratio_pr"] = _pr_series(g["sm_ratio"], lookback, min_hist)
        g["vol5_pr"] = _pr_series(g["vol5"], lookback, min_hist)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df


def _mw(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(a[m], b[m])
    return float(r), float(p)


def _print_2(label: str, df: pd.DataFrame, mask, min_n: int) -> None:
    print(f"  [{label}]", flush=True)
    _print_ret("→10:00", df.loc[mask, "ret_905_1000"], min_n)
    _print_ret("→收", df.loc[mask, "ret_905_close"], min_n)


def _print_mix(
    label: str,
    df: pd.DataFrame,
    mask,
    min_n: int,
    dead: float = 0.003,
    cols: tuple[tuple[str, str], ...] | None = None,
) -> None:
    """漲/平/跌：平 = |ret| < dead（預設 0.3%，濾掉不漲不跌）。"""
    print(f"  [{label}]", flush=True)
    pairs = cols or (("→10:00", "ret_905_1000"), ("→收", "ret_905_close"))
    for name, col in pairs:
        x = df.loc[mask, col].dropna()
        n = len(x)
        if n < min_n:
            print(f"  {name}: n={n:,}  (< min_n={min_n}，不足)", flush=True)
            continue
        up = x > dead
        dn = x < -dead
        flat = ~(up | dn)
        print(
            f"  {name}: n={n:,}  "
            f"漲%={100 * up.mean():.1f}  平%={100 * flat.mean():.1f}  "
            f"跌%={100 * dn.mean():.1f}  "
            f"mean={100 * x.mean():.3f}%  "
            f"median={100 * x.median():.3f}%  "
            f"(有動 n={int((~flat).sum()):,} mean={100 * x[~flat].mean():.3f}%)",
            flush=True,
        )


def _print_ret(label: str, x: pd.Series, min_n: int) -> None:
    x = x.dropna()
    n = len(x)
    if n < min_n:
        print(f"  {label}: n={n:,}  (< min_n={min_n}，不足)", flush=True)
        return
    print(
        f"  {label}: n={n:,}  "
        f"上漲%={100 * (x > 0).mean():.1f}  "
        f"mean={100 * x.mean():.3f}%  "
        f"median={100 * x.median():.3f}%",
        flush=True,
    )


def _prep_ib(start: str, end_excl: pd.Timestamp) -> pd.DataFrame:
    raw = _load_dated(_IB_DIR, start, end_excl)
    if raw.empty:
        return raw
    raw["net"] = raw["buy"].astype(float) - raw["sell"].astype(float)
    raw["name"] = raw["name"].astype(str)
    raw["stock_id"] = raw["stock_id"].astype(str)
    raw = raw.drop_duplicates(["stock_id", "date", "name"], keep="last")
    wide = raw.pivot_table(
        index=["stock_id", "date"],
        columns="name",
        values="net",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    wide.columns.name = None
    for c in ("Foreign_Investor", "Investment_Trust", "Dealer_self", "Dealer_Hedging", "Foreign_Dealer_Self"):
        if c not in wide.columns:
            wide[c] = 0.0
    wide["foreign_net"] = wide["Foreign_Investor"]
    wide["trust_net"] = wide["Investment_Trust"]
    wide["dealer_net"] = (
        wide["Dealer_self"] + wide["Dealer_Hedging"] + wide["Foreign_Dealer_Self"]
    )
    wide["day"] = wide["date"].dt.strftime("%Y-%m-%d")
    return wide[["stock_id", "day", "foreign_net", "trust_net", "dealer_net"]]


def _prep_margin(start: str, end_excl: pd.Timestamp) -> pd.DataFrame:
    raw = _load_dated(_MARGIN_DIR, start, end_excl)
    if raw.empty:
        return raw
    raw["stock_id"] = raw["stock_id"].astype(str)
    raw = raw.drop_duplicates(["stock_id", "date"], keep="last")
    m = raw["margin_purchase_today_balance"].astype(float)
    my = raw["margin_purchase_yesterday_balance"].astype(float)
    s = raw["short_sale_today_balance"].astype(float)
    sy = raw["short_sale_yesterday_balance"].astype(float)
    raw["margin_dlt_pct"] = (m - my) / my.replace(0, np.nan)
    raw["short_dlt_pct"] = (s - sy) / sy.replace(0, np.nan)
    raw["sm_ratio"] = np.where(m > 0, s / m, np.nan)
    raw["day"] = raw["date"].dt.strftime("%Y-%m-%d")
    return raw[["stock_id", "day", "margin_dlt_pct", "short_dlt_pct", "sm_ratio"]]


def _prep_labels(start: str, end_excl: pd.Timestamp) -> pd.DataFrame:
    m5 = load_m5_std(start_date=start)
    if m5 is None or m5.empty:
        return pd.DataFrame()
    m5 = m5.copy()
    m5["stock_id"] = m5["stock_id"].astype(str)
    m5["date"] = pd.to_datetime(m5["date"], format="mixed")
    m5 = m5[m5["date"] < end_excl]
    m5["tod"] = m5["date"].dt.time
    a = m5[m5["tod"] == _T905].drop_duplicates(["stock_id", "date"], keep="last")
    b = m5[m5["tod"] == _T1000].drop_duplicates(["stock_id", "date"], keep="last")
    a["day"] = a["date"].dt.strftime("%Y-%m-%d")
    b["day"] = b["date"].dt.strftime("%Y-%m-%d")
    a = a.rename(
        columns={
            "close": "c905",
            "open": "o905",
            "high": "h905",
            "low": "l905",
            "volume": "vol5",
        }
    )
    b = b.rename(columns={"close": "c1000"})
    lab = a[["stock_id", "day", "c905", "o905", "h905", "l905", "vol5"]].merge(
        b[["stock_id", "day", "c1000"]],
        on=["stock_id", "day"],
        how="inner",
    )
    lab = lab[lab["c905"] > 0]
    lab["ret_905_1000"] = lab["c1000"] / lab["c905"] - 1.0
    lab["open5_rng"] = (lab["h905"] - lab["l905"]) / lab["o905"].replace(0, np.nan)
    return lab


def _permute_col(df: pd.DataFrame, col: str, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby("stock_id", sort=False).groups.items():
        vals = df.loc[idx, col].to_numpy()
        out.loc[idx] = rng.permutation(vals)
    return out


def _feat_table(ok: pd.DataFrame, feat: str, ret: str, min_n: int) -> None:
    print(f"\n-- {feat} vs {ret} --", flush=True)
    x = ok[feat]
    y = ok[ret]
    _print_ret("all", y, min_n=1)
    for name, pred in BUCKETS:
        _print_ret(name, y[pred(x)], min_n)
    hi = y[x >= 0.80]
    lo = y[x < 0.20]
    rest = y[x < 0.80]
    if len(hi) >= min_n and len(lo) >= min_n:
        p = _mw(hi, lo)
        print(
            f"  ge80 vs lt20: n={len(hi):,}/{len(lo):,}  "
            f"mean {100 * hi.mean():.3f}% / {100 * lo.mean():.3f}%  "
            f"Δ={100 * (hi.mean() - lo.mean()):.3f}%  MW_p={p:.2e}",
            flush=True,
        )
    elif len(hi) < min_n or len(lo) < min_n:
        print(
            f"  ge80 vs lt20: n={len(hi):,}/{len(lo):,}  (< min_n={min_n}，不足)",
            flush=True,
        )
    if len(hi) >= min_n and len(rest) >= min_n:
        p = _mw(hi, rest)
        print(
            f"  ge80 vs rest: n={len(hi):,}/{len(rest):,}  "
            f"mean {100 * hi.mean():.3f}% / {100 * rest.mean():.3f}%  "
            f"MW_p={p:.2e}",
            flush=True,
        )
    r, p = _spearman(x, y)
    n = (x.notna() & y.notna()).sum()
    print(f"  Spearman ρ={r:.3f}  p={p:.2e}  n={n:,}", flush=True)


def build_overnight_panel(
    start_date: str,
    end_date: str,
    lookback: int = 20,
    min_hist: int = 10,
    tick_only: bool = False,
) -> pd.DataFrame:
    """ib∩margin∩m5 stock-day，特徵已對到 D-1，含 PR／ATR。"""
    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=max(lookback * 3, 45))).strftime(
        "%Y-%m-%d"
    )
    end_excl = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    print("載入 ib / margin / day / m5…", flush=True)
    ib = _prep_ib(hist, end_excl)
    margin = _prep_margin(hist, end_excl)
    print(
        f"  ib stock-days={len(ib):,} stocks={ib['stock_id'].nunique() if not ib.empty else 0}",
        flush=True,
    )
    print(
        f"  margin stock-days={len(margin):,} stocks={margin['stock_id'].nunique() if not margin.empty else 0}",
        flush=True,
    )
    if ib.empty or margin.empty:
        print("缺 db/ib 或 db/margin", flush=True)
        return pd.DataFrame()

    both = ib.merge(margin, on=["stock_id", "day"], how="inner")
    print(
        f"  ib∩margin stock-days={len(both):,} stocks={both['stock_id'].nunique()}",
        flush=True,
    )

    day = load_day(start_date=hist)
    day = day.copy()
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[(day["date"] < end_excl) & (day["open"] > 0)]
    day["day"] = day["date"].dt.strftime("%Y-%m-%d")
    g = day.sort_values(["stock_id", "date"]).groupby("stock_id", sort=False)
    day["close_lag1"] = g["close"].shift(1)
    day["close_lag5"] = g["close"].shift(5)
    day["ret_1d"] = day["close"] / day["close_lag1"] - 1.0
    day["ret_5d"] = day["close"] / day["close_lag5"] - 1.0
    prev = day["close_lag1"]
    tr = np.maximum(
        np.maximum((day["high"] - day["low"]).abs(), (day["high"] - prev).abs()),
        (day["low"] - prev).abs(),
    )
    day["_atr14"] = tr.groupby(day["stock_id"]).transform(
        lambda s: s.rolling(_ATR_N, min_periods=_ATR_N).mean()
    )
    day["day_atr"] = day.groupby("stock_id")["_atr14"].shift(1) / day["open"].replace(
        0, np.nan
    )
    vol = day[
        ["stock_id", "day", "volume", "close", "ret_1d", "ret_5d", "day_atr"]
    ].rename(
        columns={"volume": "day_vol", "close": "day_close", "day_atr": "feat_atr"}
    )

    both = both.merge(vol, on=["stock_id", "day"], how="inner")
    dv = both["day_vol"].replace(0, np.nan)
    both["foreign_turn"] = both["foreign_net"] / dv
    both["trust_turn"] = both["trust_net"] / dv
    both["dealer_turn"] = both["dealer_net"] / dv
    print(f"  +day vol 後 {len(both):,}", flush=True)

    lab = _prep_labels(hist, end_excl)
    print(
        f"  m5 09:05∩10:00 {len(lab):,} stocks={lab['stock_id'].nunique()}",
        flush=True,
    )
    if lab.empty:
        print("無 m5 label", flush=True)
        return lab
    lab = lab.merge(
        day[["stock_id", "day", "close", "day_atr"]].rename(columns={"close": "dclose"}),
        on=["stock_id", "day"],
        how="inner",
    )
    lab["ret_905_close"] = lab["dclose"] / lab["c905"] - 1.0

    cal = sorted(day["day"].unique())
    prev_map = {cal[i]: cal[i - 1] for i in range(1, len(cal))}
    lab["feat_day"] = lab["day"].map(prev_map)
    ev = lab.merge(
        both,
        left_on=["stock_id", "feat_day"],
        right_on=["stock_id", "day"],
        how="inner",
        suffixes=("", "_feat"),
    )
    if "day_feat" in ev.columns:
        ev = ev.drop(columns=["day_feat"])
    print(f"  事件 ib∩margin∩m5（特徵=D-1）{len(ev):,}", flush=True)

    if tick_only:
        tick = set(str(s) for s in load_tick_universe())
        before = len(ev)
        ev = ev[ev["stock_id"].isin(tick)].copy()
        print(f"  ∩ tick_universe {before:,}→{len(ev):,}", flush=True)

    if ev.empty:
        print("交集為空", flush=True)
        return ev

    print("算 20 日 PR…", flush=True)
    # vol5 lives on event day D; attach_prs groups by stock_id+day (event day).
    # Feature turns are already D-1 values sitting on event rows; PR along event
    # calendar ≈ trading calendar. That's D-1 series sampled on D, monotonic.
    ev = ev.sort_values(["stock_id", "day"]).copy()
    ev = _attach_prs(ev, lookback, min_hist)
    ev = ev[(ev["day"] >= start_date) & (ev["day"] <= end_date)].copy()
    n_ok = ev["foreign_pr"].notna().sum()
    print(f"  統計窗列={len(ev):,}  有 foreign_pr={n_ok:,}", flush=True)

    idx = day[day["stock_id"] == "0050"][["day", "open", "close"]].sort_values("day")
    idx["close_prev"] = idx["close"].shift(1)
    idx["idx_open_up"] = idx["open"] > idx["close_prev"]
    ev = ev.merge(idx[["day", "idx_open_up"]], on="day", how="left")
    return ev


def print_overnight_stats(ev: pd.DataFrame, min_n: int) -> None:
    ok = ev.dropna(subset=["foreign_pr", "ret_905_1000"])
    print("\n" + "=" * 64, flush=True)
    print("單因子（D-1 PR → D 路徑）", flush=True)
    print("=" * 64, flush=True)
    for feat in PR_FEATS:
        sub = ok.dropna(subset=[feat])
        if sub.empty:
            continue
        for ret in RET_COLS:
            _feat_table(sub, feat, ret, min_n)

    print("\n" + "=" * 64, flush=True)
    print("permutation 對照：foreign_pr 同股打亂日期", flush=True)
    print("=" * 64, flush=True)
    shuf = ok.copy()
    shuf["foreign_pr"] = _permute_col(shuf, "foreign_pr")
    _feat_table(shuf.dropna(subset=["foreign_pr"]), "foreign_pr", "ret_905_1000", min_n)

    print("\n" + "=" * 64, flush=True)
    print("交叉", flush=True)
    print("=" * 64, flush=True)
    hi_f = ok["foreign_pr"] >= 0.80
    lo_f = ok["foreign_pr"] < 0.20
    for flag, name in ((True, "0050今開>昨收"), (False, "0050今開≤昨收")):
        m = ok["idx_open_up"] == flag
        print(f"\n[{name}]", flush=True)
        _print_ret("foreign ge80", ok.loc[m & hi_f, "ret_905_1000"], min_n)
        _print_ret("foreign lt20", ok.loc[m & lo_f, "ret_905_1000"], min_n)

    vol_hi = ok["vol5_pr"] >= 0.80
    print("\n[外資 PR × 今 09:05 量 PR≥80]", flush=True)
    _print_ret("foreign ge80 ∩ vol5 ge80", ok.loc[hi_f & vol_hi, "ret_905_1000"], min_n)
    _print_ret("foreign lt20 ∩ vol5 ge80", ok.loc[lo_f & vol_hi, "ret_905_1000"], min_n)

    same_up = (ok["foreign_net"] > 0) & (ok["trust_net"] > 0)
    same_dn = (ok["foreign_net"] < 0) & (ok["trust_net"] < 0)
    print("\n[外資+投信同向（淨額，D-1）]", flush=True)
    _print_ret("兩者買超", ok.loc[same_up, "ret_905_1000"], min_n)
    _print_ret("兩者賣超", ok.loc[same_dn, "ret_905_1000"], min_n)

    crowd = (ok["margin_dlt_pr"] >= 0.80) & (ok["ret_5d"] > 0.03)
    print("\n[融資增 PR≥80 且 D-1 之 5日漲>3%]", flush=True)
    _print_ret("→10:00", ok.loc[crowd, "ret_905_1000"], min_n)
    _print_ret("→收", ok.loc[crowd, "ret_905_close"], min_n)

    short_up = ok["short_dlt_pct"] > 0
    fa = ok["feat_atr"]
    k = 0.5
    px_up = ok["ret_1d"] >= k * fa
    px_dn = ok["ret_1d"] <= -k * fa
    px_flat = fa.notna() & ~px_up & ~px_dn
    n_sign = int((short_up & (ok["ret_1d"] > 0)).sum())
    n_real = int((short_up & px_up).sum())
    n_flat_s = int((short_up & px_flat).sum())
    print("\n[融券漲 ∩ 股真漲（D-1，相對 ATR）]", flush=True)
    print(
        "  股真漲=ret_1d ≥ 0.5×feat_atr；真跌≤-0.5×；其餘=平。"
        " feat_atr=D-1 的 ATR14(到 D-2)/D-1 開。",
        flush=True,
    )
    print(
        f"  融券增且 close>0 的列 {n_sign:,} → 真漲 {n_real:,}；"
        f" 融券增且平 {n_flat_s:,}",
        flush=True,
    )
    _print_mix("融券增 ∩ 真漲", ok, short_up & px_up, min_n)
    _print_mix("融券增 ∩ 真跌", ok, short_up & px_dn, min_n)
    _print_mix("融券增 ∩ 平", ok, short_up & px_flat, min_n)
    _print_mix("僅真漲", ok, px_up, min_n)
    _print_mix("僅真跌", ok, px_dn, min_n)

    hi_s = ok["short_dlt_pr"] >= 0.80
    print("\n[融券增 PR≥80 ∩ 真漲]", flush=True)
    _print_mix("PR≥80 ∩ 真漲", ok, hi_s & px_up, min_n)
    _print_mix("PR≥80 ∩ 真跌", ok, hi_s & px_dn, min_n)
    _print_mix("PR≥80 ∩ ret_1d>3%", ok, hi_s & (ok["ret_1d"] > 0.03), min_n)

    print("\n[融券增 ∩ 真漲 × 今ATR%（ATR14到昨天 / 今開）]", flush=True)
    both_up = short_up & px_up
    for lo, hi, name in (
        (0.0, 0.03, "今ATR<3%"),
        (0.03, 0.05, "今ATR 3–5%"),
        (0.05, None, "今ATR≥5%"),
    ):
        m = both_up & (ok["day_atr"] >= lo)
        if hi is not None:
            m = m & (ok["day_atr"] < hi)
        _print_mix(name, ok, m, min_n)
    print("  對照：僅真漲 ∩ 今ATR≥3%", flush=True)
    _print_mix("僅真漲 ∩ 今ATR≥3%", ok, px_up & (ok["day_atr"] >= 0.03), min_n)
    _print_mix("融券增∩真漲 ∩ 今ATR≥3%", ok, both_up & (ok["day_atr"] >= 0.03), min_n)

    px_up1 = ok["ret_1d"] >= 1.0 * fa
    print("\n[更嚴：D-1 漲幅 ≥ 1×feat_atr]", flush=True)
    _print_mix("融券增 ∩ 漲≥1ATR", ok, short_up & px_up1, min_n)
    _print_mix("僅漲≥1ATR", ok, px_up1, min_n)


def run(
    start_date: str,
    end_date: str,
    lookback: int = 20,
    min_hist: int = 10,
    min_n: int = 200,
    tick_only: bool = False,
) -> pd.DataFrame:
    t0 = time.time()
    print("ib/margin D-1 → 當日 09:05→10:00 / 09:05→收", flush=True)
    print(f"窗 {start_date}～{end_date}  lookback={lookback}  min_n={min_n}", flush=True)
    print("特徵=D-1；label=D。母體=ib∩margin∩m5。無模型。未用 VWAP+壓力支撐。", flush=True)
    if tick_only:
        print("再 ∩ tick_universe", flush=True)
    print(flush=True)
    ev = build_overnight_panel(
        start_date, end_date, lookback=lookback, min_hist=min_hist, tick_only=tick_only
    )
    if ev.empty:
        return ev
    print_overnight_stats(ev, min_n)
    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)
    return ev


def main() -> None:
    p = argparse.ArgumentParser(description="D-1 法人/融資券 vs 當日 09:05 盤中路徑")
    p.add_argument("--start_date", default="2025-01-01")
    p.add_argument("--end_date", default="2026-08-14")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--min_hist", type=int, default=10)
    p.add_argument("--min_n", type=int, default=200)
    p.add_argument("--tick_only", action="store_true", help="再 ∩ tick_universe")
    args = p.parse_args()
    run(
        args.start_date,
        args.end_date,
        lookback=args.lookback,
        min_hist=args.min_hist,
        min_n=args.min_n,
        tick_only=args.tick_only,
    )


if __name__ == "__main__":
    main()
