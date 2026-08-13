"""
0050 大缺口日 vs tick_universe 截面波動，以及
09:05→10:00 效率標籤（quiet / range / trend）能否用 gap×ret5（+ 廣度／類股同步）分開；
另驗證類股開盤（leave-one-out med gap / ret5）能否判斷同類股內個股方向。

定義：
    0050 gap = (open / prev_close) - 1
    ret5    = (c0905 - o0905) / o0905
    個股窗內：net=(p1000-p905)/p905；rng=(maxH-minL)/p905；eff=|net|/rng
    日標籤：quiet if med_rng<1%；其餘依 med_eff 相對非 quiet 中位數分 range/trend
    類股：info.group；peers≥5 的 leave-one-out med_gap / med_ret5

用法：
    python -m strategy_test.idx_gap_market_vol.verify \\
        --start_date 2024-01-01 --end_date 2026-07-31
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from data.adjustment_query import load_pattern_day, load_pattern_m5_std
from finmind.tick_universe import load_tick_universe
from strategy.mkt.config import IDX_SYMBOL

GAP_THR = 0.01
RET5_THR = 0.005
QUIET_RNG = 0.01
ENTRY_T = dtime(9, 5)
EXIT_T = dtime(10, 0)
SECTOR_MIN_N = 5
TEST_DAYS = 90
FRICTION = 0.0045  # 0.45%，同 open_drive
SIGNS = ("up", "down", "flat")


def _gap_sign(gap: float) -> str:
    if gap > GAP_THR:
        return "up"
    if gap < -GAP_THR:
        return "down"
    return "flat"


def _ret5_sign(r: float) -> str:
    if r > RET5_THR:
        return "up"
    if r < -RET5_THR:
        return "down"
    return "flat"


def _regime(gap: float) -> str:
    return "big" if abs(gap) >= GAP_THR else "flat"


def _mw_pvalue(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)


def _summarize(daily: pd.DataFrame, label: str, baseline: pd.DataFrame | None) -> None:
    n = len(daily)
    if n == 0:
        print(f"  {label}: n=0", flush=True)
        return

    med_r = float(daily["med_range"].mean())
    mean_r = float(daily["mean_range"].mean())
    med_a = float(daily["med_abs_ret"].mean())
    mean_a = float(daily["mean_abs_ret"].mean())
    pct2 = float(daily["pct_range_ge2"].mean())
    pct3 = float(daily["pct_range_ge3"].mean())
    mean_gap = float(daily["idx_gap"].mean())

    line = (
        f"  {label}: n_days={n}  mean_idx_gap={100 * mean_gap:+.3f}%  "
        f"mean(med_range)={100 * med_r:.3f}%  mean(mean_range)={100 * mean_r:.3f}%  "
        f"mean(med_abs_ret)={100 * med_a:.3f}%  mean(mean_abs_ret)={100 * mean_a:.3f}%  "
        f"mean(pct≥2%)={100 * pct2:.1f}%  mean(pct≥3%)={100 * pct3:.1f}%"
    )
    if baseline is not None and len(baseline) > 0:
        base_med_r = float(baseline["med_range"].mean())
        base_med_a = float(baseline["med_abs_ret"].mean())
        ratio_r = med_r / base_med_r if base_med_r > 0 else float("nan")
        ratio_a = med_a / base_med_a if base_med_a > 0 else float("nan")
        p_r = _mw_pvalue(daily["med_range"], baseline["med_range"])
        p_a = _mw_pvalue(daily["med_abs_ret"], baseline["med_abs_ret"])
        line += (
            f"  | vs flat: range×{ratio_r:.2f} (p={p_r:.2e})  "
            f"abs_ret×{ratio_a:.2f} (p={p_a:.2e})"
        )
    print(line, flush=True)


def _load_info_group() -> pd.DataFrame:
    path = _ROOT / "db/info/info.parquet"
    info = pd.read_parquet(path, columns=["stock_id", "group"])
    info["stock_id"] = info["stock_id"].astype(str)
    info = info.dropna(subset=["group"]).drop_duplicates("stock_id", keep="last")
    return info


def _stock_window_metrics(m5: pd.DataFrame) -> pd.DataFrame:
    """每 stock×day：ret5、net/rng/eff（09:05→10:00）。"""
    m5 = m5.copy()
    m5["t"] = m5["date"].dt.time
    m5 = m5[m5["t"].between(ENTRY_T, EXIT_T)].sort_values(["stock_id", "day_str", "date"])
    m5 = m5.drop_duplicates(["stock_id", "day_str", "t"], keep="last")

    entry = m5[m5["t"] == ENTRY_T][["stock_id", "day_str", "open", "close"]].rename(
        columns={"open": "o905", "close": "p905"}
    )
    exit_ = m5[m5["t"] == EXIT_T][["stock_id", "day_str", "close"]].rename(
        columns={"close": "p1000"}
    )
    win = (
        m5.groupby(["stock_id", "day_str"], sort=False)
        .agg(win_high=("high", "max"), win_low=("low", "min"), n_bars=("close", "count"))
        .reset_index()
    )
    out = entry.merge(exit_, on=["stock_id", "day_str"], how="inner")
    out = out.merge(win, on=["stock_id", "day_str"], how="inner")
    out = out[(out["p905"] > 0) & (out["o905"] > 0) & (out["n_bars"] >= 2)].copy()
    out["ret5"] = (out["p905"] - out["o905"]) / out["o905"]
    out["net"] = (out["p1000"] - out["p905"]) / out["p905"]
    out["rng"] = (out["win_high"] - out["win_low"]) / out["p905"]
    out["eff"] = np.where(out["rng"] > 1e-6, out["net"].abs() / out["rng"], np.nan)
    return out.reset_index(drop=True)


def _sector_day_feats(stock_day: pd.DataFrame, idx_ret5: pd.Series) -> pd.DataFrame:
    """09:05 可觀測：類股 align / concentration。"""
    df = stock_day.dropna(subset=["group", "ret5"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["day_str", "sector_align", "sector_conc", "n_sectors"])

    rows = []
    for day_str, g in df.groupby("day_str", sort=False):
        ir = float(idx_ret5.get(day_str, np.nan))
        if not np.isfinite(ir) or ir == 0:
            ref = float(np.sign(g["ret5"].median()))
        else:
            ref = float(np.sign(ir))
        if ref == 0:
            ref = 1.0

        aligns = []
        concs = []
        for _, sg in g.groupby("group", sort=False):
            if len(sg) < SECTOR_MIN_N:
                continue
            mean_r = float(sg["ret5"].mean())
            aligns.append(1.0 if np.sign(mean_r) == ref or mean_r == 0 else 0.0)
            same = float((np.sign(sg["ret5"]) == ref).mean())
            concs.append(abs(same - 0.5) * 2.0)
        if not aligns:
            rows.append(
                {"day_str": day_str, "sector_align": np.nan, "sector_conc": np.nan, "n_sectors": 0}
            )
        else:
            rows.append(
                {
                    "day_str": day_str,
                    "sector_align": float(np.mean(aligns)),
                    "sector_conc": float(np.mean(concs)),
                    "n_sectors": len(aligns),
                }
            )
    return pd.DataFrame(rows)


def _loo_median(s: pd.Series) -> pd.Series:
    """同組 leave-one-out 中位；peers < SECTOR_MIN_N → NA。"""
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    if n <= SECTOR_MIN_N:
        return pd.Series(out, index=s.index)
    for i in range(n):
        peers = np.concatenate([vals[:i], vals[i + 1 :]])
        if np.isfinite(peers).sum() >= SECTOR_MIN_N:
            out[i] = float(np.nanmedian(peers))
    return pd.Series(out, index=s.index)


def _side_line(r: pd.Series, label: str) -> None:
    """r = long return；印 long/short mean、win0；|mean|≥摩擦標 opportunity。"""
    r = r.dropna()
    n = len(r)
    if n == 0:
        print(f"  {label}: n=0", flush=True)
        return
    long_mean = float(r.mean())
    short_mean = float((-r).mean())
    long_win0 = float((r > 0).mean())
    short_win0 = float((r < 0).mean())
    tag = ""
    if abs(long_mean) >= FRICTION and long_mean >= abs(short_mean):
        tag = "  → opportunity long"
    elif abs(short_mean) >= FRICTION and short_mean > abs(long_mean):
        tag = "  → opportunity short"
    print(
        f"  {label}: n={n:,}  "
        f"long mean={100 * long_mean:+.3f}% win0={100 * long_win0:.1f}%  "
        f"short mean={100 * short_mean:+.3f}% win0={100 * short_win0:.1f}%{tag}",
        flush=True,
    )


def _fit_logistic(df: pd.DataFrame, feat_cols: list[str], label: str) -> None:
    """時間切末 TEST_DAYS；印 test_acc 與 majority baseline。"""
    use = df.dropna(subset=feat_cols + ["y_trend"]).sort_values("day_str").reset_index(drop=True)
    if len(use) < TEST_DAYS + 30:
        print(f"  {label}: n={len(use)} 樣本不足", flush=True)
        return

    train = use.iloc[:-TEST_DAYS]
    test = use.iloc[-TEST_DAYS:]
    maj = float(train["y_trend"].mean())
    pred_maj = int(maj >= 0.5)
    maj_test = float((test["y_trend"] == pred_maj).mean())

    X_tr = train[feat_cols].to_numpy(dtype=float)
    y_tr = train["y_trend"].to_numpy(dtype=int)
    X_te = test[feat_cols].to_numpy(dtype=float)
    y_te = test["y_trend"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_tr_s, y_tr)
    pred = clf.predict(X_te_s)
    acc = float(accuracy_score(y_te, pred))
    coefs = {c: float(v) for c, v in zip(feat_cols, clf.coef_.ravel())}
    print(
        f"  {label}: train={len(train)} test={len(test)}  "
        f"test_acc={100 * acc:.1f}%  majority={100 * maj_test:.1f}%  "
        f"(train_trend={100 * maj:.1f}%)  "
        f"coefs={{{', '.join(f'{k}={v:+.2f}' for k, v in coefs.items())}}}",
        flush=True,
    )


def run(start_date: str, end_date: str) -> None:
    t0 = time.time()
    universe = {str(s) for s in load_tick_universe()}
    trade_universe = universe - {IDX_SYMBOL}

    print("idx_gap_market_vol — 大缺口波動 + 09:05→10:00 效率體制", flush=True)
    print(f"母體: {len(trade_universe)} 股（排除 {IDX_SYMBOL}）", flush=True)
    print(f"大缺口: |{IDX_SYMBOL} gap| ≥ {100 * GAP_THR:.0f}%", flush=True)
    print(f"區間 {start_date} ~ {end_date}\n", flush=True)

    hist = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    print(f"載入 pattern day（start={hist})...", flush=True)
    day = load_pattern_day(start_date=hist, end_date=end_date)
    day["stock_id"] = day["stock_id"].astype(str)
    day["date"] = pd.to_datetime(day["date"], format="mixed")
    day = day[day["stock_id"].isin(trade_universe | {IDX_SYMBOL})].copy()
    day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)
    day["day_str"] = day["date"].dt.strftime("%Y-%m-%d")

    g = day.groupby("stock_id", sort=False)
    day["prev_close"] = g["close"].shift(1)
    day = day[day["prev_close"].notna() & (day["prev_close"] > 0)].copy()
    day["gap"] = day["open"].astype(float) / day["prev_close"].astype(float) - 1.0
    day["range_pct"] = (day["high"].astype(float) - day["low"].astype(float)) / day[
        "prev_close"
    ].astype(float)
    day["abs_ret"] = (
        day["close"].astype(float) / day["prev_close"].astype(float) - 1.0
    ).abs()

    idx = day[
        (day["stock_id"] == IDX_SYMBOL)
        & (day["day_str"] >= start_date)
        & (day["day_str"] <= end_date)
    ][["day_str", "gap"]].rename(columns={"gap": "idx_gap"})
    idx["regime"] = idx["idx_gap"].map(_regime)
    idx["gap_sign"] = idx["idx_gap"].map(_gap_sign)
    print(f"{IDX_SYMBOL} 交易日: {len(idx)}", flush=True)
    print(
        f"  big={int((idx['regime'] == 'big').sum())}  "
        f"flat={int((idx['regime'] == 'flat').sum())}  "
        f"up={int((idx['gap_sign'] == 'up').sum())}  "
        f"down={int((idx['gap_sign'] == 'down').sum())}",
        flush=True,
    )

    stocks = day[
        (day["stock_id"].isin(trade_universe))
        & (day["day_str"] >= start_date)
        & (day["day_str"] <= end_date)
        & day["range_pct"].notna()
        & (day["range_pct"] >= 0)
    ].copy()
    print(f"個股日列: {len(stocks):,}", flush=True)

    daily = (
        stocks.groupby("day_str", sort=False)
        .agg(
            n_stocks=("stock_id", "count"),
            med_range=("range_pct", "median"),
            mean_range=("range_pct", "mean"),
            med_abs_ret=("abs_ret", "median"),
            mean_abs_ret=("abs_ret", "mean"),
            pct_range_ge2=("range_pct", lambda s: float((s >= 0.02).mean())),
            pct_range_ge3=("range_pct", lambda s: float((s >= 0.03).mean())),
        )
        .reset_index()
    )
    daily = daily.merge(idx, on="day_str", how="inner")
    print(f"可比較交易日: {len(daily)}  日均股票數≈{daily['n_stocks'].mean():.0f}\n", flush=True)

    flat = daily[daily["regime"] == "flat"]
    big = daily[daily["regime"] == "big"]
    up = daily[daily["gap_sign"] == "up"]
    down = daily[daily["gap_sign"] == "down"]

    print("[regime: big vs flat]", flush=True)
    _summarize(flat, "flat", None)
    _summarize(big, "big ", flat)

    print("\n[gap_sign: up / down vs flat]", flush=True)
    _summarize(flat, "flat", None)
    _summarize(up, "up  ", flat)
    _summarize(down, "down", flat)

    daily["gap_mag"] = daily["idx_gap"].abs().map(
        lambda x: "le1" if x <= 0.01 else ("1to2" if x <= 0.02 else ("2to5" if x <= 0.05 else "gt5"))
    )
    print("\n[gap_mag 強度]", flush=True)
    for mag in ("le1", "1to2", "2to5", "gt5"):
        sub = daily[daily["gap_mag"] == mag]
        _summarize(sub, f"{mag:4s}", flat if mag != "le1" else None)

    print("\n" + "=" * 60, flush=True)
    print("Phase regime — 09:05→10:00 效率（quiet / range / trend）", flush=True)
    print("=" * 60, flush=True)

    print("載入 pattern m5_std（09:05–10:00）...", flush=True)
    m5 = load_pattern_m5_std(start_date=start_date, end_date=end_date)
    m5["stock_id"] = m5["stock_id"].astype(str)
    m5["date"] = pd.to_datetime(m5["date"], format="mixed")
    m5["day_str"] = m5["date"].dt.strftime("%Y-%m-%d")
    m5 = m5[
        m5["stock_id"].isin(trade_universe | {IDX_SYMBOL})
        & (m5["day_str"] >= start_date)
        & (m5["day_str"] <= end_date)
        & (m5["date"].dt.time >= ENTRY_T)
        & (m5["date"].dt.time <= EXIT_T)
    ].copy()
    print(f"m5 bars: {len(m5):,}", flush=True)

    px = _stock_window_metrics(m5)
    print(f"完整 09:05+10:00 事件: {len(px):,}", flush=True)

    info = _load_info_group()
    px = px.merge(info, on="stock_id", how="left")
    n_grp = int(px.loc[px["stock_id"].isin(trade_universe), "group"].notna().sum())
    n_tr = int(px["stock_id"].isin(trade_universe).sum())
    print(f"info.group 覆蓋個股列: {n_grp:,}/{n_tr:,}", flush=True)

    idx_px = px[px["stock_id"] == IDX_SYMBOL][["day_str", "ret5"]].rename(
        columns={"ret5": "idx_ret5"}
    )
    idx_px = idx.merge(idx_px, on="day_str", how="inner")
    idx_px["ret5_sign"] = idx_px["idx_ret5"].map(_ret5_sign)
    idx_ret5_map = idx_px.set_index("day_str")["idx_ret5"]

    stock_px = px[px["stock_id"].isin(trade_universe)].copy()

    daily_eff = (
        stock_px.groupby("day_str", sort=False)
        .agg(
            n_eff=("stock_id", "count"),
            med_eff=("eff", "median"),
            med_rng=("rng", "median"),
            med_abs_net=("net", lambda s: float(s.abs().median())),
            cs_med_abs_ret5=("ret5", lambda s: float(s.abs().median())),
            cs_disp=("ret5", "std"),
        )
        .reset_index()
    )

    def _breadth(g: pd.DataFrame) -> float:
        day_str = g.name
        ir = float(idx_ret5_map.get(day_str, np.nan))
        ref = np.sign(ir) if np.isfinite(ir) and ir != 0 else np.sign(g["ret5"].median())
        if ref == 0:
            return float("nan")
        return float((np.sign(g["ret5"]) == ref).mean())

    br = stock_px.groupby("day_str", sort=False).apply(_breadth, include_groups=False)
    br.name = "breadth"
    daily_eff = daily_eff.merge(br.reset_index(), on="day_str", how="left")

    sec = _sector_day_feats(stock_px, idx_ret5_map)
    daily_eff = daily_eff.merge(sec, on="day_str", how="left")
    daily_eff = daily_eff.merge(
        idx_px[["day_str", "idx_gap", "gap_sign", "idx_ret5", "ret5_sign", "regime"]],
        on="day_str",
        how="inner",
    )

    quiet_mask = daily_eff["med_rng"] < QUIET_RNG
    non_quiet = daily_eff.loc[~quiet_mask, "med_eff"]
    eff_cut = float(non_quiet.median()) if len(non_quiet) else float("nan")
    labels = np.where(
        quiet_mask, "quiet", np.where(daily_eff["med_eff"] >= eff_cut, "trend", "range")
    )
    daily_eff["label"] = labels
    daily_eff["y_trend"] = (daily_eff["label"] == "trend").astype(int)

    print(
        f"\n標籤: quiet if med_rng<{100 * QUIET_RNG:.0f}%; "
        f"其餘 med_eff 中位切={eff_cut:.3f}",
        flush=True,
    )
    for lab in ("quiet", "range", "trend"):
        sub = daily_eff[daily_eff["label"] == lab]
        print(
            f"  {lab:5s}: n={len(sub)}  "
            f"mean(med_eff)={sub['med_eff'].mean():.3f}  "
            f"mean(med_rng)={100 * sub['med_rng'].mean():.3f}%  "
            f"mean(med_abs_net)={100 * sub['med_abs_net'].mean():.3f}%",
            flush=True,
        )

    print("\n[0050 gap_sign × ret5_sign → med_eff / 標籤]", flush=True)
    print(
        f"  {'gap':5s} {'ret5':5s}  {'n':>4s}  "
        f"{'med_eff':>8s} {'med_rng%':>8s} {'abs_net%':>8s}  "
        f"{'quiet%':>7s} {'range%':>7s} {'trend%':>7s}",
        flush=True,
    )
    for gs in SIGNS:
        for rs in SIGNS:
            sub = daily_eff[(daily_eff["gap_sign"] == gs) & (daily_eff["ret5_sign"] == rs)]
            n = len(sub)
            if n == 0:
                print(f"  {gs:5s} {rs:5s}  {0:4d}", flush=True)
                continue
            print(
                f"  {gs:5s} {rs:5s}  {n:4d}  "
                f"{sub['med_eff'].mean():8.3f} "
                f"{100 * sub['med_rng'].mean():8.3f} "
                f"{100 * sub['med_abs_net'].mean():8.3f}  "
                f"{100 * (sub['label'] == 'quiet').mean():6.1f}% "
                f"{100 * (sub['label'] == 'range').mean():6.1f}% "
                f"{100 * (sub['label'] == 'trend').mean():6.1f}%",
                flush=True,
            )

    print("\n[相關：特徵 vs med_eff（全日）]", flush=True)
    for col in (
        "idx_gap",
        "idx_ret5",
        "cs_med_abs_ret5",
        "breadth",
        "cs_disp",
        "sector_align",
        "sector_conc",
        "med_rng",
    ):
        s = daily_eff[[col, "med_eff"]].dropna()
        if len(s) < 30:
            print(f"  {col}: n={len(s)} skip", flush=True)
            continue
        x = s[col].abs() if col in ("idx_gap", "idx_ret5") else s[col]
        r, p = stats.spearmanr(x, s["med_eff"])
        if col in ("idx_gap", "idx_ret5"):
            r2, p2 = stats.spearmanr(s[col], s["med_eff"])
            print(
                f"  |{col}| vs med_eff: ρ={r:+.3f} (p={p:.2e})  "
                f"signed ρ={r2:+.3f} (p={p2:.2e})  n={len(s)}",
                flush=True,
            )
        else:
            print(f"  {col} vs med_eff: ρ={r:+.3f} (p={p:.2e})  n={len(s)}", flush=True)

    print(
        f"\n[logistic trend vs range，時間切末 {TEST_DAYS} 日；丟 quiet]",
        flush=True,
    )
    binary = daily_eff[daily_eff["label"].isin(("trend", "range"))].copy()
    binary["abs_gap"] = binary["idx_gap"].abs()
    binary["abs_ret5"] = binary["idx_ret5"].abs()
    binary["gap_up"] = (binary["gap_sign"] == "up").astype(float)
    binary["gap_down"] = (binary["gap_sign"] == "down").astype(float)
    binary["ret5_up"] = (binary["ret5_sign"] == "up").astype(float)
    binary["ret5_down"] = (binary["ret5_sign"] == "down").astype(float)

    feats_a = ["abs_gap", "abs_ret5", "gap_up", "gap_down", "ret5_up", "ret5_down"]
    feats_b = feats_a + ["cs_med_abs_ret5", "breadth", "cs_disp"]
    feats_c = feats_b + ["sector_align", "sector_conc"]

    print(f"  binary n={len(binary)}  trend%={100 * binary['y_trend'].mean():.1f}%", flush=True)
    _fit_logistic(binary, feats_a, "A 0050 only")
    _fit_logistic(binary, feats_b, "A+B +CS")
    _fit_logistic(binary, feats_c, "A+B+C +sector")

    # ---------- Phase sector：類股開盤 → 個股方向 ----------
    print("\n" + "=" * 60, flush=True)
    print("Phase sector — 類股開盤（LOO med）→ 個股方向", flush=True)
    print("=" * 60, flush=True)

    day_ret = stocks[["stock_id", "day_str", "gap", "open", "close"]].copy()
    day_ret = day_ret[(day_ret["open"] > 0)].copy()
    day_ret["day_oc"] = day_ret["close"].astype(float) / day_ret["open"].astype(float) - 1.0
    day_ret["stock_gap_sign"] = day_ret["gap"].map(_gap_sign)

    sev = stock_px.merge(
        day_ret[["stock_id", "day_str", "gap", "stock_gap_sign", "day_oc"]],
        on=["stock_id", "day_str"],
        how="inner",
    )
    sev = sev.dropna(subset=["group", "gap", "ret5", "net"]).copy()
    sev["r_1000"] = sev["net"]
    sev["stock_ret5_sign"] = sev["ret5"].map(_ret5_sign)

    print(
        f"事件列: {len(sev):,}  groups={sev['group'].nunique()}  "
        f"摩擦 |mean|≥{100 * FRICTION:.2f}%",
        flush=True,
    )
    print("算 leave-one-out 類股 med_gap / med_ret5...", flush=True)
    sev["sector_gap"] = sev.groupby(["day_str", "group"], sort=False)["gap"].transform(
        _loo_median
    )
    sev["sector_ret5"] = sev.groupby(["day_str", "group"], sort=False)["ret5"].transform(
        _loo_median
    )
    sev = sev.dropna(subset=["sector_gap", "sector_ret5"]).copy()
    sev["sector_gap_sign"] = sev["sector_gap"].map(_gap_sign)
    sev["sector_ret5_sign"] = sev["sector_ret5"].map(_ret5_sign)
    print(f"有類股 LOO 特徵: {len(sev):,}", flush=True)

    print("\n[baseline 無類股濾網 @10:00]", flush=True)
    _side_line(sev["r_1000"], "all")

    print("\n[sector_gap_sign → 個股 r_1000]", flush=True)
    for gs in SIGNS:
        _side_line(sev.loc[sev["sector_gap_sign"] == gs, "r_1000"], f"sec_gap={gs}")

    print("\n[sector_gap_sign → 個股 day_oc (close/open-1)]", flush=True)
    for gs in SIGNS:
        _side_line(sev.loc[sev["sector_gap_sign"] == gs, "day_oc"], f"sec_gap={gs}")

    print("\n[sector_gap × sector_ret5 → 個股 r_1000]", flush=True)
    for gs in SIGNS:
        for rs in SIGNS:
            mask = (sev["sector_gap_sign"] == gs) & (sev["sector_ret5_sign"] == rs)
            _side_line(sev.loc[mask, "r_1000"], f"sec[{gs}/{rs}]")

    print(
        "\n[sector_gap × stock_gap → 個股 r_1000（看類股是否多餘）]",
        flush=True,
    )
    for gs in SIGNS:
        for ss in SIGNS:
            mask = (sev["sector_gap_sign"] == gs) & (sev["stock_gap_sign"] == ss)
            _side_line(sev.loc[mask, "r_1000"], f"sec={gs} stock={ss}")

    print(f"\n耗時 {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="0050 大缺口波動 + 09:05→10:00 效率體制")
    p.add_argument("--start_date", default="2024-01-01")
    p.add_argument("--end_date", default="2026-07-31")
    args = p.parse_args()
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
