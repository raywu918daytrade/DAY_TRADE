"""
實驗：驗證「跟大盤(0050)差距越大，之後越有機會收斂」這個假設，值不值得
往下做（見 2026-07-14 討論）。

假設
    當大盤漲、個股沒跟上（ret_vs_idx 很負，落後大盤），推測個股接下來有
    機會補漲、往大盤靠近。這支腳本先驗證最單純的版本：不看 lag/趨勢，
    只看「現在的 ret_vs_idx」跟「接下來 N 分鐘的個股報酬率」有沒有負相關
    （落後越多，之後漲越多）。

    如果這個最單純的版本就有訊號，才值得繼續往「lag1~5 + 收斂訊號」那個
    更複雜的方向做；如果完全沒相關性，方向可能要重新想，不用先把複雜特徵
    都做完才發現沒用。

做法
    把 ret_vs_idx 依大小切成 10 等分（decile），看每一組「接下來 N 分鐘
    個股報酬率」的平均值——如果假設成立，應該看到 ret_vs_idx 越負（落後
    越多）那組，未來報酬率平均越高（負相關、單調遞減的關係）。

用法
    python strategy/mkt_idx/experiments/ret_vs_idx_signal_check.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from data.query import load_m1
from strategy.mkt_idx.config import IDX_SYMBOL
from strategy.mkt_idx.features import add_ret_vs_idx, top_n_by_prev_day_volume


def _load_base(top_n: int | None = None) -> pd.DataFrame:
    """載入分K + 算好 ret_vs_idx，各 forward_minutes 共用同一份，不用重複載入。

    top_n: 留空=不過濾；設數字則只留每天依前一日成交量排序前 top_n 名的股票
    （見 strategy/mkt_idx/features.py::top_n_by_prev_day_volume，2026-07-14
    驗證過流動性過濾能提升訊號密度，這裡進一步驗證過濾後 decile 的優勢
    是否也更明顯）。"""
    print("載入分K...")
    m1 = load_m1()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    print("算 ret_vs_idx...")
    # 0050 自己要留著算完 ret_vs_idx 才能排除——add_ret_vs_idx() 需要 0050
    # 的列才算得出大盤那一分鐘的走勢，先過濾掉會導致 ret_vs_idx 整欄變 NaN。
    df = add_ret_vs_idx(m1)
    df = df[df["stock_id"] != IDX_SYMBOL]  # 排除 0050 自己（ret_vs_idx 恆為 0）

    if top_n:
        print(f"流動性過濾：只留每天前一日量排名前{top_n}名的股票...")
        before = df["stock_id"].nunique()
        df = top_n_by_prev_day_volume(df, n=top_n)
        print(f"  股票數: {before} → {df['stock_id'].nunique()}（依日期各自篩選）")

    return df


def run(df_base: pd.DataFrame, forward_minutes: int = 5):
    df = df_base.copy()

    # 未來 N 分鐘個股報酬率（日內，不跨日；用 shift(-N) 往後看）
    g_day = df.groupby(["stock_id", "day_date"], group_keys=False)
    df["_fwd_close"] = g_day["close"].shift(-forward_minutes)
    df["fwd_ret"] = df["_fwd_close"] / df["close"] - 1

    df = df.dropna(subset=["ret_vs_idx", "fwd_ret"])
    print(f"有效樣本: {len(df):,} 筆")

    corr = df["ret_vs_idx"].corr(df["fwd_ret"])
    print(f"\nret_vs_idx 對 未來{forward_minutes}分鐘報酬率 的相關係數: {corr:.4f}")
    print("（假設成立的話，應該是負值——落後越多，之後漲越多）")

    # ── 十等分 decile 分析 ──────────────────────────────────────────────
    df["decile"] = pd.qcut(df["ret_vs_idx"], 10, labels=False, duplicates="drop")
    df["_up"] = (df["fwd_ret"] > 0).astype(int)
    df["_flat"] = (df["fwd_ret"] == 0).astype(int)
    df["_down"] = (df["fwd_ret"] < 0).astype(int)
    grp = df.groupby("decile").agg(
        樣本數=("fwd_ret", "count"),
        ret_vs_idx平均=("ret_vs_idx", "mean"),
        未來報酬率平均=("fwd_ret", "mean"),
        上漲比例=("_up", "mean"),
        持平比例=("_flat", "mean"),
        下跌比例=("_down", "mean"),
    )
    grp["ret_vs_idx平均"] = (grp["ret_vs_idx平均"] * 100).round(3)
    grp["未來報酬率平均"] = (grp["未來報酬率平均"] * 100).round(4)
    grp["上漲比例"] = (grp["上漲比例"] * 100).round(2)
    grp["持平比例"] = (grp["持平比例"] * 100).round(2)
    grp["下跌比例"] = (grp["下跌比例"] * 100).round(2)
    print(f"\n── ret_vs_idx 十等分，看未來{forward_minutes}分鐘報酬率 ──")
    print("（decile 0 = 落後大盤最多那組，decile 9 = 超前大盤最多那組）")
    print("（上漲/持平/下跌比例 = 接下來 N 分鐘收盤價分別比現在高/相同/低的比例，三者加總=100%）")
    print(grp.to_string())
    print(
        f"\n  decile 0 vs decile 9  上漲比例差: "
        f"{grp['上漲比例'].iloc[0] - grp['上漲比例'].iloc[-1]:+.2f} 個百分點"
        f"　下跌比例差: {grp['下跌比例'].iloc[0] - grp['下跌比例'].iloc[-1]:+.2f} 個百分點"
    )

    return df, grp


if __name__ == "__main__":
    # HOLD_BARS=10 已經定案，這裡只比較 5/10 分鐘（跟持有時間對得上的窗口），
    # 不重跑 15/30（那兩個之前已經驗證過訊號會衰退/反轉，跟這次要驗證的
    # 「流動性過濾後優勢是否更明顯」無關）。
    print(f"\n{'#'*70}\n不過濾（全部股票）\n{'#'*70}")
    base_all = _load_base()
    for _fwd in [5, 10]:
        print(f"\n{'='*70}\nforward_minutes = {_fwd}\n{'='*70}")
        run(base_all, forward_minutes=_fwd)

    print(f"\n{'#'*70}\n前一日量排名前100\n{'#'*70}")
    base_top100 = _load_base(top_n=100)
    for _fwd in [5, 10]:
        print(f"\n{'='*70}\nforward_minutes = {_fwd}\n{'='*70}")
        run(base_top100, forward_minutes=_fwd)
