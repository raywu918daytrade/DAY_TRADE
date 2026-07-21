"""
實驗：hourly_event_distribution.py 驗證出訊號集中在9點這一小時，這支腳本
再往下切成10分鐘一組（9:00~9:10、9:10~9:20...9:50~10:00），看9點這一小時
裡面訊號是不是又更集中在剛開盤那幾分鐘（見 2026-07-14 討論）。

用法
    python strategy/mkt/experiments/opening_hour_10min_distribution.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from data.query import load_m1
from strategy.mkt.config import IDX_SYMBOL
from strategy.mkt.features import make_barrier_labels_3class, top_n_by_prev_day_volume


def run(top_n: int | None = None):
    """top_n: 留空=不過濾（全部股票）；設數字則只留每天依前一日成交量排序
    前 top_n 名的股票（見 strategy/mkt/features.py::top_n_by_prev_day_volume）。"""
    print("載入分K...")
    m1 = load_m1()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date
    m1 = m1[m1["stock_id"] != IDX_SYMBOL]

    if top_n:
        print(f"流動性過濾：只留每天前一日量排名前{top_n}名的股票...")
        before = m1["stock_id"].nunique()
        m1 = top_n_by_prev_day_volume(m1, n=top_n)
        print(f"  股票數: {before} → {m1['stock_id'].nunique()}（依日期各自篩選）")

    print("計算3分類 triple barrier label...")
    m1["label"] = make_barrier_labels_3class(m1)

    # 只留9:00~10:00，切成6組10分鐘區間
    sub = m1[(m1["date"].dt.hour == 9)].copy()
    sub["bucket_start"] = 9 * 60 + (sub["date"].dt.minute // 10) * 10
    sub["bucket_label"] = sub["bucket_start"].apply(lambda m: f"{m//60}:{m%60:02d}~{(m+10)//60}:{(m+10)%60:02d}")

    print("\n── 9:00~10:00，10分鐘一組 (2=漲 1=平 0=跌) ──")
    rows = []
    for start in sorted(sub["bucket_start"].unique()):
        grp = sub[sub["bucket_start"] == start]
        label_str = grp["bucket_label"].iloc[0]
        total = len(grp)
        up = (grp["label"] == 2).sum()
        down = (grp["label"] == 0).sum()
        flat = (grp["label"] == 1).sum()
        nan = grp["label"].isna().sum()
        up_pct, down_pct, flat_pct, nan_pct = (
            up / total * 100,
            down / total * 100,
            flat / total * 100,
            nan / total * 100,
        )
        # 漲+跌+平+NaN 四類加總=100%（訊號密度是另外加總漲+跌兩類，不是第5個類別，
        # 跟上面四類分開看，避免跟「四類加總=100%」搞混，2026-07-14 討論過這點）
        signal_density = up_pct + down_pct
        rows.append(
            {
                "區間": label_str,
                "total": total,
                "up_pct": up_pct,
                "down_pct": down_pct,
                "flat_pct": flat_pct,
                "nan_pct": nan_pct,
                "signal_density_up_plus_down": signal_density,
            }
        )
        print(
            f"{label_str}: 總數{total:,}  漲{up:,}({up_pct:.2f}%)  "
            f"跌{down:,}({down_pct:.2f}%)  平{flat:,}({flat_pct:.2f}%)  "
            f"NaN{nan:,}({nan_pct:.2f}%)  "
            f"[四類加總={up_pct+down_pct+flat_pct+nan_pct:.2f}%  訊號密度(漲+跌)={signal_density:.2f}%]"
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("\n========== 不過濾（全部股票）==========")
    run()
    print("\n========== 前一日量排名前500 ==========")
    run(top_n=500)
    print("\n========== 前一日量排名前100 ==========")
    run(top_n=100)
