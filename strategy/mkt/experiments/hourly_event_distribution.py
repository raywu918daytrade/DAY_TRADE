"""
實驗：3分類 triple barrier 標籤（漲/平/跌）裡，漲/跌這兩個稀有事件在
一天裡是怎麼分佈的（見 2026-07-14 討論）。

背景
    strategy/mkt/features.py::make_barrier_labels_3class() 驗證過，
    HOLD_BARS=10、TP_PCT=SL_PCT=3% 這組參數下，「平」（10分鐘內都沒碰到
    ±3%）佔了九成以上，「漲」「跌」是很稀有的事件（各不到1%）——這跟台股
    盤中「開盤活躍、10:00~13:00盤整」的日內型態吻合。這支腳本進一步驗證：
    這些稀有的漲/跌事件，是不是集中在特定時段（例如開盤），值不值得把
    SESSION_START/SESSION_END 縮小到只涵蓋訊號密集的時段，而不是整天都跑。

    2026-07-14 實測結果：訊號濃度從9點開始一路單調遞減到13點（9點漲跌
    合計2.87%，13點只剩0.29%，將近10倍差距），沒有尾盤第二個高峰。
    13點這一組要留意：越接近13:30收盤，NaN（樣本不足10根K棒）比例越高
    （13點NaN高達56%），可信度打折。

用法
    python strategy/mkt/experiments/hourly_event_distribution.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from data.raw_query import load_m1
from strategy.mkt.config import IDX_SYMBOL
from strategy.mkt.features import make_barrier_labels_3class


def run(hours: list[int] | None = None):
    if hours is None:
        hours = [9, 10, 11, 12, 13]

    print("載入分K...")
    m1 = load_m1()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date
    m1 = m1[m1["stock_id"] != IDX_SYMBOL]

    print("計算3分類 triple barrier label...")
    m1["label"] = make_barrier_labels_3class(m1)
    m1["hour"] = m1["date"].dt.hour

    print("\n整體分佈:")
    print((m1["label"].value_counts(dropna=False, normalize=True) * 100).round(2))

    print(f"\n── 按小時分組 (2=漲 1=平 0=跌) ──")
    rows = []
    for h in hours:
        sub = m1[m1["hour"] == h]
        total = len(sub)
        up = (sub["label"] == 2).sum()
        down = (sub["label"] == 0).sum()
        flat = (sub["label"] == 1).sum()
        nan = sub["label"].isna().sum()
        up_pct, down_pct, flat_pct, nan_pct = (
            up / total * 100,
            down / total * 100,
            flat / total * 100,
            nan / total * 100,
        )
        # 漲+跌+平+NaN 四類加總=100%；訊號密度是另外加總漲+跌兩類，不是第5個
        # 類別，別跟「四類加總=100%」搞混（2026-07-14 討論過這點）
        signal_density = up_pct + down_pct
        rows.append(
            {
                "hour": h,
                "total": total,
                "up_pct": up_pct,
                "down_pct": down_pct,
                "flat_pct": flat_pct,
                "nan_pct": nan_pct,
                "signal_density_up_plus_down": signal_density,
            }
        )
        print(
            f"{h}點: 總數{total:,}  漲{up:,}({up_pct:.2f}%)  "
            f"跌{down:,}({down_pct:.2f}%)  平{flat:,}({flat_pct:.2f}%)  "
            f"NaN{nan:,}({nan_pct:.2f}%)  "
            f"[四類加總={up_pct+down_pct+flat_pct+nan_pct:.2f}%  訊號密度(漲+跌)={signal_density:.2f}%]"
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
