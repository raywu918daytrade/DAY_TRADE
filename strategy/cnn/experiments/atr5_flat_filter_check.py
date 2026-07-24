"""
實驗：「平」樣本的atr5分布長怎樣？——為了決定要不要、以及用什麼門檻，
只對「平」（target=1）做atr5篩選（跌/漲全部保留，見討論：平占比過高、
class_weights只調loss權重、每個batch裡跌/漲訊號還是太稀疏，想用atr5挑掉
「本來就沒什麼波動、幾乎注定不會動」的平樣本，同時降低平的數量、也提高
留下來的平樣本的資訊量）。

只算9~10點的分布，不是全天——train()現在只訓練9~10點的候選，門檻要用
「訓練時實際會看到的population」去算，用全天分布會被其他時段的波動特性
稀釋、算出來的門檻套到9-10點子集上篩選力道會不準（9-10點通常波動比全天
均值更大）。

只是看分布，不動train.py/dataset.py——先看數字再決定門檻，不要憑空猜。
放在experiments/底下，是一次性的診斷，不是要長期維護的核心流程。

用法：python -m strategy.cnn.experiments.atr5_flat_filter_check
"""

import numpy as np
import pandas as pd

from strategy.cnn.dataset import _M1_COLS, available_months, load_shard_branch, load_shard_meta

_ATR5_CHANNEL = _M1_COLS.index("atr5")
_PERCENTILES = [10, 25, 50, 75, 90, 95, 99]
# 好讀好改：9:10~10:00（不是粗略的hour==9整個小時）——候選實際上要到9:13
# 左右才開始出現（見m5自己5分鐘暖機的說明），9:00~9:10這段本來就沒有候選，
# 但精準卡在9:10~10:00才跟train()未來實際會用的窗口完全對齊。
_WINDOW_START = pd.Timestamp("00:00:00") + pd.Timedelta(hours=9, minutes=00)
_WINDOW_END = pd.Timestamp("00:00:00") + pd.Timedelta(hours=10)


def _in_window(date: pd.Series) -> np.ndarray:
    """對外看得到好讀的時間常數（_WINDOW_START/_WINDOW_END），內部轉成
    分鐘數做純數字向量化比較——逐筆比較datetime.time物件在幾百萬筆資料上
    會明顯變慢。"""
    minutes = date.dt.hour.to_numpy() * 60 + date.dt.minute.to_numpy()
    start = _WINDOW_START.hour * 60 + _WINDOW_START.minute
    end = _WINDOW_END.hour * 60 + _WINDOW_END.minute
    return (minutes >= start) & (minutes < end)


def load_target_and_atr5() -> pd.DataFrame:
    months = available_months()
    parts = []
    for month in months:
        meta = load_shard_meta(month)
        mask = _in_window(meta["date"])
        m1 = load_shard_branch(month, "m1", mmap=True)
        atr5 = np.array(m1[mask, _ATR5_CHANNEL, -1])  # 候選當下這一步的atr5（已除day_open正規化）
        parts.append(pd.DataFrame({"target": meta.loc[mask, "target"].to_numpy(), "atr5": atr5}))
    return pd.concat(parts, ignore_index=True)


def main():
    df = load_target_and_atr5()
    label_names = {0: "跌", 1: "平", 2: "漲"}

    print(f"總樣本數: {len(df):,}")
    for target, name in label_names.items():
        sub = df.loc[df["target"] == target, "atr5"]
        print(f"\n── {name}（{len(sub):,} 筆） ──")
        pct = np.percentile(sub, _PERCENTILES)
        for p, v in zip(_PERCENTILES, pct):
            print(f"  p{p}: {v:.5f}")
        print(f"  atr5==0（暖機期NaN補0）比例: {(sub == 0).mean():.4f}")

    flat = df.loc[df["target"] == 1, "atr5"]
    non_flat_count = (df["target"] != 1).sum()
    print(f"\n── 只篩「平」：不同門檻下平還會剩多少筆（跌/漲固定{non_flat_count:,}筆不變） ──")
    for p in _PERCENTILES:
        threshold = np.percentile(flat, p)
        kept = (flat >= threshold).sum()
        print(
            f"  門檻=p{p}分位數({threshold:.5f}): 平剩 {kept:,} 筆（保留{kept / len(flat):.2%}），"
            f"平:跌+漲 比例 = {kept / non_flat_count:.2f} : 1"
        )


if __name__ == "__main__":
    main()
