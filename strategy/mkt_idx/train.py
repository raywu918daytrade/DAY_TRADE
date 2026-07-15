"""
mkt_idx 模型訓練 — RandomForest（先用最精簡的設定：只有 ret_vs_idx 一個特徵）

2026-07-14 討論：先用最簡單的設定跑一次，確認整條 pipeline（特徵→標籤→
流動性過濾→時段過濾→訓練）真的串得起來、看得懂結果，之後再逐步加特徵、
調參數（先精簡、慢慢加，避免一次寫太多debug不動）。

目前設定（都是前面幾支 experiments/ 腳本驗證出來的結論）：
    FEATURES = ["ret_vs_idx"]     單一特徵，個股跟0050累積報酬率差
    流動性過濾：前一日量前100名   訊號密度提升最明顯的門檻
    時段：只留9:00~9:10           訊號集中在開盤頭10分鐘，之後快速衰退
    3分類標籤：漲/平/跌           HOLD_BARS=10、TP_PCT=SL_PCT=3%（見config.py）
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data.query import load_m1
from strategy.mkt_idx.config import IDX_SYMBOL
from strategy.mkt_idx.features import add_ret_vs_idx, make_barrier_labels_3class, top_n_by_prev_day_volume

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "models/mkt_idx_rfc.pkl"

FEATURES = ["ret_vs_idx"]  # 先精簡，只用這一個特徵


def _prepare_data(top_n: int = 100, hour: int = 9, minute_max: int = 10) -> pd.DataFrame:
    print("載入分K...")
    m1 = load_m1()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    print("算 ret_vs_idx...")
    # 0050 自己要留著算完才能排除，理由見 features.py::add_ret_vs_idx() 的說明
    df = add_ret_vs_idx(m1)
    df = df[df["stock_id"] != IDX_SYMBOL]

    print(f"流動性過濾：前一日量前{top_n}名...")
    before = df["stock_id"].nunique()
    df = top_n_by_prev_day_volume(df, n=top_n)
    print(f"  股票數: {before} → {df['stock_id'].nunique()}（依日期各自篩選）")

    print("計算3分類標籤（漲=2/平=1/跌=0）...")
    df["target"] = make_barrier_labels_3class(df)

    print(f"時段過濾：{hour}:00~{hour}:{minute_max:02d}...")
    df = df[(df["date"].dt.hour == hour) & (df["date"].dt.minute < minute_max)].copy()

    df = df.dropna(subset=FEATURES + ["target"])
    df["target"] = df["target"].astype(int)
    print(f"有效樣本: {len(df):,} 筆")
    return df


def train(test_days: int = 10):
    df = _prepare_data()

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff]
    print(
        f"\n訓練: {len(train_df):,} ({train_df['date'].min().strftime('%Y-%m-%d')} ~ "
        f"{train_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"測試: {len(test_df):,} ({test_df['date'].min().strftime('%Y-%m-%d')} ~ "
        f"{test_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(f"\n訓練集標籤分佈:\n{(train_df['target'].value_counts(normalize=True)*100).round(2)}")

    # class_weight="balanced"：標籤極度不平衡（平佔九成以上），不加權的話模型
    # 會偷懶全部猜「平」就能拿到很高的準確率，卻毫無用處——用 balanced 讓
    # 稀有的漲/跌類別在訓練時獲得更高權重，強迫模型認真學怎麼分辨它們。
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    print(f"\nAccuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
    print("\n混淆矩陣（列=實際，欄=預測，順序 跌/平/漲）:")
    print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
    print("\n分類報告:")
    print(classification_report(test_df["target"], y_pred, labels=[0, 1, 2], target_names=["跌", "平", "漲"]))

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}")
    return model


def load_model():
    if not _MODEL_PATH.exists():
        raise FileNotFoundError("找不到模型，請先執行 train()")
    return joblib.load(_MODEL_PATH)


if __name__ == "__main__":
    train()
