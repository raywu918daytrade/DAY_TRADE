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

== Main 模式 ==

train        訓練模型（RandomForest），存至 models/mkt_idx_rfc.pkl
importance   顯示特徵重要性（讀已存的模型，不重訓）
evaluate     單獨印混淆矩陣 + 分類報告（讀已存的模型 + 跟訓練時同一套切分
             邏輯重新切出測試集，不重訓）
confidence   漲/跌兩個稀有類別的信心度門檻掃描，看不同機率門檻下的
             precision/recall（讀已存的模型，不重訓）

用法：
    python -m strategy.mkt_idx.train train
    python -m strategy.mkt_idx.train importance
    python -m strategy.mkt_idx.train evaluate
    python -m strategy.mkt_idx.train confidence

⚠️ evaluate/confidence 用的 test_days 要跟當初訓練那個模型用的 test_days
一致（都預設30），不然測試集會跟模型訓練時看過的資料重疊，評估結果不可信
（見 _split_data() 的說明）。
"""

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
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


def _split_data(test_days: int = 30):
    """跟 train() 共用同一套切分邏輯，讓 evaluate()/confidence_report() 能
    用「跟訓練時同一份」測試集去評估已存的模型，不用每次都重新訓練一次
    （2026-07-17 討論：混淆矩陣/信心度報表要能單獨印，不用綁著train()）。

    ⚠️ test_days 要跟當初訓練那個模型用的 test_days 一致，不然測試集會
    跟模型訓練時看過的資料重疊，評估結果不可信。
    """
    df = _prepare_data()
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    return df[df["date"] <= cutoff], df[df["date"] > cutoff]


def train(test_days: int = 20):
    # test_days=20（2026-07-16 從10調大）：0050歷史資料補齊後，訓練資料從
    # 約1.5個月變成6個月以上，但漲/跌是很稀有的類別（各僅佔2~5%），test_days
    # 太小的話測試集裡稀有類別樣本數太少、precision/recall算出來不穩定，
    # 不是「訓練資料變多就要按比例放大測試集」的邏輯。
    train_df, test_df = _split_data(test_days)
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


def _predict_with_threshold(model, test_df, threshold: float | None):
    """threshold=None 用 model.predict() 原本的判法（機率最高的類別勝出，
    沒有門檻概念）；設 0~1 的數字時，改成「信心度夠了才判漲/跌，不然算平」
    的門檻式判法——P(漲)≥threshold 判漲、P(跌)≥threshold 判跌（兩個都過
    門檻取機率較高的那個），否則一律判平。"""
    if threshold is None:
        return model.predict(test_df[FEATURES]), "未設信心度門檻"

    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    p_down = proba[:, class_idx[0]]
    p_up = proba[:, class_idx[2]]
    # 預設判平；P(跌)/P(漲) 過門檻才改判，兩個都過門檻取較高機率那個
    y_pred = np.ones(len(test_df), dtype=int)
    down_pass = p_down >= threshold
    up_pass = p_up >= threshold
    y_pred[down_pass & ~up_pass] = 0
    y_pred[up_pass & ~down_pass] = 2
    both = down_pass & up_pass
    y_pred[both] = np.where(p_up[both] >= p_down[both], 2, 0)
    return y_pred, f"信心度門檻={threshold:.2f}"


_DEFAULT_THRESHOLDS = [None, 0.5, 0.6, 0.7, 0.8]  # None = 完全沒有信心度門檻（model.predict()原本判法）


def evaluate(
    model=None, test_days: int = 30, threshold: float | None | list[float | None] = _DEFAULT_THRESHOLDS
):
    """單獨印混淆矩陣/分類報告，用已存的模型評估，不用重新跑一次 train()
    （2026-07-17 討論：之前混淆矩陣只有 train() 裡才有，每次要看都要重訓
    一次，浪費時間）。

    threshold：預設掃 _DEFAULT_THRESHOLDS 這組門檻，各自印一次混淆矩陣，
    方便一次比較不同門檻下漲/平/跌互相誤判的狀況怎麼變化（2026-07-18
    討論，跟 confidence_report() 的門檻掃描表是互補視角：那支只看單一漲
    或跌類別的precision/recall，這支是選定門檻後完整的3x3矩陣）。list裡的
    None代表完全沒有信心度門檻（model.predict()原本判法，機率最高的類別
    勝出），跟其他數字門檻放在同一組掃描結果裡方便直接比較。想看單一門檻
    就傳一個數字（例如 threshold=0.6）；只想看沒有門檻的版本就傳
    threshold=None。
    """
    if model is None:
        model = load_model()
    _, test_df = _split_data(test_days)

    print(
        f"測試: {len(test_df):,} 筆 ({test_df['date'].min().strftime('%Y-%m-%d')} ~ "
        f"{test_df['date'].max().strftime('%Y-%m-%d')})"
    )

    thresholds = threshold if isinstance(threshold, list) else [threshold]
    results = []
    for thr in thresholds:
        y_pred, thr_label = _predict_with_threshold(model, test_df, thr)
        print(f"\n{'='*60}\n{thr_label}\n{'='*60}")
        print(f"Accuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
        print("\n混淆矩陣（列=實際，欄=預測，順序 跌/平/漲）:")
        print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
        print("\n分類報告:")
        print(classification_report(test_df["target"], y_pred, labels=[0, 1, 2], target_names=["跌", "平", "漲"]))
        results.append(y_pred)

    return test_df, (results[0] if len(results) == 1 else results)


def confidence_report(model=None, test_days: int = 30, thresholds: list[float] | None = None):
    """依信心度（模型算出來的機率）門檻掃描，分別看「漲」「跌」這兩個稀有
    類別在不同信心度下的 precision/recall，比照 strategy/rally/validate.py
    的 confidence_report() 概念，改成3分類版（要分開看漲/跌，不是單一
    proba≥threshold 就好）。

    門檻越高，預測數越少，但理論上precision應該要越高——如果掃出來precision
    沒有隨門檻提升，代表模型的機率輸出沒有校準好、"信心度"不能拿來當篩選依據。
    """
    if model is None:
        model = load_model()
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    _, test_df = _split_data(test_days)
    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    test_df = test_df.copy()
    test_df["proba_down"] = proba[:, class_idx[0]]
    test_df["proba_up"] = proba[:, class_idx[2]]

    for cls, col, name in [(2, "proba_up", "漲"), (0, "proba_down", "跌")]:
        total_actual = int((test_df["target"] == cls).sum())
        print(f"\n── {name}（class={cls}）信心度門檻掃描 ──")
        print(f"  測試集裡實際{name}的樣本數: {total_actual:,}")
        print(f"  {'門檻':>6}  {'預測數':>7}  {'猜中數':>7}  {'precision':>9}  {'recall':>6}")
        for thr in thresholds:
            sub = test_df[test_df[col] >= thr]
            n = len(sub)
            if n == 0:
                print(f"  {thr:.2f}  {0:>7,}  {0:>7,}  {'--':>9}  {'--':>6}")
                continue
            tp = int((sub["target"] == cls).sum())
            precision = tp / n
            recall = tp / total_actual if total_actual else 0
            print(f"  {thr:.2f}  {n:>7,}  {tp:>7,}  {precision*100:>8.2f}%  {recall*100:>5.2f}%")

    return test_df


def feature_importance(model=None, top_n: int = 10):
    """顯示 RandomForest 特徵重要性。目前 FEATURES 只有 ret_vs_idx 一個
    （見檔頭說明：先精簡、慢慢加），這裡會顯示 100%——先把這支函式準備好，
    之後陸續加特徵時直接能用，不用等特徵變多才回頭補。"""
    if model is None:
        model = load_model()

    print(f"\n── 特徵重要性（目前 FEATURES: {FEATURES}）──")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:top_n]:
        print(f"  {name:20s}  {imp:.4f}")


def main(mode: str = "", test_days: int = 30, threshold: float | list[float] | None = None):
    """CLI進入點，比照 strategy/rally/entry.py 的 mode 切換方式（mkt_idx
    目前只有 train.py，還沒有 predict.py/live.py/entry.py，等那些補齊了
    再考慮要不要拆成獨立的 entry.py）。可用模式見檔頭的「== Main 模式 ==」。

    兩種用法都支援，互不衝突（2026-07-18 討論）：
      1. VS Code按F5：__main__ 裡直接寫死 mode 變數，不帶任何CLI參數
         （sys.argv長度=1，只有腳本路徑本身），這裡就直接用傳進來的 mode，
         不會被argparse覆蓋。
      2. 終端機帶參數：python -m strategy.mkt_idx.train evaluate --threshold 0.6
         （sys.argv長度>1），這裡改用argparse解析，CLI帶的參數會覆蓋掉
         __main__ 裡寫死的值。
    """
    # 只有終端機真的帶了CLI參數才用argparse覆蓋；F5直接執行（sys.argv只有
    # 腳本路徑本身，長度=1）就尊重呼叫端傳進來的 mode/test_days/threshold，
    # 不要看mode是不是空字串來判斷（之前這樣寫，F5執行時只要mode沒填就會
    # 誤觸發argparse，去讀根本不存在的CLI參數而出錯或用到不對的預設值）。
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="mkt_idx 策略 — RandomForest")
        parser.add_argument(
            "mode",
            nargs="?",
            default="train",
            choices=["train", "importance", "evaluate", "confidence"],
            help="執行模式（預設train）",
        )
        parser.add_argument("--test_days", type=int, default=30, help="測試集天數")
        parser.add_argument(
            "--threshold",
            type=float,
            nargs="*",
            default=None,
            help="evaluate專用：信心度門檻(0~1)，可帶多個一次比較（例：--threshold 0.5 0.6 0.7 0.8），"
            "留空=用predict()原本的判法（見evaluate()說明）",
        )
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        # nargs="*" 沒帶值時是 None；帶一個值時是長度1的list，跟F5那邊
        # 「單一float」的用法一致傳進 evaluate() 就好，不用特別拆成純數字
        threshold = args.threshold
    elif not mode:
        mode = "train"  # F5執行、__main__也沒寫mode時的保底預設

    if mode == "train":
        train(test_days=test_days)
    elif mode == "importance":
        feature_importance()
    elif mode == "evaluate":
        # threshold 沒特別指定（None）時不傳進去，讓 evaluate() 自己的預設值
        # （_DEFAULT_THRESHOLDS）生效，不要在這裡另外複製一份預設值
        # （2026-07-18 討論：預設值該由 evaluate() 自己負責，main()/CLI只在
        # 使用者真的想覆蓋時才傳）。
        if threshold is not None:
            evaluate(test_days=test_days, threshold=threshold)
        else:
            evaluate(test_days=test_days)
    elif mode == "confidence":
        confidence_report(test_days=test_days)
    else:
        print(f"未知模式: {mode}，可用: train / importance / evaluate / confidence")


if __name__ == "__main__":
    """
    兩種執行方式都支援（2026-07-18 討論，main() 會依 sys.argv 長度自動判斷
    要用哪一種，兩者不會互相打架）：

    1. VS Code按F5（開發時常用）：直接改下面這幾行變數，不用打字：
        mode        "train" / "importance" / "evaluate" / "confidence"
        test_days   測試集天數（要跟訓練那個模型用的 test_days 一致，
                    不然測試集會跟訓練時看過的資料重疊，評估結果不可信）
        threshold   只有 evaluate 會用到，留 None 就好——會直接用
                    evaluate() 自己的預設值（見 _DEFAULT_THRESHOLDS）。
                    真的想覆蓋才在這裡改，例如單一數字 0.6，或自己的
                    list（例如 [0.5, 0.6, 0.7, 0.8, 0.9]）。

    2. 終端機帶參數（會覆蓋掉下面寫死的值）：
        python -m strategy.mkt_idx.train train
        python -m strategy.mkt_idx.train importance
        python -m strategy.mkt_idx.train evaluate
        python -m strategy.mkt_idx.train evaluate --threshold 0.6
        python -m strategy.mkt_idx.train confidence
    """
    mode = "evaluate"  # F5時改這裡：train / importance / evaluate / confidence
    test_days = 30
    threshold = None  # 只有 mode="evaluate" 用得到；留 None = 用 evaluate() 自己的預設值
    main(mode=mode, test_days=test_days, threshold=threshold)
