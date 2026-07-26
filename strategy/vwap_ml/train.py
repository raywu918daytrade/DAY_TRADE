"""
vwap_ml 模型訓練 — LightGBM 三分類（0=回歸VWAP/1=無訊號/2=延續突破）

2026-07-26 討論：v1 只實際做了 LightGBM 一種演算法，但比照
strategy/mkt/train.py 先把 MODEL_TYPE 切換機制建起來（_LOAD_MODEL_BY_TYPE/
_TRAIN_BY_TYPE 字典＋load_model_by_type()），之後要加 rfc/xgb 只要補一組
train_xxx()/load_model_xxx() 函式並登記進字典，不用改 predict.py、
run_backtest.py、up/down/live.py 任何一行。

== Main 模式 ==

train        訓練模型（--model_type 選演算法，目前只有 lgbm），存至
             models/vwap_ml_{model_type}.pkl
importance   顯示特徵重要性（讀已存的模型，不重訓）
evaluate     單獨印混淆矩陣 + 分類報告（讀已存的模型，不重訓）
confidence   回歸/延續兩個類別的信心度門檻掃描，看不同機率門檻下的
             precision/recall（讀已存的模型，不重訓）

用法：
    python -m strategy.vwap_ml.train train
    python -m strategy.vwap_ml.train train --model_type lgbm
    python -m strategy.vwap_ml.train importance
    python -m strategy.vwap_ml.train evaluate
    python -m strategy.vwap_ml.train confidence

⚠️ evaluate/confidence 用的 test_days 要跟當初訓練那個模型用的 test_days
一致，不然測試集會跟訓練時看過的資料重疊，評估結果不可信（見
_split_data() 的說明）。
"""

import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data.query import load_m1, load_m3, load_m5
from strategy.vwap_ml.config import ATR5_FILTER_THRESHOLD, MODEL_TYPE, STD_MULT
from strategy.vwap_ml.features import FEATURES, make_features

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH_LGBM = _ROOT / "models/vwap_ml_lgbm.pkl"
_CACHE_PATH = _ROOT / "cache/vwap_ml_prepared.parquet"
_SOURCE_DIRS = [_ROOT / "db/m1", _ROOT / "db/m3", _ROOT / "db/m5"]

_TARGET_NAMES = ["回歸", "無訊號", "延續"]


def _source_mtime() -> float:
    """db/m1 裡所有檔案中最新的修改時間戳，比照 strategy/mkt/train.py 的
    _source_mtime() 寫法。"""
    mtimes = [f.stat().st_mtime for d in _SOURCE_DIRS if d.exists() for f in d.iterdir() if f.suffix == ".parquet"]
    return max(mtimes) if mtimes else 0


def _cache_is_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    return _CACHE_PATH.stat().st_mtime >= _source_mtime()


def _prepare_data(
    use_cache: bool = False,
    std_mult: float = STD_MULT,
    atr5_threshold: float = ATR5_FILTER_THRESHOLD,
) -> pd.DataFrame:
    """
    use_cache（2026-07-26 改：跟 strategy/mkt/train.py 的慣例相反，故意
    改成「預設不信任 cache」）：
        False（預設）= 不管 cache 存不存在、新不新，一律重新計算——
                cache 的新鮮度比對（_cache_is_fresh()）只看 db/m1/db/m3/
                db/m5 這些資料檔案的時間戳，偵測不到「features.py 裡的
                計算邏輯改了」這種情況（這個階段密集調整 label 定義，已經
                因為這樣誤用過舊cache兩次），預設一律重算才不會沒發現
                自己在用舊邏輯算出來的結果。
        True  = 才做新鮮度比對——cache比來源資料新就直接讀，否則照樣重算。
                只有明確知道「這段期間沒有改過任何計算邏輯，只是想連續
                看幾次 evaluate/confidence 的結果」時才手動設 True，享受
                省下重算時間的好處。

    std_mult/atr5_threshold（2026-07-26新增，比照
    strategy/mkt/train.py::_prepare_data() 的 atr5_threshold 參數化做法）：
    只給實驗用（例如跑 walk-forward 比較 std_mult=1.0/1.5/2.0），預設值就是
    config.py 的正式設定。⚠️ 只要任一個傳的值跟正式設定不同，就完全跳過
    cache讀寫（不讀舊cache、也不寫檔案）——cache只認預設設定這一組，避免
    非預設參數的實驗結果弄髒正式pipeline在用的cache檔案。這代表用非預設
    參數呼叫這支函式每次都會重新跑一次完整流程，比較慢，但只有做實驗時
    才會這樣用，不影響正式train/predict。
    """
    skip_cache = std_mult != STD_MULT or atr5_threshold != ATR5_FILTER_THRESHOLD
    if not skip_cache and use_cache and _cache_is_fresh():
        print("cache比來源資料新，直接讀取cache...")
        return pd.read_parquet(_CACHE_PATH)

    print("載入分K...")
    m1 = load_m1()
    # 讀現成的 m3/m5 批次快取（data/build_m3_m5_rolling.py 預先算好），不要
    # 讓 make_features() 內部用 compute_m3()/compute_m5() 對全市場全歷史
    # 重新現算一次——那樣等於又製造出兩份跟 m1 一樣大的 dataframe，是
    # 2026-07-26 訓練全歷史資料時記憶體爆掉的主因之一，見
    # features.py::add_vwap_features() 的說明。
    m3 = load_m3()
    m5 = load_m5()

    # stock_id 原本是 object（Python字串）dtype，全歷史規模下記憶體被嚴重
    # 放大（2026-07-26實測：db/m1 全歷史1.56億列只有3090種不重複代號，
    # deep memory profile顯示這一欄單獨就佔了m1總記憶體12.4GB裡的8.3GB，
    # 是造成訓練時記憶體衝到40GB的主因——m1/m3/m5三份dataframe的stock_id
    # 欄位加起來將近25GB）。轉成category（整數編碼+一份代號對照表）大幅
    # 降低記憶體。三份dataframe要共用同一組categories（各自先
    # drop_duplicates()取出代號集合再取聯集，不要直接對上億列呼叫
    # unique()，那一步本身也很花記憶體/時間），這樣後面
    # merge(on=["stock_id","date"])才不會因為categories不一致，把
    # category悄悄轉回object，記憶體效果就白做了。
    stock_ids = pd.unique(
        pd.concat(
            [m1["stock_id"].drop_duplicates(), m3["stock_id"].drop_duplicates(), m5["stock_id"].drop_duplicates()],
            ignore_index=True,
        )
    )
    stock_dtype = pd.CategoricalDtype(categories=stock_ids)
    m1["stock_id"] = m1["stock_id"].astype(stock_dtype)
    m3["stock_id"] = m3["stock_id"].astype(stock_dtype)
    m5["stock_id"] = m5["stock_id"].astype(stock_dtype)

    print("算 VWAP z-score / 候選觸發 / 三分類標籤...")
    df = make_features(m1, std_mult=std_mult, atr5_threshold=atr5_threshold, m3=m3, m5=m5)
    df = df.dropna(subset=FEATURES + ["target"])
    df["target"] = df["target"].astype(int)
    print(f"有效樣本: {len(df):,} 筆")

    if skip_cache:
        print("std_mult/atr5_threshold非正式設定，跳過cache寫入（避免弄髒正式cache）")
        return df

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE_PATH)
    print(f"cache已存至 {_CACHE_PATH}")
    return df


def _split_data(test_days: int = 30, use_cache: bool = False, start_date: str | None = None):
    """跟 train() 共用同一套切分邏輯，讓 evaluate()/confidence_report()
    能用「跟訓練時同一份」測試集去評估已存的模型，比照
    strategy/mkt/train.py::_split_data() 的說明。

    start_date：只篩 train_df（不影響 test_df），限制訓練資料從這個日期
    開始，留空＝用全部歷史。
    """
    df = _prepare_data(use_cache=use_cache)
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df, test_df = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    if start_date is not None:
        train_df = train_df[train_df["date"] >= pd.Timestamp(start_date)]
    return train_df, test_df


def train_lgbm(test_days: int = 30, start_date: str | None = None, use_cache: bool = False):
    import lightgbm as lgb

    train_df, test_df = _split_data(test_days, use_cache=use_cache, start_date=start_date)
    print(
        f"\n訓練: {len(train_df):,} ({train_df['date'].min().strftime('%Y-%m-%d')} ~ "
        f"{train_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"測試: {len(test_df):,} ({test_df['date'].min().strftime('%Y-%m-%d')} ~ "
        f"{test_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(f"\n訓練集標籤分佈:\n{(train_df['target'].value_counts(normalize=True) * 100).round(2)}")

    # class_weight="balanced"：三分類裡「無訊號」大概率佔絕大多數，不加權
    # 模型會偷懶全部猜「無訊號」，用 balanced 強迫模型認真學回歸/延續。
    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    print(f"\nAccuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
    print("\n混淆矩陣（列=實際，欄=預測，順序 回歸/無訊號/延續）:")
    print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
    print("\n分類報告:")
    print(
        classification_report(test_df["target"], y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0)
    )

    _MODEL_PATH_LGBM.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_LGBM)
    print(f"模型已存至 {_MODEL_PATH_LGBM}")
    return model


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError("找不到 LGBM 模型，請先執行 train_lgbm()")
    return joblib.load(_MODEL_PATH_LGBM)


_LOAD_MODEL_BY_TYPE = {"lgbm": load_model_lgbm}
_TRAIN_BY_TYPE = {"lgbm": train_lgbm}


_model_cache: dict[str, object] = {}


def load_model_by_type(model_type: str):
    """依 config.MODEL_TYPE（目前只有 "lgbm"）載入對應模型，
    run_backtest.py 跟 up/down/live.py 共用這支，切換模型只要改
    config.py::MODEL_TYPE 一個地方，比照 strategy/mkt/train.py 的做法。

    加記憶體快取（同一個 model_type 只從磁碟讀一次）：strategy/vwap_ml/up、
    strategy/vwap_ml/down 這種「同一個模型、只是過濾方向不同」的變體，
    開機時會各自呼叫一次 load_model_by_type(MODEL_TYPE)，沒有快取的話會
    重複 joblib.load() 出兩個不同物件，多佔記憶體；main/live_trader.py 的
    predict_live() 結果快取也是靠「model物件是否同一個」判斷能不能共用
    同一分鐘的推論結果，見 strategy/mkt/train.py::load_model_by_type() 的
    同樣說明。
    """
    if model_type not in _LOAD_MODEL_BY_TYPE:
        raise ValueError(f"未知 model_type: {model_type!r}，可用: {list(_LOAD_MODEL_BY_TYPE)}")
    if model_type not in _model_cache:
        _model_cache[model_type] = _LOAD_MODEL_BY_TYPE[model_type]()
    return _model_cache[model_type]


def _predict_with_threshold(model, test_df, threshold: float | None):
    """threshold=None 用 model.predict() 原本的判法（機率最高的類別勝出）；
    設 0~1 的數字時，改成「信心度夠了才判回歸/延續，不然算無訊號」的門檻式
    判法，比照 strategy/mkt/train.py::_predict_with_threshold() 的寫法。"""
    if threshold is None:
        return model.predict(test_df[FEATURES]), "未設信心度門檻"

    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    p_revert = proba[:, class_idx[0]]
    p_continue = proba[:, class_idx[2]]
    y_pred = np.ones(len(test_df), dtype=int)  # 預設判無訊號
    revert_pass = p_revert >= threshold
    continue_pass = p_continue >= threshold
    y_pred[revert_pass & ~continue_pass] = 0
    y_pred[continue_pass & ~revert_pass] = 2
    both = revert_pass & continue_pass
    y_pred[both] = np.where(p_continue[both] >= p_revert[both], 2, 0)
    return y_pred, f"信心度門檻={threshold:.2f}"


_DEFAULT_THRESHOLDS = [None, 0.5, 0.6, 0.7, 0.8]


def evaluate(
    model=None,
    test_days: int = 30,
    threshold: float | None | list[float | None] = _DEFAULT_THRESHOLDS,
    use_cache: bool = False,
):
    """單獨印混淆矩陣/分類報告，用已存的模型評估，不用重新跑一次 train()。"""
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    _, test_df = _split_data(test_days, use_cache=use_cache)

    print(
        f"測試: {len(test_df):,} 筆 ({test_df['date'].min().strftime('%Y-%m-%d')} ~ "
        f"{test_df['date'].max().strftime('%Y-%m-%d')})"
    )

    thresholds = threshold if isinstance(threshold, list) else [threshold]
    results = []
    for thr in thresholds:
        y_pred, thr_label = _predict_with_threshold(model, test_df, thr)
        print(f"\n{'=' * 60}\n{thr_label}\n{'=' * 60}")
        print(f"Accuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
        print("\n混淆矩陣（列=實際，欄=預測，順序 回歸/無訊號/延續）:")
        print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
        print("\n分類報告:")
        print(
            classification_report(
                test_df["target"], y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0
            )
        )
        results.append(y_pred)

    return test_df, (results[0] if len(results) == 1 else results)


def confidence_report(model=None, test_days: int = 30, thresholds: list[float] | None = None, use_cache: bool = False):
    """依信心度門檻掃描，分別看「回歸」「延續」這兩個類別在不同信心度下的
    precision/recall，比照 strategy/mkt/train.py::confidence_report() 的
    3分類版寫法。"""
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    _, test_df = _split_data(test_days, use_cache=use_cache)
    proba = model.predict_proba(test_df[FEATURES])
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    test_df = test_df.copy()
    test_df["proba_revert"] = proba[:, class_idx[0]]
    test_df["proba_continue"] = proba[:, class_idx[2]]

    for cls, col, name in [(0, "proba_revert", "回歸"), (2, "proba_continue", "延續")]:
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
            print(f"  {thr:.2f}  {n:>7,}  {tp:>7,}  {precision * 100:>8.2f}%  {recall * 100:>5.2f}%")

    return test_df


def feature_importance(model=None, top_n: int = 20):
    """顯示 LightGBM 特徵重要性，FEATURES 清單見 features.py（單一事實
    來源，這裡不重複列一份）。"""
    if model is None:
        model = load_model_by_type(MODEL_TYPE)

    print(f"\n── 特徵重要性（目前 FEATURES: {FEATURES}）──")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:top_n]:
        print(f"  {name:20s}  {imp:.4f}")


def main(
    mode: str = "",
    test_days: int = 30,
    threshold: float | list[float] | None = None,
    model_type: str = "lgbm",
    use_cache: bool = False,
    start_date: str | None = None,
):
    """CLI進入點，比照 strategy/mkt/train.py::main() 的 mode 切換方式。

    model_type：train 用哪個演算法（目前只有 lgbm）；importance/evaluate/
    confidence 則是讀哪個演算法已經訓練好的模型來評估，比照
    strategy/mkt/train.py::main() 的說明——三個模式共用同一套
    FEATURES/切分邏輯，只有這個參數決定要用哪個演算法。

    兩種用法都支援，互不衝突：
      1. VS Code按F5：__main__ 裡直接寫死 mode 變數，不帶任何CLI參數。
      2. 終端機帶參數：python -m strategy.vwap_ml.train evaluate --threshold 0.6
    """
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="vwap_ml 策略 — LightGBM")
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
            help="evaluate專用：信心度門檻(0~1)，可帶多個一次比較，留空=用predict()原本的判法",
        )
        parser.add_argument(
            "--model_type", type=str, default="lgbm", choices=["lgbm"], help="模型演算法（目前只有lgbm）"
        )
        parser.add_argument(
            "--use_cache",
            action="store_true",
            help="cache比來源資料新就直接讀（省時間）；不加這個旗標＝無條件重新計算並覆蓋cache（改過計算邏輯後要用這個）",
        )
        parser.add_argument(
            "--start_date",
            type=str,
            default=None,
            help="train專用：只篩訓練資料起始日期（例：2025-01-01），留空=用全部歷史",
        )
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        threshold = args.threshold
        model_type = args.model_type
        use_cache = args.use_cache
        start_date = args.start_date
    elif not mode:
        mode = "train"

    if mode == "train":
        _TRAIN_BY_TYPE[model_type](test_days=test_days, start_date=start_date, use_cache=use_cache)
    elif mode == "importance":
        feature_importance(model=_LOAD_MODEL_BY_TYPE[model_type]())
    elif mode == "evaluate":
        model = _LOAD_MODEL_BY_TYPE[model_type]()
        if threshold is not None:
            evaluate(model=model, test_days=test_days, threshold=threshold, use_cache=use_cache)
        else:
            evaluate(model=model, test_days=test_days, use_cache=use_cache)
    elif mode == "confidence":
        confidence_report(model=_LOAD_MODEL_BY_TYPE[model_type](), test_days=test_days, use_cache=use_cache)
    else:
        print(f"未知模式: {mode}，可用: train / importance / evaluate / confidence")


if __name__ == "__main__":
    """
    VS Code按F5（開發時常用）：直接改下面這幾行變數，不用打字。
    終端機帶參數會覆蓋掉下面寫死的值，見 main() 的說明。
    """
    mode = "train"  # train / importance / evaluate / confidence
    test_days = 30
    threshold = None  # 只有 mode="evaluate" 用得到；留 None = 用 evaluate() 自己的預設值
    model_type = "lgbm"  # 目前只有 lgbm
    use_cache = False
    start_date = "2024-01-01"  # 只有 mode="train" 用得到；留 None = 用全部歷史
    main(
        mode=mode,
        test_days=test_days,
        threshold=threshold,
        model_type=model_type,
        use_cache=use_cache,
        start_date=start_date,
    )
