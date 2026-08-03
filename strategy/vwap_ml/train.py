"""
vwap_ml 模型訓練 — LightGBM / XGBoost 三分類（0=回歸VWAP/1=無訊號/2=延續突破）

2026-07-26 討論：先比照 strategy/mkt/train.py 把 MODEL_TYPE 切換機制建起來
（_LOAD_MODEL_BY_TYPE/_TRAIN_BY_TYPE 字典＋load_model_by_type()），之後要
再加其他演算法（例如rfc）只要補一組 train_xxx()/load_model_xxx() 函式並
登記進字典，不用改 predict.py、run_backtest.py、up/down/live.py 任何一行。

== Main 模式 ==

train        訓練模型（--model_type 選演算法，lgbm/xgb），存至
             models/vwap_ml_{model_type}.pkl
importance   顯示特徵重要性（讀已存的模型，不重訓）
evaluate     單獨印混淆矩陣 + 分類報告（讀已存的模型，不重訓）
confidence   回歸/延續兩個類別的信心度門檻掃描，看不同機率門檻下的
             precision/recall（讀已存的模型，不重訓）
direction    把回歸/延續的預測依 trigger_side（上軌/下軌，對應實際做空/
             做多）拆開看precision/recall（讀已存的模型，不重訓）

用法：
    python -m strategy.vwap_ml.train train
    python -m strategy.vwap_ml.train train --model_type lgbm
    python -m strategy.vwap_ml.train importance
    python -m strategy.vwap_ml.train evaluate
    python -m strategy.vwap_ml.train confidence
    python -m strategy.vwap_ml.train direction

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

from data.raw_query import iter_m1_months, iter_m3_months, iter_m5_months
from strategy.vwap_ml.config import ATR5_FILTER_THRESHOLD, MODEL_TYPE, STD_MULT
from strategy.vwap_ml.features import FEATURES, make_features

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH_LGBM = _ROOT / "models/vwap_ml_lgbm.pkl"
_MODEL_PATH_XGB = _ROOT / "models/vwap_ml_xgb.pkl"
_CACHE_PATH = _ROOT / "cache/vwap_ml_prepared.parquet"
_SOURCE_DIRS = [_ROOT / "db/m1", _ROOT / "db/m3", _ROOT / "db/m5"]


def _cache_path_for(start_date: str | None) -> Path:
    """start_date=None（全部歷史）用預設的共用cache路徑；其他start_date
    各自存獨立檔案，比照 strategy/mkt/train.py::_cache_path_for(top_n) 的
    做法——start_date 是在載入資料時就套用的範圍限制（不是事後才篩），
    如果共用同一份cache，不同start_date會互相覆蓋、讀到錯的資料。"""
    if start_date is None:
        return _CACHE_PATH
    return _CACHE_PATH.parent / f"vwap_ml_prepared_{start_date}.parquet"


_TARGET_NAMES = ["回歸", "無訊號", "延續"]


def _source_mtime() -> float:
    """db/m1/db/m3/db/m5 裡「檔名代表最新月份」那個檔案的修改時間戳，
    不是掃全部檔案取最大值。

    2026-07-26 討論：原本掃全部檔案取最大mtime，會被完全無關、正在補
    2019~2023歷史資料缺口的背景程式（finmind.backfill_m1_history）誤觸發
    ——它會持續改到db/m1裡的舊月份檔案（例如2023_05.parquet），這種
    更新跟vwap_ml實際會用到的近期資料無關，卻讓cache被判定過期、觸發
    不必要的10幾分鐘重算（訓練時常用 start_date=2024-01-01之後，2019~
    2023的資料根本沒被用到）。只看「檔名最新的月份」那個檔案的時間戳，
    才能正確反映「有沒有新的近期資料進來」這個真正該偵測的情況，不會
    被補久遠歷史資料的背景作業影響。

    檔名格式 YYYY_MM.parquet，字串排序剛好等於時間排序，按檔名排序取
    最後一個即可，不用額外解析日期。
    """
    latest_mtimes = []
    for d in _SOURCE_DIRS:
        if not d.exists():
            continue
        files = sorted(f for f in d.iterdir() if f.suffix == ".parquet")
        if files:
            latest_mtimes.append(files[-1].stat().st_mtime)
    return max(latest_mtimes) if latest_mtimes else 0


def _cache_is_fresh(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= _source_mtime()


def _prepare_data(
    use_cache: bool = False,
    std_mult: float = STD_MULT,
    atr5_threshold: float = ATR5_FILTER_THRESHOLD,
    start_date: str | None = None,
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

    start_date（2026-07-26新增）：直接限制 iter_m1_months()/iter_m3_months()/
    iter_m5_months() 載入的範圍，不是像過去那樣先載入全部歷史、事後才篩選
    train_df——
    使用者實際上固定用 start_date="2024-01-01"，先載全部歷史（目前含
    2019~2023，正在被另一個獨立的 finmind.backfill_m1_history 背景作業
    持續補資料）再事後過濾，等於白白多算了完全用不到的資料，也讓
    _cache_is_fresh() 更容易被那個背景作業誤觸發成「過期」。不同
    start_date 存在各自獨立的cache檔案（見 _cache_path_for()），互不
    覆蓋。
    """
    skip_cache = std_mult != STD_MULT or atr5_threshold != ATR5_FILTER_THRESHOLD
    cache_path = _cache_path_for(start_date)
    if not skip_cache and use_cache and _cache_is_fresh(cache_path):
        print(f"cache比來源資料新，直接讀取cache... [{cache_path.name}]")
        return pd.read_parquet(cache_path)

    # 逐月讀取＋逐月跑 make_features()，不要一次把 start_date 到現在全部
    # 月份的 m1/m3/m5 讀進記憶體（2026-08-03 vwap_ml 回測記憶體爆炸時發現：
    # db/m3、db/m5 是 build_m3_m5_rolling.py 預先算好的「每分鐘一列」批次
    # 快取，跟 m1 同量級，2024-01起到現在三份一次讀完是32個月×3張近1.3億列
    # 的表，遠超一次全部讀進pandas能負擔的記憶體）。make_features() 內部
    # 的 _trim_to_needed_window() 只留當日開盤到 SESSION_END+
    # LABEL_HORIZON_MINUTES 這段（約全交易日的1/3），但這個裁切原本是在
    # 全部月份都讀完之後才做，幫不上尖峰記憶體；改成逐月讀取，每個月讀完
    # 立刻跑完整pipeline（含裁切、VWAP z-score、候選觸發、label、ATR5
    # 過濾），只保留篩選後很小的候選事件，讀下一個月前，這個月的原始
    # m1/m3/m5就可以被回收——尖峰記憶體從「32個月全部」降到「單月」等級。
    # VWAP/label 都是當日內計算（VWAP從當天開盤累積、label最多看未來
    # LABEL_HORIZON_MINUTES分鐘），不會跨月甚至跨日，逐月切開處理不影響
    # 任何數值結果。
    print(f"逐月載入分K並算特徵...{f'（start_date={start_date}）' if start_date else '（全部歷史）'}")
    month_dfs = []
    for m1, m3, m5 in zip(iter_m1_months(start_date), iter_m3_months(start_date), iter_m5_months(start_date)):
        month_df = make_features(m1, std_mult=std_mult, atr5_threshold=atr5_threshold, m3=m3, m5=m5)
        if not month_df.empty:
            month_dfs.append(month_df)

    print("合併各月候選事件 / 算三分類標籤結果...")
    df = pd.concat(month_dfs, ignore_index=True) if month_dfs else pd.DataFrame(columns=FEATURES + ["target"])
    df = df.dropna(subset=FEATURES + ["target"])
    df["target"] = df["target"].astype(int)
    print(f"有效樣本: {len(df):,} 筆")

    if skip_cache:
        print("std_mult/atr5_threshold非正式設定，跳過cache寫入（避免弄髒正式cache）")
        return df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"cache已存至 {cache_path}")
    return df


def _split_data(test_days: int = 30, use_cache: bool = False, start_date: str | None = None):
    """跟 train() 共用同一套切分邏輯，讓 evaluate()/confidence_report()
    能用「跟訓練時同一份」測試集去評估已存的模型，比照
    strategy/mkt/train.py::_split_data() 的說明。

    start_date 現在直接轉傳給 _prepare_data()，在載入資料的源頭就限制
    範圍（見 _prepare_data() 的說明），不再是「先載全部歷史、事後篩掉
    train_df」的做法，所以 train_df/test_df 都已經是限制範圍後的結果，
    不用再對 train_df 額外過濾一次。留空＝用全部歷史。
    """
    df = _prepare_data(use_cache=use_cache, start_date=start_date)
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df, test_df = df[df["date"] <= cutoff], df[df["date"] > cutoff]
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
    #
    # 2026-07-26：參數改用 experiments/tune_lgbm.py（Optuna，
    # start_date=2024-01-01、45天val/45天test三段式切分）搜出來的最佳組合
    # ——選 LGBM 不選 XGB 當正式模型：XGB調參後測試集precision看起來更高
    # (79.22%)，但依trigger_side拆開一看，下軌(做多)完全沒有任何預測
    # （n=0），是Optuna挑了一組極保守參數只在少數最有把握的做空案例出手
    # 取巧堆出來的數字，不能實際交易；LGBM這組雖然precision略低(73.51%)，
    # 但上軌(做空)876筆/下軌(做多)317筆都有紮實的訊號量，多空都能真的
    # 交易，見 experiments/tune_lgbm.py 執行紀錄與
    # strategy/vwap_ml/README.md 的說明。原始參數(n_estimators=300/
    # num_leaves=31/max_depth=6/learning_rate=0.05/min_child_samples=50/
    # subsample=0.8/colsample_bytree=0.8)沒調過，只是隨手設的起始值，
    # 不是什麼值得堅持的基準。
    model = lgb.LGBMClassifier(
        n_estimators=600,
        num_leaves=48,
        max_depth=6,
        learning_rate=0.05405420735163498,
        min_child_samples=85,
        subsample=0.9370412323040285,
        subsample_freq=1,
        colsample_bytree=0.5052284606961193,
        reg_lambda=0.16155957844464003,
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


def train_xgb(test_days: int = 30, start_date: str | None = None, use_cache: bool = False):
    """訓練 XGBoost 模型，跟 train_lgbm() 共用 FEATURES/切分邏輯，比照
    strategy/mkt/train.py::train_xgb() 的寫法（2026-07-26新增）。

    XGBClassifier 沒有像 LGBMClassifier 那樣原生支援多分類的
    class_weight="balanced"，要自己用
    sklearn.utils.class_weight.compute_sample_weight("balanced", y) 算出
    每筆樣本的權重，再用 fit(..., sample_weight=...) 傳進去，效果對應
    train_lgbm() 的 class_weight="balanced"。

    ⚠️ 目前參數是先抓 orb/mkt 用過的起始值（未針對vwap_ml調過），之後要
    調參請用 experiments/tune_xgb.py。
    """
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

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

    sample_weight = compute_sample_weight("balanced", train_df["target"])
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(train_df[FEATURES], train_df["target"], sample_weight=sample_weight)

    y_pred = model.predict(test_df[FEATURES])
    print(f"\nAccuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
    print("\n混淆矩陣（列=實際，欄=預測，順序 回歸/無訊號/延續）:")
    print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
    print("\n分類報告:")
    print(
        classification_report(test_df["target"], y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0)
    )

    _MODEL_PATH_XGB.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_XGB)
    print(f"模型已存至 {_MODEL_PATH_XGB}")
    return model


def load_model_xgb():
    if not _MODEL_PATH_XGB.exists():
        raise FileNotFoundError("找不到 XGB 模型，請先執行 train_xgb()")
    return joblib.load(_MODEL_PATH_XGB)


_LOAD_MODEL_BY_TYPE = {"lgbm": load_model_lgbm, "xgb": load_model_xgb}
_TRAIN_BY_TYPE = {"lgbm": train_lgbm, "xgb": train_xgb}


_model_cache: dict[str, object] = {}


def load_model_by_type(model_type: str):
    """依 config.MODEL_TYPE（"lgbm"/"xgb"）載入對應模型，
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
    start_date: str | None = None,
):
    """單獨印混淆矩陣/分類報告，用已存的模型評估，不用重新跑一次 train()。

    start_date 要跟訓練那個模型用的 start_date 一致，才會讀到同一份
    cache（見 _prepare_data()::_cache_path_for() 的說明），不然會因為
    cache路徑對不上而觸發不必要的重新計算。
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    _, test_df = _split_data(test_days, use_cache=use_cache, start_date=start_date)

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


def confidence_report(
    model=None,
    test_days: int = 30,
    thresholds: list[float] | None = None,
    use_cache: bool = False,
    start_date: str | None = None,
):
    """依信心度門檻掃描，分別看「回歸」「延續」這兩個類別在不同信心度下的
    precision/recall，比照 strategy/mkt/train.py::confidence_report() 的
    3分類版寫法。

    start_date 要跟訓練那個模型用的 start_date 一致，見 evaluate() 的說明。
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    _, test_df = _split_data(test_days, use_cache=use_cache, start_date=start_date)
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


_SIDE_DIRECTION = {
    0: {"upper": "做空", "lower": "做多"},  # 回歸：上軌觸發=從高點拉回=做空；下軌觸發=從低點反彈=做多
    2: {"upper": "做多", "lower": "做空"},  # 延續：上軌觸發=繼續往上=做多；下軌觸發=繼續往下=做空
}


def direction_breakdown(
    model=None,
    test_days: int = 30,
    threshold: float | None = None,
    use_cache: bool = False,
    start_date: str | None = None,
):
    """
    2026-07-26 新增：把「回歸」「延續」這兩個類別的預測結果，依觸發時的
    trigger_side（upper=上軌/+2觸發、lower=下軌/-2觸發）拆開來看precision/
    recall，比照 predict.py::_direction_probas() 的方向對應表（見
    _SIDE_DIRECTION）換算成實際做多/做空。

    動機：evaluate()/confidence_report() 印出來的 precision 是「回歸」
    整體混在一起算的（不分上軌/下軌），但上軌回歸(做空)跟下軌回歸(做多)
    對應到完全不同的實際交易方向，兩者的準確度可能不一樣（例如台股放空
    限制多、下軌回歸做多可能表現不同於上軌回歸做空），只看整體
    precision會被平均掉、看不出來哪個方向比較可靠。

    threshold=None 用 model.predict() 原本判法（機率最高的類別勝出）；
    設 0~1 的門檻則比照 evaluate()::_predict_with_threshold() 的判法。

    start_date 要跟訓練那個模型用的 start_date 一致，見 evaluate() 的說明。

    df 需已有 trigger_side 欄位（_split_data() 回傳的 test_df 本來就有，
    不在 FEATURES 清單裡但沒被 dropna() 濾掉）。
    """
    if model is None:
        model = load_model_by_type(MODEL_TYPE)
    _, test_df = _split_data(test_days, use_cache=use_cache, start_date=start_date)

    y_pred, thr_label = _predict_with_threshold(model, test_df, threshold)
    test_df = test_df.copy()
    test_df["_pred"] = y_pred

    print(f"── 依 trigger_side 拆解方向準確度（{thr_label}）──")
    for cls, name in [(0, "回歸"), (2, "延續")]:
        print(f"\n{name}（class={cls}）:")
        for side in ("upper", "lower"):
            sub = test_df[test_df["trigger_side"] == side]
            actual_cnt = int((sub["target"] == cls).sum())
            pred_cnt = int((sub["_pred"] == cls).sum())
            tp = int(((sub["target"] == cls) & (sub["_pred"] == cls)).sum())
            precision = tp / pred_cnt if pred_cnt else float("nan")
            recall = tp / actual_cnt if actual_cnt else float("nan")
            direction = _SIDE_DIRECTION[cls][side]
            print(
                f"  trigger_side={side}（{direction}）: 實際{name}={actual_cnt:,}  "
                f"預測{name}={pred_cnt:,}  猜中={tp:,}  precision={precision * 100:.2f}%  recall={recall * 100:.2f}%"
            )

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

    model_type：train 用哪個演算法（lgbm/xgb）；importance/evaluate/
    confidence/direction 則是讀哪個演算法已經訓練好的模型來評估，比照
    strategy/mkt/train.py::main() 的說明——這些模式共用同一套
    FEATURES/切分邏輯，只有這個參數決定要用哪個演算法。

    start_date：所有模式都會用到（2026-07-26改），決定讀/建哪一份cache
    （見 _prepare_data()::_cache_path_for() 的說明）——evaluate/confidence/
    direction 要跟當初 train 那次用同一個 start_date，才會讀到同一份
    cache，不然會因為cache路徑對不上而觸發不必要的重新計算。

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
            choices=["train", "importance", "evaluate", "confidence", "direction"],
            help="執行模式（預設train）",
        )
        parser.add_argument("--test_days", type=int, default=30, help="測試集天數")
        parser.add_argument(
            "--threshold",
            type=float,
            nargs="*",
            default=None,
            help="evaluate/direction專用：信心度門檻(0~1)，evaluate可帶多個一次比較、"
            "direction只吃第一個，留空=用predict()原本的判法",
        )
        parser.add_argument(
            "--model_type", type=str, default="lgbm", choices=["lgbm", "xgb"], help="模型演算法（lgbm/xgb）"
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
            help="限制載入範圍的起始日期（例：2024-01-01），留空=用全部歷史；"
            "train/evaluate/confidence/direction 都要傳同一個值才會讀到同一份cache",
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
            evaluate(model=model, test_days=test_days, threshold=threshold, use_cache=use_cache, start_date=start_date)
        else:
            evaluate(model=model, test_days=test_days, use_cache=use_cache, start_date=start_date)
    elif mode == "confidence":
        confidence_report(
            model=_LOAD_MODEL_BY_TYPE[model_type](), test_days=test_days, use_cache=use_cache, start_date=start_date
        )
    elif mode == "direction":
        # direction_breakdown() 只吃單一門檻（不像evaluate那樣掃一串），
        # --threshold 帶多個值時只取第一個
        single_threshold = threshold[0] if isinstance(threshold, list) and threshold else threshold
        direction_breakdown(
            model=_LOAD_MODEL_BY_TYPE[model_type](),
            test_days=test_days,
            threshold=single_threshold,
            use_cache=use_cache,
            start_date=start_date,
        )
    else:
        print(f"未知模式: {mode}，可用: train / importance / evaluate / confidence / direction")


if __name__ == "__main__":
    """
    VS Code按F5（開發時常用）：直接改下面這幾行變數，不用打字。
    終端機帶參數會覆蓋掉下面寫死的值，見 main() 的說明。
    """
    mode = "evaluate"  # train / importance / evaluate / confidence / direction
    test_days = 30
    threshold = None  # mode="evaluate"/"direction" 用得到；留 None = 用各自的預設判法
    model_type = "lgbm"  # lgbm / xgb
    use_cache = True
    start_date = "2024-01-01"  # 全部mode都會用到，決定讀哪一份cache；留 None = 用全部歷史
    main(
        mode=mode,
        test_days=test_days,
        threshold=threshold,
        model_type=model_type,
        use_cache=use_cache,
        start_date=start_date,
    )
