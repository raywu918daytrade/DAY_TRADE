"""
mkt 模型訓練 — RandomForest / XGBoost / LightGBM

2026-07-14 討論：先用最簡單的設定跑一次，確認整條 pipeline（特徵→標籤→
流動性過濾→時段過濾→訓練）真的串得起來、看得懂結果，之後再逐步加特徵、
調參數（先精簡、慢慢加，避免一次寫太多debug不動）。2026-07-21 加了
XGBoost/LightGBM，三個模型共用同一份 FEATURES/_split_data() 切分邏輯，
比照 strategy/rally/train.py 的三模型設計。

目前設定（都是前面幾支 experiments/ 腳本驗證出來的結論）：
    FEATURES        見 strategy/mkt/features.py（單一事實來源，這裡
                    不重複列一份，避免兩邊各自維護一份不一致）
    流動性過濾：前一日量前100名   訊號密度提升最明顯的門檻
    時段：9:11~9:30                9:00~9:10被判定為開盤集合競價剛結束的
                    雜訊時段、沒有方向性，已棄用（見 config.py 的說明）
    3分類標籤：漲/平/跌           HOLD_BARS=10、TP_PCT=SL_PCT=3%（見config.py）

== Main 模式 ==

train        訓練模型，存至 models/mkt_{rfc,xgb,lgbm}.pkl（用
             --model_type 選演算法，預設rfc）
importance   顯示特徵重要性（讀已存的模型，不重訓）
evaluate     單獨印混淆矩陣 + 分類報告（讀已存的模型 + 跟訓練時同一套切分
             邏輯重新切出測試集，不重訓）
confidence   漲/跌兩個稀有類別的信心度門檻掃描，看不同機率門檻下的
             precision/recall（讀已存的模型，不重訓）

用法：
    python -m strategy.mkt.train train
    python -m strategy.mkt.train train --model_type xgb
    python -m strategy.mkt.train train --model_type lgbm
    python -m strategy.mkt.train importance --model_type xgb
    python -m strategy.mkt.train evaluate
    python -m strategy.mkt.train confidence

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

from data.query import load_day, load_m1, load_m3, load_m3_std, load_m5, load_m5_std
from strategy.mkt.config import ATR5_FILTER_THRESHOLD, IDX_SYMBOL, TOP_N
from strategy.mkt.features import (
    FEATURES,
    add_atr5,
    add_bar_features,
    add_idx_gap_pct,
    add_m3_m5_features,
    add_m3_m5_std_features,
    add_ret_vs_idx,
    make_barrier_labels_3class,
    top_n_by_prev_day_volume,
)

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "models/mkt_rfc.pkl"
_MODEL_PATH_XGB = _ROOT / "models/mkt_xgb.pkl"
_MODEL_PATH_LGBM = _ROOT / "models/mkt_lgbm.pkl"
_CACHE_PATH = _ROOT / "cache/mkt_prepared.parquet"
_SOURCE_DIRS = [
    _ROOT / "db/m1",
    _ROOT / "db/m3",
    _ROOT / "db/m5",
    _ROOT / "db/m3_std",
    _ROOT / "db/m5_std",
    _ROOT / "db/fugle_day",
]


def _source_mtime() -> float:
    """db/m1、db/m3、db/m5 裡所有檔案中最新的修改時間戳，比照
    strategy/rally/features.py 的 _m1_mtime() 寫法。"""
    mtimes = [f.stat().st_mtime for d in _SOURCE_DIRS if d.exists() for f in d.iterdir() if f.suffix == ".parquet"]
    return max(mtimes) if mtimes else 0


def _cache_path_for(top_n: int) -> Path:
    """top_n=TOP_N（config.py正式設定，2026-07-25起=300）用原本那份共用
    cache（mkt_prepared.parquet）；其他top_n值各自存獨立檔案（2026-07-22
    討論）——top_n是在_prepare_data()內部就套用的流動性過濾，不像
    start_date是事後才篩，如果共用同一份cache，不同top_n會互相覆蓋、
    讀到錯的資料（例如拿top_n=100算出來的cache，被誤判成新鮮的
    top_n=300結果直接回傳）。"""
    return _CACHE_PATH if top_n == TOP_N else _ROOT / f"cache/mkt_prepared_top{top_n}.parquet"


def _cache_is_fresh(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= _source_mtime()


def _prepare_data(
    top_n: int = TOP_N,
    hour: int = 9,
    minute_min: int = 11,
    minute_max: int = 30,
    use_cache: bool = False,
    atr5_threshold: float = ATR5_FILTER_THRESHOLD,
) -> pd.DataFrame:
    """
    use_cache（2026-07-21討論）：
        True  = 不管 db/m1、db/m3、db/m5 有沒有更新，只要cache檔案存在
                就直接用（最快，但可能用到過期資料，適合快速反覆測試
                evaluate()/confidence_report() 這種下游邏輯、不在意資料
                新不新的情境）
        False（預設）= 自動判斷：cache存在且比三個來源資料夾裡最新的檔案
                還新，就直接用cache；只要來源有任何檔案比cache新（例如
                db/m1 今天寫進新的即時資料），就視為過期、重新跑一次完整
                流程並覆蓋掉舊cache。不用手動清cache、不用設定過期時間。

    top_n 不同值各自有獨立cache，見 _cache_path_for() 的說明。

    atr5_threshold（2026-07-25新增）：只給 experiments/ 拿來測試不同ATR5
    門檻用，預設就是正式設定 config.ATR5_FILTER_THRESHOLD。⚠️ 只要傳的值
    跟正式設定不同，就完全跳過cache讀寫（不讀舊cache、也不寫檔案）——
    cache只用 top_n 當key，不知道atr5_threshold是什麼，如果讓非預設門檻
    也去讀寫同一份cache，會弄髒正式pipeline在用的cache檔案。這代表用非
    預設門檻呼叫這支函式每次都會重新跑一次完整流程，比較慢，但只有
    experiments/ 會這樣用，不影響正式train/predict。
    """
    cache_path = _cache_path_for(top_n)
    skip_cache = atr5_threshold != ATR5_FILTER_THRESHOLD
    if not skip_cache:
        if use_cache and cache_path.exists():
            print(f"use_cache=True，直接讀取cache（不檢查來源資料有沒有更新）... [{cache_path.name}]")
            return pd.read_parquet(cache_path)
        if not use_cache and _cache_is_fresh(cache_path):
            print(f"cache比來源資料新，直接讀取cache... [{cache_path.name}]")
            return pd.read_parquet(cache_path)

    print("載入分K...")
    m1 = load_m1()
    m1["date"] = pd.to_datetime(m1["date"])
    m1 = m1.sort_values(["stock_id", "date"]).reset_index(drop=True)
    m1["day_date"] = m1["date"].dt.date

    print("算 ret_vs_idx...")
    # 0050 自己要留著算完才能排除，理由見 features.py::add_ret_vs_idx() 的說明
    df = add_ret_vs_idx(m1)

    print("算 0050 開盤跳空缺口...")
    # 同樣要在排除0050之前算，理由見 features.py::add_idx_gap_pct() 的說明
    df = add_idx_gap_pct(df, load_day())
    df = df[df["stock_id"] != IDX_SYMBOL]

    print(f"流動性過濾：前一日量前{top_n}名...")
    before = df["stock_id"].nunique()
    df = top_n_by_prev_day_volume(df, n=top_n)
    print(f"  股票數: {before} → {df['stock_id'].nunique()}（依日期各自篩選）")

    print("算當根K棒量能/波動特徵...")
    df = add_bar_features(df)

    print("算m3/m5收盤價/成交量特徵...")
    # load_m3()/load_m5() 是 db/m3、db/m5 全部歷史（build_m3_m5_rolling.py 預先聚合，
    # 見 data/query.py 的說明），這裡在流動性過濾之後才merge，減少資料量。
    df = add_m3_m5_features(df, load_m3(), load_m5())

    print("算m3_std/m5_std收盤價/成交量特徵...")
    # db/m3_std、db/m5_std（build_m3_m5_std.py 預先聚合）是標準獨立K棒，跟
    # 上面 rolling 版意義不同，見 features.py::add_m3_m5_std_features() 的說明。
    df = add_m3_m5_std_features(df, load_m3_std(), load_m5_std())

    print("算atr5（只拿來過濾用，不進FEATURES，見add_atr5()的說明）...")
    df = add_atr5(df)

    print("計算3分類標籤（漲=2/平=1/跌=0）...")
    df["target"] = make_barrier_labels_3class(df)

    print(f"時段過濾：{hour}:{minute_min:02d}~{hour}:{minute_max:02d}...")
    # 2026-07-20 討論：9:00~9:10改成不用（開盤集合競價剛結束，波動雖大但
    # 沒有方向性，不是要進場的時機），訓練/推論時段改成9:11~9:30。
    df = df[
        (df["date"].dt.hour == hour) & (df["date"].dt.minute >= minute_min) & (df["date"].dt.minute < minute_max)
    ].copy()

    print(f"ATR5平盤過濾：atr5>={atr5_threshold}（絕對門檻，跌/平/漲三類都篩，2026-07-23討論）...")
    # ⚠️ 一定要放在所有lag/rolling特徵（add_bar_features/add_m3_m5_features/
    # add_m3_m5_std_features）都算完、時段過濾之後才篩——這裡會刪列，放
    # 太早會破壞shift(1)這類lag運算的分鐘連續性，見
    # strategy/mkt/features.py::filter_by_atr5_percentile() 的說明（雖然
    # 這裡沒有直接呼叫那支函式，但原理跟警告完全一樣）。
    before_atr5 = len(df)
    df = df[df["atr5"] >= atr5_threshold].copy()
    print(f"  樣本數: {before_atr5:,} → {len(df):,}")

    df = df.dropna(subset=FEATURES + ["target"])
    df["target"] = df["target"].astype(int)
    print(f"有效樣本: {len(df):,} 筆")

    if skip_cache:
        print("atr5_threshold非正式設定，跳過cache寫入（避免弄髒正式cache）")
        return df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"cache已存至 {cache_path}")
    return df


def _split_data(test_days: int = 30, use_cache: bool = False, start_date: str | None = None):
    """跟 train() 共用同一套切分邏輯，讓 evaluate()/confidence_report() 能
    用「跟訓練時同一份」測試集去評估已存的模型，不用每次都重新訓練一次
    （2026-07-17 討論：混淆矩陣/信心度報表要能單獨印，不用綁著train()）。

    ⚠️ test_days 要跟當初訓練那個模型用的 test_days 一致，不然測試集會
    跟模型訓練時看過的資料重疊，評估結果不可信。

    use_cache：見 _prepare_data() 的說明。

    start_date（2026-07-22討論）：只篩 train_df（不影響 test_df），限制
    訓練資料從這個日期開始，不用全部歷史。⚠️ 故意放在 _prepare_data() 之後
    才篩，不是塞進 _prepare_data() 裡面——_prepare_data() 的cache新鮮度只看
    來源檔案mtime，不知道 start_date 是什麼，如果篩選寫在 _prepare_data()
    裡，改了 start_date 但來源沒更新，會拿到用舊start_date算出來的cache、
    不會生效也不會報錯。放在這裡篩是對記憶體裡的DataFrame做，改start_date
    馬上生效，且cache本身永遠存全部歷史、不用重建。留空＝用全部歷史（現在
    的行為，不受影響）。這是固定絕對日期，不是「最近N天」——隨時間推移
    train_df的範圍會越變越長（新資料進來、start_date不變），不是恆定窗口，
    要保持「只看最近N個月」的話要定期回來手動調整這個日期。

    evaluate()/confidence_report() 只用這支函式回傳的 test_df，不會用到
    train_df，所以不用把 start_date 傳給它們。
    """
    df = _prepare_data(use_cache=use_cache)
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df, test_df = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    if start_date is not None:
        train_df = train_df[train_df["date"] >= pd.Timestamp(start_date)]
    return train_df, test_df


def train(test_days: int = 20, start_date: str | None = None, use_cache: bool = False):
    # test_days=20（2026-07-16 從10調大）：0050歷史資料補齊後，訓練資料從
    # 約1.5個月變成6個月以上，但漲/跌是很稀有的類別（各僅佔2~5%），test_days
    # 太小的話測試集裡稀有類別樣本數太少、precision/recall算出來不穩定，
    # 不是「訓練資料變多就要按比例放大測試集」的邏輯。
    # start_date：只篩訓練資料起始日期，見 _split_data() 的說明。
    # use_cache（2026-07-22新增）：True＝不管背景有沒有寫入新的m1資料都直接
    # 用現成cache，見 _prepare_data() 的說明——反覆調整 start_date/超參數時
    # 不想每次都被背景寫入觸發整包重建，設True。
    train_df, test_df = _split_data(test_days, use_cache=use_cache, start_date=start_date)
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
    # zero_division=0：高門檻時模型可能對某類別完全不預測（例如threshold=0.8
    # 時漲/跌都沒半筆），precision算出來是0/0，sklearn預設會噴
    # UndefinedMetricWarning警告——這種情況語意上就是「0%」，明確設定
    # zero_division=0 讓它安靜地回報0.0，不用每次都跳一堆警告訊息。
    print(
        classification_report(
            test_df["target"], y_pred, labels=[0, 1, 2], target_names=["跌", "平", "漲"], zero_division=0
        )
    )

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}")
    return model


def train_xgb(test_days: int = 20, start_date: str | None = None, use_cache: bool = False):
    """訓練 XGBoost 模型，跟 train()（RandomForest）共用 FEATURES/切分邏輯，
    比照 strategy/rally/train.py 的 train_xgb() 寫法（2026-07-21 討論）。

    XGBClassifier 沒有像 RandomForestClassifier 那樣的 class_weight="balanced"
    參數（那個只支援二分類的 scale_pos_weight，這裡是3分類）——要達到
    同樣「稀有類別權重加大」的效果，要自己用
    sklearn.utils.class_weight.compute_sample_weight("balanced", y) 算出
    每筆樣本的權重，再用 fit(..., sample_weight=...) 傳進去。

    2026-07-21 第二輪：window_days拉大到45天（約30個交易日）重新tune+walk-
    forward，比第一輪的20天窗口更穩定（val/test落差從13個百分點縮小到5個
    百分點）。5個45天窗口下，新參數在0.5/0.6/0.7/0.8全部門檻都贏（0.5~0.7
    各4/5、0.8滿貫5/5），而且門檻越高贏越多（0.8：14.69%→21.26%），第一輪
    參數在0.7/0.8這兩個高信心度門檻其實是輸給最原始參數的，這輪明確修正
    了這個問題。第一輪參數（n_estimators=600/max_depth=10/learning_rate≈
    0.100/subsample≈0.923/colsample_bytree≈0.556/min_child_weight=74/
    reg_lambda≈1.926/gamma≈0.396）、最原始參數（n_estimators=300/max_depth=6/
    learning_rate=0.05/subsample=0.8/colsample_bytree=0.8/min_child_weight=50/
    reg_lambda=1.0，沒有gamma）都紀錄在這裡以防之後要retune時當對照組。
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
    print(f"\n訓練集標籤分佈:\n{(train_df['target'].value_counts(normalize=True)*100).round(2)}")

    sample_weight = compute_sample_weight("balanced", train_df["target"])
    model = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.12770505395846093,
        subsample=0.9541586311700712,
        colsample_bytree=0.9978337571109724,
        min_child_weight=22,
        reg_lambda=1.847987791882633,
        gamma=0.2824283269109099,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(train_df[FEATURES], train_df["target"], sample_weight=sample_weight)

    y_pred = model.predict(test_df[FEATURES])
    print(f"\nAccuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
    print("\n混淆矩陣（列=實際，欄=預測，順序 跌/平/漲）:")
    print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
    print("\n分類報告:")
    print(
        classification_report(
            test_df["target"], y_pred, labels=[0, 1, 2], target_names=["跌", "平", "漲"], zero_division=0
        )
    )

    _MODEL_PATH_XGB.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_XGB)
    print(f"模型已存至 {_MODEL_PATH_XGB}")
    return model


def train_lgbm(test_days: int = 20, start_date: str | None = None, use_cache: bool = False):
    """訓練 LightGBM 模型，跟 train()（RandomForest）共用 FEATURES/切分邏輯，
    比照 strategy/rally/train.py 的 train_lgbm() 寫法（2026-07-21 討論）。
    LGBMClassifier 的 class_weight="balanced" 是原生支援多分類的，跟
    RandomForestClassifier 用法一致，不用像 XGBoost 那樣另外算sample_weight。

    2026-07-23：超參數換成 strategy/mkt/experiments/tune_lgbm.py（Optuna，
    45天val/test切分）+ walk_forward_lgbm.py（5個45天窗口）驗證過的一組。
    ⚠️ 跟train_xgb()那次不同，這組不是全面勝利：threshold=0.5/0.6全面贏
    （mean precision分別5.21%→8.41%、6.18%→8.42%），0.7小贏（8.78%→9.89%，
    但std下降代表比較穩），threshold=0.8打平偏輸（15.03%→13.33%，只有3/5
    窗口贏）——貼這組上來是因為低/中門檻確實比舊參數穩定變好，且原本的
    n_estimators=300/learning_rate=0.05/num_leaves=31/min_child_samples=50/
    subsample=0.8（沒有subsample_freq，subsample其實從未真的生效過，見
    tune_lgbm.py檔頭說明）根本沒調過，不是什麼值得堅持的基準。原始參數
    紀錄在這裡以防之後要retune時當對照組。
    """
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
    print(f"\n訓練集標籤分佈:\n{(train_df['target'].value_counts(normalize=True)*100).round(2)}")

    model = lgb.LGBMClassifier(
        n_estimators=800,
        num_leaves=101,
        max_depth=8,
        learning_rate=0.17376155767832735,
        min_child_samples=16,
        subsample=0.8372082477437949,
        subsample_freq=1,
        colsample_bytree=0.6699451641790991,
        reg_lambda=1.1052479063154046,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    print(f"\nAccuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
    print("\n混淆矩陣（列=實際，欄=預測，順序 跌/平/漲）:")
    print(confusion_matrix(test_df["target"], y_pred, labels=[0, 1, 2]))
    print("\n分類報告:")
    print(
        classification_report(
            test_df["target"], y_pred, labels=[0, 1, 2], target_names=["跌", "平", "漲"], zero_division=0
        )
    )

    _MODEL_PATH_LGBM.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_LGBM)
    print(f"模型已存至 {_MODEL_PATH_LGBM}")
    return model


def load_model():
    if not _MODEL_PATH.exists():
        raise FileNotFoundError("找不到模型，請先執行 train()")
    return joblib.load(_MODEL_PATH)


def load_model_xgb():
    if not _MODEL_PATH_XGB.exists():
        raise FileNotFoundError("找不到 XGB 模型，請先執行 train_xgb()")
    return joblib.load(_MODEL_PATH_XGB)


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError("找不到 LGBM 模型，請先執行 train_lgbm()")
    return joblib.load(_MODEL_PATH_LGBM)


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
    model=None,
    test_days: int = 30,
    threshold: float | None | list[float | None] = _DEFAULT_THRESHOLDS,
    use_cache: bool = False,
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

    use_cache：見 _prepare_data() 的說明。evaluate() 常常要反覆呼叫比較不同
    門檻/模型，預設還是False（自動判斷新不新），想加速反覆測試時才設True
    （2026-07-21討論）。
    """
    if model is None:
        model = load_model()
    _, test_df = _split_data(test_days, use_cache=use_cache)

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
        # zero_division=0：高門檻時模型可能對某類別完全不預測（例如threshold=0.8
        # 時漲/跌都沒半筆），precision算出來是0/0，sklearn預設會噴
        # UndefinedMetricWarning警告——這種情況語意上就是「0%」，明確設定
        # zero_division=0 讓它安靜地回報0.0，不用每次都跳一堆警告訊息。
        print(
            classification_report(
                test_df["target"], y_pred, labels=[0, 1, 2], target_names=["跌", "平", "漲"], zero_division=0
            )
        )
        results.append(y_pred)

    return test_df, (results[0] if len(results) == 1 else results)


def confidence_report(model=None, test_days: int = 30, thresholds: list[float] | None = None, use_cache: bool = False):
    """依信心度（模型算出來的機率）門檻掃描，分別看「漲」「跌」這兩個稀有
    類別在不同信心度下的 precision/recall，比照 strategy/rally/validate.py
    的 confidence_report() 概念，改成3分類版（要分開看漲/跌，不是單一
    proba≥threshold 就好）。

    門檻越高，預測數越少，但理論上precision應該要越高——如果掃出來precision
    沒有隨門檻提升，代表模型的機率輸出沒有校準好、"信心度"不能拿來當篩選依據。

    use_cache：見 _prepare_data() 的說明。
    """
    if model is None:
        model = load_model()
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    _, test_df = _split_data(test_days, use_cache=use_cache)
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
    """顯示 RandomForest 特徵重要性，FEATURES 清單見 features.py（單一事實
    來源，這裡不重複列一份）。"""
    if model is None:
        model = load_model()

    print(f"\n── 特徵重要性（目前 FEATURES: {FEATURES}）──")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:top_n]:
        print(f"  {name:20s}  {imp:.4f}")


_LOAD_MODEL_BY_TYPE = {"rfc": load_model, "xgb": load_model_xgb, "lgbm": load_model_lgbm}
_TRAIN_BY_TYPE = {"rfc": train, "xgb": train_xgb, "lgbm": train_lgbm}


_model_cache: dict[str, object] = {}


def load_model_by_type(model_type: str):
    """依 config.MODEL_TYPE（"rfc"/"xgb"/"lgbm"）載入對應模型，
    run_backtest.py 跟 live.py 共用這支，切換模型只要改 config.py 一個地方，
    比照 strategy/rally/train.py、strategy/orb/train.py 的做法。

    2026-07-25加記憶體快取（同一個 model_type 只從磁碟讀一次）：strategy/mkt/up、
    strategy/mkt/down 這種「同一個模型、只是过濾方向不同」的變體，開機時會各自
    呼叫一次 load_model_by_type(MODEL_TYPE)，沒有快取的話會重複 joblib.load()
    出兩個不同物件，多佔記憶體；main/live_trader.py::on_minute() 的 predict_live()
    結果快取也是靠「model物件是否同一個」判斷能不能共用同一分鐘的推論結果
    （見該檔案說明），這裡回傳同一個物件，那層快取才會生效。
    """
    if model_type not in _LOAD_MODEL_BY_TYPE:
        raise ValueError(f"未知 model_type: {model_type!r}，可用: {list(_LOAD_MODEL_BY_TYPE)}")
    if model_type not in _model_cache:
        _model_cache[model_type] = _LOAD_MODEL_BY_TYPE[model_type]()
    return _model_cache[model_type]


def main(
    mode: str = "",
    test_days: int = 30,
    threshold: float | list[float] | None = None,
    model_type: str = "rfc",
    use_cache: bool = False,
    start_date: str | None = None,
):
    """CLI進入點，比照 strategy/rally/entry.py 的 mode 切換方式（mkt
    目前只有 train.py，還沒有 predict.py/live.py/entry.py，等那些補齊了
    再考慮要不要拆成獨立的 entry.py）。可用模式見檔頭的「== Main 模式 ==」。

    model_type：train 用哪個演算法（rfc/xgb/lgbm，預設rfc）；importance/
    evaluate/confidence 則是讀哪個演算法已經訓練好的模型來評估（2026-07-21
    討論：三個模型共用同一套 FEATURES/切分邏輯，只有這個參數決定要用哪個
    演算法，比照 strategy/rally 的 train/train_xgb/train_lgbm 三選一設計）。

    兩種用法都支援，互不衝突（2026-07-18 討論）：
      1. VS Code按F5：__main__ 裡直接寫死 mode 變數，不帶任何CLI參數
         （sys.argv長度=1，只有腳本路徑本身），這裡就直接用傳進來的 mode，
         不會被argparse覆蓋。
      2. 終端機帶參數：python -m strategy.mkt.train evaluate --threshold 0.6
         （sys.argv長度>1），這裡改用argparse解析，CLI帶的參數會覆蓋掉
         __main__ 裡寫死的值。
    """
    # 只有終端機真的帶了CLI參數才用argparse覆蓋；F5直接執行（sys.argv只有
    # 腳本路徑本身，長度=1）就尊重呼叫端傳進來的 mode/test_days/threshold，
    # 不要看mode是不是空字串來判斷（之前這樣寫，F5執行時只要mode沒填就會
    # 誤觸發argparse，去讀根本不存在的CLI參數而出錯或用到不對的預設值）。
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="mkt 策略 — RandomForest/XGBoost/LightGBM")
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
        parser.add_argument(
            "--model_type", type=str, default="rfc", choices=["rfc", "xgb", "lgbm"], help="模型演算法（預設rfc）"
        )
        parser.add_argument(
            "--use_cache",
            action="store_true",
            help="不管來源資料有沒有更新，只要cache存在就直接用（見_prepare_data()說明）",
        )
        parser.add_argument(
            "--start_date",
            type=str,
            default=None,
            help="train專用：只篩訓練資料起始日期（例：2025-01-01），留空=用全部歷史（見_split_data()說明）",
        )
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        # nargs="*" 沒帶值時是 None；帶一個值時是長度1的list，跟F5那邊
        # 「單一float」的用法一致傳進 evaluate() 就好，不用特別拆成純數字
        threshold = args.threshold
        model_type = args.model_type
        use_cache = args.use_cache
        start_date = args.start_date
    elif not mode:
        mode = "train"  # F5執行、__main__也沒寫mode時的保底預設

    if mode == "train":
        _TRAIN_BY_TYPE[model_type](test_days=test_days, start_date=start_date, use_cache=use_cache)
    elif mode == "importance":
        feature_importance(model=_LOAD_MODEL_BY_TYPE[model_type]())
    elif mode == "evaluate":
        # threshold 沒特別指定（None）時不傳進去，讓 evaluate() 自己的預設值
        # （_DEFAULT_THRESHOLDS）生效，不要在這裡另外複製一份預設值
        # （2026-07-18 討論：預設值該由 evaluate() 自己負責，main()/CLI只在
        # 使用者真的想覆蓋時才傳）。
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
        model_type  "rfc" / "xgb" / "lgbm"（2026-07-21新增，預設rfc）。
                    train時決定訓練哪個演算法；importance/evaluate/
                    confidence時決定讀哪個演算法已存的模型。
        use_cache   True/False（2026-07-21新增，只有evaluate/confidence
                    會用到，train不受影響）。見 _prepare_data() 的說明：
                    True=不管來源資料新不新都直接用cache（快，適合反覆
                    測試evaluate/confidence）；False=自動判斷，來源資料
                    有更新才重算。
        start_date  只有 mode="train" 會用到（2026-07-22新增）。只篩訓練
                    資料起始日期（例："2025-01-01"），留 None = 用全部
                    歷史（現在的行為）。⚠️ 固定絕對日期，不是「最近N天」，
                    隨時間推移train_df範圍會越來越長，要保持「只看最近N
                    個月」得定期回來手動調整，見 _split_data() 的說明。

    2. 終端機帶參數（會覆蓋掉下面寫死的值）：
        python -m strategy.mkt.train train
        python -m strategy.mkt.train train --model_type xgb
        python -m strategy.mkt.train train --model_type lgbm
        python -m strategy.mkt.train train --model_type xgb --start_date 2025-01-01
        python -m strategy.mkt.train importance --model_type xgb
        python -m strategy.mkt.train evaluate --threshold 0.6
        python -m strategy.mkt.train evaluate --use_cache
        python -m strategy.mkt.train confidence
    """
    mode = "evaluate"  # train / importance / evaluate / confidence
    test_days = 30
    threshold = None  # 只有 mode="evaluate" 用得到；留 None = 用 evaluate() 自己的預設值
    model_type = "xgb"  # rfc / xgb / lgbm
    use_cache = True  # 只有 mode="evaluate"/"confidence" 用得到
    start_date = "2024-01-01"  # 只有 mode="train" 用得到；留 None = 用全部歷史
    main(
        mode=mode,
        test_days=test_days,
        threshold=threshold,
        model_type=model_type,
        use_cache=use_cache,
        start_date=start_date,
    )
