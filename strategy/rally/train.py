"""
模型訓練 — RandomForest / XGBoost / LightGBM（CLI 進入點）

== 策略重點（破底翻，rally） ==

抓「先跌後漲」的當日反彈型態：股價當天先破底、動能轉弱後又開始翻揚，賭這股
反彈動能會延續。不是規則式硬過濾，而是用機器學習分類器（RandomForest/XGBoost/
LightGBM 三選一比較）學這個型態加上一整套盤中動能/量能/位置特徵，去預測「進場
後 30 根分K內（HOLD_BARS），會先漲 3%（TP_PCT）還是先跌 3%（SL_PCT）」
（triple barrier 標籤，target=1/0，見 features.py）。

breakout_signal（第1根5分鐘K跌、第2根5分鐘K漲＝先跌後漲）原本是即時交易的硬
過濾規則，2026-07-09 驗證後拿掉了——加這道規則反而讓勝率下降 8-10 個百分點
（見 experiments/breakout_filter_eval.py），現在只是模型的輸入特徵之一，讓
樹模型自己決定要不要用、怎麼用，不再強制訊號一定要先跌後漲。

全天訓練（不鎖 9:14~9:30 黃金窗口），靠 minutes_since_open 這個特徵讓模型自己
判斷開盤動能期 vs 中午盤整期，交易時段限制交給呼叫端（回測用 first_entry_time/
last_entry_time，即時交易用 config.py 的 SESSION_START/END）。當沖策略，
不留倉，收盤前強制平倉（live_trader.py 的 _force_close_eod）。

股票母體：db/tickers/tick_universe.parquet 固定 400 支（2026-08-06起，比照
strategy/mkt/train.py、strategy/orb/features.py 的做法，見 features.py 的
_compute_month_features() 說明），不再用全市場~2700支，篩選在 features.py
算特徵之前就做，訓練/回測/即時推論都是同一份固定清單，不會因為 db/m1 涵蓋
的股票範圍隨時間變動就跟著變。

三個模型共用 features.py 的 FEATURES 與 triple barrier 標籤，用同一份
_prepare_train_test() 切分全天訓練/測試集，方便公平比較。

實際邏輯拆到同資料夾底下：
    config.py     交易相關設定（TP/SL/HOLD_BARS、SESSION 時段、ATR_FILTER_THRESHOLD）
    features.py   特徵工程、triple barrier 標籤、FEATURES 清單、load_features() cache
    validate.py   信心度/召回率/模型×時段×信心度交叉報表、特徵重要性
    predict.py    批次與即時推論（predict_live 是正式對外入口）
    experiments/  一次性假設驗證（例如破底翻要不要拆專門模型/硬過濾值不值得留、
                  超參數調參 tune_xgb.py/tune_lgbm.py、walk-forward 驗證），
                  還沒有定論、不是核心流程，獨立於上面幾支之外

2026-08-06：entry.py 併進本檔（比照 strategy/mkt/train.py、strategy/orb/train.py
沒有獨立 entry.py 的做法），CLI 進入點跟訓練邏輯放同一支檔案，不用再跳兩個
檔案對照參數。

== Main 模式 ==

train / train_xgb / train_lgbm   訓練模型
validate                         信心度分析 + 召回率分析 + 模型×時段×信心度交叉報表
signals                          每小時訊號數 vs 抓到筆數（固定門檻，原始筆數，見 validate.hour_signal_report）
importance                       特徵重要性
build_m3_m5                      補 db/m3/db/m5（增量，只重算比 db/m1 舊的月份）——訓練前務必先跑，
                                  不然 load_features() fallback 讀到的 m3/m5 可能比 db/m1 舊，訓練資料
                                  會悄悄漏掉最新幾天（2026-07-13 實際發生過：db/m1 到 07-09、
                                  db/m3/db/m5 卡在 07-07，少算了2個交易日）

用法：
    python -m strategy.rally.train train
    python -m strategy.rally.train train_xgb --start_date 2021-01-01
    python -m strategy.rally.train train_lgbm
    python -m strategy.rally.train validate
    python -m strategy.rally.train importance
    python -m strategy.rally.train build_m3_m5
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.build_m3_m5_rolling import build as build_m3_m5
from strategy.rally.config import (  # noqa: F401  (re-export：交易參數一眼看到全部)
    ATR_FILTER_THRESHOLD,
    HOLD_BARS,
    SESSION_END,
    SESSION_START,
    SL_PCT,
    TP_PCT,
)
from strategy.rally.features import FEATURES, load_features

# strategy.rally.validate 反過來 import 這支檔案的 _MODEL_PATH*/load_model*
# （信心度報表要載入已訓練好的模型），放頂層 import 會形成循環 import，
# 延後到 main() 裡面才 import（那時候這支檔案的模組層級程式碼都已經跑完，
# 不會卡在「模組還沒初始化完成」那個問題）。

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "models/m1_rfc.pkl"
_MODEL_PATH_XGB = _ROOT / "models/m1_xgb.pkl"
_MODEL_PATH_LGBM = _ROOT / "models/m1_lgbm_breakout.pkl"


def train(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """
    訓練 RandomForest 模型。

    Parameters
    ----------
    test_days : int
        以最後 N 天為測試集（預設 10）。若指定 start_date/end_date 則忽略此參數。
    start_date : str
        訓練資料起日，格式 "YYYY-MM-DD"。留空表示不限制。會往下傳給
        load_features()，在算特徵之前就篩掉更早的資料，減少計算量（見
        features.load_features() 的說明），不是算完全部歷史才篩。
    end_date : str
        訓練資料迄日，格式 "YYYY-MM-DD"。留空表示不限制（只在算完特徵後篩選，
        不影響特徵計算量）。
    use_cache : bool
        False 時不管 cache 現在是什麼狀態，一律重新計算特徵（見 features.load_features()）。
    """
    print("特徵工程...")
    df = load_features(use_cache=use_cache, start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵: {FEATURES}")
    print(f"  全天有效樣本: {len(df):,} 筆")

    # ATR 平盤過濾（見 config.py::ATR_FILTER_THRESHOLD 的說明）：篩掉波動太小、
    # 幾乎注定不會觸發 triple barrier 停利/停損的樣本，train/predict/
    # predict_live 三邊要用同一個門檻，不能只改這裡。
    before_atr = len(df)
    df = df[df["m1_atr"] >= ATR_FILTER_THRESHOLD]
    print(f"  ATR過濾（m1_atr>={ATR_FILTER_THRESHOLD}）: {before_atr:,} → {len(df):,} 筆")

    # 日期過濾
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    print(f"  日期區間: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

    # 時間切割（若有指定明確日期範圍則以日期切割，否則用 test_days）
    if start_date and end_date:
        cutoff = df["date"].quantile(0.8)  # 前 80% 訓練、後 20% 測試
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    else:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    print(
        f"  訓練: {len(train_df):,} ({train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"  測試: {len(test_df):,} ({test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')})"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        # 2026-08-06：train_df 現在是2021年起+固定400支股票，全天樣本數
        # 2500萬筆量級，n_jobs=-1（全部10核平行造樹）+ 預設 bootstrap 用全部
        # 樣本，實測把 swap 吃到94.8%（機器只有25.8GB RAM），改成 n_jobs=4
        # （降低同時平行造樹的記憶體疊加）+ max_samples=0.3（每棵樹bootstrap
        # 只抽30%樣本，每棵樹記憶體需求大幅下降，對模型品質通常影響不大，
        # 樹之間差異變大甚至可能略有幫助）。
        n_jobs=4,
        max_samples=0.3,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}")

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


_MODEL_LOADERS = {
    "rfc": load_model,
    "xgb": load_model_xgb,
    "lgbm": load_model_lgbm,
}


def load_model_by_type(model_type: str):
    """依 config.MODEL_TYPE（"rfc"/"xgb"/"lgbm"）載入對應模型，
    run_backtest.py 跟 live.py 共用這支，切換模型只要改 config.py 一個地方。"""
    if model_type not in _MODEL_LOADERS:
        raise ValueError(f"未知 model_type: {model_type!r}，可用: {list(_MODEL_LOADERS)}")
    return _MODEL_LOADERS[model_type]()


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost / LightGBM 訓練（與 RFC 共用同一份 FEATURES 與標籤）
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare_train_test(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """載入特徵並切分全天訓練/測試集（三模型共用）。"""
    print("特徵工程...")
    df = load_features(use_cache=use_cache, start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵數: {len(FEATURES)}")
    print(f"  全天有效樣本: {len(df):,} 筆")

    # ATR 平盤過濾，見 train() 裡同樣邏輯的說明。
    before_atr = len(df)
    df = df[df["m1_atr"] >= ATR_FILTER_THRESHOLD]
    print(f"  ATR過濾（m1_atr>={ATR_FILTER_THRESHOLD}）: {before_atr:,} → {len(df):,} 筆")

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    print(f"  日期區間: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

    if start_date and end_date:
        cutoff = df["date"].quantile(0.8)
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    else:
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)
        train_df = df[df["date"] <= cutoff]
        test_df = df[df["date"] > cutoff]
    print(
        f"  訓練: {len(train_df):,} ({train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"  測試: {len(test_df):,} ({test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')})"
    )
    return train_df, test_df


def train_xgb(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """訓練 XGBoost 模型（與 RFC 共用 FEATURES）。"""
    from xgboost import XGBClassifier

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date, use_cache)

    model = XGBClassifier(
        # 2026-07-21 用 experiments/tune_xgb.py（Optuna，train/val/test 三段式
        # 切分）調過，再用 experiments/walk_forward_xgb.py（4個獨立窗口）驗證：
        # threshold>=0.60 時新參數 4/4 窗口都贏舊參數（0.60: 55.18% vs 47.00%，
        # 0.65: 66.03% vs 49.16%），才貼進來取代原本隨便設的預設值。這組參數是
        # 用 2026-01~07、全市場股票調的，2026-08-06 改成固定400支+2021年起
        # 重訓後，建議重新跑一次 tune_xgb.py/walk_forward_xgb.py 確認還適用。
        n_estimators=500,
        max_depth=3,
        learning_rate=0.013674939977956308,
        subsample=0.603928001148621,
        colsample_bytree=0.8222495158859673,
        min_child_weight=25,
        reg_lambda=1.9326921144453604,
        gamma=2.5044241331307036,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_XGB.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_XGB)
    print(f"模型已存至 {_MODEL_PATH_XGB}")
    return model


def train_lgbm(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """訓練 LightGBM 模型（與 RFC 共用 FEATURES）。"""
    import lightgbm as lgb

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date, use_cache)

    model = lgb.LGBMClassifier(
        # 2026-07-21 用 experiments/tune_lgbm.py（Optuna，train/val/test 三段式
        # 切分）調過：val precision 62.83% → test（完全沒被調參碰過）60.11%，
        # 掉幅小、generalize 得好，才貼進來取代原本隨便設的預設值。同一次也
        # 調過 XGB，但那組參數 val 62.12% → test 只剩 44.29%（掉了17.8個百分
        # 點、幾乎跟瞎猜差不多），明顯是過擬合到 val 期間，沒有採用。這組參數
        # 是用 2026-01~07、全市場股票調的，2026-08-06 改成固定400支+2021年起
        # 重訓後，建議重新跑一次 tune_lgbm.py/walk_forward_lgbm.py 確認還適用。
        n_estimators=200,
        num_leaves=101,
        max_depth=3,
        learning_rate=0.03657843946200939,
        min_child_samples=28,
        subsample=0.8600035132620599,
        subsample_freq=1,  # LightGBM sklearn API 沒設這個 subsample 不會生效（bagging_freq
        # 預設0，bagging_fraction 會被忽略當1.0處理）——2026-07-21 發現，見
        # experiments/tune_lgbm.py 的說明，補上才讓 subsample 真的有作用
        colsample_bytree=0.8319671794969992,
        reg_lambda=0.5187813029919602,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"\nAccuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH_LGBM.parent.mkdir(exist_ok=True)
    joblib.dump(model, _MODEL_PATH_LGBM)
    print(f"模型已存至 {_MODEL_PATH_LGBM}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 進入點
# ═══════════════════════════════════════════════════════════════════════════════


def main(
    mode: str = "",
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
):
    """
    當沖策略 RandomForest 主程式。

    支援兩種用法：
      1. 直接傳參數：main(mode="train")
      2. CLI 執行：python -m strategy.rally.train train

    Parameters
    ----------
    mode : str
        執行模式。留空則從 CLI 讀取。
        train / train_xgb / train_lgbm / validate / signals / importance / build_m3_m5
    test_days : int
        測試集天數（預設 10）
    start_date : str
        資料起日，格式 "YYYY-MM-DD"。留空不限制。
    end_date : str
        資料迄日，格式 "YYYY-MM-DD"。留空不限制。
    use_cache : bool
        False 時不管特徵 cache 現在是什麼狀態，一律重新計算（見
        features.load_features()）。只影響 train / train_xgb / train_lgbm。
    """
    if not mode:
        parser = argparse.ArgumentParser(
            description="當沖策略 — RandomForest（close + volume + 過去5天日K）",
        )
        parser.add_argument(
            "mode",
            choices=[
                "train",
                "train_xgb",
                "train_lgbm",
                "validate",
                "signals",
                "importance",
                "build_m3_m5",
            ],
            help="執行模式",
        )
        parser.add_argument("--test_days", type=int, default=10, help="測試集天數")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        parser.add_argument(
            "--use_cache",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="是否用特徵 cache（預設 True）；--no-use_cache 強制重新計算（只影響 train 系列 mode）",
        )
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        start_date = args.start_date
        end_date = args.end_date
        use_cache = args.use_cache

    if mode == "train":
        train(test_days=test_days, start_date=start_date, end_date=end_date, use_cache=use_cache)

    elif mode == "train_xgb":
        train_xgb(test_days=test_days, start_date=start_date, end_date=end_date, use_cache=use_cache)

    elif mode == "train_lgbm":
        train_lgbm(test_days=test_days, start_date=start_date, end_date=end_date, use_cache=use_cache)

    elif mode == "validate":
        from strategy.rally.validate import confidence_report, coverage_report, model_hour_confidence_report

        confidence_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        coverage_report(test_days=test_days, start_date=start_date, end_date=end_date)
        print()
        model_hour_confidence_report(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "signals":
        from strategy.rally.validate import hour_signal_report

        hour_signal_report(test_days=test_days, start_date=start_date, end_date=end_date)

    elif mode == "importance":
        from strategy.rally.validate import feature_importance

        feature_importance()

    elif mode == "build_m3_m5":
        # 增量重算：只補比 db/m1 舊的月份，訓練前跑這個確保 load_features()
        # fallback 讀到的 db/m3/db/m5 沒有漏掉最新幾天。
        build_m3_m5(incremental=True)

    else:
        print(f"未知模式: {mode}，可用: train / train_xgb / train_lgbm / validate / signals / importance / build_m3_m5")


if __name__ == "__main__":
    """
    直接執行本檔的進入點。

    在這裡直接改 mode 變數即可切換模式，不用每次打 CLI 參數。
    也可從 terminal 用 argparse 呼叫，例如：
        python -m strategy.rally.train train
        python -m strategy.rally.train validate
        python -m strategy.rally.train importance

    可用 mode:
        "train"       訓練 RandomForest 模型（存至 models/m1_rfc.pkl）
        "validate"    信心度分析 + 召回率分析 + 模型×時段×信心度交叉報表
        "signals"     每小時訊號數 vs 抓到筆數（固定門檻，原始筆數）
        "importance"  顯示特徵重要性
        "train_xgb"   訓練 XGBoost 模型（存至 models/m1_xgb.pkl）
        "train_lgbm"  訓練 LightGBM 模型（存至 models/m1_lgbm_breakout.pkl）
        "build_m3_m5" 增量補 db/m3/db/m5（只重算比 db/m1 舊的月份），train 前先跑這個
        ""            走 CLI argparse（terminal 下帶參數）

    可選參數（CLI 或下方變數）：
        test_days     測試集天數（預設 10，取最後 N 天）
        start_date    資料起日 YYYY-MM-DD（留空 = 不限制）
        end_date      資料迄日 YYYY-MM-DD（留空 = 不限制）
        use_cache     False 時不管特徵 cache 現在是什麼狀態，一律重新計算
                      （只影響 train/train_xgb/train_lgbm，見 features.load_features()）

    備註：
        - breakout_signal（先跌後漲的破底翻型態）是模型的普通輸入特徵之一，
          不是訓練目標；它的相關驗證報表跟「要不要拆專門模型」的實驗都在
          strategy/rally/experiments/ 底下，還沒有定論，不是這裡的核心模式
        - 模型是全天訓練的（features.py 的 minutes_since_open 讓模型自己判斷時段），
          進場時段限制交給 backtest/intraday_backtest.py 的
          first_entry_time/last_entry_time 控制，不再鎖死在 make_features()/predict()
        - predict.py 的 predict_live(use_breakout_filter=True) 為即時推論強過濾入口
          （預設開啟），交易時段限制交給呼叫端（例如 live_trader.py 的
          SESSION_START/END），這裡不再額外鎖 9:14~9:30 黃金窗口
        - 直接執行本檔不會觸發即時下單，僅做訓練 / 驗證 / 報表
        - 股票母體固定 400 支 tick_universe（見檔頭說明），不是全市場
    """
    # ══════════════════════════════════════════════════════════════════════
    #  在這裡直接改 mode，不用每次打 CLI
    # ══════════════════════════════════════════════════════════════════════
    mode = "train_xgb"  # build_m3_m5,train_xgb,validate
    test_days = 30
    start_date = "2021-01-01"
    end_date = ""
    use_cache = True  # False 時不管 cache 狀態一律重新計算特徵

    main(
        mode=mode,
        test_days=test_days,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
    )
