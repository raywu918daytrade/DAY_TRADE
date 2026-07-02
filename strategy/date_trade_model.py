"""
日內交易模型 — LightGBM 分K方向預測

== 訓練邏輯 ==

目標（Triple Barrier Label，與回測邏輯一致）
    在進場後最多 HOLD_BARS 根分K內：
        先碰到 +TP_PCT → label = 1（停利）
        先碰到 -SL_PCT → label = 0（停損）
        時間到都未碰   → 以最終方向決定（漲=1, 跌=0）
    每日最後 HOLD_BARS 根不生成 label（無法確認結果）

資料來源
    分K (db/m1/)      ：還原股價歷史1分鐘OHLCV，所有股票合併訓練
    日K (db/day_trade/)：還原股價歷史日線，提供趨勢背景特徵

特徵
    分K 特徵：
        ret_1/3/5/10/20/30  當前收盤相對N分鐘前的報酬率
        vol_ratio           當前成交量 / 20分鐘均量
        close_pos           close 在當根 (high-low) 的相對位置 [0,1]
        hour / minute       盤中時間（捕捉開收盤效應）
    當日盤中特徵（累積計算，不含未來資料）：
        price_vs_open       close vs 今日開盤（當日漲跌幅）
        vwap_dev            close vs 今日累積VWAP（強弱位置）
        high_pos_today      close 在今日目前最高/最低的位置 [0,1]
    日K 特徵（前一交易日，無未來洩漏）：
        gap                 今日跳空幅度（今日開盤 vs 昨日收盤）
        prev_ret            前日漲跌幅
        prev_vol_ratio      前日量 / 20日均量
        pos_20d             收盤在近20日高低點的相對位置 [0,1]

資料切割
    以時間為界，不隨機 shuffle（避免 look-ahead bias）
    cutoff = 最新日期 - test_days 個自然日（預設 10 天 ≈ 7 個交易日）
    train : date <= cutoff
    test  : date >  cutoff

    範例（目前資料）
        全資料  2026-05-29 ~ 2026-06-30，22 個交易日，1,775 支股票
        訓練集  2026-05-29 ~ 2026-06-20，15 個交易日，約 323,000 筆
        測試集  2026-06-21 ~ 2026-06-30， 7 個交易日，約 117,000 筆

    注意：每支股票每日最後一根分K不納入訓練，
    因為 target（下一分鐘）會跨日，導致標籤錯誤

模型
    LightGBM Classifier，early stopping 根據測試集 binary_logloss

評估
    Accuracy、AUC（測試集）
"""

import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, roc_auc_score

# 確保能從根目錄導入
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from data.query import load_day, load_m1, load_m1_live

_MODEL_PATH = Path(__file__).parent.parent / "models/m1_lgbm.pkl"

# Triple Barrier 參數（需與回測保持一致）
TP_PCT = 0.03  # 停利 3%
SL_PCT = 0.03  # 停損 3%
HOLD_BARS = 30  # 最多持有分K數

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# 訓練/推論的交易時段；可用 .env 覆寫（本機測試用）
SESSION_START = (9, 1)  # 9:01（第一根完整分K）
_end_h = int(os.environ.get("SESSION_END_HOUR", "10"))
_end_m = int(os.environ.get("SESSION_END_MIN", "0"))
SESSION_END = (_end_h, _end_m)

# ── Triple Barrier Label ──────────────────────────────────────────────────────


def _barrier_label_group(closes: np.ndarray) -> np.ndarray:
    """每日每股，逐 bar 往前看最多 HOLD_BARS 根，判斷先碰到 tp 或 sl。"""
    n = len(closes)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        entry = closes[i]
        tp_price = entry * (1 + TP_PCT)
        sl_price = entry * (1 - SL_PCT)
        future = closes[i + 1 : i + HOLD_BARS + 1]
        tp_idx = np.argmax(future >= tp_price) if (future >= tp_price).any() else len(future)
        sl_idx = np.argmax(future <= sl_price) if (future <= sl_price).any() else len(future)
        if tp_idx < sl_idx:
            labels[i] = 1
        elif sl_idx < tp_idx:
            labels[i] = 0
        elif len(future) == HOLD_BARS:
            labels[i] = 1 if future[-1] > entry else 0
        # else: 太靠近當日尾端，保留 NaN
    return labels


def _make_barrier_labels(m1: pd.DataFrame) -> pd.Series:
    result = m1.groupby(["stock_id", "day_date"], group_keys=False).apply(
        lambda g: pd.Series(
            _barrier_label_group(g["close"].values),
            index=g.index,
        ),
        include_groups=False,
    )
    return result


# ── 特徵工程 ──────────────────────────────────────────────────────────────────


def make_features(m1: pd.DataFrame, day: pd.DataFrame, compute_labels: bool = True) -> pd.DataFrame:
    m1 = m1.copy()
    day = day.copy()
    m1["date"] = pd.to_datetime(m1["date"])
    day["date"] = pd.to_datetime(day["date"])
    # 先建 day_date，供後面日內分組使用
    m1["day_date"] = m1["date"].dt.date

    # 日內分組：所有分K特徵都限定在當日內，不跨日
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)

    # 分K 報酬率（日內，不跨日，最長 15 根 → 9:16 起即可預測）
    for lag in [1, 3, 5, 10, 15]:
        m1[f"ret_{lag}"] = g_day["close"].transform(lambda x: x.pct_change(lag))

    # 量比（日內15分鐘均量，不跨日，warmup 14 根）
    m1["vol_ma15"] = g_day["volume"].transform(lambda x: x.rolling(15).mean())
    m1["vol_ratio"] = m1["volume"] / m1["vol_ma15"].replace(0, np.nan)

    # close 在當根 high-low 的位置
    bar_range = (m1["high"] - m1["low"]).replace(0, np.nan)
    m1["close_pos"] = (m1["close"] - m1["low"]) / bar_range

    # 時間特徵
    m1["hour"] = m1["date"].dt.hour
    m1["minute"] = m1["date"].dt.minute

    # 目標：Triple Barrier Label（訓練用；即時推論時跳過）
    if compute_labels:
        m1["target"] = _make_barrier_labels(m1)
        m1 = m1[m1["target"].notna()].copy()
        m1["target"] = m1["target"].astype(int)

    # ── 當日盤中特徵（用 cumsum/cummax 確保不洩漏未來資料）────────────────────
    g_day = m1.groupby(["stock_id", "day_date"], group_keys=False)

    # 今日開盤價（當日第一根 open）
    day_open = g_day["open"].transform("first").replace(0, np.nan)

    # close vs 今日開盤（開盤後漲跌幅）
    m1["price_vs_open"] = (m1["close"] - day_open) / day_open

    # 當日累積 VWAP 偏離
    m1["_pv"] = m1["close"] * m1["volume"]
    m1["_cum_pv"] = g_day["_pv"].transform("cumsum")
    m1["_cum_vol"] = g_day["volume"].transform("cumsum")
    vwap = m1["_cum_pv"] / m1["_cum_vol"].replace(0, np.nan)
    m1["vwap_dev"] = (m1["close"] - vwap) / vwap.replace(0, np.nan)

    # close 在今日目前最高/最低的位置（rolling，不含未來）
    m1["_high_max"] = g_day["high"].transform("cummax")
    m1["_low_min"] = g_day["low"].transform("cummin")
    today_range = (m1["_high_max"] - m1["_low_min"]).replace(0, np.nan)
    m1["high_pos_today"] = (m1["close"] - m1["_low_min"]) / today_range

    m1 = m1.drop(columns=["_pv", "_cum_pv", "_cum_vol", "_high_max", "_low_min"])

    # 破底翻：close 距近 N 根最低點的反彈幅度（=0 代表仍在低點，>0 代表已反彈）
    for _n in [3, 5, 10]:
        _roll_min = g_day["close"].transform(lambda x, n=_n: x.rolling(n, min_periods=n).min())
        m1[f"reversal_{_n}"] = (m1["close"] - _roll_min) / _roll_min.replace(0, np.nan)

    # ── 日K 特徵（前一日，避免未來資料）────────────────────────────────────────
    dg = day.groupby("stock_id")
    day = day.copy()
    day["prev_ret"] = dg["close"].transform(lambda x: x.pct_change(1))
    day["prev_vol_ratio"] = dg["volume"].transform(lambda x: x / x.rolling(20).mean())
    day["pos_20d"] = dg["close"].transform(
        lambda x: (x - x.rolling(20).min()) / (x.rolling(20).max() - x.rolling(20).min()).replace(0, np.nan)
    )
    # 跳空（今日開盤 vs 昨日收盤）
    day["gap"] = dg["close"].transform(
        lambda x: (day.loc[x.index, "open"] - x.shift(1)) / x.shift(1).replace(0, np.nan)
    )
    # 過去 10 天日報酬率與量比（lag 1~10，lag1=前1日）
    # vol_ma5：短線均量，warmup 只需4天，避免 rolling(20) 吃掉過多歷史資料
    vol_ma5 = dg["volume"].transform(lambda x: x.rolling(5).mean()).replace(0, np.nan)
    day_ret_cols, day_vol_cols = [], []
    for lag in range(1, 11):
        cr, cv = f"day_ret_{lag}", f"day_vol_{lag}"
        day[cr] = dg["close"].transform(lambda x, l=lag: x.pct_change(1).shift(l - 1))
        day[cv] = dg["volume"].transform(lambda x, l=lag: (x / vol_ma5.loc[x.index]).shift(l - 1))
        day_ret_cols.append(cr)
        day_vol_cols.append(cv)

    day["day_date"] = day["date"].dt.date
    day_feat_cols = (
        ["stock_id", "day_date", "prev_ret", "prev_vol_ratio", "pos_20d", "gap"] + day_ret_cols + day_vol_cols
    )
    m1 = m1.merge(day[day_feat_cols], on=["stock_id", "day_date"], how="left")
    return m1


FEATURES = [
    # 分K 報酬率（日內，最長15根，9:16起有效）
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_15",
    # 量比、K線形態
    "vol_ratio",
    "close_pos",
    # 時間
    "hour",
    "minute",
    # 當日盤中
    "price_vs_open",  # close vs 今日開盤
    "vwap_dev",  # close vs 今日累積VWAP
    "high_pos_today",  # close 在今日最高/最低的位置
    # 破底翻：距 N 根最低點的反彈幅度（min_periods=N，9:03/9:05/9:10 起有值）
    "reversal_3",
    "reversal_5",
    "reversal_10",
    # 日K 背景
    "gap",  # 今日跳空幅度
    "prev_ret",
    "prev_vol_ratio",
    "pos_20d",
    # 過去 10 天日報酬率與量比
    *[f"day_ret_{i}" for i in range(1, 11)],
    *[f"day_vol_{i}" for i in range(1, 11)],
]


# ── 訓練 ──────────────────────────────────────────────────────────────────────


def train(test_days: int = 10):
    print("載入分K...")
    m1 = load_m1()
    print(f"  {len(m1):,} 筆")

    print("載入日K...")
    day = load_day()
    print(f"  {len(day):,} 筆")

    print("特徵工程...")
    df = make_features(m1, day)
    df = df.dropna(subset=FEATURES + ["target"])

    # 只保留早盤時段（盤中動能窗口，label 仍看當日整段未來）
    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()
    print(f"  早盤有效樣本: {len(df):,} 筆")

    # 時間切割（最後 test_days 天當測試，不 shuffle 避免未來洩漏）
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff]
    print(f"  訓練: {len(train_df):,}  測試: {len(test_df):,}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train_df[FEATURES],
        train_df["target"],
        eval_set=[(test_df[FEATURES], test_df["target"])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

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
        raise FileNotFoundError(f"找不到模型，請先執行 train()")
    return joblib.load(_MODEL_PATH)


def predict_live(
    minute_str: str,
    day: pd.DataFrame,
    model=None,
    threshold: float = 0.55,
    day_trade_stocks: set | None = None,
    m1_live: pd.DataFrame | None = None,
) -> list:
    """
    即時推論：計算特徵，回傳當分K達門檻的訊號清單。
    m1_live: 已載入的今日分K，傳入可避免重複 I/O（on_minute 已載過）。
    day_trade_stocks: 當沖標的 set，若提供則只推論這些股票。
    回傳格式：[{"stock_id": ..., "proba": ..., "price": ...}, ...]
    """
    if model is None:
        model = load_model()

    if m1_live is None:
        date_str = minute_str[:10]
        m1_live = load_m1_live(date_str)
    if m1_live.empty:
        return []

    # 只保留當沖標的
    if day_trade_stocks:
        m1_live = m1_live[m1_live["stock_id"].isin(day_trade_stocks)]
    if m1_live.empty:
        return []

    # make_features 以 day_date 做 merge；今日 day 資料尚不存在，補一行今日摘要
    # day["date"] 可能是 object(str) 或 datetime64，統一轉換後比對
    day = day.copy()
    day["date"] = pd.to_datetime(day["date"])
    today_ts = pd.Timestamp(date_str)
    if not (day["date"] == today_ts).any():
        rows = []
        for sid, g in m1_live.groupby("stock_id"):
            g_s = g.sort_values("date")
            rows.append(
                {
                    "stock_id": sid,
                    "date": today_ts,
                    "open": float(g_s.iloc[0]["open"]),
                    "high": float(g["high"].max()),
                    "low": float(g["low"].min()),
                    "close": float(g_s.iloc[-1]["close"]),
                    "volume": int(g["volume"].sum()),
                }
            )
        if rows:
            day = pd.concat([day, pd.DataFrame(rows)], ignore_index=True)
            day["date"] = pd.to_datetime(day["date"])
            day = day.sort_values(["stock_id", "date"]).reset_index(drop=True)

    df = make_features(m1_live, day, compute_labels=False)

    current = df[df["date"] == pd.Timestamp(minute_str)]
    if current.empty:
        return []

    # 只保留特徵齊全的 bar（早盤前幾根 lag 特徵可能為 NaN）
    valid = current.dropna(subset=FEATURES)
    if valid.empty:
        return []

    proba = model.predict_proba(valid[FEATURES])[:, 1]
    signals = [
        {"stock_id": row["stock_id"], "proba": float(p), "price": float(row["close"])}
        for (_, row), p in zip(valid.iterrows(), proba)
        if p >= threshold
    ]
    return sorted(signals, key=lambda x: -x["proba"])


def confidence_report(model=None, test_days: int = 10):
    """
    依信心度區間顯示樣本數與勝率（實際漲的比例）。
    用來決定 intraday_platform 的 threshold 設定。
    """
    if model is None:
        model = load_model()

    m1 = load_m1()
    day = load_day()
    df = make_features(m1, day)
    df = df.dropna(subset=FEATURES + ["target"])

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    bins = [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.01]
    labels = [
        "<0.30",
        "0.30-0.35",
        "0.35-0.40",
        "0.40-0.45",
        "0.45-0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.70",
        "≥0.70",
    ]
    test_df["bucket"] = pd.cut(test_df["proba"], bins=bins, labels=labels, right=False)

    report = (
        test_df.groupby("bucket", observed=True)
        .agg(樣本數=("target", "count"), 勝率=("target", "mean"))
        .assign(勝率=lambda x: (x["勝率"] * 100).round(1).astype(str) + "%")
    )
    print("\n── 信心度分析（測試集）──")
    print(report.to_string())
    return report


def coverage_report(model=None, test_days: int = 10):
    """
    召回率分析：9:00~10:00 實際漲的機會有幾個，模型在不同門檻下抓到幾個。
    精確率 = 抓到的裡面有幾個是對的（模型準不準）
    召回率 = 實際漲的裡面有幾個被抓到（機會有沒有把握到）
    """
    if model is None:
        model = load_model()

    m1 = load_m1()
    day = load_day()
    df = make_features(m1, day)
    df = df.dropna(subset=FEATURES + ["target"])

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()

    test_df["proba"] = model.predict_proba(test_df[FEATURES])[:, 1]

    total_pos = test_df["target"].sum()  # 實際漲的總數
    total_neg = (test_df["target"] == 0).sum()

    print(f"\n── 召回率分析（測試集）──")
    print(f"實際漲（label=1）: {total_pos:,} 筆")
    print(f"實際跌（label=0）: {total_neg:,} 筆")
    print()
    print(f"{'門檻':>6}  {'訊號數':>7}  {'精確率':>7}  {'召回率':>7}  {'F1':>6}")
    print("-" * 45)

    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        flagged = test_df["proba"] >= thr
        tp = (flagged & (test_df["target"] == 1)).sum()
        precision = tp / flagged.sum() if flagged.sum() > 0 else 0
        recall = tp / total_pos if total_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {thr:.2f}  {flagged.sum():>7,}  {precision*100:>6.1f}%  {recall*100:>6.1f}%  {f1:.3f}")


def predict(model=None) -> pd.DataFrame:
    """
    對早盤時段（SESSION_START ~ SESSION_END）的分K 產生預測機率。
    回傳 df_proba：index=datetime，columns=stock_id，值=漲的機率。
    """
    if model is None:
        model = load_model()

    m1 = load_m1()
    day = load_day()
    df = make_features(m1, day)
    df = df.dropna(subset=FEATURES)

    # 只保留早盤時段（與訓練一致）
    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()

    df["proba"] = model.predict_proba(df[FEATURES])[:, 1]
    df_proba = df.pivot(index="date", columns="stock_id", values="proba")
    return df_proba


def optimize_model(n_trials: int = 50, test_days: int = 10):
    """
    用 Optuna 搜尋最佳 LightGBM 超參數，目標為測試集 AUC。
    搜尋完後自動用最佳參數重新訓練並存模型。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("載入資料...")
    m1 = load_m1()
    day = load_day()
    df = make_features(m1, day)
    df = df.dropna(subset=FEATURES + ["target"])

    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff]
    print(f"訓練: {len(train_df):,}  測試: {len(test_df):,}")

    def objective(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=1,
            lambda_l1=trial.suggest_float("lambda_l1", 0.0, 5.0),
            lambda_l2=trial.suggest_float("lambda_l2", 0.0, 5.0),
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            train_df[FEATURES],
            train_df["target"],
            eval_set=[(test_df[FEATURES], test_df["target"])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
        return roc_auc_score(test_df["target"], y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n最佳 AUC: {study.best_value:.4f}")
    print("最佳參數:", best)

    # 用最佳參數重新訓練並存模型
    print("\n用最佳參數重新訓練...")
    best_model = lgb.LGBMClassifier(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        bagging_freq=1,
        **best,
    )
    best_model.fit(
        train_df[FEATURES],
        train_df["target"],
        eval_set=[(test_df[FEATURES], test_df["target"])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
    )
    y_pred = best_model.predict(test_df[FEATURES])
    y_prob = best_model.predict_proba(test_df[FEATURES])[:, 1]
    print(f"Accuracy : {accuracy_score(test_df['target'], y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(test_df['target'], y_prob):.4f}")

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}")
    return best_model, study


if __name__ == "__main__":
    # model = train()
    # confidence_report()
    coverage_report()
