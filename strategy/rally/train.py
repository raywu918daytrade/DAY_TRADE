"""
模型訓練 — RandomForest / XGBoost / LightGBM

三個模型共用 features.py 的 FEATURES 與 triple barrier 標籤，用同一份
_prepare_train_test() 切分早盤訓練/測試集，方便公平比較。
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.rally.config import BREAKOUT_TRADE_END, BREAKOUT_TRADE_START, SESSION_END, SESSION_START
from strategy.rally.features import FEATURES, load_features

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "models/m1_rfc.pkl"
_MODEL_PATH_XGB = _ROOT / "models/m1_xgb.pkl"
_MODEL_PATH_LGBM = _ROOT / "models/m1_lgbm_breakout.pkl"


def train(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    訓練 RandomForest 模型。

    Parameters
    ----------
    test_days : int
        以最後 N 天為測試集（預設 10）。若指定 start_date/end_date 則忽略此參數。
    start_date : str
        訓練資料起日，格式 "YYYY-MM-DD"。留空表示不限制。
    end_date : str
        訓練資料迄日，格式 "YYYY-MM-DD"。留空表示不限制。
    """
    print("特徵工程...")
    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵: {FEATURES}")

    # 只保留早盤時段
    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()
    print(f"  早盤有效樣本: {len(df):,} 筆")

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
        n_jobs=-1,
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
        raise FileNotFoundError(f"找不到模型，請先執行 train()")
    return joblib.load(_MODEL_PATH)


def load_model_xgb():
    if not _MODEL_PATH_XGB.exists():
        raise FileNotFoundError(f"找不到 XGB 模型，請先執行 train_xgb()")
    return joblib.load(_MODEL_PATH_XGB)


def load_model_lgbm():
    if not _MODEL_PATH_LGBM.exists():
        raise FileNotFoundError(f"找不到 LGBM 模型，請先執行 train_lgbm()")
    return joblib.load(_MODEL_PATH_LGBM)


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost / LightGBM 訓練（與 RFC 共用同一份 FEATURES 與標籤）
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare_train_test(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """載入特徵並切分早盤訓練/測試集（三模型共用）。"""
    print("特徵工程...")
    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  使用特徵數: {len(FEATURES)}")

    hhmm = df["hour"] * 100 + df["minute"]
    df = df[
        (hhmm >= SESSION_START[0] * 100 + SESSION_START[1]) & (hhmm <= SESSION_END[0] * 100 + SESSION_END[1])
    ].copy()
    print(f"  早盤有效樣本: {len(df):,} 筆")

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
):
    """訓練 XGBoost 模型（與 RFC 共用 FEATURES）。"""
    from xgboost import XGBClassifier

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_lambda=1.0,
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
):
    """訓練 LightGBM 模型（與 RFC 共用 FEATURES）。"""
    import lightgbm as lgb

    train_df, test_df = _prepare_train_test(test_days, start_date, end_date)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
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


def _breakout_scan(model, test_df: pd.DataFrame) -> dict:
    """對單一模型在破底翻黃金窗口做門檻掃描，回傳各門檻指標。"""
    sh, sm = BREAKOUT_TRADE_START
    eh, em = BREAKOUT_TRADE_END
    sub = test_df[test_df["breakout_signal"]].copy()
    mask = (sub["hour"] == sh) & (sub["minute"] >= sm) & (sub["hour"] == eh) & (sub["minute"] <= em)
    sub = sub[mask].copy()
    if sub.empty:
        return {}
    sub["proba"] = model.predict_proba(sub[FEATURES])[:, 1]
    total_pos = int(sub["target"].sum())
    out = {"total_pos": total_pos}
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        s = sub[sub["proba"] >= thr]
        tp = int((s["target"] == 1).sum()) if len(s) else 0
        out[thr] = {
            "n": len(s),
            "tp": tp,
            "win": s["target"].mean() if len(s) else float("nan"),
            "recall": tp / total_pos if total_pos else 0,
        }
    return out


def compare_breakout(
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
):
    """
    三模型（RFC / XGB / LGBM）在破底翻黃金窗口的門檻掃描對照。
    使用同一份測試集與同一組 FEATURES，公平比較。
    """
    models = {
        "RFC ": (load_model, _MODEL_PATH),
        "XGB ": (load_model_xgb, _MODEL_PATH_XGB),
        "LGBM": (load_model_lgbm, _MODEL_PATH_LGBM),
    }
    loaded = {}
    for name, (loader, path) in models.items():
        if not path.exists():
            print(f"  （跳過 {name.strip()}：模型不存在，請先 train_{name.strip().lower()}）")
            continue
        loaded[name] = loader()

    if not loaded:
        print("  無可用模型，請先訓練 RFC / XGB / LGBM。")
        return

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()
    print(f"  測試區間: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")

    scans = {name: _breakout_scan(m, test_df) for name, m in loaded.items()}

    print("\n── 破底翻黃金窗口 門檻掃描對照（9:14~9:30）──")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        print(f"\n  門檻 {thr:.2f}")
        print(f"    {'模型':<5} {'訊號數':>7} {'抓到漲':>7} {'勝率':>7} {'召回率':>7}")
        print("    " + "-" * 36)
        for name, sc in scans.items():
            if not sc or thr not in sc:
                continue
            r = sc[thr]
            print(f"    {name:<5} {r['n']:>7,} {r['tp']:>7,} {r['win']*100:>6.1f}% {r['recall']*100:>6.1f}%")
