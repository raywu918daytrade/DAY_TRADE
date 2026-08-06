"""
實驗：拆成兩個模型划不划算？——「破底翻專門模型」vs「通用模型」

背景
    strategy/rally 的通用模型（train.py 的 train_xgb() 等）用全天資料訓練，
    breakout_signal（先跌後漲的破底翻型態）只是 FEATURES 裡的其中一個特徵。
    即時交易時另外有 use_breakout_filter 這道硬規則，只在 breakout_signal=True
    才出訊號。

    這支實驗要驗證的假設：既然最終只會用到 breakout_signal=True 的訊號，
    那訓練時是不是該直接只用這部分樣本、訓練一個「專門」模型，而不是讓通用
    模型把 breakout_signal 當成 63 個特徵之一去學？

結論怎麼看
    跑 compare() 之後看兩個模型在同一份「breakout_signal=True + 黃金窗口
    9:14~9:30」測試集上的 AUC/勝率/召回率——如果專門模型沒有明顯贏過通用
    模型，代表樹模型本來就能透過 breakout_signal 這個特徵自己學會分岔，
    不需要真的拆成兩個模型維護（訓練/驗證/部署都要雙倍成本）。

跟 strategy/rally 核心檔案的關係
    這支檔案故意放在 experiments/ 底下、獨立於 train.py/validate.py 之外——
    這是「驗證一個假設，可能會證明沒必要」的一次性實驗，不是已經決定要長期
    維護的核心流程。結論如果是「有幫助」，才把邏輯正式搬進 train.py/
    validate.py；如果是「沒必要」，直接整個資料夾砍掉，核心檔案完全不受
    影響。

模型檔案存在 models/experiments/m1_xgb_breakout_only.pkl（跟正式模型
models/*.pkl 分開放，同樣道理：這是實驗產物，不是正式要上線的模型）。

用法
    python strategy/rally/experiments/breakout_specialist.py
"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from strategy.rally.features import FEATURES, load_features
from strategy.rally.train import _MODEL_PATH_XGB, load_model_xgb

_ROOT = Path(__file__).parent.parent.parent.parent
_MODEL_PATH_XGB_BREAKOUT_ONLY = _ROOT / "models/experiments/m1_xgb_breakout_only.pkl"

# 破底翻硬過濾的「黃金窗口」，只有這支跟 breakout_filter_eval.py 兩支實驗
# 腳本還在用（2026-08-06 從 config.py 搬過來，理由同 breakout_filter_eval.py
# 檔頭的說明）。
_BREAKOUT_TRADE_START = "9:14"
_BREAKOUT_TRADE_END = "9:30"


def _prepare_breakout_train_test(
    test_days: int = 10,
    hhmm_range: tuple[str, str] | None = (_BREAKOUT_TRADE_START, _BREAKOUT_TRADE_END),
):
    """只保留 breakout_signal=True（預設再限制在黃金窗口）的訓練/測試集。

    hhmm_range：一對 "H:MM" 字串（見本檔頭的 _BREAKOUT_TRADE_START/END 說明）。"""
    print("特徵工程...")
    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    df = df[df["breakout_signal"]]
    print(f"  只保留 breakout_signal=True 樣本")
    if hhmm_range is not None:
        start_str, end_str = hhmm_range
        sh, sm = (int(x) for x in start_str.split(":"))
        eh, em = (int(x) for x in end_str.split(":"))
        hhmm = df["hour"] * 100 + df["minute"]
        df = df[(hhmm >= sh * 100 + sm) & (hhmm <= eh * 100 + em)]
        print(f"  只保留 {sh}:{sm:02d}~{eh}:{em:02d} 樣本")
    print(f"  破底翻有效樣本: {len(df):,} 筆")

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


def train_breakout_only(
    test_days: int = 10,
    hhmm_range: tuple[str, str] | None = (_BREAKOUT_TRADE_START, _BREAKOUT_TRADE_END),
):
    """
    訓練「只用破底翻樣本」的專門 XGBoost 模型，跟 train.py::train_xgb() 用
    完全一樣的超參數、同一份 FEATURES，唯一差別是訓練/測試資料只保留
    breakout_signal=True（預設再限制在黃金窗口 9:14~9:30，破底翻訊號全天
    都算得出來，但真正有效的型態集中在開盤那段，中午之後的多半是雜訊）。
    """
    from xgboost import XGBClassifier

    train_df, test_df = _prepare_breakout_train_test(test_days, hhmm_range)

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

    _MODEL_PATH_XGB_BREAKOUT_ONLY.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _MODEL_PATH_XGB_BREAKOUT_ONLY)
    print(f"模型已存至 {_MODEL_PATH_XGB_BREAKOUT_ONLY}")
    return model


def load_model_breakout_only():
    if not _MODEL_PATH_XGB_BREAKOUT_ONLY.exists():
        raise FileNotFoundError("找不到破底翻專門模型，請先執行 train_breakout_only()")
    return joblib.load(_MODEL_PATH_XGB_BREAKOUT_ONLY)


def compare(test_days: int = 10):
    """
    通用 XGB（strategy/rally/train.py 的 train_xgb()）vs 破底翻專門 XGB
    （本檔 train_breakout_only()）在同一份「breakout_signal=True + 黃金窗口
    9:14~9:30」測試集上直接對打，看專門化到底有沒有幫助。
    """
    models = {}
    if _MODEL_PATH_XGB.exists():
        models["通用 XGB    "] = load_model_xgb()
    else:
        print("  （跳過 通用 XGB：模型不存在，請先跑 python -m strategy.rally.train train_xgb）")
    if _MODEL_PATH_XGB_BREAKOUT_ONLY.exists():
        models["破底翻專門 XGB"] = load_model_breakout_only()
    else:
        print("  （跳過 破底翻專門 XGB：模型不存在，請先執行 train_breakout_only()）")

    if not models:
        print("  無可用模型可比較")
        return

    df = load_features()
    df = df.dropna(subset=FEATURES + ["target"])
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff].copy()

    # 限制在跟專門模型訓練時同一個子集：breakout_signal=True + 黃金窗口
    test_df = test_df[test_df["breakout_signal"]]
    sh, sm = (int(x) for x in _BREAKOUT_TRADE_START.split(":"))
    eh, em = (int(x) for x in _BREAKOUT_TRADE_END.split(":"))
    mask = (test_df["hour"] == sh) & (test_df["minute"] >= sm) & (test_df["hour"] == eh) & (test_df["minute"] <= em)
    test_df = test_df[mask].copy()

    if test_df.empty:
        print(f"  該區間（{sh}:{sm:02d}~{eh}:{em:02d}，breakout_signal=True）無樣本")
        return

    total_pos = int(test_df["target"].sum())
    print(f"  測試集: {sh}:{sm:02d}~{eh}:{em:02d} + breakout_signal=True，共 {len(test_df):,} 筆，實際上漲 {total_pos:,} 筆")

    for name, model in models.items():
        proba = model.predict_proba(test_df[FEATURES])[:, 1]
        auc = roc_auc_score(test_df["target"], proba)
        print(f"\n  ── {name.strip()} ──  AUC={auc:.4f}")
        print(f"    {'門檻':>6}  {'訊號數':>7}  {'勝率':>6}  {'召回率':>6}")
        print("    " + "-" * 34)
        for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            flagged = proba >= thr
            n = int(flagged.sum())
            if n == 0:
                print(f"    {thr:.2f}  {0:>7,}  {'--':>6}  {'--':>6}")
                continue
            tp = int(test_df["target"][flagged].sum())
            win = tp / n
            recall = tp / total_pos if total_pos else 0
            print(f"    {thr:.2f}  {n:>7,}  {win*100:>5.1f}%  {recall*100:>5.1f}%")


if __name__ == "__main__":
    train_breakout_only(test_days=10)
    print()
    compare(test_days=10)
