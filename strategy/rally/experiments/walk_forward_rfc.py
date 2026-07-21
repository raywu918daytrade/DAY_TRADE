"""
Walk-forward驗證：RFC版，邏輯比照 strategy/rally/experiments/walk_forward_xgb.py
（訓練永遠在測試之前、expanding window、多窗口、同時掃 _THRESHOLDS 整組門檻）。

2026-07-21 背景：跟 XGB/LGBM 不同，RFC 從沒被調過參數（沒有 tune_rfc.py），
但 validate.py 的 model_hour_confidence_report() 顯示 RFC 目前三個模型裡表現
最好，尤其 9 點時段高信心度區間（0.60-0.70）precision 88.4%（單一測試月，
n=2392）特別突出。這支腳本主要用來確認：(1) RFC 整體表現在多個窗口下是否
穩定比單一個月更可信，(2) 之後如果要幫 RFC 調參，把 _CANDIDATE_PARAMS 換成
調參結果就能重複利用這支腳本，不用重寫。

⚠️ _CANDIDATE_PARAMS 目前是佔位用（先複製 _BASELINE_PARAMS，比照
strategy/mkt/experiments/walk_forward_lgbm.py 同樣做法）——RFC 還沒調過參數，
現在跑起來 baseline/candidate 兩欄會一樣，只是用來看 RFC 現有參數本身跨窗口
的穩定度。之後如果做了 RFC 調參，把候選參數貼到 _CANDIDATE_PARAMS 就能直接
比較。

用法：
    python strategy/rally/experiments/walk_forward_rfc.py
"""

import statistics
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from strategy.rally.experiments.tune_xgb import _precision_at_thresholds
from strategy.rally.features import FEATURES, load_features

# train.py::train() 現在實際在用的參數，當作比較基準。
_BASELINE_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=50,
)

# 佔位用，見檔頭說明——還沒調過參數，先複製 baseline。
_CANDIDATE_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=50,
)

# rally 的機率分布比 mkt 集中在較低區間（見 validate.py 的 coverage_report，
# RFC/XGB/LGBM 過了 0.65~0.70 之後訊號就很稀薄），門檻組取比較密集在這個範圍。
_THRESHOLDS = [0.45, 0.5, 0.55, 0.6, 0.65]


def _fit_rfc(train_df: pd.DataFrame, params: dict) -> RandomForestClassifier:
    model = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])
    return model


def _walk_forward_windows(df: pd.DataFrame, n_windows: int, window_days: int) -> list[tuple]:
    """從最新日期往回切 n_windows 個連續、不重疊的 window_days 天測試窗口，
    回傳依時間由舊到新排序的 (test_start, test_end) list。"""
    max_date = df["date"].max()
    windows = []
    for i in range(n_windows):
        test_end = max_date - pd.Timedelta(days=window_days * i)
        test_start = test_end - pd.Timedelta(days=window_days)
        windows.append((test_start, test_end))
    return list(reversed(windows))


def run(
    n_windows: int = 4,
    window_days: int = 30,
    min_train_days: int = 30,
    start_date: str = "2026-01-01",
    use_cache: bool = True,
    train_window_days: int | None = None,
):
    """train_window_days：見 walk_forward_xgb.py::run() 的說明，用法完全一致
    （預設None＝expanding訓練資料；設數字則每個窗口只留最近這麼多天）。"""
    df = load_features(use_cache=use_cache, start_date=start_date)
    df = df.dropna(subset=FEATURES + ["target"])
    windows = _walk_forward_windows(df, n_windows, window_days)

    results: dict[float, list[dict]] = {t: [] for t in _THRESHOLDS}

    for test_start, test_end in windows:
        train_df = df[df["date"] < test_start]
        if train_window_days is not None:
            train_start = test_start - pd.Timedelta(days=train_window_days)
            train_df = train_df[train_df["date"] >= train_start]
        test_df = df[(df["date"] >= test_start) & (df["date"] < test_end)]
        label = f"{test_start.date()}~{test_end.date()}"

        if train_df.empty or test_df.empty:
            print(f"[{label}] 跳過：訓練或測試資料為空")
            continue
        train_days_span = (train_df["date"].max() - train_df["date"].min()).days
        if train_days_span < min_train_days:
            print(f"[{label}] 跳過：訓練資料只橫跨{train_days_span}天，未達min_train_days={min_train_days}")
            continue

        baseline_model = _fit_rfc(train_df, _BASELINE_PARAMS)
        candidate_model = _fit_rfc(train_df, _CANDIDATE_PARAMS)

        base_by_thr = _precision_at_thresholds(baseline_model, test_df, _THRESHOLDS)
        cand_by_thr = _precision_at_thresholds(candidate_model, test_df, _THRESHOLDS)
        total_up = int((test_df["target"] == 1).sum())

        print(f"[{label}] 實際漲={total_up}")
        for thr in _THRESHOLDS:
            p_base, n_base = base_by_thr[thr]
            p_cand, n_cand = cand_by_thr[thr]
            results[thr].append(
                {
                    "window": label,
                    "actual_up": total_up,
                    "baseline_precision": p_base,
                    "baseline_n": n_base,
                    "candidate_precision": p_cand,
                    "candidate_n": n_cand,
                }
            )
            print(
                f"    threshold={thr:.2f}  舊參數: precision={p_base:>6.2%}(n={n_base:>4})  "
                f"新參數: precision={p_cand:>6.2%}(n={n_cand:>4})"
            )

    if not results[_THRESHOLDS[0]]:
        print("沒有任何窗口跑得動，檢查 n_windows/window_days/min_train_days 是否合理")
        return results

    print(f"\n=== 總結（{len(results[_THRESHOLDS[0]])}個窗口，各門檻分別統計） ===")
    for thr in _THRESHOLDS:
        rows = results[thr]
        base_p = [r["baseline_precision"] for r in rows]
        print(f"\n-- threshold={thr:.2f} --")
        print(f"precision： mean={statistics.mean(base_p):.2%}  std={statistics.pstdev(base_p):.2%}")
        print(f"各窗口: {[f'{p:.1%}' for p in base_p]}")

    return results


if __name__ == "__main__":
    run()
