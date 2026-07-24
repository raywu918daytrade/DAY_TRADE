"""
cnn 策略 — 多解析度 1D CNN（CLI 進入點）

== 策略重點 ==

不手工設計特徵，改把 5 種不同時間解析度的原始報價序列（m1/m3/m5 rolling、
m3_std/m5_std 獨立K棒）各自接一個 Conv1D 分支（見 model.py），讓網路自己從
多尺度原始序列裡學形狀。候選/label 比照 rally：全天每一根 m1 K 都是候選，
label 是3類 triple barrier（未來 HOLD_BARS 根先漲 TP_PCT% =漲、先跌 SL_PCT%
=跌、都沒碰到=平，2026-07-24從2類改成3類），不像 orb 那樣預先定義「事件
觸發」規則去篩候選——用 CNN 的目的就是讓網路自己找形狀，人工先篩事件等於
先用規則過濾掉可能學到的模式（rally 已實測過事件篩選的小樣本專門模型輸給
全樣本通用模型，見 rally/experiments/breakout_specialist.py）。

本檔比照 strategy/mkt/train.py 的慣例：argparse + __main__ 直接寫在這支檔案
自己身上，不另外開 entry.py 純轉發（cnn 是全新模組，直接照 mkt 這個較新、
較精簡的慣例走）。

本輪只做「最基本架構」——資料組裝 + 模型 + 訓練跑得通、印得出 test set 的
基本指標，驗證多分支設計本身可不可行。predict.py/live.py/backtest整合/
confidence報表都還沒做，等這裡先證明有訊號了再回頭補。

== Main 模式 ==

train             分chunk漸進訓練（固定筆數切成好幾個chunk，一次只載入一個
                  chunk訓練、練完立刻釋放），記憶體不會隨資料總量增加而長大，
                  範圍再大也不會OOM（2026-07-23新增、2026-07-24確認沒有明顯
                  缺點後取代掉原本「所有月份合併成一包」的舊版train）。
evaluate          讀已存模型，在 test set 上印 accuracy/AUC/混淆矩陣
evaluate_hours    跟 evaluate 一樣，但只看候選時間戳落在特定時段（例如
                  9~10點）的樣本（2026-07-24新增）。
confidence        掃描不同信心度門檻（P(跌)/P(漲)要多高才判斷方向，不是只看
                  argmax），驗證信心度高的預測是不是真的比較準，比照
                  rally/mkt的confidence_report慣例（2026-07-24新增）。
confidence_hours  跟 confidence 一樣，但只看候選時間戳落在特定時段（例如
                  9~10點）的樣本（2026-07-24新增）。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.cnn.config import DEVICE
from strategy.cnn.dataset import (
    BRANCH_NAMES,
    N_CLASSES,
    available_months,
    build_dataset,
    load_shard_branch,
    load_shard_meta,
)
from strategy.cnn.model import MultiScaleCNN

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "models/cnn_multiscale.pt"


class ShardedMultiScaleDataset(Dataset):
    """跨月份 shard 讀取的 torch Dataset。每個分支的 tensor 用
    np.load(mmap_mode='r') 開成 memmap（見 dataset.py 檔頭「為什麼按月分片存」
    的說明）——訓練時 DataLoader 每次只讀 __getitem__ 要的那幾筆，作業系統才會
    把對應那一小段從磁碟 page 進記憶體，不會把整個月份/整個資料集都常駐 RAM，
    半年甚至全量資料 train 階段才不會又 OOM 一次。

    row_filters：{month: 該月要保留的 local row index array}，由呼叫端
    （_split_data() 的 train/test 切分）決定，這裡不管切分邏輯。
    """

    def __init__(self, months: list[str], row_filters: dict[str, np.ndarray]):
        self._branch_arrays: dict[tuple[str, str], np.ndarray] = {}
        self._index: list[tuple[str, int]] = []
        meta_parts = []
        for month in months:
            keep = row_filters.get(month)
            if keep is None or len(keep) == 0:
                continue
            meta = load_shard_meta(month)
            for branch in BRANCH_NAMES:
                self._branch_arrays[(month, branch)] = load_shard_branch(month, branch, mmap=True)
            meta_parts.append(meta.iloc[keep])
            self._index.extend((month, int(i)) for i in keep)

        self.meta = (
            pd.concat(meta_parts, ignore_index=True)
            if meta_parts
            else pd.DataFrame(columns=["stock_id", "date", "target"])
        )
        self.target = self.meta["target"].to_numpy()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i):
        month, local_idx = self._index[i]
        x = {
            branch: torch.from_numpy(np.array(self._branch_arrays[(month, branch)][local_idx])).float()
            for branch in BRANCH_NAMES
        }
        y = torch.tensor(self.target[i], dtype=torch.long)  # CrossEntropyLoss 要class index（long），不是float
        return x, y


def _month_bound(date_str: str) -> str:
    """ "2026-07-15" → "2026_07"，跟 dataset.py 的月份shard key格式對齊。"""
    return f"{date_str[:4]}_{date_str[5:7]}"


def _resolve_months_and_cutoff(test_days: int, start_date: str, end_date: str, force_rebuild: bool):
    """共用邏輯：確保cache建好、篩出符合範圍的月份、算出test/train切分的
    cutoff日期。_load_test_ds()/_load_test_ds_hours()/train() 都靠這支
    共用，避免每個地方各自重寫一份月份篩選+cutoff計算邏輯。"""
    build_dataset(start_date=start_date, end_date=end_date, force_rebuild=force_rebuild)

    months = available_months()
    if start_date:
        months = [m for m in months if m >= _month_bound(start_date)]
    if end_date:
        months = [m for m in months if m <= _month_bound(end_date)]
    if not months:
        raise RuntimeError("cache/cnn/ 沒有可用的月份 shard，請確認 build_dataset() 有跑成功")
    months = sorted(months)

    metas = {m: load_shard_meta(m) for m in months}
    global_max_date = max(meta["date"].max() for meta in metas.values())
    cutoff = global_max_date - pd.Timedelta(days=test_days)
    return months, metas, cutoff


def _load_test_ds(
    test_days: int, start_date: str = "", end_date: str = "", force_rebuild: bool = False
) -> ShardedMultiScaleDataset:
    """只建立test_ds，不建train_ds——evaluate()專用。2026-07-23發現：原本
    evaluate()借用_split_data()，會連根本用不到的train_ds也一起建出來，範圍
    一大（例如18個月）單純評估一個模型也會踩到跟訓練一樣的記憶體風險。"""
    months, metas, cutoff = _resolve_months_and_cutoff(test_days, start_date, end_date, force_rebuild)
    test_filters = {m: np.nonzero(meta["date"].to_numpy() > cutoff)[0] for m, meta in metas.items()}
    return ShardedMultiScaleDataset(months, test_filters)


def _load_test_ds_hours(
    test_days: int,
    hour_start: int,
    hour_end: int,
    start_date: str = "",
    end_date: str = "",
    force_rebuild: bool = False,
) -> ShardedMultiScaleDataset:
    """跟 _load_test_ds() 一樣，但只保留候選時間戳落在 [hour_start, hour_end)
    的樣本（例如9~10點窗口），evaluate_hours() 專用（2026-07-24新增）——
    用來驗證模型在特定時段（例如最活躍的開盤前段）表現是否比整體平均更好。"""
    months, metas, cutoff = _resolve_months_and_cutoff(test_days, start_date, end_date, force_rebuild)
    test_filters = {}
    for m, meta in metas.items():
        dates = meta["date"]
        hour = dates.dt.hour
        mask = (dates.to_numpy() > cutoff) & (hour >= hour_start).to_numpy() & (hour < hour_end).to_numpy()
        test_filters[m] = np.nonzero(mask)[0]
    return ShardedMultiScaleDataset(months, test_filters)


def _class_weights(target: np.ndarray) -> torch.Tensor:
    """3類（跌/平/漲）的類別不平衡加權，跟rally/mkt樹模型的
    class_weight="balanced"是同樣的概念：weight_c = 總筆數 / (類別數 × 該類別筆數)，
    數量越少的類別權重越高，逼模型認真學稀有類別，不要偷懶只猜多數類別
    （2026-07-22實測：不加權時模型會學會「無腦猜多數」這個偷懶策略）。"""
    counts = np.array([(target == c).sum() for c in range(N_CLASSES)], dtype=np.float64)
    counts = np.maximum(counts, 1)  # 避免除以0
    weights = len(target) / (N_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def _build_row_chunks(
    months: list[str], metas: dict[str, pd.DataFrame], cutoff, chunk_size: int
) -> list[dict[str, np.ndarray]]:
    """把所有月份裡日期<=cutoff（訓練集部分）的列，依時間順序（月份順序、
    月內列順序）攤平成一條列表，每 chunk_size 筆切一個chunk（最後一個chunk
    可能不滿）。一個chunk可能橫跨2個以上的月份（例如某月份剩下30萬筆不夠湊
    滿100萬，就借下個月份的前70萬筆補滿），回傳的每個dict可以直接餵給
    ShardedMultiScaleDataset(months, row_filters) 建構子。

    2026-07-23討論：改成固定列數切、不照月份切的原因——每個月份實際筆數差
    滿多（例如2月106萬筆、6月233萬筆，差超過2倍），照月份切的話每個chunk的
    記憶體用量會忽大忽小；固定列數切可以讓每個chunk的大小更一致、更好預測。
    """
    flat: list[tuple[str, int]] = []
    for month in months:
        dates = metas[month]["date"].to_numpy()
        train_idx = np.nonzero(dates <= cutoff)[0]
        flat.extend((month, int(i)) for i in train_idx)

    chunks: list[dict[str, np.ndarray]] = []
    for start in range(0, len(flat), chunk_size):
        batch = flat[start : start + chunk_size]
        by_month: dict[str, list[int]] = {}
        for month, idx in batch:
            by_month.setdefault(month, []).append(idx)
        chunks.append({m: np.array(idxs, dtype=np.int64) for m, idxs in by_month.items()})
    return chunks


def train(
    test_days: int = 30,
    max_rounds: int = 10,
    chunk_size: int = 1_000_000,
    eval_every: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 3,
    start_date: str = "",
    end_date: str = "",
    force_rebuild: bool = False,
):
    """分批漸進訓練——不會把所有月份合併成一個大的 ShardedMultiScaleDataset。
    任何時間點記憶體裡只有「當下這一個chunk（固定 chunk_size 筆，可能橫跨
    多個月份）的訓練資料」+「固定的小測試集」，不會隨著資料總量增加而長大
    （2026-07-23實測：一次開全部月份memmap、shuffle打亂全範圍樣本的做法，
    7個月資料在24GB機器上就會逼近OOM邊緣，18個月以上幾乎必炸，2026-07-24
    確認分chunk版本沒有明顯缺點後直接取代掉那個版本，不再保留）。

    改成固定筆數切（不是按月份切）的原因：每個月份實際筆數差滿多（例如
    2月106萬筆、6月233萬筆，差超過2倍），按月切的話每個chunk記憶體用量會
    忽大忽小；固定筆數切能讓每個chunk大小更一致、更好預測（見
    _build_row_chunks() 的說明）。

    做法：外層跑最多 max_rounds 輪，每一輪依時間順序把訓練集切成的所有
    chunk依序各訓練1個epoch、訓練完立刻釋放該chunk資料，換下一個chunk；
    每 eval_every 個chunk才用固定測試集評估一次，記錄目前最佳checkpoint。

    eval_every（2026-07-24新增）：2026-07-24實測發現，對165萬筆固定測試集
    做一次推論本身要花不少時間，而原本「每個chunk都評估」等於這筆開銷被
    重複幾十次（一次19個月的訓練切成35個chunk，就要重複評估35次以上），
    很可能是訓練總耗時的主要來源。改成每eval_every個chunk才評估一次可以
    大幅省下這筆重複開銷；代價是「抓到的best checkpoint」可能不是絕對最佳
    點、而是附近稍微次佳的點（loss曲線通常連續變化，差異不大），且early
    stopping會晚一點觸發（中間跳過的幾個chunk如果剛好在變差，要等下一次
    評估點才會發現）——但因為存的一律是「目前為止最佳」的checkpoint，不是
    「最後一個」，最終存下來的模型品質不會受影響，只是可能多訓練了幾個
    無謂的chunk才停止（見討論）。

    ⚠️ Early stopping只在跑完「第一輪」（模型第一次看過完整時間範圍的廣度）
    之後才開始生效——第一輪進行中，模型可能只看過前面幾個chunk，評估結果
    本來就還不穩定、廣度也還不完整，不能拿來跟「一次性訓練epoch 1（本來就
    看過全部資料一遍）」相提並論，用來判斷提早停止並不公平（2026-07-23討論）。
    """
    months, metas, cutoff = _resolve_months_and_cutoff(test_days, start_date, end_date, force_rebuild)

    # 固定測試集：載入一次、全程留在記憶體（體積只跟test_days有關，不會隨
    # 訓練資料總量增加而變大），每個chunk訓練完都用同一組評估，才能公平比較。
    test_filters = {m: np.nonzero(meta["date"].to_numpy() > cutoff)[0] for m, meta in metas.items()}
    test_ds = ShardedMultiScaleDataset(months, test_filters)
    print(f"固定測試集: {len(test_ds):,} 筆（最後{test_days}天，全程不變）")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    chunks = _build_row_chunks(months, metas, cutoff, chunk_size)
    print(f"訓練集共切成 {len(chunks)} 個chunk（每個約{chunk_size:,}筆）")

    model = MultiScaleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 類別不平衡加權：跟 train() 一樣的概念，這裡用全部月份訓練集部分的
    # target分佈先算好，訓練過程中固定不變。
    train_targets_per_month = [meta["target"].to_numpy()[meta["date"].to_numpy() <= cutoff] for meta in metas.values()]
    all_train_targets = np.concatenate(train_targets_per_month)
    class_weights = _class_weights(all_train_targets)
    print(
        f"訓練集標籤數: {[(all_train_targets == c).sum() for c in range(N_CLASSES)]}，"
        f"class_weights={class_weights.tolist()}"
    )
    del train_targets_per_month, all_train_targets
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_test_loss = float("inf")
    best_state = None
    epochs_without_improve = 0
    stopped = False

    chunk_counter = 0
    for round_idx in range(1, max_rounds + 1):
        for chunk_idx, row_filters in enumerate(chunks, start=1):
            chunk_months = list(row_filters.keys())
            train_ds = ShardedMultiScaleDataset(chunk_months, row_filters)
            if len(train_ds) == 0:
                del train_ds
                continue
            # num_workers=0：2026-07-23實測，每個chunk開num_workers=2時，2個
            # worker process各自吃了12.45GB（比主程序10.33GB本身還重），根因
            # 還沒查清楚，先求穩不開worker。chunk本身量體不大（50~100萬筆），
            # 不像train()合併版要面對1300萬筆，I/O瓶頸的嚴重度應該小很多。
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

            train_t0 = time.time()
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x = {k: v.to(DEVICE) for k, v in x.items()}
                y = y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
            train_loss = total_loss / len(train_ds)
            n_train = len(train_ds)
            train_sec = time.time() - train_t0
            del train_loader, train_ds  # 釋放這個chunk的memmap參照，才能真正回收記憶體

            chunk_counter += 1
            if chunk_counter % eval_every != 0:
                # 跳過這次評估（見eval_every說明）：165萬筆測試集做一次推論本身
                # 不便宜，每個chunk都評估等於這筆開銷被重複幾十次，是訓練總
                # 耗時的主要來源之一。
                print(
                    f"round {round_idx}/{max_rounds}  chunk {chunk_idx}/{len(chunks)}  train_n={n_train:,}  "
                    f"train_loss={train_loss:.4f}  train_sec={train_sec:.1f}  (跳過評估，eval_every={eval_every})"
                )
                continue

            eval_t0 = time.time()
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x = {k: v.to(DEVICE) for k, v in x.items()}
                    y = y.to(DEVICE)
                    logits = model(x)
                    total_test_loss += criterion(logits, y).item() * len(y)
            test_loss = total_test_loss / len(test_ds)
            eval_sec = time.time() - eval_t0
            print(
                f"round {round_idx}/{max_rounds}  chunk {chunk_idx}/{len(chunks)}  train_n={n_train:,}  "
                f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  train_sec={train_sec:.1f}  eval_sec={eval_sec:.1f}"
            )

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            elif round_idx > 1:
                # 第一輪還沒看過全部資料的完整廣度，評估結果本來就不穩定，
                # 不拿來判斷提早停止（見函式說明）。
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(f"test_loss連續{patience}次沒有改善，第{round_idx}輪、chunk {chunk_idx}提早停止")
                    stopped = True
                    break
        if stopped:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}（best test_loss={best_test_loss:.4f}）")
    return model


def load_model() -> MultiScaleCNN:
    if not _MODEL_PATH.exists():
        raise FileNotFoundError("找不到模型，請先執行 train")
    model = MultiScaleCNN().to(DEVICE)
    model.load_state_dict(torch.load(_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def _run_inference_and_report(model: MultiScaleCNN, test_ds: ShardedMultiScaleDataset, batch_size: int) -> None:
    """對 test_ds 跑推論，印出 accuracy/AUC/混淆矩陣/分類報告——evaluate()跟
    evaluate_hours()共用同一份報表邏輯，只差在test_ds怎麼篩出來的。"""
    test_target = test_ds.target
    print(f"\n測試集: {len(test_target):,} 筆")
    if len(test_target) == 0:
        print("這個範圍沒有樣本可以評估。")
        return

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for x, _y in loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)  # (N, 3)
    y_pred = probs.argmax(axis=1)

    print(f"Accuracy: {accuracy_score(test_target, y_pred):.4f}")
    # multi_class="ovr"：3分類沒有單一條ROC曲線，改用one-vs-rest各類別分別算
    # AUC再平均，是sklearn處理多分類AUC的標準做法。
    print(f"AUC (one-vs-rest): {roc_auc_score(test_target, probs, multi_class='ovr', labels=[0, 1, 2]):.4f}")
    print("\n混淆矩陣（列=實際，欄=預測，順序 跌/平/漲）:")
    print(confusion_matrix(test_target, y_pred, labels=[0, 1, 2]))
    print("\n分類報告:")
    print(
        classification_report(test_target, y_pred, labels=[0, 1, 2], target_names=["跌", "平", "漲"], zero_division=0)
    )


def evaluate(test_days: int = 10, start_date: str = "", end_date: str = "", batch_size: int = 256):
    model = load_model()
    test_ds = _load_test_ds(test_days, start_date, end_date)
    _run_inference_and_report(model, test_ds, batch_size)


def evaluate_hours(
    hour_start: int = 9,
    hour_end: int = 10,
    test_days: int = 10,
    start_date: str = "",
    end_date: str = "",
    batch_size: int = 256,
):
    """只評估候選時間戳落在 [hour_start, hour_end) 的樣本（預設9~10點），
    2026-07-24新增——用來驗證模型在特定時段（例如當沖最活躍的開盤前段）
    表現是否比整體平均更好或更差，跟一開始討論LOOKBACK_MINUTES時想讓推論
    盡量涵蓋9-10點活躍時段的初衷呼應。"""
    model = load_model()
    test_ds = _load_test_ds_hours(test_days, hour_start, hour_end, start_date, end_date)
    print(f"時段篩選: {hour_start}:00~{hour_end}:00")
    _run_inference_and_report(model, test_ds, batch_size)


def _predict_with_threshold(probs: np.ndarray, threshold: float | None) -> np.ndarray:
    """threshold=None 用 argmax（機率最高的類別勝出，沒有信心度門檻概念）；
    設 0~1 的數字時，改成「信心度夠了才判跌/漲，不然算平」的門檻式判法——
    P(跌)>=threshold 判跌、P(漲)>=threshold 判漲（兩個都過門檻取機率較高的
    那個），否則一律判平（class 1）。跟 strategy/mkt/train.py 的
    _predict_with_threshold() 同樣的概念（2026-07-24新增）。"""
    if threshold is None:
        return probs.argmax(axis=1)

    p_down = probs[:, 0]
    p_up = probs[:, 2]
    y_pred = np.ones(len(probs), dtype=int)  # 預設判平
    down_pass = p_down >= threshold
    up_pass = p_up >= threshold
    y_pred[down_pass & ~up_pass] = 0
    y_pred[up_pass & ~down_pass] = 2
    both = down_pass & up_pass
    y_pred[both] = np.where(p_up[both] >= p_down[both], 2, 0)
    return y_pred


_DEFAULT_THRESHOLDS: list[float | None] = [None, 0.4, 0.5, 0.6, 0.7, 0.8]


def confidence_report(
    test_days: int = 10,
    thresholds: list[float | None] | None = None,
    hour_start: int | None = None,
    hour_end: int | None = None,
    start_date: str = "",
    end_date: str = "",
    batch_size: int = 256,
):
    """掃描不同信心度門檻，看precision/recall/覆蓋率會不會隨門檻拉高而變好
    ——驗證「模型自己覺得有把握的那些預測，是不是真的比準argmax（無門檻）
    的預測還準」，比照 rally/mkt 的 confidence_report 慣例（2026-07-24新增）。
    只跑一次推論、算好機率矩陣後，每個門檻只是重新篩選、不用重跑模型。

    hour_start/hour_end 都給定時（2026-07-24新增），只評估候選時間戳落在
    [hour_start, hour_end) 的樣本，跟 evaluate_hours() 同樣的篩選邏輯；
    留 None（預設）就是全天，不篩時段。"""
    thresholds = thresholds if thresholds is not None else _DEFAULT_THRESHOLDS

    model = load_model()
    if hour_start is not None and hour_end is not None:
        test_ds = _load_test_ds_hours(test_days, hour_start, hour_end, start_date, end_date)
        print(f"時段篩選: {hour_start}:00~{hour_end}:00")
    else:
        test_ds = _load_test_ds(test_days, start_date, end_date)
    test_target = test_ds.target
    print(f"\n測試集: {len(test_target):,} 筆")
    if len(test_target) == 0:
        print("測試集沒有樣本可以評估。")
        return

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for x, _y in loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            p = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(p)
    probs = np.concatenate(all_probs)

    for threshold in thresholds:
        y_pred = _predict_with_threshold(probs, threshold)
        label = "argmax（無門檻）" if threshold is None else f"信心度門檻={threshold:.2f}"
        coverage = (y_pred != 1).mean()  # 判斷成跌/漲（不是平）的比例
        print(f"\n── {label}，判斷覆蓋率(非平)={coverage:.4f} ──")
        print(f"Accuracy: {accuracy_score(test_target, y_pred):.4f}")
        print(confusion_matrix(test_target, y_pred, labels=[0, 1, 2]))
        print(
            classification_report(
                test_target, y_pred, labels=[0, 1, 2], target_names=["跌", "平", "漲"], zero_division=0
            )
        )


def main(
    mode: str = "",
    test_days: int = 30,
    max_rounds: int = 10,
    chunk_size: int = 1_000_000,
    eval_every: int = 1,
    hour_start: int = 9,
    hour_end: int = 10,
    thresholds: list[float] | None = None,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 3,
    start_date: str = "",
    end_date: str = "",
    force_rebuild: bool = False,
):
    """cnn 策略主程式。比照 strategy/mkt/train.py 的兩種執行方式（sys.argv
    長度自動判斷用哪一種，兩者不會互相打架）：

      1. VS Code按F5：直接改下面 __main__ 裡的變數，不用打字。
      2. 終端機帶參數：python -m strategy.cnn.train train --test_days 30

    start_date/end_date：限制資料範圍（YYYY-MM-DD），第一次跑建議先限縮
    月份範圍，避免全量22個月資料第一次就跑很久（見 dataset.py 的效能備註）。

    train 依固定筆數（chunk_size）把資料切成好幾個chunk、一次只載入一個chunk
    訓練、練完立刻釋放，記憶體不會隨資料總量增加而長大（2026-07-24：原本
    另外有一個「一整包訓練」的版本，範圍一大就容易在24GB機器上OOM，實測後
    確認分chunk版本沒有明顯缺點就直接取代掉，只留這一個）。

    evaluate_hours：跟 evaluate 一樣讀已存模型評估，但只看候選時間戳落在
    [hour_start, hour_end) 的樣本（預設9~10點，2026-07-24新增）。

    confidence：掃描不同信心度門檻（P(跌)/P(漲)要多高才判斷方向，不是只看
    argmax），驗證信心度高的預測是不是真的比較準（2026-07-24新增）。
    """
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="cnn 策略 — 多解析度 1D CNN")
        parser.add_argument(
            "mode",
            nargs="?",
            default="train",
            choices=["train", "evaluate", "evaluate_hours", "confidence", "confidence_hours"],
            help="執行模式（預設train）",
        )
        parser.add_argument("--test_days", type=int, default=30, help="測試集天數")
        parser.add_argument("--max_rounds", type=int, default=10, help="最多跑幾輪全部資料（只影響train）")
        parser.add_argument("--chunk_size", type=int, default=1_000_000, help="每個chunk的筆數（只影響train）")
        parser.add_argument(
            "--eval_every",
            type=int,
            default=1,
            help="每幾個chunk才評估一次（只影響train，預設1=每個都評估）",
        )
        parser.add_argument("--hour_start", type=int, default=9, help="只影響evaluate_hours：起始小時（含）")
        parser.add_argument("--hour_end", type=int, default=10, help="只影響evaluate_hours：結束小時（不含）")
        parser.add_argument(
            "--thresholds",
            type=float,
            nargs="*",
            default=None,
            help="只影響confidence：信心度門檻清單（例：--thresholds 0.5 0.6 0.7），留空用預設清單",
        )
        parser.add_argument("--batch_size", type=int, default=256, help="batch size")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate（只影響train）")
        parser.add_argument(
            "--patience",
            type=int,
            default=3,
            help="early stopping耐心值：test_loss連續幾次沒改善就停止（只影響train）",
        )
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        parser.add_argument("--force_rebuild", action="store_true", help="略過cache新鮮度檢查，強制重算tensor")
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        max_rounds = args.max_rounds
        chunk_size = args.chunk_size
        eval_every = args.eval_every
        hour_start = args.hour_start
        hour_end = args.hour_end
        thresholds = args.thresholds
        batch_size = args.batch_size
        lr = args.lr
        patience = args.patience
        start_date = args.start_date
        end_date = args.end_date
        force_rebuild = args.force_rebuild
    elif not mode:
        mode = "train"

    if mode == "train":
        train(
            test_days=test_days,
            max_rounds=max_rounds,
            chunk_size=chunk_size,
            eval_every=eval_every,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            start_date=start_date,
            end_date=end_date,
            force_rebuild=force_rebuild,
        )
    elif mode == "evaluate":
        evaluate(test_days=test_days, start_date=start_date, end_date=end_date, batch_size=batch_size)
    elif mode == "evaluate_hours":
        evaluate_hours(
            hour_start=hour_start,
            hour_end=hour_end,
            test_days=test_days,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
        )
    elif mode == "confidence":
        confidence_report(
            test_days=test_days,
            thresholds=thresholds if thresholds else None,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
        )
    elif mode == "confidence_hours":
        confidence_report(
            test_days=test_days,
            thresholds=thresholds if thresholds else None,
            hour_start=hour_start,
            hour_end=hour_end,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
        )
    else:
        print(f"未知模式: {mode}，可用: train / evaluate / evaluate_hours / confidence / confidence_hours")


if __name__ == "__main__":
    # ══════════════════════════════════════════════════════════════════════
    #  VS Code按F5：在這裡直接改變數，不用每次打 CLI
    # ══════════════════════════════════════════════════════════════════════
    mode = "confidence"  # train / evaluate / evaluate_hours / confidence
    test_days = 30
    max_rounds = 10  # 最多跑幾輪全部資料
    chunk_size = 1_000_000  # 每個chunk的筆數
    eval_every = 1  # 每幾個chunk評估一次（1=每個都評估）
    hour_start = 9  # 只影響 evaluate_hours
    hour_end = 10  # 只影響 evaluate_hours
    thresholds = None  # 只影響 confidence，留None用預設清單[None,0.4,0.5,0.6,0.7,0.8]
    batch_size = 256
    lr = 1e-3
    patience = 3
    start_date = "2025-01-01"  # 先限縮範圍測試基本架構跑不跑得通，之後再放寬
    end_date = ""
    force_rebuild = False

    main(
        mode=mode,
        test_days=test_days,
        max_rounds=max_rounds,
        chunk_size=chunk_size,
        eval_every=eval_every,
        hour_start=hour_start,
        hour_end=hour_end,
        thresholds=thresholds,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        start_date=start_date,
        end_date=end_date,
        force_rebuild=force_rebuild,
    )
