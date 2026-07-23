"""
cnn 策略 — 多解析度 1D CNN（CLI 進入點）

== 策略重點 ==

不手工設計特徵，改把 5 種不同時間解析度的原始報價序列（m1/m3/m5 rolling、
m3_std/m5_std 獨立K棒）各自接一個 Conv1D 分支（見 model.py），讓網路自己從
多尺度原始序列裡學形狀。候選/label 比照 rally：全天每一根 m1 K 都是候選，
label 是通用 triple barrier（未來 HOLD_BARS 根先漲 TP_PCT% 還是跌 SL_PCT%），
不像 orb 那樣預先定義「事件觸發」規則去篩候選——用 CNN 的目的就是讓網路自己
找形狀，人工先篩事件等於先用規則過濾掉可能學到的模式（rally 已實測過事件
篩選的小樣本專門模型輸給全樣本通用模型，見 rally/experiments/breakout_specialist.py）。

本檔比照 strategy/mkt/train.py 的慣例：argparse + __main__ 直接寫在這支檔案
自己身上，不另外開 entry.py 純轉發（cnn 是全新模組，直接照 mkt 這個較新、
較精簡的慣例走）。

本輪只做「最基本架構」——資料組裝 + 模型 + 訓練跑得通、印得出 test set 的
基本指標，驗證多分支設計本身可不可行。predict.py/live.py/backtest整合/
confidence報表都還沒做，等這裡先證明有訊號了再回頭補。

== Main 模式 ==

train             一整包訓練（所有月份合併成一個資料集）。範圍小（1~2個月）
                  適用，範圍大（7個月以上）在24GB機器上容易OOM，見
                  train_sequential 的說明。
train_sequential  分月漸進訓練，同一時間只有一個月的資料在記憶體裡，不會
                  隨月份數增加而長大，範圍大時改用這個（2026-07-23新增）。
evaluate          讀已存模型，在 test set 上印 accuracy/AUC/混淆矩陣
"""

import argparse
import sys
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
from strategy.cnn.dataset import BRANCH_NAMES, available_months, build_dataset, load_shard_branch, load_shard_meta
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
        y = torch.tensor(self.target[i], dtype=torch.float32)
        return x, y


def _month_bound(date_str: str) -> str:
    """ "2026-07-15" → "2026_07"，跟 dataset.py 的月份shard key格式對齊。"""
    return f"{date_str[:4]}_{date_str[5:7]}"


def _resolve_months_and_cutoff(test_days: int, start_date: str, end_date: str, force_rebuild: bool):
    """共用邏輯：確保cache建好、篩出符合範圍的月份、算出test/train切分的
    cutoff日期。_split_data()/_load_test_ds()/train_sequential() 都靠這支
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


def _split_data(test_days: int, start_date: str = "", end_date: str = "", force_rebuild: bool = False):
    """依日期切分 train/test，跟 rally/mkt 一致：test_days 是最後 N 天當測試集，
    避免 in-sample 資料把績效灌水。回傳兩個 ShardedMultiScaleDataset（train/test），
    不會把任何分支tensor整包讀進RAM（見 ShardedMultiScaleDataset 說明）。

    只有 train()（一整包訓練，同時需要train_ds+test_ds）才呼叫這支——單純
    評估用 _load_test_ds()，不要在這裡多建根本用不到的 train_ds。"""
    months, metas, cutoff = _resolve_months_and_cutoff(test_days, start_date, end_date, force_rebuild)

    train_filters, test_filters = {}, {}
    for m, meta in metas.items():
        dates = meta["date"].to_numpy()
        train_filters[m] = np.nonzero(dates <= cutoff)[0]
        test_filters[m] = np.nonzero(dates > cutoff)[0]

    train_ds = ShardedMultiScaleDataset(months, train_filters)
    test_ds = ShardedMultiScaleDataset(months, test_filters)
    return train_ds, test_ds


def _load_test_ds(
    test_days: int, start_date: str = "", end_date: str = "", force_rebuild: bool = False
) -> ShardedMultiScaleDataset:
    """只建立test_ds，不建train_ds——evaluate()專用。2026-07-23發現：原本
    evaluate()借用_split_data()，會連根本用不到的train_ds也一起建出來，範圍
    一大（例如18個月）單純評估一個模型也會踩到跟訓練一樣的記憶體風險。"""
    months, metas, cutoff = _resolve_months_and_cutoff(test_days, start_date, end_date, force_rebuild)
    test_filters = {m: np.nonzero(meta["date"].to_numpy() > cutoff)[0] for m, meta in metas.items()}
    return ShardedMultiScaleDataset(months, test_filters)


def train(
    test_days: int = 10,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 3,
    start_date: str = "",
    end_date: str = "",
    force_rebuild: bool = False,
):
    """patience：test_loss連續這麼多個epoch沒有改善就提早停止訓練（early
    stopping），避免像2026-07-22半年/雙月測試那次一樣，train_loss一路降、
    test_loss後段卻暴衝（過擬合），卻還是把最後一個epoch的model存下來。
    最後存檔的一律是「test_loss最低那個epoch」的權重，不是最後一個epoch。"""
    train_ds, test_ds = _split_data(test_days, start_date, end_date, force_rebuild)
    print(f"\n訓練: {len(train_ds):,} 筆  測試: {len(test_ds):,} 筆")
    print(f"訓練集標籤分佈: {pd.Series(train_ds.target).value_counts(normalize=True).round(4).to_dict()}")

    # num_workers>0：ShardedMultiScaleDataset的資料是memmap（磁碟後端），shuffle=True
    # 打亂順序後每個batch要湊的256筆會分散在磁碟各處，單一process逐筆讀取會變I/O瓶頸
    # （2026-07-23實測：20分鐘CPU使用率只有~20%，一個epoch都跑不完）。開worker
    # process平行讀取可以大幅緩解，不會改變任何訓練結果，只是加速資料讀取。
    #
    # ⚠️ 只有 train_loader 需要：test_loader 是 shuffle=False（循序讀取，本來就
    # 沒有隨機存取I/O瓶頸）。第一次修的時候兩邊都開了 num_workers=4+
    # persistent_workers=True，等於同時常駐 4+4=8 個worker process，每個都要
    # 各自映射整批shard檔案，疊起來直接把訓練process搞死（2026-07-23實測：
    # process被強制終止，只留下17個洩漏的semaphore警告）。train_loader也只用
    # num_workers=2（不是4），降低同時存在的process數，避免重蹈覆轍。
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MultiScaleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 類別不平衡加權：2026-07-22實測發現模型會偷懶學「無腦猜多數類別」
    # （訓練集跌:漲=62:38時，AUC≈0.53、漲類別recall≈0）。pos_weight=負/正
    # 樣本數比例，跟rally/mkt樹模型的class_weight="balanced"是同樣的概念，
    # 只是BCEWithLogitsLoss這裡要自己算比例傳進去。
    n_pos = int((train_ds.target == 1).sum())
    n_neg = int((train_ds.target == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    print(f"訓練集 漲:跌 = {n_pos}:{n_neg}，pos_weight={pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_test_loss = float("inf")
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
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

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = {k: v.to(DEVICE) for k, v in x.items()}
                y = y.to(DEVICE)
                logits = model(x)
                total_test_loss += criterion(logits, y).item() * len(y)
        test_loss = total_test_loss / len(test_ds)
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"test_loss連續{patience}個epoch沒有改善，第{epoch}個epoch提早停止")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}（best test_loss={best_test_loss:.4f}）")
    return model


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


def train_sequential(
    test_days: int = 30,
    max_rounds: int = 10,
    chunk_size: int = 1_000_000,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 3,
    start_date: str = "",
    end_date: str = "",
    force_rebuild: bool = False,
):
    """分批漸進訓練——跟 train() 不同，不會把所有月份合併成一個大的
    ShardedMultiScaleDataset。任何時間點記憶體裡只有「當下這一個chunk
    （固定 chunk_size 筆，可能橫跨多個月份）的訓練資料」+「固定的小測試集」，
    不會隨著資料總量增加而長大（2026-07-23實測：train() 那種一次開全部
    月份memmap、shuffle打亂全範圍樣本的做法，7個月資料在24GB機器上就會逼近
    OOM邊緣，18個月以上幾乎必炸）。

    改成固定筆數切（不是按月份切）的原因：每個月份實際筆數差滿多（例如
    2月106萬筆、6月233萬筆，差超過2倍），按月切的話每個chunk記憶體用量會
    忽大忽小；固定筆數切能讓每個chunk大小更一致、更好預測（見
    _build_row_chunks() 的說明）。

    做法：外層跑最多 max_rounds 輪，每一輪依時間順序把訓練集切成的所有
    chunk依序各訓練1個epoch、訓練完立刻釋放該chunk資料，換下一個chunk；
    每個chunk訓練完都用同一組固定測試集評估一次，記錄目前最佳checkpoint。

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
    n_pos = sum(
        int((meta["target"].to_numpy()[meta["date"].to_numpy() <= cutoff] == 1).sum()) for meta in metas.values()
    )
    n_neg = sum(
        int((meta["target"].to_numpy()[meta["date"].to_numpy() <= cutoff] == 0).sum()) for meta in metas.values()
    )
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    print(f"訓練集 漲:跌 = {n_pos}:{n_neg}，pos_weight={pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_test_loss = float("inf")
    best_state = None
    epochs_without_improve = 0
    stopped = False

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
            del train_loader, train_ds  # 釋放這個chunk的memmap參照，才能真正回收記憶體

            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x = {k: v.to(DEVICE) for k, v in x.items()}
                    y = y.to(DEVICE)
                    logits = model(x)
                    total_test_loss += criterion(logits, y).item() * len(y)
            test_loss = total_test_loss / len(test_ds)
            print(
                f"round {round_idx}/{max_rounds}  chunk {chunk_idx}/{len(chunks)}  train_n={n_train:,}  "
                f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}"
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


def evaluate(test_days: int = 10, start_date: str = "", end_date: str = "", batch_size: int = 256):
    model = load_model()
    test_ds = _load_test_ds(test_days, start_date, end_date)
    test_target = test_ds.target
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for x, _y in loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)
    y_pred = (probs >= 0.5).astype(int)

    print(f"\n測試集: {len(test_target):,} 筆")
    print(f"Accuracy: {accuracy_score(test_target, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(test_target, probs):.4f}")
    print("\n混淆矩陣（列=實際，欄=預測，順序 跌/漲）:")
    print(confusion_matrix(test_target, y_pred, labels=[0, 1]))
    print("\n分類報告:")
    print(classification_report(test_target, y_pred, labels=[0, 1], target_names=["跌", "漲"], zero_division=0))


def main(
    mode: str = "",
    test_days: int = 10,
    epochs: int = 10,
    max_rounds: int = 10,
    chunk_size: int = 1_000_000,
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
      2. 終端機帶參數：python -m strategy.cnn.train train --test_days 10

    start_date/end_date：限制資料範圍（YYYY-MM-DD），第一次跑建議先限縮
    月份範圍，避免全量22個月資料第一次就跑很久（見 dataset.py 的效能備註）。

    train 跟 train_sequential 的差異（2026-07-23討論）：train 把所有請求月份
    合併成一包訓練，範圍一大（7個月以上）在24GB機器上容易OOM；train_sequential
    改成依固定筆數（chunk_size）切成好幾個chunk、一次只載入一個chunk訓練、練完
    立刻釋放，記憶體不會隨資料總量增加而長大，範圍大時改用這個。
    """
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="cnn 策略 — 多解析度 1D CNN")
        parser.add_argument(
            "mode",
            nargs="?",
            default="train",
            choices=["train", "train_sequential", "evaluate"],
            help="執行模式（預設train）",
        )
        parser.add_argument("--test_days", type=int, default=10, help="測試集天數")
        parser.add_argument("--epochs", type=int, default=10, help="訓練epoch數（只影響train）")
        parser.add_argument("--max_rounds", type=int, default=10, help="最多跑幾輪全部資料（只影響train_sequential）")
        parser.add_argument(
            "--chunk_size", type=int, default=1_000_000, help="每個chunk的筆數（只影響train_sequential）"
        )
        parser.add_argument("--batch_size", type=int, default=256, help="batch size")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate（只影響train/train_sequential）")
        parser.add_argument(
            "--patience",
            type=int,
            default=3,
            help="early stopping耐心值：test_loss連續幾次沒改善就停止（只影響train/train_sequential）",
        )
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        parser.add_argument("--force_rebuild", action="store_true", help="略過cache新鮮度檢查，強制重算tensor")
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        epochs = args.epochs
        max_rounds = args.max_rounds
        chunk_size = args.chunk_size
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
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            start_date=start_date,
            end_date=end_date,
            force_rebuild=force_rebuild,
        )
    elif mode == "train_sequential":
        train_sequential(
            test_days=test_days,
            max_rounds=max_rounds,
            chunk_size=chunk_size,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            start_date=start_date,
            end_date=end_date,
            force_rebuild=force_rebuild,
        )
    elif mode == "evaluate":
        evaluate(test_days=test_days, start_date=start_date, end_date=end_date, batch_size=batch_size)
    else:
        print(f"未知模式: {mode}，可用: train / train_sequential / evaluate")


if __name__ == "__main__":
    # ══════════════════════════════════════════════════════════════════════
    #  VS Code按F5：在這裡直接改變數，不用每次打 CLI
    # ══════════════════════════════════════════════════════════════════════
    mode = "evaluate"  # train / train_sequential / evaluate
    test_days = 30
    epochs = 10  # 只影響 train（一整包）
    max_rounds = 10  # 只影響 train_sequential（分chunk）
    chunk_size = 1_000_000  # 只影響 train_sequential，每個chunk的筆數
    batch_size = 256
    lr = 1e-3
    patience = 3
    start_date = "2026-01-01"  # 先限縮範圍測試基本架構跑不跑得通，之後再放寬
    end_date = ""
    force_rebuild = False

    main(
        mode=mode,
        test_days=test_days,
        epochs=epochs,
        max_rounds=max_rounds,
        chunk_size=chunk_size,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        start_date=start_date,
        end_date=end_date,
        force_rebuild=force_rebuild,
    )
