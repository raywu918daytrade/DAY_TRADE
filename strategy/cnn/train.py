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

train      訓練模型（存至 models/cnn_multiscale.pt）
evaluate   讀已存模型，在 test set 上印 accuracy/AUC/混淆矩陣
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
from strategy.cnn.dataset import load_dataset
from strategy.cnn.model import MultiScaleCNN

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "models/cnn_multiscale.pt"


class MultiScaleTensorDataset(Dataset):
    """把 dataset.py 產出的 {branch_name: (N,5,L) ndarray} + target 包成
    torch Dataset，__getitem__ 回傳的 dict 會被 DataLoader 預設 collate
    自動遞迴 batch 成 {branch_name: (B,5,L) tensor}，不用自己寫 collate_fn。"""

    def __init__(self, branches: dict, target: np.ndarray):
        self.branches = branches
        self.target = target

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx):
        x = {name: torch.from_numpy(arr[idx]).float() for name, arr in self.branches.items()}
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return x, y


def _split_data(test_days: int, start_date: str = "", end_date: str = "", force_rebuild: bool = False):
    """依日期切分 train/test，跟 rally/mkt 一致：test_days 是最後 N 天當測試集，
    避免 in-sample 資料把績效灌水。"""
    branches, meta = load_dataset(start_date=start_date, end_date=end_date, force_rebuild=force_rebuild)
    cutoff = meta["date"].max() - pd.Timedelta(days=test_days)
    train_mask = (meta["date"] <= cutoff).to_numpy()
    test_mask = ~train_mask

    train_branches = {k: v[train_mask] for k, v in branches.items()}
    test_branches = {k: v[test_mask] for k, v in branches.items()}
    train_target = meta["target"].to_numpy()[train_mask]
    test_target = meta["target"].to_numpy()[test_mask]
    return (train_branches, train_target), (test_branches, test_target), meta


def train(
    test_days: int = 10,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    start_date: str = "",
    end_date: str = "",
    force_rebuild: bool = False,
):
    (train_branches, train_target), (test_branches, test_target), meta = _split_data(
        test_days, start_date, end_date, force_rebuild
    )
    print(
        f"\n訓練: {len(train_target):,} 筆  測試: {len(test_target):,} 筆  "
        f"（資料範圍 {meta['date'].min()} ~ {meta['date'].max()}）"
    )
    print(f"訓練集標籤分佈: {pd.Series(train_target).value_counts(normalize=True).round(4).to_dict()}")

    train_ds = MultiScaleTensorDataset(train_branches, train_target)
    test_ds = MultiScaleTensorDataset(test_branches, test_target)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MultiScaleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

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

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}")
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
    (_, _), (test_branches, test_target), _ = _split_data(test_days, start_date, end_date)
    test_ds = MultiScaleTensorDataset(test_branches, test_target)
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
    batch_size: int = 256,
    lr: float = 1e-3,
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
    """
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="cnn 策略 — 多解析度 1D CNN")
        parser.add_argument(
            "mode", nargs="?", default="train", choices=["train", "evaluate"], help="執行模式（預設train）"
        )
        parser.add_argument("--test_days", type=int, default=10, help="測試集天數")
        parser.add_argument("--epochs", type=int, default=10, help="訓練epoch數（只影響train）")
        parser.add_argument("--batch_size", type=int, default=256, help="batch size")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate（只影響train）")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        parser.add_argument("--force_rebuild", action="store_true", help="略過cache新鮮度檢查，強制重算tensor")
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
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
            start_date=start_date,
            end_date=end_date,
            force_rebuild=force_rebuild,
        )
    elif mode == "evaluate":
        evaluate(test_days=test_days, start_date=start_date, end_date=end_date, batch_size=batch_size)
    else:
        print(f"未知模式: {mode}，可用: train / evaluate")


if __name__ == "__main__":
    # ══════════════════════════════════════════════════════════════════════
    #  VS Code按F5：在這裡直接改變數，不用每次打 CLI
    # ══════════════════════════════════════════════════════════════════════
    mode = "evaluate"  # train / evaluate
    test_days = 30
    epochs = 10
    batch_size = 256
    lr = 1e-3
    start_date = "2026-01-01"  # 先限縮範圍測試基本架構跑不跑得通，之後再放寬
    end_date = ""
    force_rebuild = False

    main(
        mode=mode,
        test_days=test_days,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        start_date=start_date,
        end_date=end_date,
        force_rebuild=force_rebuild,
    )
