"""
vwap_dl 策略訓練 — ResNet + GRU 混合模型（CLI 進入點）。

ResNet 看近 10 分鐘原始 OHLCV，GRU 從 9:00 到當下逐分鐘累積 VWAP 軌跡。
"""

import argparse
import sys
import time
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from strategy.vwap_dl.config import DEVICE
from strategy.vwap_dl.dataset import available_months, build_dataset, load_shard_data, load_shard_meta
from strategy.vwap_dl.model import ResNetGRUModel, N_CLASSES

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "models/vwap_dl_cnngru.pt"

_TARGET_NAMES = ["回歸", "無訊號", "延續"]


class ShardedDataset(Dataset):
    """跨月份 shard 讀取的 torch Dataset。

    resnet_x / gru_lengths 用 mmap，gru_seq 是 object array（list of arrays，
    用 allow_pickle 載入）。
    """

    def __init__(self, months: list[str], row_filters: dict[str, np.ndarray]):
        self._data: dict[str, dict] = {}
        self._index: list[tuple[str, int]] = []
        meta_parts = []
        for month in months:
            keep = row_filters.get(month)
            if keep is None or len(keep) == 0:
                continue
            self._data[month] = load_shard_data(month, mmap=True)
            meta = load_shard_meta(month)
            meta_parts.append(meta.iloc[keep])
            self._index.extend((month, int(i)) for i in keep)

        self.meta = (
            pd.concat(meta_parts, ignore_index=True)
            if meta_parts
            else pd.DataFrame(columns=["stock_id", "date", "target", "atr5", "trigger_side"])
        )
        self.target = self.meta["target"].to_numpy()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i):
        month, local_idx = self._index[i]
        d = self._data[month]

        resnet_x = torch.from_numpy(np.array(d["resnet_x"][local_idx])).float()  # (5, 10)
        gru_seq = torch.from_numpy(np.array(d["gru_seq"][local_idx])).float()  # (seq_len, 14)
        gru_len = torch.tensor(len(gru_seq), dtype=torch.long)
        y = torch.tensor(self.target[i], dtype=torch.long)

        return resnet_x, gru_seq, gru_len, y


def collate_fn(batch):
    """自訂 collate：對不同長度的 gru_seq 做 padding。"""
    resnet_x = torch.stack([b[0] for b in batch], dim=0)  # (batch, 5, 10)
    gru_lens = torch.tensor([b[2] for b in batch], dtype=torch.long)  # (batch,)
    # gru_seq 先 pad 到 batch 內最長
    max_len = gru_lens.max().item()
    gru_seqs = []
    for b in batch:
        seq = b[1]  # (seq_len, 14)
        pad_len = max_len - len(seq)
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, seq.size(1), dtype=seq.dtype)], dim=0)
        gru_seqs.append(seq)
    gru_seq = torch.stack(gru_seqs, dim=0)  # (batch, max_len, 14)
    y = torch.stack([b[3] for b in batch], dim=0)  # (batch,)
    return resnet_x, gru_seq, gru_lens, y


def _month_bound(date_str: str) -> str:
    return f"{date_str[:4]}_{date_str[5:7]}"


def _class_weights(target: np.ndarray) -> torch.Tensor:
    counts = np.array([(target == c).sum() for c in range(N_CLASSES)], dtype=np.float64)
    counts = np.maximum(counts, 1)
    weights = len(target) / (N_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def _build_row_chunks(
    months: list[str],
    metas: dict[str, pd.DataFrame],
    cutoff,
    chunk_size: int,
) -> list[dict[str, np.ndarray]]:
    flat: list[tuple[str, int]] = []
    for month in months:
        dates = metas[month]["date"].to_numpy()
        mask = dates <= cutoff
        train_idx = np.nonzero(mask)[0]
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
    std_mult: float = 2.0,
):
    # 確保 cache 建好
    months = build_dataset(start_date=start_date, end_date=end_date, std_mult=std_mult, force_rebuild=force_rebuild)
    months = available_months()
    if start_date:
        months = [m for m in months if m >= _month_bound(start_date)]
    if end_date:
        months = [m for m in months if m <= _month_bound(end_date)]
    if not months:
        raise RuntimeError("cache/vwap_dl/ 沒有可用的月份 shard")
    months = sorted(months)
    print(f"可用月份: {months}")

    metas = {m: load_shard_meta(m) for m in months}
    global_max_date = max(meta["date"].max() for meta in metas.values())
    cutoff = global_max_date - pd.Timedelta(days=test_days)
    print(f"cutoff date = {cutoff}")

    # 固定測試集
    test_filters = {m: np.nonzero(meta["date"].to_numpy() > cutoff)[0] for m, meta in metas.items()}
    test_ds = ShardedDataset(months, test_filters)
    print(f"固定測試集: {len(test_ds):,} 筆（最後 {test_days} 天）")
    if len(test_ds) == 0:
        print("測試集為空，跳過訓練。")
        return None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 訓練集 chunk
    chunks = _build_row_chunks(months, metas, cutoff, chunk_size)
    print(f"訓練集共切成 {len(chunks)} 個 chunk（每個約 {chunk_size:,} 筆）")
    if len(chunks) == 0:
        print("訓練集為空，跳過訓練。")
        return None

    model = ResNetGRUModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_targets = np.concatenate(
        [meta["target"].to_numpy()[meta["date"].to_numpy() <= cutoff] for meta in metas.values()]
    )
    weights = _class_weights(train_targets)
    print(
        f"訓練集標籤數: {[(train_targets == c).sum() for c in range(N_CLASSES)]}, " f"class_weights={weights.tolist()}"
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_test_loss = float("inf")
    best_state = None
    epochs_without_improve = 0
    stopped = False
    chunk_counter = 0

    for round_idx in range(1, max_rounds + 1):
        for chunk_idx, row_filters in enumerate(chunks, start=1):
            chunk_months = list(row_filters.keys())
            train_ds = ShardedDataset(chunk_months, row_filters)
            if len(train_ds) == 0:
                del train_ds
                continue
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
            )

            train_t0 = time.time()
            model.train()
            total_loss = 0.0
            for resnet_x, gru_seq, gru_len, y in train_loader:
                resnet_x = resnet_x.to(DEVICE)
                gru_seq = gru_seq.to(DEVICE)
                gru_len = gru_len.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(resnet_x, gru_seq, gru_len)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
            train_loss = total_loss / len(train_ds)
            n_train = len(train_ds)
            train_sec = time.time() - train_t0
            del train_loader, train_ds

            chunk_counter += 1
            if chunk_counter % eval_every != 0:
                print(
                    f"round {round_idx}/{max_rounds}  chunk {chunk_idx}/{len(chunks)}  "
                    f"train_n={n_train:,}  train_loss={train_loss:.4f}  "
                    f"train_sec={train_sec:.1f}  (跳過評估)"
                )
                continue

            eval_t0 = time.time()
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for resnet_x, gru_seq, gru_len, y in test_loader:
                    resnet_x = resnet_x.to(DEVICE)
                    gru_seq = gru_seq.to(DEVICE)
                    gru_len = gru_len.to(DEVICE)
                    y = y.to(DEVICE)
                    logits = model(resnet_x, gru_seq, gru_len)
                    total_test_loss += criterion(logits, y).item() * len(y)
            test_loss = total_test_loss / len(test_ds)
            eval_sec = time.time() - eval_t0

            print(
                f"round {round_idx}/{max_rounds}  chunk {chunk_idx}/{len(chunks)}  "
                f"train_n={n_train:,}  train_loss={train_loss:.4f}  "
                f"test_loss={test_loss:.4f}  train_sec={train_sec:.1f}  eval_sec={eval_sec:.1f}"
            )

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            elif round_idx > 1:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(f"test_loss 連續 {patience} 次沒有改善，提早停止")
                    stopped = True
                    break
        if stopped:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), _MODEL_PATH)
    print(f"模型已存至 {_MODEL_PATH}（best test_loss={best_test_loss:.4f}）")

    print("\n── 最終測試集評估 ──")
    _run_inference_and_report(model, test_ds, batch_size)
    return model


def load_model() -> ResNetGRUModel:
    if not _MODEL_PATH.exists():
        raise FileNotFoundError("找不到模型，請先執行 train")
    model = ResNetGRUModel().to(DEVICE)
    model.load_state_dict(torch.load(_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def _run_inference_and_report(model: ResNetGRUModel, test_ds: ShardedDataset, batch_size: int) -> None:
    test_target = test_ds.target
    print(f"測試集: {len(test_target):,} 筆")
    if len(test_target) == 0:
        return

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_probs = []
    with torch.no_grad():
        for resnet_x, gru_seq, gru_len, _y in loader:
            resnet_x = resnet_x.to(DEVICE)
            gru_seq = gru_seq.to(DEVICE)
            gru_len = gru_len.to(DEVICE)
            probs = torch.softmax(model(resnet_x, gru_seq, gru_len), dim=1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)
    y_pred = probs.argmax(axis=1)

    print(f"Accuracy: {accuracy_score(test_target, y_pred):.4f}")
    print(f"AUC (one-vs-rest): {roc_auc_score(test_target, probs, multi_class='ovr', labels=[0, 1, 2]):.4f}")
    print("\n混淆矩陣（列=實際，欄=預測，順序 回歸/無訊號/延續）:")
    print(confusion_matrix(test_target, y_pred, labels=[0, 1, 2]))
    print("\n分類報告:")
    print(classification_report(test_target, y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0))


def evaluate(
    test_days: int = 30,
    start_date: str = "",
    end_date: str = "",
    batch_size: int = 256,
):
    model = load_model()
    months = available_months()
    if start_date:
        months = [m for m in months if m >= _month_bound(start_date)]
    if end_date:
        months = [m for m in months if m <= _month_bound(end_date)]
    months = sorted(months)
    if not months:
        print("沒有可用月份。")
        return

    metas = {m: load_shard_meta(m) for m in months}
    global_max_date = max(meta["date"].max() for meta in metas.values())
    cutoff = global_max_date - pd.Timedelta(days=test_days)
    test_filters = {m: np.nonzero(meta["date"].to_numpy() > cutoff)[0] for m, meta in metas.items()}
    test_ds = ShardedDataset(months, test_filters)
    print(f"測試集: {len(test_ds):,} 筆（最後 {test_days} 天）")
    _run_inference_and_report(model, test_ds, batch_size)


def confidence_report(
    test_days: int = 30,
    thresholds: list[float | None] | None = None,
    start_date: str = "",
    end_date: str = "",
    batch_size: int = 256,
):
    thresholds = thresholds if thresholds is not None else [None, 0.4, 0.5, 0.6, 0.7, 0.8]
    model = load_model()
    months = available_months()
    if start_date:
        months = [m for m in months if m >= _month_bound(start_date)]
    if end_date:
        months = [m for m in months if m <= _month_bound(end_date)]
    months = sorted(months)
    if not months:
        print("沒有可用月份。")
        return

    metas = {m: load_shard_meta(m) for m in months}
    global_max_date = max(meta["date"].max() for meta in metas.values())
    cutoff = global_max_date - pd.Timedelta(days=test_days)
    test_filters = {m: np.nonzero(meta["date"].to_numpy() > cutoff)[0] for m, meta in metas.items()}
    test_ds = ShardedDataset(months, test_filters)
    test_target = test_ds.target
    print(f"測試集: {len(test_target):,} 筆")
    if len(test_target) == 0:
        return

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_probs = []
    with torch.no_grad():
        for resnet_x, gru_seq, gru_len, _y in loader:
            resnet_x = resnet_x.to(DEVICE)
            gru_seq = gru_seq.to(DEVICE)
            gru_len = gru_len.to(DEVICE)
            p = torch.softmax(model(resnet_x, gru_seq, gru_len), dim=1).cpu().numpy()
            all_probs.append(p)
    probs = np.concatenate(all_probs)

    for threshold in thresholds:
        y_pred = _predict_with_threshold(probs, threshold)
        label = "argmax（無門檻）" if threshold is None else f"信心度門檻={threshold:.2f}"
        coverage = (y_pred != 1).mean()
        print(f"\n── {label}，判斷覆蓋率(非無訊號)={coverage:.4f} ──")
        print(f"Accuracy: {accuracy_score(test_target, y_pred):.4f}")
        print(confusion_matrix(test_target, y_pred, labels=[0, 1, 2]))
        print(classification_report(test_target, y_pred, labels=[0, 1, 2], target_names=_TARGET_NAMES, zero_division=0))


def _predict_with_threshold(probs: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None:
        return probs.argmax(axis=1)
    p_revert = probs[:, 0]
    p_continue = probs[:, 2]
    y_pred = np.ones(len(probs), dtype=int)
    revert_pass = p_revert >= threshold
    continue_pass = p_continue >= threshold
    y_pred[revert_pass & ~continue_pass] = 0
    y_pred[continue_pass & ~revert_pass] = 2
    both = revert_pass & continue_pass
    y_pred[both] = np.where(p_continue[both] >= p_revert[both], 2, 0)
    return y_pred


_LOAD_MODEL = load_model


def main(
    mode: str = "",
    test_days: int = 30,
    max_rounds: int = 10,
    chunk_size: int = 1_000_000,
    eval_every: int = 1,
    thresholds: list[float] | None = None,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 3,
    start_date: str = "",
    end_date: str = "",
    force_rebuild: bool = False,
    std_mult: float = 2.0,
):
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="vwap_dl 策略 — ResNet + GRU")
        parser.add_argument(
            "mode",
            nargs="?",
            default="train",
            choices=["train", "evaluate", "confidence"],
            help="執行模式（預設train）",
        )
        parser.add_argument("--test_days", type=int, default=30, help="測試集天數")
        parser.add_argument("--max_rounds", type=int, default=10, help="最多跑幾輪全部資料")
        parser.add_argument("--chunk_size", type=int, default=1_000_000, help="每個 chunk 筆數")
        parser.add_argument("--eval_every", type=int, default=1, help="每幾個 chunk 評估一次")
        parser.add_argument("--thresholds", type=float, nargs="*", default=None, help="信心度門檻清單")
        parser.add_argument("--batch_size", type=int, default=256, help="batch size")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
        parser.add_argument("--start_date", type=str, default="", help="資料起日 YYYY-MM-DD")
        parser.add_argument("--end_date", type=str, default="", help="資料迄日 YYYY-MM-DD")
        parser.add_argument("--force_rebuild", action="store_true", help="強制重建 cache")
        parser.add_argument("--std_mult", type=float, default=2.0, help="VWAP z-score 門檻")
        args = parser.parse_args()
        mode = args.mode
        test_days = args.test_days
        max_rounds = args.max_rounds
        chunk_size = args.chunk_size
        eval_every = args.eval_every
        thresholds = args.thresholds
        batch_size = args.batch_size
        lr = args.lr
        patience = args.patience
        start_date = args.start_date
        end_date = args.end_date
        force_rebuild = args.force_rebuild
        std_mult = args.std_mult
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
            std_mult=std_mult,
        )
    elif mode == "evaluate":
        evaluate(test_days=test_days, start_date=start_date, end_date=end_date, batch_size=batch_size)
    elif mode == "confidence":
        confidence_report(
            test_days=test_days,
            thresholds=thresholds if thresholds else None,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
        )
    else:
        print(f"未知模式: {mode}")


if __name__ == "__main__":
    # VS Code F5 用
    mode = "train"
    test_days = 30
    max_rounds = 10
    chunk_size = 500_000
    eval_every = 3
    thresholds = None
    batch_size = 256
    lr = 1e-3
    patience = 3
    start_date = "2026-01-01"
    end_date = ""
    force_rebuild = False
    std_mult = 2.0

    main(
        mode=mode,
        test_days=test_days,
        max_rounds=max_rounds,
        chunk_size=chunk_size,
        eval_every=eval_every,
        thresholds=thresholds,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        start_date=start_date,
        end_date=end_date,
        force_rebuild=force_rebuild,
        std_mult=std_mult,
    )
