"""
ResNet + GRU 混合模型 — ResNet 看近 10 分鐘的原始價格型態，GRU 從 9:00
到當下逐分鐘看累積 VWAP 偏離軌跡，兩者 concat 後接 MLP head 做 3 分類。

設計概念（2026-07-27 討論）：
  1. ResNet（1D Conv + skip connection）：看候選前 10 分鐘的原始 OHLCV
     （5 channels × 10 步），提取價格形狀（急拉、緩漲、震盪等）。
  2. GRU：從 9:00 到觸發當下，每步 14 維特徵（含 VWAP z-score），
     累積越長的偏離歷史軌跡。
  3. Concat 兩個 embedding → MLP → 3 分類。

這樣 ResNet 負責「近 10 分鐘價格形狀」，GRU 負責「從開盤到現在累積軌跡」，
各司其職。9:11 觸發時 GRU 看 11 步、9:50 觸發時 GRU 看 50 步，支援可變長度。
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from strategy.vwap_dl.config import CNN_EMBED_DIM, DROPOUT, GRU_HIDDEN_DIM, GRU_N_LAYERS, GRU_INPUT_DIM

N_CLASSES = 3  # 回歸=0 / 無訊號=1 / 延續=2


class _ResNetBlock(nn.Module):
    """1D ResNet 基本塊：兩個 Conv1d + BatchNorm + ReLU + skip connection。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # skip connection
        out = self.relu(out)
        return out


class _ResNetBranch(nn.Module):
    """ResNet 分支：Conv1d 升 channel → 2 個 ResNetBlock → AdaptiveAvgPool → embedding。

    輸入：(batch, 5, 10) 原始 OHLCV
    輸出：(batch, embed_dim)
    """

    def __init__(self, embed_dim: int = CNN_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            _ResNetBlock(16),
            _ResNetBlock(16),
            nn.Conv1d(16, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5, 10)
        return self.net(x).squeeze(-1)  # (batch, embed_dim)


class ResNetGRUModel(nn.Module):
    """ResNet + GRU 混合模型。

    ResNet 看近 10 分鐘原始 OHLCV → local embedding。
    GRU 從 9:00 到當下逐分鐘看 18 維特徵（含 VWAP z-score + 大盤 VWAP 特徵）。
    Concat → MLP → 3 分類 logits。

    2026-08-04 曾經試過在GRU分支加 _TemporalAttention（對逐步輸出做加權
    平均取代單純用h_n[-1]，理由是解決「最後hidden state瓶頸」）——離線
    指標進步（Accuracy 0.50→0.55、AUC 0.68→0.71），但回測在30天/90天
    兩個窗口都一致變差（30天10筆/80.0%/+1.20%→7筆/71.4%/+0.32%；90天
    21筆/71.4%/+1.60%→19筆/68.4%/+1.25%），判斷是label定義的「回歸」
    跟backtest的「賺錢」本來就是兩件事、AUC整體排序進步不保證
    threshold=0.6這個切點附近的樣本也變好，改回不用。備份在
    models/vwap_dl_cnngru.pt.bak_pre_attn，之後想重新驗證可以直接復用
    這份說明，不用重新設計。
    """

    def __init__(
        self,
        embed_dim: int = CNN_EMBED_DIM,
        gru_hidden: int = GRU_HIDDEN_DIM,
        gru_layers: int = GRU_N_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.resnet = _ResNetBranch(embed_dim)
        self.gru = nn.GRU(
            input_size=GRU_INPUT_DIM,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, resnet_x: torch.Tensor, gru_seq: torch.Tensor, gru_lengths: torch.Tensor) -> torch.Tensor:
        """resnet_x: (batch, 5, 10) 原始 OHLCV 視窗
        gru_seq: (batch, max_seq_len, 14) 逐分鐘特徵（含 VWAP z-score）
        gru_lengths: (batch,) 實際序列長度（9:00 到觸發的分鐘數）
        """
        # ResNet 分支：近 10 分鐘價格形狀
        local_embed = self.resnet(resnet_x)  # (batch, embed_dim)

        # GRU 分支：從 9:00 到當下的累積軌跡
        packed = rnn_utils.pack_padded_sequence(gru_seq, gru_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # h_n: (num_layers, batch, gru_hidden)
        global_hidden = h_n[-1]  # (batch, gru_hidden) 取最後一層

        # Concat + MLP
        combined = torch.cat([local_embed, global_hidden], dim=1)  # (batch, embed_dim + gru_hidden)
        logits = self.head(combined)  # (batch, 3)
        return logits
