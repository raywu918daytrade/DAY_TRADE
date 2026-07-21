"""
多解析度 1D CNN 模型定義。

五路分支（m1/m3/m5/m3_std/m5_std）結構相同、各自獨立權重的 Conv1D，最後用
AdaptiveAvgPool1d 把不同長度的分支都壓成同樣維度的 embedding，concat 起來
接 MLP head 做二元分類（target 0/1，triple barrier）。
"""

import torch
import torch.nn as nn

from strategy.cnn.dataset import BRANCH_NAMES, N_CHANNELS


class _Branch(nn.Module):
    """單一分支：Conv1d → BatchNorm1d → ReLU，重複幾層，最後 AdaptiveAvgPool1d(1)
    壓成固定長度 embedding——不用手算每個分支卷積後的長度，五路輸入長度不同
    （m1/m3/m5=60，m3_std=20，m5_std=12）也能共用同一個 class。"""

    def __init__(self, in_channels: int = N_CHANNELS, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch, embed_dim)


class MultiScaleCNN(nn.Module):
    """五路多解析度 Conv1D，輸入為 dict[branch_name] -> (batch, 5, branch_len)。"""

    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.branches = nn.ModuleDict({name: _Branch(N_CHANNELS, embed_dim) for name in BRANCH_NAMES})
        self.head = nn.Sequential(
            nn.Linear(embed_dim * len(BRANCH_NAMES), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        embeds = [self.branches[name](inputs[name]) for name in BRANCH_NAMES]
        x = torch.cat(embeds, dim=1)
        return self.head(x).squeeze(-1)  # (batch,) logits
