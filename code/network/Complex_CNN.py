"""Complex_CNN：多尺度残差卷积去噪网络（对应原版 EEGdenoiseNet 的 complex_CNN）。

结构：

1. **主干 stem**：``Conv1d(1 -> 32, k=5) + BatchNorm + ReLU``；
2. **多尺度残差分支**：3 条并行分支（卷积核 3、5、7），每条分支堆叠 2 个残差块，
   每个残差块内部为 ``Conv(32->32) - Conv(32->16) - Conv(16->32)`` 的残差连接；
3. **融合 head**：把 3 条分支输出（32x3=96 通道）拼接后用 ``Conv1d(96 -> 32, k=1)`` 融合；
4. **输出层**：展平后全连接映射回原始长度。

输入 ``[B, L]``，输出 ``[B, L]``。
"""
from __future__ import annotations

from typing import Any, List

import torch
from torch import nn

NETWORK_NAME = "Complex_CNN"
DEFAULT_INIT_ARGS = {"stem_channels": 32, "dropout": 0.0}
IO_SHAPE = "[B, L] -> [B, L]"


def _glorot_init(module: nn.Module) -> None:
    """Xavier(Glorot) 均匀初始化。"""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    """一个残差块：32 -> 32 -> 16 -> 32，最后与原输入相加。"""

    def __init__(self, channels: int = 32, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels // 2, kernel_size, padding=padding),
            nn.BatchNorm1d(channels // 2), nn.ReLU(),
            nn.Conv1d(channels // 2, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        # 可选 dropout（原版没有，默认 0 表示关闭）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.apply(_glorot_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差连接：输出与输入形状相同，可直接相加。"""
        return self.dropout(self.body(x)) + x


class MultiScaleBranch(nn.Module):
    """一条多尺度分支：两个相同卷积核的残差块串联。"""

    def __init__(self, kernel_size: int, channels: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(channels, kernel_size, dropout),
            ResidualBlock(channels, kernel_size, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ComplexCNN(nn.Module):
    """多尺度（k=3/5/7）残差卷积网络。"""

    def __init__(
        self,
        datanum: int,
        stem_channels: int = 32,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.datanum = int(datanum)
        self.stem_channels = int(stem_channels)

        # 1) 主干：把单通道信号升到 stem_channels 通道
        self.stem = nn.Sequential(
            nn.Conv1d(1, self.stem_channels, 5, padding=2),
            nn.BatchNorm1d(self.stem_channels),
            nn.ReLU(),
        )
        # 2) 三条不同卷积核尺度的残差分支
        self.branches = nn.ModuleList(
            [MultiScaleBranch(k, self.stem_channels, dropout) for k in (3, 5, 7)]
        )
        # 3) 融合层：96 = 32 x 3 通道 -> 32 通道
        self.head = nn.Sequential(
            nn.Conv1d(self.stem_channels * len(self.branches), self.stem_channels, 1),
            nn.BatchNorm1d(self.stem_channels),
            nn.ReLU(),
        )
        # 4) 输出层：展平后映射回原始长度
        self.output = nn.Linear(self.stem_channels * self.datanum, self.datanum)
        self.apply(_glorot_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, L]`` -> 返回 ``[B, L]``。"""
        if x.dim() == 3:  # 容错
            x = x.reshape(x.shape[0], -1)
        x = x.unsqueeze(1)                          # [B, 1, L]
        y = self.stem(x)                            # [B, 32, L]
        branch_outputs: List[torch.Tensor] = [branch(y) for branch in self.branches]
        y = torch.cat(branch_outputs, dim=1)          # [B, 96, L]
        y = self.head(y)                            # [B, 32, L]
        return self.output(torch.flatten(y, 1))     # [B, L]


def build(datanum: int, **kwargs: Any) -> nn.Module:
    """模型工厂函数。"""
    return ComplexCNN(datanum, **kwargs)
