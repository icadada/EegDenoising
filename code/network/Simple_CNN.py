"""Simple_CNN：4 层一维卷积去噪网络（对应原版 EEGdenoiseNet 的 simple_CNN）。

结构：连续 4 个 ``Conv1d(→64, k=3) + BatchNorm + ReLU + Dropout``，随后展平接一层全连接
映射回原始长度。输入 ``[B, L]``，内部升维成 ``[B, 1, L]``，输出 ``[B, L]``。
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

NETWORK_NAME = "Simple_CNN"
DEFAULT_INIT_ARGS = {"channels": 64, "kernel_size": 3, "dropout": 0.3}
IO_SHAPE = "[B, L] -> [B, L]"


def _glorot_init(module: nn.Module) -> None:
    """Xavier(Glorot) 均匀初始化（与原版 Keras 行为一致）。"""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SimpleCNN(nn.Module):
    """4 层 Conv1d 特征提取 + 1 层全连接输出。"""

    def __init__(
        self,
        datanum: int,
        channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.datanum = int(datanum)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)

        layer_list = []
        in_channels = 1
        for _ in range(4):
            layer_list += [
                nn.Conv1d(in_channels, self.channels, self.kernel_size,
                          stride=1, padding=self.kernel_size // 2),
                nn.BatchNorm1d(self.channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_channels = self.channels
        self.features = nn.Sequential(*layer_list)
        # 展平后长度 = 通道数 x 时间长度
        self.output = nn.Linear(self.channels * self.datanum, self.datanum)
        self.apply(_glorot_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, L]`` -> 返回 ``[B, L]``。"""
        if x.dim() == 3:  # 容错：接受 [B,1,L] 或 [B,L,1]
            x = x.reshape(x.shape[0], -1)
        x = x.unsqueeze(1)                 # [B, L] -> [B, 1, L]
        y = self.features(x)               # [B, C, L]
        return self.output(torch.flatten(y, 1))


def build(datanum: int, **kwargs: Any) -> nn.Module:
    """模型工厂函数。"""
    return SimpleCNN(datanum, **kwargs)