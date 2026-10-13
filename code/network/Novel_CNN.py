"""Novel_CNN：深层逐级池化卷积去噪网络（对应原版 EEGdenoiseNet 的 Novel_CNN）。

结构：7 个卷积块，每个块为 ``Conv-ReLU-Conv-ReLU``，通道数逐级翻倍
（1 -> 32 -> 64 -> ... -> 2048），通道数 >= 128 的块后接 ``Dropout(0.5)``。
前 6 个块后接 ``AvgPool1d(2)`` 做下采样（与原版一致：第 7 块之后不再池化），
最后展平接全连接层映射回原始长度。

与旧实现的区别：这里通过一次**空跑（dry-run）**自动推断展平后的特征维度，
因此对任意信号长度 ``L`` 都成立，不再写死 ``L // 64``。

输入 ``[B, L]``，输出 ``[B, L]``。
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch
from torch import nn

NETWORK_NAME = "Novel_CNN"
DEFAULT_INIT_ARGS = {"dropout": 0.5}
IO_SHAPE = "[B, L] -> [B, L]"

#: 每个卷积块的 (输入通道, 输出通道, 是否加 Dropout)，共 7 块
CHANNEL_CONFIG: Sequence[Tuple[int, int, bool]] = (
    (1, 32, False),
    (32, 64, False),
    (64, 128, False),
    (128, 256, True),
    (256, 512, True),
    (512, 1024, True),
    (1024, 2048, True),
)


def _he_init(module: nn.Module) -> None:
    """He(Kaiming) 正态初始化，与原版 Novel_CNN 使用的初始化一致。"""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class NovelCNN(nn.Module):
    """7 块深层卷积网络（6 次平均池化 + 1 层全连接输出）。"""

    def __init__(self, datanum: int, dropout: float = 0.5, **kwargs: Any) -> None:
        super().__init__()
        self.datanum = int(datanum)

        module_list: List[nn.Module] = []
        for index, (in_channels, out_channels, use_dropout) in enumerate(CHANNEL_CONFIG):
            module_list += [
                nn.Conv1d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
            ]
            if use_dropout:
                module_list.append(nn.Dropout(dropout))
            if index < len(CHANNEL_CONFIG) - 1:      # 与原版一致：最后一块之后不池化
                module_list.append(nn.AvgPool1d(2))
        self.features = nn.Sequential(*module_list)

        # 空跑一次，自动得到展平后的特征维度（避免写死 L//64）
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.datanum)
            feature_dim = int(self.features(dummy_input).numel())
        self.output = nn.Linear(feature_dim, self.datanum)
        self.apply(_he_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, L]`` -> 返回 ``[B, L]``。"""
        if x.dim() == 3:  # 容错
            x = x.reshape(x.shape[0], -1)
        x = x.unsqueeze(1)                          # [B, 1, L]
        y = self.features(x)                        # [B, 2048, L/64]
        return self.output(torch.flatten(y, 1))     # [B, L]


def build(datanum: int, **kwargs: Any) -> nn.Module:
    """模型工厂函数。"""
    return NovelCNN(datanum, **kwargs)
