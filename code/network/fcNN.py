"""fcNN：全连接去噪网络（对应原版 EEGdenoiseNet 的 fcNN）。

结构：4 层全连接，前 3 层后接 ReLU 与 Dropout，最后一层直接输出。
输入输出均为 ``[B, L]``，不需要任何维度变换。
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

#: 模型名（同时也是命令行 --network 的取值）
NETWORK_NAME = "fcNN"
#: 本模型的默认初始化参数（可被 --model-args / --model-kwargs 覆盖）
DEFAULT_INIT_ARGS = {"dropout": 0.3}
#: 文档用：说明输入输出形状
IO_SHAPE = "[B, L] -> [B, L]"


def _glorot_init(module: nn.Module) -> None:
    """Xavier(Glorot) 均匀初始化，与原版 Keras 默认初始化保持一致。"""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FcNN(nn.Module):
    """4 层全连接网络。"""

    def __init__(self, datanum: int, dropout: float = 0.3, **kwargs: Any) -> None:
        super().__init__()
        # kwargs 里可能包含其他模型才需要的参数，这里忽略即可（便于命令行统一传参）
        self.datanum = int(datanum)
        self.net = nn.Sequential(
            nn.Linear(self.datanum, self.datanum), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.datanum, self.datanum), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.datanum, self.datanum), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.datanum, self.datanum),
        )
        self.apply(_glorot_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：``x`` 形状 ``[B, L]``，返回 ``[B, L]``。"""
        if x.dim() == 3:  # 容错：接受 [B,1,L] 或 [B,L,1]
            x = x.reshape(x.shape[0], -1)
        return self.net(x)


def build(datanum: int, **kwargs: Any) -> nn.Module:
    """模型工厂函数（本项目统一入口名）。"""
    return FcNN(datanum, **kwargs)
