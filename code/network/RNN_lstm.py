"""RNN_lstm：LSTM + 全连接去噪网络（对应原版 EEGdenoiseNet 的 RNN_lstm）。

结构：单层 LSTM（``input_size=1, hidden_size=1``）沿时间轴处理每个采样点，随后接 3 层
全连接（前两层带 ReLU 与 Dropout）映射回原始长度。

输入 ``[B, L]``，内部升维成 ``[B, L, 1]``（Keras LSTM 的样本优先格式），输出 ``[B, L]``。
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

NETWORK_NAME = "RNN_lstm"
DEFAULT_INIT_ARGS = {"hidden_size": 1, "dropout": 0.3}
IO_SHAPE = "[B, L] -> [B, L]"


def _glorot_init(module: nn.Module) -> None:
    """Xavier(Glorot) 均匀初始化。注意 LSTM 的参数不在此列，保持 PyTorch 默认初始化。"""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RNNLSTM(nn.Module):
    """LSTM(1->1) + 全连接网络的混合结构。"""

    def __init__(
        self,
        datanum: int,
        hidden_size: int = 1,
        dropout: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.datanum = int(datanum)
        self.hidden_size = int(hidden_size)
        # batch_first=True 表示输入形状为 [B, L, 1]
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(self.datanum * self.hidden_size, self.datanum), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.datanum, self.datanum), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.datanum, self.datanum),
        )
        self.apply(_glorot_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, L]`` -> 返回 ``[B, L]``。"""
        if x.dim() == 3:  # 容错：接受 [B,1,L] 或 [B,L,1]
            if x.shape[-1] == 1:
                x = x.squeeze(-1)          # [B, L, 1] -> [B, L]
            else:
                x = x.reshape(x.shape[0], -1)
        x = x.unsqueeze(-1)                # [B, L] -> [B, L, 1]
        y, _ = self.lstm(x)                # [B, L, hidden]
        y = y.flatten(1)                   # [B, L*hidden]
        return self.net(y)                 # [B, L]


def build(datanum: int, **kwargs: Any) -> nn.Module:
    """模型工厂函数。"""
    return RNNLSTM(datanum, **kwargs)
