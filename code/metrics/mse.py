"""MSE（均方误差）指标。

公式（逐样本，T 为时间点数）：

.. math::

    \\mathrm{MSE} = \\frac{1}{T}\\sum_{i=1}^{T}(\\hat{x}_i - x_i)^2

其中 :math:`x` 为干净 EEG，:math:`\\hat{x}` 为去噪 EEG。MSE 越小说明去噪越接近干净信号。
"""
from __future__ import annotations

import numpy as np

#: 指标名称（命令行选择与文件名使用）
NAME = "MSE"
#: 中文名，仅用于显示
DISPLAY_NAME = "均方误差"
#: 是否越大越好（False 表示越小越好）
HIGHER_IS_BETTER = False


def compute(clean: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """逐样本计算 MSE。

    :param clean: 干净 EEG，形状 ``[N, T]``
    :param denoised: 去噪 EEG，形状 ``[N, T]``
    :return: 形状 ``[N]``，每个样本一个 MSE 值
    """
    error = denoised - clean
    return np.mean(error * error, axis=1)
