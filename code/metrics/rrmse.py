"""RRMSE（相对均方根误差）指标。

公式（论文式 (9)、(10)，逐样本）：

.. math::

    \\mathrm{RRMSE} = \\frac{\\mathrm{rms}(\\hat{X} - X)}{\\mathrm{rms}(X)},\\qquad
    \\mathrm{rms}(\\hat{X}-X)=\\sqrt{\\frac{1}{T}\\sum_{i=1}^{T}(\\hat{x}_i-x_i)^2}

RRMSE 越小说明相对误差越小、去噪效果越好；等于 0 表示完全无误差。

边界处理：若干净信号的 rms 为 0，返回 ``nan``。
"""
from __future__ import annotations

import numpy as np

NAME = "RRMSE"
DISPLAY_NAME = "相对均方根误差"
HIGHER_IS_BETTER = False


def compute(clean: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """逐样本计算 RRMSE。

    :param clean: 干净 EEG，形状 ``[N, T]``
    :param denoised: 去噪 EEG，形状 ``[N, T]``
    :return: 形状 ``[N]``，每个样本一个 RRMSE 值
    """
    error_rms = np.sqrt(np.mean((denoised - clean) ** 2, axis=1))
    clean_rms = np.sqrt(np.mean(clean ** 2, axis=1))

    result = np.full(error_rms.shape, np.nan, dtype=np.float64)
    valid = clean_rms > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        result[valid] = error_rms[valid] / clean_rms[valid]
    return result
