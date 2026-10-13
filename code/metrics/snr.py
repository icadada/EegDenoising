"""SNR（信噪比，dB）指标。

公式（逐样本，T 为时间点数）：

.. math::

    \\mathrm{SNR(dB)} = 10\\log_{10}\\frac{\\sum_{i=1}^{T} x_i^2}
                                          {\\sum_{i=1}^{T} (x_i - \\hat{x}_i)^2}

含义：干净 EEG 的能量 与 **残余噪声**（干净减去去噪）能量 之比。
注意这里衡量的是**去噪后残留了多少噪声**，因此数值越大说明去噪越彻底。

边界处理：当场点全部为 0（干净信号无能量）时返回 ``nan``；当残余噪声为 0（完美去噪）时
返回 ``inf``（无穷大信噪比）。
"""
from __future__ import annotations

import numpy as np

NAME = "SNR"
DISPLAY_NAME = "信噪比(dB)"
HIGHER_IS_BETTER = True


def compute(clean: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """逐样本计算 SNR(dB)。

    :param clean: 干净 EEG，形状 ``[N, T]``
    :param denoised: 去噪 EEG，形状 ``[N, T]``
    :return: 形状 ``[N]``，每个样本一个 SNR 值（dB）
    """
    signal_energy = np.sum(clean * clean, axis=1)
    residual_noise_energy = np.sum((clean - denoised) ** 2, axis=1)

    result = np.full(signal_energy.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = signal_energy > 0
        result[valid] = 10.0 * np.log10(signal_energy[valid] / residual_noise_energy[valid])
        # 残余噪声为 0 表示完美去噪 -> 正无穷
        perfect = valid & (residual_noise_energy == 0)
        result[perfect] = np.inf
    return result
