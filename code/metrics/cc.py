"""CC（相关系数）指标。

公式（论文式 (8)，逐样本）：

.. math::

    \\mathrm{CC} = \\frac{\\mathrm{cov}(\\hat{X}, X)}{\\sigma_{\\hat{X}}\\cdot\\sigma_{X}}

由于分子分母中的 ``1/T`` 会相互抵消，该式等价于**皮尔逊相关系数（Pearson r）**：

.. math::

    \\mathrm{CC} = \\frac{\\sum_i (x_i-\\bar{x})(\\hat{x}_i-\\bar{\\hat{x}})}
                        {\\sqrt{\\sum_i (x_i-\\bar{x})^2}\\sqrt{\\sum_i (\\hat{x}_i-\\bar{\\hat{x}})^2}}

取值范围为 ``[-1, 1]``，越接近 1 表示去噪波形与干净波形越相似（论文中说明为 0~1，实际使用时
去噪结果与干净信号高度正相关，一般都在 0 附近到 1 之间）。

边界处理：若干净信号或去噪信号在时间轴上为常数（标准差为 0），相关系数没有定义，返回 ``nan``。
"""
from __future__ import annotations

import numpy as np

NAME = "CC"
DISPLAY_NAME = "相关系数"
HIGHER_IS_BETTER = True


def compute(clean: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """逐样本计算皮尔逊相关系数。

    :param clean: 干净 EEG，形状 ``[N, T]``
    :param denoised: 去噪 EEG，形状 ``[N, T]``
    :return: 形状 ``[N]``，每个样本一个 CC 值
    """
    # 去均值
    clean_centered = clean - clean.mean(axis=1, keepdims=True)
    denoised_centered = denoised - denoised.mean(axis=1, keepdims=True)

    numerator = np.sum(clean_centered * denoised_centered, axis=1)
    denominator = np.sqrt(np.sum(clean_centered ** 2, axis=1) * np.sum(denoised_centered ** 2, axis=1))

    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        result[valid] = numerator[valid] / denominator[valid]
    # 浮点误差可能让结果略微超出 [-1, 1]，这里裁剪回来
    return np.clip(result, -1.0, 1.0)
