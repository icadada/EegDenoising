"""评价指标包（自动发现 + 命令行选择）。

设计约定
--------
本目录下**每个指标一个 .py 文件**，文件名即指标文件名（如 ``mse.py``）。
只要在文件里定义以下三个变量/函数，本包就会**自动发现**该指标并注册进
:data:`METRIC_REGISTRY`，无需修改任何其他代码：

.. code-block:: python

    NAME = "MSE"                 # 唯一标识：命令行选择用、文件名用
    DISPLAY_NAME = "均方误差"      # 仅用于显示
    HIGHER_IS_BETTER = False     # 决定绘图与“最优模型”方向的说明
    def compute(clean, denoised):  # 输入 [N, T]，返回每个样本一个值 [N]
        ...

因此以后要新增指标（例如 PSD 误差），只需往 ``metrics/`` 里扔一个 ``psd_error.py``。

本模块还提供把“逐样本指标”按 SNR 分档求平均的工具函数，用于绘制横轴为 SNR 的曲线。
"""
from __future__ import annotations

import importlib
import os
import pkgutil
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: 指标注册表：``{指标名: {"文件": 模块, "计算": 函数, "中文名": str, "是否越大越好": bool}}``
METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {}

#: 发现过程中的错误信息（便于排查新增指标文件写错的情况）
discovery_warnings: List[str] = []


def _discover_all_metrics() -> None:
    """扫描本包目录，把所有合法指标模块注册到 :data:`METRIC_REGISTRY`。"""
    if METRIC_REGISTRY:
        return
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for info in pkgutil.iter_modules([package_dir]):
        name = info.name
        if name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:  # 单个指标文件出错不应影响其他指标
            discovery_warnings.append(f"导入指标模块 {name}.py 失败：{exc}")
            continue

        # 优先使用英文标识符，缺失时回退到旧的中文属性名（保持向后兼容）
        metric_name = getattr(module, "NAME", getattr(module, "名称", None))
        compute = getattr(module, "compute", getattr(module, "计算", None))
        if not metric_name or not callable(compute):
            discovery_warnings.append(f"指标文件 {name}.py 缺少 名称 或 计算()，已跳过")
            continue
        METRIC_REGISTRY[str(metric_name)] = {
            "文件": module,
            "计算": compute,
            "中文名": getattr(module, "DISPLAY_NAME", getattr(module, "中文名", str(metric_name))),
            "是否越大越好": bool(
                getattr(module, "HIGHER_IS_BETTER", getattr(module, "是否越大越好", False))
            ),
        }


_discover_all_metrics()


def available_metrics() -> List[str]:
    """返回当前可用的指标名列表（已按固定顺序排序，便于命令行展示）。"""
    _discover_all_metrics()
    preferred_order = ["MSE", "SNR", "CC", "RRMSE"]
    remaining = [k for k in METRIC_REGISTRY if k not in preferred_order]
    return [k for k in preferred_order if k in METRIC_REGISTRY] + sorted(remaining)


def resolve_metric_selection(selection: Sequence[str]) -> List[str]:
    """把命令行传入的指标选择解析成具体的指标名列表。

    支持写法：

    - ``["全部"]`` / ``["all"]``：使用全部指标；
    - ``["MSE", "CC"]``：使用指定指标；
    - ``["MSE", "全部"]``：等价于全部。
    """
    _discover_all_metrics()
    selection = [str(s) for s in (selection or [])]
    if any(s.lower() in ("全部", "all", "*") for s in selection):
        return available_metrics()

    valid = available_metrics()
    upper_map = {k.upper(): k for k in valid}
    result: List[str] = []
    for s in selection:
        key = s.upper()
        if key not in upper_map:
            raise ValueError(f"未知指标 {s}，可选：{valid}")
        if upper_map[key] not in result:
            result.append(upper_map[key])
    if not result:
        raise ValueError("没有选择任何指标")
    return result


def compute_single_metric(metric_name: str, clean: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """计算单个指标的**逐样本**数值。

    :param metric_name: 指标注册表中的名称，如 ``"RRMSE"``
    :param clean: 干净 EEG，形状 ``[N, T]``
    :param denoised: 去噪 EEG，形状 ``[N, T]``
    :return: 形状 ``[N]`` 的逐样本指标值
    """
    _discover_all_metrics()
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"未知指标 {metric_name}，可选：{available_metrics()}")
    clean = np.asarray(clean, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    if clean.shape != denoised.shape:
        raise ValueError(f"干净与去噪数据形状不一致：{clean.shape} vs {denoised.shape}")
    return np.asarray(METRIC_REGISTRY[metric_name]["计算"](clean, denoised), dtype=np.float64)


def average_by_snr(
    per_sample_values: np.ndarray,
    snr_db: np.ndarray,
    snr_levels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """把逐样本指标按 SNR 档位分组求平均。

    - 验证集/测试集：``snr_db`` 恰好等于 10 个固定档位之一，分组是精确的；
    - 训练集：``snr_db`` 是 ``U(-7, 2)`` 的随机值，这里就近归入最接近的档位，
      因此训练集曲线表示“该 SNR 附近的平均表现”。

    :return: 形状 ``[len(档位)]`` 的数组；某档位没有样本时该位置为 ``nan``。
    """
    snr_levels = np.asarray(
        snr_levels if snr_levels is not None else np.linspace(-7.0, 2.0, num=10), dtype=np.float64
    )
    per_sample_values = np.asarray(per_sample_values, dtype=np.float64).reshape(-1)
    snr_db = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if per_sample_values.shape[0] != snr_db.shape[0]:
        raise ValueError(
            f"逐样本指标数量({per_sample_values.shape[0]})与 SNR 数量({snr_db.shape[0]})不一致"
        )

    # 就近归入档位
    nearest = np.abs(snr_db[:, None] - snr_levels[None, :]).argmin(axis=1)

    result = np.full(snr_levels.shape[0], np.nan, dtype=np.float64)
    for i in range(snr_levels.shape[0]):
        mask = nearest == i
        if np.any(mask):
            with np.errstate(invalid="ignore"):
                result[i] = np.nanmean(per_sample_values[mask])
    return result


def compute_metrics(
    metric_names: Sequence[str],
    clean: np.ndarray,
    denoised: np.ndarray,
    snr_db: np.ndarray,
    snr_levels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """一次计算多个指标，并返回 ``{指标名: 逐SNR档位的平均值[10]}``。"""
    result: Dict[str, np.ndarray] = {}
    for name in metric_names:
        per_sample = compute_single_metric(name, clean, denoised)
        result[name] = average_by_snr(per_sample, snr_db, snr_levels)
    return result


def overall_metric_value(metric_name: str, clean: np.ndarray, denoised: np.ndarray) -> float:
    """计算整个数据集上的**整体**指标值（用于柱状图/汇总表）。"""
    per_sample = compute_single_metric(metric_name, clean, denoised)
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(per_sample))
