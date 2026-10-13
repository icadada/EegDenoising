"""网络模型包（自动发现 + 统一接口）。

**统一输入输出约定（需求 6）**

本项目所有模型都必须满足：

- 输入：``[Batch_size, len]``，即二维张量（不加通道维、不加时间维）；
- 输出：``[Batch_size, len]``，与输入形状完全一致。

模型内部需要 ``[B, 1, L]``（Conv1d）或 ``[B, L, 1]``（LSTM）时，自己在 ``forward`` 里
``unsqueeze``，输出前再还原成 ``[B, L]``。这样训练/评估代码不需要为任何模型写特例。

**如何新增一个模型（会被自动发现，无需改训练代码）**

在本目录新建 ``MyNet.py``，写入：

.. code-block:: python

    import torch
    from torch import nn

    NETWORK_NAME = "MyNet"                 # 可选，默认使用文件名
    DEFAULT_INIT_ARGS = {"width": 32}      # 可选，本模型的默认初始化参数
    IO_SHAPE = "[B, L] -> [B, L]"          # 可选，仅作文档说明

    class MyNetModule(nn.Module):
        def __init__(self, datanum, width=32, **kwargs):
            super().__init__()
            ...
        def forward(self, x):          # x: [B, L]
            ...
            return y                   # [B, L]

    def build(datanum, **kwargs):        # 推荐入口名
        return MyNetModule(datanum, **kwargs)

    # 或者直接定义与文件名同名的函数：
    # def MyNet(datanum, **kwargs): return MyNetModule(datanum, **kwargs)

之后命令行 ``--network MyNet`` 即可直接训练，``--model-arg width=64`` 可以覆盖默认参数。
"""
from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from typing import Any, Callable, Dict, List

#: 模型注册表：``{模型名: {"构建": 可调用对象, "默认初始化参数": dict, "文件": 模块}}``
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

#: 发现过程中的警告信息（例如某个模型文件写错），便于排查
discovery_warnings: List[str] = []

#: 这些模块名不当作模型处理
_IGNORED_MODULES = {"__init__"}


def _get_builder(module: Any, model_name: str) -> Callable[..., Any] | None:
    """从模块中找出“构建模型”的可调用对象。

    依次尝试：``build`` -> 与文件名同名的函数/类 -> ``构建`` -> ``MODEL`` -> ``NET``。
    """
    for candidate in ("build", "构建", model_name, "MODEL", "NET"):
        if not hasattr(module, candidate):
            continue
        obj = getattr(module, candidate)
        if callable(obj) and not inspect.ismodule(obj):
            return obj
    return None


def _discover_all_models() -> None:
    """扫描本包目录，把每个 .py 文件注册为一个可用模型。"""
    if MODEL_REGISTRY:
        return
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for info in pkgutil.iter_modules([package_dir]):
        filename = info.name
        if filename.startswith("_") or filename in _IGNORED_MODULES:
            continue
        try:
            module = importlib.import_module(f"{__name__}.{filename}")
        except Exception as exc:  # 单个模型报错不影响其他模型
            discovery_warnings.append(f"导入模型文件 {filename}.py 失败：{exc}")
            continue

        # 优先使用英文标识符，缺失时回退到旧的中文属性名（保持向后兼容）
        model_name = str(
            getattr(module, "NETWORK_NAME", getattr(module, "网络名称", filename))
        )
        builder = _get_builder(module, model_name)
        if builder is None:
            discovery_warnings.append(
                f"模型文件 {filename}.py 中找不到构建函数（需要 build() 或与文件同名的函数/类），已跳过"
            )
            continue
        MODEL_REGISTRY[model_name] = {
            "构建": builder,
            "默认初始化参数": dict(
                getattr(module, "DEFAULT_INIT_ARGS", getattr(module, "默认初始化参数", {})) or {}
            ),
            "文件": module,
            "来源": f"{filename}.py",
        }


_discover_all_models()


def available_networks() -> List[str]:
    """返回自动发现的模型名列表。"""
    _discover_all_models()
    return list(MODEL_REGISTRY.keys())


def model_default_args(model_name: str) -> Dict[str, Any]:
    """返回某个模型文件里声明的默认初始化参数（没有则返回空字典）。"""
    _discover_all_models()
    if model_name not in MODEL_REGISTRY:
        return {}
    return dict(MODEL_REGISTRY[model_name]["默认初始化参数"])


def create_model(model_name: str, datanum: int, **kwargs: Any) -> Any:
    """按名字创建模型实例，并把额外参数透传给模型构造函数（需求 7）。

    :param model_name: ``network/`` 下自动发现的模型名
    :param datanum: 信号长度 len（EMG 为 1024，EOG 为 512）
    :param kwargs: 透传给模型构造函数的初始化参数
    """
    _discover_all_models()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知网络 {model_name}，可用：{available_networks()}")
    builder = MODEL_REGISTRY[model_name]["构建"]
    return builder(datanum, **kwargs)
