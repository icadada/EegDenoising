"""通用工具模块。

本模块集中放置与具体模型/指标无关的小工具，主要包含三类内容：

1. **统一文件命名规则**（本项目的核心约定）
   所有产出文件的名字都由这里生成，保证命名风格一致，便于批量读取与对比。
   命名示例（EMG、512Hz、Simple_CNN、第 1 次实验）：:

        实验文件夹 : 测试集_EMG去噪_512HZ_试验次数_1
        Loss     : 测试集_EMG去噪_512HZ_Loss.npy            <- 训练过程 loss
        权重      : 测试集_EMG去噪_512HZ_Simple_CNN.pt       <- 最优模型权重
        断点      : 测试集_EMG去噪_512HZ_断点续训.pt          <- 用于恢复训练
        指标      : 测试集_EMG去噪_512HZ_MSE_指标.npy        <- 测试集 MSE 指标
                    训练集_EMG去噪_512HZ_MSE_指标.npy        <- 训练集 MSE 指标
                    验证集_EMG去噪_512HZ_MSE_指标.npy        <- 验证集 MSE 指标

2. **路径/IO 小工具**：建目录、存读 JSON、存读 npy、设备选择、随机种子。

3. **日志**：既打印到终端，也追加写入实验结果文件夹下的 ``训练日志.txt``。

注意：本模块只依赖 numpy，不依赖 torch，方便单独测试。
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# 一、统一命名规则
# --------------------------------------------------------------------------------------

#: 三个数据划分对应的中文名，同时也是指标文件名里的前缀
SPLIT_LABELS: Dict[str, str] = {
    "train": "训练集",
    "val": "验证集",
    "test": "测试集",
}

#: 噪声类型 -> 采样率（Hz）。EMG 用 512Hz，EOG 用 256Hz，与原版 EEGdenoiseNet 保持一致。
NOISE_SAMPLE_RATE: Dict[str, int] = {
    "EMG": 512,
    "EOG": 256,
}

#: 噪声类型 -> 该噪声对应的 EEG / 噪声数据文件名
NOISE_DATA_FILES: Dict[str, Dict[str, str]] = {
    "EMG": {"eeg": "EEG_all_epochs_512hz.npy", "noise": "EMG_all_epochs_512hz.npy"},
    "EOG": {"eeg": "EEG_all_epochs.npy", "noise": "EOG_all_epochs.npy"},
}


def noise_label(noise_type: str) -> str:
    """返回形如 ``EMG去噪_512HZ`` 的噪声标签。"""
    nt = noise_type.upper()
    if nt not in NOISE_SAMPLE_RATE:
        raise ValueError(f"不支持的噪声类型 {noise_type}，可选：{list(NOISE_SAMPLE_RATE)}")
    return f"{nt}去噪_{NOISE_SAMPLE_RATE[nt]}HZ"


def default_file_prefix(noise_type: str) -> str:
    """返回默认的统一文件前缀，形如 ``测试集_EMG去噪_512HZ``。

    该前缀用于 Loss / 权重 / 断点 / 实验文件夹名，按需求文档的命名示例设定。
    可以通过命令行 ``--文件前缀`` 覆盖（例如改成 ``训练集`` 开头）。
    """
    return f"{SPLIT_LABELS['test']}_{noise_label(noise_type)}"


def metric_filename(split: str, noise_type: str, metric: str) -> str:
    """生成指标 npy 文件名，例如 ``测试集_EMG去噪_512HZ_MSE_指标.npy``。"""
    return f"{SPLIT_LABELS[split]}_{noise_label(noise_type)}_{metric}_指标.npy"


def loss_filename(prefix: str) -> str:
    """生成训练过程 loss 文件名，例如 ``测试集_EMG去噪_512HZ_Loss.npy``。"""
    return f"{prefix}_Loss.npy"


def weight_filename(prefix: str, model_name: str) -> str:
    """生成最优模型权重文件名，例如 ``测试集_EMG去噪_512HZ_Simple_CNN.pt``。"""
    return f"{prefix}_{model_name}.pt"


def checkpoint_filename(prefix: str) -> str:
    """生成断点续训文件名，例如 ``测试集_EMG去噪_512HZ_断点续训.pt``。"""
    return f"{prefix}_断点续训.pt"


def experiment_dirname(prefix: str, trial: int) -> str:
    """生成实验结果文件夹名，例如 ``测试集_EMG去噪_512HZ_试验次数_1``。"""
    return f"{prefix}_试验次数_{int(trial)}"


def experiment_config_filename() -> str:
    """实验配置（命令行参数落盘）文件名。"""
    return "训练配置.json"


def experiment_summary_filename() -> str:
    """实验摘要（最优 epoch、整体指标等）文件名。"""
    return "实验摘要.json"


def train_history_filename(prefix: str) -> str:
    """每个 epoch 的 loss / 指标历史文件名（便于画曲线、恢复训练）。"""
    return f"{prefix}_训练历史.npy"


def split_data_dirname() -> str:
    """保存“已划分数据集”的子目录名，用于恢复训练时保证数据完全一致。"""
    return "划分数据"


# --------------------------------------------------------------------------------------
# 二、路径与 IO
# --------------------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """确保目录存在（递归创建），并返回该目录。"""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Any) -> None:
    """保存 JSON（UTF-8、缩进、保留中文），常用于保存命令行参数。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def load_json(path: str) -> Dict[str, Any]:
    """读取 JSON 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: str, array: np.ndarray) -> None:
    """保存 numpy 数组（npy 二进制格式）。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.save(path, array)


def load_npy(path: str) -> np.ndarray:
    """读取 numpy 数组（npy 二进制格式）。"""
    return np.load(path, allow_pickle=True)


def load_npy_dict(path: str) -> Dict[str, Any]:
    """读取用 ``np.save`` 保存的 python 字典（需要 allow_pickle=True）。"""
    obj = np.load(path, allow_pickle=True)
    # 0 维 object 数组需要 .item() 才能还原成字典
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    return obj


# --------------------------------------------------------------------------------------
# 三、设备、随机种子、时间
# --------------------------------------------------------------------------------------

def select_device(device_arg: str) -> str:
    """解析设备字符串。``auto`` 表示有 CUDA 就用 cuda:0，否则用 cpu。"""
    if device_arg is None or str(device_arg).lower() == "auto":
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return str(device_arg)


def set_random_seed(seed: Optional[int]) -> None:
    """统一设置 python / numpy / torch 的随机种子，保证实验可复现。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def timestamp() -> str:
    """返回 ``YYYY-MM-DD HH:MM:SS`` 形式的当前时间字符串。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    """同时输出到终端与日志文件的极简日志器。"""

    def __init__(self, log_file: Optional[str] = None) -> None:
        self.log_file = log_file
        if log_file:
            ensure_dir(os.path.dirname(os.path.abspath(log_file)))

    def __call__(self, *items: Any) -> None:
        text = " ".join(str(c) for c in items)
        line = f"[{timestamp()}] {text}"
        print(line, flush=True)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def next_trial_index(result_dir: str, prefix: str) -> int:
    """扫描结果目录，返回下一个可用的试验次数（用于避免覆盖之前的实验结果）。

    例如已存在 ``..._试验次数_1``、``..._试验次数_2``，则返回 3。
    """
    if not os.path.isdir(result_dir):
        return 1
    trials = []
    for name in os.listdir(result_dir):
        base = experiment_dirname(prefix, 0).rsplit("_", 1)[0]  # 去掉末尾数字后作为前缀
        if name.startswith(base + "_"):
            tail = name[len(base) + 1 :]
            if tail.isdigit():
                trials.append(int(tail))
    return (max(trials) + 1) if trials else 1


def print_table(rows: Iterable[Iterable[Any]], header: Optional[Iterable[Any]] = None) -> None:
    """按列对齐打印一个简单表格（用于终端汇报指标）。"""
    rows = [[str(c) for c in row] for row in rows]
    if header is not None:
        rows = [[str(c) for c in header]] + rows
    if not rows:
        return
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for idx, row in enumerate(rows):
        print("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if idx == 0 and header is not None:
            print("  " + "  ".join("-" * w for w in widths))


def get_script_root() -> str:
    """返回 ``code/`` 目录的绝对路径（本文件位于 ``code/utils.py``）。"""
    return os.path.dirname(os.path.abspath(__file__))


def get_project_root() -> str:
    """返回 ``EEG_denoise_New/`` 的绝对路径。"""
    return os.path.dirname(get_script_root())
