"""通用工具模块。

本模块集中放置与具体模型/指标无关的小工具，主要包含三类内容：

1. **统一文件命名规则 + 结果目录结构**（本项目的核心约定）
   所有产出文件的名字与目录都由这里生成，保证命名风格一致，便于批量读取与对比。

   ``results/`` 与 ``code/``、``data/`` **同级**，目录结构如下（EMG、512Hz、Simple_CNN、K=10）::

       results/
       └── Simple_CNN/                                  <- ① 模型名
           └── EMG去噪_512HZ/                            <- ② 噪声类型（<噪声>去噪_<采样率>HZ）
               ├── 指标汇总_均值标准差中位数IQR.json        <- K 次实验各指标的 Mean/STD/Median/IQR
               ├── EMG_显著性检验.json / .csv             <- ③ ANOVA + Tukey HSD 结果
               ├── 图像/                                  <- ④ 显著性标注的箱线图
               ├── 试验次数_1/                            <- ⑤ 第 1 次独立重复实验
               │   ├── 权重/        测试集_EMG去噪_512HZ_Simple_CNN.pt / *_断点续训.pt
               │   ├── 指标/        *_Loss.npy / <划分>_EMG去噪_512HZ_<指标>_指标.npy
               │   ├── 图像/        *.png
               │   ├── 训练配置/    训练配置.json / 实验摘要.json
               │   ├── 训练日志.txt
               │   └── 划分数据/    划分信息.json / *.npy
               └── 试验次数_2/ ... 试验次数_K/

   命名示例（EMG、512Hz、Simple_CNN、第 1 次实验）::

        Loss     : 测试集_EMG去噪_512HZ_Loss.npy            <- 训练过程 loss
        权重      : 测试集_EMG去噪_512HZ_Simple_CNN.pt       <- 最优模型权重
        断点      : 测试集_EMG去噪_512HZ_断点续训.pt          <- 用于恢复训练
        指标      : 测试集_EMG去噪_512HZ_MSE_指标.npy        <- 测试集 MSE 指标
                    训练集_EMG去噪_512HZ_MSE_指标.npy        <- 训练集 MSE 指标
                    验证集_EMG去噪_512HZ_MSE_指标.npy        <- 验证集 MSE 指标

2. **路径/IO 小工具**：建目录、存读 JSON、存读 npy、设备选择、随机种子、结果目录布局。

3. **日志**：既打印到终端，也追加写入结果文件夹下的 ``训练日志.txt``。

注意：本模块只依赖 numpy，不依赖 torch，方便单独测试。
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
# 一（补）、结果目录结构（模型 / 噪声类型 / 试验次数 三层）
# --------------------------------------------------------------------------------------

#: 单次实验文件夹内部的子目录名（== 需求 2：权重、指标、图像、配置、划分数据分开存放）
SUBDIR_WEIGHTS = "权重"
SUBDIR_METRICS = "指标"
SUBDIR_PLOTS = "图像"
SUBDIR_CONFIG = "训练配置"
SUBDIR_SPLIT = "划分数据"

#: 汇总文件名
AGGREGATE_FILENAME = "指标汇总_均值标准差中位数IQR.json"


def clean_subdir_names() -> Tuple[str, ...]:
    """返回需要在“清空实验结果”时一并删除的、可自动重建的子目录名。"""
    return (SUBDIR_WEIGHTS, SUBDIR_METRICS, SUBDIR_PLOTS, SUBDIR_SPLIT)


def results_root_default() -> str:
    """默认结果根目录：``<项目根>/results``（与 ``code/``、``data/`` 同级）。"""
    return os.path.join(get_project_root(), "results")


def resolve_result_root(result_dir: Optional[str]) -> str:
    """解析结果根目录。

    - 传入 ``None`` / 空字符串：返回默认的 ``<项目根>/results``（与 code、data 同级）；
    - 传入相对路径：相对于**项目根目录**解析（而不是当前工作目录），
      这样从任何地方执行 ``python code/main.py`` 结果都落在同一个地方；
    - 传入绝对路径：原样返回。
    """
    text = str(result_dir).strip() if result_dir is not None else ""
    if not text:
        return results_root_default()
    if os.path.isabs(text):
        return os.path.normpath(text)
    return os.path.normpath(os.path.join(get_project_root(), text))


def model_results_dir(result_dir: str, model_name: str) -> str:
    """返回某模型的结果目录：``<结果根>/<模型名>``。"""
    return os.path.join(result_dir, str(model_name))


def noise_label_dir(noise_type: str) -> str:
    """返回噪声文件夹名，形如 ``EMG去噪_512HZ``（``<噪声类型>去噪_<采样率>HZ``）。"""
    return noise_label(noise_type)


def noise_results_dir(result_dir: str, model_name: str, noise_type: str) -> str:
    """返回某模型某噪声类型的结果目录：``<结果根>/<模型名>/<噪声>去噪_<采样率>HZ``。"""
    return os.path.join(model_results_dir(result_dir, model_name), noise_label_dir(noise_type))


def make_layout(result_dir: str, model_name: str, noise_type: str, trial: int) -> "ResultLayout":
    """创建（并返回）一次独立重复实验的结果目录布局对象。"""
    return ResultLayout(
        result_dir=result_dir, model_name=str(model_name), noise_type=noise_type, trial=int(trial)
    )


@dataclass
class ResultLayout:
    """一次独立重复实验（模型 + 噪声类型 + 第 N 次实验）的完整目录布局。

    目录结构::

        <结果根>/<模型名>/<噪声>去噪_<采样率>HZ/试验次数_<N>/
        ├── 权重/        最优权重与断点续训文件
        ├── 指标/        Loss / 训练历史 / 指标 npy / SNR档位.npy
        ├── 图像/        自动生成的 *.png
        ├── 训练配置/     训练配置.json / 实验摘要.json / 各划分整体指标.json
        ├── 训练日志.txt
        └── 划分数据/     划分好的数据集
    """

    result_dir: str
    model_name: str
    noise_type: str
    trial: int

    # ---------------- 目录 ----------------
    @property
    def noise_dir(self) -> str:
        """噪声类型目录：``<结果根>/<模型名>/<噪声>去噪_<采样率>HZ``。"""
        return noise_results_dir(self.result_dir, self.model_name, self.noise_type)

    @property
    def model_dir(self) -> str:
        """模型目录：``<结果根>/<模型名>``。"""
        return model_results_dir(self.result_dir, self.model_name)

    @property
    def trial_dirname(self) -> str:
        """单次实验文件夹名，形如 ``试验次数_1``。"""
        return format_trial_name(self.trial)

    @property
    def experiment_dir(self) -> str:
        """单次实验文件夹（trial 级目录）。"""
        return os.path.join(self.noise_dir, self.trial_dirname)

    @property
    def weights_dir(self) -> str:
        return os.path.join(self.experiment_dir, SUBDIR_WEIGHTS)

    @property
    def metrics_dir(self) -> str:
        return os.path.join(self.experiment_dir, SUBDIR_METRICS)

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.experiment_dir, SUBDIR_PLOTS)

    @property
    def config_dir(self) -> str:
        return os.path.join(self.experiment_dir, SUBDIR_CONFIG)

    @property
    def split_dir(self) -> str:
        return os.path.join(self.experiment_dir, SUBDIR_SPLIT)

    @property
    def significance_dir(self) -> str:
        """显著性检验结果目录：``<噪声目录>/统计分析``。"""
        return os.path.join(self.noise_dir, "统计分析")

    @property
    def noise_plots_dir(self) -> str:
        """噪声类型级别的图像目录（存放箱线图等跨实验图像）。"""
        return os.path.join(self.noise_dir, "图像")

    # ---------------- 文件 ----------------
    @property
    def prefix(self) -> str:
        """统一文件前缀，形如 ``测试集_EMG去噪_512HZ``。"""
        return default_file_prefix(self.noise_type)

    @property
    def weights_path(self) -> str:
        return os.path.join(self.weights_dir, weight_filename(self.prefix, self.model_name))

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.weights_dir, checkpoint_filename(self.prefix))

    @property
    def loss_path(self) -> str:
        return os.path.join(self.metrics_dir, loss_filename(self.prefix))

    @property
    def history_path(self) -> str:
        return os.path.join(self.metrics_dir, train_history_filename(self.prefix))

    @property
    def snr_levels_path(self) -> str:
        return os.path.join(self.metrics_dir, "SNR档位.npy")

    @property
    def config_path(self) -> str:
        return os.path.join(self.config_dir, experiment_config_filename())

    @property
    def summary_path(self) -> str:
        return os.path.join(self.config_dir, experiment_summary_filename())

    @property
    def overall_metrics_path(self) -> str:
        return os.path.join(self.config_dir, "各划分整体指标.json")

    @property
    def log_path(self) -> str:
        return os.path.join(self.experiment_dir, "训练日志.txt")

    @property
    def aggregate_path(self) -> str:
        """K 次实验汇总文件（Mean/STD/Median/IQR），位于噪声类型目录下。"""
        return os.path.join(self.noise_dir, AGGREGATE_FILENAME)

    def metric_path(self, split: str, metric: str, per_sample: bool = False) -> str:
        """指标 npy 路径，例如 ``指标/测试集_EMG去噪_512HZ_MSE_指标.npy``。"""
        name = metric_filename(split, self.noise_type, metric)
        if per_sample:
            name = name.replace("_指标.npy", "_逐样本指标.npy")
        return os.path.join(self.metrics_dir, name)

    def ensure_dirs(self) -> "ResultLayout":
        """创建全部子目录。"""
        for directory in (
            self.weights_dir, self.metrics_dir, self.plots_dir, self.config_dir, self.split_dir
        ):
            ensure_dir(directory)
        return self

    def describe(self) -> str:
        """返回人可读的目录说明（写进日志用）。"""
        return (
            f"模型目录：{self.model_dir}\n"
            f"噪声目录：{self.noise_dir}\n"
            f"本次实验：{self.experiment_dir}"
        )


def format_trial_name(trial: int) -> str:
    """试验次数的显示名，形如 ``试验次数_1``。"""
    return f"试验次数_{int(trial)}"


def parse_trial_name(name: str) -> Optional[int]:
    """从 ``实验文件夹名`` 或 ``试验次数_3`` 中解析出试验编号，失败返回 None。"""
    text = str(name)
    for prefix in ("试验次数_", "_试验次数_"):
        index = text.rfind(prefix)
        if index >= 0:
            tail = text[index + len(prefix):]
            if tail.isdigit():
                return int(tail)
    return None


def find_result_layouts(result_dir: str) -> List[ResultLayout]:
    """扫描结果目录，返回全部 ``试验次数_N`` 实验的布局对象（按路径排序）。

    兼容两种结构：

    - 新结构：``<结果根>/<模型名>/<噪声>去噪_<采样率>HZ/试验次数_N/``
    - 旧结构：``<结果根>/<模型名>/试验次数_N/``

    参数既可以是**结果根目录**，也可以是**单个模型目录**（``<结果根>/<模型名>``），
    两种写法都能正确找出其下的全部实验。
    """
    layouts: List[ResultLayout] = []
    seen = set()

    def add(result_root: str, model_name: str, noise_type: str, trial: int) -> None:
        key = (os.path.abspath(result_root), model_name, noise_type, int(trial))
        if key in seen:
            return
        seen.add(key)
        layouts.append(ResultLayout(result_root, model_name, noise_type, int(trial)))

    def scan_trials(result_root: str, model_name: str, noise_type: str, container: str) -> None:
        """扫描 ``container`` 下的 ``试验次数_N`` 目录并登记。"""
        try:
            names = sorted(os.listdir(container))
        except OSError:
            return
        for name in names:
            trial = parse_trial_name(name) if name.startswith("试验次数_") else None
            if trial is not None and os.path.isdir(os.path.join(container, name)):
                add(result_root, model_name, noise_type, trial)

    if not os.path.isdir(result_dir):
        return layouts

    root = os.path.abspath(result_dir)

    # 情况 0：传进来的本身就是“噪声目录”（<模型>/<噪声>去噪_<采样率>HZ）
    root_noise_type = _noise_type_from_dirname(os.path.basename(root))
    if root_noise_type:
        scan_trials(os.path.dirname(os.path.dirname(root)), os.path.basename(os.path.dirname(root)),
                    root_noise_type, root)
        return layouts

    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        noise_type = _noise_type_from_dirname(name)
        if noise_type:
            # 传进来的就是“模型目录”（root），其下直接是 <噪声>去噪_<采样率>HZ
            model_name = os.path.basename(root)
            result_root = os.path.dirname(root)
            scan_trials(result_root, model_name, noise_type, path)
            continue
        # 传进来的是“结果根目录”，name 是模型名
        for sub_name in sorted(os.listdir(path)):
            sub_path = os.path.join(path, sub_name)
            if not os.path.isdir(sub_path):
                continue
            sub_noise_type = _noise_type_from_dirname(sub_name)
            if sub_noise_type:
                scan_trials(root, name, sub_noise_type, sub_path)
            else:
                # 旧结构：<模型>/试验次数_N（没有噪声层级，按目录名猜噪声类型）
                trial = parse_trial_name(sub_name)
                if trial is not None:
                    add(root, name, _noise_type_from_dirname(root) or "EMG", trial)
    return layouts


def _noise_type_from_dirname(dirname: str) -> Optional[str]:
    """从 ``EMG去噪_512HZ`` 这样的目录名里还原噪声类型（``EMG`` / ``EOG``）。"""
    upper = str(dirname).upper()
    for noise_type in NOISE_SAMPLE_RATE:
        if upper.startswith(noise_type):
            return noise_type
    return None


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


def find_in_run_dir(experiment_dir: str, filename: str) -> Optional[str]:
    """在单次实验文件夹里查找文件。

    由于需求 2 把结果分到了 ``权重/指标/图像/训练配置/划分数据`` 等子目录，
    这里先在已知子目录里找，再退回实验文件夹根目录（兼容旧结构）。
    """
    for subdir in clean_subdir_names() + (SUBDIR_CONFIG,):
        candidate = os.path.join(experiment_dir, subdir, filename)
        if os.path.isfile(candidate):
            return candidate
    candidate = os.path.join(experiment_dir, filename)
    if os.path.isfile(candidate):
        return candidate
    return None


def find_summary_path(experiment_dir: str) -> Optional[str]:
    """查找 ``实验摘要.json``（新结构在 ``训练配置/`` 下，旧结构在根目录）。"""
    return find_in_run_dir(experiment_dir, experiment_summary_filename())


def find_overall_metrics_path(experiment_dir: str) -> Optional[str]:
    """查找 ``各划分整体指标.json``（由 trainer 落盘，K 次实验汇总时读取）。"""
    return find_in_run_dir(experiment_dir, "各划分整体指标.json")


def find_loss_path(experiment_dir: str, prefix: str) -> Optional[str]:
    """查找 ``<前缀>_Loss.npy``。"""
    return find_in_run_dir(experiment_dir, loss_filename(prefix))


def find_train_history_path(experiment_dir: str, prefix: str) -> Optional[str]:
    """查找 ``<前缀>_训练历史.npy``。"""
    return find_in_run_dir(experiment_dir, train_history_filename(prefix))


def find_metric_path(experiment_dir: str, split: str, noise_type: str, metric: str,
                     per_sample: bool = False) -> Optional[str]:
    """查找某个划分/指标的 npy 文件。"""
    name = metric_filename(split, noise_type, metric)
    if per_sample:
        name = name.replace("_指标.npy", "_逐样本指标.npy")
    return find_in_run_dir(experiment_dir, name)


def read_overall_metrics(experiment_dir: str) -> Dict[str, Dict[str, float]]:
    """读取一次实验在三个划分上的整体指标。

    优先读 ``训练配置/各划分整体指标.json``（内容最完整），其次退回 ``实验摘要.json``。
    """
    path = find_overall_metrics_path(experiment_dir)
    if path:
        try:
            data = load_json(path)
            if isinstance(data, dict) and data:
                return {k: dict(v) for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            pass
    path = find_summary_path(experiment_dir)
    if path:
        try:
            summary = load_json(path)
            overall = summary.get("各划分整体指标", {}) or {}
            return {k: dict(v) for k, v in overall.items() if isinstance(v, dict)}
        except Exception:
            pass
    return {}


def read_trial_metrics(experiment_dir: str, metric: str, split: str = "test") -> Optional[float]:
    """读取一次实验在指定划分上的某个指标的整体值（K 次统计与方差分析用）。

    优先读落盘的整体指标；若缺失，则用 ``<划分>_<噪声>去噪_<采样率>HZ_<指标>_逐样本指标.npy``
    就地重新计算（整体值 = 逐样本指标的 nanmean）。
    """
    overall = read_overall_metrics(experiment_dir)
    value = (overall.get(split) or {}).get(metric)
    if value is not None and np.isfinite(float(value)):
        return float(value)

    noise_type = detect_noise_type(experiment_dir)
    if noise_type is None:
        return None if value is None else float(value)
    path = find_metric_path(experiment_dir, split, noise_type, metric, per_sample=True)
    if path:
        values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if values.size:
            with np.errstate(invalid="ignore"):
                return float(np.nanmean(values))
    if value is None:
        return None
    return float(value)


def detect_noise_type(experiment_dir: str) -> Optional[str]:
    """推断一次实验用的噪声类型：先看目录名，再看实验摘要。"""
    upper = os.path.abspath(str(experiment_dir)).upper()
    for noise_type in NOISE_SAMPLE_RATE:
        if f"{noise_type}去噪_" in upper:
            return noise_type
    path = find_summary_path(experiment_dir)
    if path:
        try:
            noise_type = str(load_json(path).get("噪声类型", "")).upper()
            if noise_type in NOISE_SAMPLE_RATE:
                return noise_type
        except Exception:
            pass
    return None


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

    扫描时**递归**查找 ``试验次数_N`` 目录，因此新旧两种目录结构都能正确递增。
    """
    trials = []
    for root, dirnames, _ in os.walk(result_dir):
        if os.path.abspath(root).count(os.sep) - os.path.abspath(result_dir).count(os.sep) > 4:
            dirnames[:] = []
            continue
        for name in dirnames:
            if name.startswith("试验次数_"):
                tail = name[len("试验次数_"):]
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
