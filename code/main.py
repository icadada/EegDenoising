#!/usr/bin/env python3
"""训练入口（EEGdenoiseNet 去噪网络重构版）。

本入口同时负责：

1. **K 次独立重复实验**（不是 K 折交叉验证）：同一模型重复训练 K 次，每次实验都用不同的
   随机种子重新随机混合生成半仿真数据、重新划分训练/验证/测试集，并随机初始化网络权重；
   每次实验结束后在当次测试集上计算指标，得到 K 个独立数值；
2. **结果目录分层**：``results/<模型名>/<噪声>去噪_<采样率>HZ/试验次数_N/``，单次实验内部再按
   权重 / 指标 / 图像 / 训练配置 / 划分数据 分子目录保存；
3. **K 次实验的统计汇总**：均值、标准差、中位数、四分位距写入噪声类型目录；
4. **多个模型跑完后自动做单因素方差分析（ANOVA）+ Tukey HSD 事后检验**，并画带显著性标注的箱线图。

常用命令
--------
默认（Simple_CNN + EMG，K=10 次独立重复实验）::

    python code/main.py

一次跑完五个模型（每个模型 K 次独立重复实验，最后自动做方差分析）::

    python code/main.py --models fcNN Simple_CNN Complex_CNN RNN_lstm Novel_CNN \
        --noise-type EMG --repeat-k 10 --device cuda:0

选择最优模型的准则（默认验证集 MSE）::

    python code/main.py --select-split val --select-metric MSE
    python code/main.py --select-split test --select-metric CC

从断点恢复训练（需求 10）::

    python code/main.py --resume 测试集_EMG去噪_512HZ_试验次数_1 --epochs 80

只对已有实验重新绘图::

    python code/main.py --plot-only Simple_CNN/EMG去噪_512HZ
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# 把 code/ 目录加入模块搜索路径，保证「python code/main.py」与「cd code && python main.py」都能运行
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np  # noqa: E402

from config import (  # noqa: E402
    build_parser,
    config_to_dict,
    model_default_hyperparams,
    runtime_env_info,
    save_config,
)
from data_prepare import DataSplit, load_split, prepare_data, save_split  # noqa: E402
from metrics import available_metrics, discovery_warnings as metric_discovery_warnings  # noqa: E402
from utils import (  # noqa: E402
    AGGREGATE_FILENAME,
    NOISE_DATA_FILES,
    SPLIT_LABELS,
    Logger,
    ResultLayout,
    clean_subdir_names,
    default_file_prefix,
    ensure_dir,
    experiment_config_filename,
    find_result_layouts,
    find_summary_path,
    load_json,
    load_npy,
    make_layout,
    noise_results_dir,
    print_table,
    read_trial_metrics,
    resolve_result_root,
    save_json,
    set_random_seed,
    weight_filename,
)

#: 汇总/统计分析产生的文件名（清空旧结果时一并删除）
_SIGNIFICANCE_SUFFIX_JSON = "显著性检验.json"
_SIGNIFICANCE_SUFFIX_CSV = "显著性检验.csv"
_AGGREGATE_SPLIT_FILENAME = "该次实验汇总.json"
_DEFAULT_NOISE_PLOT_DIRNAME = "图像"
_STATS_SUBDIR = "统计分析"


# --------------------------------------------------------------------------------------
# 定位实验 / 结果目录
# --------------------------------------------------------------------------------------

def _looks_like_trial_dir(path: str) -> bool:
    """判断目录是否是一次实验（试验次数_N）目录。"""
    if not os.path.isdir(path):
        return False
    from utils import parse_trial_name

    if parse_trial_name(os.path.basename(os.path.normpath(path))) is None:
        return False
    if os.path.isfile(os.path.join(path, "训练日志.txt")):
        return True
    if find_summary_path(path):
        return True
    for subdir in clean_subdir_names():
        if os.path.isdir(os.path.join(path, subdir)):
            return True
    return False


def _layout_from_trial_dir(trial_dir: str, noise_type: str) -> ResultLayout:
    """根据 ``<结果根>/<模型>/<噪声目录>/试验次数_N`` 还原出一个 ResultLayout。

    噪声类型以**父目录名**（``EMG去噪_512HZ``）为准；若父目录不像噪声目录，
    则回退到调用方给出的 ``noise_type``。
    """
    from utils import _noise_type_from_dirname, parse_trial_name

    trial_dir = os.path.abspath(trial_dir)
    noise_dir = os.path.dirname(trial_dir)
    model_dir = os.path.dirname(noise_dir)
    result_dir = os.path.dirname(model_dir)
    inferred = _noise_type_from_dirname(os.path.basename(noise_dir))
    trial = parse_trial_name(os.path.basename(trial_dir)) or 1
    return ResultLayout(result_dir, os.path.basename(model_dir), inferred or noise_type, trial)


def locate_target_layouts(
    specified: str,
    result_dir: str,
    file_prefix: Optional[str] = None,
    noise_type: str = "EMG",
) -> List[ResultLayout]:
    """把 ``--resume`` / ``--plot-only`` 的取值解析成一组实验布局。

    支持：完整路径 / ``试验次数_N`` 文件夹名 / 模型文件夹（如 ``Simple_CNN``）/
    模型+噪声（如 ``Simple_CNN/EMG去噪_512HZ``）/ ``latest`` / ``最新``。
    """
    specified = str(specified)
    noise_type = str(noise_type or "EMG").upper()

    if specified.lower() in ("latest", "auto", "最新"):
        layouts = find_result_layouts(result_dir)
        if not layouts:
            raise FileNotFoundError(f"结果目录 {result_dir} 下没有找到任何实验")
        latest_model = max(
            {layout.noise_dir for layout in layouts}, key=lambda p: os.path.getmtime(p)
        )
        return [layout for layout in layouts if layout.noise_dir == latest_model]

    candidates = [specified, os.path.join(result_dir, specified)]
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        candidate = os.path.abspath(candidate)
        if _looks_like_trial_dir(candidate):
            return [_layout_from_trial_dir(candidate, noise_type)]
        # 模型目录 / 模型+噪声目录 / 结果根目录：find_result_layouts 都能正确处理
        layouts = find_result_layouts(candidate)
        layouts = [layout for layout in layouts if layout.noise_type == noise_type]
        if layouts:
            return layouts
    raise FileNotFoundError(
        f"找不到实验文件夹：{specified}\n（已尝试：{specified} 与 {os.path.join(result_dir, specified)}）"
    )


# --------------------------------------------------------------------------------------
# 数据与权重
# --------------------------------------------------------------------------------------

def load_raw_data(config: Any) -> Tuple[np.ndarray, np.ndarray]:
    """按噪声类型读取干净 EEG 与噪声 npy。"""
    noise_type = config.noise_type.upper()
    file = NOISE_DATA_FILES[noise_type]
    eeg_path = os.path.join(config.data_dir, file["eeg"])
    noise_path = os.path.join(config.data_dir, file["noise"])
    for path in (eeg_path, noise_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到数据文件：{path}\n请用 --data-dir 指定正确的数据目录。"
            )
    return load_npy(eeg_path).astype(np.float64), load_npy(noise_path).astype(np.float64)


def build_split(config: Any, EEG_all: np.ndarray, noise_all: np.ndarray, logger: Any) -> DataSplit:
    """调用数据准备模块生成划分。"""
    seed = None if (config.seed is None or int(config.seed) < 0) else int(config.seed)
    logger(
        f"开始生成数据划分：噪声类型={config.noise_type}，combin_num={config.combin_num}，"
        f"train_per={config.train_per}，随机种子={seed}"
    )
    data = prepare_data(
        EEG_all=EEG_all,
        noise_all=noise_all,
        combin_num=int(config.combin_num),
        train_per=float(config.train_per),
        noise_type=config.noise_type,
        seed=seed,
        max_train_samples=config.max_train_samples,
    )
    logger(f"数据划分完成：{data.describe()}")
    return data


def resolve_weights_path(config: Any, experiment_dir: Optional[str]) -> str:
    """解析 ``--resume-weights`` 指定的权重文件路径。"""
    value = str(config.resume_weights)
    if value.lower() in ("best", "最优"):
        if not experiment_dir:
            raise ValueError("--resume-weights best 需要同时给出 --resume 或在结果目录中定位实验")
        layout = _layout_from_trial_dir(experiment_dir, config.noise_type)
        candidate = layout.weights_path
        if os.path.isfile(candidate):
            return candidate
        legacy = os.path.join(experiment_dir, weight_filename(config.file_prefix, config.network))
        if os.path.isfile(legacy):
            return legacy
        raise FileNotFoundError(f"找不到最优权重：{candidate}")
    if os.path.isfile(value):
        return value
    if experiment_dir:
        for base in (experiment_dir, os.path.join(experiment_dir, "权重")):
            candidate = os.path.join(base, value)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"找不到权重文件：{value}")


def print_result_summary(result: Dict[str, Any], logger: Any) -> None:
    """在终端打印各划分的整体指标表格。"""
    summary = result.get("各划分整体指标", {}) or {}
    metric_name = sorted({k for v in summary.values() for k in v if k != "loss"})
    row: List[List[str]] = []
    for split in ("train", "val", "test"):
        if split not in summary:
            continue
        row.append(
            [SPLIT_LABELS[split]]
            + [f"{summary[split].get('loss', float('nan')):.6g}"]
            + [f"{summary[split].get(m, float('nan')):.6g}" for m in metric_name]
        )
    logger("========== 各数据划分整体指标 ==========")
    print_table(row, header=["划分", "loss(MSE)"] + metric_name)
    logger(
        f"最优模型：第 {result['历史最优轮次']} 轮，"
        f"{result['挑选划分']}_{result['挑选指标']}={result['历史最优挑选指标值']:.8g}，"
        f"val_loss={result['历史最优验证损失']:.8g}"
    )
    logger(f"结果目录：{result['实验目录']}")


# --------------------------------------------------------------------------------------
# 单次实验
# --------------------------------------------------------------------------------------

def make_trial_config(
    base_config: Any,
    parser: Any,
    model_name: str,
    noise_type: str,
    trial: int,
    seed: Optional[int],
) -> Any:
    """基于基础配置派生“第 trial 次实验”的配置（只训练一个模型、一次实验）。"""
    import copy

    config = copy.deepcopy(base_config)
    config.network = str(model_name)
    config.repeat_k = 1
    config.trial = int(trial)
    config.seed = seed
    config.file_prefix = default_file_prefix(noise_type)
    # 每个模型在该噪声类型下的默认超参数：只有当用户**没有显式指定**（命令行或 --config 文件）
    # 时才套用该模型的默认值，否则用户显式给出的值优先。
    explicit = set(getattr(base_config, "_explicit_dests", set()) or set())
    file_keys = set(getattr(base_config, "_file_config_keys", set()) or set())
    for key, value in model_default_hyperparams(model_name, noise_type).items():
        if key in explicit or key in file_keys:
            continue
        setattr(config, key, value)
    # 模型初始化参数按新模型名重新合并（模型文件默认 < 训练配置 < --model-kwargs < --model-arg）
    file_model_args: Dict[str, Any] = {}
    config_path = getattr(base_config, "config", None)
    if config_path and os.path.isfile(str(config_path)):
        try:
            file_model_args = dict(load_json(str(config_path)).get("模型初始化参数", {}) or {})
        except Exception:
            file_model_args = {}
    config.model_args = parser._merge_model_args(config, model_name, file_model_args)
    return config


def prepare_trial_split(
    layout: ResultLayout,
    config: Any,
    logger: Any,
) -> Tuple[DataSplit, bool]:
    """准备某次实验的数据划分。

    :return: ``(划分数据, 是否本次新建)``。若同一次实验的其他模型已经生成过划分数据，
        则直接复用（同一试验次数下所有模型共用同一份数据，保证比较公平且不重复耗内存）。
    """
    split_dir = layout.split_dir
    info_path = os.path.join(split_dir, "划分信息.json")
    if os.path.isfile(info_path):
        try:
            data = load_split(split_dir)
            logger(f"复用本次实验已保存的数据划分：{data.describe()}")
            return data, False
        except Exception as error:  # 文件损坏时重新生成
            logger(f"加载已保存的划分数据失败（{error}），将重新生成")
    EEG_all, noise_all = load_raw_data(config)
    logger(f"读取数据完成：EEG {EEG_all.shape}，噪声 {noise_all.shape}")
    data = build_split(config, EEG_all, noise_all, logger)
    if int(getattr(config, "save_splits", 1)) == 1:
        ensure_dir(split_dir)
        save_split(split_dir, data)
        logger(f"划分数据已保存到：{split_dir}（本次实验的其他模型将直接复用）")
    return data, True


def run_one_experiment(
    layout: ResultLayout,
    config: Any,
    data: DataSplit,
    device: str,
) -> Dict[str, Any]:
    """训练一个模型的一次独立实验（新建模式）。"""
    from trainer import DenoiseTrainer

    layout.ensure_dirs()
    logger = Logger(layout.log_path)
    logger("=" * 90)
    logger(f"新建实验：{layout.experiment_dir}")
    logger(f"模型：{config.network} | 噪声：{config.noise_type} | 第 {layout.trial} 次独立重复实验")
    logger(f"本次随机种子：{config.seed}")
    logger(f"最优模型挑选准则：{config.select_split}_{config.select_metric}")
    logger(f"运行环境：{runtime_env_info()}")
    logger(f"训练参数：{config_to_dict(config)['训练参数']}")
    logger(f"模型初始化参数：{config.model_args}")

    save_config(config, layout.config_path)
    trainer = DenoiseTrainer(config, data, layout.experiment_dir, logger, device, layout=layout)
    extra_weights = resolve_weights_path(config, layout.experiment_dir) if config.resume_weights else None
    result = trainer.run(resume=False, extra_weights=extra_weights)
    if int(config.save_plot) == 1:
        try:
            from plotting import plot_all

            plot_all(layout.experiment_dir, lang=config.plot_lang, split_list=config.eval_splits,
                     logger=logger, output_dir=layout.plots_dir)
        except Exception as error:  # 绘图失败不应让训练结果丢失
            logger(f"绘图失败（训练结果已保存）：{error}")
    print_result_summary(result, logger)
    return result


def run_resume(config: Any, layout: ResultLayout, device: str) -> int:
    """从断点恢复训练（需求 10）。"""
    from trainer import DenoiseTrainer

    layout.ensure_dirs()
    logger = Logger(layout.log_path)
    logger("=" * 90)
    logger(f"恢复训练模式：{layout.experiment_dir}")
    logger(f"运行环境：{runtime_env_info()}")

    data: Optional[DataSplit] = None
    info_path = os.path.join(layout.split_dir, "划分信息.json")
    if os.path.isfile(info_path):
        try:
            data = load_split(layout.split_dir)
            logger(f"已加载保存的数据划分：{data.describe()}")
        except Exception as error:
            logger(f"加载已保存的划分数据失败（{error}），将重新生成")
    if data is None:
        EEG_all, noise_all = load_raw_data(config)
        data = build_split(config, EEG_all, noise_all, logger)
        if int(config.save_splits) == 1:
            save_split(layout.split_dir, data)
            logger(f"划分数据已保存到：{layout.split_dir}")

    save_config(config, layout.config_path)
    trainer = DenoiseTrainer(config, data, layout.experiment_dir, logger, device, layout=layout)
    extra_weights = resolve_weights_path(config, layout.experiment_dir) if config.resume_weights else None
    result = trainer.run(resume=True, extra_weights=extra_weights)
    if int(config.save_plot) == 1:
        try:
            from plotting import plot_all

            plot_all(layout.experiment_dir, lang=config.plot_lang, split_list=config.eval_splits,
                     logger=logger, output_dir=layout.plots_dir)
        except Exception as error:
            logger(f"绘图失败（训练结果已保存）：{error}")
    print_result_summary(result, logger)
    return 0


# --------------------------------------------------------------------------------------
# K 次实验的统计汇总
# --------------------------------------------------------------------------------------

def read_trial_values(layout: ResultLayout, metric_names: List[str], k: int,
                      splits: Tuple[str, ...] = ("train", "val", "test")
                      ) -> Dict[str, Dict[str, List[Optional[float]]]]:
    """读取该模型 K 次实验在指定划分上的各指标原始值。"""
    values: Dict[str, Dict[str, List[Optional[float]]]] = {
        split: {metric: [] for metric in metric_names} for split in splits
    }
    for trial in range(1, int(k) + 1):
        trial_layout = ResultLayout(layout.result_dir, layout.model_name, layout.noise_type, trial)
        for split in splits:
            for metric in metric_names:
                value = read_trial_metrics(trial_layout.experiment_dir, metric, split)
                values[split][metric].append(None if value is None else float(value))
    return values


def compute_statistics(values: List[Optional[float]]) -> Dict[str, Any]:
    """对 K 个独立数值计算 均值 / 标准差 / 中位数 / 四分位距（需求 4）。"""
    array = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    valid = array[np.isfinite(array)]
    result: Dict[str, Any] = {
        "K": int(array.size),
        "有效个数": int(valid.size),
        "原始值": [None if not np.isfinite(v) else float(v) for v in array],
    }
    if valid.size == 0:
        result.update({"均值": None, "标准差": None, "中位数": None,
                       "四分位距IQR": None, "Q1": None, "Q3": None, "最小值": None, "最大值": None})
        return result
    q1, q3 = np.percentile(valid, [25, 75])
    result.update(
        {
            "均值": float(np.mean(valid)),
            "标准差": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
            "中位数": float(np.median(valid)),
            "Q1": float(q1),
            "Q3": float(q3),
            "四分位距IQR": float(q3 - q1),
            "最小值": float(np.min(valid)),
            "最大值": float(np.max(valid)),
        }
    )
    return result


def write_aggregate_statistics(
    result_dir: str,
    noise_type: str,
    metric_names: List[str],
    k: int,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    model_name: Optional[str] = None,
) -> str:
    """把 K 次实验的 均值/标准差/中位数/IQR 写入噪声类型目录下（需求 4）。

    :param result_dir: 结果根目录
    :param model_name: 模型名（同时决定读取哪个 ``<模型名>/<噪声目录>/试验次数_*``）
    """
    model_name = str(model_name or os.path.basename(os.path.normpath(result_dir)))
    layout = ResultLayout(result_dir, model_name, noise_type, 1)
    aggregate: Dict[str, Any] = {
        "模型": layout.model_name,
        "噪声类型": noise_type,
        "独立实验次数K": int(k),
        "统计量说明": "均值/标准差/中位数/IQR 均由 K 次独立重复实验（不同随机种子、不同数据划分）的结果计算",
        "各划分": {split: {} for split in splits},
    }
    for split in splits:
        values = read_trial_values(layout, metric_names, k, (split,))[split]
        for metric in metric_names:
            aggregate["各划分"][split][metric] = compute_statistics(values[metric])

    # 每次实验一行的明细，便于人工核对
    per_trial: List[Dict[str, Any]] = []
    for trial in range(1, int(k) + 1):
        trial_layout = ResultLayout(layout.result_dir, layout.model_name, noise_type, trial)
        entry: Dict[str, Any] = {"试验次数": trial, "随机种子": None, "指标": {}}
        summary_path = find_summary_path(trial_layout.experiment_dir)
        if summary_path:
            try:
                summary = load_json(summary_path)
                entry["随机种子"] = summary.get("随机种子")
                entry["最优轮次"] = summary.get("最优轮次")
            except Exception:
                pass
        for split in splits:
            entry["指标"][split] = {}
            for metric in metric_names:
                entry["指标"][split][metric] = read_trial_metrics(
                    trial_layout.experiment_dir, metric, split
                )
        per_trial.append(entry)
    aggregate["每次实验明细"] = per_trial

    path = os.path.join(layout.noise_dir, AGGREGATE_FILENAME)
    save_json(path, aggregate)
    save_json(os.path.join(layout.noise_dir, "该次实验汇总.json"), {"各次实验": per_trial})
    return path


# --------------------------------------------------------------------------------------
# 清空旧结果
# --------------------------------------------------------------------------------------

def clean_noise_results(result_dir: str, model_name: str, noise_type: str,
                        logger: Any = None) -> None:
    """清空该模型在该噪声类型下的全部历史结果（只保留 ``README.md``）。

    删除内容：``试验次数_*`` 文件夹、``统计分析/``、以及目录下的汇总/箱线图文件。

    :param result_dir: 结果根目录
    :param model_name: 模型名
    """
    import shutil

    noise_dir = noise_results_dir(result_dir, model_name, noise_type)
    if not os.path.isdir(noise_dir):
        return
    removed: List[str] = []
    for name in sorted(os.listdir(noise_dir)):
        path = os.path.join(noise_dir, name)
        if name == "README.md":
            continue
        if os.path.isdir(path):
            if name.startswith("试验次数_") or name == _STATS_SUBDIR:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(name)
            continue
        if (
            name == AGGREGATE_FILENAME
            or name == _AGGREGATE_SPLIT_FILENAME
            or name.endswith(_SIGNIFICANCE_SUFFIX_JSON)
            or name.endswith(_SIGNIFICANCE_SUFFIX_CSV)
            or name.lower().endswith(".png")
        ):
            os.remove(path)
            removed.append(name)
    # 清理可能由旧版本直接写在模型目录下的汇总/箱线图/日志文件
    _remove_stray_model_root_files(os.path.dirname(noise_dir), logger, removed=removed)
    if callable(logger):
        if removed:
            logger(f"[清空] 已删除 {len(removed)} 项旧结果：{noise_dir}")
        else:
            logger(f"[清空] {noise_dir} 下没有需要删除的旧结果")


def _fix_misplaced_log_dir(model_dir: str, model_name: str, noise_type: str) -> None:
    """修复历史版本写歪的结果目录。

    早期版本误把“模型目录”当成“结果根目录”传给 ``noise_results_dir``，于是生成了::

        results/<模型名>/<模型名>/<噪声>去噪_<采样率>HZ/<模型名>_运行日志.txt

    这里在训练开始前把这种目录整体搬迁到正确位置
    （``results/<模型名>/``），再删掉空的错误目录。
    """
    import shutil

    model_dir = os.path.abspath(model_dir)
    result_root = os.path.dirname(model_dir)
    wrong_dir = os.path.join(model_dir, model_name)
    if not os.path.isdir(wrong_dir):
        return

    for noise_name in sorted(os.listdir(wrong_dir)):
        src_dir = os.path.join(wrong_dir, noise_name)
        if not os.path.isdir(src_dir) or not str(noise_name).startswith(noise_type):
            continue
        dst_dir = ensure_dir(noise_results_dir(result_root, model_name, noise_type))
        for name in sorted(os.listdir(src_dir)):
            src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(src, ignore_errors=True)
                else:
                    shutil.move(src, dst)
            elif os.path.isfile(src):
                if os.path.isfile(dst):
                    with open(src, "r", encoding="utf-8", errors="ignore") as src_handle:
                        text = src_handle.read()
                    with open(dst, "a", encoding="utf-8") as dst_handle:
                        dst_handle.write(text)
                    os.remove(src)
                else:
                    shutil.move(src, dst)
    # 搬空后删掉错误目录（README.md 之外没有内容就整体删除）
    leftover = [n for n in os.listdir(wrong_dir) if n != "README.md"]
    if not leftover:
        shutil.rmtree(wrong_dir, ignore_errors=True)
    else:
        # 只可能还剩空的噪声目录：把去掉子目录后空掉的目录一并清理
        for name in list(leftover):
            path = os.path.join(wrong_dir, name)
            if os.path.isdir(path) and not os.listdir(path):
                shutil.rmtree(path, ignore_errors=True)
        if [n for n in os.listdir(wrong_dir) if n != "README.md"] == []:
            shutil.rmtree(wrong_dir, ignore_errors=True)


def _remove_stray_model_root_files(model_dir: str, logger: Any = None,
                                   removed: Optional[List[str]] = None) -> None:
    """删除残留在 ``<结果根>/<模型名>/`` 这一层的旧版文件。

    正常结构下这一层**只应该有** ``<噪声>去噪_<采样率>HZ/`` 文件夹与 ``README.md``；
    旧版本会把运行日志、汇总 JSON、箱线图写在模型名这一层，这里顺手清掉。
    """
    if not os.path.isdir(model_dir):
        return
    dropped: List[str] = []
    for name in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path) or name == "README.md":
            continue
        if (
            name.endswith("_运行日志.txt")
            or name.endswith("_统计汇总.json")
            or name.endswith(_SIGNIFICANCE_SUFFIX_JSON)
            or name.endswith(_SIGNIFICANCE_SUFFIX_CSV)
            or name.lower().endswith(".png")
        ):
            os.remove(path)
            dropped.append(name)
    if dropped:
        if removed is not None:
            removed.extend(dropped)
        if callable(logger):
            logger(f"[清理] 已删除模型目录下不该存在的旧文件：{dropped}")


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # ---------- 第一轮迷你解析：只为定位 --resume / --plot-only 指向的实验文件夹 ----------
    mini_parser = argparse.ArgumentParser(add_help=False)
    mini_parser.add_argument("--resume", default=None)
    mini_parser.add_argument("--plot-only", default=None)
    mini_parser.add_argument("--result-dir", default=None)
    mini_parser.add_argument("--file-prefix", default=None)
    mini_parser.add_argument("--noise-type", default="EMG")
    mini_parser.add_argument("--config", default=None)
    mini_args, _ = mini_parser.parse_known_args(raw_args)

    target_layouts: List[ResultLayout] = []
    resume_layout: Optional[ResultLayout] = None
    target = mini_args.resume or mini_args.plot_only
    if target:
        try:
            target_layouts = locate_target_layouts(
                target, resolve_result_root(mini_args.result_dir), mini_args.file_prefix,
                mini_args.noise_type,
            )
        except FileNotFoundError as error:
            print(f"[错误] {error}", file=sys.stderr)
            return 2
        if mini_args.resume:
            resume_layout = target_layouts[0]
        if mini_args.config is None:
            config_path = os.path.join(resume_layout.config_dir, experiment_config_filename()) if resume_layout else None
            if config_path and not os.path.isfile(config_path):
                for name in (experiment_config_filename(),):
                    legacy = os.path.join(resume_layout.experiment_dir, name)
                    if os.path.isfile(legacy):
                        config_path = legacy
                        break
            if config_path and os.path.isfile(config_path):
                raw_args = ["--config", config_path] + raw_args

    # ---------- 正式解析 ----------
    from network import available_networks, discovery_warnings as network_discovery_warnings

    parser = build_parser(available_networks(), available_metrics())
    config = parser.parse_typed(raw_args)

    for warning in list(network_discovery_warnings) + list(metric_discovery_warnings):
        print(f"[警告] {warning}", flush=True)

    # 结果目录优先按命令行/配置文件给出的值
    config.result_dir = resolve_result_root(
        getattr(config, "result_dir", None) or mini_args.result_dir
    )

    # ---------- 只绘图模式 ----------
    if config.plot_only:
        from plotting import plot_all

        for layout in target_layouts:
            logger = Logger(layout.log_path)
            logger("只绘图模式：不进行训练")
            plot_all(layout.experiment_dir, lang=config.plot_lang, split_list=config.eval_splits,
                     logger=logger, output_dir=layout.plots_dir)
        print(f"已重新绘图，共 {len(target_layouts)} 次实验")
        return 0

    # ---------- 断点恢复模式 ----------
    device = select_device_safe(config.device)
    if resume_layout is not None:
        config.network = resume_layout.model_name
        config.file_prefix = default_file_prefix(resume_layout.noise_type)
        config.result_dir = resume_layout.result_dir
        seed = None if (config.seed is None or int(config.seed) < 0) else int(config.seed)
        set_random_seed(seed)
        return run_resume(config, resume_layout, device)

    # ---------- K 次独立重复实验模式 ----------
    return run_experiment_campaign(config, parser, device)


def select_device_safe(device_arg: str) -> str:
    """选择训练设备（包一层异常保护）。"""
    from utils import select_device

    return select_device(device_arg)


def run_experiment_campaign(config: Any, parser: Any, device: str) -> int:
    """执行“多个模型 × K 次独立重复实验”的完整流程。"""
    from network import available_networks

    models: List[str] = list(getattr(config, "models", None) or [config.network])
    for model_name in models:
        if model_name not in available_networks():
            print(f"[错误] 未知网络 {model_name}，可用：{available_networks()}", file=sys.stderr)
            return 2

    k = max(1, int(getattr(config, "repeat_k", 1) or 1))
    noise_type = str(config.noise_type).upper()
    print(f"待训练模型：{models}")
    print(f"噪声类型：{noise_type} | 独立重复实验次数 K={k} | 设备：{device}")
    print(f"结果根目录：{config.result_dir}")
    if k < 2:
        print("[提示] K < 2，无法做统计学检验；只有 K >= 2 才能计算标准差与方差分析。")

    # 解析要评估/统计的指标（把“全部”展开成具体指标名）
    from metrics import resolve_metric_selection

    metric_names = list(getattr(config, "metrics", ["MSE", "SNR", "CC", "RRMSE"]))
    if any(str(m).lower() in ("全部", "all", "*") for m in metric_names):
        metric_names = resolve_metric_selection(metric_names)
    # 需要统计的指标：所选全部指标 + 挑选指标（若不在其中）
    stats_metrics: List[str] = list(dict.fromkeys(
        metric_names + ([config.select_metric] if str(config.select_metric).lower() != "loss" else [])
    ))

    for model_name in models:
        model_dir = ensure_dir(os.path.join(config.result_dir, model_name))
        # 注意：noise_results_dir 的第一个参数是**结果根目录**（它内部会再拼模型名），
        # 传成 model_dir 会拼出 …/<模型>/<模型>/… 的错误结构。
        noise_dir = ensure_dir(noise_results_dir(config.result_dir, model_name, noise_type))
        # 修掉历史版本可能已经写歪的日志目录：…/<模型>/<模型>/<噪声目录>/<模型>_运行日志.txt
        _fix_misplaced_log_dir(model_dir, model_name, noise_type)
        # 运行日志保存在“噪声目录”内部（<模型名>/<噪声>去噪_<采样率>HZ/），
        # 保证 <模型名>/ 这一层只有噪声类型文件夹，不再出现别的文件
        bootstrap_logger = Logger(os.path.join(noise_dir, f"{model_name}_运行日志.txt"))
        _remove_stray_model_root_files(model_dir, bootstrap_logger)

        if int(getattr(config, "clean", 1)) == 1:
            clean_noise_results(config.result_dir, model_name, noise_type, bootstrap_logger)
        bootstrap_logger("=" * 90)
        bootstrap_logger(
            f"开始 {model_name} 的 {k} 次独立重复实验（噪声类型 {noise_type}）"
        )

        for trial in range(1, k + 1):
            base_seed = getattr(config, "seed", 42)
            trial_seed: Optional[int]
            if base_seed is None or int(base_seed) < 0:
                trial_seed = None
            else:
                trial_seed = int(base_seed) + (trial - 1)
            trial_config = make_trial_config(config, parser, model_name, noise_type, trial, trial_seed)
            layout = make_layout(config.result_dir, model_name, noise_type, trial)
            trial_logger = Logger(layout.log_path)
            trial_logger("=" * 90)
            trial_logger(
                f"{model_name} 第 {trial}/{k} 次独立重复实验 | 随机种子={trial_seed} | "
                f"结果目录={layout.experiment_dir}"
            )

            # 每次实验都重新设置随机种子：数据划分与网络初始化都会不同
            set_random_seed(trial_seed)
            data, _ = prepare_trial_split(layout, trial_config, trial_logger)
            set_random_seed(trial_seed)
            run_one_experiment(layout, trial_config, data, device)
            del data

        path = write_aggregate_statistics(config.result_dir, noise_type, stats_metrics, k,
                                          model_name=model_name)
        bootstrap_logger(f"{model_name} 的 K 次实验统计汇总已保存：{path}")

    # ---------- 所有模型训练完成：单因素方差分析 + Tukey HSD ----------
    if int(getattr(config, "run_significance", 1)) == 1 and len(models) > 1:
        try:
            from significance_test import run_significance_analysis

            for path in run_significance_analysis(
                result_dir=config.result_dir,
                noise_type=noise_type,
                models=models,
                metrics=stats_metrics,
                lang=config.plot_lang,
                highlight=getattr(config, "highlight", None),
            ):
                print(f"  - 显著性检验产物：{path}")
        except Exception as error:
            print(f"[警告] 显著性检验失败（训练结果不受影响）：{error}", flush=True)
    elif len(models) == 1:
        print("[提示] 只训练了一个模型，跳过方差分析（ANOVA 需要 >= 2 个模型）")

    print("全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
