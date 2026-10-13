#!/usr/bin/env python3
"""训练入口（EEGdenoiseNet 去噪网络重构版）。

常用命令
--------
新建实验并训练（默认 Simple_CNN + EMG + 512Hz）::

    python code/main.py --network Simple_CNN --noise-type EMG --epochs 50

选择要评估的指标（默认全部：MSE/SNR/CC/RRMSE）::

    python code/main.py --network fcNN --metrics MSE SNR CC RRMSE

给模型传自定义初始化参数（需求 7）::

    python code/main.py --network Simple_CNN --model-arg channels=32 dropout=0.2

从断点恢复训练（需求 10）::

    python code/main.py --resume 测试集_EMG去噪_512HZ_试验次数_1 --epochs 80
    python code/main.py --resume latest --epochs 80

只对已有实验重新绘图::

    python code/main.py --plot-only 测试集_EMG去噪_512HZ_试验次数_1
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

from config import save_config, build_parser, runtime_env_info, config_to_dict  # noqa: E402
from data_prepare import load_split, prepare_data, save_split, DataSplit  # noqa: E402
from metrics import available_metrics, discovery_warnings as metric_discovery_warnings  # noqa: E402
from network import available_networks, discovery_warnings as network_discovery_warnings  # noqa: E402
from plotting import plot_all  # noqa: E402
from trainer import DenoiseTrainer  # noqa: E402
from utils import (  # noqa: E402
    load_json,
    load_npy,
    NOISE_DATA_FILES,
    experiment_dirname,
    experiment_config_filename,
    ensure_dir,
    print_table,
    Logger,
    weight_filename,
    next_trial_index,
    set_random_seed,
    split_data_dirname,
    SPLIT_LABELS,
    select_device,
    default_file_prefix,
)

# --------------------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------------------

def locate_experiment_dir(
    specified: str,
    result_dir: str,
    file_prefix: Optional[str] = None,
    noise_type: str = "EMG",
) -> str:
    """把 ``--resume`` / ``--plot-only`` 的取值解析成实际存在的实验文件夹路径。

    支持三种写法：

    - ``latest`` / ``auto`` / ``最新``：自动选择结果目录下**最新修改**的实验文件夹；
    - 实验文件夹名（如 ``测试集_EMG去噪_512HZ_试验次数_1``）：在结果目录下查找；
    - 完整路径：直接使用。
    """
    specified = str(specified)
    if specified.lower() in ("latest", "auto", "最新"):
        if not os.path.isdir(result_dir):
            raise FileNotFoundError(f"结果目录不存在：{result_dir}")
        prefix = file_prefix or default_file_prefix(noise_type)
        candidate = [
            os.path.join(result_dir, name)
            for name in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, name))
            and name.startswith(prefix)
            and "_试验次数_" in name
        ]
        if not candidate:
            raise FileNotFoundError(
                f"在 {result_dir} 中找不到以 {prefix} 开头的实验文件夹，无法使用 latest"
            )
        return max(candidate, key=os.path.getmtime)

    if os.path.isdir(specified):
        return os.path.abspath(specified)

    candidate_path = os.path.join(result_dir, specified)
    if os.path.isdir(candidate_path):
        return os.path.abspath(candidate_path)
    raise FileNotFoundError(
        f"找不到实验文件夹：{specified}\n（已尝试：{specified} 与 {candidate_path}）"
    )


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
        return os.path.join(experiment_dir, weight_filename(config.file_prefix, config.network))
    if os.path.isfile(value):
        return value
    if experiment_dir:
        candidate = os.path.join(experiment_dir, value)
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
    logger(f"最优模型：第 {result['历史最优轮次']} 轮，val_loss={result['历史最优验证损失']:.8g}")
    logger(f"结果目录：{result['实验目录']}")


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # ---------- 第一轮迷你解析：只为定位 --resume / --plot-only 指向的实验文件夹 ----------
    mini_parser = argparse.ArgumentParser(add_help=False)
    mini_parser.add_argument("--resume", default=None)
    mini_parser.add_argument("--plot-only", default=None)
    mini_parser.add_argument("--result-dir", default=os.path.join(CODE_DIR, "results"))
    mini_parser.add_argument("--file-prefix", default=None)
    mini_parser.add_argument("--noise-type", default="EMG")
    mini_parser.add_argument("--config", default=None)
    mini_args, _ = mini_parser.parse_known_args(raw_args)

    resume_dir: Optional[str] = None
    target = mini_args.resume or mini_args.plot_only
    if target:
        resume_dir = locate_experiment_dir(
            target, mini_args.result_dir, mini_args.file_prefix, mini_args.noise_type
        )
        # 把上次实验保存的参数作为默认值注入（命令行仍可覆盖）
        if mini_args.config is None:
            config_path = os.path.join(resume_dir, experiment_config_filename())
            if os.path.isfile(config_path):
                raw_args = ["--config", config_path] + raw_args

    # ---------- 正式解析 ----------
    parser = build_parser(available_networks(), available_metrics())
    config = parser.parse_typed(raw_args)

    for warning in list(network_discovery_warnings) + list(metric_discovery_warnings):
        print(f"[警告] {warning}", flush=True)

    # ---------- 只绘图模式 ----------
    if config.plot_only:
        experiment_dir = resume_dir or locate_experiment_dir(
            config.plot_only, config.result_dir, config.file_prefix, config.noise_type
        )
        logger = Logger(os.path.join(experiment_dir, "训练日志.txt"))
        logger("只绘图模式：不进行训练")
        plot_all(experiment_dir, lang=config.plot_lang, split_list=config.eval_splits, logger=logger)
        return 0

    # ---------- 训练（新建或恢复） ----------
    device = select_device(config.device)
    seed = None if (config.seed is None or int(config.seed) < 0) else int(config.seed)
    set_random_seed(seed)

    data: Optional[DataSplit] = None
    if resume_dir:
        experiment_dir = resume_dir
        logger = Logger(os.path.join(experiment_dir, "训练日志.txt"))
        logger("=" * 90)
        logger(f"恢复训练模式：{experiment_dir}")
        logger(f"运行环境：{runtime_env_info()}")
        # 优先加载上次保存的划分数据，保证训练/验证/测试集完全一致
        split_dir = os.path.join(experiment_dir, split_data_dirname())
        if os.path.isfile(os.path.join(split_dir, "划分信息.json")):
            try:
                data = load_split(split_dir)
                logger(f"已加载保存的数据划分：{data.describe()}")
            except Exception as error:
                logger(f"加载已保存的划分数据失败（{error}），将重新生成")
        if data is None:
            EEG_all, noise_all = load_raw_data(config)
            data = build_split(config, EEG_all, noise_all, logger)
            if int(config.save_splits) == 1:
                save_split(split_dir, data)
                logger(f"划分数据已保存到：{split_dir}")
        trainer = DenoiseTrainer(config, data, experiment_dir, logger, device)
        # --resume-weights 指定的权重会在断点恢复之后加载（优先级最高），由 运行() 内部处理
        extra_weights = resolve_weights_path(config, experiment_dir) if config.resume_weights else None
        result = trainer.run(resume=True, extra_weights=extra_weights)
    else:
        trial = config.trial if config.trial else next_trial_index(config.result_dir, config.file_prefix)
        experiment_dir = ensure_dir(os.path.join(config.result_dir, experiment_dirname(config.file_prefix, int(trial))))
        logger = Logger(os.path.join(experiment_dir, "训练日志.txt"))
        logger("=" * 90)
        logger(f"新建实验：{experiment_dir}")
        logger(f"运行环境：{runtime_env_info()}")
        logger(f"训练参数：{config_to_dict(config)['训练参数']}")
        logger(f"模型初始化参数：{config.model_args}")
        # 目录复用提醒：同名文件会被覆盖，避免静默丢失上一次实验的结果
        existing_config_path = os.path.join(experiment_dir, experiment_config_filename())
        if os.path.isfile(existing_config_path):
            try:
                old_network = (load_json(existing_config_path).get("训练参数", {}) or {}).get("network")
            except Exception:
                old_network = None
            if old_network and old_network != config.network:
                logger(
                    f"[警告] 该实验文件夹已属于网络 {old_network}，本次训练的是 {config.network}，"
                    "同名指标/Loss 文件会被覆盖。如需并行保留，请用 --trial 指定新的编号。"
                )
            else:
                logger(f"[提示] 实验文件夹已存在（网络 {old_network}），同名文件将被本次结果覆盖。")
        save_config(config, os.path.join(experiment_dir, experiment_config_filename()))
        EEG_all, noise_all = load_raw_data(config)
        logger(f"读取数据完成：EEG {EEG_all.shape}，噪声 {noise_all.shape}")
        data = build_split(config, EEG_all, noise_all, logger)
        if int(config.save_splits) == 1:
            split_dir = os.path.join(experiment_dir, split_data_dirname())
            save_split(split_dir, data)
            logger(f"划分数据已保存到：{split_dir}（恢复训练时将直接加载）")

        trainer = DenoiseTrainer(config, data, experiment_dir, logger, device)
        # --resume-weights：只用已有权重初始化本次新实验，不恢复优化器与 epoch
        extra_weights = resolve_weights_path(config, experiment_dir) if config.resume_weights else None
        result = trainer.run(resume=False, extra_weights=extra_weights)

    # ---------- 训练完成后自动绘图 ----------
    if int(config.save_plot) == 1:
        try:
            plot_all(experiment_dir, lang=config.plot_lang, split_list=config.eval_splits, logger=logger)
        except Exception as error:  # 绘图失败不应让训练结果丢失
            logger(f"绘图失败（训练结果已保存）：{error}")

    print_result_summary(result, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
