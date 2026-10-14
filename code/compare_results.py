#!/usr/bin/env python3
"""实验结果对比绘图工具（需求 11）。

功能：从 ``results/`` 下的多个实验结果文件夹中，分别读取 loss、MSE、SNR(dB)、
CC、RRMSE 等指标（npy 文件），绘制成对比图。**本脚本不会自动运行**，需要手动执行。

目录结构说明：结果按 ``results/<模型名>/<噪声>去噪_<采样率>HZ/试验次数_N/`` 分层存放，
每个实验内部再分为 ``权重/ 指标/ 图像/ 训练配置/ 划分数据/`` 子目录。本脚本会
**递归发现** 全部 ``试验次数_N`` 实验（也兼容旧的平铺结构）。

用法示例
--------
对比全部实验（测试集、全部指标）::

    python code/compare_results.py --result-dir results --split test

只对比某个模型的全部 K 次实验::

    python code/compare_results.py --result-dir results/Simple_CNN --split test

只对比指定的几个实验文件夹，并指定输出目录::

    python code/compare_results.py --result-dir results \
        --experiments results/Simple_CNN/EMG去噪_512HZ/试验次数_1 \
                      results/fcNN/EMG去噪_512HZ/试验次数_1 \
        --metrics MSE SNR CC RRMSE --split test --out results/对比图

产出
----
- ``对比_<指标>_<划分>.png``：横轴 SNR，一张图里多条曲线（每个实验一条）；
- ``对比_指标总览_<划分>.png``：各实验整体指标的柱状图；
- ``对比_训练Loss.png``：各实验的训练/验证 loss 曲线对比；
- ``对比汇总_<划分>.csv``：整体指标的表格，便于直接写进论文。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plotting import (  # noqa: E402
    split_styles,
    metric_display,
    safe_label,
    configure_fonts,
    read_loss,
    read_summary,
    read_metrics,
    read_snr_levels,
    read_train_history,
)
from utils import (  # noqa: E402
    SPLIT_LABELS,
    ensure_dir,
    find_result_layouts,
    find_summary_path,
    read_overall_metrics as read_overall_metrics_from_dir,
    resolve_result_root,
)

#: 从文件夹名里提取“试验次数”的正则
_TRIAL_RE = re.compile(r"_试验次数_(\d+)")

#: 对比图使用的颜色循环
color_cycle = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# --------------------------------------------------------------------------------------
# 发现与读取
# --------------------------------------------------------------------------------------

def discover_experiments(result_dir: str, specified: Optional[Sequence[str]] = None) -> List[str]:
    """返回实验结果文件夹列表（按名字排序）。

    :param specified: 需要对比的实验文件夹名或路径；为 ``None`` 时自动递归发现结果目录下所有实验
    """
    if specified:
        result: List[str] = []
        for item in specified:
            path = item if os.path.isdir(item) else os.path.join(result_dir, item)
            if not os.path.isdir(path):
                raise FileNotFoundError(f"找不到实验文件夹：{item}")
            result.append(os.path.abspath(path))
        return result

    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"结果目录不存在：{result_dir}")

    experiment_list: List[str] = []
    # 结果目录本身就是一个实验文件夹的情况
    if _TRIAL_RE.search(os.path.basename(os.path.normpath(result_dir))) and find_summary_path(result_dir):
        experiment_list.append(os.path.abspath(result_dir))
    # ① 推荐方式：按 <模型>/<噪声>/试验次数_N 结构递归发现
    for layout in find_result_layouts(result_dir):
        if os.path.isdir(layout.experiment_dir):
            experiment_list.append(os.path.abspath(layout.experiment_dir))
    # ② 兼容旧的平铺结构：任何含“实验摘要.json”的一级/二级子目录
    for depth in (1, 2):
        pattern_root = result_dir
        for name in sorted(os.listdir(pattern_root)):
            subdir = os.path.join(pattern_root, name)
            if os.path.isdir(subdir) and find_summary_path(subdir):
                path = os.path.abspath(subdir)
                if path not in experiment_list:
                    experiment_list.append(path)
    experiment_list = sorted(set(experiment_list))
    if not experiment_list:
        raise FileNotFoundError(
            f"{result_dir} 下没有找到任何实验结果（缺少 训练配置/实验摘要.json）"
        )
    return experiment_list


def experiment_display_name(experiment_dir: str, lang: str = "en") -> str:
    """实验在图例里的显示名。

    英文模式使用全 ASCII 的 ``网络 (噪声+采样率 #试验次数)``，避免图中出现缺字方块；
    中文模式使用 ``网络 @ 文件前缀``。
    """
    dirname = os.path.basename(experiment_dir.rstrip("/"))
    trial_match = _TRIAL_RE.search(dirname)
    trial = trial_match.group(1) if trial_match else "?"
    try:
        summary = read_summary(experiment_dir)
        network = summary.get("网络", "网络")
        noise_type = summary.get("噪声类型", "")
        sample_rate = summary.get("采样率", "")
        prefix = summary.get("文件前缀", "")
    except Exception:
        return safe_label(dirname, lang)

    if lang == "zh":
        return safe_label(f"{network} @ {prefix}", lang)
    return safe_label(f"{network} ({noise_type}{sample_rate}Hz #{trial})", lang)


def read_overall_metrics(experiment_dir: str, split: str) -> Dict[str, float]:
    """读取实验的整体指标（新结构读 ``训练配置/各划分整体指标.json``，兼容实验摘要）。"""
    overall = read_overall_metrics_from_dir(experiment_dir)
    return dict(overall.get(split, {}) or {})


# --------------------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------------------

def plot_metric_comparison(
    experiment_dirs: List[str],
    output_dir: str,
    split: str,
    metric_names: List[str],
    lang: str = "en",
) -> List[str]:
    """为每个指标画一张“横轴 SNR、多条曲线”的对比图。"""
    artifacts: List[str] = []
    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        has_curves = False
        for index, experiment_dir in enumerate(experiment_dirs):
            try:
                summary = read_summary(experiment_dir)
                noise_type = summary.get("噪声类型", "EMG")
                snr = read_snr_levels(experiment_dir)
                value = read_metrics(experiment_dir, split, noise_type, metric)
            except Exception:
                continue
            ax.plot(snr[:value.size], value, marker="o", markersize=4, linewidth=1.6,
                    color=color_cycle[index % len(color_cycle)],
                    label=experiment_display_name(experiment_dir, lang))
            has_curves = True
        if not has_curves:
            plt.close(fig)
            continue
        display = metric_display.get(metric, {"en": metric, "zh": metric})
        ax.set_xlabel("Input SNR (dB, nominal)" if lang == "en" else "输入 SNR（dB，标称值）")
        ax.set_ylabel(display["en"] if lang == "en" else display["zh"])
        ax.set_title(
            (f"{display['en']} comparison on {split_styles.get(split, {}).get('标签_en', split)} set"
             if lang == "en" else
             f"各实验在{SPLIT_LABELS.get(split, split)}上的{display['zh']}对比")
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(output_dir, f"对比_{metric}_{split}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        artifacts.append(path)
    return artifacts


def plot_overall_bar_comparison(
    experiment_dirs: List[str],
    output_dir: str,
    split: str,
    metric_names: List[str],
    lang: str = "en",
) -> Optional[str]:
    """绘制各实验整体指标的柱状对比图（每个指标一个子图）。"""
    data = [read_overall_metrics(directory, split) for directory in experiment_dirs]
    label = [experiment_display_name(directory, lang) for directory in experiment_dirs]
    if not any(data):
        return None

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.6 * len(metric_names), 4.6), squeeze=False)
    for i, metric in enumerate(metric_names):
        ax = axes[0][i]
        get_value = [d.get(metric, np.nan) for d in data]
        position = np.arange(len(get_value))
        ax.bar(position, get_value, color=[color_cycle[j % len(color_cycle)] for j in range(len(get_value))])
        ax.set_xticks(position)
        ax.set_xticklabels(label, rotation=25, ha="right", fontsize=7)
        display = metric_display.get(metric, {"en": metric, "zh": metric})
        ax.set_title(display["en"] if lang == "en" else display["zh"])
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        ("Overall metric comparison: " + split_styles.get(split, {}).get("标签_en", split))
        if lang == "en" else
        ("整体指标对比：" + SPLIT_LABELS.get(split, split))
    )
    fig.tight_layout()
    path = os.path.join(output_dir, f"对比_指标总览_{split}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_loss_comparison(
    experiment_dirs: List[str], output_dir: str, lang: str = "en"
) -> Optional[str]:
    """绘制各实验的训练/验证 loss 曲线对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    has_data = False
    for index, experiment_dir in enumerate(experiment_dirs):
        try:
            summary = read_summary(experiment_dir)
            history = read_train_history(experiment_dir, summary.get("文件前缀"))
        except Exception:
            try:
                history = read_loss(experiment_dir, read_summary(experiment_dir).get("文件前缀"))
            except Exception:
                continue
        epoch = np.asarray(history.get("epochs", []), dtype=np.float64)
        if not epoch.size:
            continue
        color = color_cycle[index % len(color_cycle)]
        label = experiment_display_name(experiment_dir, lang)
        axes[0].plot(epoch, np.asarray(history.get("train_loss", []), dtype=np.float64),
                     linewidth=1.6, color=color, label=label)
        axes[1].plot(epoch, np.asarray(history.get("val_loss", []), dtype=np.float64),
                     linewidth=1.6, color=color, label=label)
        has_data = True
    if not has_data:
        plt.close(fig)
        return None

    for ax, title_en, title_zh in (
        (axes[0], "Train loss", "训练 loss"),
        (axes[1], "Validation loss", "验证 loss"),
    ):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch" if lang == "en" else "训练轮次")
        ax.set_ylabel("Loss (MSE, log scale)" if lang == "en" else "Loss（MSE，对数刻度）")
        ax.set_title(title_en if lang == "en" else title_zh)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Loss comparison of experiments" if lang == "en" else "各实验 Loss 曲线对比")
    fig.tight_layout()
    path = os.path.join(output_dir, "对比_训练Loss.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_comparison_csv(
    experiment_dirs: List[str], output_dir: str, split: str, metric_names: List[str]
) -> str:
    """把各实验的整体指标汇总成 CSV，便于直接放进论文表格。"""
    path = os.path.join(output_dir, f"对比汇总_{split}.csv")
    header = ["实验", "实验目录", "loss(MSE)"] + list(metric_names)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for experiment_dir in experiment_dirs:
            overall = read_overall_metrics(experiment_dir, split)
            writer.writerow(
                [experiment_display_name(experiment_dir, 'zh'), experiment_dir]
                + [f"{overall.get('loss', float('nan')):.8g}"]
                + [f"{overall.get(k, float('nan')):.8g}" for k in metric_names]
            )
    return path


# --------------------------------------------------------------------------------------
# 命令行入口
# --------------------------------------------------------------------------------------

def build_comparison_parser() -> argparse.ArgumentParser:
    """构建对比脚本的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从 results 中读取多次实验的指标并绘制对比图（需求 11）"
    )
    parser.add_argument("--result-dir", default=None,
                      help="实验结果根目录（默认 <项目根>/results，与 code、data 同级）")
    parser.add_argument("--experiments", nargs="*", default=None,
                      help="要对比的实验文件夹名（默认自动发现全部）")
    parser.add_argument("--metrics", nargs="+", default=["MSE", "SNR", "CC", "RRMSE"],
                      help="要对比的指标")
    parser.add_argument("--split", nargs="+", default=["test"],
                      choices=["train", "val", "test"],
                      help="对比哪个数据划分（可多选）")
    parser.add_argument("--out", default=None,
                      help="输出目录，默认 <result-dir>/对比图")
    parser.add_argument("--lang", default="en", choices=["en", "zh"], help="图像语言")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_comparison_parser().parse_args(argv)
    lang = configure_fonts(args.lang)
    args.result_dir = resolve_result_root(args.result_dir)
    output_dir = ensure_dir(args.out or os.path.join(args.result_dir, "对比图"))

    experiment_list = discover_experiments(args.result_dir, args.experiments)
    print(f"共发现 {len(experiment_list)} 个实验：")
    for directory in experiment_list:
        print(f"  - {experiment_display_name(directory, lang)}  ({directory})")

    artifacts: List[str] = []
    for split in args.split:
        artifacts += plot_metric_comparison(experiment_list, output_dir, split, args.metrics, lang)
        for path in (
            plot_overall_bar_comparison(experiment_list, output_dir, split, args.metrics, lang),
            save_comparison_csv(experiment_list, output_dir, split, args.metrics),
        ):
            if path:
                artifacts.append(path)
    path = plot_loss_comparison(experiment_list, output_dir, lang)
    if path:
        artifacts.append(path)

    print(f"\n对比图已保存到：{output_dir}")
    for path in artifacts:
        print(f"  - {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
