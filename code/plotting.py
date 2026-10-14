"""绘图模块：训练完成后自动绘制 loss 曲线与各项指标曲线。

产出图片（都保存在实验结果文件夹里）：

===========================  ==================================================
文件                          内容
===========================  ==================================================
``Loss曲线.png``             每个 epoch 的训练/验证 loss 曲线（纵轴对数刻度）
``训练过程指标曲线.png``       每个 epoch 在验证集上的指标曲线（MSE/SNR/CC/RRMSE 子图）
``MSE-SNR.png``              训练/验证/测试三个划分的 MSE 随输入 SNR 的变化
``SNR-SNR.png``              去噪后 SNR 随输入 SNR 的变化（含 y=x 参考线）
``CC-SNR.png``               相关系数 CC 随输入 SNR 的变化
``RRMSE-SNR.png``            RRMSE 随输入 SNR 的变化
``指标总览.png``              三个划分的整体指标柱状图
===========================  ==================================================

说明：横轴 SNR 使用**标称 SNR**（与原版 EEGdenoiseNet 代码一致，见 ``data_prepare`` 的文档）。
训练集的 SNR 是随机值，其曲线表示“就近归入该档位的样本平均值”。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # 无显示环境（服务器）也能出图
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from utils import (  # noqa: E402
    ensure_dir,
    find_loss_path,
    find_metric_path,
    find_summary_path,
    find_train_history_path,
    load_json,
    load_npy,
    load_npy_dict,
)

#: 各划分在图上显示的名字与颜色
split_styles = {
    "train": {"标签_en": "Train", "标签_zh": "训练集", "颜色": "#1f77b4"},
    "val": {"标签_en": "Validation", "标签_zh": "验证集", "颜色": "#ff7f0e"},
    "test": {"标签_en": "Test", "标签_zh": "测试集", "颜色": "#2ca02c"},
}

#: 指标的英文/中文显示名与“越大越好”标记
metric_display = {
    "MSE": {"en": "MSE", "zh": "MSE（均方误差）", "越大越好": False},
    "SNR": {"en": "SNR (dB)", "zh": "SNR（dB）", "越大越好": True},
    "CC": {"en": "CC", "zh": "CC（相关系数）", "越大越好": True},
    "RRMSE": {"en": "RRMSE", "zh": "RRMSE（相对均方根误差）", "越大越好": False},
}

#: 常用中文字体候选（按优先级）
CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei", "Droid Sans Fallback",
    "AR PL UMing CN",
]


def configure_fonts(lang: str = "en") -> str:
    """根据语言配置 matplotlib 字体，返回实际使用的语言（``en`` 或 ``zh``）。

    若请求中文但系统没有中文字体，会自动退回英文，避免图上出现方块乱码。
    """
    lang = (lang or "en").lower()
    if lang not in ("zh", "en"):
        lang = "en"
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if lang == "zh":
        for name in CJK_FONT_CANDIDATES:
            if name in available_fonts:
                matplotlib.rcParams["font.sans-serif"] = [name] + list(
                    matplotlib.rcParams.get("font.sans-serif", [])
                )
                matplotlib.rcParams["axes.unicode_minus"] = False
                return "zh"
        return "en"  # 没有中文字体 -> 退回英文
    matplotlib.rcParams["axes.unicode_minus"] = False
    return "en"


def _text(lang: str, en: str, zh: str) -> str:
    """根据语言选择显示文本。"""
    return zh if lang == "zh" else en


def safe_label(text: Any, lang: str = "en") -> str:
    """英文模式下剔除标签里的非 ASCII 字符，避免图上出现“缺字方块”。

    实验文件夹名/前缀通常含中文（例如 ``测试集_EMG去噪_512HZ``），而英文模式默认字体
    （DejaVu Sans）不含中文字形，因此这里做一次净化；中文模式原样返回。
    """
    text = str(text)
    if lang == "zh":
        return text
    keep = "".join(char if ord(char) < 128 else " " for char in text)
    keep = re.sub(r"\s+", " ", keep).strip()
    return keep or "experiment"


# --------------------------------------------------------------------------------------
# 读取实验结果
# --------------------------------------------------------------------------------------

def read_summary(experiment_dir: str) -> Dict[str, Any]:
    """读取实验摘要 JSON（新结构在 ``训练配置/``，旧结构在实验文件夹根目录）。"""
    path = find_summary_path(experiment_dir)
    if not path:
        raise FileNotFoundError(f"找不到实验摘要：{experiment_dir}/训练配置/实验摘要.json")
    return load_json(path)


def read_train_history(experiment_dir: str, prefix: str) -> Dict[str, Any]:
    """读取 epoch 级的训练历史（loss + 每轮验证指标）。"""
    path = find_train_history_path(experiment_dir, prefix)
    if not path:
        raise FileNotFoundError(f"找不到训练历史：{experiment_dir}/指标/{prefix}_训练历史.npy")
    return load_npy_dict(path)


def read_loss(experiment_dir: str, prefix: str) -> Dict[str, Any]:
    """读取 ``<前缀>_Loss.npy``。"""
    path = find_loss_path(experiment_dir, prefix)
    if not path:
        raise FileNotFoundError(f"找不到 Loss 文件：{experiment_dir}/指标/{prefix}_Loss.npy")
    return load_npy_dict(path)


def read_metrics(experiment_dir: str, split: str, noise_type: str, metric: str) -> np.ndarray:
    """读取某个划分某个指标的 npy（形状 = SNR 档位数）。"""
    path = find_metric_path(experiment_dir, split, noise_type, metric)
    if not path:
        raise FileNotFoundError(f"找不到指标文件：{split}/{noise_type}/{metric}")
    return np.asarray(load_npy(path), dtype=np.float64)


def read_snr_levels(experiment_dir: str) -> np.ndarray:
    """读取 SNR 档位；若文件不存在则按默认 10 档返回。"""
    from utils import find_in_run_dir

    path = find_in_run_dir(experiment_dir, "SNR档位.npy")
    if path:
        return np.asarray(np.load(path), dtype=np.float64)
    return np.linspace(-7.0, 2.0, num=10)


# --------------------------------------------------------------------------------------
# 逐个图
# --------------------------------------------------------------------------------------

def plot_loss_curve(experiment_dir: str, history: Dict[str, Any], lang: str = "en",
                    output_dir: Optional[str] = None) -> str:
    """绘制训练/验证 loss 随 epoch 的变化曲线（默认写入实验文件夹的 ``图像/`` 子目录）。"""
    epochs = np.asarray(history.get("epochs", []), dtype=np.float64)
    train_loss = np.asarray(history.get("train_loss", []), dtype=np.float64)
    val_loss = np.asarray(history.get("val_loss", []), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    if epochs.size:
        ax.plot(epochs, train_loss, marker="o", markersize=3,
                label=_text(lang, "Train loss", "训练 loss"), color="#1f77b4")
        ax.plot(epochs, val_loss, marker="s", markersize=3,
                label=_text(lang, "Validation loss", "验证 loss"), color="#ff7f0e")
        ax.set_yscale("log")
    ax.set_xlabel(_text(lang, "Epoch", "训练轮次"))
    ax.set_ylabel(_text(lang, "Loss (MSE, log scale)", "Loss（MSE，对数刻度）"))
    ax.set_title(_text(lang, "Training and validation loss", "训练与验证 Loss 曲线"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir or os.path.join(experiment_dir, "图像"), "Loss曲线.png")
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_training_curves(
    experiment_dir: str, history: Dict[str, Any], metric_names: List[str], lang: str = "en",
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """绘制每个 epoch 在验证集上的各项指标曲线（子图形式）。"""
    epochs = np.asarray(history.get("epochs", []), dtype=np.float64)
    val_metrics = history.get("val_metrics", {}) or {}
    if not epochs.size or not metric_names:
        return None

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(8, 3.0 * len(metric_names)),
                             sharex=True, squeeze=False)
    for i, metric in enumerate(metric_names):
        ax = axes[i][0]
        values = np.asarray(val_metrics.get(metric, []), dtype=np.float64)
        if values.size == epochs.size:
            ax.plot(epochs, values, marker="o", markersize=3, color="#d62728")
        display = metric_display.get(metric, {"en": metric, "zh": metric})
        ax.set_ylabel(_text(lang, f"{display['en']}", f"{display['zh']}"))
        ax.grid(True, alpha=0.3)
    axes[-1][0].set_xlabel(_text(lang, "Epoch", "训练轮次"))
    fig.suptitle(_text(lang, "Validation metrics per epoch", "验证集指标随训练轮次的变化"))
    fig.tight_layout()
    path = os.path.join(output_dir or os.path.join(experiment_dir, "图像"), "训练过程指标曲线.png")
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_metrics_vs_snr(
    experiment_dir: str,
    metric: str,
    noise_type: str,
    split_list: List[str],
    snr_levels: np.ndarray,
    lang: str = "en",
    filename: Optional[str] = None,
    draw_reference_line: bool = False,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """绘制某个指标随输入 SNR 的变化曲线（训练/验证/测试三条曲线）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    for split in split_list:
        path = find_metric_path(experiment_dir, split, noise_type, metric)
        if not path:
            continue
        values = np.asarray(np.load(path), dtype=np.float64)
        style = split_styles.get(split, {"标签_en": split, "标签_zh": split, "颜色": None})
        ax.plot(snr_levels[:values.size], values, marker="o", markersize=4, linewidth=1.6,
                color=style["颜色"], label=_text(lang, style["标签_en"], style["标签_zh"]))
        has_data = True
    if not has_data:
        plt.close(fig)
        return None

    if draw_reference_line:  # SNR-SNR 图上画 y=x 参考线
        ax.plot(snr_levels, snr_levels, linestyle="--", color="gray", linewidth=1.2,
                label=_text(lang, "Input SNR (reference)", "输入 SNR（参考线）"))

    display = metric_display.get(metric, {"en": metric, "zh": metric})
    ax.set_xlabel(_text(lang, "Input SNR (dB, nominal)", "输入 SNR（dB，标称值）"))
    ax.set_ylabel(_text(lang, display["en"], display["zh"]))
    ax.set_title(_text(lang, f"{display['en']} versus input SNR", f"{display['zh']} 随输入 SNR 的变化"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    filename = filename or f"{metric}-SNR.png"
    path = os.path.join(output_dir or os.path.join(experiment_dir, "图像"), filename)
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_metrics_overview(
    experiment_dir: str,
    summary: Dict[str, Any],
    split_list: List[str],
    metric_names: List[str],
    lang: str = "en",
    output_dir: Optional[str] = None,
) -> str:
    """绘制三个划分的“整体指标”柱状图（每个指标一个子图）。"""
    overall = summary.get("各划分整体指标", {}) or {}
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.2 * len(metric_names), 4.2), squeeze=False)
    for i, metric in enumerate(metric_names):
        ax = axes[0][i]
        values = []
        labels = []
        for split in split_list:
            if split in overall and metric in overall[split]:
                values.append(overall[split][metric])
                style = split_styles.get(split, {"标签_en": split, "标签_zh": split})
                labels.append(_text(lang, style["标签_en"], style["标签_zh"]))
        colors = [split_styles.get(p, {}).get("颜色") for p in split_list[: len(values)]]
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels)
        display = metric_display.get(metric, {"en": metric, "zh": metric})
        ax.set_title(_text(lang, display["en"], display["zh"]))
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(_text(lang, "Overall metrics by split", "各数据划分的整体指标"))
    fig.tight_layout()
    path = os.path.join(output_dir or os.path.join(experiment_dir, "图像"), "指标总览.png")
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# 总入口
# --------------------------------------------------------------------------------------

def plot_all(
    experiment_dir: str,
    lang: str = "en",
    split_list: Optional[List[str]] = None,
    logger: Any = None,
    output_dir: Optional[str] = None,
) -> List[str]:
    """训练完成后调用：绘制全部图像并返回文件路径列表（需求 6、7）。

    :param output_dir: 图像输出目录，默认写入 ``<实验目录>/图像/``（需求 2 的分目录保存）。
    """
    lang = configure_fonts(lang)
    summary = read_summary(experiment_dir)
    prefix = summary.get("文件前缀")
    noise_type = summary.get("噪声类型", "EMG")
    metric_names = list(summary.get("评价指标") or ["MSE", "SNR", "CC", "RRMSE"])
    split_list = split_list or ["train", "val", "test"]
    output_dir = ensure_dir(output_dir or os.path.join(experiment_dir, "图像"))

    artifacts: List[str] = []
    history = None
    try:
        history = read_train_history(experiment_dir, prefix)
    except FileNotFoundError:
        # 历史文件缺失时退回用 Loss 文件画 loss 曲线
        try:
            history = read_loss(experiment_dir, prefix)
        except FileNotFoundError:
            history = None

    if history is not None:
        artifacts.append(plot_loss_curve(experiment_dir, history, lang, output_dir))
        path = plot_training_curves(experiment_dir, history, metric_names, lang, output_dir)
        if path:
            artifacts.append(path)

    snr_levels = read_snr_levels(experiment_dir)
    for metric in metric_names:  # MSE-SNR / SNR-SNR / CC-SNR / RRMSE-SNR
        path = plot_metrics_vs_snr(
            experiment_dir, metric, noise_type, split_list, snr_levels, lang,
            filename=f"{metric}-SNR.png", draw_reference_line=(metric == "SNR"),
            output_dir=output_dir,
        )
        if path:
            artifacts.append(path)

    artifacts.append(
        plot_metrics_overview(experiment_dir, summary, split_list, metric_names, lang, output_dir)
    )

    if callable(logger):
        logger(f"已绘制图像 {len(artifacts)} 张：" + "，".join(os.path.basename(p) for p in artifacts))
    return artifacts
