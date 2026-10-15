#!/usr/bin/env python3
"""推理与可视化：用「最后一次（第 K 次）实验保存的权重」在当次测试集上推理，逐样本出图。

为什么可以直接叠三条曲线？
--------------------------
K 次独立重复实验中，**同一次实验的划分数据是被所有模型复用的**（第一个模型训练时用
``seed + 试验次数 - 1`` 生成并落盘，其余模型直接加载），所以：

- 第 K 次实验里，所有模型的测试集 **逐元素完全相同**（噪声、干净信号、SNR 档位都一致）；
- 因此可以放心地把 “Noisy EEG / Pure EEG / Denoised EEG” 画在同一张图上对比。

如果第 K 次的 ``划分数据/`` 已被自动清理，本脚本会用该次实验 ``训练配置.json`` 里记录的
``seed / combin_num / train_per`` **重新推导**出完全相同的测试集（已验证逐元素一致）。

目录结构
--------
::

    results/Inference Result/
    └── EMG去噪_512Hz/                      # <噪声类型>去噪_<采样率>Hz
        ├── 推理配置.json                    # 本次推理用的权重、测试集来源、样本数等
        ├── 划分数据/                        # 本次推理实际使用的测试集（可直接核对）
        └── 测试样本_0001/                   # 按测试集样本顺序编号 1..K
            ├── fcNN.png                     # 每个模型一张图（文件名 = 模型名）
            ├── Simple_CNN.png
            └── ...

每张图里：**Noisy EEG = 灰色、Pure EEG = 绿色、Denoised EEG = 蓝色**，
横轴 ``Time(sample)``、纵轴 ``Amplitude``。

用法
----
全部模型、最后一次实验（默认）::

    python code/inference.py --noise-type EMG

指定实验次数、只跑前 20 个样本、只要两个模型::

    python code/inference.py --noise-type EMG --trial 10 --max-samples 20 \
        --models fcNN Novel_CNN

先看有哪些模型/实验次数可用::

    python code/inference.py --noise-type EMG --list
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np  # noqa: E402

from utils import (  # noqa: E402
    NOISE_DATA_FILES,
    ResultLayout,
    ensure_dir,
    load_json,
    noise_label,
    resolve_result_root,
    save_json,
)

#: 推理结果根目录名（放在 results/ 下）
INFERENCE_ROOT_NAME = "Inference Result"

#: 三条曲线的颜色与标签（按需求固定：噪声灰、纯净绿、去噪蓝）
CURVE_STYLE: Tuple[Tuple[str, str, str], ...] = (
    ("Noisy EEG", "#9E9E9E", "Noisy EEG"),
    ("Pure EEG", "#2CA02C", "Pure EEG"),
    ("Denoised EEG", "#1F77B4", "Denoised EEG"),
)

#: 推理图尺寸（比论文用图宽一些，因为要显示整段 512/1024 点的时间序列）
FIG_SIZE: Tuple[float, float] = (7.6, 4.2)
DPI = 150


# --------------------------------------------------------------------------------------
# SNR 档位选择
# --------------------------------------------------------------------------------------

#: ``--snr`` 支持的关键字
SNR_KEYWORDS: Dict[str, str] = {
    "worst": "最低 SNR（噪声最强，去噪收益最明显）",
    "best": "最高 SNR（噪声最弱）",
    "mid": "中间档位",
    "all": "全部档位",
}


def resolve_snr_levels(selection: Optional[Sequence[str]], levels: np.ndarray
                       ) -> Tuple[List[float], bool]:
    """把 ``--snr`` 的取值解析成具体档位列表。

    支持 ``worst`` / ``best`` / ``mid`` / ``all`` 以及显式数值（如 ``-4`` ``0`` ``1.5``）。

    :return: ``(档位值列表, 是否为默认的单档模式)``
    """
    available = [float(v) for v in np.asarray(levels, dtype=np.float64)]
    if not available:
        return [], True
    if not selection:  # 默认：最难的那一档
        return [min(available)], True

    chosen: List[float] = []
    for raw in selection:
        text = str(raw).strip().lower()
        if text in SNR_KEYWORDS:
            if text == "worst":
                wanted = [min(available)]
            elif text == "best":
                wanted = [max(available)]
            elif text == "mid":
                middle = available[len(available) // 2]
                wanted = [middle]
            else:  # all
                wanted = list(available)
        else:
            try:
                value = float(text)
            except ValueError as error:
                raise ValueError(
                    f"--snr 只能是 {list(SNR_KEYWORDS)} 或具体数值（dB），收到：{raw}"
                ) from error
            # 允许写 -4 而档位是 -4.0；容差取档位间距的一小部分
            step = (max(available) - min(available)) / max(len(available) - 1, 1)
            nearest = min(available, key=lambda item: abs(item - value))
            if abs(nearest - value) > max(step * 0.25, 1e-6):
                raise ValueError(
                    f"--snr {raw} 不在测试集的 SNR 档位里，可选：{available}"
                )
            wanted = [nearest]
        for value in wanted:
            if value not in chosen:
                chosen.append(value)
    return chosen, len(chosen) == 1 and len(selection) == 0


def snr_dir_name(snr_value: float) -> str:
    """返回每个 SNR 档位对应的子文件夹名，例如 ``-7dB`` / ``+0dB`` / ``1.5dB``。

    每个样本文件夹下按 ``--snr`` 选中的档位各建一个子文件夹，
    对应档位的推理图就放在里面。
    """
    value = float(snr_value)
    text = f"{value:+.0f}" if value.is_integer() else f"{value:+.1f}"
    return f"{text}dB"


# --------------------------------------------------------------------------------------
# 发现「第 K 次实验」的权重
# --------------------------------------------------------------------------------------

def noise_tag(noise_type: str, sample_rate: Optional[int] = None) -> str:
    """返回目录标签，形如 ``EMG去噪_512Hz``（用 Hz，与 results/对比图/ 一致）。"""
    if sample_rate:
        return f"{noise_type.upper()}去噪_{int(sample_rate)}Hz"
    return noise_label(noise_type).replace("HZ", "Hz")


def list_trials(result_dir: str, model_name: str, noise_type: str) -> List[int]:
    """列出某模型在某噪声类型下已完成的实验次数（有目录即算）。"""
    layout = ResultLayout(result_dir, model_name, noise_type, 1)
    if not os.path.isdir(layout.noise_dir):
        return []
    trials: List[int] = []
    for name in sorted(os.listdir(layout.noise_dir)):
        if not name.startswith("试验次数_"):
            continue
        tail = name[len("试验次数_"):]
        if tail.isdigit() and os.path.isdir(os.path.join(layout.noise_dir, name)):
            trials.append(int(tail))
    return sorted(trials)


def discover_models(result_dir: str, noise_type: str) -> List[str]:
    """找出在某噪声类型下有训练结果的模型（按名字排序）。"""
    models: List[str] = []
    if not os.path.isdir(result_dir):
        return models
    for name in sorted(os.listdir(result_dir)):
        if name in ("对比图", INFERENCE_ROOT_NAME, "README.md"):
            continue
        model_dir = os.path.join(result_dir, name)
        if not os.path.isdir(model_dir):
            continue
        if list_trials(result_dir, name, noise_type):
            models.append(name)
    return models


def collect_inference_targets(
    result_dir: str,
    noise_type: str,
    models: Optional[Sequence[str]] = None,
    trial: Optional[int] = None,
) -> Tuple[int, List[Tuple[str, ResultLayout, Dict[str, Any]]]]:
    """确定要用哪一次实验、以及每个模型对应的权重文件。

    :return: ``(试验次数, [(模型名, 布局, 权重内容), ...])``
    """
    import torch

    candidates = list(models) if models else discover_models(result_dir, noise_type)
    if not candidates:
        raise FileNotFoundError(f"在 {result_dir} 下没找到 {noise_type} 的任何模型训练结果")

    # 试验次数：不指定就用“所有模型里最大的那一次”（= 最后一次独立重复实验）
    if trial is None:
        available = [t for model in candidates for t in list_trials(result_dir, model, noise_type)]
        if not available:
            raise FileNotFoundError(f"{noise_type}: 没有找到任何 试验次数_N 目录")
        trial = max(available)

    targets: List[Tuple[str, ResultLayout, Dict[str, Any]]] = []
    missing: List[str] = []
    for model in candidates:
        layout = ResultLayout(result_dir, model, noise_type, int(trial))
        if not os.path.isfile(layout.weights_path):
            missing.append(f"{model}（缺 {os.path.relpath(layout.weights_path, result_dir)}）")
            continue
        content = torch.load(layout.weights_path, map_location="cpu", weights_only=False)
        targets.append((model, layout, content))
    if missing:
        raise FileNotFoundError(
            f"以下模型在第 {trial} 次实验里找不到权重文件：\n  - " + "\n  - ".join(missing)
        )
    return int(trial), targets


# --------------------------------------------------------------------------------------
# 复现第 K 次实验的测试集
# --------------------------------------------------------------------------------------

def read_trial_config(layout: ResultLayout, trial: int) -> Dict[str, Any]:
    """读取该次实验的训练配置（取不到时用全局默认值兜底）。"""
    config: Dict[str, Any] = {}
    if os.path.isfile(layout.config_path):
        config = dict(load_json(layout.config_path).get("训练参数", {}) or {})
    elif os.path.isfile(layout.summary_path):
        config = dict(load_json(layout.summary_path) or {})
    return config


def prepare_test_split(
    layout: ResultLayout,
    config: Dict[str, Any],
    noise_type: str,
    logger: Any = None,
    save_dir: Optional[str] = None,
):
    """复现该次实验的测试集。

    优先使用该实验目录里仍然存在的 ``划分数据/``；若已被清理，则按训练配置里的
    ``seed / combin_num / train_per`` 重新推导（与训练时逐元素一致）。
    """
    from data_prepare import load_split, prepare_data, save_split

    if os.path.isfile(os.path.join(layout.split_dir, "划分信息.json")):
        data = load_split(layout.split_dir)
        source = f"复用该实验已保存的划分数据：{layout.split_dir}"
    else:
        data_dir = config.get("data_dir") or os.path.join(
            os.path.dirname(CODE_DIR), "data"
        )
        files = NOISE_DATA_FILES[noise_type.upper()]
        eeg = np.load(os.path.join(data_dir, files["eeg"])).astype(np.float64)
        noise = np.load(os.path.join(data_dir, files["noise"])).astype(np.float64)
        seed = config.get("seed")
        seed = None if seed is None or int(seed) < 0 else int(seed)
        data = prepare_data(
            EEG_all=eeg, noise_all=noise,
            combin_num=int(config.get("combin_num", 10)),
            train_per=float(config.get("train_per", 0.8)),
            noise_type=noise_type.upper(),
            seed=seed,
        )
        source = (f"按训练配置重新推导（seed={seed}，combin_num={config.get('combin_num')}，"
                  f"train_per={config.get('train_per')}）")
    if callable(logger):
        logger(f"测试集来源：{source}；{data.describe()}")
    if save_dir:
        ensure_dir(save_dir)
        save_split(save_dir, data)
    return data, source


def purge_dir_contents(directory: str) -> int:
    """删除目录下的全部内容（只用于脚本自己创建的样本/SNR 目录），返回删除的条目数。"""
    import shutil

    if not os.path.isdir(directory):
        return 0
    count = 0
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
        count += 1
    return count


def clean_previous_images(out_root: str, models: Sequence[str], logger: Any = None) -> int:
    """清理上次推理留下的旧图，避免目录里混入过期结果。

    由于本次运行会重建每个 ``测试样本_*`` 目录下的全部内容（含各 SNR 子文件夹与旧版本
    直接写在样本目录里的图片），这里把除 ``划分数据`` 之外的样本目录内容整体清掉。
    推理是幂等的：每个样本 × 每个选中 SNR × 每个模型都会被重新生成。
    """
    removed = 0
    if not os.path.isdir(out_root):
        return removed
    for name in sorted(os.listdir(out_root)):
        folder = os.path.join(out_root, name)
        if not name.startswith("测试样本_") or not os.path.isdir(folder):
            continue
        removed += purge_dir_contents(folder)
    if removed and callable(logger):
        logger(f"已清理上次推理留下的 {removed} 项旧内容（模型：{sorted(models)}）")
    return removed


# --------------------------------------------------------------------------------------
# 推理
# --------------------------------------------------------------------------------------

def denoise_all(
    model: Any,
    noisy: np.ndarray,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """分块前向推理，返回去噪后的波形（float32，形状与输入一致）。"""
    import torch

    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, noisy.shape[0], batch_size):
            stop = min(start + batch_size, noisy.shape[0])
            batch = torch.as_tensor(noisy[start:stop], dtype=torch.float32, device=device)
            prediction = model(batch)
            if prediction.shape != batch.shape:
                raise RuntimeError(
                    f"模型输出形状 {tuple(prediction.shape)} 与输入 {tuple(batch.shape)} 不一致；"
                    "框架要求所有模型满足 输入[B,L] -> 输出[B,L]"
                )
            outputs.append(prediction.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros_like(noisy, dtype=np.float32)


# --------------------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------------------

def plot_sample(
    noisy: np.ndarray,
    clean: np.ndarray,
    denoised: np.ndarray,
    path: str,
    snr_db: Optional[float] = None,
) -> str:
    """画一个样本的三条曲线（Noisy 灰 / Pure 绿 / Denoised 蓝）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plotting import configure_fonts

    configure_fonts("en")  # 图内只用英文，避免中文字体依赖

    x = np.arange(noisy.shape[0])
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(x, noisy, color=CURVE_STYLE[0][1], linewidth=0.9, alpha=0.85,
            label=CURVE_STYLE[0][2], zorder=1)
    ax.plot(x, clean, color=CURVE_STYLE[1][1], linewidth=1.1, alpha=0.95,
            label=CURVE_STYLE[1][2], zorder=2)
    ax.plot(x, denoised, color=CURVE_STYLE[2][1], linewidth=1.1, alpha=0.95,
            label=CURVE_STYLE[2][2], zorder=3)

    ax.set_xlabel("Time(sample)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")
    ax.tick_params(direction="in", length=4, width=0.9, labelsize=9.5)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#999999", loc="best", ncol=3)
    if snr_db is not None and np.isfinite(snr_db):
        ax.set_title(f"Input SNR = {snr_db:+.0f} dB (nominal)", fontsize=9.5, color="#444444")

    fig.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def run_inference(
    result_dir: str,
    noise_type: str,
    models: Optional[Sequence[str]] = None,
    trial: Optional[int] = None,
    max_samples: Optional[int] = None,
    snr_selection: Optional[Sequence[str]] = None,
    annotate: bool = True,
    sample_index_offset: int = 1,
    device: str = "auto",
    batch_size: int = 128,
    output_root: Optional[str] = None,
    logger: Any = print,
) -> Dict[str, Any]:
    """完整跑一遍推理并出图，返回结果摘要（供 main.py 串起来调用）。"""
    from metrics import compute_single_metric
    from network import create_model
    from utils import select_device

    result_dir = resolve_result_root(result_dir)
    noise_type = str(noise_type).upper()
    device = select_device(device)

    trial, targets = collect_inference_targets(result_dir, noise_type, models, trial)
    logger(f"推理设置：噪声类型={noise_type}，实验次数={trial}（最后一次独立重复实验），"
           f"模型={[m for m, _, _ in targets]}，设备={device}")

    # 用第一个模型的训练配置复现测试集（所有模型同一次实验共用同一份划分）
    first_model, first_layout, _ = targets[0]
    config = read_trial_config(first_layout, trial)
    sample_rate = None
    try:
        sample_rate = int(load_json(first_layout.summary_path).get("采样率"))
    except Exception:
        sample_rate = None

    tag = noise_tag(noise_type, sample_rate)
    out_root = ensure_dir(output_root or os.path.join(result_dir, INFERENCE_ROOT_NAME, tag))

    data, split_source = prepare_test_split(
        first_layout, config, noise_type, logger,
        save_dir=os.path.join(out_root, "划分数据"),
    )
    noisy, clean, snr = data.test_noisy, data.test_clean, data.test_snr
    # 测试集排布 = 外层 SNR 档位、内层样本（见 data_prepare），所以每档样本数 = 总数 / 档位数
    num_levels = int(len(np.unique(snr))) if snr.size else 1
    per_level = noisy.shape[0] // max(num_levels, 1)
    num_samples = per_level
    if max_samples is not None:
        num_samples = min(num_samples, int(max_samples))
    logger(f"测试集：{per_level} 个样本 x {num_levels} 个 SNR 档位 = {noisy.shape[0]} 条波形；"
           f"本次为每个样本出图（共 {num_samples} 个样本）")

    # 先清掉本次要重画的模型的旧图，避免样本数变化后留下过期图片
    clean_previous_images(out_root, [m for m, _, _ in targets], logger)

    # 每个模型在自己的 SNR 档位上做一次推理，得到逐样本波形
    # 测试集排布：外层 SNR 档位、内层样本，因此第 i 个样本分布在 snr == level 的每个块里
    # 测试集排布是“外层 SNR 档位、内层样本”，因此每个档位的行区间是连续的；
    # np.unique 会返回升序，正好对应块 0..N-1（下面还会做一次一致性断言）
    levels = np.unique(snr) if snr.size else np.array([np.nan])
    try:
        snr_levels, default_single = resolve_snr_levels(snr_selection, levels)
    except ValueError as error:
        raise ValueError(str(error)) from error
    logger(f"使用的 SNR 档位：{snr_levels} dB（测试集共 {len(levels)} 档 "
           f"{float(np.min(levels)):.0f}~{float(np.max(levels)):.0f} dB；"
           f"{'默认取最难的一档' if default_single else '按 --snr 指定'}）；"
           f"每个样本下按 SNR 建 {len(snr_levels)} 个子文件夹："
           + "、".join(snr_dir_name(v) for v in snr_levels))
    sample_positions = np.arange(num_samples)

    summary: Dict[str, Any] = {
        "噪声类型": noise_type,
        "实验次数": int(trial),
        "测试集来源": split_source,
        "测试集样本数": int(num_samples),
        "测试集全部SNR档位": [float(v) for v in levels],
        "本次使用的SNR档位": [float(v) for v in snr_levels],
        "设备": device,
        "输出目录": out_root,
        "模型": {},
    }

    for model_name, layout, content in targets:
        model = create_model(
            content.get("network", model_name),
            int(content.get("datanum", data.signal_length)),
            **dict(content.get("模型初始化参数", {}) or {}),
        ).to(device)
        model.load_state_dict(content["state_dict"])
        logger(f"[{model_name}] 已加载权重（第 {content.get('epoch', '?')} 轮，"
               f"挑选准则 {content.get('挑选划分', '?')}_{content.get('挑选指标', '?')}）")

        # 一致性断言：确保“块号 x 每档样本数”真的落在该 SNR 档位的行区间内
        for block, value in enumerate(levels):
            probe = block * per_level
            if probe < snr.size and abs(float(snr[probe]) - float(value)) > 1e-6:
                raise RuntimeError(
                    "测试集的 SNR 分块与预期不一致（块 "
                    f"{block} 首个样本的 SNR = {float(snr[probe])}，期望 {float(value)}）；"
                    "请检查 划分数据/ 是否与本次实验匹配"
                )

        denoised = denoise_all(model, noisy, device, batch_size)
        metrics: Dict[str, float] = {}
        for metric in ("MSE", "SNR", "CC", "RRMSE"):
            try:
                with np.errstate(invalid="ignore"):
                    metrics[metric] = float(np.nanmean(compute_single_metric(metric, clean, denoised)))
            except Exception as error:  # 指标算不出来不影响出图
                logger(f"[{model_name}] 指标 {metric} 计算失败：{error}")

        # 逐样本出图：样本 i 在每个 SNR 档位块里都位于第 i 个位置
        block_of: Dict[float, int] = {float(value): block for block, value in enumerate(levels)}
        written = 0
        for index in sample_positions:
            folder = ensure_dir(os.path.join(
                out_root, f"测试样本_{index + sample_index_offset:04d}"))
            for snr_value in snr_levels:
                block = block_of.get(float(snr_value))
                if block is None:
                    continue
                row = block * per_level + index
                if row >= noisy.shape[0]:
                    continue
                path = os.path.join(folder, snr_dir_name(snr_value), f"{model_name}.png")
                plot_sample(
                    noisy[row], clean[row], denoised[row], path,
                    snr_db=snr_value if annotate and snr.size else None,
                )
                written += 1

        summary["模型"][model_name] = {
            "权重": os.path.relpath(layout.weights_path, result_dir),
            "权重轮次": int(content.get("epoch", 0) or 0),
            "挑选划分": content.get("挑选划分"),
            "挑选指标": content.get("挑选指标"),
            "随机种子": content.get("随机种子"),
            "整体指标(全部测试样本)": metrics,
            "图片数量": int(written),
            "图片目录结构": "测试样本_XXXX/<SNR>dB/<模型名>.png",
        }
        logger(f"[{model_name}] 已生成 {written} 张图"
               f"（{len(sample_positions)} 个样本 x {len(snr_levels)} 个 SNR 档位）；"
               f"整体指标 " + "，".join(f"{k}={v:.6g}" for k, v in metrics.items()))

    save_json(os.path.join(out_root, "推理配置.json"), summary)
    logger(f"推理结果已保存到：{out_root}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(
        description="用最后一次实验的权重在测试集上推理，逐样本保存可视化图片"
    )
    parser.add_argument("--result-dir", default=None,
                        help="结果根目录（默认 <项目根>/results）")
    parser.add_argument("--noise-type", default="EMG", choices=["EMG", "EOG", "emg", "eog"],
                        help="噪声类型（EMG=512Hz，EOG=256Hz）")
    parser.add_argument("--models", nargs="+", default=None,
                        help="要推理的模型（默认结果目录下该噪声类型的全部模型）")
    parser.add_argument("--trial", type=int, default=None,
                        help="用第几次实验的权重（默认取各模型里最大的，即最后一次）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多画多少个测试样本（默认全部；调试时可用 --max-samples 5）")
    parser.add_argument("--snr", nargs="+", default=None, metavar="SNR",
                        help="用测试集的哪个 SNR 档位：worst(默认，最难 -7dB) / best / mid / all，"
                             "或具体数值（如 -7 -4 0）；每个档位在样本目录下建一个 <SNR>dB 子文件夹")
    parser.add_argument("--no-annotate", action="store_true",
                        help="不在图上写输入 SNR（默认会写一行小标题）")
    parser.add_argument("--sample-index-offset", type=int, default=1,
                        help="样本文件夹编号起始值，默认 1（即 测试样本_0001）")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda:0")
    parser.add_argument("--batch-size", type=int, default=128, help="推理批大小")
    parser.add_argument("--out", default=None,
                        help="输出根目录（默认 <result-dir>/Inference Result/<噪声标签>）")
    parser.add_argument("--list", action="store_true", help="只列出可用的模型与实验次数，不推理")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result_dir = resolve_result_root(args.result_dir)
    noise_type = args.noise_type.upper()

    if args.list:
        models = discover_models(result_dir, noise_type)
        if not models:
            print(f"[错误] {result_dir} 下没有找到 {noise_type} 的训练结果", file=sys.stderr)
            return 2
        from data_prepare import SNR_LEVELS

        print(f"{noise_type} 可用模型与实验次数（测试集 SNR 档位："
              f"{np.round(SNR_LEVELS, 1).tolist()} dB）：")
        for model in models:
            trials = list_trials(result_dir, model, noise_type)
            layout = ResultLayout(result_dir, model, noise_type, max(trials))
            has_weights = os.path.isfile(layout.weights_path)
            print(f"  - {model:14} 实验次数 {trials}，最后一次权重"
                  f"{'存在' if has_weights else '缺失'}")
        return 0

    try:
        run_inference(
            result_dir=result_dir,
            noise_type=noise_type,
            models=args.models,
            trial=args.trial,
            max_samples=args.max_samples,
            snr_selection=args.snr,
            annotate=not args.no_annotate,
            sample_index_offset=max(1, int(args.sample_index_offset)),
            device=args.device,
            batch_size=int(args.batch_size),
            output_root=args.out,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"[错误] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
