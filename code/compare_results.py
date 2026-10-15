#!/usr/bin/env python3
"""实验结果对比绘图工具。

功能：从 ``results/`` 下的实验里读取 loss 与各指标（npy），为**每个噪声类型**生成
一套论文可直接使用的对比图。**本脚本不会自动运行**，需要手动执行。

核心约定
--------
1. **K 次独立重复实验聚合成每个模型一条曲线**，两个参数分别控制“中心曲线”和“误差带”：
   - ``--aggregate mean``（默认，均值）/ ``median``（中位数）/ ``all``（K 次全部画细线）；
   - ``--shade std``（默认，±标准差）/ ``sem``（±标准误）/ ``iqr``（±四分位距/2）/
     ``none``（**不画阴影**，只要均值曲线）。
   两个参数自由组合，例如 ``--aggregate mean --shade none`` 就是只有均值折线。
2. 输出按**噪声类型**分目录（EMG/EOG 的 SNR 范围与图尺寸不同，混在一起无法统一）::

       results/对比图/
       ├── EMG去噪_512Hz/
       │   ├── EMG去噪_512Hz_对比_CC_test.png
       │   ├── EMG去噪_512Hz_对比_MSE_test.png
       │   ├── EMG去噪_512Hz_对比_SNR_test.png
       │   ├── EMG去噪_512Hz_对比_RRMSE_test.png
       │   ├── EMG去噪_512Hz_对比_训练Loss.png   ← 每个模型只画“K 次里最好的那一次”
       │   ├── EMG去噪_512Hz_对比_验证Loss.png
       │   └── EMG去噪_512Hz_对比汇总_test.csv
       └── EOG去噪_256Hz/ ...

3. 所有图片**尺寸统一**（5.4 x 5.4 英寸方图，200 dpi），**不画标题**，
   横轴统一 ``SNR(dB)``，纵轴只写指标缩写（``CC`` / ``MSE`` / ``SNR`` / ``RRMSE`` / ``Loss``）。
   如果模型本身对不上，可用 ``--highlight Novel_CNN`` 把它标红并在图例写 ``Novel_CNN*``。
4. ``对比_训练Loss.png`` / ``对比_验证Loss.png``：每个模型只取 **K 次实验中验证集 MSE 最低
   （即最终保存下来的那次权重）** 的 loss 曲线，避免 5 个模型 x 10 次 = 50 条曲线糊在一起。

用法示例
--------
对比全部实验（自动分出 EMG / EOG 两套图）::

    python code/compare_results.py --result-dir results

只对比 EMG、突出显示自己提出的模型::

    python code/compare_results.py --noise-type EMG --highlight Novel_CNN

只画均值曲线、不要阴影::

    python code/compare_results.py --shade none

10 次实验全部画出来（不聚合，细线）::

    python code/compare_results.py --aggregate all
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plotting import (  # noqa: E402
    configure_fonts,
    read_metrics,
    read_snr_levels,
    read_summary,
    read_train_history,
)
from utils import (  # noqa: E402
    NOISE_SAMPLE_RATE,
    ResultLayout,
    SPLIT_LABELS,
    ensure_dir,
    find_result_layouts,
    read_trial_metrics,
    resolve_result_root,
)

#: 从实验文件夹名里提取试验次数（兼容 ``试验次数_3`` 与旧版 ``..._试验次数_3``）
_TRIAL_RE = re.compile(r"试验次数_(\d+)")

#: 统一图片尺寸（英寸）与 dpi，保证同一批图在论文里大小一致
FIG_SIZE: Tuple[float, float] = (5.4, 5.4)
DPI = 200

#: 曲线颜色循环：取“高区分度 + 色盲友好”的一组颜色（相邻两个模型不会撞色）
COLOR_CYCLE = [
    "#0072B2",  # 深蓝
    "#E69F00",  # 琥珀黄
    "#009E73",  # 翠绿
    "#8B0000",  # 深红（比 --highlight 的红色更暗，两者不会混淆）
    "#6A3D9A",  # 深紫
    "#8C564B",  # 棕
    "#E377C2",  # 粉
    "#7F7F7F",  # 中灰
    "#BCBD22",  # 橄榄黄
    "#17BECF",  # 青
]

#: 重点模型专用色（--highlight）
HIGHLIGHT_COLOR = "#d62728"

#: 纵轴标签只写指标缩写
METRIC_AXIS_LABEL = {
    "MSE": "MSE",
    "SNR": "SNR(dB)",
    "CC": "CC",
    "RRMSE": "RRMSE",
    "loss": "Loss",
}


# --------------------------------------------------------------------------------------
# 发现实验
# --------------------------------------------------------------------------------------

def find_trial_number(path: str) -> Optional[int]:
    """从实验路径里提取试验次数（``试验次数_3`` → 3）。"""
    match = _TRIAL_RE.search(str(path))
    return int(match.group(1)) if match else None


def group_by_noise_type(result_dir: str) -> Dict[str, List[ResultLayout]]:
    """按噪声类型把结果目录下全部实验分组（只保留目录真实存在的）。"""
    grouped: Dict[str, List[ResultLayout]] = {}
    for layout in find_result_layouts(result_dir):
        if not os.path.isdir(layout.experiment_dir):
            continue
        grouped.setdefault(layout.noise_type, []).append(layout)
    for layouts in grouped.values():
        layouts.sort(key=lambda item: (item.model_name, item.trial))
    return grouped


def filter_layouts(layouts: Sequence[ResultLayout], models: Optional[Sequence[str]],
                   trials: Optional[Sequence[int]]) -> List[ResultLayout]:
    """按模型名与试验次数筛选。"""
    result = list(layouts)
    if models:
        wanted = {str(m) for m in models}
        result = [item for item in result if item.model_name in wanted]
    if trials:
        wanted_trials = {int(t) for t in trials}
        result = [item for item in result if item.trial in wanted_trials]
    return result


def model_names_in_order(layouts: Sequence[ResultLayout]) -> List[str]:
    """稳定的模型顺序：按每个模型第一次出现的先后。"""
    names: List[str] = []
    for layout in layouts:
        if layout.model_name not in names:
            names.append(layout.model_name)
    return names


def noise_tag(layouts: Sequence[ResultLayout], noise_type: str) -> str:
    """返回文件名前缀，形如 ``EMG去噪_512Hz``（采样率优先取实验摘要里的值）。"""
    sample_rate = 0
    for layout in layouts:
        try:
            sample_rate = int(read_summary(layout.experiment_dir).get("采样率") or 0)
        except Exception:
            sample_rate = 0
        if sample_rate:
            break
    if not sample_rate:  # 摘要缺失时退回默认采样率
        sample_rate = NOISE_SAMPLE_RATE.get(noise_type.upper(), 0)
    return f"{noise_type.upper()}去噪_{sample_rate}Hz"


def comparison_dir(result_dir: str, tag: str) -> str:
    """对比图输出目录：``results/对比图/<噪声标签>/``。"""
    return ensure_dir(os.path.join(result_dir, "对比图", tag))


# --------------------------------------------------------------------------------------
# 读取：每个划分、每个指标的 K 次重复
# --------------------------------------------------------------------------------------

def load_metric_curves(
    layouts: Sequence[ResultLayout], noise_type: str, split: str, metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    """读取某划分/指标在全部实验里的曲线，返回 ``(snr档位, 形状(K,N)的数值)``。

    缺文件的实验会被跳过；SNR 档位以第一个实验为准。
    """
    snr_levels: Optional[np.ndarray] = None
    rows: List[np.ndarray] = []
    for layout in layouts:
        try:
            values = read_metrics(layout.experiment_dir, split, noise_type, metric)
        except FileNotFoundError:
            continue
        if snr_levels is None:
            snr_levels = read_snr_levels(layout.experiment_dir)
        rows.append(np.asarray(values, dtype=np.float64))
    if not rows:
        return np.asarray([]), np.asarray([])
    width = min(row.size for row in rows)  # 防御：形状不一致时截齐
    matrix = np.vstack([row[:width] for row in rows])
    if snr_levels is not None:
        snr_levels = np.asarray(snr_levels, dtype=np.float64)[:width]
    else:
        snr_levels = np.arange(width, dtype=np.float64)
    return snr_levels, matrix


def summarise(matrix: np.ndarray, aggregate: str = "mean", shade: str = "std"
              ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """把 ``(K,N)`` 的 K 次重复聚合成 ``(中心曲线, 误差带或None)``。

    :param aggregate: ``mean``（均值）或 ``median``（中位数）
    :param shade: 误差带类型；``none`` 表示**不画阴影**，只画中心曲线
        - ``std``：标准差
        - ``sem``：标准误（std/√K）
        - ``iqr``：四分位距/2
    """
    with np.errstate(invalid="ignore"):
        center = (np.nanmedian(matrix, axis=0) if aggregate == "median"
                  else np.nanmean(matrix, axis=0))
        if shade == "none" or matrix.shape[0] < 2:
            return center, None
        if shade == "iqr":
            q1 = np.nanpercentile(matrix, 25, axis=0)
            q3 = np.nanpercentile(matrix, 75, axis=0)
            return center, (q3 - q1) / 2.0
        std = np.nanstd(matrix, axis=0, ddof=1)
        if shade == "sem":
            return center, std / np.sqrt(max(matrix.shape[0], 1))
        return center, std


def center_label(aggregate: str, shrink: float = 1.0) -> str:
    """中心曲线的说明文字（``shrink`` 用于 ``--aggregate all`` 时的曲线变细说明）。"""
    if aggregate == "median":
        return "median"
    return "mean"


def band_label(aggregate: str, shade: str) -> str:
    """中心曲线 + 误差带的说明文字，例如 ``mean±STD`` / ``mean``（不画阴影）。"""
    name = center_label(aggregate)
    if shade == "none" or aggregate == "all":
        return name
    return f"{name}±{shade.upper() if shade != 'iqr' else 'IQR/2'}"


# --------------------------------------------------------------------------------------
# 逐次实验的汇总表
# --------------------------------------------------------------------------------------

def build_summary_rows(
    layouts: Sequence[ResultLayout], metrics: Sequence[str], splits: Sequence[str]
) -> List[Dict[str, Any]]:
    """每次实验一行：模型、试验次数、各划分各指标的整体值。"""
    rows: List[Dict[str, Any]] = []
    for layout in layouts:
        seed, best_epoch, best_val = None, None, None
        try:
            summary = read_summary(layout.experiment_dir)
            seed = summary.get("随机种子")
            best_epoch = summary.get("最优轮次")
            best_val = summary.get("最优验证损失")
        except Exception:
            pass
        row: Dict[str, Any] = {
            "模型": layout.model_name,
            "试验次数": layout.trial,
            "随机种子": seed,
            "最优轮次": best_epoch,
            "最优验证损失": best_val,
            "实验目录": layout.experiment_dir,
        }
        for split in splits:
            for metric in metrics:
                row[f"{split}_{metric}"] = read_trial_metrics(layout.experiment_dir, metric, split)
        rows.append(row)
    return rows


def save_comparison_csv(rows: Sequence[Dict[str, Any]], metrics: Sequence[str],
                        splits: Sequence[str], path: str) -> str:
    """把每次实验的整体指标汇总成 CSV（UTF-8-BOM，Excel 可直接打开）。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    header = ["模型", "试验次数", "随机种子", "最优轮次", "最优验证损失", "实验目录"]
    for split in splits:
        for metric in metrics:
            header.append(f"{SPLIT_LABELS.get(split, split)}_{metric}")
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            values = [row.get("模型"), row.get("试验次数"), row.get("随机种子"),
                      row.get("最优轮次"), row.get("最优验证损失"), row.get("实验目录")]
            for split in splits:
                for metric in metrics:
                    value = row.get(f"{split}_{metric}")
                    values.append("" if value is None else f"{float(value):.10g}")
            writer.writerow(values)
    return path


# --------------------------------------------------------------------------------------
# 绘图：统一风格
# --------------------------------------------------------------------------------------

def style_axes(ax: Any) -> None:
    """统一坐标轴风格：细边框、内向刻度、浅网格。"""
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")
    ax.tick_params(direction="in", length=4, width=0.9, colors="#333333", labelsize=10)


def save_square(fig: Any, path: str) -> str:
    """按统一尺寸保存方图。"""
    fig.set_size_inches(*FIG_SIZE)
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fig.savefig(path, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return path


def metric_axis_label(metric: str) -> str:
    """纵轴标签：只写指标缩写。"""
    return METRIC_AXIS_LABEL.get(metric, metric)


def plot_metric_vs_snr(
    layouts: Sequence[ResultLayout],
    noise_type: str,
    split: str,
    metric: str,
    output_dir: str,
    tag: str,
    aggregate: str = "mean",
    shade: str = "std",
    highlight: Optional[str] = None,
) -> Optional[str]:
    """画“横轴 SNR、纵轴某指标”的模型对比图。

    ``aggregate`` 决定中心曲线（mean/median，``all`` = 每条实验各画一条），
    ``shade`` 决定是否画误差带（std/sem/iqr/none）。
    """
    fig, ax = plt.subplots()
    has_data = False
    for index, model in enumerate(model_names_in_order(layouts)):
        model_layouts = [item for item in layouts if item.model_name == model]
        snr, matrix = load_metric_curves(model_layouts, noise_type, split, metric)
        if matrix.size == 0:
            continue
        color = HIGHLIGHT_COLOR if (highlight and model == highlight) else \
            COLOR_CYCLE[index % len(COLOR_CYCLE)]
        label = f"{model}*" if (highlight and model == highlight) else model
        if aggregate == "all":
            # 每一次实验各画一条细线（只用于细节核查，牺牲可读性换信息量）
            for run_index in range(matrix.shape[0]):
                ax.plot(snr, matrix[run_index], color=color, linewidth=0.8, alpha=0.28,
                        label=label if run_index == 0 else None)
        else:
            center, band = summarise(matrix, aggregate, shade)
            if band is not None:  # shade == "none" 时 band 为 None，即不画阴影
                ax.fill_between(snr, center - band, center + band, color=color, alpha=0.18,
                                linewidth=0)
            ax.plot(snr, center, color=color, linewidth=1.9, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, label=label)
        has_data = True
    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel("SNR(dB)", fontsize=11)
    ax.set_ylabel(metric_axis_label(metric), fontsize=11)
    style_axes(ax)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#999999", loc="best")
    if highlight:
        ax.text(0.985, 0.02, f"* {highlight}   {band_label(aggregate, shade)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color=HIGHLIGHT_COLOR)
    return save_square(fig, os.path.join(output_dir, f"{tag}_对比_{metric}_{split}.png"))


def plot_loss_comparison(
    layouts: Sequence[ResultLayout],
    output_dir: str,
    tag: str,
    highlight: Optional[str] = None,
) -> List[str]:
    """画训练/验证 loss 对比：每个模型只取 K 次里**验证集 MSE 最低**的那一次。"""
    series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model in model_names_in_order(layouts):
        model_layouts = [item for item in layouts if item.model_name == model]
        # 选出验证集 MSE 最低（即最终权重对应的）一次实验
        scored: List[Tuple[float, ResultLayout]] = []
        for layout in model_layouts:
            try:
                value = float(read_summary(layout.experiment_dir).get("最优验证损失"))
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                scored.append((value, layout))
        if not scored:
            continue
        _, best = min(scored, key=lambda item: item[0])
        try:
            summary = read_summary(best.experiment_dir)
            history = read_train_history(best.experiment_dir, summary.get("文件前缀"))
        except Exception:
            continue
        epochs = np.asarray(history.get("epochs", []), dtype=np.float64)
        if not epochs.size:
            continue
        series[model] = (
            epochs,
            np.asarray(history.get("train_loss", []), dtype=np.float64),
            np.asarray(history.get("val_loss", []), dtype=np.float64),
        )
    if not series:
        return []

    artifacts: List[str] = []
    for key, filename in (
        ("train", f"{tag}_对比_训练Loss.png"),
        ("val", f"{tag}_对比_验证Loss.png"),
    ):
        fig, ax = plt.subplots()
        for index, model in enumerate(series):
            epochs, train_loss, val_loss = series[model]
            curve = train_loss if key == "train" else val_loss
            color = HIGHLIGHT_COLOR if (highlight and model == highlight) else \
                COLOR_CYCLE[index % len(COLOR_CYCLE)]
            label = f"{model}*" if (highlight and model == highlight) else model
            ax.plot(epochs, curve, color=color, linewidth=1.6, label=label)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        style_axes(ax)
        ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#999999", loc="best")
        if highlight:
            ax.text(0.985, 0.02, "* best of K runs", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=HIGHLIGHT_COLOR)
        artifacts.append(save_square(fig, os.path.join(output_dir, filename)))
    return artifacts


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """构建对比脚本的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="按噪声类型生成论文用的模型对比图（K 次独立实验聚合成均值±标准差）"
    )
    parser.add_argument("--result-dir", default=None,
                        help="结果根目录（默认 <项目根>/results）")
    parser.add_argument("--noise-type", nargs="+", default=None,
                        choices=["EMG", "EOG", "emg", "eog"],
                        help="只对比指定噪声类型（默认自动分出 EMG / EOG 两套图）")
    parser.add_argument("--models", nargs="+", default=None,
                        help="只对比指定模型（默认全部模型）")
    parser.add_argument("--trials", nargs="+", type=int, default=None,
                        help="只用指定的试验次数（默认 1..K 全部）")
    parser.add_argument("--metrics", nargs="+", default=["MSE", "SNR", "CC", "RRMSE"],
                        help="要对比的指标")
    parser.add_argument("--split", nargs="+", default=["test"],
                        choices=["train", "val", "test"], help="对比哪个数据划分（可多选）")
    parser.add_argument("--aggregate", default="mean", choices=["mean", "median", "all"],
                        help="K 次实验聚合成中心曲线的方式：mean=均值（默认）/ median=中位数 / "
                             "all=把 K 次全部画出来（细线）")
    parser.add_argument("--shade", default="std", choices=["std", "sem", "iqr", "none"],
                        help="中心曲线周围的误差带：std=±标准差（默认）/ sem=±标准误 / "
                             "iqr=±四分位距/2 / none=不画阴影（只要均值曲线）")
    parser.add_argument("--highlight", default=None,
                        help="重点模型名（红色突出显示、图例写 <模型>*，一般用来标出自己提出的模型）")
    parser.add_argument("--out", default=None,
                        help="输出目录，默认 <result-dir>/对比图")
    parser.add_argument("--lang", default="en", choices=["en", "zh"], help="图像语言")
    parser.add_argument("--no-loss", action="store_true", help="不画训练/验证 loss 对比图")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_fonts(args.lang)
    result_dir = resolve_result_root(args.result_dir)
    out_root = args.out or os.path.join(result_dir, "对比图")

    grouped = group_by_noise_type(result_dir)
    if args.noise_type:
        wanted = {str(item).upper() for item in args.noise_type}
        grouped = {key: value for key, value in grouped.items() if key in wanted}
    if not grouped:
        print(f"[错误] {result_dir} 下没有找到符合条件的实验结果", file=sys.stderr)
        return 2

    artifacts: List[str] = []
    for noise_type in sorted(grouped):
        layouts = filter_layouts(grouped[noise_type], args.models, args.trials)
        if not layouts:
            print(f"[跳过] {noise_type}: 按当前筛选条件没有实验")
            continue
        tag = noise_tag(layouts, noise_type)
        output_dir = comparison_dir(result_dir, tag) if args.out is None else ensure_dir(args.out)
        models = model_names_in_order(layouts)
        print(f"\n[{noise_type}] {tag}：{len(layouts)} 次实验，模型 {models}")
        print(f"  中心曲线：{center_label(args.aggregate)}；误差带：{args.shade}"
              f"（{band_label(args.aggregate, args.shade)}）；输出目录：{output_dir}")

        for split in args.split:
            for metric in args.metrics:
                path = plot_metric_vs_snr(layouts, noise_type, split, metric, output_dir, tag,
                                          aggregate=args.aggregate, shade=args.shade,
                                          highlight=args.highlight)
                if path:
                    artifacts.append(path)
                    print(f"  - {os.path.basename(path)}")

        if not args.no_loss:
            for path in plot_loss_comparison(layouts, output_dir, tag, highlight=args.highlight):
                artifacts.append(path)
                print(f"  - {os.path.basename(path)}")

        rows = build_summary_rows(layouts, args.metrics, args.split)
        for split in args.split:
            csv_path = save_comparison_csv(
                rows, args.metrics, [split],
                os.path.join(output_dir, f"{tag}_对比汇总_{split}.csv"),
            )
            artifacts.append(csv_path)
            print(f"  - {os.path.basename(csv_path)}")

    print(f"\n对比图已保存到：{out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
