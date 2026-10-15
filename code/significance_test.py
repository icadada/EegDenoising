#!/usr/bin/env python3
"""显著性检验：单因素方差分析（One-way ANOVA）+ Tukey HSD 事后检验 + 带显著性标注的箱线图。

背景（需求 3 的统计学部分）
--------------------------
单次实验的结果不可靠，因此每个模型都在 **K 次独立重复实验**下训练：每次实验使用不同的随机
种子，重新随机混合生成半仿真数据、重新划分训练/验证/测试集并随机初始化网络权重。于是每个
模型在每个指标上都得到 K 个**互相独立**的数值。

本脚本对这些原始数据做两件事：

1. **单因素方差分析（One-way ANOVA）**
   - 因子（自变量）：模型（如 fcNN / Simple_CNN / Complex_CNN / RNN_lstm / Novel_CNN）；
   - 因变量：某个具体指标（如 CC 或 RRMSE）在 K 次实验里的 K 个原始数值（不是均值）；
   - 原假设 H0：各模型的该指标**均值相同**；若 ``p < 0.05`` 则认为模型间存在显著差异。
2. **事后检验（Post-hoc, Tukey HSD）**：ANOVA 显著后做两两比较，给出“谁和谁有显著差异”，
   例如 ``Complex_CNN 的 CC 显著高于 fcNN（p = 0.011）``。

并据此画箱线图：在 Tukey HSD 显著的模型对之间画横线并标注 ``*`` / ``**`` / ``***``
（分别代表 ``p < 0.05`` / ``p < 0.01`` / ``p < 0.001``），箱体本身展示 K 次实验的
中位数（红线）、四分位距（箱体，代表方差/分布）、上下边缘（极值）与异常值。

用法
----
只对某个噪声类型做检验（模型自动发现）::

    python code/significance_test.py --noise-type EMG

指定模型与指标::

    python code/significance_test.py --noise-type EMG \
        --models fcNN Simple_CNN Complex_CNN RNN_lstm Novel_CNN \
        --metrics MSE SNR CC RRMSE

指定用哪个数据划分的 K 个结果做检验（默认测试集）::

    python code/significance_test.py --noise-type EOG --split test

产出：

- 箱线图统一放在 ``results/对比图/<噪声>去噪_<采样率>Hz/`` 下，
  文件名形如 ``EMG去噪_512Hz_显著性检验_SNR_箱线图.png``；
- 箱线图的统计量（n / 中位数 / 均值 / 标准差 / IQR）不再画在图上，
  而是输出成 ``EMG去噪_512Hz_显著性检验_统计表.csv``（同一目录下），方便直接做成论文表格；
- ANOVA + Tukey HSD 的完整结果仍写在 ``<模型>/<噪声>去噪_<采样率>HZ/统计分析/`` 下：
  ``<噪声类型>_显著性检验.json`` / ``.csv`` / ``_摘要.json``。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np  # noqa: E402

from utils import (  # noqa: E402
    ResultLayout,
    ensure_dir,
    load_json,
    noise_label,
    noise_results_dir,
    read_trial_metrics,
    resolve_result_root,
)

#: 显著性水平与星号标记（需求 3）
ALPHA = 0.05
STAR_LEVELS: Tuple[Tuple[float, str], ...] = ((0.001, "***"), (0.01, "**"), (0.05, "*"))

#: 箱线图输出尺寸/dpi：与 compare_results.py 完全一致，保证论文里所有图同尺寸
FIG_SIZE: Tuple[float, float] = (5.4, 5.4)
BOXPLOT_DPI = 200

#: 指标的显示名（纵轴只写指标缩写，避免中文字体依赖）
METRIC_DISPLAY_EN: Dict[str, str] = {
    "MSE": "MSE",
    "SNR": "SNR(dB)",
    "CC": "CC",
    "RRMSE": "RRMSE",
}

#: 箱线图颜色循环（与 compare_results.py 完全一致的高区分度配色）
COLOR_CYCLE = [
    "#0072B2",  # 深蓝
    "#E69F00",  # 琥珀黄
    "#009E73",  # 翠绿
    "#8B0000",  # 深红（比 highlight 的红色更暗，避免混淆）
    "#6A3D9A",  # 深紫
    "#8C564B",  # 棕
    "#E377C2",  # 粉
    "#7F7F7F",  # 中灰
    "#BCBD22",  # 橄榄黄
    "#17BECF",  # 青
]


# --------------------------------------------------------------------------------------
# 数据准备
# --------------------------------------------------------------------------------------

def discover_models(noise_dir: str, result_dir: str) -> List[str]:
    """扫描结果目录，找出在 ``<模型>/<噪声目录>/`` 下真正有实验的模型名。"""
    models: List[str] = []
    if not os.path.isdir(result_dir):
        return models
    noise_name = os.path.basename(os.path.normpath(noise_dir))
    for name in sorted(os.listdir(result_dir)):
        model_dir = os.path.join(result_dir, name)
        if os.path.isdir(os.path.join(model_dir, noise_name)):
            models.append(name)
    return models


def discover_metrics(noise_dir: str, models: Sequence[str]) -> List[str]:
    """从汇总文件或实验摘要里推断需要检验的指标列表。"""
    aggregate_path = os.path.join(noise_dir, "指标汇总_均值标准差中位数IQR.json")
    if os.path.isfile(aggregate_path):
        try:
            aggregate = load_json(aggregate_path)
            metrics = list(((aggregate.get("各划分") or {}).get("test") or {}).keys())
            if metrics:
                return metrics
        except Exception:
            pass
    for model in models:
        from utils import find_result_layouts

        for layout in find_result_layouts(os.path.join(os.path.dirname(noise_dir))) or []:
            summary_path = layout.summary_path
            if os.path.isfile(summary_path):
                try:
                    metrics = list(load_json(summary_path).get("评价指标") or [])
                    if metrics:
                        return metrics
                except Exception:
                    pass
    return ["MSE", "SNR", "CC", "RRMSE"]


def collect_trial_values(
    result_dir: str,
    noise_type: str,
    models: Sequence[str],
    metric: str,
    split: str = "test",
) -> Dict[str, np.ndarray]:
    """收集每个模型在 K 次独立实验里的该指标原始值（需求 3 的“数据输入”）。

    返回 ``{模型名: ndarray(K)}``，缺失或非有限值会被剔除。
    """
    values: Dict[str, np.ndarray] = {}
    for model in models:
        layout = ResultLayout(result_dir, model, noise_type, 1)
        collected: List[float] = []
        if os.path.isdir(layout.noise_dir):
            for name in sorted(os.listdir(layout.noise_dir)):
                if not name.startswith("试验次数_"):
                    continue
                trial_dir = os.path.join(layout.noise_dir, name)
                value = read_trial_metrics(trial_dir, metric, split)
                if value is None:
                    continue
                value = float(value)
                if np.isfinite(value):
                    collected.append(value)
        values[model] = np.asarray(collected, dtype=np.float64)
    return values


# --------------------------------------------------------------------------------------
# 统计检验
# --------------------------------------------------------------------------------------

def _import_scipy_stats() -> Any:
    """延迟导入 scipy.stats（便于给出更友好的错误信息）。"""
    from scipy import stats

    return stats


def _tukey_hsd(groups: Sequence[np.ndarray], labels: Sequence[str]) -> Dict[str, Any]:
    """Tukey HSD 事后检验。

    优先使用 ``scipy.stats.tukey_hsd``（SciPy >= 1.8），否则退回 ``statsmodels``。
    返回 ``{"方法": str, "两两比较": [ {模型A, 模型B, 均值差, p值, 显著, 星号} ]}``。
    """
    stats = _import_scipy_stats()
    result: Dict[str, Any] = {"方法": None, "两两比较": []}

    means = {label: float(np.mean(group)) for label, group in zip(labels, groups)}
    pair_result: Optional[Any] = None
    method = None

    if hasattr(stats, "tukey_hsd"):
        try:
            pair_result = stats.tukey_hsd(*[np.asarray(g, dtype=np.float64) for g in groups])
            method = "scipy.stats.tukey_hsd"
        except Exception:
            pair_result = None

    if pair_result is None:
        try:  # 退回 statsmodels
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            flat = np.concatenate([np.asarray(g, dtype=np.float64) for g in groups])
            labels_flat = np.concatenate(
                [np.full(len(g), label, dtype=object) for g, label in zip(groups, labels)]
            )
            outcome = pairwise_tukeyhsd(flat, labels_flat, alpha=ALPHA)
            for row in outcome.summary().data[1:]:
                group_a, group_b, meandiff, p_value = str(row[0]), str(row[1]), float(row[2]), float(row[3])
                result["两两比较"].append(
                    {
                        "模型A": group_a,
                        "模型B": group_b,
                        "均值差(A-B)": meandiff,
                        "p值": p_value,
                        "显著(p<0.05)": bool(p_value < ALPHA),
                        "星号": significance_stars(p_value),
                    }
                )
            result["方法"] = "statsmodels.pairwise_tukeyhsd"
            return result
        except Exception:
            return result

    result["方法"] = method
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            try:
                p_value = float(pair_result.pvalue[i, j])
            except Exception:
                continue
            label_a, label_b = labels[i], labels[j]
            result["两两比较"].append(
                {
                    "模型A": label_a,
                    "模型B": label_b,
                    "均值差(A-B)": float(means[label_a] - means[label_b]),
                    "p值": p_value,
                    "显著(p<0.05)": bool(p_value < ALPHA),
                    "星号": significance_stars(p_value),
                }
            )
    return result


def significance_stars(p_value: float) -> str:
    """把 p 值映射成 ``*`` / ``**`` / ``***`` / ``ns``。"""
    if p_value is None or not np.isfinite(p_value):
        return "ns"
    for threshold, stars in STAR_LEVELS:
        if p_value < threshold:
            return stars
    return "ns"


def anova_one_metric(
    model_values: Dict[str, np.ndarray], metric: str, split: str = "test", alpha: float = ALPHA
) -> Dict[str, Any]:
    """对单个指标做单因素方差分析 + Tukey HSD。"""
    labels = [model for model, values in model_values.items() if values.size >= 2]
    groups = [np.asarray(model_values[model], dtype=np.float64) for model in labels]

    record: Dict[str, Any] = {
        "指标": metric,
        "数据划分": split,
        "因子": "模型",
        "因变量": f"{split} 集上的 {metric}（每个模型 K 次独立实验的原始值）",
        "各组样本数": {model: int(values.size) for model, values in model_values.items()},
        "各组均值": {model: (float(np.mean(values)) if values.size else None)
                     for model, values in model_values.items()},
        "各组标准差": {model: (float(np.std(values, ddof=1)) if values.size > 1 else None)
                       for model, values in model_values.items()},
    }

    if len(labels) < 2:
        record.update({"可检验": False, "说明": "有效模型少于 2 个（每个模型至少需要 2 次实验），无法做方差分析",
                       "ANOVA": None, "TukeyHSD": None})
        return record

    if any(values.size < 2 for values in groups):
        record.update({"可检验": False, "说明": "存在有效实验次数少于 2 的模型，无法做方差分析",
                       "ANOVA": None, "TukeyHSD": None})
        return record

    if all(np.allclose(values, values[0]) for values in groups) and len(
        {round(float(values[0]), 12) for values in groups}
    ) == 1:
        record.update({"可检验": False, "说明": "所有组的数值完全相同（组内方差为 0），方差分析无定义",
                       "ANOVA": None, "TukeyHSD": None})
        return record

    stats = _import_scipy_stats()
    try:
        f_stat, p_value = stats.f_oneway(*groups)
    except Exception as error:
        record.update({"可检验": False, "说明": f"方差分析失败：{error}", "ANOVA": None, "TukeyHSD": None})
        return record

    # 效应量 eta^2（组间平方和 / 总平方和）
    all_values = np.concatenate(groups)
    grand_mean = float(np.mean(all_values))
    ss_between = float(sum(group.size * (float(np.mean(group)) - grand_mean) ** 2 for group in groups))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    eta_squared = (ss_between / ss_total) if ss_total > 0 else float("nan")

    anova = {
        "F统计量": float(f_stat),
        "p值": float(p_value),
        "自由度(组间)": int(len(groups) - 1),
        "自由度(组内)": int(sum(group.size for group in groups) - len(groups)),
        "eta方(效应量)": float(eta_squared),
        "显著(p<0.05)": bool(p_value < alpha),
        "结论": (
            f"p = {p_value:.4g} < {alpha}：不同模型在 {metric} 上存在统计学显著差异"
            if p_value < alpha else
            f"p = {p_value:.4g} >= {alpha}：不同模型在 {metric} 上没有显著差异"
        ),
    }
    record.update(
        {
            "可检验": True,
            "ANOVA": anova,
            "TukeyHSD": _tukey_hsd(groups, labels) if p_value < alpha else None,
            "TukeyHSD跳过原因": None if p_value < alpha else "ANOVA 不显著，按常规不做事后检验",
        }
    )
    return record


def analyze(
    result_dir: str,
    noise_type: str,
    models: Sequence[str],
    metrics: Sequence[str],
    split: str = "test",
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """对全部指标做方差分析，返回可直接序列化的结果字典。"""
    from utils import timestamp

    noise_type = noise_type.upper()
    analysis: Dict[str, Any] = {
        "噪声类型": noise_type,
        "数据划分": split,
        "显著性水平": alpha,
        "因子": "模型",
        "模型": list(models),
        "生成时间": timestamp(),
        "检验方法": "单因素方差分析 One-way ANOVA；事后检验 Tukey HSD（HSD 校正）",
        "标记说明": {"***": "p < 0.001", "**": "p < 0.01", "*": "p < 0.05", "ns": "不显著"},
        "各指标": {},
    }
    for metric in metrics:
        model_values = collect_trial_values(result_dir, noise_type, models, metric, split)
        analysis["各指标"][metric] = anova_one_metric(model_values, metric, split, alpha)
    return analysis


# --------------------------------------------------------------------------------------
# 保存结果
# --------------------------------------------------------------------------------------

def _format_number(value: Any, digits: int = 6) -> str:
    """把数值格式化成适合放进 CSV/JSON 的字符串。"""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(number)
    return f"{number:.{digits}g}"


def save_analysis_json(analysis: Dict[str, Any], path: str) -> str:
    """保存完整分析结果到 JSON。"""
    from utils import save_json

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    save_json(path, analysis)
    return path


def save_analysis_csv(analysis: Dict[str, Any], path: str) -> str:
    """把 ANOVA 与两两比较整理成 CSV（UTF-8-BOM，Excel 可直接打开）。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["噪声类型", "数据划分", analysis["噪声类型"], analysis["数据划分"]])
        writer.writerow(["检验方法", analysis["检验方法"]])
        writer.writerow([])
        writer.writerow([
            "指标", "检验项", "模型A", "模型B", "均值A", "均值B", "均值差(A-B)",
            "F统计量", "p值", "显著(p<0.05)", "星号", "结论",
        ])
        for metric, record in analysis["各指标"].items():
            anova = record.get("ANOVA") or {}
            means = record.get("各组均值", {}) or {}
            writer.writerow([
                metric, "ANOVA", "", "", "", "", "",
                _format_number(anova.get("F统计量")), _format_number(anova.get("p值")),
                anova.get("显著(p<0.05)", ""), significance_stars(anova.get("p值", float("nan"))),
                anova.get("结论", record.get("说明", "")),
            ])
            tukey = (record.get("TukeyHSD") or {}) if record.get("TukeyHSD") else {}
            for pair in tukey.get("两两比较", []) or []:
                writer.writerow([
                    metric, "TukeyHSD", pair["模型A"], pair["模型B"],
                    _format_number(means.get(pair["模型A"])), _format_number(means.get(pair["模型B"])),
                    _format_number(pair["均值差(A-B)"]), "",
                    _format_number(pair["p值"]), pair["显著(p<0.05)"], pair["星号"],
                    f"{pair['模型A']} vs {pair['模型B']}: p={_format_number(pair['p值'], 4)}",
                ])
    return path


# --------------------------------------------------------------------------------------
# 箱线图（带 Tukey HSD 显著性标注）
# --------------------------------------------------------------------------------------

def _bracket_levels(
    pairs: Sequence[Dict[str, Any]],
    x_positions: Dict[str, float],
    box_highs: Dict[str, float],
    lower: float,
    upper: float,
    text_height: float,
    gap: float,
) -> List[Dict[str, Any]]:
    """为每一对显著差异计算横线位置，保证互不重叠、也不压住箱体/星号。

    算法：按“横跨宽度”从小到大处理每一对；先把横线放在
    ``max(两端组的上缘)`` 之上，若与**已放好的横线或它们的星号文字**在水平方向有重叠，
    就继续往上抬，直到找到不冲突的高度。

    :param lower/upper: 数据（含离群点）的上下界，用来换算内部高度
    :param text_height: 星号文字大约占多少数据高度（由字号换算）
    :param gap: 横线与箱体、文字之间的最小间距
    """
    ordered = sorted(
        pairs,
        key=lambda pair: abs(x_positions[pair["模型A"]] - x_positions[pair["模型B"]]),
    )
    line_height = max(text_height * 0.35, gap)  # 横线自身占的高度（含一点余量）
    span = max(upper - lower, 1e-12)
    floor = upper + gap * 0.6  # 第一层横线的最低高度：刚在最高箱体上方

    placed: List[Dict[str, Any]] = []

    def overlaps(x1: float, x2: float, item: Dict[str, Any]) -> bool:
        """水平方向是否与已放好的横线/星号重叠。"""
        return not (x2 < item["x1"] - 1e-9 or x1 > item["x2"] + 1e-9)

    for pair in ordered:
        x1 = min(x_positions[pair["模型A"]], x_positions[pair["模型B"]])
        x2 = max(x_positions[pair["模型A"]], x_positions[pair["模型B"]])
        base = max(float(box_highs.get(pair["模型A"], 0.0)),
                   float(box_highs.get(pair["模型B"], 0.0)))
        y = max(base + gap, floor)

        # 逐次上抬，直到：既不与已放好的横线重叠、也不与它们的星号文字重叠
        for _ in range(200):
            if not placed:
                break
            conflict = False
            for item in placed:
                if not overlaps(x1, x2, item):
                    continue
                # ① 与本横线自身（线 + 竖钩）区间冲突
                if y < item["y"] + item["h"] + gap and item["y"] < y + line_height + gap:
                    conflict = True
                    break
                # ② 与本横线的星号文字区间冲突（中心 ± 半个星号宽度）
                if y < item["text_top"] + gap and item["text_bottom"] < y + text_height + gap:
                    conflict = True
                    break
            if not conflict:
                break
            y += text_height * 0.6

        placed.append(
            {
                "模型A": pair["模型A"],
                "模型B": pair["模型B"],
                "x1": x1,
                "x2": x2,
                "p值": pair["p值"],
                "星号": pair["星号"],
                "level": int(round((y - floor) / max(text_height * 0.6, 1e-12))),
                "y": float(y),
                "h": float(max(span * 0.018, gap * 0.5)),
                "text_bottom": float(y + max(span * 0.018, gap * 0.5)),
                "text_top": float(y + max(span * 0.018, gap * 0.5) + text_height),
            }
        )
    return placed


def boxplot_statistics(labels: Sequence[str], data: Sequence[np.ndarray],
                       metric: str, split: str) -> List[Dict[str, Any]]:
    """计算箱线图的统计量（n / 中位数 / 均值 / 标准差 / Q1 / Q3 / IQR / 极值）。"""
    rows: List[Dict[str, Any]] = []
    for model, values in zip(labels, data):
        values = np.asarray(values, dtype=np.float64)
        q1, q3 = np.percentile(values, [25, 75])
        rows.append(
            {
                "指标": metric,
                "数据划分": split,
                "模型": model,
                "n": int(values.size),
                "中位数": float(np.median(values)),
                "均值": float(np.mean(values)),
                "标准差": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "Q1": float(q1),
                "Q3": float(q3),
                "四分位距IQR": float(q3 - q1),
                "最小值": float(np.min(values)),
                "最大值": float(np.max(values)),
            }
        )
    return rows


def save_reading_note(analysis: Dict[str, Any], tag: str, path: str,
                      stats_rows: Sequence[Dict[str, Any]]) -> str:
    """写一份“怎么看这些图”的说明文件（图上 ``*`` 的含义 + ANOVA/Tukey 结论 + 统计量）。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    split_label = {"train": "训练集", "val": "验证集", "test": "测试集"}.get(
        str(analysis.get("数据划分")), str(analysis.get("数据划分")))
    lines = [
        f"# {tag} 显著性检验说明",
        "",
        f"- 生成时间：{analysis.get('生成时间')}",
        f"- 噪声类型：{analysis.get('噪声类型')}",
        f"- 数据划分：{split_label}（用 K 次独立重复实验在该划分上的 **K 个原始值**做检验）",
        "- 因子（自变量）：模型；因变量：某个评价指标",
        f"- 检验方法：{analysis.get('检验方法')}",
        f"- 显著性水平：{analysis.get('显著性水平')}",
        "",
        "## 图上的标记怎么读",
        "",
        "| 标记 | 含义 |",
        "|---|---|",
        "| `*` | p < 0.05（差异显著） |",
        "| `**` | p < 0.01（差异很显著） |",
        "| `***` | p < 0.001（差异极显著） |",
        "| 无横线 | 该模型对之间差异不显著（p ≥ 0.05） |",
        "",
        "横线两端连接的即被比较的两个模型；箱体 = 四分位距 IQR，红线 = 中位数，",
        "须 = 非异常值范围，散点 = K 次实验的原始数值。",
        "图上不写标题、不标注 ANOVA 的 F/p，正文里按下表引用即可。",
        "",
        "## 方差分析结果",
        "",
        "| 指标 | F | p | 是否显著 | 显著模型对（Tukey HSD） |",
        "|---|---|---|---|---|",
    ]
    for metric, record in analysis.get("各指标", {}).items():
        anova = record.get("ANOVA") or {}
        pairs = [pair for pair in ((record.get("TukeyHSD") or {}).get("两两比较") or [])
                 if pair.get("显著(p<0.05)")]
        pair_text = "；".join(
            f"{pair['模型A']} vs {pair['模型B']} ({pair['星号']}, p={pair['p值']:.3g})"
            for pair in pairs
        ) or ("—" if anova.get("显著(p<0.05)") else "不显著，按常规不做事后检验")
        f_stat, p_value = anova.get("F统计量"), anova.get("p值")
        if isinstance(f_stat, (int, float)) and isinstance(p_value, (int, float)):
            lines.append(f"| {metric} | {f_stat:.3f} | {p_value:.3g} | "
                         f"{'是' if anova.get('显著(p<0.05)') else '否'} | {pair_text} |")
        else:
            lines.append(f"| {metric} | — | — | {record.get('说明', '不可检验')} | — |")
    lines += [
        "",
        "## 各模型统计量（由 K 次独立实验计算）",
        "",
        "| 指标 | 模型 | n | 中位数 | 均值 | 标准差 | Q1 | Q3 | IQR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in stats_rows:
        lines.append(
            f"| {row['指标']} | {row['模型']} | {row['n']} | {row['中位数']:.4g} | "
            f"{row['均值']:.4g} | {row['标准差']:.4g} | {row['Q1']:.4g} | {row['Q3']:.4g} | "
            f"{row['四分位距IQR']:.4g} |"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def save_statistics_csv(rows: Sequence[Dict[str, Any]], path: str) -> str:
    """把箱线图统计量写成 CSV（UTF-8-BOM，Excel 可直接打开）。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    header = ["指标", "数据划分", "模型", "n", "中位数", "均值", "标准差",
              "Q1", "Q3", "四分位距IQR", "最小值", "最大值"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row["指标"], row["数据划分"], row["模型"], row["n"],
                f"{row['中位数']:.10g}", f"{row['均值']:.10g}", f"{row['标准差']:.10g}",
                f"{row['Q1']:.10g}", f"{row['Q3']:.10g}", f"{row['四分位距IQR']:.10g}",
                f"{row['最小值']:.10g}", f"{row['最大值']:.10g}",
            ])
    return path


def plot_boxplot_with_significance(
    analysis: Dict[str, Any],
    metric: str,
    result_dir: str,
    noise_type: str,
    models: Sequence[str],
    split: str = "test",
    output_dir: Optional[str] = None,
    lang: str = "en",
    tag: Optional[str] = None,
    highlight: Optional[str] = None,
) -> Optional[str]:
    """画某个指标的箱线图（方形、无标题），并按 Tukey HSD 结果标注显著性。

    图上只保留坐标轴、箱体/散点与 ``*``/``**``/``***`` 横线；
    具体统计量（n/中位数/均值/标准差/IQR）由 :func:`save_statistics_csv` 输出成 CSV。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plotting import configure_fonts

    lang = configure_fonts(lang)
    tag = tag or noise_label(noise_type).replace("HZ", "Hz")

    model_values = collect_trial_values(result_dir, noise_type, models, metric, split)
    labels = [model for model in models
              if model_values.get(model) is not None and model_values[model].size]
    if not labels:
        return None

    data = [model_values[model] for model in labels]
    positions = np.arange(1, len(labels) + 1, dtype=np.float64)
    # 统一方形画布（与 compare_results.py 一致，5.4x5.4 英寸 @200dpi = 1080x1080）
    span = max(1.0, float(len(labels)) + 0.2)          # 横轴跨度
    unit = FIG_SIZE[0] / 5.6                            # 每个单位宽度占多少英寸
    fig, ax = plt.subplots(figsize=(max(FIG_SIZE[0], span * unit), FIG_SIZE[1]))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.66,
        patch_artist=True,
        showmeans=False,
        tick_labels=labels,
        # 中位线要细：箱体本身较扁时，粗中位线会把整个箱体占满
        medianprops={"color": "#FF7F0E", "linewidth": 1.1},
        boxprops={"linewidth": 1.1, "edgecolor": "#222222"},
        whiskerprops={"linewidth": 1.0, "linestyle": "-"},
        capprops={"linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "#888888",
                    "markeredgecolor": "#555555", "alpha": 0.85},
    )
    for index, patch in enumerate(box["boxes"]):
        if highlight and labels[index] == highlight:
            patch.set_facecolor("#d62728")
            patch.set_alpha(0.85)
        else:
            patch.set_facecolor(COLOR_CYCLE[index % len(COLOR_CYCLE)])
            patch.set_alpha(0.75)

    # ---- 坐标范围：用“箱线图自身算出来的上下须”作为数据范围，让箱体尽量高 ----
    core_values: List[float] = []
    for values in data:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        low = float(np.min(values[values >= q1 - 1.5 * iqr])) if np.any(values >= q1 - 1.5 * iqr) \
            else float(np.min(values))
        high = float(np.max(values[values <= q3 + 1.5 * iqr])) if np.any(values <= q3 + 1.5 * iqr) \
            else float(np.max(values))
        core_values += [low, high, float(np.min(values)), float(np.max(values))]
    lower, upper = min(core_values), max(core_values)
    pad = max((upper - lower) * 0.06, abs(upper) * 0.015, 1e-9)
    span = max(upper - lower, abs(upper) * 0.05, 1e-12)

    # ---- 显著性横线：按 Tukey HSD 结果逐对标注 ----
    tukey = (analysis["各指标"].get(metric, {}) or {}).get("TukeyHSD") or {}
    significant_pairs = [pair for pair in (tukey.get("两两比较") or []) if pair.get("显著(p<0.05)")]
    x_positions = {model: float(position) for model, position in zip(labels, positions)}
    box_highs = {model: float(np.max(values)) for model, values in zip(labels, data)}

    # 把“字号”换算成“数据坐标高度”，这样留白与行距在任何量纲下都不会撞车
    star_fontsize = 13.0
    axes_height_in = FIG_SIZE[1] * 0.86  # 粗略的绘图区高度（英寸）
    span_for_text = max(span, 1e-12)
    text_height = span_for_text * (star_fontsize / 72.0) / max(axes_height_in, 1e-6) * 2.0
    gap = text_height * 0.35

    brackets = _bracket_levels(significant_pairs, x_positions, box_highs,
                               lower, upper, text_height, gap)
    highest = upper
    for bracket in brackets:
        x1, x2, y, h = bracket["x1"], bracket["x2"], bracket["y"], bracket["h"]
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1.2, color="#222222")
        ax.text((x1 + x2) / 2.0, y + h, bracket["星号"], ha="center", va="bottom",
                fontsize=star_fontsize, fontweight="bold", color="#B22222",
                clip_on=False)
        highest = max(highest, bracket["text_top"])

    # 顶部留白 = 星号文字高度 + 一点余量；保证最后一层 *** 不会贴边/被裁掉
    top_margin = text_height * 0.8 if brackets else pad
    ax.set_ylim(lower - pad * 1.1, highest + top_margin)
    ax.set_xlim(0.45, len(labels) + 0.55)
    ax.set_ylabel(METRIC_DISPLAY_EN.get(metric, metric), fontsize=11)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")
    ax.tick_params(direction="in", length=4, width=0.9)
    label_size = 10.0 if len(labels) <= 4 else max(7.0, 42.0 / max(len(name) for name in labels))
    ax.set_xticklabels(labels, fontsize=label_size, rotation=12, ha="right")
    ax.tick_params(axis="y", labelsize=10)

    # 图上不写标题、不写 ANOVA 结论、不画统计表、不加图例
    # （* / ** / *** 的含义写在 <标签>_显著性检验说明.md 里）

    output_dir = ensure_dir(output_dir or os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(output_dir, f"{tag}_显著性检验_{metric}_箱线图.png")
    fig.set_size_inches(*FIG_SIZE)
    # 顶部留出足够空间给星号（不依赖 tight_layout，避免把星号裁掉）
    fig.subplots_adjust(left=0.13, right=0.975, top=0.94, bottom=0.115)
    fig.savefig(path, dpi=BOXPLOT_DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------------------

def run_significance_analysis(
    result_dir: str,
    noise_type: str,
    models: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    split: str = "test",
    alpha: float = ALPHA,
    lang: str = "en",
    stats_dir: Optional[str] = None,
    plots_dir: Optional[str] = None,
    output_prefix: Optional[str] = None,
    highlight: Optional[str] = None,
) -> List[str]:
    """完整执行“ANOVA + Tukey HSD + 箱线图 + 统计表”，返回全部产物路径。

    - 箱线图、统计表 CSV、ANOVA/Tukey 结果统一放在
      ``results/对比图/<噪声>去噪_<采样率>Hz/`` 下（ANOVA/Tukey 的 JSON/CSV/摘要放
      ``统计结果/`` 子目录），与 ``compare_results.py`` 的对比图同目录、同尺寸；
    - 每张箱线图都不带标题，只保留坐标轴与显著性横线。
    """
    from utils import timestamp

    result_dir = resolve_result_root(result_dir)
    noise_type = str(noise_type).upper()
    label = noise_label(noise_type)
    tag = noise_label(noise_type).replace("HZ", "Hz")

    # 用“第一个找到的模型目录”定位噪声目录
    if models:
        noise_dir = noise_results_dir(result_dir, list(models)[0], noise_type)
    else:
        candidates = [
            noise_results_dir(result_dir, name, noise_type)
            for name in sorted(os.listdir(result_dir)) if os.path.isdir(os.path.join(result_dir, name))
        ] if os.path.isdir(result_dir) else []
        noise_dir = next((path for path in candidates if os.path.isdir(path)), None)
        if noise_dir is None:
            raise FileNotFoundError(
                f"在 {result_dir} 下找不到任何 <模型>/<{label}> 结果目录，无法做显著性检验"
            )

    model_list = list(models) if models else discover_models(noise_dir, result_dir)
    if len(model_list) < 2:
        raise ValueError(
            f"显著性检验至少需要 2 个模型，当前在 {noise_dir} 下只发现 {model_list}"
        )
    metric_list = list(metrics) if metrics else discover_metrics(noise_dir, model_list)
    if not metric_list:
        raise ValueError("没有可检验的指标")

    analysis = analyze(result_dir, noise_type, model_list, metric_list, split, alpha)
    prefix = output_prefix or noise_type
    # 图像与统计结果统一放在 results/对比图/<噪声>去噪_<采样率>Hz/ 下
    comparison_root = os.path.join(result_dir, "对比图", tag)
    plots_dir = ensure_dir(plots_dir or comparison_root)
    stats_dir = ensure_dir(stats_dir or os.path.join(comparison_root, "统计结果"))

    artifacts = [
        save_analysis_json(analysis, os.path.join(stats_dir, f"{prefix}_显著性检验.json")),
        save_analysis_csv(analysis, os.path.join(stats_dir, f"{prefix}_显著性检验.csv")),
    ]

    # 箱线图 + 统计表 CSV（统计表取代原来画在图下方的表格）
    stat_rows: List[Dict[str, Any]] = []
    for metric in metric_list:
        model_values = collect_trial_values(result_dir, noise_type, model_list, metric, split)
        labels = [m for m in model_list if model_values.get(m) is not None and model_values[m].size]
        stat_rows += boxplot_statistics(labels, [model_values[m] for m in labels], metric, split)
        path = plot_boxplot_with_significance(
            analysis, metric, result_dir, noise_type, model_list,
            split=split, output_dir=plots_dir, lang=lang, tag=tag, highlight=highlight,
        )
        if path:
            artifacts.append(path)
    if stat_rows:
        artifacts.append(save_statistics_csv(
            stat_rows, os.path.join(plots_dir, f"{tag}_显著性检验_统计表.csv")
        ))
        artifacts.append(save_reading_note(
            analysis, tag, os.path.join(plots_dir, f"{tag}_显著性检验说明.md"), stat_rows
        ))

    # 把结论摘要也写进汇总里，方便一眼看到“谁和谁有显著差异”
    summary = {
        "生成时间": timestamp(),
        "文件": [os.path.basename(path) for path in artifacts],
        "各指标ANOVA": {
            metric: {
                "F": (record.get("ANOVA") or {}).get("F统计量"),
                "p": (record.get("ANOVA") or {}).get("p值"),
                "显著": (record.get("ANOVA") or {}).get("显著(p<0.05)"),
                "显著模型对": [
                    f"{pair['模型A']} vs {pair['模型B']} ({pair['星号']}, p={pair['p值']:.3g})"
                    for pair in ((record.get("TukeyHSD") or {}).get("两两比较") or [])
                    if pair.get("显著(p<0.05)")
                ],
            }
            for metric, record in analysis["各指标"].items()
        },
    }
    from utils import save_json

    save_json(os.path.join(stats_dir, f"{prefix}_显著性检验摘要.json"), summary)
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    """构建本脚本的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对 K 次独立重复实验的结果做单因素方差分析 + Tukey HSD，并画带显著性标注的箱线图"
    )
    parser.add_argument("--result-dir", default=None,
                        help="结果根目录，默认 <项目根>/results")
    parser.add_argument("--noise-type", default="EMG", choices=["EMG", "EOG", "emg", "eog"],
                        help="噪声类型（EMG=512Hz，EOG=256Hz）")
    parser.add_argument("--models", nargs="+", default=None,
                        help="参与比较的模型名（默认自动发现结果目录下的全部模型）")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help='要检验的指标（默认取汇总文件里的全部指标，也可写 全部）')
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="用哪个数据划分的 K 个结果做检验，默认 test（测试集）")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="显著性水平，默认 0.05")
    parser.add_argument("--highlight", default=None,
                        help="重点模型名（红色突出显示，一般用来标出自己提出的模型）")
    parser.add_argument("--lang", default="en", choices=["en", "zh"], help="图像语言")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result_dir = resolve_result_root(args.result_dir)
    metrics = args.metrics
    if metrics and any(str(m).lower() in ("全部", "all", "*") for m in metrics):
        metrics = None  # 交给自动发现

    try:
        artifacts = run_significance_analysis(
            result_dir=result_dir,
            noise_type=args.noise_type.upper(),
            models=args.models,
            metrics=metrics,
            split=args.split,
            alpha=float(args.alpha),
            lang=args.lang,
            highlight=args.highlight,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"[错误] {error}", file=sys.stderr)
        return 2

    print("显著性检验完成，产物：")
    for path in artifacts:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
