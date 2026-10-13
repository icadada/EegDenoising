"""训练参数配置模块（typed_argument_parser 风格）。

本模块提供 :class:`TypedArgumentParser`，它是一个**带类型**的命令行参数解析器，特点：

1. 每个参数都显式声明 ``type``（int / float / str / 布尔开关），解析后即为正确类型；
2. 支持**每个模型各自的默认值**（见 :data:`MODEL_DEFAULT_HYPERPARAMS`、:data:`MODEL_DEFAULT_INIT_ARGS`）；
3. 支持把命令行参数**以 JSON 文件形式保存**，并从 JSON 文件**加载**（用于复现实验、恢复训练）；
4. 支持**透传额外参数**给模型构造函数（``--model-kwargs`` / ``--model-arg``），
   以后新增的自定义模型可以直接在命令行里传初始化参数，无需改训练代码。

参数优先级（从高到低）::

    命令行显式指定  >  --config 指定的 JSON 配置文件  >  该模型的默认值  >  全局默认值

用法示例::

    python code/main.py --network Simple_CNN --noise-type EMG --epochs 50
    python code/main.py --network MyNewNet --model-kwargs '{"width": 32, "use_bn": true}'
    python code/main.py --config code/results/xxx/训练配置.json      # 完全复现某次实验
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
from typing import Any, Dict, List, Optional

from utils import (
    save_json,
    load_json,
    get_script_root,
    get_project_root,
    default_file_prefix,
)

# --------------------------------------------------------------------------------------
# 一、默认值定义
# --------------------------------------------------------------------------------------

#: 所有模型共用的全局默认值。
#: 数值取自原版 EEGdenoiseNet V2 的 ``main.py``（epochs=50、batch_size=40、RMSprop lr=5e-5）。
GLOBAL_DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "epochs": 50,          # 训练轮数
    "batch_size": 40,      # 批大小
    "lr": 5e-5,            # 学习率
    "optimizer": "rmsprop",  # 优化器：rmsprop / adam / sgd
    "combin_num": 10,      # EEG 与噪声随机重配次数（数据增强倍数）
    "train_per": 0.8,      # 训练集占全部配对数据的比例
    "loss": "mse",         # 损失函数：mse / rmse
}

#: 每个模型的**训练超参数默认值**（需求 4：写好每个模型训练时候的默认值）。
#: 五个基准网络在原论文/原代码中都用 50 epoch、batch 40、RMSprop lr=5e-5，
#: 这里按模型分开存放，方便以后单独微调某一个模型而不影响其他模型。
MODEL_DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "fcNN":        {"epochs": 50, "batch_size": 40, "lr": 5e-5, "optimizer": "rmsprop"},
    "Simple_CNN":  {"epochs": 50, "batch_size": 40, "lr": 5e-5, "optimizer": "rmsprop"},
    "Complex_CNN": {"epochs": 50, "batch_size": 40, "lr": 5e-5, "optimizer": "rmsprop"},
    "RNN_lstm":    {"epochs": 50, "batch_size": 40, "lr": 5e-5, "optimizer": "rmsprop"},
    "Novel_CNN":   {"epochs": 50, "batch_size": 40, "lr": 5e-5, "optimizer": "rmsprop"},
}

#: 每个模型的**结构初始化参数默认值**，会在创建模型时透传给模型构造函数。
#: 这些值同时也可以通过 ``--model-kwargs`` / ``--model-arg`` 在命令行覆盖。
MODEL_DEFAULT_INIT_ARGS: Dict[str, Dict[str, Any]] = {
    "fcNN":        {"dropout": 0.3},
    "Simple_CNN":  {"channels": 64, "kernel_size": 3, "dropout": 0.3},
    "Complex_CNN": {"stem_channels": 32, "dropout": 0.0},
    "RNN_lstm":    {"hidden_size": 1, "dropout": 0.3},
    "Novel_CNN":   {"dropout": 0.5},
}

#: 允许选择的评价指标（真实可用列表以 ``metrics`` 包自动发现的结果为准）
DEFAULT_METRICS: List[str] = ["MSE", "SNR", "CC", "RRMSE"]

#: 默认参与评估的数据划分
DEFAULT_EVAL_SPLITS: List[str] = ["train", "val", "test"]


# --------------------------------------------------------------------------------------
# 二、typed_argument_parser
# --------------------------------------------------------------------------------------

class TypedArgumentParser(argparse.ArgumentParser):
    """带类型的参数解析器：参数有类型、能按模型给默认值、能整体存成文件。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)
        # 记录每个参数的元信息（类型/全局默认/每模型默认），用于按模型注入默认值
        self._meta: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def add_typed(
        self,
        *flags: str,
        type: Any = str,
        default: Any = None,
        group: Any = None,
        **kwargs: Any,
    ) -> None:
        """注册一个带类型的参数。

        额外支持三个自定义关键字：

        - ``per_model_default``：``{模型名: 默认值}``，实现“每个模型各自的默认值”；
        - ``choices``：等价于 argparse 的 ``choices``；
        - ``group``：把参数放进某个 argparse 参数组，仅影响 ``--help`` 的显示。
        """
        per_model_default = kwargs.pop("per_model_default", None)
        # 先用 argparse.SUPPRESS 作为默认值占位，真正的默认值在 parse_typed() 里按优先级注入
        target = group if group is not None else self
        target.add_argument(*flags, type=type, default=argparse.SUPPRESS, **kwargs)
        dest = self._dest(flags)
        self._meta[dest] = {"type": type, "全局默认": default, "每模型默认": per_model_default or {}}

    @staticmethod
    def _dest(flags: Any) -> str:
        """把 ``--noise-type`` 这类选项名转换成命名空间里的属性名 ``noise_type``。"""
        for f in flags:
            if f.startswith("--"):
                return f[2:].replace("-", "_")
        raise ValueError(f"参数名必须以 -- 开头：{flags}")

    # ------------------------------------------------------------------
    def parse_typed(self, argv: Optional[List[str]] = None) -> argparse.Namespace:
        """按优先级解析参数，并返回类型正确的命名空间。"""
        argv = list(sys.argv[1:] if argv is None else argv)

        # ① 预解析，先拿到 --config 与 --network，用于确定“默认值表”
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument("--config", default=None)
        pre_parser.add_argument("--network", default=None)
        pre_args, _ = pre_parser.parse_known_args(argv)

        full_config: Dict[str, Any] = {}
        if pre_args.config:
            full_config = load_json(pre_args.config)
        # 允许配置文件里存放“训练参数”子字典或平铺的键
        file_config: Dict[str, Any] = full_config.get("训练参数", full_config)
        # 配置文件中保存的模型初始化参数（恢复训练时用于还原模型结构）
        file_model_args: Dict[str, Any] = dict(full_config.get("模型初始化参数", {}) or {})

        # 模型名优先级：命令行 > 配置文件 > 解析器全局默认值
        model_name = (
            pre_args.network
            or file_config.get("network")
            or self._meta.get("network", {}).get("全局默认")
        )

        # ② 组装最终默认值：全局默认 <- 该模型默认 <- 配置文件
        defaults: Dict[str, Any] = {}
        for dest, meta in self._meta.items():
            if meta["全局默认"] is not None or dest in GLOBAL_DEFAULT_HYPERPARAMS:
                defaults[dest] = meta["全局默认"]
        defaults.update(GLOBAL_DEFAULT_HYPERPARAMS)
        if model_name and model_name in MODEL_DEFAULT_HYPERPARAMS:
            defaults.update(MODEL_DEFAULT_HYPERPARAMS[model_name])
        # 配置文件里出现的键全部生效（含 --model-kwargs 这类字符串参数）
        for key, value in file_config.items():
            if key in self._meta:
                defaults[key] = value
        self.set_defaults(**defaults)

        # ③ 正式解析命令行（命令行显式值会自动覆盖上面的默认值）
        config = self.parse_args(argv)

        # ④ 补齐没有默认值的参数，避免 AttributeError
        for dest, meta in self._meta.items():
            if not hasattr(config, dest):
                setattr(config, dest, meta["全局默认"])

        # ⑤ 合并模型初始化参数：
        #    模型文件默认 <- 配置文件保存的 <- --model-kwargs(JSON) <- --model-arg(k=v)
        config.model_args = self._merge_model_args(
            config, getattr(config, "network", None) or model_name, file_model_args
        )

        # ⑥ 补全派生字段：文件前缀、采样率等
        if not getattr(config, "file_prefix", None):
            config.file_prefix = default_file_prefix(config.noise_type)
        return config

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_key_values(key_values: Any) -> Dict[str, Any]:
        """把 ``["width=32", "use_bn=true"]`` 解析成 ``{"width": 32, "use_bn": True}``。

        ``--model-arg`` 使用 ``nargs="*" + action="append"``，因此可能是嵌套列表，这里先拍平。
        """
        import json

        flat: List[Any] = []
        for item in key_values or []:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)

        result: Dict[str, Any] = {}
        for item in flat:
            if "=" not in str(item):
                raise ValueError(f"额外模型参数需要 key=value 形式，收到：{item}")
            key, _, raw = str(item).partition("=")
            key = key.strip()
            raw = raw.strip()
            try:  # 尽量解析成数字/布尔，解析不了就当字符串
                result[key] = json.loads(raw)
            except json.JSONDecodeError:
                result[key] = raw
        return result

    def _merge_model_args(
        self,
        config: argparse.Namespace,
        model_name: Optional[str],
        file_model_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """合并模型初始化参数，优先级：模型文件默认 < 配置文件 < --model-kwargs < --model-arg。"""
        import json

        merged: Dict[str, Any] = {}
        # ① 模型文件里声明的默认初始化参数（新增模型可以自带默认值）
        try:
            from network import model_default_args

            if model_name:
                merged.update(model_default_args(model_name))
        except Exception:
            pass
        # ② config.py 里维护的兜底默认值（保证不依赖 network 包也能工作）
        if model_name and model_name in MODEL_DEFAULT_INIT_ARGS:
            for key, value in MODEL_DEFAULT_INIT_ARGS[model_name].items():
                merged.setdefault(key, value)
        # ③ 配置文件（同一个实验恢复训练时保证模型结构一致）
        merged.update(dict(file_model_args or {}))

        # ④ --model-kwargs 支持单个 JSON 字符串，也支持多次传入
        raw = getattr(config, "model_kwargs", None)
        if raw:
            texts = raw if isinstance(raw, list) else [raw]
            for text in texts:
                try:
                    merged.update(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"--model-kwargs 不是合法 JSON：{text}") from exc

        # ⑤ --model-arg key=value 优先级最高
        merged.update(self._parse_key_values(getattr(config, "model_arg", None)))
        return merged


# --------------------------------------------------------------------------------------
# 三、构建解析器
# --------------------------------------------------------------------------------------

def build_parser(available_networks: Optional[List[str]] = None, available_metrics: Optional[List[str]] = None) -> TypedArgumentParser:
    """构建本项目使用的类型化参数解析器。

    :param available_networks: 由 ``network`` 包自动发现的模型名列表（支持后续新增模型自动出现）
    :param available_metrics: 由 ``metrics`` 包自动发现的指标名列表
    """
    available_networks = available_networks or list(MODEL_DEFAULT_HYPERPARAMS)
    available_metrics = available_metrics or DEFAULT_METRICS

    parser = TypedArgumentParser(
        description="EEGdenoiseNet 去噪网络训练框架（PyTorch 版，重构实现）",
        epilog="示例：python code/main.py --network Simple_CNN --noise-type EMG --metrics MSE SNR CC RRMSE",
    )

    # ---------------- 数据与路径 ----------------
    group = parser.add_argument_group("数据与路径")
    parser.add_typed("--data-dir", type=str, default=os.path.join(get_project_root(), "data"),
              help="数据目录（存放 EEG_all_epochs_512hz.npy 等）", group=group)
    parser.add_typed("--result-dir", type=str, default=os.path.join(get_script_root(), "results"),
              help="实验结果根目录，每次实验在其中新建一个文件夹", group=group)
    parser.add_typed("--noise-type", type=str, default="EMG", choices=["EMG", "EOG"],
              help="噪声类型：EMG(512Hz) 或 EOG(256Hz)", group=group)
    parser.add_typed("--file-prefix", type=str, default=None,
              help="统一文件前缀（Loss/权重/实验文件夹名），默认 测试集_<噪声>去噪_<采样率>HZ",
              group=group)
    parser.add_typed("--trial", type=int, default=None,
              help="试验次数编号，用于区分多次重复实验；不填则自动取下一个编号", group=group)
    parser.add_typed("--config", type=str, default=None,
              help="从 JSON 配置文件加载参数（优先级低于命令行）", group=group)

    # ---------------- 网络与模型 ----------------
    group = parser.add_argument_group("网络与模型")
    parser.add_typed("--network", type=str, default="Simple_CNN", choices=available_networks,
              help="要训练的网络名（network/ 目录下新增 .py 会自动出现在这里）", group=group)
    parser.add_typed("--model-kwargs", type=str, default=None, action="append",
              help='透传给模型构造函数的额外参数，JSON 字符串，如 \'{"dropout":0.2}\'；'
                   "可重复传入，后者覆盖前者", group=group)
    parser.add_typed("--model-arg", type=str, default=None, nargs="*", action="append",
              help="透传给模型构造函数的额外参数，key=value 形式，如 --model-arg width=32 use_bn=true",
              group=group)

    # ---------------- 训练超参数 ----------------
    group = parser.add_argument_group("训练超参数（按模型有各自默认值）")
    parser.add_typed("--epochs", type=int, default=GLOBAL_DEFAULT_HYPERPARAMS["epochs"],
              per_model_default={m: v.get("epochs") for m, v in MODEL_DEFAULT_HYPERPARAMS.items()},
              help="训练轮数", group=group)
    parser.add_typed("--batch-size", type=int, default=GLOBAL_DEFAULT_HYPERPARAMS["batch_size"],
              per_model_default={m: v.get("batch_size") for m, v in MODEL_DEFAULT_HYPERPARAMS.items()},
              help="批大小", group=group)
    parser.add_typed("--lr", type=float, default=GLOBAL_DEFAULT_HYPERPARAMS["lr"],
              per_model_default={m: v.get("lr") for m, v in MODEL_DEFAULT_HYPERPARAMS.items()},
              help="学习率", group=group)
    parser.add_typed("--optimizer", type=str, default="rmsprop", choices=["rmsprop", "adam", "sgd"],
              per_model_default={m: v.get("optimizer") for m, v in MODEL_DEFAULT_HYPERPARAMS.items()},
              help="优化器：rmsprop(alpha=0.9) / adam(betas=0.5,0.9) / sgd(momentum=0.9)", group=group)
    parser.add_typed("--loss", type=str, default="mse", choices=["mse", "rmse"],
              help="损失函数", group=group)
    parser.add_typed("--seed", type=int, default=42, help="随机种子（设为 -1 表示不固定种子）", group=group)
    parser.add_typed("--device", type=str, default="auto",
              help="训练设备：auto / cpu / cuda:0", group=group)
    parser.add_typed("--shuffle", type=int, default=1, choices=[0, 1],
              help="每个 epoch 是否打乱训练集顺序（1=打乱，0=不打乱）", group=group)
    parser.add_typed("--best-start-ratio", type=float, default=0.0,
              help="从第几比例的 epoch 之后才开始挑选最优模型（原版为 0.8，默认 0 表示全程挑选）",
              group=group)
    parser.add_typed("--max-train-samples", type=int, default=None,
              help="限制训练样本数量（用于快速冒烟测试，None 表示使用全部训练数据）", group=group)
    parser.add_typed("--max-eval-samples", type=int, default=None,
              help="限制每个评估集使用的样本数量（用于快速冒烟测试）", group=group)
    parser.add_typed("--checkpoint-every", type=int, default=1,
              help="每隔多少轮保存一次断点文件（恢复训练用），1 表示每轮都保存", group=group)

    # ---------------- 数据生成 ----------------
    group = parser.add_argument_group("数据生成")
    parser.add_typed("--combin-num", type=int, default=GLOBAL_DEFAULT_HYPERPARAMS["combin_num"],
              help="EEG 与噪声随机重配次数，训练集样本数 = 配对数 x combin_num", group=group)
    parser.add_typed("--train-per", type=float, default=GLOBAL_DEFAULT_HYPERPARAMS["train_per"],
              help="训练集比例，剩余部分平均分为验证集与测试集", group=group)
    parser.add_typed("--save-splits", type=int, default=1, choices=[0, 1],
              help="是否把划分好的数据集保存到结果文件夹（恢复训练时保证数据一致），1=保存", group=group)

    # ---------------- 评估与输出 ----------------
    group = parser.add_argument_group("评估与输出")
    parser.add_typed("--metrics", type=str, default=["全部"], nargs="+",
              help=f"选择要计算并保存的评价指标，可选 {available_metrics}，或写 全部", group=group)
    parser.add_typed("--eval-splits", type=str, default=list(DEFAULT_EVAL_SPLITS), nargs="+",
              choices=["train", "val", "test"], help="对哪些数据划分计算指标", group=group)
    parser.add_typed("--eval-batch-size", type=int, default=256,
              help="评估时的批大小（分块推理以节省显存/内存）", group=group)
    parser.add_typed("--save-plot", type=int, default=1, choices=[0, 1],
              help="训练完成后是否自动绘图，1=绘制", group=group)
    parser.add_typed("--plot-lang", type=str, default="en", choices=["en", "zh"],
              help="图像文字语言：en=英文（默认，无需中文字体），zh=中文（需系统装有中文字体）",
              group=group)
    parser.add_typed("--plot-only", type=str, default=None,
              help="只对已有实验重新绘图，不训练。填结果文件夹名、完整路径或 latest", group=group)

    # ---------------- 恢复训练 ----------------
    group = parser.add_argument_group("恢复训练")
    parser.add_typed("--resume", type=str, default=None,
              help="恢复训练：填结果文件夹名（如 测试集_EMG去噪_512HZ_试验次数_1）、"
                   "完整路径，或 latest/auto（自动选择最新一次实验）", group=group)
    parser.add_typed("--resume-weights", type=str, default=None,
              help="只加载权重（不恢复优化器/epoch），可填权重文件路径或 best", group=group)
    return parser


# --------------------------------------------------------------------------------------
# 四、配置落盘 / 读取
# --------------------------------------------------------------------------------------

def config_to_dict(config: argparse.Namespace) -> Dict[str, Any]:
    """把命名空间转成可 JSON 序列化的字典（区分“训练参数”与“模型初始化参数”）。"""
    train_params: Dict[str, Any] = {}
    others: Dict[str, Any] = {}
    for key, value in vars(config).items():
        if key == "model_args":
            continue
        try:  # 过滤掉不可序列化的对象
            import json

            json.dumps(value)
            train_params[key] = value
        except TypeError:
            others[key] = str(value)
    return {"训练参数": train_params, "模型初始化参数": dict(config.model_args), "其他": others}


def save_config(config: argparse.Namespace, path: str) -> None:
    """把本次训练的命令行参数以 JSON 形式保存到文件（需求 4）。"""
    save_json(path, config_to_dict(config))


def runtime_env_info() -> Dict[str, Any]:
    """收集运行环境信息，随配置一起落盘，便于日后复现。"""
    info: Dict[str, Any] = {
        "时间": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "主机": platform.node(),
        "操作系统": platform.platform(),
        "Python": sys.version.split()[0],
        "命令行": " ".join([sys.executable] + sys.argv),
    }
    try:
        import torch

        info["PyTorch"] = torch.__version__
        info["CUDA可用"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["GPU"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - 仅作信息记录
        info["PyTorch"] = f"不可用({exc})"
    return info
