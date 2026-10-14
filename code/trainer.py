"""训练模块：训练 / 验证 / 评估 / 断点保存 / 恢复训练。

本模块负责：

1. 按配置创建模型、优化器、损失函数；
2. 逐 epoch 训练，并在验证集上计算验证 loss 与所选指标；
3. 保存最优模型权重（``<前缀>_<模型名>.pt``）与断点文件（``<前缀>_断点续训.pt``）；
4. 训练结束后对 **训练集/验证集/测试集** 分别计算指标，保存为 npy；
5. 支持从断点恢复训练（含优化器状态、epoch、历史、随机数状态）。

**模型统一接口**：输入 ``[B, L]``，输出 ``[B, L]``。若某个模型输出形状不一致，
推理统计方法会直接抛出带说明的错误，提醒模型作者修正。
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from config import save_config, runtime_env_info
from data_prepare import SNR_LEVELS, DataSplit
from metrics import (
    METRIC_REGISTRY,
    average_by_snr,
    compute_single_metric,
    resolve_metric_selection,
)
from network import create_model
from utils import (
    ResultLayout,
    SPLIT_LABELS,
    ensure_dir,
    save_json,
)


def metric_higher_is_better(metric_name: str) -> bool:
    """返回某个指标是否“越大越好”（读 ``metrics`` 包里各指标声明的方向）。

    ``loss`` 一律是越小越好。
    """
    if str(metric_name).lower() == "loss":
        return False
    from metrics import _discover_all_metrics

    _discover_all_metrics()
    return bool(METRIC_REGISTRY.get(metric_name, {}).get("是否越大越好", False))


def metric_direction_text(metric_name: str) -> str:
    """返回“越大越好 / 越小越好”的中文描述。"""
    return "越大越好" if metric_higher_is_better(metric_name) else "越小越好"


def is_better(score: float, best: float, metric_name: str) -> bool:
    """判断当前得分是否优于历史最优（按指标方向）。nan 永远不优于任何值。"""
    if score is None:
        return False
    score = float(score)
    if not math.isfinite(score):
        return False
    if not math.isfinite(best):
        return True
    return score > best if metric_higher_is_better(metric_name) else score < best

# --------------------------------------------------------------------------------------
# 优化器与损失函数
# --------------------------------------------------------------------------------------

def build_optimizer(name: str, params: Any, lr: float) -> torch.optim.Optimizer:
    """按名称构建优化器（超参与原版 EEGdenoiseNet 保持一致）。

    - ``rmsprop``：``alpha=0.9``
    - ``adam``：``betas=(0.5, 0.9), eps=1e-8``
    - ``sgd``：``momentum=0.9, nesterov=False``
    """
    name = name.lower()
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9), eps=1e-8)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=False)
    raise ValueError(f"未知优化器 {name}，可选 rmsprop/adam/sgd")


def build_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """按名称构建损失函数（需要可反向传播，因此用 torch 实现）。"""
    name = name.lower()
    if name == "mse":
        return lambda prediction, target: torch.mean((prediction - target) ** 2)
    if name == "rmse":
        return lambda prediction, target: torch.sqrt(torch.mean((prediction - target) ** 2) + 1e-12)
    raise ValueError(f"未知损失函数 {name}，可选 mse/rmse")


# --------------------------------------------------------------------------------------
# 训练器
# --------------------------------------------------------------------------------------

class DenoiseTrainer:
    """一次实验的训练器：负责从零训练或从断点恢复训练。"""

    def __init__(
        self,
        config: Any,
        data: DataSplit,
        experiment_dir: str,
        logger: Callable[..., None],
        device: str,
        layout: Optional[ResultLayout] = None,
    ) -> None:
        self.config = config
        self.data = data
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.device = device
        # 结果目录布局：权重/指标/图像/训练配置/划分数据 分目录保存（需求 2）
        self.layout: ResultLayout = layout or ResultLayout(
            result_dir=getattr(config, "result_dir", os.path.dirname(experiment_dir)),
            model_name=getattr(config, "network", "model"),
            noise_type=data.noise_type,
            trial=int(getattr(config, "trial", 1) or 1),
        )

        self.prefix: str = config.file_prefix
        self.metric_names: List[str] = resolve_metric_selection(config.metrics)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn = build_loss(config.loss)

        # ---- 最优模型挑选准则（需求 3）----
        self.select_metric: str = str(getattr(config, "select_metric", "MSE") or "MSE")
        self.select_split: str = str(getattr(config, "select_split", "val") or "val")
        if self.select_metric.lower() == "loss":
            self.select_metric = "loss"
        elif self.select_metric not in self.metric_names:
            # 命令行挑选指标与 --metrics 不一致时，自动补进评估指标，保证每轮都能算出来
            self.metric_names.append(self.select_metric)
        self.select_higher_is_better = metric_higher_is_better(self.select_metric)
        self.best_score: float = -float("inf") if self.select_higher_is_better else float("inf")
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_done = 0

        # 训练历史：每个 epoch 记录一项，便于画曲线
        self.history: Dict[str, Any] = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "耗时秒": [],
            "lr": [],
            "val_metrics": {name: [] for name in self.metric_names},
            "各划分指标": {split: {name: [] for name in self.metric_names}
                        for split in ("train", "val", "test")},
            "挑选指标值": [],
            "挑选指标": self.select_metric,
            "挑选划分": self.select_split,
            "挑选指标方向": "越大越好" if self.select_higher_is_better else "越小越好",
        }

    # ------------------------------------------------------------------
    # 路径（需求 2：分目录保存）
    # ------------------------------------------------------------------
    def _ensure_dirs(self) -> None:
        """按需创建结果目录（兼容旧结构：layout 与 experiment_dir 不一致时不建新目录）。"""
        self.layout.ensure_dirs()
        ensure_dir(self.experiment_dir)

    @property
    def weights_path(self) -> str:
        return self.layout.weights_path

    @property
    def checkpoint_path(self) -> str:
        return self.layout.checkpoint_path

    @property
    def history_path(self) -> str:
        return self.layout.history_path

    @property
    def split_dir(self) -> str:
        return self.layout.split_dir

    @property
    def metric_dir(self) -> str:
        return self.layout.metrics_dir

    def _metric_path(self, split: str, metric: str, per_sample: bool = False) -> str:
        return self.layout.metric_path(split, metric, per_sample=per_sample)

    # ------------------------------------------------------------------
    # 构建模型 / 优化器
    # ------------------------------------------------------------------
    def build(self) -> None:
        """创建模型、优化器，并把模型结构写入日志。"""
        self.model = create_model(
            self.config.network, self.data.signal_length, **self.config.model_args
        ).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger(
            f"模型 {self.config.network} 创建完成：总参数 {param_count:,}，"
            f"可训练参数 {trainable_param_count:,}，模型初始化参数 {self.config.model_args}"
        )
        self.optimizer = build_optimizer(
            self.config.optimizer, self.model.parameters(), float(self.config.lr)
        )
        self.logger(
            f"优化器 {self.config.optimizer}，学习率 {self.config.lr}，"
            f"损失函数 {self.config.loss}，评估指标 {self.metric_names}"
        )

    # ------------------------------------------------------------------
    # 推理与统计
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference_stats(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
        metric_names: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        need_output: bool = False,
    ) -> Tuple[float, Dict[str, np.ndarray], Optional[np.ndarray]]:
        """分块推理，同时统计 MSE 与所选指标的**逐样本**数值。

        :return: ``(平均MSE, {指标名: 逐样本数组}, 去噪输出或None)``
        """
        assert self.model is not None, "请先调用 build()"
        batch_size = int(batch_size or self.config.eval_batch_size)
        metric_names = list(metric_names or [])
        N = noisy.shape[0]

        self.model.eval()
        sum_squared_error = 0.0
        numel = 0
        per_sample: Dict[str, List[np.ndarray]] = {name: [] for name in metric_names}
        output_list: List[np.ndarray] = []

        for start in range(0, N, batch_size):
            stop = min(start + batch_size, N)
            xb = torch.as_tensor(noisy[start:stop], dtype=torch.float32, device=self.device)
            yb = torch.as_tensor(clean[start:stop], dtype=torch.float32, device=self.device)
            prediction = self.model(xb)

            # 统一接口校验：输出必须是 [B, L]
            if prediction.shape != yb.shape:
                raise RuntimeError(
                    f"模型 {self.config.network} 的输出形状 {tuple(prediction.shape)} 与目标形状 "
                    f"{tuple(yb.shape)} 不一致。本框架要求所有模型满足 输入[B,L] -> 输出[B,L]，"
                    "请在模型 forward 中把输出还原为 [Batch_size, len]。"
                )

            sum_squared_error += float(torch.sum((prediction - yb) ** 2).detach().cpu())
            numel += yb.numel()

            if metric_names:
                clean_np = yb.detach().cpu().numpy().astype(np.float64)
                pred_np = prediction.detach().cpu().numpy().astype(np.float64)
                for name in metric_names:
                    per_sample[name].append(compute_single_metric(name, clean_np, pred_np))
            if need_output:
                output_list.append(prediction.detach().cpu().numpy().astype(np.float32))

        mean_mse = sum_squared_error / max(numel, 1)
        per_sample_merged = {name: np.concatenate(values) if values else np.array([])
                    for name, values in per_sample.items()}
        outputs = np.concatenate(output_list, axis=0) if output_list else None
        return mean_mse, per_sample_merged, outputs

    # ------------------------------------------------------------------
    # 单个 epoch
    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个 epoch，返回 ``(平均训练loss, 该轮耗时秒)``。"""
        assert self.model is not None and self.optimizer is not None
        start_time = time.time()
        self.model.train()

        N = self.data.train_noisy.shape[0]
        batch_size = int(self.config.batch_size)
        num_batches = math.ceil(N / batch_size)

        # 是否需要打乱（打乱顺序由 torch 随机数控制，其状态会被写入断点）
        order = torch.randperm(N) if int(self.config.shuffle) == 1 else torch.arange(N)

        sum_squared_error = 0.0
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)
        for batch_index in progress_bar:
            index = order[batch_index * batch_size:(batch_index + 1) * batch_size]
            xb = torch.as_tensor(self.data.train_noisy[index.numpy()], dtype=torch.float32, device=self.device)
            yb = torch.as_tensor(self.data.train_clean[index.numpy()], dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(xb)
            if prediction.shape != yb.shape:
                raise RuntimeError(
                    f"模型输出形状 {tuple(prediction.shape)} 与目标 {tuple(yb.shape)} 不一致，"
                    "要求所有模型满足 输入[B,L] -> 输出[B,L]。"
                )
            loss = self.loss_fn(prediction, yb)
            loss.backward()
            self.optimizer.step()

            sum_squared_error += float(loss.detach().cpu()) * xb.shape[0]
            progress_bar.set_postfix({"损失": f"{float(loss.detach().cpu()):.6g}"})

        avg_train_loss = sum_squared_error / max(N, 1)
        return avg_train_loss, time.time() - start_time

    def evaluate_split(
        self, split: str, metric_names: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """在指定划分上计算 loss 与所选指标（整体值），返回 ``(loss, {指标: 值})``。

        ``loss`` 恒为 MSE（与训练损失同一尺度），用于日志与兼容旧结果。
        """
        names = list(metric_names or [])
        noisy, clean, _ = self.data.get(split)
        if self.config.max_eval_samples:  # 冒烟测试时可限制样本量
            limit = int(self.config.max_eval_samples)
            noisy, clean = noisy[:limit], clean[:limit]
        loss, per_sample, _ = self.inference_stats(noisy, clean, metric_names=names)
        overall_metrics: Dict[str, float] = {}
        for name in names:
            with np.errstate(invalid="ignore"):
                overall_metrics[name] = (
                    float(np.nanmean(per_sample[name])) if per_sample[name].size else float("nan")
                )
        return loss, overall_metrics

    def validate_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """在验证集上计算 loss 与所选指标（保留旧接口）。"""
        return self.evaluate_split("val", self.metric_names)

    # ------------------------------------------------------------------
    # 主训练流程
    # ------------------------------------------------------------------
    def train(self, start_epoch: int = 1) -> None:
        """从 ``start_epoch`` 训练到 ``config.epochs``，并保存最优权重与断点。

        最优模型的挑选规则（需求 3）由 ``--select-split`` 与 ``--select-metric`` 决定：
        每个 epoch 都会在训练/验证/测试集上计算所选指标，然后按
        ``<划分>_<指标>``（默认 ``val_MSE``）的最优值保存权重。
        """
        assert self.model is not None
        self._ensure_dirs()
        self.logger(
            f"开始训练：轮次 {start_epoch} -> {self.config.epochs}；"
            f"最优模型挑选准则 = {self.select_split}_{self.select_metric}"
            f"（{metric_direction_text(self.select_metric)}）"
        )
        if self.select_split == "train":
            self.logger("[提示] 当前按训练集指标挑选最优模型，容易过拟合，通常不推荐。")
        elif self.select_split == "test":
            self.logger("[提示] 当前按测试集指标挑选最优模型，测试集信息会泄漏进模型选择，写论文时务必说明。")

        for epoch in range(start_epoch, int(self.config.epochs) + 1):
            train_loss, elapsed = self.train_one_epoch(epoch)

            # ① 先算“挑选准则”所在的划分（默认验证集），拿到 loss 与指标
            criterion_loss, criterion_metrics = self.evaluate_split(
                self.select_split, self.metric_names
            )
            val_loss = (
                criterion_loss if self.select_split == "val"
                else self.evaluate_split("val", [])[0]
            )
            # ② 其余划分也各算一次所选指标，便于事后比较不同挑选准则的差异
            split_metrics: Dict[str, Dict[str, float]] = {}
            for split in ("train", "val", "test"):
                if split == self.select_split:
                    split_metrics[split] = dict(criterion_metrics)
                else:
                    split_metrics[split] = self.evaluate_split(split, self.metric_names)[1]

            criterion_value = (
                float(criterion_loss) if self.select_metric == "loss"
                else float(criterion_metrics.get(self.select_metric, float("nan")))
            )

            # 记录历史
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["耗时秒"].append(float(elapsed))
            self.history["lr"].append(float(self.optimizer.param_groups[0]["lr"]))
            self.history["挑选指标值"].append(float(criterion_value))
            for name in self.metric_names:
                self.history["val_metrics"][name].append(
                    float(split_metrics["val"].get(name, float("nan")))
                )
                for split in ("train", "val", "test"):
                    self.history["各划分指标"][split][name].append(
                        float(split_metrics[split].get(name, float("nan")))
                    )
            self.epochs_done = epoch

            metric_text = "，".join(
                f"{name}={split_metrics['val'][name]:.6g}" for name in self.metric_names
            )
            self.logger(
                f"Epoch {epoch}/{self.config.epochs} 完成，耗时 {elapsed:.1f}s，"
                f"train_loss={train_loss:.6g}，val_loss={val_loss:.6g}，"
                f"挑选值({self.select_split}_{self.select_metric})={criterion_value:.6g}"
                + (f"，验证集指标：{metric_text}" if metric_text else "")
            )

            # 挑选并保存最优模型（best_start_ratio 用于兼容原版“只在前 80% 之后才挑最优”的规则）
            is_eligible = epoch > float(self.config.epochs) * float(self.config.best_start_ratio)
            if is_eligible and is_better(criterion_value, self.best_score, self.select_metric):
                self.best_score = float(criterion_value)
                if math.isfinite(criterion_loss):
                    self.best_val_loss = float(criterion_loss)
                self.best_epoch = epoch
                self.save_weights(self.weights_path)
                self.logger(
                    f"  ↳ 新的最优模型：{self.select_split}_{self.select_metric}="
                    f"{criterion_value:.8g}（第 {epoch} 轮）"
                )

            # 保存断点（默认每个 epoch 都保存，可用 --checkpoint-every 降低频率）
            every_n_epochs = max(1, int(getattr(self.config, "checkpoint_every", 1)))
            if epoch % every_n_epochs == 0 or epoch == int(self.config.epochs):
                self.save_checkpoint(epoch)

        # 若一次都没被选中（例如 epochs 很小或指标全是 nan），用最后一轮兜底
        if self.best_epoch == 0:
            self.best_epoch = self.epochs_done
            if self.history["挑选指标值"]:
                self.best_score = float(self.history["挑选指标值"][-1])
            if self.history["val_loss"]:
                self.best_val_loss = float(self.history["val_loss"][-1])
            self.save_weights(self.weights_path)
            self.logger(
                f"没有任何一轮满足挑选条件，已用最后一轮权重作为最优模型"
                f"（{self.select_split}_{self.select_metric}={self.best_score:.8g}）"
            )
        self.logger(
            f"最优模型：第 {self.best_epoch} 轮，{self.select_split}_{self.select_metric}="
            f"{self.best_score:.8g}，val_loss={self.best_val_loss:.8g}"
        )

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------
    def save_weights(self, path: str) -> None:
        """保存最优模型权重，文件名形如 ``测试集_EMG去噪_512HZ_Simple_CNN.pt``。"""
        assert self.model is not None
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "network": self.config.network,
                "模型初始化参数": dict(self.config.model_args),
                "datanum": int(self.data.signal_length),
                "noise_type": self.data.noise_type,
                "epoch": int(self.best_epoch),
                # 需求 3：记录最优模型是按哪个数据集、哪个指标挑选出来的
                "挑选划分": self.select_split,
                "挑选指标": self.select_metric,
                "挑选指标方向": metric_direction_text(self.select_metric),
                "挑选指标值": float(self.best_score),
                "val_loss": float(self.best_val_loss),
                "随机种子": getattr(self.config, "seed", None),
            },
            path,
        )

    def load_weights(self, path: str, strict: bool = True) -> None:
        """把权重加载进当前模型。"""
        assert self.model is not None
        content = torch.load(path, map_location=self.device, weights_only=False)
        state = content.get("state_dict", content) if isinstance(content, dict) else content
        self.model.load_state_dict(state, strict=strict)
        self.logger(f"已加载权重：{path}")

    def save_checkpoint(self, epoch: int) -> None:
        """保存断点文件（模型 + 优化器 + epoch + 历史 + 随机数状态），用于恢复训练。"""
        assert self.model is not None and self.optimizer is not None
        ensure_dir(os.path.dirname(os.path.abspath(self.checkpoint_path)))
        torch.save(
            {
                "epoch": int(epoch),
                "network": self.config.network,
                "模型初始化参数": dict(self.config.model_args),
                "datanum": int(self.data.signal_length),
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": float(self.best_val_loss),
                "best_epoch": int(self.best_epoch),
                "best_score": float(self.best_score),
                "挑选划分": self.select_split,
                "挑选指标": self.select_metric,
                "history": self.history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_random_state": np.random.get_state(),
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self, path: Optional[str] = None) -> int:
        """加载断点，恢复模型、优化器、历史与随机数状态，返回“下一次该训练的轮次”。"""
        assert self.model is not None and self.optimizer is not None
        path = path or self.checkpoint_path
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到断点文件：{path}\n"
                "提示：恢复训练需要该实验文件夹里存在 <前缀>_断点续训.pt"
            )
        content = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(content["state_dict"])
        if "optimizer_state_dict" in content:
            self.optimizer.load_state_dict(content["optimizer_state_dict"])
        self.best_val_loss = float(content.get("best_val_loss", float("inf")))
        self.best_epoch = int(content.get("best_epoch", 0))
        self.epochs_done = int(content.get("epoch", 0))
        self.best_score = float(
            content.get(
                "best_score",
                -float("inf") if self.select_higher_is_better else float("inf"),
            )
        )
        if content.get("history"):
            self.history = content["history"]
            # 兼容：历史里可能缺少新选择的指标 / 新增的字段
            for name in self.metric_names:
                self.history.setdefault("val_metrics", {}).setdefault(name, [])
            self.history.setdefault("各划分指标", {})
            for split in ("train", "val", "test"):
                self.history["各划分指标"].setdefault(split, {})
                for name in self.metric_names:
                    self.history["各划分指标"][split].setdefault(name, [])
            self.history.setdefault("挑选指标值", [])
            self.history.setdefault("挑选指标", self.select_metric)
            self.history.setdefault("挑选划分", self.select_split)
        if content.get("torch_rng_state") is not None:
            try:
                torch.set_rng_state(content["torch_rng_state"].cpu())
                if content.get("cuda_rng_state") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(content["cuda_rng_state"])
                if content.get("numpy_random_state") is not None:
                    np.random.set_state(content["numpy_random_state"])
            except Exception as exc:  # 随机状态恢复失败不影响训练
                self.logger(f"恢复随机数状态失败（可忽略）：{exc}")

        self.logger(
            f"已从断点恢复：已完成 {self.epochs_done} 轮，"
            f"历史最优 {self.select_split}_{self.select_metric}={self.best_score:.8g}"
            f"（第 {self.best_epoch} 轮），val_loss={self.best_val_loss:.8g}"
        )
        return self.epochs_done + 1

    # ------------------------------------------------------------------
    # 结束后：保存 loss / 指标 / 摘要
    # ------------------------------------------------------------------
    def save_loss(self) -> str:
        """保存训练过程的 loss（需求 5：``<前缀>_Loss.npy``，写入 ``指标/`` 子目录）。"""
        assert self.model is not None
        self._ensure_dirs()
        path = self.layout.loss_path
        np.save(
            path,
            {
                "epochs": self.history["epochs"],
                "train_loss": self.history["train_loss"],
                "val_loss": self.history["val_loss"],
                "耗时秒": self.history["耗时秒"],
                "lr": self.history["lr"],
            },
        )
        # 另外保存完整训练历史（含逐轮验证指标），便于画指标-epoch 曲线
        np.save(self.layout.history_path, self.history)
        return path

    def evaluate_and_save_metrics(self, weights: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """对训练/验证/测试集计算全部所选指标并保存 npy（需求 3）。

        :return: ``{划分: {指标: 整体平均值}}``
        """
        assert self.model is not None
        if weights:
            self.load_weights(weights)

        summary: Dict[str, Dict[str, float]] = {}
        self._ensure_dirs()
        snr_levels_path = self.layout.snr_levels_path
        np.save(snr_levels_path, SNR_LEVELS)

        for split in self.config.eval_splits:
            noisy, clean, snr = self.data.get(split)
            if self.config.max_eval_samples:  # 冒烟测试时可限制样本量
                limit = int(self.config.max_eval_samples)
                noisy, clean, snr = noisy[:limit], clean[:limit], snr[:limit]

            loss, per_sample, _ = self.inference_stats(
                noisy, clean, metric_names=self.metric_names, need_output=False
            )
            overall: Dict[str, float] = {}
            for metric in self.metric_names:
                # ① 主文件：按 SNR 分档平均后的曲线数据（10 个档位）
                by_level = average_by_snr(per_sample[metric], snr, SNR_LEVELS)
                np.save(self._metric_path(split, metric), by_level)
                # ② 附带文件：逐样本指标（便于做配对检验等二次分析）
                np.save(self._metric_path(split, metric, per_sample=True), per_sample[metric])
                with np.errstate(invalid="ignore"):
                    overall[metric] = float(np.nanmean(per_sample[metric]))

            summary[split] = {"loss": float(loss), **overall}
            self.logger(
                f"{SPLIT_LABELS[split]}评估完成：loss(MSE)={loss:.6g}，"
                + "，".join(
                    f"{name}={overall[name]:.6g}" for name in self.metric_names
                )
            )

        # 整体指标单独落盘：K 次独立实验的统计汇总与方差分析都直接读这个文件
        save_json(self.layout.overall_metrics_path, summary)
        return summary

    def save_summary(self, summary: Dict[str, Dict[str, float]], elapsed_seconds: float) -> str:
        """保存实验摘要 JSON（最优轮次、各划分整体指标、挑选准则、环境信息等）。"""
        self._ensure_dirs()
        path = self.layout.summary_path
        save_json(
            path,
            {
                "网络": self.config.network,
                "模型初始化参数": dict(self.config.model_args),
                "噪声类型": self.data.noise_type,
                "采样率": self.data.sample_rate,
                "信号长度": self.data.signal_length,
                "文件前缀": self.prefix,
                "实验目录": self.experiment_dir,
                "试验次数": int(self.layout.trial),
                "随机种子": getattr(self.config, "seed", None),
                "最优模型挑选划分": self.select_split,
                "最优模型挑选指标": self.select_metric,
                "最优模型挑选方向": metric_direction_text(self.select_metric),
                "最优轮次": int(self.best_epoch),
                "最优挑选指标值": float(self.best_score),
                "最优验证损失": float(self.best_val_loss),
                "已完成轮次": int(self.epochs_done),
                "总训练耗时秒": float(elapsed_seconds),
                "各划分整体指标": summary,
                "数据规模": {
                    "训练集": list(self.data.train_noisy.shape),
                    "验证集": list(self.data.val_noisy.shape),
                    "测试集": list(self.data.test_noisy.shape),
                },
                "SNR档位": SNR_LEVELS.tolist(),
                "评价指标": self.metric_names,
                "运行环境": runtime_env_info(),
            },
        )
        return path

    # ------------------------------------------------------------------
    def run(self, resume: bool = False, extra_weights: Optional[str] = None) -> Dict[str, Any]:
        """完整的“训练 -> 保存 -> 评估 -> 保存指标”流程。

        :param resume: True 表示从本实验文件夹中的断点继续训练（恢复模型/优化器/epoch/历史）
        :param extra_weights: 可选的权重文件路径。会在（可能的）断点恢复**之后**加载，
            因此用户显式指定的权重优先级最高（对应 ``--resume-weights``）。
        """
        start = time.time()
        if self.model is None:
            self.build()
        self._ensure_dirs()

        start_epoch = 1
        if resume:
            start_epoch = self.load_checkpoint()
            if start_epoch > int(self.config.epochs):
                self.logger(
                    f"断点已完成 {self.epochs_done} 轮，>= 目标轮数 {self.config.epochs}，"
                    "本次只做评估与保存。"
                )

        # 用户显式指定的额外权重优先级最高（在断点恢复之后加载）
        if extra_weights:
            self.load_weights(extra_weights)

        # 重新保存一次配置（恢复训练时相当于记录本次使用的参数）
        save_config(self.config, self.layout.config_path)

        self.train(start_epoch=start_epoch)
        self.save_loss()

        # 用历史最优权重做最终评估
        weights = self.weights_path if os.path.isfile(self.weights_path) else None
        summary = self.evaluate_and_save_metrics(weights=weights)
        summary_path = self.save_summary(summary, time.time() - start)

        result = {
            "实验目录": self.experiment_dir,
            "权重": self.weights_path,
            "摘要": summary_path,
            "历史最优轮次": int(self.best_epoch),
            "历史最优验证损失": float(self.best_val_loss),
            "历史最优挑选指标值": float(self.best_score),
            "挑选划分": self.select_split,
            "挑选指标": self.select_metric,
            "各划分整体指标": summary,
        }
        self.logger(f"训练结束，结果保存在：{self.experiment_dir}")
        return result
