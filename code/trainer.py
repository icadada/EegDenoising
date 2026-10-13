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
from metrics import compute_single_metric, average_by_snr, resolve_metric_selection
from network import create_model
from utils import (
    loss_filename,
    save_json,
    experiment_summary_filename,
    experiment_config_filename,
    checkpoint_filename,
    weight_filename,
    SPLIT_LABELS,
    metric_filename,
    ensure_dir,
    train_history_filename,
)

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
    ) -> None:
        self.config = config
        self.data = data
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.device = device

        self.prefix: str = config.file_prefix
        self.metric_names: List[str] = resolve_metric_selection(config.metrics)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn = build_loss(config.loss)

        # 训练历史：每个 epoch 记录一项，便于画曲线
        self.history: Dict[str, Any] = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "耗时秒": [],
            "lr": [],
            "val_metrics": {name: [] for name in self.metric_names},
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_done = 0

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------
    @property
    def weights_path(self) -> str:
        return os.path.join(self.experiment_dir, weight_filename(self.prefix, self.config.network))

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.experiment_dir, checkpoint_filename(self.prefix))

    @property
    def history_path(self) -> str:
        return os.path.join(self.experiment_dir, train_history_filename(self.prefix))

    def _metric_path(self, split: str, metric: str, per_sample: bool = False) -> str:
        name = metric_filename(split, self.data.noise_type, metric)
        if per_sample:
            name = name.replace("_指标.npy", "_逐样本指标.npy")
        return os.path.join(self.experiment_dir, name)

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

    def validate_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """在验证集上计算 loss 与所选指标（整体值），返回 ``(验证loss, {指标: 值})``。"""
        val_loss, per_sample, _ = self.inference_stats(
            self.data.val_noisy,
            self.data.val_clean,
            metric_names=self.metric_names,
        )
        overall_metrics: Dict[str, float] = {}
        for name in self.metric_names:
            with np.errstate(invalid="ignore"):
                overall_metrics[name] = float(np.nanmean(per_sample[name])) if per_sample[name].size else float("nan")
        return val_loss, overall_metrics

    # ------------------------------------------------------------------
    # 主训练流程
    # ------------------------------------------------------------------
    def train(self, start_epoch: int = 1) -> None:
        """从 ``start_epoch`` 训练到 ``config.epochs``，并保存最优权重与断点。"""
        assert self.model is not None
        self.logger(f"开始训练：轮次 {start_epoch} -> {self.config.epochs}")
        for epoch in range(start_epoch, int(self.config.epochs) + 1):
            train_loss, elapsed = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate_one_epoch()

            # 记录历史
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["耗时秒"].append(float(elapsed))
            self.history["lr"].append(float(self.optimizer.param_groups[0]["lr"]))
            for name in self.metric_names:
                self.history["val_metrics"][name].append(float(val_metrics[name]))
            self.epochs_done = epoch

            metric_text = "，".join(f"{name}={val_metrics[name]:.6g}" for name in self.metric_names)
            self.logger(
                f"Epoch {epoch}/{self.config.epochs} 完成，耗时 {elapsed:.1f}s，"
                f"train_loss={train_loss:.6g}，val_loss={val_loss:.6g}"
                + (f"，{metric_text}" if metric_text else "")
            )

            # 挑选并保存最优模型（best_start_ratio 用于兼容原版“只在前 80% 之后才挑最优”的规则）
            is_eligible = epoch > float(self.config.epochs) * float(self.config.best_start_ratio)
            if is_eligible and val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self.best_epoch = epoch
                self.save_weights(self.weights_path)
                self.logger(f"  ↳ 新的最优模型：val_loss={val_loss:.8g}（第 {epoch} 轮）")

            # 保存断点（默认每个 epoch 都保存，可用 --checkpoint-every 降低频率）
            every_n_epochs = max(1, int(getattr(self.config, "checkpoint_every", 1)))
            if epoch % every_n_epochs == 0 or epoch == int(self.config.epochs):
                self.save_checkpoint(epoch)

        # 若一次都没被选中（例如 epochs 很小），用最后一轮兜底
        if self.best_epoch == 0:
            if self.history["val_loss"]:
                self.best_val_loss = float(self.history["val_loss"][-1])
            self.best_epoch = self.epochs_done
            self.save_weights(self.weights_path)
            self.logger(f"训练轮数较少，已用最后一轮权重作为最优模型（val_loss={self.best_val_loss:.8g}）")

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
                "val_loss": float(self.best_val_loss),
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
        if content.get("history"):
            self.history = content["history"]
            # 兼容：历史里可能缺少新选择的指标
            for name in self.metric_names:
                self.history.setdefault("val_metrics", {}).setdefault(name, [])
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
            f"历史最优 val_loss={self.best_val_loss:.8g}（第 {self.best_epoch} 轮）"
        )
        return self.epochs_done + 1

    # ------------------------------------------------------------------
    # 结束后：保存 loss / 指标 / 摘要
    # ------------------------------------------------------------------
    def save_loss(self) -> str:
        """保存训练过程的 loss（需求 5：``<前缀>_Loss.npy``）。"""
        assert self.model is not None
        path = os.path.join(self.experiment_dir, loss_filename(self.prefix))
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
        np.save(self.history_path, self.history)
        return path

    def evaluate_and_save_metrics(self, weights: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """对训练/验证/测试集计算全部所选指标并保存 npy（需求 3）。

        :return: ``{划分: {指标: 整体平均值}}``
        """
        assert self.model is not None
        if weights:
            self.load_weights(weights)

        summary: Dict[str, Dict[str, float]] = {}
        snr_levels_path = os.path.join(self.experiment_dir, "SNR档位.npy")
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
        return summary

    def save_summary(self, summary: Dict[str, Dict[str, float]], elapsed_seconds: float) -> str:
        """保存实验摘要 JSON（最优轮次、各划分整体指标、环境信息等）。"""
        path = os.path.join(self.experiment_dir, experiment_summary_filename())
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
                "最优轮次": int(self.best_epoch),
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
        save_config(self.config, os.path.join(self.experiment_dir, experiment_config_filename()))

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
            "各划分整体指标": summary,
        }
        self.logger(f"训练结束，结果保存在：{self.experiment_dir}")
        return result
