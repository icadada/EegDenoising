"""数据准备模块（与原版 EEGdenoiseNet V2 完全一致的加噪逻辑，但做了向量化）。

流程
----
1. 读取干净 EEG 与噪声（EMG/EOG）epoch；
2. 先把两个数据集整体随机打乱，再按 ``train_per`` 划分 训练 / 验证 / 测试；
3. EMG 比 EEG 多时复制 EEG、EOG 比 EEG 少时复制噪声，保证两者一一配对；
4. 训练集：EEG 与噪声各自随机重排 ``combin_num`` 次扩大样本量，**每个样本随机取 SNR ∈ [-7, 2] dB**；
5. 验证集/测试集：使用 **10 个固定 SNR 档位** ``linspace(-7, 2, 10)``，每档枚举全部样本；
6. 每个样本按 ``snr`` 缩放噪声后相加，再用**该样本加噪信号的标准差**统一归一化
   （加噪信号与干净信号用同一个 std，保证两者幅度关系不变）。

关于 SNR 的定义细节
-------------------
原版代码使用 ``snr_linear = 10 ** (0.1 * snr_db)``（功率比）直接作为**幅度**缩放的分母，
因此实际达到的功率信噪比是标称值的 2 倍（标称 -7 dB 实际约 -14 dB）。本模块**刻意保持与原版
一致**，以保证与 EEGdenoiseNet 的基准结果可比。绘图时会在横轴标注“标称 SNR”。

为节省内存，本模块统一使用 ``float32`` 存储信号，逐样本 RMS/std 用 ``float64`` 分块计算。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils import load_json, save_json, ensure_dir

#: 训练/验证/测试共用的 SNR 档位（dB，标称值）
SNR_LEVELS = np.linspace(-7.0, 2.0, num=10)
#: 训练集随机 SNR 的取值范围（dB）
TRAIN_SNR_RANGE = (-7.0, 2.0)


# --------------------------------------------------------------------------------------
# 一、基础工具
# --------------------------------------------------------------------------------------

def get_rms(records: np.ndarray, axis: int = -1) -> np.ndarray:
    """求均方根（RMS）。"""
    records = np.asarray(records, dtype=np.float64)
    return np.sqrt(np.mean(np.square(records), axis=axis))


def _per_sample_rms(signal: np.ndarray, block_size: int = 4096) -> np.ndarray:
    """分块用 float64 计算每个样本的 RMS，避免一次性生成巨大的 float64 临时数组。"""
    signal = np.asarray(signal)
    result = np.empty(signal.shape[0], dtype=np.float64)
    for start in range(0, signal.shape[0], block_size):
        stop = min(start + block_size, signal.shape[0])
        result[start:stop] = get_rms(signal[start:stop].astype(np.float64), axis=1)
    return result


def _per_sample_std(signal: np.ndarray, block_size: int = 4096) -> np.ndarray:
    """分块计算每个样本沿时间轴的标准差。"""
    signal = np.asarray(signal)
    result = np.empty(signal.shape[0], dtype=np.float64)
    for start in range(0, signal.shape[0], block_size):
        stop = min(start + block_size, signal.shape[0])
        result[start:stop] = np.std(signal[start:stop].astype(np.float64), axis=1)
    return result


def random_signal(signal: np.ndarray, combin_num: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """把 epoch 顺序随机置换 ``combin_num`` 次。

    :return: 形状 ``(combin_num, n_epochs, n_points)`` 的数组
    """
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal 必须是 (N, T) 二维数组，收到 {signal.shape}")
    if combin_num < 1:
        raise ValueError("combin_num 必须 >= 1")
    rng = rng or np.random.default_rng()

    result = np.empty((combin_num,) + signal.shape, dtype=signal.dtype)
    for i in range(combin_num):
        result[i] = signal[rng.permutation(signal.shape[0])]
    return result


def add_noise_and_normalize(
    eeg: np.ndarray,
    noise: np.ndarray,
    snr_linear: np.ndarray,
    block_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按目标 SNR 把噪声加到 EEG 上，并用加噪信号的 std 归一化。

    :param eeg: 干净 EEG，``[N, T]``
    :param noise: 噪声，``[N, T]``
    :param snr_linear: 每个样本的线性 SNR，``[N]``（= ``10**(0.1*snr_db)``）
    :return: ``(加噪并归一化后的信号, 归一化后的干净信号, 每个样本使用的std)``
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    snr_linear = np.asarray(snr_linear, dtype=np.float64).reshape(-1)

    eeg_rms = _per_sample_rms(eeg)
    noise_rms = _per_sample_rms(noise)

    # 缩放系数：与原版 coe = rms(eeg) / (rms(noise) * SNR) 完全一致
    coe = np.zeros_like(eeg_rms)
    valid = noise_rms > 0
    coe[valid] = eeg_rms[valid] / (noise_rms[valid] * snr_linear[valid])

    noisy = eeg + noise * coe.astype(np.float32)[:, None]
    std = _per_sample_std(noisy)
    std = np.where(std == 0.0, 1.0, std)  # 防止全零样本导致除零

    noisy_normalized = (noisy / std.astype(np.float32)[:, None]).astype(np.float32)
    clean_normalized = (eeg / std.astype(np.float32)[:, None]).astype(np.float32)
    return noisy_normalized, clean_normalized, std


# --------------------------------------------------------------------------------------
# 二、数据划分
# --------------------------------------------------------------------------------------

@dataclass
class DataSplit:
    """保存一次实验中划分好的全部数据（含每个样本的 SNR，用于按 SNR 分档评估）。"""

    train_noisy: np.ndarray
    train_clean: np.ndarray
    train_snr: np.ndarray
    val_noisy: np.ndarray
    val_clean: np.ndarray
    val_snr: np.ndarray
    test_noisy: np.ndarray
    test_clean: np.ndarray
    test_snr: np.ndarray
    test_norm_std: np.ndarray
    noise_type: str = "EMG"
    sample_rate: int = 512
    signal_length: int = 1024
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def get(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按 ``train``/``val``/``test`` 取出 ``(加噪, 干净, SNR)``。"""
        mapping = {
            "train": (self.train_noisy, self.train_clean, self.train_snr),
            "val": (self.val_noisy, self.val_clean, self.val_snr),
            "test": (self.test_noisy, self.test_clean, self.test_snr),
        }
        if split not in mapping:
            raise ValueError(f"未知划分 {split}，可选 {list(mapping)}")
        return mapping[split]

    def describe(self) -> str:
        """返回各集合形状的可读描述。"""
        return (
            f"训练集 {self.train_noisy.shape} | "
            f"验证集 {self.val_noisy.shape} | "
            f"测试集 {self.test_noisy.shape} | "
            f"采样率 {self.sample_rate}Hz | 信号长度 {self.signal_length}"
        )


def prepare_data(
    EEG_all: np.ndarray,
    noise_all: np.ndarray,
    combin_num: int = 10,
    train_per: float = 0.8,
    noise_type: str = "EMG",
    seed: Optional[int] = None,
    max_train_samples: Optional[int] = None,
) -> DataSplit:
    """生成训练/验证/测试数据（与原版逻辑一致，但向量化实现）。

    :param max_train_samples: 只保留前若干个训练样本，用于快速冒烟测试（``None`` 表示不限制）
    """
    from utils import NOISE_SAMPLE_RATE

    eeg = np.asarray(EEG_all)
    noise = np.asarray(noise_all)
    if eeg.ndim != 2 or noise.ndim != 2:
        raise ValueError("EEG_all 与 noise_all 都必须是 (N, T) 二维数组")
    if eeg.shape[1] != noise.shape[1]:
        raise ValueError(f"EEG 与噪声的时间长度不一致：{eeg.shape} vs {noise.shape}")
    if not (0 < train_per < 1):
        raise ValueError("train_per 必须在 (0, 1) 之间")

    rng = np.random.default_rng(seed)

    # ① 先整体随机打乱，避免原始排列顺序影响划分
    eeg_random = eeg[rng.permutation(eeg.shape[0])]
    noise_random = noise[rng.permutation(noise.shape[0])]

    # ② 数量对齐（与原版一致：EMG 多则复制 EEG，EOG 少则复制噪声）
    if noise_type.upper() == "EMG" and noise_random.shape[0] > eeg_random.shape[0]:
        repeat_count = noise_random.shape[0] - eeg_random.shape[0]
        eeg_random = np.vstack([eeg_random[:repeat_count], eeg_random])
    if noise_type.upper() == "EOG" and noise_random.shape[0] < eeg_random.shape[0]:
        repeat_count = eeg_random.shape[0] - noise_random.shape[0]
        noise_random = np.vstack([noise_random[:repeat_count], noise_random])
    if eeg_random.shape[0] != noise_random.shape[0]:
        raise ValueError(
            f"EEG 与 {noise_type.upper()} 数据量没有匹配成功："
            f"{eeg_random.shape[0]} vs {noise_random.shape[0]}"
        )

    n_points = noise_random.shape[1]
    n_train = round(train_per * eeg_random.shape[0])
    n_val = round((eeg_random.shape[0] - n_train) / 2)

    train_eeg = eeg_random[:n_train]
    val_eeg = eeg_random[n_train:n_train + n_val]
    test_eeg = eeg_random[n_train + n_val:]

    train_noise = noise_random[:n_train]
    val_noise = noise_random[n_train:n_train + n_val]
    test_noise = noise_random[n_train + n_val:]

    # ③ 训练集：combin_num 次随机重配 + 每样本随机 SNR
    eeg_train = random_signal(train_eeg, combin_num, rng).reshape(combin_num * train_eeg.shape[0], n_points)
    noise_train = random_signal(train_noise, combin_num, rng).reshape(combin_num * train_noise.shape[0], n_points)
    if max_train_samples is not None:
        eeg_train = eeg_train[:max_train_samples]
        noise_train = noise_train[:max_train_samples]

    snr_train_db = rng.uniform(TRAIN_SNR_RANGE[0], TRAIN_SNR_RANGE[1], eeg_train.shape[0])
    train_noisy, train_clean, _ = add_noise_and_normalize(eeg_train, noise_train, 10 ** (0.1 * snr_train_db))

    # ④ 验证集：10 个固定 SNR 档位，外层 SNR、内层样本（与原版循环顺序一致）
    snr_val_db = np.repeat(SNR_LEVELS, val_eeg.shape[0])
    val_noisy, val_clean, _ = add_noise_and_normalize(
        np.tile(val_eeg, (SNR_LEVELS.size, 1)),
        np.tile(val_noise, (SNR_LEVELS.size, 1)),
        10 ** (0.1 * snr_val_db),
    )

    # ⑤ 测试集：同上，另外保存归一化用的 std（需要还原物理量纲时使用）
    snr_test_db = np.repeat(SNR_LEVELS, test_eeg.shape[0])
    test_noisy, test_clean, test_std = add_noise_and_normalize(
        np.tile(test_eeg, (SNR_LEVELS.size, 1)),
        np.tile(test_noise, (SNR_LEVELS.size, 1)),
        10 ** (0.1 * snr_test_db),
    )

    return DataSplit(
        train_noisy=train_noisy,
        train_clean=train_clean,
        train_snr=snr_train_db.astype(np.float64),
        val_noisy=val_noisy,
        val_clean=val_clean,
        val_snr=snr_val_db.astype(np.float64),
        test_noisy=test_noisy,
        test_clean=test_clean,
        test_snr=snr_test_db.astype(np.float64),
        test_norm_std=test_std,
        noise_type=noise_type.upper(),
        sample_rate=NOISE_SAMPLE_RATE[noise_type.upper()],
        signal_length=int(n_points),
        extra_info={
            "combin_num": int(combin_num),
            "train_per": float(train_per),
            "原始EEG数量": int(eeg.shape[0]),
            "原始噪声数量": int(noise.shape[0]),
        },
    )


# --------------------------------------------------------------------------------------
# 三、划分数据的保存与加载（需求 10：恢复训练时保证数据完全一致）
# --------------------------------------------------------------------------------------

#: 每个集合需要落盘的内容：中文名后缀 -> 数据划分的属性名
_SAVED_FIELDS = {
    "训练集": ("train_noisy", "train_clean", "train_snr"),
    "验证集": ("val_noisy", "val_clean", "val_snr"),
    "测试集": ("test_noisy", "test_clean", "test_snr"),
}


def save_split(directory: str, split: DataSplit) -> Dict[str, str]:
    """把划分好的数据保存到目录（npy 二进制），返回 ``{描述: 文件路径}``。"""
    ensure_dir(directory)
    artifacts: Dict[str, str] = {}
    for label, (noisy_attr, clean_attr, snr_attr) in _SAVED_FIELDS.items():
        for suffix, attr in (("加噪", noisy_attr), ("干净", clean_attr), ("SNR", snr_attr)):
            path = os.path.join(directory, f"{label}_{suffix}.npy")
            np.save(path, getattr(split, attr))
            artifacts[f"{label}_{suffix}"] = path
    path = os.path.join(directory, "测试集_归一化STD.npy")
    np.save(path, split.test_norm_std)
    artifacts["测试集_归一化STD"] = path

    # 元信息（形状、采样率等）
    save_json(
        os.path.join(directory, "划分信息.json"),
        {
            "噪声类型": split.noise_type,
            "采样率": split.sample_rate,
            "信号长度": split.signal_length,
            "训练集形状": list(split.train_noisy.shape),
            "验证集形状": list(split.val_noisy.shape),
            "测试集形状": list(split.test_noisy.shape),
            "SNR档位": SNR_LEVELS.tolist(),
            **split.extra_info,
        },
    )
    return artifacts


def load_split(directory: str) -> DataSplit:
    """从目录加载之前保存的划分数据（恢复训练时使用）。"""
    info_path = os.path.join(directory, "划分信息.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"找不到划分信息文件：{info_path}")
    info = load_json(info_path)

    data: Dict[str, np.ndarray] = {}
    for label in _SAVED_FIELDS:
        for suffix in ("加噪", "干净", "SNR"):
            path = os.path.join(directory, f"{label}_{suffix}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"缺少划分数据文件：{path}")
            data[f"{label}_{suffix}"] = np.load(path)

    std_path = os.path.join(directory, "测试集_归一化STD.npy")
    data["测试集_归一化STD"] = np.load(std_path) if os.path.isfile(std_path) else np.array([])

    return DataSplit(
        train_noisy=data["训练集_加噪"], train_clean=data["训练集_干净"], train_snr=data["训练集_SNR"],
        val_noisy=data["验证集_加噪"], val_clean=data["验证集_干净"], val_snr=data["验证集_SNR"],
        test_noisy=data["测试集_加噪"], test_clean=data["测试集_干净"], test_snr=data["测试集_SNR"],
        test_norm_std=data["测试集_归一化STD"],
        noise_type=info.get("噪声类型", "EMG"),
        sample_rate=int(info.get("采样率", 512)),
        signal_length=int(info.get("信号长度", 1024)),
        extra_info={k: v for k, v in info.items()
                    if k not in ("噪声类型", "采样率", "信号长度", "SNR档位")},
    )
