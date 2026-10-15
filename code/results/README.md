# 实验结果目录

本目录用于存放**全部训练与统计结果**，由训练脚本自动创建，位于**与 `code/`、`data/` 同级**的位置。

## 目录结构

```
results/
├── <模型名>/                                  # 例如 Simple_CNN、fcNN、RNN_lstm
│   └── <噪声类型>去噪_<采样率>HZ/               # 例如 EMG去噪_512HZ、EOG去噪_256HZ
│       ├── 指标汇总_均值标准差中位数IQR.json      # K 次独立实验各指标的 Mean/STD/Median/IQR
│       ├── 该次实验汇总.json                    # 每次实验一行（含随机种子、最优轮次、各指标）
│       ├── <模型名>_运行日志.txt                # 该模型 K 次实验的总日志
│       └── 试验次数_1/ … 试验次数_K/            # 第 i 次独立重复实验
│           ├── 权重/                            # 最优模型权重（*.pt）、断点（*_断点续训.pt）
│           ├── 指标/                            # Loss（*_Loss.npy）、各划分各指标（*.npy）
│           ├── 图像/                            # 单次实验的诊断图（7 张 *.png）
│           ├── 训练配置/                        # 训练配置.json、实验摘要.json、各划分整体指标.json
│           ├── 训练日志.txt                     # 本次实验的完整日志
│           └── 划分数据/                        # 本次实验划分好的数据集
├── Inference Result/                          # ★ 测试集逐样本推理图（用最后一次实验的权重）
│   └── <噪声类型>去噪_<采样率>Hz/               # 例如 EOG去噪_256Hz
│       ├── 推理配置.json                        # 实验次数、测试集来源、各模型权重与整体指标
│       ├── 划分数据/                            # 本次推理实际使用的测试集
│       └── 测试样本_0001/<SNR>dB/<模型名>.png     # 每个样本按 SNR 档位建子文件夹（默认 -7dB）
│                                                # --snr all 则建 -7dB/-6dB/…/+2dB 共 10 个
└── 对比图/                                     # ★ 所有“论文用图”集中在这里（尺寸统一 1080x1080）
    └── <噪声类型>去噪_<采样率>Hz/               # 例如 EMG去噪_512Hz（注意是 Hz）
        ├── <标签>_对比_CC_test.png              # 横轴 SNR(dB)、纵轴 CC；曲线 = K 次均值，阴影 = ±1 标准差（--shade none 可关）
        │                                        # 图内无标题无注记，重点模型用红色 + 图例 <模型>* 区分
        ├── <标签>_对比_MSE_test.png
        ├── <标签>_对比_SNR_test.png
        ├── <标签>_对比_RRMSE_test.png
        ├── <标签>_对比_训练Loss.png             # 每个模型只画 K 次里验证集 MSE 最低的那一次
        ├── <标签>_对比_验证Loss.png
        ├── <标签>_显著性检验_<指标>_箱线图.png   # 带 Tukey HSD 逐对显著性标注（* / ** / ***）；不画 K 次散点，
        │                                        # 横线自动分层，星号不会被裁掉也不会互相重叠
        ├── <标签>_显著性检验_统计表.csv          # 箱线图的 n/中位数/均值/标准差/Q1/Q3/IQR
        ├── <标签>_显著性检验说明.md              # 图上 * 的含义 + ANOVA/Tukey 结论
        ├── <标签>_对比汇总_test.csv              # 每一次实验一行（含随机种子），便于自己再统计
        └── 统计结果/                            # ANOVA 完整结果
            ├── <噪声>_显著性检验.json
            ├── <噪声>_显著性检验.csv
            └── <噪声>_显著性检验摘要.json
```

> `<标签>` = `<噪声类型>去噪_<采样率>Hz`，例如 `EMG去噪_512Hz`。
> 每种噪声类型下的 K 次实验是**独立重复实验**（不是 K 折交叉验证）：每次实验使用不同的随机种子，
> 重新随机混合生成半仿真数据、重新划分训练/验证/测试集，并随机初始化网络权重。
>
> `results/<模型名>/` 这一层**只有** `<噪声类型>去噪_<采样率>HZ/` 文件夹，不会出现散落文件
> （运行日志也放在噪声目录内部）。

## 相关命令

```bash
# ① 一个模型的 K 次独立重复实验（K 默认 10）
python code/main.py --network Simple_CNN --noise-type EMG --repeat-k 10

# ② 五个模型各跑 K 次，跑完自动做 ANOVA + Tukey HSD 并画箱线图
python code/main.py --models fcNN Simple_CNN Complex_CNN RNN_lstm Novel_CNN \
  --noise-type EMG --repeat-k 10 --device cuda:0 --highlight Novel_CNN

# ③ 生成论文用对比图（按噪声类型分目录；曲线 = K 次均值 ± 标准差）
python code/compare_results.py --result-dir results --highlight Novel_CNN

# ③' 只要均值曲线、不画阴影；或换中位数±IQR
python code/compare_results.py --result-dir results --shade none
python code/compare_results.py --result-dir results --aggregate median --shade iqr

# ④ 单独做显著性检验（可对已有结果反复执行）
python code/significance_test.py --noise-type EMG --highlight Novel_CNN

# ⑤ 从某次实验恢复训练
python code/main.py --resume Simple_CNN/EMG去噪_512HZ/试验次数_1 --epochs 80
python code/main.py --resume latest --epochs 80

# ⑥ 对某次实验重新绘图（只画单次实验的诊断图，不动对比图）
python code/main.py --plot-only Simple_CNN/EMG去噪_512HZ

# ⑦ 测试集逐样本推理图（最后一次实验的权重；Noisy灰 / Pure绿 / Denoised蓝）
python code/inference.py --noise-type EOG
python code/inference.py --noise-type EMG --max-samples 20 --models fcNN Novel_CNN
python code/inference.py --noise-type EMG --snr -4 0        # 指定 SNR 档位（默认 worst=-7dB）

# ⑧ 训练时顺带做推理：训练 → ANOVA → 推理 一条命令
python code/main.py --models fcNN Simple_CNN Complex_CNN RNN_lstm Novel_CNN \
  --noise-type EOG --repeat-k 10 --device cuda:0 --infer 1
```

详细说明见项目根目录的 `说明文档.md`。
