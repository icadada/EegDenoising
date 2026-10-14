# 实验结果目录

本目录用于存放**全部训练与统计结果**，由训练脚本自动创建，位于**与 `code/`、`data/` 同级**的位置。

## 目录结构

```
results/
└── <模型名>/                                  # 例如 Simple_CNN、fcNN、RNN_lstm
    └── <噪声类型>去噪_<采样率>HZ/               # 例如 EMG去噪_512HZ、EOG去噪_256HZ
        ├── 指标汇总_均值标准差中位数IQR.json      # K 次独立实验各指标的 Mean/STD/Median/IQR
        ├── 该次实验汇总.json                    # 每次实验一行（含随机种子、最优轮次、各指标）
        ├── <模型名>_运行日志.txt                # 该模型 K 次实验的总日志
        ├── 统计分析/                            # ANOVA + Tukey HSD 结果
        │   ├── <噪声类型>_显著性检验.json
        │   ├── <噪声类型>_显著性检验.csv
        │   └── <噪声类型>_显著性检验摘要.json
        ├── 图像/                                # 跨实验的统计图（带显著性标注的箱线图）
        │   └── <噪声类型>_显著性检验_<指标>_箱线图.png
        ├── 试验次数_1/                          # 第 1 次独立重复实验
        │   ├── 权重/                            # 最优模型权重（*.pt）、断点（*_断点续训.pt）
        │   ├── 指标/                            # Loss（*_Loss.npy）、各划分各指标（*.npy）
        │   ├── 图像/                            # 自动生成的 *.png
        │   ├── 训练配置/                        # 训练配置.json、实验摘要.json、各划分整体指标.json
        │   ├── 训练日志.txt                     # 本次实验的完整日志
        │   └── 划分数据/                        # 本次实验划分好的数据集
        ├── 试验次数_2/ ...
        └── 试验次数_K/                          # K 默认 10
```

> 每种噪声类型下的 K 次实验是**独立重复实验**（不是 K 折交叉验证）：每次实验使用不同的随机种子，
> 重新随机混合生成半仿真数据、重新划分训练/验证/测试集，并随机初始化网络权重。

## 相关命令

```bash
# ① 一个模型的 K 次独立重复实验（K 默认 10）
python code/main.py --network Simple_CNN --noise-type EMG --repeat-k 10

# ② 五个模型各跑 K 次，跑完自动做 ANOVA + Tukey HSD 并画箱线图
python code/main.py --models fcNN Simple_CNN Complex_CNN RNN_lstm Novel_CNN \
  --noise-type EMG --repeat-k 10 --device cuda:0

# ③ 单独做显著性检验（可对已有结果反复执行）
python code/significance_test.py --noise-type EMG

# ④ 从某次实验恢复训练
python code/main.py --resume Simple_CNN/EMG去噪_512HZ/试验次数_1 --epochs 80
python code/main.py --resume latest --epochs 80

# ⑤ 对某次实验重新绘图
python code/main.py --plot-only Simple_CNN/EMG去噪_512HZ

# ⑥ 多次实验的结果对比（loss / MSE / SNR / CC / RRMSE）
python code/compare_results.py --result-dir results --split test
```

详细说明见项目根目录的 `说明文档.md`。
