# 实验结果目录

本目录用于存放每一次训练的**全部结果**，由训练脚本自动创建子文件夹，命名规则为：

```
<文件前缀>_试验次数_<N>/
例如：测试集_EMG去噪_512HZ_试验次数_1
```

每个子文件夹包含：最优模型权重（`*.pt`）、断点文件（`*_断点续训.pt`）、
Loss（`*_Loss.npy`）、各数据划分的各指标（`<划分>_<噪声>去噪_<采样率>HZ_<指标>_指标.npy`）、
训练配置（`训练配置.json`）、实验摘要（`实验摘要.json`）、训练日志（`训练日志.txt`）、
自动生成的图像（`*.png`）以及划分好的数据集（`划分数据/`）。

相关命令：

```bash
# 训练（自动新建 _试验次数_N 文件夹，不会覆盖已有结果）
python code/main.py --network Simple_CNN --noise-type EMG

# 从某次实验恢复训练
python code/main.py --resume 测试集_EMG去噪_512HZ_试验次数_1 --epochs 80
python code/main.py --resume latest --epochs 80

# 对某次实验重新绘图
python code/main.py --plot-only 测试集_EMG去噪_512HZ_试验次数_1

# 多次实验结果对比（loss / MSE / SNR / CC / RRMSE）
python code/compare_results.py --result-dir code/results --split test
```

详细说明见项目根目录的 `说明文档.md`。
