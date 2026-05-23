# ConvNeXt-Small-with-Mamba-for-Video-Crowd-Counting
## 模型简介
本项目提出一种面向固定摄像头视频人群计数的预训练增强 Mamba 模型。整体结构由三部分组成：  
1. `ConvNeXt-Small` 作为逐帧视觉编码器，提取稳定的空间特征；  
2. `ROI` 掩码与场景先验，用于过滤固定场景中的无关背景；  
3. 双向 `Temporal Mamba` 模块，在视频片段内融合前后文时序信息，缓解遮挡、交叠和局部噪声带来的计数波动。  
模型最终输出帧级人数预测，并通过片段级聚合得到视频片段人数估计。
## 实验配置
- 训练平台：AutoDL
- GPU：NVIDIA RTX 4090 24GB
- CPU：Intel Xeon Gold 6430
- 内存：120GB
- Python：3.10.14
- PyTorch：2.1.1
- CUDA：11.8
- `mamba-ssm`：2.2.2
- `causal-conv1d`：1.4.0
主要实验设置：
- 视觉骨干：ImageNet 预训练 `ConvNeXt-Small`
- 时序模块：双向 `Temporal Mamba`
- 输入片段长度：`clip_len=16`
- 评价指标：`MAE`、`RMSE`
## 实验结果
### Mall 数据集
Mall 是固定摄像头单场景视频人群计数数据集，场景背景稳定，适合验证时序建模和 ROI 先验的作用。
| 方法 | MAE | RMSE |
| --- | ---: | ---: |
| 本文方法 | 1.41 | 1.78 |
### UCSD 数据集
UCSD 用于补充验证模型在另一类固定场景下的泛化能力。
| 方法 | MAE | RMSE |
| --- | ---: | ---: |
| 本文方法 | 1.09 | 1.43 |
## 结论
实验表明，`ConvNeXt-Small` 的强视觉表征能力与 `Temporal Mamba` 的高效时序建模能力可以形成互补，在固定摄像头视频人群计数任务中取得稳定表现。
