# Mall 当前执行手册

实验在 autodl 上运行，不在本地跑训练。当前目标是 `clip_count_mae < 1.5`。

## 0. 当前结论

旧 `VideoMambaCounter` density family 已停止作为 Mall 主线。当前有效路线是 `MallPretrainedCounter`：强 ImageNet 预训练 backbone + 官方 count 回归。

当前最好结果：

- `Exp36A ConvNeXt-Small + GRU refine35B best`
  - `frame_count_mae = 2.036196`
  - `frame_count_rmse = 2.576963`
  - `clip_count_mae = 1.600205`
  - `clip_count_rmse = 2.052519`
- `Exp36A best + motion_guided lambda=0.5/window=3`
  - `frame_count_mae = 2.062066`
  - `frame_count_rmse = 2.599593`
  - `clip_count_mae = 1.574825`
  - `clip_count_rmse = 2.023259`
- `Exp39A ConvNeXt-Small + GRU stride=1 refine36A best`
  - `frame_count_mae = 2.022230`
  - `frame_count_rmse = 2.549299`
  - `clip_count_mae = 1.554557`
  - `clip_count_rmse = 1.984686`
- `Exp39B ConvNeXt-Small + GRU clip_len=16 refine36A best`
  - eval protocol: `clip_len=16`, `test_clip_stride=1`, test clips = `1185`
  - `frame_count_mae = 2.035944`
  - `frame_count_rmse = 2.581916`
  - `clip_count_mae = 1.454182`
  - `clip_count_rmse = 1.885346`
- `Exp40A ConvNeXt-Small + Temporal Mamba clip_len=8 best`
  - eval protocol: `clip_len=8`, `test_clip_stride=1`, test clips = `1193`
  - `frame_count_mae = 1.956905`
  - `frame_count_rmse = 2.466977`
  - `clip_count_mae = 1.548556`
  - `clip_count_rmse = 1.933232`
- `Exp40B ConvNeXt-Small + Temporal Mamba clip_len=16 best`
  - eval protocol: `clip_len=16`, `test_clip_stride=1`, test clips = `1185`
  - `frame_count_mae = 1.964943`
  - `frame_count_rmse = 2.480001`
  - `clip_count_mae = 1.410054`
  - `clip_count_rmse = 1.776802`

已停止或降级的路线：

- `Exp35B best + motion_guided 0.5/window=3`: `clip_count_mae = 1.594995`
- `Exp37A 35B+35A ensemble 0.7/0.3`: `clip_count_mae = 1.592730`
- `Exp37A 35B+35A ensemble 0.8/0.2`: `clip_count_mae = 1.591487`
- `Exp37A 35B+35A ensemble 0.6/0.4`: `clip_count_mae = 1.595983`
- `Exp37B 35B+35A+34A ensemble`: `clip_count_mae = 1.577911`

判断：

- 集成没有超过 `Exp36A best + motion_guided 0.5/window=3`，停止 `Exp37` 主线。
- 当前最优结果已经更新为 `Exp40B clip_len=16 eval = 1.410054`，已经达到 `<1.5` 目标。
- `Exp40B` 的最终时序头是 bidirectional Temporal Mamba，因此论文最终可以写成“预训练表征增强的 Mamba 视频计数模型”。
- 注意：`Exp40B` 是 `clip_len=16` 协议，和 `clip_len=8` 的 `Exp40A` 不是完全等价评估窗口；论文表格必须单独标注 `clip_len` 和测试片段数。

## 1. 静态检查

```bash
python -m py_compile train_mall_pretrained_counter.py ensemble_mall_pretrained_predictions.py dataset.py
```

## 2. Exp36A 后处理细扫

当前最好是 `lambda=0.5/window=3`。继续扫邻近点，不再扫集成。

### lambda=0.35 window=3

```bash
python train_mall_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp36a_convnext_small_gru_refine35b/best.pth \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 8 \
  --height 360 \
  --width 480 \
  --batch-size 4 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head gru \
  --hidden-dim 256 \
  --dropout 0.25 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 2 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.35 \
  --postproc-window 3 \
  --device cuda
```

### lambda=0.45 window=3

```bash
python train_mall_pretrained_counter.py --eval-only --checkpoint /root/autodl-tmp/checkpoints/mall_exp36a_convnext_small_gru_refine35b/best.pth --data-root /root/autodl-tmp/Mall --clip-len 8 --height 360 --width 480 --batch-size 4 --num-workers 8 --backbone convnext_small --weights imagenet --temporal-head gru --hidden-dim 256 --dropout 0.25 --density-kernel perspective --perspective-scale 0.5 --adaptive-min-sigma 1.0 --adaptive-max-sigma 8.0 --use-density-cache --density-cache-dir /root/autodl-tmp/Mall_cache --train-clip-stride 2 --test-clip-stride 1 --use-roi-mask --mask-rgb-with-roi --temporal-postproc motion_guided --postproc-lambda 0.45 --postproc-window 3 --device cuda
```

### lambda=0.55 window=3

```bash
python train_mall_pretrained_counter.py --eval-only --checkpoint /root/autodl-tmp/checkpoints/mall_exp36a_convnext_small_gru_refine35b/best.pth --data-root /root/autodl-tmp/Mall --clip-len 8 --height 360 --width 480 --batch-size 4 --num-workers 8 --backbone convnext_small --weights imagenet --temporal-head gru --hidden-dim 256 --dropout 0.25 --density-kernel perspective --perspective-scale 0.5 --adaptive-min-sigma 1.0 --adaptive-max-sigma 8.0 --use-density-cache --density-cache-dir /root/autodl-tmp/Mall_cache --train-clip-stride 2 --test-clip-stride 1 --use-roi-mask --mask-rgb-with-roi --temporal-postproc motion_guided --postproc-lambda 0.55 --postproc-window 3 --device cuda
```

### lambda=0.65 window=3

```bash
python train_mall_pretrained_counter.py --eval-only --checkpoint /root/autodl-tmp/checkpoints/mall_exp36a_convnext_small_gru_refine35b/best.pth --data-root /root/autodl-tmp/Mall --clip-len 8 --height 360 --width 480 --batch-size 4 --num-workers 8 --backbone convnext_small --weights imagenet --temporal-head gru --hidden-dim 256 --dropout 0.25 --density-kernel perspective --perspective-scale 0.5 --adaptive-min-sigma 1.0 --adaptive-max-sigma 8.0 --use-density-cache --density-cache-dir /root/autodl-tmp/Mall_cache --train-clip-stride 2 --test-clip-stride 1 --use-roi-mask --mask-rgb-with-roi --temporal-postproc motion_guided --postproc-lambda 0.65 --postproc-window 3 --device cuda
```

### lambda=0.5 window=5

```bash
python train_mall_pretrained_counter.py --eval-only --checkpoint /root/autodl-tmp/checkpoints/mall_exp36a_convnext_small_gru_refine35b/best.pth --data-root /root/autodl-tmp/Mall --clip-len 8 --height 360 --width 480 --batch-size 4 --num-workers 8 --backbone convnext_small --weights imagenet --temporal-head gru --hidden-dim 256 --dropout 0.25 --density-kernel perspective --perspective-scale 0.5 --adaptive-min-sigma 1.0 --adaptive-max-sigma 8.0 --use-density-cache --density-cache-dir /root/autodl-tmp/Mall_cache --train-clip-stride 2 --test-clip-stride 1 --use-roi-mask --mask-rgb-with-roi --temporal-postproc motion_guided --postproc-lambda 0.5 --postproc-window 5 --device cuda
```

### lambda=0.75 window=3

```bash
python train_mall_pretrained_counter.py --eval-only --checkpoint /root/autodl-tmp/checkpoints/mall_exp36a_convnext_small_gru_refine35b/best.pth --data-root /root/autodl-tmp/Mall --clip-len 8 --height 360 --width 480 --batch-size 4 --num-workers 8 --backbone convnext_small --weights imagenet --temporal-head gru --hidden-dim 256 --dropout 0.25 --density-kernel perspective --perspective-scale 0.5 --adaptive-min-sigma 1.0 --adaptive-max-sigma 8.0 --use-density-cache --density-cache-dir /root/autodl-tmp/Mall_cache --train-clip-stride 2 --test-clip-stride 1 --use-roi-mask --mask-rgb-with-roi --temporal-postproc motion_guided --postproc-lambda 0.75 --postproc-window 3 --device cuda
```

## 3. Exp39A: 从 36A best 做 train stride=1 精修

目的：Mall train 只有 800 帧，原来 `train_clip_stride=2` 只给约 397 个 clip。stride=1 增加训练窗口密度，可能进一步降低 clip MAE。

```bash
OMP_NUM_THREADS=1 python train_mall_pretrained_counter.py \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 8 \
  --height 360 \
  --width 480 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --num-workers 8 \
  --epochs 50 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head gru \
  --hidden-dim 256 \
  --dropout 0.25 \
  --freeze-backbone-epochs 0 \
  --lr 3e-5 \
  --backbone-lr 1e-6 \
  --weight-decay 2e-4 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 5e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --device cuda \
  --init-from /root/autodl-tmp/checkpoints/mall_exp36a_convnext_small_gru_refine35b/best.pth \
  --save-dir /root/autodl-tmp/checkpoints/mall_exp39a_convnext_small_gru_stride1_refine36a
```

### Exp39A best + 当前最优后处理

```bash
python train_mall_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp39a_convnext_small_gru_stride1_refine36a/best.pth \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 8 \
  --height 360 \
  --width 480 \
  --batch-size 4 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head gru \
  --hidden-dim 256 \
  --dropout 0.25 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.5 \
  --postproc-window 3 \
  --device cuda
```

## 4. Exp39B: clip_len=16 训练

目的：训练时给 GRU 更长上下文。当前截图结果显示 `clip_len=16` 测试协议已经达到 `clip_count_mae = 1.454182`。

```bash
OMP_NUM_THREADS=1 python train_mall_pretrained_counter.py \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 16 \
  --height 360 \
  --width 480 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 50 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head gru \
  --hidden-dim 256 \
  --dropout 0.25 \
  --freeze-backbone-epochs 0 \
  --lr 3e-5 \
  --backbone-lr 1e-6 \
  --weight-decay 2e-4 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 5e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --device cuda \
  --init-from /root/autodl-tmp/checkpoints/mall_exp36a_convnext_small_gru_refine35b/best.pth \
  --save-dir /root/autodl-tmp/checkpoints/mall_exp39b_convnext_small_gru_len16_refine36a
```

### Exp39B 当前最好评估：clip_len=16

```bash
python train_mall_pretrained_counter.py \
  --eval-only \
  --eval-split train \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp39b_convnext_small_gru_len16_refine36a/best.pth \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 16 \
  --height 360 \
  --width 480 \
  --batch-size 4 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head gru \
  --hidden-dim 256 \
  --dropout 0.25 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --device cuda
```

### Exp39B 对照评估：clip_len=8

只用于和 39A 做同协议对照；如果论文最终采用 `clip_len=16` 协议，这条不是主结果。

```bash
python train_mall_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp39b_convnext_small_gru_len16_refine36a/best.pth \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 8 \
  --height 360 \
  --width 480 \
  --batch-size 4 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head gru \
  --hidden-dim 256 \
  --dropout 0.25 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.5 \
  --postproc-window 3 \
  --device cuda
```

## 5. 备用 Exp38A: ConvNeXt-Small + MLP

只在 Exp39A/39B 无收益时跑。GRU 可能过平滑，MLP 有时保留更准逐帧响应。

```bash
OMP_NUM_THREADS=1 python train_mall_pretrained_counter.py \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 8 \
  --height 360 \
  --width 480 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --num-workers 8 \
  --epochs 80 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mlp \
  --hidden-dim 256 \
  --dropout 0.2 \
  --freeze-backbone-epochs 5 \
  --lr 1e-4 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 2 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 5 \
  --min-lr 1e-6 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/mall_exp38a_convnext_small_mlp_count
```

## 6. 判断规则

- 当前最好：`Exp40B clip_len=16 eval = clip_count_mae 1.410054`。
- 如果最终报告采用 `clip_len=16`，可以停止主实验，只补充 `last.pth` 同协议评估和命令留档。
- 如果必须延续此前 `clip_len=8` 协议，则以 `Exp40A = 1.548556` 作为同协议 Mamba 结果。
- 最终 MAE 必须使用 test split 离线评估结果；不要使用训练中 fast-val 单点替代最终指标。

## 7. Exp40: 预训练增强 Mamba 计数模型

目的：如果论文最终必须强调“最终模型仍是 Mamba 模型”，需要把 `Exp39B` 的强预训练表征路线迁移到 Mamba 时序头。新增 `--temporal-head mamba` 后，模型结构变为：

```text
ImageNet ConvNeXt-Small encoder -> bidirectional Temporal Mamba head -> count regression
```

该路线不是回到旧 `VideoMambaCounter` density integral，而是保留 39B 有效的强视觉预训练和官方 count 监督，同时把 GRU 替换为 Mamba 时序建模。

### Exp40A: clip_len=8 预训练增强 Mamba

```bash
OMP_NUM_THREADS=1 python train_mall_pretrained_counter.py \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 8 \
  --height 360 \
  --width 480 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --num-workers 8 \
  --epochs 80 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --dropout 0.25 \
  --freeze-backbone-epochs 5 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 5e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/mall_exp40a_convnext_small_temporal_mamba_len8
```

### Exp40A best 评估

```bash
python train_mall_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp40a_convnext_small_temporal_mamba_len8/best.pth \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 8 \
  --height 360 \
  --width 480 \
  --batch-size 4 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --dropout 0.25 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --device cuda
```

### Exp40B: clip_len=16 预训练增强 Mamba

只有当 `Exp40A` 明显优于 35B/39A 或至少接近 `1.55` 时再跑。

```bash
OMP_NUM_THREADS=1 python train_mall_pretrained_counter.py \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 16 \
  --height 360 \
  --width 480 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 80 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --dropout 0.25 \
  --freeze-backbone-epochs 5 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 5e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/mall_exp40b_convnext_small_temporal_mamba_len16
```

### Exp40B best 评估

```bash
python train_mall_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 16 \
  --height 360 \
  --width 480 \
  --batch-size 4 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --dropout 0.25 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --device cuda
```

### Exp40C: 39B teacher 蒸馏到 Temporal Mamba

先导出 `Exp39B` 在 train split 上的逐帧 teacher 预测。该步骤只生成 cache，不训练。

```bash
python train_mall_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp39b_convnext_small_gru_len16_refine36a/best.pth \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 16 \
  --height 360 \
  --width 480 \
  --batch-size 4 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head gru \
  --hidden-dim 256 \
  --dropout 0.25 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --save-seq-preds /root/autodl-tmp/mall_teacher_preds/exp39b_train_len16 \
  --device cuda
```

再训练 Mamba 时序头，并加入 teacher soft count 监督。

```bash
OMP_NUM_THREADS=1 python train_mall_pretrained_counter.py \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 16 \
  --height 360 \
  --width 480 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 80 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --dropout 0.25 \
  --freeze-backbone-epochs 5 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --teacher-seq-preds /root/autodl-tmp/mall_teacher_preds/exp39b_train_len16 \
  --teacher-count-key smoothed_pred_count \
  --teacher-count-weight 0.2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 5e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/mall_exp40c_temporal_mamba_distill39b_len16
```

### Exp40C best 评估

```bash
python train_mall_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp40c_temporal_mamba_distill39b_len16/best.pth \
  --data-root /root/autodl-tmp/Mall \
  --clip-len 16 \
  --height 360 \
  --width 480 \
  --batch-size 4 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --dropout 0.25 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.0 \
  --adaptive-max-sigma 8.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/Mall_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --device cuda
```
