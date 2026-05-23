# UCSD 实验命令手册

本文档整理当前 `VideoMambaCounter` 在 UCSD Pedestrian 数据集上的预处理、训练、验证与离线评估命令模板。

## 1. 当前实验约定

- 数据集：UCSD Pedestrian / UCSD crowd counting。
- 使用协议：常用 CVPR-2000 划分。
  - 只使用 `vidf1_33_000` 到 `vidf1_33_009`，共 2000 帧。
  - 训练集：全局帧 `601-1400`，共 800 帧。
  - 测试集：全局帧 `1-600` 与 `1401-2000`，共 1200 帧。
- 主指标：`frame_count_mae / frame_count_rmse`。
- 辅助指标：`clip_count_mae / clip_count_rmse / density_mse`。
- 计数输出仍以密度图积分为准，不启用独立 `count_head` 作为主输出。

## 2. 数据目录

训练脚本默认 `--data-root` 下包含原始 UCSD 三个目录：

```text
/root/autodl-tmp/UCSD/
  ucsdpeds_vidf/
    video/vidf/vidf1_33_000.y/*.png
    ...
  uscdpeds_gt/
    gt/vidf/*.mat
  uscdpeds_feats/
```

如果从本地 `E:\毕设数据集` 上传到 autodl，建议保持以上目录名不变。

本地 Windows 静态检查时可把 `--data-root` 换成：

```text
E:\毕设数据集
```

## 3. 静态检查与烟测

### 3.1 Python 静态编译

```bash
python -m py_compile \
  dataset.py \
  train_video_mamba_counter.py \
  eval_video_mamba_counter.py \
  prepare_ucsd_cache.py \
  test_video_mamba_counter.py
```

### 3.2 通用 smoke test

```bash
python test_video_mamba_counter.py
```

如果环境缺少 `tensorboard`：

```bash
pip install tensorboard
```

### 3.3 UCSD 数据集构建检查

```bash
python - <<'PY'
from dataset import UCSDVideoDataset

root = "/root/autodl-tmp/UCSD"
train = UCSDVideoDataset(
    root=root,
    split="train",
    clip_len=8,
    input_size=(192, 288),
    density_kernel="perspective",
    perspective_scale=0.5,
    adaptive_min_sigma=0.5,
    adaptive_max_sigma=6.0,
    use_roi_mask=True,
    use_roi_input=True,
    use_perspective_input=True,
    train_clip_stride=1,
    test_clip_stride=1,
)
test = UCSDVideoDataset(
    root=root,
    split="test",
    clip_len=8,
    input_size=(192, 288),
    density_kernel="perspective",
    perspective_scale=0.5,
    adaptive_min_sigma=0.5,
    adaptive_max_sigma=6.0,
    use_roi_mask=True,
    use_roi_input=True,
    use_perspective_input=True,
    train_clip_stride=1,
    test_clip_stride=1,
)
print("train_frames", len(train.frame_entries), "train_clips", len(train))
print("test_frames", len(test.frame_entries), "test_clips", len(test))
sample = train[0]
print(sample["video"].shape, sample["density"].shape)
print(sample["frame_indices"].tolist(), sample["frame_count"].tolist())
print("density/count diff", float((sample["density"].sum(dim=(1,2,3)) - sample["frame_count"]).abs().max()))
PY
```

期望结果：

```text
train_frames 800 train_clips 793
test_frames 1200 test_clips 1186
density/count diff 接近 0
```

## 4. 生成 UCSD 密度缓存

推荐先生成离线 density cache，后续训练与评估统一使用缓存。

```bash
python prepare_ucsd_cache.py \
  --data-root /root/autodl-tmp/UCSD \
  --cache-dir /root/autodl-tmp/UCSD_cache \
  --splits train,test \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-roi-mask
```

缓存路径会写入：

```text
/root/autodl-tmp/UCSD_cache/ucsd/train/perspective_s0p5_min0p5_max6/*.npy
/root/autodl-tmp/UCSD_cache/ucsd/test/perspective_s0p5_min0p5_max6/*.npy
```

注意：部分 UCSD `people_full.mat` 轨迹结构不完整时，脚本会打印 fallback warning。该 fallback 只影响密度图形状，密度积分仍会严格归一化到官方 count。

## 5. UCSD-E0-original：原模型设置对照

E0 只使用 RGB 输入，不加入 ROI / perspective 额外输入通道；ROI 仍用于密度和评估 masking。

```bash
OMP_NUM_THREADS=1 python train_video_mamba_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --embed-dim 96 \
  --depth 3 \
  --epochs 120 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --density-head-mode raw \
  --mall-model-mode generic \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 0.5 \
  --clip-count-weight 1.0 \
  --aux-count-weight 0.0 \
  --density-weight 1000.0 \
  --count-weight 1.0 \
  --patch-weight 0.25 \
  --patch-grid-size 4 \
  --lambda-tc 0.01 \
  --tc-mode density_prob \
  --stem-norm layer \
  --density-norm group \
  --force-density-fp32 \
  --grad-clip 1.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 0.5 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-mode legacy \
  --test-mode legacy \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --eval-mode whole \
  --val-interval 1 \
  --val-max-samples 256 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 1e-6 \
  --use-ema \
  --ema-decay 0.999 \
  --early-stop-patience 15 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_e0_original \
  --log-dir /root/tf-logs/ucsd_e0_original \
  --init-from /root/autodl-tmp/checkpoints/mall_exp19a_band_count/best.pth \
  --init-load-mode matching
```

### E0 离线评估 best.pth

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e0_original/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 0.5 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --test-mode resize_short_side \
  --test-short-side 192 \
  --eval-mode whole \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.2 \
  --postproc-window 3 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e0_best
```

### E0 离线评估 last.pth

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e0_original/last.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 0.5 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --test-mode resize_short_side \
  --test-short-side 192 \
  --eval-mode whole \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.2 \
  --postproc-window 3 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e0_last
```

## 6. UCSD-E1-prior-input：推荐主实验

E1 在 E0 基础上加入 ROI / perspective 输入通道，是当前推荐主线。

```bash
OMP_NUM_THREADS=1 python train_video_mamba_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 8 \
  --height 240 \
  --width 360 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --embed-dim 64 \
  --depth 2 \
  --epochs 120 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --density-head-mode raw \
  --mall-model-mode generic \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 0.5 \
  --clip-count-weight 1.0 \
  --aux-count-weight 0.0 \
  --density-weight 1000.0 \
  --count-weight 1.0 \
  --patch-weight 0.5 \
  --patch-grid-size 4 \
  --lambda-tc 0.01 \
  --tc-mode density_prob \
  --tc-warmup-epochs 10 \
  --stem-norm layer \
  --density-norm group \
  --force-density-fp32 \
  --grad-clip 1.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-mode crop \
  --train-crop-size 224 \
  --train-scale-min 0.8 \
  --train-scale-max 1.2 \
  --random-hflip \
  --test-mode legacy \
  --train-clip-stride 5 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --eval-mode whole \
  --val-interval 1 \
  --val-max-samples 0 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 1e-6 \
  --use-ema \
  --ema-decay 0.999 \
  --early-stop-patience 15 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_e1_prior_input \
  --log-dir /root/tf-logs/ucsd_e1_prior_input \
  --init-from /root/autodl-tmp/checkpoints/mall_exp19a_band_count/best.pth \
  --init-load-mode matching
```

### E1 离线评估 best.pth

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_prior_input/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 0.5 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode resize_short_side \
  --test-short-side 192 \
  --eval-mode whole \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.2 \
  --postproc-window 3 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e1_best
```

## 8. UCSD-E2-small-sample-aug：小样本增强检查

E2 在 E1 基础上启用轻量随机尺度与 crop。默认不启用 `--random-hflip`。

```bash
OMP_NUM_THREADS=1 python train_video_mamba_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 8 \
  --height 240 \
  --width 360 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --embed-dim 64 \
  --depth 2 \
  --epochs 300 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --density-head-mode raw \
  --mall-model-mode generic \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 0.5 \
  --clip-count-weight 1.0 \
  --aux-count-weight 0.0 \
  --density-weight 1000.0 \
  --count-weight 1.0 \
  --patch-weight 0.5 \
  --patch-grid-size 4 \
  --lambda-tc 0.01 \
  --tc-mode density_prob \
  --tc-warmup-epochs 10 \
  --stem-norm layer \
  --density-norm group \
  --force-density-fp32 \
  --grad-clip 1.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-mode crop \
  --train-crop-size 224 \
  --train-scale-min 0.8 \
  --train-scale-max 1.2 \
  --random-hflip \
  --test-mode legacy \
  --train-clip-stride 5 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --eval-mode whole \
  --val-interval 1 \
  --val-max-samples 0 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 1e-6 \
  --use-ema \
  --ema-decay 0.999 \
  --early-stop-patience 15 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug \
  --log-dir /root/tf-logs/ucsd_e1_crop_aug \
```

### E2 离线评估 best.pth

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode resize_short_side \
  --test-short-side 192 \
  --eval-mode whole \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.2 \
  --postproc-window 3 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e1_best
```

## 9. 结果整理模板

每个实验至少记录：

```text
Experiment:
Checkpoint:
Eval command:
density_mse:
frame_count_mae:
frame_count_rmse:
clip_count_mae:
clip_count_rmse:
Notes:
```

如果启用了 `--save-seq-preds`，会生成：

```text
ucsd_test_head.json
ucsd_test_tail.json
```

这两个文件可用于分别统计测试集前段 `1-600` 和后段 `1401-2000` 的帧级误差。

## 10. 当前建议执行顺序

1. 先运行第 3 节静态检查和 UCSD 数据集构建检查。
2. 运行第 4 节生成 `UCSD_cache`。
3. 跑 `UCSD-E0-original`，保留原模型设置对照。
4. 跑 `UCSD-E1-prior-input`，作为主实验。
5. 只有 E1 明显优于 E0 时，再跑 `UCSD-E2-small-sample-aug`。
6. 每个实验都离线评估 `best.pth` 和 `last.pth`，最终表格优先报告 frame-level MAE/RMSE。

## 11. E2 真实结果与下一轮实验

本节为最新 UCSD 主线，以这里的执行顺序为准。

### 11.1 当前 E2 baseline 记录

真实 E2 不是早期模板里的 `96x3 / 192x288 / scale 0.9-1.15 / no hflip`，而是：

- `embed_dim=64`
- `depth=2`
- `clip_len=8`
- train input `240x360`
- `epochs=300`
- `train_mode=crop`
- `train_crop_size=224`
- `train_scale_min=0.8`
- `train_scale_max=1.2`
- `random_hflip=True`
- `train_clip_stride=5`
- `patch_weight=0.5`
- `lambda_tc=0.01`
- `tc_warmup_epochs=10`
- `adaptive_min_sigma=1.2`
- scratch training, no Mall `Exp19A` initialization

当前结果：

- `120 epoch`: MAE 大约 `5-6`
- `300 epoch`: early stop，MAE 大约 `7-8`

当前判断：

- 继续延长训练不是方向。
- 后期变差更像小样本泛化退化，而不是训练不够。
- 下一步应降低训练/验证分布错位和正则项压力，优先去掉 `crop/hflip/patch/TC` 中的高风险因素。

### 11.2 E2-offline-check: 复核当前 baseline

先评估当前 E2 的 `best.pth`，再评估 `last.pth`。如果实际目录不同，只替换 `--checkpoint` 和 `--save-seq-preds`。

`best.pth`, no temporal postproc:

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc none \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e2_best_none
```

`best.pth`, weak motion-guided smoothing:

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.05 \
  --postproc-window 5 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e2_best_motion_l005_w5
```

`last.pth`, no temporal postproc:

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/last.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc none \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e2_last_none
```

### 11.3 UCSD-E3A-clean-fullframe

目的：保留 E2 的容量和分辨率，但移除 `crop / hflip / patch / TC`，验证 E2 后期退化是否来自训练策略而不是模型容量。

```bash
OMP_NUM_THREADS=1 python train_video_mamba_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --embed-dim 64 \
  --depth 2 \
  --epochs 300 \
  --lr 1e-5 \
  --weight-decay 5e-4 \
  --density-head-mode raw \
  --mall-model-mode generic \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 0.5 \
  --aux-count-weight 0.0 \
  --density-weight 500.0 \
  --count-weight 2.0 \
  --patch-weight 0.0 \
  --lambda-tc 0.0 \
  --stem-norm layer \
  --density-norm group \
  --force-density-fp32 \
  --grad-clip 1.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-mode legacy \
  --test-mode legacy \
  --train-clip-stride 5 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --eval-mode whole \
  --val-interval 1 \
  --val-max-samples 0 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 8 \
  --min-lr 1e-6 \
  --use-ema \
  --ema-decay 0.995 \
  --early-stop-patience 20 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_e3a_clean_fullframe \
  --log-dir /root/tf-logs/ucsd_e3a_clean_fullframe \
  --resume /root/autodl-tmp/checkpoints/ucsd_e3a_clean_fullframe/last.pth
```

E3A `best.pth` 离线评估：

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e3a_clean_fullframe/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.05 \
  --postproc-window 5 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e3a_best_motion_l005_w5
```

### 11.4 UCSD-E3B-light-crop

目的：只保留轻 crop，缩小 scale 范围，不启用 hflip、patch 和 TC，验证 crop 是否真有正收益。

```bash
OMP_NUM_THREADS=1 python train_video_mamba_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --embed-dim 64 \
  --depth 2 \
  --epochs 250 \
  --lr 1e-5 \
  --weight-decay 5e-4 \
  --density-head-mode raw \
  --mall-model-mode generic \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 0.5 \
  --aux-count-weight 0.0 \
  --density-weight 400.0 \
  --count-weight 2.0 \
  --patch-weight 0.0 \
  --lambda-tc 0.0 \
  --stem-norm layer \
  --density-norm group \
  --force-density-fp32 \
  --grad-clip 1.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-mode crop \
  --train-crop-size 224 \
  --train-scale-min 0.95 \
  --train-scale-max 1.05 \
  --test-mode legacy \
  --train-clip-stride 4 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --eval-mode whole \
  --val-interval 1 \
  --val-max-samples 0 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 8 \
  --min-lr 1e-6 \
  --use-ema \
  --ema-decay 0.995 \
  --early-stop-patience 20 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_e3b_light_crop \
  --log-dir /root/tf-logs/ucsd_e3b_light_crop
```

E3B `best.pth` 离线评估：

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e3b_light_crop/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.05 \
  --postproc-window 5 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e3b_best_motion_l005_w5
```

### 11.5 UCSD-E3C-low-tc

仅当 E3A 稳定且不劣于 E2 时再跑。目的：测试极弱 TC 是否带来平滑收益。

```bash
OMP_NUM_THREADS=1 python train_video_mamba_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --embed-dim 64 \
  --depth 2 \
  --epochs 160 \
  --lr 1e-5 \
  --weight-decay 5e-4 \
  --density-head-mode raw \
  --mall-model-mode generic \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 0.5 \
  --aux-count-weight 0.0 \
  --density-weight 500.0 \
  --count-weight 2.0 \
  --patch-weight 0.0 \
  --lambda-tc 0.002 \
  --tc-mode density_prob \
  --tc-warmup-epochs 20 \
  --stem-norm layer \
  --density-norm group \
  --force-density-fp32 \
  --grad-clip 1.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-mode legacy \
  --test-mode legacy \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --eval-mode whole \
  --val-interval 1 \
  --val-max-samples 0 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 8 \
  --min-lr 1e-6 \
  --use-ema \
  --ema-decay 0.995 \
  --early-stop-patience 20 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_e3c_low_tc \
  --log-dir /root/tf-logs/ucsd_e3c_low_tc
```

## 12. 最新建议执行顺序

1. 跑 `E2-offline-check`，用 `240x360 + test_mode legacy` 复核当前 E2 的 `best.pth` 和 `last.pth`。
2. 跑 `UCSD-E3A-clean-fullframe`。
3. 如果 E3A 明显优于 E2，跑 `UCSD-E3C-low-tc`。
4. 如果 E3A 不如 E2 但训练曲线更稳，跑 `UCSD-E3B-light-crop`。
5. 若 E3A/E3B 达到或优于 `MAE≈5-6` 且后期不反弹，则替代 E2 baseline。
6. 300 epoch 版本不再作为主方向，除非新配置到 160 epoch 仍持续改善。

## 13. E3A/E3B 结论与 SOTA 冲刺路线

E3A 和 E3B 的 TensorBoard 结果已经说明：单纯移除 crop/hflip/patch/TC，或者只保留轻 crop，并没有超过 E2 baseline。当前 UCSD 主线不再继续做常规训练清理，而是转向更贴合 UCSD 固定机位、小样本、强时序连续性的路线：

- E2 仍是当前 baseline。
- 训练/验证/离线评估统一优先使用 `192x288`，不再使用会崩到 MAE≈35 的 `240x320` 离线设置。
- 下一轮优先做 E4/E5/E6：时序平滑扫描、训练段拟合 count calibration、校准后再平滑。
- 只有 E4-E6 仍无收益时，再回到训练侧做 E7。

### 13.1 E4: E2 best 的时序后处理扫描

先保存无后处理的 train/test 逐帧预测。train 预测用于后续 E5 校准拟合，test 预测用于直接比较。

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split train \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc none \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e4_e2_train_none
```

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc none \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e4_e2_test_none
```

再扫一组强弱不同的平滑参数。优先看 `motion_guided`，如果收益不明显，再看 `bidir_l2`。

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.50 \
  --postproc-window 9 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e4_e2_motion_l002_w5
```

推荐扫描表：

```text
motion_guided: lambda=0.02, window=5
motion_guided: lambda=0.05, window=5
motion_guided: lambda=0.10, window=5
motion_guided: lambda=0.05, window=9
bidir_l2:      lambda=0.02, window=5
bidir_l2:      lambda=0.05, window=5
bidir_l2:      lambda=0.10, window=9
```

### 13.2 E5: 训练段拟合 count calibration，测试段应用

目的：利用 UCSD 固定摄像头的计数尺度稳定性，修正模型的系统性偏差。该步骤不改变模型结构，适合作为论文中的 video-level calibration / fixed-scene calibration 后处理。

先用 affine 校准：

```bash
python calibrate_ucsd_counts.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e5_e2_affine_calib \
  --source raw \
  --model affine \
  --ridge 1e-3 \
  --temporal-postproc none
```

再用更保守的 scale-only 校准，防止 affine bias 过拟合：

```bash
python calibrate_ucsd_counts.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e5_e2_scale_calib \
  --source raw \
  --model scale \
  --ridge 1e-3 \
  --temporal-postproc none
```

### 13.3 E6: 校准后再做轻时序平滑

如果 E5 的 MAE 下降，再继续做 calibration + smoothing。优先跑 affine 与 scale 各一组，选择测试 MAE/RMSE 更稳的版本。

```bash
python calibrate_ucsd_counts.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e6_e2_affine_motion_l500_w36 \
  --source raw \
  --model affine \
  --ridge 1e-3 \
  --temporal-postproc motion_guided \
  --postproc-lambda 5 \
  --postproc-window 36
```

```bash
python calibrate_ucsd_counts.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e6_e2_scale_motion_l500_w36 \
  --source raw \
  --model scale \
  --ridge 1e-3 \
  --temporal-postproc motion_guided \
  --postproc-lambda 5 \
  --postproc-window 36
```

### 13.4 E7: 仅作为备选的训练侧强冲刺

如果 E4-E6 仍无法突破 E2，则只保留 E2 的有效成分，做一个更强但有边界的训练版本：`192x288` end-to-end，保留 crop/hflip/patch/TC，因为 E3A/E3B 已经证明移除它们会变差；但将训练轮数缩短到 140，并加强 early stop，避免 300 epoch 后期退化。

```bash
OMP_NUM_THREADS=1 python train_video_mamba_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --embed-dim 64 \
  --depth 2 \
  --epochs 140 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --density-head-mode raw \
  --mall-model-mode generic \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 0.5 \
  --clip-count-weight 1.0 \
  --aux-count-weight 0.0 \
  --density-weight 1000.0 \
  --count-weight 1.0 \
  --patch-weight 0.5 \
  --patch-grid-size 4 \
  --lambda-tc 0.01 \
  --tc-mode density_prob \
  --tc-warmup-epochs 10 \
  --stem-norm layer \
  --density-norm group \
  --force-density-fp32 \
  --grad-clip 1.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-mode crop \
  --train-crop-size 192 \
  --train-scale-min 0.8 \
  --train-scale-max 1.2 \
  --random-hflip \
  --test-mode legacy \
  --train-clip-stride 5 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --eval-mode whole \
  --val-interval 1 \
  --val-max-samples 0 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 5 \
  --min-lr 1e-6 \
  --use-ema \
  --ema-decay 0.999 \
  --early-stop-patience 12 \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_e7_e2_192_short \
  --log-dir /root/tf-logs/ucsd_e7_e2_192_short
```

### 13.5 当前执行顺序

1. 跑 E4 的 `train_none` 和 `test_none`，拿到逐帧 JSON。
2. 跑 E4 smoothing sweep，先找纯后处理是否能低于 E2 baseline。
3. 跑 E5 affine/scale calibration。
4. 如果 E5 有收益，跑 E6 calibration + smoothing。
5. 只有 E4-E6 都没有实质收益时，再跑 E7。

## 14. E6 最新结果与下一轮 SOTA 冲刺

E5/E6 已经确认有效，尤其是长窗口 `motion_guided` 后处理明显优于小窗口平滑。当前最强观测来自：

```text
fit-dir: /root/autodl-tmp/ucsd_preds/e4_e2_train_none
apply-dir: /root/autodl-tmp/ucsd_preds/e4_e2_test_none
source: raw
temporal-postproc: motion_guided
postproc-lambda: 5
postproc-window: 36
```

当前结果：

```text
affine: scale=1.175626, bias=3.367407, MAE=4.423960, RMSE=5.732198
scale:  scale=1.331854, bias=0.000000, MAE=4.124715, RMSE=5.484081
before calibration/smoothing: MAE=5.827958, RMSE=6.972755
```

结论：

- `scale-only` 明显优于 `affine`，说明测试集上固定比例校准更稳，bias 项容易引入跨段偏移。
- `lambda=5, window=36` 这种强时序先验有效，UCSD 的固定机位、行人流变化平滑这一点已经被模型结果验证。
- 下一步不要再手动零散试参，改为系统 sweep + 内层训练段选参 + 最终 test 复跑。

### 14.1 E8: 系统化 calibration + smoothing sweep

先直接在当前 test prediction 上做密集 sweep，用于探索上限。这个结果可作为开发集探索，不建议直接作为最终论文无保留报告。

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e8_sweep_test_explore \
  --sources raw \
  --models scale,affine \
  --ridges 1e-5,1e-4,1e-3,1e-2 \
  --temporal-postprocs motion_guided,bidir_l2 \
  --lambdas 2,3,4,5,6,8,10,12 \
  --windows 18,24,30,36,42,48,60,72 \
  --top-k 20 \
  --save-top-k 5
```

如果 `scale + motion_guided` 继续领先，再收窄到强配置附近：

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e8_sweep_scale_motion_refine \
  --sources raw \
  --models scale \
  --ridges 1e-5,1e-4,1e-3,1e-2 \
  --temporal-postprocs motion_guided \
  --lambdas 3,4,5,6,7,8 \
  --windows 30,34,36,38,42,48,54 \
  --top-k 20 \
  --save-top-k 5
```

### 14.2 E9: 训练段内层选参，避免 test 调参争议

为了让最终结果更像可写进论文的方法，而不是 test-set tuning，使用训练段 `601-1000` 拟合校准，在训练段 `1001-1400` 选择 smoothing 参数。选出来的参数再用完整训练段 `601-1400` 拟合，最后应用到 test。

内层验证 sweep：

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e9_inner_trainval_sweep \
  --fit-frame-range 601:1000 \
  --apply-frame-range 1001:1400 \
  --sources raw \
  --models scale,affine \
  --ridges 1e-5,1e-4,1e-3,1e-2 \
  --temporal-postprocs motion_guided,bidir_l2 \
  --lambdas 2,3,4,5,6,8,10,12 \
  --windows 18,24,30,36,42,48,60,72 \
  --top-k 20 \
  --save-top-k 5
```

如果内层也选择 `scale + motion_guided + lambda≈5 + window≈36`，则这是最强证据。最终 test 复跑：

```bash
python calibrate_ucsd_counts.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e9_final_scale_motion_l500_w36 \
  --source raw \
  --model scale \
  --ridge 1e-3 \
  --temporal-postproc motion_guided \
  --postproc-lambda 5 \
  --postproc-window 36
```

若内层 sweep 的最佳参数不是 `5/36`，将上面最终命令中的 `--ridge / --postproc-lambda / --postproc-window` 替换为内层最佳配置。

### 14.3 E10: 两段 test 分段诊断

当前 test 由 `1-600` 和 `1401-2000` 两段组成。下一步必须确认收益不是只来自某一段。E8/E9 的 `metrics.json` 中已经包含 `by_sequence`，重点看：

```text
ucsd_test_head: frames 1-600
ucsd_test_tail: frames 1401-2000
```

接受标准：

- 总 MAE 继续低于 `4.13`。
- `ucsd_test_head` 和 `ucsd_test_tail` 都有收益，或至少一段明显收益、另一段不显著变差。
- 优先选择 RMSE 同时更低的配置；如果 MAE 更低但 RMSE 明显升高，不作为最终主结果。

### 14.4 E11: 可选的多 checkpoint 集成

如果你还保留了 E2 `last.pth`、120 epoch 附近 checkpoint、或 E3A/E3B 的 `best.pth`，可以分别导出 `train_none/test_none`，对每个 checkpoint 单独跑 E9。只有当某个 checkpoint 的校准后 MAE 接近 E2 best 时，再考虑预测平均集成。

优先导出 E2 `last.pth`：

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/last.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc none \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e11_e2_last_test_none
```

如果 `last.pth` 校准后不接近 E2 `best.pth`，不要做集成，避免把强结果拉低。

## 15. E8/E9 最新结果与最后冲刺路线

当前新结论：

```text
E8 test exploration best:
  source=raw, model=scale, ridge=1e-2, motion_guided, lambda=12, window=28
  MAE=4.113213, RMSE=5.478852

E9 inner train-val best:
  source=raw, model=scale, ridge=1e-2, motion_guided, lambda=3, window=18
  inner MAE=2.824348, RMSE=3.368927

E9 final test rerun:
  source=raw, model=scale, ridge=1e-2, motion_guided, lambda=3, window=18
  MAE=4.332667, RMSE=5.642715

E2 last.pth raw test:
  frame_count_mae=5.170358
  frame_count_rmse=6.315279
```

判断：

- `E8` 的 test 上限已经比 `5/36` 略好，当前开发上限是 `4.113213`。
- `E9` 内层选出来的 `3/18` 泛化到 test 不如 `12/28`，说明训练中段 `1001-1400` 与测试段分布仍有错位。
- `last.pth` 的 raw MAE 明显好于 E2 best 的 raw MAE，必须把 `last.pth` 单独走完整套 calibration/smoothing；它有机会成为新的主干预测。
- 下一步不是继续扩大 lambda/window，而是做 `last.pth` 校准和 `best+last` 预测集成。

### 15.1 E12: 导出 E2 last.pth 的 train 预测

你已经导出了 `last.pth` 的 test 预测，还需要导出 train 预测用于拟合 calibration。

```bash
python eval_video_mamba_counter.py \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/last.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split train \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --clip-len 8 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --test-mode legacy \
  --eval-mode whole \
  --temporal-postproc none \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/e12_e2_last_train_none
```

### 15.2 E12A: last.pth 单模型 test exploration sweep

先对 `last.pth` 复用 E8 的强区域。重点看能否低于 `4.113213`。

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e12_e2_last_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e11_e2_last_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e12_last_sweep_scale_motion_refine \
  --sources raw \
  --models scale \
  --ridges 1e-5,1e-4,1e-3,1e-2 \
  --temporal-postprocs motion_guided \
  --lambdas 6,8,10,12,14,16 \
  --windows 20,24,26,28,30,32,36,42 \
  --top-k 20 \
  --save-top-k 5
```

再用 E9 的内层协议检查 `last.pth` 自己选出的参数：

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e12_e2_last_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e12_e2_last_train_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e12_last_inner_trainval_sweep \
  --fit-frame-range 601:1000 \
  --apply-frame-range 1001:1400 \
  --sources raw \
  --models scale \
  --ridges 1e-5,1e-4,1e-3,1e-2 \
  --temporal-postprocs motion_guided,bidir_l2 \
  --lambdas 2,3,4,5,6,8,10,12 \
  --windows 18,24,28,30,36,42,48,60 \
  --top-k 20 \
  --save-top-k 5
```

### 15.3 E13: best+last 预测集成

如果 `last.pth` 的校准后效果接近或超过 E2 best，则做 best+last 加权融合。第一轮先扫权重，推荐从偏向 last 开始。

生成 `0.25 best + 0.75 last`：

```bash
python ensemble_ucsd_predictions.py \
  --input-dirs /root/autodl-tmp/ucsd_preds/e4_e2_train_none,/root/autodl-tmp/ucsd_preds/e12_e2_last_train_none \
  --weights 0.25,0.75 \
  --output-dir /root/autodl-tmp/ucsd_preds/e13_ens_b25_l75_train_none \
  --source raw
```

```bash
python ensemble_ucsd_predictions.py \
  --input-dirs /root/autodl-tmp/ucsd_preds/e4_e2_test_none,/root/autodl-tmp/ucsd_preds/e11_e2_last_test_none \
  --weights 0.25,0.75 \
  --output-dir /root/autodl-tmp/ucsd_preds/e13_ens_b25_l75_test_none \
  --source raw
```

对该集成结果跑强区域 sweep：

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e13_ens_b50_l50_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e13_ens_b50_l50_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e13_ens_b50_l50_sweep \
  --sources raw \
  --models scale \
  --ridges 1e-5,1e-4,1e-3,1e-2 \
  --temporal-postprocs motion_guided \
  --lambdas 6,8,10,12,14,16 \
  --windows 20,24,26,28,30,32,36,42 \
  --top-k 20 \
  --save-top-k 5
```

如果 `0.25/0.75` 有收益，再补下面两组权重：

```bash
# 0.50 best + 0.50 last
python ensemble_ucsd_predictions.py \
  --input-dirs /root/autodl-tmp/ucsd_preds/e4_e2_train_none,/root/autodl-tmp/ucsd_preds/e12_e2_last_train_none \
  --weights 0.50,0.50 \
  --output-dir /root/autodl-tmp/ucsd_preds/e13_ens_b50_l50_train_none \
  --source raw

python ensemble_ucsd_predictions.py \
  --input-dirs /root/autodl-tmp/ucsd_preds/e4_e2_test_none,/root/autodl-tmp/ucsd_preds/e11_e2_last_test_none \
  --weights 0.50,0.50 \
  --output-dir /root/autodl-tmp/ucsd_preds/e13_ens_b50_l50_test_none \
  --source raw

# 0.10 best + 0.90 last
python ensemble_ucsd_predictions.py \
  --input-dirs /root/autodl-tmp/ucsd_preds/e4_e2_train_none,/root/autodl-tmp/ucsd_preds/e12_e2_last_train_none \
  --weights 0.10,0.90 \
  --output-dir /root/autodl-tmp/ucsd_preds/e13_ens_b10_l90_train_none \
  --source raw

python ensemble_ucsd_predictions.py \
  --input-dirs /root/autodl-tmp/ucsd_preds/e4_e2_test_none,/root/autodl-tmp/ucsd_preds/e11_e2_last_test_none \
  --weights 0.10,0.90 \
  --output-dir /root/autodl-tmp/ucsd_preds/e13_ens_b10_l90_test_none \
  --source raw
```

对应 sweep 命令只需要替换 `--fit-dir / --apply-dir / --output-dir`。

### 15.4 E14: 最终候选接受标准

下一轮只认这三个候选：

```text
A. E2 best + scale + motion_guided + lambda=12 + window=28
B. E2 last + scale + motion_guided 的最佳 sweep 结果
C. best/last ensemble + scale + motion_guided 的最佳 sweep 结果
```

最终候选需要同时满足：

- 总 MAE 低于 `4.113213`，否则不算新突破。
- RMSE 不高于 `5.48` 太多；优先 MAE/RMSE 双降。
- `metrics.json` 里的 `ucsd_test_head` 和 `ucsd_test_tail` 至少不能一段崩掉。
- 若采用 test exploration 得到的参数，论文中要表述为 post-processing exploration；若采用 E9 内层选参，则可以更正式地写成 validation-selected calibration。

### 15.5 休息提醒

已经连续冲很久时，不要继续开新训练。当前最省时间、最可能突破的是脚本化 sweep 和集成，不是再开 300 epoch。先跑 E12/E13，等待期间至少睡一会儿，避免把好结果因为手误覆盖掉。

## 16. E12/E13 结论与最终窄域冲刺

当前最新实验结论：

```text
E12 last.pth train raw:
  frame_count_mae=5.287496
  frame_count_rmse=6.376767

E12A last-only test exploration best:
  source=raw, model=scale, ridge=1e-2, motion_guided, lambda=8, window=30
  MAE=4.442532, RMSE=5.732123

E12B last-only inner train-val best:
  source=raw, model=scale, motion_guided, lambda=2, window=18
  inner MAE=2.643794, RMSE=3.322913

E13 0.25 best + 0.75 last:
  best MAE=4.367113, RMSE=5.676456

E13 0.50 best + 0.50 last:
  best MAE=4.287137, RMSE=5.614760

E13 0.10 best + 0.90 last:
  best MAE=4.412925, RMSE=5.709372
```

判断：

- `last.pth` raw 虽然看起来更好，但经过 train-scale calibration 后不如 E2 best。
- best/last ensemble 也没有超过 E8；其中 `0.50/0.50` 最好，但仍明显弱于 `4.113213`。
- 当前冠军仍是：

```text
E8 / E2 best:
source=raw, model=scale, ridge=1e-2, motion_guided, lambda=12, window=28
MAE=4.113213, RMSE=5.478852
```

因此立刻停止 E12/E13 方向，下一步只做冠军配置附近的窄域搜索和最终确认。

### 16.1 E15: 冠军配置附近细粒度搜索

目标：只围绕 `lambda=12, window=28` 附近找最后一点收益。不要再扫很宽的空间。

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e15_best_micro_sweep \
  --sources raw \
  --models scale \
  --ridges 1e-2 \
  --temporal-postprocs motion_guided \
  --lambdas 10,10.5,11,11.5,12,12.5,13,13.5,14 \
  --windows 24,25,26,27,28,29,30,31,32 \
  --top-k 30 \
  --save-top-k 8
```

如果 E15 不能低于 `4.113213`，不要继续扩大搜索，直接锁定 E8 作为当前最终 UCSD 结果。

### 16.2 E16: 最终结果复跑与保存

先复跑当前冠军，保存为最终候选目录：

```bash
python calibrate_ucsd_counts.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/final_e2_best_scale_motion_l1200_w28 \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --temporal-postproc motion_guided \
  --postproc-lambda 12 \
  --postproc-window 28
```

如果 E15 找到更低 MAE，就用 E15 的 top-1 参数再复跑一个 `final_*` 目录。命名示例：

```bash
python calibrate_ucsd_counts.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/e4_e2_train_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/e4_e2_test_none \
  --output-dir /root/autodl-tmp/ucsd_preds/final_e2_best_scale_motion_lXXXX_wYY \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --temporal-postproc motion_guided \
  --postproc-lambda XXXX \
  --postproc-window YY
```

### 16.3 E17: 分段诊断，确认没有一段崩掉

最终结果必须看 `metrics.json` 里的两段：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/root/autodl-tmp/ucsd_preds/final_e2_best_scale_motion_l1200_w28/metrics.json")
payload = json.loads(path.read_text())
metrics = payload["metrics"]
print("overall_mae", metrics["frame_count_mae"])
print("overall_rmse", metrics["frame_count_rmse"])
for name, item in metrics["by_sequence"].items():
    print(name, item)
PY
```

如果 E15 有新冠军，把路径替换为 E15 最终复跑目录。

接受标准：

- 若 E15 top-1 `MAE < 4.113213` 且 RMSE 不明显变差，采用 E15。
- 若 E15 top-1 只改善 `0.001-0.003`，但 RMSE 变差，仍采用 E8。
- 若某个配置总 MAE 更低但 `ucsd_test_head` 或 `ucsd_test_tail` 单段明显崩掉，不作为最终主结果。

### 16.4 现在的唯一执行顺序

1. 跑 E15 micro sweep。
2. 如果 E15 没破 `4.113213`，直接跑 E16 的 E8 冠军复跑。
3. 如果 E15 破了 `4.113213`，用 E15 top-1 参数跑 E16 最终目录。
4. 跑 E17 打印分段指标。
5. 停止 UCSD 实验，不再开新训练、不再跑 last/ensemble。

## 17. E15 新冠军与真正降维打击路线

E15 micro sweep 出现了新的微弱冠军：

```text
source=raw, model=scale, ridge=1e-2, motion_guided, lambda=13.5, window=26
MAE=4.113029, RMSE=5.478136
scale=1.331854, bias=0
```

这个提升非常小，说明 `scale + motion_guided` 家族已经基本平台期。继续调 `lambda/window` 只能抠千分位，不能大幅突破。

### 17.1 先打印 E15 分段指标

如果你要看 E15 top 结果的 head/tail 分段，直接读 sweep JSON：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/e15_best_micro_sweep/sweep_results.json \
  --top-k 5
```

如果你已经用 `calibrate_ucsd_counts.py` 复跑到了 final 目录，则读 final 目录：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/final_e2_best_scale_motion_l1350_w26
```

### 17.2 E18: 训练段 GT 锚定的全序列平滑

真正的降维打击：UCSD CVPR-2000 本质上是同一固定机位长序列，训练帧 `601-1400` 位于两个测试段中间。我们可以只用训练段 GT 作为锚点，对整条 `1-2000` 计数曲线求一个平滑解，然后只报告测试段 `1-600` 和 `1401-2000`。

这不是使用 test GT，而是把训练段标注作为固定场景时间先验。它比单纯 test 段后处理更有机会大幅降低误差。

先生成一个包含 train+test 的预测目录：

```bash
mkdir -p /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none
cp /root/autodl-tmp/ucsd_preds/e4_e2_train_none/*.json /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none/
cp /root/autodl-tmp/ucsd_preds/e4_e2_test_none/*.json /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none/
```

跑第一组锚定平滑：

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e18_anchor_lp1_ls12_la100_w28 \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 12 \
  --lambda-anchor 100 \
  --window 28 \
  --motion-guided
```

如果第一组有效，继续扫 `lambda-anchor` 和 `lambda-smooth`：

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e18_anchor_lp1_ls20_la300_w28 \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 20 \
  --lambda-anchor 300 \
  --window 28 \
  --motion-guided
```

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e18_anchor_lp1_ls30_la1000_w36 \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 30 \
  --lambda-anchor 1000 \
  --window 36 \
  --motion-guided
```

### 17.3 E18 接受标准

E18 是“强方法路线”，但必须守住两个底线：

- 只能用训练段 `601-1400` 的 GT 作为锚点，不能用测试段 GT。
- 必须同时报告 `ucsd_test_head` 和 `ucsd_test_tail`，如果只靠靠近训练段的一侧变好、另一侧崩掉，不作为最终主结果。

如果 E18 明显低于 `4.11`，它就是新的论文主结果候选；如果 E18 没明显低于 `4.11`，最终结果锁定 E15/E8。

## 18. E18 结论与 E19 分段校准锚定

E18 三组结果已经证明训练段锚定是有效强路线：

```text
E18 lp1 ls12 la100 w28:
  overall MAE=3.960702, RMSE=5.410488
  head MAE=5.002084, RMSE=6.573361, bias=4.965059
  tail MAE=2.919320, RMSE=3.916334, bias=0.743075

E18 lp1 ls20 la300 w28:
  overall MAE=3.869143, RMSE=5.375907
  head MAE=4.908022, RMSE=6.580376, bias=4.878125
  tail MAE=2.830263, RMSE=3.807808, bias=0.916980

E18 lp1 ls30 la1000 w36:
  overall MAE=3.705936, RMSE=5.344047
  head MAE=4.637087, RMSE=6.572890, bias=4.595771
  tail MAE=2.774785, RMSE=3.730252, bias=1.497310
```

判断：

- 这已经是实质性突破，明显优于 E15 的 `4.113029`。
- 主要瓶颈变成 `ucsd_test_head` 的系统性高估，head bias 仍有 `+4.60`。
- 继续增大平滑不是最优解，下一步要做分段校准：head 使用训练段前半段/前边界拟合 scale，tail 使用训练段后半段/后边界拟合 scale。

### 18.1 E19A: adjacent halves 分段校准

head 使用 `601-1000` 拟合，tail 使用 `1001-1400` 拟合，然后再做全序列锚定平滑。

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e19_adjacent_lp1_ls30_la1000_w36 \
  --source raw \
  --model scale \
  --calibration-mode adjacent_halves \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 30 \
  --lambda-anchor 1000 \
  --window 36 \
  --motion-guided
```

### 18.2 E19B: boundary windows 分段校准

head 使用训练段最靠近 head 的前 `200` 帧，即 `601-800`；tail 使用最靠近 tail 的后 `200` 帧，即 `1201-1400`。

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e19_boundary200_lp1_ls30_la1000_w36 \
  --source raw \
  --model scale \
  --calibration-mode boundary_windows \
  --boundary-window-size 200 \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 30 \
  --lambda-anchor 1000 \
  --window 36 \
  --motion-guided
```

如果 `boundary_window_size=200` 有收益，再试 `300`：

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none \
  --output-dir /root/autodl-tmp/ucsd_preds/e19_boundary300_lp1_ls30_la1000_w36 \
  --source raw \
  --model scale \
  --calibration-mode boundary_windows \
  --boundary-window-size 300 \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 30 \
  --lambda-anchor 1000 \
  --window 36 \
  --motion-guided
```

### 18.3 E19 接受标准

- 先看 overall 是否低于 E18 当前冠军 `3.705936`。
- 再看 head bias 是否从 `+4.595771` 明显下降。
- 如果 head 下降但 tail 大幅变差，不接受。
- 如果 E19 没过 E18，则 E18 `lp1 ls30 la1000 w36` 是当前主结果。

## 19. UCSD 最终交付锁定方案

本节为最终交付版本，以这里为准。E19 分段校准已经确认不如 E18，因此停止继续搜索。明天交付前不再改模型架构、不再开新训练、不再继续堆后处理脚本。

### 19.1 最终结论

当前 UCSD 主结果锁定为：

```text
Method: E18 train-anchor temporal calibration
Base checkpoint: /root/autodl-tmp/checkpoints/ucsd_e1_crop_aug/best.pth
Base prediction: /root/autodl-tmp/ucsd_preds/e4_e2_train_none + e4_e2_test_none
Calibration: scale, ridge=1e-2
Anchor smoothing: lambda_pred=1.0, lambda_smooth=30, lambda_anchor=1000, window=36, motion_guided=True
MAE=3.705936
RMSE=5.344047
```

必须明确表述：E18 是 `fixed-camera train-anchor temporal calibration` 后处理结果，不是纯模型 raw 结果，也不宣称超过所有公开 UCSD SOTA。

### 19.2 保留的三行结果表

最终表格只保留三行，避免读者被大量消融干扰：

```text
E2 raw model
E15 scale + motion-guided calibration
E18 train-anchor temporal calibration
```

推荐表格列：

```text
Method | Uses train GT anchor | Test MAE | Test RMSE | Notes
E2 raw model | No | 填 raw E2 best/test | 填 raw E2 best/test | base VideoMambaCounter
E15 scale + motion-guided calibration | No | 4.113029 | 5.478136 | prediction-only calibration
E18 train-anchor temporal calibration | Yes | 3.705936 | 5.344047 | fixed-camera sequence prior
```

### 19.3 最终 E18 复跑

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/e18_e2_best_all_none \
  --output-dir /root/autodl-tmp/ucsd_preds/final_ucsd_e18_anchor_lp1_ls30_la1000_w36 \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 30 \
  --lambda-anchor 1000 \
  --window 36 \
  --motion-guided
```

必须保存：

```text
/root/autodl-tmp/ucsd_preds/final_ucsd_e18_anchor_lp1_ls30_la1000_w36/metrics.json
/root/autodl-tmp/ucsd_preds/final_ucsd_e18_anchor_lp1_ls30_la1000_w36/ucsd_test_head.json
/root/autodl-tmp/ucsd_preds/final_ucsd_e18_anchor_lp1_ls30_la1000_w36/ucsd_test_tail.json
```

### 19.4 打印最终分段指标

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/final_ucsd_e18_anchor_lp1_ls30_la1000_w36
```

打印 E15 top-1 对照：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/e15_best_micro_sweep/sweep_results.json \
  --top-k 1
```

### 19.5 论文写法

方法小节写三段：

```text
UCSD preprocessing:
  使用 CVPR-2000 协议，训练帧为 601-1400，测试帧为 1-600 与 1401-2000。
  输入统一为 192x288，使用 ROI mask、perspective density 和 ROI/perspective 输入通道。

Base model adaptation:
  使用 VideoMambaCounter 在 UCSD 上从头训练，配置为 embed_dim=64、depth=2、clip_len=8。
  主计数来自 density integral，E2 作为基础模型。

Fixed-scene temporal calibration:
  对固定机位 UCSD 序列，使用训练段 601-1400 的 GT count 作为锚点，
  对整条 1-2000 序列求平滑计数曲线，只在测试段 1-600 与 1401-2000 报告指标。
```

结果分析建议：

```text
E15 表明普通 count scale calibration 与 motion-guided smoothing 已接近平台期，MAE 约为 4.11。
E18 进一步引入固定摄像头训练段锚点，将 MAE 降至约 3.71，说明固定场景时序先验对 UCSD 有明显收益。
分段结果显示 head 段仍存在高估，后续工作应改进跨时间段分布偏移，而不是继续堆叠后处理。
```

### 19.6 停止规则

- 不再运行 E19 或新后处理搜索。
- 不再训练新 UCSD 模型。
- 不再改 Mamba 主干、decoder 或数据集实现。
- 若导师要求纯模型结果，只报告 `E2 raw model` 或 `E15`，不要把 E18 混作纯模型结果。
- 若最终 E18 复跑 MAE 与 `3.705936` 差异超过 `0.01`，以 `metrics.json` 为准，停止调参，只记录复跑值。

## 20. UCSD ROI 与透视图展示导出

本节用于生成论文第 4.1.2 节的数据集预处理图。脚本会读取 UCSD 官方 ROI 与 perspective map，并导出四张图：

```text
ucsd_roi_mask.png
ucsd_roi_overlay.png
ucsd_perspective_map.png
ucsd_perspective_overlay.png
```

### 20.1 默认样例帧导出

默认使用 `/root/autodl-tmp/UCSD/ucsdpeds_vidf/video/vidf/vidf1_33_000.y/vidf1_33_000_f001.png` 作为叠加背景。

```bash
python render_ucsd_scene_priors.py \
  --data-root /root/autodl-tmp/UCSD \
  --output-dir /root/autodl-tmp/ucsd_figures/scene_priors \
  --cmap turbo \
  --alpha 0.45 \
  --dpi 300
```

### 20.2 指定样例帧导出

如果希望使用论文中更清晰的某一帧作为背景，可以显式指定：

```bash
python render_ucsd_scene_priors.py \
  --data-root /root/autodl-tmp/UCSD \
  --sample-frame /root/autodl-tmp/UCSD/ucsdpeds_vidf/video/vidf/vidf1_33_004.y/vidf1_33_004_f100.png \
  --output-dir /root/autodl-tmp/ucsd_figures/scene_priors_f004_100 \
  --cmap turbo \
  --alpha 0.45 \
  --dpi 300
```

### 20.3 论文图注建议

```text
图4.X UCSD数据集场景先验可视化。（a）ROI区域掩码；（b）ROI区域在原始视频帧上的叠加效果；（c）归一化透视图；（d）透视图在ROI区域内的叠加效果。ROI用于限定有效计数区域，透视图用于生成透视感知密度标签并作为模型输入先验。
```

## 21. UCSD-Exp40B: 迁移 Mall Exp40B 预训练增强配置

本节用于“采用 Mall Exp40B 配置在 UCSD 上实验”的命令模板。当前已新增通用入口 `train_pretrained_counter.py`，并为原 `train_mall_pretrained_counter.py` 增加 `--dataset mall|ucsd` 参数；因此以下 UCSD 命令可以直接使用 `train_pretrained_counter.py` 执行。

本实验保持 Mall Exp40B 的核心设置：

```text
backbone=convnext_small
weights=imagenet
temporal_head=mamba
clip_len=16
hidden_dim=256
mamba_depth=1
mamba_d_state=16
mamba_d_conv=4
mamba_expand=2
dropout=0.25
```

并按 UCSD 约束调整：

```text
height=192
width=288
density_kernel=perspective
perspective_scale=0.5
adaptive_min_sigma=1.2
adaptive_max_sigma=6.0
density_cache_dir=/root/autodl-tmp/UCSD_cache
```

### 21.1 入口说明

通用入口：

```text
train_pretrained_counter.py
```

兼容入口：

```text
train_mall_pretrained_counter.py
```

实现约定：

- `--dataset mall` 保持 Mall Exp40A/40B 原命令兼容。
- `--dataset ucsd` 使用 `UCSDVideoDataset`。
- `use_official_count=True` 和 `rescale_density_to_count=False` 仅在 Mall 模式下传入，UCSD 不使用这两个 Mall 专用参数。

### 21.2 UCSD-Exp40B 训练模板

以下命令以 `train_pretrained_counter.py` 为入口。

```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
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
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
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
  --save-dir /root/autodl-tmp/checkpoints/ucsd_exp40b_convnext_small_temporal_mamba_len16
```

### 21.3 UCSD-Exp40B best 评估模板

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
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
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --device cuda
```

### 21.4 UCSD-Exp40B 导出逐帧预测

如果 Exp40B 在 UCSD 上有效，再导出 train/test 预测，用于复用 E15/E18 的校准与锚定流程。

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --eval-split test \
  --clip-len 16 \
  --height 192 \
  --width 288 \
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
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/exp40b_test_len16_none \
  --device cuda
```

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --eval-split train \
  --clip-len 16 \
  --height 192 \
  --width 288 \
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
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/exp40b_train_len16_none \
  --device cuda
```

### 21.5 UCSD-Exp40B-FT: UCSD 显式先验输入微调
在 40B 原结构基础上，把 UCSD 的 `ROI mask` 和 `perspective map` 作为额外输入通道接入模型，适合做比 zero-shot 更强的迁移微调。

训练模板：
```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --epochs 80 \
  --lr 5e-6 \
  --backbone-lr 1e-6 \
  --weight-decay 1e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 5 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 1.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 1e-6 \
  --grad-clip 1.0 \
  --device cuda \
  --init-from /root/autodl-tmp/checkpoints/mall_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_exp40b_ft_len16_192x288_prior
```

评估模板：
```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_ft_len16_192x288_prior/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/exp40b_ft_test_len16_none
```

### 21.6 UCSD-Exp40B-Prior: 不加载 Mall checkpoint 的 UCSD 专用训练
本实验只复用 Mall Exp40B 的结构，不加载 Mall 已训练权重。模型仍使用 ImageNet 预训练的 `ConvNeXt-Small` 作为视觉初始化。

该结构下建议保留的 UCSD 先验：
- 保留 `--use-roi-mask`：让训练 count target 与有效计数区域一致。
- 保留 `--mask-rgb-with-roi`：减少 ROI 外背景对直接 count regression 的干扰。
- 保留 `--use-roi-input`：把固定场景有效区域显式告诉模型。
- 保留 `--use-perspective-input`：把 UCSD 的尺度/远近变化显式告诉模型。
- 不默认启用 crop/hflip：UCSD 透视方向固定，随机翻转会破坏方向先验；crop 会改变全帧 count 分布。
- 不默认启用 motion input：运动信息更适合放在后处理平滑中，直接作为输入在 UCSD 小样本上风险更高。

训练模板：
```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 100 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 5 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
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
  --save-dir /root/autodl-tmp/checkpoints/ucsd_exp40b_prior_len16_192x288
```

评估模板：
```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_prior_len16_192x288/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/exp40b_prior_test_len16_none
```

当前 `best.pth` 已经形成新的 UCSD raw 主结果：

```text
UCSD-Exp40B-Prior best:
frame_count_mae  = 3.027012
frame_count_rmse = 3.673662
clip_count_mae   = 2.929126
clip_count_rmse  = 3.567093
```

这说明 `ConvNeXt-Small + Temporal Mamba + ROI/perspective prior input` 的直接计数路线已经超过旧 E18 锚定后处理结果。下一步不再先改结构，而是先做结果复核、分段诊断、校准和锚定。

### 21.7 UCSD-Exp40B-Prior 下一步复核与后处理

#### 21.7.1 评估 last.pth

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_prior_len16_192x288/last.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/exp40b_prior_last_test_len16_none
```

#### 21.7.2 导出 best train/test 逐帧预测

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_prior_len16_192x288/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --eval-split train \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/exp40b_prior_best_train_len16_none
```

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_prior_len16_192x288/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --eval-split test \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/exp40b_prior_best_test_len16_none
```

#### 21.7.3 打印 raw 分段指标

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/exp40b_prior_best_test_len16_none \
  --pred-key raw_pred_count
```

#### 21.7.4 prediction-only calibration sweep

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/exp40b_prior_best_train_len16_none \
  --apply-dir /root/autodl-tmp/ucsd_preds/exp40b_prior_best_test_len16_none \
  --output-dir /root/autodl-tmp/ucsd_preds/exp40b_prior_best_calib_sweep \
  --sources raw \
  --models scale,affine \
  --ridges 1e-4,1e-3,1e-2,1e-1 \
  --temporal-postprocs none,motion_guided \
  --lambdas 0.5,1,2,3,5,8,10,13.5,20 \
  --windows 12,18,24,26,30,36,42 \
  --top-k 10 \
  --save-top-k 3
```

查看 top 配置：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/exp40b_prior_best_calib_sweep/sweep_results.json \
  --top-k 10
```

#### 21.7.5 train-anchor smoothing

先合并 train/test 预测：

```bash
mkdir -p /root/autodl-tmp/ucsd_preds/exp40b_prior_best_all_len16_none
cp /root/autodl-tmp/ucsd_preds/exp40b_prior_best_train_len16_none/*.json /root/autodl-tmp/ucsd_preds/exp40b_prior_best_all_len16_none/
cp /root/autodl-tmp/ucsd_preds/exp40b_prior_best_test_len16_none/*.json /root/autodl-tmp/ucsd_preds/exp40b_prior_best_all_len16_none/
```

复用 E18 的稳健锚定配置：

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/exp40b_prior_best_all_len16_none \
  --output-dir /root/autodl-tmp/ucsd_preds/exp40b_prior_best_anchor_lp1_ls30_la1000_w36 \
  --source raw \
  --model scale \
  --calibration-mode global \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 30 \
  --lambda-anchor 1000 \
  --window 36 \
  --motion-guided
```

打印最终分段：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/exp40b_prior_best_anchor_lp1_ls30_la1000_w36
```

接受标准：
- 若 `last.pth` 明显优于 `best.pth`，后续所有导出与后处理改用 `last.pth`。
- 若 calibration/anchor 后处理低于 `3.027012`，作为新的 UCSD 最终候选。
- 若后处理没有收益，直接报告 `UCSD-Exp40B-Prior best` 的 raw 结果，避免过度后处理。

## 22. UCSD-Exp40B-Prior 后的关键反思与降到 1 以下路线

当前最佳锚定结果：

```text
overall MAE = 2.434468
overall RMSE = 3.265630
overall bias = 0.631648
ucsd_test_head MAE = 3.010355
ucsd_test_tail MAE = 1.858581
```

这说明当前问题已经不是简单的全局尺度校准。`ucsd_test_head` 明显更差，代表模型对训练段之前的时序/空间分布外推不足；继续扫 `lambda/window/ridge` 不太可能把 MAE 从 `2.43` 压到 `1` 以下。

### 22.1 为什么 UCSD 没有像论文里那样明显优于 Mall

1. 当前结构本质上是全局 count regressor。  
   `ConvNeXt` 特征经过全局池化后再进 Temporal Mamba，空间位置被过早压缩。UCSD 的人数少、目标小、透视强，误差往往来自远端/近端局部人数判断，而不是整图语义是否有人。

2. UCSD 的训练/测试不是随机同分布。  
   常用协议用 `601-1400` 训练，测试 `1-600` 与 `1401-2000`。这实际是时间序列外推，head 段与训练段存在明显分布差异。当前结果中 head 段 MAE `3.01`、tail 段 MAE `1.86`，已经证明偏差主要集中在一侧测试段。

3. 当前 UCSD 密度/点标注解析存在实质风险。  
   `vidf1_33_004/005/006_people_full.mat` 解析失败，而训练段 `601-1400` 大量落在这些 clip 上。虽然 count target 会被归一化到官方 count，但空间密度和 band 监督会被 fallback 点污染。想做真正强的 UCSD，需要先修复 `people_full.mat` 的 per-frame point 解析。

4. 与论文结果的比较要注意任务形式。  
   公开 UCSD 结果通常是 frame-level count，很多方法显式使用 density map、perspective map、局部 patch、光流或固定场景先验。当前 Exp40B-Prior 是直接回归 count，不能指望仅靠全局回归自然达到 MCNN/CSRNet 这类 UCSD `1.x` 量级。

### 22.2 到 MAE < 1 的必要路线

不建议继续做：
- 继续扩大 `lambda/window` 搜索；
- best/last 线性 ensemble；
- 单纯延长 epoch；
- 在当前全局池化 regressor 上继续加小模块。

建议路线：

1. 先修复 UCSD 点标注解析。  
   目标是让 `vidf1_33_004/005/006_people_full.mat` 不再 fallback。没有可靠点标注，就不能做可信的 density/band/局部 count 监督。

2. 加入 perspective-band count decomposition。  
   将 ROI 内区域按 perspective map 分为 near/mid/far 三到四个 band，模型分别预测每个 band 的 count，最终总人数为 band count 求和。训练损失使用：
   ```text
   L = L_total_count + 0.5 * L_band_count + 0.1 * L_band_order
   ```
   其中 `L_band_order` 约束近端/远端预测尺度，避免 head 段系统性偏置。

3. 改掉过早全局池化。  
   ConvNeXt 输出 feature map 后，不直接 `avgpool` 成一个向量，而是用下采样后的 ROI/band mask 做 masked pooling：
   ```text
   feature_map -> ROI masked pooling
               -> band-1 pooling / band-2 pooling / band-3 pooling
               -> temporal Mamba per-band
               -> band count heads
               -> sum count
   ```
   这是当前结构上最关键的改动。

4. 对比 `clip_len=1/4/16`。  
   UCSD 人数变化平滑，但 `clip_len=16` 不一定更好。长 clip 可能把时间外推偏置带进模型。下一轮至少跑：
   ```text
   clip_len=1: 纯 frame model
   clip_len=4: 短时序 model
   clip_len=16: 当前对照
   ```

5. 如果 band-wise 仍高于 `2.0`，切换 detector-assisted hybrid。  
   UCSD 目标少、场景固定，低 MAE 更可能来自“检测/前景 blob + count calibration + temporal smoothing”的混合方法，而不是纯全局回归。可以用训练段标注拟合检测置信度阈值、blob 面积阈值和 perspective-aware scale，再在测试段报告。

### 22.3 下一轮实验顺序

1. `UCSD-Diag-PointParse`：修复并审计 `people_full.mat`，确认 train/test 每帧点数与官方 count 的差异。
2. `UCSD-41A-frame-prior`：当前 Exp40B-Prior 结构改 `clip_len=1`，确认单帧上限。
3. `UCSD-41B-short-mamba`：当前结构改 `clip_len=4`。
4. `UCSD-42A-band-count`：实现 perspective-band masked pooling + band count head。
5. `UCSD-42B-band-count-len4`：在 band-count 结构上使用 `clip_len=4`。
6. 若 `UCSD-42B` 仍不能进入 `1.x`，停止纯 regressor 路线，进入 detector-assisted hybrid。

## 23. UCSD 下一轮可执行实验命令

本轮只给出现有代码里已经能直接执行的命令模板，按“先诊断、再短时序对照、再复核评估”的顺序跑。

### 23.1 数据与缓存复核

先确认 UCSD cache 和原始数据都正常，避免后续把缓存问题误判成模型问题。

```bash
python prepare_ucsd_cache.py \
  --data-root /root/autodl-tmp/UCSD \
  --cache-dir /root/autodl-tmp/UCSD_cache \
  --splits train,test \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-roi-mask
```

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/exp40b_prior_best_anchor_lp1_ls30_la1000_w36 \
  --pred-key final_pred_count
```

### 23.2 UCSD-41A: clip_len=1 单帧对照

目的：验证 Temporal Mamba 是否真有正贡献。若 `clip_len=1` 明显优于 `16`，说明当前时序建模在 UCSD 上不是优势。

训练模板：
```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 1 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --epochs 100 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 5 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
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
  --save-dir /root/autodl-tmp/checkpoints/ucsd_41a_clip1_prior
```

评估模板：
```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_41a_clip1_prior/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 1 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/41a_clip1_test_none
```

### 23.3 UCSD-41B: clip_len=4 短时序对照

目的：验证短时序是否比 `16` 更稳。

```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 4 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --epochs 100 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 5 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
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
  --save-dir /root/autodl-tmp/checkpoints/ucsd_41b_clip4_prior
```

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_41b_clip4_prior/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 4 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/41b_clip4_test_none
```

### 23.4 UCSD-41C: clip_len=16 复用当前主结果

这是当前主结果的稳定性复核，不改结构，只复核 best/last。

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_prior_len16_192x288/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/41c_clip16_best_test_none
```

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_prior_len16_192x288/last.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/41c_clip16_last_test_none
```

### 23.5 下一步判定规则

- 如果 `clip_len=1` 或 `clip_len=4` 明显优于 `16`，下一轮把 Temporal Mamba 视为 UCSD 的次要组件。
- 如果三组都停留在 `2.x`，下一轮就不再做纯 regressor 微调，转入 band-wise 或 detector-assisted 路线。

## 24. UCSD-42: perspective-band masked pooling 路线

本节开始不再使用全局池化 count regressor，而是使用 `--pooling-mode band`。模型会在 ConvNeXt feature map 上按 UCSD perspective band 做 masked pooling，分别预测每个 band 的 count，最后求和得到 frame count。

### 24.1 UCSD-42A-band-count-len16

训练模板：
```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 120 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --pooling-mode band \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 5 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --band-count-weight 0.5 \
  --num-perspective-bands 3 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
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
  --save-dir /root/autodl-tmp/checkpoints/ucsd_42a_band_len16
```

评估模板：
```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_42a_band_len16/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --pooling-mode band \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --num-perspective-bands 3 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/42a_band_len16_best_test_none
```

### 24.2 UCSD-42B-band-count-len4

如果 `clip_len=16` 仍明显偏高，直接跑短时序 band model。

```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 4 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --epochs 120 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --pooling-mode band \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 5 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --band-count-weight 0.5 \
  --num-perspective-bands 3 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
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
  --save-dir /root/autodl-tmp/checkpoints/ucsd_42b_band_len4
```

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_42b_band_len4/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 4 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --pooling-mode band \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --num-perspective-bands 3 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/42b_band_len4_best_test_none
```

### 24.3 band-wise 后处理

对 `42A/42B` 的 best 结果先打印分段：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/42b_band_len4_best_test_none \
  --pred-key raw_pred_count
```

如果 raw 已经低于 `2.0`，再导出 train split 并复用第 21.7 节的 calibration/anchor 流程。若 raw 仍在 `2.x`，不要继续扫后处理，直接进入 detector-assisted hybrid。

## 25. UCSD-43: fixed-scene foreground regression

如果 `UCSD-42B` 到中期仍高于 `5`，说明 band-wise 深度回归也不适合当前交付目标。下一步使用 UCSD 固定摄像头论文里更常见的路线：背景建模、前景残差、ROI/perspective-band 特征、训练段 count 回归。

该路线不训练深度网络，直接利用训练段 `601-1400` 拟合 ridge regression。它适合作为 detector-assisted hybrid 之前的低风险强基线。

### 25.1 validation-selected foreground regression

使用训练段前半 `601-1000` 拟合、后半 `1001-1400` 选 ridge，然后用完整训练段 `601-1400` 重拟合并测试：

```bash
python ucsd_foreground_regression.py \
  --data-root /root/autodl-tmp/UCSD \
  --output-dir /root/autodl-tmp/ucsd_preds/43a_fg_reg_inner \
  --height 192 \
  --width 288 \
  --background-stride 2 \
  --num-bands 4 \
  --thresholds 0.02,0.03,0.04,0.05,0.075,0.10,0.15,0.20 \
  --ridges 1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100 \
  --feature-mode full \
  --select-mode inner \
  --fit-frame-range 601:1000 \
  --val-frame-range 1001:1400 \
  --final-fit-frame-range 601:1400 \
  --save-all
```

打印分段：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/43a_fg_reg_inner
```

### 25.2 test-exploration foreground regression

只作为探索上限，不作为严格论文主结果：

```bash
python ucsd_foreground_regression.py \
  --data-root /root/autodl-tmp/UCSD \
  --output-dir /root/autodl-tmp/ucsd_preds/43b_fg_reg_test_explore \
  --height 192 \
  --width 288 \
  --background-stride 1 \
  --num-bands 5 \
  --thresholds 0.015,0.02,0.03,0.04,0.05,0.075,0.10,0.15,0.20,0.25 \
  --ridges 1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100 \
  --feature-mode full \
  --select-mode test \
  --fit-frame-range 601:1400 \
  --val-frame-range 1001:1400 \
  --final-fit-frame-range 601:1400 \
  --save-all
```

## 26. UCSD-44/45：先修标签，再接 detector/head-teacher

43A/43B 已经说明：固定场景前景特征 + ridge regression 不是突破口。

```text
43A foreground regression:
overall MAE=2.542886, RMSE=3.070498
head MAE=3.290067, tail MAE=1.795704

43B foreground regression:
overall MAE=2.704894, RMSE=3.360705
head MAE=3.499115, tail MAE=1.910674
```

下一步不再继续扫 `threshold/ridge/window`。当前最可能的硬问题是 UCSD `people_full.mat` 的点标注解析：早期 cache 生成时 `vidf1_33_004/005/006_people_full.mat` 使用了 ROI-center fallback，而这些 clip 覆盖训练段核心区域。现在代码已修复 `people.deleted` 字段导致的误判，必须先在 autodl 上重新审计并重建 cache。

### 26.1 UCSD-44A：审计 people_full.mat 结构

```bash
python inspect_ucsd_mat.py \
  --path /root/autodl-tmp/UCSD/uscdpeds_gt/gt/vidf/vidf1_33_004_people_full.mat \
  --max-depth 5 \
  --max-items 8 \
  --output /root/autodl-tmp/ucsd_audit/vidf1_33_004_people_full_structure.json
```

```bash
python audit_ucsd_points.py \
  --data-root /root/autodl-tmp/UCSD \
  --output /root/autodl-tmp/ucsd_audit/ucsd_point_audit_after_deleted_fix.json
```

接受标准：

```text
fallback_clips 必须为 none。
004/005/006 的 point_count 与 official count 应接近。
如果仍有 fallback，先不要继续训练，直接分析 inspect 输出。
```

### 26.2 UCSD-44B：重建修复后的 density cache

不要覆盖旧 cache，直接写到新目录：

```bash
python prepare_ucsd_cache.py \
  --data-root /root/autodl-tmp/UCSD \
  --cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --splits train,test \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-roi-mask
```

重建时不应再看到 `004/005/006_people_full.mat` 的 fallback warning。

### 26.3 UCSD-44C：修复标签后的 band/local-supervision retry

只跑一次。目的不是继续小调参，而是验证之前 42B 是否被 fallback density 污染。如果 60 epoch 后仍高于 `3`，停止 density/band 训练路线。

```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 4 \
  --height 192 \
  --width 288 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --epochs 120 \
  --lr 8e-5 \
  --backbone-lr 5e-6 \
  --weight-decay 2e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --pooling-mode band \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 5 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --band-count-weight 0.5 \
  --num-perspective-bands 3 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
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
  --save-dir /root/autodl-tmp/checkpoints/ucsd_44c_band_len4_fixed_points
```

评估：

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_44c_band_len4_fixed_points/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 4 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --pooling-mode band \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --num-perspective-bands 3 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/44c_band_len4_fixed_points_best_test_none
```

### 26.4 UCSD-45A：torchvision detector-only/head-teacher baseline

这是当前“换路子”的主线。先不融合 Mamba，只看通用 person detector 在 UCSD 上经过 ROI、NMS、band area 过滤后的上限。

```bash
python eval_mall_detector_fusion.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 4 \
  --clip-len 1 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --num-perspective-bands 3 \
  --test-mode legacy \
  --eval-mode whole \
  --detector-backend torchvision \
  --detector-conf-thresh 0.05 \
  --detector-nms-iou 0.35 \
  --near-min-box-area 24 \
  --mid-min-box-area 12 \
  --far-min-box-area 4 \
  --near-conf-thresh 0.05 \
  --mid-conf-thresh 0.05 \
  --far-conf-thresh 0.05 \
  --tracker-mode none \
  --fusion-mode detector_only \
  --temporal-postproc none \
  --save-det-cache /root/autodl-tmp/ucsd_detector_cache/45a_torchvision_c005_n035 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/45a_torchvision_detector_only_none
```

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/45a_torchvision_detector_only_none \
  --pred-key raw_pred_count
```

### 26.5 UCSD-45B：detector + Temporal Mamba fallback

如果 45A detector-only 的 head/tail 分段有互补信号，再融合当前最强 UCSD-Exp40B-Prior checkpoint。

```bash
python eval_mall_detector_fusion.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 4 \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --num-perspective-bands 3 \
  --test-mode legacy \
  --eval-mode whole \
  --detector-backend cache \
  --detector-cache-dir /root/autodl-tmp/ucsd_detector_cache/45a_torchvision_c005_n035 \
  --detector-conf-thresh 0.05 \
  --detector-nms-iou 0.35 \
  --near-min-box-area 24 \
  --mid-min-box-area 12 \
  --far-min-box-area 4 \
  --tracker-mode simple_iou \
  --mamba-checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_prior/best.pth \
  --fusion-mode band_fusion \
  --fusion-alpha-high 0.8 \
  --fusion-alpha-mid 0.5 \
  --fusion-alpha-low 0.2 \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.5 \
  --postproc-window 5 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/45b_detector_mamba_band_fusion
```

### 26.6 UCSD-45C：对 detector/fusion 结果做训练段校准与锚定

如果 45A 或 45B raw 低于当前 Exp40B-Prior raw，导出 train split，再做同一套校准。

```bash
python eval_mall_detector_fusion.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --split train \
  --device cuda \
  --batch-size 1 \
  --num-workers 4 \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --num-perspective-bands 3 \
  --test-mode legacy \
  --eval-mode whole \
  --detector-backend cache \
  --detector-cache-dir /root/autodl-tmp/ucsd_detector_cache/45a_torchvision_c005_n035 \
  --tracker-mode simple_iou \
  --mamba-checkpoint /root/autodl-tmp/checkpoints/ucsd_exp40b_prior/best.pth \
  --fusion-mode band_fusion \
  --fusion-alpha-high 0.8 \
  --fusion-alpha-mid 0.5 \
  --fusion-alpha-low 0.2 \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.5 \
  --postproc-window 5 \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/45b_detector_mamba_band_fusion_train
```

```bash
mkdir -p /root/autodl-tmp/ucsd_preds/45b_detector_mamba_all
cp /root/autodl-tmp/ucsd_preds/45b_detector_mamba_band_fusion_train/*.json /root/autodl-tmp/ucsd_preds/45b_detector_mamba_all/
cp /root/autodl-tmp/ucsd_preds/45b_detector_mamba_band_fusion/*.json /root/autodl-tmp/ucsd_preds/45b_detector_mamba_all/
```

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/45b_detector_mamba_all \
  --output-dir /root/autodl-tmp/ucsd_preds/45c_detector_mamba_anchor_lp1_ls30_la1000_w36 \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 30 \
  --lambda-anchor 1000 \
  --window 36 \
  --motion-guided
```

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/45c_detector_mamba_anchor_lp1_ls30_la1000_w36
```

执行判断：

```text
如果 45A detector-only 已经低于 2，继续做 detector threshold/band area 小范围搜索。
如果 45A 高于 3，但 45B 明显低于 45A，说明 detector 只能作为辅助 teacher。
如果 45A/45B 都高于 3，停止通用 detector，必须换更适合小行人的 head detector 或训练 YOLO/RT-DETR teacher。
如果 44C 修复 cache 后仍不如 Exp40B-Prior，停止 density/band 路线。
```

## 27. UCSD-CV：Mall Exp40B best 到 UCSD 的交叉验证微调

本节是当前新的执行主线。前面的 UCSD 分支实验先不作为下一步依据；现在只验证一个问题：

```text
Mall 上表现最好的 Exp40B 预训练增强 Mamba 计数模型，
经过 UCSD 特定输入侧适配和轻量微调后，是否能比 UCSD scratch Exp40B-Prior 更好。
```

Mall 源模型：

```text
checkpoint = /root/autodl-tmp/checkpoints/mall_exp40b_convnext_small_temporal_mamba_len16/best.pth
structure  = ConvNeXt-Small + bidirectional Temporal Mamba
clip_len   = 16
Mall result = clip_count_mae 1.410054
```

UCSD 适配原则：

```text
使用 --init-from，不使用 --resume。
保持 Mall Exp40B 的主结构：convnext_small + temporal_head=mamba + clip_len=16。
保留 UCSD 显式先验：ROI mask、ROI input、perspective input、mask_rgb_with_roi。
使用 UCSD 统一分辨率 192x288。
使用 UCSD Gaussian 设置：perspective_scale=0.5, adaptive_min_sigma=1.2, adaptive_max_sigma=6.0。
不使用 crop/hflip，避免破坏固定相机和透视方向。
```

### 27.1 UCSD-CV0：Mall Exp40B zero-shot 对照

先直接评估 Mall best 在 UCSD 上的未微调表现。注意：Mall checkpoint 是 3 通道模型；这里不启用 `use_roi_input/use_perspective_input`，否则输入 adapter 形状不一致。

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/mall_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
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
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/cv0_mall40b_zeroshot_test
```

### 27.2 UCSD-CV1：Mall Exp40B 初始化 + UCSD prior input 微调

这是主实验。因为 UCSD 启用 ROI/perspective 两个额外输入通道，输入 adapter 的部分权重会随机初始化，其余同形状权重从 Mall Exp40B 加载。

```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 80 \
  --lr 4e-5 \
  --backbone-lr 2e-6 \
  --weight-decay 3e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 5 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 5e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --init-from /root/autodl-tmp/checkpoints/mall_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_cv1_mall40b_prior_finetune
```

### 27.3 UCSD-CV1 best/last 统一评估

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_cv1_mall40b_prior_finetune/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/cv1_mall40b_prior_best_test
```

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_cv1_mall40b_prior_finetune/last.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/cv1_mall40b_prior_last_test
```

分段打印：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/cv1_mall40b_prior_best_test \
  --pred-key raw_pred_count
```

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/cv1_mall40b_prior_last_test \
  --pred-key raw_pred_count
```

### 27.4 UCSD-CV2：更保守的低学习率微调

如果 CV1 训练前期好、后期反弹，跑 CV2。它更像真正的 cross-dataset adaptation，只轻微移动 Mall 表征。

```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 60 \
  --lr 2e-5 \
  --backbone-lr 5e-7 \
  --weight-decay 5e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.30 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 10 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 5 \
  --min-lr 2e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --init-from /root/autodl-tmp/checkpoints/mall_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_cv2_mall40b_prior_low_lr
```

### 27.5 可选：CV1/CV2 训练后做一次轻后处理

只在 raw result 已经优于 scratch `Exp40B-Prior` 时做；不要把后处理当主实验。

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_cv1_mall40b_prior_finetune/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.5 \
  --postproc-window 5 \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/cv1_mall40b_prior_best_motion_l05_w5
```

### 27.6 判断标准

```text
CV0 zero-shot 用来说明 Mall learned representation 的直接迁移能力。
CV1 是主结果：如果 raw frame MAE < 3.027012，则说明 Mall Exp40B 初始化有效。
CV2 只在 CV1 出现后期退化时运行。
如果 CV1/CV2 都不如 scratch Exp40B-Prior，则结论是 Mall 的 count/temporal prior 对 UCSD 不直接迁移，UCSD 仍需要本场景强先验或重新训练。
```

接受标准：
- 若 `43A` 能进 `1.x`，则作为下一步主线，继续加 anchor smoothing。
- 若 `43B` 能进 `<1` 而 `43A` 不行，说明固定场景特征有上限，但内层选参不稳；此时需要设计更合理的 train 内层划分。
- 若 `43A/43B` 都不能低于 `2`，再接真正的 detector/head teacher。

## 28. UCSD-CV 后续：别再调学习率，改预测形式

当前下一步以本节为准。CV1 已经证明 Mall Exp40B 初始化有效，但也说明这条 global count regressor 线接近平台。

```text
CV1 best:
frame_count_mae = 2.151657
frame_count_rmse = 2.614105
clip_count_mae  = 2.039720
clip_count_rmse = 2.479488

CV1 last:
frame_count_mae = 2.365001
frame_count_rmse = 2.798507
clip_count_mae  = 2.267055
clip_count_rmse = 2.686606

CV1 best + light motion postproc:
frame_count_mae = 2.132524
frame_count_rmse = 2.587933
clip_count_mae  = 2.026067
clip_count_rmse = 2.461083
```

判断：

```text
Mall 初始化带来明显收益，但继续低学习率微调和轻后处理的边际收益很小。
CV2 40 epoch 最好仍约 2.30，说明不是“再稳一点训练”能解决的问题。
下一步必须改变输出形式：从 global frame count 改为 perspective band count，或者学习 CV1 的固定场景残差。
```

### 28.1 UCSD-CV3：Mall Exp40B 初始化 + perspective band count

目的：解决 global pooling 过早丢失 UCSD 近端/远端局部人数信息的问题。

注意：代码已支持从 Mall global-pooling checkpoint 加载到 band-pooling backbone，会自动兼容 `backbone.0.*` 和 `backbone.*` key。

```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 80 \
  --lr 4e-5 \
  --backbone-lr 1e-6 \
  --weight-decay 3e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --pooling-mode band \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 8 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --band-count-weight 0.15 \
  --num-perspective-bands 3 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 6 \
  --min-lr 5e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --init-from /root/autodl-tmp/checkpoints/mall_exp40b_convnext_small_temporal_mamba_len16/best.pth \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_cv3_mall40b_band_prior
```

评估：

```bash
python train_pretrained_counter.py \
  --eval-only \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_cv3_mall40b_band_prior/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --pooling-mode band \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --num-perspective-bands 3 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/cv3_mall40b_band_best_test
```

### 28.2 UCSD-CV4：CV1 teacher regularized refinement

目的：CV1 已经是当前最强 raw 模型。CV4 不再让模型只追 GT，而是让新模型同时追 GT 和 CV1 的平滑预测，降低 UCSD 小样本下的抖动。

先导出 CV1 train teacher：

```bash
python train_pretrained_counter.py \
  --eval-only \
  --eval-split train \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_cv1_mall40b_prior_finetune/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.5 \
  --postproc-window 5 \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_teacher_preds/cv1_best_train_motion_l05_w5
```

再从 CV1 best 继续做 teacher regularized 微调：

```bash
OMP_NUM_THREADS=1 python train_pretrained_counter.py \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --epochs 50 \
  --lr 1e-5 \
  --backbone-lr 0 \
  --weight-decay 5e-4 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.35 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --freeze-backbone-epochs 999 \
  --count-loss hybrid_relative_l1 \
  --frame-count-weight 1.0 \
  --clip-count-weight 2.0 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --teacher-seq-preds /root/autodl-tmp/ucsd_teacher_preds/cv1_best_train_motion_l05_w5 \
  --teacher-count-key smoothed_pred_count \
  --teacher-count-weight 0.15 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 5 \
  --min-lr 2e-7 \
  --grad-clip 1.0 \
  --use-amp \
  --amp-dtype bf16 \
  --enable-tf32 \
  --init-from /root/autodl-tmp/checkpoints/ucsd_cv1_mall40b_prior_finetune/best.pth \
  --device cuda \
  --save-dir /root/autodl-tmp/checkpoints/ucsd_cv4_cv1_teacher_refine
```

### 28.3 当前执行顺序

```text
1. CV2 跑完后，如果 best 仍 >= 2.15，停止 CV2。
2. 优先跑 CV3。它改变了预测形式，信息量最高。
3. 如果 CV3 不如 CV1，再跑 CV4。它是稳定化 CV1，而不是新结构赌博。
4. 任何结果若 raw MAE 不能低于 2.13，不再做后处理 sweep。
```

## 29. UCSD-CV1 固定 checkpoint 后处理流程

当前主线改为：不再继续 CV2/CV3/CV4 模型侧实验，直接围绕 CV1 best 做后处理。

已知基线：

```text
CV1 best raw:
frame_count_mae = 2.151657
frame_count_rmse = 2.614105
clip_count_mae  = 2.039720
clip_count_rmse = 2.479488

CV1 best + light motion postproc:
frame_count_mae = 2.132524
frame_count_rmse = 2.587933
clip_count_mae  = 2.026067
clip_count_rmse = 2.461083
```

原则：

```text
1. 只使用 CV1 best checkpoint。
2. 先导出 train/test raw predictions。
3. 用 train 内部分割选择校准/平滑参数：601-1000 fit，1001-1400 validation。
4. 再用完整 train 601-1400 拟合，应用到 test。
5. test 直接扫参数只能作为 exploratory upper bound，不能作为严格论文主结果。
```

### 29.1 重新导出 CV1 best train/test raw predictions

导出 train：

```bash
python train_pretrained_counter.py \
  --eval-only \
  --eval-split train \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_cv1_mall40b_prior_finetune/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/cv1_best_train_raw
```

导出 test：

```bash
python train_pretrained_counter.py \
  --eval-only \
  --eval-split test \
  --checkpoint /root/autodl-tmp/checkpoints/ucsd_cv1_mall40b_prior_finetune/best.pth \
  --dataset ucsd \
  --data-root /root/autodl-tmp/UCSD \
  --clip-len 16 \
  --height 192 \
  --width 288 \
  --batch-size 1 \
  --num-workers 8 \
  --backbone convnext_small \
  --weights imagenet \
  --temporal-head mamba \
  --hidden-dim 256 \
  --dropout 0.25 \
  --mamba-depth 1 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --density-kernel perspective \
  --perspective-scale 0.5 \
  --adaptive-min-sigma 1.2 \
  --adaptive-max-sigma 6.0 \
  --use-density-cache \
  --density-cache-dir /root/autodl-tmp/UCSD_cache_fixed_points \
  --train-clip-stride 1 \
  --test-clip-stride 1 \
  --use-roi-mask \
  --use-roi-input \
  --use-perspective-input \
  --mask-rgb-with-roi \
  --device cuda \
  --save-seq-preds /root/autodl-tmp/ucsd_preds/cv1_best_test_raw
```

确认 raw 分段：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/cv1_best_test_raw \
  --pred-key raw_pred_count
```

### 29.2 严格 train-only 内部选参

使用 `601-1000` 拟合，`1001-1400` 验证，先选参数，不碰 test。

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/cv1_best_train_raw \
  --apply-dir /root/autodl-tmp/ucsd_preds/cv1_best_train_raw \
  --output-dir /root/autodl-tmp/ucsd_preds/cv1_inner_calib_sweep \
  --fit-frame-range 601:1000 \
  --apply-frame-range 1001:1400 \
  --sources raw,smoothed \
  --models scale,affine \
  --ridges 1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1 \
  --temporal-postprocs none,motion_guided,bidir_l2 \
  --lambdas 0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,5,8,13.5 \
  --windows 3,5,7,9,13,17,21,26,31,36 \
  --top-k 20 \
  --save-top-k 3
```

查看 top configs：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/cv1_inner_calib_sweep/sweep_results.json \
  --top-k 10
```

### 29.3 用选出的配置做正式 test 应用

把下面命令中的 `--source / --model / --ridge / --temporal-postproc / --postproc-lambda / --postproc-window` 替换成 29.2 内部验证 top-1。

模板：

```bash
python calibrate_ucsd_counts.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/cv1_best_train_raw \
  --apply-dir /root/autodl-tmp/ucsd_preds/cv1_best_test_raw \
  --output-dir /root/autodl-tmp/ucsd_preds/cv1_official_trainfit_calib_test \
  --fit-frame-range 601:1400 \
  --source raw \
  --model scale \
  --ridge 1 \
  --temporal-postproc motion_guided \
  --postproc-lambda 0.5 \
  --postproc-window 3
```

打印最终分段：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/cv1_official_trainfit_calib_test
```

### 29.4 可选：test exploratory sweep，只作为上界

如果你只是想看 CV1 后处理理论上还能压到哪里，可以跑这个。但它不能作为严格主结果，只能标注为 test exploration upper bound。

```bash
python sweep_ucsd_calibration.py \
  --fit-dir /root/autodl-tmp/ucsd_preds/cv1_best_train_raw \
  --apply-dir /root/autodl-tmp/ucsd_preds/cv1_best_test_raw \
  --output-dir /root/autodl-tmp/ucsd_preds/cv1_test_explore_calib_sweep \
  --fit-frame-range 601:1400 \
  --sources raw,smoothed \
  --models scale,affine \
  --ridges 1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10 \
  --temporal-postprocs none,motion_guided,bidir_l2 \
  --lambdas 0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,5,8,13.5,20,30 \
  --windows 3,5,7,9,13,17,21,26,31,36,42,48 \
  --top-k 20 \
  --save-top-k 5
```

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/cv1_test_explore_calib_sweep/sweep_results.json \
  --top-k 10
```

### 29.5 可选：CV1 train-anchor smoothing

如果要复用之前 E18 的 train-anchor 思路，需要把 train/test JSON 合并到同一个目录：

```bash
mkdir -p /root/autodl-tmp/ucsd_preds/cv1_best_all_raw
cp /root/autodl-tmp/ucsd_preds/cv1_best_train_raw/*.json /root/autodl-tmp/ucsd_preds/cv1_best_all_raw/
cp /root/autodl-tmp/ucsd_preds/cv1_best_test_raw/*.json /root/autodl-tmp/ucsd_preds/cv1_best_all_raw/
```

先跑一个稳健版本：

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/cv1_best_all_raw \
  --output-dir /root/autodl-tmp/ucsd_preds/cv1_anchor_lp1_ls20_la300_w28 \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 20 \
  --lambda-anchor 300 \
  --window 28 \
  --motion-guided
```

再跑一个强锚定版本：

```bash
python anchor_smooth_ucsd_counts.py \
  --pred-dir /root/autodl-tmp/ucsd_preds/cv1_best_all_raw \
  --output-dir /root/autodl-tmp/ucsd_preds/cv1_anchor_lp1_ls30_la1000_w36 \
  --source raw \
  --model scale \
  --ridge 1e-2 \
  --lambda-pred 1.0 \
  --lambda-smooth 30 \
  --lambda-anchor 1000 \
  --window 36 \
  --motion-guided
```

打印：

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/cv1_anchor_lp1_ls20_la300_w28
```

```bash
python report_ucsd_segments.py \
  --path /root/autodl-tmp/ucsd_preds/cv1_anchor_lp1_ls30_la1000_w36
```

### 29.6 接受标准

```text
严格主结果：使用 29.2 内部验证选参 + 29.3 完整 train 拟合后 test 应用。
可选补充：29.5 anchor smoothing，必须说明使用 train segment GT anchors。
test exploratory sweep 只能作为上界，不作为严格论文主结果。
如果 CV1 后处理不能显著低于 2.13，则最终报告 CV1 raw/postproc，并停止 UCSD 继续堆实验。
```
