from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset import build_video_dataset, shanghai_video_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Pretrained frame-count regressor for fixed-camera crowd counting")
    parser.add_argument("--dataset", type=str, default="mall", choices=["mall", "ucsd"], help="Dataset type")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50", "convnext_tiny", "convnext_small"])
    parser.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--temporal-head", type=str, default="gru", choices=["mlp", "gru", "mamba"])
    parser.add_argument("--pooling-mode", type=str, default="global", choices=["global", "band"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mamba-depth", type=int, default=1)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--count-loss", type=str, default="hybrid_relative_l1", choices=["l1", "smooth_l1", "relative_l1", "hybrid_relative_l1"])
    parser.add_argument("--frame-count-weight", type=float, default=1.0)
    parser.add_argument("--clip-count-weight", type=float, default=2.0)
    parser.add_argument("--band-count-weight", type=float, default=0.0)
    parser.add_argument("--num-perspective-bands", type=int, default=3)
    parser.add_argument("--density-kernel", type=str, default="perspective", choices=["fixed", "adaptive", "perspective"])
    parser.add_argument("--adaptive-k", type=int, default=3)
    parser.add_argument("--adaptive-beta", type=float, default=0.3)
    parser.add_argument("--adaptive-min-sigma", type=float, default=1.0)
    parser.add_argument("--adaptive-max-sigma", type=float, default=8.0)
    parser.add_argument("--perspective-scale", type=float, default=0.5)
    parser.add_argument("--use-density-cache", action="store_true")
    parser.add_argument("--density-cache-dir", type=str, default="")
    parser.add_argument("--train-clip-stride", type=int, default=2)
    parser.add_argument("--test-clip-stride", type=int, default=1)
    parser.add_argument("--use-roi-mask", action="store_true")
    parser.add_argument("--use-roi-input", action="store_true")
    parser.add_argument("--use-perspective-input", action="store_true")
    parser.add_argument("--mask-rgb-with-roi", action="store_true", help="Multiply RGB input by ROI before the pretrained backbone")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="/root/autodl-tmp/checkpoints/mall_exp34a_pretrained_counter")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init-from", type=str, default="", help="Load model weights only and start a fresh training run")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--temporal-postproc", type=str, default="none", choices=["none", "bidir_l2", "motion_guided"])
    parser.add_argument("--postproc-lambda", type=float, default=0.0)
    parser.add_argument("--postproc-window", type=int, default=3)
    parser.add_argument("--save-seq-preds", type=str, default="")
    parser.add_argument("--teacher-seq-preds", type=str, default="", help="Directory with teacher sequence JSON predictions")
    parser.add_argument(
        "--teacher-count-key",
        type=str,
        default="smoothed_pred_count",
        choices=["raw_pred_count", "smoothed_pred_count"],
    )
    parser.add_argument("--teacher-count-weight", type=float, default=0.0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def build_resnet_backbone(name: str, weights_mode: str, *, output_maps: bool = False) -> tuple[nn.Module, int]:
    try:
        from torchvision.models import (
            ConvNeXt_Small_Weights,
            ConvNeXt_Tiny_Weights,
            ResNet18_Weights,
            ResNet50_Weights,
            convnext_small,
            convnext_tiny,
            resnet18,
            resnet50,
        )
    except Exception as exc:  # pragma: no cover - depends on autodl environment
        raise RuntimeError("torchvision is required for train_mall_pretrained_counter.py") from exc

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if weights_mode == "imagenet" else None
        model = resnet18(weights=weights)
        feature_dim = int(model.fc.in_features)
        children = list(model.children())[:-2 if output_maps else -1]
        backbone = nn.Sequential(*children)
        return backbone, feature_dim
    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if weights_mode == "imagenet" else None
        model = resnet50(weights=weights)
        feature_dim = int(model.fc.in_features)
        children = list(model.children())[:-2 if output_maps else -1]
        backbone = nn.Sequential(*children)
        return backbone, feature_dim
    if name == "convnext_small":
        weights = ConvNeXt_Small_Weights.DEFAULT if weights_mode == "imagenet" else None
        model = convnext_small(weights=weights)
    else:
        weights = ConvNeXt_Tiny_Weights.DEFAULT if weights_mode == "imagenet" else None
        model = convnext_tiny(weights=weights)
    feature_dim = int(model.classifier[-1].in_features)
    backbone = model.features if output_maps else nn.Sequential(model.features, model.avgpool)
    return backbone, feature_dim


class PretrainedCounter(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        weights: str,
        temporal_head: str,
        pooling_mode: str,
        hidden_dim: int,
        dropout: float,
        input_channels: int = 3,
        num_count_bands: int = 3,
        mamba_depth: int = 1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ) -> None:
        super().__init__()
        self.pooling_mode = str(pooling_mode)
        if self.pooling_mode not in {"global", "band"}:
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")
        self.num_count_bands = int(num_count_bands)
        if self.num_count_bands < 1:
            raise ValueError(f"num_count_bands must be >= 1, got {self.num_count_bands}")
        self.backbone, feature_dim = build_resnet_backbone(
            backbone_name,
            weights,
            output_maps=(self.pooling_mode == "band"),
        )
        self.input_channels = int(input_channels)
        self.input_adapter: nn.Module | None = None
        if self.input_channels < 3:
            raise ValueError(f"input_channels must be >= 3, got {self.input_channels}")
        if self.input_channels > 3:
            self.input_adapter = nn.Conv2d(self.input_channels, 3, kernel_size=1, bias=True)
            adapter_conv = self.input_adapter
            with torch.no_grad():
                adapter_conv.weight.zero_()
                for channel in range(3):
                    adapter_conv.weight[channel, channel, 0, 0] = 1.0
                adapter_conv.bias.zero_()
        self.temporal_head = temporal_head
        if temporal_head == "gru":
            self.temporal = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            head_in = hidden_dim * 2
        elif temporal_head == "mamba":
            self.temporal = BidirectionalTemporalMambaHead(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                depth=mamba_depth,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=dropout,
            )
            head_in = hidden_dim
        else:
            self.temporal = None
            head_in = feature_dim
        self.count_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = bool(trainable)
        if self.input_adapter is not None:
            for param in self.input_adapter.parameters():
                param.requires_grad = True

    def forward(
        self,
        video: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
        band_masks: torch.Tensor | None = None,
        mask_rgb_with_roi: bool = False,
    ) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(f"Expected video shape (B,T,C,H,W), got {tuple(video.shape)}")
        if mask_rgb_with_roi and roi_mask is not None:
            video = video.clone()
            video[:, :, :3] = video[:, :, :3] * roi_mask.float()
        b, t, c, h, w = video.shape
        if self.input_adapter is not None:
            if c != self.input_channels:
                raise ValueError(
                    f"Expected input_channels={self.input_channels}, but got video with {c} channels"
                )
            rgb = self.input_adapter(video.reshape(b * t, c, h, w)).reshape(b, t, 3, h, w)
        else:
            rgb = video[:, :, :3].contiguous()
        b, t, c, h, w = rgb.shape
        feat_raw = self.backbone(rgb.reshape(b * t, c, h, w))
        band_count = None
        if self.pooling_mode == "band":
            if feat_raw.ndim != 4:
                raise ValueError(f"Band pooling expects feature maps, got shape={tuple(feat_raw.shape)}")
            if band_masks is None:
                raise ValueError("--pooling-mode band requires band_masks from the dataset")
            _, feature_dim, feat_h, feat_w = feat_raw.shape
            feat_map = feat_raw.reshape(b, t, feature_dim, feat_h, feat_w)
            masks = band_masks.float()
            if masks.ndim == 5:
                masks = masks.squeeze(2)
            if masks.ndim != 4:
                raise ValueError(f"Expected band_masks shape (B,K,1,H,W) or (B,K,H,W), got {tuple(band_masks.shape)}")
            masks = F.interpolate(masks, size=(feat_h, feat_w), mode="nearest").clamp(0.0, 1.0)
            if masks.shape[1] != self.num_count_bands:
                raise ValueError(
                    f"Expected {self.num_count_bands} band masks, got {masks.shape[1]}; "
                    "set --num-perspective-bands to match"
                )
            weights = masks[:, None, :, None, :, :]
            denom = weights.sum(dim=(-1, -2)).clamp_min(1e-6)
            pooled = (feat_map[:, :, None] * weights).sum(dim=(-1, -2)) / denom
            feat = pooled.permute(0, 2, 1, 3).reshape(b * self.num_count_bands, t, feature_dim)
        else:
            feat = feat_raw.flatten(1).reshape(b, t, -1)
        with autocast(enabled=False):
            feat = feat.float()
            if self.temporal_head == "gru":
                feat, _ = self.temporal(feat)
            elif self.temporal_head == "mamba":
                feat = self.temporal(feat)
            raw = self.count_head(feat).squeeze(-1)
        if self.pooling_mode == "band":
            band_count = F.softplus(raw).reshape(b, self.num_count_bands, t).permute(0, 2, 1).contiguous()
            frame_count = band_count.sum(dim=-1)
        else:
            frame_count = F.softplus(raw)
        clip_count = frame_count.mean(dim=1)
        outputs = {"frame_count": frame_count, "clip_count": clip_count}
        if band_count is not None:
            outputs["band_count"] = band_count
        return outputs


class TemporalMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:  # pragma: no cover - depends on autodl environment
            raise RuntimeError("mamba_ssm is required when --temporal-head mamba is used") from exc

        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fuse = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        forward = self.forward_mamba(x_norm)
        backward = torch.flip(self.backward_mamba(torch.flip(x_norm, dims=[1])), dims=[1])
        x = residual + self.dropout(self.fuse(torch.cat([forward, backward], dim=-1)))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class BidirectionalTemporalMambaHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("mamba_depth must be >= 1")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TemporalMambaBlock(
                    hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(feat)
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


def build_loader(args: argparse.Namespace, split: str, device: torch.device) -> DataLoader:
    dataset_kwargs: dict[str, Any] = dict(
        root=args.data_root,
        split=split,
        clip_len=args.clip_len,
        input_size=(args.height, args.width),
        density_kernel=args.density_kernel,
        adaptive_k=args.adaptive_k,
        adaptive_beta=args.adaptive_beta,
        adaptive_min_sigma=args.adaptive_min_sigma,
        adaptive_max_sigma=args.adaptive_max_sigma,
        perspective_scale=args.perspective_scale,
        use_density_cache=args.use_density_cache,
        density_cache_dir=args.density_cache_dir,
        train_mode="legacy",
        test_mode="legacy",
        train_clip_stride=args.train_clip_stride,
        test_clip_stride=args.test_clip_stride,
        use_roi_mask=args.use_roi_mask,
        use_roi_input=args.use_roi_input,
        use_perspective_input=args.use_perspective_input,
        num_perspective_bands=args.num_perspective_bands,
    )
    if args.dataset == "mall":
        dataset_kwargs.update(
            use_official_count=True,
            rescale_density_to_count=False,
        )
    dataset = build_video_dataset(
        dataset_name=args.dataset,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train" and not args.eval_only),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(split == "train" and not args.eval_only),
        collate_fn=shanghai_video_collate_fn,
    )


def count_loss(
    pred_frame: torch.Tensor,
    gt_frame: torch.Tensor,
    *,
    mode: str,
    frame_weight: float,
    clip_weight: float,
) -> torch.Tensor:
    pred_clip = pred_frame.mean(dim=1)
    gt_clip = gt_frame.mean(dim=1)
    if mode == "l1":
        return F.l1_loss(pred_frame, gt_frame)
    if mode == "smooth_l1":
        return F.smooth_l1_loss(pred_frame, gt_frame)
    frame_rel = torch.mean(torch.abs(pred_frame - gt_frame) / (gt_frame + 1.0))
    clip_rel = torch.mean(torch.abs(pred_clip - gt_clip) / (gt_clip + 1.0))
    if mode == "relative_l1":
        return clip_rel
    return float(frame_weight) * frame_rel + float(clip_weight) * clip_rel


def compute_band_frame_counts(density: torch.Tensor, band_masks: torch.Tensor) -> torch.Tensor:
    if density.ndim != 5:
        raise ValueError(f"Expected density shape (B,T,1,H,W), got {tuple(density.shape)}")
    if band_masks.ndim == 5:
        masks = band_masks.squeeze(2)
    elif band_masks.ndim == 4:
        masks = band_masks
    else:
        raise ValueError(f"Expected band_masks shape (B,K,1,H,W) or (B,K,H,W), got {tuple(band_masks.shape)}")
    if masks.shape[-2:] != density.shape[-2:]:
        masks = F.interpolate(masks.float(), size=density.shape[-2:], mode="nearest")
    masked = density[:, :, None] * masks[:, None, :, None, :, :]
    return masked.sum(dim=(-1, -2, -3)).float()


def band_count_loss(pred_band: torch.Tensor, gt_band: torch.Tensor) -> torch.Tensor:
    if pred_band.shape != gt_band.shape:
        raise ValueError(f"Band count shape mismatch: pred={tuple(pred_band.shape)} gt={tuple(gt_band.shape)}")
    rel = torch.abs(pred_band - gt_band) / (gt_band + 1.0)
    return rel.mean()


def load_teacher_count_cache(path: str, count_key: str) -> dict[str, dict[int, float]]:
    if not path:
        return {}
    cache_dir = Path(path)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Teacher sequence prediction directory not found: {cache_dir}")
    cache: dict[str, dict[int, float]] = {}
    for json_path in sorted(cache_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        sequence_id = str(payload.get("sequence_id", json_path.stem))
        frame_table: dict[int, float] = {}
        for item in payload.get("frames", []):
            if count_key not in item:
                raise KeyError(f"Missing key '{count_key}' in {json_path}")
            frame_table[int(item["frame_index"])] = float(item[count_key])
        cache[sequence_id] = frame_table
    if not cache:
        raise RuntimeError(f"No teacher prediction JSON files found in: {cache_dir}")
    return cache


def build_teacher_frame_tensor(
    batch: dict[str, Any],
    teacher_cache: dict[str, dict[int, float]],
    device: torch.device,
) -> torch.Tensor | None:
    if not teacher_cache or "frame_indices" not in batch:
        return None
    frame_indices = batch["frame_indices"]
    batch_size = len(frame_indices)
    sequence_ids = batch.get("sequence_id", ["default_train"] * batch_size)
    values: list[list[float]] = []
    for batch_idx in range(batch_size):
        sequence_id = str(sequence_ids[batch_idx])
        table = teacher_cache.get(sequence_id)
        if table is None and len(teacher_cache) == 1:
            table = next(iter(teacher_cache.values()))
        if table is None:
            raise KeyError(f"Teacher cache does not contain sequence_id={sequence_id}")
        indices = frame_indices[batch_idx].detach().cpu().tolist()
        values.append([float(table[int(frame_idx)]) for frame_idx in indices])
    return torch.tensor(values, dtype=torch.float32, device=device)


def compute_motion_strength(video: torch.Tensor) -> torch.Tensor:
    rgb = video[:, :, :3].detach().float().cpu()
    gray = 0.2989 * rgb[:, :, 0:1] + 0.5870 * rgb[:, :, 1:2] + 0.1140 * rgb[:, :, 2:3]
    b, t = gray.shape[:2]
    motion = torch.zeros((b, t), dtype=torch.float32)
    if t > 1:
        motion[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs().mean(dim=(2, 3, 4))
    return motion


def smooth_frame_sequence(
    frame_counts: torch.Tensor,
    *,
    mode: str,
    lambda_value: float,
    window: int,
    motion_strength: torch.Tensor | None = None,
) -> torch.Tensor:
    frame_counts = frame_counts.detach().cpu().double().flatten()
    num_frames = int(frame_counts.numel())
    if mode == "none" or lambda_value <= 0 or num_frames <= 1:
        return frame_counts.float()

    window = max(int(window), 1)
    laplacian = torch.zeros((num_frames, num_frames), dtype=torch.float64)
    if motion_strength is not None:
        motion_strength = motion_strength.detach().cpu().double().flatten()
        motion_strength = motion_strength / motion_strength.mean().clamp_min(1e-6)

    for distance in range(1, window + 1):
        scale = 1.0 / float(distance)
        for right in range(distance, num_frames):
            left = right - distance
            pair_weight = scale
            if mode == "motion_guided" and motion_strength is not None:
                local_motion = motion_strength[left + 1 : right + 1].mean()
                pair_weight *= 1.0 / (1.0 + float(local_motion))
            laplacian[left, left] += pair_weight
            laplacian[right, right] += pair_weight
            laplacian[left, right] -= pair_weight
            laplacian[right, left] -= pair_weight

    system = torch.eye(num_frames, dtype=torch.float64) + float(lambda_value) * laplacian
    return torch.linalg.solve(system, frame_counts).float()


def save_sequence_predictions(
    save_dir: Path,
    sequence_id: str,
    frame_indices: list[int],
    gt_counts: list[float],
    raw_counts: list[float],
    smoothed_counts: list[float],
    motion_strength: list[float],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequence_id": sequence_id,
        "num_frames": len(frame_indices),
        "frames": [
            {
                "frame_index": int(frame_idx),
                "gt_count": float(gt_count),
                "raw_pred_count": float(raw_count),
                "smoothed_pred_count": float(smoothed_count),
                "motion_strength": float(motion_value),
            }
            for frame_idx, gt_count, raw_count, smoothed_count, motion_value in zip(
                frame_indices,
                gt_counts,
                raw_counts,
                smoothed_counts,
                motion_strength,
            )
        ],
    }
    (save_dir / f"{sequence_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate(model: PretrainedCounter, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    frame_abs = 0.0
    frame_sq = 0.0
    frame_n = 0
    clip_abs = 0.0
    clip_sq = 0.0
    clip_n = 0
    sequence_frames: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    clip_records: list[dict[str, Any]] = []
    seq_pred_dir = Path(args.save_seq_preds) if args.save_seq_preds else None
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        gt_frame = batch["frame_count"].to(device, non_blocking=True)
        roi_mask = batch.get("roi_mask")
        if roi_mask is not None:
            roi_mask = roi_mask.to(device, non_blocking=True)
        band_masks = batch.get("band_masks")
        if band_masks is not None:
            band_masks = band_masks.to(device, non_blocking=True)
        outputs = model(video, roi_mask=roi_mask, band_masks=band_masks, mask_rgb_with_roi=args.mask_rgb_with_roi)
        pred_frame = outputs["frame_count"]
        pred_clip = outputs["clip_count"]
        gt_clip = gt_frame.mean(dim=1)
        frame_err = pred_frame - gt_frame
        clip_err = pred_clip - gt_clip
        frame_abs += float(frame_err.abs().sum().detach().cpu())
        frame_sq += float(frame_err.square().sum().detach().cpu())
        frame_n += int(frame_err.numel())
        clip_abs += float(clip_err.abs().sum().detach().cpu())
        clip_sq += float(clip_err.square().sum().detach().cpu())
        clip_n += int(clip_err.numel())
        if "frame_indices" in batch:
            motion = compute_motion_strength(batch["video"])
            batch_size = video.shape[0]
            for idx in range(batch_size):
                sequence_id = batch.get("sequence_id", ["default_test"] * batch_size)[idx]
                frame_indices = batch["frame_indices"][idx].detach().cpu().tolist()
                pred_counts = pred_frame[idx].detach().cpu().tolist()
                gt_counts = gt_frame[idx].detach().cpu().tolist()
                motion_values = motion[idx].tolist()
                clip_records.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_indices": [int(value) for value in frame_indices],
                        "gt_clip_count": float(gt_clip[idx].detach().cpu()),
                    }
                )
                table = sequence_frames[sequence_id]
                for frame_index, pred_value, gt_value, motion_value in zip(
                    frame_indices,
                    pred_counts,
                    gt_counts,
                    motion_values,
                ):
                    frame_index = int(frame_index)
                    if frame_index not in table:
                        table[frame_index] = {"pred_values": [], "motion_values": [], "gt_count": float(gt_value)}
                    table[frame_index]["pred_values"].append(float(pred_value))
                    table[frame_index]["motion_values"].append(float(motion_value))
    if args.temporal_postproc != "none" and clip_records:
        frame_abs = 0.0
        frame_sq = 0.0
        frame_n = 0
        clip_abs = 0.0
        clip_sq = 0.0
        clip_n = 0
        sequence_outputs: dict[str, dict[int, float]] = {}
        for sequence_id, table in sequence_frames.items():
            ordered = sorted(table)
            raw_counts = torch.tensor([float(sum(table[i]["pred_values"]) / len(table[i]["pred_values"])) for i in ordered])
            gt_counts = torch.tensor([float(table[i]["gt_count"]) for i in ordered])
            motion_counts = torch.tensor([float(sum(table[i]["motion_values"]) / len(table[i]["motion_values"])) for i in ordered])
            smoothed = smooth_frame_sequence(
                raw_counts,
                mode=args.temporal_postproc,
                lambda_value=args.postproc_lambda,
                window=args.postproc_window,
                motion_strength=motion_counts if args.temporal_postproc == "motion_guided" else None,
            )
            sequence_outputs[sequence_id] = {int(frame_idx): float(smoothed[pos]) for pos, frame_idx in enumerate(ordered)}
            frame_error = smoothed - gt_counts
            frame_abs += float(frame_error.abs().sum())
            frame_sq += float(frame_error.square().sum())
            frame_n += int(frame_error.numel())
            if seq_pred_dir is not None:
                save_sequence_predictions(
                    save_dir=seq_pred_dir,
                    sequence_id=sequence_id,
                    frame_indices=ordered,
                    gt_counts=gt_counts.tolist(),
                    raw_counts=raw_counts.tolist(),
                    smoothed_counts=smoothed.tolist(),
                    motion_strength=motion_counts.tolist(),
                )
        for record in clip_records:
            sequence_id = str(record["sequence_id"])
            pred_clip = sum(sequence_outputs[sequence_id][int(i)] for i in record["frame_indices"]) / len(record["frame_indices"])
            err = pred_clip - float(record["gt_clip_count"])
            clip_abs += abs(err)
            clip_sq += err * err
            clip_n += 1
    elif seq_pred_dir is not None and clip_records:
        for sequence_id, table in sequence_frames.items():
            ordered = sorted(table)
            raw_counts = [float(sum(table[i]["pred_values"]) / len(table[i]["pred_values"])) for i in ordered]
            gt_counts = [float(table[i]["gt_count"]) for i in ordered]
            motion_counts = [float(sum(table[i]["motion_values"]) / len(table[i]["motion_values"])) for i in ordered]
            save_sequence_predictions(
                save_dir=seq_pred_dir,
                sequence_id=sequence_id,
                frame_indices=ordered,
                gt_counts=gt_counts,
                raw_counts=raw_counts,
                smoothed_counts=raw_counts,
                motion_strength=motion_counts,
            )
    return {
        "frame_count_mae": frame_abs / max(frame_n, 1),
        "frame_count_rmse": math.sqrt(frame_sq / max(frame_n, 1)),
        "clip_count_mae": clip_abs / max(clip_n, 1),
        "clip_count_rmse": math.sqrt(clip_sq / max(clip_n, 1)),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_clip_mae: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_clip_mae": best_clip_mae,
            "args": vars(args),
        },
        str(path),
    )


def load_model_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", -1)) + 1, float(checkpoint.get("best_clip_mae", float("inf")))


def load_model_weights(path: str, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    current_state = model.state_dict()
    matched_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        candidate_keys = [key]
        if key.startswith("backbone.0."):
            candidate_keys.append("backbone." + key[len("backbone.0.") :])
        elif key.startswith("backbone."):
            candidate_keys.append("backbone.0." + key[len("backbone.") :])

        matched_key = None
        for candidate_key in candidate_keys:
            if candidate_key in current_state and current_state[candidate_key].shape == value.shape:
                matched_key = candidate_key
                break
        if matched_key is not None:
            matched_state[matched_key] = value
        else:
            skipped_keys.append(key)
    current_state.update(matched_state)
    model.load_state_dict(current_state)
    print(
        f"Loaded {len(matched_state)} tensors from init checkpoint {path}; "
        f"skipped {len(skipped_keys)} tensors with missing or mismatched shapes"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.enable_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.enable_tf32)
    model = PretrainedCounter(
        backbone_name=args.backbone,
        weights=args.weights,
        temporal_head=args.temporal_head,
        pooling_mode=args.pooling_mode,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        input_channels=3 + int(args.use_roi_input) + int(args.use_perspective_input),
        num_count_bands=args.num_perspective_bands,
        mamba_depth=args.mamba_depth,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": [p for name, p in model.named_parameters() if not name.startswith("backbone.")], "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler: Any
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    start_epoch = 0
    best_clip_mae = float("inf")
    if args.init_from:
        load_model_weights(args.init_from, model, device)
        print(f"Initialized model weights from: {args.init_from}")
    if args.resume:
        start_epoch, best_clip_mae = load_model_checkpoint(args.resume, model, optimizer, device)
    if args.checkpoint:
        start_epoch, best_clip_mae = load_model_checkpoint(args.checkpoint, model, None if args.eval_only else optimizer, device)

    val_loader = build_loader(args, split=args.eval_split if args.eval_only else "test", device=device)
    if args.eval_only:
        metrics = evaluate(model, val_loader, args, device)
        print("==> Pretrained counter evaluation")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Dataset: {args.dataset}")
        print(f"Split: {args.eval_split}")
        for key, value in metrics.items():
            print(f"{key}: {value:.6f}")
        return

    train_loader = build_loader(args, split="train", device=device)
    teacher_cache = load_teacher_count_cache(args.teacher_seq_preds, args.teacher_count_key)
    if teacher_cache:
        print(
            f"Loaded teacher count cache from {args.teacher_seq_preds} "
            f"using key={args.teacher_count_key} weight={args.teacher_count_weight}"
        )
    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda" and args.amp_dtype == "fp16")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    save_dir = Path(args.save_dir)
    accum_steps = max(int(args.grad_accum_steps), 1)
    print(
        f"Training PretrainedCounter dataset={args.dataset} backbone={args.backbone} weights={args.weights} "
        f"temporal_head={args.temporal_head} pooling={args.pooling_mode} batch={args.batch_size} accum={accum_steps}"
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.set_backbone_trainable(epoch >= args.freeze_backbone_epochs)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            video = batch["video"].to(device, non_blocking=True)
            gt_frame = batch["frame_count"].to(device, non_blocking=True)
            roi_mask = batch.get("roi_mask")
            if roi_mask is not None:
                roi_mask = roi_mask.to(device, non_blocking=True)
            band_masks = batch.get("band_masks")
            if band_masks is not None:
                band_masks = band_masks.to(device, non_blocking=True)
            with autocast(enabled=args.use_amp and device.type == "cuda", dtype=amp_dtype):
                outputs = model(video, roi_mask=roi_mask, band_masks=band_masks, mask_rgb_with_roi=args.mask_rgb_with_roi)
                loss = count_loss(
                    outputs["frame_count"],
                    gt_frame,
                    mode=args.count_loss,
                    frame_weight=args.frame_count_weight,
                    clip_weight=args.clip_count_weight,
                )
                if args.band_count_weight > 0.0:
                    if band_masks is None or "density" not in batch or "band_count" not in outputs:
                        raise RuntimeError("--band-count-weight requires band_masks, density, and model band_count output")
                    gt_band = compute_band_frame_counts(batch["density"].to(device, non_blocking=True), band_masks)
                    loss = loss + float(args.band_count_weight) * band_count_loss(outputs["band_count"], gt_band)
                if teacher_cache and args.teacher_count_weight > 0:
                    teacher_frame = build_teacher_frame_tensor(batch, teacher_cache, device)
                    if teacher_frame is not None:
                        teacher_clip = teacher_frame.mean(dim=1)
                        pred_clip = outputs["frame_count"].mean(dim=1)
                        teacher_loss = F.smooth_l1_loss(outputs["frame_count"], teacher_frame) + F.smooth_l1_loss(
                            pred_clip,
                            teacher_clip,
                        )
                        loss = loss + float(args.teacher_count_weight) * teacher_loss
            scaler.scale(loss / accum_steps).backward()
            should_step = ((step + 1) % accum_steps == 0) or (step + 1 == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu())
        metrics = evaluate(model, val_loader, args, device)
        if args.scheduler == "plateau":
            scheduler.step(metrics["clip_count_mae"])
        else:
            scheduler.step()
        is_best = metrics["clip_count_mae"] < best_clip_mae
        if is_best:
            best_clip_mae = metrics["clip_count_mae"]
        save_checkpoint(save_dir / "last.pth", model, optimizer, epoch, best_clip_mae, args)
        if is_best:
            save_checkpoint(save_dir / "best.pth", model, optimizer, epoch, best_clip_mae, args)
        print(
            f"[Epoch {epoch}] loss={running_loss / max(len(train_loader), 1):.6f} "
            f"frame_mae={metrics['frame_count_mae']:.6f} clip_mae={metrics['clip_count_mae']:.6f} "
            f"clip_rmse={metrics['clip_count_rmse']:.6f} best={best_clip_mae:.6f}"
        )


if __name__ == "__main__":
    main()
