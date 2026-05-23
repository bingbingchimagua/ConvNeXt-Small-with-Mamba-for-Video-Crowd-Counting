from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from dataset import (
    UCSDVideoDataset,
    density_cache_variant_name,
    generate_density_map,
    load_ucsd_counts,
    load_ucsd_points,
    load_ucsd_scene_priors,
    fallback_ucsd_points_from_roi,
    scale_density_to_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare cached UCSD density maps")
    parser.add_argument("--data-root", type=str, required=True, help="UCSD root directory")
    parser.add_argument("--splits", type=str, default="train,test", help="Comma-separated splits to cache")
    parser.add_argument("--density-kernel", type=str, default="perspective", choices=["fixed", "adaptive", "perspective"])
    parser.add_argument("--sigma", type=float, default=4.0, help="Sigma used when density-kernel=fixed")
    parser.add_argument("--adaptive-k", type=int, default=3)
    parser.add_argument("--adaptive-beta", type=float, default=0.3)
    parser.add_argument("--adaptive-min-sigma", type=float, default=0.5)
    parser.add_argument("--adaptive-max-sigma", type=float, default=6.0)
    parser.add_argument("--perspective-scale", type=float, default=0.5)
    parser.add_argument("--use-roi-mask", action="store_true", help="Apply ROI mask to cached density maps")
    parser.add_argument("--cache-dir", type=str, required=True, help="Directory to store cached density maps")
    parser.add_argument(
        "--count-tolerance",
        type=float,
        default=1e-3,
        help="Maximum allowed density-sum/count mismatch after cache normalization",
    )
    return parser.parse_args()


def build_cache_dir(args: argparse.Namespace, split: str) -> Path:
    variant = density_cache_variant_name(
        density_kernel=args.density_kernel,
        adaptive_k=args.adaptive_k,
        adaptive_beta=args.adaptive_beta,
        adaptive_min_sigma=args.adaptive_min_sigma,
        adaptive_max_sigma=args.adaptive_max_sigma,
        perspective_scale=args.perspective_scale,
    )
    return Path(args.cache_dir) / "ucsd" / split / variant


def list_ucsd_samples(data_root: Path, split: str, use_roi_mask: bool) -> list[dict]:
    video_root = data_root / "ucsdpeds_vidf" / "video" / "vidf"
    gt_dir = data_root / "uscdpeds_gt" / "gt" / "vidf"
    if not video_root.exists():
        raise FileNotFoundError(f"UCSD video directory not found: {video_root}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"UCSD ground-truth directory not found: {gt_dir}")

    roi_mask, perspective_map, _ = load_ucsd_scene_priors(gt_dir)
    samples: list[dict] = []
    for clip_index in range(UCSDVideoDataset.NUM_CLIPS):
        clip_id = f"vidf1_33_{clip_index:03d}"
        image_dir = video_root / f"{clip_id}.y"
        people_path = gt_dir / f"{clip_id}_people_full.mat"
        count_path = gt_dir / f"{clip_id}_count_2K_roi_mainwalkway.mat"
        if not image_dir.exists():
            raise FileNotFoundError(f"UCSD clip image directory not found: {image_dir}")
        if not people_path.exists():
            raise FileNotFoundError(f"UCSD people annotation file not found: {people_path}")
        if not count_path.exists():
            raise FileNotFoundError(f"UCSD count annotation file not found: {count_path}")

        gt_counts = load_ucsd_counts(count_path, expected_frames=UCSDVideoDataset.FRAMES_PER_CLIP)
        try:
            points_per_frame = load_ucsd_points(
                people_path,
                expected_frames=UCSDVideoDataset.FRAMES_PER_CLIP,
                gt_counts=gt_counts,
                roi_mask=None,
            )
        except ValueError as exc:
            print(f"[UCSD Cache] warning: {exc}; using ROI-center fallback points for density labels.")
            points_per_frame = fallback_ucsd_points_from_roi(roi_mask, UCSDVideoDataset.FRAMES_PER_CLIP)
        image_paths = sorted(image_dir.glob("*.png"))
        image_by_local = {UCSDVideoDataset._local_frame_number(path.stem): path for path in image_paths}
        for local_frame in range(1, UCSDVideoDataset.FRAMES_PER_CLIP + 1):
            global_frame = UCSDVideoDataset._global_frame_number(clip_index, local_frame)
            if not UCSDVideoDataset.is_split_global_frame(split, global_frame):
                continue
            image_path = image_by_local.get(local_frame)
            if image_path is None:
                raise FileNotFoundError(f"Missing UCSD frame {clip_id}_f{local_frame:03d}.png under {image_dir}")
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise RuntimeError(f"Invalid UCSD image file: {image_path}") from exc
            samples.append(
                {
                    "image_path": image_path,
                    "frame_stem": f"{clip_id}_f{local_frame:03d}",
                    "global_frame": global_frame,
                    "points": points_per_frame[local_frame - 1],
                    "gt_count": float(gt_counts[local_frame - 1]),
                    "roi_mask": roi_mask,
                    "perspective_map": perspective_map,
                }
            )
    return samples


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    for split in splits:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        samples = list_ucsd_samples(data_root=data_root, split=split, use_roi_mask=args.use_roi_mask)
        cache_dir = build_cache_dir(args, split)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[UCSD Cache] split={split} samples={len(samples)} -> {cache_dir}")

        for idx, sample in enumerate(samples, start=1):
            with Image.open(sample["image_path"]) as image:
                width, height = image.size

            density = generate_density_map(
                points=sample["points"],
                height=height,
                width=width,
                sigma=args.sigma,
                density_kernel=args.density_kernel,
                adaptive_k=args.adaptive_k,
                adaptive_beta=args.adaptive_beta,
                adaptive_min_sigma=args.adaptive_min_sigma,
                adaptive_max_sigma=args.adaptive_max_sigma,
                perspective_map=sample["perspective_map"],
                perspective_scale=args.perspective_scale,
            )
            if args.use_roi_mask:
                density = density * np.asarray(sample["roi_mask"], dtype=np.float32)
            density = scale_density_to_count(density, float(sample["gt_count"]))
            count_error = abs(float(density.sum()) - float(sample["gt_count"]))
            if count_error > args.count_tolerance:
                raise RuntimeError(
                    f"Density/count mismatch for {sample['frame_stem']}: "
                    f"density_sum={float(density.sum()):.6f} gt={float(sample['gt_count']):.6f}"
                )

            save_path = cache_dir / f"{sample['frame_stem']}.npy"
            np.save(str(save_path), density.astype(np.float32, copy=False))
            if idx == 1 or idx % 100 == 0 or idx == len(samples):
                print(
                    f"[UCSD Cache] split={split} progress={idx}/{len(samples)} "
                    f"global_frame={sample['global_frame']} file={save_path.name}"
                )


if __name__ == "__main__":
    main()
