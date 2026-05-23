from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import (
    UCSDVideoDataset,
    fallback_ucsd_points_from_roi,
    load_ucsd_counts,
    load_ucsd_points,
    load_ucsd_scene_priors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit UCSD people_full point parsing against official counts")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def split_name(global_frame: int) -> str:
    if 601 <= global_frame <= 1400:
        return "train"
    if global_frame <= 600 or global_frame >= 1401:
        return "test"
    return "ignored"


def summarize_errors(errors: list[float]) -> dict[str, float]:
    if not errors:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "num_frames": 0}
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(arr))),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "bias": float(np.mean(arr)),
        "num_frames": int(arr.size),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    gt_dir = root / "uscdpeds_gt" / "gt" / "vidf"
    roi_mask, _, _ = load_ucsd_scene_priors(gt_dir)

    rows: list[dict] = []
    fallback_clips: list[str] = []
    split_errors: dict[str, list[float]] = {"train": [], "test": [], "all": []}

    for clip_index in range(UCSDVideoDataset.NUM_CLIPS):
        clip_id = f"vidf1_33_{clip_index:03d}"
        people_path = gt_dir / f"{clip_id}_people_full.mat"
        count_path = gt_dir / f"{clip_id}_count_2K_roi_mainwalkway.mat"
        gt_counts = load_ucsd_counts(count_path, expected_frames=UCSDVideoDataset.FRAMES_PER_CLIP)
        used_fallback = False
        try:
            points_per_frame = load_ucsd_points(
                people_path,
                expected_frames=UCSDVideoDataset.FRAMES_PER_CLIP,
                gt_counts=gt_counts,
                roi_mask=roi_mask,
            )
        except ValueError as exc:
            if not args.allow_fallback:
                raise RuntimeError(f"{clip_id} failed point parsing: {exc}") from exc
            used_fallback = True
            fallback_clips.append(clip_id)
            points_per_frame = fallback_ucsd_points_from_roi(roi_mask, UCSDVideoDataset.FRAMES_PER_CLIP)

        frame_point_counts = np.asarray([len(points) for points in points_per_frame], dtype=np.float32)
        frame_errors = frame_point_counts - gt_counts.astype(np.float32)
        for local_idx, error in enumerate(frame_errors, start=1):
            global_frame = UCSDVideoDataset._global_frame_number(clip_index, local_idx)
            split = split_name(global_frame)
            if split in {"train", "test"}:
                split_errors[split].append(float(error))
                split_errors["all"].append(float(error))

        rows.append(
            {
                "clip_id": clip_id,
                "used_fallback": used_fallback,
                "mean_gt_count": float(np.mean(gt_counts)),
                "mean_point_count": float(np.mean(frame_point_counts)),
                "mae_point_vs_count": float(np.mean(np.abs(frame_errors))),
                "max_abs_point_vs_count": float(np.max(np.abs(frame_errors))),
                "num_zero_point_frames": int(np.sum(frame_point_counts == 0)),
            }
        )

    summary = {
        "data_root": str(root),
        "fallback_clips": fallback_clips,
        "by_split": {name: summarize_errors(errors) for name, errors in split_errors.items()},
        "clips": rows,
    }
    print("==> UCSD point parsing audit")
    print(f"fallback_clips: {fallback_clips if fallback_clips else 'none'}")
    for name in ("train", "test", "all"):
        item = summary["by_split"][name]
        print(
            f"{name}: frames={item['num_frames']} "
            f"MAE(point_count-count)={item['mae']:.6f} "
            f"RMSE={item['rmse']:.6f} bias={item['bias']:.6f}"
        )
    for row in rows:
        flag = " FALLBACK" if row["used_fallback"] else ""
        print(
            f"{row['clip_id']}{flag}: mean_gt={row['mean_gt_count']:.3f} "
            f"mean_points={row['mean_point_count']:.3f} "
            f"MAE={row['mae_point_vs_count']:.3f} max={row['max_abs_point_vs_count']:.3f} "
            f"zero_frames={row['num_zero_point_frames']}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
