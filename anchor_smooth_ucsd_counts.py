from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from calibrate_ucsd_counts import (
    compute_metrics,
    fit_calibration,
    flatten_counts,
    load_sequences,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UCSD train-anchor sequence smoothing")
    parser.add_argument("--pred-dir", type=str, required=True, help="Prediction dir containing train+test sequence JSON files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for anchored predictions and metrics")
    parser.add_argument("--source", type=str, default="raw", choices=["raw", "smoothed"], help="Prediction field to use")
    parser.add_argument("--model", type=str, default="scale", choices=["scale", "affine"], help="Calibration model")
    parser.add_argument(
        "--calibration-mode",
        type=str,
        default="global",
        choices=["global", "adjacent_halves", "boundary_windows"],
        help="How to fit/apply count calibration before anchor smoothing",
    )
    parser.add_argument(
        "--boundary-window-size",
        type=int,
        default=200,
        help="Train-frame window size for calibration_mode=boundary_windows",
    )
    parser.add_argument("--ridge", type=float, default=1e-2, help="Ridge used for global calibration")
    parser.add_argument("--lambda-pred", type=float, default=1.0, help="Penalty for staying close to model predictions")
    parser.add_argument("--lambda-smooth", type=float, default=12.0, help="Temporal smoothness strength")
    parser.add_argument("--lambda-anchor", type=float, default=100.0, help="Training GT anchor strength")
    parser.add_argument("--window", type=int, default=28, help="Temporal smoothness window")
    parser.add_argument("--motion-guided", action="store_true", help="Downweight smoothing edges with strong motion")
    parser.add_argument("--clamp-min", type=float, default=0.0, help="Minimum final count")
    return parser.parse_args()


def sequence_key(frame_index: int) -> str:
    if frame_index <= 600:
        return "ucsd_test_head"
    if frame_index >= 1401:
        return "ucsd_test_tail"
    return "ucsd_train_mid"


def merge_sequences(pred_dir: Path, source: str) -> dict[int, dict[str, float]]:
    sequences = load_sequences(pred_dir, source)
    by_frame: dict[int, dict[str, float]] = {}
    for frames in sequences.values():
        for frame in frames:
            frame_index = int(frame["frame_index"])
            by_frame[frame_index] = {
                "frame_index": float(frame_index),
                "gt_count": float(frame["gt_count"]),
                "pred_count": float(frame["pred_count"]),
                "motion_strength": float(frame.get("motion_strength", 0.0)),
            }
    if not by_frame:
        raise FileNotFoundError(f"No frames found under {pred_dir}")
    return by_frame


def fit_scale_for_mask(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    *,
    model: str,
    ridge: float,
) -> tuple[float, float]:
    if not bool(np.any(mask)):
        raise ValueError("Cannot fit calibration from an empty frame mask")
    return fit_calibration(pred[mask], gt[mask], model, ridge)


def calibrate_predictions(
    pred_raw: np.ndarray,
    gt: np.ndarray,
    frame_indices: list[int],
    train_mask: np.ndarray,
    *,
    model: str,
    ridge: float,
    calibration_mode: str,
    boundary_window_size: int,
    clamp_min: float,
) -> tuple[np.ndarray, dict[str, float]]:
    indices = np.asarray(frame_indices, dtype=np.int64)
    if calibration_mode == "global":
        scale, bias = fit_scale_for_mask(pred_raw, gt, train_mask, model=model, ridge=ridge)
        pred = pred_raw * scale + bias
        metadata = {"global_scale": scale, "global_bias": bias}
        return np.maximum(pred, float(clamp_min)), metadata

    if calibration_mode == "adjacent_halves":
        head_fit_mask = (indices >= 601) & (indices <= 1000)
        tail_fit_mask = (indices >= 1001) & (indices <= 1400)
    elif calibration_mode == "boundary_windows":
        size = max(int(boundary_window_size), 1)
        head_fit_mask = (indices >= 601) & (indices <= min(1400, 600 + size))
        tail_fit_mask = (indices >= max(601, 1401 - size)) & (indices <= 1400)
    else:
        raise ValueError(f"Unsupported calibration_mode: {calibration_mode}")

    global_scale, global_bias = fit_scale_for_mask(pred_raw, gt, train_mask, model=model, ridge=ridge)
    head_scale, head_bias = fit_scale_for_mask(pred_raw, gt, head_fit_mask, model=model, ridge=ridge)
    tail_scale, tail_bias = fit_scale_for_mask(pred_raw, gt, tail_fit_mask, model=model, ridge=ridge)

    pred = pred_raw * global_scale + global_bias
    head_apply = indices <= 600
    tail_apply = indices >= 1401
    train_head_apply = (indices >= 601) & (indices <= 1000)
    train_tail_apply = (indices >= 1001) & (indices <= 1400)
    pred[head_apply | train_head_apply] = pred_raw[head_apply | train_head_apply] * head_scale + head_bias
    pred[tail_apply | train_tail_apply] = pred_raw[tail_apply | train_tail_apply] * tail_scale + tail_bias
    metadata = {
        "global_scale": global_scale,
        "global_bias": global_bias,
        "head_scale": head_scale,
        "head_bias": head_bias,
        "tail_scale": tail_scale,
        "tail_bias": tail_bias,
    }
    return np.maximum(pred, float(clamp_min)), metadata


def build_laplacian(
    num_frames: int,
    *,
    window: int,
    motion_strength: np.ndarray,
    motion_guided: bool,
) -> np.ndarray:
    laplacian = np.zeros((num_frames, num_frames), dtype=np.float64)
    motion = motion_strength.astype(np.float64, copy=False).reshape(-1)
    if motion_guided:
        motion = motion / max(float(np.mean(motion)), 1e-6)

    for distance in range(1, max(int(window), 1) + 1):
        distance_weight = 1.0 / float(distance)
        for right in range(distance, num_frames):
            left = right - distance
            pair_weight = distance_weight
            if motion_guided:
                local_motion = float(np.mean(motion[left + 1 : right + 1]))
                pair_weight *= 1.0 / (1.0 + local_motion)
            laplacian[left, left] += pair_weight
            laplacian[right, right] += pair_weight
            laplacian[left, right] -= pair_weight
            laplacian[right, left] -= pair_weight
    return laplacian


def solve_anchored_counts(
    pred: np.ndarray,
    gt: np.ndarray,
    train_mask: np.ndarray,
    motion_strength: np.ndarray,
    *,
    lambda_pred: float,
    lambda_smooth: float,
    lambda_anchor: float,
    window: int,
    motion_guided: bool,
    clamp_min: float,
) -> np.ndarray:
    num_frames = int(pred.size)
    laplacian = build_laplacian(
        num_frames,
        window=window,
        motion_strength=motion_strength,
        motion_guided=motion_guided,
    )
    pred_weight = max(float(lambda_pred), 1e-12)
    anchor_weight = np.zeros(num_frames, dtype=np.float64)
    anchor_weight[train_mask] = max(float(lambda_anchor), 0.0)
    lhs = pred_weight * np.eye(num_frames, dtype=np.float64)
    lhs += float(lambda_smooth) * laplacian
    lhs += np.diag(anchor_weight)
    rhs = pred_weight * pred + anchor_weight * gt
    final = np.linalg.solve(lhs, rhs)
    return np.maximum(final, float(clamp_min))


def make_sequence_outputs(
    frame_indices: list[int],
    gt: np.ndarray,
    pred: np.ndarray,
    final: np.ndarray,
    motion: np.ndarray,
) -> dict[str, list[dict[str, float]]]:
    outputs: dict[str, list[dict[str, float]]] = {
        "ucsd_test_head": [],
        "ucsd_train_mid": [],
        "ucsd_test_tail": [],
    }
    for position, frame_index in enumerate(frame_indices):
        item = {
            "frame_index": int(frame_index),
            "gt_count": float(gt[position]),
            "input_pred_count": float(pred[position]),
            "calibrated_pred_count": float(pred[position]),
            "final_pred_count": float(final[position]),
            "motion_strength": float(motion[position]),
        }
        outputs[sequence_key(frame_index)].append(item)
    return {key: value for key, value in outputs.items() if value}


def subset_for_test(sequences: dict[str, list[dict[str, float]]]) -> dict[str, list[dict[str, float]]]:
    return {key: value for key, value in sequences.items() if key in {"ucsd_test_head", "ucsd_test_tail"}}


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    all_frames = merge_sequences(pred_dir, args.source)
    frame_indices = sorted(all_frames)
    pred_raw = np.asarray([all_frames[idx]["pred_count"] for idx in frame_indices], dtype=np.float64)
    gt = np.asarray([all_frames[idx]["gt_count"] for idx in frame_indices], dtype=np.float64)
    motion = np.asarray([all_frames[idx]["motion_strength"] for idx in frame_indices], dtype=np.float64)
    train_mask = np.asarray([601 <= idx <= 1400 for idx in frame_indices], dtype=bool)

    pred, calibration_metadata = calibrate_predictions(
        pred_raw,
        gt,
        frame_indices,
        train_mask,
        model=args.model,
        ridge=args.ridge,
        calibration_mode=args.calibration_mode,
        boundary_window_size=args.boundary_window_size,
        clamp_min=args.clamp_min,
    )
    final = solve_anchored_counts(
        pred,
        gt,
        train_mask,
        motion,
        lambda_pred=args.lambda_pred,
        lambda_smooth=args.lambda_smooth,
        lambda_anchor=args.lambda_anchor,
        window=args.window,
        motion_guided=args.motion_guided,
        clamp_min=args.clamp_min,
    )
    all_sequences = make_sequence_outputs(frame_indices, gt, pred, final, motion)
    test_sequences = subset_for_test(all_sequences)
    metrics = compute_metrics(test_sequences, "final_pred_count")
    metadata: dict[str, Any] = {
        "pred_dir": args.pred_dir,
        "source": args.source,
        "model": args.model,
        "calibration_mode": args.calibration_mode,
        "boundary_window_size": args.boundary_window_size,
        "ridge": args.ridge,
        "lambda_pred": args.lambda_pred,
        "lambda_smooth": args.lambda_smooth,
        "lambda_anchor": args.lambda_anchor,
        "window": args.window,
        "motion_guided": args.motion_guided,
        "clamp_min": args.clamp_min,
        "calibration": calibration_metadata,
    }
    save_outputs(Path(args.output_dir), test_sequences, metrics, metadata)
    (Path(args.output_dir) / "all_sequences.json").write_text(json.dumps(all_sequences, indent=2), encoding="utf-8")

    print("==> UCSD anchor smoothing summary")
    print(f"Pred dir: {args.pred_dir}")
    print(f"Model: {args.model}, calibration_mode={args.calibration_mode}")
    for key, value in calibration_metadata.items():
        print(f"{key}: {value:.6f}")
    print(
        f"lambda_pred={args.lambda_pred} lambda_smooth={args.lambda_smooth} "
        f"lambda_anchor={args.lambda_anchor} window={args.window} motion_guided={args.motion_guided}"
    )
    print(f"frame_count_mae: {metrics['frame_count_mae']:.6f}")
    print(f"frame_count_rmse: {metrics['frame_count_rmse']:.6f}")
    for name, item in metrics["by_sequence"].items():
        print(
            f"{name}: frames={item['num_frames']} "
            f"MAE={item['mae']:.6f} RMSE={item['rmse']:.6f} bias={item['bias']:.6f}"
        )


if __name__ == "__main__":
    main()
