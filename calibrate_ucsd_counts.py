from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Post-hoc count calibration for UCSD sequence predictions")
    parser.add_argument("--fit-dir", type=str, required=True, help="Directory containing train split prediction JSON files")
    parser.add_argument("--apply-dir", type=str, required=True, help="Directory containing prediction JSON files to calibrate")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for calibrated predictions and metrics")
    parser.add_argument(
        "--fit-frame-range",
        type=str,
        default="",
        help="Optional inclusive frame range for fitting, formatted as start:end",
    )
    parser.add_argument(
        "--apply-frame-range",
        type=str,
        default="",
        help="Optional inclusive frame range for application/evaluation, formatted as start:end",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="raw",
        choices=["raw", "smoothed"],
        help="Prediction field used before calibration",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="affine",
        choices=["affine", "scale"],
        help="Calibration model: affine fits y=a*x+b; scale fits y=a*x",
    )
    parser.add_argument("--ridge", type=float, default=1e-4, help="L2 regularization used when fitting calibration")
    parser.add_argument("--clamp-min", type=float, default=0.0, help="Minimum calibrated count")
    parser.add_argument(
        "--temporal-postproc",
        type=str,
        default="none",
        choices=["none", "bidir_l2", "motion_guided"],
        help="Optional smoothing applied after calibration",
    )
    parser.add_argument("--postproc-lambda", type=float, default=0.0, help="Temporal smoothing strength")
    parser.add_argument("--postproc-window", type=int, default=3, help="Temporal smoothing window")
    return parser.parse_args()


def parse_frame_range(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    if ":" not in value:
        raise ValueError(f"Frame range must be formatted as start:end, got {value!r}")
    start_text, end_text = value.split(":", 1)
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"Invalid frame range {value!r}: end must be >= start")
    return start, end


def prediction_key(source: str) -> str:
    return "raw_pred_count" if source == "raw" else "smoothed_pred_count"


def load_sequences(
    input_dir: Path,
    source: str,
    frame_range: tuple[int, int] | None = None,
) -> dict[str, list[dict[str, float]]]:
    key = prediction_key(source)
    sequences: dict[str, list[dict[str, float]]] = {}
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sequence_id = str(payload.get("sequence_id", path.stem))
        frames = []
        for frame in payload.get("frames", []):
            frame_index = int(frame["frame_index"])
            if frame_range is not None and not (frame_range[0] <= frame_index <= frame_range[1]):
                continue
            if key not in frame:
                raise KeyError(f"{path} frame is missing prediction field: {key}")
            frames.append(
                {
                    "frame_index": frame_index,
                    "gt_count": float(frame["gt_count"]),
                    "pred_count": float(frame[key]),
                    "motion_strength": float(frame.get("motion_strength", 0.0)),
                }
            )
        frames.sort(key=lambda item: item["frame_index"])
        if frames:
            sequences[sequence_id] = frames
    if not sequences:
        range_text = "" if frame_range is None else f" in frame range {frame_range[0]}:{frame_range[1]}"
        raise FileNotFoundError(f"No prediction frames found under {input_dir}{range_text}")
    return sequences


def flatten_counts(sequences: dict[str, list[dict[str, float]]]) -> tuple[np.ndarray, np.ndarray]:
    pred = []
    gt = []
    for frames in sequences.values():
        for frame in frames:
            pred.append(float(frame["pred_count"]))
            gt.append(float(frame["gt_count"]))
    return np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)


def fit_calibration(pred: np.ndarray, gt: np.ndarray, model: str, ridge: float) -> tuple[float, float]:
    if pred.size == 0:
        raise ValueError("Cannot fit calibration from empty predictions")
    if model == "scale":
        denom = float(np.dot(pred, pred) + ridge)
        scale = float(np.dot(pred, gt) / max(denom, 1e-12))
        return scale, 0.0

    design = np.stack([pred, np.ones_like(pred)], axis=1)
    reg = np.diag([float(ridge), float(ridge)])
    lhs = design.T @ design + reg
    rhs = design.T @ gt
    scale, bias = np.linalg.solve(lhs, rhs)
    return float(scale), float(bias)


def smooth_sequence(
    counts: np.ndarray,
    *,
    mode: str,
    lambda_value: float,
    window: int,
    motion_strength: np.ndarray,
) -> np.ndarray:
    counts = counts.astype(np.float64, copy=False).reshape(-1)
    num_frames = int(counts.size)
    if mode == "none" or lambda_value <= 0.0 or num_frames <= 1:
        return counts.copy()

    window = max(int(window), 1)
    laplacian = np.zeros((num_frames, num_frames), dtype=np.float64)
    motion = motion_strength.astype(np.float64, copy=False).reshape(-1)
    if mode == "motion_guided":
        motion = motion / max(float(np.mean(motion)), 1e-6)

    for distance in range(1, window + 1):
        distance_weight = 1.0 / float(distance)
        for right in range(distance, num_frames):
            left = right - distance
            pair_weight = distance_weight
            if mode == "motion_guided":
                local_motion = float(np.mean(motion[left + 1 : right + 1]))
                pair_weight *= 1.0 / (1.0 + local_motion)
            laplacian[left, left] += pair_weight
            laplacian[right, right] += pair_weight
            laplacian[left, right] -= pair_weight
            laplacian[right, left] -= pair_weight

    system = np.eye(num_frames, dtype=np.float64) + float(lambda_value) * laplacian
    return np.linalg.solve(system, counts)


def compute_metrics(sequences: dict[str, list[dict[str, float]]], pred_key: str) -> dict[str, Any]:
    all_errors = []
    by_sequence = {}
    for sequence_id, frames in sequences.items():
        errors = np.asarray([float(frame[pred_key]) - float(frame["gt_count"]) for frame in frames], dtype=np.float64)
        mae = float(np.mean(np.abs(errors))) if errors.size else 0.0
        rmse = float(math.sqrt(float(np.mean(errors * errors)))) if errors.size else 0.0
        by_sequence[sequence_id] = {
            "num_frames": int(errors.size),
            "mae": mae,
            "rmse": rmse,
            "bias": float(np.mean(errors)) if errors.size else 0.0,
        }
        all_errors.extend(errors.tolist())

    errors_np = np.asarray(all_errors, dtype=np.float64)
    return {
        "frame_count_mae": float(np.mean(np.abs(errors_np))) if errors_np.size else 0.0,
        "frame_count_rmse": float(math.sqrt(float(np.mean(errors_np * errors_np)))) if errors_np.size else 0.0,
        "frame_count_bias": float(np.mean(errors_np)) if errors_np.size else 0.0,
        "num_frames": int(errors_np.size),
        "by_sequence": by_sequence,
    }


def apply_calibration(
    sequences: dict[str, list[dict[str, float]]],
    *,
    scale: float,
    bias: float,
    clamp_min: float,
    temporal_postproc: str,
    postproc_lambda: float,
    postproc_window: int,
) -> dict[str, list[dict[str, float]]]:
    calibrated: dict[str, list[dict[str, float]]] = {}
    for sequence_id, frames in sequences.items():
        raw = np.asarray([float(frame["pred_count"]) for frame in frames], dtype=np.float64)
        motion = np.asarray([float(frame["motion_strength"]) for frame in frames], dtype=np.float64)
        calibrated_counts = np.maximum(raw * float(scale) + float(bias), float(clamp_min))
        final_counts = smooth_sequence(
            calibrated_counts,
            mode=temporal_postproc,
            lambda_value=postproc_lambda,
            window=postproc_window,
            motion_strength=motion,
        )
        final_counts = np.maximum(final_counts, float(clamp_min))
        calibrated[sequence_id] = []
        for frame, calibrated_count, final_count in zip(frames, calibrated_counts, final_counts):
            calibrated[sequence_id].append(
                {
                    "frame_index": int(frame["frame_index"]),
                    "gt_count": float(frame["gt_count"]),
                    "input_pred_count": float(frame["pred_count"]),
                    "calibrated_pred_count": float(calibrated_count),
                    "final_pred_count": float(final_count),
                    "motion_strength": float(frame["motion_strength"]),
                }
            )
    return calibrated


def save_outputs(
    output_dir: Path,
    sequences: dict[str, list[dict[str, float]]],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sequence_id, frames in sequences.items():
        payload = {
            "sequence_id": sequence_id,
            "num_frames": len(frames),
            "frames": frames,
        }
        (output_dir / f"{sequence_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"metadata": metadata, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    fit_frame_range = parse_frame_range(args.fit_frame_range)
    apply_frame_range = parse_frame_range(args.apply_frame_range)
    fit_sequences = load_sequences(Path(args.fit_dir), args.source, fit_frame_range)
    apply_sequences = load_sequences(Path(args.apply_dir), args.source, apply_frame_range)
    fit_pred, fit_gt = flatten_counts(fit_sequences)
    scale, bias = fit_calibration(fit_pred, fit_gt, args.model, args.ridge)

    before_metrics = compute_metrics(apply_sequences, "pred_count")
    calibrated_sequences = apply_calibration(
        apply_sequences,
        scale=scale,
        bias=bias,
        clamp_min=args.clamp_min,
        temporal_postproc=args.temporal_postproc,
        postproc_lambda=args.postproc_lambda,
        postproc_window=args.postproc_window,
    )
    after_metrics = compute_metrics(calibrated_sequences, "final_pred_count")
    metadata = {
        "fit_dir": args.fit_dir,
        "apply_dir": args.apply_dir,
        "fit_frame_range": args.fit_frame_range,
        "apply_frame_range": args.apply_frame_range,
        "source": args.source,
        "model": args.model,
        "scale": scale,
        "bias": bias,
        "ridge": args.ridge,
        "clamp_min": args.clamp_min,
        "temporal_postproc": args.temporal_postproc,
        "postproc_lambda": args.postproc_lambda,
        "postproc_window": args.postproc_window,
        "before": before_metrics,
    }
    save_outputs(Path(args.output_dir), calibrated_sequences, after_metrics, metadata)

    print("==> UCSD count calibration summary")
    print(f"Fit dir: {args.fit_dir}")
    print(f"Apply dir: {args.apply_dir}")
    print(f"Model: {args.model}, scale={scale:.6f}, bias={bias:.6f}")
    print(f"Before MAE/RMSE: {before_metrics['frame_count_mae']:.6f} / {before_metrics['frame_count_rmse']:.6f}")
    print(f"After MAE/RMSE: {after_metrics['frame_count_mae']:.6f} / {after_metrics['frame_count_rmse']:.6f}")
    print(f"Saved calibrated predictions to: {args.output_dir}")


if __name__ == "__main__":
    main()
