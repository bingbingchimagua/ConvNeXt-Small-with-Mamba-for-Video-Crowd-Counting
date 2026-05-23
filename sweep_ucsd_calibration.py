from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from calibrate_ucsd_counts import (
    apply_calibration,
    compute_metrics,
    fit_calibration,
    flatten_counts,
    load_sequences,
    parse_frame_range,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sweep UCSD count calibration and temporal smoothing settings")
    parser.add_argument("--fit-dir", type=str, required=True, help="Directory containing train split prediction JSON files")
    parser.add_argument("--apply-dir", type=str, required=True, help="Directory containing prediction JSON files to calibrate")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for sweep CSV/JSON outputs")
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
    parser.add_argument("--sources", type=str, default="raw", help="Comma-separated sources: raw,smoothed")
    parser.add_argument("--models", type=str, default="scale,affine", help="Comma-separated models: scale,affine")
    parser.add_argument("--ridges", type=str, default="1e-4,1e-3,1e-2", help="Comma-separated ridge values")
    parser.add_argument(
        "--temporal-postprocs",
        type=str,
        default="motion_guided",
        help="Comma-separated postproc modes: none,bidir_l2,motion_guided",
    )
    parser.add_argument("--lambdas", type=str, default="1,2,3,4,5,6,8,10", help="Comma-separated smoothing strengths")
    parser.add_argument("--windows", type=str, default="18,24,30,36,42,48,60", help="Comma-separated smoothing windows")
    parser.add_argument("--clamp-min", type=float, default=0.0, help="Minimum calibrated count")
    parser.add_argument("--top-k", type=int, default=10, help="Number of best configs to print and optionally save")
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=0,
        help="Save calibrated sequence JSON files for the best K configs under output-dir/top_configs",
    )
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse"], help="Ranking metric")
    return parser.parse_args()


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in parse_str_list(value)]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_str_list(value)]


def safe_name(value: float | int | str) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return text.replace("+", "").replace("e", "e")


def result_sort_key(result: dict[str, Any], metric: str) -> tuple[float, float]:
    primary = float(result["frame_count_mae"] if metric == "mae" else result["frame_count_rmse"])
    secondary = float(result["frame_count_rmse"] if metric == "mae" else result["frame_count_mae"])
    return primary, secondary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = parse_str_list(args.sources)
    models = parse_str_list(args.models)
    ridges = parse_float_list(args.ridges)
    temporal_postprocs = parse_str_list(args.temporal_postprocs)
    lambdas = parse_float_list(args.lambdas)
    windows = parse_int_list(args.windows)
    fit_frame_range = parse_frame_range(args.fit_frame_range)
    apply_frame_range = parse_frame_range(args.apply_frame_range)

    results: list[dict[str, Any]] = []
    sequence_cache: dict[str, tuple[dict[str, list[dict[str, float]]], dict[str, list[dict[str, float]]]]] = {}

    for source in sources:
        fit_sequences = load_sequences(Path(args.fit_dir), source, fit_frame_range)
        apply_sequences = load_sequences(Path(args.apply_dir), source, apply_frame_range)
        sequence_cache[source] = (fit_sequences, apply_sequences)
        fit_pred, fit_gt = flatten_counts(fit_sequences)
        before_metrics = compute_metrics(apply_sequences, "pred_count")

        for model in models:
            for ridge in ridges:
                scale, bias = fit_calibration(fit_pred, fit_gt, model, ridge)
                for temporal_postproc in temporal_postprocs:
                    lambda_values = [0.0] if temporal_postproc == "none" else lambdas
                    window_values = [1] if temporal_postproc == "none" else windows
                    for lambda_value in lambda_values:
                        for window in window_values:
                            calibrated = apply_calibration(
                                apply_sequences,
                                scale=scale,
                                bias=bias,
                                clamp_min=args.clamp_min,
                                temporal_postproc=temporal_postproc,
                                postproc_lambda=lambda_value,
                                postproc_window=window,
                            )
                            metrics = compute_metrics(calibrated, "final_pred_count")
                            results.append(
                                {
                                    "source": source,
                                    "model": model,
                                    "ridge": ridge,
                                    "scale": scale,
                                    "bias": bias,
                                    "temporal_postproc": temporal_postproc,
                                    "postproc_lambda": lambda_value,
                                    "postproc_window": window,
                                    "before_mae": before_metrics["frame_count_mae"],
                                    "before_rmse": before_metrics["frame_count_rmse"],
                                    "frame_count_mae": metrics["frame_count_mae"],
                                    "frame_count_rmse": metrics["frame_count_rmse"],
                                    "frame_count_bias": metrics["frame_count_bias"],
                                    "num_frames": metrics["num_frames"],
                                    "by_sequence": metrics["by_sequence"],
                                }
                            )

    results.sort(key=lambda item: result_sort_key(item, args.metric))
    csv_path = output_dir / "sweep_results.csv"
    json_path = output_dir / "sweep_results.json"
    csv_fields = [
        "rank",
        "source",
        "model",
        "ridge",
        "scale",
        "bias",
        "temporal_postproc",
        "postproc_lambda",
        "postproc_window",
        "before_mae",
        "before_rmse",
        "frame_count_mae",
        "frame_count_rmse",
        "frame_count_bias",
        "num_frames",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = {key: result[key] for key in csv_fields if key != "rank"}
            row["rank"] = rank
            writer.writerow(row)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    save_count = max(0, min(int(args.save_top_k), len(results)))
    if save_count > 0:
        top_root = output_dir / "top_configs"
        top_root.mkdir(parents=True, exist_ok=True)
        for rank, result in enumerate(results[:save_count], start=1):
            source = str(result["source"])
            _, apply_sequences = sequence_cache[source]
            calibrated = apply_calibration(
                apply_sequences,
                scale=float(result["scale"]),
                bias=float(result["bias"]),
                clamp_min=args.clamp_min,
                temporal_postproc=str(result["temporal_postproc"]),
                postproc_lambda=float(result["postproc_lambda"]),
                postproc_window=int(result["postproc_window"]),
            )
            config_name = (
                f"rank{rank:02d}_{source}_{result['model']}_ridge{safe_name(result['ridge'])}_"
                f"{result['temporal_postproc']}_l{safe_name(result['postproc_lambda'])}_"
                f"w{safe_name(result['postproc_window'])}"
            )
            metadata = {
                "fit_dir": args.fit_dir,
                "apply_dir": args.apply_dir,
                "fit_frame_range": args.fit_frame_range,
                "apply_frame_range": args.apply_frame_range,
                "source": source,
                "model": result["model"],
                "scale": result["scale"],
                "bias": result["bias"],
                "ridge": result["ridge"],
                "clamp_min": args.clamp_min,
                "temporal_postproc": result["temporal_postproc"],
                "postproc_lambda": result["postproc_lambda"],
                "postproc_window": result["postproc_window"],
                "before": {
                    "frame_count_mae": result["before_mae"],
                    "frame_count_rmse": result["before_rmse"],
                },
            }
            metrics = {
                "frame_count_mae": result["frame_count_mae"],
                "frame_count_rmse": result["frame_count_rmse"],
                "frame_count_bias": result["frame_count_bias"],
                "num_frames": result["num_frames"],
                "by_sequence": result["by_sequence"],
            }
            save_outputs(top_root / config_name, calibrated, metrics, metadata)

    print("==> UCSD calibration sweep summary")
    print(f"Fit dir: {args.fit_dir}")
    print(f"Apply dir: {args.apply_dir}")
    print(f"Results CSV: {csv_path}")
    print(f"Results JSON: {json_path}")
    for rank, result in enumerate(results[: max(args.top_k, 0)], start=1):
        print(
            f"#{rank:02d} {result['source']} {result['model']} ridge={result['ridge']} "
            f"{result['temporal_postproc']} lambda={result['postproc_lambda']} "
            f"window={result['postproc_window']} "
            f"MAE={result['frame_count_mae']:.6f} RMSE={result['frame_count_rmse']:.6f} "
            f"scale={result['scale']:.6f} bias={result['bias']:.6f}"
        )


if __name__ == "__main__":
    main()
