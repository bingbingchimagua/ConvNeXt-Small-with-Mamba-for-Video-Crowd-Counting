from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_PAD_FILL = tuple(int(round(value * 255.0)) for value in IMAGENET_MEAN)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    buffer = bytearray(image.tobytes())
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).view(height, width, 3).clone()
    return tensor.permute(2, 0, 1).contiguous().float().div(255.0)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image - mean) / std


def density_numpy_to_tensor(density: np.ndarray) -> torch.Tensor:
    height, width = density.shape
    density_bytes = bytearray(density.astype(np.float32, copy=False).tobytes())
    tensor = torch.frombuffer(density_bytes, dtype=torch.float32).view(height, width).clone()
    return tensor.unsqueeze(0)


def load_density_npy_as_tensor(path: Path) -> torch.Tensor:
    density = np.load(str(path))
    if density.ndim != 2:
        raise ValueError(f"Cached density map must be 2D, but got shape={density.shape} from {path}")
    return density_numpy_to_tensor(density.astype(np.float32, copy=False))


def density_cache_variant_name(
    density_kernel: str,
    adaptive_k: int,
    adaptive_beta: float,
    adaptive_min_sigma: float,
    adaptive_max_sigma: float,
    perspective_scale: float = 0.5,
) -> str:
    if density_kernel == "adaptive":
        beta_str = f"{adaptive_beta:g}".replace(".", "p")
        min_str = f"{adaptive_min_sigma:g}".replace(".", "p")
        max_str = f"{adaptive_max_sigma:g}".replace(".", "p")
        return f"adaptive_k{adaptive_k}_b{beta_str}_min{min_str}_max{max_str}"
    if density_kernel == "perspective":
        scale_str = f"{perspective_scale:g}".replace(".", "p")
        min_str = f"{adaptive_min_sigma:g}".replace(".", "p")
        max_str = f"{adaptive_max_sigma:g}".replace(".", "p")
        return f"perspective_s{scale_str}_min{min_str}_max{max_str}"
    return "fixed"


def load_shanghaitech_points(gt_path: Path) -> np.ndarray:
    mat = loadmat(str(gt_path))
    points = mat["image_info"][0, 0][0, 0][0]
    return np.asarray(points, dtype=np.float32)


def _unwrap_singleton_array(value: object) -> object:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def _ensure_points_array(points: np.ndarray | list) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return points.reshape(-1, 2).astype(np.float32, copy=False)


def load_points_from_txt(annotation_path: Path) -> np.ndarray:
    text = annotation_path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 2), dtype=np.float32)
    rows: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) >= 2:
            rows.append([float(parts[0]), float(parts[1])])
    return _ensure_points_array(rows)


def _load_points_from_via_dict(data: dict) -> np.ndarray:
    points: list[list[float]] = []
    for item in data.values():
        if not isinstance(item, dict):
            continue
        regions = item.get("regions", [])
        if isinstance(regions, dict):
            region_iterable = regions.values()
        else:
            region_iterable = regions
        for region in region_iterable:
            if not isinstance(region, dict):
                continue
            shape_attributes = region.get("shape_attributes", {})
            if not isinstance(shape_attributes, dict):
                continue
            has_rect_fields = all(key in shape_attributes for key in ("x", "y", "width", "height"))
            if not has_rect_fields:
                continue
            shape_name = shape_attributes.get("name")
            if shape_name not in (None, "", "rect"):
                continue
            x = float(shape_attributes["x"])
            y = float(shape_attributes["y"])
            width = float(shape_attributes["width"])
            height = float(shape_attributes["height"])
            points.append([x + width / 2.0, y + height / 2.0])
    return _ensure_points_array(points)


def load_points_from_json(annotation_path: Path) -> np.ndarray:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "points" in data:
            points = data.get("points", [])
        else:
            return _load_points_from_via_dict(data)
    elif isinstance(data, list):
        points = data
    else:
        raise ValueError(f"Unsupported JSON annotation structure in {annotation_path}")
    return _ensure_points_array(points)


def load_points_from_mat(annotation_path: Path) -> np.ndarray:
    data = loadmat(str(annotation_path))
    if "points" in data:
        return _ensure_points_array(data["points"])
    for key, value in data.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[-1] == 2:
            return _ensure_points_array(array)
        if array.ndim >= 2 and array.size > 0:
            flat = np.asarray(array, dtype=np.float32).reshape(-1, 2)
            if flat.shape[-1] == 2:
                return _ensure_points_array(flat)
    raise ValueError(f"Could not find point annotations in mat file: {annotation_path}")


def load_fdst_points(annotation_path: Path) -> np.ndarray:
    suffix = annotation_path.suffix.lower()
    if suffix == ".txt":
        return load_points_from_txt(annotation_path)
    if suffix == ".json":
        return load_points_from_json(annotation_path)
    if suffix == ".mat":
        return load_points_from_mat(annotation_path)
    raise ValueError(f"Unsupported FDST annotation format: {annotation_path}")


def load_mall_ground_truth(gt_path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    data = loadmat(str(gt_path))
    if "frame" not in data or "count" not in data:
        raise ValueError(f"Mall ground truth file must contain 'frame' and 'count': {gt_path}")

    frame_entries = np.asarray(data["frame"], dtype=object).reshape(-1)
    counts = np.asarray(data["count"], dtype=np.float32).reshape(-1)
    if len(frame_entries) != len(counts):
        raise ValueError(
            f"Mall ground truth frame/count length mismatch in {gt_path}: "
            f"len(frame)={len(frame_entries)} len(count)={len(counts)}"
        )

    points_per_frame: list[np.ndarray] = []
    for frame_entry in frame_entries:
        value = _unwrap_singleton_array(frame_entry)
        if isinstance(value, np.void):
            if value.dtype.names is None or "loc" not in value.dtype.names:
                raise ValueError(f"Unsupported Mall frame entry structure in {gt_path}")
            value = value["loc"]
        elif isinstance(value, np.ndarray) and value.dtype.names and "loc" in value.dtype.names:
            value = value["loc"]
        value = _unwrap_singleton_array(value)
        points_per_frame.append(_ensure_points_array(value))
    return points_per_frame, counts.astype(np.float32, copy=False)


def load_mall_scene_priors(prior_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(str(prior_path))
    if "roi" not in data or "pMapN" not in data:
        raise ValueError(f"Mall scene prior file must contain 'roi' and 'pMapN': {prior_path}")

    roi = np.asarray(data["roi"])
    roi_value = _unwrap_singleton_array(roi)
    if isinstance(roi_value, np.void):
        if roi_value.dtype.names is None or "mask" not in roi_value.dtype.names:
            raise ValueError(f"Unsupported Mall roi structure in {prior_path}")
        roi_value = roi_value["mask"]
    elif isinstance(roi_value, np.ndarray) and roi_value.dtype.names and "mask" in roi_value.dtype.names:
        roi_value = roi_value["mask"]
    roi_mask = np.asarray(_unwrap_singleton_array(roi_value), dtype=np.float32)
    if roi_mask.ndim != 2:
        raise ValueError(f"Mall roi mask must be 2D, but got shape={roi_mask.shape} from {prior_path}")
    roi_mask = np.clip(roi_mask, 0.0, 1.0).astype(np.float32, copy=False)

    perspective_map = np.asarray(data["pMapN"], dtype=np.float32)
    if perspective_map.ndim != 2:
        raise ValueError(
            f"Mall perspective map must be 2D, but got shape={perspective_map.shape} from {prior_path}"
        )
    p_min = float(perspective_map.min())
    p_max = float(perspective_map.max())
    normalized_perspective = (perspective_map - p_min) / max(p_max - p_min, 1e-6)
    return roi_mask, perspective_map.astype(np.float32, copy=False), normalized_perspective.astype(np.float32, copy=False)


def load_bus_frame_info(annotation_path: Path) -> tuple[np.ndarray, float]:
    data = loadmat(str(annotation_path))
    if "image_info" not in data:
        raise ValueError(f"Bus annotation file must contain 'image_info': {annotation_path}")

    info = _unwrap_singleton_array(data["image_info"])
    field_names = _mat_struct_field_names(info)
    if "number" not in field_names or "location" not in field_names:
        raise ValueError(f"Bus image_info must contain 'number' and 'location': {annotation_path}")

    count_value = _unwrap_singleton_array(_mat_struct_get_field(info, "number"))
    points_value = _unwrap_singleton_array(_mat_struct_get_field(info, "location"))
    points = _ensure_points_array(points_value)
    count = float(np.asarray(count_value, dtype=np.float32).reshape(-1)[0])
    return points, count


def load_bus_scene_priors(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_path = Path(root)
    roi_path = root_path / "bus_roi.npy"
    if not roi_path.exists():
        raise FileNotFoundError(f"Bus ROI mask not found: {roi_path}")

    roi_mask = np.asarray(np.load(str(roi_path)), dtype=np.float32)
    if roi_mask.ndim != 2:
        raise ValueError(f"Bus ROI mask must be 2D, but got shape={roi_mask.shape} from {roi_path}")
    roi_mask = np.clip(roi_mask, 0.0, 1.0).astype(np.float32, copy=False)

    prior_path = root_path / "bus_perspective_density_prior.npy"
    if prior_path.exists():
        perspective_map = np.asarray(np.load(str(prior_path)), dtype=np.float32)
    else:
        train_image_dir = root_path / "train" / "images"
        train_density_dir = root_path / "train" / "amb_gt"
        image_paths = sorted(train_image_dir.glob("*.jpg"), key=lambda path: _parse_bus_frame_number(path.stem))
        if not image_paths:
            raise FileNotFoundError(
                f"Bus perspective prior not found at {prior_path}, and no train images found under {train_image_dir}"
            )
        density_sum = None
        for image_path in image_paths:
            density_path = train_density_dir / f"{image_path.stem}.npy"
            if not density_path.exists():
                raise FileNotFoundError(f"Bus density map not found for train image {image_path.name}: {density_path}")
            density = np.asarray(np.load(str(density_path)), dtype=np.float32)
            if density.shape != roi_mask.shape:
                raise ValueError(
                    f"Bus density prior source shape mismatch: {density_path} has {density.shape}, "
                    f"expected {roi_mask.shape}"
                )
            density_sum = density if density_sum is None else density_sum + density
        perspective_map = density_sum / max(float(len(image_paths)), 1.0)

    if perspective_map.shape != roi_mask.shape:
        raise ValueError(
            f"Bus perspective prior shape must match ROI: prior={perspective_map.shape}, roi={roi_mask.shape}"
        )
    perspective_map = np.asarray(perspective_map, dtype=np.float32) * roi_mask
    roi_values = perspective_map[roi_mask > 0.5]
    if roi_values.size > 0:
        p_min = float(roi_values.min())
        p_max = float(roi_values.max())
        normalized = np.zeros_like(perspective_map, dtype=np.float32)
        normalized[roi_mask > 0.5] = (roi_values - p_min) / max(p_max - p_min, 1e-6)
    else:
        normalized = np.zeros_like(perspective_map, dtype=np.float32)
    return roi_mask, perspective_map.astype(np.float32, copy=False), normalized.astype(np.float32, copy=False)


def _parse_bus_frame_number(frame_stem: str) -> int:
    try:
        return int(frame_stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported Bus frame name: {frame_stem}") from exc


def _mat_struct_field_names(value: object) -> list[str]:
    field_names = getattr(value, "_fieldnames", None)
    if field_names is not None:
        return list(field_names)
    dtype_names = getattr(getattr(value, "dtype", None), "names", None)
    if dtype_names is not None:
        return list(dtype_names)
    return []


def _mat_struct_get_field(value: object, field_name: str) -> object:
    if hasattr(value, field_name):
        return getattr(value, field_name)
    return value[field_name]


def _load_numeric_mat_variable(mat_path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    for key in preferred_keys:
        if key in data:
            return np.asarray(data[key], dtype=np.float32)
    for key, value in data.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            return array.astype(np.float32, copy=False)
    raise ValueError(f"Could not find a numeric variable in mat file: {mat_path}")


def _collect_numeric_arrays(value: object, arrays: list[np.ndarray]) -> None:
    value = _unwrap_singleton_array(value)
    field_names = _mat_struct_field_names(value)
    if field_names:
        for field_name in field_names:
            _collect_numeric_arrays(_mat_struct_get_field(value, field_name), arrays)
        return
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.reshape(-1):
                _collect_numeric_arrays(item, arrays)
            return
        if np.issubdtype(value.dtype, np.number):
            arrays.append(np.asarray(value, dtype=np.float32))


def _pick_2d_numeric_array(value: object, expected_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    arrays: list[np.ndarray] = []
    _collect_numeric_arrays(value, arrays)
    arrays = [array for array in arrays if array.ndim == 2]
    if not arrays:
        return None
    if expected_shape is not None:
        matching = [array for array in arrays if array.shape == expected_shape]
        if matching:
            return max(matching, key=lambda array: array.size)
    return max(arrays, key=lambda array: array.size)


def _normalize_map(map_array: np.ndarray) -> np.ndarray:
    values = np.asarray(map_array, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values, dtype=np.float32)
    finite_values = values[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    normalized = (values - min_value) / max(max_value - min_value, 1e-6)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return normalized.astype(np.float32, copy=False)


def load_ucsd_scene_priors(gt_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roi_path = gt_dir / "vidf1_33_roi_mainwalkway.mat"
    perspective_path = gt_dir / "vidf1_33_dmap3.mat"
    if not roi_path.exists():
        raise FileNotFoundError(f"UCSD ROI file not found: {roi_path}")
    if not perspective_path.exists():
        raise FileNotFoundError(f"UCSD perspective file not found: {perspective_path}")

    roi_data = loadmat(str(roi_path), squeeze_me=True, struct_as_record=False)
    if "roi" not in roi_data:
        raise ValueError(f"UCSD ROI file must contain 'roi': {roi_path}")
    roi_value = _unwrap_singleton_array(roi_data["roi"])
    if _mat_struct_field_names(roi_value) and "mask" in _mat_struct_field_names(roi_value):
        roi_value = _mat_struct_get_field(roi_value, "mask")
    roi_mask = np.asarray(_unwrap_singleton_array(roi_value), dtype=np.float32)
    if roi_mask.ndim != 2:
        raise ValueError(f"UCSD ROI mask must be 2D, got shape={roi_mask.shape} from {roi_path}")
    roi_mask = np.clip(roi_mask, 0.0, 1.0).astype(np.float32, copy=False)

    perspective_data = loadmat(str(perspective_path), squeeze_me=True, struct_as_record=False)
    perspective = None
    for key in ("dmap", "pmapxy", "pmapy", "pmapx"):
        if key in perspective_data:
            candidate = _pick_2d_numeric_array(perspective_data[key], expected_shape=roi_mask.shape)
            if candidate is not None:
                perspective = candidate
                break
    if perspective is None:
        for key, value in perspective_data.items():
            if key.startswith("__"):
                continue
            candidate = _pick_2d_numeric_array(value, expected_shape=roi_mask.shape)
            if candidate is not None:
                perspective = candidate
                break
    if perspective is None:
        raise ValueError(f"Could not find a 2D UCSD perspective map in {perspective_path}")
    if perspective.shape != roi_mask.shape:
        raise ValueError(
            f"UCSD ROI and perspective shapes must match, got roi={roi_mask.shape} perspective={perspective.shape}"
        )
    normalized_perspective = _normalize_map(perspective)
    return roi_mask, perspective.astype(np.float32, copy=False), normalized_perspective


def load_ucsd_counts(count_path: Path, expected_frames: int = 200) -> np.ndarray:
    data = loadmat(str(count_path), squeeze_me=True, struct_as_record=False)
    def pick_count_array(value: object) -> np.ndarray | None:
        arrays: list[np.ndarray] = []
        _collect_numeric_arrays(value, arrays)
        candidates = [np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays if array.size == expected_frames]
        if not candidates:
            return None
        if len(candidates) >= 3:
            summed = candidates[0] + candidates[1]
            if np.allclose(candidates[2], summed, atol=1e-4, rtol=0.0):
                return candidates[2]
        for idx, candidate in enumerate(candidates):
            others = [other for other_idx, other in enumerate(candidates) if other_idx != idx]
            if len(others) >= 2:
                summed = np.zeros_like(candidate)
                for other in others:
                    summed += other
                if np.allclose(candidate, summed, atol=1e-4, rtol=0.0):
                    return candidate
        if len(candidates) >= 2:
            return candidates[0] + candidates[1]
        return candidates[0]

    values = None
    for key in ("count", "cgt"):
        if key in data:
            candidate = pick_count_array(data[key])
            if candidate is not None:
                values = candidate
                break
    if values is None:
        for key, value in data.items():
            if key.startswith("__"):
                continue
            candidate = pick_count_array(value)
            if candidate is not None:
                values = candidate
                break
    if values is None:
        raise ValueError(f"Could not find a numeric count array in mat file: {count_path}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        counts = values.reshape(1)
    elif values.ndim == 1:
        counts = values.reshape(-1)
    elif values.ndim == 2 and values.shape[0] == expected_frames:
        counts = values.sum(axis=1) if values.shape[1] > 1 else values.reshape(-1)
    elif values.ndim == 2 and values.shape[1] == expected_frames:
        counts = values.sum(axis=0) if values.shape[0] > 1 else values.reshape(-1)
    elif values.ndim == 2 and 2 in values.shape:
        counts = values.sum(axis=0 if values.shape[0] == 2 else 1).reshape(-1)
    else:
        counts = values.reshape(-1)
    counts = counts.astype(np.float32, copy=False)
    if counts.size != expected_frames:
        raise ValueError(
            f"UCSD count file must contain {expected_frames} frame counts, got {counts.size} from {count_path}"
        )
    return counts


def _numeric_points_from_array(array: np.ndarray) -> np.ndarray | None:
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return np.zeros((0, 2), dtype=np.float32) if array.size == 0 else None
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        if array.size == 2:
            return array.reshape(1, 2)
        if array.size == 4:
            return np.array([[(array[0] + array[2]) * 0.5, (array[1] + array[3]) * 0.5]], dtype=np.float32)
        return None
    if array.ndim == 2:
        if array.shape[-1] >= 2:
            points = array[:, :2]
            if array.shape[-1] >= 4:
                width_like = np.nanmedian(array[:, 2])
                height_like = np.nanmedian(array[:, 3])
                if width_like > 0 and height_like > 0:
                    points = np.stack([array[:, 0] + array[:, 2] * 0.5, array[:, 1] + array[:, 3] * 0.5], axis=1)
            return _ensure_points_array(points)
        if array.shape[0] >= 2:
            transposed = array[:2, :].T
            return _ensure_points_array(transposed)
    return None


def _collect_points_from_mat_value(value: object, candidates: list[np.ndarray]) -> None:
    value = _unwrap_singleton_array(value)
    field_names = _mat_struct_field_names(value)
    if field_names:
        lower_to_name = {field.lower(): field for field in field_names}
        x_key = next((lower_to_name[key] for key in ("x", "cx", "col", "column") if key in lower_to_name), None)
        y_key = next((lower_to_name[key] for key in ("y", "cy", "row") if key in lower_to_name), None)
        if x_key is not None and y_key is not None:
            xs = np.asarray(_mat_struct_get_field(value, x_key), dtype=np.float32).reshape(-1)
            ys = np.asarray(_mat_struct_get_field(value, y_key), dtype=np.float32).reshape(-1)
            if xs.size == ys.size:
                candidates.append(np.stack([xs, ys], axis=1).astype(np.float32, copy=False))
        for field_name in field_names:
            _collect_points_from_mat_value(_mat_struct_get_field(value, field_name), candidates)
        return

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.reshape(-1):
                _collect_points_from_mat_value(item, candidates)
            return
        points = _numeric_points_from_array(value)
        if points is not None:
            candidates.append(points)
        return


def _pick_point_candidate(candidates: list[np.ndarray], target_count: float | None = None) -> np.ndarray:
    clean_candidates = [_ensure_points_array(candidate) for candidate in candidates if candidate is not None]
    clean_candidates = [candidate for candidate in clean_candidates if candidate.ndim == 2 and candidate.shape[1] == 2]
    if not clean_candidates:
        return np.zeros((0, 2), dtype=np.float32)
    if target_count is not None:
        target = max(float(target_count), 0.0)
        return min(clean_candidates, key=lambda candidate: abs(float(len(candidate)) - target))
    return max(clean_candidates, key=lambda candidate: len(candidate))


def _extract_ucsd_points_sequence(
    value: object,
    *,
    expected_frames: int,
    gt_counts: np.ndarray | None,
) -> list[np.ndarray] | None:
    value = _unwrap_singleton_array(value)
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == expected_frames:
        points_per_frame = []
        for frame_idx, item in enumerate(value.reshape(-1)):
            candidates: list[np.ndarray] = []
            _collect_points_from_mat_value(item, candidates)
            target_count = None if gt_counts is None else float(gt_counts[frame_idx])
            points_per_frame.append(_pick_point_candidate(candidates, target_count=target_count))
        return points_per_frame
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == expected_frames:
        points_per_frame = []
        for frame_idx in range(expected_frames):
            candidates = []
            _collect_points_from_mat_value(value[frame_idx], candidates)
            target_count = None if gt_counts is None else float(gt_counts[frame_idx])
            points_per_frame.append(_pick_point_candidate(candidates, target_count=target_count))
        return points_per_frame
    return None


def load_ucsd_points(
    people_path: Path,
    *,
    expected_frames: int = 200,
    gt_counts: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    data = loadmat(str(people_path), squeeze_me=True, struct_as_record=False)
    if "people" in data:
        track_points = _load_ucsd_people_tracks(data["people"], expected_frames=expected_frames)
        if track_points is not None:
            return _filter_ucsd_points_with_roi(track_points, roi_mask)
    for key in ("people", "frame", "frames", "loc", "points"):
        if key in data:
            points_per_frame = _extract_ucsd_points_sequence(
                data[key],
                expected_frames=expected_frames,
                gt_counts=gt_counts,
            )
            if points_per_frame is not None:
                return _filter_ucsd_points_with_roi(points_per_frame, roi_mask)

    for key, value in data.items():
        if key.startswith("__"):
            continue
        points_per_frame = _extract_ucsd_points_sequence(value, expected_frames=expected_frames, gt_counts=gt_counts)
        if points_per_frame is not None:
            return _filter_ucsd_points_with_roi(points_per_frame, roi_mask)
    raise ValueError(f"Could not parse UCSD per-frame point annotations from {people_path}")


def _load_ucsd_people_tracks(value: object, *, expected_frames: int) -> list[np.ndarray] | None:
    value = _unwrap_singleton_array(value)
    if not isinstance(value, np.ndarray):
        return None
    people = value.reshape(-1)
    if people.size == expected_frames:
        return None

    def collect_tracks(*, include_deleted: bool) -> tuple[list[list[list[float]]], bool]:
        frame_points: list[list[list[float]]] = [[] for _ in range(expected_frames)]
        found_track = False
        for person in people:
            person = _unwrap_singleton_array(person)
            field_names = _mat_struct_field_names(person)
            if "loc" not in field_names:
                continue
            deleted = False
            if "deleted" in field_names:
                deleted_value = np.asarray(_mat_struct_get_field(person, "deleted")).reshape(-1)
                if deleted_value.size == 1:
                    deleted = bool(deleted_value[0])
            if deleted and not include_deleted:
                continue
            loc = np.asarray(_mat_struct_get_field(person, "loc"), dtype=np.float32)
            if loc.ndim != 2 or loc.shape[1] < 3:
                continue
            found_track = True
            frame_numbers = np.rint(loc[:, 2]).astype(np.int64)
            for row, frame_number in zip(loc, frame_numbers):
                if 1 <= frame_number <= expected_frames:
                    frame_points[frame_number - 1].append([float(row[0]), float(row[1])])
        return frame_points, found_track

    frame_points, found_track = collect_tracks(include_deleted=False)
    if not found_track:
        frame_points, found_track = collect_tracks(include_deleted=True)
    if not found_track:
        return None
    return [_ensure_points_array(points) for points in frame_points]


def _filter_ucsd_points_with_roi(points_per_frame: list[np.ndarray], roi_mask: np.ndarray | None) -> list[np.ndarray]:
    if roi_mask is None:
        return points_per_frame
    height, width = roi_mask.shape
    filtered: list[np.ndarray] = []
    for points in points_per_frame:
        points = _ensure_points_array(points)
        if points.size == 0:
            filtered.append(points)
            continue
        xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
        keep = roi_mask[ys, xs] > 0.5
        filtered.append(points[keep].astype(np.float32, copy=False))
    return filtered


def fallback_ucsd_points_from_roi(roi_mask: np.ndarray, expected_frames: int) -> list[np.ndarray]:
    ys, xs = np.where(np.asarray(roi_mask) > 0.5)
    if xs.size == 0:
        point = np.zeros((1, 2), dtype=np.float32)
    else:
        point = np.asarray([[float(np.median(xs)), float(np.median(ys))]], dtype=np.float32)
    return [point.copy() for _ in range(expected_frames)]


def scale_density_to_count(density: np.ndarray, target_count: float) -> np.ndarray:
    density = np.asarray(density, dtype=np.float32)
    target = max(float(target_count), 0.0)
    current = float(density.sum())
    if target == 0.0:
        return np.zeros_like(density, dtype=np.float32)
    if current <= 1e-8:
        raise ValueError(f"Cannot scale an empty density map to positive target count {target:.4f}")
    return (density * (target / current)).astype(np.float32, copy=False)


def load_background_image(background_cache_dir: str | Path, split: str) -> np.ndarray:
    cache_root = Path(background_cache_dir)
    split_root = cache_root / "mall" / split
    candidates = [
        split_root / "background.npy",
        split_root / "background.png",
        split_root / "background.jpg",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".npy":
            background = np.load(str(path))
            if background.ndim == 2:
                background = np.repeat(background[..., None], 3, axis=2)
            if background.ndim != 3 or background.shape[2] != 3:
                raise ValueError(f"Background cache must be HxWx3 or HxW, got shape={background.shape} from {path}")
            if background.dtype != np.float32:
                background = background.astype(np.float32, copy=False)
            if background.max() > 1.0:
                background = background / 255.0
            return np.clip(background, 0.0, 1.0).astype(np.float32, copy=False)
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    raise FileNotFoundError(
        f"Background cache not found under {split_root}. Expected one of: "
        f"{', '.join(path.name for path in candidates)}"
    )


def build_foreground_maps(rgb: torch.Tensor, background_rgb: torch.Tensor, roi_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb.shape[-2:] != background_rgb.shape[-2:]:
        raise ValueError(
            f"Foreground map size mismatch between frame={tuple(rgb.shape)} and background={tuple(background_rgb.shape)}"
        )
    rgb = rgb.float().clamp(0.0, 1.0)
    background_rgb = background_rgb.float().clamp(0.0, 1.0)
    roi_mask = roi_mask.float().clamp(0.0, 1.0)
    frame_gray = 0.2989 * rgb[0:1] + 0.5870 * rgb[1:2] + 0.1140 * rgb[2:3]
    background_gray = 0.2989 * background_rgb[0:1] + 0.5870 * background_rgb[1:2] + 0.1140 * background_rgb[2:3]
    residual = torch.abs(frame_gray - background_gray)
    residual = residual * roi_mask
    conf_max = residual.amax(dim=(-2, -1), keepdim=True)
    confidence = residual / conf_max.clamp_min(1e-6)
    confidence = confidence * roi_mask
    return residual.contiguous(), confidence.contiguous()


def build_perspective_band_masks(
    perspective_map: torch.Tensor,
    roi_mask: torch.Tensor,
    num_bands: int,
) -> torch.Tensor:
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")
    if perspective_map.ndim == 3 and perspective_map.shape[0] == 1:
        perspective_map = perspective_map.squeeze(0)
    if roi_mask.ndim == 3 and roi_mask.shape[0] == 1:
        roi_mask = roi_mask.squeeze(0)
    if perspective_map.ndim != 2 or roi_mask.ndim != 2:
        raise ValueError(
            "build_perspective_band_masks expects 2D perspective_map and roi_mask, "
            f"got {tuple(perspective_map.shape)} and {tuple(roi_mask.shape)}"
        )

    perspective_map = perspective_map.float()
    roi_mask = roi_mask.float().clamp(0.0, 1.0)
    roi_values = perspective_map[roi_mask > 0.5]
    if roi_values.numel() == 0:
        bands = perspective_map.new_zeros((num_bands, 1, perspective_map.shape[0], perspective_map.shape[1]))
        return bands

    if num_bands == 1:
        return roi_mask.unsqueeze(0).unsqueeze(0)

    quantiles = torch.linspace(0.0, 1.0, steps=num_bands + 1, device=perspective_map.device)
    boundaries = torch.quantile(roi_values, quantiles)
    masks: list[torch.Tensor] = []
    for idx in range(num_bands):
        lower = boundaries[idx]
        upper = boundaries[idx + 1]
        if idx == num_bands - 1:
            band = (perspective_map >= lower) & (perspective_map <= upper)
        else:
            band = (perspective_map >= lower) & (perspective_map < upper)
        masks.append((band.float() * roi_mask).unsqueeze(0))
    return torch.stack(masks, dim=0).contiguous()


def load_soft_count_value(cache_dir: str | Path, split: str, frame_stem: str) -> float:
    path = Path(cache_dir) / "mall" / split / f"{frame_stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Soft count cache not found: {path}")
    value = np.load(str(path))
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def load_soft_density_tensor(cache_dir: str | Path, split: str, frame_stem: str) -> torch.Tensor:
    path = Path(cache_dir) / "mall" / split / f"{frame_stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Soft density cache not found: {path}")
    return load_density_npy_as_tensor(path)


def generate_fixed_density_map(points: np.ndarray, height: int, width: int, sigma: float = 4.0) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    if points.size == 0:
        return density

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    for point in points:
        x = min(width - 1, max(0, int(round(point[0]))))
        y = min(height - 1, max(0, int(round(point[1]))))
        density[y, x] += 1.0

    density = gaussian_filter(density, sigma=sigma, mode="constant")
    if density.sum() > 0:
        density = density * (len(points) / max(float(density.sum()), 1e-6))
    return density.astype(np.float32)


def _add_gaussian_to_density(density: np.ndarray, center_x: float, center_y: float, sigma: float) -> None:
    sigma = max(float(sigma), 1e-6)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    height, width = density.shape
    x0 = max(0, int(math.floor(center_x)) - radius)
    x1 = min(width - 1, int(math.floor(center_x)) + radius)
    y0 = max(0, int(math.floor(center_y)) - radius)
    y1 = min(height - 1, int(math.floor(center_y)) + radius)
    if x0 > x1 or y0 > y1:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    kernel = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma))
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0:
        density[y0 : y1 + 1, x0 : x1 + 1] += (kernel / kernel_sum).astype(np.float32)


def compute_adaptive_sigmas(
    points: np.ndarray,
    height: int,
    width: int,
    k: int = 3,
    beta: float = 0.3,
    min_sigma: float = 1.0,
    max_sigma: float = 15.0,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    num_points = len(points)
    if num_points == 0:
        return np.zeros((0,), dtype=np.float32)

    fallback_sigma = np.clip(0.25 * ((height + width) * 0.5), min_sigma, max_sigma)
    if num_points == 1:
        return np.full((1,), fallback_sigma, dtype=np.float32)

    query_k = min(max(k, 1) + 1, num_points)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=query_k)
    distances = np.asarray(distances, dtype=np.float32)
    if distances.ndim == 1:
        distances = distances[:, None]
    neighbor_distances = distances[:, 1:]
    if neighbor_distances.shape[1] == 0:
        sigmas = np.full((num_points,), fallback_sigma, dtype=np.float32)
    else:
        mean_distances = neighbor_distances.mean(axis=1)
        sigmas = beta * mean_distances
        sigmas = np.where(np.isfinite(sigmas), sigmas, fallback_sigma)
        sigmas = np.clip(sigmas, min_sigma, max_sigma)
    return sigmas.astype(np.float32)


def generate_adaptive_density_map(
    points: np.ndarray,
    height: int,
    width: int,
    k: int = 3,
    beta: float = 0.3,
    min_sigma: float = 1.0,
    max_sigma: float = 15.0,
) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    if points.size == 0:
        return density

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    sigmas = compute_adaptive_sigmas(points, height, width, k, beta, min_sigma, max_sigma)
    for point, sigma in zip(points, sigmas):
        center_x = float(np.clip(point[0], 0.0, max(width - 1, 0)))
        center_y = float(np.clip(point[1], 0.0, max(height - 1, 0)))
        _add_gaussian_to_density(density, center_x, center_y, float(sigma))

    if density.sum() > 0:
        density = density * (len(points) / max(float(density.sum()), 1e-6))
    return density.astype(np.float32)


def generate_perspective_density_map(
    points: np.ndarray,
    height: int,
    width: int,
    perspective_map: np.ndarray,
    perspective_scale: float = 0.5,
    min_sigma: float = 1.0,
    max_sigma: float = 8.0,
) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    if points.size == 0:
        return density

    pmap = np.asarray(perspective_map, dtype=np.float32)
    if pmap.shape != (height, width):
        raise ValueError(
            f"Perspective map shape must match density size, got pmap={pmap.shape} vs expected={(height, width)}"
        )

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    for point in points:
        x_idx = int(np.clip(round(float(point[0])), 0, max(width - 1, 0)))
        y_idx = int(np.clip(round(float(point[1])), 0, max(height - 1, 0)))
        sigma = float(np.clip(perspective_scale * float(pmap[y_idx, x_idx]), min_sigma, max_sigma))
        center_x = float(np.clip(point[0], 0.0, max(width - 1, 0)))
        center_y = float(np.clip(point[1], 0.0, max(height - 1, 0)))
        _add_gaussian_to_density(density, center_x, center_y, sigma)

    if density.sum() > 0:
        density = density * (len(points) / max(float(density.sum()), 1e-6))
    return density.astype(np.float32)


def generate_density_map(
    points: np.ndarray,
    height: int,
    width: int,
    sigma: float = 4.0,
    density_kernel: str = "fixed",
    adaptive_k: int = 3,
    adaptive_beta: float = 0.3,
    adaptive_min_sigma: float = 1.0,
    adaptive_max_sigma: float = 15.0,
    perspective_map: np.ndarray | None = None,
    perspective_scale: float = 0.5,
) -> np.ndarray:
    if density_kernel == "adaptive":
        return generate_adaptive_density_map(
            points=points,
            height=height,
            width=width,
            k=adaptive_k,
            beta=adaptive_beta,
            min_sigma=adaptive_min_sigma,
            max_sigma=adaptive_max_sigma,
        )
    if density_kernel == "perspective":
        if perspective_map is None:
            raise ValueError("perspective_map is required when density_kernel='perspective'")
        return generate_perspective_density_map(
            points=points,
            height=height,
            width=width,
            perspective_map=perspective_map,
            perspective_scale=perspective_scale,
            min_sigma=adaptive_min_sigma,
            max_sigma=adaptive_max_sigma,
        )
    return generate_fixed_density_map(points=points, height=height, width=width, sigma=sigma)


def _pad_to_multiple(size: int, multiple: int) -> int:
    if multiple <= 1:
        return size
    return int(math.ceil(size / multiple) * multiple)


def resize_density_tensor(density: torch.Tensor, dst_h: int, dst_w: int) -> torch.Tensor:
    if density.ndim != 3 or density.shape[0] != 1:
        raise ValueError(f"Expected density tensor with shape (1, H, W), but got {tuple(density.shape)}")
    src_h, src_w = density.shape[-2:]
    if src_h == dst_h and src_w == dst_w:
        return density

    density_4d = density.unsqueeze(0)
    resized = F.interpolate(density_4d, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
    scale = (src_h * src_w) / max(float(dst_h * dst_w), 1.0)
    return resized.squeeze(0) * scale


def resize_feature_tensor(
    feature: torch.Tensor,
    dst_h: int,
    dst_w: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    if feature.ndim != 3 or feature.shape[0] != 1:
        raise ValueError(f"Expected feature tensor with shape (1, H, W), but got {tuple(feature.shape)}")
    src_h, src_w = feature.shape[-2:]
    if src_h == dst_h and src_w == dst_w:
        return feature
    resized = F.interpolate(
        feature.unsqueeze(0),
        size=(dst_h, dst_w),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    return resized.squeeze(0)


def shanghai_video_collate_fn(batch: list[dict]) -> dict:
    max_h = max(item["video"].shape[-2] for item in batch)
    max_w = max(item["video"].shape[-1] for item in batch)
    stacked_tensors: dict[str, list[torch.Tensor]] = {}
    list_values: dict[str, list] = {"image_id": [], "part": [], "split": []}
    optional_spatial_keys = {
        "roi_mask",
        "perspective_map",
        "motion_map",
        "background_residual",
        "foreground_confidence",
        "foreground_mask",
        "soft_density",
        "band_masks",
    }

    for item in batch:
        pad_h = max_h - item["video"].shape[-2]
        pad_w = max_w - item["video"].shape[-1]
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                tensor_value = value
                if key in {"video", "density"} or key in optional_spatial_keys:
                    tensor_value = F.pad(tensor_value, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
                stacked_tensors.setdefault(key, []).append(tensor_value)
            else:
                list_values.setdefault(key, []).append(value)

    collated = {key: torch.stack(values, dim=0) for key, values in stacked_tensors.items()}
    collated.update(list_values)
    return collated


class _BaseVideoDataset(Dataset):
    def __init__(
        self,
        *,
        root: str,
        split: str,
        clip_len: int,
        input_size: tuple[int, int],
        sigma: float,
        density_kernel: str,
        adaptive_k: int,
        adaptive_beta: float,
        adaptive_min_sigma: float,
        adaptive_max_sigma: float,
        perspective_scale: float,
        use_density_cache: bool,
        density_cache_dir: str,
        train_mode: str,
        train_crop_size: int,
        train_scale_min: float,
        train_scale_max: float,
        random_hflip: bool,
        train_clip_stride: int,
        test_clip_stride: int,
        test_mode: str,
        test_short_side: int,
        pad_to_multiple: int,
        normalize: bool,
        image_transform: Callable | None,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        if clip_len < 1:
            raise ValueError("clip_len must be >= 1.")
        if density_kernel not in {"fixed", "adaptive", "perspective"}:
            raise ValueError("density_kernel must be 'fixed', 'adaptive', or 'perspective'.")
        if train_mode not in {"legacy", "crop"}:
            raise ValueError("train_mode must be 'legacy' or 'crop'.")
        if test_mode not in {"legacy", "resize_short_side"}:
            raise ValueError("test_mode must be 'legacy' or 'resize_short_side'.")
        if train_scale_min <= 0 or train_scale_max <= 0 or train_scale_min > train_scale_max:
            raise ValueError("train_scale_min and train_scale_max must be positive and ordered.")
        if train_clip_stride < 1 or test_clip_stride < 1:
            raise ValueError("train_clip_stride and test_clip_stride must be >= 1.")

        self.root = Path(root)
        self.split = split
        self.clip_len = clip_len
        self.input_size = input_size
        self.sigma = sigma
        self.density_kernel = density_kernel
        self.adaptive_k = adaptive_k
        self.adaptive_beta = adaptive_beta
        self.adaptive_min_sigma = adaptive_min_sigma
        self.adaptive_max_sigma = adaptive_max_sigma
        self.perspective_scale = perspective_scale
        self.use_density_cache = use_density_cache
        self.density_cache_dir = Path(density_cache_dir) if density_cache_dir else None
        self.train_mode = train_mode
        self.train_crop_size = train_crop_size
        self.train_scale_min = train_scale_min
        self.train_scale_max = train_scale_max
        self.random_hflip = random_hflip
        self.train_clip_stride = train_clip_stride
        self.test_clip_stride = test_clip_stride
        self.test_mode = test_mode
        self.test_short_side = test_short_side
        self.pad_to_multiple = pad_to_multiple
        self.normalize = normalize
        self.image_transform = image_transform

    @staticmethod
    def _list_image_paths(image_dir: Path) -> list[Path]:
        image_paths: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(image_dir.glob(f"*{ext}"))
        return sorted(image_paths)

    @staticmethod
    def _verify_image_file(image_path: Path) -> str | None:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            return f"{image_path.name} ({type(exc).__name__})"
        return None

    def _resize_image_and_density(
        self,
        image: Image.Image,
        density: torch.Tensor,
        dst_size: tuple[int, int],
    ) -> tuple[Image.Image, torch.Tensor]:
        dst_w, dst_h = dst_size
        image = image.resize((dst_w, dst_h), Image.BILINEAR)
        density = resize_density_tensor(density, dst_h=dst_h, dst_w=dst_w)
        return image, density

    def _resize_clip(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
        dst_size: tuple[int, int],
    ) -> tuple[list[Image.Image], list[torch.Tensor]]:
        resized_images: list[Image.Image] = []
        resized_densities: list[torch.Tensor] = []
        for image, density in zip(images, densities):
            resized_image, resized_density = self._resize_image_and_density(image, density, dst_size)
            resized_images.append(resized_image)
            resized_densities.append(resized_density)
        return resized_images, resized_densities

    def _apply_train_crop(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
    ) -> tuple[list[Image.Image], list[torch.Tensor]]:
        crop_size = self.train_crop_size
        src_w, src_h = images[0].size
        scale = random.uniform(self.train_scale_min, self.train_scale_max)
        scaled_w = max(1, int(round(src_w * scale)))
        scaled_h = max(1, int(round(src_h * scale)))
        images, densities = self._resize_clip(images, densities, (scaled_w, scaled_h))

        if scaled_w < crop_size or scaled_h < crop_size:
            min_scale = max(crop_size / max(scaled_w, 1), crop_size / max(scaled_h, 1))
            scaled_w = max(crop_size, int(round(scaled_w * min_scale)))
            scaled_h = max(crop_size, int(round(scaled_h * min_scale)))
            images, densities = self._resize_clip(images, densities, (scaled_w, scaled_h))

        if self.random_hflip and random.random() < 0.5:
            images = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
            densities = [torch.flip(density, dims=[-1]) for density in densities]

        max_left = images[0].size[0] - crop_size
        max_top = images[0].size[1] - crop_size
        left = 0 if max_left <= 0 else random.randint(0, max_left)
        top = 0 if max_top <= 0 else random.randint(0, max_top)
        cropped_images = [image.crop((left, top, left + crop_size, top + crop_size)) for image in images]
        cropped_densities = [
            density[:, top : top + crop_size, left : left + crop_size].contiguous() for density in densities
        ]
        return cropped_images, cropped_densities

    def _apply_test_resize_short_side(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
    ) -> tuple[list[Image.Image], list[torch.Tensor]]:
        src_w, src_h = images[0].size
        short_side = min(src_w, src_h)
        scale = self.test_short_side / max(short_side, 1)
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        return self._resize_clip(images, densities, (dst_w, dst_h))

    def _process_clip(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
    ) -> tuple[list[Image.Image], list[torch.Tensor]]:
        if self.split == "train" and self.train_mode == "crop":
            return self._apply_train_crop(images, densities)
        if self.split != "train" and self.test_mode == "resize_short_side":
            return self._apply_test_resize_short_side(images, densities)
        dst_h, dst_w = self.input_size
        return self._resize_clip(images, densities, (dst_w, dst_h))

    def _to_image_tensor(self, image: Image.Image) -> torch.Tensor:
        if self.image_transform is not None:
            return self.image_transform(image)
        image_tensor = pil_to_float_tensor(image)
        if self.normalize:
            image_tensor = normalize_image(image_tensor)
        return image_tensor

    def _finalize_clip(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
        *,
        image_id: str,
        part_label: str,
    ) -> dict:
        pad_h = 0
        pad_w = 0
        if self.split != "train" and self.test_mode == "resize_short_side" and self.pad_to_multiple > 1:
            padded_h = _pad_to_multiple(densities[0].shape[-2], self.pad_to_multiple)
            padded_w = _pad_to_multiple(densities[0].shape[-1], self.pad_to_multiple)
            pad_h = padded_h - densities[0].shape[-2]
            pad_w = padded_w - densities[0].shape[-1]

        image_tensors = [self._to_image_tensor(image) for image in images]
        density_tensors = [density.float() for density in densities]
        if pad_h > 0 or pad_w > 0:
            image_tensors = [F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for image in image_tensors]
            density_tensors = [F.pad(density, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for density in density_tensors]
        frame_count = torch.stack([density.sum().reshape(()) for density in density_tensors]).float()
        clip_count = frame_count.mean().reshape(1).float()
        return {
            "video": torch.stack(image_tensors, dim=0),
            "density": torch.stack(density_tensors, dim=0),
            "frame_count": frame_count,
            "clip_count": clip_count,
            "image_id": image_id,
            "part": part_label,
            "split": self.split,
        }


class ShanghaiTechVideoDataset(_BaseVideoDataset):
    """
    Convert ShanghaiTech image counting samples into pseudo-video clips.

    Output:
      - video:       (T, C, H, W)
      - density:     (T, 1, H, W)
      - frame_count: (T,)
      - clip_count:  (1,)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        part: str = "A",
        clip_len: int = 4,
        input_size: tuple[int, int] = (256, 256),
        sigma: float = 15.0,
        density_kernel: str = "fixed",
        adaptive_k: int = 3,
        adaptive_beta: float = 0.3,
        adaptive_min_sigma: float = 1.0,
        adaptive_max_sigma: float = 15.0,
        perspective_scale: float = 0.5,
        use_density_cache: bool = False,
        density_cache_dir: str = "",
        train_mode: str = "legacy",
        train_crop_size: int = 256,
        train_scale_min: float = 0.75,
        train_scale_max: float = 1.25,
        random_hflip: bool = False,
        train_clip_stride: int = 1,
        test_clip_stride: int = 1,
        test_mode: str = "legacy",
        test_short_side: int = 512,
        pad_to_multiple: int = 32,
        normalize: bool = True,
        repeat_mode: str = "repeat",
        image_transform: Callable | None = None,
    ) -> None:
        if part not in {"A", "B"}:
            raise ValueError("part must be 'A' or 'B'.")
        if repeat_mode != "repeat":
            raise ValueError("Only repeat_mode='repeat' is supported in this version.")

        super().__init__(
            root=root,
            split=split,
            clip_len=clip_len,
            input_size=input_size,
            sigma=sigma,
            density_kernel=density_kernel,
            adaptive_k=adaptive_k,
            adaptive_beta=adaptive_beta,
            adaptive_min_sigma=adaptive_min_sigma,
            adaptive_max_sigma=adaptive_max_sigma,
            perspective_scale=perspective_scale,
            use_density_cache=use_density_cache,
            density_cache_dir=density_cache_dir,
            train_mode=train_mode,
            train_crop_size=train_crop_size,
            train_scale_min=train_scale_min,
            train_scale_max=train_scale_max,
            random_hflip=random_hflip,
            train_clip_stride=train_clip_stride,
            test_clip_stride=test_clip_stride,
            test_mode=test_mode,
            test_short_side=test_short_side,
            pad_to_multiple=pad_to_multiple,
            normalize=normalize,
            image_transform=image_transform,
        )
        self.part = part
        self.repeat_mode = repeat_mode

        split_dir = self.root / f"part_{part}" / f"{split}_data"
        self.image_dir = split_dir / "images"
        self.gt_dir = split_dir / "ground-truth"
        self.samples = self._build_samples()

        if not self.samples:
            raise FileNotFoundError(f"No ShanghaiTech samples found under {self.image_dir}")

    def _build_samples(self) -> list[dict]:
        image_paths = self._list_image_paths(self.image_dir)
        samples: list[dict] = []
        invalid_images: list[str] = []
        for image_path in image_paths:
            gt_path = self.gt_dir / f"GT_{image_path.stem}.mat"
            if not gt_path.exists():
                continue

            invalid = self._verify_image_file(image_path)
            if invalid is not None:
                invalid_images.append(invalid)
                continue

            samples.append({"image_path": image_path, "gt_path": gt_path})

        skipped = len(invalid_images)
        print(
            f"[ShanghaiTechVideoDataset] split={self.split} part={self.part} "
            f"total_images={len(image_paths)} valid_samples={len(samples)} skipped_bad_images={skipped}"
        )
        if invalid_images:
            preview = ", ".join(invalid_images[:5])
            print(f"[ShanghaiTechVideoDataset] skipped examples: {preview}")
        return samples

    def _resolve_cache_path(self, sample: dict) -> Path:
        if self.density_cache_dir is None:
            raise ValueError("density_cache_dir is required when use_density_cache=True")
        variant = density_cache_variant_name(
            density_kernel=self.density_kernel,
            adaptive_k=self.adaptive_k,
            adaptive_beta=self.adaptive_beta,
            adaptive_min_sigma=self.adaptive_min_sigma,
            adaptive_max_sigma=self.adaptive_max_sigma,
            perspective_scale=self.perspective_scale,
        )
        return (
            self.density_cache_dir
            / f"part_{self.part}"
            / self.split
            / variant
            / f"{sample['image_path'].stem}.npy"
        )

    def _load_single_frame(self, sample: dict) -> tuple[Image.Image, torch.Tensor]:
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(
                f"Failed to read image after dataset verification: {sample['image_path']}"
            ) from exc

        if self.use_density_cache:
            cache_path = self._resolve_cache_path(sample)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Density cache not found for sample {sample['image_path'].name}: {cache_path}"
                )
            density = load_density_npy_as_tensor(cache_path)
        else:
            points = load_shanghaitech_points(sample["gt_path"])
            width, height = image.size
            density = density_numpy_to_tensor(
                generate_density_map(
                    points=points,
                    height=height,
                    width=width,
                    sigma=self.sigma,
                    density_kernel=self.density_kernel,
                    adaptive_k=self.adaptive_k,
                    adaptive_beta=self.adaptive_beta,
                    adaptive_min_sigma=self.adaptive_min_sigma,
                    adaptive_max_sigma=self.adaptive_max_sigma,
                    perspective_scale=self.perspective_scale,
                )
            )
        return image, density

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image, density = self._load_single_frame(self.samples[index])
        images = [image.copy() for _ in range(self.clip_len)]
        densities = [density.clone() for _ in range(self.clip_len)]
        images, densities = self._process_clip(images, densities)
        return self._finalize_clip(
            images,
            densities,
            image_id=self.samples[index]["image_path"].stem,
            part_label=self.part,
        )


class FDSTVideoDataset(_BaseVideoDataset):
    """
    Real video dataset loader for FDST-style sequence folders.

    Preferred directory convention:
      root/train_data/<sequence_id>/<frame_stem>.jpg|png
      root/train_data/<sequence_id>/<frame_stem>.json
      root/test_data/<sequence_id>/<frame_stem>.jpg|png
      root/test_data/<sequence_id>/<frame_stem>.json

    Backward-compatible fallback:
      root/train/<sequence_id>/images/<frame_stem>.jpg|png
      root/train/<sequence_id>/annotations/<frame_stem>.txt|json|mat
      root/test/<sequence_id>/images/<frame_stem>.jpg|png
      root/test/<sequence_id>/annotations/<frame_stem>.txt|json|mat
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 4,
        input_size: tuple[int, int] = (256, 256),
        sigma: float = 15.0,
        density_kernel: str = "fixed",
        adaptive_k: int = 3,
        adaptive_beta: float = 0.3,
        adaptive_min_sigma: float = 1.0,
        adaptive_max_sigma: float = 15.0,
        perspective_scale: float = 0.5,
        use_density_cache: bool = False,
        density_cache_dir: str = "",
        train_mode: str = "legacy",
        train_crop_size: int = 256,
        train_scale_min: float = 0.75,
        train_scale_max: float = 1.25,
        random_hflip: bool = False,
        train_clip_stride: int = 1,
        test_clip_stride: int = 1,
        test_mode: str = "legacy",
        test_short_side: int = 512,
        pad_to_multiple: int = 32,
        normalize: bool = True,
        image_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            clip_len=clip_len,
            input_size=input_size,
            sigma=sigma,
            density_kernel=density_kernel,
            adaptive_k=adaptive_k,
            adaptive_beta=adaptive_beta,
            adaptive_min_sigma=adaptive_min_sigma,
            adaptive_max_sigma=adaptive_max_sigma,
            perspective_scale=perspective_scale,
            use_density_cache=use_density_cache,
            density_cache_dir=density_cache_dir,
            train_mode=train_mode,
            train_crop_size=train_crop_size,
            train_scale_min=train_scale_min,
            train_scale_max=train_scale_max,
            random_hflip=random_hflip,
            train_clip_stride=train_clip_stride,
            test_clip_stride=test_clip_stride,
            test_mode=test_mode,
            test_short_side=test_short_side,
            pad_to_multiple=pad_to_multiple,
            normalize=normalize,
            image_transform=image_transform,
        )
        split_dir_name = "train_data" if split == "train" else "test_data"
        self.sequence_dir = self.root / split_dir_name
        if not self.sequence_dir.exists():
            fallback_dir = self.root / split
            if fallback_dir.exists():
                self.sequence_dir = fallback_dir
        self.samples = self._build_samples()

        if not self.samples:
            raise FileNotFoundError(f"No FDST clips found under {self.sequence_dir}")

    @staticmethod
    def _resolve_annotation_path(annotation_dir: Path, frame_stem: str) -> Path | None:
        for suffix in (".txt", ".json", ".mat"):
            candidate = annotation_dir / f"{frame_stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def _resolve_cache_path(self, sequence_id: str, frame_stem: str) -> Path:
        if self.density_cache_dir is None:
            raise ValueError("density_cache_dir is required when use_density_cache=True")
        variant = density_cache_variant_name(
            density_kernel=self.density_kernel,
            adaptive_k=self.adaptive_k,
            adaptive_beta=self.adaptive_beta,
            adaptive_min_sigma=self.adaptive_min_sigma,
            adaptive_max_sigma=self.adaptive_max_sigma,
            perspective_scale=self.perspective_scale,
        )
        return self.density_cache_dir / "fdst" / self.split / variant / sequence_id / f"{frame_stem}.npy"

    def _build_samples(self) -> list[dict]:
        if not self.sequence_dir.exists():
            raise FileNotFoundError(f"FDST split directory not found: {self.sequence_dir}")

        samples: list[dict] = []
        total_frames = 0
        invalid_images: list[str] = []
        sequence_dirs = sorted(path for path in self.sequence_dir.iterdir() if path.is_dir())

        for sequence_dir in sequence_dirs:
            image_dir = sequence_dir
            annotation_dir = sequence_dir
            if (sequence_dir / "images").exists():
                image_dir = sequence_dir / "images"
            if (sequence_dir / "annotations").exists():
                annotation_dir = sequence_dir / "annotations"

            frame_entries: list[dict] = []
            for image_path in self._list_image_paths(image_dir):
                annotation_path = self._resolve_annotation_path(annotation_dir, image_path.stem)
                if annotation_path is None:
                    continue
                invalid = self._verify_image_file(image_path)
                if invalid is not None:
                    invalid_images.append(f"{sequence_dir.name}/{invalid}")
                    continue
                frame_entries.append(
                    {
                        "image_path": image_path,
                        "annotation_path": annotation_path,
                        "frame_stem": image_path.stem,
                    }
                )

            total_frames += len(frame_entries)
            if len(frame_entries) < self.clip_len:
                continue

            clip_stride = self.train_clip_stride if self.split == "train" else self.test_clip_stride
            for start in range(0, len(frame_entries) - self.clip_len + 1, clip_stride):
                samples.append(
                    {
                        "sequence_id": sequence_dir.name,
                        "frames": frame_entries[start : start + self.clip_len],
                        "start_index": start,
                    }
                )

        print(
            f"[FDSTVideoDataset] split={self.split} sequences={len(sequence_dirs)} "
            f"total_frames={total_frames} clips={len(samples)} skipped_bad_images={len(invalid_images)}"
        )
        if invalid_images:
            preview = ", ".join(invalid_images[:5])
            print(f"[FDSTVideoDataset] skipped examples: {preview}")
        return samples

    def _load_frame(self, sequence_id: str, frame_entry: dict) -> tuple[Image.Image, torch.Tensor]:
        try:
            image = Image.open(frame_entry["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(
                f"Failed to read FDST image after dataset verification: {frame_entry['image_path']}"
            ) from exc

        if self.use_density_cache:
            cache_path = self._resolve_cache_path(sequence_id, frame_entry["frame_stem"])
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Density cache not found for FDST frame {sequence_id}/{frame_entry['frame_stem']}: {cache_path}"
                )
            density = load_density_npy_as_tensor(cache_path)
        else:
            points = load_fdst_points(frame_entry["annotation_path"])
            width, height = image.size
            density = density_numpy_to_tensor(
                generate_density_map(
                    points=points,
                    height=height,
                    width=width,
                    sigma=self.sigma,
                    density_kernel=self.density_kernel,
                    adaptive_k=self.adaptive_k,
                    adaptive_beta=self.adaptive_beta,
                    adaptive_min_sigma=self.adaptive_min_sigma,
                    adaptive_max_sigma=self.adaptive_max_sigma,
                    perspective_scale=self.perspective_scale,
                )
            )
        return image, density

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        images: list[Image.Image] = []
        densities: list[torch.Tensor] = []
        for frame_entry in sample["frames"]:
            image, density = self._load_frame(sample["sequence_id"], frame_entry)
            images.append(image)
            densities.append(density)

        images, densities = self._process_clip(images, densities)
        frame_stems = [frame["frame_stem"] for frame in sample["frames"]]
        image_id = f"{sample['sequence_id']}/{frame_stems[0]}-{frame_stems[-1]}"
        return self._finalize_clip(images, densities, image_id=image_id, part_label="FDST")


class MallVideoDataset(_BaseVideoDataset):
    """
    Real video dataset loader for the Mall dataset.

    Expected directory convention:
      root/frames/frames/seq_000001.jpg ... seq_002000.jpg
      root/mall_gt.mat
      root/perspective_roi.mat

    Split convention:
      train -> frames 1..800
      test  -> frames 801..2000
    """

    TRAIN_END_FRAME = 800
    TEST_START_FRAME = 801

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 4,
        input_size: tuple[int, int] = (256, 256),
        sigma: float = 15.0,
        density_kernel: str = "fixed",
        adaptive_k: int = 3,
        adaptive_beta: float = 0.3,
        adaptive_min_sigma: float = 1.0,
        adaptive_max_sigma: float = 15.0,
        perspective_scale: float = 0.5,
        use_density_cache: bool = False,
        density_cache_dir: str = "",
        train_mode: str = "legacy",
        train_crop_size: int = 256,
        train_scale_min: float = 0.75,
        train_scale_max: float = 1.25,
        random_hflip: bool = False,
        train_clip_stride: int = 1,
        test_clip_stride: int = 1,
        test_mode: str = "legacy",
        test_short_side: int = 512,
        pad_to_multiple: int = 32,
        normalize: bool = True,
        use_roi_mask: bool = False,
        use_roi_input: bool = False,
        use_perspective_input: bool = False,
        use_motion_input: bool = False,
        use_background_input: bool = False,
        use_foreground_input: bool = False,
        background_cache_dir: str = "",
        use_foreground_head: bool = False,
        foreground_threshold: float = 0.2,
        num_perspective_bands: int = 3,
        use_soft_count_target: bool = False,
        soft_count_cache_dir: str = "",
        use_soft_density_target: bool = False,
        soft_density_cache_dir: str = "",
        use_official_count: bool = False,
        rescale_density_to_count: bool = False,
        image_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            clip_len=clip_len,
            input_size=input_size,
            sigma=sigma,
            density_kernel=density_kernel,
            adaptive_k=adaptive_k,
            adaptive_beta=adaptive_beta,
            adaptive_min_sigma=adaptive_min_sigma,
            adaptive_max_sigma=adaptive_max_sigma,
            perspective_scale=perspective_scale,
            use_density_cache=use_density_cache,
            density_cache_dir=density_cache_dir,
            train_mode=train_mode,
            train_crop_size=train_crop_size,
            train_scale_min=train_scale_min,
            train_scale_max=train_scale_max,
            random_hflip=random_hflip,
            train_clip_stride=train_clip_stride,
            test_clip_stride=test_clip_stride,
            test_mode=test_mode,
            test_short_side=test_short_side,
            pad_to_multiple=pad_to_multiple,
            normalize=normalize,
            image_transform=image_transform,
        )
        self.use_roi_mask = use_roi_mask
        self.use_roi_input = use_roi_input
        self.use_perspective_input = use_perspective_input
        self.use_motion_input = use_motion_input
        self.use_background_input = use_background_input
        self.use_foreground_input = use_foreground_input
        self.background_cache_dir = Path(background_cache_dir) if background_cache_dir else None
        self.use_foreground_head = use_foreground_head
        self.foreground_threshold = float(foreground_threshold)
        self.num_perspective_bands = int(num_perspective_bands)
        self.use_soft_count_target = use_soft_count_target
        self.soft_count_cache_dir = Path(soft_count_cache_dir) if soft_count_cache_dir else None
        self.use_soft_density_target = use_soft_density_target
        self.soft_density_cache_dir = Path(soft_density_cache_dir) if soft_density_cache_dir else None
        self.use_official_count = bool(use_official_count)
        self.rescale_density_to_count = bool(rescale_density_to_count)
        if self.use_soft_density_target and self.split == "train" and self.train_mode == "crop":
            raise ValueError("Mall soft density targets currently support train_mode='legacy' only.")
        self.image_dir = self.root / "frames" / "frames"
        if not self.image_dir.exists():
            fallback_dir = self.root / "frames"
            if fallback_dir.exists():
                self.image_dir = fallback_dir
        self.gt_path = self.root / "mall_gt.mat"
        self.roi_path = self.root / "perspective_roi.mat"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Mall image directory not found: {self.image_dir}")
        if not self.gt_path.exists():
            raise FileNotFoundError(f"Mall ground truth file not found: {self.gt_path}")
        if not self.roi_path.exists():
            raise FileNotFoundError(f"Mall scene prior file not found: {self.roi_path}")

        self.points_per_frame, self.gt_counts = load_mall_ground_truth(self.gt_path)
        self.roi_mask_np, self.perspective_map_np, self.perspective_map_normalized_np = load_mall_scene_priors(self.roi_path)
        self.background_rgb_np = None
        if self.use_background_input or self.use_foreground_input or self.use_foreground_head:
            if self.background_cache_dir is None:
                raise ValueError(
                    "background_cache_dir is required when using background input, foreground input, or foreground head"
                )
            if self.split == "train" and self.train_mode == "crop":
                raise ValueError("Mall background/foreground inputs currently support train_mode='legacy' only.")
            self.background_rgb_np = load_background_image(self.background_cache_dir, self.split)
        self.samples = self._build_samples()
        if not self.samples:
            raise FileNotFoundError(f"No Mall clips found under {self.image_dir}")

    @staticmethod
    def _parse_frame_number(frame_stem: str) -> int:
        try:
            return int(frame_stem.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Unsupported Mall frame name: {frame_stem}") from exc

    def _is_split_frame(self, frame_number: int) -> bool:
        if self.split == "train":
            return 1 <= frame_number <= self.TRAIN_END_FRAME
        return self.TEST_START_FRAME <= frame_number <= len(self.points_per_frame)

    def _resolve_cache_path(self, frame_stem: str) -> Path:
        if self.density_cache_dir is None:
            raise ValueError("density_cache_dir is required when use_density_cache=True")
        variant = density_cache_variant_name(
            density_kernel=self.density_kernel,
            adaptive_k=self.adaptive_k,
            adaptive_beta=self.adaptive_beta,
            adaptive_min_sigma=self.adaptive_min_sigma,
            adaptive_max_sigma=self.adaptive_max_sigma,
            perspective_scale=self.perspective_scale,
        )
        return self.density_cache_dir / "mall" / self.split / variant / f"{frame_stem}.npy"

    @staticmethod
    def _resize_map(map_tensor: torch.Tensor, dst_h: int, dst_w: int, mode: str) -> torch.Tensor:
        return resize_feature_tensor(map_tensor, dst_h=dst_h, dst_w=dst_w, mode=mode)

    def _resize_clip_with_maps(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
        roi_masks: list[torch.Tensor],
        perspective_maps: list[torch.Tensor],
        dst_size: tuple[int, int],
    ) -> tuple[list[Image.Image], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        dst_w, dst_h = dst_size
        resized_images: list[Image.Image] = []
        resized_densities: list[torch.Tensor] = []
        resized_roi: list[torch.Tensor] = []
        resized_pmap: list[torch.Tensor] = []
        for image, density, roi_mask, perspective_map in zip(images, densities, roi_masks, perspective_maps):
            resized_image, resized_density = self._resize_image_and_density(image, density, (dst_w, dst_h))
            resized_images.append(resized_image)
            resized_densities.append(resized_density)
            resized_roi.append(self._resize_map(roi_mask, dst_h=dst_h, dst_w=dst_w, mode="nearest"))
            resized_pmap.append(self._resize_map(perspective_map, dst_h=dst_h, dst_w=dst_w, mode="bilinear"))
        return resized_images, resized_densities, resized_roi, resized_pmap

    def _process_clip_with_scene_maps(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
        roi_masks: list[torch.Tensor],
        perspective_maps: list[torch.Tensor],
    ) -> tuple[list[Image.Image], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if self.split == "train" and self.train_mode == "crop":
            crop_size = self.train_crop_size
            src_w, src_h = images[0].size
            scale = random.uniform(self.train_scale_min, self.train_scale_max)
            scaled_w = max(1, int(round(src_w * scale)))
            scaled_h = max(1, int(round(src_h * scale)))
            images, densities, roi_masks, perspective_maps = self._resize_clip_with_maps(
                images, densities, roi_masks, perspective_maps, (scaled_w, scaled_h)
            )

            if scaled_w < crop_size or scaled_h < crop_size:
                min_scale = max(crop_size / max(scaled_w, 1), crop_size / max(scaled_h, 1))
                scaled_w = max(crop_size, int(round(scaled_w * min_scale)))
                scaled_h = max(crop_size, int(round(scaled_h * min_scale)))
                images, densities, roi_masks, perspective_maps = self._resize_clip_with_maps(
                    images, densities, roi_masks, perspective_maps, (scaled_w, scaled_h)
                )

            if self.random_hflip and random.random() < 0.5:
                images = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
                densities = [torch.flip(density, dims=[-1]) for density in densities]
                roi_masks = [torch.flip(roi_mask, dims=[-1]) for roi_mask in roi_masks]
                perspective_maps = [torch.flip(pmap, dims=[-1]) for pmap in perspective_maps]

            max_left = images[0].size[0] - crop_size
            max_top = images[0].size[1] - crop_size
            left = 0 if max_left <= 0 else random.randint(0, max_left)
            top = 0 if max_top <= 0 else random.randint(0, max_top)
            images = [image.crop((left, top, left + crop_size, top + crop_size)) for image in images]
            densities = [density[:, top : top + crop_size, left : left + crop_size].contiguous() for density in densities]
            roi_masks = [roi[:, top : top + crop_size, left : left + crop_size].contiguous() for roi in roi_masks]
            perspective_maps = [p[:, top : top + crop_size, left : left + crop_size].contiguous() for p in perspective_maps]
            return images, densities, roi_masks, perspective_maps

        if self.split != "train" and self.test_mode == "resize_short_side":
            src_w, src_h = images[0].size
            short_side = min(src_w, src_h)
            scale = self.test_short_side / max(short_side, 1)
            dst_w = max(1, int(round(src_w * scale)))
            dst_h = max(1, int(round(src_h * scale)))
            return self._resize_clip_with_maps(images, densities, roi_masks, perspective_maps, (dst_w, dst_h))

        dst_h, dst_w = self.input_size
        return self._resize_clip_with_maps(images, densities, roi_masks, perspective_maps, (dst_w, dst_h))

    def _process_background_image(self, image: Image.Image) -> Image.Image:
        if self.split == "train" and self.train_mode == "crop":
            raise ValueError("Background image processing for Mall does not support train_mode='crop'.")

        if self.split != "train" and self.test_mode == "resize_short_side":
            src_w, src_h = image.size
            short_side = min(src_w, src_h)
            scale = self.test_short_side / max(short_side, 1)
            dst_w = max(1, int(round(src_w * scale)))
            dst_h = max(1, int(round(src_h * scale)))
            return image.resize((dst_w, dst_h), resample=Image.BILINEAR)

        dst_h, dst_w = self.input_size
        return image.resize((dst_w, dst_h), resample=Image.BILINEAR)

    def _build_samples(self) -> list[dict]:
        image_paths = self._list_image_paths(self.image_dir)
        frame_entries: list[dict] = []
        invalid_images: list[str] = []

        for image_path in image_paths:
            frame_number = self._parse_frame_number(image_path.stem)
            if frame_number < 1 or frame_number > len(self.points_per_frame):
                continue
            if not self._is_split_frame(frame_number):
                continue

            invalid = self._verify_image_file(image_path)
            if invalid is not None:
                invalid_images.append(invalid)
                continue

            frame_entries.append(
                {
                    "image_path": image_path,
                    "frame_stem": image_path.stem,
                    "frame_number": frame_number,
                    "points": self.points_per_frame[frame_number - 1],
                    "gt_count": float(self.gt_counts[frame_number - 1]),
                }
            )

        samples: list[dict] = []
        if len(frame_entries) >= self.clip_len:
            clip_stride = self.train_clip_stride if self.split == "train" else self.test_clip_stride
            for start in range(0, len(frame_entries) - self.clip_len + 1, clip_stride):
                samples.append(
                    {
                        "frames": frame_entries[start : start + self.clip_len],
                        "start_index": start,
                    }
                )

        print(
            f"[MallVideoDataset] split={self.split} total_frames={len(frame_entries)} "
            f"clips={len(samples)} skipped_bad_images={len(invalid_images)}"
        )
        if invalid_images:
            preview = ", ".join(invalid_images[:5])
            print(f"[MallVideoDataset] skipped examples: {preview}")
        return samples

    def _load_frame(self, frame_entry: dict) -> tuple[Image.Image, torch.Tensor]:
        try:
            image = Image.open(frame_entry["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(
                f"Failed to read Mall image after dataset verification: {frame_entry['image_path']}"
            ) from exc

        if self.use_density_cache:
            cache_path = self._resolve_cache_path(frame_entry["frame_stem"])
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Density cache not found for Mall frame {frame_entry['frame_stem']}: {cache_path}"
                )
            density = load_density_npy_as_tensor(cache_path)
        else:
            width, height = image.size
            density = density_numpy_to_tensor(
                generate_density_map(
                    points=frame_entry["points"],
                    height=height,
                    width=width,
                    sigma=self.sigma,
                    density_kernel=self.density_kernel,
                    adaptive_k=self.adaptive_k,
                    adaptive_beta=self.adaptive_beta,
                    adaptive_min_sigma=self.adaptive_min_sigma,
                    adaptive_max_sigma=self.adaptive_max_sigma,
                    perspective_map=self.perspective_map_np,
                    perspective_scale=self.perspective_scale,
                )
            )
        return image, density

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        images: list[Image.Image] = []
        raw_images: list[Image.Image] = []
        densities: list[torch.Tensor] = []
        soft_densities: list[torch.Tensor] = []
        roi_masks = []
        perspective_maps = []
        raw_roi_masks = []
        raw_perspective_maps = []
        for frame_entry in sample["frames"]:
            image, density = self._load_frame(frame_entry)
            images.append(image)
            raw_images.append(image.copy())
            densities.append(density)
            if self.use_soft_density_target:
                if self.soft_density_cache_dir is None:
                    raise ValueError("soft_density_cache_dir is required when use_soft_density_target=True")
                soft_densities.append(load_soft_density_tensor(self.soft_density_cache_dir, self.split, frame_entry["frame_stem"]))
            roi_tensor = density_numpy_to_tensor(self.roi_mask_np)
            perspective_tensor = density_numpy_to_tensor(self.perspective_map_normalized_np)
            roi_masks.append(roi_tensor)
            perspective_maps.append(perspective_tensor)
            raw_roi_masks.append(roi_tensor.clone())
            raw_perspective_maps.append(perspective_tensor.clone())

        images, densities, roi_masks, perspective_maps = self._process_clip_with_scene_maps(
            images,
            densities,
            roi_masks,
            perspective_maps,
        )
        if self.use_soft_density_target:
            _, soft_densities, _, _ = self._process_clip_with_scene_maps(
                raw_images,
                soft_densities,
                raw_roi_masks,
                raw_perspective_maps,
            )

        if self.use_roi_mask:
            densities = [density * roi_mask for density, roi_mask in zip(densities, roi_masks)]
            if self.use_soft_density_target:
                soft_densities = [density * roi_mask for density, roi_mask in zip(soft_densities, roi_masks)]

        official_counts = [float(frame["gt_count"]) for frame in sample["frames"]]
        if self.rescale_density_to_count:
            densities = [
                self._scale_density_tensor_to_count(density, official_count)
                for density, official_count in zip(densities, official_counts)
            ]
            if self.use_soft_density_target:
                soft_densities = [
                    self._scale_density_tensor_to_count(density, official_count)
                    for density, official_count in zip(soft_densities, official_counts)
                ]

        pad_h = 0
        pad_w = 0
        if self.split != "train" and self.test_mode == "resize_short_side" and self.pad_to_multiple > 1:
            padded_h = _pad_to_multiple(densities[0].shape[-2], self.pad_to_multiple)
            padded_w = _pad_to_multiple(densities[0].shape[-1], self.pad_to_multiple)
            pad_h = padded_h - densities[0].shape[-2]
            pad_w = padded_w - densities[0].shape[-1]

        rgb_tensors = []
        for image in images:
            if self.image_transform is not None:
                rgb_tensors.append(self.image_transform(image))
            else:
                rgb_tensors.append(pil_to_float_tensor(image))

        background_rgb = None
        foreground_residuals: list[torch.Tensor] = []
        foreground_confidences: list[torch.Tensor] = []
        foreground_masks: list[torch.Tensor] = []
        if self.background_rgb_np is not None:
            background_image = Image.fromarray((self.background_rgb_np * 255.0).round().astype(np.uint8), mode="RGB")
            background_image = self._process_background_image(background_image)
            background_rgb = pil_to_float_tensor(background_image)

        motion_maps: list[torch.Tensor] = []
        if self.use_motion_input:
            grayscale = []
            for rgb in rgb_tensors:
                gray = 0.2989 * rgb[0:1] + 0.5870 * rgb[1:2] + 0.1140 * rgb[2:3]
                grayscale.append(gray)
            for frame_idx, gray in enumerate(grayscale):
                if frame_idx == 0:
                    motion_maps.append(torch.zeros_like(gray))
                else:
                    motion_maps.append(torch.abs(gray - grayscale[frame_idx - 1]))

        video_tensors: list[torch.Tensor] = []
        for frame_idx, rgb in enumerate(rgb_tensors):
            if background_rgb is not None:
                residual, confidence = build_foreground_maps(rgb, background_rgb, roi_masks[frame_idx].float())
                foreground_residuals.append(residual.float())
                foreground_confidences.append(confidence.float())
                foreground_masks.append((confidence >= self.foreground_threshold).float().contiguous())
            rgb_tensor = normalize_image(rgb) if self.normalize and self.image_transform is None else rgb
            extras = []
            if self.use_roi_input:
                extras.append(roi_masks[frame_idx].float())
            if self.use_perspective_input:
                extras.append(perspective_maps[frame_idx].float())
            if self.use_motion_input:
                extras.append(motion_maps[frame_idx].float())
            if self.use_background_input:
                if not foreground_residuals:
                    raise RuntimeError("Foreground residuals were not prepared for use_background_input")
                extras.append(foreground_residuals[frame_idx])
            if self.use_foreground_input:
                if not foreground_confidences:
                    raise RuntimeError("Foreground confidences were not prepared for use_foreground_input")
                extras.append(foreground_confidences[frame_idx])
            if extras:
                rgb_tensor = torch.cat([rgb_tensor] + extras, dim=0)
            video_tensors.append(rgb_tensor)

        density_tensors = [density.float() for density in densities]
        roi_tensors = [roi.float() for roi in roi_masks]
        perspective_tensors = [p.float() for p in perspective_maps]
        if self.use_motion_input:
            motion_maps = [motion.float() for motion in motion_maps]
        if self.use_soft_density_target:
            soft_density_tensors = [density.float() for density in soft_densities]
        else:
            soft_density_tensors = []

        if pad_h > 0 or pad_w > 0:
            video_tensors = [F.pad(video, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for video in video_tensors]
            density_tensors = [F.pad(density, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for density in density_tensors]
            roi_tensors = [F.pad(roi, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for roi in roi_tensors]
            perspective_tensors = [F.pad(p, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for p in perspective_tensors]
            if self.use_motion_input:
                motion_maps = [F.pad(motion, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for motion in motion_maps]
            if self.use_soft_density_target:
                soft_density_tensors = [F.pad(density, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for density in soft_density_tensors]
            if foreground_residuals:
                foreground_residuals = [F.pad(residual, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for residual in foreground_residuals]
                foreground_confidences = [F.pad(conf, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for conf in foreground_confidences]
                foreground_masks = [F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for mask in foreground_masks]

        frame_stems = [frame["frame_stem"] for frame in sample["frames"]]
        frame_indices = torch.tensor([int(frame["frame_number"]) for frame in sample["frames"]], dtype=torch.long)
        image_id = f"mall/{frame_stems[0]}-{frame_stems[-1]}"
        density_frame_count = torch.stack([density.sum().reshape(()) for density in density_tensors]).float()
        official_frame_count = torch.tensor(official_counts, dtype=torch.float32)
        frame_count = official_frame_count if self.use_official_count else density_frame_count
        clip_count = frame_count.mean().reshape(1).float()
        band_masks = build_perspective_band_masks(
            perspective_tensors[0],
            roi_tensors[0],
            num_bands=self.num_perspective_bands,
        ).float()
        output = {
            "video": torch.stack(video_tensors, dim=0),
            "density": torch.stack(density_tensors, dim=0),
            "frame_count": frame_count,
            "clip_count": clip_count,
            "density_frame_count": density_frame_count,
            "official_frame_count": official_frame_count,
            "official_clip_count": official_frame_count.mean().reshape(1),
            "image_id": image_id,
            "frame_stems": frame_stems,
            "frame_indices": frame_indices,
            "sequence_id": f"mall_{self.split}",
            "part": "Mall",
            "split": self.split,
            "roi_mask": torch.stack(roi_tensors, dim=0),
            "perspective_map": torch.stack(perspective_tensors, dim=0),
            "band_masks": band_masks,
        }
        if self.use_motion_input:
            output["motion_map"] = torch.stack(motion_maps, dim=0)
        if foreground_confidences:
            output["foreground_confidence"] = torch.stack(foreground_confidences, dim=0)
            output["foreground_mask"] = torch.stack(foreground_masks, dim=0)
            output["background_residual"] = torch.stack(foreground_residuals, dim=0)
        if self.use_soft_count_target:
            if self.soft_count_cache_dir is None:
                raise ValueError("soft_count_cache_dir is required when use_soft_count_target=True")
            soft_frame_count = torch.tensor(
                [load_soft_count_value(self.soft_count_cache_dir, self.split, frame_stems[i]) for i in range(len(frame_stems))],
                dtype=torch.float32,
            )
            output["soft_frame_count"] = soft_frame_count
            output["soft_clip_count"] = soft_frame_count.mean().reshape(1)
        if self.use_soft_density_target:
            output["soft_density"] = torch.stack(soft_density_tensors, dim=0)
        return output

    @staticmethod
    def _scale_density_tensor_to_count(density: torch.Tensor, target_count: float) -> torch.Tensor:
        target = max(float(target_count), 0.0)
        if target == 0.0:
            return torch.zeros_like(density)
        current = float(density.sum())
        if current <= 1e-8:
            return density
        return density * (target / current)


class BusVideoDataset(_BaseVideoDataset):
    """
    Fixed-camera Bus dataset loader.

    Expected directory convention:
      root/train/images/bus_0.jpg, root/train/images/bus_0.mat, ...
      root/train/amb_gt/bus_0.npy, ...
      root/test/images/bus_14130.jpg, root/test/images/bus_14130.mat, ...
      root/test/amb_gt/bus_14130.npy, ...
      root/bus_roi.npy
      root/bus_perspective_density_prior.npy (optional; generated from train/amb_gt if absent)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 8,
        input_size: tuple[int, int] = (192, 288),
        sigma: float = 15.0,
        density_kernel: str = "fixed",
        adaptive_k: int = 3,
        adaptive_beta: float = 0.3,
        adaptive_min_sigma: float = 1.0,
        adaptive_max_sigma: float = 15.0,
        perspective_scale: float = 0.5,
        use_density_cache: bool = False,
        density_cache_dir: str = "",
        train_mode: str = "legacy",
        train_crop_size: int = 192,
        train_scale_min: float = 0.9,
        train_scale_max: float = 1.1,
        random_hflip: bool = False,
        train_clip_stride: int = 1,
        test_clip_stride: int = 1,
        test_mode: str = "legacy",
        test_short_side: int = 192,
        pad_to_multiple: int = 32,
        normalize: bool = True,
        use_roi_mask: bool = False,
        use_roi_input: bool = False,
        use_perspective_input: bool = False,
        use_motion_input: bool = False,
        use_background_input: bool = False,
        use_foreground_input: bool = False,
        background_cache_dir: str = "",
        use_foreground_head: bool = False,
        foreground_threshold: float = 0.2,
        num_perspective_bands: int = 3,
        use_soft_count_target: bool = False,
        soft_count_cache_dir: str = "",
        use_soft_density_target: bool = False,
        soft_density_cache_dir: str = "",
        image_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            clip_len=clip_len,
            input_size=input_size,
            sigma=sigma,
            density_kernel=density_kernel,
            adaptive_k=adaptive_k,
            adaptive_beta=adaptive_beta,
            adaptive_min_sigma=adaptive_min_sigma,
            adaptive_max_sigma=adaptive_max_sigma,
            perspective_scale=perspective_scale,
            use_density_cache=use_density_cache,
            density_cache_dir=density_cache_dir,
            train_mode=train_mode,
            train_crop_size=train_crop_size,
            train_scale_min=train_scale_min,
            train_scale_max=train_scale_max,
            random_hflip=random_hflip,
            train_clip_stride=train_clip_stride,
            test_clip_stride=test_clip_stride,
            test_mode=test_mode,
            test_short_side=test_short_side,
            pad_to_multiple=pad_to_multiple,
            normalize=normalize,
            image_transform=image_transform,
        )
        if use_background_input or use_foreground_input or use_foreground_head:
            raise ValueError("Bus dataset does not support background/foreground input paths yet.")
        if use_soft_count_target or use_soft_density_target:
            raise ValueError("Bus dataset does not support soft target distillation caches yet.")
        if background_cache_dir or soft_count_cache_dir or soft_density_cache_dir:
            pass
        if foreground_threshold <= 0:
            pass

        self.use_roi_mask = use_roi_mask
        self.use_roi_input = use_roi_input
        self.use_perspective_input = use_perspective_input
        self.use_motion_input = use_motion_input
        self.num_perspective_bands = int(num_perspective_bands)

        self.image_dir = self.root / self.split / "images"
        self.density_dir = self.root / self.split / "amb_gt"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Bus image directory not found: {self.image_dir}")
        if not self.density_dir.exists():
            raise FileNotFoundError(f"Bus density directory not found: {self.density_dir}")

        self.roi_mask_np, self.perspective_map_np, self.perspective_map_normalized_np = load_bus_scene_priors(self.root)
        self.frame_entries = self._load_frame_entries()
        self.samples = self._build_samples()
        if not self.samples:
            raise FileNotFoundError(f"No Bus clips found under {self.image_dir} for split={self.split}")

    @staticmethod
    def _resize_map(map_tensor: torch.Tensor, dst_h: int, dst_w: int, mode: str) -> torch.Tensor:
        return resize_feature_tensor(map_tensor, dst_h=dst_h, dst_w=dst_w, mode=mode)

    def _resolve_cache_path(self, frame_stem: str) -> Path:
        if self.density_cache_dir is None:
            raise ValueError("density_cache_dir is required when use_density_cache=True")
        return self.density_cache_dir / "bus" / self.split / "amb_gt_rescaled" / f"{frame_stem}.npy"

    def _load_frame_entries(self) -> list[dict]:
        image_paths = sorted(self._list_image_paths(self.image_dir), key=lambda path: _parse_bus_frame_number(path.stem))
        entries: list[dict] = []
        invalid_images: list[str] = []
        for image_path in image_paths:
            annotation_path = image_path.with_suffix(".mat")
            density_path = self.density_dir / f"{image_path.stem}.npy"
            if not annotation_path.exists() or not density_path.exists():
                continue
            invalid = self._verify_image_file(image_path)
            if invalid is not None:
                invalid_images.append(invalid)
                continue
            points, gt_count = load_bus_frame_info(annotation_path)
            entries.append(
                {
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                    "density_path": density_path,
                    "frame_stem": image_path.stem,
                    "frame_number": _parse_bus_frame_number(image_path.stem),
                    "points": points,
                    "gt_count": gt_count,
                }
            )
        print(
            f"[BusVideoDataset] split={self.split} frames={len(entries)} "
            f"skipped_bad_images={len(invalid_images)}"
        )
        if invalid_images:
            preview = ", ".join(invalid_images[:5])
            print(f"[BusVideoDataset] skipped examples: {preview}")
        return entries

    def _build_samples(self) -> list[dict]:
        samples: list[dict] = []
        clip_stride = self.train_clip_stride if self.split == "train" else self.test_clip_stride
        if len(self.frame_entries) >= self.clip_len:
            for start in range(0, len(self.frame_entries) - self.clip_len + 1, clip_stride):
                samples.append({"frames": self.frame_entries[start : start + self.clip_len], "start_index": start})
        print(f"[BusVideoDataset] split={self.split} clips={len(samples)}")
        return samples

    def _load_frame(self, frame_entry: dict) -> tuple[Image.Image, torch.Tensor]:
        try:
            image = Image.open(frame_entry["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to read Bus image after verification: {frame_entry['image_path']}") from exc

        if self.use_density_cache:
            density = load_density_npy_as_tensor(self._resolve_cache_path(frame_entry["frame_stem"]))
        else:
            density = load_density_npy_as_tensor(frame_entry["density_path"])
        density_np = density.squeeze(0).numpy()
        density_np = scale_density_to_count(density_np, float(frame_entry["gt_count"]))
        if self.use_roi_mask:
            density_np = density_np * self.roi_mask_np
            density_np = scale_density_to_count(density_np, float(frame_entry["gt_count"]))
        return image, density_numpy_to_tensor(density_np)

    def _resize_clip_with_maps(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
        roi_masks: list[torch.Tensor],
        perspective_maps: list[torch.Tensor],
        dst_size: tuple[int, int],
    ) -> tuple[list[Image.Image], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        dst_w, dst_h = dst_size
        resized_images: list[Image.Image] = []
        resized_densities: list[torch.Tensor] = []
        resized_roi: list[torch.Tensor] = []
        resized_pmap: list[torch.Tensor] = []
        for image, density, roi_mask, perspective_map in zip(images, densities, roi_masks, perspective_maps):
            resized_image, resized_density = self._resize_image_and_density(image, density, (dst_w, dst_h))
            resized_images.append(resized_image)
            resized_densities.append(resized_density)
            resized_roi.append(self._resize_map(roi_mask, dst_h=dst_h, dst_w=dst_w, mode="nearest"))
            resized_pmap.append(self._resize_map(perspective_map, dst_h=dst_h, dst_w=dst_w, mode="bilinear"))
        return resized_images, resized_densities, resized_roi, resized_pmap

    def _process_clip_with_scene_maps(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
        roi_masks: list[torch.Tensor],
        perspective_maps: list[torch.Tensor],
    ) -> tuple[list[Image.Image], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if self.split == "train" and self.train_mode == "crop":
            crop_size = self.train_crop_size
            src_w, src_h = images[0].size
            scale = random.uniform(self.train_scale_min, self.train_scale_max)
            scaled_w = max(1, int(round(src_w * scale)))
            scaled_h = max(1, int(round(src_h * scale)))
            images, densities, roi_masks, perspective_maps = self._resize_clip_with_maps(
                images, densities, roi_masks, perspective_maps, (scaled_w, scaled_h)
            )
            if scaled_w < crop_size or scaled_h < crop_size:
                min_scale = max(crop_size / max(scaled_w, 1), crop_size / max(scaled_h, 1))
                scaled_w = max(crop_size, int(round(scaled_w * min_scale)))
                scaled_h = max(crop_size, int(round(scaled_h * min_scale)))
                images, densities, roi_masks, perspective_maps = self._resize_clip_with_maps(
                    images, densities, roi_masks, perspective_maps, (scaled_w, scaled_h)
                )
            if self.random_hflip and random.random() < 0.5:
                images = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
                densities = [torch.flip(density, dims=[-1]) for density in densities]
                roi_masks = [torch.flip(roi_mask, dims=[-1]) for roi_mask in roi_masks]
                perspective_maps = [torch.flip(pmap, dims=[-1]) for pmap in perspective_maps]
            max_left = images[0].size[0] - crop_size
            max_top = images[0].size[1] - crop_size
            left = 0 if max_left <= 0 else random.randint(0, max_left)
            top = 0 if max_top <= 0 else random.randint(0, max_top)
            images = [image.crop((left, top, left + crop_size, top + crop_size)) for image in images]
            densities = [density[:, top : top + crop_size, left : left + crop_size].contiguous() for density in densities]
            roi_masks = [roi[:, top : top + crop_size, left : left + crop_size].contiguous() for roi in roi_masks]
            perspective_maps = [p[:, top : top + crop_size, left : left + crop_size].contiguous() for p in perspective_maps]
            return images, densities, roi_masks, perspective_maps

        if self.split != "train" and self.test_mode == "resize_short_side":
            src_w, src_h = images[0].size
            short_side = min(src_w, src_h)
            scale = self.test_short_side / max(short_side, 1)
            dst_w = max(1, int(round(src_w * scale)))
            dst_h = max(1, int(round(src_h * scale)))
            return self._resize_clip_with_maps(images, densities, roi_masks, perspective_maps, (dst_w, dst_h))

        dst_h, dst_w = self.input_size
        return self._resize_clip_with_maps(images, densities, roi_masks, perspective_maps, (dst_w, dst_h))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        images: list[Image.Image] = []
        densities: list[torch.Tensor] = []
        roi_masks: list[torch.Tensor] = []
        perspective_maps: list[torch.Tensor] = []
        for frame_entry in sample["frames"]:
            image, density = self._load_frame(frame_entry)
            images.append(image)
            densities.append(density)
            roi_masks.append(density_numpy_to_tensor(self.roi_mask_np))
            perspective_maps.append(density_numpy_to_tensor(self.perspective_map_normalized_np))

        images, densities, roi_masks, perspective_maps = self._process_clip_with_scene_maps(
            images, densities, roi_masks, perspective_maps
        )
        if self.use_roi_mask:
            densities = [density * roi_mask for density, roi_mask in zip(densities, roi_masks)]
        if not (self.split == "train" and self.train_mode == "crop"):
            densities = [
                self._scale_density_tensor_to_count(density, float(frame_entry["gt_count"]))
                for density, frame_entry in zip(densities, sample["frames"])
            ]

        pad_h = 0
        pad_w = 0
        if self.split != "train" and self.test_mode == "resize_short_side" and self.pad_to_multiple > 1:
            padded_h = _pad_to_multiple(densities[0].shape[-2], self.pad_to_multiple)
            padded_w = _pad_to_multiple(densities[0].shape[-1], self.pad_to_multiple)
            pad_h = padded_h - densities[0].shape[-2]
            pad_w = padded_w - densities[0].shape[-1]

        rgb_tensors = [pil_to_float_tensor(image) for image in images]
        motion_maps: list[torch.Tensor] = []
        if self.use_motion_input:
            grayscale = [0.2989 * rgb[0:1] + 0.5870 * rgb[1:2] + 0.1140 * rgb[2:3] for rgb in rgb_tensors]
            for frame_idx, gray in enumerate(grayscale):
                motion_maps.append(torch.zeros_like(gray) if frame_idx == 0 else torch.abs(gray - grayscale[frame_idx - 1]))

        video_tensors: list[torch.Tensor] = []
        for frame_idx, rgb in enumerate(rgb_tensors):
            rgb_tensor = normalize_image(rgb) if self.normalize and self.image_transform is None else rgb
            extras = []
            if self.use_roi_input:
                extras.append(roi_masks[frame_idx].float())
            if self.use_perspective_input:
                extras.append(perspective_maps[frame_idx].float())
            if self.use_motion_input:
                extras.append(motion_maps[frame_idx].float())
            if extras:
                rgb_tensor = torch.cat([rgb_tensor] + extras, dim=0)
            video_tensors.append(rgb_tensor)

        density_tensors = [density.float() for density in densities]
        roi_tensors = [roi.float() for roi in roi_masks]
        perspective_tensors = [p.float() for p in perspective_maps]
        if pad_h > 0 or pad_w > 0:
            video_tensors = [F.pad(video, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for video in video_tensors]
            density_tensors = [F.pad(density, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for density in density_tensors]
            roi_tensors = [F.pad(roi, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for roi in roi_tensors]
            perspective_tensors = [F.pad(p, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for p in perspective_tensors]
            if self.use_motion_input:
                motion_maps = [F.pad(motion, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for motion in motion_maps]

        frame_stems = [frame["frame_stem"] for frame in sample["frames"]]
        frame_indices = torch.tensor([int(frame["frame_number"]) for frame in sample["frames"]], dtype=torch.long)
        frame_count = torch.stack([density.sum().reshape(()) for density in density_tensors]).float()
        clip_count = frame_count.mean().reshape(1).float()
        band_masks = build_perspective_band_masks(
            perspective_tensors[0],
            roi_tensors[0],
            num_bands=self.num_perspective_bands,
        ).float()
        output = {
            "video": torch.stack(video_tensors, dim=0),
            "density": torch.stack(density_tensors, dim=0),
            "frame_count": frame_count,
            "clip_count": clip_count,
            "image_id": f"bus/{frame_stems[0]}-{frame_stems[-1]}",
            "frame_stems": frame_stems,
            "frame_indices": frame_indices,
            "sequence_id": f"bus_{self.split}",
            "part": "Bus",
            "split": self.split,
            "roi_mask": torch.stack(roi_tensors, dim=0),
            "perspective_map": torch.stack(perspective_tensors, dim=0),
            "band_masks": band_masks,
        }
        if self.use_motion_input:
            output["motion_map"] = torch.stack([motion.float() for motion in motion_maps], dim=0)
        return output

    @staticmethod
    def _scale_density_tensor_to_count(density: torch.Tensor, target_count: float) -> torch.Tensor:
        target = max(float(target_count), 0.0)
        if target == 0.0:
            return torch.zeros_like(density)
        current = float(density.sum())
        if current <= 1e-8:
            return density
        return density * (target / current)


class UCSDVideoDataset(_BaseVideoDataset):
    """
    UCSD pedestrian counting loader using the common CVPR-2000 protocol.

    Expected directory convention under root:
      root/ucsdpeds_vidf/video/vidf/vidf1_33_000.y/*.png ... vidf1_33_009.y/*.png
      root/uscdpeds_gt/gt/vidf/vidf1_33_000_people_full.mat ...
      root/uscdpeds_gt/gt/vidf/vidf1_33_000_count_2K_roi_mainwalkway.mat ...
      root/uscdpeds_gt/gt/vidf/vidf1_33_roi_mainwalkway.mat
      root/uscdpeds_gt/gt/vidf/vidf1_33_dmap3.mat

    Split convention:
      train -> global frames 601..1400
      test  -> global frames 1..600 and 1401..2000
    """

    NUM_CLIPS = 10
    FRAMES_PER_CLIP = 200
    TRAIN_START_FRAME = 601
    TRAIN_END_FRAME = 1400

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 4,
        input_size: tuple[int, int] = (192, 288),
        sigma: float = 4.0,
        density_kernel: str = "fixed",
        adaptive_k: int = 3,
        adaptive_beta: float = 0.3,
        adaptive_min_sigma: float = 0.5,
        adaptive_max_sigma: float = 6.0,
        perspective_scale: float = 0.5,
        use_density_cache: bool = False,
        density_cache_dir: str = "",
        train_mode: str = "legacy",
        train_crop_size: int = 192,
        train_scale_min: float = 0.9,
        train_scale_max: float = 1.15,
        random_hflip: bool = False,
        train_clip_stride: int = 1,
        test_clip_stride: int = 1,
        test_mode: str = "legacy",
        test_short_side: int = 192,
        pad_to_multiple: int = 32,
        normalize: bool = True,
        use_roi_mask: bool = False,
        use_roi_input: bool = False,
        use_perspective_input: bool = False,
        use_motion_input: bool = False,
        use_background_input: bool = False,
        use_foreground_input: bool = False,
        background_cache_dir: str = "",
        use_foreground_head: bool = False,
        foreground_threshold: float = 0.2,
        num_perspective_bands: int = 3,
        use_soft_count_target: bool = False,
        soft_count_cache_dir: str = "",
        use_soft_density_target: bool = False,
        soft_density_cache_dir: str = "",
        image_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            clip_len=clip_len,
            input_size=input_size,
            sigma=sigma,
            density_kernel=density_kernel,
            adaptive_k=adaptive_k,
            adaptive_beta=adaptive_beta,
            adaptive_min_sigma=adaptive_min_sigma,
            adaptive_max_sigma=adaptive_max_sigma,
            perspective_scale=perspective_scale,
            use_density_cache=use_density_cache,
            density_cache_dir=density_cache_dir,
            train_mode=train_mode,
            train_crop_size=train_crop_size,
            train_scale_min=train_scale_min,
            train_scale_max=train_scale_max,
            random_hflip=random_hflip,
            train_clip_stride=train_clip_stride,
            test_clip_stride=test_clip_stride,
            test_mode=test_mode,
            test_short_side=test_short_side,
            pad_to_multiple=pad_to_multiple,
            normalize=normalize,
            image_transform=image_transform,
        )
        if use_background_input or use_foreground_input or use_foreground_head:
            raise ValueError("UCSD dataset does not support background/foreground input paths yet.")
        if use_soft_count_target or use_soft_density_target:
            raise ValueError("UCSD dataset does not support soft target distillation caches yet.")

        self.use_roi_mask = use_roi_mask
        self.use_roi_input = use_roi_input
        self.use_perspective_input = use_perspective_input
        self.use_motion_input = use_motion_input
        self.num_perspective_bands = int(num_perspective_bands)

        self.video_root = self.root / "ucsdpeds_vidf" / "video" / "vidf"
        self.gt_dir = self.root / "uscdpeds_gt" / "gt" / "vidf"
        if not self.video_root.exists():
            raise FileNotFoundError(f"UCSD video directory not found: {self.video_root}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"UCSD ground-truth directory not found: {self.gt_dir}")

        self.roi_mask_np, self.perspective_map_np, self.perspective_map_normalized_np = load_ucsd_scene_priors(self.gt_dir)
        self.frame_entries = self._load_frame_entries()
        self.samples = self._build_samples()
        if not self.samples:
            raise FileNotFoundError(f"No UCSD clips found under {self.video_root} for split={self.split}")

    @staticmethod
    def _clip_id(clip_index: int) -> str:
        return f"vidf1_33_{clip_index:03d}"

    @staticmethod
    def _local_frame_number(frame_stem: str) -> int:
        try:
            return int(frame_stem.split("_f")[-1])
        except ValueError as exc:
            raise ValueError(f"Unsupported UCSD frame name: {frame_stem}") from exc

    @classmethod
    def _global_frame_number(cls, clip_index: int, local_frame: int) -> int:
        return clip_index * cls.FRAMES_PER_CLIP + local_frame

    @classmethod
    def is_split_global_frame(cls, split: str, global_frame: int) -> bool:
        if split == "train":
            return cls.TRAIN_START_FRAME <= global_frame <= cls.TRAIN_END_FRAME
        return global_frame < cls.TRAIN_START_FRAME or global_frame > cls.TRAIN_END_FRAME

    def _load_frame_entries(self) -> list[dict]:
        entries: list[dict] = []
        invalid_images: list[str] = []
        for clip_index in range(self.NUM_CLIPS):
            clip_id = self._clip_id(clip_index)
            image_dir = self.video_root / f"{clip_id}.y"
            people_path = self.gt_dir / f"{clip_id}_people_full.mat"
            count_path = self.gt_dir / f"{clip_id}_count_2K_roi_mainwalkway.mat"
            if not image_dir.exists():
                raise FileNotFoundError(f"UCSD clip image directory not found: {image_dir}")
            if not people_path.exists():
                raise FileNotFoundError(f"UCSD people annotation file not found: {people_path}")
            if not count_path.exists():
                raise FileNotFoundError(f"UCSD count annotation file not found: {count_path}")

            gt_counts = load_ucsd_counts(count_path, expected_frames=self.FRAMES_PER_CLIP)
            try:
                points_per_frame = load_ucsd_points(
                    people_path,
                    expected_frames=self.FRAMES_PER_CLIP,
                    gt_counts=gt_counts,
                    roi_mask=None,
                )
            except ValueError as exc:
                print(f"[UCSDVideoDataset] warning: {exc}; using ROI-center fallback points for density labels.")
                points_per_frame = fallback_ucsd_points_from_roi(self.roi_mask_np, self.FRAMES_PER_CLIP)
            image_paths = self._list_image_paths(image_dir)
            if len(image_paths) < self.FRAMES_PER_CLIP:
                raise FileNotFoundError(
                    f"UCSD clip {clip_id} must contain at least {self.FRAMES_PER_CLIP} frames, got {len(image_paths)}"
                )

            image_by_local = {self._local_frame_number(path.stem): path for path in image_paths}
            for local_frame in range(1, self.FRAMES_PER_CLIP + 1):
                image_path = image_by_local.get(local_frame)
                if image_path is None:
                    raise FileNotFoundError(f"Missing UCSD frame {clip_id}_f{local_frame:03d}.png under {image_dir}")
                invalid = self._verify_image_file(image_path)
                if invalid is not None:
                    invalid_images.append(f"{clip_id}/{invalid}")
                    continue
                global_frame = self._global_frame_number(clip_index, local_frame)
                if not self.is_split_global_frame(self.split, global_frame):
                    continue
                entries.append(
                    {
                        "image_path": image_path,
                        "frame_stem": f"{clip_id}_f{local_frame:03d}",
                        "clip_id": clip_id,
                        "clip_index": clip_index,
                        "local_frame": local_frame,
                        "global_frame": global_frame,
                        "points": points_per_frame[local_frame - 1],
                        "gt_count": float(gt_counts[local_frame - 1]),
                    }
                )

        print(
            f"[UCSDVideoDataset] split={self.split} frames={len(entries)} "
            f"skipped_bad_images={len(invalid_images)}"
        )
        if invalid_images:
            preview = ", ".join(invalid_images[:5])
            print(f"[UCSDVideoDataset] skipped examples: {preview}")
        return entries

    def _build_samples(self) -> list[dict]:
        samples: list[dict] = []
        clip_stride = self.train_clip_stride if self.split == "train" else self.test_clip_stride
        contiguous_runs: list[list[dict]] = []
        current_run: list[dict] = []
        previous_frame = None
        for entry in self.frame_entries:
            frame_number = int(entry["global_frame"])
            if previous_frame is None or frame_number == previous_frame + 1:
                current_run.append(entry)
            else:
                if current_run:
                    contiguous_runs.append(current_run)
                current_run = [entry]
            previous_frame = frame_number
        if current_run:
            contiguous_runs.append(current_run)

        for run in contiguous_runs:
            if len(run) < self.clip_len:
                continue
            for start in range(0, len(run) - self.clip_len + 1, clip_stride):
                samples.append(
                    {
                        "frames": run[start : start + self.clip_len],
                        "start_index": start,
                        "sequence_id": self._sequence_id_for_run(run),
                    }
                )
        print(f"[UCSDVideoDataset] split={self.split} clips={len(samples)}")
        return samples

    @staticmethod
    def _sequence_id_for_run(run: list[dict]) -> str:
        first_frame = int(run[0]["global_frame"])
        if first_frame < UCSDVideoDataset.TRAIN_START_FRAME:
            return "ucsd_test_head"
        if first_frame > UCSDVideoDataset.TRAIN_END_FRAME:
            return "ucsd_test_tail"
        return "ucsd_train_mid"

    def _resolve_cache_path(self, frame_stem: str) -> Path:
        if self.density_cache_dir is None:
            raise ValueError("density_cache_dir is required when use_density_cache=True")
        variant = density_cache_variant_name(
            density_kernel=self.density_kernel,
            adaptive_k=self.adaptive_k,
            adaptive_beta=self.adaptive_beta,
            adaptive_min_sigma=self.adaptive_min_sigma,
            adaptive_max_sigma=self.adaptive_max_sigma,
            perspective_scale=self.perspective_scale,
        )
        return self.density_cache_dir / "ucsd" / self.split / variant / f"{frame_stem}.npy"

    @staticmethod
    def _resize_map(map_tensor: torch.Tensor, dst_h: int, dst_w: int, mode: str) -> torch.Tensor:
        return resize_feature_tensor(map_tensor, dst_h=dst_h, dst_w=dst_w, mode=mode)

    def _resize_clip_with_maps(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
        roi_masks: list[torch.Tensor],
        perspective_maps: list[torch.Tensor],
        dst_size: tuple[int, int],
    ) -> tuple[list[Image.Image], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        dst_w, dst_h = dst_size
        resized_images: list[Image.Image] = []
        resized_densities: list[torch.Tensor] = []
        resized_roi: list[torch.Tensor] = []
        resized_pmap: list[torch.Tensor] = []
        for image, density, roi_mask, perspective_map in zip(images, densities, roi_masks, perspective_maps):
            resized_image, resized_density = self._resize_image_and_density(image, density, (dst_w, dst_h))
            resized_images.append(resized_image)
            resized_densities.append(resized_density)
            resized_roi.append(self._resize_map(roi_mask, dst_h=dst_h, dst_w=dst_w, mode="nearest"))
            resized_pmap.append(self._resize_map(perspective_map, dst_h=dst_h, dst_w=dst_w, mode="bilinear"))
        return resized_images, resized_densities, resized_roi, resized_pmap

    def _process_clip_with_scene_maps(
        self,
        images: list[Image.Image],
        densities: list[torch.Tensor],
        roi_masks: list[torch.Tensor],
        perspective_maps: list[torch.Tensor],
    ) -> tuple[list[Image.Image], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if self.split == "train" and self.train_mode == "crop":
            crop_size = self.train_crop_size
            src_w, src_h = images[0].size
            scale = random.uniform(self.train_scale_min, self.train_scale_max)
            scaled_w = max(1, int(round(src_w * scale)))
            scaled_h = max(1, int(round(src_h * scale)))
            images, densities, roi_masks, perspective_maps = self._resize_clip_with_maps(
                images, densities, roi_masks, perspective_maps, (scaled_w, scaled_h)
            )

            if scaled_w < crop_size or scaled_h < crop_size:
                min_scale = max(crop_size / max(scaled_w, 1), crop_size / max(scaled_h, 1))
                scaled_w = max(crop_size, int(round(scaled_w * min_scale)))
                scaled_h = max(crop_size, int(round(scaled_h * min_scale)))
                images, densities, roi_masks, perspective_maps = self._resize_clip_with_maps(
                    images, densities, roi_masks, perspective_maps, (scaled_w, scaled_h)
                )

            if self.random_hflip and random.random() < 0.5:
                images = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
                densities = [torch.flip(density, dims=[-1]) for density in densities]
                roi_masks = [torch.flip(roi_mask, dims=[-1]) for roi_mask in roi_masks]
                perspective_maps = [torch.flip(pmap, dims=[-1]) for pmap in perspective_maps]

            max_left = images[0].size[0] - crop_size
            max_top = images[0].size[1] - crop_size
            left = 0 if max_left <= 0 else random.randint(0, max_left)
            top = 0 if max_top <= 0 else random.randint(0, max_top)
            images = [image.crop((left, top, left + crop_size, top + crop_size)) for image in images]
            densities = [density[:, top : top + crop_size, left : left + crop_size].contiguous() for density in densities]
            roi_masks = [roi[:, top : top + crop_size, left : left + crop_size].contiguous() for roi in roi_masks]
            perspective_maps = [p[:, top : top + crop_size, left : left + crop_size].contiguous() for p in perspective_maps]
            return images, densities, roi_masks, perspective_maps

        if self.split != "train" and self.test_mode == "resize_short_side":
            src_w, src_h = images[0].size
            short_side = min(src_w, src_h)
            scale = self.test_short_side / max(short_side, 1)
            dst_w = max(1, int(round(src_w * scale)))
            dst_h = max(1, int(round(src_h * scale)))
            return self._resize_clip_with_maps(images, densities, roi_masks, perspective_maps, (dst_w, dst_h))

        dst_h, dst_w = self.input_size
        return self._resize_clip_with_maps(images, densities, roi_masks, perspective_maps, (dst_w, dst_h))

    def _load_frame(self, frame_entry: dict) -> tuple[Image.Image, torch.Tensor]:
        try:
            image = Image.open(frame_entry["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to read UCSD image after verification: {frame_entry['image_path']}") from exc

        if self.use_density_cache:
            cache_path = self._resolve_cache_path(frame_entry["frame_stem"])
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Density cache not found for UCSD frame {frame_entry['frame_stem']}: {cache_path}"
                )
            density = load_density_npy_as_tensor(cache_path)
        else:
            width, height = image.size
            density_np = generate_density_map(
                points=frame_entry["points"],
                height=height,
                width=width,
                sigma=self.sigma,
                density_kernel=self.density_kernel,
                adaptive_k=self.adaptive_k,
                adaptive_beta=self.adaptive_beta,
                adaptive_min_sigma=self.adaptive_min_sigma,
                adaptive_max_sigma=self.adaptive_max_sigma,
                perspective_map=self.perspective_map_np,
                perspective_scale=self.perspective_scale,
            )
            if self.use_roi_mask:
                density_np = density_np * self.roi_mask_np
            density_np = scale_density_to_count(density_np, float(frame_entry["gt_count"]))
            density = density_numpy_to_tensor(density_np)
        return image, density

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        images: list[Image.Image] = []
        densities: list[torch.Tensor] = []
        roi_masks = []
        perspective_maps = []
        for frame_entry in sample["frames"]:
            image, density = self._load_frame(frame_entry)
            images.append(image)
            densities.append(density)
            roi_masks.append(density_numpy_to_tensor(self.roi_mask_np))
            perspective_maps.append(density_numpy_to_tensor(self.perspective_map_normalized_np))

        images, densities, roi_masks, perspective_maps = self._process_clip_with_scene_maps(
            images,
            densities,
            roi_masks,
            perspective_maps,
        )
        if self.use_roi_mask:
            densities = [density * roi_mask for density, roi_mask in zip(densities, roi_masks)]
        if not (self.split == "train" and self.train_mode == "crop"):
            densities = [
                self._scale_density_tensor_to_count(density, float(frame_entry["gt_count"]))
                for density, frame_entry in zip(densities, sample["frames"])
            ]

        pad_h = 0
        pad_w = 0
        if self.split != "train" and self.test_mode == "resize_short_side" and self.pad_to_multiple > 1:
            padded_h = _pad_to_multiple(densities[0].shape[-2], self.pad_to_multiple)
            padded_w = _pad_to_multiple(densities[0].shape[-1], self.pad_to_multiple)
            pad_h = padded_h - densities[0].shape[-2]
            pad_w = padded_w - densities[0].shape[-1]

        rgb_tensors = [pil_to_float_tensor(image) for image in images]
        motion_maps: list[torch.Tensor] = []
        if self.use_motion_input:
            grayscale = [0.2989 * rgb[0:1] + 0.5870 * rgb[1:2] + 0.1140 * rgb[2:3] for rgb in rgb_tensors]
            for frame_idx, gray in enumerate(grayscale):
                if frame_idx == 0:
                    motion_maps.append(torch.zeros_like(gray))
                else:
                    motion_maps.append(torch.abs(gray - grayscale[frame_idx - 1]))

        video_tensors: list[torch.Tensor] = []
        for frame_idx, rgb in enumerate(rgb_tensors):
            rgb_tensor = normalize_image(rgb) if self.normalize and self.image_transform is None else rgb
            extras = []
            if self.use_roi_input:
                extras.append(roi_masks[frame_idx].float())
            if self.use_perspective_input:
                extras.append(perspective_maps[frame_idx].float())
            if self.use_motion_input:
                extras.append(motion_maps[frame_idx].float())
            if extras:
                rgb_tensor = torch.cat([rgb_tensor] + extras, dim=0)
            video_tensors.append(rgb_tensor)

        density_tensors = [density.float() for density in densities]
        roi_tensors = [roi.float() for roi in roi_masks]
        perspective_tensors = [p.float() for p in perspective_maps]
        if pad_h > 0 or pad_w > 0:
            video_tensors = [F.pad(video, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for video in video_tensors]
            density_tensors = [F.pad(density, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for density in density_tensors]
            roi_tensors = [F.pad(roi, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for roi in roi_tensors]
            perspective_tensors = [F.pad(p, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for p in perspective_tensors]
            if self.use_motion_input:
                motion_maps = [F.pad(motion, (0, pad_w, 0, pad_h), mode="constant", value=0.0) for motion in motion_maps]

        frame_stems = [frame["frame_stem"] for frame in sample["frames"]]
        global_indices = torch.tensor([int(frame["global_frame"]) for frame in sample["frames"]], dtype=torch.long)
        frame_count = torch.stack([density.sum().reshape(()) for density in density_tensors]).float()
        clip_count = frame_count.mean().reshape(1).float()
        band_masks = build_perspective_band_masks(
            perspective_tensors[0],
            roi_tensors[0],
            num_bands=self.num_perspective_bands,
        ).float()
        output = {
            "video": torch.stack(video_tensors, dim=0),
            "density": torch.stack(density_tensors, dim=0),
            "frame_count": frame_count,
            "clip_count": clip_count,
            "image_id": f"ucsd/{frame_stems[0]}-{frame_stems[-1]}",
            "frame_stems": frame_stems,
            "frame_indices": global_indices,
            "sequence_id": sample.get("sequence_id", "ucsd_cvpr2000"),
            "part": "UCSD",
            "split": self.split,
            "roi_mask": torch.stack(roi_tensors, dim=0),
            "perspective_map": torch.stack(perspective_tensors, dim=0),
            "band_masks": band_masks,
        }
        if self.use_motion_input:
            output["motion_map"] = torch.stack([motion.float() for motion in motion_maps], dim=0)
        return output

    @staticmethod
    def _scale_density_tensor_to_count(density: torch.Tensor, target_count: float) -> torch.Tensor:
        target = max(float(target_count), 0.0)
        if target == 0.0:
            return torch.zeros_like(density)
        current = float(density.sum())
        if current <= 1e-8:
            return density
        return density * (target / current)


def build_video_dataset(dataset_name: str, **kwargs) -> Dataset:
    dataset_name = str(dataset_name).lower()
    if dataset_name == "shanghaitech":
        for key in (
            "use_roi_mask",
            "use_roi_input",
            "use_perspective_input",
            "use_motion_input",
            "use_background_input",
            "use_foreground_input",
            "background_cache_dir",
            "use_foreground_head",
            "foreground_threshold",
            "num_perspective_bands",
            "use_soft_count_target",
            "soft_count_cache_dir",
            "use_soft_density_target",
            "soft_density_cache_dir",
            "use_official_count",
            "rescale_density_to_count",
            "train_clip_stride",
            "test_clip_stride",
        ):
            kwargs.pop(key, None)
        return ShanghaiTechVideoDataset(**kwargs)
    if dataset_name == "fdst":
        kwargs.pop("part", None)
        kwargs.pop("repeat_mode", None)
        for key in (
            "use_roi_mask",
            "use_roi_input",
            "use_perspective_input",
            "use_motion_input",
            "use_background_input",
            "use_foreground_input",
            "background_cache_dir",
            "use_foreground_head",
            "foreground_threshold",
            "num_perspective_bands",
            "use_soft_count_target",
            "soft_count_cache_dir",
            "use_soft_density_target",
            "soft_density_cache_dir",
            "use_official_count",
            "rescale_density_to_count",
        ):
            kwargs.pop(key, None)
        return FDSTVideoDataset(**kwargs)
    if dataset_name == "mall":
        kwargs.pop("part", None)
        kwargs.pop("repeat_mode", None)
        return MallVideoDataset(**kwargs)
    if dataset_name == "bus":
        kwargs.pop("part", None)
        kwargs.pop("repeat_mode", None)
        kwargs.pop("use_official_count", None)
        kwargs.pop("rescale_density_to_count", None)
        return BusVideoDataset(**kwargs)
    if dataset_name == "ucsd":
        kwargs.pop("part", None)
        kwargs.pop("repeat_mode", None)
        kwargs.pop("use_official_count", None)
        kwargs.pop("rescale_density_to_count", None)
        return UCSDVideoDataset(**kwargs)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


__all__ = [
    "FDSTVideoDataset",
    "BusVideoDataset",
    "IMAGENET_MEAN",
    "IMAGENET_PAD_FILL",
    "IMAGENET_STD",
    "MallVideoDataset",
    "ShanghaiTechVideoDataset",
    "UCSDVideoDataset",
    "build_video_dataset",
    "density_cache_variant_name",
    "density_numpy_to_tensor",
    "generate_density_map",
    "load_density_npy_as_tensor",
    "load_bus_frame_info",
    "load_bus_scene_priors",
    "load_mall_ground_truth",
    "load_mall_scene_priors",
    "load_ucsd_counts",
    "load_ucsd_points",
    "load_ucsd_scene_priors",
    "fallback_ucsd_points_from_roi",
    "normalize_image",
    "pil_to_float_tensor",
    "resize_feature_tensor",
    "resize_density_tensor",
    "shanghai_video_collate_fn",
]
