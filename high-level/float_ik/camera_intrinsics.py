#!/usr/bin/env python3
"""Camera-intrinsics profiles and deterministic pinhole image remapping."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover - reported explicitly when remap is requested.
    cv2 = None


LEGACY_MODE = "legacy"
REAL_K_REMAP_MODE = "real_k_remap"
SUPPORTED_MODES = (LEGACY_MODE, REAL_K_REMAP_MODE)
REMAP_VERSION = "pinhole_k_render_to_k_target_v1"


def pinhole_intrinsics_from_horizontal_fov(width: int, height: int, horizontal_fov_deg: float) -> dict:
    width = int(width)
    height = int(height)
    horizontal_fov_deg = float(horizontal_fov_deg)
    if width <= 0 or height <= 0:
        raise ValueError(f"Camera resolution must be positive, got {(width, height)}.")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError(f"Horizontal FOV must be in (0, 180), got {horizontal_fov_deg}.")
    focal = float(width) / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return {
        "fx": focal,
        "fy": focal,
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "width": width,
        "height": height,
        "horizontal_fov_deg": horizontal_fov_deg,
    }


def _validated_intrinsics(value: dict, *, width: int, height: int, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} intrinsics must be a mapping.")
    result = {key: float(value[key]) for key in ("fx", "fy", "cx", "cy")}
    if result["fx"] <= 0.0 or result["fy"] <= 0.0:
        raise ValueError(f"{name} focal lengths must be positive, got {result}.")
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError(f"{name} intrinsics must be finite, got {result}.")
    result.update({"width": int(width), "height": int(height)})
    return result


def load_real_camera_intrinsics_config(path: str | Path) -> dict:
    raw_path = Path(path).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        high_level_root = Path(__file__).resolve().parents[1]
        candidates.extend([high_level_root / raw_path, high_level_root.parent / raw_path])
    path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), raw_path.resolve())
    if not path.is_file():
        raise FileNotFoundError(
            f"Camera intrinsics config does not exist: {raw_path}; checked "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    resolution = raw.get("resolution", [640, 480])
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError(f"resolution must be [width, height], got {resolution!r}.")
    width, height = int(resolution[0]), int(resolution[1])
    render_fov = float(raw.get("render_horizontal_fov_deg", 60.0))
    render_intrinsics = pinhole_intrinsics_from_horizontal_fov(width, height, render_fov)
    cameras = {
        name: _validated_intrinsics(raw.get(name), width=width, height=height, name=name)
        for name in ("front", "wrist")
    }
    return {
        "config_path": str(path),
        "resolution": [width, height],
        "render_horizontal_fov_deg": render_fov,
        "render_intrinsics": render_intrinsics,
        "camera_intrinsics": cameras,
    }


def reverse_remap_coordinates(source_intrinsics: dict, target_intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    width = int(target_intrinsics["width"])
    height = int(target_intrinsics["height"])
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = (
        float(source_intrinsics["fx"])
        * (u - float(target_intrinsics["cx"]))
        / float(target_intrinsics["fx"])
        + float(source_intrinsics["cx"])
    )
    map_y = (
        float(source_intrinsics["fy"])
        * (v - float(target_intrinsics["cy"]))
        / float(target_intrinsics["fy"])
        + float(source_intrinsics["cy"])
    )
    return map_x.astype(np.float32), map_y.astype(np.float32)


def remap_coverage(map_x: np.ndarray, map_y: np.ndarray, source_intrinsics: dict) -> float:
    width = int(source_intrinsics["width"])
    height = int(source_intrinsics["height"])
    valid = (map_x >= 0.0) & (map_x <= width - 1) & (map_y >= 0.0) & (map_y <= height - 1)
    return float(np.mean(valid))


def remap_image(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, *, interpolation: str) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for --camera_intrinsics_mode real_k_remap.")
    interpolation_flag = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
    }.get(str(interpolation))
    if interpolation_flag is None:
        raise ValueError(f"Unsupported remap interpolation {interpolation!r}.")
    return cv2.remap(
        np.asarray(image),
        map_x,
        map_y,
        interpolation=interpolation_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
