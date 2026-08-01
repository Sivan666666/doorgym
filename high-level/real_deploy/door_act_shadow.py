#!/usr/bin/env python3
"""Run Door ACT inference on a Jetson for real deployment or offline checks.

By default this runner publishes ACT base commands to ROS and full ACT actions
to the local Z1 bridge, matching the real robot deployment path.  Pass
``--no_enable_ros_base_bridge`` and ``--no_enable_z1_action_bridge`` for a pure
offline/shadow run; also pass ``--no_enable_z1_state_receiver`` to avoid
listening for Z1 EE feedback.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
DEPTH_FPS = 30
POLICY_HZ = 25.0
DEFAULT_ACTION_HORIZON = 10
DEPTH_LOWER_M = 0.2
DEPTH_FAR_M = 1.5
DEFAULT_CROP_LEFT = 0
DEFAULT_CROP_RIGHT = 0
DEFAULT_CROP_TOP = 0
DEFAULT_CROP_BOTTOM = 0
DEFAULT_RS_SPATIAL_MAGNITUDE = 2
DEFAULT_RS_SPATIAL_SMOOTH_DELTA = 50
DEFAULT_RS_SPATIAL_HOLES_FILL = 5
DEPTH_INPAINT_MODES = ("off", "realsense", "rgb_guided", "opencv_k", "opencv_k_no_rs")
DEFAULT_DEPTH_INPAINT_MODE = "opencv_k_no_rs"
DEFAULT_DEPTH_INPAINT_MAX_DISTANCE_PX = 64.0
DEFAULT_DEPTH_INPAINT_ITERATIONS = 2
DEFAULT_DEPTH_INPAINT_RGB_SIGMA = 0.10
DEFAULT_OPENCV_K_SMALL_MAX_AREA = 15000
DEFAULT_OPENCV_K_SMALL_MAX_SPAN_PX = 280
DEFAULT_OPENCV_K_SMALL_BORDER_MARGIN_PX = -1
DEFAULT_OPENCV_K_WRIST_FRINGE_PX = 10
DEFAULT_OPENCV_K_FRONT_FRINGE_PX = 40
DEFAULT_OPENCV_K_WHITE_HOLE_THRESHOLD = 250
DEFAULT_OPENCV_K_WHITE_HOLE_MAX_AREA = 2500
DEFAULT_OPENCV_K_WHITE_HOLE_MAX_SPAN_PX = 90
DEFAULT_OPENCV_K_WHITE_HOLE_BORDER_MARGIN_PX = 2
DEFAULT_OPENCV_K_WHITE_HOLE_RING_RADIUS_PX = 3
DEFAULT_OPENCV_K_WHITE_HOLE_MIN_GRAY_RING_PX = 8
DEFAULT_OPENCV_K_WHITE_HOLE_MIN_GRAY_RING_RATIO = 0.35
DEFAULT_OPENCV_K_REALTIME_PROCESS_SCALE = 1.0
DEFAULT_DEPTH_GAUSSIAN_BLUR_KSIZE = 0
DEFAULT_DEPTH_GAUSSIAN_BLUR_SIGMA = 0.0
DEFAULT_WRIST_REALSENSE_SERIAL = "261222075130"
DEFAULT_FRONT_REALSENSE_SERIAL = "261222075566"
EE_STATE_ACTION_DIM = 10
JOINT_STATE_ACTION_DIM = 9
STATE_ACTION_MODES = ("ee10", "joint9")
INTERACTION_STATE_NAMES = (
    "contact_probability",
    "handle_progress",
    "door_progress",
)


def normalize_state_action_mode(value: str) -> str:
    mode = str(value).strip().lower()
    mode = {"ee": "ee10", "joint": "joint9", "joint_state9": "joint9"}.get(mode, mode)
    if mode not in STATE_ACTION_MODES:
        raise ValueError(f"Unsupported state/action mode {value!r}; expected {STATE_ACTION_MODES}.")
    return mode


def state_action_dim(mode: str) -> int:
    return JOINT_STATE_ACTION_DIM if normalize_state_action_mode(mode) == "joint9" else EE_STATE_ACTION_DIM


def infer_state_action_mode(controller, requested: str = "auto") -> str:
    requested = str(requested).strip().lower()
    if requested != "auto":
        mode = normalize_state_action_mode(requested)
    else:
        state_dim = int(controller.config.get("state_dim", len(controller.state_feature_names)))
        action_dim = int(controller.action_dim)
        action_frame = str(getattr(controller, "action_frame", "")).lower()
        if state_dim == JOINT_STATE_ACTION_DIM and action_dim == JOINT_STATE_ACTION_DIM:
            mode = "joint9"
        elif state_dim == EE_STATE_ACTION_DIM and action_dim == EE_STATE_ACTION_DIM:
            mode = "ee10"
        elif "joint" in action_frame and action_dim == JOINT_STATE_ACTION_DIM:
            mode = "joint9"
        else:
            raise ValueError(
                "Cannot infer real-deployment state/action schema from checkpoint: "
                f"state_dim={state_dim}, action_dim={action_dim}, action_frame={action_frame!r}. "
                "Pass --z1_state_action_mode explicitly."
            )
    expected = state_action_dim(mode)
    state_dim = int(controller.config.get("state_dim", len(controller.state_feature_names)))
    if int(controller.action_dim) != expected or state_dim != expected:
        raise ValueError(
            f"Checkpoint is incompatible with {mode}: state_dim={state_dim}, "
            f"action_dim={controller.action_dim}, expected both {expected}."
        )
    return mode


def add_repo_paths(repo_root: Path) -> None:
    high_level = repo_root / "high-level"
    candidates = [
        high_level,
        high_level / "dp",
        high_level / "lerobot" / "src",
    ]
    for path in candidates:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


@dataclass(frozen=True)
class DepthCrop:
    left: int = DEFAULT_CROP_LEFT
    right: int = DEFAULT_CROP_RIGHT
    top: int = DEFAULT_CROP_TOP
    bottom: int = DEFAULT_CROP_BOTTOM

    def validate(self, width: int, height: int) -> None:
        values = (self.left, self.right, self.top, self.bottom)
        if any(int(value) < 0 for value in values):
            raise ValueError(f"Depth crop values must be non-negative, got {values}.")
        if self.left + self.right >= int(width):
            raise ValueError(f"Horizontal crop {self.left}+{self.right} must be less than width {width}.")
        if self.top + self.bottom >= int(height):
            raise ValueError(f"Vertical crop {self.top}+{self.bottom} must be less than height {height}.")

    def as_dict(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "right": int(self.right),
            "top": int(self.top),
            "bottom": int(self.bottom),
        }


def crop_and_resize_depth(
    depth_m: np.ndarray,
    crop: DepthCrop,
    output_width: int = DEPTH_WIDTH,
    output_height: int = DEPTH_HEIGHT,
) -> tuple[np.ndarray, tuple[int, int]]:
    import cv2

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth image, got shape {depth.shape}.")
    height, width = depth.shape
    crop.validate(width, height)
    y1 = int(crop.top)
    y2 = height - int(crop.bottom)
    x1 = int(crop.left)
    x2 = width - int(crop.right)
    cropped = depth[y1:y2, x1:x2]
    resized = cv2.resize(
        cropped,
        (int(output_width), int(output_height)),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.asarray(resized, dtype=np.float32), tuple(int(x) for x in cropped.shape)


def depth_m_to_display_u8(
    depth_m: np.ndarray,
    depth_lower_m: float = DEPTH_LOWER_M,
    depth_far_m: float = DEPTH_FAR_M,
) -> np.ndarray:
    depth_lower_m = float(depth_lower_m)
    depth_far_m = float(depth_far_m)
    if not depth_far_m > depth_lower_m:
        raise ValueError(f"depth_far_m must be greater than depth_lower_m, got {depth_lower_m}, {depth_far_m}.")
    depth = np.array(depth_m, dtype=np.float32, copy=True)
    depth = np.nan_to_num(depth, nan=0.0, posinf=depth_far_m, neginf=0.0)
    valid = depth >= depth_lower_m
    depth = np.clip(depth, depth_lower_m, depth_far_m)
    scaled = (depth - depth_lower_m) / max(depth_far_m - depth_lower_m, 1.0e-6)
    scaled[~valid] = 0.0
    return (255.0 * np.clip(scaled, 0.0, 1.0)).astype(np.uint8)


def depth_m_to_policy_u8(
    depth_m: np.ndarray,
    depth_lower_m: float = DEPTH_LOWER_M,
    depth_far_m: float = DEPTH_FAR_M,
) -> np.ndarray:
    u8 = depth_m_to_display_u8(depth_m, depth_lower_m, depth_far_m)
    return np.repeat(u8[..., None], 3, axis=-1)


def make_realsense_filters(
    rs_module,
    enabled: bool,
    fill_holes: bool = True,
    spatial_magnitude: int = DEFAULT_RS_SPATIAL_MAGNITUDE,
    spatial_smooth_delta: int = DEFAULT_RS_SPATIAL_SMOOTH_DELTA,
    spatial_holes_fill: int = DEFAULT_RS_SPATIAL_HOLES_FILL,
) -> list:
    if not enabled:
        return []
    spatial = rs_module.spatial_filter()
    spatial.set_option(rs_module.option.filter_magnitude, int(spatial_magnitude))
    spatial.set_option(rs_module.option.filter_smooth_alpha, 0.75)
    spatial.set_option(rs_module.option.filter_smooth_delta, int(spatial_smooth_delta))
    spatial.set_option(rs_module.option.holes_fill, int(spatial_holes_fill) if fill_holes else 0)
    temporal = rs_module.temporal_filter()
    temporal.set_option(rs_module.option.filter_smooth_alpha, 0.75)
    temporal.set_option(rs_module.option.filter_smooth_delta, 1)
    filters = [spatial]
    if fill_holes:
        filters.append(rs_module.hole_filling_filter())
    filters.append(temporal)
    return filters


def _small_component_mask_from_binary(
    binary: np.ndarray,
    *,
    max_area: int,
    max_span_px: int,
    min_area: int = 1,
    border_margin_px: int = -1,
) -> tuple[np.ndarray, dict[str, int | float]]:
    import cv2

    mask = np.asarray(binary, dtype=np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape[:2]
    selected = np.zeros_like(mask, dtype=np.uint8)
    out: dict[str, int | float] = {
        "total_px": int(mask.sum()),
        "components": max(0, int(n_labels) - 1),
        "selected_components": 0,
        "selected_px": 0,
        "skipped_area": 0,
        "skipped_span": 0,
        "skipped_border": 0,
    }
    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area) or area > int(max_area):
            out["skipped_area"] = int(out["skipped_area"]) + 1
            continue
        if w > int(max_span_px) or h > int(max_span_px):
            out["skipped_span"] = int(out["skipped_span"]) + 1
            continue
        if (
            x <= int(border_margin_px)
            or y <= int(border_margin_px)
            or x + w >= width - int(border_margin_px)
            or y + h >= height - int(border_margin_px)
        ):
            out["skipped_border"] = int(out["skipped_border"]) + 1
            continue
        selected[labels == label] = 255
        out["selected_components"] = int(out["selected_components"]) + 1
        out["selected_px"] = int(out["selected_px"]) + area
    out["selected_ratio"] = float(out["selected_px"]) / max(float(out["total_px"]), 1.0)
    return selected, out


def _opencv_k_add_large_component_fringe(
    depth_u8: np.ndarray,
    base_mask: np.ndarray,
    *,
    small_max_area: int = DEFAULT_OPENCV_K_SMALL_MAX_AREA,
    small_max_span_px: int = DEFAULT_OPENCV_K_SMALL_MAX_SPAN_PX,
    fringe_px: int = 0,
) -> tuple[np.ndarray, dict[str, int]]:
    import cv2

    out = np.asarray(base_mask, dtype=np.uint8).copy()
    fringe_px = int(fringe_px)
    stats = {
        "large_components": 0,
        "fringe_px": fringe_px,
        "fringe_added_px": 0,
        "black_mask_px": int(np.count_nonzero(out)),
    }
    if fringe_px <= 0:
        return out, stats
    black = (np.asarray(depth_u8, dtype=np.uint8) == 0).astype(np.uint8)
    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    for label in range(1, n_labels):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        width = int(cc_stats[label, cv2.CC_STAT_WIDTH])
        height = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
        if area <= int(small_max_area) and width <= int(small_max_span_px) and height <= int(small_max_span_px):
            continue
        component = (labels == label).astype(np.uint8)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 3)
        fringe = ((distance > 0.0) & (distance <= float(fringe_px))).astype(np.uint8) * 255
        stats["fringe_added_px"] += int(np.count_nonzero((fringe > 0) & (out == 0)))
        out = np.maximum(out, fringe)
        stats["large_components"] += 1
    stats["black_mask_px"] = int(np.count_nonzero(out))
    return out, stats


def _opencv_k_small_white_hole_mask(
    depth_u8: np.ndarray,
    *,
    white_threshold: int = DEFAULT_OPENCV_K_WHITE_HOLE_THRESHOLD,
    max_area: int = DEFAULT_OPENCV_K_WHITE_HOLE_MAX_AREA,
    max_span_px: int = DEFAULT_OPENCV_K_WHITE_HOLE_MAX_SPAN_PX,
    border_margin_px: int = DEFAULT_OPENCV_K_WHITE_HOLE_BORDER_MARGIN_PX,
    ring_radius_px: int = DEFAULT_OPENCV_K_WHITE_HOLE_RING_RADIUS_PX,
    min_gray_ring_px: int = DEFAULT_OPENCV_K_WHITE_HOLE_MIN_GRAY_RING_PX,
    min_gray_ring_ratio: float = DEFAULT_OPENCV_K_WHITE_HOLE_MIN_GRAY_RING_RATIO,
) -> tuple[np.ndarray, dict[str, int | float]]:
    import cv2

    image = np.asarray(depth_u8, dtype=np.uint8)
    white = (image >= int(white_threshold)).astype(np.uint8)
    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    height, width = white.shape[:2]
    selected = np.zeros_like(white, dtype=np.uint8)
    out: dict[str, int | float] = {
        "white_threshold": int(white_threshold),
        "total_white_px": int(white.sum()),
        "components": max(0, int(n_labels) - 1),
        "selected_components": 0,
        "selected_px": 0,
        "skipped_area": 0,
        "skipped_span": 0,
        "skipped_border": 0,
        "skipped_ring": 0,
    }
    kernel_radius = max(1, int(ring_radius_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * kernel_radius + 1, 2 * kernel_radius + 1),
    )
    for label in range(1, n_labels):
        x = int(cc_stats[label, cv2.CC_STAT_LEFT])
        y = int(cc_stats[label, cv2.CC_STAT_TOP])
        w = int(cc_stats[label, cv2.CC_STAT_WIDTH])
        h = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > int(max_area):
            out["skipped_area"] = int(out["skipped_area"]) + 1
            continue
        if w > int(max_span_px) or h > int(max_span_px):
            out["skipped_span"] = int(out["skipped_span"]) + 1
            continue
        if (
            x <= int(border_margin_px)
            or y <= int(border_margin_px)
            or x + w >= width - int(border_margin_px)
            or y + h >= height - int(border_margin_px)
        ):
            out["skipped_border"] = int(out["skipped_border"]) + 1
            continue
        component = labels == label
        dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & ~component
        ring_count = int(ring.sum())
        gray_ring = ring & (image > 0) & (image < int(white_threshold))
        gray_count = int(gray_ring.sum())
        if (
            ring_count <= 0
            or gray_count < int(min_gray_ring_px)
            or gray_count / max(ring_count, 1) < float(min_gray_ring_ratio)
        ):
            out["skipped_ring"] = int(out["skipped_ring"]) + 1
            continue
        selected[component] = 255
        out["selected_components"] = int(out["selected_components"]) + 1
        out["selected_px"] = int(out["selected_px"]) + area
    out["selected_ratio_of_white"] = float(out["selected_px"]) / max(float(out["total_white_px"]), 1.0)
    return selected, out


def _opencv_k_inpaint_policy_gray_impl(
    depth_u8: np.ndarray,
    *,
    camera_name: str,
    small_max_area: int,
    small_max_span_px: int,
    wrist_fringe_px: int,
    front_fringe_px: int,
    white_hole_max_area: int,
    white_hole_max_span_px: int,
    white_hole_border_margin_px: int,
    white_hole_ring_radius_px: int,
    white_hole_min_gray_ring_px: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    import cv2

    image = np.asarray(depth_u8, dtype=np.uint8)
    black_mask, black_stats = _small_component_mask_from_binary(
        image == 0,
        max_area=small_max_area,
        max_span_px=small_max_span_px,
        min_area=1,
        border_margin_px=DEFAULT_OPENCV_K_SMALL_BORDER_MARGIN_PX,
    )
    name = str(camera_name).lower()
    fringe_px = front_fringe_px if name == "front" else wrist_fringe_px
    black_mask, fringe_stats = _opencv_k_add_large_component_fringe(
        image,
        black_mask,
        small_max_area=small_max_area,
        small_max_span_px=small_max_span_px,
        fringe_px=fringe_px,
    )
    white_mask, white_stats = _opencv_k_small_white_hole_mask(
        image,
        max_area=white_hole_max_area,
        max_span_px=white_hole_max_span_px,
        border_margin_px=white_hole_border_margin_px,
        ring_radius_px=white_hole_ring_radius_px,
        min_gray_ring_px=white_hole_min_gray_ring_px,
    )
    combined_mask = np.maximum(black_mask, white_mask)
    if not np.any(combined_mask):
        return image.copy(), {
            "camera": name,
            "fringe_px": int(fringe_px),
            "black_px": int(np.count_nonzero(image == 0)),
            "white_px": int(np.count_nonzero(image >= DEFAULT_OPENCV_K_WHITE_HOLE_THRESHOLD)),
            "black_selected_px": 0,
            "fringe_added_px": 0,
            "white_selected_px": 0,
            "combined_mask_px": 0,
            "changed_px": 0,
        }
    inpainted = cv2.inpaint(image, combined_mask, 3.0, cv2.INPAINT_TELEA)
    return inpainted, {
        "camera": name,
        "fringe_px": int(fringe_px),
        "black_px": int(np.count_nonzero(image == 0)),
        "white_px": int(np.count_nonzero(image >= DEFAULT_OPENCV_K_WHITE_HOLE_THRESHOLD)),
        "black_selected_px": int(black_stats.get("selected_px", 0)),
        "fringe_added_px": int(fringe_stats.get("fringe_added_px", 0)),
        "white_selected_px": int(white_stats.get("selected_px", 0)),
        "combined_mask_px": int(np.count_nonzero(combined_mask)),
        "changed_px": int(np.count_nonzero(inpainted != image)),
    }


def opencv_k_inpaint_policy_gray(
    depth_u8: np.ndarray,
    *,
    camera_name: str,
    process_scale: float = DEFAULT_OPENCV_K_REALTIME_PROCESS_SCALE,
) -> tuple[np.ndarray, dict[str, int | float]]:
    import cv2

    image = np.asarray(depth_u8, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = float(process_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"process_scale must be in (0, 1], got {process_scale}")
    if scale >= 0.999:
        out, stats = _opencv_k_inpaint_policy_gray_impl(
            image,
            camera_name=camera_name,
            small_max_area=DEFAULT_OPENCV_K_SMALL_MAX_AREA,
            small_max_span_px=DEFAULT_OPENCV_K_SMALL_MAX_SPAN_PX,
            wrist_fringe_px=DEFAULT_OPENCV_K_WRIST_FRINGE_PX,
            front_fringe_px=DEFAULT_OPENCV_K_FRONT_FRINGE_PX,
            white_hole_max_area=DEFAULT_OPENCV_K_WHITE_HOLE_MAX_AREA,
            white_hole_max_span_px=DEFAULT_OPENCV_K_WHITE_HOLE_MAX_SPAN_PX,
            white_hole_border_margin_px=DEFAULT_OPENCV_K_WHITE_HOLE_BORDER_MARGIN_PX,
            white_hole_ring_radius_px=DEFAULT_OPENCV_K_WHITE_HOLE_RING_RADIUS_PX,
            white_hole_min_gray_ring_px=DEFAULT_OPENCV_K_WHITE_HOLE_MIN_GRAY_RING_PX,
        )
        stats = dict(stats)
        stats["process_scale"] = 1.0
        stats["process_shape_hw"] = [height, width]
        stats["fringe_px_fullres"] = (
            DEFAULT_OPENCV_K_FRONT_FRINGE_PX if str(camera_name).lower() == "front" else DEFAULT_OPENCV_K_WRIST_FRINGE_PX
        )
        stats["fringe_px_process"] = stats.get("fringe_px", stats["fringe_px_fullres"])
        return out, stats

    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    small = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    area_scale = scale * scale
    small_out, stats = _opencv_k_inpaint_policy_gray_impl(
        small,
        camera_name=camera_name,
        small_max_area=max(1, int(round(DEFAULT_OPENCV_K_SMALL_MAX_AREA * area_scale))),
        small_max_span_px=max(1, int(round(DEFAULT_OPENCV_K_SMALL_MAX_SPAN_PX * scale))),
        wrist_fringe_px=max(1, int(round(DEFAULT_OPENCV_K_WRIST_FRINGE_PX * scale))),
        front_fringe_px=max(1, int(round(DEFAULT_OPENCV_K_FRONT_FRINGE_PX * scale))),
        white_hole_max_area=max(1, int(round(DEFAULT_OPENCV_K_WHITE_HOLE_MAX_AREA * area_scale))),
        white_hole_max_span_px=max(1, int(round(DEFAULT_OPENCV_K_WHITE_HOLE_MAX_SPAN_PX * scale))),
        white_hole_border_margin_px=max(1, int(round(DEFAULT_OPENCV_K_WHITE_HOLE_BORDER_MARGIN_PX * scale))),
        white_hole_ring_radius_px=max(1, int(round(DEFAULT_OPENCV_K_WHITE_HOLE_RING_RADIUS_PX * scale))),
        white_hole_min_gray_ring_px=max(1, int(round(DEFAULT_OPENCV_K_WHITE_HOLE_MIN_GRAY_RING_PX * area_scale))),
    )
    out = cv2.resize(small_out, (width, height), interpolation=cv2.INTER_LINEAR)
    stats = dict(stats)
    stats["process_scale"] = scale
    stats["process_shape_hw"] = [scaled_height, scaled_width]
    stats["fringe_px_fullres"] = (
        DEFAULT_OPENCV_K_FRONT_FRINGE_PX if str(camera_name).lower() == "front" else DEFAULT_OPENCV_K_WRIST_FRINGE_PX
    )
    stats["fringe_px_process"] = stats.get("fringe_px", 0)
    stats["changed_px_fullres"] = int(np.count_nonzero(out != image))
    return out, stats


def validate_depth_inpaint_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in DEPTH_INPAINT_MODES:
        raise ValueError(f"Unsupported depth inpaint mode {mode!r}; expected one of {DEPTH_INPAINT_MODES}.")
    return normalized


def is_opencv_k_depth_mode(mode: str) -> bool:
    return validate_depth_inpaint_mode(mode) in ("opencv_k", "opencv_k_no_rs")


def validate_gaussian_blur_args(ksize: int, sigma: float) -> tuple[int, float]:
    k = int(ksize)
    s = float(sigma)
    if k < 0:
        raise ValueError("--depth_gaussian_blur_ksize must be non-negative.")
    if k > 0 and k % 2 == 0:
        raise ValueError("--depth_gaussian_blur_ksize must be odd when enabled.")
    if s < 0:
        raise ValueError("--depth_gaussian_blur_sigma must be non-negative.")
    return k, s


def maybe_gaussian_blur_depth_u8(
    depth_u8: np.ndarray,
    *,
    ksize: int,
    sigma: float,
) -> tuple[np.ndarray, dict[str, int | float | None]]:
    import cv2

    k, s = validate_gaussian_blur_args(ksize, sigma)
    if k <= 0:
        return np.asarray(depth_u8, dtype=np.uint8), {"enabled": False, "ksize": 0, "sigma": None}
    image = np.asarray(depth_u8, dtype=np.uint8)
    blurred = cv2.GaussianBlur(image, (k, k), s, borderType=cv2.BORDER_DEFAULT)
    delta = np.abs(blurred.astype(np.int16) - image.astype(np.int16))
    return blurred, {
        "enabled": True,
        "ksize": int(k),
        "sigma": float(s),
        "changed_px": int(np.count_nonzero(delta)),
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": int(delta.max()) if delta.size else 0,
    }


def save_depth_debug_images(
    out_dir: Path,
    wrist_policy_depth: np.ndarray,
    front_policy_depth: np.ndarray,
    meta: dict,
    debug_frames: dict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    depth_clip_m = meta.get("depth_clip_m", [DEPTH_LOWER_M, DEPTH_FAR_M])
    depth_lower_m, depth_far_m = float(depth_clip_m[0]), float(depth_clip_m[1])
    cv2.imwrite(str(out_dir / "wrist_policy_depth_u8.png"), wrist_policy_depth)
    cv2.imwrite(str(out_dir / "front_policy_depth_u8.png"), front_policy_depth)
    wrist_gray = wrist_policy_depth[..., 0] if wrist_policy_depth.ndim == 3 else wrist_policy_depth
    front_gray = front_policy_depth[..., 0] if front_policy_depth.ndim == 3 else front_policy_depth
    cv2.imwrite(str(out_dir / "wrist_policy_depth_colormap.png"), cv2.applyColorMap(wrist_gray, cv2.COLORMAP_JET))
    cv2.imwrite(str(out_dir / "front_policy_depth_colormap.png"), cv2.applyColorMap(front_gray, cv2.COLORMAP_JET))
    if debug_frames:
        for name, frames in debug_frames.items():
            aligned_depth_m = frames.get("aligned_depth_m")
            raw_cropped_depth_m = frames.get("raw_cropped_depth_m")
            nearest_depth_m = frames.get("nearest_depth_m")
            inpainted_cropped_depth_m = frames.get("inpainted_cropped_depth_m")
            processed_depth_m = frames.get("processed_depth_m")
            invalid_mask = frames.get("invalid_mask")
            fillable_mask = frames.get("fillable_mask")
            color_rgb = frames.get("color_rgb")
            color_crop_rgb = frames.get("color_crop_rgb")
            crop = frames.get("crop")
            if aligned_depth_m is not None:
                aligned_depth_m = np.asarray(aligned_depth_m, dtype=np.float32)
                aligned_mm = np.clip(aligned_depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
                cv2.imwrite(str(out_dir / f"{name}_aligned_depth_mm.png"), aligned_mm)
                aligned_display = depth_m_to_policy_u8(
                    aligned_depth_m,
                    depth_lower_m,
                    depth_far_m,
                )
                cv2.imwrite(str(out_dir / f"{name}_01_original_aligned_depth_display_u8.png"), aligned_display)
            if raw_cropped_depth_m is not None:
                raw_cropped_depth_m = np.asarray(raw_cropped_depth_m, dtype=np.float32)
                raw_cropped_mm = np.clip(raw_cropped_depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
                cv2.imwrite(str(out_dir / f"{name}_04_raw_cropped_depth_mm.png"), raw_cropped_mm)
                cv2.imwrite(
                    str(out_dir / f"{name}_04_raw_cropped_depth_display_u8.png"),
                    depth_m_to_policy_u8(raw_cropped_depth_m, depth_lower_m, depth_far_m),
                )
            if invalid_mask is not None:
                cv2.imwrite(
                    str(out_dir / f"{name}_05_raw_invalid_mask.png"),
                    np.asarray(invalid_mask, dtype=np.uint8) * 255,
                )
            if fillable_mask is not None:
                cv2.imwrite(
                    str(out_dir / f"{name}_06_inpaint_fillable_mask.png"),
                    np.asarray(fillable_mask, dtype=np.uint8) * 255,
                )
            if nearest_depth_m is not None:
                nearest_depth_m = np.asarray(nearest_depth_m, dtype=np.float32)
                nearest_mm = np.clip(nearest_depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
                cv2.imwrite(str(out_dir / f"{name}_07_nearest_initialized_depth_mm.png"), nearest_mm)
                cv2.imwrite(
                    str(out_dir / f"{name}_07_nearest_initialized_depth_display_u8.png"),
                    depth_m_to_policy_u8(nearest_depth_m, depth_lower_m, depth_far_m),
                )
            if inpainted_cropped_depth_m is not None:
                inpainted_cropped_depth_m = np.asarray(inpainted_cropped_depth_m, dtype=np.float32)
                inpainted_mm = np.clip(inpainted_cropped_depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
                cv2.imwrite(str(out_dir / f"{name}_08_rgb_guided_inpainted_depth_mm.png"), inpainted_mm)
                cv2.imwrite(
                    str(out_dir / f"{name}_08_rgb_guided_inpainted_depth_display_u8.png"),
                    depth_m_to_policy_u8(inpainted_cropped_depth_m, depth_lower_m, depth_far_m),
                )
                if raw_cropped_depth_m is not None:
                    difference_mm = np.abs(inpainted_cropped_depth_m - raw_cropped_depth_m) * 1000.0
                    difference_u16 = np.clip(difference_mm, 0.0, 65535.0).astype(np.uint16)
                    cv2.imwrite(str(out_dir / f"{name}_09_inpaint_difference_mm.png"), difference_u16)
                    difference_u8 = (255.0 * np.clip(difference_mm / 500.0, 0.0, 1.0)).astype(np.uint8)
                    cv2.imwrite(
                        str(out_dir / f"{name}_09_inpaint_difference_colormap.png"),
                        cv2.applyColorMap(difference_u8, cv2.COLORMAP_TURBO),
                    )
            if processed_depth_m is not None:
                processed_depth_m = np.asarray(processed_depth_m, dtype=np.float32)
                processed_mm = np.clip(processed_depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
                cv2.imwrite(str(out_dir / f"{name}_cropped_resized_depth_mm.png"), processed_mm)
                processed_display = depth_m_to_policy_u8(
                    processed_depth_m,
                    depth_lower_m,
                    depth_far_m,
                )
                cv2.imwrite(str(out_dir / f"{name}_02_cropped_resized_depth_display_u8.png"), processed_display)
            policy_depth = frames.get("policy_u8")
            if policy_depth is None:
                policy_depth = wrist_policy_depth if name == "wrist" else front_policy_depth
            cv2.imwrite(str(out_dir / f"{name}_03_policy_normalized_depth_u8.png"), policy_depth)
            if color_rgb is not None:
                color_bgr = cv2.cvtColor(np.asarray(color_rgb), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / f"{name}_color.png"), color_bgr)
                if crop is not None:
                    crop_view = color_bgr.copy()
                    height, width = crop_view.shape[:2]
                    x1 = int(crop.left)
                    x2 = width - int(crop.right) - 1
                    y1 = int(crop.top)
                    y2 = height - int(crop.bottom) - 1
                    cv2.rectangle(crop_view, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.imwrite(str(out_dir / f"{name}_color_crop_roi.png"), crop_view)
            if color_crop_rgb is not None:
                cv2.imwrite(
                    str(out_dir / f"{name}_color_crop.png"),
                    cv2.cvtColor(np.asarray(color_crop_rgb), cv2.COLOR_RGB2BGR),
                )
    with (out_dir / "depth_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


class AsyncPolicyDepthVideoRecorder:
    """Asynchronously record the exact uint8 depth images fed to the policy.

    The inference loop only copies the two small uint8 frames into a bounded
    queue. Encoding happens in a background thread; if the encoder falls behind,
    frames are dropped instead of delaying robot control.
    """

    def __init__(
        self,
        out_dir: Path,
        fps: float,
        queue_size: int = 128,
        codec: str = "mp4v",
        separate: bool = False,
        video_stem: str = "policy_depth_u8",
        metadata_stem: str = "policy_depth_video",
        layout: str = "side_by_side_left_wrist_right_front",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.fps = float(fps)
        self.queue_size = int(queue_size)
        self.codec = str(codec)
        self.separate = bool(separate)
        self.video_stem = str(video_stem)
        self.metadata_stem = str(metadata_stem)
        self.layout = str(layout)
        self.queue: queue.Queue | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.meta_lock = threading.Lock()
        self.enqueued = 0
        self.written = 0
        self.dropped = 0
        self.error: str | None = None

    @staticmethod
    def _gray_u8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"policy depth image must be HxW or HxWxC, got {arr.shape}.")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _bgr(gray: np.ndarray) -> np.ndarray:
        return np.repeat(gray[..., None], 3, axis=-1)

    def start(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"Policy depth video fps must be positive, got {self.fps}.")
        if self.queue_size <= 0:
            raise ValueError(f"Policy depth video queue size must be positive, got {self.queue_size}.")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=self.queue_size)
        self.thread = threading.Thread(target=self._loop, name="policy-depth-video", daemon=True)
        self.thread.start()

    def submit(
        self,
        step: int,
        wall_time: float,
        wrist_policy_depth: np.ndarray,
        front_policy_depth: np.ndarray,
    ) -> dict:
        if self.queue is None:
            raise RuntimeError("AsyncPolicyDepthVideoRecorder.start() was not called.")
        try:
            wrist = self._gray_u8(wrist_policy_depth).copy()
            front = self._gray_u8(front_policy_depth).copy()
        except Exception as exc:
            with self.meta_lock:
                self.error = f"copy_failed:{exc}"
            return self.stats(extra={"queued": False, "drop_reason": "copy_failed"})
        item = {
            "step": int(step),
            "wall_time": float(wall_time),
            "wrist": wrist,
            "front": front,
        }
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            with self.meta_lock:
                self.dropped += 1
                enqueued = self.enqueued
                written = self.written
                dropped = self.dropped
                error = self.error
            return {
                "enabled": True,
                "queued": False,
                "drop_reason": "queue_full",
                "enqueued": enqueued,
                "written": written,
                "dropped": dropped,
                "error": error,
            }
        with self.meta_lock:
            self.enqueued += 1
            enqueued = self.enqueued
            written = self.written
            dropped = self.dropped
            error = self.error
        return {
            "enabled": True,
            "queued": True,
            "enqueued": enqueued,
            "written": written,
            "dropped": dropped,
            "error": error,
        }

    def stats(self, extra: dict | None = None) -> dict:
        with self.meta_lock:
            out = {
                "enabled": True,
                "dir": str(self.out_dir),
                "fps": self.fps,
                "enqueued": self.enqueued,
                "written": self.written,
                "dropped": self.dropped,
                "error": self.error,
            }
        if extra:
            out.update(extra)
        return out

    def stop(self, timeout_s: float = 5.0) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=float(timeout_s))
        return self.stats(extra={"thread_alive": bool(self.thread and self.thread.is_alive())})

    def _open_writer(self, path: Path, size: tuple[int, int]):
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*self.codec[:4])
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, size, True)
        if not writer.isOpened():
            fallback_path = path.with_suffix(".avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(fallback_path), fourcc, self.fps, size, True)
            path = fallback_path
        if not writer.isOpened():
            raise RuntimeError(f"failed to open policy depth video writer: {path}")
        return writer, path

    def _loop(self) -> None:
        writers = {}
        paths = {}
        meta_file = None
        try:
            meta_file = (self.out_dir / f"{self.metadata_stem}_frames.jsonl").open("w", encoding="utf-8")
            while not self.stop_event.is_set() or (self.queue is not None and not self.queue.empty()):
                try:
                    item = self.queue.get(timeout=0.1) if self.queue is not None else None
                except queue.Empty:
                    continue
                if item is None:
                    continue
                wrist = np.asarray(item["wrist"], dtype=np.uint8)
                front = np.asarray(item["front"], dtype=np.uint8)
                if wrist.shape != front.shape:
                    raise ValueError(f"wrist/front policy depth shapes differ: {wrist.shape} vs {front.shape}")
                h, w = wrist.shape
                if "side_by_side" not in writers:
                    writers["side_by_side"], paths["side_by_side"] = self._open_writer(
                        self.out_dir / f"{self.video_stem}_side_by_side.mp4",
                        (w * 2, h),
                    )
                    if self.separate:
                        writers["wrist"], paths["wrist"] = self._open_writer(
                            self.out_dir / f"wrist_{self.video_stem}.mp4",
                            (w, h),
                        )
                        writers["front"], paths["front"] = self._open_writer(
                            self.out_dir / f"front_{self.video_stem}.mp4",
                            (w, h),
                        )
                    with self.meta_lock:
                        self.error = None
                combined = np.concatenate([wrist, front], axis=1)
                writers["side_by_side"].write(self._bgr(combined))
                if self.separate:
                    writers["wrist"].write(self._bgr(wrist))
                    writers["front"].write(self._bgr(front))
                meta_file.write(
                    json.dumps(
                        {
                            "step": int(item["step"]),
                            "wall_time": float(item["wall_time"]),
                            "layout": self.layout,
                            "shape_hw": [int(h), int(w)],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                with self.meta_lock:
                    self.written += 1
        except Exception as exc:
            with self.meta_lock:
                self.error = str(exc)
        finally:
            if meta_file is not None:
                meta_file.close()
            for writer in writers.values():
                writer.release()
            if paths:
                try:
                    (self.out_dir / f"{self.metadata_stem}_manifest.json").write_text(
                        json.dumps(
                            {
                                "fps": self.fps,
                                "codec_requested": self.codec,
                                "layout": self.layout,
                                "video_stem": self.video_stem,
                                "videos": {key: str(path) for key, path in paths.items()},
                                "separate": self.separate,
                                "stats": self.stats(),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    with self.meta_lock:
                        self.error = f"manifest_failed:{exc}"


class DummyDepthPair:
    def __init__(
        self,
        value_m: float = 0.0,
        depth_lower_m: float = DEPTH_LOWER_M,
        depth_far_m: float = DEPTH_FAR_M,
    ) -> None:
        self.value_m = float(value_m)
        self.depth_lower_m = float(depth_lower_m)
        self.depth_far_m = float(depth_far_m)
        self.depth = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), self.value_m, dtype=np.float32)
        self.last_raw_unfiltered_depth_u8: tuple[np.ndarray, np.ndarray] | None = None

    def start(self) -> None:
        pass

    def read(self, capture_debug: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        now = time.time()
        image = depth_m_to_policy_u8(self.depth, self.depth_lower_m, self.depth_far_m)
        self.last_raw_unfiltered_depth_u8 = (image[..., 0].copy(), image[..., 0].copy())
        return image, image.copy(), {
            "wrist_ts": now,
            "front_ts": now,
            "dt_ms": 0.0,
            "backend": "dummy",
            "depth_clip_m": [self.depth_lower_m, self.depth_far_m],
        }

    def raw_unfiltered_depth_u8(self) -> tuple[np.ndarray, np.ndarray]:
        if self.last_raw_unfiltered_depth_u8 is None:
            image = depth_m_to_policy_u8(self.depth, self.depth_lower_m, self.depth_far_m)[..., 0]
            return image.copy(), image.copy()
        return tuple(image.copy() for image in self.last_raw_unfiltered_depth_u8)

    def debug_snapshots(self) -> dict[str, dict[str, np.ndarray]]:
        return {}

    def stop(self) -> None:
        pass


class AsyncRealSenseDepthCamera:
    """Continuously update the latest D435 depth frame in a background thread."""

    def __init__(
        self,
        serial: str,
        name: str,
        use_filters: bool = True,
        warmup_frames: int = 10,
        align_depth_to_color: bool = True,
        crop: DepthCrop = DepthCrop(),
        depth_lower_m: float = DEPTH_LOWER_M,
        depth_far_m: float = DEPTH_FAR_M,
        spatial_magnitude: int = DEFAULT_RS_SPATIAL_MAGNITUDE,
        depth_inpaint_mode: str = DEFAULT_DEPTH_INPAINT_MODE,
        depth_gaussian_blur_ksize: int = DEFAULT_DEPTH_GAUSSIAN_BLUR_KSIZE,
        depth_gaussian_blur_sigma: float = DEFAULT_DEPTH_GAUSSIAN_BLUR_SIGMA,
    ) -> None:
        if not serial:
            raise ValueError(f"{name} RealSense serial is empty.")
        import pyrealsense2 as rs

        self.rs = rs
        self.serial = str(serial)
        self.name = str(name)
        self.use_filters = bool(use_filters)
        self.warmup_frames = max(0, int(warmup_frames))
        self.align_depth_to_color = bool(align_depth_to_color)
        self.crop = crop
        self.depth_lower_m = float(depth_lower_m)
        self.depth_far_m = float(depth_far_m)
        self.spatial_magnitude = int(spatial_magnitude)
        self.depth_inpaint_mode = validate_depth_inpaint_mode(depth_inpaint_mode)
        self.depth_gaussian_blur_ksize, self.depth_gaussian_blur_sigma = validate_gaussian_blur_args(
            depth_gaussian_blur_ksize,
            depth_gaussian_blur_sigma,
        )
        self.crop.validate(DEPTH_WIDTH, DEPTH_HEIGHT)
        self.pipeline = None
        self.align = None
        self.scale = 0.001
        self.filters = []
        self.device_info: dict[str, str] = {}
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_depth_m: np.ndarray | None = None
        self.latest_aligned_depth_m: np.ndarray | None = None
        self.latest_policy_u8: np.ndarray | None = None
        self.latest_raw_unfiltered_policy_u8: np.ndarray | None = None
        self.latest_color_rgb: np.ndarray | None = None
        self.latest_raw_cropped_depth_m: np.ndarray | None = None
        self.latest_inpaint_input_depth_m: np.ndarray | None = None
        self.latest_color_crop_rgb: np.ndarray | None = None
        self.latest_invalid_mask: np.ndarray | None = None
        self.latest_output_invalid_mask: np.ndarray | None = None
        self.latest_opencv_k_stats: dict[str, int | float] | None = None
        self.latest_crop_shape: tuple[int, int] | None = None
        self.latest_timestamp_s: float | None = None
        self.latest_arrival_s: float | None = None
        self.frame_count = 0
        self.error: str = ""

    def _make_filters(self) -> list:
        return make_realsense_filters(
            self.rs,
            self.use_filters,
            fill_holes=self.depth_inpaint_mode == "realsense",
            spatial_magnitude=self.spatial_magnitude,
        )

    def start(self) -> None:
        self.pipeline = self.rs.pipeline()
        cfg = self.rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(
            self.rs.stream.depth,
            DEPTH_WIDTH,
            DEPTH_HEIGHT,
            self.rs.format.z16,
            DEPTH_FPS,
        )
        if self.align_depth_to_color:
            cfg.enable_stream(
                self.rs.stream.color,
                DEPTH_WIDTH,
                DEPTH_HEIGHT,
                self.rs.format.rgb8,
                DEPTH_FPS,
            )
        profile = self.pipeline.start(cfg)
        device = profile.get_device()
        sensor = device.first_depth_sensor()
        self.scale = float(sensor.get_depth_scale())
        self.device_info = self._device_info(device)
        self.align = self.rs.align(self.rs.stream.color) if self.align_depth_to_color else None
        self.filters = self._make_filters()
        for _ in range(self.warmup_frames):
            frames = self.pipeline.wait_for_frames(5000)
            if self.align is not None:
                frames = self.align.process(frames)
            if not frames.get_depth_frame():
                continue
        self.thread = threading.Thread(target=self._loop, name=f"rs-{self.name}", daemon=True)
        self.thread.start()

    def _device_info(self, device) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, label in (
            (self.rs.camera_info.name, "name"),
            (self.rs.camera_info.serial_number, "serial"),
            (self.rs.camera_info.firmware_version, "firmware"),
            (self.rs.camera_info.usb_type_descriptor, "usb"),
            (self.rs.camera_info.physical_port, "physical_port"),
            (self.rs.camera_info.product_line, "product_line"),
        ):
            try:
                if device.supports(key):
                    out[label] = device.get_info(key)
            except Exception:
                pass
        return out

    def _loop(self) -> None:
        import cv2

        assert self.pipeline is not None
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(5000)
                if self.align is not None:
                    frames = self.align.process(frames)
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                color_frame = frames.get_color_frame() if self.align_depth_to_color else None
                aligned_depth_raw = np.asanyarray(depth_frame.get_data())
                aligned_depth_m = aligned_depth_raw.astype(np.float32) * self.scale
                raw_invalid_mask = ~np.isfinite(aligned_depth_m) | (aligned_depth_m <= 0.0)
                for rs_filter in self.filters:
                    depth_frame = rs_filter.process(depth_frame)
                depth_raw = np.asanyarray(depth_frame.get_data())
                filtered_aligned_depth_m = depth_raw.astype(np.float32) * self.scale
                height, width = filtered_aligned_depth_m.shape
                y1, y2 = int(self.crop.top), height - int(self.crop.bottom)
                x1, x2 = int(self.crop.left), width - int(self.crop.right)
                raw_cropped_depth_m = aligned_depth_m[y1:y2, x1:x2]
                raw_resized_depth_m, _ = crop_and_resize_depth(aligned_depth_m, self.crop)
                raw_unfiltered_policy_u8 = depth_m_to_display_u8(
                    raw_resized_depth_m,
                    self.depth_lower_m,
                    self.depth_far_m,
                )
                inpaint_input_depth_m = filtered_aligned_depth_m[y1:y2, x1:x2]
                invalid_mask = raw_invalid_mask[y1:y2, x1:x2]
                output_invalid_mask = cv2.resize(
                    invalid_mask.astype(np.uint8),
                    (DEPTH_WIDTH, DEPTH_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                depth_m, crop_shape = crop_and_resize_depth(filtered_aligned_depth_m, self.crop)
                policy_u8 = depth_m_to_policy_u8(depth_m, self.depth_lower_m, self.depth_far_m)
                opencv_k_stats = None
                gaussian_blur_stats = None
                policy_gray = policy_u8[..., 0]
                if is_opencv_k_depth_mode(self.depth_inpaint_mode):
                    policy_gray, opencv_k_stats = opencv_k_inpaint_policy_gray(
                        policy_gray,
                        camera_name=self.name,
                    )
                policy_gray, gaussian_blur_stats = maybe_gaussian_blur_depth_u8(
                    policy_gray,
                    ksize=self.depth_gaussian_blur_ksize,
                    sigma=self.depth_gaussian_blur_sigma,
                )
                if opencv_k_stats is not None:
                    opencv_k_stats = dict(opencv_k_stats)
                    opencv_k_stats["gaussian_blur"] = gaussian_blur_stats
                policy_u8 = np.repeat(policy_gray[..., None], 3, axis=-1)
                color_rgb = np.asanyarray(color_frame.get_data()).copy() if color_frame else None
                color_crop_rgb = (
                    color_rgb[y1:y2, x1:x2]
                    if color_rgb is not None
                    else np.zeros((*inpaint_input_depth_m.shape, 3), dtype=np.uint8)
                )
                timestamp_s = float(depth_frame.get_timestamp()) / 1000.0
                arrival_s = time.time()
                with self.lock:
                    self.latest_depth_m = depth_m
                    self.latest_aligned_depth_m = aligned_depth_m
                    self.latest_policy_u8 = policy_u8
                    self.latest_raw_unfiltered_policy_u8 = raw_unfiltered_policy_u8
                    self.latest_color_rgb = color_rgb
                    self.latest_raw_cropped_depth_m = raw_cropped_depth_m.copy()
                    self.latest_inpaint_input_depth_m = inpaint_input_depth_m.copy()
                    self.latest_color_crop_rgb = color_crop_rgb.copy()
                    self.latest_invalid_mask = invalid_mask.copy()
                    self.latest_output_invalid_mask = output_invalid_mask.copy()
                    self.latest_opencv_k_stats = None if opencv_k_stats is None else dict(opencv_k_stats)
                    self.latest_crop_shape = crop_shape
                    self.latest_timestamp_s = timestamp_s
                    self.latest_arrival_s = arrival_s
                    self.frame_count += 1
                    self.error = ""
            except Exception as exc:
                with self.lock:
                    self.error = repr(exc)
                time.sleep(0.01)

    def read_latest(self, timeout_s: float = 2.0) -> tuple[np.ndarray, dict]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            with self.lock:
                if self.latest_policy_u8 is not None:
                    now = time.time()
                    depth_m = self.latest_depth_m
                    valid = depth_m[np.isfinite(depth_m)] if depth_m is not None else np.asarray([], dtype=np.float32)
                    positive = valid[valid > 0.0]
                    meta = {
                        "serial": self.serial,
                        "name": self.name,
                        "timestamp_s": self.latest_timestamp_s,
                        "arrival_s": self.latest_arrival_s,
                        "age_ms": None if self.latest_arrival_s is None else (now - self.latest_arrival_s) * 1000.0,
                        "frame_count": self.frame_count,
                        "depth_scale": self.scale,
                        "filters": self.use_filters,
                        "depth_inpaint_mode": self.depth_inpaint_mode,
                        "depth_gaussian_blur": {
                            "enabled": self.depth_gaussian_blur_ksize > 0,
                            "ksize": int(self.depth_gaussian_blur_ksize),
                            "sigma": float(self.depth_gaussian_blur_sigma),
                        },
                        "spatial_magnitude": self.spatial_magnitude if self.use_filters else 0,
                        "worker_mode": "thread",
                        "align_depth_to": "color" if self.align_depth_to_color else "none",
                        "crop": self.crop.as_dict(),
                        "crop_shape_hw": list(self.latest_crop_shape) if self.latest_crop_shape else None,
                        "output_shape_hw": [DEPTH_HEIGHT, DEPTH_WIDTH],
                        "depth_clip_m": [self.depth_lower_m, self.depth_far_m],
                        "device": self.device_info,
                        "depth_min_m": float(positive.min()) if positive.size else 0.0,
                        "depth_max_m": float(positive.max()) if positive.size else 0.0,
                        "depth_mean_m": float(positive.mean()) if positive.size else 0.0,
                        "opencv_k": None if self.latest_opencv_k_stats is None else dict(self.latest_opencv_k_stats),
                        "error": self.error,
                    }
                    return self.latest_policy_u8.copy(), meta
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {self.name} RealSense depth frame.")
            time.sleep(0.002)

    def read_latest_with_raw(
        self,
        timeout_s: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        policy, meta = self.read_latest(timeout_s=timeout_s)
        with self.lock:
            if self.latest_raw_unfiltered_policy_u8 is None:
                raise RuntimeError(f"{self.name} raw unfiltered depth is unavailable.")
            raw = self.latest_raw_unfiltered_policy_u8.copy()
        return policy, raw, meta

    def read_latest_inputs(self, timeout_s: float = 2.0) -> tuple[dict[str, np.ndarray], dict]:
        _, meta = self.read_latest(timeout_s=timeout_s)
        with self.lock:
            if (
                self.latest_inpaint_input_depth_m is None
                or self.latest_color_crop_rgb is None
                or self.latest_invalid_mask is None
                or self.latest_output_invalid_mask is None
                or self.latest_raw_unfiltered_policy_u8 is None
            ):
                raise RuntimeError(f"{self.name} RealSense inpaint inputs are unavailable.")
            meta = dict(meta)
            meta["timestamp_s"] = self.latest_timestamp_s
            meta["arrival_s"] = self.latest_arrival_s
            meta["age_ms"] = (
                None if self.latest_arrival_s is None else (time.time() - self.latest_arrival_s) * 1000.0
            )
            meta["frame_count"] = self.frame_count
            return {
                "raw_cropped_depth_m": self.latest_raw_cropped_depth_m.copy(),
                "inpaint_input_depth_m": self.latest_inpaint_input_depth_m.copy(),
                "color_crop_rgb": self.latest_color_crop_rgb.copy(),
                "invalid_mask": self.latest_invalid_mask.copy(),
                "output_invalid_mask": self.latest_output_invalid_mask.copy(),
                "base_policy_u8": self.latest_policy_u8[..., 0].copy(),
                "raw_unfiltered_policy_u8": self.latest_raw_unfiltered_policy_u8.copy(),
            }, meta

    def debug_snapshot(self, timeout_s: float = 2.0) -> dict[str, np.ndarray | DepthCrop]:
        self.read_latest(timeout_s=timeout_s)
        with self.lock:
            out: dict[str, np.ndarray | DepthCrop] = {"crop": self.crop}
            if self.latest_aligned_depth_m is not None:
                out["aligned_depth_m"] = self.latest_aligned_depth_m.copy()
            if self.latest_depth_m is not None:
                out["processed_depth_m"] = self.latest_depth_m.copy()
            if self.latest_policy_u8 is not None:
                out["policy_u8"] = self.latest_policy_u8.copy()
            if self.latest_color_rgb is not None:
                out["color_rgb"] = self.latest_color_rgb.copy()
            if self.latest_raw_cropped_depth_m is not None:
                out["raw_cropped_depth_m"] = self.latest_raw_cropped_depth_m.copy()
            if self.latest_color_crop_rgb is not None:
                out["color_crop_rgb"] = self.latest_color_crop_rgb.copy()
            if self.latest_invalid_mask is not None:
                out["invalid_mask"] = self.latest_invalid_mask.copy()
            return out

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.pipeline is not None:
            self.pipeline.stop()


def _realsense_process_worker(
    serial: str,
    name: str,
    use_filters: bool,
    warmup_frames: int,
    align_depth_to_color: bool,
    crop_values: tuple[int, int, int, int],
    depth_lower_m: float,
    depth_far_m: float,
    spatial_magnitude: int,
    depth_inpaint_mode: str,
    depth_gaussian_blur_ksize: int,
    depth_gaussian_blur_sigma: float,
    policy_buffer,
    raw_unfiltered_policy_buffer,
    processed_depth_buffer,
    aligned_depth_buffer,
    color_buffer,
    raw_cropped_depth_buffer,
    inpaint_input_depth_buffer,
    color_crop_buffer,
    invalid_mask_buffer,
    output_invalid_mask_buffer,
    data_lock,
    timestamp_value,
    arrival_value,
    frame_count_value,
    has_frame_value,
    depth_scale_value,
    stop_event,
    status_queue,
) -> None:
    # Ctrl+C belongs to the parent inference process. The parent sets this
    # worker's shared stop event and joins it cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import cv2
    import pyrealsense2 as rs

    crop = DepthCrop(*crop_values)
    depth_gaussian_blur_ksize, depth_gaussian_blur_sigma = validate_gaussian_blur_args(
        depth_gaussian_blur_ksize,
        depth_gaussian_blur_sigma,
    )
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
    if align_depth_to_color:
        cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.rgb8, DEPTH_FPS)

    try:
        profile = pipeline.start(cfg)
        device = profile.get_device()
        depth_scale = float(device.first_depth_sensor().get_depth_scale())
        depth_scale_value.value = depth_scale
        device_info: dict[str, str] = {}
        for key, label in (
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial"),
            (rs.camera_info.firmware_version, "firmware"),
            (rs.camera_info.usb_type_descriptor, "usb"),
            (rs.camera_info.physical_port, "physical_port"),
            (rs.camera_info.product_line, "product_line"),
        ):
            try:
                if device.supports(key):
                    device_info[label] = device.get_info(key)
            except Exception:
                pass

        align = rs.align(rs.stream.color) if align_depth_to_color else None
        filters = make_realsense_filters(
            rs,
            use_filters,
            fill_holes=depth_inpaint_mode == "realsense",
            spatial_magnitude=spatial_magnitude,
        )
        for _ in range(max(0, int(warmup_frames))):
            frames = pipeline.wait_for_frames(5000)
            if align is not None:
                frames = align.process(frames)

        status_queue.put(
            {
                "type": "started",
                "name": name,
                "serial": serial,
                "device": device_info,
            }
        )

        policy_view = np.frombuffer(policy_buffer, dtype=np.uint8).reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 3)
        raw_unfiltered_policy_view = np.frombuffer(
            raw_unfiltered_policy_buffer,
            dtype=np.uint8,
        ).reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
        processed_view = np.frombuffer(processed_depth_buffer, dtype=np.float32).reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
        aligned_view = np.frombuffer(aligned_depth_buffer, dtype=np.float32).reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
        color_view = np.frombuffer(color_buffer, dtype=np.uint8).reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 3)
        crop_height = DEPTH_HEIGHT - crop.top - crop.bottom
        crop_width = DEPTH_WIDTH - crop.left - crop.right
        raw_cropped_view = np.frombuffer(raw_cropped_depth_buffer, dtype=np.float32).reshape(
            crop_height, crop_width
        )
        inpaint_input_view = np.frombuffer(inpaint_input_depth_buffer, dtype=np.float32).reshape(
            crop_height, crop_width
        )
        color_crop_view = np.frombuffer(color_crop_buffer, dtype=np.uint8).reshape(
            crop_height, crop_width, 3
        )
        invalid_mask_view = np.frombuffer(invalid_mask_buffer, dtype=np.uint8).reshape(
            crop_height, crop_width
        )
        output_invalid_mask_view = np.frombuffer(output_invalid_mask_buffer, dtype=np.uint8).reshape(
            DEPTH_HEIGHT, DEPTH_WIDTH
        )
        local_frame_count = 0

        while not stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(5000)
                if align is not None:
                    frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                color_frame = frames.get_color_frame() if align_depth_to_color else None
                aligned_depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                raw_invalid_mask = ~np.isfinite(aligned_depth_m) | (aligned_depth_m <= 0.0)
                for rs_filter in filters:
                    depth_frame = rs_filter.process(depth_frame)
                filtered_depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

                height, width = filtered_depth_m.shape
                y1, y2 = int(crop.top), height - int(crop.bottom)
                x1, x2 = int(crop.left), width - int(crop.right)
                raw_cropped_depth_m = aligned_depth_m[y1:y2, x1:x2]
                raw_resized_depth_m = cv2.resize(
                    raw_cropped_depth_m,
                    (DEPTH_WIDTH, DEPTH_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
                raw_unfiltered_policy_u8 = depth_m_to_display_u8(
                    raw_resized_depth_m,
                    depth_lower_m,
                    depth_far_m,
                )
                inpaint_input_depth_m = filtered_depth_m[y1:y2, x1:x2]
                invalid_mask = raw_invalid_mask[y1:y2, x1:x2]
                output_invalid_mask = cv2.resize(
                    invalid_mask.astype(np.uint8),
                    (DEPTH_WIDTH, DEPTH_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )
                processed_depth_m = cv2.resize(
                    inpaint_input_depth_m,
                    (DEPTH_WIDTH, DEPTH_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
                policy_u8 = depth_m_to_policy_u8(processed_depth_m, depth_lower_m, depth_far_m)
                opencv_k_stats = None
                gaussian_blur_stats = None
                policy_gray = policy_u8[..., 0]
                if is_opencv_k_depth_mode(depth_inpaint_mode):
                    policy_gray, opencv_k_stats = opencv_k_inpaint_policy_gray(
                        policy_gray,
                        camera_name=name,
                    )
                policy_gray, gaussian_blur_stats = maybe_gaussian_blur_depth_u8(
                    policy_gray,
                    ksize=depth_gaussian_blur_ksize,
                    sigma=depth_gaussian_blur_sigma,
                )
                if opencv_k_stats is not None:
                    opencv_k_stats = dict(opencv_k_stats)
                    opencv_k_stats["gaussian_blur"] = gaussian_blur_stats
                policy_u8 = np.repeat(policy_gray[..., None], 3, axis=-1)
                color_rgb = (
                    np.asanyarray(color_frame.get_data())
                    if color_frame
                    else np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH, 3), dtype=np.uint8)
                )
                color_crop_rgb = color_rgb[y1:y2, x1:x2]
                timestamp_s = float(depth_frame.get_timestamp()) / 1000.0
                arrival_s = time.time()
                local_frame_count += 1

                with data_lock:
                    policy_view[...] = policy_u8
                    raw_unfiltered_policy_view[...] = raw_unfiltered_policy_u8
                    # ACT needs policy_u8 every frame.  The larger raw/color
                    # buffers are only for occasional debug snapshots; copying
                    # all ~3.4MB per camera every frame needlessly consumes NX
                    # memory bandwidth and slows the second camera.
                    publish_inpaint_inputs = depth_inpaint_mode == "rgb_guided"
                    if publish_inpaint_inputs or local_frame_count == 1 or local_frame_count % DEPTH_FPS == 0:
                        raw_cropped_view[...] = raw_cropped_depth_m
                        inpaint_input_view[...] = inpaint_input_depth_m
                        color_crop_view[...] = color_crop_rgb
                        invalid_mask_view[...] = invalid_mask.astype(np.uint8)
                        output_invalid_mask_view[...] = output_invalid_mask
                    if local_frame_count == 1 or local_frame_count % DEPTH_FPS == 0:
                        processed_view[...] = processed_depth_m
                        aligned_view[...] = aligned_depth_m
                        color_view[...] = color_rgb
                    timestamp_value.value = timestamp_s
                    arrival_value.value = arrival_s
                    frame_count_value.value = local_frame_count
                    has_frame_value.value = 1
                if (
                    opencv_k_stats is not None
                    and (local_frame_count == 1 or local_frame_count % DEPTH_FPS == 0)
                ):
                    try:
                        status_queue.put_nowait(
                            {
                                "type": "opencv_k_stats",
                                "name": name,
                                "frame_count": local_frame_count,
                                "opencv_k": opencv_k_stats,
                            }
                        )
                    except queue.Full:
                        pass
            except Exception as exc:
                try:
                    status_queue.put_nowait({"type": "error", "error": repr(exc)})
                except queue.Full:
                    pass
                time.sleep(0.01)
    except Exception as exc:
        try:
            status_queue.put({"type": "startup_error", "error": repr(exc)})
        except Exception:
            pass
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


class AsyncRealSenseDepthProcessCamera:
    """Run one D435 capture/filter pipeline in its own process.

    pyrealsense2 processing filters hold the Python GIL long enough that two
    filtered cameras in threads serialize.  A process per camera keeps the two
    30Hz pipelines independent while exposing the latest frames through shared
    memory.
    """

    def __init__(
        self,
        serial: str,
        name: str,
        use_filters: bool = True,
        warmup_frames: int = 10,
        align_depth_to_color: bool = True,
        crop: DepthCrop = DepthCrop(),
        depth_lower_m: float = DEPTH_LOWER_M,
        depth_far_m: float = DEPTH_FAR_M,
        spatial_magnitude: int = DEFAULT_RS_SPATIAL_MAGNITUDE,
        depth_inpaint_mode: str = DEFAULT_DEPTH_INPAINT_MODE,
        depth_gaussian_blur_ksize: int = DEFAULT_DEPTH_GAUSSIAN_BLUR_KSIZE,
        depth_gaussian_blur_sigma: float = DEFAULT_DEPTH_GAUSSIAN_BLUR_SIGMA,
    ) -> None:
        if not serial:
            raise ValueError(f"{name} RealSense serial is empty.")
        self.serial = str(serial)
        self.name = str(name)
        self.use_filters = bool(use_filters)
        self.warmup_frames = max(0, int(warmup_frames))
        self.align_depth_to_color = bool(align_depth_to_color)
        self.crop = crop
        self.depth_lower_m = float(depth_lower_m)
        self.depth_far_m = float(depth_far_m)
        self.spatial_magnitude = int(spatial_magnitude)
        self.depth_inpaint_mode = validate_depth_inpaint_mode(depth_inpaint_mode)
        self.depth_gaussian_blur_ksize, self.depth_gaussian_blur_sigma = validate_gaussian_blur_args(
            depth_gaussian_blur_ksize,
            depth_gaussian_blur_sigma,
        )
        self.crop.validate(DEPTH_WIDTH, DEPTH_HEIGHT)
        self.crop_height = DEPTH_HEIGHT - self.crop.top - self.crop.bottom
        self.crop_width = DEPTH_WIDTH - self.crop.left - self.crop.right

        # Do not fork after the parent has imported librealsense: inherited SDK
        # state can leave the child pipeline alive but unable to receive frames.
        self.ctx = mp.get_context("spawn")
        self.policy_buffer = self.ctx.RawArray(ctypes.c_uint8, DEPTH_HEIGHT * DEPTH_WIDTH * 3)
        self.raw_unfiltered_policy_buffer = self.ctx.RawArray(
            ctypes.c_uint8,
            DEPTH_HEIGHT * DEPTH_WIDTH,
        )
        self.processed_depth_buffer = self.ctx.RawArray(ctypes.c_float, DEPTH_HEIGHT * DEPTH_WIDTH)
        self.aligned_depth_buffer = self.ctx.RawArray(ctypes.c_float, DEPTH_HEIGHT * DEPTH_WIDTH)
        self.color_buffer = self.ctx.RawArray(ctypes.c_uint8, DEPTH_HEIGHT * DEPTH_WIDTH * 3)
        crop_pixels = self.crop_height * self.crop_width
        self.raw_cropped_depth_buffer = self.ctx.RawArray(ctypes.c_float, crop_pixels)
        self.inpaint_input_depth_buffer = self.ctx.RawArray(ctypes.c_float, crop_pixels)
        self.color_crop_buffer = self.ctx.RawArray(ctypes.c_uint8, crop_pixels * 3)
        self.invalid_mask_buffer = self.ctx.RawArray(ctypes.c_uint8, crop_pixels)
        self.output_invalid_mask_buffer = self.ctx.RawArray(
            ctypes.c_uint8,
            DEPTH_HEIGHT * DEPTH_WIDTH,
        )
        self.data_lock = self.ctx.Lock()
        self.timestamp_value = self.ctx.Value("d", 0.0, lock=False)
        self.arrival_value = self.ctx.Value("d", 0.0, lock=False)
        self.frame_count_value = self.ctx.Value("q", 0, lock=False)
        self.has_frame_value = self.ctx.Value("b", 0, lock=False)
        self.depth_scale_value = self.ctx.Value("d", 0.001, lock=False)
        self.stop_event = self.ctx.Event()
        self.status_queue = self.ctx.Queue(maxsize=8)
        self.process: mp.Process | None = None
        self.device_info: dict[str, str] = {}
        self.latest_opencv_k_stats: dict[str, int | float] | None = None
        self.error = ""

    def _drain_status(self) -> None:
        while True:
            try:
                item = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if item.get("type") == "started":
                self.device_info = dict(item.get("device") or {})
            elif item.get("type") == "opencv_k_stats":
                self.latest_opencv_k_stats = dict(item.get("opencv_k") or {})
            elif item.get("type") in {"error", "startup_error"}:
                self.error = str(item.get("error") or "")

    def start(self) -> None:
        self.process = self.ctx.Process(
            target=_realsense_process_worker,
            args=(
                self.serial,
                self.name,
                self.use_filters,
                self.warmup_frames,
                self.align_depth_to_color,
                (self.crop.left, self.crop.right, self.crop.top, self.crop.bottom),
                self.depth_lower_m,
                self.depth_far_m,
                self.spatial_magnitude,
                self.depth_inpaint_mode,
                self.depth_gaussian_blur_ksize,
                self.depth_gaussian_blur_sigma,
                self.policy_buffer,
                self.raw_unfiltered_policy_buffer,
                self.processed_depth_buffer,
                self.aligned_depth_buffer,
                self.color_buffer,
                self.raw_cropped_depth_buffer,
                self.inpaint_input_depth_buffer,
                self.color_crop_buffer,
                self.invalid_mask_buffer,
                self.output_invalid_mask_buffer,
                self.data_lock,
                self.timestamp_value,
                self.arrival_value,
                self.frame_count_value,
                self.has_frame_value,
                self.depth_scale_value,
                self.stop_event,
                self.status_queue,
            ),
            name=f"rs-process-{self.name}",
            daemon=True,
        )
        self.process.start()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self._drain_status()
            if self.has_frame_value.value:
                return
            if self.process is not None and not self.process.is_alive():
                self._drain_status()
                raise RuntimeError(f"{self.name} RealSense process exited during startup: {self.error}")
            time.sleep(0.01)
        raise TimeoutError(f"Timed out starting {self.name} RealSense process: {self.error}")

    def read_latest(self, timeout_s: float = 2.0) -> tuple[np.ndarray, dict]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            self._drain_status()
            if self.has_frame_value.value:
                with self.data_lock:
                    policy = np.frombuffer(self.policy_buffer, dtype=np.uint8).reshape(
                        DEPTH_HEIGHT, DEPTH_WIDTH, 3
                    ).copy()
                    depth_m = np.frombuffer(self.processed_depth_buffer, dtype=np.float32).reshape(
                        DEPTH_HEIGHT, DEPTH_WIDTH
                    ).copy()
                    timestamp_s = float(self.timestamp_value.value)
                    arrival_s = float(self.arrival_value.value)
                    frame_count = int(self.frame_count_value.value)
                positive = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
                now = time.time()
                return policy, {
                    "serial": self.serial,
                    "name": self.name,
                    "timestamp_s": timestamp_s,
                    "arrival_s": arrival_s,
                    "age_ms": (now - arrival_s) * 1000.0,
                    "frame_count": frame_count,
                    "depth_scale": float(self.depth_scale_value.value),
                    "filters": self.use_filters,
                    "depth_inpaint_mode": self.depth_inpaint_mode,
                    "depth_gaussian_blur": {
                        "enabled": self.depth_gaussian_blur_ksize > 0,
                        "ksize": int(self.depth_gaussian_blur_ksize),
                        "sigma": float(self.depth_gaussian_blur_sigma),
                    },
                    "spatial_magnitude": self.spatial_magnitude if self.use_filters else 0,
                    "worker_mode": "process",
                    "align_depth_to": "color" if self.align_depth_to_color else "none",
                    "crop": self.crop.as_dict(),
                    "crop_shape_hw": [
                        DEPTH_HEIGHT - self.crop.top - self.crop.bottom,
                        DEPTH_WIDTH - self.crop.left - self.crop.right,
                    ],
                    "output_shape_hw": [DEPTH_HEIGHT, DEPTH_WIDTH],
                    "depth_clip_m": [self.depth_lower_m, self.depth_far_m],
                    "device": self.device_info,
                    "depth_min_m": float(positive.min()) if positive.size else 0.0,
                    "depth_max_m": float(positive.max()) if positive.size else 0.0,
                    "depth_mean_m": float(positive.mean()) if positive.size else 0.0,
                    "opencv_k": None if self.latest_opencv_k_stats is None else dict(self.latest_opencv_k_stats),
                    "error": self.error,
                }
            if self.process is not None and not self.process.is_alive():
                raise RuntimeError(f"{self.name} RealSense process exited: {self.error}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {self.name} RealSense process frame.")
            time.sleep(0.002)

    def read_latest_with_raw(
        self,
        timeout_s: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        policy, meta = self.read_latest(timeout_s=timeout_s)
        with self.data_lock:
            raw = np.frombuffer(
                self.raw_unfiltered_policy_buffer,
                dtype=np.uint8,
            ).reshape(DEPTH_HEIGHT, DEPTH_WIDTH).copy()
        return policy, raw, meta

    def read_latest_inputs(self, timeout_s: float = 2.0) -> tuple[dict[str, np.ndarray], dict]:
        _, meta = self.read_latest(timeout_s=timeout_s)
        with self.data_lock:
            meta = dict(meta)
            meta["timestamp_s"] = float(self.timestamp_value.value)
            meta["arrival_s"] = float(self.arrival_value.value)
            meta["age_ms"] = (time.time() - meta["arrival_s"]) * 1000.0
            meta["frame_count"] = int(self.frame_count_value.value)
            return {
                "raw_cropped_depth_m": np.frombuffer(
                    self.raw_cropped_depth_buffer,
                    dtype=np.float32,
                )
                .reshape(self.crop_height, self.crop_width)
                .copy(),
                "inpaint_input_depth_m": np.frombuffer(
                    self.inpaint_input_depth_buffer,
                    dtype=np.float32,
                )
                .reshape(self.crop_height, self.crop_width)
                .copy(),
                "color_crop_rgb": np.frombuffer(self.color_crop_buffer, dtype=np.uint8)
                .reshape(self.crop_height, self.crop_width, 3)
                .copy(),
                "invalid_mask": np.frombuffer(self.invalid_mask_buffer, dtype=np.uint8)
                .reshape(self.crop_height, self.crop_width)
                .astype(bool),
                "output_invalid_mask": np.frombuffer(
                    self.output_invalid_mask_buffer,
                    dtype=np.uint8,
                )
                .reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
                .astype(bool),
                "base_policy_u8": np.frombuffer(self.policy_buffer, dtype=np.uint8)
                .reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 3)[..., 0]
                .copy(),
                "raw_unfiltered_policy_u8": np.frombuffer(
                    self.raw_unfiltered_policy_buffer,
                    dtype=np.uint8,
                )
                .reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
                .copy(),
            }, meta

    def debug_snapshot(self, timeout_s: float = 2.0) -> dict[str, np.ndarray | DepthCrop]:
        self.read_latest(timeout_s=timeout_s)
        with self.data_lock:
            return {
                "crop": self.crop,
                "aligned_depth_m": np.frombuffer(self.aligned_depth_buffer, dtype=np.float32)
                .reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
                .copy(),
                "processed_depth_m": np.frombuffer(self.processed_depth_buffer, dtype=np.float32)
                .reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
                .copy(),
                "policy_u8": np.frombuffer(self.policy_buffer, dtype=np.uint8)
                .reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 3)
                .copy(),
                "color_rgb": np.frombuffer(self.color_buffer, dtype=np.uint8)
                .reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 3)
                .copy(),
                "raw_cropped_depth_m": np.frombuffer(
                    self.raw_cropped_depth_buffer,
                    dtype=np.float32,
                )
                .reshape(self.crop_height, self.crop_width)
                .copy(),
                "color_crop_rgb": np.frombuffer(self.color_crop_buffer, dtype=np.uint8)
                .reshape(self.crop_height, self.crop_width, 3)
                .copy(),
                "invalid_mask": np.frombuffer(self.invalid_mask_buffer, dtype=np.uint8)
                .reshape(self.crop_height, self.crop_width)
                .astype(bool),
            }

    def stop(self) -> None:
        self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
        try:
            self.status_queue.close()
        except Exception:
            pass


class RealSenseDepthPair:
    def __init__(
        self,
        wrist_serial: str,
        front_serial: str,
        allow_single_duplicate: bool = False,
        use_filters: bool = True,
        warmup_frames: int = 10,
        align_depth_to_color: bool = True,
        crop: DepthCrop = DepthCrop(),
        depth_lower_m: float = DEPTH_LOWER_M,
        depth_far_m: float = DEPTH_FAR_M,
        worker_mode: str = "auto",
        spatial_magnitude: int = DEFAULT_RS_SPATIAL_MAGNITUDE,
        depth_inpaint_mode: str = DEFAULT_DEPTH_INPAINT_MODE,
        depth_inpaint_device: str = "cuda:0",
        depth_inpaint_max_distance_px: float = DEFAULT_DEPTH_INPAINT_MAX_DISTANCE_PX,
        depth_inpaint_iterations: int = DEFAULT_DEPTH_INPAINT_ITERATIONS,
        depth_inpaint_rgb_sigma: float = DEFAULT_DEPTH_INPAINT_RGB_SIGMA,
        depth_gaussian_blur_ksize: int = DEFAULT_DEPTH_GAUSSIAN_BLUR_KSIZE,
        depth_gaussian_blur_sigma: float = DEFAULT_DEPTH_GAUSSIAN_BLUR_SIGMA,
    ) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.allow_single_duplicate = bool(allow_single_duplicate)
        self.use_filters = bool(use_filters)
        self.warmup_frames = int(warmup_frames)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.crop = crop
        self.depth_lower_m = float(depth_lower_m)
        self.depth_far_m = float(depth_far_m)
        self.spatial_magnitude = int(spatial_magnitude)
        self.depth_inpaint_mode = validate_depth_inpaint_mode(depth_inpaint_mode)
        self.depth_inpaint_device = str(depth_inpaint_device)
        self.depth_inpaint_max_distance_px = float(depth_inpaint_max_distance_px)
        self.depth_inpaint_iterations = int(depth_inpaint_iterations)
        self.depth_inpaint_rgb_sigma = float(depth_inpaint_rgb_sigma)
        self.depth_gaussian_blur_ksize, self.depth_gaussian_blur_sigma = validate_gaussian_blur_args(
            depth_gaussian_blur_ksize,
            depth_gaussian_blur_sigma,
        )
        self.worker_mode = str(worker_mode)
        if self.worker_mode not in {"auto", "thread", "process"}:
            raise ValueError(f"Unsupported RealSense worker mode: {self.worker_mode}")
        self.active_worker_mode = self.worker_mode
        self.single_duplicate = False
        wrist_serial = str(wrist_serial or "")
        front_serial = str(front_serial or "")
        if not wrist_serial and not front_serial:
            devices = self.list_devices()
            if len(devices) == 1 and self.allow_single_duplicate:
                wrist_serial = front_serial = str(devices[0]["serial"])
            else:
                raise ValueError("RealSense mode requires --wrist_serial/--front_serial or one camera with --allow_single_realsense_duplicate.")
        if not wrist_serial or not front_serial:
            only = wrist_serial or front_serial
            if self.allow_single_duplicate and only:
                wrist_serial = front_serial = only
            else:
                raise ValueError("RealSense mode requires both serials unless --allow_single_realsense_duplicate is set.")
        if wrist_serial == front_serial:
            if not self.allow_single_duplicate:
                raise ValueError("wrist/front RealSense serials must be different unless --allow_single_realsense_duplicate is set.")
            self.single_duplicate = True
        self.serials = {"wrist": wrist_serial, "front": front_serial}
        self.cameras: dict[
            str,
            AsyncRealSenseDepthCamera | AsyncRealSenseDepthProcessCamera,
        ] = {}
        self.inpainter = None
        self.last_inpaint_debug: dict[str, dict[str, np.ndarray | DepthCrop]] = {}
        self.last_raw_unfiltered_depth_u8: tuple[np.ndarray, np.ndarray] | None = None

    @staticmethod
    def list_devices() -> list[dict]:
        import pyrealsense2 as rs

        out = []
        for dev in rs.context().query_devices():
            info = {}
            for key, label in (
                (rs.camera_info.name, "name"),
                (rs.camera_info.serial_number, "serial"),
                (rs.camera_info.firmware_version, "firmware"),
                (rs.camera_info.usb_type_descriptor, "usb"),
                (rs.camera_info.physical_port, "physical_port"),
                (rs.camera_info.product_line, "product_line"),
            ):
                try:
                    if dev.supports(key):
                        info[label] = dev.get_info(key)
                except Exception:
                    pass
            out.append(info)
        return out

    def start(self) -> None:
        worker_mode = self.worker_mode
        if worker_mode == "auto":
            worker_mode = (
                "process"
                if self.use_filters or self.depth_inpaint_mode == "rgb_guided" or is_opencv_k_depth_mode(self.depth_inpaint_mode)
                else "thread"
            )
        self.active_worker_mode = worker_mode
        camera_cls = (
            AsyncRealSenseDepthProcessCamera
            if worker_mode == "process"
            else AsyncRealSenseDepthCamera
        )
        if self.single_duplicate:
            cam = camera_cls(
                self.serials["wrist"],
                "single",
                use_filters=self.use_filters,
                warmup_frames=self.warmup_frames,
                align_depth_to_color=self.align_depth_to_color,
                crop=self.crop,
                depth_lower_m=self.depth_lower_m,
                depth_far_m=self.depth_far_m,
                spatial_magnitude=self.spatial_magnitude,
                depth_inpaint_mode=self.depth_inpaint_mode,
                depth_gaussian_blur_ksize=self.depth_gaussian_blur_ksize,
                depth_gaussian_blur_sigma=self.depth_gaussian_blur_sigma,
            )
            cam.start()
            self.cameras["single"] = cam
        else:
            for name, serial in self.serials.items():
                cam = camera_cls(
                    serial,
                    name,
                    use_filters=self.use_filters,
                    warmup_frames=self.warmup_frames,
                    align_depth_to_color=self.align_depth_to_color,
                    crop=self.crop,
                    depth_lower_m=self.depth_lower_m,
                    depth_far_m=self.depth_far_m,
                    spatial_magnitude=self.spatial_magnitude,
                    depth_inpaint_mode=self.depth_inpaint_mode,
                    depth_gaussian_blur_ksize=self.depth_gaussian_blur_ksize,
                    depth_gaussian_blur_sigma=self.depth_gaussian_blur_sigma,
                )
                cam.start()
                self.cameras[name] = cam
        if self.depth_inpaint_mode == "rgb_guided":
            from depth_inpaint import RGBGuidedDepthInpainter

            self.inpainter = RGBGuidedDepthInpainter(
                device=self.depth_inpaint_device,
                max_distance_px=self.depth_inpaint_max_distance_px,
                iterations=self.depth_inpaint_iterations,
                rgb_sigma=self.depth_inpaint_rgb_sigma,
                output_size_hw=(DEPTH_HEIGHT, DEPTH_WIDTH),
                depth_lower_m=self.depth_lower_m,
                depth_far_m=self.depth_far_m,
            )

    def read(self, capture_debug: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.depth_inpaint_mode == "rgb_guided":
            return self._read_rgb_guided(capture_debug=capture_debug)
        if self.single_duplicate:
            depth, raw_unfiltered, meta = self.cameras["single"].read_latest_with_raw()
            self.last_raw_unfiltered_depth_u8 = (
                raw_unfiltered.copy(),
                raw_unfiltered.copy(),
            )
            return depth, depth.copy(), {
                "wrist_ts": meta["timestamp_s"],
                "front_ts": meta["timestamp_s"],
                "dt_ms": 0.0,
                "backend": "realsense_async",
                "worker_mode": meta.get("worker_mode", self.active_worker_mode),
                "single_duplicate": True,
                "wrist": meta,
                "front": meta,
                "resolution": [DEPTH_WIDTH, DEPTH_HEIGHT],
                "fps": DEPTH_FPS,
                "align_depth_to": "color" if self.align_depth_to_color else "none",
                "crop": self.crop.as_dict(),
                "depth_clip_m": [self.depth_lower_m, self.depth_far_m],
                "spatial_magnitude": self.spatial_magnitude if self.use_filters else 0,
                "depth_inpaint_mode": self.depth_inpaint_mode,
                "depth_gaussian_blur": {
                    "enabled": self.depth_gaussian_blur_ksize > 0,
                    "ksize": int(self.depth_gaussian_blur_ksize),
                    "sigma": float(self.depth_gaussian_blur_sigma),
                },
            }
        wrist, wrist_raw_unfiltered, wrist_meta = self.cameras["wrist"].read_latest_with_raw()
        front, front_raw_unfiltered, front_meta = self.cameras["front"].read_latest_with_raw()
        self.last_raw_unfiltered_depth_u8 = (
            wrist_raw_unfiltered,
            front_raw_unfiltered,
        )
        wrist_ts = wrist_meta["timestamp_s"] or 0.0
        front_ts = front_meta["timestamp_s"] or 0.0
        wrist_arrival = wrist_meta["arrival_s"] or 0.0
        front_arrival = front_meta["arrival_s"] or 0.0
        return wrist, front, {
            "wrist_ts": wrist_ts,
            "front_ts": front_ts,
            # Independent D435 hardware clocks are not guaranteed to share an
            # epoch.  Use host arrival times for cross-camera freshness.
            "dt_ms": abs(wrist_arrival - front_arrival) * 1000.0,
            "device_timestamp_dt_ms": abs(wrist_ts - front_ts) * 1000.0,
            "backend": "realsense_async",
            "worker_mode": wrist_meta.get("worker_mode", self.active_worker_mode),
            "single_duplicate": False,
            "wrist": wrist_meta,
            "front": front_meta,
            "resolution": [DEPTH_WIDTH, DEPTH_HEIGHT],
            "fps": DEPTH_FPS,
            "align_depth_to": "color" if self.align_depth_to_color else "none",
            "crop": self.crop.as_dict(),
            "depth_clip_m": [self.depth_lower_m, self.depth_far_m],
            "spatial_magnitude": self.spatial_magnitude if self.use_filters else 0,
            "depth_inpaint_mode": self.depth_inpaint_mode,
            "depth_gaussian_blur": {
                "enabled": self.depth_gaussian_blur_ksize > 0,
                "ksize": int(self.depth_gaussian_blur_ksize),
                "sigma": float(self.depth_gaussian_blur_sigma),
            },
        }

    def _read_rgb_guided(self, capture_debug: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.inpainter is None:
            raise RuntimeError("RGB-guided inpainter has not been initialized; call start() first.")
        names = ["single"] if self.single_duplicate else ["wrist", "front"]
        packets = []
        metas = []
        for name in names:
            packet, meta = self.cameras[name].read_latest_inputs()
            packets.append(packet)
            metas.append(meta)
        if self.single_duplicate:
            raw_unfiltered = packets[0]["raw_unfiltered_policy_u8"]
            self.last_raw_unfiltered_depth_u8 = (
                raw_unfiltered.copy(),
                raw_unfiltered.copy(),
            )
        else:
            self.last_raw_unfiltered_depth_u8 = (
                packets[0]["raw_unfiltered_policy_u8"].copy(),
                packets[1]["raw_unfiltered_policy_u8"].copy(),
            )
        depth_batch = np.stack([packet["inpaint_input_depth_m"] for packet in packets], axis=0)
        color_batch = np.stack([packet["color_crop_rgb"] for packet in packets], axis=0)
        invalid_batch = np.stack([packet["invalid_mask"] for packet in packets], axis=0)
        base_policy_batch = np.stack([packet["base_policy_u8"] for packet in packets], axis=0)
        output_invalid_batch = np.stack(
            [packet["output_invalid_mask"] for packet in packets],
            axis=0,
        )
        result = self.inpainter.process(
            depth_batch,
            color_batch,
            invalid_mask=invalid_batch,
            base_policy_u8=base_policy_batch,
            base_invalid_mask=output_invalid_batch,
            return_torch_policy=not capture_debug,
            return_debug=capture_debug,
        )

        logical_names = ["wrist"] if self.single_duplicate else ["wrist", "front"]
        if capture_debug:
            if (
                result.nearest_depth_m is None
                or result.inpainted_depth_m is None
                or result.resized_depth_m is None
            ):
                raise RuntimeError("RGB-guided debug capture did not return intermediate depth arrays.")
            self.last_inpaint_debug = {}
            for index, logical_name in enumerate(logical_names):
                packet = packets[index]
                self.last_inpaint_debug[logical_name] = {
                    "crop": self.crop,
                    "raw_cropped_depth_m": packet["raw_cropped_depth_m"],
                    "color_crop_rgb": packet["color_crop_rgb"],
                    "invalid_mask": result.invalid_mask[index],
                    "fillable_mask": result.fillable_mask[index],
                    "nearest_depth_m": result.nearest_depth_m[index],
                    "inpainted_cropped_depth_m": result.inpainted_depth_m[index],
                    "processed_depth_m": result.resized_depth_m[index],
                    "policy_u8": result.policy_u8[index],
                }

        inpaint_stats = dict(result.stats)
        per_camera = list(inpaint_stats.pop("per_camera"))
        copy_policy = lambda image: image.clone() if hasattr(image, "clone") else image.copy()
        if self.single_duplicate:
            wrist_policy = result.policy_u8[0]
            front_policy = copy_policy(wrist_policy)
            wrist_meta = dict(metas[0])
            front_meta = dict(metas[0])
            per_named = {"wrist": per_camera[0], "front": dict(per_camera[0])}
            if capture_debug:
                self.last_inpaint_debug["front"] = {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in self.last_inpaint_debug["wrist"].items()
                }
        else:
            wrist_policy, front_policy = result.policy_u8[0], result.policy_u8[1]
            wrist_meta, front_meta = dict(metas[0]), dict(metas[1])
            per_named = {"wrist": per_camera[0], "front": per_camera[1]}
        wrist_meta["inpaint"] = per_named["wrist"]
        front_meta["inpaint"] = per_named["front"]
        wrist_ts = wrist_meta["timestamp_s"] or 0.0
        front_ts = front_meta["timestamp_s"] or 0.0
        wrist_arrival = wrist_meta["arrival_s"] or 0.0
        front_arrival = front_meta["arrival_s"] or 0.0
        inpaint_stats["per_camera"] = per_named
        return copy_policy(wrist_policy), copy_policy(front_policy), {
            "wrist_ts": wrist_ts,
            "front_ts": front_ts,
            "dt_ms": abs(wrist_arrival - front_arrival) * 1000.0,
            "device_timestamp_dt_ms": abs(wrist_ts - front_ts) * 1000.0,
            "backend": "realsense_async",
            "worker_mode": wrist_meta.get("worker_mode", self.active_worker_mode),
            "single_duplicate": self.single_duplicate,
            "wrist": wrist_meta,
            "front": front_meta,
            "resolution": [DEPTH_WIDTH, DEPTH_HEIGHT],
            "fps": DEPTH_FPS,
            "align_depth_to": "color",
            "crop": self.crop.as_dict(),
            "depth_clip_m": [self.depth_lower_m, self.depth_far_m],
            "spatial_magnitude": self.spatial_magnitude if self.use_filters else 0,
            "depth_inpaint_mode": self.depth_inpaint_mode,
            "inpaint": inpaint_stats,
        }

    def raw_unfiltered_depth_u8(self) -> tuple[np.ndarray, np.ndarray]:
        if self.last_raw_unfiltered_depth_u8 is None:
            raise RuntimeError("Raw unfiltered depth is unavailable before the first camera read.")
        wrist, front = self.last_raw_unfiltered_depth_u8
        return wrist.copy(), front.copy()

    def debug_snapshots(self) -> dict[str, dict[str, np.ndarray | DepthCrop]]:
        if self.single_duplicate:
            debug = self.cameras["single"].debug_snapshot()
            snapshots = {"wrist": debug, "front": debug.copy()}
        else:
            snapshots = {
            "wrist": self.cameras["wrist"].debug_snapshot(),
            "front": self.cameras["front"].debug_snapshot(),
        }
        for name, inpaint_debug in self.last_inpaint_debug.items():
            snapshots.setdefault(name, {}).update(inpaint_debug)
        return snapshots

    def stop(self) -> None:
        for cam in self.cameras.values():
            cam.stop()


def zero_state(state_action_mode: str = "ee10") -> np.ndarray:
    mode = normalize_state_action_mode(state_action_mode)
    state = np.zeros(state_action_dim(mode), dtype=np.float32)
    # State convention: [last_vx, last_vyaw, ee_xyz, ee_quat_xyzw, gripper].
    # If the Z1 state stream is not available yet, an identity quaternion is a
    # safer fallback than an all-zero quaternion.  Runtime should normally
    # overwrite state[2:10] from Z1ActStateReceiver before inference.
    if mode == "ee10":
        state[8] = 1.0
    return state


def quaternion_angle_error_deg(target_xyzw: np.ndarray, actual_xyzw: np.ndarray) -> float:
    target = np.asarray(target_xyzw, dtype=np.float64).reshape(4)
    actual = np.asarray(actual_xyzw, dtype=np.float64).reshape(4)
    target_norm = float(np.linalg.norm(target))
    actual_norm = float(np.linalg.norm(actual))
    if (
        not np.isfinite(target_norm)
        or not np.isfinite(actual_norm)
        or target_norm < 1.0e-9
        or actual_norm < 1.0e-9
    ):
        return float("nan")
    target /= target_norm
    actual /= actual_norm
    dot = float(np.clip(abs(np.dot(target, actual)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def compute_arm_tracking_error(
    actual_state: np.ndarray,
    target_action: np.ndarray | None,
    z1_state_meta: dict | None,
    *,
    target_step: int | None,
    current_step: int,
    target_age_s: float | None,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    gripper_tolerance_rad: float,
    state_action_mode: str = "ee10",
    joint_tolerance_rad: float = 0.10,
) -> dict:
    mode = normalize_state_action_mode(state_action_mode)
    meta = z1_state_meta or {}
    valid_feedback = bool(
        target_action is not None
        and not bool(meta.get("stale", True))
        and int(meta.get("count", 0) or 0) > 0
    )
    result = {
        "valid": False,
        "feedback_age_s": meta.get("age_s"),
        "target_step": target_step,
        "lag_steps": None if target_step is None else int(current_step - target_step),
        "target_age_s": target_age_s,
        "ik_ok": meta.get("ik_ok"),
        "ik_source": meta.get("ik_source"),
        "ik_fail_count": meta.get("ik_fail_count"),
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_deg": float(orientation_tolerance_deg),
        "gripper_tolerance_rad": float(gripper_tolerance_rad),
        "state_action_mode": mode,
        "joint_tolerance_rad": float(joint_tolerance_rad),
        "reached": False,
    }
    if not valid_feedback:
        return result

    actual = np.asarray(actual_state, dtype=np.float64).reshape(-1)
    target = np.asarray(target_action, dtype=np.float64).reshape(-1)
    expected_dim = state_action_dim(mode)
    if actual.shape[0] < expected_dim or target.shape[0] < expected_dim:
        return result
    if mode == "joint9":
        actual_values = actual[:JOINT_STATE_ACTION_DIM]
        target_values = target[:JOINT_STATE_ACTION_DIM]
        if not (np.isfinite(actual_values).all() and np.isfinite(target_values).all()):
            return result
        joint_error = target[2:8] - actual[2:8]
        joint_abs_error = np.abs(joint_error)
        joint_max_abs_error_rad = float(np.max(joint_abs_error))
        joint_rmse_rad = float(np.sqrt(np.mean(np.square(joint_error))))
        gripper_error_rad = float(target[8] - actual[8])
        gripper_abs_error_rad = abs(gripper_error_rad)
        result.update(
            {
                "valid": True,
                "actual_q": np.round(actual[2:8], 6).tolist(),
                "target_q": np.round(target[2:8], 6).tolist(),
                "joint_error_rad": np.round(joint_error, 6).tolist(),
                "joint_abs_error_rad": np.round(joint_abs_error, 6).tolist(),
                "joint_max_abs_error_rad": joint_max_abs_error_rad,
                "joint_rmse_rad": joint_rmse_rad,
                "actual_gripper": round(float(actual[8]), 6),
                "target_gripper": round(float(target[8]), 6),
                "gripper_error_rad": gripper_error_rad,
                "gripper_abs_error_rad": gripper_abs_error_rad,
                "reached": bool(
                    joint_max_abs_error_rad <= float(joint_tolerance_rad)
                    and gripper_abs_error_rad <= float(gripper_tolerance_rad)
                ),
            }
        )
        return result
    actual_pose = actual[2:10]
    target_pose = target[2:10]
    if not (np.isfinite(actual_pose).all() and np.isfinite(target_pose).all()):
        return result

    position_error_xyz = target[2:5] - actual[2:5]
    position_error_m = float(np.linalg.norm(position_error_xyz))
    orientation_error_deg = quaternion_angle_error_deg(target[5:9], actual[5:9])
    gripper_error_rad = float(target[9] - actual[9])
    gripper_abs_error_rad = abs(gripper_error_rad)
    reached = bool(
        np.isfinite(orientation_error_deg)
        and position_error_m <= float(position_tolerance_m)
        and orientation_error_deg <= float(orientation_tolerance_deg)
        and gripper_abs_error_rad <= float(gripper_tolerance_rad)
    )
    result.update(
        {
            "valid": True,
            "actual_ee_pos": np.round(actual[2:5], 6).tolist(),
            "actual_ee_quat_xyzw": np.round(actual[5:9], 6).tolist(),
            "actual_gripper": round(float(actual[9]), 6),
            "target_ee_pos": np.round(target[2:5], 6).tolist(),
            "target_ee_quat_xyzw": np.round(target[5:9], 6).tolist(),
            "target_gripper": round(float(target[9]), 6),
            "position_error_xyz_m": np.round(position_error_xyz, 6).tolist(),
            "position_error_m": position_error_m,
            "position_error_mm": position_error_m * 1000.0,
            "orientation_error_deg": orientation_error_deg,
            "gripper_error_rad": gripper_error_rad,
            "gripper_abs_error_rad": gripper_abs_error_rad,
            "reached": reached,
        }
    )
    return result


def shortest_path_slerp_xyzw(
    old_quat_xyzw: np.ndarray,
    new_quat_xyzw: np.ndarray,
    new_weight: float,
) -> np.ndarray:
    """SLERP from old to new using the shortest quaternion arc."""
    old = np.asarray(old_quat_xyzw, dtype=np.float64).reshape(4)
    new = np.asarray(new_quat_xyzw, dtype=np.float64).reshape(4)
    old_norm = float(np.linalg.norm(old))
    new_norm = float(np.linalg.norm(new))
    if not np.isfinite(old_norm) or old_norm < 1.0e-9:
        old = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        old /= old_norm
    if not np.isfinite(new_norm) or new_norm < 1.0e-9:
        new = old.copy()
    else:
        new /= new_norm

    dot = float(np.dot(old, new))
    if dot < 0.0:
        new = -new
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))
    t = float(np.clip(new_weight, 0.0, 1.0))
    if dot > 0.9995:
        blended = (1.0 - t) * old + t * new
    else:
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        blended = (
            np.sin((1.0 - t) * theta) / sin_theta * old
            + np.sin(t * theta) / sin_theta * new
        )
    blended /= max(float(np.linalg.norm(blended)), 1.0e-9)
    if blended[3] < 0.0:
        blended = -blended
    return blended.astype(np.float32)


def blend_ee_actions(
    old_action: np.ndarray,
    new_action: np.ndarray,
    *,
    old_weight: float = 0.3,
    new_weight: float = 0.7,
) -> np.ndarray:
    """Blend overlapping 10D ACT EE actions without corrupting quaternions."""
    old = np.asarray(old_action, dtype=np.float32).reshape(-1)
    new = np.asarray(new_action, dtype=np.float32).reshape(-1)
    if old.shape[0] < 10 or new.shape[0] < 10:
        raise ValueError(f"EE action blending requires 10D actions, got {old.shape} and {new.shape}.")
    total = float(old_weight) + float(new_weight)
    if total <= 0.0:
        raise ValueError("EE action blend weights must have a positive sum.")
    old_alpha = float(old_weight) / total
    new_alpha = float(new_weight) / total
    blended = new[:10].copy()
    # Base vx/vyaw and EE xyz are ordinary Euclidean values.
    blended[0:5] = old_alpha * old[0:5] + new_alpha * new[0:5]
    # EE orientation must remain on S^3 and follow the shortest rotation arc.
    blended[5:9] = shortest_path_slerp_xyzw(old[5:9], new[5:9], new_alpha)
    # Gripper is intentionally latest-only; blending can delay grasp/release.
    blended[9] = new[9]
    return blended


def blend_joint_actions(
    old_action: np.ndarray,
    new_action: np.ndarray,
    *,
    old_weight: float = 0.3,
    new_weight: float = 0.7,
) -> np.ndarray:
    """Blend vx/vyaw/q1..q6 and keep the newest gripper command."""
    old = np.asarray(old_action, dtype=np.float32).reshape(-1)
    new = np.asarray(new_action, dtype=np.float32).reshape(-1)
    if old.shape[0] < 9 or new.shape[0] < 9:
        raise ValueError(f"Joint action blending requires 9D actions, got {old.shape} and {new.shape}.")
    total = float(old_weight) + float(new_weight)
    if total <= 0.0:
        raise ValueError("Joint action blend weights must have a positive sum.")
    old_alpha = float(old_weight) / total
    new_alpha = float(new_weight) / total
    blended = new[:9].copy()
    blended[0:8] = old_alpha * old[0:8] + new_alpha * new[0:8]
    blended[8] = new[8]
    return blended


def blend_actions(
    old_action: np.ndarray,
    new_action: np.ndarray,
    *,
    state_action_mode: str,
    old_weight: float = 0.3,
    new_weight: float = 0.7,
) -> np.ndarray:
    if normalize_state_action_mode(state_action_mode) == "joint9":
        return blend_joint_actions(old_action, new_action, old_weight=old_weight, new_weight=new_weight)
    return blend_ee_actions(old_action, new_action, old_weight=old_weight, new_weight=new_weight)


@dataclass
class TimedEEAction:
    timestep: int
    action: np.ndarray
    source: str = "chunk"
    blend_count: int = 1
    interaction_state: np.ndarray | None = None
    interaction_chunk_ids: tuple[str, ...] = ()


class EEActionOverlapBuffer:
    """Timestamped action buffer with schema-aware old/new chunk aggregation."""

    def __init__(
        self,
        old_weight: float = 0.3,
        new_weight: float = 0.7,
        state_action_mode: str = "ee10",
    ) -> None:
        self.old_weight = float(old_weight)
        self.new_weight = float(new_weight)
        self.state_action_mode = normalize_state_action_mode(state_action_mode)
        self.action_dim = state_action_dim(self.state_action_mode)
        self._queue: deque[TimedEEAction] = deque()
        self.last_popped_timestep = -1

    def reset(self) -> None:
        self._queue.clear()
        self.last_popped_timestep = -1

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def first_timestep(self) -> int | None:
        return None if not self._queue else int(self._queue[0].timestep)

    @property
    def last_timestep(self) -> int | None:
        return None if not self._queue else int(self._queue[-1].timestep)

    def ingest(
        self,
        actions: list[np.ndarray] | np.ndarray,
        *,
        start_timestep: int,
        current_timestep: int,
        interaction_states: np.ndarray | list[np.ndarray] | None = None,
        chunk_id: str | None = None,
    ) -> dict:
        rows = np.asarray(actions, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] < self.action_dim:
            raise ValueError(
                f"Expected {self.state_action_mode} action chunk with shape "
                f"(T, >={self.action_dim}), got {rows.shape}."
            )
        interaction_rows = None
        if interaction_states is not None:
            interaction_rows = np.asarray(interaction_states, dtype=np.float32)
            if (
                interaction_rows.ndim != 2
                or interaction_rows.shape[1] != len(INTERACTION_STATE_NAMES)
                or interaction_rows.shape[0] < rows.shape[0]
            ):
                raise ValueError(
                    "Expected interaction states with shape "
                    f"(T, {len(INTERACTION_STATE_NAMES)}) and T >= {rows.shape[0]}, "
                    f"got {interaction_rows.shape}."
                )
            if not np.isfinite(interaction_rows[: rows.shape[0]]).all():
                raise ValueError("Interaction-state chunk contains non-finite values.")
        normalized_chunk_id = None if chunk_id is None else str(chunk_id)

        future = {
            int(item.timestep): item
            for item in self._queue
            if int(item.timestep) > self.last_popped_timestep
        }
        stale_skipped = 0
        overlap_blended = 0
        appended = 0
        for offset, row in enumerate(rows):
            timestep = int(start_timestep) + int(offset)
            new_interaction = (
                None
                if interaction_rows is None
                else np.asarray(interaction_rows[offset], dtype=np.float32).copy()
            )
            new_chunk_ids = () if normalized_chunk_id is None else (normalized_chunk_id,)
            if timestep < int(current_timestep) or timestep <= self.last_popped_timestep:
                stale_skipped += 1
                continue
            if timestep in future:
                old_item = future[timestep]
                if old_item.interaction_state is not None and new_interaction is not None:
                    total_weight = self.old_weight + self.new_weight
                    blended_interaction = (
                        (self.old_weight / total_weight) * old_item.interaction_state
                        + (self.new_weight / total_weight) * new_interaction
                    ).astype(np.float32)
                elif new_interaction is not None:
                    blended_interaction = new_interaction
                elif old_item.interaction_state is not None:
                    blended_interaction = old_item.interaction_state.copy()
                else:
                    blended_interaction = None
                blended_chunk_ids = tuple(
                    dict.fromkeys((*old_item.interaction_chunk_ids, *new_chunk_ids))
                )
                future[timestep] = TimedEEAction(
                    timestep=timestep,
                    action=blend_actions(
                        old_item.action,
                        row,
                        state_action_mode=self.state_action_mode,
                        old_weight=self.old_weight,
                        new_weight=self.new_weight,
                    ),
                    source="overlap_0.3_old_0.7_new",
                    blend_count=int(old_item.blend_count) + 1,
                    interaction_state=blended_interaction,
                    interaction_chunk_ids=blended_chunk_ids,
                )
                overlap_blended += 1
            else:
                future[timestep] = TimedEEAction(
                    timestep=timestep,
                    action=np.asarray(row[: self.action_dim], dtype=np.float32).copy(),
                    source="new_chunk",
                    blend_count=1,
                    interaction_state=new_interaction,
                    interaction_chunk_ids=new_chunk_ids,
                )
                appended += 1

        self._queue = deque(sorted(future.values(), key=lambda item: item.timestep))
        return {
            "start_timestep": int(start_timestep),
            "ingest_timestep": int(current_timestep),
            "stale_skipped": int(stale_skipped),
            "overlap_blended": int(overlap_blended),
            "appended": int(appended),
            "queue_size": self.queue_size,
            "queue_first_timestep": self.first_timestep,
            "queue_last_timestep": self.last_timestep,
            "old_weight": self.old_weight,
            "new_weight": self.new_weight,
            "chunk_id": normalized_chunk_id,
            "interaction_rows_available": (
                0 if interaction_rows is None else int(interaction_rows.shape[0])
            ),
        }

    def pop(self, expected_timestep: int) -> TimedEEAction:
        if not self._queue:
            raise RuntimeError("EE action overlap buffer is empty.")
        item = self._queue.popleft()
        if int(item.timestep) != int(expected_timestep):
            raise RuntimeError(
                f"EE action timeline mismatch: expected step {expected_timestep}, got {item.timestep}."
            )
        self.last_popped_timestep = int(item.timestep)
        return item


class RosBaseVelocityBridge:
    """ROS2 bridge for base vx/yaw command and previous-command velocity state."""

    def __init__(
        self,
        cmd_vel_topic: str = "/cmd_vel_safe",
        vel_state_topic: str = "/vel_state",
        node_name: str = "door_act_base_bridge",
        vel_state_timeout_s: float = 0.5,
    ) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.executors import SingleThreadedExecutor

        self.rclpy = rclpy
        self.Twist = Twist
        self.cmd_vel_topic = str(cmd_vel_topic)
        self.vel_state_topic = str(vel_state_topic)
        self.vel_state_timeout_s = float(vel_state_timeout_s)
        self._lock = threading.Lock()
        self._vel_state = np.zeros(2, dtype=np.float32)
        self._vel_state_stamp_mono = 0.0
        self._vel_state_count = 0
        self._last_cmd = np.zeros(2, dtype=np.float32)
        self._last_cmd_stamp_mono = 0.0
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            # Keep Ctrl+C under the ACT main thread so it can publish a final
            # zero base command before shutting ROS down.
            from rclpy.signals import SignalHandlerOptions

            rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node(str(node_name))
        self.publisher = self.node.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.subscription = self.node.create_subscription(
            Twist,
            self.vel_state_topic,
            self._on_vel_state,
            10,
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(
            target=self._spin,
            name="ros-base-velocity",
            daemon=True,
        )
        self.thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except Exception as exc:
            # rclpy may process SIGINT before our orderly stop() call.
            if exc.__class__.__name__ != "ExternalShutdownException":
                print(f"warning: ROS base executor stopped unexpectedly: {exc}", file=sys.stderr, flush=True)

    def _on_vel_state(self, msg) -> None:
        vel = np.array([float(msg.linear.x), float(msg.angular.z)], dtype=np.float32)
        with self._lock:
            self._vel_state = vel
            self._vel_state_stamp_mono = time.monotonic()
            self._vel_state_count += 1

    def get_vel_state(self) -> tuple[np.ndarray, dict]:
        now = time.monotonic()
        with self._lock:
            vel = self._vel_state.copy()
            stamp = float(self._vel_state_stamp_mono)
            count = int(self._vel_state_count)
        age_s = None if count <= 0 or stamp <= 0.0 else now - stamp
        stale = (
            count <= 0
            or (
                self.vel_state_timeout_s > 0.0
                and age_s is not None
                and age_s > self.vel_state_timeout_s
            )
        )
        if stale:
            vel[:] = 0.0
        return vel, {
            "topic": self.vel_state_topic,
            "count": count,
            "age_s": age_s,
            "stale": bool(stale),
        }

    def publish_action(self, action: np.ndarray) -> dict:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape[0] < 2:
            raise ValueError(f"ACT action needs at least 2 values for vx/vyaw, got {values.shape}.")
        vx = float(values[0])
        vyaw = float(values[1])
        msg = self.Twist()
        msg.linear.x = vx
        msg.angular.z = vyaw
        self.publisher.publish(msg)
        with self._lock:
            self._last_cmd[:] = (vx, vyaw)
            self._last_cmd_stamp_mono = time.monotonic()
        return {
            "topic": self.cmd_vel_topic,
            "vx": vx,
            "vyaw": vyaw,
        }

    def publish_zero(self) -> None:
        msg = self.Twist()
        self.publisher.publish(msg)

    def stop(self, send_zero: bool = True) -> None:
        if send_zero:
            try:
                self.publish_zero()
                time.sleep(0.02)
            except Exception:
                pass
        try:
            self.executor.shutdown()
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self._owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()


class UdpActionPublisher:
    """Send full EE10 or joint9 ACT packets to the Z1 bridge."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 15011,
        state_action_mode: str = "ee10",
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.state_action_mode = normalize_state_action_mode(state_action_mode)
        self.action_dim = state_action_dim(self.state_action_mode)
        self.addr = (self.host, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish_action(self, action: np.ndarray) -> dict:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape[0] < self.action_dim:
            raise ValueError(
                f"{self.state_action_mode} ACT action needs {self.action_dim} values, got {values.shape}."
            )
        payload = {
            "state_action_mode": self.state_action_mode,
            "action": [float(x) for x in values[: self.action_dim]],
        }
        self.sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), self.addr)
        return {
            "host": self.host,
            "port": self.port,
            "action": payload["action"],
        }

    def publish_hold_zero_base(self, action_template: np.ndarray | None = None) -> None:
        action = zero_state(self.state_action_mode)
        if action_template is not None:
            values = np.asarray(action_template, dtype=np.float32).reshape(-1)
            if values.shape[0] >= self.action_dim:
                action[:] = values[: self.action_dim]
                action[0:2] = 0.0
        self.publish_action(action)

    def request_shutdown_back_to_start(self, repeat: int = 3, interval_s: float = 0.02) -> dict:
        payload = {"bridge_command": "shutdown_back_to_start"}
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        sends = max(1, int(repeat))
        for index in range(sends):
            self.sock.sendto(encoded, self.addr)
            if index + 1 < sends and interval_s > 0.0:
                time.sleep(float(interval_s))
        return {
            "host": self.host,
            "port": self.port,
            "bridge_command": payload["bridge_command"],
            "sent": sends,
        }

    def close(self) -> None:
        self.sock.close()


class Z1ActStateReceiver:
    """Receive assembled EE10 or joint9 ACT state JSON from the Z1 bridge."""

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = 15013,
        timeout_s: float = 0.5,
        state_action_mode: str = "ee10",
    ) -> None:
        self.bind_host = str(bind_host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.state_action_mode = normalize_state_action_mode(state_action_mode)
        self.state_dim = state_action_dim(self.state_action_mode)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_host, self.port))
        self.sock.settimeout(0.1)
        self._lock = threading.Lock()
        self._state = zero_state(self.state_action_mode)
        self._stamp_mono = 0.0
        self._count = 0
        self._source = ""
        self._last_error = ""
        self._payload_meta: dict[str, object] = {}
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="z1-act-state", daemon=True)
        self.thread.start()

    def _state_from_payload(self, payload: dict) -> np.ndarray:
        if "state" in payload:
            values = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
            if values.shape[0] >= self.state_dim:
                state = values[: self.state_dim].astype(np.float32, copy=True)
                if self.state_action_mode == "joint9":
                    return state
                quat = state[5:9]
                norm = float(np.linalg.norm(quat))
                if np.isfinite(norm) and norm > 1.0e-6:
                    state[5:9] = quat / norm
                else:
                    state[5:9] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                return state
        state = zero_state(self.state_action_mode)
        if self.state_action_mode == "joint9":
            q_value = payload.get("q", payload.get("joint_state", None))
            if q_value is not None:
                state[2:8] = np.asarray(q_value, dtype=np.float32).reshape(-1)[:6]
            if "gripper" in payload:
                state[8] = float(payload["gripper"])
            return state
        if "ee_pos" in payload:
            state[2:5] = np.asarray(payload["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        quat_value = payload.get("ee_quat_xyzw", payload.get("ee_quat", None))
        if quat_value is not None:
            quat = np.asarray(quat_value, dtype=np.float32).reshape(-1)[:4]
            norm = float(np.linalg.norm(quat))
            state[5:9] = quat / norm if np.isfinite(norm) and norm > 1.0e-6 else np.asarray(
                [0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        if "gripper" in payload:
            state[9] = float(payload["gripper"])
        return state

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Z1 state payload must be a JSON object.")
                state = self._state_from_payload(payload)
                payload_meta = {
                    key: payload.get(key)
                    for key in (
                        "action_age_s",
                        "q",
                        "qd",
                        "front_camera_pose_base",
                        "wrist_camera_pose_base",
                        "has_camera_pose",
                        "startup_zero_requested",
                        "startup_zero_active",
                        "startup_zero_done",
                        "startup_zero_max_err",
                        "startup_home_q",
                        "startup_home_max_speed",
                        "startup_zero_error",
                        "shutdown_requested",
                        "shutdown_active",
                        "shutdown_done",
                        "shutdown_error",
                        "ik_ok",
                        "ik_source",
                        "ik_fail_count",
                        "control_count",
                        "arm_enabled",
                        "dry_run",
                        "last_error",
                    )
                    if key in payload
                }
                with self._lock:
                    self._state = state
                    self._stamp_mono = time.monotonic()
                    self._count += 1
                    self._source = f"{addr[0]}:{addr[1]}"
                    self._payload_meta = payload_meta
                    self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._last_error = repr(exc)

    def get_state_tail(self) -> tuple[np.ndarray, dict]:
        now = time.monotonic()
        with self._lock:
            state = self._state.copy()
            stamp = float(self._stamp_mono)
            count = int(self._count)
            source = str(self._source)
            last_error = str(self._last_error)
            payload_meta = dict(self._payload_meta)
        age_s = None if count <= 0 or stamp <= 0.0 else now - stamp
        stale = (
            count <= 0
            or (
                self.timeout_s > 0.0
                and age_s is not None
                and age_s > self.timeout_s
            )
        )
        tail = zero_state(self.state_action_mode)[2 : self.state_dim]
        if not stale:
            tail = state[2 : self.state_dim].astype(np.float32, copy=True)
        return tail, {
            "bind": f"{self.bind_host}:{self.port}",
            "count": count,
            "age_s": age_s,
            "stale": bool(stale),
            "source": source,
            "last_error": last_error,
            **payload_meta,
        }

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.sock.close()
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = "") -> None:
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument("--no_" + dest, dest=dest, action="store_false")
    parser.set_defaults(**{dest: bool(default)})


def wait_for_z1_startup_zero(receiver: Z1ActStateReceiver, timeout_s: float) -> dict:
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    last_meta: dict = {}
    while True:
        _tail, meta = receiver.get_state_tail()
        last_meta = meta
        if int(meta.get("count", 0) or 0) > 0:
            done = meta.get("startup_zero_done")
            active = meta.get("startup_zero_active")
            error = str(meta.get("startup_zero_error") or "")
            if done is True:
                return meta
            if error and active is False:
                raise RuntimeError(f"Z1 startup zero failed before ACT inference: {error}")
        if timeout_s <= 0.0 or time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for Z1 startup zero before ACT inference; "
                f"last_meta={last_meta}"
            )
        time.sleep(0.05)


def wait_for_z1_shutdown_back_to_start(receiver: Z1ActStateReceiver, timeout_s: float) -> dict:
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    last_meta: dict = {}
    while True:
        _tail, meta = receiver.get_state_tail()
        last_meta = meta
        if int(meta.get("count", 0) or 0) > 0:
            done = meta.get("shutdown_done")
            active = meta.get("shutdown_active")
            error = str(meta.get("shutdown_error") or "")
            if done is True:
                return meta
            if error and active is False:
                raise RuntimeError(f"Z1 backToStart failed during ACT shutdown: {error}")
        if timeout_s <= 0.0 or time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for Z1 backToStart after ACT shutdown; "
                f"last_meta={last_meta}"
            )
        time.sleep(0.05)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow ACT inference for Door policy on Jetson.")
    parser.add_argument("--repo_root", type=Path, default=Path("/home/anx/door_act_deploy/visual_whole_body"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/anx/door_act_deploy/checkpoints/door_act_model_latest"))
    parser.add_argument("--log_path", type=Path, default=Path("/home/anx/door_act_deploy/logs/door_act_shadow.jsonl"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--hz", type=float, default=POLICY_HZ)
    parser.add_argument(
        "--action_horizon",
        "--dp_action_horizon",
        dest="action_horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help=(
            "How many actions from each ACT-predicted chunk to execute before running inference again. "
            f"Default {DEFAULT_ACTION_HORIZON} means replan every {DEFAULT_ACTION_HORIZON} control steps."
        ),
    )
    parser.add_argument(
        "--warmup_policy_iters",
        type=int,
        default=2,
        help=(
            "Run this many ACT chunk predictions on live observations before step 0, "
            "without publishing base or Z1 commands. This warms CUDA/model kernels."
        ),
    )
    add_bool_argument(
        parser,
        "--policy_amp",
        default=True,
        help_text="Use CUDA FP16 autocast for ACT inference. Disable with --no_policy_amp.",
    )
    add_bool_argument(
        parser,
        "--async_policy_inference",
        default=True,
        help_text=(
            "Prefetch the next ACT action chunk in a background thread while the current chunk executes. "
            "Disable with --no_async_policy_inference."
        ),
    )
    parser.add_argument(
        "--policy_prefetch_actions",
        type=int,
        default=3,
        help=(
            "Start asynchronous inference when this many actions remain in the active chunk. "
            "At 25 Hz, 3 remaining actions provide about 160 ms before the next chunk is needed."
        ),
    )
    parser.add_argument(
        "--log_flush_interval",
        type=int,
        default=25,
        help="Flush the JSONL log every N control steps; 1 restores per-step fsync-style flushing.",
    )
    parser.add_argument(
        "--arm_tracking_position_tolerance_m",
        type=float,
        default=0.02,
        help="Position-error threshold used to mark a Z1 EE target as reached.",
    )
    parser.add_argument(
        "--arm_tracking_orientation_tolerance_deg",
        type=float,
        default=5.0,
        help="Quaternion angular-error threshold used to mark a Z1 EE target as reached.",
    )
    parser.add_argument(
        "--arm_tracking_gripper_tolerance_rad",
        type=float,
        default=0.10,
        help="Absolute gripper-error threshold used to mark a Z1 target as reached.",
    )
    parser.add_argument(
        "--arm_tracking_joint_tolerance_rad",
        type=float,
        default=0.10,
        help="Maximum per-joint absolute error used to mark a joint9 target as reached.",
    )
    parser.add_argument("--camera_mode", choices=["dummy", "realsense"], default="dummy")
    parser.add_argument("--dummy_depth_m", type=float, default=0.0)
    parser.add_argument("--wrist_serial", type=str, default=DEFAULT_WRIST_REALSENSE_SERIAL)
    parser.add_argument("--front_serial", type=str, default=DEFAULT_FRONT_REALSENSE_SERIAL)
    parser.add_argument("--allow_single_realsense_duplicate", action="store_true")
    parser.add_argument("--rs_filters", dest="rs_filters", action="store_true")
    parser.add_argument("--no_rs_filters", dest="rs_filters", action="store_false")
    parser.set_defaults(rs_filters=None)
    parser.add_argument(
        "--camera_worker_mode",
        choices=["auto", "thread", "process"],
        default="auto",
        help="auto uses one process per camera when RS filters are enabled, avoiding pyrealsense2 filter GIL serialization.",
    )
    parser.add_argument(
        "--rs_spatial_magnitude",
        type=int,
        default=DEFAULT_RS_SPATIAL_MAGNITUDE,
        choices=[1, 2, 3, 4, 5],
        help="RealSense spatial-filter iteration count. 2 keeps both current D435 streams near 30Hz; the Parkour reference uses 5.",
    )
    parser.add_argument(
        "--depth_inpaint_mode",
        choices=DEPTH_INPAINT_MODES,
        default=DEFAULT_DEPTH_INPAINT_MODE,
        help=(
            "off keeps RS denoising without hole propagation; realsense keeps the current RS hole filling; "
            "rgb_guided uses bounded CUDA RGB guidance; opencv_k uses the selected OpenCV policy "
            "(wrist fringe=10px, front fringe=40px, plus tiny white-hole filling); opencv_k_no_rs "
            "uses the same full-resolution OpenCV policy and always disables RealSense filters."
        ),
    )
    parser.add_argument(
        "--depth_inpaint_max_distance_px",
        type=float,
        default=DEFAULT_DEPTH_INPAINT_MAX_DISTANCE_PX,
        help="Maximum crop-image distance from a reliable sensor depth that rgb_guided may fill.",
    )
    parser.add_argument(
        "--depth_inpaint_iterations",
        type=int,
        default=DEFAULT_DEPTH_INPAINT_ITERATIONS,
        help="Number of full-resolution RGB edge-aware refinement iterations.",
    )
    parser.add_argument(
        "--depth_inpaint_rgb_sigma",
        type=float,
        default=DEFAULT_DEPTH_INPAINT_RGB_SIGMA,
        help="RGB edge scale in [0,1]; smaller values resist propagation across color edges more strongly.",
    )
    parser.add_argument(
        "--depth_gaussian_blur_ksize",
        type=int,
        default=DEFAULT_DEPTH_GAUSSIAN_BLUR_KSIZE,
        help=(
            "Optional odd GaussianBlur kernel size applied to policy depth u8 after inpainting. "
            "0 disables it; e.g. 15 with --depth_gaussian_blur_sigma 4.0."
        ),
    )
    parser.add_argument(
        "--depth_gaussian_blur_sigma",
        type=float,
        default=DEFAULT_DEPTH_GAUSSIAN_BLUR_SIGMA,
        help="GaussianBlur sigma used when --depth_gaussian_blur_ksize > 0.",
    )
    parser.add_argument("--no_align_depth_to_color", action="store_true")
    parser.add_argument("--camera_warmup_frames", type=int, default=10)
    parser.add_argument("--depth_snapshot_settle_s", type=float, default=1.0)
    parser.add_argument("--depth_lower_m", type=float, default=DEPTH_LOWER_M)
    parser.add_argument("--depth_far_m", type=float, default=DEPTH_FAR_M)
    parser.add_argument("--crop_left", type=int, default=DEFAULT_CROP_LEFT)
    parser.add_argument("--crop_right", type=int, default=DEFAULT_CROP_RIGHT)
    parser.add_argument("--crop_top", type=int, default=DEFAULT_CROP_TOP)
    parser.add_argument("--crop_bottom", type=int, default=DEFAULT_CROP_BOTTOM)
    parser.add_argument("--save_depth_debug_dir", type=Path, default=None)
    parser.add_argument(
        "--record_policy_depth_video_dir",
        type=Path,
        default=None,
        help=(
            "If set, asynchronously record the two policy_depth_u8 inputs as video. "
            "The side-by-side video layout is left=wrist, right=front."
        ),
    )
    parser.add_argument(
        "--record_policy_depth_video_fps",
        type=float,
        default=0.0,
        help="Video fps for policy-depth recording. 0 means use --hz.",
    )
    parser.add_argument(
        "--record_policy_depth_video_queue_size",
        type=int,
        default=128,
        help="Bounded async encoding queue. If full, frames are dropped instead of blocking inference.",
    )
    parser.add_argument(
        "--record_policy_depth_video_codec",
        type=str,
        default="mp4v",
        help="FourCC codec for policy-depth video; falls back to MJPG/AVI if unavailable.",
    )
    add_bool_argument(
        parser,
        "--record_policy_depth_video_separate",
        default=False,
        help_text="Also write separate wrist/front videos in addition to the side-by-side video.",
    )
    add_bool_argument(
        parser,
        "--record_raw_unfiltered_depth_video",
        default=True,
        help_text=(
            "When policy-depth video recording is enabled, also record aligned "
            "depth before all RealSense filters. Disable with "
            "--no_record_raw_unfiltered_depth_video."
        ),
    )
    parser.add_argument(
        "--record_raw_unfiltered_depth_video_dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for raw aligned, unfiltered depth video. "
            "By default it is created beside --record_policy_depth_video_dir."
        ),
    )
    parser.add_argument("--depth_snapshot_only", action="store_true")
    parser.add_argument("--camera_benchmark_s", type=float, default=0.0)
    parser.add_argument("--list_realsense", action="store_true")
    add_bool_argument(
        parser,
        "--enable_ros_base_bridge",
        default=True,
        help_text="Publish ACT action[0]/action[1] to /cmd_vel_safe and subscribe /vel_state into observation.state[0:2].",
    )
    parser.add_argument("--ros_cmd_vel_topic", type=str, default="/cmd_vel_safe")
    parser.add_argument("--ros_vel_state_topic", type=str, default="/vel_state")
    parser.add_argument("--ros_node_name", type=str, default="door_act_base_bridge")
    parser.add_argument(
        "--vel_state_timeout_s",
        type=float,
        default=0.5,
        help="If no /vel_state arrives for this long, feed zeros into ACT state[0:2]. Set 0 to never time out.",
    )
    add_bool_argument(
        parser,
        "--enable_z1_action_bridge",
        default=True,
        help_text="Send the full checkpoint action to the local Z1 bridge UDP action port.",
    )
    parser.add_argument(
        "--z1_state_action_mode",
        choices=("auto",) + STATE_ACTION_MODES,
        default="auto",
        help=(
            "Z1 state/action schema. auto selects joint9 for 9D joint checkpoints and ee10 for "
            "legacy 10D EE checkpoints."
        ),
    )
    parser.add_argument("--z1_action_udp_host", type=str, default="127.0.0.1")
    parser.add_argument("--z1_action_udp_port", type=int, default=15011)
    add_bool_argument(
        parser,
        "--enable_z1_state_receiver",
        default=True,
        help_text="Receive Z1 bridge state JSON and fill the arm/gripper state asynchronously.",
    )
    parser.add_argument("--z1_state_udp_bind_host", type=str, default="0.0.0.0")
    parser.add_argument("--z1_state_udp_port", type=int, default=15013)
    parser.add_argument(
        "--z1_state_timeout_s",
        type=float,
        default=0.5,
        help="If no Z1 ACT state arrives for this long, use the schema-specific zero fallback. Set 0 to never time out.",
    )
    add_bool_argument(
        parser,
        "--wait_for_z1_startup_zero",
        default=True,
        help_text="Before loading camera/inference loop, wait until z1_act_ee_bridge reports startup zero-joint pose done.",
    )
    parser.add_argument("--z1_startup_zero_wait_timeout_s", type=float, default=20.0)
    add_bool_argument(
        parser,
        "--z1_back_to_start_on_exit",
        default=True,
        help_text=(
            "On normal completion or Ctrl+C, request the Z1 bridge to call "
            "backToStart(), enter passive, and exit cleanly."
        ),
    )
    parser.add_argument(
        "--z1_back_to_start_wait_timeout_s",
        type=float,
        default=20.0,
        help="Seconds to wait for the Z1 bridge to report shutdown backToStart completion.",
    )
    return parser.parse_args(argv)


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.action_horizon <= 0:
        raise ValueError("--action_horizon must be positive.")
    if args.warmup_policy_iters < 0:
        raise ValueError("--warmup_policy_iters must be non-negative.")
    if args.policy_prefetch_actions < 0:
        raise ValueError("--policy_prefetch_actions must be non-negative.")
    if args.policy_prefetch_actions >= args.action_horizon:
        raise ValueError("--policy_prefetch_actions must be smaller than --action_horizon.")
    if args.log_flush_interval <= 0:
        raise ValueError("--log_flush_interval must be positive.")
    if args.arm_tracking_position_tolerance_m < 0:
        raise ValueError("--arm_tracking_position_tolerance_m must be non-negative.")
    if args.arm_tracking_orientation_tolerance_deg < 0:
        raise ValueError("--arm_tracking_orientation_tolerance_deg must be non-negative.")
    if args.arm_tracking_gripper_tolerance_rad < 0:
        raise ValueError("--arm_tracking_gripper_tolerance_rad must be non-negative.")
    if args.arm_tracking_joint_tolerance_rad < 0:
        raise ValueError("--arm_tracking_joint_tolerance_rad must be non-negative.")
    args.depth_inpaint_mode = validate_depth_inpaint_mode(args.depth_inpaint_mode)
    if args.rs_filters is None:
        args.rs_filters = args.depth_inpaint_mode != "opencv_k_no_rs"
    elif args.depth_inpaint_mode == "opencv_k_no_rs" and bool(args.rs_filters):
        print(
            "warning: --depth_inpaint_mode opencv_k_no_rs always disables RealSense filters; ignoring --rs_filters.",
            flush=True,
        )
        args.rs_filters = False
    if args.depth_inpaint_max_distance_px < 0:
        raise ValueError("--depth_inpaint_max_distance_px must be non-negative.")
    if args.depth_inpaint_iterations < 0:
        raise ValueError("--depth_inpaint_iterations must be non-negative.")
    if args.depth_inpaint_rgb_sigma <= 0:
        raise ValueError("--depth_inpaint_rgb_sigma must be positive.")
    args.depth_gaussian_blur_ksize, args.depth_gaussian_blur_sigma = validate_gaussian_blur_args(
        args.depth_gaussian_blur_ksize,
        args.depth_gaussian_blur_sigma,
    )
    if (
        args.camera_mode == "realsense"
        and args.depth_inpaint_mode == "rgb_guided"
        and args.no_align_depth_to_color
    ):
        raise ValueError("rgb_guided inpainting requires depth aligned to color; remove --no_align_depth_to_color.")
    if args.vel_state_timeout_s < 0:
        raise ValueError("--vel_state_timeout_s must be non-negative; use 0 to disable timeout.")
    if args.z1_state_timeout_s < 0:
        raise ValueError("--z1_state_timeout_s must be non-negative; use 0 to disable timeout.")
    if args.z1_startup_zero_wait_timeout_s < 0:
        raise ValueError("--z1_startup_zero_wait_timeout_s must be non-negative.")
    if args.z1_back_to_start_wait_timeout_s < 0:
        raise ValueError("--z1_back_to_start_wait_timeout_s must be non-negative.")
    if args.record_policy_depth_video_fps < 0:
        raise ValueError("--record_policy_depth_video_fps must be non-negative; use 0 to follow --hz.")
    if args.record_policy_depth_video_queue_size <= 0:
        raise ValueError("--record_policy_depth_video_queue_size must be positive.")
    if args.record_policy_depth_video_codec and len(args.record_policy_depth_video_codec) < 4:
        raise ValueError("--record_policy_depth_video_codec must be a FourCC string with at least 4 characters.")
    if args.wait_for_z1_startup_zero and not args.enable_z1_state_receiver:
        raise ValueError("--wait_for_z1_startup_zero requires --enable_z1_state_receiver.")
    for name in ("z1_action_udp_port", "z1_state_udp_port"):
        value = int(getattr(args, name))
        if value <= 0 or value > 65535:
            raise ValueError(f"--{name} must be in [1, 65535].")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DOOR_ACT_DISABLE_BACKBONE_PRETRAINED", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("LEROBOT_MINIMAL_ACT_IMPORTS", "1")
    add_repo_paths(args.repo_root)
    validate_runtime_args(args)

    if args.list_realsense:
        print(json.dumps(RealSenseDepthPair.list_devices(), indent=2), flush=True)
        return

    crop = DepthCrop(
        left=args.crop_left,
        right=args.crop_right,
        top=args.crop_top,
        bottom=args.crop_bottom,
    )
    crop.validate(DEPTH_WIDTH, DEPTH_HEIGHT)

    camera = (
        DummyDepthPair(args.dummy_depth_m, args.depth_lower_m, args.depth_far_m)
        if args.camera_mode == "dummy"
        else RealSenseDepthPair(
            args.wrist_serial,
            args.front_serial,
            allow_single_duplicate=args.allow_single_realsense_duplicate,
            use_filters=bool(args.rs_filters),
            warmup_frames=args.camera_warmup_frames,
            align_depth_to_color=not bool(args.no_align_depth_to_color),
            crop=crop,
            depth_lower_m=args.depth_lower_m,
            depth_far_m=args.depth_far_m,
            worker_mode=args.camera_worker_mode,
            spatial_magnitude=args.rs_spatial_magnitude,
            depth_inpaint_mode=args.depth_inpaint_mode,
            depth_inpaint_device=args.device,
            depth_inpaint_max_distance_px=args.depth_inpaint_max_distance_px,
            depth_inpaint_iterations=args.depth_inpaint_iterations,
            depth_inpaint_rgb_sigma=args.depth_inpaint_rgb_sigma,
            depth_gaussian_blur_ksize=args.depth_gaussian_blur_ksize,
            depth_gaussian_blur_sigma=args.depth_gaussian_blur_sigma,
        )
    )

    if args.camera_benchmark_s > 0:
        camera.start()
        try:
            time.sleep(1.0)
            _, _, first = camera.read()
            t0 = time.monotonic()
            deadline = t0 + float(args.camera_benchmark_s)
            samples = 0
            last = first
            while time.monotonic() < deadline:
                sample_start = time.monotonic()
                _, _, last = camera.read()
                samples += 1
                sleep_s = (1.0 / POLICY_HZ) - (time.monotonic() - sample_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            elapsed = time.monotonic() - t0
            result = {
                "elapsed_s": elapsed,
                "read_samples": samples,
                "read_hz": samples / elapsed,
                "backend": last.get("backend"),
                "worker_mode": last.get("worker_mode"),
                "filters": last["wrist"].get("filters"),
                "spatial_magnitude": last["wrist"].get("spatial_magnitude"),
                "depth_inpaint_mode": last.get("depth_inpaint_mode", args.depth_inpaint_mode),
                "inpaint": last.get("inpaint"),
                "align_depth_to": last.get("align_depth_to"),
                "wrist_serial": last["wrist"].get("serial"),
                "wrist_fps": (
                    int(last["wrist"]["frame_count"]) - int(first["wrist"]["frame_count"])
                )
                / elapsed,
                "wrist_age_ms": last["wrist"].get("age_ms"),
                "front_serial": last["front"].get("serial"),
                "front_fps": (
                    int(last["front"]["frame_count"]) - int(first["front"]["frame_count"])
                )
                / elapsed,
                "front_age_ms": last["front"].get("age_ms"),
                "crop": last.get("crop"),
                "depth_clip_m": last.get("depth_clip_m"),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        finally:
            camera.stop()
        return

    if args.depth_snapshot_only:
        if args.save_depth_debug_dir is None:
            raise ValueError("--depth_snapshot_only requires --save_depth_debug_dir.")
        camera.start()
        try:
            if args.depth_snapshot_settle_s > 0:
                time.sleep(float(args.depth_snapshot_settle_s))
            wrist_depth, front_depth, cam_meta = camera.read(capture_debug=True)
            save_depth_debug_images(
                args.save_depth_debug_dir,
                wrist_depth,
                front_depth,
                cam_meta,
                debug_frames=camera.debug_snapshots(),
            )
            print(json.dumps(cam_meta, ensure_ascii=False, indent=2), flush=True)
            print(f"depth_snapshot_saved dir={args.save_depth_debug_dir}", flush=True)
        finally:
            camera.stop()
        return

    import torch
    from dp.door_policy_backend import DoorPolicyController

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    controller = DoorPolicyController(args.checkpoint, device=args.device, action_horizon=args.action_horizon)
    state_action_mode = infer_state_action_mode(controller, args.z1_state_action_mode)
    runtime_state_dim = state_action_dim(state_action_mode)
    period = 1.0 / max(float(args.hz), 1.0e-6)

    print(
        f"shadow_start checkpoint={args.checkpoint} device={args.device} "
        f"vision={controller.vision_mode} steps={args.steps} camera={args.camera_mode} "
        f"state_action_mode={state_action_mode} state_dim={runtime_state_dim} action_dim={controller.action_dim} "
        f"action_horizon={controller.action_horizon} "
        f"warmup_policy_iters={args.warmup_policy_iters} "
        f"policy_amp={args.policy_amp} "
        f"async_policy_inference={args.async_policy_inference} "
        f"policy_prefetch_actions={args.policy_prefetch_actions} "
        f"depth_inpaint={args.depth_inpaint_mode} "
        f"ros_base_bridge={args.enable_ros_base_bridge} "
        f"z1_action_bridge={args.enable_z1_action_bridge} "
        f"z1_state_receiver={args.enable_z1_state_receiver} "
        f"policy_depth_video_dir={args.record_policy_depth_video_dir} "
        f"raw_unfiltered_depth_video={args.record_raw_unfiltered_depth_video}",
        flush=True,
    )

    base_bridge = None
    z1_action_pub = None
    z1_state_receiver = None
    policy_depth_recorder = None
    raw_unfiltered_depth_recorder = None
    policy_executor = None
    run_t0 = time.monotonic()
    next_t = run_t0
    completed_steps = 0
    loop_elapsed = 0.0
    first_record_wall_time = None
    last_record_wall_time = None
    interrupted = False
    shutdown_back_to_start_requested = False
    try:
        if args.enable_ros_base_bridge:
            base_bridge = RosBaseVelocityBridge(
                cmd_vel_topic=args.ros_cmd_vel_topic,
                vel_state_topic=args.ros_vel_state_topic,
                node_name=args.ros_node_name,
                vel_state_timeout_s=args.vel_state_timeout_s,
            )
            print(
                f"ros_base_bridge_start pub={args.ros_cmd_vel_topic} sub={args.ros_vel_state_topic}",
                flush=True,
            )
        if args.enable_z1_action_bridge:
            z1_action_pub = UdpActionPublisher(
                args.z1_action_udp_host,
                args.z1_action_udp_port,
                state_action_mode=state_action_mode,
            )
            print(
                f"z1_action_bridge_start udp={args.z1_action_udp_host}:{args.z1_action_udp_port}",
                flush=True,
            )
        if args.enable_z1_state_receiver:
            z1_state_receiver = Z1ActStateReceiver(
                bind_host=args.z1_state_udp_bind_host,
                port=args.z1_state_udp_port,
                timeout_s=args.z1_state_timeout_s,
                state_action_mode=state_action_mode,
            )
            print(
                f"z1_state_receiver_start bind={args.z1_state_udp_bind_host}:{args.z1_state_udp_port}",
                flush=True,
            )
        if args.wait_for_z1_startup_zero:
            if z1_state_receiver is None:
                raise RuntimeError("wait_for_z1_startup_zero requested but Z1 state receiver is disabled.")
            meta = wait_for_z1_startup_zero(z1_state_receiver, args.z1_startup_zero_wait_timeout_s)
            print(
                "z1_startup_zero_ready "
                f"max_err={meta.get('startup_zero_max_err')} "
                f"home_q={meta.get('startup_home_q')} "
                f"max_speed={meta.get('startup_home_max_speed')} "
                f"arm_enabled={meta.get('arm_enabled')} dry_run={meta.get('dry_run')}",
                flush=True,
            )
        if args.record_policy_depth_video_dir is not None:
            video_fps = float(args.record_policy_depth_video_fps or args.hz)
            policy_depth_recorder = AsyncPolicyDepthVideoRecorder(
                args.record_policy_depth_video_dir,
                fps=video_fps,
                queue_size=args.record_policy_depth_video_queue_size,
                codec=args.record_policy_depth_video_codec,
                separate=args.record_policy_depth_video_separate,
            )
            policy_depth_recorder.start()
            print(
                "policy_depth_video_start "
                f"dir={args.record_policy_depth_video_dir} fps={video_fps:.3f} "
                f"queue={args.record_policy_depth_video_queue_size}",
                flush=True,
            )
        raw_unfiltered_video_dir = args.record_raw_unfiltered_depth_video_dir
        if (
            raw_unfiltered_video_dir is None
            and args.record_raw_unfiltered_depth_video
            and args.record_policy_depth_video_dir is not None
        ):
            raw_unfiltered_video_dir = (
                args.record_policy_depth_video_dir.parent
                / "raw_unfiltered_depth_video"
            )
        if raw_unfiltered_video_dir is not None:
            video_fps = float(args.record_policy_depth_video_fps or args.hz)
            raw_unfiltered_depth_recorder = AsyncPolicyDepthVideoRecorder(
                raw_unfiltered_video_dir,
                fps=video_fps,
                queue_size=args.record_policy_depth_video_queue_size,
                codec=args.record_policy_depth_video_codec,
                separate=args.record_policy_depth_video_separate,
                video_stem="raw_unfiltered_depth_u8",
                metadata_stem="raw_unfiltered_depth_video",
            )
            raw_unfiltered_depth_recorder.start()
            print(
                "raw_unfiltered_depth_video_start "
                f"dir={raw_unfiltered_video_dir} fps={video_fps:.3f} "
                f"queue={args.record_policy_depth_video_queue_size}",
                flush=True,
            )
        camera.start()
        with args.log_path.open("w", encoding="utf-8") as f:
            def read_policy_observation(capture_debug: bool = False):
                wrist_depth, front_depth, cam_meta = camera.read(capture_debug=capture_debug)
                wrist_raw_unfiltered, front_raw_unfiltered = camera.raw_unfiltered_depth_u8()
                state = zero_state(state_action_mode)
                vel_state_meta = None
                z1_state_meta = None
                if z1_state_receiver is not None:
                    state_tail, z1_state_meta = z1_state_receiver.get_state_tail()
                    state[2:runtime_state_dim] = state_tail
                if base_bridge is not None:
                    vel_state, vel_state_meta = base_bridge.get_vel_state()
                    state[0:2] = vel_state
                return (
                    state,
                    wrist_depth,
                    front_depth,
                    wrist_raw_unfiltered,
                    front_raw_unfiltered,
                    cam_meta,
                    vel_state_meta,
                    z1_state_meta,
                )

            policy_amp_enabled = bool(args.policy_amp and controller.device.type == "cuda")

            def infer_policy_chunk(
                state: np.ndarray,
                wrist_depth: np.ndarray,
                front_depth: np.ndarray,
                front_camera_pose_base: np.ndarray | None,
                wrist_camera_pose_base: np.ndarray | None,
                observation_timestep: int,
                start_timestep: int,
            ) -> dict:
                prep_t0 = time.perf_counter()
                controller.append_observation(
                    state,
                    wrist_depth,
                    wrist_depth,
                    front_depth,
                    front_depth,
                    front_camera_pose_base,
                    wrist_camera_pose_base,
                )
                policy_prep_s = time.perf_counter() - prep_t0
                forward_t0 = time.perf_counter()
                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=policy_amp_enabled,
                    ):
                        controller.sample_action_chunk()
                policy_forward_s = time.perf_counter() - forward_t0
                actions = [np.asarray(row, dtype=np.float32).copy() for row in controller.action_queue]
                controller.action_queue.clear()
                interaction_state_chunk = getattr(
                    controller, "last_interaction_state_chunk", None
                )
                if interaction_state_chunk is not None:
                    interaction_state_chunk = np.asarray(
                        interaction_state_chunk, dtype=np.float32
                    )
                    if interaction_state_chunk.ndim == 3:
                        if interaction_state_chunk.shape[0] != 1:
                            raise RuntimeError(
                                "Real deployment expects one interaction chunk, got "
                                f"{interaction_state_chunk.shape}."
                            )
                        interaction_state_chunk = interaction_state_chunk[0]
                    if (
                        interaction_state_chunk.ndim != 2
                        or interaction_state_chunk.shape[1] != len(INTERACTION_STATE_NAMES)
                    ):
                        raise RuntimeError(
                            "Interaction-state prediction must have shape Hx3, got "
                            f"{interaction_state_chunk.shape}."
                        )
                    if not np.isfinite(interaction_state_chunk).all():
                        raise RuntimeError("Interaction-state prediction contains non-finite values.")
                    interaction_state_chunk = interaction_state_chunk.copy()
                interaction_states_for_actions = None
                if interaction_state_chunk is not None and actions:
                    if interaction_state_chunk.shape[0] >= len(actions):
                        interaction_states_for_actions = interaction_state_chunk[: len(actions)].copy()
                    elif interaction_state_chunk.shape[0] == 1:
                        # encoder_current predicts one current interaction
                        # state rather than an H-step decoder chunk.
                        interaction_states_for_actions = np.repeat(
                            interaction_state_chunk, len(actions), axis=0
                        )
                    else:
                        raise RuntimeError(
                            "Interaction-state horizon is shorter than the executable action horizon: "
                            f"interaction={interaction_state_chunk.shape[0]}, actions={len(actions)}."
                        )
                chunk_id = (
                    f"obs{int(observation_timestep):06d}_start{int(start_timestep):06d}_"
                    f"{time.monotonic_ns()}"
                )
                return {
                    "actions": actions,
                    "chunk_id": chunk_id,
                    "interaction_state_chunk": interaction_state_chunk,
                    "interaction_states_for_actions": interaction_states_for_actions,
                    "policy_prep_s": policy_prep_s,
                    "policy_forward_s": policy_forward_s,
                    "observation_timestep": int(observation_timestep),
                    "start_timestep": int(start_timestep),
                    "completed_monotonic": time.monotonic(),
                }

            def enrich_chunk_ingest_log(ingest_meta: dict, policy_result: dict) -> dict:
                interaction_chunk = policy_result.get("interaction_state_chunk")
                interaction_shape = None
                interaction_values = None
                if interaction_chunk is not None:
                    interaction_array = np.asarray(interaction_chunk, dtype=np.float32)
                    interaction_shape = [int(value) for value in interaction_array.shape]
                    interaction_values = np.round(interaction_array, 6).tolist()
                ingest_meta.update(
                    {
                        "chunk_id": str(policy_result["chunk_id"]),
                        "interaction_state_names": list(INTERACTION_STATE_NAMES),
                        "interaction_state_chunk_shape": interaction_shape,
                        "interaction_state_chunk": interaction_values,
                        "interaction_state_action_aligned_count": (
                            0
                            if policy_result.get("interaction_states_for_actions") is None
                            else int(
                                np.asarray(
                                    policy_result["interaction_states_for_actions"]
                                ).shape[0]
                            )
                        ),
                    }
                )
                return ingest_meta

            if args.async_policy_inference:
                policy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-policy")

            def submit_policy_chunk(
                state: np.ndarray,
                wrist_depth: np.ndarray,
                front_depth: np.ndarray,
                z1_state_meta: dict | None,
                *,
                observation_timestep: int,
                start_timestep: int,
            ) -> Future | dict:
                front_camera_pose_base = None
                wrist_camera_pose_base = None
                if controller.plucker_conditioning:
                    meta = z1_state_meta or {}
                    if not bool(meta.get("has_camera_pose", False)):
                        raise RuntimeError(
                            "Plücker checkpoint requires live camera poses, but the Z1 bridge has not "
                            "published a valid FK-derived wrist pose yet."
                        )
                    front_camera_pose_base = np.asarray(
                        meta.get("front_camera_pose_base"), dtype=np.float32
                    ).reshape(7)
                    wrist_camera_pose_base = np.asarray(
                        meta.get("wrist_camera_pose_base"), dtype=np.float32
                    ).reshape(7)
                    if not (
                        np.isfinite(front_camera_pose_base).all()
                        and np.isfinite(wrist_camera_pose_base).all()
                    ):
                        raise RuntimeError("Z1 bridge returned a non-finite Plücker camera pose.")
                inputs = (
                    np.asarray(state, dtype=np.float32).copy(),
                    np.asarray(wrist_depth).copy(),
                    np.asarray(front_depth).copy(),
                    None if front_camera_pose_base is None else front_camera_pose_base.copy(),
                    None if wrist_camera_pose_base is None else wrist_camera_pose_base.copy(),
                    int(observation_timestep),
                    int(start_timestep),
                )
                if policy_executor is not None:
                    return policy_executor.submit(infer_policy_chunk, *inputs)
                return infer_policy_chunk(*inputs)

            for warmup_idx in range(int(args.warmup_policy_iters)):
                obs_t0 = time.perf_counter()
                (
                    state,
                    wrist_depth,
                    front_depth,
                    _wrist_raw_unfiltered,
                    _front_raw_unfiltered,
                    cam_meta,
                    vel_state_meta,
                    z1_state_meta,
                ) = read_policy_observation(capture_debug=False)
                camera_read_s = time.perf_counter() - obs_t0
                warmup_result = submit_policy_chunk(
                    state,
                    wrist_depth,
                    front_depth,
                    z1_state_meta,
                    observation_timestep=-(warmup_idx + 1),
                    start_timestep=0,
                )
                if isinstance(warmup_result, Future):
                    warmup_result = warmup_result.result()
                policy_prep_s = float(warmup_result["policy_prep_s"])
                policy_forward_s = float(warmup_result["policy_forward_s"])
                warmup_action_count = len(warmup_result["actions"])
                warmup_interaction_chunk = warmup_result.get("interaction_state_chunk")
                infer_s = policy_prep_s + policy_forward_s
                record_wall_time = time.time()
                record = {
                    "phase": "warmup",
                    "state_action_mode": state_action_mode,
                    "warmup_iter": warmup_idx,
                    "wall_time": record_wall_time,
                    "obs_s": camera_read_s,
                    "camera_read_s": camera_read_s,
                    "infer_s": infer_s,
                    "policy_prep_s": policy_prep_s,
                    "policy_forward_s": policy_forward_s,
                    "policy_replan": True,
                    "camera": cam_meta,
                    "state": np.round(state, 6).tolist(),
                    "vel_state": vel_state_meta,
                    "z1_state": z1_state_meta,
                    "warmup_action_count": int(warmup_action_count),
                    "chunk_id": str(warmup_result["chunk_id"]),
                    "interaction_state_names": list(INTERACTION_STATE_NAMES),
                    "interaction_state_chunk_shape": (
                        None
                        if warmup_interaction_chunk is None
                        else [
                            int(value)
                            for value in np.asarray(warmup_interaction_chunk).shape
                        ]
                    ),
                    "interaction_state_chunk": (
                        None
                        if warmup_interaction_chunk is None
                        else np.round(
                            np.asarray(warmup_interaction_chunk, dtype=np.float32), 6
                        ).tolist()
                    ),
                    "base_cmd": None,
                    "z1_action": None,
                    "published": False,
                    "cuda_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
                    if torch.cuda.is_available()
                    else 0.0,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"warmup={warmup_idx:02d} prep_ms={policy_prep_s * 1000.0:.1f} "
                    f"forward_ms={policy_forward_s * 1000.0:.1f} "
                    f"state_vx={float(state[0]):+.4f} state_vyaw={float(state[1]):+.4f} "
                    f"actions={warmup_action_count} published=false",
                    flush=True,
                )
            controller.reset()
            action_buffer = EEActionOverlapBuffer(
                old_weight=0.3,
                new_weight=0.7,
                state_action_mode=state_action_mode,
            )
            pending_policy_future: Future | None = None
            pending_policy_result: dict | None = None
            previous_z1_target: np.ndarray | None = None
            previous_z1_target_step: int | None = None
            previous_z1_target_stamp_mono: float | None = None
            tracking_position_errors_m: list[float] = []
            tracking_orientation_errors_deg: list[float] = []
            tracking_joint_max_errors_rad: list[float] = []
            tracking_joint_rmse_errors_rad: list[float] = []
            tracking_gripper_errors_rad: list[float] = []
            tracking_reached_count = 0
            tracking_valid_count = 0
            next_t = time.monotonic()
            for step in range(int(args.steps)):
                obs_t0 = time.perf_counter()
                capture_debug = step == 0 and args.save_depth_debug_dir is not None
                (
                    state,
                    wrist_depth,
                    front_depth,
                    wrist_raw_unfiltered,
                    front_raw_unfiltered,
                    cam_meta,
                    vel_state_meta,
                    z1_state_meta,
                ) = read_policy_observation(capture_debug=capture_debug)
                camera_read_s = time.perf_counter() - obs_t0
                if step == 0 and args.save_depth_debug_dir is not None:
                    save_depth_debug_images(
                        args.save_depth_debug_dir,
                        wrist_depth,
                        front_depth,
                        cam_meta,
                        debug_frames=camera.debug_snapshots(),
                    )
                policy_replan = False
                policy_prefetch_submitted = False
                policy_observation_used = False
                policy_prep_s = 0.0
                policy_forward_s = 0.0
                policy_wait_s = 0.0
                policy_result_age_s = None
                policy_chunk_ingest = None
                policy_chunk_ingests = []
                policy_prefetch_start_timestep = None
                policy_prefetch_observation_timestep = None

                if pending_policy_future is not None and pending_policy_future.done():
                    pending_policy_result = pending_policy_future.result()
                    pending_policy_future = None
                if pending_policy_result is not None:
                    policy_prep_s = float(pending_policy_result["policy_prep_s"])
                    policy_forward_s = float(pending_policy_result["policy_forward_s"])
                    policy_result_age_s = max(
                        0.0,
                        time.monotonic() - float(pending_policy_result["completed_monotonic"]),
                    )
                    policy_chunk_ingest = action_buffer.ingest(
                        pending_policy_result["actions"],
                        start_timestep=int(pending_policy_result["start_timestep"]),
                        current_timestep=step,
                        interaction_states=pending_policy_result.get(
                            "interaction_states_for_actions"
                        ),
                        chunk_id=pending_policy_result.get("chunk_id"),
                    )
                    policy_chunk_ingest.update(
                        {
                            "observation_timestep": int(
                                pending_policy_result["observation_timestep"]
                            ),
                            "completed_to_ingest_age_s": max(
                                0.0, policy_result_age_s
                            ),
                            "policy_prep_s": policy_prep_s,
                            "policy_forward_s": policy_forward_s,
                        }
                    )
                    policy_chunk_ingest = enrich_chunk_ingest_log(
                        policy_chunk_ingest, pending_policy_result
                    )
                    policy_chunk_ingests.append(policy_chunk_ingest)
                    pending_policy_result = None

                if action_buffer.queue_size == 0:
                    policy_replan = True
                    if pending_policy_future is None:
                        request = submit_policy_chunk(
                            state,
                            wrist_depth,
                            front_depth,
                            z1_state_meta,
                            observation_timestep=step,
                            start_timestep=step,
                        )
                        policy_observation_used = True
                        if isinstance(request, Future):
                            pending_policy_future = request
                        else:
                            pending_policy_result = request
                    if pending_policy_result is None:
                        wait_t0 = time.perf_counter()
                        pending_policy_result = pending_policy_future.result()
                        policy_wait_s = time.perf_counter() - wait_t0
                        pending_policy_future = None
                    policy_prep_s = float(pending_policy_result["policy_prep_s"])
                    policy_forward_s = float(pending_policy_result["policy_forward_s"])
                    policy_result_age_s = max(
                        0.0,
                        time.monotonic() - float(pending_policy_result["completed_monotonic"]),
                    )
                    policy_chunk_ingest = action_buffer.ingest(
                        pending_policy_result["actions"],
                        start_timestep=int(pending_policy_result["start_timestep"]),
                        current_timestep=step,
                        interaction_states=pending_policy_result.get(
                            "interaction_states_for_actions"
                        ),
                        chunk_id=pending_policy_result.get("chunk_id"),
                    )
                    policy_chunk_ingest.update(
                        {
                            "observation_timestep": int(
                                pending_policy_result["observation_timestep"]
                            ),
                            "completed_to_ingest_age_s": policy_result_age_s,
                            "policy_prep_s": policy_prep_s,
                            "policy_forward_s": policy_forward_s,
                        }
                    )
                    policy_chunk_ingest = enrich_chunk_ingest_log(
                        policy_chunk_ingest, pending_policy_result
                    )
                    policy_chunk_ingests.append(policy_chunk_ingest)
                    pending_policy_result = None

                if action_buffer.queue_size == 0:
                    raise RuntimeError("ACT inference returned an empty action chunk.")
                action_pop_t0 = time.perf_counter()
                timed_action = action_buffer.pop(expected_timestep=step)
                action = timed_action.action
                executed_interaction_state = None
                if timed_action.interaction_state is not None:
                    interaction_values = np.asarray(
                        timed_action.interaction_state, dtype=np.float32
                    )
                    executed_interaction_state = {
                        "names": list(INTERACTION_STATE_NAMES),
                        "values": np.round(interaction_values, 6).tolist(),
                        "contact_probability": round(float(interaction_values[0]), 6),
                        "handle_progress": round(float(interaction_values[1]), 6),
                        "door_progress": round(float(interaction_values[2]), 6),
                        "action_timestep": int(timed_action.timestep),
                        "action_source": timed_action.source,
                        "blend_count": int(timed_action.blend_count),
                        "chunk_ids": list(timed_action.interaction_chunk_ids),
                    }
                action_pop_s = time.perf_counter() - action_pop_t0
                infer_s = policy_prep_s + policy_forward_s

                now_mono = time.monotonic()
                previous_target_tracking = compute_arm_tracking_error(
                    state,
                    previous_z1_target,
                    z1_state_meta,
                    target_step=previous_z1_target_step,
                    current_step=step,
                    target_age_s=(
                        None
                        if previous_z1_target_stamp_mono is None
                        else max(0.0, now_mono - previous_z1_target_stamp_mono)
                    ),
                    position_tolerance_m=args.arm_tracking_position_tolerance_m,
                    orientation_tolerance_deg=args.arm_tracking_orientation_tolerance_deg,
                    gripper_tolerance_rad=args.arm_tracking_gripper_tolerance_rad,
                    state_action_mode=state_action_mode,
                    joint_tolerance_rad=args.arm_tracking_joint_tolerance_rad,
                )
                current_target_pre_send_error = compute_arm_tracking_error(
                    state,
                    np.asarray(action, dtype=np.float32),
                    z1_state_meta,
                    target_step=step,
                    current_step=step,
                    target_age_s=0.0,
                    position_tolerance_m=args.arm_tracking_position_tolerance_m,
                    orientation_tolerance_deg=args.arm_tracking_orientation_tolerance_deg,
                    gripper_tolerance_rad=args.arm_tracking_gripper_tolerance_rad,
                    state_action_mode=state_action_mode,
                    joint_tolerance_rad=args.arm_tracking_joint_tolerance_rad,
                )
                if previous_target_tracking["valid"]:
                    tracking_valid_count += 1
                    tracking_reached_count += int(previous_target_tracking["reached"])
                    if state_action_mode == "joint9":
                        tracking_joint_max_errors_rad.append(
                            float(previous_target_tracking["joint_max_abs_error_rad"])
                        )
                        tracking_joint_rmse_errors_rad.append(
                            float(previous_target_tracking["joint_rmse_rad"])
                        )
                    else:
                        tracking_position_errors_m.append(float(previous_target_tracking["position_error_m"]))
                        tracking_orientation_errors_deg.append(
                            float(previous_target_tracking["orientation_error_deg"])
                        )
                    tracking_gripper_errors_rad.append(
                        float(previous_target_tracking["gripper_abs_error_rad"])
                    )

                if (
                    args.async_policy_inference
                    and action_buffer.queue_size <= int(args.policy_prefetch_actions)
                    and (int(args.steps) - step - 1) > action_buffer.queue_size
                    and pending_policy_future is None
                    and pending_policy_result is None
                ):
                    # Observation at control step k is sampled after step k-1
                    # has executed and before the step-k command is published.
                    # Therefore new action[0] belongs to global step k. If the
                    # async result arrives later, ingest() drops the expired
                    # prefix and aligns the remaining rows by global timestep.
                    policy_prefetch_observation_timestep = int(step)
                    policy_prefetch_start_timestep = int(step)
                    request = submit_policy_chunk(
                        state,
                        wrist_depth,
                        front_depth,
                        z1_state_meta,
                        observation_timestep=policy_prefetch_observation_timestep,
                        start_timestep=policy_prefetch_start_timestep,
                    )
                    policy_observation_used = True
                    policy_prefetch_submitted = True
                    if isinstance(request, Future):
                        pending_policy_future = request
                    else:
                        pending_policy_result = request
                base_cmd_meta = None
                if base_bridge is not None:
                    base_cmd_meta = base_bridge.publish_action(np.asarray(action, dtype=np.float32))
                z1_action_meta = None
                if z1_action_pub is not None:
                    z1_action_meta = z1_action_pub.publish_action(np.asarray(action, dtype=np.float32))
                    previous_z1_target = np.asarray(action, dtype=np.float32).copy()
                    previous_z1_target_step = int(step)
                    previous_z1_target_stamp_mono = time.monotonic()
                record_wall_time = time.time()
                if first_record_wall_time is None:
                    first_record_wall_time = record_wall_time
                last_record_wall_time = record_wall_time
                policy_depth_video_meta = None
                if policy_depth_recorder is not None:
                    policy_depth_video_meta = policy_depth_recorder.submit(
                        step,
                        record_wall_time,
                        wrist_depth,
                        front_depth,
                    )
                raw_unfiltered_depth_video_meta = None
                if raw_unfiltered_depth_recorder is not None:
                    raw_unfiltered_depth_video_meta = raw_unfiltered_depth_recorder.submit(
                        step,
                        record_wall_time,
                        wrist_raw_unfiltered,
                        front_raw_unfiltered,
                    )
                record = {
                    "step": step,
                    "state_action_mode": state_action_mode,
                    "wall_time": record_wall_time,
                    "obs_s": camera_read_s,
                    "camera_read_s": camera_read_s,
                    "infer_s": infer_s,
                    "policy_prep_s": policy_prep_s,
                    "policy_forward_s": policy_forward_s,
                    "policy_replan": policy_replan,
                    "policy_prefetch_submitted": policy_prefetch_submitted,
                    "policy_observation_used": policy_observation_used,
                    "policy_wait_s": policy_wait_s,
                    "policy_result_age_s": policy_result_age_s,
                    "policy_chunk_ingest": policy_chunk_ingest,
                    "policy_chunk_ingests": policy_chunk_ingests,
                    "policy_prefetch_observation_timestep": policy_prefetch_observation_timestep,
                    "policy_prefetch_start_timestep": policy_prefetch_start_timestep,
                    "policy_amp": policy_amp_enabled,
                    "async_policy_inference": bool(args.async_policy_inference),
                    "action_pop_s": action_pop_s,
                    "action_timestep": int(timed_action.timestep),
                    "action_source": timed_action.source,
                    "action_blend_count": int(timed_action.blend_count),
                    "executed_interaction_state": executed_interaction_state,
                    "actions_remaining": action_buffer.queue_size,
                    "action_queue_first_timestep": action_buffer.first_timestep,
                    "action_queue_last_timestep": action_buffer.last_timestep,
                    "camera": cam_meta,
                    "state": np.round(state, 6).tolist(),
                    "vel_state": vel_state_meta,
                    "z1_state": z1_state_meta,
                    "arm_tracking": {
                        "comparison": "feedback_sampled_before_current_publish",
                        "previous_command": previous_target_tracking,
                        "current_command_pre_send": current_target_pre_send_error,
                    },
                    "policy_depth_video": policy_depth_video_meta,
                    "raw_unfiltered_depth_video": raw_unfiltered_depth_video_meta,
                    "base_cmd": base_cmd_meta,
                    "z1_action": z1_action_meta,
                    "action": np.round(np.asarray(action, dtype=np.float32), 6).tolist(),
                    "action_finite": bool(np.isfinite(action).all()),
                    "cuda_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
                    if torch.cuda.is_available()
                    else 0.0,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (step + 1) % int(args.log_flush_interval) == 0:
                    f.flush()
                tracking_text = (
                    f"arm_track_joint_max_rad={float(previous_target_tracking.get('joint_max_abs_error_rad', float('nan'))):.4f} "
                    f"arm_track_joint_rmse_rad={float(previous_target_tracking.get('joint_rmse_rad', float('nan'))):.4f} "
                    if state_action_mode == "joint9"
                    else (
                        f"arm_track_mm={float(previous_target_tracking.get('position_error_mm', float('nan'))):.1f} "
                        f"arm_track_deg={float(previous_target_tracking.get('orientation_error_deg', float('nan'))):.1f} "
                    )
                )
                print(
                    f"step={step:04d} mode={state_action_mode} replan={str(policy_replan).lower()} "
                    f"prep_ms={policy_prep_s * 1000.0:.1f} "
                    f"forward_ms={policy_forward_s * 1000.0:.1f} "
                    f"wait_ms={policy_wait_s * 1000.0:.1f} "
                    f"state_vx={float(state[0]):+.4f} state_vyaw={float(state[1]):+.4f} "
                    f"action0={float(action[0]):+.4f} action1={float(action[1]):+.4f} "
                    f"{tracking_text}"
                    f"arm_reached={str(bool(previous_target_tracking.get('reached', False))).lower()} "
                    f"cam_dt_ms={float(cam_meta['dt_ms']):.1f} "
                    f"inpaint_ms={float((cam_meta.get('inpaint') or {}).get('total_ms', 0.0)):.1f}",
                    flush=True,
                )
                completed_steps += 1
                next_t += period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # If model loading / a transient camera stall makes one
                    # cycle late, do not "catch up" by sending several actions
                    # faster than the requested control rate on the real robot.
                    next_t = time.monotonic()
            def tracking_percentile(values: list[float], percentile: float) -> float | None:
                if not values:
                    return None
                return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))

            tracking_summary = {
                "phase": "arm_tracking_summary",
                "state_action_mode": state_action_mode,
                "wall_time": time.time(),
                "comparison": "feedback_at_step_k_vs_command_sent_at_step_k_minus_1",
                "valid_samples": tracking_valid_count,
                "reached_samples": tracking_reached_count,
                "reached_ratio": (
                    float(tracking_reached_count) / float(tracking_valid_count)
                    if tracking_valid_count > 0
                    else None
                ),
                "position_error_mm": {
                    "p50": (
                        None
                        if not tracking_position_errors_m
                        else tracking_percentile(tracking_position_errors_m, 50.0) * 1000.0
                    ),
                    "p95": (
                        None
                        if not tracking_position_errors_m
                        else tracking_percentile(tracking_position_errors_m, 95.0) * 1000.0
                    ),
                    "max": (
                        None
                        if not tracking_position_errors_m
                        else max(tracking_position_errors_m) * 1000.0
                    ),
                },
                "orientation_error_deg": {
                    "p50": tracking_percentile(tracking_orientation_errors_deg, 50.0),
                    "p95": tracking_percentile(tracking_orientation_errors_deg, 95.0),
                    "max": max(tracking_orientation_errors_deg) if tracking_orientation_errors_deg else None,
                },
                "joint_max_abs_error_rad": {
                    "p50": tracking_percentile(tracking_joint_max_errors_rad, 50.0),
                    "p95": tracking_percentile(tracking_joint_max_errors_rad, 95.0),
                    "max": max(tracking_joint_max_errors_rad) if tracking_joint_max_errors_rad else None,
                },
                "joint_rmse_rad": {
                    "p50": tracking_percentile(tracking_joint_rmse_errors_rad, 50.0),
                    "p95": tracking_percentile(tracking_joint_rmse_errors_rad, 95.0),
                    "max": max(tracking_joint_rmse_errors_rad) if tracking_joint_rmse_errors_rad else None,
                },
                "gripper_abs_error_rad": {
                    "p50": tracking_percentile(tracking_gripper_errors_rad, 50.0),
                    "p95": tracking_percentile(tracking_gripper_errors_rad, 95.0),
                    "max": max(tracking_gripper_errors_rad) if tracking_gripper_errors_rad else None,
                },
                "thresholds": {
                    "position_m": float(args.arm_tracking_position_tolerance_m),
                    "orientation_deg": float(args.arm_tracking_orientation_tolerance_deg),
                    "joint_rad": float(args.arm_tracking_joint_tolerance_rad),
                    "gripper_rad": float(args.arm_tracking_gripper_tolerance_rad),
                },
            }
            f.write(json.dumps(tracking_summary, ensure_ascii=False) + "\n")
            f.flush()
            print(f"arm_tracking_summary {json.dumps(tracking_summary, ensure_ascii=False)}", flush=True)
        loop_elapsed = time.monotonic() - run_t0
    except KeyboardInterrupt:
        interrupted = True
        loop_elapsed = time.monotonic() - run_t0
        print("shadow_interrupt received; stopping base and returning Z1 to calibrated home", flush=True)
    finally:
        if base_bridge is not None:
            try:
                base_bridge.publish_zero()
            except Exception:
                pass
        if (
            z1_action_pub is not None
            and args.enable_z1_action_bridge
            and args.z1_back_to_start_on_exit
        ):
            try:
                shutdown_meta = z1_action_pub.request_shutdown_back_to_start()
                shutdown_back_to_start_requested = True
                print(
                    f"z1_shutdown_back_to_start_requested {json.dumps(shutdown_meta, ensure_ascii=False)}",
                    flush=True,
                )
            except Exception as exc:
                print(f"warning: failed to request Z1 backToStart: {exc}", file=sys.stderr, flush=True)
        try:
            camera.stop()
        finally:
            if policy_depth_recorder is not None:
                stats = policy_depth_recorder.stop()
                print(f"policy_depth_video_done {json.dumps(stats, ensure_ascii=False)}", flush=True)
            if raw_unfiltered_depth_recorder is not None:
                stats = raw_unfiltered_depth_recorder.stop()
                print(
                    f"raw_unfiltered_depth_video_done {json.dumps(stats, ensure_ascii=False)}",
                    flush=True,
                )
            if policy_executor is not None:
                policy_executor.shutdown(wait=True, cancel_futures=False)
            if shutdown_back_to_start_requested and z1_state_receiver is not None:
                try:
                    shutdown_meta = wait_for_z1_shutdown_back_to_start(
                        z1_state_receiver,
                        args.z1_back_to_start_wait_timeout_s,
                    )
                    print(
                        "z1_shutdown_back_to_start_done "
                        f"done={shutdown_meta.get('shutdown_done')} "
                        f"error={shutdown_meta.get('shutdown_error')}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"warning: Z1 backToStart completion was not confirmed: {exc}", file=sys.stderr, flush=True)
            if z1_action_pub is not None:
                try:
                    z1_action_pub.close()
                except Exception:
                    pass
            if z1_state_receiver is not None:
                z1_state_receiver.stop()
            if base_bridge is not None:
                base_bridge.stop(send_zero=True)
    run_elapsed = loop_elapsed or (time.monotonic() - run_t0)
    steady_elapsed = (
        float(last_record_wall_time - first_record_wall_time)
        if first_record_wall_time is not None
        and last_record_wall_time is not None
        and completed_steps > 1
        else 0.0
    )
    steady_hz = (completed_steps - 1) / steady_elapsed if steady_elapsed > 0.0 else 0.0
    print(
        f"shadow_done log={args.log_path} steps={completed_steps} "
        f"elapsed_s={run_elapsed:.3f} steady_hz={steady_hz:.3f} "
        f"startup_inclusive_hz={completed_steps / max(run_elapsed, 1.0e-9):.3f} "
        f"interrupted={str(interrupted).lower()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
