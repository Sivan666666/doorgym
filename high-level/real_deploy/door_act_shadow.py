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
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
DEPTH_FPS = 30
POLICY_HZ = 25.0
DEPTH_LOWER_M = 0.2
DEPTH_FAR_M = 1.5
DEFAULT_CROP_LEFT = 60
DEFAULT_CROP_RIGHT = 30
DEFAULT_CROP_TOP = 30
DEFAULT_CROP_BOTTOM = 30
DEFAULT_RS_SPATIAL_MAGNITUDE = 2
DEFAULT_RS_SPATIAL_SMOOTH_DELTA = 50
DEFAULT_RS_SPATIAL_HOLES_FILL = 5
DEPTH_INPAINT_MODES = ("off", "realsense", "rgb_guided")
DEFAULT_DEPTH_INPAINT_MODE = "realsense"
DEFAULT_DEPTH_INPAINT_MAX_DISTANCE_PX = 64.0
DEFAULT_DEPTH_INPAINT_ITERATIONS = 2
DEFAULT_DEPTH_INPAINT_RGB_SIGMA = 0.10
DEFAULT_WRIST_REALSENSE_SERIAL = "261222075130"
DEFAULT_FRONT_REALSENSE_SERIAL = "261222075566"


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


def depth_m_to_policy_u8(
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
    u8 = (255.0 * np.clip(scaled, 0.0, 1.0)).astype(np.uint8)
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


def validate_depth_inpaint_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in DEPTH_INPAINT_MODES:
        raise ValueError(f"Unsupported depth inpaint mode {mode!r}; expected one of {DEPTH_INPAINT_MODES}.")
    return normalized


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

    def start(self) -> None:
        pass

    def read(self, capture_debug: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        now = time.time()
        image = depth_m_to_policy_u8(self.depth, self.depth_lower_m, self.depth_far_m)
        return image, image.copy(), {
            "wrist_ts": now,
            "front_ts": now,
            "dt_ms": 0.0,
            "backend": "dummy",
            "depth_clip_m": [self.depth_lower_m, self.depth_far_m],
        }

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
        self.latest_color_rgb: np.ndarray | None = None
        self.latest_raw_cropped_depth_m: np.ndarray | None = None
        self.latest_inpaint_input_depth_m: np.ndarray | None = None
        self.latest_color_crop_rgb: np.ndarray | None = None
        self.latest_invalid_mask: np.ndarray | None = None
        self.latest_output_invalid_mask: np.ndarray | None = None
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
                inpaint_input_depth_m = filtered_aligned_depth_m[y1:y2, x1:x2]
                invalid_mask = raw_invalid_mask[y1:y2, x1:x2]
                output_invalid_mask = cv2.resize(
                    invalid_mask.astype(np.uint8),
                    (DEPTH_WIDTH, DEPTH_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                depth_m, crop_shape = crop_and_resize_depth(filtered_aligned_depth_m, self.crop)
                policy_u8 = depth_m_to_policy_u8(depth_m, self.depth_lower_m, self.depth_far_m)
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
                    self.latest_color_rgb = color_rgb
                    self.latest_raw_cropped_depth_m = raw_cropped_depth_m.copy()
                    self.latest_inpaint_input_depth_m = inpaint_input_depth_m.copy()
                    self.latest_color_crop_rgb = color_crop_rgb.copy()
                    self.latest_invalid_mask = invalid_mask.copy()
                    self.latest_output_invalid_mask = output_invalid_mask.copy()
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
                        "error": self.error,
                    }
                    return self.latest_policy_u8.copy(), meta
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {self.name} RealSense depth frame.")
            time.sleep(0.002)

    def read_latest_inputs(self, timeout_s: float = 2.0) -> tuple[dict[str, np.ndarray], dict]:
        _, meta = self.read_latest(timeout_s=timeout_s)
        with self.lock:
            if (
                self.latest_inpaint_input_depth_m is None
                or self.latest_color_crop_rgb is None
                or self.latest_invalid_mask is None
                or self.latest_output_invalid_mask is None
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
    policy_buffer,
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
    import cv2
    import pyrealsense2 as rs

    crop = DepthCrop(*crop_values)
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
        self.crop.validate(DEPTH_WIDTH, DEPTH_HEIGHT)
        self.crop_height = DEPTH_HEIGHT - self.crop.top - self.crop.bottom
        self.crop_width = DEPTH_WIDTH - self.crop.left - self.crop.right

        # Do not fork after the parent has imported librealsense: inherited SDK
        # state can leave the child pipeline alive but unable to receive frames.
        self.ctx = mp.get_context("spawn")
        self.policy_buffer = self.ctx.RawArray(ctypes.c_uint8, DEPTH_HEIGHT * DEPTH_WIDTH * 3)
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
        self.error = ""

    def _drain_status(self) -> None:
        while True:
            try:
                item = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if item.get("type") == "started":
                self.device_info = dict(item.get("device") or {})
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
                self.policy_buffer,
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
                    "error": self.error,
                }
            if self.process is not None and not self.process.is_alive():
                raise RuntimeError(f"{self.name} RealSense process exited: {self.error}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {self.name} RealSense process frame.")
            time.sleep(0.002)

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
            worker_mode = "process" if self.use_filters or self.depth_inpaint_mode == "rgb_guided" else "thread"
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
            depth, meta = self.cameras["single"].read_latest()
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
            }
        wrist, wrist_meta = self.cameras["wrist"].read_latest()
        front, front_meta = self.cameras["front"].read_latest()
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


def zero_state() -> np.ndarray:
    state = np.zeros(10, dtype=np.float32)
    # State convention: [last_vx, last_vyaw, ee_xyz, ee_quat_xyzw, gripper].
    # If the Z1 state stream is not available yet, an identity quaternion is a
    # safer fallback than an all-zero quaternion.  Runtime should normally
    # overwrite state[2:10] from Z1ActStateReceiver before inference.
    state[8] = 1.0
    return state


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
            rclpy.init(args=None)
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
            target=self.executor.spin,
            name="ros-base-velocity",
            daemon=True,
        )
        self.thread.start()

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
    """Send full ACT action[10] packets to the Z1 bridge."""

    def __init__(self, host: str = "127.0.0.1", port: int = 15011) -> None:
        self.host = str(host)
        self.port = int(port)
        self.addr = (self.host, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish_action(self, action: np.ndarray) -> dict:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape[0] < 10:
            raise ValueError(f"ACT action needs 10 values for Z1 bridge, got {values.shape}.")
        payload = {"action": [float(x) for x in values[:10]]}
        self.sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), self.addr)
        return {
            "host": self.host,
            "port": self.port,
            "action": payload["action"],
        }

    def publish_hold_zero_base(self, action_template: np.ndarray | None = None) -> None:
        action = np.zeros(10, dtype=np.float32)
        action[8] = 1.0
        if action_template is not None:
            values = np.asarray(action_template, dtype=np.float32).reshape(-1)
            if values.shape[0] >= 10:
                action[:] = values[:10]
                action[0:2] = 0.0
        self.publish_action(action)

    def close(self) -> None:
        self.sock.close()


class Z1ActStateReceiver:
    """Receive assembled ACT state JSON from z1_act_ee_bridge over UDP."""

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = 15013,
        timeout_s: float = 0.5,
    ) -> None:
        self.bind_host = str(bind_host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_host, self.port))
        self.sock.settimeout(0.1)
        self._lock = threading.Lock()
        self._state = zero_state()
        self._stamp_mono = 0.0
        self._count = 0
        self._source = ""
        self._last_error = ""
        self._payload_meta: dict[str, object] = {}
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="z1-act-state", daemon=True)
        self.thread.start()

    @staticmethod
    def _state_from_payload(payload: dict) -> np.ndarray:
        if "state" in payload:
            values = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
            if values.shape[0] >= 10:
                state = values[:10].astype(np.float32, copy=True)
                quat = state[5:9]
                norm = float(np.linalg.norm(quat))
                if np.isfinite(norm) and norm > 1.0e-6:
                    state[5:9] = quat / norm
                else:
                    state[5:9] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                return state
        state = zero_state()
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
                        "startup_zero_requested",
                        "startup_zero_active",
                        "startup_zero_done",
                        "startup_zero_max_err",
                        "startup_zero_error",
                        "ik_ok",
                        "ik_source",
                        "arm_enabled",
                        "dry_run",
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
        tail = zero_state()[2:10]
        if not stale:
            tail = state[2:10].astype(np.float32, copy=True)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow ACT inference for Door policy on Jetson.")
    parser.add_argument("--repo_root", type=Path, default=Path("/home/anx/door_act_deploy/visual_whole_body"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/anx/door_act_deploy/checkpoints/door_act_model_latest"))
    parser.add_argument("--log_path", type=Path, default=Path("/home/anx/door_act_deploy/logs/door_act_shadow.jsonl"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--hz", type=float, default=POLICY_HZ)
    parser.add_argument("--camera_mode", choices=["dummy", "realsense"], default="dummy")
    parser.add_argument("--dummy_depth_m", type=float, default=0.0)
    parser.add_argument("--wrist_serial", type=str, default=DEFAULT_WRIST_REALSENSE_SERIAL)
    parser.add_argument("--front_serial", type=str, default=DEFAULT_FRONT_REALSENSE_SERIAL)
    parser.add_argument("--allow_single_realsense_duplicate", action="store_true")
    parser.add_argument("--rs_filters", dest="rs_filters", action="store_true")
    parser.add_argument("--no_rs_filters", dest="rs_filters", action="store_false")
    parser.set_defaults(rs_filters=True)
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
        help="off keeps RS denoising without hole propagation; realsense keeps the current RS hole filling; rgb_guided uses bounded CUDA RGB guidance.",
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
        help_text="Send full ACT action[10] to the local z1_act_ee_bridge UDP action port.",
    )
    parser.add_argument("--z1_action_udp_host", type=str, default="127.0.0.1")
    parser.add_argument("--z1_action_udp_port", type=int, default=15011)
    add_bool_argument(
        parser,
        "--enable_z1_state_receiver",
        default=True,
        help_text="Receive Z1 bridge ACT state JSON and fill observation.state[2:10] asynchronously.",
    )
    parser.add_argument("--z1_state_udp_bind_host", type=str, default="0.0.0.0")
    parser.add_argument("--z1_state_udp_port", type=int, default=15013)
    parser.add_argument(
        "--z1_state_timeout_s",
        type=float,
        default=0.5,
        help="If no Z1 ACT state arrives for this long, use identity-quat zero EE fallback. Set 0 to never time out.",
    )
    add_bool_argument(
        parser,
        "--wait_for_z1_startup_zero",
        default=True,
        help_text="Before loading camera/inference loop, wait until z1_act_ee_bridge reports startup zero-joint pose done.",
    )
    parser.add_argument("--z1_startup_zero_wait_timeout_s", type=float, default=20.0)
    return parser.parse_args(argv)


def validate_runtime_args(args: argparse.Namespace) -> None:
    args.depth_inpaint_mode = validate_depth_inpaint_mode(args.depth_inpaint_mode)
    if args.depth_inpaint_max_distance_px < 0:
        raise ValueError("--depth_inpaint_max_distance_px must be non-negative.")
    if args.depth_inpaint_iterations < 0:
        raise ValueError("--depth_inpaint_iterations must be non-negative.")
    if args.depth_inpaint_rgb_sigma <= 0:
        raise ValueError("--depth_inpaint_rgb_sigma must be positive.")
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

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    controller = DoorPolicyController(args.checkpoint, device=args.device, action_horizon=50)
    period = 1.0 / max(float(args.hz), 1.0e-6)

    print(
        f"shadow_start checkpoint={args.checkpoint} device={args.device} "
        f"vision={controller.vision_mode} steps={args.steps} camera={args.camera_mode} "
        f"depth_inpaint={args.depth_inpaint_mode} "
        f"ros_base_bridge={args.enable_ros_base_bridge} "
        f"z1_action_bridge={args.enable_z1_action_bridge} "
        f"z1_state_receiver={args.enable_z1_state_receiver}",
        flush=True,
    )

    base_bridge = None
    z1_action_pub = None
    z1_state_receiver = None
    run_t0 = time.monotonic()
    next_t = run_t0
    completed_steps = 0
    loop_elapsed = 0.0
    first_record_wall_time = None
    last_record_wall_time = None
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
            z1_action_pub = UdpActionPublisher(args.z1_action_udp_host, args.z1_action_udp_port)
            print(
                f"z1_action_bridge_start udp={args.z1_action_udp_host}:{args.z1_action_udp_port}",
                flush=True,
            )
        if args.enable_z1_state_receiver:
            z1_state_receiver = Z1ActStateReceiver(
                bind_host=args.z1_state_udp_bind_host,
                port=args.z1_state_udp_port,
                timeout_s=args.z1_state_timeout_s,
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
                f"arm_enabled={meta.get('arm_enabled')} dry_run={meta.get('dry_run')}",
                flush=True,
            )
        camera.start()
        with args.log_path.open("w", encoding="utf-8") as f:
            for step in range(int(args.steps)):
                obs_t0 = time.perf_counter()
                capture_debug = step == 0 and args.save_depth_debug_dir is not None
                wrist_depth, front_depth, cam_meta = camera.read(capture_debug=capture_debug)
                if step == 0 and args.save_depth_debug_dir is not None:
                    save_depth_debug_images(
                        args.save_depth_debug_dir,
                        wrist_depth,
                        front_depth,
                        cam_meta,
                        debug_frames=camera.debug_snapshots(),
                    )
                state = zero_state()
                vel_state_meta = None
                z1_state_meta = None
                if z1_state_receiver is not None:
                    state_tail, z1_state_meta = z1_state_receiver.get_state_tail()
                    state[2:10] = state_tail
                if base_bridge is not None:
                    vel_state, vel_state_meta = base_bridge.get_vel_state()
                    state[0:2] = vel_state
                infer_t0 = time.perf_counter()
                action = controller.act(state, wrist_depth, wrist_depth, front_depth, front_depth)
                infer_s = time.perf_counter() - infer_t0
                base_cmd_meta = None
                if base_bridge is not None:
                    base_cmd_meta = base_bridge.publish_action(np.asarray(action, dtype=np.float32))
                z1_action_meta = None
                if z1_action_pub is not None:
                    z1_action_meta = z1_action_pub.publish_action(np.asarray(action, dtype=np.float32))
                record_wall_time = time.time()
                if first_record_wall_time is None:
                    first_record_wall_time = record_wall_time
                last_record_wall_time = record_wall_time
                record = {
                    "step": step,
                    "wall_time": record_wall_time,
                    "obs_s": time.perf_counter() - obs_t0,
                    "infer_s": infer_s,
                    "camera": cam_meta,
                    "state": np.round(state, 6).tolist(),
                    "vel_state": vel_state_meta,
                    "z1_state": z1_state_meta,
                    "base_cmd": base_cmd_meta,
                    "z1_action": z1_action_meta,
                    "action": np.round(np.asarray(action, dtype=np.float32), 6).tolist(),
                    "action_finite": bool(np.isfinite(action).all()),
                    "cuda_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
                    if torch.cuda.is_available()
                    else 0.0,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"step={step:04d} infer_ms={infer_s * 1000.0:.1f} "
                    f"state_vx={float(state[0]):+.4f} state_vyaw={float(state[1]):+.4f} "
                    f"action0={float(action[0]):+.4f} action1={float(action[1]):+.4f} "
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
        loop_elapsed = time.monotonic() - run_t0
    finally:
        try:
            camera.stop()
        finally:
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
        f"startup_inclusive_hz={completed_steps / max(run_elapsed, 1.0e-9):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
