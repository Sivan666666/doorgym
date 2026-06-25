#!/usr/bin/env python3
"""Replay raw RealSense bags through multiple offline depth-filter pipelines.

The input should be raw D435 .bag files recorded with depth z16 and color rgb8.
This script replays each bag without a live camera, runs the selected
librealsense filter chains, then writes policy-style depth-u8 videos/snapshots.

Default sweep:

    A_no_filters
    B_spatial_only
    C_spatial_temporal
    D_spatial_hole_filling
    E_spatial_hole_filling_temporal
    F_spatial_temporal_selective_fill

The processing order intentionally matches the current deployment path:

    align depth to color -> RealSense filters -> crop/resize -> 0.2-1.5m to u8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from door_act_shadow import (  # type: ignore
        DEFAULT_RS_SPATIAL_HOLES_FILL,
        DEFAULT_RS_SPATIAL_MAGNITUDE,
        DEFAULT_RS_SPATIAL_SMOOTH_DELTA,
        DEPTH_FAR_M,
        DEPTH_FPS,
        DEPTH_HEIGHT,
        DEPTH_LOWER_M,
        DEPTH_WIDTH,
        depth_m_to_display_u8,
    )
except Exception:
    DEFAULT_RS_SPATIAL_HOLES_FILL = 5
    DEFAULT_RS_SPATIAL_MAGNITUDE = 2
    DEFAULT_RS_SPATIAL_SMOOTH_DELTA = 50
    DEPTH_LOWER_M = 0.2
    DEPTH_FAR_M = 1.5
    DEPTH_WIDTH = 640
    DEPTH_HEIGHT = 480
    DEPTH_FPS = 30

    def depth_m_to_display_u8(depth_m: np.ndarray, depth_lower_m: float = 0.2, depth_far_m: float = 1.5) -> np.ndarray:
        depth = np.array(depth_m, dtype=np.float32, copy=True)
        depth = np.nan_to_num(depth, nan=0.0, posinf=depth_far_m, neginf=0.0)
        valid = depth >= depth_lower_m
        depth = np.clip(depth, depth_lower_m, depth_far_m)
        scaled = (depth - depth_lower_m) / max(depth_far_m - depth_lower_m, 1.0e-6)
        scaled[~valid] = 0.0
        return (255.0 * np.clip(scaled, 0.0, 1.0)).astype(np.uint8)


@dataclass(frozen=True)
class DepthCrop:
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        y1 = int(self.top)
        y2 = h - int(self.bottom)
        x1 = int(self.left)
        x2 = w - int(self.right)
        if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1:
            raise ValueError(f"invalid crop {self} for image shape {image.shape}")
        return image[y1:y2, x1:x2]

    def as_dict(self) -> dict[str, int]:
        return {"left": self.left, "right": self.right, "top": self.top, "bottom": self.bottom}


@dataclass(frozen=True)
class FilterMode:
    key: str
    label: str
    spatial: bool
    temporal: bool
    hole_filling: bool
    selective_hole_fill: bool = False


ALL_MODES: tuple[FilterMode, ...] = (
    FilterMode("A_no_filters", "A no filters", spatial=False, temporal=False, hole_filling=False),
    FilterMode("B_spatial_only", "B spatial only", spatial=True, temporal=False, hole_filling=False),
    FilterMode("C_spatial_temporal", "C spatial + temporal", spatial=True, temporal=True, hole_filling=False),
    FilterMode("D_spatial_hole_filling", "D spatial + hole filling", spatial=True, temporal=False, hole_filling=True),
    FilterMode(
        "E_spatial_hole_filling_temporal",
        "E spatial + hole filling + temporal",
        spatial=True,
        temporal=True,
        hole_filling=True,
    ),
    FilterMode(
        "F_spatial_temporal_selective_fill",
        "F spatial + temporal + selective fill",
        spatial=True,
        temporal=True,
        hole_filling=False,
        selective_hole_fill=True,
    ),
)
MODE_BY_KEY = {mode.key: mode for mode in ALL_MODES}


def make_filters(rs: Any, mode: FilterMode, args: argparse.Namespace) -> list[Any]:
    filters: list[Any] = []
    if mode.spatial:
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, int(args.rs_spatial_magnitude))
        spatial.set_option(rs.option.filter_smooth_alpha, float(args.rs_spatial_alpha))
        spatial.set_option(rs.option.filter_smooth_delta, int(args.rs_spatial_smooth_delta))
        spatial.set_option(
            rs.option.holes_fill,
            int(args.rs_spatial_holes_fill) if mode.hole_filling else 0,
        )
        filters.append(spatial)
    if mode.hole_filling:
        filters.append(rs.hole_filling_filter())
    if mode.temporal:
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, float(args.rs_temporal_alpha))
        temporal.set_option(rs.option.filter_smooth_delta, int(args.rs_temporal_smooth_delta))
        filters.append(temporal)
    return filters


def selective_hole_fill_depth_m(depth_m: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill only bounded, locally smooth invalid-depth components.

    This is deliberately more conservative than RealSense hole_filling_filter().
    Large foreground invalid regions such as the wrist-camera gripper silhouette
    should stay black, while small pinholes inside otherwise smooth surfaces can
    be filled from their local valid-depth ring.
    """

    import cv2

    depth = np.asarray(depth_m, dtype=np.float32)
    out = np.array(depth, dtype=np.float32, copy=True)
    invalid = ~np.isfinite(out) | (out <= 0.0)
    height, width = invalid.shape[:2]
    stats_out: dict[str, Any] = {
        "invalid_px_before": int(invalid.sum()),
        "filled_px": 0,
        "filled_components": 0,
        "total_components": 0,
        "skipped_area": 0,
        "skipped_span": 0,
        "skipped_border": 0,
        "skipped_ring": 0,
        "skipped_depth_variation": 0,
        "skipped_near_protect": 0,
    }
    if stats_out["invalid_px_before"] <= 0:
        stats_out["invalid_px_after"] = 0
        return out, stats_out

    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        invalid.astype(np.uint8),
        connectivity=8,
    )
    stats_out["total_components"] = max(0, int(n_labels) - 1)
    ring_radius = max(1, int(args.selective_fill_ring_radius_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1),
    )
    max_area = int(args.selective_fill_max_component_area)
    max_span = int(args.selective_fill_max_component_span_px)
    border_margin = int(args.selective_fill_border_margin_px)
    min_ring_ratio = float(args.selective_fill_min_valid_ring_ratio)
    min_ring_px = int(args.selective_fill_min_valid_ring_px)
    range_thresh_m = float(args.selective_fill_depth_range_thresh_m)
    std_thresh_m = float(args.selective_fill_depth_std_thresh_m)
    large_area = int(args.selective_fill_large_component_area)
    protect_near_depth_m = float(args.selective_fill_protect_near_depth_m)
    protect_near_area = int(args.selective_fill_protect_near_area)

    for label in range(1, n_labels):
        x = int(cc_stats[label, cv2.CC_STAT_LEFT])
        y = int(cc_stats[label, cv2.CC_STAT_TOP])
        w = int(cc_stats[label, cv2.CC_STAT_WIDTH])
        h = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        if area > max_area:
            stats_out["skipped_area"] += 1
            continue
        if w > max_span or h > max_span:
            stats_out["skipped_span"] += 1
            continue
        if (
            x <= border_margin
            or y <= border_margin
            or x + w >= width - border_margin
            or y + h >= height - border_margin
        ):
            stats_out["skipped_border"] += 1
            continue

        component = labels == label
        dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & ~component
        ring_count = int(ring.sum())
        ring_valid = ring & np.isfinite(out) & (out > 0.0)
        valid_count = int(ring_valid.sum())
        if ring_count <= 0 or valid_count < min_ring_px or valid_count / max(ring_count, 1) < min_ring_ratio:
            stats_out["skipped_ring"] += 1
            continue

        ring_depths = out[ring_valid].astype(np.float32)
        median_depth = float(np.median(ring_depths))
        p10, p90 = np.percentile(ring_depths, [10.0, 90.0])
        depth_range = float(p90 - p10)
        depth_std = float(np.std(ring_depths))

        if area >= protect_near_area and median_depth < protect_near_depth_m:
            stats_out["skipped_near_protect"] += 1
            continue
        # Very tiny edge speckles can be filled even when their ring sees a
        # depth edge.  Larger holes require a smooth, same-surface support ring.
        if area >= large_area and (depth_range > range_thresh_m or depth_std > std_thresh_m):
            stats_out["skipped_depth_variation"] += 1
            continue

        out[component] = median_depth
        stats_out["filled_px"] += area
        stats_out["filled_components"] += 1

    invalid_after = ~np.isfinite(out) | (out <= 0.0)
    stats_out["invalid_px_after"] = int(invalid_after.sum())
    stats_out["filled_ratio_of_invalid"] = (
        float(stats_out["filled_px"]) / max(float(stats_out["invalid_px_before"]), 1.0)
    )
    return out, stats_out


def accumulate_selective_stats(total: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key == "filled_ratio_of_invalid":
            continue
        if isinstance(value, (int, np.integer)):
            total[key] = int(total.get(key, 0)) + int(value)
        elif isinstance(value, (float, np.floating)):
            total[key] = float(total.get(key, 0.0)) + float(value)


class BagReader:
    def __init__(self, bag_path: Path, align_to_color: bool, filters: list[Any]) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.bag_path = Path(bag_path)
        self.align_to_color = bool(align_to_color)
        self.filters = filters
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device_from_file(str(self.bag_path), repeat_playback=False)
        self.profile = self.pipeline.start(cfg)
        playback = self.profile.get_device().as_playback()
        playback.set_real_time(False)
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        self.align = rs.align(rs.stream.color) if self.align_to_color else None

    def read_depth_m(self, timeout_ms: int = 5000) -> np.ndarray | None:
        try:
            frames = self.pipeline.wait_for_frames(int(timeout_ms))
        except Exception:
            return None
        if self.align is not None:
            try:
                frames = self.align.process(frames)
            except Exception:
                return None
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None
        for rs_filter in self.filters:
            depth_frame = rs_filter.process(depth_frame)
        return np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale

    def stop(self) -> None:
        self.pipeline.stop()


def process_depth_to_u8(
    depth_m: np.ndarray,
    crop: DepthCrop,
    output_width: int,
    output_height: int,
    depth_lower_m: float,
    depth_far_m: float,
) -> np.ndarray:
    import cv2

    cropped = crop.apply(depth_m)
    resized = cv2.resize(cropped, (int(output_width), int(output_height)), interpolation=cv2.INTER_LINEAR)
    return depth_m_to_display_u8(resized, depth_lower_m, depth_far_m)


def gray_to_bgr_with_label(gray: np.ndarray, label: str) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        bgr,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return bgr


def parse_snapshot_steps(text: str) -> set[int]:
    out: set[int] = set()
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.add(max(0, int(chunk)))
    return out


def open_writer(path: Path, fps: float, width: int, height: int, codec: str):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)), True)
    if not writer.isOpened():
        fallback = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (int(width), int(height)), True)
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {path} or {fallback}")
        return writer, fallback
    return writer, path


def run_one_mode(
    mode: FilterMode,
    wrist_bag: Path,
    front_bag: Path,
    args: argparse.Namespace,
    crop: DepthCrop,
    snapshot_steps: set[int],
    snapshot_store: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    import cv2
    import pyrealsense2 as rs

    wrist = BagReader(wrist_bag, align_to_color=args.align_to_color, filters=make_filters(rs, mode, args))
    front = BagReader(front_bag, align_to_color=args.align_to_color, filters=make_filters(rs, mode, args))
    writer = None
    output_path = None
    raw_frame = -1
    written = 0
    selective_stats_total: dict[str, Any] = {}
    start = time.time()
    try:
        video_path = args.out_dir / "videos" / f"{mode.key}_side_by_side.mp4"
        writer, output_path = open_writer(
            video_path,
            fps=float(args.output_fps),
            width=int(args.output_width) * 2,
            height=int(args.output_height),
            codec=args.codec,
        )
        while True:
            raw_frame += 1
            wrist_depth = wrist.read_depth_m(timeout_ms=args.timeout_ms)
            front_depth = front.read_depth_m(timeout_ms=args.timeout_ms)
            if wrist_depth is None or front_depth is None:
                break
            if mode.selective_hole_fill:
                wrist_depth, wrist_fill_stats = selective_hole_fill_depth_m(wrist_depth, args)
                front_depth, front_fill_stats = selective_hole_fill_depth_m(front_depth, args)
                accumulate_selective_stats(selective_stats_total, wrist_fill_stats)
                accumulate_selective_stats(selective_stats_total, front_fill_stats)
            if raw_frame < int(args.skip_frames):
                continue
            if int(args.sample_stride) > 1 and ((raw_frame - int(args.skip_frames)) % int(args.sample_stride) != 0):
                continue
            wrist_u8 = process_depth_to_u8(
                wrist_depth,
                crop,
                args.output_width,
                args.output_height,
                args.depth_lower_m,
                args.depth_far_m,
            )
            front_u8 = process_depth_to_u8(
                front_depth,
                crop,
                args.output_width,
                args.output_height,
                args.depth_lower_m,
                args.depth_far_m,
            )
            if raw_frame in snapshot_steps:
                snapshot_store.setdefault(raw_frame, {})[mode.key] = (wrist_u8.copy(), front_u8.copy())
            frame = np.concatenate(
                [
                    gray_to_bgr_with_label(wrist_u8, f"wrist {mode.label}"),
                    gray_to_bgr_with_label(front_u8, f"front {mode.label}"),
                ],
                axis=1,
            )
            writer.write(frame)
            written += 1
            if args.max_frames > 0 and written >= int(args.max_frames):
                break
    finally:
        if writer is not None:
            writer.release()
        wrist.stop()
        front.stop()
    summary = {
        "mode": mode.key,
        "label": mode.label,
        "video": str(output_path),
        "raw_frames_seen": raw_frame + 1,
        "frames_written": written,
        "elapsed_s": time.time() - start,
    }
    if mode.selective_hole_fill:
        frames_for_stats = max(int(raw_frame + 1), 1)
        total_invalid = float(selective_stats_total.get("invalid_px_before", 0))
        total_filled = float(selective_stats_total.get("filled_px", 0))
        selective_stats_total["filled_ratio_of_invalid"] = total_filled / max(total_invalid, 1.0)
        summary["selective_fill_stats_total"] = selective_stats_total
        per_dual_frame: dict[str, float] = {}
        for key, value in selective_stats_total.items():
            if not isinstance(value, (int, float)):
                continue
            if key == "filled_ratio_of_invalid":
                per_dual_frame[key] = float(value)
            else:
                per_dual_frame[key] = float(value) / frames_for_stats
        summary["selective_fill_stats_per_dual_frame"] = per_dual_frame
    return summary


def write_snapshot_grids(
    snapshot_store: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    modes: list[FilterMode],
    out_dir: Path,
) -> list[str]:
    import cv2

    out_paths: list[str] = []
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for step in sorted(snapshot_store):
        cols = []
        for mode in modes:
            pair = snapshot_store[step].get(mode.key)
            if pair is None:
                continue
            wrist_u8, front_u8 = pair
            col = np.concatenate(
                [
                    gray_to_bgr_with_label(wrist_u8, f"wrist {mode.label}"),
                    gray_to_bgr_with_label(front_u8, f"front {mode.label}"),
                ],
                axis=0,
            )
            cols.append(col)
        if not cols:
            continue
        grid = np.concatenate(cols, axis=1)
        path = snap_dir / f"filter_sweep_step_{step:06d}.png"
        cv2.imwrite(str(path), grid)
        out_paths.append(str(path))
    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline RealSense filter sweep from dual raw .bag files.")
    parser.add_argument("--wrist_bag", type=Path, required=True)
    parser.add_argument("--front_bag", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--modes",
        default=",".join(mode.key for mode in ALL_MODES),
        help=f"Comma-separated modes. Available: {','.join(MODE_BY_KEY)}",
    )
    parser.add_argument("--align_to_color", dest="align_to_color", action="store_true")
    parser.add_argument("--no_align_to_color", dest="align_to_color", action="store_false")
    parser.set_defaults(align_to_color=True)
    parser.add_argument("--output_width", type=int, default=DEPTH_WIDTH)
    parser.add_argument("--output_height", type=int, default=DEPTH_HEIGHT)
    parser.add_argument("--output_fps", type=float, default=DEPTH_FPS)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--depth_lower_m", type=float, default=DEPTH_LOWER_M)
    parser.add_argument("--depth_far_m", type=float, default=DEPTH_FAR_M)
    parser.add_argument("--crop_left", type=int, default=0)
    parser.add_argument("--crop_right", type=int, default=0)
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    parser.add_argument("--rs_spatial_magnitude", type=int, default=DEFAULT_RS_SPATIAL_MAGNITUDE)
    parser.add_argument("--rs_spatial_alpha", type=float, default=0.75)
    parser.add_argument("--rs_spatial_smooth_delta", type=int, default=DEFAULT_RS_SPATIAL_SMOOTH_DELTA)
    parser.add_argument("--rs_spatial_holes_fill", type=int, default=DEFAULT_RS_SPATIAL_HOLES_FILL)
    parser.add_argument("--rs_temporal_alpha", type=float, default=0.75)
    parser.add_argument("--rs_temporal_smooth_delta", type=int, default=1)
    parser.add_argument("--selective_fill_max_component_area", type=int, default=1200)
    parser.add_argument("--selective_fill_max_component_span_px", type=int, default=90)
    parser.add_argument("--selective_fill_ring_radius_px", type=int, default=5)
    parser.add_argument("--selective_fill_min_valid_ring_ratio", type=float, default=0.35)
    parser.add_argument("--selective_fill_min_valid_ring_px", type=int, default=12)
    parser.add_argument("--selective_fill_depth_range_thresh_m", type=float, default=0.14)
    parser.add_argument("--selective_fill_depth_std_thresh_m", type=float, default=0.07)
    parser.add_argument(
        "--selective_fill_large_component_area",
        type=int,
        default=32,
        help="Components at or above this area must pass the depth smoothness thresholds.",
    )
    parser.add_argument("--selective_fill_border_margin_px", type=int, default=1)
    parser.add_argument(
        "--selective_fill_protect_near_depth_m",
        type=float,
        default=0.35,
        help="Do not fill larger holes whose support ring is closer than this depth.",
    )
    parser.add_argument("--selective_fill_protect_near_area", type=int, default=80)
    parser.add_argument("--skip_frames", type=int, default=0)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--timeout_ms", type=int, default=5000)
    parser.add_argument("--snapshot_steps", default="0,30,60,120,240,360,480")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop = DepthCrop(args.crop_left, args.crop_right, args.crop_top, args.crop_bottom)
    requested = [chunk.strip() for chunk in args.modes.split(",") if chunk.strip()]
    unknown = [name for name in requested if name not in MODE_BY_KEY]
    if unknown:
        raise ValueError(f"unknown modes {unknown}; available={list(MODE_BY_KEY)}")
    modes = [MODE_BY_KEY[name] for name in requested]
    snapshot_steps = parse_snapshot_steps(args.snapshot_steps)
    snapshot_store: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    summaries = []
    for mode in modes:
        print(f"filter_sweep_mode_start {mode.key}", flush=True)
        summary = run_one_mode(
            mode,
            args.wrist_bag,
            args.front_bag,
            args,
            crop,
            snapshot_steps,
            snapshot_store,
        )
        summaries.append(summary)
        print("filter_sweep_mode_done " + json.dumps(summary, ensure_ascii=False), flush=True)
    snapshot_paths = write_snapshot_grids(snapshot_store, modes, args.out_dir)
    manifest = {
        "script": Path(__file__).name,
        "wrist_bag": str(args.wrist_bag),
        "front_bag": str(args.front_bag),
        "out_dir": str(args.out_dir),
        "align_to_color": bool(args.align_to_color),
        "output_resolution": [int(args.output_width), int(args.output_height)],
        "depth_clip_m": [float(args.depth_lower_m), float(args.depth_far_m)],
        "crop": crop.as_dict(),
        "filter_params": {
            "spatial_magnitude": int(args.rs_spatial_magnitude),
            "spatial_alpha": float(args.rs_spatial_alpha),
            "spatial_smooth_delta": int(args.rs_spatial_smooth_delta),
            "spatial_holes_fill": int(args.rs_spatial_holes_fill),
            "temporal_alpha": float(args.rs_temporal_alpha),
            "temporal_smooth_delta": int(args.rs_temporal_smooth_delta),
            "selective_fill": {
                "max_component_area": int(args.selective_fill_max_component_area),
                "max_component_span_px": int(args.selective_fill_max_component_span_px),
                "ring_radius_px": int(args.selective_fill_ring_radius_px),
                "min_valid_ring_ratio": float(args.selective_fill_min_valid_ring_ratio),
                "min_valid_ring_px": int(args.selective_fill_min_valid_ring_px),
                "depth_range_thresh_m": float(args.selective_fill_depth_range_thresh_m),
                "depth_std_thresh_m": float(args.selective_fill_depth_std_thresh_m),
                "large_component_area": int(args.selective_fill_large_component_area),
                "border_margin_px": int(args.selective_fill_border_margin_px),
                "protect_near_depth_m": float(args.selective_fill_protect_near_depth_m),
                "protect_near_area": int(args.selective_fill_protect_near_area),
            },
        },
        "modes": [mode.__dict__ for mode in modes],
        "videos": summaries,
        "snapshots": snapshot_paths,
    }
    manifest_path = args.out_dir / "filter_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("filter_sweep_done " + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
