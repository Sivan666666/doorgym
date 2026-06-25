#!/usr/bin/env python3
"""Generate a full selected-K OpenCV inpaint comparison video from dual D435 bags.

Preferred offline policy being evaluated:

    wrist: K1 = I + large-component internal fringe 40 px
    front: K2 = I + large-component internal fringe 60 px

where C is:

    align(color) -> spatial_filter(holes_fill=0) -> temporal -> 0.2-1.5m u8

and I is OpenCV TELEA inpaint on small/mid black connected components.

Output video layout:

    wrist C   | wrist K1
    front C   | front K2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from door_act_shadow import opencv_k_inpaint_policy_gray
from offline_opencv_dual_aggressive_step_compare import (
    add_large_component_fringe_mask,
    labelled_tile,
)
from offline_opencv_inpaint_step_compare import (
    depth_m_to_policy_u8,
    make_small_black_component_mask,
    make_small_white_component_mask,
)


@dataclass
class RunningStats:
    frames: int = 0
    black_px_sum: int = 0
    mask_px_sum: int = 0
    changed_px_sum: int = 0
    mask_px_max: int = 0
    changed_px_max: int = 0

    def add(self, black_px: int, mask_px: int, changed_px: int) -> None:
        self.frames += 1
        self.black_px_sum += int(black_px)
        self.mask_px_sum += int(mask_px)
        self.changed_px_sum += int(changed_px)
        self.mask_px_max = max(self.mask_px_max, int(mask_px))
        self.changed_px_max = max(self.changed_px_max, int(changed_px))

    def as_dict(self) -> dict[str, Any]:
        denom = max(self.frames, 1)
        return {
            "frames": int(self.frames),
            "black_px_mean": float(self.black_px_sum) / denom,
            "mask_px_mean": float(self.mask_px_sum) / denom,
            "changed_px_mean": float(self.changed_px_sum) / denom,
            "mask_px_max": int(self.mask_px_max),
            "changed_px_max": int(self.changed_px_max),
        }


class CBagReader:
    def __init__(self, bag_path: Path, args: argparse.Namespace) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.rs_filters = bool(args.rs_filters)
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device_from_file(str(bag_path), repeat_playback=False)
        self.profile = self.pipeline.start(cfg)
        playback = self.profile.get_device().as_playback()
        playback.set_real_time(False)
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, int(args.rs_spatial_magnitude))
        self.spatial.set_option(rs.option.filter_smooth_alpha, float(args.rs_spatial_alpha))
        self.spatial.set_option(rs.option.filter_smooth_delta, int(args.rs_spatial_delta))
        self.spatial.set_option(rs.option.holes_fill, 0)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, float(args.rs_temporal_alpha))
        self.temporal.set_option(rs.option.filter_smooth_delta, int(args.rs_temporal_delta))

    def read_u8(self, args: argparse.Namespace) -> np.ndarray | None:
        try:
            frames = self.pipeline.wait_for_frames(int(args.timeout_ms))
        except Exception:
            return None
        try:
            frames = self.align.process(frames)
        except Exception:
            return None
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None
        if self.rs_filters:
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        depth_m = cv2.resize(depth_m, (args.output_width, args.output_height), interpolation=cv2.INTER_LINEAR)
        return depth_m_to_policy_u8(depth_m, args.depth_lower_m, args.depth_far_m)

    def stop(self) -> None:
        self.pipeline.stop()


def apply_selected_k(depth_u8: np.ndarray, fringe_px: int, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if bool(args.use_runtime_opencv_k):
        out, stats = opencv_k_inpaint_policy_gray(
            depth_u8,
            camera_name=str(args.current_camera_name),
        )
        stats = dict(stats)
        stats["black_px"] = int(stats.get("black_px", np.count_nonzero(depth_u8 == 0)))
        stats["white_px"] = int(stats.get("white_px", np.count_nonzero(depth_u8 >= int(args.white_hole_threshold))))
        stats["combined_mask_px"] = int(stats.get("combined_mask_px", 0))
        stats["changed_px"] = int(stats.get("changed_px_fullres", stats.get("changed_px", 0)))
        return out, stats

    small_mask, small_stats = make_small_black_component_mask(
        depth_u8,
        max_area=int(args.small_max_area),
        max_span_px=int(args.small_max_span_px),
        min_area=1,
        border_margin_px=int(args.small_border_margin_px),
    )
    mask, fringe_stats = add_large_component_fringe_mask(
        depth_u8,
        small_mask,
        small_max_area=int(args.small_max_area),
        small_max_span_px=int(args.small_max_span_px),
        fringe_px=int(fringe_px),
    )
    white_stats: dict[str, Any] = {
        "selected_px": 0,
        "selected_components": 0,
        "total_white_px": int(np.count_nonzero(depth_u8 >= int(args.white_hole_threshold))),
    }
    if bool(args.fill_small_white_holes):
        white_mask, white_stats = make_small_white_component_mask(
            depth_u8,
            white_threshold=int(args.white_hole_threshold),
            max_area=int(args.white_hole_max_area),
            max_span_px=int(args.white_hole_max_span_px),
            min_area=1,
            border_margin_px=int(args.white_hole_border_margin_px),
            ring_radius_px=int(args.white_hole_ring_radius_px),
            min_gray_ring_px=int(args.white_hole_min_gray_ring_px),
            min_gray_ring_ratio=float(args.white_hole_min_gray_ring_ratio),
        )
        mask = np.maximum(mask, white_mask)
    out = cv2.inpaint(depth_u8, mask, 3.0, cv2.INPAINT_TELEA)
    stats = {
        "black_px": int(np.count_nonzero(depth_u8 == 0)),
        "white_px": int(np.count_nonzero(depth_u8 >= int(args.white_hole_threshold))),
        "small_selected_px": int(small_stats["selected_px"]),
        "small_selected_components": int(small_stats["selected_components"]),
        "white_selected_px": int(white_stats["selected_px"]),
        "white_selected_components": int(white_stats["selected_components"]),
        **fringe_stats,
        "combined_mask_px": int(np.count_nonzero(mask)),
        "changed_px": int(np.count_nonzero(out != depth_u8)),
    }
    return out, stats


def open_writer(path: Path, fps: float, width: int, height: int, codec: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec[:4]),
        float(fps),
        (int(width), int(height)),
        True,
    )
    if writer.isOpened():
        return writer, path
    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(fallback),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (int(width), int(height)),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {path} or {fallback}")
    return writer, fallback


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wrist_reader = CBagReader(args.wrist_bag, args)
    front_reader = CBagReader(args.front_bag, args)
    frame_w = int(args.tile_width) * 2
    frame_h = (int(args.tile_height) + int(args.label_height)) * 2
    writer = None
    actual_video_path = None
    wrist_stats_acc = RunningStats()
    front_stats_acc = RunningStats()
    frames_written = 0
    try:
        video_path = args.out_dir / "selected_wristK1_frontK2_full.mp4"
        writer, actual_video_path = open_writer(video_path, args.output_fps, frame_w, frame_h, args.codec)
        while True:
            wrist_c = wrist_reader.read_u8(args)
            front_c = front_reader.read_u8(args)
            if wrist_c is None or front_c is None:
                break
            raw_idx = frames_written
            if raw_idx < int(args.skip_frames):
                continue
            if int(args.sample_stride) > 1 and ((raw_idx - int(args.skip_frames)) % int(args.sample_stride) != 0):
                continue
            args.current_camera_name = "wrist"
            wrist_k, wrist_stats = apply_selected_k(wrist_c, int(args.wrist_fringe_px), args)
            args.current_camera_name = "front"
            front_k, front_stats = apply_selected_k(front_c, int(args.front_fringe_px), args)
            wrist_stats_acc.add(wrist_stats["black_px"], wrist_stats["combined_mask_px"], wrist_stats["changed_px"])
            front_stats_acc.add(front_stats["black_px"], front_stats["combined_mask_px"], front_stats["changed_px"])
            wrist_fringe_label = int(wrist_stats.get("fringe_px_fullres", args.wrist_fringe_px))
            front_fringe_label = int(front_stats.get("fringe_px_fullres", args.front_fringe_px))
            top = np.concatenate(
                [
                    labelled_tile(
                        wrist_c,
                        f"wrist frame {raw_idx:04d} C",
                        f"{args.base_label}; black={wrist_stats['black_px']}",
                        width=args.tile_width,
                        height=args.tile_height,
                        label_h=args.label_height,
                    ),
                    labelled_tile(
                        wrist_k,
                        f"wrist frame {raw_idx:04d} K1",
                        f"fringe={wrist_fringe_label}px mask={wrist_stats['combined_mask_px']} white={wrist_stats['white_selected_px']} chg={wrist_stats['changed_px']}",
                        width=args.tile_width,
                        height=args.tile_height,
                        label_h=args.label_height,
                    ),
                ],
                axis=1,
            )
            bottom = np.concatenate(
                [
                    labelled_tile(
                        front_c,
                        f"front frame {raw_idx:04d} C",
                        f"{args.base_label}; black={front_stats['black_px']}",
                        width=args.tile_width,
                        height=args.tile_height,
                        label_h=args.label_height,
                    ),
                    labelled_tile(
                        front_k,
                        f"front frame {raw_idx:04d} K2",
                        f"fringe={front_fringe_label}px mask={front_stats['combined_mask_px']} white={front_stats['white_selected_px']} chg={front_stats['changed_px']}",
                        width=args.tile_width,
                        height=args.tile_height,
                        label_h=args.label_height,
                    ),
                ],
                axis=1,
            )
            frame = np.concatenate([top, bottom], axis=0)
            writer.write(frame)
            frames_written += 1
            if args.max_frames > 0 and frames_written >= int(args.max_frames):
                break
            if frames_written % int(args.progress_every) == 0:
                print(
                    f"selected_k_video_progress frames={frames_written} "
                    f"wrist_mask={wrist_stats['combined_mask_px']} front_mask={front_stats['combined_mask_px']} "
                    f"wrist_white={wrist_stats['white_selected_px']} front_white={front_stats['white_selected_px']}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.release()
        wrist_reader.stop()
        front_reader.stop()

    manifest = {
        "script": Path(__file__).name,
        "wrist_bag": str(args.wrist_bag),
        "front_bag": str(args.front_bag),
        "base_C": (
            "align(color) -> clip 0.2-1.5m -> u8"
            if not bool(args.rs_filters)
            else "align(color) -> spatial_filter(holes_fill=0) -> temporal -> clip 0.2-1.5m -> u8"
        ),
        "rs_filters": bool(args.rs_filters),
        "selected_policy": {
            "use_runtime_opencv_k": bool(args.use_runtime_opencv_k),
            "wrist": {"variant": "K1", "fringe_px": 10 if args.use_runtime_opencv_k else int(args.wrist_fringe_px)},
            "front": {"variant": "K2", "fringe_px": 40 if args.use_runtime_opencv_k else int(args.front_fringe_px)},
        },
        "rs_spatial_magnitude": int(args.rs_spatial_magnitude),
        "component_mask": {
            "small_max_area": int(args.small_max_area),
            "small_max_span_px": int(args.small_max_span_px),
            "small_border_margin_px": int(args.small_border_margin_px),
        },
        "small_white_holes": {
            "enabled": bool(args.fill_small_white_holes),
            "threshold": int(args.white_hole_threshold),
            "max_area": int(args.white_hole_max_area),
            "max_span_px": int(args.white_hole_max_span_px),
            "border_margin_px": int(args.white_hole_border_margin_px),
            "ring_radius_px": int(args.white_hole_ring_radius_px),
            "min_gray_ring_px": int(args.white_hole_min_gray_ring_px),
            "min_gray_ring_ratio": float(args.white_hole_min_gray_ring_ratio),
        },
        "video": str(actual_video_path),
        "frames_written": int(frames_written),
        "output_fps": float(args.output_fps),
        "output_resolution": [frame_w, frame_h],
        "wrist_stats": wrist_stats_acc.as_dict(),
        "front_stats": front_stats_acc.as_dict(),
        "elapsed_s": time.time() - t0,
    }
    manifest_path = args.out_dir / "selected_wristK1_frontK2_full_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("selected_k_video_done " + json.dumps(manifest, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrist_bag", type=Path, required=True)
    p.add_argument("--front_bag", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--output_width", type=int, default=640)
    p.add_argument("--output_height", type=int, default=480)
    p.add_argument("--tile_width", type=int, default=480)
    p.add_argument("--tile_height", type=int, default=360)
    p.add_argument("--label_height", type=int, default=58)
    p.add_argument("--output_fps", type=float, default=30.0)
    p.add_argument("--codec", default="mp4v")
    p.add_argument("--depth_lower_m", type=float, default=0.2)
    p.add_argument("--depth_far_m", type=float, default=1.5)
    p.add_argument("--small_max_area", type=int, default=15000)
    p.add_argument("--small_max_span_px", type=int, default=280)
    p.add_argument("--small_border_margin_px", type=int, default=-1)
    p.add_argument("--wrist_fringe_px", type=int, default=10)
    p.add_argument("--front_fringe_px", type=int, default=40)
    p.add_argument(
        "--use_runtime_opencv_k",
        action="store_true",
        help="Use door_act_shadow.opencv_k_inpaint_policy_gray(), matching the realtime deployment path.",
    )
    p.add_argument("--fill_small_white_holes", dest="fill_small_white_holes", action="store_true")
    p.add_argument("--no_fill_small_white_holes", dest="fill_small_white_holes", action="store_false")
    p.set_defaults(fill_small_white_holes=True)
    p.add_argument("--white_hole_threshold", type=int, default=250)
    p.add_argument("--white_hole_max_area", type=int, default=2500)
    p.add_argument("--white_hole_max_span_px", type=int, default=90)
    p.add_argument("--white_hole_border_margin_px", type=int, default=2)
    p.add_argument("--white_hole_ring_radius_px", type=int, default=3)
    p.add_argument("--white_hole_min_gray_ring_px", type=int, default=8)
    p.add_argument("--white_hole_min_gray_ring_ratio", type=float, default=0.35)
    p.add_argument("--rs_spatial_magnitude", type=int, default=2)
    p.add_argument("--rs_spatial_alpha", type=float, default=0.75)
    p.add_argument("--rs_spatial_delta", type=int, default=50)
    p.add_argument("--rs_temporal_alpha", type=float, default=0.75)
    p.add_argument("--rs_temporal_delta", type=int, default=1)
    p.add_argument("--rs_filters", dest="rs_filters", action="store_true")
    p.add_argument("--no_rs_filters", dest="rs_filters", action="store_false")
    p.set_defaults(rs_filters=True)
    p.add_argument("--timeout_ms", type=int, default=5000)
    p.add_argument("--skip_frames", type=int, default=0)
    p.add_argument("--sample_stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=50)
    args = p.parse_args()
    args.base_label = "spatial+temporal hfill=0" if bool(args.rs_filters) else "no_rs_filters"
    return args


if __name__ == "__main__":
    main()
