#!/usr/bin/env python3
"""Clearly labelled dual-camera aggressive depth inpaint comparison.

This is an offline visual-debug helper for a single frame from recorded dual
D435 bags.  It regenerates labelled wrist/front tiles for:

    A: align(color), no RealSense filters
    C: spatial(holes_fill=0) + temporal
    E: spatial(holes_fill>0) + hole_filling + temporal
    I: C + OpenCV TELEA on small/mid black connected components
    K1-K4: I + OpenCV TELEA on an internal fringe of large black components

K variants are intentionally aggressive.  They are for visual tuning before
changing the realtime deployment path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from offline_opencv_inpaint_step_compare import depth_m_to_policy_u8, make_small_black_component_mask


def read_depth_at_step(
    bag_path: Path,
    step: int,
    *,
    spatial: bool,
    temporal: bool,
    hole_filling: bool,
    timeout_ms: int,
    spatial_magnitude: int,
    spatial_alpha: float,
    spatial_delta: int,
    spatial_holes_fill: int,
    temporal_alpha: float,
    temporal_delta: int,
) -> np.ndarray:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(str(bag_path), repeat_playback=False)
    profile = pipeline.start(cfg)
    try:
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        align = rs.align(rs.stream.color)
        filters: list[Any] = []
        if spatial:
            spatial_filter = rs.spatial_filter()
            spatial_filter.set_option(rs.option.filter_magnitude, int(spatial_magnitude))
            spatial_filter.set_option(rs.option.filter_smooth_alpha, float(spatial_alpha))
            spatial_filter.set_option(rs.option.filter_smooth_delta, int(spatial_delta))
            spatial_filter.set_option(rs.option.holes_fill, int(spatial_holes_fill) if hole_filling else 0)
            filters.append(spatial_filter)
        if hole_filling:
            filters.append(rs.hole_filling_filter())
        if temporal:
            temporal_filter = rs.temporal_filter()
            temporal_filter.set_option(rs.option.filter_smooth_alpha, float(temporal_alpha))
            temporal_filter.set_option(rs.option.filter_smooth_delta, int(temporal_delta))
            filters.append(temporal_filter)

        for frame_idx in range(step + 1):
            frames = pipeline.wait_for_frames(int(timeout_ms))
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError(f"no depth frame at index {frame_idx} in {bag_path}")
            for f in filters:
                depth_frame = f.process(depth_frame)
        return np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    finally:
        pipeline.stop()


def add_large_component_fringe_mask(
    depth_u8: np.ndarray,
    base_mask: np.ndarray,
    *,
    small_max_area: int,
    small_max_span_px: int,
    fringe_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    black = (depth_u8 == 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    out = np.array(base_mask, dtype=np.uint8, copy=True)
    added_total = 0
    large_components = 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area <= small_max_area and width <= small_max_span_px and height <= small_max_span_px:
            continue
        component = (labels == label).astype(np.uint8)
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 3)
        fringe = ((dist > 0.0) & (dist <= float(fringe_px))).astype(np.uint8) * 255
        added_total += int(np.count_nonzero((fringe > 0) & (out == 0)))
        out = np.maximum(out, fringe)
        large_components += 1
    return out, {
        "large_components": int(large_components),
        "fringe_px": int(fringe_px),
        "fringe_added_px": int(added_total),
        "mask_px": int(np.count_nonzero(out)),
    }


def labelled_tile(gray: np.ndarray, title: str, subtitle: str, *, width: int, height: int, label_h: int) -> np.ndarray:
    img = cv2.resize(gray, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bar = np.zeros((int(label_h), int(width), 3), dtype=np.uint8)
    cv2.putText(bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, subtitle, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([bar, bgr], axis=0)


def make_opencv_variants(depth_u8: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base_mask, base_stats = make_small_black_component_mask(
        depth_u8,
        max_area=int(args.small_max_area),
        max_span_px=int(args.small_max_span_px),
        min_area=1,
        border_margin_px=int(args.small_border_margin_px),
    )
    variants: dict[str, np.ndarray] = {
        "I": cv2.inpaint(depth_u8, base_mask, 3.0, cv2.INPAINT_TELEA),
    }
    stats: dict[str, Any] = {
        "I": {
            "mask_px": int(np.count_nonzero(base_mask)),
            "changed_px": int(np.count_nonzero(variants["I"] != depth_u8)),
            "base_stats": base_stats,
        }
    }
    for idx, fringe_px in enumerate(args.fringe_px, start=1):
        mask, mask_stats = add_large_component_fringe_mask(
            depth_u8,
            base_mask,
            small_max_area=int(args.small_max_area),
            small_max_span_px=int(args.small_max_span_px),
            fringe_px=int(fringe_px),
        )
        key = f"K{idx}"
        variants[key] = cv2.inpaint(depth_u8, mask, 3.0, cv2.INPAINT_TELEA)
        stats[key] = {
            **mask_stats,
            "changed_px": int(np.count_nonzero(variants[key] != depth_u8)),
        }
    return variants, stats


def process_camera(name: str, bag: Path, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    depths: dict[str, np.ndarray] = {}
    mode_defs = {
        "A": dict(spatial=False, temporal=False, hole_filling=False),
        "C": dict(spatial=True, temporal=True, hole_filling=False),
        "E": dict(spatial=True, temporal=True, hole_filling=True),
    }
    for key, cfg in mode_defs.items():
        depth = read_depth_at_step(
            bag,
            args.step,
            timeout_ms=args.timeout_ms,
            spatial_magnitude=args.rs_spatial_magnitude,
            spatial_alpha=args.rs_spatial_alpha,
            spatial_delta=args.rs_spatial_delta,
            spatial_holes_fill=args.rs_spatial_holes_fill,
            temporal_alpha=args.rs_temporal_alpha,
            temporal_delta=args.rs_temporal_delta,
            **cfg,
        )
        depths[key] = depth_m_to_policy_u8(
            cv2.resize(depth, (args.output_width, args.output_height), interpolation=cv2.INTER_LINEAR),
            args.depth_lower_m,
            args.depth_far_m,
        )
    opencv_variants, opencv_stats = make_opencv_variants(depths["C"], args)
    depths.update(opencv_variants)
    stats = {
        "camera": name,
        "black_px_C": int(np.count_nonzero(depths["C"] == 0)),
        "opencv": opencv_stats,
    }
    return depths, stats


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wrist_depths, wrist_stats = process_camera("wrist", args.wrist_bag, args)
    front_depths, front_stats = process_camera("front", args.front_bag, args)

    columns = [
        ("A", "A no filters", "align(color), raw"),
        ("C", "C spatial+temporal", "holes_fill=0"),
        ("E", "E RS holefill+temporal", f"holes_fill={args.rs_spatial_holes_fill}+hole_fill"),
        ("I", "I OpenCV small/mid", f"area<={args.small_max_area}, span<={args.small_max_span_px}"),
    ]
    for idx, fringe_px in enumerate(args.fringe_px, start=1):
        columns.append((f"K{idx}", f"K{idx} OpenCV fringe {int(fringe_px)}px", "I + large black edge fringe"))

    rows = []
    for cam_name, depths, stats in (("wrist", wrist_depths, wrist_stats), ("front", front_depths, front_stats)):
        tiles = []
        for key, title, subtitle in columns:
            if key in stats.get("opencv", {}):
                st = stats["opencv"][key]
                subtitle = f"{subtitle}; mask={st.get('mask_px', 0)} chg={st.get('changed_px', 0)}"
            elif key == "C":
                subtitle = f"{subtitle}; black={stats['black_px_C']}"
            tiles.append(
                labelled_tile(
                    depths[key],
                    f"{cam_name} {title}",
                    subtitle,
                    width=args.tile_width,
                    height=args.tile_height,
                    label_h=args.label_height,
                )
            )
        rows.append(np.concatenate(tiles, axis=1))
    grid = np.concatenate(rows, axis=0)

    out_png = args.out_dir / f"dual_A_C_E_I_K_aggressive_step_{args.step:06d}.png"
    out_jpg = args.out_dir / f"dual_A_C_E_I_K_aggressive_step_{args.step:06d}.jpg"
    cv2.imwrite(str(out_png), grid)
    cv2.imwrite(str(out_jpg), grid, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])

    manifest = {
        "script": Path(__file__).name,
        "step": int(args.step),
        "base_pipeline_C": "align(color) -> spatial_filter(holes_fill=0) -> temporal_filter -> clip 0.2-1.5m -> u8",
        "modes": {key: {"title": title, "subtitle": subtitle} for key, title, subtitle in columns},
        "component_mask": {
            "small_max_area": int(args.small_max_area),
            "small_max_span_px": int(args.small_max_span_px),
            "small_border_margin_px": int(args.small_border_margin_px),
            "fringe_px": [int(v) for v in args.fringe_px],
        },
        "wrist_stats": wrist_stats,
        "front_stats": front_stats,
        "outputs": {"png": str(out_png), "jpg": str(out_jpg)},
        "elapsed_s": time.time() - t0,
    }
    manifest_path = args.out_dir / f"dual_A_C_E_I_K_aggressive_step_{args.step:06d}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("dual_aggressive_step_done " + json.dumps(manifest, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrist_bag", type=Path, required=True)
    p.add_argument("--front_bag", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--step", type=int, default=360)
    p.add_argument("--output_width", type=int, default=640)
    p.add_argument("--output_height", type=int, default=480)
    p.add_argument("--tile_width", type=int, default=480)
    p.add_argument("--tile_height", type=int, default=360)
    p.add_argument("--label_height", type=int, default=62)
    p.add_argument("--depth_lower_m", type=float, default=0.2)
    p.add_argument("--depth_far_m", type=float, default=1.5)
    p.add_argument("--small_max_area", type=int, default=15000)
    p.add_argument("--small_max_span_px", type=int, default=280)
    p.add_argument("--small_border_margin_px", type=int, default=-1)
    p.add_argument("--fringe_px", type=int, nargs="+", default=[40, 60, 80, 120])
    p.add_argument("--rs_spatial_magnitude", type=int, default=2)
    p.add_argument("--rs_spatial_alpha", type=float, default=0.75)
    p.add_argument("--rs_spatial_delta", type=int, default=50)
    p.add_argument("--rs_spatial_holes_fill", type=int, default=5)
    p.add_argument("--rs_temporal_alpha", type=float, default=0.75)
    p.add_argument("--rs_temporal_delta", type=int, default=1)
    p.add_argument("--timeout_ms", type=int, default=5000)
    p.add_argument("--jpeg_quality", type=int, default=92)
    return p.parse_args()


if __name__ == "__main__":
    main()
