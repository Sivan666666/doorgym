#!/usr/bin/env python3
"""One-step OpenCV inpaint comparison on recorded dual-D435 raw bags.

This is an offline debugging helper.  It replays one frame index from the raw
RealSense bags, applies the deployment-style base filter

    align(color) -> spatial_filter(holes_fill=0) -> temporal_filter

then tries small-connected-component OpenCV inpainting variants:

    G1: TELEA radius=3
    G2: TELEA radius=5
    G3: Navier-Stokes radius=3
    G4: small connected components + dilate once + TELEA radius=3

The selected mask is made from small black components in policy-depth-u8.  Large
black foreground regions such as the wrist-camera gripper silhouette are
intentionally left untouched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def depth_m_to_policy_u8(depth_m: np.ndarray, lower_m: float, far_m: float) -> np.ndarray:
    depth = np.array(depth_m, dtype=np.float32, copy=True)
    depth = np.nan_to_num(depth, nan=0.0, posinf=far_m, neginf=0.0)
    valid = depth >= lower_m
    depth = np.clip(depth, lower_m, far_m)
    scaled = (depth - lower_m) / max(far_m - lower_m, 1.0e-6)
    scaled[~valid] = 0.0
    return (255.0 * np.clip(scaled, 0.0, 1.0)).astype(np.uint8)


def label_image(gray: np.ndarray, label: str, sublabel: str | None = None) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        bgr,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if sublabel:
        cv2.putText(
            bgr,
            sublabel,
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return bgr


def make_small_black_component_mask(
    depth_u8: np.ndarray,
    *,
    max_area: int,
    max_span_px: int,
    min_area: int,
    border_margin_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    black = (depth_u8 == 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    h, w = depth_u8.shape[:2]
    selected = np.zeros_like(depth_u8, dtype=np.uint8)
    out: dict[str, Any] = {
        "total_black_px": int(black.sum()),
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
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            out["skipped_area"] += 1
            continue
        if cw > max_span_px or ch > max_span_px:
            out["skipped_span"] += 1
            continue
        if (
            x <= border_margin_px
            or y <= border_margin_px
            or x + cw >= w - border_margin_px
            or y + ch >= h - border_margin_px
        ):
            out["skipped_border"] += 1
            continue
        selected[labels == label] = 255
        out["selected_components"] += 1
        out["selected_px"] += area
    out["selected_ratio_of_black"] = float(out["selected_px"]) / max(float(out["total_black_px"]), 1.0)
    return selected, out


def make_small_white_component_mask(
    depth_u8: np.ndarray,
    *,
    white_threshold: int,
    max_area: int,
    max_span_px: int,
    min_area: int,
    border_margin_px: int,
    ring_radius_px: int,
    min_gray_ring_px: int,
    min_gray_ring_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select tiny white holes surrounded by non-white/non-black gray depth.

    Large white regions are usually far background or far-clip background and
    must not be filled.  The extra ring check keeps only white islands whose
    immediate support ring looks like a real foreground surface.
    """

    white = (depth_u8 >= int(white_threshold)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    h, w = depth_u8.shape[:2]
    selected = np.zeros_like(depth_u8, dtype=np.uint8)
    out: dict[str, Any] = {
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
    kernel_size = 2 * max(1, int(ring_radius_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            out["skipped_area"] += 1
            continue
        if cw > max_span_px or ch > max_span_px:
            out["skipped_span"] += 1
            continue
        if (
            x <= border_margin_px
            or y <= border_margin_px
            or x + cw >= w - border_margin_px
            or y + ch >= h - border_margin_px
        ):
            out["skipped_border"] += 1
            continue
        component = labels == label
        dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & ~component
        ring_count = int(ring.sum())
        # White holes should be inside a measured surface.  The ring therefore
        # should contain enough mid-range gray pixels, not mostly white far
        # background and not black invalid pixels.
        gray_ring = ring & (depth_u8 > 0) & (depth_u8 < int(white_threshold))
        gray_count = int(gray_ring.sum())
        if ring_count <= 0 or gray_count < int(min_gray_ring_px) or gray_count / max(ring_count, 1) < float(min_gray_ring_ratio):
            out["skipped_ring"] += 1
            continue
        selected[component] = 255
        out["selected_components"] += 1
        out["selected_px"] += area
    out["selected_ratio_of_white"] = float(out["selected_px"]) / max(float(out["total_white_px"]), 1.0)
    return selected, out


def read_base_filtered_depth_at_step(
    bag_path: Path,
    step: int,
    *,
    timeout_ms: int,
    spatial_magnitude: int,
    spatial_alpha: float,
    spatial_delta: int,
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

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, int(spatial_magnitude))
        spatial.set_option(rs.option.filter_smooth_alpha, float(spatial_alpha))
        spatial.set_option(rs.option.filter_smooth_delta, int(spatial_delta))
        spatial.set_option(rs.option.holes_fill, 0)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, float(temporal_alpha))
        temporal.set_option(rs.option.filter_smooth_delta, int(temporal_delta))

        for frame_idx in range(step + 1):
            frames = pipeline.wait_for_frames(int(timeout_ms))
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError(f"no depth frame at index {frame_idx} in {bag_path}")
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
        return np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    finally:
        pipeline.stop()


def inpaint_variants(depth_u8: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base_mask, mask_stats = make_small_black_component_mask(
        depth_u8,
        max_area=int(args.max_component_area),
        max_span_px=int(args.max_component_span_px),
        min_area=int(args.min_component_area),
        border_margin_px=int(args.border_margin_px),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(base_mask, kernel, iterations=1)
    variants = {
        "G1": cv2.inpaint(depth_u8, base_mask, 3.0, cv2.INPAINT_TELEA),
        "G2": cv2.inpaint(depth_u8, base_mask, 5.0, cv2.INPAINT_TELEA),
        "G3": cv2.inpaint(depth_u8, base_mask, 3.0, cv2.INPAINT_NS),
        "G4": cv2.inpaint(depth_u8, dilated_mask, 3.0, cv2.INPAINT_TELEA),
    }
    stats = {
        "mask": mask_stats,
        "dilated_mask_px": int((dilated_mask > 0).sum()),
        "changed_px": {key: int(np.count_nonzero(img != depth_u8)) for key, img in variants.items()},
    }
    return variants, stats


def build_grids(
    wrist_u8: np.ndarray,
    front_u8: np.ndarray,
    wrist_variants: dict[str, np.ndarray],
    front_variants: dict[str, np.ndarray],
    wrist_stats: dict[str, Any],
    front_stats: dict[str, Any],
    variant_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = {
        "G1": f"{variant_prefix}1 TELEA r3",
        "G2": f"{variant_prefix}2 TELEA r5",
        "G3": f"{variant_prefix}3 NS r3",
        "G4": f"{variant_prefix}4 dilate + TELEA r3",
    }
    cols = []
    for key in ("G1", "G2", "G3", "G4"):
        wrist_label = f"wrist {labels[key]}"
        front_label = f"front {labels[key]}"
        wrist_sub = f"mask {wrist_stats['mask']['selected_px']}px changed {wrist_stats['changed_px'][key]}px"
        front_sub = f"mask {front_stats['mask']['selected_px']}px changed {front_stats['changed_px'][key]}px"
        col = np.concatenate(
            [
                label_image(wrist_variants[key], wrist_label, wrist_sub),
                label_image(front_variants[key], front_label, front_sub),
            ],
            axis=0,
        )
        cols.append(col)
    g_grid = np.concatenate(cols, axis=1)

    mask_cols = []
    for name, img, stats in (("wrist", wrist_u8, wrist_stats), ("front", front_u8, front_stats)):
        mask, _ = make_small_black_component_mask(
            img,
            max_area=999999,
            max_span_px=999999,
            min_area=1,
            border_margin_px=-1,
        )
        selected, _ = make_small_black_component_mask(
            img,
            max_area=stats["args"]["max_component_area"],
            max_span_px=stats["args"]["max_component_span_px"],
            min_area=stats["args"]["min_component_area"],
            border_margin_px=stats["args"]["border_margin_px"],
        )
        dilated = cv2.dilate(selected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        mask_cols.append(label_image(mask, f"{name} all black mask"))
        mask_cols.append(label_image(selected, f"{name} selected small mask"))
        mask_cols.append(label_image(dilated, f"{name} selected+dilate mask"))
    mask_grid = np.concatenate(mask_cols, axis=1)
    return g_grid, mask_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrist_bag", type=Path, required=True)
    p.add_argument("--front_bag", type=Path, required=True)
    p.add_argument("--ae_snapshot", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--step", type=int, default=360)
    p.add_argument("--output_width", type=int, default=640)
    p.add_argument("--output_height", type=int, default=480)
    p.add_argument("--depth_lower_m", type=float, default=0.2)
    p.add_argument("--depth_far_m", type=float, default=1.5)
    p.add_argument("--max_component_area", type=int, default=5000)
    p.add_argument("--max_component_span_px", type=int, default=140)
    p.add_argument("--min_component_area", type=int, default=1)
    p.add_argument("--border_margin_px", type=int, default=1)
    p.add_argument("--rs_spatial_magnitude", type=int, default=2)
    p.add_argument("--rs_spatial_alpha", type=float, default=0.75)
    p.add_argument("--rs_spatial_delta", type=int, default=50)
    p.add_argument("--rs_temporal_alpha", type=float, default=0.75)
    p.add_argument("--rs_temporal_delta", type=int, default=1)
    p.add_argument("--timeout_ms", type=int, default=5000)
    p.add_argument("--jpeg_quality", type=int, default=92)
    p.add_argument("--variant_prefix", default="G")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    wrist_depth = read_base_filtered_depth_at_step(
        args.wrist_bag,
        args.step,
        timeout_ms=args.timeout_ms,
        spatial_magnitude=args.rs_spatial_magnitude,
        spatial_alpha=args.rs_spatial_alpha,
        spatial_delta=args.rs_spatial_delta,
        temporal_alpha=args.rs_temporal_alpha,
        temporal_delta=args.rs_temporal_delta,
    )
    front_depth = read_base_filtered_depth_at_step(
        args.front_bag,
        args.step,
        timeout_ms=args.timeout_ms,
        spatial_magnitude=args.rs_spatial_magnitude,
        spatial_alpha=args.rs_spatial_alpha,
        spatial_delta=args.rs_spatial_delta,
        temporal_alpha=args.rs_temporal_alpha,
        temporal_delta=args.rs_temporal_delta,
    )
    wrist_u8 = depth_m_to_policy_u8(
        cv2.resize(wrist_depth, (args.output_width, args.output_height), interpolation=cv2.INTER_LINEAR),
        args.depth_lower_m,
        args.depth_far_m,
    )
    front_u8 = depth_m_to_policy_u8(
        cv2.resize(front_depth, (args.output_width, args.output_height), interpolation=cv2.INTER_LINEAR),
        args.depth_lower_m,
        args.depth_far_m,
    )

    wrist_variants, wrist_stats = inpaint_variants(wrist_u8, args)
    front_variants, front_stats = inpaint_variants(front_u8, args)
    common_arg_stats = {
        "max_component_area": int(args.max_component_area),
        "max_component_span_px": int(args.max_component_span_px),
        "min_component_area": int(args.min_component_area),
        "border_margin_px": int(args.border_margin_px),
    }
    wrist_stats["args"] = common_arg_stats
    front_stats["args"] = common_arg_stats
    variant_prefix = str(args.variant_prefix or "G").strip() or "G"
    g_grid, mask_grid = build_grids(
        wrist_u8,
        front_u8,
        wrist_variants,
        front_variants,
        wrist_stats,
        front_stats,
        variant_prefix,
    )

    g_path = args.out_dir / f"opencv_inpaint_{variant_prefix}1{variant_prefix}4_step_{args.step:06d}.png"
    mask_path = args.out_dir / f"opencv_inpaint_{variant_prefix}_masks_step_{args.step:06d}.png"
    cv2.imwrite(str(g_path), g_grid)
    cv2.imwrite(str(mask_path), mask_grid)

    combined_path = None
    combined_jpg_path = None
    if args.ae_snapshot and args.ae_snapshot.exists():
        ae = cv2.imread(str(args.ae_snapshot), cv2.IMREAD_COLOR)
        if ae is None:
            raise RuntimeError(f"failed to read {args.ae_snapshot}")
        if ae.shape[0] != g_grid.shape[0]:
            g_grid_for_combined = cv2.resize(
                g_grid,
                (int(g_grid.shape[1] * ae.shape[0] / g_grid.shape[0]), ae.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            g_grid_for_combined = g_grid
        combined = np.concatenate([ae, g_grid_for_combined], axis=1)
        combined_path = args.out_dir / f"realsense_AE_plus_opencv_{variant_prefix}1{variant_prefix}4_step_{args.step:06d}.png"
        combined_jpg_path = args.out_dir / f"realsense_AE_plus_opencv_{variant_prefix}1{variant_prefix}4_step_{args.step:06d}.jpg"
        cv2.imwrite(str(combined_path), combined)
        cv2.imwrite(str(combined_jpg_path), combined, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])

    manifest = {
        "script": Path(__file__).name,
        "step": int(args.step),
        "base_pipeline": "align(color) -> spatial_filter(holes_fill=0) -> temporal_filter -> clip 0.2-1.5m -> u8",
        "opencv_variants": {
            f"{variant_prefix}1": "cv2.inpaint TELEA radius=3, mask=small connected components",
            f"{variant_prefix}2": "cv2.inpaint TELEA radius=5, mask=small connected components",
            f"{variant_prefix}3": "cv2.inpaint NS radius=3, mask=small connected components",
            f"{variant_prefix}4": "small connected components + dilate once + cv2.inpaint TELEA radius=3",
        },
        "component_mask_args": common_arg_stats,
        "wrist_stats": wrist_stats,
        "front_stats": front_stats,
        "outputs": {
            "g_grid": str(g_path),
            "mask_grid": str(mask_path),
            "combined_png": str(combined_path) if combined_path else None,
            "combined_jpg": str(combined_jpg_path) if combined_jpg_path else None,
        },
        "elapsed_s": time.time() - t0,
    }
    manifest_path = args.out_dir / f"opencv_inpaint_step_{args.step:06d}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("opencv_inpaint_step_done " + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
