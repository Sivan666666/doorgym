#!/usr/bin/env python3
"""Multi-step selected OpenCV-K depth inpaint comparison for dual D435 bags.

This is an offline visual-debug helper.  It evaluates the currently preferred
asymmetric policy:

    wrist: C baseline + K1 = I + large-component fringe 40 px
    front: C baseline + K2 = I + large-component fringe 60 px

where C is:

    align(color) -> spatial_filter(holes_fill=0) -> temporal -> 0.2-1.5m u8

and I is OpenCV TELEA inpaint on small/mid black connected components.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from offline_opencv_dual_aggressive_step_compare import (
    add_large_component_fringe_mask,
    labelled_tile,
    read_depth_at_step,
)
from offline_opencv_inpaint_step_compare import (
    depth_m_to_policy_u8,
    make_small_black_component_mask,
    make_small_white_component_mask,
)


def parse_steps(text: str) -> list[int]:
    out: list[int] = []
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(max(0, int(chunk)))
    if not out:
        raise ValueError("empty --steps")
    return out


def read_c_depth_u8(bag: Path, step: int, args: argparse.Namespace) -> np.ndarray:
    depth = read_depth_at_step(
        bag,
        step,
        spatial=True,
        temporal=True,
        hole_filling=False,
        timeout_ms=args.timeout_ms,
        spatial_magnitude=args.rs_spatial_magnitude,
        spatial_alpha=args.rs_spatial_alpha,
        spatial_delta=args.rs_spatial_delta,
        spatial_holes_fill=args.rs_spatial_holes_fill,
        temporal_alpha=args.rs_temporal_alpha,
        temporal_delta=args.rs_temporal_delta,
    )
    return depth_m_to_policy_u8(
        cv2.resize(depth, (args.output_width, args.output_height), interpolation=cv2.INTER_LINEAR),
        args.depth_lower_m,
        args.depth_far_m,
    )


def apply_selected_k(depth_u8: np.ndarray, fringe_px: int, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
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
        "black_px_c": int(np.count_nonzero(depth_u8 == 0)),
        "white_px_c": int(np.count_nonzero(depth_u8 >= int(args.white_hole_threshold))),
        "small_selected_px": int(small_stats["selected_px"]),
        "small_selected_components": int(small_stats["selected_components"]),
        "white_selected_px": int(white_stats["selected_px"]),
        "white_selected_components": int(white_stats["selected_components"]),
        **fringe_stats,
        "combined_mask_px": int(np.count_nonzero(mask)),
        "changed_px": int(np.count_nonzero(out != depth_u8)),
    }
    return out, stats


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = parse_steps(args.steps)
    t0 = time.time()

    rows = []
    all_stats: dict[str, Any] = {"wrist": {}, "front": {}}
    for cam_name, bag, fringe_px, k_name in (
        ("wrist", args.wrist_bag, int(args.wrist_fringe_px), "K1"),
        ("front", args.front_bag, int(args.front_fringe_px), "K2"),
    ):
        tiles = []
        for step in steps:
            c_u8 = read_c_depth_u8(bag, step, args)
            k_u8, stats = apply_selected_k(c_u8, fringe_px, args)
            all_stats[cam_name][str(step)] = stats
            tiles.append(
                labelled_tile(
                    c_u8,
                    f"{cam_name} step {step} C",
                    f"spatial+temporal hfill=0; black={stats['black_px_c']}",
                    width=args.tile_width,
                    height=args.tile_height,
                    label_h=args.label_height,
                )
            )
            tiles.append(
                labelled_tile(
                    k_u8,
                    f"{cam_name} step {step} {k_name}",
                    f"fringe={fringe_px}px mask={stats['combined_mask_px']} white={stats['white_selected_px']} chg={stats['changed_px']}",
                    width=args.tile_width,
                    height=args.tile_height,
                    label_h=args.label_height,
                )
            )
        rows.append(np.concatenate(tiles, axis=1))

    grid = np.concatenate(rows, axis=0)
    step_tag = "_".join(str(s) for s in steps)
    out_png = args.out_dir / f"dual_selected_wristK1_frontK2_steps_{step_tag}.png"
    out_jpg = args.out_dir / f"dual_selected_wristK1_frontK2_steps_{step_tag}.jpg"
    cv2.imwrite(str(out_png), grid)
    cv2.imwrite(str(out_jpg), grid, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])

    manifest = {
        "script": Path(__file__).name,
        "steps": steps,
        "base_C": "align(color) -> spatial_filter(holes_fill=0) -> temporal -> clip 0.2-1.5m -> u8",
        "selected_policy": {
            "wrist": {"variant": "K1", "fringe_px": int(args.wrist_fringe_px)},
            "front": {"variant": "K2", "fringe_px": int(args.front_fringe_px)},
        },
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
        "stats": all_stats,
        "outputs": {"png": str(out_png), "jpg": str(out_jpg)},
        "elapsed_s": time.time() - t0,
    }
    manifest_path = args.out_dir / f"dual_selected_wristK1_frontK2_steps_{step_tag}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("selected_k_multistep_done " + json.dumps(manifest, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrist_bag", type=Path, required=True)
    p.add_argument("--front_bag", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--steps", default="120,240,360,480")
    p.add_argument("--output_width", type=int, default=640)
    p.add_argument("--output_height", type=int, default=480)
    p.add_argument("--tile_width", type=int, default=320)
    p.add_argument("--tile_height", type=int, default=240)
    p.add_argument("--label_height", type=int, default=58)
    p.add_argument("--depth_lower_m", type=float, default=0.2)
    p.add_argument("--depth_far_m", type=float, default=1.5)
    p.add_argument("--small_max_area", type=int, default=15000)
    p.add_argument("--small_max_span_px", type=int, default=280)
    p.add_argument("--small_border_margin_px", type=int, default=-1)
    p.add_argument("--wrist_fringe_px", type=int, default=40)
    p.add_argument("--front_fringe_px", type=int, default=60)
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
    p.add_argument("--rs_spatial_holes_fill", type=int, default=5)
    p.add_argument("--rs_temporal_alpha", type=float, default=0.75)
    p.add_argument("--rs_temporal_delta", type=int, default=1)
    p.add_argument("--timeout_ms", type=int, default=5000)
    p.add_argument("--jpeg_quality", type=int, default=92)
    return p.parse_args()


if __name__ == "__main__":
    main()
