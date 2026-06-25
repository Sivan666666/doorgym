#!/usr/bin/env python3
"""Compare Gaussian blur parameters after full-res opencv_k_no_rs depth processing.

Pipeline per camera:

    raw bag -> align(color) -> no RealSense filters -> clip/normalize u8
    -> runtime opencv_k full-res inpaint -> GaussianBlur variants

The output is a labelled wrist/front grid for quick visual inspection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from door_act_shadow import opencv_k_inpaint_policy_gray
from offline_opencv_dual_aggressive_step_compare import labelled_tile, read_depth_at_step
from offline_opencv_inpaint_step_compare import depth_m_to_policy_u8


DEFAULT_BLUR_SPECS = (
    "none",
    "3:0",
    "5:0",
    "5:1.0",
    "7:1.5",
    "9:2.0",
    "11:3.0",
    "15:4.0",
)


def parse_blur_spec(text: str) -> tuple[str, int, float | None]:
    item = str(text).strip().lower()
    if item in ("none", "off", "0"):
        return "none", 0, None
    if ":" in item:
        k_text, sigma_text = item.split(":", 1)
    else:
        k_text, sigma_text = item, "0"
    ksize = int(k_text)
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError(f"Gaussian kernel size must be a positive odd integer, got {ksize!r}")
    sigma = float(sigma_text)
    label = f"k{ksize}_sigma{sigma:g}"
    return label, ksize, sigma


def make_variants(opencv_k_u8: np.ndarray, specs: list[str]) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    out: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for spec in specs:
        label, ksize, sigma = parse_blur_spec(spec)
        if ksize <= 0:
            image = np.asarray(opencv_k_u8, dtype=np.uint8).copy()
            stats = {"ksize": 0, "sigma": None, "changed_px": 0, "mean_abs_delta": 0.0}
        else:
            image = cv2.GaussianBlur(
                np.asarray(opencv_k_u8, dtype=np.uint8),
                (ksize, ksize),
                float(sigma),
                borderType=cv2.BORDER_DEFAULT,
            )
            delta = np.abs(image.astype(np.int16) - opencv_k_u8.astype(np.int16))
            stats = {
                "ksize": int(ksize),
                "sigma": float(sigma),
                "changed_px": int(np.count_nonzero(delta)),
                "mean_abs_delta": float(delta.mean()),
                "max_abs_delta": int(delta.max()),
            }
        out.append((label, image, stats))
    return out


def process_camera(
    bag_path: Path,
    camera_name: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[tuple[str, np.ndarray, dict[str, Any]]]]:
    depth_m = read_depth_at_step(
        bag_path,
        int(args.step),
        spatial=False,
        temporal=False,
        hole_filling=False,
        timeout_ms=int(args.timeout_ms),
        spatial_magnitude=int(args.rs_spatial_magnitude),
        spatial_alpha=float(args.rs_spatial_alpha),
        spatial_delta=int(args.rs_spatial_delta),
        spatial_holes_fill=int(args.rs_spatial_holes_fill),
        temporal_alpha=float(args.rs_temporal_alpha),
        temporal_delta=int(args.rs_temporal_delta),
    )
    if depth_m.shape != (int(args.output_height), int(args.output_width)):
        depth_m = cv2.resize(
            depth_m.astype(np.float32),
            (int(args.output_width), int(args.output_height)),
            interpolation=cv2.INTER_LINEAR,
        )
    base_u8 = depth_m_to_policy_u8(depth_m, float(args.depth_lower_m), float(args.depth_far_m))
    opencv_k_u8, opencv_k_stats = opencv_k_inpaint_policy_gray(base_u8, camera_name=camera_name)
    variants = make_variants(opencv_k_u8, list(args.blur))
    return opencv_k_u8, dict(opencv_k_stats), variants


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    wrist_k, wrist_k_stats, wrist_variants = process_camera(args.wrist_bag, "wrist", args)
    front_k, front_k_stats, front_variants = process_camera(args.front_bag, "front", args)

    rows = []
    all_stats: dict[str, Any] = {
        "script": Path(__file__).name,
        "step": int(args.step),
        "pipeline": "align(color) -> no_rs_filters -> clip/normalize -> full-res opencv_k -> GaussianBlur",
        "wrist_bag": str(args.wrist_bag),
        "front_bag": str(args.front_bag),
        "wrist_opencv_k": wrist_k_stats,
        "front_opencv_k": front_k_stats,
        "blur": {},
    }
    for camera_name, variants in (("wrist", wrist_variants), ("front", front_variants)):
        tiles = []
        all_stats["blur"][camera_name] = {}
        for label, image, stats in variants:
            all_stats["blur"][camera_name][label] = stats
            sub = (
                "opencv_k no blur"
                if label == "none"
                else f"Gaussian {stats['ksize']}x{stats['ksize']} sigma={stats['sigma']} "
                f"mad={stats['mean_abs_delta']:.2f}"
            )
            tiles.append(
                labelled_tile(
                    image,
                    f"{camera_name} step {int(args.step):04d} {label}",
                    sub,
                    width=int(args.tile_width),
                    height=int(args.tile_height),
                    label_h=int(args.label_height),
                )
            )
            individual_path = args.out_dir / f"{camera_name}_step{int(args.step):04d}_{label}.png"
            cv2.imwrite(str(individual_path), image)
        rows.append(np.concatenate(tiles, axis=1))

    grid = np.concatenate(rows, axis=0)
    grid_path = args.out_dir / f"dual_opencv_k_no_rs_gaussian_blur_step_{int(args.step):04d}.png"
    cv2.imwrite(str(grid_path), grid)

    all_stats["grid"] = str(grid_path)
    all_stats["elapsed_s"] = time.time() - t0
    manifest_path = args.out_dir / f"dual_opencv_k_no_rs_gaussian_blur_step_{int(args.step):04d}_manifest.json"
    manifest_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False) + "\n")
    print("gaussian_blur_step_compare_done " + json.dumps(all_stats, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrist_bag", type=Path, required=True)
    p.add_argument("--front_bag", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--step", type=int, default=313)
    p.add_argument("--blur", nargs="+", default=list(DEFAULT_BLUR_SPECS))
    p.add_argument("--output_width", type=int, default=640)
    p.add_argument("--output_height", type=int, default=480)
    p.add_argument("--tile_width", type=int, default=320)
    p.add_argument("--tile_height", type=int, default=240)
    p.add_argument("--label_height", type=int, default=58)
    p.add_argument("--depth_lower_m", type=float, default=0.2)
    p.add_argument("--depth_far_m", type=float, default=1.5)
    p.add_argument("--timeout_ms", type=int, default=5000)
    # Kept only to reuse the shared reader signature; no RS filters are enabled.
    p.add_argument("--rs_spatial_magnitude", type=int, default=2)
    p.add_argument("--rs_spatial_alpha", type=float, default=0.75)
    p.add_argument("--rs_spatial_delta", type=int, default=50)
    p.add_argument("--rs_spatial_holes_fill", type=int, default=5)
    p.add_argument("--rs_temporal_alpha", type=float, default=0.75)
    p.add_argument("--rs_temporal_delta", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    main()
