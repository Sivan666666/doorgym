#!/usr/bin/env python3
"""Front-camera aggressive OpenCV inpaint comparison on one bag frame.

This helper is intentionally narrower than offline_opencv_inpaint_step_compare.py:
it focuses on the front camera case where all small/mid black components are
already filled, but the remaining largest black component is still too large to
fill as a whole.  The tested variants add only a thin inpaint mask fringe inside
large black connected components:

    I  : small/mid components only
    J1 : I + large-component internal fringe 8 px
    J2 : I + large-component internal fringe 12 px
    J3 : I + large-component internal fringe 20 px
    J4 : I + large-component internal fringe 30 px

The goal is to fill front-camera edge holes more aggressively without erasing an
entire foreground silhouette.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from offline_opencv_inpaint_step_compare import (
    depth_m_to_policy_u8,
    label_image,
    make_small_black_component_mask,
    read_base_filtered_depth_at_step,
)


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
        is_small_like = area <= small_max_area and width <= small_max_span_px and height <= small_max_span_px
        if is_small_like:
            continue
        component = (labels == label).astype(np.uint8)
        # Pixels just inside the invalid region boundary.  This avoids filling
        # the deep core of a large foreground/unknown silhouette.
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 3)
        fringe = ((dist > 0.0) & (dist <= float(fringe_px))).astype(np.uint8) * 255
        added = int(np.count_nonzero((fringe > 0) & (out == 0)))
        out = np.maximum(out, fringe)
        added_total += added
        large_components += 1
    return out, {
        "large_components": int(large_components),
        "fringe_px": int(fringe_px),
        "fringe_added_px": int(added_total),
        "mask_px": int(np.count_nonzero(out)),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

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
    front_u8 = depth_m_to_policy_u8(
        cv2.resize(front_depth, (args.output_width, args.output_height), interpolation=cv2.INTER_LINEAR),
        args.depth_lower_m,
        args.depth_far_m,
    )
    base_mask, base_stats = make_small_black_component_mask(
        front_u8,
        max_area=args.small_max_area,
        max_span_px=args.small_max_span_px,
        min_area=1,
        border_margin_px=args.small_border_margin_px,
    )

    variants: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    i_img = cv2.inpaint(front_u8, base_mask, 3.0, cv2.INPAINT_TELEA)
    variants.append(
        (
            "I front small/mid TELEA r3",
            i_img,
            {
                "mask_px": int(np.count_nonzero(base_mask)),
                "changed_px": int(np.count_nonzero(i_img != front_u8)),
                "fringe_px": 0,
            },
        )
    )
    for idx, fringe_px in enumerate(args.fringe_px, start=1):
        mask, stats = add_large_component_fringe_mask(
            front_u8,
            base_mask,
            small_max_area=args.small_max_area,
            small_max_span_px=args.small_max_span_px,
            fringe_px=int(fringe_px),
        )
        img = cv2.inpaint(front_u8, mask, 3.0, cv2.INPAINT_TELEA)
        stats["changed_px"] = int(np.count_nonzero(img != front_u8))
        variants.append((f"J{idx} front fringe {int(fringe_px)}px", img, stats))

    front_cols = []
    if args.ae_snapshot and args.ae_snapshot.exists():
        ae = cv2.imread(str(args.ae_snapshot), cv2.IMREAD_COLOR)
        if ae is None:
            raise RuntimeError(f"failed to read {args.ae_snapshot}")
        # Existing A-E grid is two rows: wrist over front.
        front_cols.append(ae[ae.shape[0] // 2 :, :, :])
    for label, img, stats in variants:
        sublabel = f"mask {stats['mask_px']}px changed {stats['changed_px']}px"
        front_cols.append(label_image(img, label, sublabel))
    front_grid = np.concatenate(front_cols, axis=1)

    mask_cols = [
        label_image((front_u8 == 0).astype(np.uint8) * 255, "front all black mask"),
        label_image(base_mask, "front I selected mask", f"{int(np.count_nonzero(base_mask))} px"),
    ]
    for idx, fringe_px in enumerate(args.fringe_px, start=1):
        mask, stats = add_large_component_fringe_mask(
            front_u8,
            base_mask,
            small_max_area=args.small_max_area,
            small_max_span_px=args.small_max_span_px,
            fringe_px=int(fringe_px),
        )
        mask_cols.append(label_image(mask, f"J{idx} mask fringe {int(fringe_px)}px", f"{stats['mask_px']} px"))
    mask_grid = np.concatenate(mask_cols, axis=1)

    front_png = args.out_dir / f"front_AE_plus_I_J_fringe_step_{args.step:06d}.png"
    front_jpg = args.out_dir / f"front_AE_plus_I_J_fringe_step_{args.step:06d}.jpg"
    mask_png = args.out_dir / f"front_I_J_fringe_masks_step_{args.step:06d}.png"
    cv2.imwrite(str(front_png), front_grid)
    cv2.imwrite(str(front_jpg), front_grid, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])
    cv2.imwrite(str(mask_png), mask_grid)

    manifest = {
        "script": Path(__file__).name,
        "step": int(args.step),
        "base_pipeline": "align(color) -> spatial_filter(holes_fill=0) -> temporal_filter -> clip 0.2-1.5m -> u8",
        "front_black_px": int(np.count_nonzero(front_u8 == 0)),
        "small_component_mask": {
            "max_area": int(args.small_max_area),
            "max_span_px": int(args.small_max_span_px),
            "border_margin_px": int(args.small_border_margin_px),
            "stats": base_stats,
        },
        "fringe_px": [int(v) for v in args.fringe_px],
        "variants": [
            {"label": label, **stats}
            for label, _img, stats in variants
        ],
        "outputs": {
            "front_png": str(front_png),
            "front_jpg": str(front_jpg),
            "mask_png": str(mask_png),
        },
        "elapsed_s": time.time() - t0,
    }
    manifest_path = args.out_dir / f"front_I_J_fringe_step_{args.step:06d}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("front_fringe_step_done " + json.dumps(manifest, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--front_bag", type=Path, required=True)
    p.add_argument("--ae_snapshot", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--step", type=int, default=360)
    p.add_argument("--output_width", type=int, default=640)
    p.add_argument("--output_height", type=int, default=480)
    p.add_argument("--depth_lower_m", type=float, default=0.2)
    p.add_argument("--depth_far_m", type=float, default=1.5)
    p.add_argument("--small_max_area", type=int, default=15000)
    p.add_argument("--small_max_span_px", type=int, default=280)
    p.add_argument("--small_border_margin_px", type=int, default=-1)
    p.add_argument("--fringe_px", type=int, nargs="+", default=[8, 12, 20, 30])
    p.add_argument("--rs_spatial_magnitude", type=int, default=2)
    p.add_argument("--rs_spatial_alpha", type=float, default=0.75)
    p.add_argument("--rs_spatial_delta", type=int, default=50)
    p.add_argument("--rs_temporal_alpha", type=float, default=0.75)
    p.add_argument("--rs_temporal_delta", type=int, default=1)
    p.add_argument("--timeout_ms", type=int, default=5000)
    p.add_argument("--jpeg_quality", type=int, default=92)
    return p.parse_args()


if __name__ == "__main__":
    main()
