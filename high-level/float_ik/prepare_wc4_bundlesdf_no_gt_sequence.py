#!/usr/bin/env python3
"""Prepare a GT-free BundleSDF sequence from captured front-camera RGB-D.

Only RGB, metric depth, SAM3 masks, and camera intrinsics are exported to the
BundleSDF input directory.  Simulator masks and handle poses, when present in
the source capture, are never exported.  The simulator mask may optionally be
read to produce an *evaluation-only* SAM3 IoU report.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from prepare_wc4_foundationpose_model_free_refs import (
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_SAM3_PYTHON,
    DEFAULT_SAM3_ROOT,
    DEFAULT_SOURCE,
    Sam3Worker,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[0]
    / "logs/foundationpose/bundlesdf_wc4_front_sam3_no_gt_60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--sam3_root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3_python", type=Path, default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--sam3_checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--sam3_prompt", default="door handle")
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.005)
    parser.add_argument("--sam3_min_mask_pixels", type=int, default=20)
    parser.add_argument("--sam3_max_mask_pixels", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--skip_gt_mask_report",
        action="store_true",
        help="Do not even read the source simulator mask for evaluation reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("stride must be at least 1")
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_files = sorted(source_dir.glob("frame_*.npz"))[:: args.stride]
    if args.max_frames > 0:
        source_files = source_files[: args.max_frames]
    if len(source_files) < 2:
        raise ValueError(f"Need at least two source frames, found {len(source_files)}")

    # A fresh directory makes it impossible for stale pose files from an older
    # preparation run to leak into BundleSDF.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for name in ("rgb", "depth", "masks"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    worker = Sam3Worker(args, output_dir / "sam3")
    reports: list[dict] = []
    K_reference: np.ndarray | None = None
    try:
        for output_index, source_path in enumerate(source_files):
            with np.load(source_path) as frame:
                rgb = np.asarray(frame["rgb"], dtype=np.uint8)
                depth = np.asarray(frame["depth"], dtype=np.float32)
                K = np.asarray(frame["K"], dtype=np.float64).reshape(3, 3)
                gt_mask_eval_only = None
                if not args.skip_gt_mask_report and "mask" in frame.files:
                    gt_mask_eval_only = np.asarray(frame["mask"], dtype=np.uint8).astype(bool)

            mask, sam_report = worker.segment(output_index, rgb)
            valid_depth = np.isfinite(depth) & (depth >= 0.05) & (depth <= 5.0)
            mask = np.asarray(mask, dtype=bool) & valid_depth
            if int(mask.sum()) < args.sam3_min_mask_pixels:
                raise RuntimeError(
                    f"SAM3/depth intersection too small for {source_path.name}: {int(mask.sum())}"
                )

            stem = f"{output_index:06d}"
            cv2.imwrite(
                str(output_dir / "rgb" / f"{stem}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            depth_mm = np.clip(
                np.rint(depth * 1000.0), 0, np.iinfo(np.uint16).max
            ).astype(np.uint16)
            depth_mm[~valid_depth] = 0
            cv2.imwrite(str(output_dir / "depth" / f"{stem}.png"), depth_mm)
            cv2.imwrite(
                str(output_dir / "masks" / f"{stem}.png"),
                mask.astype(np.uint8) * 255,
            )

            report = {
                "output_index": output_index,
                "source_file_name": source_path.name,
                "sam3_mask_pixels": int(mask.sum()),
                "sam3_score": float(sam_report["selected_score"]),
                "sam3_selection_reason": sam_report["selection_reason"],
            }
            if gt_mask_eval_only is not None:
                intersection = int(np.logical_and(mask, gt_mask_eval_only).sum())
                union = int(np.logical_or(mask, gt_mask_eval_only).sum())
                report.update(
                    {
                        "gt_mask_pixels_eval_only": int(gt_mask_eval_only.sum()),
                        "mask_iou_eval_only": float(intersection / max(1, union)),
                    }
                )
            reports.append(report)
            print(
                f"[{output_index + 1:03d}/{len(source_files):03d}] "
                f"{source_path.name} SAM3 pixels={report['sam3_mask_pixels']} "
                f"score={report['sam3_score']:.4f}",
                flush=True,
            )

            if K_reference is None:
                K_reference = K
            elif not np.allclose(K_reference, K, atol=1e-6):
                raise ValueError("Source frames have inconsistent camera intrinsics")
    finally:
        worker.close()

    assert K_reference is not None
    np.savetxt(output_dir / "cam_K.txt", K_reference, fmt="%.10f")
    contract = {
        "pipeline": "front RGB-D + SAM3 masks + K -> BundleSDF",
        "frame_count": len(source_files),
        "source_dir": str(source_dir),
        "exported_modalities": ["rgb", "depth_uint16_mm", "sam3_mask", "camera_intrinsics"],
        "exported_pose_or_transform": False,
        "gt_camera_to_handle_used_by_tracker": False,
        "simulator_mask_used_by_tracker": False,
        "gt_mask_usage": "evaluation-only IoU report" if not args.skip_gt_mask_report else "not read",
    }
    (output_dir / "input_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "sam3_report.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared GT-free BundleSDF input: {output_dir}")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
