#!/usr/bin/env python3
"""Evaluate BundleSDF tracking after the GT-free run has completed.

The tracker receives no simulator pose.  Ground truth is loaded here only to
align BundleSDF's arbitrary first-frame object coordinates and report errors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rotation_error_deg(estimate: np.ndarray, target: np.ndarray) -> float:
    relative = estimate[:3, :3] @ target[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--bundlesdf_output", type=Path, required=True)
    parser.add_argument(
        "--pose_dir",
        type=Path,
        default=None,
        help="Optional directory of 4x4 pose txt files; defaults to bundlesdf_output/ob_in_cam.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    source_files = sorted(args.source_dir.expanduser().resolve().glob("frame_*.npz"))
    pose_dir = (
        args.pose_dir.expanduser().resolve()
        if args.pose_dir is not None
        else args.bundlesdf_output.expanduser().resolve() / "ob_in_cam"
    )
    pose_files = sorted(pose_dir.glob("*.txt"))
    if not source_files or len(source_files) != len(pose_files):
        raise RuntimeError(f"Frame/pose count mismatch: {len(source_files)} vs {len(pose_files)}")

    estimates = [np.loadtxt(path).reshape(4, 4) for path in pose_files]
    targets = []
    for path in source_files:
        with np.load(path) as frame:
            targets.append(np.asarray(frame["gt_T_camera_handle"], dtype=np.float64).reshape(4, 4))

    # BundleSDF chooses its own object frame.  One fixed first-frame transform
    # maps that arbitrary frame to the simulator handle frame.  No later GT is
    # used for alignment or correction.
    bundlesdf_object_from_handle = np.linalg.inv(estimates[0]) @ targets[0]
    aligned = [pose @ bundlesdf_object_from_handle for pose in estimates]
    translation_mm = np.asarray(
        [1000.0 * np.linalg.norm(pose[:3, 3] - gt[:3, 3]) for pose, gt in zip(aligned, targets)]
    )
    rotation_deg = np.asarray(
        [rotation_error_deg(pose, gt) for pose, gt in zip(aligned, targets)]
    )

    output = args.output or (args.bundlesdf_output / "no_gt_tracking_evaluation.json")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "tracker_inputs": ["front RGB", "front depth", "SAM3 mask", "camera intrinsics"],
        "gt_used_during_tracking_or_reconstruction": False,
        "gt_usage_here": "posthoc evaluation and one fixed first-frame canonical-frame alignment only",
        "num_frames": len(aligned),
        "translation_error_mm": {
            "mean": float(translation_mm.mean()),
            "median": float(np.median(translation_mm)),
            "p95": float(np.percentile(translation_mm, 95)),
            "max": float(translation_mm.max()),
            "final": float(translation_mm[-1]),
        },
        "rotation_error_deg": {
            "mean": float(rotation_deg.mean()),
            "median": float(np.median(rotation_deg)),
            "p95": float(np.percentile(rotation_deg, 95)),
            "max": float(rotation_deg.max()),
            "final": float(rotation_deg[-1]),
        },
        "bundlesdf_object_from_handle_eval_alignment": bundlesdf_object_from_handle.tolist(),
        "frames": [
            {
                "frame": index,
                "translation_error_mm": float(t_error),
                "rotation_error_deg": float(r_error),
            }
            for index, (t_error, r_error) in enumerate(zip(translation_mm, rotation_deg))
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure_path = output.with_suffix(".png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(translation_mm, color="#2878B5", linewidth=2)
    axes[0].set_ylabel("Translation error (mm)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(rotation_deg, color="#C82423", linewidth=2)
    axes[1].set_ylabel("Rotation error (deg)")
    axes[1].set_xlabel("Frame")
    axes[1].grid(alpha=0.25)
    fig.suptitle("WC4 BundleSDF GT-free tracking (GT used posthoc only)")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    print(f"report={output}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
