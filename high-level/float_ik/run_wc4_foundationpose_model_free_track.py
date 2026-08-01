#!/usr/bin/env python3
"""Register then track WC4 with a BundleSDF-reconstructed handle mesh.

The estimator receives a SAM3 mask only on the first frame.  Subsequent frames
use FoundationPose's official RGB-D tracking path.  Recorded simulator poses
are used solely for error reporting and never enter registration or tracking.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_FP_ROOT = Path("/home/sivan/whole_body/FoundationPose")
DEFAULT_SOURCE = Path(
    "/home/sivan/whole_body/visual_whole_body/high-level/logs/"
    "foundationpose/wc4_front_camera_dense_track_smoke/stream_inputs"
)
DEFAULT_REFS = Path(
    "/home/sivan/whole_body/visual_whole_body/high-level/logs/"
    "foundationpose/model_free_wc4_refs_sam3_16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose_root", type=Path, default=DEFAULT_FP_ROOT)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFS)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument(
        "--initial_mask",
        type=Path,
        default=None,
        help="SAM mask for first-frame registration (defaults to reference_dir/mask/000000.png).",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--register_iter", type=int, default=5)
    parser.add_argument("--track_iter", type=int, default=2)
    parser.add_argument("--overlay_interval", type=int, default=10)
    parser.add_argument("--debug", type=int, default=1)
    return parser.parse_args()


def rotation_error_deg(estimate: np.ndarray, target: np.ndarray) -> float:
    relative = estimate[:3, :3] @ target[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def main() -> None:
    args = parse_args()
    root = args.foundationpose_root.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    mesh_path = (
        args.mesh.expanduser().resolve()
        if args.mesh is not None
        else reference_dir / "model/model.obj"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else reference_dir / "tracking_reconstructed_mesh"
    )
    initial_mask_path = (
        args.initial_mask.expanduser().resolve()
        if args.initial_mask is not None
        else reference_dir / "mask/000000.png"
    )
    for required in (root, source_dir, reference_dir, mesh_path, initial_mask_path):
        if not required.exists():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(root))
    os.chdir(root)
    import cv2
    import imageio.v2 as imageio
    import nvdiffrast.torch as dr
    import trimesh
    from datareader import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    set_logging_format()
    set_seed(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    pose_dir = output_dir / "poses"
    overlay_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    _ = mesh.vertex_normals
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(output_dir / "debug"),
        debug=args.debug,
        glctx=dr.RasterizeCudaContext(),
    )

    frame_files = sorted(source_dir.glob("frame_*.npz"))
    if not frame_files:
        raise RuntimeError(f"No source frames in {source_dir}")
    initial_mask = np.asarray(
        Image.open(initial_mask_path).convert("L"), dtype=np.uint8
    ) > 0
    reports: list[dict] = []
    for frame_index, frame_path in enumerate(frame_files):
        with np.load(frame_path) as frame:
            rgb = np.asarray(frame["rgb"], dtype=np.uint8).copy()
            depth = np.asarray(frame["depth"], dtype=np.float32).copy()
            K = np.asarray(frame["K"], dtype=np.float64).reshape(3, 3)
            gt_pose = np.asarray(frame["gt_T_camera_handle"], dtype=np.float64).reshape(4, 4)
        depth[~np.isfinite(depth) | (depth < 0.05) | (depth > 5.0)] = 0.0
        if frame_index == 0:
            pose = estimator.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=initial_mask,
                iteration=args.register_iter,
            )
            mode = "register"
        else:
            pose = estimator.track_one(
                rgb=rgb,
                depth=depth,
                K=K,
                iteration=args.track_iter,
            )
            mode = "track"
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        translation_error = float(np.linalg.norm(pose[:3, 3] - gt_pose[:3, 3]))
        angle_error = rotation_error_deg(pose, gt_pose)
        report = {
            "frame_index": frame_index,
            "source_file": str(frame_path),
            "mode": mode,
            "translation_error_m": translation_error,
            "rotation_error_deg": angle_error,
            "T_camera_handle": pose.tolist(),
            "gt_T_camera_handle_eval_only": gt_pose.tolist(),
        }
        reports.append(report)
        np.savetxt(pose_dir / f"frame_{frame_index:06d}.txt", pose, fmt="%.10f")
        if (
            frame_index == 0
            or frame_index == len(frame_files) - 1
            or frame_index % max(1, args.overlay_interval) == 0
        ):
            center_pose = pose @ np.linalg.inv(to_origin)
            overlay = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
            overlay = draw_xyz_axis(
                overlay,
                ob_in_cam=center_pose,
                scale=float(max(0.03, min(0.15, np.max(extents)))),
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            imageio.imwrite(overlay_dir / f"frame_{frame_index:06d}.png", overlay)
        print(
            f"[{frame_index:03d}/{len(frame_files) - 1:03d} {mode}] "
            f"translation={1000.0 * translation_error:.2f}mm rotation={angle_error:.2f}deg",
            flush=True,
        )

    translation_errors = np.asarray([r["translation_error_m"] for r in reports])
    rotation_errors = np.asarray([r["rotation_error_deg"] for r in reports])
    summary = {
        "status": "ok",
        "method": "FoundationPose model-free reconstructed mesh: register first frame, track remainder",
        "mesh": str(mesh_path),
        "source_dir": str(source_dir),
        "num_frames": len(reports),
        "initial_mask": "SAM3 RGB text-prompt mask",
        "subsequent_mask_usage": "none",
        "gt_pose_usage": "evaluation-only",
        "translation_error_mm": {
            "mean": float(1000.0 * translation_errors.mean()),
            "median": float(1000.0 * np.median(translation_errors)),
            "p95": float(1000.0 * np.percentile(translation_errors, 95)),
            "max": float(1000.0 * translation_errors.max()),
            "final": float(1000.0 * translation_errors[-1]),
        },
        "rotation_error_deg": {
            "mean": float(rotation_errors.mean()),
            "median": float(np.median(rotation_errors)),
            "p95": float(np.percentile(rotation_errors, 95)),
            "max": float(rotation_errors.max()),
            "final": float(rotation_errors[-1]),
        },
        "frames": reports,
    }
    (output_dir / "tracking_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
