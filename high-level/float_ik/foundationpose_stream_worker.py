#!/usr/bin/env python3
"""Persistent FoundationPose worker for streamed Isaac Gym RGB-D frames."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose_root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--register_iter", type=int, default=5)
    parser.add_argument("--track_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--min_mask_pixels", type=int, default=20)
    parser.add_argument(
        "--reregister_every",
        type=int,
        default=1,
        help="Re-run global registration every N requests; 0 uses tracking after the first frame.",
    )
    return parser.parse_args()


def rotation_error_deg(estimate: np.ndarray, target: np.ndarray) -> float:
    relative = estimate[:3, :3] @ target[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = args.foundationpose_root.resolve()
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    pose_dir = args.output_dir / "poses"
    overlay_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)

    mesh = trimesh.load(args.mesh.resolve(), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
        _ = mesh.vertex_normals
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(args.output_dir / "debug"),
        debug=args.debug,
        glctx=glctx,
    )
    # FoundationPose stores the previous pose on the estimator instance.  A
    # parallel Isaac Gym rollout interleaves frames from several independent
    # cameras, so keep that state per stream and restore it before track_one.
    # Requests without a stream_id retain the original single-stream behavior.
    pose_states_by_stream: dict[str, object] = {}
    requests_by_stream: dict[str, int] = {}
    atomic_json(
        args.output_dir / "worker_ready.json",
        {"status": "ready", "mesh": str(args.mesh.resolve())},
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("command") == "shutdown":
            break
        result_path = Path(request["result_path"])
        frame_id = int(request["frame_id"])
        stream_id = str(request.get("stream_id", "default"))
        try:
            with np.load(request["input_path"]) as frame:
                rgb = np.asarray(frame["rgb"], dtype=np.uint8)
                depth = np.asarray(frame["depth"], dtype=np.float32)
                mask = np.asarray(frame["mask"], dtype=np.uint8).astype(bool)
                K = np.asarray(frame["K"], dtype=np.float64).reshape(3, 3)
                gt_pose = np.asarray(frame["gt_T_camera_handle"], dtype=np.float64).reshape(4, 4)
            depth[~np.isfinite(depth) | (depth < 0.02) | (depth > 10.0)] = 0.0
            mask_pixels = int(mask.sum())
            stream_request_index = int(requests_by_stream.get(stream_id, 0))
            requests_by_stream[stream_id] = stream_request_index + 1
            initialized = stream_id in pose_states_by_stream
            should_register = (not initialized) or (
                args.reregister_every > 0
                and stream_request_index % args.reregister_every == 0
            )
            if should_register:
                if mask_pixels < args.min_mask_pixels:
                    if not initialized:
                        atomic_json(
                            result_path,
                            {"status": "waiting_for_mask", "frame_id": frame_id, "mask_pixels": mask_pixels},
                        )
                        continue
                    should_register = False
            if should_register:
                pose = estimator.register(
                    K=K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.register_iter,
                )
                mode = "register"
            else:
                # pose_last is the centered-mesh CUDA tensor, not the public
                # object-frame 4x4 returned by register/track_one.
                estimator.pose_last = pose_states_by_stream[stream_id].clone()
                pose = estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_iter)
                mode = "track"

            pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            pose_states_by_stream[stream_id] = estimator.pose_last.detach().clone()
            safe_stream_id = "".join(
                character if character.isalnum() or character in ("-", "_") else "_"
                for character in stream_id
            )
            pose_path = pose_dir / f"{safe_stream_id}_frame_{frame_id:06d}.txt"
            np.savetxt(pose_path, pose, fmt="%.10f")
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
            overlay_path = overlay_dir / f"{safe_stream_id}_frame_{frame_id:06d}.png"
            imageio.imwrite(overlay_path, overlay)
            translation_error = float(np.linalg.norm(pose[:3, 3] - gt_pose[:3, 3]))
            angle_error = rotation_error_deg(pose, gt_pose)
            payload = {
                "status": "ok",
                "mode": mode,
                "frame_id": frame_id,
                "stream_id": stream_id,
                "stream_request_index": stream_request_index,
                "mask_pixels": mask_pixels,
                "T_camera_handle": pose.tolist(),
                "gt_T_camera_handle": gt_pose.tolist(),
                "translation_camera_m": pose[:3, 3].tolist(),
                "gt_translation_camera_m": gt_pose[:3, 3].tolist(),
                "translation_error_m": translation_error,
                "rotation_error_deg": angle_error,
                "pose_path": str(pose_path),
                "overlay_path": str(overlay_path),
            }
            atomic_json(result_path, payload)
            atomic_json(args.output_dir / "latest_pose.json", payload)
        except Exception as exc:
            atomic_json(
                result_path,
                {
                    "status": "error",
                    "frame_id": frame_id,
                    "stream_id": stream_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


if __name__ == "__main__":
    main()
