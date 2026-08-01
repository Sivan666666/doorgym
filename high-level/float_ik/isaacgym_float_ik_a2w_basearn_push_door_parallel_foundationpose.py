#!/usr/bin/env python3
"""A2W door play with asynchronous front-camera FoundationPose tracking.

This is a non-invasive wrapper around
``isaacgym_float_ik_a2w_basearn_push_door_parallel.py``. The original module is
imported unchanged; only its camera display callback is replaced at runtime.
Isaac Gym remains in the b1z1 environment while FoundationPose runs in its own
persistent Python subprocess.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import isaacgym_float_ik_a2w_basearn_push_door_parallel as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FP_ROOT = Path("/home/sivan/whole_body/FoundationPose")
DEFAULT_FP_PYTHON = Path("/home/sivan/miniconda3/envs/foundationpose/bin/python")
DEFAULT_MESH = (
    SCRIPT_DIR.parents[0]
    / "data/asset/door_set/wc4/assets/meshes/lever_handle_foundationpose.obj"
)
DEFAULT_OUTPUT = SCRIPT_DIR.parents[0] / "logs/foundationpose/wc4_front_camera"


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--foundationpose_root", type=Path, default=DEFAULT_FP_ROOT)
    parser.add_argument("--foundationpose_python", type=Path, default=DEFAULT_FP_PYTHON)
    parser.add_argument("--foundationpose_mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--foundationpose_output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--foundationpose_interval", type=int, default=1)
    parser.add_argument("--foundationpose_register_iter", type=int, default=5)
    parser.add_argument("--foundationpose_track_iter", type=int, default=2)
    parser.add_argument("--foundationpose_debug", type=int, default=1)
    parser.add_argument("--foundationpose_min_mask_pixels", type=int, default=20)
    parser.add_argument(
        "--foundationpose_reregister_every",
        type=int,
        default=0,
        help="Use the simulation mask to globally re-register every N requests; 0 is pure tracking.",
    )
    parser.add_argument("--foundationpose_startup_timeout", type=float, default=600.0)
    parser.add_argument("--foundationpose_first_pose_timeout", type=float, default=180.0)
    parser.add_argument(
        "--foundationpose_async",
        action="store_true",
        help="Do not pause simulation for tracking results; faster but may skip too much motion.",
    )
    parser.add_argument("--foundationpose_show_overlay", action="store_true", default=True)
    parser.add_argument("--no_foundationpose_show_overlay", dest="foundationpose_show_overlay", action="store_false")
    parser.add_argument("--foundationpose_show_original_cameras", action="store_true", default=False)
    return parser.parse_known_args(argv)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def gym_transform_matrix(transform) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_xyzw_to_matrix(
        np.array([transform.r.x, transform.r.y, transform.r.z, transform.r.w])
    )
    matrix[:3, 3] = [transform.p.x, transform.p.y, transform.p.z]
    return matrix


def ground_truth_camera_handle(gym, sim, env, camera_handle) -> np.ndarray:
    door_actor = -1
    handle_body = -1
    for actor_index in range(gym.get_actor_count(env)):
        actor = gym.get_actor_handle(env, actor_index)
        candidate = gym.find_actor_rigid_body_handle(env, actor, "lever_handle")
        if candidate >= 0:
            door_actor = actor
            handle_body = candidate
            break
    if door_actor < 0 or handle_body < 0:
        raise RuntimeError("Could not find an actor containing the lever_handle rigid body.")
    world_camera_gym = gym_transform_matrix(gym.get_camera_transform(sim, env, camera_handle))
    world_handle = gym_transform_matrix(gym.get_rigid_transform(env, handle_body))
    camera_gym_handle = np.linalg.inv(world_camera_gym) @ world_handle
    # Isaac Gym camera (+X forward, +Y left, +Z up) -> OpenCV camera
    # (+Z forward, +X right, +Y down).
    cv_from_gym = np.eye(4, dtype=np.float64)
    cv_from_gym[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    return cv_from_gym @ camera_gym_handle


class FoundationPoseClient:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.output_dir = cfg.foundationpose_output_dir.expanduser().resolve()
        self.input_dir = self.output_dir / "stream_inputs"
        self.result_dir = self.output_dir / "stream_results"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        ready_path = self.output_dir / "worker_ready.json"
        ready_path.unlink(missing_ok=True)
        self.log_handle = (self.output_dir / "foundationpose_worker.log").open("w", encoding="utf-8")
        command = [
            str(cfg.foundationpose_python.expanduser().resolve()),
            "-u",
            str(SCRIPT_DIR / "foundationpose_stream_worker.py"),
            "--foundationpose_root",
            str(cfg.foundationpose_root.expanduser().resolve()),
            "--mesh",
            str(cfg.foundationpose_mesh.expanduser().resolve()),
            "--output_dir",
            str(self.output_dir),
            "--register_iter",
            str(cfg.foundationpose_register_iter),
            "--track_iter",
            str(cfg.foundationpose_track_iter),
            "--debug",
            str(cfg.foundationpose_debug),
            "--min_mask_pixels",
            str(cfg.foundationpose_min_mask_pixels),
            "--reregister_every",
            str(cfg.foundationpose_reregister_every),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.pending: tuple[int, Path] | None = None
        self.frame_id = 0
        self.latest_overlay: Path | None = None
        print(f"FoundationPose worker started: pid={self.process.pid} log={self.log_handle.name}", flush=True)
        deadline = time.monotonic() + float(cfg.foundationpose_startup_timeout)
        while not ready_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"FoundationPose worker exited during startup with code {self.process.returncode}; "
                    f"see {self.log_handle.name}"
                )
            if time.monotonic() >= deadline:
                self.process.kill()
                raise TimeoutError(
                    f"FoundationPose did not initialize within {cfg.foundationpose_startup_timeout:g}s; "
                    f"see {self.log_handle.name}"
                )
            time.sleep(0.2)
        print("FoundationPose models loaded; streaming is ready.", flush=True)

    def poll(self) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"FoundationPose worker exited with code {self.process.returncode}; see {self.log_handle.name}"
            )
        if self.pending is None:
            return
        frame_id, result_path = self.pending
        if not result_path.exists():
            return
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.pending = None
        status = payload.get("status")
        if status == "ok":
            self.latest_overlay = Path(payload["overlay_path"])
            print(
                f"[FoundationPose frame={frame_id:06d} mode={payload['mode']}] "
                f"t_cam={np.round(payload['translation_camera_m'], 4).tolist()} "
                f"GT={np.round(payload['gt_translation_camera_m'], 4).tolist()} "
                f"pos_err={1000.0 * float(payload['translation_error_m']):.1f}mm "
                f"rot_err={float(payload['rotation_error_deg']):.2f}deg "
                f"mask={payload['mask_pixels']}",
                flush=True,
            )
        elif status == "waiting_for_mask":
            print(f"[FoundationPose] waiting for visible handle mask; pixels={payload['mask_pixels']}", flush=True)
        else:
            print(f"[FoundationPose] worker error: {payload.get('error')} (result={result_path})", flush=True)

    def submit(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        gt_pose: np.ndarray,
        stream_id: str = "default",
    ) -> bool:
        self.poll()
        if self.pending is not None:
            return False
        frame_id = self.frame_id
        self.frame_id += 1
        input_path = self.input_dir / f"frame_{frame_id:06d}.npz"
        result_path = self.result_dir / f"frame_{frame_id:06d}.json"
        result_path.unlink(missing_ok=True)
        np.savez(
            input_path,
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            mask=np.asarray(mask, dtype=np.uint8),
            K=np.asarray(K, dtype=np.float64),
            gt_T_camera_handle=np.asarray(gt_pose, dtype=np.float64),
        )
        request = {
            "frame_id": frame_id,
            "stream_id": str(stream_id),
            "input_path": str(input_path),
            "result_path": str(result_path),
        }
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        self.pending = (frame_id, result_path)
        return True

    def wait_for_pending_result(self, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while self.pending is not None:
            self.poll()
            if self.pending is None:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"FoundationPose first pose exceeded {timeout:g}s")
            time.sleep(0.1)

    def show_overlay(self) -> None:
        if not self.cfg.foundationpose_show_overlay or self.latest_overlay is None or base.cv2 is None:
            return
        overlay = base.cv2.imread(str(self.latest_overlay), base.cv2.IMREAD_COLOR)
        if overlay is not None:
            base.cv2.imshow("FoundationPose: Front Handle Pose", overlay)
            base.cv2.waitKey(1)

    def close(self) -> None:
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=8)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=3)
        finally:
            self.log_handle.close()


def main() -> None:
    cfg, remaining = parse_wrapper_args(sys.argv[1:])
    required = [cfg.foundationpose_python, cfg.foundationpose_root, cfg.foundationpose_mesh]
    missing = [str(path) for path in required if not path.expanduser().exists()]
    if missing:
        raise FileNotFoundError(f"Missing FoundationPose dependency: {missing}")

    # These flags keep the front sensor rendering even when original camera
    # windows are suppressed by this wrapper.
    sys.argv = [sys.argv[0], *remaining, "--enable_front_camera", "--show_camera_images"]
    original_parse_args = base.parse_args

    def tracking_parse_args():
        parsed = original_parse_args()
        # The original script intentionally disables OpenCV previews in
        # headless mode and for --no_show_seg. This dedicated wrapper still
        # needs camera rendering even when no windows are requested.
        parsed.enable_front_camera = True
        parsed.show_camera_images = True
        parsed.camera_rgb = True
        parsed.camera_depth = True
        parsed.camera_seg = True
        return parsed

    base.parse_args = tracking_parse_args
    original_show = base.show_camera_handle_images
    holder: dict[str, FoundationPoseClient] = {}
    callback_count = 0

    def tracked_show(gym, sim, env, camera_handles, args):
        nonlocal callback_count
        callback_count += 1
        if cfg.foundationpose_show_original_cameras:
            original_show(gym, sim, env, camera_handles, args)
        client = holder.get("client")
        if client is None:
            client = FoundationPoseClient(cfg)
            holder["client"] = client
        client.poll()
        client.show_overlay()
        if callback_count % max(1, int(cfg.foundationpose_interval)) != 0:
            return
        camera_handle = camera_handles.get("front")
        if camera_handle is None or client.pending is not None:
            return
        camera_cfg = base.dc.DEFAULT_FRONT_CAMERA_CFG
        width, height = map(int, camera_cfg["resolution"])
        rgb_raw = gym.get_camera_image(sim, env, camera_handle, base.gymapi.IMAGE_COLOR)
        depth_raw = gym.get_camera_image(sim, env, camera_handle, base.gymapi.IMAGE_DEPTH)
        seg_raw = gym.get_camera_image(sim, env, camera_handle, base.gymapi.IMAGE_SEGMENTATION)
        if rgb_raw is None or depth_raw is None or seg_raw is None:
            return
        rgb = base.dc.camera_color_to_rgb(rgb_raw, height, width)
        depth = np.abs(base.dc.camera_image_to_array(depth_raw, height, width).astype(np.float32))
        depth[~np.isfinite(depth)] = 0.0
        seg = base.dc.camera_image_to_array(seg_raw, height, width).astype(np.int32)
        mask = seg == int(args.handle_seg_id)
        intrinsics = base.dc.camera_intrinsics_from_cfg(camera_cfg)
        K = np.array(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        gt_pose = ground_truth_camera_handle(gym, sim, env, camera_handle)
        is_first_request = client.frame_id == 0
        submitted = client.submit(rgb, depth, mask, K, gt_pose)
        if submitted and (is_first_request or not cfg.foundationpose_async):
            client.wait_for_pending_result(cfg.foundationpose_first_pose_timeout)

    base.show_camera_handle_images = tracked_show
    try:
        base.main()
    finally:
        client = holder.get("client")
        if client is not None:
            client.poll()
            client.close()


if __name__ == "__main__":
    main()
