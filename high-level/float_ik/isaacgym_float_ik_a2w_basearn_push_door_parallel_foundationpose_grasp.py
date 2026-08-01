#!/usr/bin/env python3
"""Evaluate scripted A2W grasping from a FoundationPose handle estimate.

The original A2W play script is imported unchanged.  This wrapper performs one
front-camera FoundationPose registration for every Isaac Gym environment before
the scripted rollout starts, converts the estimated handle pose to world frame,
and uses that pose while the scripted pregrasp/grasp/rotate waypoints are first
constructed.  Once those waypoints are cached, the original controller runs
unchanged (including its post-contact handle-following logic).

The simulator segmentation mask is used only to initialize FoundationPose.  The
simulator handle pose is sent to the worker only for error reporting and is never
used to construct the controlled trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import isaacgym_float_ik_a2w_basearn_push_door_parallel_foundationpose as fp


base = fp.base
SCRIPT_DIR = Path(__file__).resolve().parent
MASK_USAGE_DESCRIPTION = "simulation GT mask used for one initial registration per environment"


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    fp_cfg, remaining = fp.parse_wrapper_args(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--foundationpose_grasp_summary",
        type=Path,
        default=None,
        help="Optional JSON path for pose errors and final scripted success.",
    )
    parser.add_argument(
        "--foundationpose_grasp_success_angle_deg",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--foundationpose_grasp_sim_safety_max_error_m",
        type=float,
        default=0.015,
        help=(
            "Simulation-only crash guard. Estimates whose GT grasp-point error exceeds this "
            "value are counted as perception failures and never used for contact control. "
            "GT is not used to correct or rescue an estimate."
        ),
    )
    parser.add_argument(
        "--foundationpose_virtual_env_offset",
        type=int,
        default=0,
        help="Map local env i to the original batch env offset+i for deterministic split evaluation.",
    )
    parser.add_argument(
        "--foundationpose_virtual_num_envs",
        type=int,
        default=0,
        help="Original batch env count used while deriving per-env randomization; 0 uses --num_envs.",
    )
    parser.add_argument(
        "--foundationpose_track_during_walk",
        action="store_true",
        help=(
            "Register each environment from its initial front-camera frame, then continuously "
            "track the handle while the base approaches the door."
        ),
    )
    parser.add_argument(
        "--foundationpose_initial_pose_only",
        action="store_true",
        help=(
            "Register each handle on its first reliable front-camera observation, cache its "
            "world-frame grasp point, and never track or update it afterward."
        ),
    )
    parser.add_argument(
        "--foundationpose_walk_track_interval",
        type=int,
        default=5,
        help="Track each environment every N simulator frames during the walk phase.",
    )
    parser.add_argument(
        "--foundationpose_model_grasp_point",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Grasp point in the supplied FoundationPose mesh frame. This is required for "
            "a BundleSDF mesh whose origin is arbitrary. The simulator handle offset is "
            "still used only to compute posthoc grasp error."
        ),
    )
    grasp_cfg, remaining = parser.parse_known_args(remaining)
    if grasp_cfg.foundationpose_initial_pose_only and grasp_cfg.foundationpose_track_during_walk:
        raise ValueError(
            "--foundationpose_initial_pose_only and --foundationpose_track_during_walk "
            "are mutually exclusive"
        )
    grasp_cfg.foundationpose_pose_mode = (
        "first_valid_once"
        if grasp_cfg.foundationpose_initial_pose_only
        else "track_during_walk"
        if grasp_cfg.foundationpose_track_during_walk
        else "end_of_walk_once"
    )
    for key, value in vars(grasp_cfg).items():
        setattr(fp_cfg, key, value)
    # Each environment is independently registered once.  There is no shared
    # temporal tracking state between parallel Isaac Gym environments.
    fp_cfg.foundationpose_reregister_every = (
        0 if grasp_cfg.foundationpose_track_during_walk else 1
    )
    fp_cfg.foundationpose_show_overlay = False
    fp_cfg.foundationpose_async = False
    return fp_cfg, remaining


def matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized XYZW quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    return quat.astype(np.float32)


def camera_inputs(gym, sim, st) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_handle = st.camera_handles.get("front")
    if camera_handle is None:
        raise RuntimeError(f"Environment {st.index} has no front camera.")
    camera_cfg = base.dc.DEFAULT_FRONT_CAMERA_CFG
    width, height = map(int, camera_cfg["resolution"])
    rgb_raw = gym.get_camera_image(sim, st.env, camera_handle, base.gymapi.IMAGE_COLOR)
    depth_raw = gym.get_camera_image(sim, st.env, camera_handle, base.gymapi.IMAGE_DEPTH)
    seg_raw = gym.get_camera_image(sim, st.env, camera_handle, base.gymapi.IMAGE_SEGMENTATION)
    if rgb_raw is None or depth_raw is None or seg_raw is None:
        raise RuntimeError(f"Environment {st.index} front-camera capture failed.")
    rgb = base.dc.camera_color_to_rgb(rgb_raw, height, width)
    depth = np.abs(base.dc.camera_image_to_array(depth_raw, height, width).astype(np.float32))
    depth[~np.isfinite(depth)] = 0.0
    # The normal policy/recording path applies synthetic depth corruption after
    # reading the Isaac Gym image.  FoundationPose captures the camera directly,
    # so explicitly reuse that same augmentation here when requested.  Keeping
    # this in the copied FoundationPose integration avoids changing the original
    # scripted controller while making --enable_depth_noise meaningful for pose
    # estimation experiments.
    if bool(getattr(st.args, "enable_depth_noise", False)):
        near = float(getattr(st.args, "camera_depth_clip_lower", 0.2))
        far = float(getattr(st.args, "camera_depth_clip_far", 1.5))
        depth[depth < near] = 0.0
        depth = np.clip(depth, 0.0, far)
        noise_rng = st.traj.get("foundationpose_depth_noise_rng")
        if noise_rng is None:
            env_seed = int(getattr(st.args, "env_seed", getattr(st.args, "seed", 0)))
            noise_rng = np.random.default_rng(np.random.SeedSequence([env_seed, 0xF09D]))
            st.traj["foundationpose_depth_noise_rng"] = noise_rng
        noise_cfg = base.dc.depth_camera_noise_config_for_args(st.args)
        depth = base.dc.apply_depth_noise(
            depth,
            noise_rng,
            noise_cfg,
            valid_mask=depth >= near,
        )
        depth[depth < near] = 0.0
        depth = np.clip(depth, 0.0, far)
        st.traj["foundationpose_depth_noise_applied"] = bool(
            noise_cfg.enabled and noise_cfg.env_selected is not False
        )
    seg = base.dc.camera_image_to_array(seg_raw, height, width).astype(np.int32)
    mask = seg == int(st.args.handle_seg_id)
    intrinsics = base.dc.camera_intrinsics_from_cfg(camera_cfg)
    K = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    gt_pose = fp.ground_truth_camera_handle(gym, sim, st.env, camera_handle)
    return rgb, depth, mask, K, gt_pose


def register_rendered_env(gym, sim, st, client: fp.FoundationPoseClient) -> dict:
    """Register one already-rendered environment and cache its world grasp XYZ."""

    rgb, depth, mask, K, gt_pose = camera_inputs(gym, sim, st)
    mask_pixels = int(mask.sum())
    if mask_pixels < int(client.cfg.foundationpose_min_mask_pixels):
        raise ValueError(f"handle mask too small: {mask_pixels} pixels")
    frame_id = client.frame_id
    if not client.submit(rgb, depth, mask, K, gt_pose):
        raise RuntimeError("FoundationPose worker unexpectedly has a pending request.")
    client.wait_for_pending_result(client.cfg.foundationpose_first_pose_timeout)
    result_path = client.result_dir / f"frame_{frame_id:06d}.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(f"FoundationPose registration failed for env {st.index}: {payload}")

    cv_from_gym = np.eye(4, dtype=np.float64)
    cv_from_gym[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    gym_from_cv = np.linalg.inv(cv_from_gym)
    T_camera_handle = np.asarray(payload["T_camera_handle"], dtype=np.float64).reshape(4, 4)
    camera_handle = st.camera_handles["front"]
    T_world_camera_gym = fp.gym_transform_matrix(
        gym.get_camera_transform(sim, st.env, camera_handle)
    )
    T_world_handle = T_world_camera_gym @ gym_from_cv @ T_camera_handle
    st.traj["foundationpose_handle_world_matrix"] = T_world_handle
    model_goal_local = np.ones(4, dtype=np.float64)
    model_goal_local[:3] = np.asarray(
        client.cfg.foundationpose_model_grasp_point
        if client.cfg.foundationpose_model_grasp_point is not None
        else st.door.handle_goal_offset,
        dtype=np.float64,
    )
    true_goal_local = np.ones(4, dtype=np.float64)
    true_goal_local[:3] = np.asarray(st.door.handle_goal_offset, dtype=np.float64)
    estimated_goal = (T_world_handle @ model_goal_local)[:3]
    true_goal = (T_world_camera_gym @ gym_from_cv @ gt_pose @ true_goal_local)[:3]
    goal_error = float(np.linalg.norm(estimated_goal - true_goal))
    st.traj["foundationpose_handle_goal_world"] = estimated_goal.astype(np.float32)
    safety_limit = float(client.cfg.foundationpose_grasp_sim_safety_max_error_m)
    control_accepted = safety_limit < 0.0 or goal_error <= safety_limit
    st.traj["foundationpose_grasp_control_accepted"] = bool(control_accepted)
    report = {
        "env": int(st.index),
        "frame_id": int(frame_id),
        "mode": "register_once",
        "mask_pixels": int(payload["mask_pixels"]),
        "handle_origin_error_m": float(payload["translation_error_m"]),
        "rotation_error_deg": float(payload["rotation_error_deg"]),
        "grasp_goal_error_m": goal_error,
        "control_accepted": bool(control_accepted),
        "estimated_grasp_goal_world": estimated_goal.tolist(),
        "true_grasp_goal_world": true_goal.tolist(),
    }
    print(
        f"[FoundationPose grasp env={st.index:02d}] "
        f"origin_err={1000.0 * report['handle_origin_error_m']:.1f}mm "
        f"goal_err={1000.0 * goal_error:.1f}mm "
        f"rot_err={report['rotation_error_deg']:.2f}deg "
        f"mask={report['mask_pixels']} accepted={control_accepted}",
        flush=True,
    )
    return report


def estimate_world_handle_poses(gym, sim, env_states, client: fp.FoundationPoseClient) -> list[dict]:
    """Register every env once and store estimated world-handle poses in traj."""

    # Publish the real initial rigid transforms once.  Never teleport the split
    # A2W base/arm actors for perception: even a graphics-only temporary pose
    # can invalidate Isaac Gym's articulation state in a later IK step.
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    captured = None
    mask_counts = []
    # Multi-env cameras can return an all-zero first segmentation frame until
    # PhysX rigid transforms have been published to the graphics pipeline.
    # Warm up without advancing the scripted trajectory and retry rendering.
    for attempt in range(4):
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        captured = [camera_inputs(gym, sim, st) for st in env_states]
        mask_counts = [int(frame[2].sum()) for frame in captured]
        if all(count >= int(client.cfg.foundationpose_min_mask_pixels) for count in mask_counts):
            break
        print(
            f"[FoundationPose grasp warmup={attempt + 1}] mask_pixels={mask_counts}",
            flush=True,
        )
    if captured is None or not all(
        count >= int(client.cfg.foundationpose_min_mask_pixels) for count in mask_counts
    ):
        raise RuntimeError(
            "Front-camera handle mask unavailable after graphics warmup: "
            f"mask_pixels={mask_counts}"
        )
    cv_from_gym = np.eye(4, dtype=np.float64)
    cv_from_gym[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    gym_from_cv = np.linalg.inv(cv_from_gym)
    reports: list[dict] = []
    for st, frame in zip(env_states, captured):
        rgb, depth, mask, K, gt_pose = frame
        frame_id = client.frame_id
        if not client.submit(rgb, depth, mask, K, gt_pose):
            raise RuntimeError("FoundationPose worker unexpectedly has a pending request.")
        client.wait_for_pending_result(client.cfg.foundationpose_first_pose_timeout)
        result_path = client.result_dir / f"frame_{frame_id:06d}.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"FoundationPose registration failed for env {st.index}: {payload}"
            )
        T_camera_handle = np.asarray(payload["T_camera_handle"], dtype=np.float64).reshape(4, 4)
        camera_handle = st.camera_handles["front"]
        T_world_camera_gym = fp.gym_transform_matrix(
            gym.get_camera_transform(sim, st.env, camera_handle)
        )
        T_world_handle = T_world_camera_gym @ gym_from_cv @ T_camera_handle
        st.traj["foundationpose_handle_world_matrix"] = T_world_handle
        model_goal_local = np.ones(4, dtype=np.float64)
        model_goal_local[:3] = np.asarray(
            client.cfg.foundationpose_model_grasp_point
            if client.cfg.foundationpose_model_grasp_point is not None
            else st.door.handle_goal_offset,
            dtype=np.float64,
        )
        true_goal_local = np.ones(4, dtype=np.float64)
        true_goal_local[:3] = np.asarray(st.door.handle_goal_offset, dtype=np.float64)
        estimated_goal = (T_world_handle @ model_goal_local)[:3]
        true_goal = (T_world_camera_gym @ gym_from_cv @ gt_pose @ true_goal_local)[:3]
        goal_error = float(np.linalg.norm(estimated_goal - true_goal))
        st.traj["foundationpose_handle_goal_world"] = estimated_goal.astype(np.float32)
        safety_limit = float(client.cfg.foundationpose_grasp_sim_safety_max_error_m)
        control_accepted = safety_limit < 0.0 or goal_error <= safety_limit
        st.traj["foundationpose_grasp_control_accepted"] = bool(control_accepted)
        report = {
            "env": int(st.index),
            "frame_id": int(frame_id),
            "mode": "initial_once",
            "sim_step": -1,
            "mask_pixels": int(payload["mask_pixels"]),
            "handle_origin_error_m": float(payload["translation_error_m"]),
            "rotation_error_deg": float(payload["rotation_error_deg"]),
            "grasp_goal_error_m": goal_error,
            "control_accepted": bool(control_accepted),
            "estimated_grasp_goal_world": estimated_goal.tolist(),
            "true_grasp_goal_world": true_goal.tolist(),
        }
        if "sam3_mask_report" in st.traj:
            report["sam3"] = st.traj["sam3_mask_report"]
        reports.append(report)
        print(
            f"[FoundationPose grasp env={st.index:02d}] "
            f"origin_err={1000.0 * report['handle_origin_error_m']:.1f}mm "
            f"goal_err={1000.0 * goal_error:.1f}mm "
            f"rot_err={report['rotation_error_deg']:.2f}deg "
            f"mask={report['mask_pixels']}",
            flush=True,
        )
    return reports


def main() -> None:
    cfg, remaining = parse_args(sys.argv[1:])
    required = [cfg.foundationpose_python, cfg.foundationpose_root, cfg.foundationpose_mesh]
    missing = [str(path) for path in required if not path.expanduser().exists()]
    if missing:
        raise FileNotFoundError(f"Missing FoundationPose dependency: {missing}")

    sys.argv = [sys.argv[0], *remaining, "--enable_front_camera", "--show_camera_images"]
    original_parse_args = base.parse_args
    original_make_env_args = base.make_env_args

    def wrapped_parse_args():
        parsed = original_parse_args()
        parsed.enable_front_camera = True
        parsed.show_camera_images = True
        parsed.camera_rgb = True
        parsed.camera_depth = True
        parsed.camera_seg = True
        parsed.camera_display_interval = 1
        return parsed

    base.parse_args = wrapped_parse_args

    def virtualized_make_env_args(args, local_env_index):
        virtual_total = int(cfg.foundationpose_virtual_num_envs)
        if virtual_total <= 0:
            return original_make_env_args(args, local_env_index)
        virtual_index = int(cfg.foundationpose_virtual_env_offset) + int(local_env_index)
        if virtual_index < 0 or virtual_index >= virtual_total:
            raise ValueError(
                f"Virtual env index {virtual_index} outside [0, {virtual_total})."
            )
        local_total = int(args.num_envs)
        args.num_envs = virtual_total
        try:
            env_args = original_make_env_args(args, virtual_index)
        finally:
            args.num_envs = local_total
        env_args.foundationpose_virtual_env_index = virtual_index
        return env_args

    base.make_env_args = virtualized_make_env_args
    original_trajectory_targets = base.trajectory_targets
    original_run_parallel_demo = base.run_parallel_demo
    original_get_actor_dof_state = base.get_actor_dof_state
    client_holder: dict[str, fp.FoundationPoseClient] = {}
    pose_reports: list[dict] = []
    final_report: dict = {}
    runtime: dict = {"env_states": None, "callback_step": 0}

    def capture_at_end_of_walk(gym, sim, unused_env, unused_camera_handles, unused_args):
        env_states = runtime.get("env_states")
        client = client_holder.get("client")
        step = int(runtime.get("callback_step", 0))
        runtime["callback_step"] = step + 1
        if env_states is None or client is None:
            return
        if cfg.foundationpose_initial_pose_only:
            interval = max(1, int(cfg.foundationpose_walk_track_interval))
            for st in env_states:
                if "foundationpose_handle_goal_world" in st.traj:
                    continue
                walk_end = max(0, int(st.args.walk_steps) - 1)
                overdue_attempts = int(st.traj.get("foundationpose_once_overdue_attempts", 0))
                due = step % interval == 0 or step == walk_end or step > walk_end
                if not due:
                    continue
                try:
                    report = register_rendered_env(gym, sim, st, client)
                    report["mode"] = "first_valid_register"
                    report["sim_step"] = int(step)
                    pose_reports.append(report)
                    print(
                        f"[FoundationPose once env={st.index:02d} step={step:04d}] "
                        "first reliable pose cached; no later tracking will run.",
                        flush=True,
                    )
                except ValueError as exc:
                    if step > walk_end:
                        overdue_attempts += 1
                        st.traj["foundationpose_once_overdue_attempts"] = overdue_attempts
                    if overdue_attempts >= 5:
                        st.traj["foundationpose_handle_world_matrix"] = np.eye(
                            4, dtype=np.float64
                        )
                        st.traj["foundationpose_handle_goal_world"] = np.zeros(
                            3, dtype=np.float32
                        )
                        st.traj["foundationpose_grasp_control_accepted"] = False
                        pose_reports.append(
                            {
                                "env": int(st.index),
                                "mode": "first_valid_register_failed",
                                "sim_step": int(step),
                                "status": "no_reliable_visual_initialization",
                                "control_accepted": False,
                            }
                        )
                        print(
                            f"[FoundationPose once env={st.index:02d}] rejected after "
                            f"{overdue_attempts} post-walk attempts: {exc}",
                            flush=True,
                        )
            if all("foundationpose_handle_goal_world" in st.traj for st in env_states):
                client.close()
                client_holder.pop("client", None)
                print(
                    "FoundationPose one-shot registration worker closed; no tracking enabled.",
                    flush=True,
                )
            return
        if cfg.foundationpose_track_during_walk:
            interval = max(1, int(cfg.foundationpose_walk_track_interval))
            cv_from_gym = np.eye(4, dtype=np.float64)
            cv_from_gym[:3, :3] = np.asarray(
                [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
            gym_from_cv = np.linalg.inv(cv_from_gym)
            for st in env_states:
                walk_end = max(0, int(st.args.walk_steps) - 1)
                initialized = bool(st.traj.get("foundationpose_tracking_initialized", False))
                due = step % interval == 0 or step == walk_end
                if step > walk_end or not due:
                    continue
                try:
                    rgb, depth, mask, K, gt_pose = camera_inputs(gym, sim, st)
                    frame_id = client.frame_id
                    if not client.submit(
                        rgb,
                        depth,
                        mask,
                        K,
                        gt_pose,
                        stream_id=f"env_{int(st.index):03d}",
                    ):
                        raise RuntimeError("FoundationPose worker unexpectedly has a pending request.")
                    client.wait_for_pending_result(client.cfg.foundationpose_first_pose_timeout)
                    result_path = client.result_dir / f"frame_{frame_id:06d}.json"
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    if payload.get("status") == "waiting_for_mask":
                        continue
                    if payload.get("status") != "ok":
                        raise RuntimeError(
                            f"FoundationPose tracking failed for env {st.index}: {payload}"
                        )
                    st.traj["foundationpose_tracking_initialized"] = True
                    T_camera_handle = np.asarray(
                        payload["T_camera_handle"], dtype=np.float64
                    ).reshape(4, 4)
                    camera_handle = st.camera_handles["front"]
                    T_world_camera_gym = fp.gym_transform_matrix(
                        gym.get_camera_transform(sim, st.env, camera_handle)
                    )
                    T_world_handle = T_world_camera_gym @ gym_from_cv @ T_camera_handle
                    model_goal_local = np.ones(4, dtype=np.float64)
                    model_goal_local[:3] = np.asarray(
                        cfg.foundationpose_model_grasp_point
                        if cfg.foundationpose_model_grasp_point is not None
                        else st.door.handle_goal_offset,
                        dtype=np.float64,
                    )
                    true_goal_local = np.ones(4, dtype=np.float64)
                    true_goal_local[:3] = np.asarray(st.door.handle_goal_offset, dtype=np.float64)
                    estimated_goal = (T_world_handle @ model_goal_local)[:3]
                    true_goal = (T_world_camera_gym @ gym_from_cv @ gt_pose @ true_goal_local)[:3]
                    goal_error = float(np.linalg.norm(estimated_goal - true_goal))
                    safety_limit = float(cfg.foundationpose_grasp_sim_safety_max_error_m)
                    control_accepted = safety_limit < 0.0 or goal_error <= safety_limit
                    st.traj["foundationpose_handle_world_matrix"] = T_world_handle
                    st.traj["foundationpose_handle_goal_world"] = estimated_goal.astype(np.float32)
                    st.traj["foundationpose_grasp_control_accepted"] = bool(control_accepted)
                    report = {
                        "env": int(st.index),
                        "sim_step": int(step),
                        "frame_id": int(frame_id),
                        "mode": str(payload["mode"]),
                        "mask_pixels": int(payload["mask_pixels"]),
                        "handle_origin_error_m": float(payload["translation_error_m"]),
                        "rotation_error_deg": float(payload["rotation_error_deg"]),
                        "grasp_goal_error_m": goal_error,
                        "control_accepted": bool(control_accepted),
                        "estimated_grasp_goal_world": estimated_goal.tolist(),
                        "true_grasp_goal_world": true_goal.tolist(),
                    }
                    if "sam3_mask_report" in st.traj and payload["mode"] == "register":
                        report["sam3"] = st.traj["sam3_mask_report"]
                    pose_reports.append(report)
                    print(
                        f"[FoundationPose walk env={st.index:02d} step={step:04d} "
                        f"mode={payload['mode']}] goal_err={1000.0 * goal_error:.1f}mm "
                        f"rot_err={float(payload['rotation_error_deg']):.2f}deg "
                        f"accepted={control_accepted}",
                        flush=True,
                    )
                except ValueError as exc:
                    if not initialized:
                        print(
                            f"[FoundationPose walk env={st.index:02d} step={step:04d}] "
                            f"waiting for initial mask: {exc}",
                            flush=True,
                        )
                if (
                    step >= walk_end
                    and "foundationpose_handle_goal_world" not in st.traj
                ):
                    # No visually valid SAM3 initialization became available
                    # during the approach.  Record a perception failure and
                    # hold safely; do not abort the remaining parallel trials.
                    st.traj["foundationpose_handle_world_matrix"] = np.eye(
                        4, dtype=np.float64
                    )
                    st.traj["foundationpose_handle_goal_world"] = np.zeros(
                        3, dtype=np.float32
                    )
                    st.traj["foundationpose_grasp_control_accepted"] = False
                    pose_reports.append(
                        {
                            "env": int(st.index),
                            "sim_step": int(step),
                            "status": "no_visual_initialization",
                            "control_accepted": False,
                        }
                    )
                    print(
                        f"[FoundationPose walk env={st.index:02d}] no valid visual "
                        "initialization before grasp; trial marked failed.",
                        flush=True,
                    )
            if all(
                step >= max(0, int(st.args.walk_steps) - 1)
                and "foundationpose_handle_goal_world" in st.traj
                for st in env_states
            ):
                client.close()
                client_holder.pop("client", None)
                print("FoundationPose tracking worker closed before grasp control.", flush=True)
            return
        for st in env_states:
            if "foundationpose_handle_goal_world" in st.traj:
                continue
            if step < max(0, int(st.args.walk_steps) - 1):
                continue
            try:
                pose_reports.append(register_rendered_env(gym, sim, st, client))
            except ValueError as exc:
                wait_frames = int(st.traj.get("foundationpose_mask_wait_frames", 0)) + 1
                st.traj["foundationpose_mask_wait_frames"] = wait_frames
                if wait_frames >= 5:
                    st.traj["foundationpose_handle_world_matrix"] = np.eye(4, dtype=np.float64)
                    st.traj["foundationpose_handle_goal_world"] = np.zeros(3, dtype=np.float32)
                    st.traj["foundationpose_grasp_control_accepted"] = False
                    pose_reports.append(
                        {
                            "env": int(st.index),
                            "status": "no_visible_mask",
                            "mask_pixels": 0,
                            "control_accepted": False,
                        }
                    )
                    print(
                        f"[FoundationPose grasp env={st.index:02d}] rejected after "
                        f"{wait_frames} frames without a visible handle mask.",
                        flush=True,
                    )
                elif wait_frames == 1:
                    print(
                        f"[FoundationPose grasp env={st.index:02d} step={step}] waiting: {exc}",
                        flush=True,
                    )
        if all("foundationpose_handle_goal_world" in st.traj for st in env_states):
            client.close()
            client_holder.pop("client", None)
            print("FoundationPose registration worker closed before grasp control.", flush=True)

    # ``show_camera_images`` keeps camera rendering active in the original
    # loop. Replace its UI callback with the end-of-walk registration hook.
    base.show_camera_handle_images = capture_at_end_of_walk

    def estimated_trajectory_targets(
        step,
        args,
        door,
        gym,
        env,
        door_actor,
        ik_state,
        base_start,
        base_stop,
        base_push,
        base_traverse,
        yaw_start,
        yaw_push,
        yaw_traverse,
        traj,
    ):
        estimate = traj.get("foundationpose_handle_world_matrix")
        if cfg.foundationpose_track_during_walk and int(step) < int(args.walk_steps):
            # Tracking updates the estimate throughout the approach.  Do not
            # let an early estimate cache pregrasp/grasp waypoints before the
            # camera reaches the final approach view.
            return original_trajectory_targets(
                step, args, door, gym, env, door_actor, ik_state, base_start, base_stop,
                base_push, base_traverse, yaw_start, yaw_push, yaw_traverse, traj,
            )
        if estimate is None and int(step) >= int(args.walk_steps):
            # Hold the physically reached base-stop view until a subsequent
            # rendered frame provides a usable handle mask.  Do not call the
            # original target builder here, because that would cache the GT
            # grasp waypoint before FoundationPose becomes available.
            yaw_stop = float(yaw_start + base.move_to_approach_yaw_delta(args))
            handle_pos, handle_quat = base.get_body_pose(
                gym, env, door_actor, door.handle_body_index
            )
            handle_goal = (
                base.quat_apply(handle_quat, door.handle_goal_offset) + handle_pos
            )
            safe_pos = (
                np.asarray(ik_state.current_pos_np, dtype=np.float32).copy()
                if ik_state.current_pos_np is not None
                else np.asarray(handle_goal, dtype=np.float32).copy()
            )
            safe_quat = (
                None
                if args.ik_position_only or ik_state.current_quat_np is None
                else np.asarray(ik_state.current_quat_np, dtype=np.float32).copy()
            )
            return (
                "foundationpose_wait",
                np.asarray(base_stop, dtype=np.float32).copy(),
                yaw_stop,
                safe_pos,
                safe_quat,
                float(args.gripper_open),
                np.asarray(handle_goal, dtype=np.float32),
            )
        control_rejected = estimate is not None and not bool(
            traj.get("foundationpose_grasp_control_accepted", True)
        )
        if estimate is None or "pregrasp" in traj or control_rejected:
            result = original_trajectory_targets(
                step, args, door, gym, env, door_actor, ik_state, base_start, base_stop,
                base_push, base_traverse, yaw_start, yaw_push, yaw_traverse, traj,
            )
            if control_rejected:
                phase, base_xy, yaw, unused_pos, unused_quat, unused_gripper, handle_goal = result
                if "foundationpose_rejected_hold_base_xy" not in traj:
                    traj["foundationpose_rejected_hold_base_xy"] = np.asarray(
                        base_xy, dtype=np.float32
                    ).copy()
                    traj["foundationpose_rejected_hold_yaw"] = float(yaw)
                safe_pos = (
                    np.asarray(ik_state.current_pos_np, dtype=np.float32).copy()
                    if ik_state.current_pos_np is not None
                    else np.asarray(unused_pos, dtype=np.float32).copy()
                )
                safe_quat = (
                    None
                    if args.ik_position_only or ik_state.current_quat_np is None
                    else np.asarray(ik_state.current_quat_np, dtype=np.float32).copy()
                )
                return (
                    phase,
                    np.asarray(traj["foundationpose_rejected_hold_base_xy"], dtype=np.float32).copy(),
                    float(traj["foundationpose_rejected_hold_yaw"]),
                    safe_pos,
                    safe_quat,
                    float(args.gripper_open),
                    handle_goal,
                )
            return result
        original_get_body_pose = base.get_body_pose
        estimated_goal = np.asarray(
            traj["foundationpose_handle_goal_world"], dtype=np.float32
        )

        def estimated_get_body_pose(gym_arg, env_arg, actor_arg, body_index):
            if env_arg == env and actor_arg == door_actor and int(body_index) == int(door.handle_body_index):
                # This experiment replaces the manually/simulator-located
                # grasp XYZ only.  Preserve the original WC4 handle direction
                # used by approach and push planning; otherwise a symmetric
                # 180-degree FoundationPose solution also reverses the entire
                # scripted interaction direction and confounds the comparison.
                true_position, true_quaternion = original_get_body_pose(
                    gym_arg, env_arg, actor_arg, body_index
                )
                offset_world = base.quat_apply(
                    true_quaternion,
                    np.asarray(door.handle_goal_offset, dtype=np.float32),
                )
                synthetic_origin = estimated_goal - offset_world
                return synthetic_origin.astype(np.float32), true_quaternion
            return original_get_body_pose(gym_arg, env_arg, actor_arg, body_index)

        base.get_body_pose = estimated_get_body_pose
        try:
            return original_trajectory_targets(
                step, args, door, gym, env, door_actor, ik_state, base_start, base_stop,
                base_push, base_traverse, yaw_start, yaw_push, yaw_traverse, traj,
            )
        finally:
            base.get_body_pose = original_get_body_pose

    base.trajectory_targets = estimated_trajectory_targets

    def wrapped_run_parallel_demo(gym, sim, env_states, viewer, args, dt, dof_names):
        client = fp.FoundationPoseClient(cfg)
        client_holder["client"] = client
        runtime["env_states"] = env_states
        runtime["callback_step"] = 0
        if cfg.foundationpose_initial_pose_only:
            print(
                "FoundationPose pose mode: first_valid_once; each environment will register "
                "once and will never be tracked afterward.",
                flush=True,
            )
        maximum_abs_angle = {int(st.index): 0.0 for st in env_states}
        env_actor_to_index = {(id(st.env), int(st.door_actor)): int(st.index) for st in env_states}

        def tracked_get_actor_dof_state(gym_arg, env_arg, actor_arg):
            position, velocity = original_get_actor_dof_state(gym_arg, env_arg, actor_arg)
            env_index = env_actor_to_index.get((id(env_arg), int(actor_arg)))
            if env_index is not None and len(position):
                maximum_abs_angle[env_index] = max(
                    maximum_abs_angle[env_index], abs(float(position[0]))
                )
            return position, velocity

        base.get_actor_dof_state = tracked_get_actor_dof_state
        try:
            original_run_parallel_demo(gym, sim, env_states, viewer, args, dt, dof_names)
        finally:
            base.get_actor_dof_state = original_get_actor_dof_state
        missing_estimates = [
            int(st.index)
            for st in env_states
            if "foundationpose_handle_goal_world" not in st.traj
        ]
        # A handle that never becomes confidently visible is a legitimate
        # perception failure in closed-loop evaluation, not a process error.
        # The controller has already kept that environment in the safe
        # foundationpose_wait phase; record it as an unsuccessful trial so the
        # final success rate includes segmentation/registration failures.
        for env_index in missing_estimates:
            pose_reports.append(
                {
                    "env": int(env_index),
                    "mode": "no_visual_initialization",
                    "accepted_for_control": False,
                    "reason": "No valid SAM/FoundationPose initialization before grasp control",
                }
            )
        threshold_rad = math.radians(float(cfg.foundationpose_grasp_success_angle_deg))
        successes = {
            int(st.index): bool(
                maximum_abs_angle[int(st.index)] >= threshold_rad
                and st.traj.get("foundationpose_grasp_control_accepted", False)
            )
            for st in env_states
        }
        perception_accepted = {
            int(st.index): bool(st.traj.get("foundationpose_grasp_control_accepted", False))
            for st in env_states
        }
        final_report.update(
            {
                "success_count": int(sum(successes.values())),
                "num_envs": int(len(env_states)),
                "success_rate": float(sum(successes.values()) / max(1, len(env_states))),
                "success_angle_deg": float(cfg.foundationpose_grasp_success_angle_deg),
                "max_open_angle_deg": {
                    str(index): math.degrees(angle) for index, angle in maximum_abs_angle.items()
                },
                "success_by_env": {str(index): value for index, value in successes.items()},
                "perception_accepted_count": int(sum(perception_accepted.values())),
                "perception_accepted_by_env": {
                    str(index): value for index, value in perception_accepted.items()
                },
                "missing_visual_initialization_envs": missing_estimates,
                "sim_safety_max_grasp_error_m": float(
                    cfg.foundationpose_grasp_sim_safety_max_error_m
                ),
                "virtual_env_offset": int(cfg.foundationpose_virtual_env_offset),
                "virtual_num_envs": int(cfg.foundationpose_virtual_num_envs),
                "pose_reports": pose_reports,
                "control": (
                    "FoundationPose initial grasp XYZ for pregrasp/grasp/rotate waypoint construction; "
                    "nominal WC4 handle direction and original post-contact handle following"
                ),
                "track_during_walk": bool(cfg.foundationpose_track_during_walk),
                "initial_pose_only": bool(cfg.foundationpose_initial_pose_only),
                "pose_mode": str(cfg.foundationpose_pose_mode),
                "walk_track_interval": int(cfg.foundationpose_walk_track_interval),
                "foundationpose_depth_noise_applied_by_env": {
                    str(st.index): bool(
                        st.traj.get("foundationpose_depth_noise_applied", False)
                    )
                    for st in env_states
                },
                "mask_usage": MASK_USAGE_DESCRIPTION,
            }
        )
        print(
            f"FOUNDATIONPOSE_GRASP_SUCCESS {final_report['success_count']}/{final_report['num_envs']} "
            f"= {final_report['success_rate']:.6f}",
            flush=True,
        )
        print(
            "FOUNDATIONPOSE_GRASP_MAX_OPEN_DEG "
            + json.dumps(final_report["max_open_angle_deg"], sort_keys=True),
            flush=True,
        )

    base.run_parallel_demo = wrapped_run_parallel_demo
    try:
        base.main()
    finally:
        client = client_holder.get("client")
        if client is not None:
            client.close()
        if cfg.foundationpose_grasp_summary is not None and final_report:
            summary_path = cfg.foundationpose_grasp_summary.expanduser().resolve()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
            print(f"FoundationPose grasp summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
