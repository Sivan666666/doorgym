#!/usr/bin/env python3
"""Solve and visualize the recorded fire-door EE trajectory with Z1 TRAC-IK.

Left:  piecewise-linear interpolation of sequential TRAC-IK joint solutions.
Right: globally quintic-smoothed trajectory through the same IK solutions.

The default episode contains world-frame replay EE poses. They are transformed
to the recorded robot-base frame, then adjusted for the mount-origin difference
between a2wz1.urdf and the standalone z1_arm.urdf used by TRAC-IK.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SCRIPT_DIR = Path(__file__).resolve().parent
IK_SOLVERS_DIR = SCRIPT_DIR / "ik_solvers"
if str(IK_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(IK_SOLVERS_DIR))

from ik_solvers.z1_tracik_kinematics import (
    DEFAULT_Z1_EE_LINK,
    DEFAULT_Z1_URDF,
    Z1TracIKKinematics,
)
from joint_trajectory_smoothing import continuity_report, smooth_joint_waypoints
from play_z1_joint_trajectory_smoothing import (
    DEFAULT_A2W_ASSET_FILE,
    DEFAULT_A2W_ASSET_ROOT,
    DEFAULT_ASSET_FILE,
    DEFAULT_ASSET_ROOT,
    Z1TrajectoryPlay,
    piecewise_linear_samples,
)


HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_EPISODE = (
    HIGH_LEVEL_ROOT
    / "data"
    / "door_dp_raw"
    / "fire_door_a2w_state10_episode0"
    / "episode_000000.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=str, default=str(DEFAULT_EPISODE))
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_stop", type=int, default=-1, help="Exclusive; -1 uses the full episode.")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--method", choices=("quintic", "quintic_hermite", "septic"), default="quintic")
    parser.add_argument("--dt", type=float, default=0.02, help="Gym playback and smoothed trajectory sample period.")
    parser.add_argument("--tracik_timeout", type=float, default=0.005)
    parser.add_argument("--tracik_epsilon", type=float, default=1.0e-5)
    parser.add_argument("--tracik_solver_type", choices=("Speed", "Distance", "Manip1", "Manip2"), default="Speed")
    parser.add_argument("--max_restarts", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--draw_waypoint_stride", type=int, default=25)
    parser.add_argument("--report_json", type=str, default="")

    parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--asset_file", type=str, default=DEFAULT_ASSET_FILE)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 1.05, -1.45, 0.75, 0.0, 0.0))
    parser.add_argument("--root_z", type=float, default=0.50)
    parser.add_argument("--hold_time", type=float, default=0.75)
    parser.add_argument("--no_loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--disable_arm_visual_flip", action="store_true")
    parser.add_argument("--show_a2w_base", dest="show_a2w_base", action="store_true", default=True)
    parser.add_argument("--no_show_a2w_base", dest="show_a2w_base", action="store_false")
    parser.add_argument("--a2w_asset_root", type=str, default=str(DEFAULT_A2W_ASSET_ROOT))
    parser.add_argument("--a2w_asset_file", type=str, default=DEFAULT_A2W_ASSET_FILE)

    # Attributes consumed by the reusable Gym player but not used for this episode.
    parser.set_defaults(
        input="",
        input_key="q",
        max_vel=0.0,
        segment_time=1.0,
        num_waypoints=5,
        waypoint_delta=0.5,
    )
    return parser.parse_args()


def joint_origin(urdf_path: Path, joint_name: str) -> np.ndarray:
    root = ET.parse(Path(urdf_path).expanduser().resolve()).getroot()
    for joint in root.findall("joint"):
        if joint.get("name") != joint_name:
            continue
        origin = joint.find("origin")
        if origin is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray([float(value) for value in origin.get("xyz", "0 0 0").split()], dtype=np.float64)
    raise KeyError(f"Joint {joint_name!r} not found in {urdf_path}")


def xyzw_pose_error(targets: np.ndarray, achieved: np.ndarray):
    targets = np.asarray(targets, dtype=np.float64)
    achieved = np.asarray(achieved, dtype=np.float64)
    pos = np.linalg.norm(targets[:, :3] - achieved[:, :3], axis=1)
    delta = Rotation.from_quat(achieved[:, 3:7]).inv() * Rotation.from_quat(targets[:, 3:7])
    return pos, delta.magnitude()


def metric_summary(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def interpolate_poses(times: np.ndarray, poses: np.ndarray, sample_t: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    sample_t = np.asarray(sample_t, dtype=np.float64)
    position = np.column_stack(
        [np.interp(sample_t, times, poses[:, axis]) for axis in range(3)]
    )
    orientation = Slerp(times, Rotation.from_quat(poses[:, 3:7]))(sample_t).as_quat()
    return np.column_stack([position, orientation])


def summarize_line(name: str, values: np.ndarray, unit: str) -> str:
    stats = metric_summary(values)
    return (
        f"  {name:<30} mean={stats['mean']:.6g} {unit} "
        f"p50={stats['p50']:.6g} p95={stats['p95']:.6g} "
        f"p99={stats['p99']:.6g} max={stats['max']:.6g}"
    )


class FireDoorTracIKPlay(Z1TrajectoryPlay):
    def __init__(self, args, tracik, gymapi, gymutil):
        self.tracik = tracik
        self.episode_report = {}
        super().__init__(args, tracik, gymapi, gymutil)
        self.viewer_controls_text = "Viewer controls: S=re-solve episode, R=replay, Space=pause/resume, Esc=quit."
        self.comparison_text = (
            "Left orange: linear TRAC-IK joints. Right blue: global quintic TRAC-IK trajectory."
        )

    def _load_episode_targets(self):
        episode_path = Path(self.args.episode).expanduser().resolve()
        if not episode_path.exists():
            raise FileNotFoundError(episode_path)

        with np.load(episode_path, allow_pickle=True) as data:
            required = ("replay_root_state", "replay_ee_pos", "replay_ee_quat", "replay_dof_pos")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"Episode is missing fields: {missing}")

            root_state = np.asarray(data["replay_root_state"], dtype=np.float64)
            ee_world_pos = np.asarray(data["replay_ee_pos"], dtype=np.float64)
            ee_world_quat = np.asarray(data["replay_ee_quat"], dtype=np.float64)
            recorded_dof = np.asarray(data["replay_dof_pos"], dtype=np.float64)
            fps = float(np.asarray(data["record_effective_fps"]).item()) if "record_effective_fps" in data.files else float(np.asarray(data["fps"]).item())
            pose_frame = str(np.asarray(data["ee_pose_frame"]).item()) if "ee_pose_frame" in data.files else "unknown"

        frame_count = len(root_state)
        start = int(np.clip(self.args.frame_start, 0, frame_count - 1))
        stop = frame_count if int(self.args.frame_stop) < 0 else int(np.clip(self.args.frame_stop, start + 1, frame_count))
        stride = max(1, int(self.args.frame_stride))
        indices = np.arange(start, stop, stride, dtype=np.int64)
        if indices.size < 2:
            raise ValueError("Selected episode range must contain at least two frames")

        root_rot = Rotation.from_quat(root_state[indices, 3:7])
        ee_rot = Rotation.from_quat(ee_world_quat[indices])
        local_pos = root_rot.inv().apply(ee_world_pos[indices] - root_state[indices, :3])
        local_quat = (root_rot.inv() * ee_rot).as_quat()
        data_poses = np.column_stack([local_pos, local_quat])

        standalone_mount = joint_origin(Path(self.args.urdf), "base_static_joint")
        a2w_urdf = Path(self.args.a2w_asset_root).expanduser().resolve() / self.args.a2w_asset_file
        a2w_mount = joint_origin(a2w_urdf, "base_static_joint")
        mount_correction = standalone_mount - a2w_mount
        tracik_poses = data_poses.copy()
        tracik_poses[:, :3] += mount_correction[None, :]

        if recorded_dof.shape[1] < 7:
            raise ValueError(f"replay_dof_pos must contain six arm joints and gripper, got {recorded_dof.shape}")
        recorded_q = recorded_dof[indices, -7:-1]
        recorded_gripper = recorded_dof[indices, -1]
        times = (indices - indices[0]).astype(np.float64) / fps
        return {
            "path": episode_path,
            "indices": indices,
            "fps": fps,
            "pose_frame": pose_frame,
            "data_poses": data_poses,
            "tracik_poses": tracik_poses,
            "recorded_q": recorded_q,
            "recorded_gripper": recorded_gripper,
            "times": times,
            "mount_correction": mount_correction,
        }

    def _solve_episode_ik(self, episode):
        targets = episode["tracik_poses"]
        seed = episode["recorded_q"][0].copy()
        rng = np.random.default_rng(int(self.args.seed))

        for _ in range(max(0, int(self.args.warmup))):
            self.tracik.ik(targets[0], seed_joints=seed, max_restarts=1, strict_seed=True)

        solutions = []
        times_ms = []
        first_pass_times_ms = []
        retry_times_ms = []
        retry_frames = []
        for local_index, target in enumerate(targets):
            start_ns = time.perf_counter_ns()
            first_start_ns = start_ns
            solution = self.tracik.ik(target, seed_joints=seed, max_restarts=1, strict_seed=True)
            first_pass_times_ms.append((time.perf_counter_ns() - first_start_ns) * 1.0e-6)
            if solution is None and int(self.args.max_restarts) > 1:
                retry_frames.append(int(episode["indices"][local_index]))
                retry_start_ns = time.perf_counter_ns()
                solution = self.tracik.ik(
                    target,
                    seed_joints=seed,
                    max_restarts=int(self.args.max_restarts),
                    strict_seed=False,
                    rng=rng,
                )
                retry_times_ms.append((time.perf_counter_ns() - retry_start_ns) * 1.0e-6)
            times_ms.append((time.perf_counter_ns() - start_ns) * 1.0e-6)
            if solution is None:
                frame = int(episode["indices"][local_index])
                raise RuntimeError(f"TRAC-IK failed at episode frame {frame}, including {self.args.max_restarts} restarts")
            solution = np.asarray(solution, dtype=np.float64).reshape(6)
            solutions.append(solution)
            seed = solution

        return {
            "q": np.asarray(solutions, dtype=np.float64),
            "times_ms": np.asarray(times_ms, dtype=np.float64),
            "first_pass_times_ms": np.asarray(first_pass_times_ms, dtype=np.float64),
            "retry_times_ms": np.asarray(retry_times_ms, dtype=np.float64),
            "retry_frames": retry_frames,
        }

    def _fk_poses(self, q_values):
        return np.asarray([self.tracik.fk(q) for q in np.asarray(q_values, dtype=np.float64)])

    def _build_report(self, episode, ik_result, raw, smooth, raw_gripper, smooth_gripper, smoothing_ms):
        ik_fk = self._fk_poses(ik_result["q"])
        ik_pos, ik_rot = xyzw_pose_error(episode["tracik_poses"], ik_fk)

        desired_samples = interpolate_poses(episode["times"], episode["tracik_poses"], smooth["t"])
        raw_fk = self._fk_poses(raw["q"])
        smooth_fk = self._fk_poses(smooth["q"])
        raw_pos, raw_rot = xyzw_pose_error(desired_samples, raw_fk)
        smooth_pos, smooth_rot = xyzw_pose_error(desired_samples, smooth_fk)
        jumps = continuity_report(smooth)
        gripper_jumps = continuity_report(smooth_gripper)
        desired_gripper = np.interp(smooth_gripper["t"], episode["times"], episode["recorded_gripper"])
        raw_gripper_error = np.abs(raw_gripper["q"][:, 0] - desired_gripper)
        smooth_gripper_error = np.abs(smooth_gripper["q"][:, 0] - desired_gripper)

        times_ms = ik_result["times_ms"]
        report = {
            "episode": str(episode["path"]),
            "frames": int(len(episode["indices"])),
            "first_frame": int(episode["indices"][0]),
            "last_frame": int(episode["indices"][-1]),
            "record_fps": float(episode["fps"]),
            "playback_dt_s": float(self.args.dt),
            "duration_s": float(episode["times"][-1]),
            "mount_correction_m": episode["mount_correction"].tolist(),
            "ik": {
                "success": int(len(ik_result["q"])),
                "failure": 0,
                "first_pass_success": int(len(ik_result["q"]) - len(ik_result["retry_frames"])),
                "retry_count": int(len(ik_result["retry_frames"])),
                "retry_frames": ik_result["retry_frames"],
                "time_ms": metric_summary(times_ms),
                "first_pass_time_ms": metric_summary(ik_result["first_pass_times_ms"]),
                "retry_time_ms": metric_summary(ik_result["retry_times_ms"]) if len(ik_result["retry_times_ms"]) else None,
                "total_time_ms": float(np.sum(times_ms)),
                "effective_hz": float(1000.0 * len(times_ms) / np.sum(times_ms)),
                "position_error_m": metric_summary(ik_pos),
                "rotation_error_rad": metric_summary(ik_rot),
            },
            "smoothing": {
                "method": self.args.method,
                "compute_time_ms": float(smoothing_ms),
                "samples": int(len(smooth["t"])),
                "max_abs_velocity_rad_s": float(np.max(np.abs(smooth["qd"]))),
                "max_abs_acceleration_rad_s2": float(np.max(np.abs(smooth["qdd"]))),
                "max_abs_jerk_rad_s3": float(np.max(np.abs(smooth["qddd"]))),
                "knot_jumps": {key: float(value) for key, value in jumps.items()},
            },
            "continuous_tracking": {
                "raw_position_error_m": metric_summary(raw_pos),
                "raw_rotation_error_rad": metric_summary(raw_rot),
                "smooth_position_error_m": metric_summary(smooth_pos),
                "smooth_rotation_error_rad": metric_summary(smooth_rot),
            },
            "gripper": {
                "source": "replay_dof_pos[:, -1]",
                "tracik_input": False,
                "recorded_range_rad": [
                    float(np.min(episode["recorded_gripper"])),
                    float(np.max(episode["recorded_gripper"])),
                ],
                "raw_method": "linear",
                "raw_range_rad": [float(np.min(raw_gripper["q"])), float(np.max(raw_gripper["q"]))],
                "raw_tracking_error_rad": metric_summary(raw_gripper_error),
                "smooth_method": self.args.method,
                "smooth_range_rad": [
                    float(np.min(smooth_gripper["q"])),
                    float(np.max(smooth_gripper["q"])),
                ],
                "smooth_tracking_error_rad": metric_summary(smooth_gripper_error),
                "max_abs_velocity_rad_s": float(np.max(np.abs(smooth_gripper["qd"]))),
                "max_abs_acceleration_rad_s2": float(np.max(np.abs(smooth_gripper["qdd"]))),
                "max_abs_jerk_rad_s3": float(np.max(np.abs(smooth_gripper["qddd"]))),
                "knot_jumps": {key: float(value) for key, value in gripper_jumps.items()},
            },
        }
        return report

    def _print_episode_report(self):
        report = self.episode_report
        ik = report["ik"]
        tracking = report["continuous_tracking"]
        print("\nFire-door replay EE -> TRAC-IK report")
        print(
            f"  frames={report['frames']} range=[{report['first_frame']}, {report['last_frame']}] "
            f"record_fps={report['record_fps']:.3f} duration={report['duration_s']:.3f}s"
        )
        print(f"  mount correction={np.asarray(report['mount_correction_m'])} m")
        print(
            f"  IK success={ik['success']}/{report['frames']} first_pass={ik['first_pass_success']} "
            f"retried={ik['retry_count']} retry_frames={ik['retry_frames']}"
        )
        print(summarize_line("TRAC-IK total time", self.ik_result["times_ms"], "ms"))
        print(f"  TRAC-IK total={ik['total_time_ms']:.3f} ms effective={ik['effective_hz']:.1f} Hz")
        print(summarize_line("IK FK position error", self.ik_position_errors, "m"))
        print(summarize_line("IK FK rotation error", self.ik_rotation_errors, "rad"))
        print(f"  quintic smoothing compute={report['smoothing']['compute_time_ms']:.3f} ms samples={report['smoothing']['samples']}")
        print(
            f"  smooth max_abs: vel={report['smoothing']['max_abs_velocity_rad_s']:.4f} rad/s "
            f"acc={report['smoothing']['max_abs_acceleration_rad_s2']:.4f} rad/s^2 "
            f"jerk={report['smoothing']['max_abs_jerk_rad_s3']:.4f} rad/s^3"
        )
        print(summarize_line("raw continuous position", self.raw_position_errors, "m"))
        print(summarize_line("raw continuous rotation", self.raw_rotation_errors, "rad"))
        print(summarize_line("smooth continuous position", self.smooth_position_errors, "m"))
        print(summarize_line("smooth continuous rotation", self.smooth_rotation_errors, "rad"))
        gripper = report["gripper"]
        print(
            f"  gripper source={gripper['source']} TRAC-IK input={gripper['tracik_input']} "
            f"recorded_range={gripper['recorded_range_rad']} rad"
        )
        print(summarize_line("gripper raw tracking error", self.raw_gripper_errors, "rad"))
        print(summarize_line("gripper smooth tracking error", self.smooth_gripper_errors, "rad"))
        print(
            f"  gripper smooth max_abs: vel={gripper['max_abs_velocity_rad_s']:.4f} rad/s "
            f"acc={gripper['max_abs_acceleration_rad_s2']:.4f} rad/s^2 "
            f"jerk={gripper['max_abs_jerk_rad_s3']:.4f} rad/s^3"
        )

    def new_trajectory(self, use_input=False):
        del use_input
        episode = self._load_episode_targets()
        ik_result = self._solve_episode_ik(episode)

        smooth_start_ns = time.perf_counter_ns()
        smooth = smooth_joint_waypoints(
            ik_result["q"],
            dt=float(self.args.dt),
            times=episode["times"],
            method=self.args.method,
            endpoint_mode="finite_difference",
        )
        smooth_gripper = smooth_joint_waypoints(
            episode["recorded_gripper"][:, None],
            dt=float(self.args.dt),
            times=episode["times"],
            method=self.args.method,
            endpoint_mode="finite_difference",
        )
        smoothing_ms = (time.perf_counter_ns() - smooth_start_ns) * 1.0e-6
        raw = piecewise_linear_samples(ik_result["q"], smooth["waypoint_t"], smooth["t"])
        raw_gripper = piecewise_linear_samples(
            episode["recorded_gripper"][:, None],
            smooth_gripper["waypoint_t"],
            smooth_gripper["t"],
        )

        if not np.allclose(smooth_gripper["t"], smooth["t"], atol=1.0e-12, rtol=0.0):
            raise RuntimeError("Arm and gripper smoothed trajectories have different sample times")

        if np.any(smooth["q"] < self.lower6[None, :] - 1.0e-7) or np.any(smooth["q"] > self.upper6[None, :] + 1.0e-7):
            raise RuntimeError("Smoothed TRAC-IK trajectory exceeds the Z1 joint limits")
        gripper_index = self.dof_names.index("jointGripper")
        if (
            np.any(smooth_gripper["q"][:, 0] < self.lower[gripper_index] - 1.0e-7)
            or np.any(smooth_gripper["q"][:, 0] > self.upper[gripper_index] + 1.0e-7)
        ):
            raise RuntimeError("Smoothed gripper trajectory exceeds the Z1 gripper limits")

        self.trajectory = {
            "waypoints": ik_result["q"],
            "raw": raw,
            "smooth": smooth,
            "gripper": {
                "raw": raw_gripper["q"][:, 0],
                "smooth": smooth_gripper["q"][:, 0],
            },
        }
        self.path_points = {
            key: np.asarray([self._world_position(key, q) for q in self.trajectory[key]["q"]])
            for key in ("raw", "smooth")
        }
        draw_stride = max(1, int(self.args.draw_waypoint_stride))
        drawn_q = ik_result["q"][::draw_stride]
        self.waypoint_points = {
            key: np.asarray([self._world_position(key, q) for q in drawn_q])
            for key in ("raw", "smooth")
        }

        desired_samples = interpolate_poses(episode["times"], episode["tracik_poses"], smooth["t"])
        ik_fk = self._fk_poses(ik_result["q"])
        raw_fk = self._fk_poses(raw["q"])
        smooth_fk = self._fk_poses(smooth["q"])
        self.ik_position_errors, self.ik_rotation_errors = xyzw_pose_error(episode["tracik_poses"], ik_fk)
        self.raw_position_errors, self.raw_rotation_errors = xyzw_pose_error(desired_samples, raw_fk)
        self.smooth_position_errors, self.smooth_rotation_errors = xyzw_pose_error(desired_samples, smooth_fk)
        desired_gripper = np.interp(smooth["t"], episode["times"], episode["recorded_gripper"])
        self.raw_gripper_errors = np.abs(raw_gripper["q"][:, 0] - desired_gripper)
        self.smooth_gripper_errors = np.abs(smooth_gripper["q"][:, 0] - desired_gripper)
        self.ik_result = ik_result
        self.episode_report = self._build_report(
            episode,
            ik_result,
            raw,
            smooth,
            raw_gripper,
            smooth_gripper,
            smoothing_ms,
        )

        self.frame = 0
        self.hold_frames = 0
        self.paused = False
        self.command_id += 1
        self._set_actors_at_frame(0)
        self._print_episode_report()
        jumps = self.episode_report["smoothing"]["knot_jumps"]
        print(
            "  smooth knot jumps: "
            f"q={jumps['q']:.3e} qd={jumps['qd']:.3e} "
            f"qdd={jumps['qdd']:.3e} qddd={jumps['qddd']:.3e}"
        )
        gripper_jumps = self.episode_report["gripper"]["knot_jumps"]
        print(
            "  gripper knot jumps: "
            f"q={gripper_jumps['q']:.3e} qd={gripper_jumps['qd']:.3e} "
            f"qdd={gripper_jumps['qdd']:.3e} qddd={gripper_jumps['qddd']:.3e}"
        )

        if self.args.report_json:
            output = Path(self.args.report_json).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as file:
                json.dump(self.episode_report, file, indent=2)
            print(f"  report_json={output}")

def main():
    args = parse_args()
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive")

    # Construct TRAC-IK before importing Isaac Gym to avoid Boost.Python type
    # registration conflicts in the shared py3.8 environment.
    tracik = Z1TracIKKinematics(
        args.urdf,
        ee_link=args.ee_link,
        timeout=float(args.tracik_timeout),
        epsilon=float(args.tracik_epsilon),
        solver_type=args.tracik_solver_type,
    )

    from isaacgym import gymapi, gymutil

    app = FireDoorTracIKPlay(args, tracik, gymapi, gymutil)
    try:
        app.run()
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
