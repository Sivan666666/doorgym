#!/usr/bin/env python3
"""Stream a recorded EE trajectory through TRAC-IK at 25 Hz in Isaac Gym.

This script does not precompute the IK trajectory. At each command tick it:
1. reads one EE pose and gripper value from the episode,
2. solves the six arm joints with TRAC-IK,
3. builds only the next short joint segment, and
4. sends segment samples to the Gym position controller.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
IK_SOLVERS_DIR = SCRIPT_DIR / "ik_solvers"
if str(IK_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(IK_SOLVERS_DIR))

from ik_solvers.z1_tracik_kinematics import DEFAULT_Z1_EE_LINK, DEFAULT_Z1_URDF, Z1TracIKKinematics
from joint_trajectory_smoothing import evaluate_polynomial, polynomial_coefficients
from play_z1_fire_door_tracik_trajectory import DEFAULT_EPISODE, joint_origin, metric_summary
from play_z1_gym_vs_pinocchio_ik import (
    A2W_DEFAULT_LEG_POS,
    build_a2w_base_visual_asset_root,
    make_transform,
    pose_error,
    pose_to_np,
    torch_orientation_error,
)
from play_z1_joint_trajectory_smoothing import (
    DEFAULT_A2W_ASSET_FILE,
    DEFAULT_A2W_ASSET_ROOT,
    DEFAULT_ASSET_FILE,
    DEFAULT_ASSET_ROOT,
)


ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=str, default=str(DEFAULT_EPISODE))
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_stop", type=int, default=-1, help="Exclusive; -1 streams to episode end.")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--command_hz", type=float, default=25.0)
    parser.add_argument("--control_hz", type=float, default=100.0)
    parser.add_argument("--interpolation", choices=("linear", "quintic", "septic"), default="quintic")
    parser.add_argument("--tracik_timeout", type=float, default=0.005)
    parser.add_argument("--tracik_epsilon", type=float, default=1.0e-5)
    parser.add_argument("--tracik_solver_type", choices=("Speed", "Distance", "Manip1", "Manip2"), default="Speed")
    parser.add_argument("--max_restarts", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stiffness", type=float, default=850.0)
    parser.add_argument("--damping", type=float, default=85.0)
    parser.add_argument("--gym_ik_damping", type=float, default=0.05)
    parser.add_argument("--gym_max_step", type=float, default=0.045)
    parser.add_argument("--ik_pos_gain", type=float, default=1.0)
    parser.add_argument("--ik_rot_gain", type=float, default=1.0)
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--print_interval", type=int, default=25, help="Report every N streamed commands; 0 disables.")
    parser.add_argument("--report_json", type=str, default="")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--no_loop", action="store_true")

    parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--asset_file", type=str, default=DEFAULT_ASSET_FILE)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--root_z", type=float, default=0.50)
    parser.add_argument("--disable_arm_visual_flip", action="store_true")
    parser.add_argument("--show_a2w_base", dest="show_a2w_base", action="store_true", default=True)
    parser.add_argument("--no_show_a2w_base", dest="show_a2w_base", action="store_false")
    parser.add_argument("--a2w_asset_root", type=str, default=str(DEFAULT_A2W_ASSET_ROOT))
    parser.add_argument("--a2w_asset_file", type=str, default=DEFAULT_A2W_ASSET_FILE)
    return parser.parse_args()


class EpisodeCommandStream:
    def __init__(self, args):
        self.path = Path(args.episode).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with np.load(self.path, allow_pickle=True) as data:
            required = ("replay_root_state", "replay_ee_pos", "replay_ee_quat", "replay_dof_pos")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"Episode is missing fields: {missing}")
            self.root_state = np.asarray(data["replay_root_state"], dtype=np.float64)
            self.ee_world_pos = np.asarray(data["replay_ee_pos"], dtype=np.float64)
            self.ee_world_quat = np.asarray(data["replay_ee_quat"], dtype=np.float64)
            self.recorded_dof = np.asarray(data["replay_dof_pos"], dtype=np.float64)
            self.record_fps = (
                float(np.asarray(data["record_effective_fps"]).item())
                if "record_effective_fps" in data.files
                else float(np.asarray(data["fps"]).item())
            )

        count = len(self.root_state)
        start = int(np.clip(args.frame_start, 0, count - 1))
        stop = count if int(args.frame_stop) < 0 else int(np.clip(args.frame_stop, start + 1, count))
        self.indices = np.arange(start, stop, max(1, int(args.frame_stride)), dtype=np.int64)
        if self.indices.size < 1:
            raise ValueError("Selected episode range is empty")

        standalone_mount = joint_origin(Path(args.urdf), "base_static_joint")
        a2w_urdf = Path(args.a2w_asset_root).expanduser().resolve() / args.a2w_asset_file
        a2w_mount = joint_origin(a2w_urdf, "base_static_joint")
        self.mount_correction = standalone_mount - a2w_mount

    def command(self, stream_index: int):
        frame = int(self.indices[stream_index])
        root_rotation = Rotation.from_quat(self.root_state[frame, 3:7])
        ee_rotation = Rotation.from_quat(self.ee_world_quat[frame])
        local_position = root_rotation.inv().apply(
            self.ee_world_pos[frame] - self.root_state[frame, :3]
        )
        local_quat = (root_rotation.inv() * ee_rotation).as_quat()
        tracik_pose = np.concatenate([local_position + self.mount_correction, local_quat])
        recorded_q = self.recorded_dof[frame, -7:-1].copy()
        gripper = float(self.recorded_dof[frame, -1])
        return frame, tracik_pose, recorded_q, gripper


class RealtimeTracIKPlay:
    def __init__(self, args, tracik, stream, gymapi, gymtorch, gymutil, torch):
        self.args = args
        self.tracik = tracik
        self.stream = stream
        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.gymutil = gymutil
        self.torch = torch
        self.gym = gymapi.acquire_gym()
        self.rng = np.random.default_rng(int(args.seed))
        self.temp_dirs = []
        self.root_poses = {
            "jacobian": np.asarray([-0.75, 0.0, float(args.root_z), 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "tracik": np.asarray([0.75, 0.0, float(args.root_z), 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        }
        self.command_period = 1.0 / float(args.command_hz)
        self.control_dt = 1.0 / float(args.control_hz)
        ratio = float(args.control_hz) / float(args.command_hz)
        self.steps_per_command = int(round(ratio))
        if self.steps_per_command < 1 or abs(ratio - self.steps_per_command) > 1.0e-9:
            raise ValueError("--control_hz must be an integer multiple of --command_hz")

        self.viewer = None
        self.paused = False
        self.finished = False
        self.stream_index = 0
        self.segment_step = 0
        self.segment_q_coeff = None
        self.segment_gripper_coeff = None
        self.segment_q_start = None
        self.segment_q_goal = None
        self.segment_gripper_start = 0.0
        self.segment_gripper_goal = 0.0
        self.current_target_pose = None
        self.current_episode_frame = -1
        self.last_solution = None
        self.last_command_q = None
        self.last_command_gripper = 0.0
        self.current_solution_pose = None
        self.current_command_pose = None
        self.visual_history = {}
        self._clear_metrics()

        self._create_sim()
        self._load_assets_and_actor()
        self._prepare_jacobian()
        self._create_viewer()
        self._build_geometries()
        self._reset_stream()

    def _clear_metrics(self):
        self.solve_ms = []
        self.ik_position_errors = []
        self.ik_rotation_errors = []
        self.gym_position_errors = []
        self.gym_rotation_errors = []
        self.gym_gripper_errors = []
        self.jacobian_position_errors = []
        self.jacobian_rotation_errors = []
        self.jacobian_gripper_errors = []
        self.jacobian_compute_ms = []
        self.retry_frames = []
        self.deadline_misses = 0
        self.jacobian_deadline_misses = 0

    def _clear_visualization(self):
        self.current_solution_pose = None
        self.current_command_pose = None
        self.visual_history = {
            "target_jacobian": [],
            "target_tracik": [],
            "solution_tracik": [],
            "command_tracik": [],
            "actual_jacobian": [],
            "actual_tracik": [],
        }

    def _append_visual_point(self, key, point):
        self.visual_history[key].append(np.asarray(point, dtype=np.float64).reshape(3).copy())

    def _local_pose_to_world(self, pose, key):
        pose = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        pose[:3] += self.root_poses[key][:3]
        return pose

    def _create_sim(self):
        params = self.gymapi.SimParams()
        params.up_axis = self.gymapi.UP_AXIS_Z
        params.gravity = self.gymapi.Vec3(0.0, 0.0, 0.0)
        params.dt = self.control_dt
        self.sim = self.gym.create_sim(0, 0, self.gymapi.SIM_PHYSX, params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")
        plane = self.gymapi.PlaneParams()
        plane.normal = self.gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane)

    def _load_assets_and_actor(self):
        options = self.gymapi.AssetOptions()
        options.fix_base_link = True
        options.collapse_fixed_joints = False
        options.disable_gravity = True
        options.default_dof_drive_mode = int(self.gymapi.DOF_MODE_POS)
        options.use_mesh_materials = True
        options.flip_visual_attachments = not bool(self.args.disable_arm_visual_flip)
        options.thickness = 0.001
        options.armature = 0.01
        asset_root = str(Path(self.args.asset_root).expanduser().resolve())
        print(
            f"Loading Z1 asset: root={asset_root}, file={self.args.asset_file}, "
            f"flip_visual_attachments={options.flip_visual_attachments}"
        )
        self.asset = self.gym.load_asset(self.sim, asset_root, self.args.asset_file, options)
        if self.asset is None:
            raise RuntimeError(f"Failed to load Z1 asset from {asset_root}/{self.args.asset_file}")

        self.base_asset = self._load_a2w_base_asset() if self.args.show_a2w_base else None
        self.env = self.gym.create_env(
            self.sim,
            self.gymapi.Vec3(-1.5, -1.5, -0.2),
            self.gymapi.Vec3(1.5, 1.5, 1.7),
            1,
        )
        self.actors = {}
        self.base_actors = {}
        actor_names = {
            "jacobian": "z1_gym_jacobian_realtime",
            "tracik": "z1_tracik_realtime",
        }
        actor_colors = {
            "jacobian": self.gymapi.Vec3(0.75, 0.35, 1.0),
            "tracik": self.gymapi.Vec3(0.12, 0.52, 1.0),
        }
        for key in ("jacobian", "tracik"):
            root = self.root_poses[key]
            root_tf = make_transform(self.gymapi, root[:3], root[3:7])
            self.root_poses[key] = pose_to_np(root_tf)
            if self.base_asset is not None:
                base_actor = self.gym.create_actor(self.env, self.base_asset, root_tf, f"a2w_{key}_realtime", 0, 0)
                self.base_actors[key] = base_actor
                self._configure_a2w_base(base_actor)
                self._paint(base_actor, self.gymapi.Vec3(0.34, 0.36, 0.39))
            actor = self.gym.create_actor(self.env, self.asset, root_tf, actor_names[key], 0, 0)
            self.actors[key] = actor
            self._paint(actor, actor_colors[key])

        self.dof_names = list(self.gym.get_asset_dof_names(self.asset))
        self.num_dofs = len(self.dof_names)
        props = self.gym.get_asset_dof_properties(self.asset)
        props["driveMode"].fill(int(self.gymapi.DOF_MODE_POS))
        props["stiffness"].fill(float(self.args.stiffness))
        props["damping"].fill(float(self.args.damping))
        for actor in self.actors.values():
            self.gym.set_actor_dof_properties(self.env, actor, props)
        self.lower = np.asarray(props["lower"], dtype=np.float64)
        self.upper = np.asarray(props["upper"], dtype=np.float64)
        self.control_indices = np.asarray([self.dof_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64)
        self.gripper_index = self.dof_names.index("jointGripper")
        self.body_names = list(self.gym.get_asset_rigid_body_names(self.asset))
        self.ee_body_handles = {
            key: self.gym.find_actor_rigid_body_handle(self.env, actor, self.args.ee_link)
            for key, actor in self.actors.items()
        }
        if any(handle < 0 for handle in self.ee_body_handles.values()):
            raise RuntimeError(f"EE link {self.args.ee_link!r} not found")

    def _prepare_jacobian(self):
        self.gym.prepare_sim(self.sim)
        tensor = self.gym.acquire_jacobian_tensor(self.sim, "z1_gym_jacobian_realtime")
        self.gym_jacobian = self.gymtorch.wrap_tensor(tensor)
        ee_asset_index = int(self.body_names.index(self.args.ee_link))
        jacobian = self.gym_jacobian[0] if self.gym_jacobian.ndim >= 4 else self.gym_jacobian
        body_dim = int(jacobian.shape[0])
        column_dim = int(jacobian.shape[-1])
        if body_dim == len(self.body_names):
            self.ee_jacobian_index = ee_asset_index
        elif body_dim == len(self.body_names) - 1:
            self.ee_jacobian_index = ee_asset_index - 1
        else:
            raise RuntimeError(
                f"Cannot map Jacobian body dim={body_dim} to asset bodies={len(self.body_names)}"
            )
        if column_dim == self.num_dofs:
            column_offset = 0
        elif column_dim == self.num_dofs + 6:
            column_offset = 6
        else:
            column_offset = column_dim - self.num_dofs
            if column_offset < 0 or column_offset > 6:
                raise RuntimeError(f"Cannot map Jacobian columns={column_dim}, DOFs={self.num_dofs}")
        self.control_jacobian_indices = self.control_indices + column_offset

    def _load_a2w_base_asset(self):
        temp_dir = tempfile.TemporaryDirectory(prefix="z1_realtime_a2w_")
        self.temp_dirs.append(temp_dir)
        root, asset_file = build_a2w_base_visual_asset_root(
            self.args.a2w_asset_root,
            self.args.a2w_asset_file,
            temp_dir.name,
        )
        options = self.gymapi.AssetOptions()
        options.fix_base_link = True
        options.collapse_fixed_joints = False
        options.disable_gravity = True
        options.default_dof_drive_mode = int(self.gymapi.DOF_MODE_POS)
        options.use_mesh_materials = True
        options.flip_visual_attachments = False
        asset = self.gym.load_asset(self.sim, str(root), asset_file, options)
        if asset is None:
            raise RuntimeError(f"Failed to load A2W base asset from {root}/{asset_file}")
        return asset

    def _configure_a2w_base(self, actor):
        count = self.gym.get_asset_dof_count(self.base_asset)
        if count <= 0:
            return
        names = list(self.gym.get_asset_dof_names(self.base_asset))
        props = self.gym.get_asset_dof_properties(self.base_asset)
        props["driveMode"].fill(int(self.gymapi.DOF_MODE_POS))
        props["stiffness"].fill(float(self.args.stiffness))
        props["damping"].fill(float(self.args.damping))
        states = np.zeros(count, dtype=self.gymapi.DofState.dtype)
        lower = np.asarray(props["lower"], dtype=np.float64)
        upper = np.asarray(props["upper"], dtype=np.float64)
        limited = np.asarray(props["hasLimits"], dtype=bool)
        for index, name in enumerate(names):
            value = float(A2W_DEFAULT_LEG_POS.get(name, 0.0))
            if limited[index] and lower[index] < upper[index]:
                value = float(np.clip(value, lower[index], upper[index]))
            states["pos"][index] = value
        self.gym.set_actor_dof_properties(self.env, actor, props)
        self.gym.set_actor_dof_states(self.env, actor, states, self.gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, actor, states["pos"])

    def _paint(self, actor, color):
        for body in range(self.gym.get_actor_rigid_body_count(self.env, actor)):
            self.gym.set_rigid_body_color(self.env, actor, body, self.gymapi.MESH_VISUAL, color)

    def _create_viewer(self):
        if self.args.headless:
            return
        camera = self.gymapi.CameraProperties()
        camera.width = 1280
        camera.height = 800
        self.viewer = self.gym.create_viewer(self.sim, camera)
        if self.viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_ESCAPE, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_SPACE, "pause")
        self.gym.viewer_camera_look_at(
            self.viewer,
            None,
            self.gymapi.Vec3(0.0, -2.4, float(self.args.root_z) + 1.0),
            self.gymapi.Vec3(0.0, 0.0, float(self.args.root_z) + 0.15),
        )

    def _build_geometries(self):
        self.target_geometry = self.gymutil.WireframeSphereGeometry(0.035, 10, 10, color=(1.0, 0.78, 0.05))
        self.current_geometries = {
            "jacobian": self.gymutil.WireframeSphereGeometry(0.026, 8, 8, color=(0.75, 0.35, 1.0)),
            "tracik": self.gymutil.WireframeSphereGeometry(0.026, 8, 8, color=(0.12, 0.52, 1.0)),
        }
        self.solution_geometry = self.gymutil.WireframeSphereGeometry(
            0.029, 9, 9, color=(0.15, 1.0, 0.25)
        )
        self.command_geometry = self.gymutil.WireframeSphereGeometry(
            0.022, 8, 8, color=(0.05, 0.85, 1.0)
        )
        self.axes_geometry = self.gymutil.AxesGeometry(scale=0.12)

    def _set_actor_state(self, actor, q6, gripper):
        states = self.gym.get_actor_dof_states(self.env, actor, self.gymapi.STATE_ALL)
        states["pos"][:] = 0.0
        states["vel"][:] = 0.0
        states["pos"][self.control_indices] = np.asarray(q6, dtype=np.float32)
        states["pos"][self.gripper_index] = float(
            np.clip(gripper, self.lower[self.gripper_index], self.upper[self.gripper_index])
        )
        self.gym.set_actor_dof_states(self.env, actor, states, self.gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, actor, states["pos"])

    def _actor_q_gripper(self, key):
        states = self.gym.get_actor_dof_states(self.env, self.actors[key], self.gymapi.STATE_ALL)
        return (
            np.asarray(states["pos"][self.control_indices], dtype=np.float64),
            float(states["pos"][self.gripper_index]),
        )

    def _reset_stream(self):
        _, _, recorded_q, gripper = self.stream.command(0)
        for actor in self.actors.values():
            self._set_actor_state(actor, recorded_q, gripper)
        self.stream_index = 0
        self.segment_step = 0
        self.segment_q_coeff = None
        self.segment_gripper_coeff = None
        self.last_solution = recorded_q.copy()
        self.last_command_q = recorded_q.copy()
        self.last_command_gripper = float(gripper)
        self.current_target_pose = None
        self.current_episode_frame = -1
        self.finished = False
        self.paused = False
        self._clear_metrics()
        self._clear_visualization()
        for _ in range(max(0, int(self.args.warmup))):
            _, target, _, _ = self.stream.command(0)
            self.tracik.ik(target, seed_joints=recorded_q, max_restarts=1, strict_seed=True)
        _, warmup_target, _, _ = self.stream.command(0)
        self.current_target_pose = warmup_target
        for _ in range(min(5, max(1, int(self.args.warmup)))):
            self.gym.refresh_jacobian_tensors(self.sim)
            self._update_gym_jacobian(gripper)
        self.jacobian_compute_ms.clear()
        self.jacobian_deadline_misses = 0
        self.current_target_pose = None
        print(
            f"Realtime stream reset: commands={len(self.stream.indices)} command_hz={self.args.command_hz:.1f} "
            f"control_hz={self.args.control_hz:.1f} steps_per_command={self.steps_per_command}"
        )

    def _plan_segment(self, q_start, q_goal, gripper_start, gripper_goal):
        zeros_q = np.zeros(6, dtype=np.float64)
        zeros_g = np.zeros(1, dtype=np.float64)
        if self.args.interpolation == "linear":
            self.segment_q_coeff = None
            self.segment_gripper_coeff = None
        else:
            method = "quintic_hermite" if self.args.interpolation == "quintic" else "septic"
            self.segment_q_coeff = polynomial_coefficients(
                q_start,
                q_goal,
                zeros_q,
                zeros_q,
                zeros_q,
                zeros_q,
                zeros_q,
                zeros_q,
                self.command_period,
                method,
            )
            self.segment_gripper_coeff = polynomial_coefficients(
                np.asarray([gripper_start]),
                np.asarray([gripper_goal]),
                zeros_g,
                zeros_g,
                zeros_g,
                zeros_g,
                zeros_g,
                zeros_g,
                self.command_period,
                method,
            )
        self.segment_q_start = np.asarray(q_start, dtype=np.float64).copy()
        self.segment_q_goal = np.asarray(q_goal, dtype=np.float64).copy()
        self.segment_gripper_start = float(gripper_start)
        self.segment_gripper_goal = float(gripper_goal)
        self.segment_step = 0

    def _receive_next_command(self):
        frame, target_pose, _, gripper_goal = self.stream.command(self.stream_index)
        seed = self.last_solution.copy()
        start_ns = time.perf_counter_ns()
        solution = self.tracik.ik(target_pose, seed_joints=seed, max_restarts=1, strict_seed=True)
        if solution is None and int(self.args.max_restarts) > 1:
            self.retry_frames.append(frame)
            solution = self.tracik.ik(
                target_pose,
                seed_joints=seed,
                max_restarts=int(self.args.max_restarts),
                strict_seed=False,
                rng=self.rng,
            )
        elapsed_ms = (time.perf_counter_ns() - start_ns) * 1.0e-6
        self.solve_ms.append(elapsed_ms)
        if elapsed_ms > self.command_period * 1000.0:
            self.deadline_misses += 1
        if solution is None:
            raise RuntimeError(f"TRAC-IK failed for streamed episode frame {frame}")

        solution = np.asarray(solution, dtype=np.float64).reshape(6)
        fk_pose = self.tracik.fk(solution)
        ik_position_error, ik_rotation_error = pose_error(target_pose, fk_pose)
        self.ik_position_errors.append(ik_position_error)
        self.ik_rotation_errors.append(ik_rotation_error)

        self._plan_segment(
            self.last_command_q,
            solution,
            self.last_command_gripper,
            gripper_goal,
        )
        self.last_solution = solution.copy()
        self.current_target_pose = target_pose.copy()
        self.current_solution_pose = self._local_pose_to_world(fk_pose, "tracik")
        for key in ("jacobian", "tracik"):
            self._append_visual_point(f"target_{key}", self._target_world_pose(key)[:3])
        self._append_visual_point("solution_tracik", self.current_solution_pose[:3])
        self.current_episode_frame = frame
        self.stream_index += 1

    def _segment_sample(self):
        tau = min((self.segment_step + 1) * self.control_dt, self.command_period)
        alpha = min(1.0, tau / self.command_period)
        if self.segment_q_coeff is None:
            q = self.segment_q_start + alpha * (self.segment_q_goal - self.segment_q_start)
            gripper = self.segment_gripper_start + alpha * (
                self.segment_gripper_goal - self.segment_gripper_start
            )
        else:
            q = evaluate_polynomial(self.segment_q_coeff, np.asarray([tau]))[0][0]
            gripper = float(evaluate_polynomial(self.segment_gripper_coeff, np.asarray([tau]))[0][0, 0])
        return q, gripper

    def _set_position_target(self, key, q6, gripper):
        target = np.zeros(self.num_dofs, dtype=np.float32)
        target[self.control_indices] = np.asarray(q6, dtype=np.float32)
        target[self.gripper_index] = float(
            np.clip(gripper, self.lower[self.gripper_index], self.upper[self.gripper_index])
        )
        self.gym.set_actor_dof_position_targets(self.env, self.actors[key], target)

    def _ee_world_pose(self, key):
        transform = self.gym.get_rigid_transform(self.env, self.ee_body_handles[key])
        return pose_to_np(transform)

    def _target_world_pose(self, key):
        if self.current_target_pose is None:
            return None
        pose = self.current_target_pose.copy()
        pose[:3] += self.root_poses[key][:3]
        return pose

    def _update_gym_jacobian(self, gripper):
        start_ns = time.perf_counter_ns()
        torch = self.torch
        target_pose = self._target_world_pose("jacobian")
        current_pose = self._ee_world_pose("jacobian")
        device = self.gym_jacobian.device
        target_position = torch.tensor(target_pose[:3], dtype=torch.float32, device=device)
        target_quat = torch.tensor(target_pose[3:7], dtype=torch.float32, device=device)
        current_position = torch.tensor(current_pose[:3], dtype=torch.float32, device=device)
        current_quat = torch.tensor(current_pose[3:7], dtype=torch.float32, device=device)
        position_error = target_position - current_position
        rotation_error = torch_orientation_error(torch, target_quat, current_quat)

        jacobian = self.gym_jacobian[0] if self.gym_jacobian.ndim >= 4 else self.gym_jacobian
        ee_jacobian = jacobian[self.ee_jacobian_index, :, :]
        columns = torch.tensor(self.control_jacobian_indices, dtype=torch.long, device=device)
        control_jacobian = ee_jacobian[:, columns]
        task_error = torch.cat(
            (
                float(self.args.ik_pos_gain) * position_error,
                float(self.args.ik_rot_gain) * rotation_error,
            )
        )
        weights = torch.tensor(
            [1.0, 1.0, 1.0, self.args.rot_weight, self.args.rot_weight, self.args.rot_weight],
            dtype=torch.float32,
            device=device,
        )
        weighted_jacobian = control_jacobian * weights.view(6, 1)
        weighted_error = task_error * weights
        transpose = torch.transpose(weighted_jacobian, 0, 1)
        damping = max(1.0e-6, float(self.args.gym_ik_damping))
        lhs = weighted_jacobian @ transpose + torch.eye(6, dtype=torch.float32, device=device) * damping**2
        delta = transpose @ torch.linalg.solve(lhs, weighted_error.unsqueeze(-1)).squeeze(-1)
        delta = torch.clamp(delta, -float(self.args.gym_max_step), float(self.args.gym_max_step))

        current_q, _ = self._actor_q_gripper("jacobian")
        next_q = np.clip(
            current_q + delta.detach().cpu().numpy().astype(np.float64),
            self.lower[self.control_indices],
            self.upper[self.control_indices],
        )
        self._set_position_target("jacobian", next_q, gripper)
        elapsed_ms = (time.perf_counter_ns() - start_ns) * 1.0e-6
        self.jacobian_compute_ms.append(elapsed_ms)
        if elapsed_ms > self.control_dt * 1000.0:
            self.jacobian_deadline_misses += 1

    def _record_deadline_tracking(self):
        if self.current_target_pose is None:
            return
        deadline_errors = {}
        for key in ("jacobian", "tracik"):
            target = self._target_world_pose(key)
            current = self._ee_world_pose(key)
            position_error, rotation_error = pose_error(target, current)
            _, gripper = self._actor_q_gripper(key)
            gripper_error = abs(gripper - self.segment_gripper_goal)
            deadline_errors[key] = (position_error, rotation_error, gripper_error)
        self.jacobian_position_errors.append(deadline_errors["jacobian"][0])
        self.jacobian_rotation_errors.append(deadline_errors["jacobian"][1])
        self.jacobian_gripper_errors.append(deadline_errors["jacobian"][2])
        self.gym_position_errors.append(deadline_errors["tracik"][0])
        self.gym_rotation_errors.append(deadline_errors["tracik"][1])
        self.gym_gripper_errors.append(deadline_errors["tracik"][2])

        count = len(self.solve_ms)
        if self.args.print_interval > 0 and count % int(self.args.print_interval) == 0:
            print(
                f"[command {count:04d}] episode_frame={self.current_episode_frame} "
                f"ik={self.solve_ms[-1]:.3f} ms "
                f"jac_pos={deadline_errors['jacobian'][0]:.5f} m "
                f"trac_pos={deadline_errors['tracik'][0]:.5f} m"
            )

    def _advance_control(self):
        if self.finished:
            return
        if self.segment_q_coeff is None and self.segment_q_start is None:
            self._receive_next_command()
        q, gripper = self._segment_sample()
        self.current_command_pose = self._local_pose_to_world(self.tracik.fk(q), "tracik")
        self._append_visual_point("command_tracik", self.current_command_pose[:3])
        self._set_position_target("tracik", q, gripper)
        self._update_gym_jacobian(gripper)
        self.segment_step += 1

    def _after_simulation(self):
        if self.finished or self.segment_q_start is None:
            return
        self._append_visual_point("actual_jacobian", self._ee_world_pose("jacobian")[:3])
        self._append_visual_point("actual_tracik", self._ee_world_pose("tracik")[:3])
        if self.segment_step < self.steps_per_command:
            return
        self._record_deadline_tracking()
        self.last_command_q = self.segment_q_goal.copy()
        self.last_command_gripper = float(self.segment_gripper_goal)
        self.segment_q_start = None
        self.segment_q_goal = None
        self.segment_q_coeff = None
        self.segment_gripper_coeff = None
        self.segment_step = 0
        if self.stream_index >= len(self.stream.indices):
            self.finished = True
            self._print_report()

    def _draw(self):
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        if self.current_target_pose is None:
            return
        colors = {
            "jacobian": self.gymapi.Vec3(0.75, 0.35, 1.0),
            "tracik": self.gymapi.Vec3(0.12, 0.52, 1.0),
        }
        trajectory_colors = {
            "target_jacobian": (1.0, 0.78, 0.05),
            "target_tracik": (1.0, 0.78, 0.05),
            "solution_tracik": (0.15, 1.0, 0.25),
            "command_tracik": (0.05, 0.85, 1.0),
            "actual_jacobian": (0.75, 0.35, 1.0),
            "actual_tracik": (1.0, 0.15, 0.12),
        }
        for history_key, color in trajectory_colors.items():
            self._draw_polyline(self.visual_history[history_key], color)
        self._draw_points(self.visual_history["solution_tracik"], (0.15, 1.0, 0.25))
        for key in ("jacobian", "tracik"):
            target = self._target_world_pose(key)
            current = self._ee_world_pose(key)
            target_tf = make_transform(self.gymapi, target[:3], target[3:7])
            current_tf = make_transform(self.gymapi, current[:3], current[3:7])
            self.gymutil.draw_lines(self.target_geometry, self.gym, self.viewer, self.env, target_tf)
            self.gymutil.draw_lines(self.current_geometries[key], self.gym, self.viewer, self.env, current_tf)
            self.gymutil.draw_lines(self.axes_geometry, self.gym, self.viewer, self.env, target_tf)
            self.gymutil.draw_lines(self.axes_geometry, self.gym, self.viewer, self.env, current_tf)
            self.gymutil.draw_line(
                self.gymapi.Vec3(*[float(value) for value in current[:3]]),
                self.gymapi.Vec3(*[float(value) for value in target[:3]]),
                colors[key],
                self.gym,
                self.viewer,
                self.env,
            )
        if self.current_solution_pose is not None:
            solution_tf = make_transform(
                self.gymapi,
                self.current_solution_pose[:3],
                self.current_solution_pose[3:7],
            )
            self.gymutil.draw_lines(self.solution_geometry, self.gym, self.viewer, self.env, solution_tf)
        if self.current_command_pose is not None:
            command_tf = make_transform(
                self.gymapi,
                self.current_command_pose[:3],
                self.current_command_pose[3:7],
            )
            self.gymutil.draw_lines(self.command_geometry, self.gym, self.viewer, self.env, command_tf)

    def _draw_polyline(self, points, color):
        if len(points) < 2:
            return
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        vertices = np.stack((points[:-1], points[1:]), axis=1)
        colors = np.repeat(np.asarray(color, dtype=np.float32)[None, :], len(vertices), axis=0)
        self.gym.add_lines(self.viewer, self.env, len(vertices), vertices, colors)

    def _draw_points(self, points, color, radius=0.008):
        if not points:
            return
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        offsets = np.eye(3, dtype=np.float32) * float(radius)
        vertices = np.stack(
            [np.stack((points - offset, points + offset), axis=1) for offset in offsets],
            axis=1,
        ).reshape(-1, 2, 3)
        colors = np.repeat(np.asarray(color, dtype=np.float32)[None, :], len(vertices), axis=0)
        self.gym.add_lines(self.viewer, self.env, len(vertices), vertices, colors)

    def _handle_events(self):
        if self.viewer is None:
            return False
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.value <= 0:
                continue
            if event.action == "quit":
                return True
            if event.action == "reset":
                self._reset_stream()
            elif event.action == "pause":
                self.paused = not self.paused
                print("Realtime stream paused." if self.paused else "Realtime stream resumed.")
        return False

    def _report(self):
        return {
            "episode": str(self.stream.path),
            "commands": int(len(self.solve_ms)),
            "command_hz": float(self.args.command_hz),
            "control_hz": float(self.args.control_hz),
            "steps_per_command": int(self.steps_per_command),
            "interpolation": self.args.interpolation,
            "ik_success": int(len(self.solve_ms)),
            "ik_retry_count": int(len(self.retry_frames)),
            "ik_retry_frames": self.retry_frames,
            "ik_deadline_ms": float(self.command_period * 1000.0),
            "ik_deadline_misses": int(self.deadline_misses),
            "ik_time_ms": metric_summary(self.solve_ms),
            "ik_position_error_m": metric_summary(self.ik_position_errors),
            "ik_rotation_error_rad": metric_summary(self.ik_rotation_errors),
            "gym_jacobian_compute_ms": metric_summary(self.jacobian_compute_ms),
            "gym_jacobian_deadline_ms": float(self.control_dt * 1000.0),
            "gym_jacobian_deadline_misses": int(self.jacobian_deadline_misses),
            "gym_jacobian_deadline_position_error_m": metric_summary(self.jacobian_position_errors),
            "gym_jacobian_deadline_rotation_error_rad": metric_summary(self.jacobian_rotation_errors),
            "gym_jacobian_deadline_gripper_error_rad": metric_summary(self.jacobian_gripper_errors),
            "tracik_deadline_position_error_m": metric_summary(self.gym_position_errors),
            "tracik_deadline_rotation_error_rad": metric_summary(self.gym_rotation_errors),
            "tracik_deadline_gripper_error_rad": metric_summary(self.gym_gripper_errors),
        }

    def _print_metric(self, name, values, unit):
        stats = metric_summary(values)
        print(
            f"  {name:<30} mean={stats['mean']:.6g} {unit} p50={stats['p50']:.6g} "
            f"p95={stats['p95']:.6g} p99={stats['p99']:.6g} max={stats['max']:.6g}"
        )

    def _print_report(self):
        print("\nRealtime 25 Hz Gym Jacobian vs TRAC-IK execution report")
        print(
            f"  commands={len(self.solve_ms)} interpolation={self.args.interpolation} "
            f"command_hz={self.args.command_hz:.1f} control_hz={self.args.control_hz:.1f}"
        )
        print(
            f"  IK success={len(self.solve_ms)}/{len(self.stream.indices)} retries={len(self.retry_frames)} "
            f"deadline={self.command_period * 1000.0:.1f} ms misses={self.deadline_misses}"
        )
        self._print_metric("TRAC-IK time", self.solve_ms, "ms")
        self._print_metric("IK FK position error", self.ik_position_errors, "m")
        self._print_metric("IK FK rotation error", self.ik_rotation_errors, "rad")
        self._print_metric("Gym Jacobian update", self.jacobian_compute_ms, "ms")
        print(
            f"  Gym Jacobian control deadline={self.control_dt * 1000.0:.1f} ms "
            f"misses={self.jacobian_deadline_misses}"
        )
        self._print_metric("Jacobian deadline position", self.jacobian_position_errors, "m")
        self._print_metric("Jacobian deadline rotation", self.jacobian_rotation_errors, "rad")
        self._print_metric("Jacobian deadline gripper", self.jacobian_gripper_errors, "rad")
        self._print_metric("TRAC-IK deadline position", self.gym_position_errors, "m")
        self._print_metric("TRAC-IK deadline rotation", self.gym_rotation_errors, "rad")
        self._print_metric("TRAC-IK deadline gripper", self.gym_gripper_errors, "rad")
        if self.args.interpolation == "quintic":
            print("  continuity=local quintic is C2; online jerk continuity is not guaranteed without future targets.")
        elif self.args.interpolation == "septic":
            print("  continuity=local septic uses zero endpoint velocity/acceleration/jerk and is C3.")

        if self.args.report_json:
            output = Path(self.args.report_json).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as file:
                json.dump(self._report(), file, indent=2)
            print(f"  report_json={output}")

    def run(self):
        print("Viewer controls: R=restart stream, Space=pause/resume, Esc=quit.")
        print("IK is solved only when each 25 Hz episode command arrives; no future IK trajectory is precomputed.")
        print("Left purple: Gym Jacobian DLS. Right blue: realtime TRAC-IK plus local trajectory.")
        print(
            "Paths: yellow=target, green crosses=TRAC-IK FK solutions, cyan=smoothed-command FK, "
            "purple/red=actual Jacobian/TRAC-IK EE."
        )
        step = 0
        while True:
            if self.viewer is not None and self.gym.query_viewer_has_closed(self.viewer):
                break
            if self.args.max_steps > 0 and step >= int(self.args.max_steps):
                break
            if self._handle_events():
                break
            if self.finished:
                if self.args.headless or self.args.no_loop:
                    break
                self._reset_stream()

            if not self.paused:
                self.gym.refresh_jacobian_tensors(self.sim)
                self._advance_control()
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self._after_simulation()

            if self.viewer is not None:
                self._draw()
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            step += 1

        if not self.finished and self.solve_ms:
            self._print_report()

    def destroy(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()


def main():
    args = parse_args()
    if args.command_hz <= 0.0 or args.control_hz <= 0.0:
        raise ValueError("--command_hz and --control_hz must be positive")
    stream = EpisodeCommandStream(args)
    if abs(stream.record_fps / max(1, int(args.frame_stride)) - float(args.command_hz)) > 1.0e-6:
        print(
            f"Warning: selected episode stream rate is {stream.record_fps / max(1, int(args.frame_stride)):.3f} Hz, "
            f"but --command_hz={args.command_hz:.3f}."
        )

    # Construct TRAC-IK before importing Isaac Gym to avoid Boost.Python type
    # registration conflicts in the shared py3.8 environment.
    tracik = Z1TracIKKinematics(
        args.urdf,
        ee_link=args.ee_link,
        timeout=float(args.tracik_timeout),
        epsilon=float(args.tracik_epsilon),
        solver_type=args.tracik_solver_type,
    )

    from isaacgym import gymapi, gymtorch, gymutil
    import torch

    app = RealtimeTracIKPlay(args, tracik, stream, gymapi, gymtorch, gymutil, torch)
    try:
        app.run()
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
