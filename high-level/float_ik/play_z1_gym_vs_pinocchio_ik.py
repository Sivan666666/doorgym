#!/usr/bin/env python3
"""Play two Z1 arms with the same 6D IK command.

Left actor: Isaac Gym Jacobian IK.
Right actor: standalone Pinocchio IK.

Press S in the viewer to sample a new reachable target pose.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from z1_pinocchio_ik import (
    DEFAULT_Z1_EE_LINK,
    DEFAULT_Z1_JOINT_NAMES,
    DEFAULT_Z1_URDF,
    Z1PinocchioIK,
    normalize_quat_xyzw,
    orientation_error_xyzw,
    quat_multiply_xyzw,
    quat_to_rot_xyzw,
)


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
DEFAULT_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "z1"
DEFAULT_ASSET_FILE = "urdf/z1_arm.urdf"
DEFAULT_EPISODE = (
    HIGH_LEVEL_ROOT
    / "data"
    / "door_dp_raw"
    / "wc4_pi05_state10_noisy_rand_100"
    / "episode_000000.npz"
)
DEFAULT_A2W_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "a2wz1"
DEFAULT_A2W_ASSET_FILE = "urdf/a2wz1.urdf"

LEFT_ACTOR_NAME = "z1_gym_jacobian"
RIGHT_ACTOR_NAME = "z1_pinocchio"
A2W_BASE_LINKS = {
    "world",
    "base_link",
    "front_bumper",
    "rear_bumper",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_wheel",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_wheel",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_wheel",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_wheel",
}
A2W_DEFAULT_LEG_POS = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--asset_file", type=str, default=DEFAULT_ASSET_FILE)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until viewer close.")
    parser.add_argument("--target_delta", type=float, default=0.45)
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 1.05, -1.45, 0.75, 0.0, 0.0))
    parser.add_argument(
        "--disable_arm_visual_flip",
        action="store_true",
        help="Match the old debug option: disable Isaac Gym visual attachment flipping for Z1 meshes.",
    )
    parser.add_argument("--stiffness", type=float, default=850.0)
    parser.add_argument("--damping", type=float, default=85.0)
    parser.add_argument("--gym_ik_damping", type=float, default=0.05)
    parser.add_argument("--gym_max_step", type=float, default=0.045)
    parser.add_argument("--ik_pos_gain", type=float, default=1.0)
    parser.add_argument("--ik_rot_gain", type=float, default=1.0)
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--pin_max_iter", type=int, default=120)
    parser.add_argument("--pin_restarts", type=int, default=16)
    parser.add_argument("--pin_dt", type=float, default=0.4)
    parser.add_argument("--pin_damping", type=float, default=1.0e-4)
    parser.add_argument("--pin_threshold", type=float, default=1.0e-5)
    parser.add_argument("--pos_tol", type=float, default=0.012)
    parser.add_argument("--rot_tol", type=float, default=0.055)
    parser.add_argument("--print_interval", type=int, default=60)
    parser.add_argument("--target_sphere_radius", type=float, default=0.035)
    parser.add_argument("--axis_scale", type=float, default=0.13)
    parser.add_argument("--root_z", type=float, default=0.50)
    parser.add_argument("--show_a2w_base", dest="show_a2w_base", action="store_true", default=True)
    parser.add_argument("--no_show_a2w_base", dest="show_a2w_base", action="store_false")
    parser.add_argument("--a2w_asset_root", type=str, default=str(DEFAULT_A2W_ASSET_ROOT))
    parser.add_argument("--a2w_asset_file", type=str, default=DEFAULT_A2W_ASSET_FILE)
    parser.add_argument(
        "--episode",
        type=str,
        default="",
        help=f"Optional raw door episode npz to replay. Default candidate: {DEFAULT_EPISODE}",
    )
    parser.add_argument("--episode_key", type=str, default="action", choices=("action", "state"))
    parser.add_argument(
        "--episode_frames",
        type=str,
        default="0,20,40,60,80",
        help="Comma list, inclusive range like 0:80:20, or all.",
    )
    parser.add_argument("--episode_stride", type=int, default=1, help="Stride used when --episode_frames=all.")
    parser.add_argument("--episode_root_mode", type=str, default="relative", choices=("relative", "static"))
    parser.add_argument("--episode_autoplay", action="store_true")
    parser.add_argument("--episode_hold_steps", type=int, default=1)
    parser.add_argument("--episode_loop", action="store_true")
    return parser.parse_args()


def torch_quat_conjugate(torch, q):
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def torch_quat_multiply(torch, lhs, rhs):
    x1, y1, z1, w1 = lhs.unbind(-1)
    x2, y2, z2, w2 = rhs.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def torch_normalize_quat(torch, q):
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-8)


def torch_orientation_error(torch, desired, current):
    desired = torch_normalize_quat(torch, desired)
    current = torch_normalize_quat(torch, current)
    delta = torch_quat_multiply(torch, desired, torch_quat_conjugate(torch, current))
    return delta[..., :3] * torch.sign(delta[..., 3:4])


def pose_error(target_pose, current_pose):
    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
    current_pose = np.asarray(current_pose, dtype=np.float64).reshape(7)
    pos_err = float(np.linalg.norm(target_pose[:3] - current_pose[:3]))
    rot_err = float(np.linalg.norm(orientation_error_xyzw(target_pose[3:7], current_pose[3:7])))
    return pos_err, rot_err


def quat_conjugate_xyzw(q):
    q = normalize_quat_xyzw(np.asarray(q, dtype=np.float64).reshape(4))
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def compose_pose(parent_pose, child_pose):
    parent_pose = np.asarray(parent_pose, dtype=np.float64).reshape(7)
    child_pose = np.asarray(child_pose, dtype=np.float64).reshape(7)
    pos = parent_pose[:3] + quat_to_rot_xyzw(parent_pose[3:7]) @ child_pose[:3]
    quat = quat_multiply_xyzw(parent_pose[3:7], child_pose[3:7])
    return np.concatenate([pos, normalize_quat_xyzw(quat)])


def inverse_pose(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    inv_quat = quat_conjugate_xyzw(pose[3:7])
    inv_pos = -(quat_to_rot_xyzw(inv_quat) @ pose[:3])
    return np.concatenate([inv_pos, inv_quat])


def relative_pose(reference_pose, pose):
    return compose_pose(inverse_pose(reference_pose), pose)


def transform_pose(root_pose, local_pose):
    return compose_pose(root_pose, local_pose)


def make_transform(gymapi, position, quat=None):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(position[0]), float(position[1]), float(position[2]))
    if quat is None:
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    else:
        quat = normalize_quat_xyzw(quat)
        pose.r = gymapi.Quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return pose


def pose_to_np(pose):
    return np.asarray([pose.p.x, pose.p.y, pose.p.z, pose.r.x, pose.r.y, pose.r.z, pose.r.w], dtype=np.float64)


def parse_frame_list(text):
    frames = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            frames.append(int(item))
    return frames


def parse_episode_frames(text, episode_len, stride=1):
    text = str(text).strip()
    stride = max(1, int(stride))
    if text.lower() in ("all", "*"):
        return list(range(0, int(episode_len), stride))

    frames = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            frames.append(int(item))
            continue

        pieces = item.split(":")
        if len(pieces) > 3:
            raise ValueError(f"Invalid episode frame range: {item!r}")
        start = int(pieces[0]) if pieces[0] else 0
        end = int(pieces[1]) if len(pieces) > 1 and pieces[1] else int(episode_len) - 1
        step = int(pieces[2]) if len(pieces) > 2 and pieces[2] else stride
        if step == 0:
            raise ValueError(f"Episode frame range step cannot be zero: {item!r}")
        stop = end + (1 if step > 0 else -1)
        frames.extend(range(start, stop, step))
    return frames


def load_episode_targets(path, key, frame_spec, stride=1):
    path = Path(path).expanduser().resolve()
    data = np.load(path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Episode key {key!r} not found in {path}. Keys: {list(data.keys())}")
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 9:
        raise ValueError(f"{key} must have shape [T, >=9], got {arr.shape}")
    frames = parse_episode_frames(frame_spec, arr.shape[0], stride=stride)
    root_states = None
    if "replay_root_state" in data:
        root_states = np.asarray(data["replay_root_state"], dtype=np.float64)
        if root_states.ndim != 2 or root_states.shape[1] < 7 or root_states.shape[0] < arr.shape[0]:
            root_states = None
    targets = []
    for frame in frames:
        if frame < 0 or frame >= arr.shape[0]:
            raise IndexError(f"Episode frame {frame} out of range [0, {arr.shape[0]})")
        row = arr[frame]
        pose = np.concatenate([row[2:5], normalize_quat_xyzw(row[5:9])])
        root_pose = None
        if root_states is not None:
            root_row = root_states[frame]
            root_pose = np.concatenate([root_row[:3], normalize_quat_xyzw(root_row[3:7])])
        targets.append(
            {
                "frame": int(frame),
                "pose": pose,
                "root_pose": root_pose,
                "gripper": float(row[9]) if row.shape[0] > 9 else 0.0,
            }
        )
    return path, targets


def find_xml_child(node, tag):
    for child in node:
        if child.tag == tag:
            return child
    return None


def populate_mesh_dir(source_mesh_dir, mesh_dir):
    source_mesh_dir = Path(source_mesh_dir)
    if not source_mesh_dir.exists():
        raise FileNotFoundError(f"A2WZ1 mesh directory not found: {source_mesh_dir}")
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_mesh_dir.iterdir():
        target_path = mesh_dir / source_path.name
        if target_path.exists():
            continue
        try:
            target_path.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def write_filtered_urdf(source_urdf, output_urdf, keep_links, robot_name):
    source_root = ET.parse(source_urdf).getroot()
    output_root = ET.Element(source_root.tag, {"name": robot_name})

    for child in source_root:
        if child.tag == "material":
            output_root.append(child)
        elif child.tag == "link" and child.get("name") in keep_links:
            output_root.append(child)
        elif child.tag == "joint":
            parent = find_xml_child(child, "parent")
            child_link = find_xml_child(child, "child")
            if parent is None or child_link is None:
                continue
            if parent.get("link") in keep_links and child_link.get("link") in keep_links:
                output_root.append(child)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(output_urdf, encoding="utf-8", xml_declaration=True)


def build_a2w_base_visual_asset_root(source_asset_root, asset_file, temp_root):
    source_asset_root = Path(source_asset_root).expanduser().resolve()
    source_urdf = source_asset_root / asset_file
    if not source_urdf.exists():
        raise FileNotFoundError(f"A2WZ1 URDF not found: {source_urdf}")

    split_root = Path(temp_root) / "a2w_base_visual"
    split_urdf_dir = split_root / "urdf"
    split_urdf_dir.mkdir(parents=True, exist_ok=True)
    populate_mesh_dir(source_asset_root / "meshes", split_root / "meshes")

    base_file = "urdf/a2w_base_visual_only.urdf"
    write_filtered_urdf(source_urdf, split_root / base_file, A2W_BASE_LINKS, "a2w_base")
    return split_root, base_file


class DualZ1IKPlay:
    def __init__(self, args, pin_ik, rng, gymapi, gymtorch, gymutil, torch):
        self.args = args
        self.pin_ik = pin_ik
        self.rng = rng
        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.gymutil = gymutil
        self.torch = torch
        self.gym = gymapi.acquire_gym()
        self.device = torch.device("cpu")
        self.command_id = 0
        self.reached_reported = False
        self.pin_goal_q6 = np.asarray(args.home_q, dtype=np.float64).copy()
        self.target_q6 = self.pin_goal_q6.copy()
        self.target_local_pose = self.pin_ik.fk(self.target_q6)
        self.last_left_error = (math.inf, math.inf)
        self.last_right_error = (math.inf, math.inf)
        self.command_step = 0
        self.temp_dirs = []
        self.left_base_actor = None
        self.right_base_actor = None
        self.left_scene_anchor_pose = np.asarray(
            [-0.75, 0.0, float(args.root_z), 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.right_scene_anchor_pose = np.asarray(
            [0.75, 0.0, float(args.root_z), 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.left_root_pose = self.left_scene_anchor_pose.copy()
        self.right_root_pose = self.right_scene_anchor_pose.copy()
        self.episode_reference_root_pose = None
        self.episode_path = None
        self.episode_targets = []
        self.episode_index = 0
        if str(args.episode).strip():
            self.episode_path, self.episode_targets = load_episode_targets(
                args.episode,
                args.episode_key,
                args.episode_frames,
                stride=int(args.episode_stride),
            )
            if self.episode_targets and self.episode_targets[0].get("root_pose") is not None:
                self.episode_reference_root_pose = self.episode_targets[0]["root_pose"].copy()

        self._create_sim()
        self._load_asset_and_actors()
        self._prepare_tensors()
        self._reset_actors()
        self._create_viewer()
        self._build_draw_geometries()

    def _create_sim(self):
        sim_params = self.gymapi.SimParams()
        sim_params.up_axis = self.gymapi.UP_AXIS_Z
        sim_params.gravity = self.gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.dt = 1.0 / 60.0
        self.sim = self.gym.create_sim(0, 0, self.gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        plane_params = self.gymapi.PlaneParams()
        plane_params.normal = self.gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _load_asset_and_actors(self):
        asset_options = self.gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = int(self.gymapi.DOF_MODE_POS)
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = not bool(self.args.disable_arm_visual_flip)
        asset_options.thickness = 0.001
        asset_options.armature = 0.01

        self.asset_root = str(Path(self.args.asset_root).expanduser().resolve())
        self.asset_file = str(self.args.asset_file)
        print(
            f"Loading Z1 arm asset: root={self.asset_root}, file={self.asset_file}, "
            f"flip_visual_attachments={asset_options.flip_visual_attachments}"
        )
        self.asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)
        if self.asset is None:
            raise RuntimeError(f"Failed to load asset root={self.asset_root} file={self.asset_file}")

        self.base_asset = self._load_a2w_base_asset() if bool(self.args.show_a2w_base) else None

        self.env = self.gym.create_env(
            self.sim,
            self.gymapi.Vec3(-2.0, -1.2, -0.2),
            self.gymapi.Vec3(2.0, 1.2, 1.5),
            1,
        )

        self.left_root_tf = make_transform(self.gymapi, self.left_root_pose[:3], self.left_root_pose[3:7])
        self.right_root_tf = make_transform(self.gymapi, self.right_root_pose[:3], self.right_root_pose[3:7])
        self.left_root_pose = pose_to_np(self.left_root_tf)
        self.right_root_pose = pose_to_np(self.right_root_tf)

        if self.base_asset is not None:
            self.left_base_actor = self.gym.create_actor(
                self.env,
                self.base_asset,
                self.left_root_tf,
                "a2w_base_gym_jacobian",
                0,
                0,
            )
            self.right_base_actor = self.gym.create_actor(
                self.env,
                self.base_asset,
                self.right_root_tf,
                "a2w_base_pinocchio",
                0,
                0,
            )
            self._paint_actor(self.left_base_actor, self.gymapi.Vec3(0.34, 0.37, 0.42))
            self._paint_actor(self.right_base_actor, self.gymapi.Vec3(0.34, 0.42, 0.36))
            self._configure_a2w_base_actor(self.left_base_actor)
            self._configure_a2w_base_actor(self.right_base_actor)

        self.left_actor = self.gym.create_actor(
            self.env,
            self.asset,
            self.left_root_tf,
            LEFT_ACTOR_NAME,
            0,
            0,
        )
        self.right_actor = self.gym.create_actor(
            self.env,
            self.asset,
            self.right_root_tf,
            RIGHT_ACTOR_NAME,
            0,
            0,
        )

        self.dof_names = list(self.gym.get_asset_dof_names(self.asset))
        self.num_dofs = len(self.dof_names)
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        self.dof_props["driveMode"].fill(int(self.gymapi.DOF_MODE_POS))
        self.dof_props["stiffness"].fill(float(self.args.stiffness))
        self.dof_props["damping"].fill(float(self.args.damping))
        self.gym.set_actor_dof_properties(self.env, self.left_actor, self.dof_props)
        self.gym.set_actor_dof_properties(self.env, self.right_actor, self.dof_props)

        self.lower = np.asarray(self.dof_props["lower"], dtype=np.float64)
        self.upper = np.asarray(self.dof_props["upper"], dtype=np.float64)
        self.control_indices = np.asarray(
            [self.dof_names.index(name) for name in DEFAULT_Z1_JOINT_NAMES if name in self.dof_names],
            dtype=np.int64,
        )
        if self.control_indices.shape[0] != len(DEFAULT_Z1_JOINT_NAMES):
            raise RuntimeError(f"Missing Z1 control DOFs. Loaded DOFs: {self.dof_names}")

        self.body_names = list(self.gym.get_asset_rigid_body_names(self.asset))
        if self.args.ee_link not in self.body_names:
            raise RuntimeError(f"EE link {self.args.ee_link!r} not in asset bodies: {self.body_names}")
        self.left_ee_body_sim_index = self.gym.find_actor_rigid_body_index(
            self.env,
            self.left_actor,
            self.args.ee_link,
            self.gymapi.DOMAIN_SIM,
        )
        self.right_ee_body_sim_index = self.gym.find_actor_rigid_body_index(
            self.env,
            self.right_actor,
            self.args.ee_link,
            self.gymapi.DOMAIN_SIM,
        )

        self._paint_actor(self.left_actor, self.gymapi.Vec3(0.25, 0.48, 1.0))
        self._paint_actor(self.right_actor, self.gymapi.Vec3(0.10, 0.85, 0.45))

    def _load_a2w_base_asset(self):
        temp_dir = tempfile.TemporaryDirectory(prefix="a2w_base_visual_")
        self.temp_dirs.append(temp_dir)
        split_root, base_file = build_a2w_base_visual_asset_root(
            self.args.a2w_asset_root,
            self.args.a2w_asset_file,
            temp_dir.name,
        )
        asset_options = self.gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = int(self.gymapi.DOF_MODE_POS)
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = False
        asset_options.thickness = 0.001
        asset_options.armature = 0.01
        print(f"Loading A2W base visual: root={split_root}, file={base_file}")
        asset = self.gym.load_asset(self.sim, str(split_root), base_file, asset_options)
        if asset is None:
            raise RuntimeError(f"Failed to load A2W base visual root={split_root} file={base_file}")
        return asset

    def _configure_a2w_base_actor(self, actor):
        if actor is None or self.base_asset is None:
            return
        num_dofs = self.gym.get_asset_dof_count(self.base_asset)
        if num_dofs <= 0:
            return
        dof_names = list(self.gym.get_asset_dof_names(self.base_asset))
        dof_props = self.gym.get_asset_dof_properties(self.base_asset)
        dof_props["driveMode"].fill(int(self.gymapi.DOF_MODE_POS))
        dof_props["stiffness"].fill(float(self.args.stiffness))
        dof_props["damping"].fill(float(self.args.damping))
        lower = np.asarray(dof_props["lower"], dtype=np.float64)
        upper = np.asarray(dof_props["upper"], dtype=np.float64)
        has_limits = np.asarray(dof_props["hasLimits"], dtype=bool)
        states = np.zeros(num_dofs, dtype=self.gymapi.DofState.dtype)
        for idx, name in enumerate(dof_names):
            value = float(A2W_DEFAULT_LEG_POS.get(name, 0.0))
            if bool(has_limits[idx]) and float(lower[idx]) < float(upper[idx]):
                value = float(np.clip(value, lower[idx], upper[idx]))
            states["pos"][idx] = value
        self.gym.set_actor_dof_properties(self.env, actor, dof_props)
        self.gym.set_actor_dof_states(self.env, actor, states, self.gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, actor, states["pos"])

    def _paint_actor(self, actor, color):
        for body_i in range(self.gym.get_actor_rigid_body_count(self.env, actor)):
            self.gym.set_rigid_body_color(self.env, actor, body_i, self.gymapi.MESH_VISUAL, color)

    def _prepare_tensors(self):
        self.gym.prepare_sim(self.sim)
        self.rb_states = self.gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.dof_state_tensor = self.gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.jacobian = self.gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, LEFT_ACTOR_NAME))
        self.ee_jacobian_index, self.control_jacobian_indices = self._jacobian_mapping(self.args.ee_link)

    def _jacobian_mapping(self, ee_link):
        ee_asset_index = int(self.body_names.index(ee_link))
        jac = self.jacobian[0] if self.jacobian.ndim >= 4 else self.jacobian
        jac_body_dim = int(jac.shape[0])
        jac_col_dim = int(jac.shape[-1])
        if jac_body_dim == len(self.body_names):
            ee_jacobian_index = ee_asset_index
        elif jac_body_dim == len(self.body_names) - 1:
            ee_jacobian_index = ee_asset_index - 1
        else:
            raise RuntimeError(
                f"Cannot map Jacobian body dim={jac_body_dim} to asset bodies={len(self.body_names)}"
            )

        if jac_col_dim == self.num_dofs:
            col_offset = 0
        elif jac_col_dim == self.num_dofs + 6:
            col_offset = 6
        else:
            col_offset = jac_col_dim - self.num_dofs
            if col_offset < 0 or col_offset > 6:
                raise RuntimeError(f"Cannot map Jacobian columns={jac_col_dim}, dofs={self.num_dofs}")
        return ee_jacobian_index, self.control_indices + col_offset

    def _reset_actors(self):
        home_q = np.zeros(self.num_dofs, dtype=np.float64)
        home_q[self.control_indices] = np.asarray(self.args.home_q, dtype=np.float64)
        home_q = np.clip(home_q, self.lower, self.upper)
        if "jointGripper" in self.dof_names:
            home_q[self.dof_names.index("jointGripper")] = 0.0
        self._set_actor_state(self.left_actor, home_q)
        self._set_actor_state(self.right_actor, home_q)
        self._simulate_once()
        self._refresh()
        if self.episode_targets:
            self.set_episode_command(0, force=True)
        else:
            self.generate_command(force=True)

    def _create_viewer(self):
        self.viewer = None
        if self.args.headless:
            return
        camera_props = self.gymapi.CameraProperties()
        camera_props.width = 1600
        camera_props.height = 900
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer. Re-run with --headless for non-graphical smoke tests.")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_ESCAPE, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_S, "new_command")
        self.gym.viewer_camera_look_at(
            self.viewer,
            None,
            self.gymapi.Vec3(0.0, -3.2, float(self.args.root_z) + 1.0),
            self.gymapi.Vec3(0.0, 0.0, float(self.args.root_z) + 0.15),
        )

    def _build_draw_geometries(self):
        gymapi = self.gymapi
        gymutil = self.gymutil

        class ThickAxesGeometry(gymutil.LineGeometry):
            def __init__(self, scale=1.0, thickness=0.004):
                offsets = {
                    0: [(0, 0, 0), (0, thickness, 0), (0, -thickness, 0), (0, 0, thickness), (0, 0, -thickness)],
                    1: [(0, 0, 0), (thickness, 0, 0), (-thickness, 0, 0), (0, 0, thickness), (0, 0, -thickness)],
                    2: [(0, 0, 0), (thickness, 0, 0), (-thickness, 0, 0), (0, thickness, 0), (0, -thickness, 0)],
                }
                axis_end = [(scale, 0, 0), (0, scale, 0), (0, 0, scale)]
                axis_color = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
                verts = np.empty((15, 2), gymapi.Vec3.dtype)
                colors = np.empty(15, gymapi.Vec3.dtype)
                idx = 0
                for axis in range(3):
                    for offset in offsets[axis]:
                        verts[idx][0] = offset
                        verts[idx][1] = tuple(axis_end[axis][j] + offset[j] for j in range(3))
                        colors[idx] = axis_color[axis]
                        idx += 1
                self.verts = verts
                self._colors = colors

            def vertices(self):
                return self.verts

            def colors(self):
                return self._colors

        radius = float(self.args.target_sphere_radius)
        self.target_geom = gymutil.WireframeSphereGeometry(
            radius=radius,
            num_lats=10,
            num_lons=10,
            color=(1.0, 0.82, 0.05),
            color2=(1.0, 0.45, 0.05),
        )
        self.left_current_geom = gymutil.WireframeSphereGeometry(
            radius=radius * 0.75,
            num_lats=8,
            num_lons=8,
            color=(0.15, 0.35, 1.0),
            color2=(0.15, 0.35, 1.0),
        )
        self.right_current_geom = gymutil.WireframeSphereGeometry(
            radius=radius * 0.75,
            num_lats=8,
            num_lons=8,
            color=(0.0, 0.9, 0.35),
            color2=(0.0, 0.9, 0.35),
        )
        self.axes_geom = ThickAxesGeometry(scale=float(self.args.axis_scale), thickness=0.004)

    def _set_actor_state(self, actor, q):
        q = np.asarray(q, dtype=np.float64).reshape(self.num_dofs)
        states = self.gym.get_actor_dof_states(self.env, actor, self.gymapi.STATE_ALL)
        states["pos"][:] = q.astype(np.float32)
        states["vel"][:] = 0.0
        self.gym.set_actor_dof_states(self.env, actor, states, self.gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, actor, q.astype(np.float32))

    def _set_actor_root_pose(self, actor, pose):
        if actor is None:
            return
        root_handle = self.gym.get_actor_root_rigid_body_handle(self.env, actor)
        if int(root_handle) < 0:
            return
        tf = make_transform(self.gymapi, pose[:3], pose[3:7])
        self.gym.set_rigid_transform(self.env, root_handle, tf)

    def _apply_scene_root_poses(self):
        self._set_actor_root_pose(self.left_actor, self.left_root_pose)
        self._set_actor_root_pose(self.right_actor, self.right_root_pose)
        self._set_actor_root_pose(self.left_base_actor, self.left_root_pose)
        self._set_actor_root_pose(self.right_base_actor, self.right_root_pose)

    def _set_roots_from_episode_item(self, item):
        if str(self.args.episode_root_mode) == "static":
            self.left_root_pose = self.left_scene_anchor_pose.copy()
            self.right_root_pose = self.right_scene_anchor_pose.copy()
            self._apply_scene_root_poses()
            return
        root_pose = item.get("root_pose")
        if root_pose is None or self.episode_reference_root_pose is None:
            self.left_root_pose = self.left_scene_anchor_pose.copy()
            self.right_root_pose = self.right_scene_anchor_pose.copy()
            self._apply_scene_root_poses()
            return
        delta_pose = relative_pose(self.episode_reference_root_pose, root_pose)
        self.left_root_pose = compose_pose(self.left_scene_anchor_pose, delta_pose)
        self.right_root_pose = compose_pose(self.right_scene_anchor_pose, delta_pose)
        self._apply_scene_root_poses()

    def _actor_q(self, actor):
        states = self.gym.get_actor_dof_states(self.env, actor, self.gymapi.STATE_ALL)
        return np.asarray(states["pos"], dtype=np.float64).copy()

    def _simulate_once(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def _refresh(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def _ee_pose(self, body_sim_index):
        state = self.rb_states[body_sim_index].detach().cpu().numpy()
        return np.concatenate(
            [
                np.asarray(state[:3], dtype=np.float64),
                normalize_quat_xyzw(np.asarray(state[3:7], dtype=np.float64)),
            ]
        )

    def _set_target_from_pose(self, local_pose, label):
        local_pose = np.asarray(local_pose, dtype=np.float64).reshape(7)
        solved = self.pin_ik.ik(
            local_pose,
            seed_joints=self._actor_q(self.right_actor)[self.control_indices],
            position_only=False,
            max_iter=int(self.args.pin_max_iter),
            max_restarts=int(self.args.pin_restarts),
            dt=float(self.args.pin_dt),
            damping=float(self.args.pin_damping),
            threshold=float(self.args.pin_threshold),
            rotation_weight=float(self.args.rot_weight),
            rng=self.rng,
        )
        if solved is None:
            print(f"{label}: Pinocchio full 6D IK failed; right actor will hold current q.")
            solved = self._actor_q(self.right_actor)[self.control_indices]

        self.command_id += 1
        self.command_step = 0
        self.reached_reported = False
        self.target_q6 = np.asarray(solved, dtype=np.float64)
        self.target_local_pose = local_pose.copy()
        self.pin_goal_q6 = np.asarray(solved, dtype=np.float64)

        target_left = transform_pose(self.left_root_pose, self.target_local_pose)
        print(
            f"[command {self.command_id}] {label} local xyz="
            f"({local_pose[0]:+.4f}, {local_pose[1]:+.4f}, {local_pose[2]:+.4f}) "
            f"quat=({local_pose[3]:+.4f}, {local_pose[4]:+.4f}, {local_pose[5]:+.4f}, {local_pose[6]:+.4f})"
        )
        print(
            f"  world target left xyz=({target_left[0]:+.4f}, {target_left[1]:+.4f}, {target_left[2]:+.4f})"
        )
        print(f"  pin_goal_q6: {np.array2string(self.pin_goal_q6, precision=4, suppress_small=True)}")

    def set_episode_command(self, index, force=False):
        del force
        if not self.episode_targets:
            return False
        if index >= len(self.episode_targets):
            if bool(self.args.episode_loop):
                index = 0
            else:
                return False
        self.episode_index = int(index)
        item = self.episode_targets[self.episode_index]
        self._set_roots_from_episode_item(item)
        label = f"episode={self.episode_path.name} frame={item['frame']} idx={self.episode_index + 1}/{len(self.episode_targets)}"
        self._set_target_from_pose(item["pose"], label)
        return True

    def generate_command(self, force=False):
        if self.episode_targets:
            next_index = self.episode_index if force else self.episode_index + 1
            return self.set_episode_command(next_index, force=force)
        del force
        lower6 = self.lower[self.control_indices]
        upper6 = self.upper[self.control_indices]
        center = self._actor_q(self.left_actor)[self.control_indices]
        margin = np.minimum(0.05, np.maximum(upper6 - lower6, 0.0) * 0.02)

        solved = None
        target_q6 = None
        target_pose = None
        for _ in range(40):
            delta = self.rng.uniform(-float(self.args.target_delta), float(self.args.target_delta), size=6)
            delta[5] *= 1.6
            q_candidate = np.clip(center + delta, lower6 + margin, upper6 - margin)
            pose_candidate = self.pin_ik.fk(q_candidate)
            q_solved = self.pin_ik.ik(
                pose_candidate,
                seed_joints=self._actor_q(self.right_actor)[self.control_indices],
                position_only=False,
                max_iter=int(self.args.pin_max_iter),
                max_restarts=int(self.args.pin_restarts),
                dt=float(self.args.pin_dt),
                damping=float(self.args.pin_damping),
                threshold=float(self.args.pin_threshold),
                rotation_weight=float(self.args.rot_weight),
                rng=self.rng,
            )
            if q_solved is not None:
                target_q6 = q_candidate
                target_pose = pose_candidate
                solved = q_solved
                break

        if solved is None:
            target_q6 = np.clip(center, lower6 + margin, upper6 - margin)
            target_pose = self.pin_ik.fk(target_q6)
            solved = target_q6.copy()
            print("Pinocchio IK did not converge for sampled targets; using the FK joint sample directly.")

        self.command_id += 1
        self.command_step = 0
        self.reached_reported = False
        self.target_q6 = np.asarray(target_q6, dtype=np.float64)
        self.target_local_pose = np.asarray(target_pose, dtype=np.float64)
        self.pin_goal_q6 = np.asarray(solved, dtype=np.float64)

        target_left = transform_pose(self.left_root_pose, self.target_local_pose)
        print(
            f"[command {self.command_id}] 6D target xyz="
            f"({target_left[0]:+.3f}, {target_left[1]:+.3f}, {target_left[2]:+.3f}) "
            f"quat=({target_left[3]:+.3f}, {target_left[4]:+.3f}, {target_left[5]:+.3f}, {target_left[6]:+.3f})"
        )
        print(f"  target_q6 from FK: {np.array2string(self.target_q6, precision=3, suppress_small=True)}")
        print(f"  pin_goal_q6:        {np.array2string(self.pin_goal_q6, precision=3, suppress_small=True)}")

    def _left_target_pose(self):
        return transform_pose(self.left_root_pose, self.target_local_pose)

    def _right_target_pose(self):
        return transform_pose(self.right_root_pose, self.target_local_pose)

    def _update_left_gym_jacobian(self):
        torch = self.torch
        target_pose = self._left_target_pose()
        target_pos = torch.tensor(target_pose[:3], dtype=torch.float32, device=self.jacobian.device)
        target_quat = torch.tensor(target_pose[3:7], dtype=torch.float32, device=self.jacobian.device)

        eef_state = self.rb_states[self.left_ee_body_sim_index]
        eef_pos = eef_state[:3]
        eef_quat = eef_state[3:7]
        pos_err = target_pos - eef_pos
        orn_err = torch_orientation_error(torch, target_quat, eef_quat)

        jac = self.jacobian[0] if self.jacobian.ndim >= 4 else self.jacobian
        j_eef = jac[self.ee_jacobian_index, :, :]
        control_cols = torch.tensor(self.control_jacobian_indices, dtype=torch.long, device=j_eef.device)
        j_control = j_eef[:, control_cols]

        dpose = torch.cat(
            (
                float(self.args.ik_pos_gain) * pos_err,
                float(self.args.ik_rot_gain) * orn_err,
            ),
            dim=0,
        )
        weights = torch.tensor(
            [1.0, 1.0, 1.0, self.args.rot_weight, self.args.rot_weight, self.args.rot_weight],
            dtype=torch.float32,
            device=j_control.device,
        )
        task_j = j_control * weights.view(6, 1)
        task_err = dpose * weights

        j_t = torch.transpose(task_j, 0, 1)
        damping = max(1.0e-6, float(self.args.gym_ik_damping))
        lhs = task_j @ j_t + torch.eye(task_j.shape[0], dtype=torch.float32, device=task_j.device) * (
            damping * damping
        )
        delta = j_t @ torch.linalg.solve(lhs, task_err.unsqueeze(-1)).squeeze(-1)
        delta = torch.clamp(delta, -float(self.args.gym_max_step), float(self.args.gym_max_step))

        current_q = self._actor_q(self.left_actor)
        next_q = current_q.copy()
        next_q[self.control_indices] += delta.detach().cpu().numpy().astype(np.float64)
        next_q = np.clip(next_q, self.lower, self.upper)
        self.gym.set_actor_dof_position_targets(self.env, self.left_actor, next_q.astype(np.float32))

    def _update_right_pinocchio(self):
        current_q = self._actor_q(self.right_actor)
        target_q = current_q.copy()
        target_q[self.control_indices] = self.pin_goal_q6
        target_q = np.clip(target_q, self.lower, self.upper)
        self.gym.set_actor_dof_position_targets(self.env, self.right_actor, target_q.astype(np.float32))

    def _update_errors(self):
        left_pose = self._ee_pose(self.left_ee_body_sim_index)
        right_pose = self._ee_pose(self.right_ee_body_sim_index)
        self.last_left_error = pose_error(self._left_target_pose(), left_pose)
        self.last_right_error = pose_error(self._right_target_pose(), right_pose)
        both_reached = (
            self.last_left_error[0] <= float(self.args.pos_tol)
            and self.last_left_error[1] <= float(self.args.rot_tol)
            and self.last_right_error[0] <= float(self.args.pos_tol)
            and self.last_right_error[1] <= float(self.args.rot_tol)
        )
        if both_reached and not self.reached_reported:
            print(
                f"[command {self.command_id}] reached: "
                f"gym pos={self.last_left_error[0]:.4f} rot={self.last_left_error[1]:.4f} | "
                f"pin pos={self.last_right_error[0]:.4f} rot={self.last_right_error[1]:.4f}"
            )
            self.reached_reported = True

    def _maybe_advance_episode(self):
        if not self.episode_targets or not bool(self.args.episode_autoplay):
            return
        hold_steps = max(1, int(self.args.episode_hold_steps))
        if self.command_step > 0 and self.command_step % hold_steps == 0:
            if not self.set_episode_command(self.episode_index + 1):
                print("[episode] finished selected frames; stopping autoplay.")
                self.args.episode_autoplay = False

    def _draw(self):
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        left_target = self._left_target_pose()
        right_target = self._right_target_pose()
        left_current = self._ee_pose(self.left_ee_body_sim_index)
        right_current = self._ee_pose(self.right_ee_body_sim_index)

        for pose in (left_target, right_target):
            tf = make_transform(self.gymapi, pose[:3], pose[3:7])
            self.gymutil.draw_lines(self.target_geom, self.gym, self.viewer, self.env, tf)
            self.gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.env, tf)

        if bool(self.args.show_a2w_base):
            for pose in (self.left_root_pose, self.right_root_pose):
                root_tf = make_transform(self.gymapi, pose[:3], pose[3:7])
                self.gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.env, root_tf)

        left_tf = make_transform(self.gymapi, left_current[:3], left_current[3:7])
        right_tf = make_transform(self.gymapi, right_current[:3], right_current[3:7])
        self.gymutil.draw_lines(self.left_current_geom, self.gym, self.viewer, self.env, left_tf)
        self.gymutil.draw_lines(self.right_current_geom, self.gym, self.viewer, self.env, right_tf)
        self.gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.env, left_tf)
        self.gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.env, right_tf)
        self._draw_error_line(left_current, left_target, (0.25, 0.48, 1.0))
        self._draw_error_line(right_current, right_target, (0.10, 0.85, 0.45))

    def _draw_error_line(self, current_pose, target_pose, color):
        p1 = self.gymapi.Vec3(float(current_pose[0]), float(current_pose[1]), float(current_pose[2]))
        p2 = self.gymapi.Vec3(float(target_pose[0]), float(target_pose[1]), float(target_pose[2]))
        c = self.gymapi.Vec3(float(color[0]), float(color[1]), float(color[2]))
        self.gymutil.draw_line(p1, p2, c, self.gym, self.viewer, self.env)

    def _handle_viewer_events(self):
        if self.viewer is None:
            return False
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "quit" and event.value > 0:
                return True
            if event.action == "new_command" and event.value > 0:
                self._refresh()
                self.generate_command()
        return False

    def run(self):
        if self.episode_targets:
            print("Viewer controls: S = next episode pose, Esc = quit.")
            frames = [item["frame"] for item in self.episode_targets]
            frame_text = frames if len(frames) <= 20 else f"{frames[:8]} ... {frames[-4:]} ({len(frames)} frames)"
            print(
                f"Episode replay: path={self.episode_path}, frames="
                f"{frame_text}, autoplay={bool(self.args.episode_autoplay)}, "
                f"root_mode={self.args.episode_root_mode}"
            )
        else:
            print("Viewer controls: S = new 6D command, Esc = quit.")
        print("Left blue actor = Isaac Gym Jacobian IK; right green actor = Pinocchio IK.")
        print("Both controllers use the same target position and orientation, transformed into each actor base.")

        step = 0
        max_steps = int(self.args.max_steps)
        while True:
            if self.viewer is not None and self.gym.query_viewer_has_closed(self.viewer):
                break
            if max_steps > 0 and step >= max_steps:
                break
            if self._handle_viewer_events():
                break

            self._refresh()
            self._update_left_gym_jacobian()
            self._update_right_pinocchio()
            self._simulate_once()
            self._refresh()
            self._update_errors()

            if self.args.print_interval > 0 and step % int(self.args.print_interval) == 0:
                print(
                    f"[step {step:05d}] "
                    f"gym pos={self.last_left_error[0]:.4f} rot={self.last_left_error[1]:.4f} | "
                    f"pin pos={self.last_right_error[0]:.4f} rot={self.last_right_error[1]:.4f}"
                )

            if self.viewer is not None:
                self._draw()
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            step += 1
            self.command_step += 1
            self._maybe_advance_episode()

        print(
            "Final errors: "
            f"gym pos={self.last_left_error[0]:.4f} rot={self.last_left_error[1]:.4f} | "
            f"pin pos={self.last_right_error[0]:.4f} rot={self.last_right_error[1]:.4f}"
        )

    def destroy(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    # Import Pinocchio before Isaac Gym/Torch in this environment; it avoids shared library collisions.
    pin_ik = Z1PinocchioIK(args.urdf, ee_link=args.ee_link)

    from isaacgym import gymapi, gymtorch, gymutil
    import torch

    app = DualZ1IKPlay(args, pin_ik, rng, gymapi, gymtorch, gymutil, torch)
    try:
        app.run()
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
