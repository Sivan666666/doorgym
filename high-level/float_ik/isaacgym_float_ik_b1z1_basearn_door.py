#!/usr/bin/env python3
"""Float-base B1Z1 base+arm IK door-pull demo.

This script intentionally reuses the known-good split-asset/IK loader from
isaacgym_visualize_b1z1_basearn.py, then adds one door actor and the scripted
door-pull trajectory from play_b1z1_walk_with_door_asset_camera.py.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = HIGH_LEVEL_ROOT.parents[0]
BASE_FLOAT_IK_SCRIPT = SCRIPT_DIR / "isaacgym_visualize_b1z1_basearn.py"
DEFAULT_DOOR_CFG = HIGH_LEVEL_ROOT / "data" / "cfg" / "b1z1_opendoor.yaml"
DEFAULT_DOOR_ASSET_NAMES = (
    "99650089960001",
    "99650089960006",
    "99655039960001",
    "99655039960006",
)


def load_base_float_ik_module():
    spec = importlib.util.spec_from_file_location("b1z1_basearn_float_ik", BASE_FLOAT_IK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {BASE_FLOAT_IK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base_ik = load_base_float_ik_module()
gymapi = base_ik.gymapi
gymutil = base_ik.gymutil


@dataclass
class DoorRuntime:
    asset_root: str
    asset_file_door: str
    spec: dict
    bounding: dict
    handle_bounding: dict
    asset: object
    body_names: list[str]
    dof_names: list[str]
    dof_lower: np.ndarray
    dof_upper: np.ndarray
    handle_body_index: int
    handle_goal_offset: np.ndarray
    handle_unlock_threshold: float
    open_stage: bool = False


def sorted_asset_items(asset_dict):
    def key_fn(item):
        key = item[0]
        return (0, int(key)) if str(key).isdigit() else (1, str(key))

    return [item[1] for item in sorted(asset_dict.items(), key=key_fn)]


def parse_args():
    args = gymutil.parse_arguments(
        description="B1Z1 base+arm float IK door-pull demo.",
        headless=True,
        no_graphics=True,
        custom_parameters=[
            {"name": "--asset_root", "type": str, "default": str(base_ik.DEFAULT_ASSET_ROOT)},
            {"name": "--asset_file", "type": str, "default": base_ik.DEFAULT_ASSET_FILE},
            {"name": "--steps", "type": int, "default": 3600},
            {"name": "--door_cfg", "type": str, "default": str(DEFAULT_DOOR_CFG)},
            {"name": "--door_name", "type": str, "default": ""},
            {"name": "--door_index", "type": int, "default": -1},
            {"name": "--door_actor_scale", "type": float, "default": 1.2},
            {"name": "--door_x", "type": float, "default": 2.5},
            {"name": "--door_y", "type": float, "default": 0.0},
            {"name": "--door_z_offset", "type": float, "default": 0.01},
            {"name": "--robot_x", "type": float, "default": 4.1},
            {"name": "--robot_y", "type": float, "default": 0.0},
            {"name": "--robot_z", "type": float, "default": 0.60},
            {"name": "--robot_yaw", "type": float, "default": math.pi},
            {"name": "--robot_front_offset", "type": float, "default": 0.55},
            {"name": "--robot_rear_offset", "type": float, "default": 0.65},
            {"name": "--stop_distance", "type": float, "default": 0.25},
            {"name": "--pull_base_distance", "type": float, "default": -0.35},
            {"name": "--base_pull_time_scale", "type": float, "default": 0.40},
            {"name": "--pull_base_yaw_delta", "type": float, "default": 0.18},
            {"name": "--door_pass_clearance", "type": float, "default": 0.55},
            {"name": "--walk_steps", "type": int, "default": 260},
            {"name": "--initial_hold_steps", "type": int, "default": 150},
            {"name": "--grasp_steps", "type": int, "default": 150},
            {"name": "--grasp_hold_steps", "type": int, "default": 100},
            {"name": "--gripper_close_steps", "type": int, "default": 120},
            {"name": "--handle_rotate_steps", "type": int, "default": 300},
            {"name": "--door_pull_steps", "type": int, "default": 960},
            {"name": "--pass_through_steps", "type": int, "default": 620},
            {"name": "--hold_steps", "type": int, "default": 300},
            {"name": "--pregrasp_offset", "type": float, "default": 0.15},
            {"name": "--grasp_offset", "type": float, "default": 0.0},
            {"name": "--grasp_x_offset", "type": float, "default": -0.03},
            {"name": "--grasp_z_offset", "type": float, "default": -0.03},
            {"name": "--handle_rotate_right_distance", "type": float, "default": 0.03},
            {"name": "--handle_rotate_down_distance", "type": float, "default": 0.03},
            {"name": "--handle_rotate_angle", "type": float, "default": 1.05},
            {"name": "--door_pull_distance", "type": float, "default": 1.10},
            {"name": "--lever_step_size", "type": float, "default": 0.06},
            {"name": "--pass_open_angle_deg", "type": float, "default": 80.0},
            {"name": "--pass_home_wait_min_steps", "type": int, "default": 120},
            {"name": "--pass_home_wait_max_steps", "type": int, "default": 420},
            {"name": "--pass_home_ready_tolerance", "type": float, "default": 0.05},
            {
                "name": "--unidoor_style_pull",
                "action": "store_true",
                "default": True,
                "help": "During pull, step from current EE along the handle pull direction.",
            },
            {"name": "--no_unidoor_style_pull", "dest": "unidoor_style_pull", "action": "store_false"},
            {"name": "--gripper_open", "type": float, "default": -1.5707963267948966},
            {"name": "--gripper_closed", "type": float, "default": 0.0},
            {"name": "--gripper_close_ratio", "type": float, "default": 0.8},
            {"name": "--handle_spring_stiffness", "type": float, "default": 0.5},
            {"name": "--handle_spring_damping", "type": float, "default": 0.1},
            {"name": "--handle_unlock_ratio", "type": float, "default": 40.0 / 45.0},
            {"name": "--door_open_resistance", "type": float, "default": 0.2},
            {"name": "--door_open_damping", "type": float, "default": 0.05},
            {"name": "--door_lock_force", "type": float, "default": 0.0},
            {"name": "--door_joint_friction", "type": float, "default": 0.5},
            {"name": "--door_joint_damping", "type": float, "default": 0.2},
            {"name": "--handle_joint_friction", "type": float, "default": 0.05},
            {"name": "--handle_joint_damping", "type": float, "default": 0.05},
            {"name": "--door_auto_open_force", "type": float, "default": 200.0},
            {"name": "--door_auto_open_sign", "type": float, "default": 1.0},
            {"name": "--door_auto_open_target_ratio", "type": float, "default": 0.95},
            {"name": "--door_vhacd_resolution", "type": int, "default": 100000},
            {"name": "--forward_ee_roll", "type": float, "default": math.pi / 2},
            {"name": "--forward_ee_pitch", "type": float, "default": 0.0},
            {"name": "--gripper_red_axis_rot", "type": float, "default": -math.pi / 2},
            {"name": "--ik_pos_gain", "type": float, "default": 1.0},
            {"name": "--ik_rot_gain", "type": float, "default": 0.7},
            {"name": "--ik_rot_weight", "type": float, "default": 0.1},
            {"name": "--ik_damping", "type": float, "default": 0.08},
            {"name": "--ik_max_step", "type": float, "default": 0.06},
            {"name": "--ik_pos_tolerance", "type": float, "default": 0.015},
            {"name": "--ik_rot_tolerance", "type": float, "default": 0.08},
            {"name": "--ik_position_only", "action": "store_true"},
            {"name": "--ik_include_gripper", "action": "store_true"},
            {"name": "--ik_ee_link", "type": str, "default": base_ik.EE_GRIPPER_LINK},
            {"name": "--stiffness", "type": float, "default": 80.0},
            {"name": "--damping", "type": float, "default": 8.0},
            {"name": "--speed_scale", "type": float, "default": 0.6},
            {"name": "--range_scale", "type": float, "default": 0.75},
            {"name": "--joint_filter", "type": str, "default": ""},
            {"name": "--single_asset", "action": "store_true"},
            {"name": "--flip_visual_attachments", "action": "store_true"},
            {"name": "--disable_arm_visual_flip", "action": "store_true"},
            {"name": "--base_visual_flip", "action": "store_true"},
            {"name": "--no_disable_gravity", "action": "store_true"},
            {"name": "--disable_self_collisions", "action": "store_true"},
            {"name": "--print_collision_summary", "action": "store_true"},
            {"name": "--log_interval", "type": int, "default": 60},
            {"name": "--draw_ik_target", "action": "store_true"},
        ],
    )

    args.ik_demo = True
    args.ik_target_pose = ""
    args.ik_demo_offset = "0 0 0"
    args.ik_keep_base_motion = False
    args.disable_base_motion = True
    args.zero_pose_seconds = 0.0
    args.zero_pose_only = False
    args.show_axis = False
    args.base_motion_amplitude = 0.0
    args.base_motion_yaw = 0.0
    args.base_motion_period = 1.0
    return args


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(a, b, t):
    return a + (b - a) * float(t)


def normalize(vec, eps=1.0e-6):
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec * 0.0
    return vec / norm


def quat_apply(q, v):
    q = base_ik.normalize_quat(q)
    v_quat = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    return base_ik.quat_multiply(
        base_ik.quat_multiply(q, v_quat),
        base_ik.quat_conjugate(q),
    )[:3]


def quat_from_angle_axis(angle, axis):
    axis = normalize(np.asarray(axis, dtype=np.float32))
    half = 0.5 * float(angle)
    return base_ik.normalize_quat(
        np.array(
            [axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half), math.cos(half)],
            dtype=np.float32,
        )
    )


def quat_axis(q, axis):
    basis = np.zeros(3, dtype=np.float32)
    basis[int(axis)] = 1.0
    return quat_apply(q, basis)


def forward_ee_quat(args, base_yaw):
    base_quat = base_ik.rpy_to_quat(args.forward_ee_roll, args.forward_ee_pitch, base_yaw)
    red_axis_quat = base_ik.rpy_to_quat(args.gripper_red_axis_rot, 0.0, 0.0)
    return base_ik.quat_multiply(base_quat, red_axis_quat)


def approach_ee_quat(args, approach_dir, fallback_yaw):
    approach_xy = np.asarray(approach_dir[:2], dtype=np.float32)
    norm = float(np.linalg.norm(approach_xy))
    if norm < 1.0e-5:
        return forward_ee_quat(args, fallback_yaw)
    # approach_dir points from handle to robot; the gripper should face back toward the handle.
    face_dir = -approach_xy / norm
    return forward_ee_quat(args, math.atan2(float(face_dir[1]), float(face_dir[0])))


def load_door_spec(args):
    cfg_path = Path(args.door_cfg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    asset_cfg = cfg["env"]["asset"]
    train_assets = asset_cfg["trainAssets"]
    load_block = asset_cfg.get("load_block") or next(iter(train_assets.keys()))
    specs = sorted_asset_items(train_assets[load_block])

    selected = None
    if args.door_name:
        selected = next((spec for spec in specs if spec.get("name") == args.door_name), None)
        if selected is None:
            raise RuntimeError(f"Door name {args.door_name!r} not found in {cfg_path}")
    elif args.door_index >= 0:
        if args.door_index >= len(specs):
            raise RuntimeError(f"--door_index={args.door_index} out of range for {len(specs)} doors")
        selected = specs[args.door_index]
    else:
        for name in DEFAULT_DOOR_ASSET_NAMES:
            selected = next((spec for spec in specs if spec.get("name") == name), None)
            if selected is not None:
                break
        if selected is None:
            selected = specs[0]

    asset_root = HIGH_LEVEL_ROOT / asset_cfg["assetRoot"]
    asset_file_door = asset_cfg["assetFileDoor"]
    door_set_root = asset_root / asset_file_door
    with (door_set_root / selected["bounding_box"]).open("r", encoding="utf-8") as f:
        bounding = json.load(f)
    with (door_set_root / selected["handle_bounding"]).open("r", encoding="utf-8") as f:
        handle_bounding = json.load(f)

    return str(asset_root), asset_file_door, selected, bounding, handle_bounding


def load_door_asset(gym, sim, args):
    asset_root, asset_file_door, spec, bounding, handle_bounding = load_door_spec(args)
    door_opts = gymapi.AssetOptions()
    door_opts.fix_base_link = True
    door_opts.collapse_fixed_joints = True
    door_opts.use_mesh_materials = True
    door_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    door_opts.override_com = True
    door_opts.override_inertia = True
    door_opts.disable_gravity = True
    door_opts.vhacd_enabled = True
    door_opts.vhacd_params = gymapi.VhacdParams()
    door_opts.vhacd_params.resolution = args.door_vhacd_resolution

    door_file = os.path.join(asset_file_door, spec["path"])
    print(f"Loading door: root={asset_root}, file={door_file}, name={spec['name']}")
    door_asset = gym.load_asset(sim, asset_root, door_file, door_opts)
    if door_asset is None:
        raise RuntimeError(f"Failed to load door asset {door_file}")

    body_names = gym.get_asset_rigid_body_names(door_asset)
    dof_names = gym.get_asset_dof_names(door_asset)
    dof_props = gym.get_asset_dof_properties(door_asset)
    if len(dof_props["upper"]) >= 2:
        dof_props["upper"][1] = min(float(dof_props["upper"][1]), math.pi / 4)
    lower = np.asarray(dof_props["lower"], dtype=np.float32)
    upper = np.asarray(dof_props["upper"], dtype=np.float32)

    shape_props = gym.get_asset_rigid_shape_properties(door_asset)
    for prop in shape_props:
        prop.friction = 2.0
    gym.set_asset_rigid_shape_properties(door_asset, shape_props)

    handle_goal_offset = args.door_actor_scale * np.asarray(handle_bounding["goal_pos"], dtype=np.float32)
    handle_range = max(1.0e-6, float(upper[1] - lower[1]) if len(upper) >= 2 else 1.0)
    handle_unlock_threshold = args.handle_unlock_ratio * handle_range
    print("door_dofs:", dof_names)
    print("door_bodies:", body_names)

    return DoorRuntime(
        asset_root=asset_root,
        asset_file_door=asset_file_door,
        spec=spec,
        bounding=bounding,
        handle_bounding=handle_bounding,
        asset=door_asset,
        body_names=body_names,
        dof_names=dof_names,
        dof_lower=lower,
        dof_upper=upper,
        handle_body_index=len(body_names) - 1,
        handle_goal_offset=handle_goal_offset,
        handle_unlock_threshold=handle_unlock_threshold,
    )


def robot_y_for_door(args, handle_bounding):
    handle_center_y = 0.5 * (
        float(handle_bounding["handle_min"][1]) + float(handle_bounding["handle_max"][1])
    )
    handle_center_world_y = args.door_y - args.door_actor_scale * handle_center_y
    return args.robot_y + handle_center_world_y


def create_env_actors(gym, sim, base_asset, arm_asset, door, dof_props, dof_states, args):
    env = gym.create_env(sim, gymapi.Vec3(-2.5, -2.5, 0.0), gymapi.Vec3(2.5, 2.5, 2.5), 1)
    robot_y = robot_y_for_door(args, door.handle_bounding)

    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(args.robot_x, robot_y, args.robot_z)
    robot_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, args.robot_yaw)

    collision_filter = 1 if args.disable_self_collisions else 0
    actor_handles = []
    if base_asset is not None:
        base_actor = gym.create_actor(env, base_asset, robot_pose, "b1_base_visual", 0, collision_filter)
        actor_handles.append(base_actor)
    arm_actor = gym.create_actor(env, arm_asset, robot_pose, base_ik.ARM_ACTOR_NAME, 0, collision_filter)
    actor_handles.append(arm_actor)
    gym.set_actor_dof_properties(env, arm_actor, dof_props)
    gym.set_actor_dof_states(env, arm_actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, arm_actor, dof_states["pos"])

    door_pose = gymapi.Transform()
    door_pose.p = gymapi.Vec3(
        float(args.door_x),
        float(args.door_y),
        float(-door.bounding["min"][2] * args.door_actor_scale + args.door_z_offset),
    )
    door_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
    door_actor = gym.create_actor(env, door.asset, door_pose, "door", 0, 0, 1)
    if abs(args.door_actor_scale - 1.0) > 1.0e-6:
        gym.set_actor_scale(env, door_actor, args.door_actor_scale)

    door_dof_props = gym.get_actor_dof_properties(env, door_actor)
    if len(door_dof_props) > 0:
        door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        n = min(len(door_dof_props), 2)
        if n >= 1:
            door_dof_props["damping"][0] = args.door_joint_damping
            door_dof_props["friction"][0] = args.door_joint_friction
        if n >= 2:
            door_dof_props["damping"][1] = args.handle_joint_damping
            door_dof_props["friction"][1] = args.handle_joint_friction
            door_dof_props["upper"][1] = min(float(door_dof_props["upper"][1]), math.pi / 4)
        gym.set_actor_dof_properties(env, door_actor, door_dof_props)

    return env, arm_actor, actor_handles, door_actor, np.array([args.robot_x, robot_y], dtype=np.float32)


def set_robot_base_pose(gym, env, actor_handles, xy, z, yaw):
    quat = base_ik.yaw_quat(yaw)
    for actor in actor_handles:
        root_handle = gym.get_actor_root_rigid_body_handle(env, actor)
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(float(xy[0]), float(xy[1]), float(z))
        transform.r = gymapi.Quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gym.set_rigid_transform(env, root_handle, transform)


def world_to_base_local(point, xy, z, yaw):
    origin = np.array([float(xy[0]), float(xy[1]), float(z)], dtype=np.float32)
    rel = np.asarray(point, dtype=np.float32) - origin
    return quat_apply(base_ik.quat_conjugate(base_ik.yaw_quat(float(yaw))), rel)


def base_local_to_world(local_point, xy, z, yaw):
    origin = np.array([float(xy[0]), float(xy[1]), float(z)], dtype=np.float32)
    return origin + quat_apply(base_ik.yaw_quat(float(yaw)), np.asarray(local_point, dtype=np.float32))


def default_ee_world_target(args, traj, base_xy, yaw, fallback):
    home_local = traj.get("home_ee_local")
    if home_local is None:
        return np.asarray(fallback, dtype=np.float32).copy()
    return base_local_to_world(home_local, base_xy, args.robot_z, yaw)


def compute_base_pass_target(args, heading):
    door_xy = np.asarray([args.door_x, args.door_y], dtype=np.float32)
    return door_xy + heading * (float(args.robot_rear_offset) + float(args.door_pass_clearance))


def get_body_pose(gym, env, actor, body_index):
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    pos_raw = states["pose"]["p"][body_index]
    quat_raw = states["pose"]["r"][body_index]
    pos = np.array([pos_raw["x"], pos_raw["y"], pos_raw["z"]], dtype=np.float32)
    quat = np.array([quat_raw["x"], quat_raw["y"], quat_raw["z"], quat_raw["w"]], dtype=np.float32)
    return pos, base_ik.normalize_quat(quat)


def get_actor_dof_state(gym, env, actor):
    states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    return np.asarray(states["pos"], dtype=np.float32), np.asarray(states["vel"], dtype=np.float32)


def door_open_angle_from_closed(door, door_angle):
    closed_angle = float(door.dof_lower[0]) if len(door.dof_lower) > 0 else 0.0
    return abs(float(door_angle) - closed_angle)


def live_handle_pull_direction(handle_quat, approach_dir):
    pull_dir = quat_axis(handle_quat, axis=2)
    pull_dir[2] = 0.0
    pull_dir = normalize(pull_dir)
    if np.linalg.norm(pull_dir) < 1.0e-5:
        pull_dir = approach_dir.copy()
    if float(np.dot(pull_dir, approach_dir)) < 0.0:
        pull_dir = -pull_dir
    return pull_dir


def compute_door_efforts(door, dof_pos, dof_vel, args):
    efforts = np.zeros(len(dof_pos), dtype=np.float32)
    if len(dof_pos) < 2:
        return efforts

    door_angle = float(dof_pos[0])
    handle_angle_from_lower = float(dof_pos[1] - door.dof_lower[1])
    if handle_angle_from_lower >= door.handle_unlock_threshold:
        door.open_stage = True

    hinge_range = max(abs(float(door.dof_upper[0])), abs(float(door.dof_lower[0])), 1.0e-3)
    if abs(door_angle) < args.door_auto_open_target_ratio * hinge_range:
        auto_torque = args.door_auto_open_force * args.door_auto_open_sign
    else:
        auto_torque = 0.0

    if door.open_stage:
        efforts[0] = auto_torque - args.door_open_resistance * door_angle - args.door_open_damping * float(dof_vel[0])
    elif args.door_lock_force > 0.0:
        efforts[0] = -args.door_lock_force

    efforts[1] = -args.handle_spring_stiffness * handle_angle_from_lower - args.handle_spring_damping * float(dof_vel[1])
    return efforts


def enforce_locked_door_hinge(gym, env, door_actor, door):
    if door.open_stage:
        return
    states = gym.get_actor_dof_states(env, door_actor, gymapi.STATE_ALL)
    if len(states) == 0:
        return
    states["pos"][0] = door.dof_lower[0]
    states["vel"][0] = 0.0
    gym.set_actor_dof_states(env, door_actor, states, gymapi.STATE_ALL)


def set_ik_target(ik_state, pos, quat):
    torch = ik_state.torch
    device = ik_state.target_pos.device
    ik_state.target_pos[:] = torch.as_tensor(pos, dtype=torch.float32, device=device)
    ik_state.target_pos_np = np.asarray(pos, dtype=np.float32).copy()
    if quat is None:
        ik_state.target_quat = None
        ik_state.target_quat_np = None
        return
    quat = base_ik.normalize_quat(quat)
    if ik_state.target_quat is None:
        ik_state.target_quat = torch.as_tensor(quat, dtype=torch.float32, device=device)
    else:
        ik_state.target_quat[:] = torch.as_tensor(quat, dtype=torch.float32, device=device)
    ik_state.target_quat = ik_state.target_quat / torch.clamp(torch.linalg.norm(ik_state.target_quat), min=1.0e-8)
    ik_state.target_quat_np = quat.copy()


def update_arm_ik_targets(gym, sim, dof_positions, ik_state, args, num_arm_dofs):
    torch = ik_state.torch
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    eef_pos = eef_state[:3]
    eef_quat = eef_state[3:7]
    pos_err = ik_state.target_pos - eef_pos
    j_eef = ik_state.jacobian[0, ik_state.eef_jacobian_index, :, :]
    j_control = j_eef[:, ik_state.control_indices]

    if ik_state.target_quat is None:
        task_j = j_control[:3, :]
        task_err = args.ik_pos_gain * pos_err
        rot_err_norm = None
    else:
        orn_err = base_ik.torch_orientation_error(torch, ik_state.target_quat, eef_quat)
        dpose = torch.cat((args.ik_pos_gain * pos_err, args.ik_rot_gain * orn_err), dim=0)
        weights = torch.tensor(
            [1.0, 1.0, 1.0, args.ik_rot_weight, args.ik_rot_weight, args.ik_rot_weight],
            dtype=torch.float32,
            device=j_control.device,
        )
        task_j = j_control * weights.view(6, 1)
        task_err = dpose * weights
        rot_err_norm = float(torch.linalg.norm(orn_err).detach().cpu())

    j_t = torch.transpose(task_j, 0, 1)
    damping = max(1.0e-6, float(args.ik_damping))
    lhs = task_j @ j_t + torch.eye(task_j.shape[0], dtype=torch.float32, device=task_j.device) * (damping * damping)
    delta = j_t @ torch.linalg.solve(lhs, task_err.unsqueeze(-1)).squeeze(-1)
    delta = torch.clamp(delta, -args.ik_max_step, args.ik_max_step)

    current_q = ik_state.dof_state_tensor[:num_arm_dofs, 0].clone()
    next_q = current_q.clone()
    next_q[ik_state.control_indices] += delta
    next_q = torch.max(torch.min(next_q, ik_state.upper), ik_state.lower)
    dof_positions[:] = next_q.detach().cpu().numpy()

    ik_state.last_pos_error = float(torch.linalg.norm(pos_err).detach().cpu())
    ik_state.last_rot_error = rot_err_norm
    ik_state.current_pos_np = eef_pos.detach().cpu().numpy().copy()


def trajectory_targets(
    step,
    args,
    door,
    gym,
    env,
    door_actor,
    ik_state,
    base_start,
    base_stop,
    base_pull,
    base_pass,
    yaw_start,
    yaw_pull,
    traj,
):
    handle_pos, handle_quat = get_body_pose(gym, env, door_actor, door.handle_body_index)
    handle_goal = quat_apply(handle_quat, door.handle_goal_offset) + handle_pos
    base_xy_current = traj.get("base_xy", base_start)
    approach_dir = np.array([base_xy_current[0], base_xy_current[1], args.robot_z], dtype=np.float32) - handle_goal
    approach_dir[2] = 0.0
    approach_dir = normalize(approach_dir)
    if np.linalg.norm(approach_dir) < 1.0e-5:
        approach_dir = np.array([math.cos(yaw_start), math.sin(yaw_start), 0.0], dtype=np.float32)

    pregrasp = handle_goal + approach_dir * args.pregrasp_offset
    grasp = handle_goal + approach_dir * args.grasp_offset
    pregrasp[0] += args.grasp_x_offset
    pregrasp[2] += args.grasp_z_offset
    grasp[0] += args.grasp_x_offset
    grasp[2] += args.grasp_z_offset
    goal_quat = approach_ee_quat(args, approach_dir, yaw_start)

    rotate_offset = np.zeros(3, dtype=np.float32)
    rotate_offset[1] = args.handle_rotate_right_distance
    rotate_offset[2] = -args.handle_rotate_down_distance
    rotate_pos = grasp + rotate_offset
    pull_dir = live_handle_pull_direction(handle_quat, approach_dir)
    pull_pos = rotate_pos + pull_dir * args.door_pull_distance

    walk_end = args.walk_steps
    initial_end = walk_end + args.initial_hold_steps
    grasp_end = initial_end + args.grasp_steps
    grasp_hold_end = grasp_end + args.grasp_hold_steps
    close_end = grasp_hold_end + args.gripper_close_steps
    rotate_end = close_end + args.handle_rotate_steps
    pull_end = rotate_end + args.door_pull_steps

    gripper_closed = args.gripper_open + (args.gripper_closed - args.gripper_open) * args.gripper_close_ratio
    target_pos = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else pregrasp.copy()
    target_quat = None if args.ik_position_only else goal_quat.copy()
    gripper = args.gripper_open
    base_xy = base_start.copy()
    yaw = yaw_start
    phase = "walk"

    if step < walk_end:
        t = smoothstep((step + 1) / max(1, args.walk_steps))
        base_xy = lerp(base_start, base_stop, t)
        target_pos = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else pregrasp.copy()
        target_quat = None if args.ik_position_only else ik_state.target_quat_np
    else:
        if "pregrasp" not in traj:
            traj["pregrasp"] = pregrasp.copy()
            traj["grasp"] = grasp.copy()
            traj["rotate"] = rotate_pos.copy()
            traj["pull"] = pull_pos.copy()
            traj["goal_quat"] = goal_quat.copy()
            traj["pull_dir"] = pull_dir.copy()
            traj["approach_dir"] = approach_dir.copy()

        base_xy = base_stop.copy()
        target_pos = traj["pregrasp"].copy()
        target_quat = None if args.ik_position_only else traj["goal_quat"].copy()
        phase = "initial_hold"

        if step < initial_end:
            pass
        elif step < grasp_end:
            t = smoothstep((step - initial_end + 1) / max(1, args.grasp_steps))
            target_pos = lerp(traj["pregrasp"], traj["grasp"], t)
            phase = "grasp"
        elif step < grasp_hold_end:
            target_pos = traj["grasp"].copy()
            phase = "grasp_hold"
        elif step < close_end:
            t = smoothstep((step - grasp_hold_end + 1) / max(1, args.gripper_close_steps))
            target_pos = traj["grasp"].copy()
            gripper = args.gripper_open + (gripper_closed - args.gripper_open) * t
            phase = "close_gripper"
        elif step < rotate_end:
            t = smoothstep((step - close_end + 1) / max(1, args.handle_rotate_steps))
            target_pos = lerp(traj["grasp"], traj["rotate"], t)
            target_quat = None if args.ik_position_only else base_ik.quat_multiply(
                traj["goal_quat"],
                quat_from_angle_axis(-t * args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            gripper = gripper_closed
            phase = "rotate_handle"
        elif step < pull_end:
            t = smoothstep((step - rotate_end + 1) / max(1, args.door_pull_steps))
            base_t = smoothstep(
                (step - rotate_end + 1) / max(1.0, args.door_pull_steps * args.base_pull_time_scale)
            )
            live_goal_quat = approach_ee_quat(args, approach_dir, yaw_start)
            turned_quat = base_ik.quat_multiply(
                live_goal_quat,
                quat_from_angle_axis(-args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            door_pos, _ = get_actor_dof_state(gym, env, door_actor)
            door_open_enough = (
                len(door_pos) > 0
                and door_open_angle_from_closed(door, float(door_pos[0])) >= math.radians(args.pass_open_angle_deg)
            )
            if door_open_enough and "pass_home_start_step" not in traj and "pass_start_step" not in traj:
                traj["pass_home_start_step"] = step
                traj["pass_home_base_xy"] = traj.get("base_xy", base_pull).copy()
                traj["pass_home_yaw"] = float(traj.get("yaw", yaw_pull))

            if "pass_start_step" in traj:
                pass_t = smoothstep((step - traj["pass_start_step"] + 1) / max(1, args.pass_through_steps))
                base_xy = lerp(traj["pass_start_base_xy"], base_pass, pass_t)
                yaw = float(lerp(np.array([traj["pass_start_yaw"]], dtype=np.float32), np.array([yaw_start], dtype=np.float32), pass_t)[0])
                fallback = ik_state.current_pos_np if ik_state.current_pos_np is not None else traj["grasp"]
                target_pos = default_ee_world_target(args, traj, base_xy, yaw, fallback)
                target_quat = None
                gripper = gripper_closed
                phase = "pass_through" if pass_t < 1.0 else "hold_pass"
            elif "pass_home_start_step" in traj:
                base_xy = traj["pass_home_base_xy"].copy()
                yaw = float(traj["pass_home_yaw"])
                fallback = ik_state.current_pos_np if ik_state.current_pos_np is not None else traj["grasp"]
                target_pos = default_ee_world_target(args, traj, base_xy, yaw, fallback)
                target_quat = None
                gripper = gripper_closed
                wait_steps = step - traj["pass_home_start_step"] + 1
                ready_by_time = wait_steps >= max(1, args.pass_home_wait_min_steps)
                ready_by_error = ik_state.last_pos_error <= float(args.pass_home_ready_tolerance)
                timed_out = wait_steps >= max(1, args.pass_home_wait_max_steps)
                if (ready_by_time and ready_by_error) or timed_out:
                    traj["pass_start_step"] = step + 1
                    traj["pass_start_base_xy"] = base_xy.copy()
                    traj["pass_start_yaw"] = float(yaw)
                phase = "return_home"
            elif door.open_stage and ik_state.current_pos_np is not None:
                live_pull_dir = live_handle_pull_direction(handle_quat, traj["approach_dir"])
                target_pos = ik_state.current_pos_np + live_pull_dir * args.lever_step_size
                target_quat = None if args.ik_position_only else turned_quat
                base_xy = lerp(base_stop, base_pull, base_t)
                yaw = float(lerp(np.array([yaw_start], dtype=np.float32), np.array([yaw_pull], dtype=np.float32), base_t)[0])
                gripper = gripper_closed
                phase = "pull_door"
            elif args.unidoor_style_pull and ik_state.current_pos_np is not None:
                pull_step_pos = ik_state.current_pos_np + traj["pull_dir"] * args.lever_step_size
                pull_max_pos = traj["rotate"] + traj["pull_dir"] * args.door_pull_distance
                progress = float(np.dot(pull_step_pos - traj["rotate"], traj["pull_dir"]))
                target_pos = pull_max_pos if progress > args.door_pull_distance else pull_step_pos
                target_quat = None if args.ik_position_only else turned_quat
                base_xy = lerp(base_stop, base_pull, base_t)
                yaw = float(lerp(np.array([yaw_start], dtype=np.float32), np.array([yaw_pull], dtype=np.float32), base_t)[0])
                gripper = gripper_closed
                phase = "pull_door"
            else:
                target_pos = lerp(traj["rotate"], traj["pull"], t)
                target_quat = None if args.ik_position_only else turned_quat
                base_xy = lerp(base_stop, base_pull, base_t)
                yaw = float(lerp(np.array([yaw_start], dtype=np.float32), np.array([yaw_pull], dtype=np.float32), base_t)[0])
                gripper = gripper_closed
                phase = "pull_door"
        else:
            door_pos, _ = get_actor_dof_state(gym, env, door_actor)
            door_open_enough = (
                len(door_pos) > 0
                and door_open_angle_from_closed(door, float(door_pos[0])) >= math.radians(args.pass_open_angle_deg)
            )
            if door_open_enough and "pass_home_start_step" not in traj and "pass_start_step" not in traj:
                traj["pass_home_start_step"] = step
                traj["pass_home_base_xy"] = traj.get("base_xy", base_pull).copy()
                traj["pass_home_yaw"] = float(traj.get("yaw", yaw_pull))

            if "pass_start_step" in traj:
                pass_t = smoothstep((step - traj["pass_start_step"] + 1) / max(1, args.pass_through_steps))
                base_xy = lerp(traj["pass_start_base_xy"], base_pass, pass_t)
                yaw = float(lerp(np.array([traj["pass_start_yaw"]], dtype=np.float32), np.array([yaw_start], dtype=np.float32), pass_t)[0])
                fallback = ik_state.current_pos_np if ik_state.current_pos_np is not None else traj["pull"]
                target_pos = default_ee_world_target(args, traj, base_xy, yaw, fallback)
                target_quat = None
                gripper = gripper_closed
                phase = "pass_through" if pass_t < 1.0 else "hold_pass"
            elif "pass_home_start_step" in traj:
                base_xy = traj["pass_home_base_xy"].copy()
                yaw = float(traj["pass_home_yaw"])
                fallback = ik_state.current_pos_np if ik_state.current_pos_np is not None else traj["pull"]
                target_pos = default_ee_world_target(args, traj, base_xy, yaw, fallback)
                target_quat = None
                gripper = gripper_closed
                wait_steps = step - traj["pass_home_start_step"] + 1
                ready_by_time = wait_steps >= max(1, args.pass_home_wait_min_steps)
                ready_by_error = ik_state.last_pos_error <= float(args.pass_home_ready_tolerance)
                timed_out = wait_steps >= max(1, args.pass_home_wait_max_steps)
                if (ready_by_time and ready_by_error) or timed_out:
                    traj["pass_start_step"] = step + 1
                    traj["pass_start_base_xy"] = base_xy.copy()
                    traj["pass_start_yaw"] = float(yaw)
                phase = "return_home"
            else:
                target_pos = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else traj["pull"].copy()
                target_quat = None if args.ik_position_only else base_ik.quat_multiply(
                    traj["goal_quat"],
                    quat_from_angle_axis(-args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                )
                base_xy = base_pull.copy()
                yaw = yaw_pull
                gripper = gripper_closed
                phase = "hold"

    traj["base_xy"] = base_xy.copy()
    traj["yaw"] = float(yaw)
    return phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal


def setup_viewer(gym, sim, args):
    viewer = base_ik.setup_viewer(gym, sim, args)
    if viewer is not None:
        gym.viewer_camera_look_at(
            viewer,
            None,
            gymapi.Vec3(4.4, -3.2, 1.8),
            gymapi.Vec3(2.8, 0.0, 0.8),
        )
    return viewer


def run_demo(gym, sim, env, arm_actor, actor_handles, door, door_actor, viewer, args, dt, dof_names, dof_positions, ik_state):
    num_arm_dofs = len(dof_positions)
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    gripper_idx = dof_dict.get("jointGripper")
    if gripper_idx is not None:
        dof_positions[gripper_idx] = args.gripper_open

    yaw_start = float(args.robot_yaw)
    heading = np.array([math.cos(yaw_start), math.sin(yaw_start)], dtype=np.float32)
    base_start = np.asarray([args.robot_x, robot_y_for_door(args, door.handle_bounding)], dtype=np.float32)
    robot_front = base_start + heading * args.robot_front_offset
    door_xy = np.asarray([args.door_x, args.door_y], dtype=np.float32)
    front_to_door = float(np.dot(door_xy - robot_front, heading))
    walk_dist = max(0.0, front_to_door - args.stop_distance)
    base_stop = base_start + heading * walk_dist
    base_pull = base_stop + heading * args.pull_base_distance
    base_pass = compute_base_pass_target(args, heading)
    yaw_pull = yaw_start + args.pull_base_yaw_delta
    initial_ee = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else np.array(
        [base_start[0], base_start[1], args.robot_z],
        dtype=np.float32,
    )
    home_ee_local = world_to_base_local(initial_ee, base_start, args.robot_z, yaw_start)
    traj = {"base_xy": base_start.copy(), "home_ee_local": home_ee_local.copy()}

    print(
        "base_start:",
        base_start.tolist(),
        "base_stop:",
        base_stop.tolist(),
        "base_pull:",
        base_pull.tolist(),
        "base_pass:",
        base_pass.tolist(),
    )
    print(
        "pass_open_angle_deg:",
        float(args.pass_open_angle_deg),
        "base_pull_time_scale:",
        float(args.base_pull_time_scale),
        "door_pass_clearance:",
        float(args.door_pass_clearance),
    )
    print(
        "pass_home_wait:",
        int(args.pass_home_wait_min_steps),
        "to",
        int(args.pass_home_wait_max_steps),
        "steps, tolerance:",
        float(args.pass_home_ready_tolerance),
    )
    print("home_ee_local_from_default_joints:", home_ee_local.tolist())
    print("Close viewer to exit.")
    start = time.time()
    step = 0
    max_steps = args.steps if args.steps > 0 else 3600
    while step < max_steps:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal = trajectory_targets(
            step,
            args,
            door,
            gym,
            env,
            door_actor,
            ik_state,
            base_start,
            base_stop,
            base_pull,
            base_pass,
            yaw_start,
            yaw_pull,
            traj,
        )
        set_robot_base_pose(gym, env, actor_handles, base_xy, args.robot_z, yaw)
        set_ik_target(ik_state, target_pos, target_quat)
        update_arm_ik_targets(gym, sim, dof_positions, ik_state, args, num_arm_dofs)
        if gripper_idx is not None:
            dof_positions[gripper_idx] = np.clip(gripper, ik_state.lower[gripper_idx].item(), ik_state.upper[gripper_idx].item())
        gym.set_actor_dof_position_targets(env, arm_actor, dof_positions)

        enforce_locked_door_hinge(gym, env, door_actor, door)
        door_pos, door_vel = get_actor_dof_state(gym, env, door_actor)
        door_efforts = compute_door_efforts(door, door_pos, door_vel, args)
        if len(door_efforts) > 0:
            gym.apply_actor_dof_efforts(env, door_actor, door_efforts)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if viewer is not None:
            if args.draw_ik_target:
                gym.clear_lines(viewer)
                base_ik.draw_ik_target(gym, viewer, env, ik_state)
                target_pose = base_ik.transform_from_arrays(handle_goal)
                goal_sphere = gymutil.WireframeSphereGeometry(
                    radius=0.035,
                    num_lats=8,
                    num_lons=8,
                    color=(0.0, 1.0, 0.2),
                    color2=(0.0, 0.7, 0.2),
                )
                gymutil.draw_lines(goal_sphere, gym, viewer, env, target_pose)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"[{step:04d}] phase={phase:14s} "
                f"ik_pos_err={ik_state.last_pos_error:.4f} "
                f"door={math.degrees(float(door_pos[0])) if len(door_pos) else 0.0:.1f}deg "
                f"handle={math.degrees(float(door_pos[1])) if len(door_pos) > 1 else 0.0:.1f}deg "
                f"open_stage={door.open_stage}",
                flush=True,
            )
        step += 1

    print(f"Done after {step} steps ({time.time() - start:.2f}s).")


def main():
    args = parse_args()
    gym = gymapi.acquire_gym()
    sim, dt = base_ik.create_sim(gym, args)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    with tempfile.TemporaryDirectory(prefix="b1z1_float_ik_door_assets_") as temp_dir:
        base_asset, arm_asset = base_ik.load_robot_assets(gym, sim, args, Path(temp_dir))
        door = load_door_asset(gym, sim, args)
        if base_asset is not None:
            base_ik.print_collision_summary(gym, base_asset, "base visual actor", verbose=args.print_collision_summary)
        base_ik.print_collision_summary(gym, arm_asset, "arm articulated actor", verbose=args.print_collision_summary)
        dof_data = base_ik.configure_dofs(gym, arm_asset, args)
        dof_names, dof_props, dof_states, dof_positions, lower, upper, defaults, speeds, selected = dof_data
        if "jointGripper" in dof_names:
            dof_states["pos"][dof_names.index("jointGripper")] = args.gripper_open
            dof_positions[dof_names.index("jointGripper")] = args.gripper_open
        env, arm_actor, actor_handles, door_actor, _ = create_env_actors(
            gym, sim, base_asset, arm_asset, door, dof_props, dof_states, args
        )
        ik_state = base_ik.setup_ik_controller(gym, sim, env, arm_actor, arm_asset, dof_names, lower, upper, args)
        viewer = setup_viewer(gym, sim, args)
        try:
            run_demo(
                gym,
                sim,
                env,
                arm_actor,
                actor_handles,
                door,
                door_actor,
                viewer,
                args,
                dt,
                dof_names,
                dof_positions,
                ik_state,
            )
        finally:
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
