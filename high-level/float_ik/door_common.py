#!/usr/bin/env python3
"""Shared helpers for float-base B1Z1 door IK controllers."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

try:
    import cv2
except ImportError:
    cv2 = None


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
DEFAULT_WRIST_CAMERA_CFG = {
    "horizontal_fov": 69,
    "resolution": [96, 54],
    "position": [0.0955, 0.22, -0.03175],
    "rotation": [-1.57, 0.0, -0.87],
}
DEFAULT_FRONT_CAMERA_CFG = {
    "horizontal_fov": 69,
    "resolution": [96, 54],
    "position": [0.425, 0.04, 0.12],
    "rotation": [0.0, 0.0, 0.0],
}

DP_NUM_DOFS = 19
DP_NUM_ACTIONS = 18
B1Z1_DEFAULT_DOF_POS = np.asarray(
    [
        -0.2,
        0.8,
        -1.5,
        0.2,
        0.8,
        -1.5,
        -0.2,
        0.8,
        -1.5,
        0.2,
        0.8,
        -1.5,
        0.0,
        1.48,
        -0.63,
        -0.84,
        0.0,
        1.57,
        -0.785,
    ],
    dtype=np.float32,
)
FLOAT_ARM_TO_DP_DOF = {
    "joint1": 12,
    "joint2": 13,
    "joint3": 14,
    "joint4": 15,
    "joint5": 16,
    "joint6": 17,
    "jointGripper": 18,
    "z1_waist": 12,
    "z1_shoulder": 13,
    "z1_elbow": 14,
    "z1_wrist_angle": 15,
    "z1_forearm_roll": 16,
    "z1_wrist_rotate": 17,
    "z1_jointGripper": 18,
}


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


class ThickAxesGeometry(gymutil.LineGeometry):
    def __init__(self, scale=1.0, thickness=0.006, pose=None):
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
        self.verts = pose.transform_points(verts) if pose is not None else verts
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


@dataclass
class DoorRuntime:
    asset_root: str
    asset_file_door: str
    asset_index: int
    spec: dict
    bounding: dict
    handle_bounding: dict
    asset: object
    body_names: list[str]
    dof_names: list[str]
    dof_lower: np.ndarray
    dof_upper: np.ndarray
    handle_body_index: int
    door_body_index: int
    handle_goal_offset: np.ndarray
    handle_unlock_threshold: float
    open_stage: bool = False


@dataclass
class ParallelDoorEnvState:
    index: int
    args: object
    env: object
    arm_actor: int
    actor_handles: list
    door: DoorRuntime
    door_actor: int
    camera_handles: dict
    ik_state: object
    dof_positions: np.ndarray
    home_positions: np.ndarray
    base_start: np.ndarray
    base_stop: np.ndarray
    base_goal: np.ndarray
    yaw_start: float
    yaw_goal: float
    traj: dict
    prev_base_xy: object = None
    prev_yaw: object = None
    last_phase: str = "init"
    last_handle_goal: object = None
    last_door_pos: object = None
    last_target_pos: object = None
    last_target_quat: object = None
    last_gripper: float = 0.0
    success: bool = False
    dp_recorder: object = None
    dp_record_success: bool = False
    dp_record_warned_no_camera: bool = False
    last_dp_action: object = None
    dp_action_frame: str = "base"


def sorted_asset_entries(asset_dict):
    def key_fn(item):
        key = item[0]
        return (0, int(key)) if str(key).isdigit() else (1, str(key))

    return [(idx, item[1]) for idx, item in enumerate(sorted(asset_dict.items(), key=key_fn))]


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(a, b, t):
    return a + (b - a) * float(t)


def normalize(vec, eps=1.0e-6):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec * 0.0
    return vec / norm


def quat_nlerp(a, b, t):
    qa = base_ik.normalize_quat(np.asarray(a, dtype=np.float32))
    qb = base_ik.normalize_quat(np.asarray(b, dtype=np.float32))
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    return base_ik.normalize_quat(lerp(qa, qb, t)).astype(np.float32)


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


def ee_quat_from_forward_xy(args, forward_xy, roll_about_x=0.0, fallback_quat=None):
    forward_xy = normalize(np.asarray(forward_xy, dtype=np.float32)[:2])
    if np.linalg.norm(forward_xy) < 1.0e-5:
        return None if fallback_quat is None else base_ik.normalize_quat(fallback_quat).astype(np.float32)
    yaw = math.atan2(float(forward_xy[1]), float(forward_xy[0]))
    quat = forward_ee_quat(args, yaw)
    if abs(float(roll_about_x)) > 1.0e-6:
        quat = base_ik.quat_multiply(
            quat,
            quat_from_angle_axis(float(roll_about_x), np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        )
    return base_ik.normalize_quat(quat).astype(np.float32)


def chase_target_to_current_ee(traj, ik_state, args, fallback_pos, fallback_quat=None):
    alpha = float(np.clip(getattr(args, "return_home_target_chase_alpha", 0.08), 0.0, 1.0))
    prev_pos = traj.get("last_target_pos")
    if prev_pos is None:
        prev_pos = fallback_pos
    prev_pos = np.asarray(prev_pos, dtype=np.float32)
    current_pos = (
        np.asarray(ik_state.current_pos_np, dtype=np.float32)
        if ik_state.current_pos_np is not None
        else prev_pos
    )
    target_pos = lerp(prev_pos, current_pos, alpha).astype(np.float32)

    target_quat = None
    if not args.ik_position_only:
        prev_quat = traj.get("last_target_quat")
        if prev_quat is None:
            prev_quat = fallback_quat
        current_quat = ik_state.current_quat_np if ik_state.current_quat_np is not None else prev_quat
        if prev_quat is not None and current_quat is not None:
            target_quat = quat_nlerp(prev_quat, current_quat, alpha)
        elif current_quat is not None:
            target_quat = base_ik.normalize_quat(current_quat).astype(np.float32)

    return target_pos, target_quat


def load_door_specs(args):
    cfg_path = Path(args.door_cfg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    asset_cfg = cfg["env"]["asset"]
    train_assets = asset_cfg["trainAssets"]
    load_block = asset_cfg.get("load_block") or next(iter(train_assets.keys()))
    specs = sorted_asset_entries(train_assets[load_block])

    selected_entries = []
    if args.door_name:
        selected_entries = [(idx, spec) for idx, spec in specs if spec.get("name") == args.door_name]
        if not selected_entries:
            raise RuntimeError(f"Door name {args.door_name!r} not found in {cfg_path}")
    elif args.door_index >= 0:
        if args.door_index >= len(specs):
            raise RuntimeError(f"--door_index={args.door_index} out of range for {len(specs)} doors")
        selected_entries = [specs[args.door_index]]
    else:
        for name in DEFAULT_DOOR_ASSET_NAMES:
            selected_entries.extend((idx, spec) for idx, spec in specs if spec.get("name") == name)
        if not selected_entries:
            selected_entries = specs
    if not selected_entries:
        raise RuntimeError(f"No door assets found in {cfg_path}")

    asset_root = HIGH_LEVEL_ROOT / asset_cfg["assetRoot"]
    asset_file_door = asset_cfg["assetFileDoor"]
    door_set_root = asset_root / asset_file_door
    loaded_specs = []
    for asset_index, selected in selected_entries:
        with (door_set_root / selected["bounding_box"]).open("r", encoding="utf-8") as f:
            bounding = json.load(f)
        with (door_set_root / selected["handle_bounding"]).open("r", encoding="utf-8") as f:
            handle_bounding = json.load(f)
        loaded_specs.append((int(asset_index), selected, bounding, handle_bounding))

    return str(asset_root), asset_file_door, loaded_specs


def load_door_assets(gym, sim, args):
    asset_root, asset_file_door, specs = load_door_specs(args)
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

    doors = []
    for asset_index, spec, bounding, handle_bounding in specs:
        door_file = os.path.join(asset_file_door, spec["path"])
        print(f"Loading door[{asset_index}]: root={asset_root}, file={door_file}, name={spec['name']}")
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

        doors.append(
            DoorRuntime(
                asset_root=asset_root,
                asset_file_door=asset_file_door,
                asset_index=int(asset_index),
                spec=spec,
                bounding=bounding,
                handle_bounding=handle_bounding,
                asset=door_asset,
                body_names=body_names,
                dof_names=dof_names,
                dof_lower=lower,
                dof_upper=upper,
                handle_body_index=len(body_names) - 1,
                door_body_index=max(0, len(body_names) - 2),
                handle_goal_offset=handle_goal_offset,
                handle_unlock_threshold=handle_unlock_threshold,
            )
        )
    print(f"Loaded {len(doors)} door asset(s) for env cycling.", flush=True)
    return doors


def load_door_asset(gym, sim, args):
    return load_door_assets(gym, sim, args)[0]


def clone_door_runtime(door):
    return replace(
        door,
        dof_lower=np.asarray(door.dof_lower, dtype=np.float32).copy(),
        dof_upper=np.asarray(door.dof_upper, dtype=np.float32).copy(),
        handle_goal_offset=np.asarray(door.handle_goal_offset, dtype=np.float32).copy(),
        open_stage=False,
    )


def robot_y_for_door(args, handle_bounding):
    handle_center_y = 0.5 * (
        float(handle_bounding["handle_min"][1]) + float(handle_bounding["handle_max"][1])
    )
    handle_center_world_y = args.door_y - args.door_actor_scale * handle_center_y
    return args.robot_y + handle_center_world_y


def configure_door_actor_dofs(gym, env, door_actor, door, args):
    door_dof_props = gym.get_actor_dof_properties(env, door_actor)
    if len(door_dof_props) == 0:
        return

    door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
    n = min(len(door_dof_props), 2)
    if n >= 1:
        if float(args.door_motion_sign) < 0.0:
            door_dof_props["lower"][0] = -math.pi / 2
            door_dof_props["upper"][0] = 0.0
        else:
            door_dof_props["lower"][0] = 0.0
            door_dof_props["upper"][0] = math.pi / 2
        door_dof_props["damping"][0] = args.door_joint_damping
        door_dof_props["friction"][0] = args.door_joint_friction
    if n >= 2:
        door_dof_props["damping"][1] = args.handle_joint_damping
        door_dof_props["friction"][1] = args.handle_joint_friction
        door_dof_props["upper"][1] = min(float(door_dof_props["upper"][1]), math.pi / 4)
    gym.set_actor_dof_properties(env, door_actor, door_dof_props)

    door.dof_lower = np.asarray(door_dof_props["lower"], dtype=np.float32).copy()
    door.dof_upper = np.asarray(door_dof_props["upper"], dtype=np.float32).copy()
    handle_range = max(1.0e-6, float(door.dof_upper[1] - door.dof_lower[1]) if len(door.dof_upper) >= 2 else 1.0)
    door.handle_unlock_threshold = args.handle_unlock_ratio * handle_range


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
    try:
        gym.set_rigid_body_segmentation_id(env, door_actor, door.handle_body_index, int(args.handle_seg_id))
    except AttributeError:
        print("set_rigid_body_segmentation_id is not available; handle mask will use the door actor segmentation.")

    configure_door_actor_dofs(gym, env, door_actor, door, args)
    return env, arm_actor, actor_handles, door_actor, np.array([args.robot_x, robot_y], dtype=np.float32)


def create_parallel_env_actors(
    gym,
    sim,
    base_asset,
    arm_asset,
    door,
    dof_props,
    dof_states,
    args,
    env_index,
    envs_per_row,
):
    env = gym.create_env(
        sim,
        gymapi.Vec3(-2.5, -2.5, 0.0),
        gymapi.Vec3(2.5, 2.5, 2.5),
        int(envs_per_row),
    )
    robot_y = robot_y_for_door(args, door.handle_bounding)

    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(args.robot_x, robot_y, args.robot_z)
    robot_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, args.robot_yaw)

    collision_filter = 1 if args.disable_self_collisions else 0
    actor_handles = []
    if base_asset is not None:
        base_actor = gym.create_actor(env, base_asset, robot_pose, f"b1_base_visual_{env_index}", env_index, collision_filter)
        actor_handles.append(base_actor)
    arm_actor = gym.create_actor(env, arm_asset, robot_pose, base_ik.ARM_ACTOR_NAME, env_index, collision_filter)
    actor_handles.append(arm_actor)
    gym.set_actor_dof_properties(env, arm_actor, dof_props)
    arm_dof_states = dof_states.copy()
    gym.set_actor_dof_states(env, arm_actor, arm_dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, arm_actor, arm_dof_states["pos"])

    door_pose = gymapi.Transform()
    door_pose.p = gymapi.Vec3(
        float(args.door_x),
        float(args.door_y),
        float(-door.bounding["min"][2] * args.door_actor_scale + args.door_z_offset),
    )
    door_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
    door_actor = gym.create_actor(env, door.asset, door_pose, f"door_{env_index}", env_index, 0, 1)
    if abs(args.door_actor_scale - 1.0) > 1.0e-6:
        gym.set_actor_scale(env, door_actor, args.door_actor_scale)
    try:
        gym.set_rigid_body_segmentation_id(env, door_actor, door.handle_body_index, int(args.handle_seg_id))
    except AttributeError:
        if env_index == 0:
            print("set_rigid_body_segmentation_id is not available; handle mask will use the door actor segmentation.")

    configure_door_actor_dofs(gym, env, door_actor, door, args)
    return env, arm_actor, actor_handles, door_actor, np.array([args.robot_x, robot_y], dtype=np.float32)


def resolve_seed(args):
    seed = int(getattr(args, "seed", -1))
    if seed < 0:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    seed = seed % (2**32)
    args.seed = seed
    np.random.seed(seed % (2**32 - 1))
    return seed


def seed_for_env(args, env_index):
    base_seed = int(getattr(args, "seed", 0))
    seq = np.random.SeedSequence([base_seed, int(env_index)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def sample_with_half_range(rng, center, half_range, lower=None, upper=None):
    value = float(center)
    half_range = abs(float(half_range))
    if half_range > 0.0:
        if lower is not None and value <= float(lower) + 1.0e-8:
            value += float(rng.uniform(0.0, half_range))
        elif upper is not None and value >= float(upper) - 1.0e-8:
            value -= float(rng.uniform(0.0, half_range))
        else:
            value += float(rng.uniform(-half_range, half_range))
    if lower is not None:
        value = max(float(lower), value)
    if upper is not None:
        value = min(float(upper), value)
    return value


def set_robot_base_pose(gym, env, actor_handles, xy, z, yaw):
    quat = base_ik.yaw_quat(yaw)
    for actor in actor_handles:
        root_handle = gym.get_actor_root_rigid_body_handle(env, actor)
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(float(xy[0]), float(xy[1]), float(z))
        transform.r = gymapi.Quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gym.set_rigid_transform(env, root_handle, transform)


def compute_base_walk_targets(args, door):
    yaw_start = float(args.robot_yaw)
    heading = np.array([math.cos(yaw_start), math.sin(yaw_start)], dtype=np.float32)
    base_start = np.asarray([args.robot_x, robot_y_for_door(args, door.handle_bounding)], dtype=np.float32)
    robot_front = base_start + heading * args.robot_front_offset
    door_xy = np.asarray([args.door_x, args.door_y], dtype=np.float32)
    front_to_door = float(np.dot(door_xy - robot_front, heading))
    walk_dist = max(0.0, front_to_door - args.stop_distance)
    base_stop = base_start + heading * walk_dist
    return yaw_start, heading, base_start, base_stop


def compute_base_push_target(args, base_stop, heading):
    requested_progress = max(0.0, float(args.push_base_distance))
    if getattr(args, "pass_through_door", False):
        clear_center = compute_base_pass_target(args, heading)
        pass_progress = float(np.dot(clear_center - base_stop, heading))
        requested_progress = max(requested_progress, pass_progress)
    return base_stop + heading * requested_progress


def compute_base_pull_target(args, base_stop, heading):
    return base_stop - heading * max(0.0, float(args.pull_base_distance))


def compute_base_pass_target(args, heading):
    door_xy = np.asarray([args.door_x, args.door_y], dtype=np.float32)
    return door_xy + np.asarray(heading, dtype=np.float32) * (
        float(args.robot_rear_offset) + float(args.door_pass_clearance)
    )


def handle_open_tangent_dir(gym, env, door_actor, door, handle_goal, fallback_dir, args):
    hinge_pos, _hinge_quat = get_body_pose(gym, env, door_actor, door.door_body_index)
    radial = np.asarray(handle_goal, dtype=np.float32) - np.asarray(hinge_pos, dtype=np.float32)
    radial[2] = 0.0
    if np.linalg.norm(radial) < 1.0e-5:
        return np.asarray(fallback_dir, dtype=np.float32).copy()

    tangent = np.cross(
        np.array([0.0, 0.0, float(args.door_motion_sign)], dtype=np.float32),
        radial,
    ).astype(np.float32)
    tangent[2] = 0.0
    tangent = normalize(tangent)
    if np.linalg.norm(tangent) < 1.0e-5:
        return np.asarray(fallback_dir, dtype=np.float32).copy()
    if float(np.dot(tangent, fallback_dir)) < 0.0:
        tangent = -tangent
    return tangent


def get_body_pose(gym, env, actor, body_index):
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    pos_raw = states["pose"]["p"][body_index]
    quat_raw = states["pose"]["r"][body_index]
    pos = np.array([pos_raw["x"], pos_raw["y"], pos_raw["z"]], dtype=np.float32)
    quat = np.array([quat_raw["x"], quat_raw["y"], quat_raw["z"], quat_raw["w"]], dtype=np.float32)
    return pos, base_ik.normalize_quat(quat)


def actor_body_handle_set(gym, env, actor):
    return set(actor_body_handle_name_map(gym, env, actor).keys())


def actor_body_handle_name_map(gym, env, actor):
    handle_names = {}
    try:
        body_dict = gym.get_actor_rigid_body_dict(env, actor)
    except Exception:
        body_dict = {}
    for body_name in body_dict.keys():
        try:
            handle = gym.find_actor_rigid_body_handle(env, actor, body_name)
        except Exception:
            handle = -1
        if int(handle) >= 0:
            handle_names[int(handle)] = str(body_name)
    return handle_names


def get_actor_body_index(gym, env, actor, body_name):
    try:
        body_dict = gym.get_actor_rigid_body_dict(env, actor)
    except Exception:
        return None
    body_index = body_dict.get(body_name)
    if body_index is None:
        return None
    return int(body_index)


def gym_quat_to_np(quat):
    return np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)


def contact_field(contact, name):
    if hasattr(contact, name):
        return getattr(contact, name)
    dtype = getattr(contact, "dtype", None)
    if dtype is not None and dtype.names and name in dtype.names:
        return contact[name]
    try:
        return contact[name]
    except Exception:
        return None


def point_segment_distance_2d(point, a, b):
    point = np.asarray(point, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1.0e-8:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def _cross_2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _point_on_segment_2d(point, a, b, eps=1.0e-7):
    point = np.asarray(point, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if abs(_cross_2d(point - a, b - a)) > eps:
        return False
    return (
        min(float(a[0]), float(b[0])) - eps <= float(point[0]) <= max(float(a[0]), float(b[0])) + eps
        and min(float(a[1]), float(b[1])) - eps <= float(point[1]) <= max(float(a[1]), float(b[1])) + eps
    )


def _segments_intersect_2d(a, b, c, d):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    ab = b - a
    cd = d - c
    ac = c - a
    ad = d - a
    ca = a - c
    cb = b - c
    o1 = _cross_2d(ab, ac)
    o2 = _cross_2d(ab, ad)
    o3 = _cross_2d(cd, ca)
    o4 = _cross_2d(cd, cb)
    eps = 1.0e-7
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    return (
        _point_on_segment_2d(c, a, b, eps)
        or _point_on_segment_2d(d, a, b, eps)
        or _point_on_segment_2d(a, c, d, eps)
        or _point_on_segment_2d(b, c, d, eps)
    )


def segment_rect_distance_2d(a, b, x_min, x_max, y_min, y_max):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    def inside(point):
        return x_min <= float(point[0]) <= x_max and y_min <= float(point[1]) <= y_max

    if inside(a) or inside(b):
        return 0.0

    corners = [
        np.array([x_min, y_min], dtype=np.float32),
        np.array([x_min, y_max], dtype=np.float32),
        np.array([x_max, y_max], dtype=np.float32),
        np.array([x_max, y_min], dtype=np.float32),
    ]
    rect_edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    if any(_segments_intersect_2d(a, b, edge_a, edge_b) for edge_a, edge_b in rect_edges):
        return 0.0

    distances = [point_segment_distance_2d(corner, a, b) for corner in corners]
    distances.extend(point_segment_distance_2d(a, edge_a, edge_b) for edge_a, edge_b in rect_edges)
    distances.extend(point_segment_distance_2d(b, edge_a, edge_b) for edge_a, edge_b in rect_edges)
    return float(min(distances))


def world_xy_to_base_local(point_xy, base_xy, yaw):
    rel = np.asarray(point_xy, dtype=np.float32) - np.asarray(base_xy, dtype=np.float32)
    forward = np.asarray([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)
    left = np.asarray([-math.sin(float(yaw)), math.cos(float(yaw))], dtype=np.float32)
    return np.asarray([float(np.dot(rel, forward)), float(np.dot(rel, left))], dtype=np.float32)


def rotate_xy(vec, angle):
    vec = np.asarray(vec, dtype=np.float32)
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.asarray([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float32)


def door_sweep_segment_distance_to_base(
    hinge_xy,
    closed_tip_xy,
    base_xy,
    yaw,
    door_motion_sign,
    max_open_angle_deg,
    base_front_extent,
    base_rear_extent,
    base_half_width,
    clearance=0.0,
    samples=91,
):
    hinge_xy = np.asarray(hinge_xy, dtype=np.float32)
    closed_tip_xy = np.asarray(closed_tip_xy, dtype=np.float32)
    base_xy = np.asarray(base_xy, dtype=np.float32)
    closed_vec = closed_tip_xy - hinge_xy
    max_angle = max(0.0, math.radians(float(max_open_angle_deg)))
    sample_count = max(2, int(samples))
    min_distance = float("inf")
    worst_angle = 0.0
    for angle in np.linspace(0.0, max_angle, sample_count):
        tip_xy = hinge_xy + rotate_xy(closed_vec, float(door_motion_sign) * float(angle))
        hinge_local = world_xy_to_base_local(hinge_xy, base_xy, yaw)
        tip_local = world_xy_to_base_local(tip_xy, base_xy, yaw)
        distance = segment_rect_distance_2d(
            hinge_local,
            tip_local,
            -float(base_rear_extent) - float(clearance),
            float(base_front_extent) + float(clearance),
            -float(base_half_width) - float(clearance),
            float(base_half_width) + float(clearance),
        )
        if distance < min_distance:
            min_distance = float(distance)
            worst_angle = math.degrees(float(angle))
    return float(min_distance), float(worst_angle)


def door_segment_distance_to_base(
    hinge_xy,
    tip_xy,
    base_xy,
    yaw,
    base_front_extent,
    base_rear_extent,
    base_half_width,
    clearance=0.0,
):
    hinge_local = world_xy_to_base_local(hinge_xy, base_xy, yaw)
    tip_local = world_xy_to_base_local(tip_xy, base_xy, yaw)
    return segment_rect_distance_2d(
        hinge_local,
        tip_local,
        -float(base_rear_extent) - float(clearance),
        float(base_front_extent) + float(clearance),
        -float(base_half_width) - float(clearance),
        float(base_half_width) + float(clearance),
    )


def limit_base_motion_by_door_segment(
    args,
    start_xy,
    desired_xy,
    yaw,
    hinge_xy,
    tip_xy,
    clearance=None,
    samples=None,
):
    start_xy = np.asarray(start_xy, dtype=np.float32)
    desired_xy = np.asarray(desired_xy, dtype=np.float32)
    delta = desired_xy - start_xy
    if float(np.linalg.norm(delta)) < 1.0e-6:
        distance = door_segment_distance_to_base(
            hinge_xy,
            tip_xy,
            start_xy,
            yaw,
            float(getattr(args, "base_collision_front_extent", 0.55)),
            float(getattr(args, "base_collision_rear_extent", 0.65)),
            float(getattr(args, "base_collision_half_width", 0.24)),
            clearance=float(clearance if clearance is not None else getattr(args, "base_motion_door_clearance", 0.06)),
        )
        return start_xy.copy(), {
            "limited": False,
            "motion_fraction": 1.0,
            "clearance_distance": float(distance),
            "clearance": float(clearance if clearance is not None else getattr(args, "base_motion_door_clearance", 0.06)),
        }

    motion_clearance = float(
        clearance if clearance is not None else getattr(args, "base_motion_door_clearance", 0.06)
    )
    sample_count = max(2, int(samples if samples is not None else getattr(args, "base_motion_limit_samples", 41)))
    base_front = float(getattr(args, "base_collision_front_extent", 0.55))
    base_rear = float(getattr(args, "base_collision_rear_extent", 0.65))
    base_half_width = float(getattr(args, "base_collision_half_width", 0.24))

    def distance_at(ratio):
        candidate = start_xy + delta * float(ratio)
        return door_segment_distance_to_base(
            hinge_xy,
            tip_xy,
            candidate,
            yaw,
            base_front,
            base_rear,
            base_half_width,
            clearance=motion_clearance,
        )

    start_distance = distance_at(0.0)
    if start_distance <= 0.0:
        best_ratio = 0.0
        best_distance = start_distance
        first_safe = None
        for i in range(1, sample_count + 1):
            ratio = float(i) / float(sample_count)
            distance = distance_at(ratio)
            if distance > best_distance:
                best_ratio = ratio
                best_distance = distance
            if distance > 0.0:
                first_safe = ratio
                break
        if first_safe is not None:
            low = 0.0
            high = first_safe
            for _ in range(16):
                mid = 0.5 * (low + high)
                if distance_at(mid) > 0.0:
                    high = mid
                else:
                    low = mid
            limited_xy = start_xy + delta * float(high)
            return limited_xy.astype(np.float32), {
                "limited": True,
                "motion_fraction": float(high),
                "clearance_distance": float(distance_at(high)),
                "clearance": motion_clearance,
            }
        if best_ratio > 0.0 and best_distance > start_distance + 1.0e-5:
            limited_xy = start_xy + delta * float(best_ratio)
            return limited_xy.astype(np.float32), {
                "limited": True,
                "motion_fraction": float(best_ratio),
                "clearance_distance": float(best_distance),
                "clearance": motion_clearance,
            }
        return start_xy.copy(), {
            "limited": True,
            "motion_fraction": 0.0,
            "clearance_distance": float(start_distance),
            "clearance": motion_clearance,
        }

    best_ratio = 0.0
    best_distance = start_distance
    first_unsafe = None
    for i in range(1, sample_count + 1):
        ratio = float(i) / float(sample_count)
        distance = distance_at(ratio)
        if distance > 0.0:
            best_ratio = ratio
            best_distance = distance
            continue
        first_unsafe = ratio
        break

    if first_unsafe is None:
        return desired_xy.copy(), {
            "limited": False,
            "motion_fraction": 1.0,
            "clearance_distance": float(best_distance),
            "clearance": motion_clearance,
        }

    low = best_ratio
    high = first_unsafe
    for _ in range(16):
        mid = 0.5 * (low + high)
        if distance_at(mid) > 0.0:
            low = mid
        else:
            high = mid
    limited_xy = start_xy + delta * float(low)
    return limited_xy.astype(np.float32), {
        "limited": True,
        "motion_fraction": float(low),
        "clearance_distance": float(distance_at(low)),
        "clearance": motion_clearance,
    }


def base_path_progress_ratio(start_xy, goal_xy, point_xy):
    start_xy = np.asarray(start_xy, dtype=np.float32)
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    point_xy = np.asarray(point_xy, dtype=np.float32)
    delta = goal_xy - start_xy
    denom = float(np.dot(delta, delta))
    if denom < 1.0e-8:
        return 1.0
    return float(np.clip(np.dot(point_xy - start_xy, delta) / denom, 0.0, 1.0))


def clamp_world_pos_to_base_box(pos_world, base_xy, base_z, yaw, min_xyz, max_xyz):
    local = world_pos_to_base(pos_world, base_xy, base_z, yaw)
    clamped = np.minimum(np.maximum(local, np.asarray(min_xyz, dtype=np.float32)), np.asarray(max_xyz, dtype=np.float32))
    return base_pos_to_world(clamped, base_xy, base_z, yaw)


def compute_safe_base_retreat_for_door_sweep(
    args,
    base_stop,
    heading,
    yaw,
    hinge_xy,
    closed_tip_xy,
    max_open_angle_deg,
):
    heading = normalize(np.asarray(heading, dtype=np.float32))
    base_stop = np.asarray(base_stop, dtype=np.float32)
    clearance = float(getattr(args, "door_sweep_clearance", 0.06))
    extra = max(0.0, float(getattr(args, "safe_retreat_extra", 0.04)))
    search_max = max(0.05, float(getattr(args, "safe_retreat_search_max", 1.5)))
    samples = int(getattr(args, "door_sweep_samples", 91))
    min_distance = max(0.0, float(getattr(args, "pull_base_distance", 0.0)))

    def check(distance):
        candidate = base_stop - heading * float(distance)
        sweep_distance, worst_angle = door_sweep_segment_distance_to_base(
            hinge_xy=hinge_xy,
            closed_tip_xy=closed_tip_xy,
            base_xy=candidate,
            yaw=yaw,
            door_motion_sign=float(args.door_motion_sign),
            max_open_angle_deg=max_open_angle_deg,
            base_front_extent=float(getattr(args, "base_collision_front_extent", 0.55)),
            base_rear_extent=float(getattr(args, "base_collision_rear_extent", 0.65)),
            base_half_width=float(getattr(args, "base_collision_half_width", 0.24)),
            clearance=clearance,
            samples=samples,
        )
        return sweep_distance > 0.0, sweep_distance, worst_angle

    safe, sweep_distance, worst_angle = check(min_distance)
    if safe:
        safe_distance = min_distance
    else:
        low = min_distance
        high = min_distance
        safe_high = False
        while high < search_max:
            high = min(search_max, max(high + 0.05, high * 1.35 + 0.02))
            safe_high, sweep_distance, worst_angle = check(high)
            if safe_high:
                break
            low = high
        if not safe_high:
            safe_distance = search_max
        else:
            for _ in range(24):
                mid = 0.5 * (low + high)
                mid_safe, _mid_distance, _mid_angle = check(mid)
                if mid_safe:
                    high = mid
                else:
                    low = mid
            safe_distance = high
            safe, sweep_distance, worst_angle = check(safe_distance)

    target_distance = min(search_max + extra, safe_distance + extra)
    target_xy = base_stop - heading * float(target_distance)
    final_sweep_distance, final_worst_angle = check(target_distance)[1:]
    return target_xy.astype(np.float32), {
        "safe_distance": float(safe_distance),
        "target_distance": float(target_distance),
        "clearance": float(clearance),
        "extra": float(extra),
        "target_open_angle_deg": float(max_open_angle_deg),
        "sweep_distance_after_extra": float(final_sweep_distance),
        "worst_angle_deg": float(final_worst_angle),
        "search_max": float(search_max),
    }


def monitor_command_jumps(gym, step, st, base_xy, yaw, target_pos, phase):
    _ = gym
    args = st.args
    if not bool(getattr(args, "enable_command_jump_check", True)):
        st.traj["_prev_cmd_base_xy"] = np.asarray(base_xy, dtype=np.float32).copy()
        st.traj["_prev_cmd_yaw"] = float(yaw)
        st.traj["_prev_cmd_target_pos"] = (
            None if target_pos is None else np.asarray(target_pos, dtype=np.float32).copy()
        )
        st.traj["_prev_cmd_phase"] = phase
        return False

    prev_base = st.traj.get("_prev_cmd_base_xy")
    prev_yaw = st.traj.get("_prev_cmd_yaw")
    prev_target = st.traj.get("_prev_cmd_target_pos")
    prev_phase = st.traj.get("_prev_cmd_phase")
    base_xy = np.asarray(base_xy, dtype=np.float32)
    target = None if target_pos is None else np.asarray(target_pos, dtype=np.float32)
    base_delta = 0.0 if prev_base is None else float(np.linalg.norm(base_xy - np.asarray(prev_base, dtype=np.float32)))
    yaw_delta = 0.0 if prev_yaw is None else abs(float(yaw) - float(prev_yaw))
    target_delta = 0.0
    if prev_target is not None and target is not None:
        target_delta = float(np.linalg.norm(target - np.asarray(prev_target, dtype=np.float32)))

    base_threshold = float(getattr(args, "base_command_jump_distance", 0.08))
    target_threshold = float(getattr(args, "target_command_jump_distance", 0.18))
    yaw_threshold = float(getattr(args, "yaw_command_jump_distance", 0.08))
    jumped = (
        prev_base is not None
        and (
            base_delta > base_threshold
            or target_delta > target_threshold
            or yaw_delta > yaw_threshold
        )
    )
    if jumped:
        st.command_jump_detected = True
        interval = max(1, int(getattr(args, "command_jump_log_interval", 20)))
        if int(step) - int(getattr(st, "command_jump_log_step", -10**9)) >= interval:
            st.command_jump_log_step = int(step)
            print(
                "[CommandJump]"
                f" step={int(step)} env={int(st.index)} phase={phase} prev_phase={prev_phase}"
                f" base_delta={base_delta:.4f} yaw_delta={yaw_delta:.4f}"
                f" target_delta={target_delta:.4f}"
                f" base_xy={np.round(base_xy, 4).tolist()}"
                f" prev_base={np.round(prev_base, 4).tolist() if prev_base is not None else None}"
                f" target={np.round(target, 4).tolist() if target is not None else None}",
                flush=True,
            )

    st.traj["_prev_cmd_base_xy"] = base_xy.copy()
    st.traj["_prev_cmd_yaw"] = float(yaw)
    st.traj["_prev_cmd_target_pos"] = None if target is None else target.copy()
    st.traj["_prev_cmd_phase"] = phase
    return bool(jumped)


def monitor_base_door_collision(gym, step, st):
    args = st.args
    if not bool(getattr(args, "enable_base_door_collision_check", True)):
        return False
    if not hasattr(st, "base_body_handles"):
        base_actor = st.actor_handles[0] if len(st.actor_handles) > 1 else st.arm_actor
        st.base_body_names_by_handle = actor_body_handle_name_map(gym, st.env, base_actor)
        st.door_body_names_by_handle = actor_body_handle_name_map(gym, st.env, st.door_actor)
        st.base_body_handles = set(st.base_body_names_by_handle.keys())
        st.door_body_handles = set(st.door_body_names_by_handle.keys())
        st.base_door_collision_detected = False
        st.base_door_collision_log_step = -10**9

    rigid_contact = False
    frame_contact = False
    contact_pair = None
    try:
        contacts = gym.get_env_rigid_contacts(st.env)
    except Exception:
        contacts = []
    for contact in contacts:
        body0 = contact_field(contact, "body0")
        body1 = contact_field(contact, "body1")
        if body0 is None or body1 is None:
            continue
        body0 = int(body0)
        body1 = int(body1)
        if (
            body0 in st.base_body_handles
            and body1 in st.door_body_handles
            or body1 in st.base_body_handles
            and body0 in st.door_body_handles
        ):
            rigid_contact = True
            if body0 in st.base_body_handles:
                base_body = st.base_body_names_by_handle.get(body0, str(body0))
                door_body = st.door_body_names_by_handle.get(body1, str(body1))
            else:
                base_body = st.base_body_names_by_handle.get(body1, str(body1))
                door_body = st.door_body_names_by_handle.get(body0, str(body0))
            frame_contact = str(door_body) == "base"
            contact_pair = {
                "base_body": str(base_body),
                "door_body": str(door_body),
                "min_dist": contact_field(contact, "min_dist"),
                "initial_overlap": contact_field(contact, "initial_overlap"),
            }
            break

    base_xy = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32)
    yaw = float(st.traj.get("yaw", st.yaw_start))
    hinge_pos, _hinge_quat = get_body_pose(gym, st.env, st.door_actor, st.door.door_body_index)
    handle_pos, handle_quat = get_body_pose(gym, st.env, st.door_actor, st.door.handle_body_index)
    handle_goal = quat_apply(handle_quat, st.door.handle_goal_offset) + handle_pos
    hinge_local = world_xy_to_base_local(hinge_pos[:2], base_xy, yaw)
    handle_local = world_xy_to_base_local(handle_goal[:2], base_xy, yaw)
    geom_distance = segment_rect_distance_2d(
        hinge_local,
        handle_local,
        -float(getattr(args, "base_collision_rear_extent", 0.65)),
        float(getattr(args, "base_collision_front_extent", 0.55)),
        -float(getattr(args, "base_collision_half_width", 0.24)),
        float(getattr(args, "base_collision_half_width", 0.24)),
    )
    threshold = float(getattr(args, "base_door_collision_distance", 0.04))
    geom_collision = geom_distance <= threshold
    rigid_gate = float(getattr(args, "rigid_contact_geom_gate", max(0.15, threshold)))
    gated_rigid_contact = bool(rigid_contact and geom_distance <= rigid_gate)
    collision = bool(frame_contact or gated_rigid_contact or geom_collision)
    if collision:
        st.base_door_collision_detected = True
        interval = max(1, int(getattr(args, "collision_log_interval", 30)))
        if int(step) - int(getattr(st, "base_door_collision_log_step", -10**9)) >= interval:
            st.base_door_collision_log_step = int(step)
            print(
                "[BaseDoorCollision]"
                f" step={int(step)} env={int(st.index)} phase={getattr(st, 'last_phase', 'unknown')}"
                f" rigid_contact={bool(rigid_contact)} geom_distance={geom_distance:.4f}"
                f" threshold={threshold:.4f}"
                f" contact_pair={contact_pair}"
                f" base_xy={np.round(base_xy, 4).tolist()}"
                f" door_local=({np.round(hinge_local, 4).tolist()}, {np.round(handle_local, 4).tolist()})"
                f" open_deg={door_open_degrees(getattr(st, 'last_door_pos', None), args):.1f}",
                flush=True,
            )
    elif rigid_contact:
        interval = max(1, int(getattr(args, "collision_log_interval", 30)))
        if int(step) - int(getattr(st, "base_door_contact_filtered_log_step", -10**9)) >= interval:
            st.base_door_contact_filtered_log_step = int(step)
            print(
                "[BaseDoorRigidContactFiltered]"
                f" step={int(step)} env={int(st.index)} phase={getattr(st, 'last_phase', 'unknown')}"
                f" rigid_contact=True geom_distance={geom_distance:.4f}"
                f" gate={rigid_gate:.4f}"
                f" contact_pair={contact_pair}"
                f" open_deg={door_open_degrees(getattr(st, 'last_door_pos', None), args):.1f}",
                flush=True,
            )
    return collision



def local_camera_pose_from_cfg(camera_cfg, local_rot_override=None):
    local_pos = np.asarray(camera_cfg.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    local_rot = list(camera_cfg.get("rotation", [0.0, 0.0, 0.0]))
    if local_rot_override is not None:
        local_rot = list(local_rot_override)
    local_quat = gym_quat_to_np(gymapi.Quat.from_euler_zyx(*local_rot))
    return local_pos, base_ik.normalize_quat(local_quat)


def draw_local_camera_axes(gym, viewer, env, actor, body_name, local_pos, local_quat, scale, thickness):
    body_index = get_actor_body_index(gym, env, actor, body_name)
    if body_index is None:
        return False
    body_pos, body_quat = get_body_pose(gym, env, actor, body_index)
    camera_pos = body_pos + quat_apply(body_quat, local_pos)
    camera_quat = base_ik.quat_multiply(body_quat, local_quat)
    pose = gymapi.Transform(
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Quat(float(camera_quat[0]), float(camera_quat[1]), float(camera_quat[2]), float(camera_quat[3])),
    )
    axes_geom = ThickAxesGeometry(scale=scale, thickness=thickness, pose=pose)
    gymutil.draw_lines(axes_geom, gym, viewer, env, gymapi.Transform())
    return True


def draw_low_level_camera_axes(gym, viewer, env, arm_actor, actor_handles, args):
    wrist_rot = list(DEFAULT_WRIST_CAMERA_CFG["rotation"])
    wrist_rot[2] -= float(args.wrist_camera_down_tilt)
    wrist_pos, wrist_quat = local_camera_pose_from_cfg(DEFAULT_WRIST_CAMERA_CFG, wrist_rot)
    draw_local_camera_axes(
        gym,
        viewer,
        env,
        arm_actor,
        "link06",
        wrist_pos,
        wrist_quat,
        args.camera_axis_scale,
        args.camera_axis_thickness,
    )

    front_rot = [
        math.radians(float(args.front_camera_yaw_deg)),
        math.radians(float(args.front_camera_pitch_deg)),
        math.radians(float(args.front_camera_roll_deg)),
    ]
    front_pos, front_quat = local_camera_pose_from_cfg(DEFAULT_FRONT_CAMERA_CFG, front_rot)
    base_actor = actor_handles[0] if len(actor_handles) > 1 else arm_actor
    if not draw_local_camera_axes(
        gym,
        viewer,
        env,
        base_actor,
        "trunk",
        front_pos,
        front_quat,
        args.camera_axis_scale,
        args.camera_axis_thickness,
    ):
        draw_local_camera_axes(
            gym,
            viewer,
            env,
            base_actor,
            "base",
            front_pos,
            front_quat,
            args.camera_axis_scale,
            args.camera_axis_thickness,
        )


def make_camera_properties(camera_cfg):
    props = gymapi.CameraProperties()
    props.width = int(camera_cfg.get("resolution", [96, 54])[0])
    props.height = int(camera_cfg.get("resolution", [96, 54])[1])
    if camera_cfg.get("horizontal_fov", None) is not None:
        props.horizontal_fov = float(camera_cfg["horizontal_fov"])
    return props


def attach_camera_to_actor_body(gym, env, actor, body_name, camera_cfg, local_rot_override=None):
    body_handle = gym.find_actor_rigid_body_handle(env, actor, body_name)
    if body_handle < 0:
        return None
    local_pos, local_quat = local_camera_pose_from_cfg(camera_cfg, local_rot_override)
    local_transform = gymapi.Transform(
        gymapi.Vec3(float(local_pos[0]), float(local_pos[1]), float(local_pos[2])),
        gymapi.Quat(float(local_quat[0]), float(local_quat[1]), float(local_quat[2]), float(local_quat[3])),
    )
    camera_handle = gym.create_camera_sensor(env, make_camera_properties(camera_cfg))
    if camera_handle < 0:
        return None
    gym.attach_camera_to_body(camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
    return camera_handle


def create_low_level_cameras(gym, env, arm_actor, actor_handles, args):
    cameras = {}
    if args.enable_wrist_camera:
        wrist_rot = list(DEFAULT_WRIST_CAMERA_CFG["rotation"])
        wrist_rot[2] -= float(args.wrist_camera_down_tilt)
        wrist_camera = attach_camera_to_actor_body(
            gym, env, arm_actor, "link06", DEFAULT_WRIST_CAMERA_CFG, wrist_rot
        )
        if wrist_camera is None:
            print("⚠️📷 Wrist camera sensor creation failed; wrist camera image display is disabled.", flush=True)
        else:
            cameras["wrist"] = wrist_camera
            print(f"Wrist camera sensor enabled: handle={wrist_camera}")

    if args.enable_front_camera:
        front_rot = [
            math.radians(float(args.front_camera_yaw_deg)),
            math.radians(float(args.front_camera_pitch_deg)),
            math.radians(float(args.front_camera_roll_deg)),
        ]
        base_actor = actor_handles[0] if len(actor_handles) > 1 else arm_actor
        front_camera = attach_camera_to_actor_body(
            gym, env, base_actor, "trunk", DEFAULT_FRONT_CAMERA_CFG, front_rot
        )
        if front_camera is None:
            front_camera = attach_camera_to_actor_body(
                gym, env, base_actor, "base", DEFAULT_FRONT_CAMERA_CFG, front_rot
            )
        if front_camera is None:
            print("⚠️📷 Front camera sensor creation failed; front camera image display is disabled.", flush=True)
        else:
            cameras["front"] = front_camera
            print(f"Front camera sensor enabled: handle={front_camera}")
    if args.show_camera_images:
        if cv2 is None:
            print("⚠️📷 cv2 is not available; camera image windows are disabled.", flush=True)
        elif not cameras:
            print("⚠️📷 No camera sensors were created; camera image windows are disabled.", flush=True)
        else:
            pair_names = (
                ", ".join(f"{name}_rgb/{name}_mask" for name in cameras.keys())
                if args.rgb
                else ", ".join(f"{name}_mask/{name}_full_depth" for name in cameras.keys())
            )
            print(
                "Camera image windows enabled:",
                pair_names,
            )
    return cameras


def camera_image_to_array(image, height, width):
    array = np.asarray(image)
    if array.size == height * width:
        return array.reshape(height, width)
    if array.size >= height * width:
        return array.reshape(height, -1)[:, :width]
    return array


def camera_color_to_rgb(image, height, width):
    array = np.asarray(image)
    if array.size == height * width * 4:
        array = array.reshape(height, width, 4)
    elif array.size == height * width * 3:
        array = array.reshape(height, width, 3)
    elif array.ndim == 3:
        array = array[:height, :width]
    elif array.size >= height * width:
        channels = max(1, array.size // (height * width))
        array = array.reshape(height, width, channels)
    else:
        array = camera_image_to_array(array, height, width)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        if array.size == 0 or np.max(array) <= 1.0:
            array = 255.0 * array
    return np.clip(array, 0, 255).astype(np.uint8)


def show_camera_handle_images(gym, sim, env, camera_handles, args):
    if not args.show_camera_images or not camera_handles or cv2 is None:
        return
    gym.render_all_camera_sensors(sim)
    display_scale = max(1, int(args.camera_display_scale))
    for prefix, camera_handle in camera_handles.items():
        seg_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)
        if seg_raw is None:
            continue

        camera_cfg = DEFAULT_WRIST_CAMERA_CFG if prefix == "wrist" else DEFAULT_FRONT_CAMERA_CFG
        width = int(camera_cfg.get("resolution", [96, 54])[0])
        height = int(camera_cfg.get("resolution", [96, 54])[1])
        seg_image = camera_image_to_array(seg_raw, height, width).astype(np.int32)
        handle_mask = (seg_image == int(args.handle_seg_id)).astype(np.float32)
        mask_vis = (255.0 * handle_mask).astype(np.uint8)

        if args.rgb:
            rgb_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
            if rgb_raw is None:
                continue
            rgb_image = camera_color_to_rgb(rgb_raw, height, width)
            rgb_nonzero = int(np.count_nonzero(rgb_image))
            printed = getattr(args, "_camera_image_stats_printed", set())
            if prefix not in printed:
                print(
                    f"{prefix} camera image stats: "
                    f"rgb_shape={tuple(rgb_image.shape)} "
                    f"mask_pixels={int(handle_mask.sum())} "
                    f"rgb_nonzero_pixels={rgb_nonzero}",
                    flush=True,
                )
                printed.add(prefix)
                args._camera_image_stats_printed = printed
            if rgb_nonzero == 0:
                blank_printed = getattr(args, "_camera_blank_warned", set())
                if prefix not in blank_printed:
                    print(
                        "⚠️📷 Camera render is blank: RGB image has no nonzero pixels. "
                        "Check graphics_device_id/GPU rendering if this persists.",
                        flush=True,
                    )
                    blank_printed.add(prefix)
                    args._camera_blank_warned = blank_printed

            if display_scale > 1:
                mask_vis = cv2.resize(mask_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
                rgb_image = cv2.resize(rgb_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f"{prefix.capitalize()} RGB", rgb_image[..., ::-1].copy())
            cv2.imshow(f"{prefix.capitalize()} Handle Mask", mask_vis)
            continue

        depth_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
        if depth_raw is None:
            continue
        depth_image = camera_image_to_array(depth_raw, height, width).astype(np.float32)
        depth_image = np.abs(depth_image)
        depth_image = np.nan_to_num(
            depth_image,
            nan=0.0,
            posinf=float(args.camera_depth_clip_far),
            neginf=float(args.camera_depth_clip_far),
        )
        depth_image[depth_image < float(args.camera_depth_clip_lower)] = 0.0
        depth_image = np.clip(depth_image, 0.0, float(args.camera_depth_clip_far))

        depth_vis = np.zeros_like(depth_image, dtype=np.uint8)
        valid_depth = depth_image[np.isfinite(depth_image) & (depth_image > 0.0)]
        valid_depth = valid_depth[np.isfinite(valid_depth) & (valid_depth > 0.0)]
        if valid_depth.size > 0:
            depth_scaled = (depth_image - float(args.camera_depth_clip_lower)) / max(
                float(args.camera_depth_clip_far) - float(args.camera_depth_clip_lower),
                1.0e-4,
            )
            depth_vis = (255.0 * np.clip(depth_scaled, 0.0, 1.0)).astype(np.uint8)

        printed = getattr(args, "_camera_image_stats_printed", set())
        if prefix not in printed:
            print(
                f"{prefix} camera image stats: "
                f"seg_shape={tuple(seg_image.shape)} "
                f"mask_pixels={int(handle_mask.sum())} "
                f"valid_depth_pixels={int(valid_depth.size)}",
                flush=True,
            )
            printed.add(prefix)
            args._camera_image_stats_printed = printed

        visible_printed = getattr(args, "_camera_handle_visible_printed", set())
        if prefix not in visible_printed and handle_mask.sum() > 0:
            print(
                f"{prefix} camera sees handle: "
                f"mask_pixels={int(handle_mask.sum())} "
                f"valid_depth_pixels={int(valid_depth.size)}",
                flush=True,
            )
            visible_printed.add(prefix)
            args._camera_handle_visible_printed = visible_printed
        if valid_depth.size == 0:
            blank_printed = getattr(args, "_camera_blank_warned", set())
            if prefix not in blank_printed:
                print(
                    "⚠️📷 Camera render has no valid depth pixels. "
                    "Check handle visibility, segmentation id, and graphics rendering if this persists.",
                    flush=True,
                )
                blank_printed.add(prefix)
                args._camera_blank_warned = blank_printed

        if display_scale > 1:
            mask_vis = cv2.resize(mask_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
            depth_vis = cv2.resize(
                depth_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST
            )
        cv2.imshow(f"{prefix.capitalize()} Handle Mask", mask_vis)
        cv2.imshow(f"{prefix.capitalize()} Full Depth", depth_vis)
    cv2.waitKey(1)


def mask_to_rgb(mask):
    mask_u8 = (255.0 * np.clip(mask, 0.0, 1.0)).astype(np.uint8)
    return np.repeat(mask_u8[..., None], 3, axis=-1)


def depth_to_rgb(depth_image, depth_lower, depth_far):
    depth_u8 = np.zeros_like(depth_image, dtype=np.uint8)
    valid = depth_image[np.isfinite(depth_image) & (depth_image > 0.0)]
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    if valid.size > 0:
        scaled = (depth_image - float(depth_lower)) / max(float(depth_far) - float(depth_lower), 1.0e-4)
        depth_u8 = (255.0 * np.clip(scaled, 0.0, 1.0)).astype(np.uint8)
    return np.repeat(depth_u8[..., None], 3, axis=-1), int(valid.size)


def capture_dp_camera_images(gym, sim, env, camera_handles, args):
    if not camera_handles:
        return {}
    gym.render_all_camera_sensors(sim)
    return capture_dp_camera_images_from_rendered(gym, sim, env, camera_handles, args)


def capture_dp_camera_images_from_rendered(gym, sim, env, camera_handles, args):
    if not camera_handles:
        return {}
    images = {}
    for prefix, camera_handle in camera_handles.items():
        seg_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)
        if seg_raw is None:
            continue
        camera_cfg = DEFAULT_WRIST_CAMERA_CFG if prefix == "wrist" else DEFAULT_FRONT_CAMERA_CFG
        width = int(camera_cfg.get("resolution", [96, 54])[0])
        height = int(camera_cfg.get("resolution", [96, 54])[1])
        seg_image = camera_image_to_array(seg_raw, height, width).astype(np.int32)
        handle_mask = (seg_image == int(args.handle_seg_id)).astype(np.float32)
        images[f"{prefix}_handle_mask"] = mask_to_rgb(handle_mask)

        if args.rgb:
            rgb_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
            if rgb_raw is None:
                continue
            rgb_image = camera_color_to_rgb(rgb_raw, height, width)
            images[f"{prefix}_rgb"] = rgb_image
            if args.headless and not getattr(args, f"_{prefix}_headless_rgb_checked", False):
                if int(np.count_nonzero(rgb_image)) == 0:
                    print(
                        "⚠️📷 Headless camera render is blank: RGB image has no nonzero pixels. "
                        "Check graphics_device_id/GPU rendering if this persists.",
                        flush=True,
                    )
                setattr(args, f"_{prefix}_headless_rgb_checked", True)
            continue

        depth_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
        if depth_raw is None:
            continue
        depth_image = camera_image_to_array(depth_raw, height, width).astype(np.float32)
        depth_image = np.abs(depth_image)
        depth_image = np.nan_to_num(
            depth_image,
            nan=0.0,
            posinf=float(args.camera_depth_clip_far),
            neginf=float(args.camera_depth_clip_far),
        )
        depth_image[depth_image < float(args.camera_depth_clip_lower)] = 0.0
        depth_image = np.clip(depth_image, 0.0, float(args.camera_depth_clip_far))
        depth_rgb, _valid_depth_count = depth_to_rgb(
            depth_image,
            args.camera_depth_clip_lower,
            args.camera_depth_clip_far,
        )
        images[f"{prefix}_masked_depth"] = depth_rgb
        if args.headless and not getattr(args, f"_{prefix}_headless_depth_checked", False):
            if _valid_depth_count == 0:
                print(
                    "⚠️📷 Headless camera render has no valid depth pixels. "
                    "Check graphics_device_id/GPU rendering if this persists.",
                    flush=True,
                )
            setattr(args, f"_{prefix}_headless_depth_checked", True)
    return images


def dp_image_inputs_from_cpu_cameras(camera_images, args):
    if args.rgb:
        return (
            camera_images.get("wrist_handle_mask"),
            camera_images.get("wrist_rgb"),
            camera_images.get("front_handle_mask"),
            camera_images.get("front_rgb"),
        )
    return (
        camera_images.get("wrist_handle_mask"),
        camera_images.get("wrist_masked_depth"),
        camera_images.get("front_handle_mask"),
        camera_images.get("front_masked_depth"),
    )


def get_actor_dof_state(gym, env, actor):
    states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    return np.asarray(states["pos"], dtype=np.float32), np.asarray(states["vel"], dtype=np.float32)


def wrap_to_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def current_ee_pose(gym, sim, ik_state):
    gym.refresh_rigid_body_state_tensor(sim)
    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    pos = eef_state[:3].detach().cpu().numpy().astype(np.float32).copy()
    quat = eef_state[3:7].detach().cpu().numpy().astype(np.float32).copy()
    quat = base_ik.normalize_quat(quat)
    ik_state.current_pos_np = pos.copy()
    ik_state.current_quat_np = quat.copy()
    return pos, quat


def current_ee_pose_from_refreshed_tensors(ik_state):
    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    pos = eef_state[:3].detach().cpu().numpy().astype(np.float32).copy()
    quat = eef_state[3:7].detach().cpu().numpy().astype(np.float32).copy()
    quat = base_ik.normalize_quat(quat)
    ik_state.current_pos_np = pos.copy()
    ik_state.current_quat_np = quat.copy()
    return pos, quat


def refresh_current_ee_pose(gym, sim, ik_state):
    gym.refresh_rigid_body_state_tensor(sim)
    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    ik_state.current_pos_np = eef_state[:3].detach().cpu().numpy().copy()
    ik_state.current_quat_np = eef_state[3:7].detach().cpu().numpy().copy()


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


def update_arm_ik_targets_for_env(gym, env, arm_actor, env_index, dof_positions, ik_state, args, num_arm_dofs):
    torch = ik_state.torch
    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    eef_pos = eef_state[:3]
    eef_quat = eef_state[3:7]
    pos_err = ik_state.target_pos - eef_pos

    jacobian_env_idx = int(env_index) if ik_state.jacobian.ndim >= 4 and ik_state.jacobian.shape[0] > int(env_index) else 0
    j_eef = ik_state.jacobian[jacobian_env_idx, ik_state.eef_jacobian_index, :, :]
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

    actor_states = gym.get_actor_dof_states(env, arm_actor, gymapi.STATE_ALL)
    current_q = torch.as_tensor(actor_states["pos"][:num_arm_dofs], dtype=torch.float32, device=j_control.device)
    next_q = current_q.clone()
    next_q[ik_state.control_indices] += delta
    next_q = torch.max(torch.min(next_q, ik_state.upper), ik_state.lower)
    dof_positions[:] = next_q.detach().cpu().numpy()

    ik_state.last_pos_error = float(torch.linalg.norm(pos_err).detach().cpu())
    ik_state.last_rot_error = rot_err_norm
    ik_state.current_pos_np = eef_pos.detach().cpu().numpy().copy()
    ik_state.current_quat_np = eef_quat.detach().cpu().numpy().copy()


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
    ik_state.current_quat_np = eef_quat.detach().cpu().numpy().copy()


def base_position(base_xy, base_z):
    return np.asarray([base_xy[0], base_xy[1], base_z], dtype=np.float32)


def world_pos_to_base(pos_world, base_xy, base_z, yaw):
    rel = np.asarray(pos_world, dtype=np.float32) - base_position(base_xy, base_z)
    return quat_apply(base_ik.quat_conjugate(base_ik.yaw_quat(float(yaw))), rel).astype(np.float32)


def base_pos_to_world(pos_base, base_xy, base_z, yaw):
    return (base_position(base_xy, base_z) + quat_apply(base_ik.yaw_quat(float(yaw)), pos_base)).astype(np.float32)


def world_quat_to_base(quat_world, yaw):
    return base_ik.normalize_quat(
        base_ik.quat_multiply(base_ik.quat_conjugate(base_ik.yaw_quat(float(yaw))), quat_world)
    ).astype(np.float32)


def base_quat_to_world(quat_base, yaw):
    return base_ik.normalize_quat(base_ik.quat_multiply(base_ik.yaw_quat(float(yaw)), quat_base)).astype(np.float32)


def map_float_dofs_to_dp(dof_names, dof_pos, dof_vel):
    dp_pos = np.zeros(DP_NUM_DOFS, dtype=np.float32)
    dp_vel = np.zeros(DP_NUM_DOFS, dtype=np.float32)
    for src_idx, name in enumerate(dof_names):
        dst_idx = FLOAT_ARM_TO_DP_DOF.get(name)
        if dst_idx is None:
            continue
        if src_idx < len(dof_pos):
            dp_pos[dst_idx] = float(dof_pos[src_idx])
        if src_idx < len(dof_vel):
            dp_vel[dst_idx] = float(dof_vel[src_idx])
    return dp_pos, dp_vel


def base_command_from_targets(base_xy, yaw, prev_base_xy, prev_yaw, dt):
    if prev_base_xy is None or prev_yaw is None or dt <= 0.0:
        return 0.0, 0.0
    delta_xy = (np.asarray(base_xy, dtype=np.float32) - np.asarray(prev_base_xy, dtype=np.float32)) / float(dt)
    forward = np.asarray([math.cos(float(prev_yaw)), math.sin(float(prev_yaw))], dtype=np.float32)
    vx = float(np.dot(delta_xy, forward))
    yaw_rate = wrap_to_pi(float(yaw) - float(prev_yaw)) / float(dt)
    return vx, yaw_rate


def target_quat_for_dp(target_quat, ik_state, ee_quat):
    if target_quat is not None:
        return base_ik.normalize_quat(target_quat)
    if ik_state.target_quat_np is not None:
        return base_ik.normalize_quat(ik_state.target_quat_np)
    return base_ik.normalize_quat(ee_quat)


def make_last_low_action_from_dp(last_dp_action):
    last_low_action = np.zeros(DP_NUM_ACTIONS, dtype=np.float32)
    if last_dp_action is None:
        return last_low_action
    values = np.asarray(last_dp_action, dtype=np.float32).reshape(-1)
    n = min(10, DP_NUM_ACTIONS, values.shape[0])
    last_low_action[:n] = values[:n]
    return last_low_action


def make_float_dp_state(
    dof_names,
    dof_pos,
    dof_vel,
    ee_pos,
    ee_quat,
    base_xy,
    base_z,
    yaw,
    yaw_rate,
    gripper,
    last_dp_action=None,
):
    dp_dof_pos, dp_dof_vel = map_float_dofs_to_dp(dof_names, dof_pos, dof_vel)
    base_roll_pitch = np.asarray([0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.asarray([0.0, 0.0, yaw_rate], dtype=np.float32)
    last_low_action = make_last_low_action_from_dp(last_dp_action)
    foot_contacts = np.zeros(4, dtype=np.float32)
    base_pos = np.asarray([base_xy[0], base_xy[1], base_z], dtype=np.float32)
    rel = np.asarray(ee_pos, dtype=np.float32) - base_pos
    c = math.cos(-float(yaw))
    s = math.sin(-float(yaw))
    ee_base = np.asarray([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]], dtype=np.float32)
    return np.concatenate(
        [
            base_roll_pitch,
            base_ang_vel,
            dp_dof_pos,
            dp_dof_vel,
            last_low_action,
            foot_contacts,
            ee_base,
            base_ik.normalize_quat(ee_quat).astype(np.float32),
            np.asarray([gripper], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def make_float_dp_action(vx, yaw_rate, target_pos, target_quat, gripper, base_xy, base_z, yaw):
    target_pos_base = world_pos_to_base(target_pos, base_xy, base_z, yaw)
    target_quat_base = world_quat_to_base(base_ik.normalize_quat(target_quat), yaw)
    return np.concatenate(
        [
            np.asarray([vx, yaw_rate], dtype=np.float32),
            target_pos_base.reshape(3),
            target_quat_base.reshape(4),
            np.asarray([gripper], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def apply_float_dp_action(action, base_xy, base_z, yaw, dt, action_frame="base"):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 10:
        raise ValueError(f"Door DP action must have at least 10 values, got shape {action.shape}")
    vx = float(action[0])
    yaw_rate = float(action[1])
    yaw_next = float(yaw) + yaw_rate * float(dt)
    heading = np.asarray([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)
    base_xy_next = np.asarray(base_xy, dtype=np.float32) + heading * (vx * float(dt))
    target_pos_action = np.asarray(action[2:5], dtype=np.float32).copy()
    target_quat_action = base_ik.normalize_quat(np.asarray(action[5:9], dtype=np.float32))
    action_frame = str(action_frame or "base").lower()
    if action_frame == "base":
        target_pos = base_pos_to_world(target_pos_action, base_xy, base_z, yaw)
        target_quat = base_quat_to_world(target_quat_action, yaw)
    elif action_frame == "world":
        target_pos = target_pos_action
        target_quat = target_quat_action
    else:
        raise ValueError(f"Unsupported float_ik action_frame={action_frame!r}; expected 'base' or 'world'.")
    gripper = float(action[9])
    return base_xy_next.astype(np.float32), yaw_next, target_pos, target_quat, gripper


def _round_list(value, precision=5):
    return np.round(np.asarray(value, dtype=np.float64), precision).tolist()


def make_float_dp_policy_log_record(step, st, dp_action, dp_state, ee_pos, ee_quat, door_pos, phase, action_names=None):
    action_frame = str(getattr(st, "dp_action_frame", "base"))
    return {
        "step": int(step),
        "controlled_env_id": int(st.index),
        "num_envs": int(st.args.num_envs),
        "phase_name": str(phase),
        "dp_action_names": list(action_names or []),
        "action_frame": action_frame,
        "dp_action_raw": _round_list(dp_action),
        "state": _round_list(dp_state),
        "base": {
            "xy": _round_list(st.traj.get("base_xy", st.base_start)),
            "yaw": float(st.traj.get("yaw", st.yaw_start)),
        },
        "ee": {
            "target_pos_world": _round_list(st.last_target_pos),
            "target_pos_action": _round_list(dp_action[2:5]),
            "target_quat": None if st.last_target_quat is None else _round_list(st.last_target_quat),
            "target_quat_action": _round_list(dp_action[5:9]),
            "actual_pos_world": _round_list(ee_pos),
            "actual_quat": _round_list(ee_quat),
        },
        "gripper": {"target": float(st.last_gripper)},
        "door": {"dof": _round_list(door_pos) if door_pos is not None else []},
    }


def print_float_dp_policy_log_record(record):
    action = record["dp_action_raw"]
    ee = record["ee"]
    print(
        "[DoorDP-FloatIK]"
        f" step={record['step']}"
        f" env={record['controlled_env_id']}/{record['num_envs']}"
        f" phase={record['phase_name']}"
        f" action_frame={record.get('action_frame')}"
        f" action(vx,yaw,ee,grip)=({action[0]:.3f}, {action[1]:.3f}, "
        f"[{action[2]:.3f}, {action[3]:.3f}, {action[4]:.3f}], {action[9]:.3f})"
        f" ee_target_action={ee['target_pos_action']}"
        f" ee_target_world={ee['target_pos_world']}"
        f" ee_actual={ee['actual_pos_world']}",
        flush=True,
    )


def make_float_replay_snapshot(args, door, dof_names, dof_pos, dof_vel, door_pos, door_vel, ee_pos, ee_quat, base_xy, yaw, vx, yaw_rate):
    dp_dof_pos, dp_dof_vel = map_float_dofs_to_dp(dof_names, dof_pos, dof_vel)
    root_state = np.zeros(13, dtype=np.float32)
    root_state[:3] = np.asarray([base_xy[0], base_xy[1], args.robot_z], dtype=np.float32)
    root_state[3:7] = base_ik.yaw_quat(float(yaw))
    root_state[7:10] = np.asarray([vx * math.cos(float(yaw)), vx * math.sin(float(yaw)), 0.0], dtype=np.float32)
    root_state[10:13] = np.asarray([0.0, 0.0, yaw_rate], dtype=np.float32)

    door_root_state = np.zeros(13, dtype=np.float32)
    door_root_state[:3] = np.asarray(
        [
            args.door_x,
            args.door_y,
            -float(door.bounding["min"][2]) * args.door_actor_scale + args.door_z_offset,
        ],
        dtype=np.float32,
    )
    door_root_state[3:7] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return {
        "replay_root_state": root_state,
        "replay_dof_pos": dp_dof_pos,
        "replay_dof_vel": dp_dof_vel,
        "replay_ee_pos": np.asarray(ee_pos, dtype=np.float32).copy(),
        "replay_ee_quat": base_ik.normalize_quat(ee_quat).astype(np.float32),
        "replay_door_root_state": door_root_state,
        "replay_door_dof_pos": np.asarray(door_pos, dtype=np.float32).copy(),
        "replay_door_dof_vel": np.asarray(door_vel, dtype=np.float32).copy(),
    }


def float_dp_vision_mode(args, normalize_vision_mode_fn=None):
    vision_mode = "rgb" if bool(getattr(args, "rgb", False)) else "depth"
    if normalize_vision_mode_fn is not None:
        vision_mode = normalize_vision_mode_fn(vision_mode)
    return vision_mode


def float_dp_record_env_ids(args):
    return set(range(int(args.num_envs))) if bool(getattr(args, "dp_record_all_envs", False)) else {int(args.dp_record_env_id)}


def require_float_dp_recording_deps(args, raw_recorder_cls, make_state_feature_names_fn):
    if getattr(args, "record_dp_dataset", False) and (raw_recorder_cls is None or make_state_feature_names_fn is None):
        raise RuntimeError("DP raw recording requires high-level/dp/door_dp_common.py.")


def make_float_dp_recorder(
    args,
    door,
    env_index,
    vision_mode,
    phase_names,
    mode_name,
    state_version,
    raw_recorder_cls,
    make_state_feature_names_fn,
    randomization_metadata_key=None,
    extra_metadata=None,
):
    metadata = {
        "door_asset_index": int(getattr(door, "asset_index", 0)),
        "door_asset_name": door.spec.get("name", ""),
        "door_asset_path": door.spec.get("path", ""),
        "door_cfg": str(args.door_cfg),
        "source_script": Path(sys.argv[0]).name,
        "action_frame": "base",
        "action_pose_frame": "base",
        "target_pose_frame": "base",
        "ikpush_state_version": str(state_version),
        "door_dp_mode": str(mode_name),
        "controller_mode": str(mode_name),
        "parallel_env_id": int(env_index),
        "parallel_num_envs": int(args.num_envs),
        "seed": int(getattr(args, "seed", -1)),
        "env_seed": int(getattr(args, "env_seed", -1)),
    }
    for name in (
        "robot_x",
        "robot_y",
        "robot_yaw",
        "pregrasp_offset",
        "grasp_x_offset",
        "grasp_z_offset",
        "handle_rotate_angle",
        "door_push_distance",
        "door_pull_distance",
        "door_joint_friction",
        "door_joint_damping",
        "handle_joint_friction",
        "handle_joint_damping",
        "handle_spring_stiffness",
        "handle_spring_damping",
    ):
        if hasattr(args, name):
            metadata[name] = float(getattr(args, name))
    if randomization_metadata_key:
        metadata[randomization_metadata_key] = str(getattr(args, f"{randomization_metadata_key}_json", ""))
    if extra_metadata:
        metadata.update(extra_metadata)
    return raw_recorder_cls(
        raw_root=args.dp_raw_root,
        fps=args.dp_fps,
        state_feature_names=make_state_feature_names_fn(DP_NUM_DOFS, DP_NUM_ACTIONS, phase_names),
        task=args.dp_task,
        vision_mode=vision_mode,
        metadata=metadata,
    )


def print_float_dp_recording_start(args, record_env_ids, vision_mode):
    if getattr(args, "headless", False):
        print(
            "⚠️📷 Headless DP recording needs camera rendering; if the graphics device cannot render cameras, "
            "raw frames will be skipped with a camera-unavailable warning.",
            flush=True,
        )
    print(
        f"Recording raw Door DP dataset to {args.dp_raw_root} task={args.dp_task!r} "
        f"env_ids={sorted(record_env_ids)} success_angle_deg={args.pass_open_angle_deg} "
        f"vision_mode={vision_mode}",
        flush=True,
    )


def setup_float_dp_policy_controller(
    args,
    env_states,
    door_dp_policy_controller_cls,
    door_dp_jsonl_logger_cls,
    mode_name,
    state_version,
):
    if not getattr(args, "dp_policy_checkpoint", ""):
        return None, None, None, [], set()
    if door_dp_policy_controller_cls is None:
        raise RuntimeError("DP policy execution requires high-level/dp/door_dp_common.py and diffusers.")
    dp_control_env_ids = list(range(args.num_envs)) if args.dp_control_all_envs else [int(args.dp_control_env_id)]
    dp_control_env_id_set = set(dp_control_env_ids)
    dp_controller = door_dp_policy_controller_cls(
        args.dp_policy_checkpoint,
        device=args.rl_device,
        num_inference_steps=args.dp_inference_steps,
        action_horizon=args.dp_action_horizon,
        noise_scheduler_type=args.dp_noise_scheduler_type,
    )
    expected_vision_mode = "rgb" if args.rgb else "depth"
    if getattr(dp_controller, "vision_mode", "depth") != expected_vision_mode:
        raise ValueError(
            f"DP checkpoint vision_mode={getattr(dp_controller, 'vision_mode', 'depth')!r}, "
            f"but {mode_name} play was run with {expected_vision_mode!r}."
        )
    if getattr(dp_controller, "action_frame", "world") not in ("world", "base"):
        raise ValueError(
            f"DP checkpoint action_frame={getattr(dp_controller, 'action_frame', None)!r}; "
            "expected 'world' or 'base'."
        )
    checkpoint_state_version = str(dp_controller.config.get("ikpush_state_version", "legacy"))
    if checkpoint_state_version != str(state_version):
        raise ValueError(
            f"DP checkpoint ikpush_state_version={checkpoint_state_version!r}, "
            f"but this {mode_name} play script emits {state_version!r}. "
            "Reconvert/retrain with the current float_ik state format."
        )
    checkpoint_mode = str(
        dp_controller.config.get(
            "door_dp_mode",
            dp_controller.config.get("controller_mode", ""),
        )
        or ""
    )
    if checkpoint_mode and checkpoint_mode not in (str(mode_name), "unknown", "legacy"):
        raise ValueError(f"DP checkpoint door_dp_mode={checkpoint_mode!r}, but this script is --mode {mode_name}.")
    controlled_states = [env_states[env_id] for env_id in dp_control_env_ids]
    for controlled_state in controlled_states:
        controlled_state.dp_action_frame = getattr(dp_controller, "action_frame", "world")
        if not controlled_state.camera_handles:
            raise RuntimeError(f"{mode_name} DP policy execution requires camera sensors; do not disable wrist/front cameras.")
    print(
        f"Loaded Door DP policy from {args.dp_policy_checkpoint} "
        f"action_frame={getattr(dp_controller, 'action_frame', 'world')}",
        flush=True,
    )
    if args.dp_control_all_envs:
        print(f"Door DP controls all {len(dp_control_env_ids)} envs with one batched policy.", flush=True)
    else:
        print(
            f"Door DP controls only env {args.dp_control_env_id}; other envs keep the scripted target trajectory.",
            flush=True,
        )
    dp_logger = None
    if getattr(args, "dp_log_path", ""):
        if door_dp_jsonl_logger_cls is None:
            raise RuntimeError("DP policy logging requires high-level/dp/door_dp_common.py")
        dp_logger = door_dp_jsonl_logger_cls(args.dp_log_path)
        print(f"Door DP log: {args.dp_log_path}", flush=True)
    return dp_controller, dp_logger, controlled_states[0], dp_control_env_ids, dp_control_env_id_set


def collect_float_dp_policy_actions(gym, sim, env_states, dof_names, gripper_idx, dt, dp_controller, dp_control_env_id_set, mode_name):
    dp_policy_inputs_by_env = {}
    dp_actions_by_env = {}
    if dp_controller is None:
        return dp_policy_inputs_by_env, dp_actions_by_env
    batch_env_ids = []
    batch_states = []
    batch_wrist_masks = []
    batch_wrist_seconds = []
    batch_front_masks = []
    batch_front_seconds = []
    for st in env_states:
        if st.index not in dp_control_env_id_set:
            continue
        base_xy_current = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32)
        yaw_current = float(st.traj.get("yaw", st.yaw_start))
        handle_pos, handle_quat = get_body_pose(gym, st.env, st.door_actor, st.door.handle_body_index)
        handle_goal = quat_apply(handle_quat, st.door.handle_goal_offset) + handle_pos
        ee_pos, ee_quat = current_ee_pose_from_refreshed_tensors(st.ik_state)
        dof_pos_actual, dof_vel_actual = get_actor_dof_state(gym, st.env, st.arm_actor)
        gripper_actual = (
            float(dof_pos_actual[gripper_idx])
            if gripper_idx is not None and gripper_idx < len(dof_pos_actual)
            else float(st.args.gripper_open)
        )
        _vx_state, yaw_rate_state = base_command_from_targets(
            base_xy_current,
            yaw_current,
            st.prev_base_xy,
            st.prev_yaw,
            dt,
        )
        dp_state = make_float_dp_state(
            dof_names,
            dof_pos_actual,
            dof_vel_actual,
            ee_pos,
            ee_quat,
            base_xy_current,
            st.args.robot_z,
            yaw_current,
            yaw_rate_state,
            gripper_actual,
            st.last_dp_action,
        )
        camera_images = capture_dp_camera_images_from_rendered(gym, sim, st.env, st.camera_handles, st.args)
        wrist_mask_rgb, wrist_second_rgb, front_mask_rgb, front_second_rgb = dp_image_inputs_from_cpu_cameras(
            camera_images, st.args
        )
        missing_required_camera = wrist_mask_rgb is None or wrist_second_rgb is None or (
            st.args.rgb and (front_mask_rgb is None or front_second_rgb is None)
        )
        if missing_required_camera:
            missing_desc = "wrist/front camera RGB or mask images" if st.args.rgb else "wrist camera mask/depth images"
            raise RuntimeError(f"{mode_name} DP policy execution cannot run because {missing_desc} are missing.")
        dp_policy_inputs_by_env[st.index] = {
            "base_xy_current": base_xy_current,
            "yaw_current": yaw_current,
            "handle_goal": handle_goal,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "dp_state": dp_state,
        }
        batch_env_ids.append(st.index)
        batch_states.append(dp_state)
        batch_wrist_masks.append(wrist_mask_rgb)
        batch_wrist_seconds.append(wrist_second_rgb)
        batch_front_masks.append(front_mask_rgb)
        batch_front_seconds.append(front_second_rgb)
    if batch_env_ids:
        dp_actions = dp_controller.act_batch(
            batch_env_ids,
            batch_states,
            batch_wrist_masks,
            batch_wrist_seconds,
            batch_front_masks,
            batch_front_seconds,
        )
        for env_id, dp_action in zip(batch_env_ids, dp_actions):
            dp_actions_by_env[int(env_id)] = np.asarray(dp_action, dtype=np.float32)
    return dp_policy_inputs_by_env, dp_actions_by_env


def record_float_dp_frame(gym, sim, st, dof_names, gripper_idx, dt, phase_id, door_pos_record, door_vel_record):
    if st.dp_recorder is None:
        return False
    ee_pos, ee_quat = current_ee_pose_from_refreshed_tensors(st.ik_state)
    dof_pos_actual, dof_vel_actual = get_actor_dof_state(gym, st.env, st.arm_actor)
    gripper_actual = (
        float(dof_pos_actual[gripper_idx])
        if gripper_idx is not None and gripper_idx < len(dof_pos_actual)
        else float(st.args.gripper_open)
    )
    base_xy = st.traj.get("base_xy", st.base_start)
    yaw = float(st.traj.get("yaw", st.yaw_start))
    vx_cmd, yaw_rate_cmd = base_command_from_targets(base_xy, yaw, st.prev_base_xy, st.prev_yaw, dt)
    target_quat = target_quat_for_dp(st.last_target_quat, st.ik_state, ee_quat)
    camera_images = capture_dp_camera_images_from_rendered(gym, sim, st.env, st.camera_handles, st.args)
    wrist_mask_rgb, wrist_second_rgb, front_mask_rgb, front_second_rgb = dp_image_inputs_from_cpu_cameras(
        camera_images, st.args
    )
    missing_required_camera = wrist_mask_rgb is None or wrist_second_rgb is None or (
        st.args.rgb and (front_mask_rgb is None or front_second_rgb is None)
    )
    if missing_required_camera:
        if not st.dp_record_warned_no_camera:
            missing_desc = "wrist/front camera RGB or mask images" if st.args.rgb else "wrist camera mask/depth images"
            print(
                f"⚠️📷 env {st.index}: Camera unavailable: skipped DP frame because {missing_desc} are missing. "
                "No raw DP frame can be saved until camera images are available.",
                flush=True,
            )
            st.dp_record_warned_no_camera = True
        return False

    dp_state = make_float_dp_state(
        dof_names,
        dof_pos_actual,
        dof_vel_actual,
        ee_pos,
        ee_quat,
        base_xy,
        st.args.robot_z,
        yaw,
        yaw_rate_cmd,
        gripper_actual,
        st.last_dp_action,
    )
    dp_action = make_float_dp_action(
        vx_cmd,
        yaw_rate_cmd,
        st.last_target_pos,
        target_quat,
        st.last_gripper,
        base_xy,
        st.args.robot_z,
        yaw,
    )
    st.dp_recorder.add_frame(
        dp_state,
        wrist_mask_rgb,
        wrist_second_rgb,
        dp_action,
        phase_id,
        front_mask_rgb=front_mask_rgb,
        front_second_rgb=front_second_rgb,
        replay_snapshot=make_float_replay_snapshot(
            st.args,
            st.door,
            dof_names,
            dof_pos_actual,
            dof_vel_actual,
            door_pos_record,
            door_vel_record,
            ee_pos,
            ee_quat,
            base_xy,
            yaw,
            vx_cmd,
            yaw_rate_cmd,
        ),
    )
    st.last_dp_action = dp_action.copy()
    st.dp_record_success = st.dp_record_success or door_success(door_pos_record, st.args)
    return True


def finish_float_dp_recorders(env_states, args):
    saved = 0
    total_recorders = sum(st.dp_recorder is not None for st in env_states)
    for st in env_states:
        if st.dp_recorder is None:
            continue
        if st.dp_record_success and st.dp_recorder.frame_count > 0:
            st.dp_recorder.save_episode()
            saved += 1
            print(
                f"Finished raw Door DP recording env={st.index}: saved_successful=1/1 "
                f"frames={st.dp_recorder.frame_count}",
                flush=True,
            )
        elif st.dp_recorder.frame_count == 0:
            print(
                f"⚠️📷 Finished raw Door DP recording env={st.index}: "
                "saved_successful=0/1 frames=0 reason=camera_unavailable",
                flush=True,
            )
        else:
            print(
                f"Finished raw Door DP recording env={st.index}: saved_successful=0/1 "
                f"frames={st.dp_recorder.frame_count} reason=door did not reach {args.pass_open_angle_deg} deg",
                flush=True,
            )
        st.dp_recorder.finalize()
    if total_recorders:
        print(f"Finished parallel raw Door DP recording: saved_successful={saved}/{total_recorders}", flush=True)
    return saved, total_recorders


def closed_hinge_angle(door, args):
    if len(door.dof_lower) == 0 or len(door.dof_upper) == 0:
        return 0.0
    return float(door.dof_upper[0]) if float(args.door_motion_sign) < 0.0 else float(door.dof_lower[0])


def door_hinge_open_ratio(door, door_angle, args):
    if len(door.dof_lower) == 0 or len(door.dof_upper) == 0:
        return 0.0
    if float(args.door_motion_sign) < 0.0:
        closed_angle = float(door.dof_upper[0])
        open_limit = float(door.dof_lower[0])
    else:
        closed_angle = float(door.dof_lower[0])
        open_limit = float(door.dof_upper[0])
    hinge_range = max(abs(open_limit - closed_angle), 1.0e-6)
    return float(np.clip(abs(float(door_angle) - closed_angle) / hinge_range, 0.0, 1.0))


def door_open_degrees(door_pos, args):
    if door_pos is None or len(door_pos) == 0:
        return 0.0
    return math.degrees(float(args.door_motion_sign) * float(door_pos[0]))


def door_success(door_pos, args):
    return door_open_degrees(door_pos, args) >= float(args.pass_open_angle_deg)


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
        auto_torque = args.door_auto_open_force * args.door_motion_sign * args.door_auto_open_sign
    else:
        auto_torque = 0.0

    if door.open_stage:
        efforts[0] = auto_torque - args.door_open_resistance * door_angle - args.door_open_damping * float(dof_vel[0])
    elif args.door_lock_force > 0.0:
        efforts[0] = -args.door_motion_sign * args.door_lock_force

    efforts[1] = -args.handle_spring_stiffness * handle_angle_from_lower - args.handle_spring_damping * float(dof_vel[1])
    return efforts


def enforce_locked_door_hinge(gym, env, door_actor, door, args):
    if door.open_stage:
        return
    states = gym.get_actor_dof_states(env, door_actor, gymapi.STATE_ALL)
    if len(states) == 0:
        return
    states["pos"][0] = closed_hinge_angle(door, args)
    states["vel"][0] = 0.0
    gym.set_actor_dof_states(env, door_actor, states, gymapi.STATE_ALL)


def setup_viewer(gym, sim, args):
    viewer = base_ik.setup_viewer(gym, sim, args)
    if viewer is not None:
        gym.viewer_camera_look_at(
            viewer,
            None,
            gymapi.Vec3(float(args.door_x + 1.9), float(args.door_y + 3.2), 1.8),
            gymapi.Vec3(float(args.door_x + 0.3), float(args.door_y), 0.8),
        )
    return viewer


def finalize_float_ik_args(args, argv):
    default_true_flags = (
        ("draw_ik_target", "--draw_ik_target", "--no_draw_ik_target"),
        ("draw_camera_axes", "--draw_camera_axes", "--no_draw_camera_axes"),
        ("enable_wrist_camera", "--enable_wrist_camera", "--no_enable_wrist_camera"),
        ("enable_front_camera", "--enable_front_camera", "--no_enable_front_camera"),
        ("show_camera_images", "--show_camera_images", "--no_show_camera_images"),
    )
    for attr, positive_flag, negative_flag in default_true_flags:
        if negative_flag in argv:
            setattr(args, attr, False)
        elif positive_flag in argv:
            setattr(args, attr, True)
        else:
            setattr(args, attr, True)

    if "--no_show_seg" in argv:
        args.show_camera_images = False
    elif "--show_seg" in argv:
        args.show_camera_images = True
    args.camera_rgb = bool(args.camera_rgb or args.rgb)
    args.camera_depth = bool((args.camera_depth or not args.no_camera_depth) and not args.rgb)
    args.camera_seg = bool(args.camera_seg or not args.no_camera_seg)
    args._explicit_cli_flags = set(argv)

    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")

    if args.headless and args.show_camera_images:
        print("[camera] Headless mode disables OpenCV camera preview windows; run without --headless to view images.", flush=True)
        args.show_camera_images = False

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
