#!/usr/bin/env python3
"""Shared helpers for float-base B1Z1 door IK controllers."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from camera_intrinsics import (
    LEGACY_MODE as CAMERA_INTRINSICS_LEGACY_MODE,
    REAL_K_REMAP_MODE as CAMERA_INTRINSICS_REAL_K_REMAP_MODE,
    REMAP_VERSION as CAMERA_INTRINSICS_REMAP_VERSION,
    load_real_camera_intrinsics_config,
    remap_coverage,
    remap_image,
    reverse_remap_coordinates,
)

try:
    import cv2
except ImportError:
    cv2 = None


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = HIGH_LEVEL_ROOT.parents[0]
if str(HIGH_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_ROOT))
BASE_FLOAT_IK_SCRIPT = SCRIPT_DIR / "isaacgym_visualize_b1z1_basearn.py"
DEFAULT_DOOR_CFG = HIGH_LEVEL_ROOT / "data" / "cfg" / "b1z1_opendoor.yaml"
from dp.depth_camera_aug import (
    DEPTH_CAMERA_RESOLUTION,
    apply_depth_aug_config_defaults,
    apply_depth_noise,
    configure_depth_noise_for_env,
    depth_aug_metadata_from_args,
    depth_aug_custom_parameters,
    depth_noise_config_from_args,
    jitter_camera_pose,
)
DEFAULT_DOOR_ASSET_NAMES = (
    "99650089960001",
    "99650089960006",
    "99655039960001",
    "99655039960006",
    "wc4",
)
DEFAULT_PARTNET_NUMERIC_GRASP_Z_OFFSET = -0.042
DEFAULT_UNSAFE_DOOR_ASSET_NAMES = (
    # This asset can segfault PhysX convex cooking in Isaac Gym after VHACD:
    # Cooking::cookConvexMesh: user-provided convex mesh descriptor is invalid.
    "99690419960003",
)


def _unsafe_door_asset_names(args=None):
    names = set(DEFAULT_UNSAFE_DOOR_ASSET_NAMES)
    extra = str(getattr(args, "door_exclude_names", "") or "").strip()
    if extra:
        names.update(name.strip() for name in extra.split(",") if name.strip())
    allow_unsafe = bool(getattr(args, "allow_unsafe_door_assets", False))
    return set() if allow_unsafe else names


def _filter_unsafe_door_entries(entries, args=None):
    unsafe = _unsafe_door_asset_names(args)
    if not unsafe:
        return list(entries)
    kept = []
    skipped = []
    for entry in entries:
        name = str(entry[1].get("name", ""))
        if name in unsafe:
            skipped.append(name)
        else:
            kept.append(entry)
    if skipped:
        skipped_names = ", ".join(sorted(set(skipped)))
        print(f"Skipping unsafe door asset(s): {skipped_names}", flush=True)
    return kept


def _reorder_preferred_door(entries, preferred_name):
    if not preferred_name:
        return list(entries)
    preferred = [entry for entry in entries if entry[1].get("name") == preferred_name]
    if not preferred:
        print(
            f"Warning: preferred door {preferred_name!r} was not found; keeping configured door order.",
            flush=True,
        )
    others = [entry for entry in entries if entry[1].get("name") != preferred_name]
    return preferred + others


def _select_diverse_door_entries(entries, max_count, preferred_name=""):
    entries = list(entries)
    if not entries:
        return []

    preferred = []
    pool = entries
    if preferred_name:
        preferred = [entry for entry in entries if entry[1].get("name") == preferred_name]
        if preferred:
            preferred = preferred[:1]
            pool = [entry for entry in entries if entry[1].get("name") != preferred_name]
        else:
            preferred = []

    if max_count is None or int(max_count) <= 0:
        max_count = len(entries)
    max_count = max(1, min(int(max_count), len(entries)))
    remaining = max_count - len(preferred)
    if remaining <= 0:
        return preferred[:max_count]
    if remaining >= len(pool):
        return preferred + pool

    if remaining == 1:
        sampled = [pool[0]]
    else:
        sampled = []
        used = set()
        for i in range(remaining):
            idx = int(round(i * (len(pool) - 1) / float(remaining - 1)))
            while idx in used and idx + 1 < len(pool):
                idx += 1
            while idx in used and idx > 0:
                idx -= 1
            used.add(idx)
            sampled.append(pool[idx])
    return preferred + sampled
DEFAULT_WRIST_CAMERA_CFG = {
    "horizontal_fov": 55,
    "resolution": DEPTH_CAMERA_RESOLUTION,
    # link06 local frame: X points along the gripper, Y is lateral, Z is up.
    # Center the camera over the gripper instead of mounting it to the side.
    "position": [0.093, 0.031, 0.22],
    # Euler ZYX angles in degrees, i.e. yaw, pitch, roll.
    "rotation_deg": [0.0, 60.0, 0.0],
}
DEFAULT_FRONT_CAMERA_CFG = {
    "horizontal_fov": 55,
    "resolution": DEPTH_CAMERA_RESOLUTION,
    "position": [0.29, 0.031, 0.165],
    "rotation_deg": [0.0, -45.0, 0.0],
}
DEFAULT_REAL_CAMERA_INTRINSICS_CONFIG = (
    HIGH_LEVEL_ROOT / "data" / "cfg" / "a2w_real_camera_intrinsics_640x480.yaml"
)
_DOOR_SIDE_WALL_ASSET_CACHE = {}

DP_NUM_DOFS = 19
DP_NUM_ACTIONS = 18
FLOAT_DP_STATE_MODE_FULL = "full"
FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10 = "pi05_current_state10"
FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10 = "pi05_last_command_state10"
FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9 = "a2w_last_command_joint_state9"
FLOAT_DP_EE_ACTION10_NAMES = [
    "vx",
    "yaw",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
    "gripper",
]
A2W_Z1_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "jointGripper",
]
A2W_LAST_COMMAND_JOINT_STATE9_NAMES = [
    "last_command_vx",
    "last_command_vyaw",
    *A2W_Z1_JOINT_NAMES,
]
A2W_JOINT_ACTION9_NAMES = [
    "vx",
    "yaw",
    *A2W_Z1_JOINT_NAMES,
]
PI05_CURRENT_STATE10_NAMES = [
    "vx",
    "yaw_rate",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
    "gripper",
]
PI05_LAST_COMMAND_STATE10_NAMES = [
    "last_command_vx",
    "last_command_vyaw",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
    "gripper",
]


def door_asset_family(door):
    name = str(door.spec.get("name", ""))
    path = str(door.spec.get("path", ""))
    if name == "wc4" or "/wc4/" in f"/{path}/" or path.startswith("wc4/"):
        return "wc4"
    if name.startswith("rec_") or "record_materialization" in path:
        return "record_materialization"
    return "partnet_numeric"


def is_partnet_numeric_door(door):
    """True for the original PartNet numeric door assets, not wc4/generated doors."""
    name = str(door.spec.get("name", ""))
    return door_asset_family(door) == "partnet_numeric" and name.isdigit()


def door_wall_opening_axis(door):
    override = str(door.spec.get("wall_opening_axis_override", "") or "").strip().lower()
    if override in ("x", "y"):
        return override
    # In the push-door scenes, side walls should sit to the left/right of the
    # door corridor in world Y.  Legacy PartNet numeric doors keep their visual
    # actor yaw, but their bbox needs a wall-only yaw offset; see
    # door_wall_bounds_yaw_offset().
    return "y"


def door_wall_bounds_yaw_offset(door):
    override = door.spec.get("bounds_yaw_offset_override")
    if override is not None:
        return float(override)
    if door_asset_family(door) == "partnet_numeric":
        return math.pi / 2.0
    return 0.0


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
    handle_rest_angle: float | None = None
    handle_unlock_direction_sign: float = 1.0
    actor_scale: float = 1.0
    actor_yaw: float = math.pi
    actor_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_y_offset: float = 0.0
    door_motion_sign_multiplier: float = 1.0
    body_rgba: dict | None = None
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
    base_door_collision_detected: bool = False
    base_door_collision_log_step: int = -10**9
    dp_recorder: object = None
    dp_record_success: bool = False
    dp_record_warned_no_camera: bool = False
    dp_record_sim_steps: int = 0
    dp_record_prev_base_xy: object = None
    dp_record_prev_yaw: object = None
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
        specs = _filter_unsafe_door_entries(specs, args)
        selection = str(getattr(args, "door_selection", "default")).strip().lower()
        preferred_name = str(getattr(args, "door_prefer_name", "") or "")
        max_unique = int(getattr(args, "door_max_unique_assets", 0) or 0)
        if selection in ("all", "all_doors"):
            selected_entries = _reorder_preferred_door(specs, preferred_name)
        elif selection in ("push_left", "left", "left_handle"):
            selected_entries = [
                (idx, spec)
                for idx, spec in specs
                if str(spec.get("generated_variant", "")).lower() == "push_left"
                or str(spec.get("handle_side", "")).lower() == "left"
            ]
        elif selection in ("push_right", "right", "right_handle"):
            selected_entries = [
                (idx, spec)
                for idx, spec in specs
                if str(spec.get("generated_variant", "")).lower() == "push_right"
                or str(spec.get("handle_side", "")).lower() == "right"
            ]
        elif selection in ("diverse", "spread", "sample"):
            env_count = max(1, int(getattr(args, "num_envs", len(specs)) or len(specs)))
            if max_unique <= 0:
                max_unique = min(env_count, len(specs))
            else:
                max_unique = min(max_unique, env_count, len(specs))
            selected_entries = _select_diverse_door_entries(specs, max_unique, preferred_name=preferred_name)
        else:
            for name in DEFAULT_DOOR_ASSET_NAMES:
                selected_entries.extend((idx, spec) for idx, spec in specs if spec.get("name") == name)
            if not selected_entries:
                selected_entries = specs
            selected_entries = _reorder_preferred_door(selected_entries, preferred_name)
    if not selected_entries:
        raise RuntimeError(f"No door assets found in {cfg_path}")

    asset_root = HIGH_LEVEL_ROOT / asset_cfg["assetRoot"]
    asset_file_door = asset_cfg["assetFileDoor"]
    door_set_root = asset_root / asset_file_door
    loaded_specs = []
    path_override = str(getattr(args, "door_asset_path_override", "") or "").strip()
    if path_override and len(selected_entries) != 1:
        raise RuntimeError("--door_asset_path_override requires selecting exactly one door via --door_name or --door_index")
    for asset_index, selected in selected_entries:
        selected = dict(selected)
        if path_override:
            selected["path"] = path_override
        with (door_set_root / selected["bounding_box"]).open("r", encoding="utf-8") as f:
            bounding = json.load(f)
        with (door_set_root / selected["handle_bounding"]).open("r", encoding="utf-8") as f:
            handle_bounding = json.load(f)
        loaded_specs.append((int(asset_index), selected, bounding, handle_bounding))

    return str(asset_root), asset_file_door, loaded_specs


def parse_rgba_values(text):
    if not text:
        return None
    try:
        values = [float(value) for value in str(text).replace(",", " ").split()]
    except ValueError:
        return None
    if len(values) < 3:
        return None
    while len(values) < 4:
        values.append(1.0)
    return tuple(max(0.0, min(1.0, value)) for value in values[:4])


def load_urdf_link_rgba(asset_root, door_file):
    urdf_path = Path(asset_root) / door_file
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        print(f"Warning: failed to parse door URDF colors from {urdf_path}: {exc}", flush=True)
        return {}

    material_rgba = {}
    for material in root.findall("material"):
        name = material.get("name")
        color = material.find("color")
        rgba = parse_rgba_values(color.get("rgba") if color is not None else None)
        if name and rgba is not None:
            material_rgba[name] = rgba

    link_rgba = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for visual in link.findall("visual"):
            material = visual.find("material")
            if material is None:
                continue
            color = material.find("color")
            rgba = parse_rgba_values(color.get("rgba") if color is not None else None)
            if rgba is None:
                rgba = material_rgba.get(material.get("name"))
            if rgba is not None and rgba[3] > 0.01:
                link_rgba[link_name] = rgba
                break
    return link_rgba


def apply_door_urdf_rgba_colors(gym, env, door_actor, door, args):
    if not bool(getattr(args, "door_use_urdf_rgba", False)):
        return
    body_rgba = door.body_rgba or {}
    for body_index, body_name in enumerate(door.body_names):
        rgba = body_rgba.get(body_name)
        if rgba is None:
            continue
        try:
            gym.set_rigid_body_color(
                env,
                door_actor,
                body_index,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(float(rgba[0]), float(rgba[1]), float(rgba[2])),
            )
        except Exception:
            pass


def apply_door_rigid_body_property_overrides(gym, env, door_actor, door):
    """Apply per-body mass properties after actor scaling.

    Isaac Gym recomputes mass properties from collision geometry when
    ``override_com``/``override_inertia`` are enabled.  Generated doors can
    therefore feel substantially heavier than a reference door even when their
    joint friction and damping are identical.  Door specs may use this bounded
    override to make selected moving bodies match a validated reference asset.
    """

    overrides = door.spec.get("rigid_body_property_overrides", {})
    if not overrides:
        return
    if not isinstance(overrides, dict):
        raise ValueError(
            f"Door {door.spec.get('name', '')!r} rigid_body_property_overrides must be a mapping"
        )

    body_names = list(gym.get_actor_rigid_body_names(env, door_actor))
    properties = gym.get_actor_rigid_body_properties(env, door_actor)
    for body_name, override in overrides.items():
        if body_name not in body_names:
            raise ValueError(
                f"Door {door.spec.get('name', '')!r} cannot override unknown rigid body "
                f"{body_name!r}; loaded bodies: {body_names}"
            )
        if not isinstance(override, dict):
            raise ValueError(f"Rigid-body override for {body_name!r} must be a mapping")
        prop = properties[body_names.index(body_name)]

        if "mass" in override:
            mass = float(override["mass"])
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError(f"Rigid-body override mass for {body_name!r} must be positive and finite")
            prop.mass = mass

        if "center_of_mass" in override:
            center = [float(value) for value in override["center_of_mass"]]
            if len(center) != 3 or not all(math.isfinite(value) for value in center):
                raise ValueError(
                    f"Rigid-body override center_of_mass for {body_name!r} must contain 3 finite values"
                )
            prop.com.x, prop.com.y, prop.com.z = center

        if "inertia" in override:
            inertia = np.asarray(override["inertia"], dtype=np.float64)
            if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
                raise ValueError(f"Rigid-body override inertia for {body_name!r} must be a finite 3x3 matrix")
            if not np.allclose(inertia, inertia.T, atol=1.0e-7):
                raise ValueError(f"Rigid-body override inertia for {body_name!r} must be symmetric")
            if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
                raise ValueError(f"Rigid-body override inertia for {body_name!r} must be positive definite")
            rows = (prop.inertia.x, prop.inertia.y, prop.inertia.z)
            for row_index, row in enumerate(rows):
                row.x = float(inertia[row_index, 0])
                row.y = float(inertia[row_index, 1])
                row.z = float(inertia[row_index, 2])

    gym.set_actor_rigid_body_properties(env, door_actor, properties, False)


def load_door_assets(gym, sim, args):
    asset_root, asset_file_door, specs = load_door_specs(args)
    door_opts = gymapi.AssetOptions()
    door_opts.fix_base_link = True
    door_opts.collapse_fixed_joints = True
    door_opts.use_mesh_materials = not bool(getattr(args, "door_use_urdf_rgba", False))
    if hasattr(door_opts, "disable_visual_materials"):
        door_opts.disable_visual_materials = False
    door_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    door_opts.override_com = True
    door_opts.override_inertia = True
    door_opts.disable_gravity = True
    door_opts.vhacd_enabled = True
    door_opts.vhacd_params = gymapi.VhacdParams()
    door_opts.vhacd_params.resolution = args.door_vhacd_resolution
    print(
        "Door visual material source: "
        + ("URDF rgba" if bool(getattr(args, "door_use_urdf_rgba", False)) else "mesh materials"),
        flush=True,
    )

    doors = []
    for asset_index, spec, bounding, handle_bounding in specs:
        actor_scale = float(spec.get("actor_scale", args.door_actor_scale))
        if actor_scale <= 0.0:
            raise ValueError(f"Door {spec.get('name', asset_index)!r} actor_scale must be positive")
        actor_yaw = math.pi + float(spec.get("actor_yaw_offset", 0.0))
        actor_position_offset = tuple(
            float(value) for value in spec.get("actor_position_offset", (0.0, 0.0, 0.0))
        )
        if len(actor_position_offset) != 3:
            raise ValueError(
                f"Door {spec.get('name', asset_index)!r} actor_position_offset must contain exactly 3 values"
            )
        door_file = os.path.join(asset_file_door, spec["path"])
        print(f"Loading door[{asset_index}]: root={asset_root}, file={door_file}, name={spec['name']}")
        body_rgba = load_urdf_link_rgba(asset_root, door_file) if bool(getattr(args, "door_use_urdf_rgba", False)) else {}
        door_asset = gym.load_asset(sim, asset_root, door_file, door_opts)
        if door_asset is None:
            raise RuntimeError(f"Failed to load door asset {door_file}")

        body_names = gym.get_asset_rigid_body_names(door_asset)
        dof_names = gym.get_asset_dof_names(door_asset)
        if len(dof_names) < 1:
            raise RuntimeError(f"Door {spec['name']!r} must expose a door DOF; loaded DOFs: {dof_names}")
        expected_door_dof = spec.get("door_dof_name", dof_names[0])
        if dof_names[0] != expected_door_dof:
            raise RuntimeError(
                f"Door {spec['name']!r} first DOF must be {expected_door_dof!r}, loaded {dof_names[0]!r}"
            )
        expected_handle_dof = str(spec.get("handle_dof_name", "") or "")
        if expected_handle_dof:
            if len(dof_names) < 2:
                raise RuntimeError(
                    f"Door {spec['name']!r} expected handle DOF {expected_handle_dof!r}, "
                    f"but loaded DOFs: {dof_names}"
                )
            if dof_names[1] != expected_handle_dof:
                raise RuntimeError(
                    f"Door {spec['name']!r} second DOF must be {expected_handle_dof!r}, loaded {dof_names[1]!r}"
                )
        handle_body_name = spec.get("handle_body_name")
        door_body_name = spec.get("door_body_name")
        if handle_body_name and handle_body_name not in body_names:
            raise RuntimeError(
                f"Door {spec['name']!r} handle body {handle_body_name!r} not found; loaded bodies: {body_names}"
            )
        if door_body_name and door_body_name not in body_names:
            raise RuntimeError(
                f"Door {spec['name']!r} door body {door_body_name!r} not found; loaded bodies: {body_names}"
            )
        handle_body_index = body_names.index(handle_body_name) if handle_body_name else len(body_names) - 1
        door_body_index = body_names.index(door_body_name) if door_body_name else max(0, len(body_names) - 2)
        dof_props = gym.get_asset_dof_properties(door_asset)
        if len(dof_props["upper"]) >= 2:
            handle_upper_override = spec.get("handle_dof_upper_override")
            if handle_upper_override is not None:
                dof_props["upper"][1] = float(handle_upper_override)
            else:
                dof_props["upper"][1] = min(float(dof_props["upper"][1]), math.pi / 4)
        lower = np.asarray(dof_props["lower"], dtype=np.float32)
        upper = np.asarray(dof_props["upper"], dtype=np.float32)

        shape_props = gym.get_asset_rigid_shape_properties(door_asset)
        for prop in shape_props:
            prop.friction = 2.0
        gym.set_asset_rigid_shape_properties(door_asset, shape_props)

        handle_goal_offset = actor_scale * np.asarray(handle_bounding["goal_pos"], dtype=np.float32)
        handle_range = max(1.0e-6, float(upper[1] - lower[1]) if len(upper) >= 2 else 0.0)
        handle_unlock_angle = spec.get("handle_unlock_angle")
        handle_unlock_threshold = (
            float(handle_unlock_angle)
            if handle_unlock_angle is not None
            else (args.handle_unlock_ratio * handle_range if len(upper) >= 2 else 0.0)
        )
        handle_rest_angle = (
            float(spec["handle_rest_angle"])
            if spec.get("handle_rest_angle") is not None
            else (float(lower[1]) if len(lower) >= 2 else 0.0)
        )
        handle_unlock_direction_sign = (
            -1.0 if float(spec.get("handle_unlock_direction_sign", 1.0)) < 0.0 else 1.0
        )
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
                handle_body_index=handle_body_index,
                door_body_index=door_body_index,
                handle_goal_offset=handle_goal_offset,
                handle_unlock_threshold=handle_unlock_threshold,
                handle_rest_angle=handle_rest_angle,
                handle_unlock_direction_sign=handle_unlock_direction_sign,
                actor_scale=actor_scale,
                actor_yaw=actor_yaw,
                actor_position_offset=actor_position_offset,
                robot_y_offset=float(spec.get("robot_y_offset", 0.0)),
                door_motion_sign_multiplier=float(spec.get("door_motion_sign_multiplier", 1.0)),
                body_rgba=body_rgba,
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
        actor_position_offset=tuple(door.actor_position_offset),
        open_stage=False,
    )


def apply_door_runtime_overrides(args, door):
    override_key = (int(door.asset_index), str(door.spec.get("name", "")))
    if getattr(args, "_door_runtime_override_key", None) == override_key:
        return
    base_motion_sign = float(getattr(args, "_door_motion_sign_before_asset", args.door_motion_sign))
    args._door_motion_sign_before_asset = base_motion_sign
    args.door_motion_sign = base_motion_sign * float(door.door_motion_sign_multiplier)
    if bool(getattr(args, "flip_door_motion_sign", False)):
        args.door_motion_sign *= -1.0
    adjusted_controller_values = {}
    if is_partnet_numeric_door(door) and not cli_flag_was_set(args, "--grasp_z_offset"):
        # The original numeric PartNet lever doors need a slightly lower
        # scripted grasp than wc4 / button_door / generated digital-twin doors.
        # Keep this as a per-door runtime default so explicit CLI overrides and
        # per-asset custom doors are unaffected.
        args.grasp_z_offset = DEFAULT_PARTNET_NUMERIC_GRASP_Z_OFFSET
        adjusted_controller_values["grasp_z_offset"] = float(args.grasp_z_offset)
    if not bool(getattr(args, "ignore_door_controller_overrides", False)):
        for name, multiplier in door.spec.get("controller_multipliers", {}).items():
            if not hasattr(args, name):
                raise ValueError(f"Door {door.spec.get('name', '')!r} cannot multiply unknown controller arg {name!r}")
            setattr(args, name, float(getattr(args, name)) * float(multiplier))
            adjusted_controller_values[name] = float(getattr(args, name))
        for name, value in door.spec.get("controller_overrides", {}).items():
            if not hasattr(args, name):
                raise ValueError(f"Door {door.spec.get('name', '')!r} cannot override unknown controller arg {name!r}")
            setattr(args, name, value)
            adjusted_controller_values[name] = value
    for metadata_attr in ("ikpush_randomization_json", "ikpull_randomization_json"):
        raw_metadata = getattr(args, metadata_attr, "")
        if not raw_metadata or not adjusted_controller_values:
            continue
        metadata = json.loads(raw_metadata)
        metadata.update(adjusted_controller_values)
        metadata["door_asset_controller_adjusted"] = True
        setattr(args, metadata_attr, json.dumps(metadata, sort_keys=True))
    args._door_runtime_override_key = override_key


def robot_y_for_door(args, handle_bounding, door=None):
    alignment = str(getattr(args, "robot_y_alignment", "handle")).strip().lower()
    explicit_alignment_y = None
    if door is not None:
        explicit_alignment_y = door.spec.get("robot_alignment_y_offset")
        alignment_handle = door.spec.get("robot_alignment_handle")
        if explicit_alignment_y is None and isinstance(alignment_handle, dict):
            center_after_yaw = alignment_handle.get("center_after_yaw")
            if isinstance(center_after_yaw, (list, tuple)) and len(center_after_yaw) >= 2:
                explicit_alignment_y = center_after_yaw[1]
    if explicit_alignment_y is not None and alignment in (
        "auto",
        "generated_auto",
        "record_materialization_auto",
        "handle",
        "handle_center",
    ):
        actor_offset_y = float(door.actor_position_offset[1])
        # The explicit offset is an absolute asset-local target that has
        # already been rotated by actor_yaw. Adding the legacy robot_y_offset
        # here would move the robot away from the selected handle.
        return (
            float(args.robot_y)
            + float(args.door_y)
            + actor_offset_y
            + float(door.actor_scale) * float(explicit_alignment_y)
        )
    if alignment in ("auto", "generated_auto", "record_materialization_auto"):
        if door is not None:
            door_name = str(door.spec.get("name", ""))
            door_path = str(door.spec.get("path", ""))
            if door_name.startswith("rec_") or "record_materialization" in door_path:
                alignment = "door_center"
            else:
                alignment = "door_y"
        else:
            alignment = "door_y"
    robot_y_offset = float(door.robot_y_offset) if door is not None else 0.0
    if alignment in ("door_y", "spawn_center", "centerline", "center_line"):
        return float(args.robot_y) + float(args.door_y)
    if alignment in ("door_center", "door", "center"):
        if door is None:
            return float(args.robot_y) + float(args.door_y)
        _, _, world_min_y, world_max_y = door_world_xy_bounds_from_bbox(args, door)
        return float(args.robot_y) + 0.5 * (world_min_y + world_max_y)
    if alignment not in ("handle", "handle_center"):
        raise ValueError(
            f"Unsupported --robot_y_alignment={alignment!r}; expected auto, handle, door_center, or door_y"
        )
    handle_center_y = 0.5 * (
        float(handle_bounding["handle_min"][1]) + float(handle_bounding["handle_max"][1])
    )
    actor_scale = float(door.actor_scale) if door is not None else float(args.door_actor_scale)
    handle_center_world_y = args.door_y - actor_scale * handle_center_y
    return args.robot_y + handle_center_world_y + robot_y_offset


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
        handle_upper_override = door.spec.get("handle_dof_upper_override")
        if handle_upper_override is not None:
            door_dof_props["upper"][1] = float(handle_upper_override)
        else:
            door_dof_props["upper"][1] = min(float(door_dof_props["upper"][1]), math.pi / 4)
    gym.set_actor_dof_properties(env, door_actor, door_dof_props)

    door.dof_lower = np.asarray(door_dof_props["lower"], dtype=np.float32).copy()
    door.dof_upper = np.asarray(door_dof_props["upper"], dtype=np.float32).copy()
    handle_range = max(1.0e-6, float(door.dof_upper[1] - door.dof_lower[1]) if len(door.dof_upper) >= 2 else 0.0)
    handle_unlock_angle = door.spec.get("handle_unlock_angle")
    door.handle_unlock_threshold = (
        float(handle_unlock_angle)
        if handle_unlock_angle is not None
        else (args.handle_unlock_ratio * handle_range if len(door.dof_upper) >= 2 else 0.0)
    )
    door.handle_rest_angle = (
        float(door.spec["handle_rest_angle"])
        if door.spec.get("handle_rest_angle") is not None
        else (float(door.dof_lower[1]) if len(door.dof_lower) >= 2 else 0.0)
    )
    door.handle_unlock_direction_sign = (
        -1.0 if float(door.spec.get("handle_unlock_direction_sign", 1.0)) < 0.0 else 1.0
    )


def door_side_walls_enabled(args):
    return not bool(getattr(args, "no_door_side_walls", False))


def create_door_side_wall_asset(gym, sim, args, opening_axis="y"):
    thickness = max(1.0e-3, float(getattr(args, "door_wall_thickness", 0.08)))
    side_width = max(1.0e-3, float(getattr(args, "door_wall_side_width", 1.0)))
    height = max(1.0e-3, float(getattr(args, "door_wall_height", 2.2)))
    if str(opening_axis).lower() == "x":
        # The two side wall blocks sit to the left/right of the door in world X.
        dims = (side_width, thickness, height)
    else:
        # The two side wall blocks sit to the left/right of the door in world Y.
        dims = (thickness, side_width, height)
    cache_key = (id(sim), tuple(round(v, 5) for v in dims))
    if cache_key in _DOOR_SIDE_WALL_ASSET_CACHE:
        return _DOOR_SIDE_WALL_ASSET_CACHE[cache_key]

    asset_root = Path(tempfile.gettempdir()) / "b1z1_float_ik_wall_assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    dim_label = "_".join(f"{value:.3f}".replace(".", "p") for value in dims)
    file_name = f"door_side_wall_{dim_label}.urdf"
    asset_path = asset_root / file_name
    if not asset_path.exists():
        asset_path.write_text(
            f"""<?xml version="1.0"?>
<robot name="door_side_wall">
  <link name="wall">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{dims[0]:.6f} {dims[1]:.6f} {dims[2]:.6f}"/>
      </geometry>
      <material name="wall_gray">
        <color rgba="0.58 0.58 0.56 1"/>
      </material>
    </visual>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
</robot>
""",
            encoding="utf-8",
        )
    opts = gymapi.AssetOptions()
    opts.fix_base_link = True
    opts.disable_gravity = True
    asset = gym.load_asset(sim, str(asset_root), file_name, opts)
    _DOOR_SIDE_WALL_ASSET_CACHE[cache_key] = asset
    return asset


def door_world_xy_bounds_from_bbox(args, door, bounds_yaw_offset=None):
    actor_offset = door.actor_position_offset
    origin_x = float(args.door_x + actor_offset[0])
    origin_y = float(args.door_y + actor_offset[1])
    scale = float(door.actor_scale)
    if bounds_yaw_offset is None:
        bounds_yaw_offset = float(door.spec.get("bounds_yaw_offset_override", 0.0) or 0.0)
    yaw = float(door.actor_yaw) + float(bounds_yaw_offset)
    c = math.cos(yaw)
    s = math.sin(yaw)
    min_x = float(door.bounding["min"][0])
    max_x = float(door.bounding["max"][0])
    min_y = float(door.bounding["min"][1])
    max_y = float(door.bounding["max"][1])

    xs = []
    ys = []
    for local_x in (min_x, max_x):
        for local_y in (min_y, max_y):
            xs.append(origin_x + scale * (c * local_x - s * local_y))
            ys.append(origin_y + scale * (s * local_x + c * local_y))
    return min(xs), max(xs), min(ys), max(ys)


def create_door_side_walls(gym, sim, env, door, args, env_index=0):
    if not door_side_walls_enabled(args):
        return []

    opening_axis = door_wall_opening_axis(door)
    wall_asset = create_door_side_wall_asset(gym, sim, args, opening_axis=opening_axis)
    side_width = max(1.0e-3, float(getattr(args, "door_wall_side_width", 1.0)))
    height = max(1.0e-3, float(getattr(args, "door_wall_height", 2.2)))
    opening_width = float(getattr(args, "door_wall_opening_width", 0.0))
    world_min_x, world_max_x, world_min_y, world_max_y = door_world_xy_bounds_from_bbox(
        args,
        door,
        bounds_yaw_offset=door_wall_bounds_yaw_offset(door),
    )
    opening_min = world_min_x if opening_axis == "x" else world_min_y
    opening_max = world_max_x if opening_axis == "x" else world_max_y
    opening_center = 0.5 * (opening_min + opening_max)
    if opening_width <= 0.0:
        clearance = max(0.0, float(getattr(args, "door_wall_opening_clearance", 0.0)))
        opening_width = (opening_max - opening_min) + 2.0 * clearance
    opening_width = max(1.0e-3, opening_width)
    gap = max(0.0, float(getattr(args, "door_wall_gap", 0.0)))

    center_x = 0.5 * (world_min_x + world_max_x) + float(getattr(args, "door_wall_x_offset", 0.0))
    center_y = 0.5 * (world_min_y + world_max_y) + float(getattr(args, "door_wall_y_offset", 0.0))
    if opening_axis == "x":
        center_x = opening_center + float(getattr(args, "door_wall_x_offset", 0.0))
    else:
        center_y = opening_center + float(getattr(args, "door_wall_y_offset", 0.0))
    if door_asset_family(door) == "wc4":
        center_x = float(args.door_x) + float(getattr(args, "door_wall_x_offset", 0.0))
    z = float(getattr(args, "door_z_offset", 0.0)) + 0.5 * height
    side_offsets = (
        -0.5 * opening_width - gap - 0.5 * side_width,
        0.5 * opening_width + gap + 0.5 * side_width,
    )
    color = gymapi.Vec3(
        float(getattr(args, "door_wall_color_r", 0.58)),
        float(getattr(args, "door_wall_color_g", 0.58)),
        float(getattr(args, "door_wall_color_b", 0.56)),
    )

    actors = []
    for side_name, side_offset in (("left", side_offsets[0]), ("right", side_offsets[1])):
        pose = gymapi.Transform()
        if opening_axis == "x":
            pose.p = gymapi.Vec3(center_x + side_offset, center_y, z)
        else:
            pose.p = gymapi.Vec3(center_x, center_y + side_offset, z)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        actor = gym.create_actor(
            env,
            wall_asset,
            pose,
            f"door_side_wall_{env_index}_{side_name}",
            int(env_index),
            int(getattr(args, "door_wall_collision_filter", 0)),
            0,
        )
        try:
            gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL, color)
        except Exception:
            pass
        actors.append(actor)
    if int(env_index) < 4:
        print(
            "door_side_walls "
            f"env={env_index} door={door.spec.get('name')} family={door_asset_family(door)} "
            f"axis={opening_axis} opening_width={opening_width:.3f} "
            f"center=({center_x:.3f},{center_y:.3f})",
            flush=True,
        )
    return actors


def create_env_actors(gym, sim, base_asset, arm_asset, door, dof_props, dof_states, args):
    env = gym.create_env(sim, gymapi.Vec3(-2.5, -2.5, 0.0), gymapi.Vec3(2.5, 2.5, 2.5), 1)
    apply_door_runtime_overrides(args, door)
    robot_y = robot_y_for_door(args, door.handle_bounding, door)

    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(args.robot_x, robot_y, args.robot_z)
    robot_pose.r = robot_base_gym_quat(getattr(args, "robot_pitch", 0.0), args.robot_yaw)

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
    actor_offset = door.actor_position_offset
    door_pose.p = gymapi.Vec3(
        float(args.door_x + actor_offset[0]),
        float(args.door_y + actor_offset[1]),
        float(-door.bounding["min"][2] * door.actor_scale + args.door_z_offset + actor_offset[2]),
    )
    door_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, float(door.actor_yaw))
    door_actor = gym.create_actor(env, door.asset, door_pose, "door", 0, 0, 1)
    if abs(door.actor_scale - 1.0) > 1.0e-6:
        gym.set_actor_scale(env, door_actor, door.actor_scale)
    apply_door_rigid_body_property_overrides(gym, env, door_actor, door)
    try:
        gym.set_rigid_body_segmentation_id(env, door_actor, door.handle_body_index, int(args.handle_seg_id))
    except AttributeError:
        print("set_rigid_body_segmentation_id is not available; handle mask will use the door actor segmentation.")
    apply_door_urdf_rgba_colors(gym, env, door_actor, door, args)

    configure_door_actor_dofs(gym, env, door_actor, door, args)
    create_door_side_walls(gym, sim, env, door, args, env_index=0)
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
    env_spacing = max(1.0e-3, float(getattr(args, "env_spacing", 5.0)))
    env_half_spacing = 0.5 * env_spacing
    env = gym.create_env(
        sim,
        gymapi.Vec3(-env_half_spacing, -env_half_spacing, 0.0),
        gymapi.Vec3(env_half_spacing, env_half_spacing, max(2.5, env_spacing)),
        int(envs_per_row),
    )
    apply_door_runtime_overrides(args, door)
    robot_y = robot_y_for_door(args, door.handle_bounding, door)

    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(args.robot_x, robot_y, args.robot_z)
    robot_pose.r = robot_base_gym_quat(getattr(args, "robot_pitch", 0.0), args.robot_yaw)

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
    actor_offset = door.actor_position_offset
    door_pose.p = gymapi.Vec3(
        float(args.door_x + actor_offset[0]),
        float(args.door_y + actor_offset[1]),
        float(-door.bounding["min"][2] * door.actor_scale + args.door_z_offset + actor_offset[2]),
    )
    door_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, float(door.actor_yaw))
    door_actor = gym.create_actor(env, door.asset, door_pose, f"door_{env_index}", env_index, 0, 1)
    if abs(door.actor_scale - 1.0) > 1.0e-6:
        gym.set_actor_scale(env, door_actor, door.actor_scale)
    apply_door_rigid_body_property_overrides(gym, env, door_actor, door)
    try:
        gym.set_rigid_body_segmentation_id(env, door_actor, door.handle_body_index, int(args.handle_seg_id))
    except AttributeError:
        if env_index == 0:
            print("set_rigid_body_segmentation_id is not available; handle mask will use the door actor segmentation.")
    apply_door_urdf_rgba_colors(gym, env, door_actor, door, args)

    configure_door_actor_dofs(gym, env, door_actor, door, args)
    create_door_side_walls(gym, sim, env, door, args, env_index=env_index)
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


def cli_flag_was_set(args, flag):
    explicit_flags = getattr(args, "_explicit_cli_flags", set())
    return any(token == flag or str(token).startswith(f"{flag}=") for token in explicit_flags)


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


def sample_with_offset_range(rng, center, min_offset, max_offset, lower=None, upper=None):
    min_offset = float(min_offset)
    max_offset = float(max_offset)
    if max_offset < min_offset:
        min_offset, max_offset = max_offset, min_offset
    value = float(center) + float(rng.uniform(min_offset, max_offset))
    if lower is not None:
        value = max(float(lower), value)
    if upper is not None:
        value = min(float(upper), value)
    return value


def robot_base_gym_quat(pitch, yaw):
    return gymapi.Quat.from_euler_zyx(0.0, float(pitch), float(yaw))


def robot_base_quat_np(pitch, yaw):
    return gym_quat_to_np(robot_base_gym_quat(pitch, yaw))


def set_robot_base_pose(gym, env, actor_handles, xy, z, yaw, pitch=0.0):
    quat = robot_base_quat_np(pitch, yaw)
    for actor in actor_handles:
        root_handle = gym.get_actor_root_rigid_body_handle(env, actor)
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(float(xy[0]), float(xy[1]), float(z))
        transform.r = gymapi.Quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gym.set_rigid_transform(env, root_handle, transform)


def compute_base_walk_targets(args, door):
    yaw_start = float(args.robot_yaw)
    heading = np.array([math.cos(yaw_start), math.sin(yaw_start)], dtype=np.float32)
    base_start = np.asarray([args.robot_x, robot_y_for_door(args, door.handle_bounding, door)], dtype=np.float32)
    robot_front = base_start + heading * args.robot_front_offset
    door_xy = np.asarray([args.door_x, args.door_y], dtype=np.float32)
    front_to_door = float(np.dot(door_xy - robot_front, heading))
    walk_dist = max(0.0, front_to_door - args.stop_distance)
    base_stop = base_start + heading * walk_dist
    return yaw_start, heading, base_start, base_stop


def configure_dynamic_walk_steps(args, base_start, base_stop, env_index=None):
    distance = float(np.linalg.norm(np.asarray(base_stop, dtype=np.float32) - np.asarray(base_start, dtype=np.float32)))
    sim_dt = max(1.0e-6, float(getattr(args, "sim_dt", 1.0 / 50.0)))
    min_speed = max(1.0e-6, float(getattr(args, "walk_min_speed", 0.20)))
    original_steps = int(getattr(args, "walk_steps", 0))
    enabled = not bool(getattr(args, "no_dynamic_walk_steps", False)) and not cli_flag_was_set(args, "--walk_steps")

    if enabled:
        if distance <= 1.0e-6:
            walk_steps = 0
        else:
            walk_steps = max(1, int(math.floor(distance / (min_speed * sim_dt))))
        args.walk_steps = int(walk_steps)
    else:
        walk_steps = original_steps

    effective_speed = 0.0
    if int(walk_steps) > 0:
        effective_speed = distance / (float(walk_steps) * sim_dt)

    metadata_attr = "ikpush_randomization_json"
    raw_metadata = getattr(args, metadata_attr, "")
    if raw_metadata:
        try:
            metadata = json.loads(raw_metadata)
            metadata.update(
                {
                    "dynamic_walk_steps": bool(enabled),
                    "walk_distance": distance,
                    "walk_min_speed": min_speed,
                    "walk_steps": int(walk_steps),
                    "walk_effective_speed": effective_speed,
                }
            )
            setattr(args, metadata_attr, json.dumps(metadata, sort_keys=True))
        except Exception:
            pass

    if env_index is None or int(env_index) < 4:
        env_desc = "single" if env_index is None else int(env_index)
        print(
            f"dynamic_walk env={env_desc} enabled={bool(enabled)} "
            f"distance={distance:.3f}m min_speed={min_speed:.3f}m/s "
            f"steps={int(walk_steps)} effective_speed={effective_speed:.3f}m/s",
            flush=True,
        )

    return int(walk_steps)


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


def _numeric_contact_value(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if arr.shape == ():
        return float(abs(arr.item()))
    return float(np.linalg.norm(arr.reshape(-1)))


def contact_magnitude(contact):
    """Best-effort contact strength for Isaac Gym rigid contact records.

    PhysX contact record fields vary a bit across Isaac Gym builds.  Prefer
    force-like fields, then fall back to penetration/overlap, and finally to a
    unit count so old builds still produce a usable binary contact signal.
    """
    for name in (
        "lambda",
        "normal_force",
        "force",
        "contact_force",
        "impulse",
        "normal",
    ):
        magnitude = _numeric_contact_value(contact_field(contact, name))
        if magnitude is not None and magnitude > 0.0:
            return magnitude
    for name in ("initial_overlap", "min_dist"):
        magnitude = _numeric_contact_value(contact_field(contact, name))
        if magnitude is not None and magnitude > 0.0:
            return magnitude
    return 1.0


def parse_csv_names(value, default):
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [item.strip() for item in text.split(",") if item.strip()]


def _get_contact_body_handle(gym, env, actor, body_name):
    try:
        handle = gym.find_actor_rigid_body_handle(env, actor, body_name)
    except Exception:
        handle = -1
    return int(handle)


def init_gripper_handle_contact_tracking(gym, st):
    if hasattr(st, "gripper_handle_contact_initialized"):
        return
    st.gripper_handle_contact_initialized = True
    default_gripper_names = ["gripperStator", "gripperMover"]
    gripper_names = parse_csv_names(
        getattr(st.args, "gripper_handle_contact_gripper_bodies", ""),
        default_gripper_names,
    )
    gripper_handles = {}
    for body_name in gripper_names:
        body_handle = _get_contact_body_handle(gym, st.env, st.arm_actor, body_name)
        if body_handle >= 0:
            gripper_handles[str(body_name)] = body_handle

    handle_names = []
    try:
        if 0 <= int(st.door.handle_body_index) < len(st.door.body_names):
            handle_names.append(str(st.door.body_names[int(st.door.handle_body_index)]))
    except Exception:
        pass
    handle_names.extend(
        parse_csv_names(
            getattr(st.args, "gripper_handle_contact_handle_bodies", ""),
            [],
        )
    )
    # Keep order while deduplicating.
    handle_names = list(dict.fromkeys(handle_names))
    handle_handles = {}
    for body_name in handle_names:
        body_handle = _get_contact_body_handle(gym, st.env, st.door_actor, body_name)
        if body_handle >= 0:
            handle_handles[str(body_name)] = body_handle

    st.gripper_handle_contact_gripper_names = list(gripper_handles.keys())
    st.gripper_handle_contact_gripper_handles = dict(gripper_handles)
    st.gripper_handle_contact_handle_names = list(handle_handles.keys())
    st.gripper_handle_contact_handle_handles = dict(handle_handles)
    if bool(getattr(st.args, "debug_gripper_handle_contact", False)):
        print(
            f"[GripperHandleContact:init] env={int(st.index)} "
            f"gripper={st.gripper_handle_contact_gripper_handles} "
            f"handle={st.gripper_handle_contact_handle_handles}",
            flush=True,
        )


def gripper_handle_contact_snapshot(gym, st):
    """Return per-record-frame gripper-vs-handle contact features.

    The first two slots correspond to the configured/default gripper bodies
    (`gripperStator`, `gripperMover`).  If Isaac Gym does not expose a numeric
    force field, score becomes a contact count proxy.
    """
    init_gripper_handle_contact_tracking(gym, st)
    gripper_items = list(getattr(st, "gripper_handle_contact_gripper_handles", {}).items())
    handle_set = set(getattr(st, "gripper_handle_contact_handle_handles", {}).values())
    scores = np.zeros(2, dtype=np.float32)
    counts = np.zeros(2, dtype=np.float32)
    if not gripper_items or not handle_set:
        return {
            "gripper_handle_contact_score": scores,
            "gripper_handle_contact_count": counts,
            "gripper_handle_contact_any": np.asarray([0.0], dtype=np.float32),
            "gripper_handle_contact_both": np.asarray([0.0], dtype=np.float32),
        }
    gripper_by_handle = {int(handle): idx for idx, (_name, handle) in enumerate(gripper_items[:2])}
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
        idx = None
        if body0 in gripper_by_handle and body1 in handle_set:
            idx = gripper_by_handle[body0]
        elif body1 in gripper_by_handle and body0 in handle_set:
            idx = gripper_by_handle[body1]
        if idx is None or idx < 0 or idx >= 2:
            continue
        counts[idx] += 1.0
        scores[idx] += float(contact_magnitude(contact))
    min_score = float(getattr(st.args, "gripper_handle_contact_score_threshold", 1.0e-6))
    active = np.logical_or(scores > min_score, counts > 0.0)
    return {
        "gripper_handle_contact_score": scores,
        "gripper_handle_contact_count": counts,
        "gripper_handle_contact_any": np.asarray([float(np.any(active))], dtype=np.float32),
        "gripper_handle_contact_both": np.asarray([float(np.all(active))], dtype=np.float32),
    }


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
    legacy_check = bool(getattr(args, "enable_base_door_collision_check", False))
    physx_check = legacy_check or bool(getattr(args, "enable_collision_physx_check", False))
    geom_check = legacy_check or bool(getattr(args, "enable_collision_geom_check", False))
    if not physx_check and not geom_check:
        return False

    if physx_check and not hasattr(st, "base_body_handles"):
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
    if physx_check:
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
    phase = str(getattr(st, "last_phase", "unknown"))
    open_deg = door_open_degrees(getattr(st, "last_door_pos", None), args)
    ignore_open_traverse_geom = bool(
        getattr(args, "pass_through_door", False)
        and phase in ("traverse_door", "return_home", "hold_home")
        and open_deg >= float(getattr(args, "pass_open_angle_deg", 80.0))
    )
    geom_collision = bool(geom_check and not ignore_open_traverse_geom and geom_distance <= threshold)
    rigid_gate = float(getattr(args, "rigid_contact_geom_gate", max(0.15, threshold)))
    gated_rigid_contact = bool(physx_check and rigid_contact and geom_distance <= rigid_gate)
    frame_contact = bool(physx_check and frame_contact)

    # A scripted push-through task must move the base toward the door immediately
    # after the handle unlocks. The cheap rectangle/segment test can report a
    # collision before the panel has accumulated a visible opening angle even
    # though PhysX reports no contact. Keep such proximity visible in diagnostics,
    # but do not let a geometry-only post-unlock near miss overwrite an otherwise
    # valid task. Actual rigid/frame contact and every pre-unlock collision remain
    # blocking.
    tracker = getattr(st, "door_twin_tracker", None)
    handle_unlocked_before_contact = bool(
        tracker is not None and bool(getattr(tracker, "handle_unlocked", False))
    )
    door_opened_before_contact = bool(
        tracker is not None and float(getattr(tracker, "max_door_open_deg", 0.0)) >= 5.0
    )
    post_unlock = handle_unlocked_before_contact
    tolerated_post_unlock_geom_contact = bool(
        getattr(args, "allow_post_unlock_geom_contact", True)
        and post_unlock
        and geom_collision
        and not gated_rigid_contact
        and not frame_contact
    )
    contact_candidate = bool(frame_contact or gated_rigid_contact or geom_collision)
    collision = bool(contact_candidate and not tolerated_post_unlock_geom_contact)
    if contact_candidate:
        st.base_door_contact_detected = True
    if collision:
        st.base_door_collision_detected = True
        if hasattr(st, "success"):
            st.success = False
        if hasattr(st, "dp_record_success"):
            st.dp_record_success = False
    if contact_candidate:
        interval = max(1, int(getattr(args, "collision_log_interval", 30)))
        if int(step) - int(getattr(st, "base_door_collision_log_step", -10**9)) >= interval:
            st.base_door_collision_log_step = int(step)
            json_contact_pair = {}
            for key, value in (contact_pair or {}).items():
                if value is None:
                    json_contact_pair[key] = None
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    json_contact_pair[key] = float(value)
                else:
                    json_contact_pair[key] = str(value)
            if tracker is not None:
                tracker.add_artifact(
                    "base_collision_event",
                    {
                        "step": int(step),
                        "phase": phase,
                        "physx_check": bool(physx_check),
                        "geom_check": bool(geom_check),
                        "rigid_contact": bool(rigid_contact),
                        "frame_contact": bool(frame_contact),
                        "gated_rigid_contact": bool(gated_rigid_contact),
                        "geom_collision": bool(geom_collision),
                        "blocking": bool(collision),
                        "tolerated_post_unlock_geom_contact": bool(tolerated_post_unlock_geom_contact),
                        "handle_unlocked_before_contact": bool(handle_unlocked_before_contact),
                        "door_opened_before_contact": bool(door_opened_before_contact),
                        "geom_distance": float(geom_distance),
                        "threshold": float(threshold),
                        "gate": float(rigid_gate),
                        "contact_pair": json_contact_pair,
                        "base_xy": np.round(base_xy, 5).tolist(),
                        "door_local": {
                            "hinge": np.round(hinge_local, 5).tolist(),
                            "handle": np.round(handle_local, 5).tolist(),
                        },
                        "open_deg": float(open_deg),
                    },
                )
            event_label = "BaseDoorCollision" if collision else "BaseDoorContactTolerated"
            print(
                f"[{event_label}]"
                f" step={int(step)} env={int(st.index)} phase={phase}"
                f" physx_check={bool(physx_check)} geom_check={bool(geom_check)}"
                f" rigid_contact={bool(rigid_contact)} frame_contact={bool(frame_contact)}"
                f" gated_rigid_contact={bool(gated_rigid_contact)} geom_collision={bool(geom_collision)}"
                f" geom_distance={geom_distance:.4f} threshold={threshold:.4f} gate={rigid_gate:.4f}"
                f" contact_pair={contact_pair}"
                f" base_xy={np.round(base_xy, 4).tolist()}"
                f" door_local=({np.round(hinge_local, 4).tolist()}, {np.round(handle_local, 4).tolist()})"
                f" open_deg={open_deg:.1f} post_unlock={post_unlock}",
                flush=True,
            )
    elif physx_check and rigid_contact:
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



def camera_rotation_radians_from_cfg(camera_cfg):
    if "rotation_deg" in camera_cfg:
        return [math.radians(float(value)) for value in camera_cfg.get("rotation_deg", [0.0, 0.0, 0.0])]
    return list(camera_cfg.get("rotation", [0.0, 0.0, 0.0]))


def wrist_camera_rotation_radians_from_args(args):
    return [
        math.radians(float(args.wrist_camera_yaw_deg)),
        math.radians(float(args.wrist_camera_pitch_deg)),
        math.radians(float(args.wrist_camera_roll_deg)),
    ]


def front_camera_rotation_radians_from_args(args):
    return [
        math.radians(float(args.front_camera_yaw_deg)),
        math.radians(float(args.front_camera_pitch_deg)),
        math.radians(float(args.front_camera_roll_deg)),
    ]


def local_camera_pose_from_cfg(camera_cfg, local_rot_override=None):
    local_pos = np.asarray(camera_cfg.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    local_rot = camera_rotation_radians_from_cfg(camera_cfg)
    if local_rot_override is not None:
        local_rot = list(local_rot_override)
    local_quat = gym_quat_to_np(gymapi.Quat.from_euler_zyx(*local_rot))
    return local_pos, base_ik.normalize_quat(local_quat)


def camera_intrinsics_from_cfg(camera_cfg):
    resolution = camera_cfg.get("resolution", DEPTH_CAMERA_RESOLUTION)
    width = int(resolution[0])
    height = int(resolution[1])
    hfov_deg = float(camera_cfg.get("horizontal_fov", 69.0))
    fx = float(width) / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx
    return {
        "fx": fx,
        "fy": fy,
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "width": width,
        "height": height,
        "horizontal_fov_deg": hfov_deg,
    }


def camera_intrinsics_mode_from_args(args=None):
    mode = str(getattr(args, "camera_intrinsics_mode", CAMERA_INTRINSICS_LEGACY_MODE) or "legacy")
    if mode not in (CAMERA_INTRINSICS_LEGACY_MODE, CAMERA_INTRINSICS_REAL_K_REMAP_MODE):
        raise ValueError(
            f"Unsupported --camera_intrinsics_mode={mode!r}; expected 'legacy' or 'real_k_remap'."
        )
    return mode


def camera_intrinsics_profile_from_args(args):
    if camera_intrinsics_mode_from_args(args) == CAMERA_INTRINSICS_LEGACY_MODE:
        return None
    cached = getattr(args, "_camera_intrinsics_profile", None)
    if cached is not None:
        return cached
    config_path = str(
        getattr(args, "camera_intrinsics_config", str(DEFAULT_REAL_CAMERA_INTRINSICS_CONFIG))
        or DEFAULT_REAL_CAMERA_INTRINSICS_CONFIG
    )
    profile = load_real_camera_intrinsics_config(config_path)
    requested_fov = float(
        getattr(args, "camera_render_horizontal_fov_deg", profile["render_horizontal_fov_deg"])
    )
    configured_fov = float(profile["render_horizontal_fov_deg"])
    if not math.isclose(requested_fov, configured_fov, rel_tol=0.0, abs_tol=1.0e-9):
        # The public override is authoritative, while the target K remains loaded from YAML.
        width, height = profile["resolution"]
        render_cfg = {
            "resolution": [width, height],
            "horizontal_fov": requested_fov,
        }
        profile["render_horizontal_fov_deg"] = requested_fov
        profile["render_intrinsics"] = camera_intrinsics_from_cfg(render_cfg)
    maps = {}
    for camera_name in ("front", "wrist"):
        map_x, map_y = reverse_remap_coordinates(
            profile["render_intrinsics"], profile["camera_intrinsics"][camera_name]
        )
        coverage = remap_coverage(map_x, map_y, profile["render_intrinsics"])
        if coverage < 1.0:
            raise ValueError(
                f"{camera_name} target intrinsics are not fully covered by the render camera: "
                f"coverage={coverage:.6f}. Increase --camera_render_horizontal_fov_deg."
            )
        maps[camera_name] = (map_x, map_y)
    profile["remap_maps"] = maps
    profile["coverage"] = {
        name: remap_coverage(*maps[name], profile["render_intrinsics"])
        for name in ("front", "wrist")
    }
    setattr(args, "_camera_intrinsics_profile", profile)
    print(
        "Camera intrinsics real_k_remap enabled: "
        f"render={profile['resolution'][0]}x{profile['resolution'][1]} "
        f"FOV={profile['render_horizontal_fov_deg']:.6g}deg "
        f"front_coverage={profile['coverage']['front']:.6f} "
        f"wrist_coverage={profile['coverage']['wrist']:.6f}",
        flush=True,
    )
    return profile


def camera_cfg_for_args(camera_name, args):
    source = DEFAULT_WRIST_CAMERA_CFG if camera_name == "wrist" else DEFAULT_FRONT_CAMERA_CFG
    cfg = dict(source)
    if camera_intrinsics_mode_from_args(args) == CAMERA_INTRINSICS_REAL_K_REMAP_MODE:
        profile = camera_intrinsics_profile_from_args(args)
        cfg["resolution"] = list(profile["resolution"])
        cfg["horizontal_fov"] = float(profile["render_horizontal_fov_deg"])
    return cfg


def remap_camera_image_for_args(image, camera_name, args, *, interpolation):
    if camera_intrinsics_mode_from_args(args) == CAMERA_INTRINSICS_LEGACY_MODE:
        return image
    if camera_name not in ("front", "wrist"):
        return image
    profile = camera_intrinsics_profile_from_args(args)
    map_x, map_y = profile["remap_maps"][camera_name]
    return remap_image(image, map_x, map_y, interpolation=interpolation)


def depth_camera_intrinsics_metadata(args=None):
    if args is None or camera_intrinsics_mode_from_args(args) == CAMERA_INTRINSICS_LEGACY_MODE:
        return {
            "front": camera_intrinsics_from_cfg(DEFAULT_FRONT_CAMERA_CFG),
            "wrist": camera_intrinsics_from_cfg(DEFAULT_WRIST_CAMERA_CFG),
        }
    return {
        name: dict(camera_intrinsics_profile_from_args(args)["camera_intrinsics"][name])
        for name in ("front", "wrist")
    }


def camera_intrinsics_recording_metadata(args):
    mode = camera_intrinsics_mode_from_args(args)
    if mode == CAMERA_INTRINSICS_LEGACY_MODE:
        # Keep legacy sidecars byte-compatible: the absence of a mode field means legacy.
        return {"camera_intrinsics": depth_camera_intrinsics_metadata()}
    profile = camera_intrinsics_profile_from_args(args)
    metadata = {
        "camera_intrinsics_mode": mode,
    }
    metadata.update(
        {
            "camera_intrinsics": depth_camera_intrinsics_metadata(args),
            "camera_render_intrinsics": dict(profile["render_intrinsics"]),
            "camera_intrinsics_remap_version": CAMERA_INTRINSICS_REMAP_VERSION,
            "camera_intrinsics_config": str(profile["config_path"]),
            "camera_intrinsics_remap_coverage": dict(profile["coverage"]),
            "render_resolution": list(profile["resolution"]),
            "output_resolution": list(profile["resolution"]),
        }
    )
    return metadata


def _cached_or_default_local_camera_pose(args, camera_name, camera_cfg, local_rot_override=None):
    cache = getattr(args, "_camera_axis_local_poses", {}) or {}
    cached = cache.get(camera_name)
    if cached is not None:
        local_pos = np.asarray(cached.get("local_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        local_quat = base_ik.normalize_quat(
            np.asarray(cached.get("local_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
        )
        return local_pos, local_quat
    local_pos, local_quat = local_camera_pose_from_cfg(camera_cfg, local_rot_override)
    # If this function is called before the camera has been attached, mirror the
    # attach-time jitter path so --record_camera_pose still records the actual
    # randomized local pose once and caches it for all later calls.
    if bool(getattr(args, "enable_depth_camera_randomization", False)):
        local_rot = camera_rotation_radians_from_cfg(camera_cfg)
        if local_rot_override is not None:
            local_rot = list(local_rot_override)
        local_pos, local_rot = jitter_camera_pose_for_args(local_pos, local_rot, args)
        local_quat = base_ik.normalize_quat(gym_quat_to_np(gymapi.Quat.from_euler_zyx(*local_rot)))
        cache = getattr(args, "_camera_axis_local_poses", None)
        if cache is None:
            cache = {}
            setattr(args, "_camera_axis_local_poses", cache)
        cache[camera_name] = {
            "local_pos": np.asarray(local_pos, dtype=np.float32).copy(),
            "local_quat": np.asarray(local_quat, dtype=np.float32).copy(),
        }
    return local_pos, local_quat


def compose_pose_np(parent_pos, parent_quat, local_pos, local_quat):
    parent_pos = np.asarray(parent_pos, dtype=np.float32)
    parent_quat = base_ik.normalize_quat(np.asarray(parent_quat, dtype=np.float32))
    local_pos = np.asarray(local_pos, dtype=np.float32)
    local_quat = base_ik.normalize_quat(np.asarray(local_quat, dtype=np.float32))
    pos = parent_pos + quat_apply(parent_quat, local_pos)
    quat = base_ik.normalize_quat(base_ik.quat_multiply(parent_quat, local_quat))
    return pos.astype(np.float32), quat.astype(np.float32)


def relative_pose_np(parent_pos, parent_quat, child_pos, child_quat):
    parent_pos = np.asarray(parent_pos, dtype=np.float32)
    parent_quat = base_ik.normalize_quat(np.asarray(parent_quat, dtype=np.float32))
    child_pos = np.asarray(child_pos, dtype=np.float32)
    child_quat = base_ik.normalize_quat(np.asarray(child_quat, dtype=np.float32))
    inv_parent_quat = base_ik.quat_conjugate(parent_quat)
    rel_pos = quat_apply(inv_parent_quat, child_pos - parent_pos)
    rel_quat = base_ik.normalize_quat(base_ik.quat_multiply(inv_parent_quat, child_quat))
    return rel_pos.astype(np.float32), rel_quat.astype(np.float32)


def pose7_np(pos, quat):
    return np.concatenate(
        [
            np.asarray(pos, dtype=np.float32).reshape(3),
            base_ik.normalize_quat(np.asarray(quat, dtype=np.float32)).reshape(4),
        ],
        axis=0,
    ).astype(np.float32)


def float_camera_pose_base(gym, st):
    """Return (front_pose_base, wrist_pose_base), each [x,y,z,qx,qy,qz,qw].

    Poses are camera optical-frame poses expressed in the robot base/root
    rigid-body frame.  The local camera poses are the same cached values used
    for Isaac Gym camera attachment and camera-axis visualization.
    """
    base_actor = st.actor_handles[0] if len(st.actor_handles) > 1 else st.arm_actor
    base_pos, base_quat = get_body_pose(gym, st.env, base_actor, 0)

    front_rot = front_camera_rotation_radians_from_args(st.args)
    front_local_pos, front_local_quat = _cached_or_default_local_camera_pose(
        st.args,
        "front",
        DEFAULT_FRONT_CAMERA_CFG,
        front_rot,
    )
    front_body_pos, front_body_quat = get_body_pose(gym, st.env, base_actor, 0)
    front_world_pos, front_world_quat = compose_pose_np(
        front_body_pos,
        front_body_quat,
        front_local_pos,
        front_local_quat,
    )
    front_base_pos, front_base_quat = relative_pose_np(base_pos, base_quat, front_world_pos, front_world_quat)

    wrist_body_index = get_actor_body_index(gym, st.env, st.arm_actor, "link06")
    if wrist_body_index is None:
        raise RuntimeError("Cannot record wrist camera pose because arm body 'link06' was not found.")
    wrist_rot = wrist_camera_rotation_radians_from_args(st.args)
    wrist_local_pos, wrist_local_quat = _cached_or_default_local_camera_pose(
        st.args,
        "wrist",
        DEFAULT_WRIST_CAMERA_CFG,
        wrist_rot,
    )
    wrist_body_pos, wrist_body_quat = get_body_pose(gym, st.env, st.arm_actor, wrist_body_index)
    wrist_world_pos, wrist_world_quat = compose_pose_np(
        wrist_body_pos,
        wrist_body_quat,
        wrist_local_pos,
        wrist_local_quat,
    )
    wrist_base_pos, wrist_base_quat = relative_pose_np(base_pos, base_quat, wrist_world_pos, wrist_world_quat)
    return pose7_np(front_base_pos, front_base_quat), pose7_np(wrist_base_pos, wrist_base_quat)


def depth_camera_noise_config_for_args(args):
    return depth_noise_config_from_args(args)


def depth_camera_randomization_metadata(args):
    return depth_aug_metadata_from_args(args)


def jitter_camera_pose_for_args(local_pos, local_rot, args):
    return jitter_camera_pose(
        local_pos,
        local_rot,
        None,
        enabled=bool(getattr(args, "enable_depth_camera_randomization", False)),
        pos_range_m=float(getattr(args, "depth_camera_pos_rand_m", 0.01)),
        rot_range_deg=float(getattr(args, "depth_camera_rot_rand_deg", 2.0)),
    )


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
    wrist_rot = wrist_camera_rotation_radians_from_args(args)
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

    front_rot = front_camera_rotation_radians_from_args(args)
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
    props.width = int(camera_cfg.get("resolution", DEPTH_CAMERA_RESOLUTION)[0])
    props.height = int(camera_cfg.get("resolution", DEPTH_CAMERA_RESOLUTION)[1])
    if camera_cfg.get("horizontal_fov", None) is not None:
        props.horizontal_fov = float(camera_cfg["horizontal_fov"])
    return props


def attach_camera_to_actor_body(gym, env, actor, body_name, camera_cfg, local_rot_override=None, args=None):
    body_handle = gym.find_actor_rigid_body_handle(env, actor, body_name)
    if body_handle < 0:
        return None
    local_pos = np.asarray(camera_cfg.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    local_rot = camera_rotation_radians_from_cfg(camera_cfg)
    if local_rot_override is not None:
        local_rot = list(local_rot_override)
    if args is not None:
        local_pos, local_rot = jitter_camera_pose_for_args(local_pos, local_rot, args)
    local_quat = gym_quat_to_np(gymapi.Quat.from_euler_zyx(*local_rot))
    local_quat = base_ik.normalize_quat(local_quat)
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
        wrist_rot = wrist_camera_rotation_radians_from_args(args)
        wrist_camera_cfg = camera_cfg_for_args("wrist", args)
        wrist_camera = attach_camera_to_actor_body(
            gym, env, arm_actor, "link06", wrist_camera_cfg, wrist_rot, args=args
        )
        if wrist_camera is None:
            print("⚠️📷 Wrist camera sensor creation failed; wrist camera image display is disabled.", flush=True)
        else:
            cameras["wrist"] = wrist_camera
            print(f"Wrist camera sensor enabled: handle={wrist_camera}")

    if args.enable_front_camera:
        front_rot = front_camera_rotation_radians_from_args(args)
        front_camera_cfg = camera_cfg_for_args("front", args)
        base_actor = actor_handles[0] if len(actor_handles) > 1 else arm_actor
        front_camera = attach_camera_to_actor_body(
            gym, env, base_actor, "trunk", front_camera_cfg, front_rot, args=args
        )
        if front_camera is None:
            front_camera = attach_camera_to_actor_body(
                gym, env, base_actor, "base", front_camera_cfg, front_rot, args=args
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
            show_camera_masks = bool(getattr(args, "show_camera_masks", False))
            pair_names = (
                ", ".join(f"{name}_rgb" + (f"/{name}_mask" if show_camera_masks else "") for name in cameras.keys())
                if args.rgb
                else ", ".join(
                    f"{name}_mask/{name}_full_depth" if show_camera_masks else f"{name}_full_depth"
                    for name in cameras.keys()
                )
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
    show_camera_masks = bool(getattr(args, "show_camera_masks", False)) and not bool(getattr(args, "depth_only", False))
    for prefix, camera_handle in camera_handles.items():
        camera_cfg = camera_cfg_for_args(prefix, args)
        width = int(camera_cfg.get("resolution", DEPTH_CAMERA_RESOLUTION)[0])
        height = int(camera_cfg.get("resolution", DEPTH_CAMERA_RESOLUTION)[1])
        handle_mask = None
        mask_vis = None
        if show_camera_masks:
            seg_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)
            if seg_raw is None:
                continue
            seg_image = camera_image_to_array(seg_raw, height, width).astype(np.int32)
            seg_image = remap_camera_image_for_args(seg_image, prefix, args, interpolation="nearest")
            handle_mask = (seg_image == int(args.handle_seg_id)).astype(np.float32)
            mask_vis = (255.0 * handle_mask).astype(np.uint8)

        if args.rgb:
            rgb_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
            if rgb_raw is None:
                continue
            rgb_image = camera_color_to_rgb(rgb_raw, height, width)
            rgb_image = remap_camera_image_for_args(rgb_image, prefix, args, interpolation="linear")
            rgb_nonzero = int(np.count_nonzero(rgb_image))
            printed = getattr(args, "_camera_image_stats_printed", set())
            if prefix not in printed:
                if show_camera_masks:
                    print(
                        f"{prefix} camera image stats: "
                        f"rgb_shape={tuple(rgb_image.shape)} "
                        f"mask_pixels={int(handle_mask.sum())} "
                        f"rgb_nonzero_pixels={rgb_nonzero}",
                        flush=True,
                    )
                else:
                    print(
                        f"{prefix} camera image stats: "
                        f"rgb_shape={tuple(rgb_image.shape)} "
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
                if mask_vis is not None:
                    mask_vis = cv2.resize(mask_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
                rgb_image = cv2.resize(rgb_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f"{prefix.capitalize()} RGB", rgb_image[..., ::-1].copy())
            if show_camera_masks:
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
        depth_image = process_metric_depth_for_args(depth_image, prefix, args)

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
            if not show_camera_masks:
                print(
                    f"{prefix} camera image stats: "
                    f"depth_shape={tuple(depth_image.shape)} "
                    f"valid_depth_pixels={int(valid_depth.size)}",
                    flush=True,
                )
            else:
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
        if show_camera_masks and prefix not in visible_printed and handle_mask.sum() > 0:
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
            if mask_vis is not None:
                mask_vis = cv2.resize(mask_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
            depth_vis = cv2.resize(
                depth_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST
            )
        if show_camera_masks:
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


def process_metric_depth_for_args(depth_image, camera_name, args):
    """Apply the legacy pipeline exactly, or remap metric depth before augmentation."""
    depth_image = np.asarray(depth_image, dtype=np.float32)
    if camera_intrinsics_mode_from_args(args) == CAMERA_INTRINSICS_REAL_K_REMAP_MODE:
        depth_image = remap_camera_image_for_args(depth_image, camera_name, args, interpolation="nearest")
        depth_image = apply_depth_noise(
            depth_image,
            None,
            depth_camera_noise_config_for_args(args),
            valid_mask=np.isfinite(depth_image) & (depth_image > 0.0),
        )
        depth_image[depth_image < float(args.camera_depth_clip_lower)] = 0.0
        return np.clip(depth_image, 0.0, float(args.camera_depth_clip_far))

    # Preserve the original ordering and values in legacy mode.
    depth_image[depth_image < float(args.camera_depth_clip_lower)] = 0.0
    depth_image = np.clip(depth_image, 0.0, float(args.camera_depth_clip_far))
    depth_image = apply_depth_noise(
        depth_image,
        None,
        depth_camera_noise_config_for_args(args),
        valid_mask=depth_image >= float(args.camera_depth_clip_lower),
    )
    depth_image[depth_image < float(args.camera_depth_clip_lower)] = 0.0
    return np.clip(depth_image, 0.0, float(args.camera_depth_clip_far))


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
        camera_cfg = camera_cfg_for_args(prefix, args)
        width = int(camera_cfg.get("resolution", DEPTH_CAMERA_RESOLUTION)[0])
        height = int(camera_cfg.get("resolution", DEPTH_CAMERA_RESOLUTION)[1])
        seg_image = camera_image_to_array(seg_raw, height, width).astype(np.int32)
        seg_image = remap_camera_image_for_args(seg_image, prefix, args, interpolation="nearest")
        handle_mask = (seg_image == int(args.handle_seg_id)).astype(np.float32)
        images[f"{prefix}_handle_mask"] = mask_to_rgb(handle_mask)

        if args.rgb:
            rgb_raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
            if rgb_raw is None:
                continue
            rgb_image = camera_color_to_rgb(rgb_raw, height, width)
            rgb_image = remap_camera_image_for_args(rgb_image, prefix, args, interpolation="linear")
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
        depth_image = process_metric_depth_for_args(depth_image, prefix, args)
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


def _depth_rgb_to_u8(depth_rgb):
    array = np.asarray(depth_rgb)
    if array.ndim == 2:
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))
    if array.ndim == 3 and array.shape[-1] >= 1:
        return np.ascontiguousarray(array[..., 0].astype(np.uint8, copy=False))
    raise ValueError(f"Expected depth image with shape HxW or HxWxC, got {array.shape}.")


def maybe_dump_initial_depth_images(st, camera_images):
    out_dir = str(getattr(st.args, "dump_initial_depth_dir", "") or "").strip()
    max_frames = int(getattr(st.args, "dump_initial_depth_frames", 0) or 0)
    if not out_dir or max_frames <= 0:
        return
    if cv2 is None:
        raise RuntimeError("--dump_initial_depth_dir requires OpenCV/cv2 to write PNG files.")

    frame_idx = int(getattr(st, "_initial_depth_dump_count", 0))
    if frame_idx >= max_frames:
        return

    out_path = Path(out_dir).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    records = []
    for prefix in ("wrist", "front"):
        image = camera_images.get(f"{prefix}_masked_depth")
        if image is None:
            continue
        depth_u8 = _depth_rgb_to_u8(image)
        filename = f"env{int(st.index):02d}_frame{frame_idx:02d}_{prefix}_depth_u8.png"
        file_path = out_path / filename
        if not cv2.imwrite(str(file_path), depth_u8):
            raise RuntimeError(f"Failed to write depth image: {file_path}")
        records.append(
            {
                "env": int(st.index),
                "frame": int(frame_idx),
                "camera": prefix,
                "path": str(file_path),
                "shape": list(depth_u8.shape),
                "nonzero_pixels": int(np.count_nonzero(depth_u8)),
                "clip_lower_m": float(getattr(st.args, "camera_depth_clip_lower", 0.0)),
                "clip_far_m": float(getattr(st.args, "camera_depth_clip_far", 0.0)),
            }
        )

    if records:
        manifest_path = out_path / "manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"Dumped initial depth frame env={int(st.index)} frame={frame_idx} "
            f"cameras={[record['camera'] for record in records]} to {out_path}",
            flush=True,
        )
        st._initial_depth_dump_count = frame_idx + 1


def dp_image_inputs_from_cpu_cameras(camera_images, args):
    if args.rgb:
        return (
            camera_images.get("wrist_handle_mask"),
            camera_images.get("wrist_rgb"),
            camera_images.get("front_handle_mask"),
            camera_images.get("front_rgb"),
        )
    if bool(getattr(args, "depth_only", False)):
        wrist_depth = camera_images.get("wrist_masked_depth")
        front_depth = camera_images.get("front_masked_depth")
        wrist_empty = None if wrist_depth is None else np.zeros_like(wrist_depth, dtype=np.uint8)
        front_empty = None if front_depth is None else np.zeros_like(front_depth, dtype=np.uint8)
        return wrist_empty, wrist_depth, front_empty, front_depth
    return (
        camera_images.get("wrist_handle_mask"),
        camera_images.get("wrist_masked_depth"),
        camera_images.get("front_handle_mask"),
        camera_images.get("front_masked_depth"),
    )


def handle_bbox_from_mask_image(mask_image):
    """Return ([x0, y0, x1, y1], valid) from a rendered handle mask image.

    Coordinates use the half-open pixel convention [x0, y0, x1, y1).  The
    bbox is intentionally not filtered for minimum size here; visibility
    thresholds are applied when generating DINO handle-latent targets.
    """
    if mask_image is None:
        return np.zeros(4, dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    mask = np.asarray(mask_image)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        return np.zeros(4, dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    ys, xs = np.nonzero(mask > 0)
    if xs.size <= 0 or ys.size <= 0:
        return np.zeros(4, dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    bbox = np.asarray(
        [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        dtype=np.float32,
    )
    return bbox, np.asarray([1.0], dtype=np.float32)


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


FULL_BASE_ACTION_FRAMES = {
    "robot_base_full",
    "base_full",
    "arm_base",
    "robot_base",
    "true_base",
}


def normalize_float_dp_pose_frame(frame):
    frame = str(frame or "base").strip().lower()
    aliases = {
        "world_frame": "world",
        "base_yaw": "base",
        "yaw_base": "base",
        "yaw_only_base": "base",
        "full_base": "robot_base_full",
        "base_with_pitch": "robot_base_full",
        "arm_base_full": "robot_base_full",
    }
    return aliases.get(frame, frame)


def is_full_base_pose_frame(frame):
    return normalize_float_dp_pose_frame(frame) in FULL_BASE_ACTION_FRAMES


def base_frame_quat_np(yaw, base_pitch=0.0, pose_frame="base"):
    if is_full_base_pose_frame(pose_frame):
        return robot_base_quat_np(float(base_pitch), float(yaw))
    return base_ik.yaw_quat(float(yaw))


def world_pos_to_base(pos_world, base_xy, base_z, yaw, base_pitch=0.0, pose_frame="base"):
    rel = np.asarray(pos_world, dtype=np.float32) - base_position(base_xy, base_z)
    base_quat = base_frame_quat_np(yaw, base_pitch, pose_frame)
    return quat_apply(base_ik.quat_conjugate(base_quat), rel).astype(np.float32)


def base_pos_to_world(pos_base, base_xy, base_z, yaw, base_pitch=0.0, pose_frame="base"):
    base_quat = base_frame_quat_np(yaw, base_pitch, pose_frame)
    return (base_position(base_xy, base_z) + quat_apply(base_quat, pos_base)).astype(np.float32)


def world_quat_to_base(quat_world, yaw, base_pitch=0.0, pose_frame="base"):
    base_quat = base_frame_quat_np(yaw, base_pitch, pose_frame)
    return base_ik.normalize_quat(
        base_ik.quat_multiply(base_ik.quat_conjugate(base_quat), quat_world)
    ).astype(np.float32)


def base_quat_to_world(quat_base, yaw, base_pitch=0.0, pose_frame="base"):
    base_quat = base_frame_quat_np(yaw, base_pitch, pose_frame)
    return base_ik.normalize_quat(base_ik.quat_multiply(base_quat, quat_base)).astype(np.float32)


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
    base_pitch=0.0,
    pose_frame="base",
):
    dp_dof_pos, dp_dof_vel = map_float_dofs_to_dp(dof_names, dof_pos, dof_vel)
    base_roll_pitch = np.asarray([0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.asarray([0.0, 0.0, yaw_rate], dtype=np.float32)
    last_low_action = make_last_low_action_from_dp(last_dp_action)
    foot_contacts = np.zeros(4, dtype=np.float32)
    ee_base = world_pos_to_base(
        ee_pos,
        base_xy,
        base_z,
        yaw,
        base_pitch=base_pitch,
        pose_frame=pose_frame,
    )
    ee_quat_base = world_quat_to_base(
        base_ik.normalize_quat(ee_quat),
        yaw,
        base_pitch=base_pitch,
        pose_frame=pose_frame,
    )
    return np.concatenate(
        [
            base_roll_pitch,
            base_ang_vel,
            dp_dof_pos,
            dp_dof_vel,
            last_low_action,
            foot_contacts,
            ee_base,
            ee_quat_base,
            np.asarray([gripper], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def normalize_float_dp_state_mode(mode):
    mode = str(mode or FLOAT_DP_STATE_MODE_FULL).lower()
    aliases = {
        "legacy": FLOAT_DP_STATE_MODE_FULL,
        "full_state": FLOAT_DP_STATE_MODE_FULL,
        "pi0.5_current_state10": FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10,
        "pi05_current_10": FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10,
        "current_state10": FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10,
        "pi0.5_last_command_state10": FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10,
        "pi05_last_command_10": FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10,
        "last_command_state10": FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10,
        "a2w_joint_state9": FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9,
        "a2w_joint9": FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9,
        "joint_state9": FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9,
        "joint9": FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9,
        "last_command_joint_state9": FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9,
        "a2w_last_command_joint9": FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9,
    }
    mode = aliases.get(mode, mode)
    supported_modes = (
        FLOAT_DP_STATE_MODE_FULL,
        FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10,
        FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10,
        FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9,
    )
    if mode not in supported_modes:
        raise ValueError(
            f"Unsupported --dp_record_state_mode={mode!r}; expected "
            + ", ".join(repr(value) for value in supported_modes)
            + "."
        )
    return mode


def float_dp_state_feature_names(args, phase_names, make_state_feature_names_fn):
    state_mode = normalize_float_dp_state_mode(getattr(args, "dp_record_state_mode", FLOAT_DP_STATE_MODE_FULL))
    if state_mode == FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10:
        return list(PI05_CURRENT_STATE10_NAMES)
    if state_mode == FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10:
        return list(PI05_LAST_COMMAND_STATE10_NAMES)
    if state_mode == FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9:
        return list(A2W_LAST_COMMAND_JOINT_STATE9_NAMES)
    return make_state_feature_names_fn(DP_NUM_DOFS, DP_NUM_ACTIONS, phase_names)


def float_dp_state_mode_from_feature_names(state_feature_names):
    if list(state_feature_names or []) == PI05_CURRENT_STATE10_NAMES:
        return FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10
    if list(state_feature_names or []) == PI05_LAST_COMMAND_STATE10_NAMES:
        return FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10
    if list(state_feature_names or []) == A2W_LAST_COMMAND_JOINT_STATE9_NAMES:
        return FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9
    return FLOAT_DP_STATE_MODE_FULL


def float_dp_action_feature_names(args=None, state_mode=None):
    if state_mode is None:
        state_mode = normalize_float_dp_state_mode(getattr(args, "dp_record_state_mode", FLOAT_DP_STATE_MODE_FULL))
    else:
        state_mode = normalize_float_dp_state_mode(state_mode)
    if state_mode == FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9:
        return list(A2W_JOINT_ACTION9_NAMES)
    return list(FLOAT_DP_EE_ACTION10_NAMES)


def float_dp_action_mode_from_feature_names(action_feature_names):
    names = list(action_feature_names or [])
    if names == A2W_JOINT_ACTION9_NAMES:
        return "a2w_joint_action9"
    return "ee_action10"


def float_dp_action_is_a2w_joint9(action_feature_names):
    return float_dp_action_mode_from_feature_names(action_feature_names) == "a2w_joint_action9"


def a2w_joint_values_from_dofs(dof_names, dof_pos, joint_names=A2W_Z1_JOINT_NAMES):
    dof_names = list(dof_names or [])
    values = np.asarray(dof_pos, dtype=np.float32).reshape(-1)
    name_to_idx = {str(name): int(idx) for idx, name in enumerate(dof_names)}
    out = np.zeros(len(joint_names), dtype=np.float32)
    for dst_idx, joint_name in enumerate(joint_names):
        src_idx = name_to_idx.get(joint_name)
        if src_idx is not None and src_idx < values.shape[0]:
            out[dst_idx] = float(values[src_idx])
    return out


def make_pi05_current_state10(
    vx,
    yaw_rate,
    ee_pos,
    ee_quat,
    base_xy,
    base_z,
    yaw,
    gripper,
    base_pitch=0.0,
    pose_frame="base",
):
    ee_pos_base = world_pos_to_base(ee_pos, base_xy, base_z, yaw, base_pitch=base_pitch, pose_frame=pose_frame)
    ee_quat_base = world_quat_to_base(
        base_ik.normalize_quat(ee_quat),
        yaw,
        base_pitch=base_pitch,
        pose_frame=pose_frame,
    )
    return np.concatenate(
        [
            np.asarray([vx, yaw_rate], dtype=np.float32),
            ee_pos_base.reshape(3),
            ee_quat_base.reshape(4),
            np.asarray([gripper], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def make_pi05_last_command_state10(
    last_dp_action,
    ee_pos,
    ee_quat,
    base_xy,
    base_z,
    yaw,
    gripper,
    base_pitch=0.0,
    pose_frame="base",
):
    last_command = np.zeros(2, dtype=np.float32)
    if last_dp_action is not None:
        values = np.asarray(last_dp_action, dtype=np.float32).reshape(-1)
        last_command[: min(2, values.shape[0])] = values[:2]
    ee_pos_base = world_pos_to_base(ee_pos, base_xy, base_z, yaw, base_pitch=base_pitch, pose_frame=pose_frame)
    ee_quat_base = world_quat_to_base(
        base_ik.normalize_quat(ee_quat),
        yaw,
        base_pitch=base_pitch,
        pose_frame=pose_frame,
    )
    return np.concatenate(
        [
            last_command,
            ee_pos_base.reshape(3),
            ee_quat_base.reshape(4),
            np.asarray([gripper], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def make_a2w_last_command_joint_state9(last_dp_action, dof_names, dof_pos):
    last_command = np.zeros(2, dtype=np.float32)
    if last_dp_action is not None:
        values = np.asarray(last_dp_action, dtype=np.float32).reshape(-1)
        last_command[: min(2, values.shape[0])] = values[:2]
    return np.concatenate(
        [
            last_command,
            a2w_joint_values_from_dofs(dof_names, dof_pos),
        ],
        axis=0,
    ).astype(np.float32)


def make_float_dp_observation_state(
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
    vx=0.0,
    state_mode=FLOAT_DP_STATE_MODE_FULL,
    base_pitch=0.0,
    pose_frame="base",
):
    state_mode = normalize_float_dp_state_mode(state_mode)
    if state_mode == FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10:
        return make_pi05_current_state10(
            vx,
            yaw_rate,
            ee_pos,
            ee_quat,
            base_xy,
            base_z,
            yaw,
            gripper,
            base_pitch=base_pitch,
            pose_frame=pose_frame,
        )
    if state_mode == FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10:
        return make_pi05_last_command_state10(
            last_dp_action,
            ee_pos,
            ee_quat,
            base_xy,
            base_z,
            yaw,
            gripper,
            base_pitch=base_pitch,
            pose_frame=pose_frame,
        )
    if state_mode == FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9:
        return make_a2w_last_command_joint_state9(last_dp_action, dof_names, dof_pos)
    return make_float_dp_state(
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
        last_dp_action,
        base_pitch=base_pitch,
        pose_frame=pose_frame,
    )


def make_float_dp_action(
    vx,
    yaw_rate,
    target_pos,
    target_quat,
    gripper,
    base_xy,
    base_z,
    yaw,
    action_mode=None,
    dof_names=None,
    dof_pos=None,
    base_pitch=0.0,
    pose_frame="base",
):
    if str(action_mode or "").lower() == "a2w_joint_action9":
        action_names = A2W_JOINT_ACTION9_NAMES
    else:
        action_names = float_dp_action_feature_names(state_mode=action_mode)
    if float_dp_action_is_a2w_joint9(action_names):
        if dof_names is None or dof_pos is None:
            raise ValueError("A2W joint9 action recording requires dof_names and dof_pos.")
        return np.concatenate(
            [
                np.asarray([vx, yaw_rate], dtype=np.float32),
                a2w_joint_values_from_dofs(dof_names, dof_pos),
            ],
            axis=0,
        ).astype(np.float32)
    target_pos_base = world_pos_to_base(
        target_pos,
        base_xy,
        base_z,
        yaw,
        base_pitch=base_pitch,
        pose_frame=pose_frame,
    )
    target_quat_base = world_quat_to_base(
        base_ik.normalize_quat(target_quat),
        yaw,
        base_pitch=base_pitch,
        pose_frame=pose_frame,
    )
    return np.concatenate(
        [
            np.asarray([vx, yaw_rate], dtype=np.float32),
            target_pos_base.reshape(3),
            target_quat_base.reshape(4),
            np.asarray([gripper], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def apply_float_dp_action(action, base_xy, base_z, yaw, dt, action_frame="base", base_pitch=0.0):
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
    action_frame = normalize_float_dp_pose_frame(action_frame)
    if action_frame == "base":
        # Recorded float_ik actions store base velocity for prev->current, while
        # target pose is encoded in the current-frame base. Decode after applying
        # the base delta so action replay and policy rollout use the same frame.
        target_pos = base_pos_to_world(target_pos_action, base_xy_next, base_z, yaw_next)
        target_quat = base_quat_to_world(target_quat_action, yaw_next)
    elif is_full_base_pose_frame(action_frame):
        target_pos = base_pos_to_world(
            target_pos_action,
            base_xy_next,
            base_z,
            yaw_next,
            base_pitch=base_pitch,
            pose_frame=action_frame,
        )
        target_quat = base_quat_to_world(
            target_quat_action,
            yaw_next,
            base_pitch=base_pitch,
            pose_frame=action_frame,
        )
    elif action_frame == "world":
        target_pos = target_pos_action
        target_quat = target_quat_action
    else:
        raise ValueError(
            f"Unsupported float_ik action_frame={action_frame!r}; "
            "expected 'base', 'robot_base_full', or 'world'."
        )
    gripper = float(action[9])
    return base_xy_next.astype(np.float32), yaw_next, target_pos, target_quat, gripper


def apply_float_dp_joint_action9(action, base_xy, yaw, dt):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < len(A2W_JOINT_ACTION9_NAMES):
        raise ValueError(
            f"A2W joint action must have at least {len(A2W_JOINT_ACTION9_NAMES)} values, got shape {action.shape}"
        )
    vx = float(action[0])
    yaw_rate = float(action[1])
    yaw_next = float(yaw) + yaw_rate * float(dt)
    heading = np.asarray([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)
    base_xy_next = np.asarray(base_xy, dtype=np.float32) + heading * (vx * float(dt))
    joint_targets = {
        joint_name: float(action[2 + joint_idx])
        for joint_idx, joint_name in enumerate(A2W_Z1_JOINT_NAMES)
    }
    return base_xy_next.astype(np.float32), yaw_next, joint_targets


def set_a2w_joint_targets_from_action(dof_positions, dof_names, joint_targets, lower=None, upper=None):
    name_to_idx = {str(name): int(idx) for idx, name in enumerate(list(dof_names or []))}
    if lower is not None and hasattr(lower, "detach"):
        lower = lower.detach().cpu().numpy()
    if upper is not None and hasattr(upper, "detach"):
        upper = upper.detach().cpu().numpy()
    for joint_name, value in dict(joint_targets or {}).items():
        idx = name_to_idx.get(str(joint_name))
        if idx is None or idx >= len(dof_positions):
            continue
        target = float(value)
        if lower is not None and upper is not None and idx < len(lower) and idx < len(upper):
            target = float(np.clip(target, float(lower[idx]), float(upper[idx])))
        dof_positions[idx] = target
    return dof_positions


def _round_list(value, precision=5):
    return np.round(np.asarray(value, dtype=np.float64), precision).tolist()


def make_float_dp_policy_log_record(
    step,
    st,
    dp_action,
    dp_state,
    ee_pos,
    ee_quat,
    door_pos,
    door_vel,
    phase,
    action_names=None,
    camera_gates=None,
    interaction_state=None,
    gym=None,
    dof_names=None,
):
    action_frame = str(getattr(st, "dp_action_frame", "base"))
    action_names = list(action_names or [])
    action_mode = float_dp_action_mode_from_feature_names(action_names)
    if action_mode == "a2w_joint_action9":
        ee_record = {
            "target_pos_world": _round_list(st.last_target_pos),
            "target_pos_action": [],
            "target_quat": None if st.last_target_quat is None else _round_list(st.last_target_quat),
            "target_quat_action": [],
            "actual_pos_world": _round_list(ee_pos),
            "actual_quat": _round_list(ee_quat),
        }
        joint_targets = _round_list(dp_action[2:9])
    else:
        ee_record = {
            "target_pos_world": _round_list(st.last_target_pos),
            "target_pos_action": _round_list(dp_action[2:5]),
            "target_quat": None if st.last_target_quat is None else _round_list(st.last_target_quat),
            "target_quat_action": _round_list(dp_action[5:9]),
            "actual_pos_world": _round_list(ee_pos),
            "actual_quat": _round_list(ee_quat),
        }
        joint_targets = []
    record = {
        "step": int(step),
        "controlled_env_id": int(st.index),
        "num_envs": int(st.args.num_envs),
        "phase_name": str(phase),
        "dp_action_names": action_names,
        "action_frame": action_frame,
        "action_mode": action_mode,
        "dp_action_raw": _round_list(dp_action),
        "state": _round_list(dp_state),
        "base": {
            "xy": _round_list(st.traj.get("base_xy", st.base_start)),
            "yaw": float(st.traj.get("yaw", st.yaw_start)),
        },
        "ee": ee_record,
        "joint_targets": joint_targets,
        "gripper": {"target": float(st.last_gripper)},
        "door": {"dof": _round_list(door_pos) if door_pos is not None else []},
    }
    if ee_record["target_pos_world"] and ee_record["actual_pos_world"]:
        record["ee"]["pos_error"] = _round_list(
            np.asarray(ee_record["target_pos_world"], dtype=np.float32)
            - np.asarray(ee_record["actual_pos_world"], dtype=np.float32)
        )
    randomization_json = str(getattr(st.args, "ikpush_randomization_json", "") or "").strip()
    if randomization_json:
        try:
            record["ikpush_randomization"] = json.loads(randomization_json)
        except Exception:
            record["ikpush_randomization_json"] = randomization_json
    if bool(getattr(st.args, "dp_log_replay_snapshot", False)):
        dof_pos = np.asarray(getattr(st, "dof_positions", []), dtype=np.float32).reshape(-1)
        dof_vel = np.zeros_like(dof_pos, dtype=np.float32)
        if gym is not None:
            actual_dof_state = gym.get_actor_dof_states(st.env, st.arm_actor, gymapi.STATE_ALL)
            dof_pos = np.asarray(actual_dof_state["pos"], dtype=np.float32).copy()
            dof_vel = np.asarray(actual_dof_state["vel"], dtype=np.float32).copy()
        door_pos_arr = np.asarray([] if door_pos is None else door_pos, dtype=np.float32).reshape(-1)
        door_vel_arr = np.zeros_like(door_pos_arr, dtype=np.float32)
        if door_vel is not None:
            door_vel_arr = np.asarray(door_vel, dtype=np.float32).reshape(-1)
        base_xy = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32)
        yaw = float(st.traj.get("yaw", st.yaw_start))
        snapshot = make_float_replay_snapshot(
            st.args,
            st.door,
            list(dof_names or []),
            dof_pos,
            dof_vel,
            door_pos_arr,
            door_vel_arr,
            ee_pos,
            ee_quat,
            base_xy,
            yaw,
            float(dp_action[0]) if len(dp_action) > 0 else 0.0,
            float(dp_action[1]) if len(dp_action) > 1 else 0.0,
        )
        record["sim_snapshot"] = {key: _round_list(value) for key, value in snapshot.items()}
        if gym is not None:
            if getattr(st, "camera_handles", None):
                front_pose_base, wrist_pose_base = float_camera_pose_base(gym, st)
                record["sim_snapshot"]["front_camera_pose_base"] = _round_list(front_pose_base)
                record["sim_snapshot"]["wrist_camera_pose_base"] = _round_list(wrist_pose_base)
            record["extra"] = {
                key: _round_list(value)
                for key, value in gripper_handle_contact_snapshot(gym, st).items()
            }
    if camera_gates is not None:
        gates = np.asarray(camera_gates, dtype=np.float32).reshape(-1)
        if gates.size >= 2:
            record["camera_gates"] = {
                "front": float(gates[0]),
                "wrist": float(gates[1]),
                "sum": float(gates[0] + gates[1]),
            }
    if interaction_state is not None:
        values = np.asarray(interaction_state, dtype=np.float32).reshape(-1)
        if values.size >= 3:
            record["interaction_contact_probability"] = float(values[0])
            record["interaction_handle_progress"] = float(values[1])
            record["interaction_door_progress"] = float(values[2])
            record["interaction_state"] = {
                "contact_probability": float(values[0]),
                "handle_progress": float(values[1]),
                "door_progress": float(values[2]),
            }
    if bool(getattr(st.args, "dp_end_signal_monitor", False)):
        end_probability = float(getattr(st, "dp_end_probability", 0.0))
        end_consecutive_count = int(getattr(st, "dp_end_consecutive_count", 0))
        end_triggered = bool(getattr(st, "dp_end_triggered", False))
        first_end_trigger_step = getattr(st, "dp_first_end_trigger_step", None)
        door_angle_at_end_trigger = getattr(st, "dp_door_angle_at_end_trigger", None)
        phase_at_end_trigger = getattr(st, "dp_phase_at_end_trigger", None)
        record.update(
            {
                "end_probability": end_probability,
                "end_consecutive_count": end_consecutive_count,
                "end_triggered": end_triggered,
                "first_end_trigger_step": first_end_trigger_step,
                "door_angle_at_end_trigger": door_angle_at_end_trigger,
                "phase_at_end_trigger": phase_at_end_trigger,
            }
        )
        record["end_signal"] = {
            "probability": end_probability,
            "consecutive_count": end_consecutive_count,
            "triggered": end_triggered,
            "first_trigger_step": first_end_trigger_step,
            "door_angle_deg_at_trigger": door_angle_at_end_trigger,
            "phase_at_trigger": phase_at_end_trigger,
        }
    temporal_meta = getattr(st, "dp_temporal_last_action_meta", None)
    if temporal_meta is not None:
        record["temporal_ensemble"] = temporal_meta
    return record


def update_float_dp_end_signal_monitor(st, policy_output, step, door_pos, phase):
    """Advance one env's sticky end detector for one actually executed policy step."""
    if not bool(getattr(st.args, "dp_end_signal_monitor", False)):
        return
    output = np.asarray(policy_output, dtype=np.float32).reshape(-1)
    if output.size <= 10:
        raise ValueError("--dp_end_signal_monitor requires a policy output with end probability at index 10.")
    probability = float(np.clip(output[10], 0.0, 1.0))
    threshold = float(getattr(st.args, "dp_end_signal_threshold", 0.8))
    required = max(1, int(getattr(st.args, "dp_end_signal_consecutive_steps", 10)))
    count = int(getattr(st, "dp_end_consecutive_count", 0))
    count = count + 1 if probability > threshold else 0
    st.dp_end_probability = probability
    st.dp_end_consecutive_count = count
    if bool(getattr(st, "dp_end_triggered", False)) or count < required:
        return
    st.dp_end_triggered = True
    st.dp_first_end_trigger_step = int(step)
    door_values = np.asarray([] if door_pos is None else door_pos, dtype=np.float32).reshape(-1)
    st.dp_door_angle_at_end_trigger = (
        None if door_values.size == 0 else float(np.degrees(float(door_values[0])))
    )
    st.dp_phase_at_end_trigger = str(phase)


def print_float_dp_policy_log_record(record):
    action = record["dp_action_raw"]
    ee = record["ee"]
    if record.get("action_mode") == "a2w_joint_action9":
        print(
            "[DoorDP-FloatIK]"
            f" step={record['step']}"
            f" env={record['controlled_env_id']}/{record['num_envs']}"
            f" phase={record['phase_name']}"
            f" action_frame={record.get('action_frame')}"
            f" action(vx,yaw,joints)=({action[0]:.3f}, {action[1]:.3f}, {record.get('joint_targets', [])})"
            f" ee_actual={ee['actual_pos_world']}",
            flush=True,
        )
        return
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
    root_state[3:7] = robot_base_quat_np(getattr(args, "robot_pitch", 0.0), float(yaw))
    root_state[7:10] = np.asarray([vx * math.cos(float(yaw)), vx * math.sin(float(yaw)), 0.0], dtype=np.float32)
    root_state[10:13] = np.asarray([0.0, 0.0, yaw_rate], dtype=np.float32)

    door_root_state = np.zeros(13, dtype=np.float32)
    actor_offset = door.actor_position_offset
    door_root_state[:3] = np.asarray(
        [
            args.door_x + actor_offset[0],
            args.door_y + actor_offset[1],
            -float(door.bounding["min"][2]) * door.actor_scale + args.door_z_offset + actor_offset[2],
        ],
        dtype=np.float32,
    )
    door_root_state[3:7] = base_ik.yaw_quat(float(door.actor_yaw))
    return {
        "replay_root_state": root_state,
        "replay_dof_pos": dp_dof_pos,
        "replay_dof_vel": dp_dof_vel,
        "replay_ee_pos": np.asarray(ee_pos, dtype=np.float32).copy(),
        "replay_ee_quat": base_ik.normalize_quat(ee_quat).astype(np.float32),
        "replay_door_root_state": door_root_state,
        "replay_door_dof_pos": np.asarray(door_pos, dtype=np.float32).copy(),
        "replay_door_dof_vel": np.asarray(door_vel, dtype=np.float32).copy(),
        "replay_door_open_stage": np.asarray([float(bool(door.open_stage))], dtype=np.float32),
    }


def float_dp_vision_mode(args, normalize_vision_mode_fn=None):
    if bool(getattr(args, "rgb", False)):
        vision_mode = "rgb"
    elif bool(getattr(args, "depth_only", False)):
        vision_mode = "depth_only"
    else:
        vision_mode = "depth"
    if normalize_vision_mode_fn is not None:
        vision_mode = normalize_vision_mode_fn(vision_mode)
    return vision_mode


def float_dp_record_env_ids(args):
    return set(range(int(args.num_envs))) if bool(getattr(args, "dp_record_all_envs", False)) else {int(args.dp_record_env_id)}


def require_float_dp_recording_deps(args, raw_recorder_cls, make_state_feature_names_fn):
    if getattr(args, "record_dp_dataset", False) and (raw_recorder_cls is None or make_state_feature_names_fn is None):
        raise RuntimeError("DP raw recording requires high-level/dp/door_dp_common.py.")


def float_dp_camera_sample_stride(args):
    dp_fps = float(getattr(args, "dp_fps", 25))
    camera_fps = float(getattr(args, "camera_fps", 25.0))
    if dp_fps <= 0.0:
        raise ValueError("--dp_fps must be positive")
    if camera_fps <= 0.0:
        raise ValueError("--camera_fps must be positive")
    return max(1, int(round(dp_fps / camera_fps)))


def float_dp_camera_effective_fps(args):
    return float(getattr(args, "dp_fps", 25)) / float(float_dp_camera_sample_stride(args))


def float_dp_record_sample_stride(args, dt):
    dp_fps = float(getattr(args, "dp_fps", 25))
    if dp_fps <= 0.0:
        raise ValueError("--dp_fps must be positive")
    if dt <= 0.0:
        return 1
    sim_fps = 1.0 / float(dt)
    return max(1, int(round(sim_fps / dp_fps)))


def float_dp_record_effective_fps(args, dt):
    if dt <= 0.0:
        return float(getattr(args, "dp_fps", 25))
    return (1.0 / float(dt)) / float(float_dp_record_sample_stride(args, dt))


def float_dp_record_dt(args, dt):
    return float(dt) * float(float_dp_record_sample_stride(args, dt))


def float_dp_policy_sample_stride(args, dt):
    return float_dp_record_sample_stride(args, dt)


def float_dp_policy_effective_fps(args, dt):
    return float_dp_record_effective_fps(args, dt)


def float_dp_policy_update_due(step, args, dt):
    stride = float_dp_policy_sample_stride(args, dt)
    return int(step) % int(stride) == 0


def float_dp_record_frame_due(st, dt):
    if getattr(st, "dp_recorder", None) is None:
        return False
    stride = float_dp_record_sample_stride(st.args, dt)
    return int(getattr(st, "dp_record_sim_steps", 0)) % int(stride) == 0


def float_dp_camera_images_for_record_frame(gym, sim, st):
    stride = float_dp_camera_sample_stride(st.args)
    vision_mode = float_dp_vision_mode(st.args)
    should_capture = (
        getattr(st, "last_wrist_mask_rgb", None) is None
        or getattr(st, "last_wrist_second_rgb", None) is None
        or (int(st.dp_recorder.frame_count) % stride) == 0
    )
    if should_capture:
        camera_images = capture_dp_camera_images_from_rendered(gym, sim, st.env, st.camera_handles, st.args)
        if bool(getattr(st.args, "record_handle_bbox", False)):
            wrist_bbox, wrist_bbox_valid = handle_bbox_from_mask_image(camera_images.get("wrist_handle_mask"))
            front_bbox, front_bbox_valid = handle_bbox_from_mask_image(camera_images.get("front_handle_mask"))
            st.last_wrist_handle_bbox = wrist_bbox.copy()
            st.last_wrist_handle_bbox_valid = wrist_bbox_valid.copy()
            st.last_front_handle_bbox = front_bbox.copy()
            st.last_front_handle_bbox_valid = front_bbox_valid.copy()
        wrist_mask_rgb, wrist_second_rgb, front_mask_rgb, front_second_rgb = dp_image_inputs_from_cpu_cameras(
            camera_images, st.args
        )
        if vision_mode == "rgb":
            missing_required_camera = (
                wrist_mask_rgb is None
                or wrist_second_rgb is None
                or front_mask_rgb is None
                or front_second_rgb is None
            )
        elif vision_mode == "depth_only":
            missing_required_camera = wrist_second_rgb is None or front_second_rgb is None
        else:
            missing_required_camera = wrist_mask_rgb is None or wrist_second_rgb is None
        if missing_required_camera:
            if getattr(st, "last_wrist_mask_rgb", None) is None or getattr(st, "last_wrist_second_rgb", None) is None:
                return None, None, None, None
            return (
                st.last_wrist_mask_rgb,
                st.last_wrist_second_rgb,
                getattr(st, "last_front_mask_rgb", None),
                getattr(st, "last_front_second_rgb", None),
            )
        st.last_wrist_mask_rgb = np.asarray(wrist_mask_rgb, dtype=np.uint8).copy()
        st.last_wrist_second_rgb = np.asarray(wrist_second_rgb, dtype=np.uint8).copy()
        st.last_front_mask_rgb = None if front_mask_rgb is None else np.asarray(front_mask_rgb, dtype=np.uint8).copy()
        st.last_front_second_rgb = (
            None if front_second_rgb is None else np.asarray(front_second_rgb, dtype=np.uint8).copy()
        )
    return (
        st.last_wrist_mask_rgb,
        st.last_wrist_second_rgb,
        getattr(st, "last_front_mask_rgb", None),
        getattr(st, "last_front_second_rgb", None),
    )


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
    state_mode = normalize_float_dp_state_mode(getattr(args, "dp_record_state_mode", FLOAT_DP_STATE_MODE_FULL))
    action_names = float_dp_action_feature_names(state_mode=state_mode)
    pose_frame = normalize_float_dp_pose_frame(getattr(args, "ee_pose_frame", "base"))
    records_tracik_command_fk = (
        str(getattr(args, "arm_ik_solver", "gym_jacobian")) == "tracik"
        and not float_dp_action_is_a2w_joint9(action_names)
    )
    if state_mode == FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10:
        state_source = "current_vx_yaw_rate_ee_base_gripper"
    elif state_mode == FLOAT_DP_STATE_MODE_PI05_LAST_COMMAND_STATE10:
        state_source = "last_command_vx_vyaw_ee_base_gripper"
    elif state_mode == FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9:
        state_source = "last_command_vx_vyaw_a2w_z1_joint_positions"
    else:
        state_source = "full_float_ik_state_with_previous_action"
    metadata = {
        "door_asset_index": int(getattr(door, "asset_index", 0)),
        "door_asset_name": door.spec.get("name", ""),
        "door_asset_path": door.spec.get("path", ""),
        "door_actor_scale": float(door.actor_scale),
        "door_actor_yaw": float(door.actor_yaw),
        "door_actor_position_offset": list(door.actor_position_offset),
        "door_motion_sign": float(args.door_motion_sign),
        "door_cfg": str(args.door_cfg),
        "source_script": Path(sys.argv[0]).name,
        "action_frame": pose_frame,
        "action_pose_frame": pose_frame,
        "target_pose_frame": pose_frame,
        "state_pose_frame": pose_frame,
        "ee_pose_frame": pose_frame,
        "ikpush_state_version": str(state_version),
        "door_dp_mode": str(mode_name),
        "controller_mode": str(mode_name),
        "parallel_env_id": int(env_index),
        "parallel_num_envs": int(args.num_envs),
        "seed": int(getattr(args, "seed", -1)),
        "env_seed": int(getattr(args, "env_seed", -1)),
        "state_format": state_mode,
        "state_source": state_source,
        "action_format": float_dp_action_mode_from_feature_names(action_names),
        "action_source": (
            "base_command_plus_a2w_z1_joint_targets"
            if float_dp_action_is_a2w_joint9(action_names)
            else (
                "base_command_plus_tracik_smoothed_command_fk"
                if records_tracik_command_fk
                else "base_command_plus_ee_target_pose"
            )
        ),
        "tracik_smoothed_action_recording": records_tracik_command_fk,
        "state_normalized": False,
        "pi05_state_action_aligned": state_mode == FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10,
        "camera_fps": float_dp_camera_effective_fps(args),
        "camera_sample_stride": int(float_dp_camera_sample_stride(args)),
        "camera_hold_last_frame": True,
        "door_side_walls": bool(door_side_walls_enabled(args)),
        "door_use_urdf_rgba": bool(getattr(args, "door_use_urdf_rgba", False)),
        "phase_names": list(phase_names),
        "keyframe_loss_enabled": not bool(getattr(args, "no_keyframe_loss_weights", False)),
        "keyframe_loss_weight": float(getattr(args, "keyframe_loss_weight", 8.0)),
        "keyframe_loss_radius": int(getattr(args, "keyframe_loss_radius", 3)),
        "keyframe_loss_feature": "loss.action_weight",
        "keyframe_extraction_rules": {
            "start": "first recorded frame",
            "stop_before_door": "first initial_hold frame",
            "pregrasp": "first grasp frame",
            "grasp": "first grasp_hold/close_gripper frame",
            "rotate": "first push_door frame, i.e. after rotate_handle completes",
        },
        "handle_closed_angle": float(door.dof_lower[1]) if len(door.dof_lower) > 1 else 0.0,
        "door_closed_angle": (
            float(door.dof_upper[0])
            if len(door.dof_upper) > 0 and float(args.door_motion_sign) < 0.0
            else (float(door.dof_lower[0]) if len(door.dof_lower) > 0 else 0.0)
        ),
        "handle_unlock_threshold": float(getattr(door, "handle_unlock_threshold", math.radians(40.0))),
        "door_progress_goal_angle": math.radians(90.0),
    }
    if bool(getattr(args, "record_end_signal", False)):
        metadata.update(
            {
                "end_signal_enabled": True,
                "end_signal_feature": "aux.end_signal",
                "end_signal_positive_phases": parse_csv_names(
                    getattr(args, "end_signal_positive_phases", "return_home,hold_home"),
                    ["return_home", "hold_home"],
                ),
                "end_signal_version": "phase_dense_v1",
            }
        )
    if bool(getattr(args, "record_camera_pose", False)):
        metadata.update(camera_intrinsics_recording_metadata(args))
        metadata.update(
            {
                "camera_pose_frame": "robot_base",
                "camera_pose_convention": "optical_frame",
                "camera_pose_features": [
                    "observation.camera_pose.front",
                    "observation.camera_pose.wrist",
                ],
            }
        )
    if bool(getattr(args, "record_handle_bbox", False)):
        metadata.update(
            {
                "handle_bbox_features": [
                    "front_handle_bbox_xyxy",
                    "front_handle_bbox_valid",
                    "wrist_handle_bbox_xyxy",
                    "wrist_handle_bbox_valid",
                ],
                "handle_bbox_convention": "xyxy_half_open_pixels",
            }
        )
    if bool(getattr(args, "record_gripper_handle_contact", False)):
        metadata.update(
            {
                "gripper_handle_contact_features": [
                    "gripper_handle_contact_score",
                    "gripper_handle_contact_count",
                    "gripper_handle_contact_any",
                    "gripper_handle_contact_both",
                ],
                "gripper_handle_contact_gripper_bodies": parse_csv_names(
                    getattr(args, "gripper_handle_contact_gripper_bodies", ""),
                    ["gripperStator", "gripperMover"],
                ),
                "gripper_handle_contact_handle_bodies": parse_csv_names(
                    getattr(args, "gripper_handle_contact_handle_bodies", ""),
                    [],
                ),
                "gripper_handle_contact_score_threshold": float(
                    getattr(args, "gripper_handle_contact_score_threshold", 1.0e-6)
                ),
                "filter_gripper_handle_contact": bool(getattr(args, "filter_gripper_handle_contact", False)),
                "filter_gripper_handle_contact_min_frames": int(
                    getattr(args, "filter_gripper_handle_contact_min_frames", 5)
                ),
                "filter_gripper_handle_contact_require_both": bool(
                    getattr(args, "filter_gripper_handle_contact_require_both", True)
                ),
                "filter_gripper_handle_contact_phase_names": parse_csv_names(
                    getattr(args, "filter_gripper_handle_contact_phase_names", ""),
                    ["close_gripper", "rotate_handle"],
                ),
            }
        )
    metadata.update(depth_camera_randomization_metadata(args))
    sim_dt = getattr(args, "sim_dt", None)
    if sim_dt is not None:
        sim_dt = float(sim_dt)
        metadata.update(
            {
                "sim_dt": sim_dt,
                "sim_fps": 1.0 / sim_dt if sim_dt > 0.0 else 0.0,
                "record_sample_stride": int(float_dp_record_sample_stride(args, sim_dt)),
                "record_effective_fps": float_dp_record_effective_fps(args, sim_dt),
            }
        )
    for name in (
        "door_x",
        "door_y",
        "door_z_offset",
        "door_wall_height",
        "door_wall_opening_width",
        "door_wall_opening_clearance",
        "door_wall_side_width",
        "door_wall_thickness",
        "door_wall_gap",
        "door_wall_x_offset",
        "door_wall_y_offset",
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
        "door_open_resistance",
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
        state_feature_names=float_dp_state_feature_names(args, phase_names, make_state_feature_names_fn),
        action_feature_names=action_names,
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
        f"vision_mode={vision_mode} "
        f"state_mode={normalize_float_dp_state_mode(getattr(args, 'dp_record_state_mode', FLOAT_DP_STATE_MODE_FULL))} "
        f"action_mode={float_dp_action_mode_from_feature_names(float_dp_action_feature_names(args))} "
        f"record_fps={float(getattr(args, 'dp_fps', 25)):.2f} "
        f"camera_fps={float_dp_camera_effective_fps(args):.2f}",
        flush=True,
    )


def shortest_path_slerp_xyzw(old_quat_xyzw, new_quat_xyzw, new_weight):
    """SLERP from old to new using the shortest quaternion arc."""
    old = np.asarray(old_quat_xyzw, dtype=np.float64).reshape(4)
    new = np.asarray(new_quat_xyzw, dtype=np.float64).reshape(4)
    old_norm = float(np.linalg.norm(old))
    new_norm = float(np.linalg.norm(new))
    if not np.isfinite(old_norm) or old_norm < 1.0e-9:
        old = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        old /= old_norm
    if not np.isfinite(new_norm) or new_norm < 1.0e-9:
        new = old.copy()
    else:
        new /= new_norm

    dot = float(np.dot(old, new))
    if dot < 0.0:
        new = -new
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))
    t = float(np.clip(new_weight, 0.0, 1.0))
    if dot > 0.9995:
        blended = (1.0 - t) * old + t * new
    else:
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        blended = (
            np.sin((1.0 - t) * theta) / sin_theta * old
            + np.sin(t * theta) / sin_theta * new
        )
    blended /= max(float(np.linalg.norm(blended)), 1.0e-9)
    if blended[3] < 0.0:
        blended = -blended
    return blended.astype(np.float32)


def blend_float_dp_ee_actions(old_action, new_action, old_weight=0.3, new_weight=0.7):
    """Blend overlapping EE actions and preserve optional auxiliary outputs."""
    old = np.asarray(old_action, dtype=np.float32).reshape(-1)
    new = np.asarray(new_action, dtype=np.float32).reshape(-1)
    if old.shape[0] < 10 or new.shape[0] < 10:
        raise ValueError(f"EE action blending requires 10D actions, got {old.shape} and {new.shape}.")
    total = float(old_weight) + float(new_weight)
    if total <= 0.0:
        raise ValueError("EE action blend weights must have a positive sum.")
    old_alpha = float(old_weight) / total
    new_alpha = float(new_weight) / total
    if old.shape[0] != new.shape[0]:
        raise ValueError(f"Overlapping policy outputs must have equal dimensions, got {old.shape} and {new.shape}.")
    blended = new.copy()
    blended[0:5] = old_alpha * old[0:5] + new_alpha * new[0:5]
    blended[5:9] = shortest_path_slerp_xyzw(old[5:9], new[5:9], new_alpha)
    # Gripper follows the newest chunk directly; blending delays grasp/release.
    blended[9] = new[9]
    # Auxiliary outputs (currently end probability) use the same temporal
    # weighting as continuous motion values.
    if blended.shape[0] > 10:
        blended[10:] = old_alpha * old[10:] + new_alpha * new[10:]
    return blended


@dataclass
class FloatDPTimedAction:
    timestep: int
    action: np.ndarray
    source: str = "chunk"
    blend_count: int = 1


class FloatDPActionOverlapBuffer:
    """Timestamped EE action buffer with NX-style overlap aggregation."""

    def __init__(self, old_weight=0.3, new_weight=0.7):
        self.old_weight = float(old_weight)
        self.new_weight = float(new_weight)
        self._queue = deque()
        self.last_popped_timestep = -1

    @property
    def queue_size(self):
        return len(self._queue)

    @property
    def first_timestep(self):
        return None if not self._queue else int(self._queue[0].timestep)

    @property
    def last_timestep(self):
        return None if not self._queue else int(self._queue[-1].timestep)

    def ingest(self, actions, start_timestep, current_timestep):
        rows = np.asarray(actions, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] < 10:
            raise ValueError(f"Expected action chunk with shape (T, >=10), got {rows.shape}.")
        future = {
            int(item.timestep): item
            for item in self._queue
            if int(item.timestep) > self.last_popped_timestep
        }
        stale_skipped = 0
        overlap_blended = 0
        appended = 0
        for offset, row in enumerate(rows):
            timestep = int(start_timestep) + int(offset)
            if timestep < int(current_timestep) or timestep <= self.last_popped_timestep:
                stale_skipped += 1
                continue
            if timestep in future:
                old_item = future[timestep]
                future[timestep] = FloatDPTimedAction(
                    timestep=timestep,
                    action=blend_float_dp_ee_actions(
                        old_item.action,
                        row,
                        old_weight=self.old_weight,
                        new_weight=self.new_weight,
                    ),
                    source="overlap_0.3_old_0.7_new",
                    blend_count=int(old_item.blend_count) + 1,
                )
                overlap_blended += 1
            else:
                future[timestep] = FloatDPTimedAction(
                    timestep=timestep,
                    action=np.asarray(row, dtype=np.float32).copy(),
                    source="new_chunk",
                    blend_count=1,
                )
                appended += 1
        self._queue = deque(sorted(future.values(), key=lambda item: item.timestep))
        return {
            "start_timestep": int(start_timestep),
            "ingest_timestep": int(current_timestep),
            "stale_skipped": int(stale_skipped),
            "overlap_blended": int(overlap_blended),
            "appended": int(appended),
            "queue_size": self.queue_size,
            "queue_first_timestep": self.first_timestep,
            "queue_last_timestep": self.last_timestep,
            "old_weight": self.old_weight,
            "new_weight": self.new_weight,
        }

    def pop(self, expected_timestep):
        if not self._queue:
            raise RuntimeError("Float DP action overlap buffer is empty.")
        item = self._queue.popleft()
        if int(item.timestep) != int(expected_timestep):
            raise RuntimeError(
                f"Float DP action timeline mismatch: expected step {expected_timestep}, got {item.timestep}."
            )
        self.last_popped_timestep = int(item.timestep)
        return item


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
    checkpoint_config = getattr(dp_controller, "config", {}) or {}
    if (
        bool(checkpoint_config.get("plucker_conditioning", False))
        and checkpoint_config.get("plucker_intrinsics_mode", "legacy_shared_fov") == "per_camera"
    ):
        if camera_intrinsics_mode_from_args(args) != CAMERA_INTRINSICS_REAL_K_REMAP_MODE:
            raise ValueError(
                "This checkpoint uses calibrated per-camera Plücker rays. Run play/eval with "
                "--camera_intrinsics_mode real_k_remap and the matching intrinsics config."
            )
        runtime_intrinsics = depth_camera_intrinsics_metadata(args)
        mismatches = []
        for camera_name in ("front", "wrist"):
            for field in ("fx", "fy", "cx", "cy"):
                expected = float(checkpoint_config.get(f"plucker_{camera_name}_{field}", float("nan")))
                actual = float(runtime_intrinsics[camera_name][field])
                if not math.isfinite(expected) or abs(expected - actual) > 1.0e-6:
                    mismatches.append(
                        f"{camera_name}.{field}: checkpoint={expected}, runtime={actual}"
                    )
        if mismatches:
            raise ValueError("Checkpoint/runtime camera intrinsics mismatch: " + "; ".join(mismatches))
    if bool(getattr(args, "dp_end_signal_monitor", False)) and not bool(
        getattr(dp_controller, "end_signal_prediction", False)
    ):
        raise ValueError(
            "--dp_end_signal_monitor was enabled, but the loaded checkpoint has no autonomous end-signal head."
        )
    expected_vision_mode = "rgb" if args.rgb else "depth"
    if bool(getattr(args, "depth_only", False)) and not args.rgb:
        expected_vision_mode = "depth_only"
    if getattr(dp_controller, "vision_mode", "depth") != expected_vision_mode:
        raise ValueError(
            f"DP checkpoint vision_mode={getattr(dp_controller, 'vision_mode', 'depth')!r}, "
            f"but {mode_name} play was run with {expected_vision_mode!r}."
        )
    controller_action_frame = normalize_float_dp_pose_frame(getattr(dp_controller, "action_frame", "world"))
    if (
        controller_action_frame not in ("world", "base", "joint_command")
        and not is_full_base_pose_frame(controller_action_frame)
    ):
        raise ValueError(
            f"DP checkpoint action_frame={getattr(dp_controller, 'action_frame', None)!r}; "
            "expected 'world', 'base', 'robot_base_full', or 'joint_command'."
        )
    checkpoint_state_version = str(dp_controller.config.get("ikpush_state_version", "legacy"))
    controller_state_mode = float_dp_state_mode_from_feature_names(
        getattr(dp_controller, "state_feature_names", [])
    )
    joint_state_checkpoint = bool(
        checkpoint_state_version == FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9
        and controller_state_mode == FLOAT_DP_STATE_MODE_A2W_LAST_COMMAND_JOINT_STATE9
    )
    if checkpoint_state_version != str(state_version) and not joint_state_checkpoint:
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
        controlled_state.dp_action_frame = controller_action_frame
        if bool(getattr(args, "dp_temporal_ensemble", False)):
            controlled_state.dp_temporal_action_buffer = FloatDPActionOverlapBuffer(
                old_weight=float(getattr(args, "dp_temporal_old_weight", 0.3)),
                new_weight=float(getattr(args, "dp_temporal_new_weight", 0.7)),
            )
            controlled_state.dp_temporal_timestep = 0
            controlled_state.dp_temporal_warned_fallback = False
        if not controlled_state.camera_handles:
            raise RuntimeError(f"{mode_name} DP policy execution requires camera sensors.")
        if bool(getattr(dp_controller, "pointcloud_conditioning", False)):
            pointcloud_mode = str(
                getattr(dp_controller, "config", {}).get("pointcloud_mode", "single_front")
            )
            needs_front = pointcloud_mode in ("single_front", "dual_view", "dual_fused")
            needs_wrist = pointcloud_mode in ("single_wrist", "dual_view", "dual_fused")
            if needs_front and "front" not in controlled_state.camera_handles:
                raise RuntimeError(
                    "This point-cloud policy requires the front depth camera; enable --enable_front_camera.")
            if needs_wrist and "wrist" not in controlled_state.camera_handles:
                raise RuntimeError(
                    "This point-cloud policy requires the wrist depth camera; enable --enable_wrist_camera.")
    print(
        f"Loaded Door DP policy from {args.dp_policy_checkpoint} "
        f"action_frame={getattr(dp_controller, 'action_frame', 'world')}",
        flush=True,
    )
    if bool(getattr(args, "dp_temporal_ensemble", False)):
        print(
            "Door DP temporal ensemble enabled: "
            f"prefetch={int(getattr(args, 'dp_temporal_prefetch_actions', 3))} "
            f"old_weight={float(getattr(args, 'dp_temporal_old_weight', 0.3)):.3g} "
            f"new_weight={float(getattr(args, 'dp_temporal_new_weight', 0.7)):.3g}",
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


def _collect_float_dp_policy_actions_temporal(
    dp_controller,
    dp_policy_inputs_by_env,
    batch_env_ids,
    batch_states,
    batch_wrist_masks,
    batch_wrist_seconds,
    batch_front_masks,
    batch_front_seconds,
    batch_front_camera_poses,
    batch_wrist_camera_poses,
    env_state_by_id,
):
    for idx, env_id in enumerate(batch_env_ids):
        dp_controller.append_observation_for_env(
            int(env_id),
            batch_states[idx],
            batch_wrist_masks[idx],
            batch_wrist_seconds[idx],
            batch_front_masks[idx],
            batch_front_seconds[idx],
            None if batch_front_camera_poses is None else batch_front_camera_poses[idx],
            None if batch_wrist_camera_poses is None else batch_wrist_camera_poses[idx],
        )

    sample_env_ids = []
    prefetch_by_env = {}
    for env_id in batch_env_ids:
        st = env_state_by_id[int(env_id)]
        buffer = getattr(st, "dp_temporal_action_buffer", None)
        if buffer is None:
            buffer = FloatDPActionOverlapBuffer(
                old_weight=float(getattr(st.args, "dp_temporal_old_weight", 0.3)),
                new_weight=float(getattr(st.args, "dp_temporal_new_weight", 0.7)),
            )
            st.dp_temporal_action_buffer = buffer
            st.dp_temporal_timestep = 0
        prefetch = max(0, int(getattr(st.args, "dp_temporal_prefetch_actions", 3)))
        prefetch_by_env[int(env_id)] = prefetch
        if buffer.queue_size <= prefetch:
            sample_env_ids.append(int(env_id))

    if sample_env_ids:
        chunks = np.asarray(dp_controller.predict_action_chunks_for_envs(sample_env_ids), dtype=np.float32)
        if chunks.ndim != 3 or chunks.shape[-1] < 10:
            raise RuntimeError(f"Temporal ensemble expected action chunks shaped (B,T,>=10), got {chunks.shape}.")
        for row_idx, env_id in enumerate(sample_env_ids):
            st = env_state_by_id[int(env_id)]
            buffer = st.dp_temporal_action_buffer
            current_timestep = int(getattr(st, "dp_temporal_timestep", 0))
            controller_horizon = int(getattr(dp_controller, "action_horizon", chunks.shape[1]))
            args_horizon = getattr(st.args, "dp_action_horizon", None)
            action_horizon = int(args_horizon) if args_horizon is not None else controller_horizon
            action_horizon = max(1, min(int(action_horizon), int(chunks.shape[1])))
            ingest_meta = buffer.ingest(
                chunks[row_idx, :action_horizon],
                start_timestep=current_timestep,
                current_timestep=current_timestep,
            )
            st.dp_temporal_last_ingest_meta = ingest_meta

    dp_actions_by_env = {}
    for env_id in batch_env_ids:
        env_id = int(env_id)
        st = env_state_by_id[env_id]
        current_timestep = int(getattr(st, "dp_temporal_timestep", 0))
        item = st.dp_temporal_action_buffer.pop(current_timestep)
        st.dp_temporal_timestep = current_timestep + 1
        st.dp_temporal_last_action_meta = {
            "timestep": int(item.timestep),
            "source": str(item.source),
            "blend_count": int(item.blend_count),
            "queue_size_after_pop": int(st.dp_temporal_action_buffer.queue_size),
            "prefetch_actions": int(prefetch_by_env.get(env_id, 0)),
            "last_ingest": getattr(st, "dp_temporal_last_ingest_meta", None),
        }
        dp_actions_by_env[env_id] = np.asarray(item.action, dtype=np.float32)
        dp_policy_inputs_by_env[env_id]["temporal_ensemble"] = dict(st.dp_temporal_last_action_meta)
        if hasattr(dp_controller, "get_last_camera_gates_for_env"):
            dp_policy_inputs_by_env[env_id]["camera_gates"] = dp_controller.get_last_camera_gates_for_env(env_id)
        if hasattr(dp_controller, "get_last_interaction_state_for_env"):
            dp_policy_inputs_by_env[env_id]["interaction_state"] = (
                dp_controller.get_last_interaction_state_for_env(env_id)
            )
    return dp_actions_by_env


def collect_float_dp_policy_actions(gym, sim, env_states, dof_names, gripper_idx, dt, dp_controller, dp_control_env_id_set, mode_name):
    dp_policy_inputs_by_env = {}
    dp_actions_by_env = {}
    if dp_controller is None:
        return dp_policy_inputs_by_env, dp_actions_by_env
    sample_dt = float(getattr(env_states[0].args, "dp_policy_dt", dt)) if env_states else float(dt)
    sample_dt = max(1.0e-6, sample_dt)
    batch_env_ids = []
    batch_states = []
    batch_wrist_masks = []
    batch_wrist_seconds = []
    batch_front_masks = []
    batch_front_seconds = []
    batch_front_camera_poses = []
    batch_wrist_camera_poses = []
    state_mode = float_dp_state_mode_from_feature_names(getattr(dp_controller, "state_feature_names", []))
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
        vx_state, yaw_rate_state = base_command_from_targets(
            base_xy_current,
            yaw_current,
            getattr(st, "dp_policy_prev_base_xy", None),
            getattr(st, "dp_policy_prev_yaw", None),
            sample_dt,
        )
        st.dp_policy_prev_base_xy = np.asarray(base_xy_current, dtype=np.float32).copy()
        st.dp_policy_prev_yaw = float(yaw_current)
        dp_state = make_float_dp_observation_state(
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
            vx=vx_state,
            state_mode=state_mode,
            base_pitch=float(getattr(st.args, "robot_pitch", 0.0)),
            pose_frame=getattr(dp_controller, "action_frame", "base"),
        )
        camera_images = capture_dp_camera_images_from_rendered(gym, sim, st.env, st.camera_handles, st.args)
        maybe_dump_initial_depth_images(st, camera_images)
        wrist_mask_rgb, wrist_second_rgb, front_mask_rgb, front_second_rgb = dp_image_inputs_from_cpu_cameras(
            camera_images, st.args
        )
        pointcloud_conditioning = bool(getattr(dp_controller, "pointcloud_conditioning", False))
        if pointcloud_conditioning:
            pointcloud_mode = str(
                getattr(dp_controller, "config", {}).get("pointcloud_mode", "single_front")
            )
            needs_front = pointcloud_mode in ("single_front", "dual_view", "dual_fused")
            needs_wrist = pointcloud_mode in ("single_wrist", "dual_view", "dual_fused")
            missing_required_camera = (
                (needs_front and front_second_rgb is None)
                or (needs_wrist and wrist_second_rgb is None)
            )
        elif getattr(dp_controller, "vision_mode", "depth") == "rgb":
            missing_required_camera = (
                wrist_mask_rgb is None
                or wrist_second_rgb is None
                or front_mask_rgb is None
                or front_second_rgb is None
            )
        elif getattr(dp_controller, "vision_mode", "depth") == "depth_only":
            missing_required_camera = wrist_second_rgb is None or front_second_rgb is None
        else:
            missing_required_camera = wrist_mask_rgb is None or wrist_second_rgb is None
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
        if (
            bool(getattr(dp_controller, "config", {}).get("plucker_conditioning", False))
            or pointcloud_conditioning
        ):
            front_pose_base, wrist_pose_base = float_camera_pose_base(gym, st)
            pointcloud_mode = str(
                getattr(dp_controller, "config", {}).get("pointcloud_mode", "single_front")
            )
            if bool(getattr(dp_controller, "config", {}).get("plucker_conditioning", False)) or pointcloud_mode in (
                "single_front", "dual_view", "dual_fused"
            ):
                dp_policy_inputs_by_env[st.index]["front_camera_pose_base"] = front_pose_base
            else:
                front_pose_base = None
            if bool(getattr(dp_controller, "config", {}).get("plucker_conditioning", False)) or pointcloud_mode in (
                "single_wrist", "dual_view", "dual_fused"
            ):
                dp_policy_inputs_by_env[st.index]["wrist_camera_pose_base"] = wrist_pose_base
            else:
                wrist_pose_base = None
        else:
            front_pose_base = None
            wrist_pose_base = None
        batch_env_ids.append(st.index)
        batch_states.append(dp_state)
        batch_wrist_masks.append(wrist_mask_rgb)
        batch_wrist_seconds.append(wrist_second_rgb)
        batch_front_masks.append(front_mask_rgb)
        batch_front_seconds.append(front_second_rgb)
        batch_front_camera_poses.append(front_pose_base)
        batch_wrist_camera_poses.append(wrist_pose_base)
    if batch_env_ids:
        env_state_by_id = {int(st.index): st for st in env_states}
        use_temporal = bool(getattr(env_states[0].args, "dp_temporal_ensemble", False))
        can_temporal = callable(getattr(dp_controller, "predict_action_chunks_for_envs", None)) and int(
            getattr(dp_controller, "action_dim", 10)
        ) >= 10
        if use_temporal and can_temporal:
            dp_actions_by_env = _collect_float_dp_policy_actions_temporal(
                dp_controller,
                dp_policy_inputs_by_env,
                batch_env_ids,
                batch_states,
                batch_wrist_masks,
                batch_wrist_seconds,
                batch_front_masks,
                batch_front_seconds,
                batch_front_camera_poses,
                batch_wrist_camera_poses,
                env_state_by_id,
            )
        else:
            if use_temporal:
                st0 = env_state_by_id[int(batch_env_ids[0])]
                if not bool(getattr(st0, "dp_temporal_warned_fallback", False)):
                    print(
                        "Warning: --dp_temporal_ensemble requested but the loaded policy controller "
                        "does not expose compatible 10D action chunks; falling back to plain act_batch.",
                        flush=True,
                    )
                    st0.dp_temporal_warned_fallback = True
            dp_actions = dp_controller.act_batch(
                batch_env_ids,
                batch_states,
                batch_wrist_masks,
                batch_wrist_seconds,
                batch_front_masks,
                batch_front_seconds,
                None if not any(pose is not None for pose in batch_front_camera_poses) else batch_front_camera_poses,
                None if not any(pose is not None for pose in batch_wrist_camera_poses) else batch_wrist_camera_poses,
            )
            for env_id, dp_action in zip(batch_env_ids, dp_actions):
                dp_actions_by_env[int(env_id)] = np.asarray(dp_action, dtype=np.float32)
                if hasattr(dp_controller, "get_last_camera_gates_for_env"):
                    dp_policy_inputs_by_env[int(env_id)]["camera_gates"] = dp_controller.get_last_camera_gates_for_env(
                        int(env_id)
                    )
                if hasattr(dp_controller, "get_last_interaction_state_for_env"):
                    dp_policy_inputs_by_env[int(env_id)]["interaction_state"] = (
                        dp_controller.get_last_interaction_state_for_env(int(env_id))
                    )
    return dp_policy_inputs_by_env, dp_actions_by_env


def record_float_dp_frame(gym, sim, st, dof_names, gripper_idx, dt, phase_id, door_pos_record, door_vel_record):
    if st.dp_recorder is None:
        return False
    record_stride = float_dp_record_sample_stride(st.args, dt)
    sim_step = int(getattr(st, "dp_record_sim_steps", 0))
    st.dp_record_sim_steps = sim_step + 1
    if sim_step % record_stride != 0:
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
    record_dt = float(dt) * float(record_stride)
    vx_cmd, yaw_rate_cmd = base_command_from_targets(
        base_xy,
        yaw,
        getattr(st, "dp_record_prev_base_xy", None),
        getattr(st, "dp_record_prev_yaw", None),
        record_dt,
    )
    scripted_target_pos = np.asarray(st.last_target_pos, dtype=np.float32).copy()
    scripted_target_quat = target_quat_for_dp(st.last_target_quat, st.ik_state, ee_quat)
    action_target_pos = scripted_target_pos
    action_target_quat = scripted_target_quat
    use_tracik_command_fk = (
        str(getattr(st.args, "arm_ik_solver", "gym_jacobian")) == "tracik"
        and getattr(st, "tracik_command_pos_world", None) is not None
        and getattr(st, "tracik_command_quat_world", None) is not None
    )
    if use_tracik_command_fk:
        action_target_pos = np.asarray(st.tracik_command_pos_world, dtype=np.float32).copy()
        action_target_quat = base_ik.normalize_quat(
            np.asarray(st.tracik_command_quat_world, dtype=np.float32)
        )
    wrist_mask_rgb, wrist_second_rgb, front_mask_rgb, front_second_rgb = float_dp_camera_images_for_record_frame(
        gym, sim, st
    )
    vision_mode = float_dp_vision_mode(st.args)
    if vision_mode == "rgb":
        missing_required_camera = (
            wrist_mask_rgb is None
            or wrist_second_rgb is None
            or front_mask_rgb is None
            or front_second_rgb is None
        )
    elif vision_mode == "depth_only":
        missing_required_camera = wrist_second_rgb is None or front_second_rgb is None
    else:
        missing_required_camera = wrist_mask_rgb is None or wrist_second_rgb is None
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

    state_mode = normalize_float_dp_state_mode(getattr(st.args, "dp_record_state_mode", FLOAT_DP_STATE_MODE_FULL))
    dp_state = make_float_dp_observation_state(
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
        vx=vx_cmd,
        state_mode=state_mode,
        base_pitch=float(getattr(st.args, "robot_pitch", 0.0)),
        pose_frame=getattr(st.args, "ee_pose_frame", "base"),
    )
    dp_action = make_float_dp_action(
        vx_cmd,
        yaw_rate_cmd,
        action_target_pos,
        action_target_quat,
        st.last_gripper,
        base_xy,
        st.args.robot_z,
        yaw,
        action_mode=state_mode,
        dof_names=dof_names,
        dof_pos=st.dof_positions,
        base_pitch=float(getattr(st.args, "robot_pitch", 0.0)),
        pose_frame=getattr(st.args, "ee_pose_frame", "base"),
    )
    replay_snapshot = make_float_replay_snapshot(
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
    )
    replay_snapshot["scripted_target_ee_pos_world"] = scripted_target_pos
    replay_snapshot["scripted_target_ee_quat_world"] = scripted_target_quat
    if use_tracik_command_fk:
        replay_snapshot["tracik_command_q"] = np.asarray(st.tracik_command_q, dtype=np.float32).copy()
        replay_snapshot["tracik_command_ee_pos_world"] = action_target_pos
        replay_snapshot["tracik_command_ee_quat_world"] = action_target_quat
    if bool(getattr(st.args, "record_camera_pose", False)):
        front_pose_base, wrist_pose_base = float_camera_pose_base(gym, st)
        replay_snapshot["front_camera_pose_base"] = front_pose_base
        replay_snapshot["wrist_camera_pose_base"] = wrist_pose_base
    if bool(getattr(st.args, "record_handle_bbox", False)):
        replay_snapshot["front_handle_bbox_xyxy"] = np.asarray(
            getattr(st, "last_front_handle_bbox", np.zeros(4, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(4)
        replay_snapshot["front_handle_bbox_valid"] = np.asarray(
            getattr(st, "last_front_handle_bbox_valid", np.zeros(1, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1)
        replay_snapshot["wrist_handle_bbox_xyxy"] = np.asarray(
            getattr(st, "last_wrist_handle_bbox", np.zeros(4, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(4)
        replay_snapshot["wrist_handle_bbox_valid"] = np.asarray(
            getattr(st, "last_wrist_handle_bbox_valid", np.zeros(1, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1)
    if bool(getattr(st.args, "record_gripper_handle_contact", False)):
        replay_snapshot.update(gripper_handle_contact_snapshot(gym, st))
    phase_names = list(st.dp_recorder.metadata.get("phase_names", []))
    phase_name = phase_names[int(phase_id)] if 0 <= int(phase_id) < len(phase_names) else ""
    positive_end_phases = set(st.dp_recorder.metadata.get("end_signal_positive_phases", []))
    st.dp_recorder.add_frame(
        dp_state,
        wrist_mask_rgb,
        wrist_second_rgb,
        dp_action,
        phase_id,
        front_mask_rgb=front_mask_rgb,
        front_second_rgb=front_second_rgb,
        replay_snapshot=replay_snapshot,
        end_signal=(
            1.0
            if bool(getattr(st.args, "record_end_signal", False))
            and str(phase_name) in positive_end_phases
            else 0.0
        ),
    )
    st.last_dp_action = dp_action.copy()
    st.dp_record_prev_base_xy = np.asarray(base_xy, dtype=np.float32).copy()
    st.dp_record_prev_yaw = float(yaw)
    if bool(getattr(st, "base_door_collision_detected", False)):
        st.dp_record_success = False
    elif door_success(door_pos_record, st.args):
        st.dp_record_success = True
    return True


def gripper_handle_contact_quality_from_recorder(st):
    recorder = getattr(st, "dp_recorder", None)
    if recorder is None or recorder.frame_count <= 0:
        return False, {
            "reason": "no_recorder_frames",
            "contact_frames": 0,
            "required_frames": int(getattr(st.args, "filter_gripper_handle_contact_min_frames", 5)),
        }
    if "gripper_handle_contact_both" not in recorder.frames and "gripper_handle_contact_any" not in recorder.frames:
        return False, {
            "reason": "missing_gripper_handle_contact_fields",
            "contact_frames": 0,
            "required_frames": int(getattr(st.args, "filter_gripper_handle_contact_min_frames", 5)),
        }

    phase_names = list(recorder.metadata.get("phase_names", []))
    requested_phases = parse_csv_names(
        getattr(st.args, "filter_gripper_handle_contact_phase_names", ""),
        ["close_gripper", "rotate_handle"],
    )
    phase_ids = {phase_names.index(name) for name in requested_phases if name in phase_names}
    subtask = np.asarray(recorder.frames.get("subtask_index", []), dtype=np.int64).reshape(-1)
    if phase_ids:
        phase_mask = np.asarray([int(x) in phase_ids for x in subtask], dtype=bool)
    else:
        phase_mask = np.ones(recorder.frame_count, dtype=bool)

    require_both = bool(getattr(st.args, "filter_gripper_handle_contact_require_both", True))
    key = "gripper_handle_contact_both" if require_both else "gripper_handle_contact_any"
    values = recorder.frames.get(key)
    if not values:
        return False, {
            "reason": f"missing_{key}",
            "contact_frames": 0,
            "required_frames": int(getattr(st.args, "filter_gripper_handle_contact_min_frames", 5)),
        }
    contact_mask = np.asarray(values, dtype=np.float32).reshape(-1) > 0.5
    usable = contact_mask & phase_mask[: contact_mask.shape[0]]
    contact_frames = int(np.count_nonzero(usable))
    required_frames = int(getattr(st.args, "filter_gripper_handle_contact_min_frames", 5))
    score_values = recorder.frames.get("gripper_handle_contact_score", [])
    if score_values:
        score_arr = np.asarray(score_values, dtype=np.float32).reshape(len(score_values), -1)
        score_window = score_arr[phase_mask[: score_arr.shape[0]]] if phase_mask.size else score_arr
        total_score = float(np.sum(score_window))
        max_score = float(np.max(score_window)) if score_window.size else 0.0
    else:
        total_score = 0.0
        max_score = 0.0
    ok = contact_frames >= required_frames
    return ok, {
        "reason": "ok" if ok else "insufficient_gripper_handle_contact",
        "contact_frames": contact_frames,
        "required_frames": required_frames,
        "require_both": require_both,
        "phase_names": requested_phases,
        "total_score": total_score,
        "max_score": max_score,
    }


def finish_float_dp_recorders(env_states, args):
    saved = 0
    total_recorders = sum(st.dp_recorder is not None for st in env_states)
    for st in env_states:
        if st.dp_recorder is None:
            continue
        collision_detected = bool(getattr(st, "base_door_collision_detected", False))
        contact_ok = True
        contact_quality = None
        if bool(getattr(st.args, "filter_gripper_handle_contact", False)):
            contact_ok, contact_quality = gripper_handle_contact_quality_from_recorder(st)
            st.dp_recorder.metadata["gripper_handle_contact_quality"] = contact_quality
        valid_success = bool(st.dp_record_success) and not collision_detected and contact_ok
        if valid_success and st.dp_recorder.frame_count > 0:
            recorded_frames = st.dp_recorder.frame_count
            st.dp_recorder.save_episode()
            saved += 1
            print(
                f"Finished raw Door DP recording env={st.index}: saved_successful=1/1 "
                f"frames={recorded_frames}",
                flush=True,
            )
        elif st.dp_recorder.frame_count == 0:
            print(
                f"⚠️📷 Finished raw Door DP recording env={st.index}: "
                "saved_successful=0/1 frames=0 reason=camera_unavailable",
                flush=True,
            )
        else:
            if collision_detected:
                reason = "base-door collision detected"
            elif not bool(st.dp_record_success):
                reason = f"door did not reach {args.pass_open_angle_deg} deg"
            elif not contact_ok:
                reason = f"gripper-handle contact filter failed: {contact_quality}"
            else:
                reason = "unknown"
            print(
                f"Finished raw Door DP recording env={st.index}: saved_successful=0/1 "
                f"frames={st.dp_recorder.frame_count} reason={reason}",
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


def handle_unlock_progress(door, handle_angle):
    """Return signed handle travel from its configured neutral/rest angle."""
    direction = -1.0 if float(getattr(door, "handle_unlock_direction_sign", 1.0)) < 0.0 else 1.0
    configured_rest = getattr(door, "handle_rest_angle", None)
    rest = (
        float(configured_rest)
        if configured_rest is not None
        else (float(door.dof_lower[1]) if len(door.dof_lower) >= 2 else 0.0)
    )
    return direction * (float(handle_angle) - rest)


def handle_is_unlocked(door, handle_angle):
    return handle_unlock_progress(door, handle_angle) >= float(door.handle_unlock_threshold)


def compute_door_efforts(door, dof_pos, dof_vel, args):
    efforts = np.zeros(len(dof_pos), dtype=np.float32)
    if len(dof_pos) == 0:
        return efforts

    door_angle = float(dof_pos[0])
    if bool(getattr(args, "door_twin_asset_probe", False)):
        door.open_stage = True
    if len(dof_pos) < 2:
        door.open_stage = True
        hinge_range = max(abs(float(door.dof_upper[0])), abs(float(door.dof_lower[0])), 1.0e-3)
        if abs(door_angle) < args.door_auto_open_target_ratio * hinge_range:
            auto_torque = args.door_auto_open_force * args.door_motion_sign * args.door_auto_open_sign
        else:
            auto_torque = 0.0
        efforts[0] = auto_torque - args.door_open_resistance * door_angle - args.door_open_damping * float(dof_vel[0])
        return efforts

    handle_angle = float(dof_pos[1])
    if handle_is_unlocked(door, handle_angle):
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

    configured_rest = getattr(door, "handle_rest_angle", None)
    handle_rest = (
        float(configured_rest)
        if configured_rest is not None
        else float(door.dof_lower[1])
    )
    handle_displacement_from_rest = handle_angle - handle_rest
    efforts[1] = (
        -args.handle_spring_stiffness * handle_displacement_from_rest
        - args.handle_spring_damping * float(dof_vel[1])
    )
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
    if bool(getattr(args, "rgb", False)) and bool(getattr(args, "depth_only", False)):
        raise ValueError("--rgb and --depth_only are mutually exclusive.")
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
