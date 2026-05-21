#!/usr/bin/env python3
"""Parallel float-base B1Z1 base+arm IK door-push recorder.

This is the high-throughput variant of isaacgym_float_ik_b1z1_basearn_push_door.py.
It keeps the same asset/controller helpers, but records multiple Isaac Gym envs
inside one simulator process.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

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

DP_ROOT = HIGH_LEVEL_ROOT / "dp"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

try:
    from door_dp_common import (
        ACTION_NAMES,
        DoorDPJsonlLogger,
        DoorDPPolicyController,
        RawDoorDPRecorder,
        make_state_feature_names,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
    )
except ImportError:
    ACTION_NAMES = None
    DoorDPJsonlLogger = None
    DoorDPPolicyController = None
    RawDoorDPRecorder = None
    make_state_feature_names = None
    normalize_vision_mode = None
    raw_image_keys_for_vision_mode = None


DP_NUM_DOFS = 19
DP_NUM_ACTIONS = 18
DP_PHASE_NAMES = [
    "walk",
    "initial_hold",
    "grasp",
    "grasp_hold",
    "close_gripper",
    "rotate_handle",
    "push_door",
    "return_home",
    "hold_home",
]
DP_PHASE_ID = {name: idx for idx, name in enumerate(DP_PHASE_NAMES)}
IKPUSH_STATE_VERSION = "zero_leg_dof_pos_prev_action_v1"
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


def sorted_asset_items(asset_dict):
    def key_fn(item):
        key = item[0]
        return (0, int(key)) if str(key).isdigit() else (1, str(key))

    return [item[1] for item in sorted(asset_dict.items(), key=key_fn)]


def sorted_asset_entries(asset_dict):
    def key_fn(item):
        key = item[0]
        return (0, int(key)) if str(key).isdigit() else (1, str(key))

    return [(idx, item[1]) for idx, item in enumerate(sorted(asset_dict.items(), key=key_fn))]


def parse_args():
    args = gymutil.parse_arguments(
        description="B1Z1 base+arm float IK door-push demo.",
        headless=True,
        no_graphics=True,
        custom_parameters=[
            {"name": "--asset_root", "type": str, "default": str(base_ik.DEFAULT_ASSET_ROOT)},
            {"name": "--asset_file", "type": str, "default": base_ik.DEFAULT_ASSET_FILE},
            {"name": "--rl_device", "type": str, "default": "cuda:0"},
            {"name": "--num_envs", "type": int, "default": 1},
            {"name": "--steps", "type": int, "default": 2900},
            {"name": "--seed", "type": int, "default": -1},
            {"name": "--door_cfg", "type": str, "default": str(DEFAULT_DOOR_CFG)},
            {"name": "--door_name", "type": str, "default": ""},
            {"name": "--door_index", "type": int, "default": -1},
            {"name": "--door_actor_scale", "type": float, "default": 1.2},
            {"name": "--door_x", "type": float, "default": 2.5},
            {"name": "--door_y", "type": float, "default": 0.0},
            {"name": "--door_z_offset", "type": float, "default": 0.01},
            {"name": "--robot_x", "type": float, "default": 4.1},
            {"name": "--robot_y", "type": float, "default": -0.06},
            {"name": "--robot_z", "type": float, "default": 0.60},
            {"name": "--robot_yaw", "type": float, "default": math.pi},
            {"name": "--robot_front_offset", "type": float, "default": 0.55},
            {"name": "--robot_rear_offset", "type": float, "default": 0.65},
            {"name": "--stop_distance", "type": float, "default": 0.25},
            {"name": "--push_base_distance", "type": float, "default": 0.35},
            {"name": "--base_push_time_scale", "type": float, "default": 1.35},
            {"name": "--door_pass_clearance", "type": float, "default": 0.55},
            {
                "name": "--no_pass_through_door",
                "action": "store_true",
                "default": False,
                "help": "Disable the default behavior that moves the base through the door during the push phase.",
            },
            {"name": "--push_base_yaw_delta", "type": float, "default": 0.0},
            {"name": "--walk_steps", "type": int, "default": 260},
            {"name": "--initial_hold_steps", "type": int, "default": 200},
            {"name": "--grasp_steps", "type": int, "default": 150},
            {"name": "--grasp_hold_steps", "type": int, "default": 100},
            {"name": "--gripper_close_steps", "type": int, "default": 120},
            {"name": "--handle_rotate_steps", "type": int, "default": 300},
            {"name": "--door_push_steps", "type": int, "default": 1080},
            {"name": "--return_home_steps", "type": int, "default": 360},
            {"name": "--hold_steps", "type": int, "default": 300},
            {"name": "--pregrasp_offset", "type": float, "default": 0.15},
            {"name": "--grasp_offset", "type": float, "default": 0.0},
            {"name": "--grasp_x_offset", "type": float, "default": -0.03},
            {"name": "--grasp_z_offset", "type": float, "default": -0.03},
            {"name": "--handle_rotate_right_distance", "type": float, "default": 0.03},
            {"name": "--handle_rotate_down_distance", "type": float, "default": 0.03},
            {"name": "--handle_rotate_angle", "type": float, "default": 1.05},
            {"name": "--door_push_distance", "type": float, "default": 1.10},
            {"name": "--no_ikpush_env_randomization", "action": "store_true"},
            {"name": "--ikpush_robot_x_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_robot_y_rand", "type": float, "default": 0.04},
            {"name": "--ikpush_robot_yaw_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_pregrasp_offset_rand", "type": float, "default": 0.025},
            {"name": "--ikpush_grasp_x_offset_rand", "type": float, "default": 0.012},
            {"name": "--ikpush_grasp_z_offset_rand", "type": float, "default": 0.012},
            {"name": "--ikpush_handle_rotate_angle_rand", "type": float, "default": 0.04},
            {"name": "--ikpush_door_push_distance_rand", "type": float, "default": 0.06},
            {"name": "--ikpush_door_joint_friction_rand", "type": float, "default": 0.08},
            {"name": "--ikpush_door_joint_damping_rand", "type": float, "default": 0.04},
            {"name": "--ikpush_handle_joint_friction_rand", "type": float, "default": 0.005},
            {"name": "--ikpush_handle_joint_damping_rand", "type": float, "default": 0.005},
            {"name": "--ikpush_handle_spring_stiffness_rand", "type": float, "default": 0.05},
            {"name": "--ikpush_handle_spring_damping_rand", "type": float, "default": 0.01},
            {"name": "--lever_step_size", "type": float, "default": 0.06},
            {"name": "--push_contact_bias", "type": float, "default": 0.025},
            {"name": "--handle_follow_push_ratio", "type": float, "default": 0.45},
            {"name": "--door_freeze_blend_start_ratio", "type": float, "default": 0.82},
            {"name": "--door_freeze_target_ratio", "type": float, "default": 0.94},
            {
                "name": "--push_follow_orientation",
                "action": "store_true",
                "help": "After the door unlocks, also keep the end-effector orientation fixed relative to the handle. By default push contact is position-only.",
            },
            {
                "name": "--unidoor_style_push",
                "action": "store_true",
                "default": True,
                "help": "During push, step from current EE along the handle push direction.",
            },
            {"name": "--no_unidoor_style_push", "dest": "unidoor_style_push", "action": "store_false"},
            {"name": "--gripper_open", "type": float, "default": -1.5707963267948966},
            {"name": "--gripper_closed", "type": float, "default": 0.0},
            {"name": "--gripper_close_ratio", "type": float, "default": 0.8},
            {"name": "--gripper_open_stage_ratio", "type": float, "default": 0.25},
            {"name": "--gripper_loosen_steps", "type": int, "default": 120},
            {"name": "--handle_spring_stiffness", "type": float, "default": 0.5},
            {"name": "--handle_spring_damping", "type": float, "default": 0.1},
            {"name": "--handle_unlock_ratio", "type": float, "default": 40.0 / 45.0},
            {"name": "--door_open_resistance", "type": float, "default": 0.0},
            {"name": "--door_open_damping", "type": float, "default": 0.0},
            {"name": "--door_lock_force", "type": float, "default": 0.0},
            {"name": "--door_joint_friction", "type": float, "default": 0.},
            {"name": "--door_joint_damping", "type": float, "default": 0.},
            {"name": "--handle_joint_friction", "type": float, "default": 0.05},
            {"name": "--handle_joint_damping", "type": float, "default": 0.05},
            {"name": "--door_auto_open_force", "type": float, "default": 0.0},
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
            {"name": "--draw_ik_target", "dest": "draw_ik_target", "action": "store_true", "default": True},
            {"name": "--no_draw_ik_target", "dest": "draw_ik_target", "action": "store_false"},
            {"name": "--draw_camera_axes", "dest": "draw_camera_axes", "action": "store_true", "default": True},
            {"name": "--no_draw_camera_axes", "dest": "draw_camera_axes", "action": "store_false"},
            {"name": "--enable_wrist_camera", "dest": "enable_wrist_camera", "action": "store_true", "default": True},
            {"name": "--no_enable_wrist_camera", "dest": "enable_wrist_camera", "action": "store_false"},
            {"name": "--enable_front_camera", "dest": "enable_front_camera", "action": "store_true", "default": True},
            {"name": "--no_enable_front_camera", "dest": "enable_front_camera", "action": "store_false"},
            {"name": "--show_camera_images", "dest": "show_camera_images", "action": "store_true", "default": True},
            {"name": "--no_show_camera_images", "dest": "show_camera_images", "action": "store_false"},
            {"name": "--show_seg", "action": "store_true"},
            {"name": "--no_show_seg", "action": "store_true"},
            {"name": "--rgb", "action": "store_true", "help": "Show RGB+mask camera previews instead of full depth+mask."},
            {"name": "--camera_rgb", "action": "store_true"},
            {"name": "--camera_depth", "action": "store_true"},
            {"name": "--no_camera_depth", "action": "store_true"},
            {"name": "--camera_seg", "action": "store_true"},
            {"name": "--no_camera_seg", "action": "store_true"},
            {"name": "--handle_seg_id", "type": int, "default": 2},
            {"name": "--camera_depth_clip_lower", "type": float, "default": 0.02},
            {"name": "--camera_depth_clip_far", "type": float, "default": 2.0},
            {"name": "--camera_display_scale", "type": int, "default": 5},
            {"name": "--camera_display_interval", "type": int, "default": 1},
            {"name": "--camera_axis_scale", "type": float, "default": 0.10},
            {"name": "--camera_axis_thickness", "type": float, "default": 0.004},
            {"name": "--wrist_camera_down_tilt", "type": float, "default": 0.20},
            {"name": "--front_camera_yaw_deg", "type": float, "default": 0.0},
            {"name": "--front_camera_pitch_deg", "type": float, "default": -30.0},
            {"name": "--front_camera_roll_deg", "type": float, "default": 0.0},
            {"name": "--record_dp_dataset", "action": "store_true"},
            {"name": "--dp_raw_root", "type": str, "default": str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp")},
            {"name": "--dp_task", "type": str, "default": "push lever door open"},
            {"name": "--dp_record_env_id", "type": int, "default": 0},
            {"name": "--dp_record_all_envs", "action": "store_true"},
            {"name": "--no_dp_record_all_envs", "action": "store_true"},
            {"name": "--dp_fps", "type": int, "default": 50},
            {"name": "--dp_policy_checkpoint", "type": str, "default": ""},
            {"name": "--dp_control_env_id", "type": int, "default": 0},
            {"name": "--dp_inference_steps", "type": int, "default": 100},
            {"name": "--dp_action_horizon", "type": int, "default": -1},
            {"name": "--dp_log_path", "type": str, "default": ""},
            {"name": "--dp_log_interval", "type": int, "default": 25},
            {"name": "--no_dp_print", "dest": "dp_print", "action": "store_false", "default": True},
            {"name": "--dp_warmstart", "action": "store_true"},
            {"name": "--dp_warmstart_raw_episode", "type": str, "default": ""},
            {"name": "--dp_warmstart_step", "type": int, "default": -1},
            {"name": "--dp_warmstart_expert_obs", "dest": "dp_warmstart_expert_obs", "action": "store_true", "default": True},
            {"name": "--no_dp_warmstart_expert_obs", "dest": "dp_warmstart_expert_obs", "action": "store_false"},
            {"name": "--pass_open_angle_deg", "type": float, "default": 80.0},
            {"name": "--no_preview_trajectory_at_spawn", "action": "store_true"},
        ],
    )

    # gymutil's wrapper does not preserve default=True for store_true custom args,
    # so keep these visualization helpers on by default and let --no_* flags opt out.
    argv = set(sys.argv[1:])
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
    args.dp_record_all_envs = not bool(args.no_dp_record_all_envs)
    args.dp_print = "--no_dp_print" not in argv
    args.dp_action_horizon = None if int(args.dp_action_horizon) < 0 else int(args.dp_action_horizon)
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if not args.dp_record_all_envs and (args.dp_record_env_id < 0 or args.dp_record_env_id >= args.num_envs):
        raise ValueError("--dp_record_env_id must be in [0, num_envs - 1].")
    if args.dp_policy_checkpoint and (args.dp_control_env_id < 0 or args.dp_control_env_id >= args.num_envs):
        raise ValueError("--dp_control_env_id must be in [0, num_envs - 1].")
    if args.record_dp_dataset and args.dp_policy_checkpoint:
        raise ValueError("--record_dp_dataset and --dp_policy_checkpoint are separate modes; run recording or policy play, not both.")
    warmstart_params = [
        bool(args.dp_warmstart_raw_episode),
        int(args.dp_warmstart_step) >= 0,
        "--dp_warmstart_expert_obs" in argv or "--no_dp_warmstart_expert_obs" in argv,
    ]
    if not args.dp_warmstart and any(warmstart_params):
        raise ValueError("Warm-start options require --dp_warmstart.")
    if args.dp_warmstart:
        if not args.dp_policy_checkpoint:
            raise ValueError("--dp_warmstart requires --dp_policy_checkpoint.")
        if not args.dp_warmstart_raw_episode:
            raise ValueError("--dp_warmstart requires --dp_warmstart_raw_episode.")
        if int(args.dp_warmstart_step) < 0:
            raise ValueError("--dp_warmstart requires non-negative --dp_warmstart_step.")
    args._explicit_cli_flags = set(sys.argv[1:])

    if args.headless and args.show_camera_images:
        print(
            "⚠️📷 Headless mode disables OpenCV camera preview windows; run without --headless to view camera images.",
            flush=True,
        )
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
    args.door_motion_sign = -1.0
    args.pass_through_door = not bool(args.no_pass_through_door)
    return args


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(a, b, t):
    return a + (b - a) * float(t)


def quat_nlerp(a, b, t):
    qa = base_ik.normalize_quat(np.asarray(a, dtype=np.float32))
    qb = base_ik.normalize_quat(np.asarray(b, dtype=np.float32))
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    return base_ik.normalize_quat(lerp(qa, qb, t)).astype(np.float32)


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
    try:
        gym.set_rigid_body_segmentation_id(env, door_actor, door.handle_body_index, int(args.handle_seg_id))
    except AttributeError:
        print("set_rigid_body_segmentation_id is not available; handle mask will use the door actor segmentation.")

    door_dof_props = gym.get_actor_dof_properties(env, door_actor)
    if len(door_dof_props) > 0:
        door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        n = min(len(door_dof_props), 2)
        if n >= 1:
            door_dof_props["lower"][0] = -math.pi / 2
            door_dof_props["upper"][0] = 0.0
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

    door_dof_props = gym.get_actor_dof_properties(env, door_actor)
    if len(door_dof_props) > 0:
        door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        n = min(len(door_dof_props), 2)
        if n >= 1:
            door_dof_props["lower"][0] = -math.pi / 2
            door_dof_props["upper"][0] = 0.0
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

    return env, arm_actor, actor_handles, door_actor, np.array([args.robot_x, robot_y], dtype=np.float32)


@dataclass
class ParallelEnvState:
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
    base_push: np.ndarray
    yaw_start: float
    yaw_push: float
    traj: dict
    dp_recorder: object = None
    dp_record_success: bool = False
    dp_record_warned_no_camera: bool = False
    prev_base_xy: object = None
    prev_yaw: object = None
    last_dp_action: object = None
    last_phase: str = "init"
    last_handle_goal: object = None
    last_door_pos: object = None
    last_target_pos: object = None
    last_target_quat: object = None
    last_gripper: float = 0.0


def clone_door_runtime(door):
    return replace(
        door,
        dof_lower=np.asarray(door.dof_lower, dtype=np.float32).copy(),
        dof_upper=np.asarray(door.dof_upper, dtype=np.float32).copy(),
        handle_goal_offset=np.asarray(door.handle_goal_offset, dtype=np.float32).copy(),
        open_stage=False,
    )


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


IKPUSH_DEFAULT_ENV_RANGES = {
    "robot_y": (-0.07, 0.07),
    "robot_yaw": (3.10, 3.18),
    "pregrasp_offset": (0.12, 0.20),
    "grasp_x_offset": (-0.055, -0.010),
    "grasp_z_offset": (-0.055, -0.005),
    "door_push_distance": (0.95, 1.20),
    "handle_rotate_angle": (0.95, 1.15),
    "door_joint_friction": (0.05, 0.30),
    "door_joint_damping": (0.05, 0.30),
    "handle_joint_friction": (0.045, 0.055),
    "handle_joint_damping": (0.045, 0.055),
    "handle_spring_stiffness": (0.45, 0.55),
    "handle_spring_damping": (0.09, 0.11),
}


IKPUSH_ARG_FLAGS = {
    "robot_y": "--robot_y",
    "robot_yaw": "--robot_yaw",
    "pregrasp_offset": "--pregrasp_offset",
    "grasp_x_offset": "--grasp_x_offset",
    "grasp_z_offset": "--grasp_z_offset",
    "door_push_distance": "--door_push_distance",
    "handle_rotate_angle": "--handle_rotate_angle",
    "door_joint_friction": "--door_joint_friction",
    "door_joint_damping": "--door_joint_damping",
    "handle_joint_friction": "--handle_joint_friction",
    "handle_joint_damping": "--handle_joint_damping",
    "handle_spring_stiffness": "--handle_spring_stiffness",
    "handle_spring_damping": "--handle_spring_damping",
}


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


def sample_env_value(rng, args, attr, half_attr, lower=None, upper=None):
    explicit_flags = getattr(args, "_explicit_cli_flags", set())
    default_range = IKPUSH_DEFAULT_ENV_RANGES.get(attr)
    flag = IKPUSH_ARG_FLAGS.get(attr)
    if default_range is not None and flag not in explicit_flags:
        value = float(rng.uniform(float(default_range[0]), float(default_range[1])))
    else:
        value = sample_with_half_range(rng, getattr(args, attr), getattr(args, half_attr), lower=lower, upper=upper)
    if lower is not None:
        value = max(float(lower), value)
    if upper is not None:
        value = min(float(upper), value)
    return value


def make_env_args(args, env_index):
    env_args = SimpleNamespace(**vars(args))
    env_seed = seed_for_env(args, env_index)
    rng = np.random.default_rng(env_seed)
    env_args.env_seed = env_seed
    env_args.parallel_env_id = int(env_index)
    enabled = not bool(getattr(args, "no_ikpush_env_randomization", False))

    sampled = {
        "seed": int(getattr(args, "seed", 0)),
        "env_seed": int(env_seed),
        "enabled": bool(enabled),
    }

    def set_sampled(attr, half_attr, lower=None, upper=None):
        base_value = getattr(args, attr)
        value = (
            sample_env_value(rng, args, attr, half_attr, lower=lower, upper=upper)
            if enabled
            else float(base_value)
        )
        setattr(env_args, attr, value)
        sampled[attr] = value

    set_sampled("robot_x", "ikpush_robot_x_rand")
    set_sampled("robot_y", "ikpush_robot_y_rand")
    set_sampled("robot_yaw", "ikpush_robot_yaw_rand")
    set_sampled("pregrasp_offset", "ikpush_pregrasp_offset_rand", lower=0.05)
    set_sampled("grasp_x_offset", "ikpush_grasp_x_offset_rand")
    set_sampled("grasp_z_offset", "ikpush_grasp_z_offset_rand")
    set_sampled("handle_rotate_angle", "ikpush_handle_rotate_angle_rand", lower=0.75, upper=1.30)
    set_sampled("door_push_distance", "ikpush_door_push_distance_rand", lower=0.70, upper=1.40)
    set_sampled("door_joint_friction", "ikpush_door_joint_friction_rand", lower=0.0)
    set_sampled("door_joint_damping", "ikpush_door_joint_damping_rand", lower=0.0)
    set_sampled("handle_joint_friction", "ikpush_handle_joint_friction_rand", lower=0.0)
    set_sampled("handle_joint_damping", "ikpush_handle_joint_damping_rand", lower=0.0)
    set_sampled("handle_spring_stiffness", "ikpush_handle_spring_stiffness_rand", lower=0.0)
    set_sampled("handle_spring_damping", "ikpush_handle_spring_damping_rand", lower=0.0)

    env_args.ikpush_randomization_json = json.dumps(sampled, sort_keys=True)
    return env_args


def set_robot_base_pose(gym, env, actor_handles, xy, z, yaw):
    quat = base_ik.yaw_quat(yaw)
    for actor in actor_handles:
        root_handle = gym.get_actor_root_rigid_body_handle(env, actor)
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(float(xy[0]), float(xy[1]), float(z))
        transform.r = gymapi.Quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gym.set_rigid_transform(env, root_handle, transform)


def compute_base_push_target(args, base_stop, heading):
    requested_progress = max(0.0, float(args.push_base_distance))
    if args.pass_through_door:
        door_xy = np.asarray([args.door_x, args.door_y], dtype=np.float32)
        clear_center = door_xy + heading * (float(args.robot_rear_offset) + float(args.door_pass_clearance))
        pass_progress = float(np.dot(clear_center - base_stop, heading))
        requested_progress = max(requested_progress, pass_progress)
    return base_stop + heading * requested_progress


def get_body_pose(gym, env, actor, body_index):
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    pos_raw = states["pose"]["p"][body_index]
    quat_raw = states["pose"]["r"][body_index]
    pos = np.array([pos_raw["x"], pos_raw["y"], pos_raw["z"]], dtype=np.float32)
    quat = np.array([quat_raw["x"], quat_raw["y"], quat_raw["z"], quat_raw["w"]], dtype=np.float32)
    return pos, base_ik.normalize_quat(quat)


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
        depth_rgb, valid_depth_count = depth_to_rgb(
            depth_image,
            args.camera_depth_clip_lower,
            args.camera_depth_clip_far,
        )
        images[f"{prefix}_masked_depth"] = depth_rgb
        if args.headless and not getattr(args, f"_{prefix}_headless_depth_checked", False):
            if valid_depth_count == 0:
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
    return pos, base_ik.normalize_quat(quat)


def current_ee_pose_from_refreshed_tensors(ik_state):
    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    pos = eef_state[:3].detach().cpu().numpy().astype(np.float32).copy()
    quat = eef_state[3:7].detach().cpu().numpy().astype(np.float32).copy()
    return pos, base_ik.normalize_quat(quat)


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
        raise ValueError(f"Unsupported ikpush action_frame={action_frame!r}; expected 'base' or 'world'.")
    gripper = float(action[9])
    return base_xy_next.astype(np.float32), yaw_next, target_pos, target_quat, gripper


def _round_list(value, precision=5):
    return np.round(np.asarray(value, dtype=np.float64), precision).tolist()


def make_float_dp_policy_log_record(step, st, dp_action, dp_state, ee_pos, ee_quat, door_pos, phase):
    action_frame = str(getattr(st, "dp_action_frame", "base"))
    return {
        "step": int(step),
        "controlled_env_id": int(st.index),
        "num_envs": int(st.args.num_envs),
        "phase_name": str(phase),
        "dp_action_names": list(ACTION_NAMES or []),
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


def scalar_to_str(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def scalar_to_int(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return int(arr.item())
    return int(arr.reshape(-1)[0])


def raw_vision_mode_from_data(data):
    if "vision_mode" in data.files:
        mode = scalar_to_str(data["vision_mode"]).lower()
        if normalize_vision_mode is not None:
            return normalize_vision_mode(mode)
        return mode
    if "wrist_rgb" in data.files or "front_rgb" in data.files:
        return "rgb"
    return "depth"


def raw_action_frame_from_data(data):
    for key in ("action_frame", "action_pose_frame", "target_pose_frame"):
        if key in data.files:
            return scalar_to_str(data[key]).lower()
    return "world"


def raw_ikpush_state_version_from_data(data):
    if "ikpush_state_version" in data.files:
        return scalar_to_str(data["ikpush_state_version"])
    return "legacy"


def normalize_config_path_for_match(path_value):
    value = scalar_to_str(path_value)
    candidates = [Path(value).expanduser(), HIGH_LEVEL_ROOT / value, REPO_ROOT / value]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return value


def yaw_from_quat_xyzw(quat):
    q = base_ik.normalize_quat(np.asarray(quat, dtype=np.float32))
    x, y, z, w = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def set_actor_root_from_state(gym, env, actor, root_state):
    state = np.asarray(root_state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 7:
        raise ValueError(f"Root state must have at least 7 values, got shape {state.shape}")
    root_handle = gym.get_actor_root_rigid_body_handle(env, actor)
    transform = gymapi.Transform()
    transform.p = gymapi.Vec3(float(state[0]), float(state[1]), float(state[2]))
    transform.r = gymapi.Quat(float(state[3]), float(state[4]), float(state[5]), float(state[6]))
    gym.set_rigid_transform(env, root_handle, transform)


def require_warmstart_field(data, key, step):
    if key not in data.files:
        raise KeyError(f"Warm-start raw episode is missing required field {key!r}.")
    arr = np.asarray(data[key])
    if arr.ndim <= 0 or arr.shape[0] <= int(step):
        raise ValueError(f"Warm-start field {key!r} has shape {arr.shape}, cannot read step {step}.")
    return arr


def validate_warmstart_raw(data, args, controller, st):
    expected_vision_mode = "rgb" if args.rgb else "depth"
    raw_vision = raw_vision_mode_from_data(data)
    if raw_vision != expected_vision_mode:
        raise ValueError(
            f"Warm-start raw episode vision_mode={raw_vision!r}, but ikpush play is running {expected_vision_mode!r}."
        )
    raw_frame = raw_action_frame_from_data(data)
    ckpt_frame = str(getattr(controller, "action_frame", "world")).lower()
    if raw_frame != ckpt_frame:
        raise ValueError(f"Warm-start raw action_frame={raw_frame!r}, checkpoint action_frame={ckpt_frame!r}.")
    raw_state_version = raw_ikpush_state_version_from_data(data)
    ckpt_state_version = str(controller.config.get("ikpush_state_version", "legacy"))
    if raw_state_version != ckpt_state_version:
        raise ValueError(
            f"Warm-start raw ikpush_state_version={raw_state_version!r}, "
            f"checkpoint ikpush_state_version={ckpt_state_version!r}."
        )
    if "door_cfg" in data.files:
        raw_cfg = normalize_config_path_for_match(data["door_cfg"])
        play_cfg = normalize_config_path_for_match(args.door_cfg)
        if raw_cfg != play_cfg:
            raise ValueError(f"Warm-start raw door_cfg={raw_cfg!r}, play door_cfg={play_cfg!r}.")
    if "door_asset_name" in data.files:
        raw_name = scalar_to_str(data["door_asset_name"])
        play_name = str(st.door.spec.get("name", ""))
        if raw_name and play_name and raw_name != play_name:
            raise ValueError(f"Warm-start raw door_asset_name={raw_name!r}, play door_asset_name={play_name!r}.")
    elif "door_asset_index" in data.files:
        raw_index = scalar_to_int(data["door_asset_index"])
        play_index = int(getattr(st.door, "asset_index", -1))
        if play_index >= 0 and raw_index != play_index:
            raise ValueError(f"Warm-start raw door_asset_index={raw_index}, play door_asset_index={play_index}.")

    step = int(args.dp_warmstart_step)
    for key in (
        "replay_root_state",
        "replay_dof_pos",
        "replay_dof_vel",
        "replay_door_root_state",
        "replay_door_dof_pos",
        "replay_door_dof_vel",
    ):
        require_warmstart_field(data, key, step)
    if args.dp_warmstart_expert_obs:
        if raw_image_keys_for_vision_mode is None:
            raise RuntimeError("Warm-start expert observation prefill requires door_dp_common.raw_image_keys_for_vision_mode.")
        missing = [key for key in raw_image_keys_for_vision_mode(expected_vision_mode) if key not in data.files]
        if missing:
            raise KeyError(f"Warm-start expert observation prefill is missing raw image fields: {missing}")


def raw_dp_dofs_to_actor_dofs(raw_pos, raw_vel, dof_names, fallback_pos):
    pos_out = np.asarray(fallback_pos, dtype=np.float32).copy()
    vel_out = np.zeros_like(pos_out, dtype=np.float32)
    raw_pos = np.asarray(raw_pos, dtype=np.float32).reshape(-1)
    raw_vel = np.asarray(raw_vel, dtype=np.float32).reshape(-1)
    if raw_pos.shape[0] == len(dof_names):
        pos_out[:] = raw_pos[: len(pos_out)]
        vel_out[:] = raw_vel[: len(vel_out)]
        return pos_out, vel_out
    if raw_pos.shape[0] != DP_NUM_DOFS:
        raise ValueError(f"Expected replay_dof_pos length {DP_NUM_DOFS} or {len(dof_names)}, got {raw_pos.shape[0]}.")
    for src_idx, name in enumerate(dof_names):
        dp_idx = FLOAT_ARM_TO_DP_DOF.get(name)
        if dp_idx is None:
            continue
        pos_out[src_idx] = float(raw_pos[dp_idx])
        if dp_idx < raw_vel.shape[0]:
            vel_out[src_idx] = float(raw_vel[dp_idx])
    return pos_out, vel_out


def apply_warmstart_state(gym, sim, st, data, step, dof_names):
    root_state = require_warmstart_field(data, "replay_root_state", step)[step].astype(np.float32)
    for actor in st.actor_handles:
        set_actor_root_from_state(gym, st.env, actor, root_state)

    raw_dof_pos = require_warmstart_field(data, "replay_dof_pos", step)[step]
    raw_dof_vel = require_warmstart_field(data, "replay_dof_vel", step)[step]
    actor_dof_pos, actor_dof_vel = raw_dp_dofs_to_actor_dofs(raw_dof_pos, raw_dof_vel, dof_names, st.dof_positions)
    arm_states = gym.get_actor_dof_states(st.env, st.arm_actor, gymapi.STATE_ALL)
    if len(arm_states) != len(actor_dof_pos):
        raise ValueError(f"Arm DOF count mismatch: actor={len(arm_states)} warmstart={len(actor_dof_pos)}")
    arm_states["pos"][:] = actor_dof_pos
    arm_states["vel"][:] = actor_dof_vel
    gym.set_actor_dof_states(st.env, st.arm_actor, arm_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(st.env, st.arm_actor, actor_dof_pos)
    st.dof_positions[:] = actor_dof_pos

    door_root_state = require_warmstart_field(data, "replay_door_root_state", step)[step].astype(np.float32)
    set_actor_root_from_state(gym, st.env, st.door_actor, door_root_state)

    door_pos = require_warmstart_field(data, "replay_door_dof_pos", step)[step].astype(np.float32)
    door_vel = require_warmstart_field(data, "replay_door_dof_vel", step)[step].astype(np.float32)
    door_states = gym.get_actor_dof_states(st.env, st.door_actor, gymapi.STATE_ALL)
    n = min(len(door_states), len(door_pos))
    door_states["pos"][:n] = door_pos[:n]
    door_states["vel"][:n] = door_vel[:n]
    gym.set_actor_dof_states(st.env, st.door_actor, door_states, gymapi.STATE_ALL)
    if n >= 1 and abs(float(door_pos[0]) - float(st.door.dof_upper[0])) > 1.0e-4:
        st.door.open_stage = True
    if n >= 2 and float(door_pos[1] - st.door.dof_lower[1]) >= float(st.door.handle_unlock_threshold):
        st.door.open_stage = True

    yaw = yaw_from_quat_xyzw(root_state[3:7])
    st.traj["base_xy"] = np.asarray(root_state[:2], dtype=np.float32).copy()
    st.traj["yaw"] = float(yaw)
    st.base_start = np.asarray(root_state[:2], dtype=np.float32).copy()
    st.yaw_start = float(yaw)
    if int(step) > 0 and "replay_root_state" in data.files:
        prev_root = np.asarray(data["replay_root_state"][int(step) - 1], dtype=np.float32)
        st.prev_base_xy = prev_root[:2].copy()
        st.prev_yaw = yaw_from_quat_xyzw(prev_root[3:7])
    else:
        st.prev_base_xy = np.asarray(root_state[:2], dtype=np.float32).copy()
        st.prev_yaw = float(yaw)
    if "action" in data.files:
        action_index = max(0, int(step) - 1)
        st.last_dp_action = np.asarray(data["action"][action_index], dtype=np.float32).copy()
    else:
        st.last_dp_action = np.zeros(10, dtype=np.float32)
    if "replay_ee_pos" in data.files:
        st.last_target_pos = np.asarray(data["replay_ee_pos"][int(step)], dtype=np.float32).copy()
    if "replay_ee_quat" in data.files:
        st.last_target_quat = base_ik.normalize_quat(np.asarray(data["replay_ee_quat"][int(step)], dtype=np.float32))

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)


def prefill_dp_controller_from_expert_obs(controller, data, step, vision_mode):
    if raw_image_keys_for_vision_mode is None:
        raise RuntimeError("Expert observation warm-start requires door_dp_common.raw_image_keys_for_vision_mode.")
    image_keys = raw_image_keys_for_vision_mode(vision_mode)
    controller.obs_buffer.clear()
    controller.action_queue.clear()
    start = max(0, int(step) - int(controller.obs_horizon) + 1)
    for idx in range(start, int(step) + 1):
        controller.append_observation(
            np.asarray(data["state"][idx], dtype=np.float32),
            np.asarray(data[image_keys[0]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[1]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[2]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[3]][idx], dtype=np.uint8),
        )


def apply_dp_warmstart_if_requested(gym, sim, args, controller, st, dof_names):
    if not args.dp_warmstart:
        return None
    raw_path = Path(args.dp_warmstart_raw_episode).expanduser()
    if not raw_path.is_absolute():
        raw_path = (Path.cwd() / raw_path).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Warm-start raw episode not found: {raw_path}")
    data = np.load(raw_path, allow_pickle=True)
    step = int(args.dp_warmstart_step)
    validate_warmstart_raw(data, args, controller, st)
    apply_warmstart_state(gym, sim, st, data, step, dof_names)
    vision_mode = "rgb" if args.rgb else "depth"
    if args.dp_warmstart_expert_obs:
        prefill_dp_controller_from_expert_obs(controller, data, step, vision_mode)
    print(
        f"DP warm-start loaded raw={raw_path} step={step} "
        f"expert_obs_prefill={bool(args.dp_warmstart_expert_obs)} "
        f"base_xy={np.round(st.traj['base_xy'], 4).tolist()} yaw={float(st.traj['yaw']):.4f} "
        f"door_open_stage={bool(st.door.open_stage)}",
        flush=True,
    )
    return data


def door_hinge_open_ratio(door, door_angle, args):
    if len(door.dof_lower) == 0 or len(door.dof_upper) == 0:
        return 0.0
    if args.door_motion_sign < 0.0:
        closed_angle = float(door.dof_upper[0])
        open_limit = float(door.dof_lower[0])
    else:
        closed_angle = float(door.dof_lower[0])
        open_limit = float(door.dof_upper[0])
    hinge_range = max(abs(open_limit - closed_angle), 1.0e-6)
    return float(np.clip(abs(float(door_angle) - closed_angle) / hinge_range, 0.0, 1.0))


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


def enforce_locked_door_hinge(gym, env, door_actor, door):
    if door.open_stage:
        return
    states = gym.get_actor_dof_states(env, door_actor, gymapi.STATE_ALL)
    if len(states) == 0:
        return
    states["pos"][0] = door.dof_upper[0]
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
    ik_state.current_quat_np = eef_quat.detach().cpu().numpy().copy()


def refresh_current_ee_pose(gym, sim, ik_state):
    gym.refresh_rigid_body_state_tensor(sim)
    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    ik_state.current_pos_np = eef_state[:3].detach().cpu().numpy().copy()
    ik_state.current_quat_np = eef_state[3:7].detach().cpu().numpy().copy()


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
    base_push,
    yaw_start,
    yaw_push,
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
    goal_quat = forward_ee_quat(args, yaw_start)

    rotate_offset = np.zeros(3, dtype=np.float32)
    rotate_offset[1] = args.handle_rotate_right_distance
    rotate_offset[2] = -args.handle_rotate_down_distance
    rotate_pos = grasp + rotate_offset
    pull_dir = quat_axis(handle_quat, axis=2)
    pull_dir[2] = 0.0
    fallback_pull_dir = approach_dir.copy()
    pull_dir = normalize(pull_dir)
    if np.linalg.norm(pull_dir) < 1.0e-5:
        pull_dir = fallback_pull_dir
    if float(np.dot(pull_dir, approach_dir)) < 0.0:
        pull_dir = -pull_dir
    push_dir = -pull_dir
    push_pos = rotate_pos + push_dir * args.door_push_distance

    walk_end = args.walk_steps
    initial_end = walk_end + args.initial_hold_steps
    grasp_end = initial_end + args.grasp_steps
    grasp_hold_end = grasp_end + args.grasp_hold_steps
    close_end = grasp_hold_end + args.gripper_close_steps
    rotate_end = close_end + args.handle_rotate_steps
    push_end = rotate_end + args.door_push_steps
    return_home_end = push_end + args.return_home_steps

    gripper_closed = args.gripper_open + (args.gripper_closed - args.gripper_open) * args.gripper_close_ratio
    gripper_open_stage = args.gripper_open + (
        args.gripper_closed - args.gripper_open
    ) * args.gripper_open_stage_ratio
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
            traj["push"] = push_pos.copy()
            traj["goal_quat"] = goal_quat.copy()
            traj["push_dir"] = push_dir.copy()
            traj["approach_dir"] = approach_dir.copy()
            traj["initial_hold_start_pos"] = (
                ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else pregrasp.copy()
            )
            if not args.ik_position_only:
                start_quat = ik_state.target_quat_np
                if start_quat is None:
                    start_quat = ik_state.current_quat_np if ik_state.current_quat_np is not None else goal_quat
                traj["initial_hold_start_quat"] = base_ik.normalize_quat(start_quat).astype(np.float32)

        base_xy = base_stop.copy()
        target_pos = traj["pregrasp"].copy()
        target_quat = None if args.ik_position_only else traj["goal_quat"].copy()
        phase = "initial_hold"

        if step < initial_end:
            initial_step = step - walk_end
            move_steps = max(1, int(round(max(1, args.initial_hold_steps) * 0.5)))
            if initial_step < move_steps:
                t = smoothstep((initial_step + 1) / move_steps)
                target_pos = lerp(traj["initial_hold_start_pos"], traj["pregrasp"], t)
                if not args.ik_position_only:
                    target_quat = quat_nlerp(traj["initial_hold_start_quat"], traj["goal_quat"], t)
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
        elif step < push_end:
            t = smoothstep((step - rotate_end + 1) / max(1, args.door_push_steps))
            base_t = smoothstep((step - rotate_end + 1) / max(1.0, args.door_push_steps * args.base_push_time_scale))
            turned_quat = base_ik.quat_multiply(
                traj["goal_quat"],
                quat_from_angle_axis(-args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            if "handle_contact_offset_local" not in traj:
                traj["handle_contact_offset_local"] = quat_apply(
                    base_ik.quat_conjugate(handle_quat),
                    traj["rotate"] - handle_goal,
                )
                traj["handle_contact_quat_local"] = base_ik.quat_multiply(
                    base_ik.quat_conjugate(handle_quat),
                    turned_quat,
                )
            live_push_dir = traj["push_dir"].copy()
            if door.open_stage:
                live_pull_dir = quat_axis(handle_quat, axis=2)
                live_pull_dir[2] = 0.0
                live_pull_dir = normalize(live_pull_dir)
                if np.linalg.norm(live_pull_dir) >= 1.0e-5:
                    if float(np.dot(live_pull_dir, traj["approach_dir"])) < 0.0:
                        live_pull_dir = -live_pull_dir
                    live_push_dir = -live_pull_dir

            follow_handle = (
                door.open_stage
                and t <= args.handle_follow_push_ratio
                and "handle_contact_offset_local" in traj
            )
            freeze_ee_target = False
            door_pos, _ = get_actor_dof_state(gym, env, door_actor)
            door_open_ratio = (
                door_hinge_open_ratio(door, float(door_pos[0]), args) if len(door_pos) > 0 else 0.0
            )
            if "handle_contact_offset_local" in traj:
                handle_contact_offset = quat_apply(handle_quat, traj["handle_contact_offset_local"])
                handle_target_pos = handle_goal + handle_contact_offset + live_push_dir * args.push_contact_bias
            else:
                handle_target_pos = handle_goal.copy()
            if (
                door.open_stage
                and len(door_pos) > 0
                and ik_state.current_pos_np is not None
                and door_open_ratio >= args.door_freeze_target_ratio
            ):
                if "door_open_freeze_target_pos" not in traj:
                    traj["door_open_freeze_target_pos"] = handle_target_pos.copy()
                target_pos = traj["door_open_freeze_target_pos"].copy()
                target_quat = None
                freeze_ee_target = True
            elif follow_handle:
                target_pos = handle_target_pos.copy()
                target_quat = None
                if args.push_follow_orientation and not args.ik_position_only and "handle_contact_quat_local" in traj:
                    target_quat = base_ik.quat_multiply(handle_quat, traj["handle_contact_quat_local"])
            elif door.open_stage and ik_state.current_pos_np is not None:
                target_pos = ik_state.current_pos_np + live_push_dir * args.lever_step_size
                blend_start = min(float(args.door_freeze_blend_start_ratio), float(args.door_freeze_target_ratio) - 1.0e-4)
                if door_open_ratio >= blend_start:
                    blend_t = smoothstep(
                        (door_open_ratio - blend_start)
                        / max(1.0e-4, float(args.door_freeze_target_ratio) - blend_start)
                    )
                    target_pos = lerp(target_pos, handle_target_pos, blend_t)
                target_quat = None
            elif args.unidoor_style_push and ik_state.current_pos_np is not None:
                push_step_pos = ik_state.current_pos_np + traj["push_dir"] * args.lever_step_size
                push_max_pos = traj["rotate"] + traj["push_dir"] * args.door_push_distance
                progress = float(np.dot(push_step_pos - traj["rotate"], traj["push_dir"]))
                target_pos = push_max_pos if progress > args.door_push_distance else push_step_pos
            else:
                target_pos = lerp(traj["rotate"], traj["push"], t)
            if freeze_ee_target:
                target_quat = None
            else:
                target_quat = None if args.ik_position_only else turned_quat
            base_xy = lerp(base_stop, base_push, base_t)
            yaw = float(lerp(np.array([yaw_start], dtype=np.float32), np.array([yaw_push], dtype=np.float32), base_t)[0])
            gripper = gripper_closed
            if door.open_stage:
                if "gripper_loosen_start_step" not in traj:
                    traj["gripper_loosen_start_step"] = step
                loosen_t = smoothstep(
                    (step - traj["gripper_loosen_start_step"] + 1) / max(1, args.gripper_loosen_steps)
                )
                gripper = float(lerp(
                    np.array([gripper_closed], dtype=np.float32),
                    np.array([gripper_open_stage], dtype=np.float32),
                    loosen_t,
                )[0])
            phase = "push_door"
        elif step < return_home_end:
            t = smoothstep((step - push_end + 1) / max(1, args.return_home_steps))
            target_pos = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else traj["push"].copy()
            target_quat = None if args.ik_position_only else base_ik.quat_multiply(
                traj["goal_quat"],
                quat_from_angle_axis(-args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            if "return_home_start_base_xy" not in traj:
                traj["return_home_start_base_xy"] = traj.get("base_xy", base_push).copy()
                traj["return_home_start_yaw"] = float(traj.get("yaw", yaw_push))
            base_xy = lerp(traj["return_home_start_base_xy"], base_push, t)
            yaw = float(lerp(
                np.array([traj["return_home_start_yaw"]], dtype=np.float32),
                np.array([yaw_push], dtype=np.float32),
                t,
            )[0])
            gripper = args.gripper_open
            traj["return_home_alpha"] = t
            phase = "return_home"
        else:
            target_pos = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else traj["push"].copy()
            target_quat = None
            base_xy = base_push.copy()
            yaw = yaw_push
            gripper = args.gripper_open
            traj["return_home_alpha"] = 1.0
            phase = "hold_home"

    traj["base_xy"] = base_xy.copy()
    traj["yaw"] = float(yaw)
    return phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal


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


def run_demo(
    gym,
    sim,
    env,
    arm_actor,
    actor_handles,
    door,
    door_actor,
    viewer,
    camera_handles,
    args,
    dt,
    dof_names,
    dof_positions,
    defaults,
    ik_state,
):
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
    base_push = compute_base_push_target(args, base_stop, heading)
    yaw_push = yaw_start + args.push_base_yaw_delta
    traj = {"base_xy": base_start.copy()}

    print("base_start:", base_start.tolist(), "base_stop:", base_stop.tolist(), "base_push:", base_push.tolist())
    print(
        "pass_through_door:",
        bool(args.pass_through_door),
        "rear_offset:",
        float(args.robot_rear_offset),
        "door_pass_clearance:",
        float(args.door_pass_clearance),
    )
    print("Close viewer to exit.")
    start = time.time()
    step = 0
    home_positions = np.asarray(defaults, dtype=np.float32).copy()
    if gripper_idx is not None:
        home_positions[gripper_idx] = np.clip(args.gripper_open, ik_state.lower[gripper_idx].item(), ik_state.upper[gripper_idx].item())

    max_steps = args.steps if args.steps > 0 else 2900
    dp_recorder = None
    dp_record_success = False
    dp_record_warned_no_camera = False
    if args.record_dp_dataset:
        if RawDoorDPRecorder is None or make_state_feature_names is None:
            raise RuntimeError("DP raw recording requires high-level/dp/door_dp_common.py.")
        if not camera_handles:
            print(
                "⚠️📷 DP raw recording requested, but no camera sensors were created; episode frames will be discarded.",
                flush=True,
            )
        if args.headless:
            print(
                "⚠️📷 Headless DP recording needs camera rendering; if the graphics device cannot render cameras, "
                "raw frames will be skipped with a camera-unavailable warning.",
                flush=True,
            )
        vision_mode = "rgb" if args.rgb else "depth"
        if normalize_vision_mode is not None:
            vision_mode = normalize_vision_mode(vision_mode)
        dp_recorder = RawDoorDPRecorder(
            raw_root=args.dp_raw_root,
            fps=args.dp_fps,
            state_feature_names=make_state_feature_names(DP_NUM_DOFS, DP_NUM_ACTIONS, DP_PHASE_NAMES),
            task=args.dp_task,
            vision_mode=vision_mode,
            metadata={
                "door_asset_index": int(args.door_index) if int(args.door_index) >= 0 else 0,
                "door_asset_name": door.spec.get("name", ""),
                "door_asset_path": door.spec.get("path", ""),
                "door_cfg": str(args.door_cfg),
                "source_script": Path(__file__).name,
                "action_frame": "base",
                "action_pose_frame": "base",
                "target_pose_frame": "base",
                "ikpush_state_version": IKPUSH_STATE_VERSION,
            },
        )
        print(
            f"Recording raw Door DP dataset to {args.dp_raw_root} task={args.dp_task!r} "
            f"env_ids=[0] success_angle_deg={args.pass_open_angle_deg} vision_mode={vision_mode}",
            flush=True,
        )
    prev_base_xy = None
    prev_yaw = None
    prev_dp_action = np.zeros(10, dtype=np.float32)
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
            base_push,
            yaw_start,
            yaw_push,
            traj,
        )
        set_robot_base_pose(gym, env, actor_handles, base_xy, args.robot_z, yaw)
        if phase == "return_home":
            if "return_home_start_dofs" not in traj:
                traj["return_home_start_dofs"] = np.asarray(dof_positions, dtype=np.float32).copy()
            alpha = float(traj.get("return_home_alpha", 0.0))
            dof_positions[:] = lerp(traj["return_home_start_dofs"], home_positions, alpha)
            ik_state.last_pos_error = 0.0
        elif phase == "hold_home":
            dof_positions[:] = home_positions
            ik_state.last_pos_error = 0.0
        else:
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

        need_camera_render = bool(camera_handles and (args.show_camera_images or args.record_dp_dataset))
        if viewer is not None or need_camera_render:
            gym.step_graphics(sim)
        if args.show_camera_images and camera_handles and step % max(1, args.camera_display_interval) == 0:
            show_camera_handle_images(gym, sim, env, camera_handles, args)
        door_pos_record, door_vel_record = get_actor_dof_state(gym, env, door_actor)

        if dp_recorder is not None:
            ee_pos, ee_quat = current_ee_pose(gym, sim, ik_state)
            dof_pos_actual, dof_vel_actual = get_actor_dof_state(gym, env, arm_actor)
            gripper_actual = float(dof_pos_actual[gripper_idx]) if gripper_idx is not None and gripper_idx < len(dof_pos_actual) else float(gripper)
            vx_cmd, yaw_rate_cmd = base_command_from_targets(base_xy, yaw, prev_base_xy, prev_yaw, dt)
            dp_target_quat = target_quat_for_dp(target_quat, ik_state, ee_quat)
            camera_images = capture_dp_camera_images(gym, sim, env, camera_handles, args)
            wrist_mask_rgb, wrist_second_rgb, front_mask_rgb, front_second_rgb = dp_image_inputs_from_cpu_cameras(camera_images, args)
            missing_required_camera = wrist_mask_rgb is None or wrist_second_rgb is None or (
                args.rgb and (front_mask_rgb is None or front_second_rgb is None)
            )
            if missing_required_camera:
                if not dp_record_warned_no_camera:
                    missing_desc = "wrist/front camera RGB or mask images" if args.rgb else "wrist camera mask/depth images"
                    print(
                        f"⚠️📷 Camera unavailable: skipped DP frame because {missing_desc} are missing. "
                        "No raw DP frame can be saved until camera images are available.",
                        flush=True,
                    )
                    dp_record_warned_no_camera = True
            else:
                dp_state = make_float_dp_state(
                    dof_names,
                    dof_pos_actual,
                    dof_vel_actual,
                    ee_pos,
                    ee_quat,
                    base_xy,
                    args.robot_z,
                    yaw,
                    yaw_rate_cmd,
                    gripper_actual,
                    prev_dp_action,
                )
                dp_action = make_float_dp_action(
                    vx_cmd,
                    yaw_rate_cmd,
                    target_pos,
                    dp_target_quat,
                    gripper,
                    base_xy,
                    args.robot_z,
                    yaw,
                )
                dp_recorder.add_frame(
                    dp_state,
                    wrist_mask_rgb,
                    wrist_second_rgb,
                    dp_action,
                    DP_PHASE_ID.get(phase, 0),
                    front_mask_rgb=front_mask_rgb,
                    front_second_rgb=front_second_rgb,
                    replay_snapshot=make_float_replay_snapshot(
                        args,
                        door,
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
                prev_dp_action = dp_action.copy()
            if len(door_pos_record) > 0:
                signed_open_deg = math.degrees(args.door_motion_sign * float(door_pos_record[0]))
                dp_record_success = dp_record_success or signed_open_deg >= float(args.pass_open_angle_deg)

        if viewer is not None:
            if args.draw_ik_target or args.draw_camera_axes:
                gym.clear_lines(viewer)
            if args.draw_camera_axes:
                draw_low_level_camera_axes(gym, viewer, env, arm_actor, actor_handles, args)
            if args.draw_ik_target:
                if phase in ("return_home", "hold_home"):
                    refresh_current_ee_pose(gym, sim, ik_state)
                    saved_target_pos_np = ik_state.target_pos_np
                    saved_target_quat_np = ik_state.target_quat_np
                    ik_state.target_pos_np = ik_state.current_pos_np.copy()
                    ik_state.target_quat_np = None
                    try:
                        base_ik.draw_ik_target(gym, viewer, env, ik_state)
                    finally:
                        ik_state.target_pos_np = saved_target_pos_np
                        ik_state.target_quat_np = saved_target_quat_np
                else:
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
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"[{step:04d}] phase={phase:14s} "
                f"ik_pos_err={ik_state.last_pos_error:.4f} "
                f"door={math.degrees(float(door_pos[0])) if len(door_pos) else 0.0:.1f}deg "
                f"signed_push={math.degrees(args.door_motion_sign * float(door_pos[0])) if len(door_pos) else 0.0:.1f}deg "
                f"handle={math.degrees(float(door_pos[1])) if len(door_pos) > 1 else 0.0:.1f}deg "
                f"open_stage={door.open_stage}",
                flush=True,
            )
        prev_base_xy = np.asarray(base_xy, dtype=np.float32).copy()
        prev_yaw = float(yaw)
        step += 1

    print(f"Done after {step} steps ({time.time() - start:.2f}s).")
    if dp_recorder is not None:
        if dp_record_success and dp_recorder.frame_count > 0:
            dp_recorder.save_episode()
            print(f"Finished raw Door DP recording: saved_successful=1/1 frames={dp_recorder.frame_count}", flush=True)
        elif dp_recorder.frame_count == 0:
            print(
                "⚠️📷 Finished raw Door DP recording: saved_successful=0/1 frames=0 reason=camera_unavailable",
                flush=True,
            )
        else:
            print(
                f"Finished raw Door DP recording: saved_successful=0/1 frames={dp_recorder.frame_count} "
                f"reason=door did not reach {args.pass_open_angle_deg} deg",
                flush=True,
            )
        dp_recorder.finalize()


def make_parallel_dp_recorder(args, door, env_index, vision_mode):
    return RawDoorDPRecorder(
        raw_root=args.dp_raw_root,
        fps=args.dp_fps,
        state_feature_names=make_state_feature_names(DP_NUM_DOFS, DP_NUM_ACTIONS, DP_PHASE_NAMES),
        task=args.dp_task,
        vision_mode=vision_mode,
        metadata={
            "door_asset_index": int(getattr(door, "asset_index", 0)),
            "door_asset_name": door.spec.get("name", ""),
            "door_asset_path": door.spec.get("path", ""),
            "door_cfg": str(args.door_cfg),
            "source_script": Path(__file__).name,
            "action_frame": "base",
            "action_pose_frame": "base",
            "target_pose_frame": "base",
            "ikpush_state_version": IKPUSH_STATE_VERSION,
            "parallel_env_id": int(env_index),
            "parallel_num_envs": int(args.num_envs),
            "seed": int(getattr(args, "seed", -1)),
            "env_seed": int(getattr(args, "env_seed", -1)),
            "robot_x": float(args.robot_x),
            "robot_y": float(args.robot_y),
            "robot_yaw": float(args.robot_yaw),
            "pregrasp_offset": float(args.pregrasp_offset),
            "grasp_x_offset": float(args.grasp_x_offset),
            "grasp_z_offset": float(args.grasp_z_offset),
            "handle_rotate_angle": float(args.handle_rotate_angle),
            "door_push_distance": float(args.door_push_distance),
            "door_joint_friction": float(args.door_joint_friction),
            "door_joint_damping": float(args.door_joint_damping),
            "handle_joint_friction": float(args.handle_joint_friction),
            "handle_joint_damping": float(args.handle_joint_damping),
            "handle_spring_stiffness": float(args.handle_spring_stiffness),
            "handle_spring_damping": float(args.handle_spring_damping),
            "ikpush_randomization": str(getattr(args, "ikpush_randomization_json", "")),
        },
    )


def initialize_parallel_env_state(
    index,
    env,
    arm_actor,
    actor_handles,
    door,
    door_actor,
    camera_handles,
    ik_state,
    args,
    dof_names,
    dof_positions_template,
    defaults,
    dp_recorder,
):
    dof_positions = np.asarray(dof_positions_template, dtype=np.float32).copy()
    home_positions = np.asarray(defaults, dtype=np.float32).copy()
    gripper_idx = {name: i for i, name in enumerate(dof_names)}.get("jointGripper")
    if gripper_idx is not None:
        dof_positions[gripper_idx] = args.gripper_open
        home_positions[gripper_idx] = np.clip(
            args.gripper_open,
            ik_state.lower[gripper_idx].item(),
            ik_state.upper[gripper_idx].item(),
        )

    yaw_start = float(args.robot_yaw)
    heading = np.array([math.cos(yaw_start), math.sin(yaw_start)], dtype=np.float32)
    base_start = np.asarray([args.robot_x, robot_y_for_door(args, door.handle_bounding)], dtype=np.float32)
    robot_front = base_start + heading * args.robot_front_offset
    door_xy = np.asarray([args.door_x, args.door_y], dtype=np.float32)
    front_to_door = float(np.dot(door_xy - robot_front, heading))
    walk_dist = max(0.0, front_to_door - args.stop_distance)
    base_stop = base_start + heading * walk_dist
    base_push = compute_base_push_target(args, base_stop, heading)
    return ParallelEnvState(
        index=int(index),
        args=args,
        env=env,
        arm_actor=arm_actor,
        actor_handles=actor_handles,
        door=door,
        door_actor=door_actor,
        camera_handles=camera_handles,
        ik_state=ik_state,
        dof_positions=dof_positions,
        home_positions=home_positions,
        base_start=base_start,
        base_stop=base_stop,
        base_push=base_push,
        yaw_start=yaw_start,
        yaw_push=yaw_start + args.push_base_yaw_delta,
        traj={"base_xy": base_start.copy()},
        dp_recorder=dp_recorder,
    )


def create_parallel_env_states(
    gym,
    sim,
    base_asset,
    arm_asset,
    door_templates,
    dof_props,
    dof_states,
    dof_positions,
    lower,
    upper,
    defaults,
    dof_names,
    args,
):
    if args.record_dp_dataset and (RawDoorDPRecorder is None or make_state_feature_names is None):
        raise RuntimeError("DP raw recording requires high-level/dp/door_dp_common.py.")
    vision_mode = "rgb" if args.rgb else "depth"
    if normalize_vision_mode is not None:
        vision_mode = normalize_vision_mode(vision_mode)

    envs_per_row = max(1, int(math.ceil(math.sqrt(float(args.num_envs)))))
    record_env_ids = set(range(args.num_envs)) if args.dp_record_all_envs else {int(args.dp_record_env_id)}
    created = []
    for env_index in range(int(args.num_envs)):
        door_template = door_templates[env_index % len(door_templates)]
        door = clone_door_runtime(door_template)
        env_args = make_env_args(args, env_index)
        env, arm_actor, actor_handles, door_actor, _ = create_parallel_env_actors(
            gym,
            sim,
            base_asset,
            arm_asset,
            door,
            dof_props,
            dof_states,
            env_args,
            env_index,
            envs_per_row,
        )
        created.append((env_index, env_args, env, arm_actor, actor_handles, door, door_actor))

    env_states = []
    for env_index, env_args, env, arm_actor, actor_handles, door, door_actor in created:
        camera_handles = {}
        if (env_args.show_camera_images or env_args.record_dp_dataset) and (
            env_args.enable_wrist_camera or env_args.enable_front_camera
        ):
            camera_handles = create_low_level_cameras(gym, env, arm_actor, actor_handles, env_args)
        ik_state = base_ik.setup_ik_controller(gym, sim, env, arm_actor, arm_asset, dof_names, lower, upper, env_args)
        dp_recorder = None
        if env_args.record_dp_dataset and env_index in record_env_ids:
            dp_recorder = make_parallel_dp_recorder(env_args, door, env_index, vision_mode)
        env_states.append(
            initialize_parallel_env_state(
                env_index,
                env,
                arm_actor,
                actor_handles,
                door,
                door_actor,
                camera_handles,
                ik_state,
                env_args,
                dof_names,
                dof_positions,
                defaults,
                dp_recorder,
            )
        )
    if args.record_dp_dataset:
        if args.headless:
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
    shown_randomization = [json.loads(st.args.ikpush_randomization_json) for st in env_states[: min(4, len(env_states))]]
    print(
        f"ikpush per-env randomization seed={int(args.seed)} "
        f"enabled={not bool(args.no_ikpush_env_randomization)} "
        f"door_cycle={[door.spec.get('name', '') for door in door_templates]} "
        f"sample_envs={shown_randomization}",
        flush=True,
    )
    return env_states, vision_mode


def run_parallel_demo(gym, sim, env_states, viewer, args, dt, dof_names):
    if not env_states:
        raise RuntimeError("No parallel envs were created.")
    num_arm_dofs = len(env_states[0].dof_positions)
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    gripper_idx = dof_dict.get("jointGripper")
    max_steps = args.steps if args.steps > 0 else 2900
    start = time.time()
    step = 0
    print(
        f"Parallel float_ik run: num_envs={len(env_states)} steps={max_steps} "
        f"recorders={sum(st.dp_recorder is not None for st in env_states)}",
        flush=True,
    )
    first = env_states[0]
    print("base_start:", first.base_start.tolist(), "base_stop:", first.base_stop.tolist(), "base_push:", first.base_push.tolist())
    print(
        "pass_through_door:",
        bool(args.pass_through_door),
        "rear_offset:",
        float(args.robot_rear_offset),
        "door_pass_clearance:",
        float(args.door_pass_clearance),
    )
    print("Close viewer to exit.")
    dp_controller = None
    dp_logger = None
    dp_control_state = None
    if args.dp_policy_checkpoint:
        if DoorDPPolicyController is None:
            raise RuntimeError("DP policy execution requires high-level/dp/door_dp_common.py and diffusers.")
        dp_controller = DoorDPPolicyController(
            args.dp_policy_checkpoint,
            device=args.rl_device,
            num_inference_steps=args.dp_inference_steps,
            action_horizon=args.dp_action_horizon,
        )
        expected_vision_mode = "rgb" if args.rgb else "depth"
        if getattr(dp_controller, "vision_mode", "depth") != expected_vision_mode:
            raise ValueError(
                f"DP checkpoint vision_mode={getattr(dp_controller, 'vision_mode', 'depth')!r}, "
                f"but ikpush play was run with {expected_vision_mode!r}."
            )
        if getattr(dp_controller, "action_frame", "world") not in ("world", "base"):
            raise ValueError(
                f"DP checkpoint action_frame={getattr(dp_controller, 'action_frame', None)!r}; "
                "expected 'world' or 'base'."
            )
        checkpoint_state_version = str(dp_controller.config.get("ikpush_state_version", "legacy"))
        if checkpoint_state_version != IKPUSH_STATE_VERSION:
            raise ValueError(
                f"DP checkpoint ikpush_state_version={checkpoint_state_version!r}, "
                f"but this ikpush play script emits {IKPUSH_STATE_VERSION!r}. "
                "Reconvert/retrain with the current ikpush state format."
            )
        dp_control_state = env_states[int(args.dp_control_env_id)]
        dp_control_state.dp_action_frame = getattr(dp_controller, "action_frame", "world")
        if not dp_control_state.camera_handles:
            raise RuntimeError("ikpush DP policy execution requires camera sensors; do not disable wrist/front cameras.")
        print(
            f"Loaded Door DP policy from {args.dp_policy_checkpoint} "
            f"action_frame={getattr(dp_controller, 'action_frame', 'world')}",
            flush=True,
        )
        print(
            f"Door DP controls only env {args.dp_control_env_id}; other envs keep the scripted target trajectory.",
            flush=True,
        )
        if args.dp_log_path:
            if DoorDPJsonlLogger is None:
                raise RuntimeError("DP policy logging requires high-level/dp/door_dp_common.py")
            dp_logger = DoorDPJsonlLogger(args.dp_log_path)
            print(f"Door DP log: {args.dp_log_path}", flush=True)
        apply_dp_warmstart_if_requested(gym, sim, args, dp_controller, dp_control_state, dof_names)

    while step < max_steps:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        if dp_controller is not None:
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            if dp_controller is not None and st.index == int(args.dp_control_env_id):
                phase = "dp_policy"
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
                    raise RuntimeError(f"ikpush DP policy execution cannot run because {missing_desc} are missing.")
                dp_action = dp_controller.act(
                    dp_state,
                    wrist_mask_rgb,
                    wrist_second_rgb,
                    front_mask_rgb,
                    front_second_rgb,
                )
                st.last_dp_action = np.asarray(dp_action, dtype=np.float32).copy()
                base_xy, yaw, target_pos, target_quat, gripper = apply_float_dp_action(
                    dp_action,
                    base_xy_current,
                    st.args.robot_z,
                    yaw_current,
                    dt,
                    action_frame=getattr(dp_controller, "action_frame", "world"),
                )
                st.traj["base_xy"] = np.asarray(base_xy, dtype=np.float32).copy()
                st.traj["yaw"] = float(yaw)
                door_pos_for_log, _door_vel_for_log = get_actor_dof_state(gym, st.env, st.door_actor)
            else:
                phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal = trajectory_targets(
                    step,
                    st.args,
                    st.door,
                    gym,
                    st.env,
                    st.door_actor,
                    st.ik_state,
                    st.base_start,
                    st.base_stop,
                    st.base_push,
                    st.yaw_start,
                    st.yaw_push,
                    st.traj,
                )
                dp_action = None
                dp_state = None
                ee_pos = None
                ee_quat = None
                door_pos_for_log = None
            st.last_phase = phase
            st.last_handle_goal = handle_goal
            st.last_target_pos = np.asarray(target_pos, dtype=np.float32).copy()
            st.last_target_quat = None if target_quat is None else np.asarray(target_quat, dtype=np.float32).copy()
            st.last_gripper = float(gripper)
            if dp_action is not None:
                dp_record = make_float_dp_policy_log_record(
                    step,
                    st,
                    dp_action,
                    dp_state,
                    ee_pos,
                    ee_quat,
                    door_pos_for_log,
                    phase,
                )
                if dp_logger is not None:
                    dp_logger.write(dp_record)
                if args.dp_print and step % max(1, int(args.dp_log_interval)) == 0:
                    print_float_dp_policy_log_record(dp_record)
            set_robot_base_pose(gym, st.env, st.actor_handles, base_xy, st.args.robot_z, yaw)
            if phase == "return_home":
                if "return_home_start_dofs" not in st.traj:
                    st.traj["return_home_start_dofs"] = np.asarray(st.dof_positions, dtype=np.float32).copy()
                alpha = float(st.traj.get("return_home_alpha", 0.0))
                st.dof_positions[:] = lerp(st.traj["return_home_start_dofs"], st.home_positions, alpha)
                st.ik_state.last_pos_error = 0.0
            elif phase == "hold_home":
                st.dof_positions[:] = st.home_positions
                st.ik_state.last_pos_error = 0.0
            else:
                set_ik_target(st.ik_state, target_pos, target_quat)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            if st.last_phase not in ("return_home", "hold_home"):
                update_arm_ik_targets_for_env(
                    gym,
                    st.env,
                    st.arm_actor,
                    st.index,
                    st.dof_positions,
                    st.ik_state,
                    st.args,
                    num_arm_dofs,
                )
                if gripper_idx is not None:
                    st.dof_positions[gripper_idx] = np.clip(
                        st.last_gripper,
                        st.ik_state.lower[gripper_idx].item(),
                        st.ik_state.upper[gripper_idx].item(),
                    )
            gym.set_actor_dof_position_targets(st.env, st.arm_actor, st.dof_positions)

            enforce_locked_door_hinge(gym, st.env, st.door_actor, st.door)
            door_pos, door_vel = get_actor_dof_state(gym, st.env, st.door_actor)
            st.last_door_pos = door_pos
            door_efforts = compute_door_efforts(st.door, door_pos, door_vel, st.args)
            if len(door_efforts) > 0:
                gym.apply_actor_dof_efforts(st.env, st.door_actor, door_efforts)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        need_camera_render = bool(
            any(st.camera_handles for st in env_states) and (args.show_camera_images or args.record_dp_dataset)
        )
        if viewer is not None or need_camera_render:
            gym.step_graphics(sim)
        if need_camera_render:
            gym.render_all_camera_sensors(sim)
        if args.show_camera_images and env_states[0].camera_handles and step % max(1, args.camera_display_interval) == 0:
            show_camera_handle_images(gym, sim, env_states[0].env, env_states[0].camera_handles, env_states[0].args)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            door_pos_record, door_vel_record = get_actor_dof_state(gym, st.env, st.door_actor)
            st.last_door_pos = door_pos_record
            if st.dp_recorder is not None:
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
                else:
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
                        DP_PHASE_ID.get(st.last_phase, 0),
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
                if len(door_pos_record) > 0:
                    signed_open_deg = math.degrees(st.args.door_motion_sign * float(door_pos_record[0]))
                    st.dp_record_success = st.dp_record_success or signed_open_deg >= float(args.pass_open_angle_deg)
            st.prev_base_xy = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32).copy()
            st.prev_yaw = float(st.traj.get("yaw", st.yaw_start))

        if viewer is not None:
            if args.draw_ik_target or args.draw_camera_axes:
                gym.clear_lines(viewer)
            for st in env_states[: min(4, len(env_states))]:
                if args.draw_camera_axes:
                    draw_low_level_camera_axes(gym, viewer, st.env, st.arm_actor, st.actor_handles, st.args)
                if args.draw_ik_target:
                    base_ik.draw_ik_target(gym, viewer, st.env, st.ik_state)
                    if st.last_handle_goal is not None:
                        target_pose = base_ik.transform_from_arrays(st.last_handle_goal)
                        goal_sphere = gymutil.WireframeSphereGeometry(
                            radius=0.035,
                            num_lats=8,
                            num_lons=8,
                            color=(0.0, 1.0, 0.2),
                            color2=(0.0, 0.7, 0.2),
                        )
                        gymutil.draw_lines(goal_sphere, gym, viewer, st.env, target_pose)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if args.log_interval > 0 and step % args.log_interval == 0:
            shown = env_states[: min(len(env_states), 4)]
            door_deg = [
                round(math.degrees(float(st.last_door_pos[0])), 1) if st.last_door_pos is not None and len(st.last_door_pos) else 0.0
                for st in shown
            ]
            phases = [st.last_phase for st in shown]
            successes = sum(st.dp_record_success for st in env_states if st.dp_recorder is not None)
            print(
                f"[{step:04d}] phases={phases} door_deg={door_deg} "
                f"record_success={successes}/{sum(st.dp_recorder is not None for st in env_states)}",
                flush=True,
            )
        step += 1

    elapsed = time.time() - start
    print(f"Done after {step} steps ({elapsed:.2f}s).")
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
    if dp_logger is not None:
        dp_logger.close()


def main():
    args = parse_args()
    seed = resolve_seed(args)
    print(f"ikpush seed={seed}", flush=True)
    gym = gymapi.acquire_gym()
    sim, dt = base_ik.create_sim(gym, args)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    with tempfile.TemporaryDirectory(prefix="b1z1_float_ik_door_assets_") as temp_dir:
        base_asset, arm_asset = base_ik.load_robot_assets(gym, sim, args, Path(temp_dir))
        door_templates = load_door_assets(gym, sim, args)
        if base_asset is not None:
            base_ik.print_collision_summary(gym, base_asset, "base visual actor", verbose=args.print_collision_summary)
        base_ik.print_collision_summary(gym, arm_asset, "arm articulated actor", verbose=args.print_collision_summary)
        dof_data = base_ik.configure_dofs(gym, arm_asset, args)
        dof_names, dof_props, dof_states, dof_positions, lower, upper, defaults, speeds, selected = dof_data
        if "jointGripper" in dof_names:
            dof_states["pos"][dof_names.index("jointGripper")] = args.gripper_open
            dof_positions[dof_names.index("jointGripper")] = args.gripper_open
        env_states, _vision_mode = create_parallel_env_states(
            gym,
            sim,
            base_asset,
            arm_asset,
            door_templates,
            dof_props,
            dof_states,
            dof_positions,
            lower,
            upper,
            defaults,
            dof_names,
            args,
        )
        viewer = setup_viewer(gym, sim, args)
        try:
            run_parallel_demo(gym, sim, env_states, viewer, args, dt, dof_names)
        finally:
            if args.show_camera_images and cv2 is not None:
                cv2.destroyAllWindows()
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
