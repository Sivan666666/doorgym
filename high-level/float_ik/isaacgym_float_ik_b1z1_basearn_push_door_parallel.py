#!/usr/bin/env python3
"""Parallel float-base B1Z1 base+arm IK door-push recorder.

This is the high-throughput variant of isaacgym_float_ik_b1z1_basearn_push_door.py.
It keeps the same asset/controller helpers, but records multiple Isaac Gym envs
inside one simulator process.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = HIGH_LEVEL_ROOT.parents[0]

import door_common as dc

base_ik = dc.base_ik
gymapi = dc.gymapi
gymutil = dc.gymutil
DEFAULT_DOOR_CFG = dc.DEFAULT_DOOR_CFG
DP_NUM_DOFS = dc.DP_NUM_DOFS
DP_NUM_ACTIONS = dc.DP_NUM_ACTIONS
B1Z1_DEFAULT_DOF_POS = dc.B1Z1_DEFAULT_DOF_POS
FLOAT_ARM_TO_DP_DOF = dc.FLOAT_ARM_TO_DP_DOF
ThickAxesGeometry = dc.ThickAxesGeometry
DoorRuntime = dc.DoorRuntime

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
sorted_asset_entries = dc.sorted_asset_entries


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
            {"name": "--return_home_target_chase_alpha", "type": float, "default": 0.08},
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
            {"name": "--dp_control_all_envs", "action": "store_true"},
            {"name": "--no_dp_control_all_envs", "action": "store_true"},
            {"name": "--dp_inference_steps", "type": int, "default": 10},
            {"name": "--dp_noise_scheduler_type", "type": str, "default": "DDIM"},
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
    args.dp_control_all_envs = not bool(args.no_dp_control_all_envs)
    args.dp_print = "--no_dp_print" not in argv
    args.dp_action_horizon = None if int(args.dp_action_horizon) < 0 else int(args.dp_action_horizon)
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if not args.dp_record_all_envs and (args.dp_record_env_id < 0 or args.dp_record_env_id >= args.num_envs):
        raise ValueError("--dp_record_env_id must be in [0, num_envs - 1].")
    if args.dp_policy_checkpoint and (args.dp_control_env_id < 0 or args.dp_control_env_id >= args.num_envs):
        raise ValueError("--dp_control_env_id must be in [0, num_envs - 1].")
    if args.dp_policy_checkpoint and args.dp_warmstart and args.dp_control_all_envs and args.num_envs > 1:
        raise ValueError("--dp_warmstart currently supports a single controlled env; add --no_dp_control_all_envs.")
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


# Shared door/IK helpers live in door_common; keep local names to avoid changing push control code.
smoothstep = dc.smoothstep
lerp = dc.lerp
quat_nlerp = dc.quat_nlerp
chase_target_to_current_ee = dc.chase_target_to_current_ee
normalize = dc.normalize
quat_apply = dc.quat_apply
quat_from_angle_axis = dc.quat_from_angle_axis
quat_axis = dc.quat_axis
forward_ee_quat = dc.forward_ee_quat
load_door_specs = dc.load_door_specs
load_door_assets = dc.load_door_assets
load_door_asset = dc.load_door_asset
robot_y_for_door = dc.robot_y_for_door
create_env_actors = dc.create_env_actors
create_parallel_env_actors = dc.create_parallel_env_actors


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


clone_door_runtime = dc.clone_door_runtime
resolve_seed = dc.resolve_seed
seed_for_env = dc.seed_for_env
sample_with_half_range = dc.sample_with_half_range


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


set_robot_base_pose = dc.set_robot_base_pose
compute_base_push_target = dc.compute_base_push_target
get_body_pose = dc.get_body_pose
get_actor_body_index = dc.get_actor_body_index
gym_quat_to_np = dc.gym_quat_to_np
local_camera_pose_from_cfg = dc.local_camera_pose_from_cfg
draw_local_camera_axes = dc.draw_local_camera_axes
draw_low_level_camera_axes = dc.draw_low_level_camera_axes
make_camera_properties = dc.make_camera_properties
attach_camera_to_actor_body = dc.attach_camera_to_actor_body
create_low_level_cameras = dc.create_low_level_cameras
camera_image_to_array = dc.camera_image_to_array
camera_color_to_rgb = dc.camera_color_to_rgb
show_camera_handle_images = dc.show_camera_handle_images
mask_to_rgb = dc.mask_to_rgb
depth_to_rgb = dc.depth_to_rgb
capture_dp_camera_images = dc.capture_dp_camera_images
capture_dp_camera_images_from_rendered = dc.capture_dp_camera_images_from_rendered
dp_image_inputs_from_cpu_cameras = dc.dp_image_inputs_from_cpu_cameras
get_actor_dof_state = dc.get_actor_dof_state
wrap_to_pi = dc.wrap_to_pi
current_ee_pose = dc.current_ee_pose
current_ee_pose_from_refreshed_tensors = dc.current_ee_pose_from_refreshed_tensors
update_arm_ik_targets_for_env = dc.update_arm_ik_targets_for_env
map_float_dofs_to_dp = dc.map_float_dofs_to_dp
base_command_from_targets = dc.base_command_from_targets
target_quat_for_dp = dc.target_quat_for_dp
make_last_low_action_from_dp = dc.make_last_low_action_from_dp
make_float_dp_state = dc.make_float_dp_state
base_position = dc.base_position
world_pos_to_base = dc.world_pos_to_base
base_pos_to_world = dc.base_pos_to_world
world_quat_to_base = dc.world_quat_to_base
base_quat_to_world = dc.base_quat_to_world
make_float_dp_action = dc.make_float_dp_action
make_float_replay_snapshot = dc.make_float_replay_snapshot


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


def prefill_dp_controller_from_expert_obs(controller, data, step, vision_mode, env_id=None):
    if raw_image_keys_for_vision_mode is None:
        raise RuntimeError("Expert observation warm-start requires door_dp_common.raw_image_keys_for_vision_mode.")
    image_keys = raw_image_keys_for_vision_mode(vision_mode)
    if env_id is None:
        controller.obs_buffer.clear()
        controller.action_queue.clear()
    else:
        controller.reset_envs([int(env_id)])
    start = max(0, int(step) - int(controller.obs_horizon) + 1)
    for idx in range(start, int(step) + 1):
        args = (
            np.asarray(data["state"][idx], dtype=np.float32),
            np.asarray(data[image_keys[0]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[1]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[2]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[3]][idx], dtype=np.uint8),
        )
        if env_id is None:
            controller.append_observation(*args)
        else:
            controller.append_observation_for_env(int(env_id), *args)


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
        prefill_dp_controller_from_expert_obs(controller, data, step, vision_mode, env_id=st.index)
    print(
        f"DP warm-start loaded raw={raw_path} step={step} "
        f"expert_obs_prefill={bool(args.dp_warmstart_expert_obs)} "
        f"base_xy={np.round(st.traj['base_xy'], 4).tolist()} yaw={float(st.traj['yaw']):.4f} "
        f"door_open_stage={bool(st.door.open_stage)}",
        flush=True,
    )
    return data


door_hinge_open_ratio = dc.door_hinge_open_ratio
compute_door_efforts = dc.compute_door_efforts
set_ik_target = dc.set_ik_target
update_arm_ik_targets = dc.update_arm_ik_targets
refresh_current_ee_pose = dc.refresh_current_ee_pose


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
    if "home_ee_base_pos" not in traj and ik_state.current_pos_np is not None:
        home_base_xy = traj.get("base_xy", base_start)
        home_yaw = float(traj.get("yaw", yaw_start))
        traj["home_ee_base_pos"] = world_pos_to_base(
            ik_state.current_pos_np,
            home_base_xy,
            args.robot_z,
            home_yaw,
        )
        if ik_state.current_quat_np is not None:
            traj["home_ee_base_quat"] = world_quat_to_base(ik_state.current_quat_np, home_yaw)

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
            if "return_home_start_base_xy" not in traj:
                traj["return_home_start_base_xy"] = traj.get("base_xy", base_push).copy()
                traj["return_home_start_yaw"] = float(traj.get("yaw", yaw_push))
            base_xy = lerp(traj["return_home_start_base_xy"], base_push, t)
            yaw = float(lerp(
                np.array([traj["return_home_start_yaw"]], dtype=np.float32),
                np.array([yaw_push], dtype=np.float32),
                t,
            )[0])
            target_pos, target_quat = chase_target_to_current_ee(
                traj,
                ik_state,
                args,
                traj["push"].copy(),
                traj.get("goal_quat"),
            )
            gripper = args.gripper_open
            traj["return_home_alpha"] = t
            phase = "return_home"
        else:
            home_base_pos = traj.get("home_ee_base_pos")
            fallback_pos = (
                base_pos_to_world(home_base_pos, base_push, args.robot_z, yaw_push)
                if home_base_pos is not None
                else traj["push"].copy()
            )
            fallback_quat = None
            if not args.ik_position_only and "home_ee_base_quat" in traj:
                fallback_quat = base_quat_to_world(traj["home_ee_base_quat"], yaw_push)
            target_pos, target_quat = chase_target_to_current_ee(
                traj,
                ik_state,
                args,
                fallback_pos,
                fallback_quat,
            )
            base_xy = base_push.copy()
            yaw = yaw_push
            gripper = args.gripper_open
            traj["return_home_alpha"] = 1.0
            phase = "hold_home"

    traj["base_xy"] = base_xy.copy()
    traj["yaw"] = float(yaw)
    traj["last_target_pos"] = np.asarray(target_pos, dtype=np.float32).copy()
    traj["last_target_quat"] = (
        None
        if target_quat is None
        else base_ik.normalize_quat(np.asarray(target_quat, dtype=np.float32)).astype(np.float32)
    )
    return phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal


setup_viewer = dc.setup_viewer


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

    yaw_start, heading, base_start, base_stop = dc.compute_base_walk_targets(args, door)
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

        dc.enforce_locked_door_hinge(gym, env, door_actor, door, args)
        door_pos, door_vel = get_actor_dof_state(gym, env, door_actor)
        door_efforts = compute_door_efforts(door, door_pos, door_vel, args)
        if len(door_efforts) > 0:
            gym.apply_actor_dof_efforts(env, door_actor, door_efforts)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        need_camera_render = bool(camera_handles and (args.show_camera_images or args.record_dp_dataset))
        if viewer is not None and need_camera_render and (args.draw_ik_target or args.draw_camera_axes):
            # Clear viewer-only debug lines before camera rendering so depth/RGB tensors stay clean.
            gym.clear_lines(viewer)
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
                    saved_target_pos_np = ik_state.target_pos_np
                    saved_target_quat_np = ik_state.target_quat_np
                    ik_state.target_pos_np = np.asarray(target_pos, dtype=np.float32).copy()
                    ik_state.target_quat_np = (
                        None
                        if target_quat is None
                        else base_ik.normalize_quat(target_quat).astype(np.float32)
                    )
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

    yaw_start, heading, base_start, base_stop = dc.compute_base_walk_targets(args, door)
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
        if (env_args.show_camera_images or env_args.record_dp_dataset or env_args.dp_policy_checkpoint) and (
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
    dp_control_env_ids = []
    dp_control_env_id_set = set()
    if args.dp_policy_checkpoint:
        if DoorDPPolicyController is None:
            raise RuntimeError("DP policy execution requires high-level/dp/door_dp_common.py and diffusers.")
        dp_control_env_ids = list(range(args.num_envs)) if args.dp_control_all_envs else [int(args.dp_control_env_id)]
        dp_control_env_id_set = set(dp_control_env_ids)
        dp_controller = DoorDPPolicyController(
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
        controlled_states = [env_states[env_id] for env_id in dp_control_env_ids]
        for controlled_state in controlled_states:
            controlled_state.dp_action_frame = getattr(dp_controller, "action_frame", "world")
            if not controlled_state.camera_handles:
                raise RuntimeError("ikpush DP policy execution requires camera sensors; do not disable wrist/front cameras.")
        dp_control_state = controlled_states[0]
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
        if args.dp_log_path:
            if DoorDPJsonlLogger is None:
                raise RuntimeError("DP policy logging requires high-level/dp/door_dp_common.py")
            dp_logger = DoorDPJsonlLogger(args.dp_log_path)
            print(f"Door DP log: {args.dp_log_path}", flush=True)
        apply_dp_warmstart_if_requested(gym, sim, args, dp_controller, dp_control_state, dof_names)

    while step < max_steps:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        gym.refresh_rigid_body_state_tensor(sim)
        for st in env_states:
            current_ee_pose_from_refreshed_tensors(st.ik_state)

        if dp_controller is not None:
            if viewer is not None and (args.draw_ik_target or args.draw_camera_axes):
                # Clear viewer-only debug lines before camera rendering so policy observations stay clean.
                gym.clear_lines(viewer)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_jacobian_tensors(sim)

        dp_policy_inputs_by_env = {}
        dp_actions_by_env = {}
        if dp_controller is not None:
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

        for st in env_states:
            if dp_controller is not None and st.index in dp_actions_by_env:
                phase = "dp_policy"
                dp_policy_input = dp_policy_inputs_by_env[st.index]
                base_xy_current = dp_policy_input["base_xy_current"]
                yaw_current = dp_policy_input["yaw_current"]
                handle_goal = dp_policy_input["handle_goal"]
                ee_pos = dp_policy_input["ee_pos"]
                ee_quat = dp_policy_input["ee_quat"]
                dp_state = dp_policy_input["dp_state"]
                dp_action = dp_actions_by_env[st.index]
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

            dc.enforce_locked_door_hinge(gym, st.env, st.door_actor, st.door, st.args)
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
        if viewer is not None and need_camera_render and (args.draw_ik_target or args.draw_camera_axes):
            # Clear viewer-only debug lines before camera rendering so depth/RGB tensors stay clean.
            gym.clear_lines(viewer)
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
                    if st.last_phase in ("return_home", "hold_home") and st.last_target_pos is not None:
                        saved_target_pos_np = st.ik_state.target_pos_np
                        saved_target_quat_np = st.ik_state.target_quat_np
                        st.ik_state.target_pos_np = np.asarray(st.last_target_pos, dtype=np.float32).copy()
                        st.ik_state.target_quat_np = (
                            None
                            if st.last_target_quat is None
                            else base_ik.normalize_quat(st.last_target_quat).astype(np.float32)
                        )
                        try:
                            base_ik.draw_ik_target(gym, viewer, st.env, st.ik_state)
                        finally:
                            st.ik_state.target_pos_np = saved_target_pos_np
                            st.ik_state.target_quat_np = saved_target_quat_np
                    else:
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
