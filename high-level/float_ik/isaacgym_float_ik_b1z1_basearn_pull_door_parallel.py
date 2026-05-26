#!/usr/bin/env python3
"""Parallel float-base B1Z1 base+arm IK pull-door controller."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import door_common as dc


base_ik = dc.base_ik
gymapi = dc.gymapi
gymutil = dc.gymutil
cv2 = dc.cv2

PULL_PHASE_NAMES = [
    "walk",
    "initial_hold",
    "grasp",
    "grasp_hold",
    "close_gripper",
    "rotate_handle",
    "retreat_base",
    "pull_door",
    "open_hold",
    "release_handle",
    "move_to_inner_contact",
    "brace_open",
    "pass_through",
    "return_home",
    "hold_home",
]

IKPULL_DEFAULT_ENV_RANGES = {
    "robot_y": (-0.07, 0.07),
    "robot_yaw": (3.10, 3.18),
    "pregrasp_offset": (0.12, 0.20),
    "grasp_x_offset": (-0.055, -0.010),
    "grasp_z_offset": (-0.055, -0.005),
    "door_pull_distance": (1.05, 1.25),
    "handle_rotate_angle": (0.95, 1.15),
    "door_joint_friction": (0.0, 0.05),
    "door_joint_damping": (0.0, 0.05),
    "handle_joint_friction": (0.045, 0.055),
    "handle_joint_damping": (0.045, 0.055),
    "handle_spring_stiffness": (0.45, 0.55),
    "handle_spring_damping": (0.09, 0.11),
}

IKPULL_ARG_FLAGS = {
    "robot_y": "--robot_y",
    "robot_yaw": "--robot_yaw",
    "pregrasp_offset": "--pregrasp_offset",
    "grasp_x_offset": "--grasp_x_offset",
    "grasp_z_offset": "--grasp_z_offset",
    "door_pull_distance": "--door_pull_distance",
    "handle_rotate_angle": "--handle_rotate_angle",
    "door_joint_friction": "--door_joint_friction",
    "door_joint_damping": "--door_joint_damping",
    "handle_joint_friction": "--handle_joint_friction",
    "handle_joint_damping": "--handle_joint_damping",
    "handle_spring_stiffness": "--handle_spring_stiffness",
    "handle_spring_damping": "--handle_spring_damping",
}


def parse_args():
    args = gymutil.parse_arguments(
        description="B1Z1 base+arm float IK door-pull demo.",
        headless=True,
        no_graphics=True,
        custom_parameters=[
            {"name": "--asset_root", "type": str, "default": str(base_ik.DEFAULT_ASSET_ROOT)},
            {"name": "--asset_file", "type": str, "default": base_ik.DEFAULT_ASSET_FILE},
            {"name": "--rl_device", "type": str, "default": "cuda:0"},
            {"name": "--num_envs", "type": int, "default": 1},
            {"name": "--steps", "type": int, "default": 4300},
            {"name": "--seed", "type": int, "default": -1},
            {"name": "--door_cfg", "type": str, "default": str(dc.DEFAULT_DOOR_CFG)},
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
            {"name": "--door_pass_clearance", "type": float, "default": 0.55},
            {"name": "--pull_base_distance", "type": float, "default": 0.0},
            {"name": "--base_pull_time_scale", "type": float, "default": 1.0},
            {"name": "--pull_base_yaw_delta", "type": float, "default": 0.0},
            {
                "name": "--handle_rotate_base_retreat_ratio",
                "type": float,
                "default": 0.0,
                "help": "Fraction of base retreat completed during handle rotation before the pull phase starts.",
            },
            {
                "name": "--handle_rotate_base_retreat_start_ratio",
                "type": float,
                "default": 1.0,
                "help": "Normalized handle-rotation progress at which base retreat begins.",
            },
            {"name": "--pull_base_lateral_offset", "type": float, "default": 0.0},
            {"name": "--pass_base_lateral_offset", "type": float, "default": 0.0},
            {"name": "--allow_post_release_base_lateral", "action": "store_true"},
            {"name": "--post_release_yaw_clearance_delta", "type": float, "default": 0.0},
            {"name": "--pull_lateral_start_open_deg", "type": float, "default": 8.0},
            {"name": "--pull_lateral_end_open_deg", "type": float, "default": 28.0},
            {"name": "--door_sweep_clearance", "type": float, "default": 0.06},
            {"name": "--safe_retreat_extra", "type": float, "default": 0.06},
            {"name": "--safe_retreat_search_max", "type": float, "default": 1.5},
            {"name": "--door_sweep_samples", "type": int, "default": 121},
            {"name": "--safe_sweep_open_angle_deg", "type": float, "default": 86.0},
            {"name": "--base_motion_door_clearance", "type": float, "default": 0.12},
            {"name": "--base_motion_limit_samples", "type": int, "default": 41},
            {"name": "--pre_inner_open_angle_deg", "type": float, "default": 80.0},
            {"name": "--walk_steps", "type": int, "default": 260},
            {"name": "--initial_hold_steps", "type": int, "default": 200},
            {"name": "--grasp_steps", "type": int, "default": 150},
            {"name": "--grasp_hold_steps", "type": int, "default": 100},
            {"name": "--gripper_close_steps", "type": int, "default": 120},
            {"name": "--handle_rotate_steps", "type": int, "default": 300},
            {"name": "--base_retreat_steps", "type": int, "default": 260},
            {"name": "--door_pull_steps", "type": int, "default": 1080},
            {"name": "--pull_settle_steps", "type": int, "default": 5},
            {"name": "--open_hold_steps", "type": int, "default": 5},
            {"name": "--release_handle_steps", "type": int, "default": 40},
            {"name": "--move_to_inner_contact_steps", "type": int, "default": 320},
            {"name": "--inner_max_extra_steps", "type": int, "default": 480},
            {"name": "--brace_open_steps", "type": int, "default": 520},
            {"name": "--brace_max_extra_steps", "type": int, "default": 720},
            {"name": "--pass_through_steps", "type": int, "default": 150},
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
            {"name": "--door_pull_distance", "type": float, "default": 1.10},
            {"name": "--ikpull_env_randomization", "action": "store_true"},
            {"name": "--no_ikpull_env_randomization", "action": "store_true"},
            {"name": "--ikpull_robot_x_rand", "type": float, "default": 0.03},
            {"name": "--ikpull_robot_y_rand", "type": float, "default": 0.04},
            {"name": "--ikpull_robot_yaw_rand", "type": float, "default": 0.03},
            {"name": "--ikpull_pregrasp_offset_rand", "type": float, "default": 0.025},
            {"name": "--ikpull_grasp_x_offset_rand", "type": float, "default": 0.012},
            {"name": "--ikpull_grasp_z_offset_rand", "type": float, "default": 0.012},
            {"name": "--ikpull_handle_rotate_angle_rand", "type": float, "default": 0.04},
            {"name": "--ikpull_door_pull_distance_rand", "type": float, "default": 0.06},
            {"name": "--ikpull_door_joint_friction_rand", "type": float, "default": 0.08},
            {"name": "--ikpull_door_joint_damping_rand", "type": float, "default": 0.04},
            {"name": "--ikpull_handle_joint_friction_rand", "type": float, "default": 0.005},
            {"name": "--ikpull_handle_joint_damping_rand", "type": float, "default": 0.005},
            {"name": "--ikpull_handle_spring_stiffness_rand", "type": float, "default": 0.05},
            {"name": "--ikpull_handle_spring_damping_rand", "type": float, "default": 0.01},
            {"name": "--lever_step_size", "type": float, "default": 0.06},
            {"name": "--pull_contact_bias", "type": float, "default": 0.025},
            {"name": "--pull_target_max_distance", "type": float, "default": 1.10},
            {"name": "--open_hold_step_size", "type": float, "default": 0.035},
            {"name": "--brace_open_step_size", "type": float, "default": 0.055},
            {"name": "--handle_follow_pull_ratio", "type": float, "default": 0.0},
            {"name": "--release_handle_distance", "type": float, "default": 0.20},
            {"name": "--post_release_back_margin", "type": float, "default": 0.28},
            {"name": "--post_release_min_back_step", "type": float, "default": 0.08},
            {"name": "--post_release_right_margin", "type": float, "default": 0.34},
            {"name": "--post_release_min_right_step", "type": float, "default": 0.30},
            {"name": "--post_release_extra_right_distance", "type": float, "default": 0.18},
            {"name": "--post_release_escape_lift", "type": float, "default": 0.04},
            {"name": "--post_release_contact_bias", "type": float, "default": 0.04},
            {"name": "--inner_contact_radial_ratio", "type": float, "default": 0.62},
            {"name": "--inner_contact_min_radial_ratio", "type": float, "default": 0.32},
            {"name": "--inner_contact_max_radial_ratio", "type": float, "default": 0.82},
            {"name": "--inner_contact_push_bias", "type": float, "default": 0.06},
            {"name": "--inner_precontact_offset", "type": float, "default": 0.20},
            {"name": "--inner_contact_z_offset", "type": float, "default": -0.02},
            {"name": "--inner_arc_clearance", "type": float, "default": 0.22},
            {"name": "--inner_arc_max_radius", "type": float, "default": 0.54},
            {"name": "--inner_arc_z_clearance", "type": float, "default": 0.12},
            {"name": "--inner_arc_approach_step", "type": float, "default": 0.10},
            {"name": "--inner_fan_angle_ratio", "type": float, "default": 0.84},
            {"name": "--inner_fan_radial_ratio", "type": float, "default": 0.70},
            {"name": "--inner_fan_opposite_tangent_bias", "type": float, "default": 0.10},
            {"name": "--inner_fan_z_offset", "type": float, "default": 0.02},
            {"name": "--inner_fan_lift", "type": float, "default": 0.12},
            {"name": "--inner_fan_ready_distance", "type": float, "default": 0.20},
            {"name": "--inner_entry_distance", "type": float, "default": 0.10},
            {"name": "--inner_ready_distance", "type": float, "default": 0.06},
            {"name": "--inner_base_time_scale", "type": float, "default": 1.8},
            {"name": "--inner_base_advance_ratio", "type": float, "default": 0.005},
            {"name": "--brace_base_time_scale", "type": float, "default": 1.2},
            {"name": "--brace_base_advance_ratio", "type": float, "default": 0.005},
            {"name": "--inner_base_reach_x_max", "type": float, "default": 0.86},
            {"name": "--inner_base_reach_extra", "type": float, "default": 0.03},
            {"name": "--inner_base_lateral_search_max", "type": float, "default": 0.0},
            {"name": "--inner_base_lateral_samples", "type": int, "default": 25},
            {"name": "--brace_contact_push_bias", "type": float, "default": 0.24},
            {"name": "--brace_contact_hold_bias", "type": float, "default": 0.06},
            {"name": "--brace_contact_approach_step", "type": float, "default": 0.11},
            {"name": "--brace_min_contact_steps", "type": int, "default": 260},
            {"name": "--brace_target_open_angle_deg", "type": float, "default": 86.0},
            {"name": "--brace_settle_steps", "type": int, "default": 30},
            {"name": "--simple_post_release_reach", "dest": "simple_post_release_reach", "action": "store_true", "default": True},
            {"name": "--no_simple_post_release_reach", "dest": "simple_post_release_reach", "action": "store_false"},
            {"name": "--simple_brace_complete_on_steps", "dest": "simple_brace_complete_on_steps", "action": "store_true", "default": True},
            {"name": "--no_simple_brace_complete_on_steps", "dest": "simple_brace_complete_on_steps", "action": "store_false"},
            {"name": "--simple_post_release_use_base_motion_limit", "action": "store_true", "default": False},
            {"name": "--simple_post_release_base_step", "type": float, "default": 0.006},
            {"name": "--simple_reach_progress_step", "type": float, "default": 0.005},
            {"name": "--simple_reach_approach_step", "type": float, "default": 0.04},
            {"name": "--simple_reach_base_x", "type": float, "default": -0.5},
            {"name": "--simple_reach_right_y", "type": float, "default": -0.3},
            {"name": "--simple_reach_left_y", "type": float, "default": 0.33},
            {"name": "--simple_reach_z_offset", "type": float, "default": 0.2},
            {"name": "--pass_inner_hold_ratio", "type": float, "default": 0.65},
            {"name": "--pass_min_open_angle_deg", "type": float, "default": 75.0},
            {"name": "--pass_closed_progress_scale", "type": float, "default": 0.25},
            {"name": "--pass_contact_push_bias", "type": float, "default": 0.28},
            {"name": "--pass_reopen_push_extra", "type": float, "default": 0.18},
            {"name": "--pass_contact_approach_step", "type": float, "default": 0.12},
            {"name": "--pass_lateral_stage_ratio", "type": float, "default": 0.0},
            {"name": "--pass_contact_base_x", "type": float, "default": 0.58},
            {"name": "--pass_contact_base_y", "type": float, "default": 0.02},
            {"name": "--pass_home_retract_ee_inside_distance", "type": float, "default": 0.03},
            {"name": "--brace_open_direction_sign", "type": float, "default": 1.0},
            {"name": "--door_freeze_blend_start_ratio", "type": float, "default": 0.82},
            {"name": "--door_freeze_target_ratio", "type": float, "default": 0.94},
            {"name": "--enable_base_door_collision_check", "dest": "enable_base_door_collision_check", "action": "store_true", "default": True},
            {"name": "--no_base_door_collision_check", "dest": "enable_base_door_collision_check", "action": "store_false"},
            {"name": "--base_door_collision_distance", "type": float, "default": 0.04},
            {"name": "--base_collision_front_extent", "type": float, "default": 0.55},
            {"name": "--base_collision_rear_extent", "type": float, "default": 0.65},
            {"name": "--base_collision_half_width", "type": float, "default": 0.24},
            {"name": "--rigid_contact_geom_gate", "type": float, "default": 0.16},
            {"name": "--collision_log_interval", "type": int, "default": 30},
            {"name": "--enable_command_jump_check", "dest": "enable_command_jump_check", "action": "store_true", "default": True},
            {"name": "--no_command_jump_check", "dest": "enable_command_jump_check", "action": "store_false"},
            {"name": "--base_command_jump_distance", "type": float, "default": 0.08},
            {"name": "--target_command_jump_distance", "type": float, "default": 0.18},
            {"name": "--yaw_command_jump_distance", "type": float, "default": 0.08},
            {"name": "--command_jump_log_interval", "type": int, "default": 20},
            {"name": "--post_release_debug_interval", "type": int, "default": 0},
            {
                "name": "--pull_follow_orientation",
                "action": "store_true",
                "help": "During pull, keep end-effector orientation fixed relative to the handle.",
            },
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
            {"name": "--gripper_open_stage_ratio", "type": float, "default": 0.25},
            {"name": "--handle_spring_stiffness", "type": float, "default": 0.5},
            {"name": "--handle_spring_damping", "type": float, "default": 0.1},
            {"name": "--handle_unlock_ratio", "type": float, "default": 40.0 / 45.0},
            {"name": "--door_open_resistance", "type": float, "default": 0.0},
            {"name": "--door_open_damping", "type": float, "default": 0.0},
            {"name": "--door_lock_force", "type": float, "default": 0.0},
            {"name": "--door_joint_friction", "type": float, "default": 0.0},
            {"name": "--door_joint_damping", "type": float, "default": 0.0},
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
            {"name": "--pass_open_angle_deg", "type": float, "default": 80.0},
            {"name": "--no_preview_trajectory_at_spawn", "action": "store_true"},
        ],
    )

    argv = set(sys.argv[1:])
    dc.finalize_float_ik_args(args, argv)
    if "--no_base_door_collision_check" in argv:
        args.enable_base_door_collision_check = False
    else:
        args.enable_base_door_collision_check = True
    if "--no_unidoor_style_pull" in argv:
        args.unidoor_style_pull = False
    elif "--unidoor_style_pull" in argv:
        args.unidoor_style_pull = True
    else:
        args.unidoor_style_pull = True
    args.simple_post_release_reach = "--no_simple_post_release_reach" not in argv
    args.simple_brace_complete_on_steps = "--no_simple_brace_complete_on_steps" not in argv

    args.door_motion_sign = 1.0
    return args


def sample_pull_env_value(rng, args, attr, half_attr, lower=None, upper=None):
    explicit_flags = getattr(args, "_explicit_cli_flags", set())
    default_range = IKPULL_DEFAULT_ENV_RANGES.get(attr)
    flag = IKPULL_ARG_FLAGS.get(attr)
    if default_range is not None and flag not in explicit_flags:
        value = float(rng.uniform(float(default_range[0]), float(default_range[1])))
    else:
        value = dc.sample_with_half_range(rng, getattr(args, attr), getattr(args, half_attr), lower=lower, upper=upper)
    if lower is not None:
        value = max(float(lower), value)
    if upper is not None:
        value = min(float(upper), value)
    return value


def make_env_args(args, env_index):
    env_args = SimpleNamespace(**vars(args))
    env_seed = dc.seed_for_env(args, env_index)
    rng = np.random.default_rng(env_seed)
    env_args.env_seed = env_seed
    env_args.parallel_env_id = int(env_index)
    enabled = bool(getattr(args, "ikpull_env_randomization", False)) and not bool(
        getattr(args, "no_ikpull_env_randomization", False)
    )

    sampled = {
        "seed": int(getattr(args, "seed", 0)),
        "env_seed": int(env_seed),
        "enabled": bool(enabled),
    }

    def set_sampled(attr, half_attr, lower=None, upper=None):
        base_value = getattr(args, attr)
        value = (
            sample_pull_env_value(rng, args, attr, half_attr, lower=lower, upper=upper)
            if enabled
            else float(base_value)
        )
        setattr(env_args, attr, value)
        sampled[attr] = value

    set_sampled("robot_x", "ikpull_robot_x_rand")
    set_sampled("robot_y", "ikpull_robot_y_rand")
    set_sampled("robot_yaw", "ikpull_robot_yaw_rand")
    set_sampled("pregrasp_offset", "ikpull_pregrasp_offset_rand", lower=0.05)
    set_sampled("grasp_x_offset", "ikpull_grasp_x_offset_rand")
    set_sampled("grasp_z_offset", "ikpull_grasp_z_offset_rand")
    set_sampled("handle_rotate_angle", "ikpull_handle_rotate_angle_rand", lower=0.75, upper=1.30)
    set_sampled("door_pull_distance", "ikpull_door_pull_distance_rand", lower=0.45, upper=1.20)
    set_sampled("door_joint_friction", "ikpull_door_joint_friction_rand", lower=0.0)
    set_sampled("door_joint_damping", "ikpull_door_joint_damping_rand", lower=0.0)
    set_sampled("handle_joint_friction", "ikpull_handle_joint_friction_rand", lower=0.0)
    set_sampled("handle_joint_damping", "ikpull_handle_joint_damping_rand", lower=0.0)
    set_sampled("handle_spring_stiffness", "ikpull_handle_spring_stiffness_rand", lower=0.0)
    set_sampled("handle_spring_damping", "ikpull_handle_spring_damping_rand", lower=0.0)

    env_args.ikpull_randomization_json = json.dumps(sampled, sort_keys=True)
    return env_args


def signed_angle_between_xy(from_vec, to_vec):
    from_vec = dc.normalize(np.asarray(from_vec, dtype=np.float32)[:2])
    to_vec = dc.normalize(np.asarray(to_vec, dtype=np.float32)[:2])
    if float(np.linalg.norm(from_vec)) < 1.0e-6 or float(np.linalg.norm(to_vec)) < 1.0e-6:
        return 0.0
    cross = float(from_vec[0] * to_vec[1] - from_vec[1] * to_vec[0])
    dot = float(np.dot(from_vec, to_vec))
    return math.atan2(cross, dot)


def door_panel_fan_geometry(gym, env, door_actor, door, handle_goal, traj):
    hinge_pos, _hinge_quat = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
    current_vec = np.asarray(handle_goal[:2], dtype=np.float32) - np.asarray(hinge_pos[:2], dtype=np.float32)
    closed_vec = np.asarray(traj.get("closed_handle_vec_xy", current_vec) if traj is not None else current_vec, dtype=np.float32)
    if float(np.linalg.norm(closed_vec)) < 1.0e-5:
        closed_vec = current_vec.copy()
    if float(np.linalg.norm(current_vec)) < 1.0e-5:
        current_vec = closed_vec.copy()
    door_radius = max(float(np.linalg.norm(closed_vec)), float(np.linalg.norm(current_vec)), 1.0e-5)
    signed_open_angle = signed_angle_between_xy(closed_vec, current_vec)
    return hinge_pos, closed_vec.astype(np.float32), current_vec.astype(np.float32), float(door_radius), float(signed_open_angle)


def door_actual_open_tangent_dir(gym, env, door_actor, door, handle_goal, fallback_dir, args, traj=None):
    if traj is None or "closed_handle_vec_xy" not in traj:
        return dc.handle_open_tangent_dir(gym, env, door_actor, door, handle_goal, fallback_dir, args)
    _hinge_pos, _closed_vec, current_vec, _door_radius, signed_open_angle = door_panel_fan_geometry(
        gym,
        env,
        door_actor,
        door,
        handle_goal,
        traj,
    )
    current_dir = dc.normalize(current_vec)
    if float(np.linalg.norm(current_dir)) < 1.0e-5:
        return dc.handle_open_tangent_dir(gym, env, door_actor, door, handle_goal, fallback_dir, args)
    if abs(float(signed_open_angle)) < math.radians(1.0):
        fallback = dc.handle_open_tangent_dir(gym, env, door_actor, door, handle_goal, fallback_dir, args)
        sign = 1.0 if float(np.dot(fallback[:2], np.array([-current_dir[1], current_dir[0]], dtype=np.float32))) >= 0.0 else -1.0
    else:
        sign = 1.0 if signed_open_angle >= 0.0 else -1.0
    tangent_xy = np.asarray([-current_dir[1], current_dir[0]], dtype=np.float32) * float(sign)
    tangent = np.asarray([float(tangent_xy[0]), float(tangent_xy[1]), 0.0], dtype=np.float32)
    return dc.normalize(tangent)


def door_inner_contact_pose(gym, env, door_actor, door, handle_goal, fallback_dir, args, traj=None):
    hinge_pos, _hinge_quat = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
    radial = np.asarray(handle_goal, dtype=np.float32) - np.asarray(hinge_pos, dtype=np.float32)
    radial[2] = 0.0
    radial_len = float(np.linalg.norm(radial))
    if radial_len < 1.0e-5:
        radial_dir = dc.normalize(np.asarray(handle_goal, dtype=np.float32) - np.asarray([args.door_x, args.door_y, args.robot_z], dtype=np.float32))
        radial_len = 0.55
    else:
        radial_dir = radial / radial_len
    tangent = door_actual_open_tangent_dir(gym, env, door_actor, door, handle_goal, fallback_dir, args, traj)
    push_dir = dc.normalize(tangent * float(args.brace_open_direction_sign))
    if np.linalg.norm(push_dir) < 1.0e-5:
        push_dir = tangent
    contact = np.asarray(hinge_pos, dtype=np.float32) + radial_dir * (
        radial_len * float(np.clip(args.inner_contact_radial_ratio, 0.25, 1.0))
    )
    contact += push_dir * float(args.inner_contact_push_bias)
    contact[2] = float(handle_goal[2]) + float(args.inner_contact_z_offset)
    return contact.astype(np.float32), push_dir.astype(np.float32)


def door_inner_precontact_pose(gym, env, door_actor, door, handle_goal, fallback_dir, args, traj=None):
    contact, push_dir = door_inner_contact_pose(gym, env, door_actor, door, handle_goal, fallback_dir, args, traj)
    precontact = contact - push_dir * float(args.inner_precontact_offset)
    return precontact.astype(np.float32), contact.astype(np.float32), push_dir.astype(np.float32)


def door_inner_sliding_contact_pose(
    gym,
    env,
    door_actor,
    door,
    handle_goal,
    fallback_dir,
    base_xy,
    yaw,
    args,
    traj=None,
    reach_push_bias=0.0,
):
    hinge_pos, _hinge_quat = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
    radial = np.asarray(handle_goal, dtype=np.float32) - np.asarray(hinge_pos, dtype=np.float32)
    radial[2] = 0.0
    radial_len = float(np.linalg.norm(radial))
    if radial_len < 1.0e-5:
        return door_inner_contact_pose(gym, env, door_actor, door, handle_goal, fallback_dir, args, traj)

    radial_dir = radial / radial_len
    raw_push_dir = door_actual_open_tangent_dir(gym, env, door_actor, door, handle_goal, fallback_dir, args, traj)
    push_dir = dc.normalize(raw_push_dir * float(args.brace_open_direction_sign))
    if np.linalg.norm(push_dir) < 1.0e-5:
        push_dir = raw_push_dir
    preferred_base = np.asarray(
        [
            float(args.pass_contact_base_x),
            float(args.pass_contact_base_y),
            float(handle_goal[2]) + float(args.inner_contact_z_offset) - float(args.robot_z),
        ],
        dtype=np.float32,
    )
    ratio_min = float(args.inner_contact_min_radial_ratio)
    ratio_max = float(args.inner_contact_max_radial_ratio)
    preferred_world = dc.base_pos_to_world(preferred_base, base_xy, args.robot_z, yaw)
    preferred_delta = preferred_world - hinge_pos
    projected_ratio = float(np.dot(preferred_delta[:2], radial_dir[:2]) / max(radial_len, 1.0e-5))
    preferred_ratio = float(np.clip(projected_ratio, ratio_min, ratio_max))
    best_contact = None
    best_score = float("inf")
    for ratio in np.linspace(ratio_min, ratio_max, 31):
        candidate = hinge_pos + radial_dir * radial_len * float(ratio)
        candidate += push_dir * float(args.inner_contact_push_bias)
        candidate[2] = float(handle_goal[2]) + float(args.inner_contact_z_offset)
        reach_probe = candidate + push_dir * float(reach_push_bias)
        local = dc.world_pos_to_base(reach_probe, base_xy, args.robot_z, yaw)
        x_violation = max(0.0, 0.18 - float(local[0])) + max(0.0, float(local[0]) - 0.84)
        y_violation = max(0.0, abs(float(local[1])) - 0.44)
        preference = (
            0.20 * abs(float(ratio) - preferred_ratio)
            + 0.05 * abs(float(local[1]) - float(preferred_base[1]))
            + 0.03 * abs(float(local[0]) - float(preferred_base[0]))
        )
        score = 100.0 * (x_violation + y_violation) + preference
        if score < best_score:
            best_score = float(score)
            best_contact = candidate.copy()
    contact = np.asarray(best_contact if best_contact is not None else hinge_pos + radial_dir * radial_len * preferred_ratio, dtype=np.float32)
    contact[2] = float(handle_goal[2]) + float(args.inner_contact_z_offset)
    return contact.astype(np.float32), push_dir.astype(np.float32)


def inside_progress_along_heading(point_xy, args, heading):
    door_xy = np.asarray([args.door_x, args.door_y], dtype=np.float32)
    return float(np.dot(np.asarray(point_xy, dtype=np.float32) - door_xy, np.asarray(heading, dtype=np.float32)))


def ensure_inside_xy(point_xy, args, heading, min_inside):
    point_xy = np.asarray(point_xy, dtype=np.float32).copy()
    heading = dc.normalize(np.asarray(heading, dtype=np.float32))
    progress = inside_progress_along_heading(point_xy, args, heading)
    if progress < float(min_inside):
        point_xy += heading * (float(min_inside) - progress)
    return point_xy.astype(np.float32)


def signed_fan_angle_from_closed(point_xy, hinge_xy, closed_vec_xy, args):
    rel = np.asarray(point_xy, dtype=np.float32) - np.asarray(hinge_xy, dtype=np.float32)
    closed = dc.normalize(np.asarray(closed_vec_xy, dtype=np.float32))
    if float(np.linalg.norm(rel)) < 1.0e-6 or float(np.linalg.norm(closed)) < 1.0e-6:
        return 0.0
    cross = float(closed[0] * rel[1] - closed[1] * rel[0])
    dot = float(np.dot(closed, rel))
    return float(args.door_motion_sign) * math.atan2(cross, dot)


def project_xy_to_open_fan(point_xy, hinge_xy, closed_vec_xy, open_angle_rad, args):
    hinge_xy = np.asarray(hinge_xy, dtype=np.float32)
    closed_vec_xy = np.asarray(closed_vec_xy, dtype=np.float32)
    door_radius = max(1.0e-5, float(np.linalg.norm(closed_vec_xy)))
    radius = float(np.linalg.norm(np.asarray(point_xy, dtype=np.float32) - hinge_xy))
    angle = signed_fan_angle_from_closed(point_xy, hinge_xy, closed_vec_xy, args)
    max_angle = max(math.radians(3.0), float(open_angle_rad))
    angle = float(np.clip(angle, math.radians(3.0), max_angle - math.radians(3.0) if max_angle > math.radians(8.0) else max_angle))
    radius = float(np.clip(radius, door_radius * 0.30, door_radius * 0.90))
    fan_vec = dc.rotate_xy(dc.normalize(closed_vec_xy) * radius, float(args.door_motion_sign) * angle)
    return (hinge_xy + fan_vec).astype(np.float32)


def door_fan_inner_waypoint(gym, env, door_actor, door, handle_goal, fallback_dir, traj, args, start_pos=None):
    hinge_pos, closed_vec, current_vec, door_radius, signed_open_angle = door_panel_fan_geometry(
        gym,
        env,
        door_actor,
        door,
        handle_goal,
        traj,
    )
    door_pos, _ = dc.get_actor_dof_state(gym, env, door_actor)
    open_angle = math.radians(max(5.0, abs(dc.door_open_degrees(door_pos, args))))
    if abs(float(signed_open_angle)) < math.radians(5.0):
        signed_open_angle = math.copysign(open_angle, float(args.door_motion_sign))
    open_angle_abs = max(math.radians(5.0), abs(float(signed_open_angle)))
    edge_clearance_angle = min(
        math.radians(18.0),
        max(math.radians(3.0), float(args.inner_fan_opposite_tangent_bias) / max(float(door_radius), 1.0e-5)),
    )
    max_angle_ratio = max(0.40, min(0.94, 1.0 - edge_clearance_angle / max(open_angle_abs, 1.0e-5)))
    angle_ratio = float(np.clip(args.inner_fan_angle_ratio, 0.35, max_angle_ratio))
    radial_ratio = float(np.clip(args.inner_fan_radial_ratio, 0.35, 0.88))
    push_dir = door_actual_open_tangent_dir(gym, env, door_actor, door, handle_goal, fallback_dir, args, traj)
    base_xy = np.asarray(
        traj.get("inner_move_start_base_xy", traj.get("base_xy", [args.robot_x, args.robot_y])),
        dtype=np.float32,
    )
    yaw = float(traj.get("yaw", args.robot_yaw + args.pull_base_yaw_delta))
    reach_x_min = 0.18
    reach_x_max = max(reach_x_min + 0.05, float(args.inner_base_reach_x_max))
    reach_y_max = 0.46
    best_goal_xy = None
    best_local = None
    best_score = float("inf")
    best_ratio_pair = (angle_ratio, radial_ratio)
    closed_dir = dc.normalize(closed_vec)
    if float(np.linalg.norm(closed_dir)) < 1.0e-5:
        closed_dir = dc.normalize(current_vec)
    start_xy = None
    if start_pos is not None:
        start_xy = np.asarray(start_pos, dtype=np.float32)[:2]
    angle_candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.40, max_angle_ratio, 22),
                np.asarray([angle_ratio, min(max_angle_ratio, angle_ratio + 0.08)], dtype=np.float32),
            ]
        )
    )
    radial_candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.55, 0.88, 18),
                np.asarray([radial_ratio], dtype=np.float32),
            ]
        )
    )
    for candidate_angle_ratio in angle_candidates:
        candidate_angle_ratio = float(np.clip(candidate_angle_ratio, 0.35, max_angle_ratio))
        fan_angle = float(signed_open_angle) * candidate_angle_ratio
        for candidate_radial_ratio in radial_candidates:
            fan_vec = dc.rotate_xy(closed_dir * door_radius * float(candidate_radial_ratio), fan_angle)
            candidate_xy = np.asarray(hinge_pos[:2], dtype=np.float32) + fan_vec
            local = dc.world_xy_to_base_local(candidate_xy, base_xy, yaw)
            x_violation = max(0.0, reach_x_min - float(local[0])) + max(0.0, float(local[0]) - reach_x_max)
            y_violation = max(0.0, abs(float(local[1])) - reach_y_max)
            preference = (
                0.40 * abs(float(candidate_radial_ratio) - radial_ratio)
                + 0.20 * abs(float(candidate_angle_ratio) - angle_ratio)
                + 0.04 * abs(float(local[1]))
                + 0.04 * max(0.0, reach_x_max - float(local[0]))
            )
            if start_xy is not None:
                preference += 0.35 * float(np.linalg.norm(candidate_xy - start_xy))
            score = 100.0 * (x_violation + y_violation) + preference
            if score < best_score:
                best_goal_xy = candidate_xy.copy()
                best_local = local.copy()
                best_score = float(score)
                best_ratio_pair = (float(candidate_angle_ratio), float(candidate_radial_ratio))
    goal_xy = np.asarray(best_goal_xy, dtype=np.float32)
    goal = np.asarray(
        [
            float(goal_xy[0]),
            float(goal_xy[1]),
            float(handle_goal[2]) + float(args.inner_fan_z_offset),
        ],
        dtype=np.float32,
    )
    fan_angle = signed_fan_angle_from_closed(goal_xy, hinge_pos[:2], closed_vec, args)
    signed_goal_angle = signed_angle_between_xy(closed_vec, goal_xy - np.asarray(hinge_pos[:2], dtype=np.float32))
    inside_sector = (
        abs(float(signed_goal_angle)) <= open_angle_abs + math.radians(1.0)
        and float(signed_goal_angle) * float(signed_open_angle) >= -1.0e-6
    )
    return goal, push_dir.astype(np.float32), {
        "open_angle_deg": math.degrees(open_angle_abs),
        "signed_open_angle_deg": math.degrees(float(signed_open_angle)),
        "fan_angle_deg": math.degrees(fan_angle),
        "signed_goal_angle_deg": math.degrees(float(signed_goal_angle)),
        "inside_sector": bool(inside_sector),
        "edge_clearance_deg": math.degrees(edge_clearance_angle),
        "radius": float(np.linalg.norm(goal_xy - hinge_pos[:2])),
        "door_radius": float(door_radius),
        "base_local": np.round(best_local, 4).tolist() if best_local is not None else [],
        "angle_radial_ratio": [round(best_ratio_pair[0], 3), round(best_ratio_pair[1], 3)],
    }


def door_fan_inner_transition_pose(
    gym,
    env,
    door_actor,
    door,
    handle_goal,
    fallback_dir,
    start_pos,
    t,
    args,
    traj,
):
    fan_goal, push_dir, fan_info = door_fan_inner_waypoint(
        gym,
        env,
        door_actor,
        door,
        handle_goal,
        fallback_dir,
        traj,
        args,
        start_pos=start_pos,
    )
    start = np.asarray(start_pos, dtype=np.float32)
    t = float(np.clip(t, 0.0, 1.0))
    side_dir = -dc.normalize(push_dir)
    control = 0.5 * (start + fan_goal)
    control[:2] += side_dir[:2] * float(args.inner_fan_opposite_tangent_bias)
    control[2] = max(float(start[2]), float(fan_goal[2])) + float(args.inner_fan_lift)
    one_minus = 1.0 - t
    target = (one_minus * one_minus) * start + (2.0 * one_minus * t) * control + (t * t) * fan_goal
    face_dir = np.asarray(handle_goal, dtype=np.float32) - target
    face_dir[2] = 0.0
    if float(np.linalg.norm(face_dir)) < 1.0e-5:
        face_dir = push_dir.copy()
    fan_info["goal"] = np.round(fan_goal, 4).tolist()
    return target.astype(np.float32), fan_goal.astype(np.float32), push_dir.astype(np.float32), dc.normalize(face_dir).astype(np.float32), fan_info


def post_release_escape_waypoints(
    gym,
    env,
    door_actor,
    door,
    handle_goal,
    start_pos,
    heading,
    yaw,
    args,
    traj,
):
    hinge_pos, _hinge_quat = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
    start = np.asarray(start_pos, dtype=np.float32)
    heading = dc.normalize(np.asarray(heading, dtype=np.float32))
    if float(np.linalg.norm(heading)) < 1.0e-5:
        heading = np.asarray([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)
    back_dir = -heading
    right_dir = base_right_vector(yaw)
    approach_dir = np.asarray(traj.get("approach_dir", np.r_[back_dir, 0.0]), dtype=np.float32).copy()
    approach_dir[2] = 0.0
    if float(np.linalg.norm(approach_dir)) < 1.0e-5:
        approach_dir = np.asarray([back_dir[0], back_dir[1], 0.0], dtype=np.float32)
    approach_dir = dc.normalize(approach_dir)

    escape_z = max(
        float(start[2]),
        float(handle_goal[2]) + float(args.inner_fan_z_offset),
    ) + float(args.post_release_escape_lift)
    away = start.copy()
    away += approach_dir * float(args.release_handle_distance)
    away[2] = escape_z

    panel_points = [
        np.asarray(hinge_pos[:2], dtype=np.float32),
        np.asarray(handle_goal[:2], dtype=np.float32),
    ]
    back_edge = max(float(np.dot(point, back_dir)) for point in panel_points)
    back_delta = max(
        float(args.post_release_min_back_step),
        back_edge + float(args.post_release_back_margin) - float(np.dot(away[:2], back_dir)),
    )
    back_out = away.copy()
    back_out[:2] += back_dir * max(0.0, back_delta)
    back_out[2] = escape_z

    right_edge = max(float(np.dot(point, right_dir)) for point in panel_points)
    right_delta = max(
        float(args.post_release_min_right_step),
        right_edge + float(args.post_release_right_margin) - float(np.dot(back_out[:2], right_dir)),
    )
    right_safe = back_out.copy()
    right_safe[:2] += right_dir * max(0.0, right_delta)
    right_safe[2] = escape_z
    far_right = right_safe.copy()
    far_right[:2] += right_dir * max(0.0, float(getattr(args, "post_release_extra_right_distance", 0.0)))
    far_right[2] = escape_z

    info = {
        "hinge": np.round(hinge_pos, 4).tolist(),
        "handle": np.round(handle_goal, 4).tolist(),
        "away": np.round(away, 4).tolist(),
        "back_out": np.round(back_out, 4).tolist(),
        "right_safe": np.round(right_safe, 4).tolist(),
        "far_right": np.round(far_right, 4).tolist(),
        "back_delta": float(back_delta),
        "right_delta": float(right_delta),
        "extra_right": float(getattr(args, "post_release_extra_right_distance", 0.0)),
        "back_dir": np.round(back_dir, 4).tolist(),
        "right_dir": np.round(right_dir, 4).tolist(),
    }
    return away.astype(np.float32), back_out.astype(np.float32), right_safe.astype(np.float32), far_right.astype(np.float32), info


def sample_arc_min_distance(center_xy, radius, start_angle, delta, hinge_xy, handle_xy, samples=17):
    min_distance = float("inf")
    for ratio in np.linspace(0.0, 1.0, max(3, int(samples))):
        angle = float(start_angle) + float(delta) * float(ratio)
        point = np.asarray(
            [
                float(center_xy[0]) + float(radius) * math.cos(angle),
                float(center_xy[1]) + float(radius) * math.sin(angle),
            ],
            dtype=np.float32,
        )
        min_distance = min(min_distance, dc.point_segment_distance_2d(point, hinge_xy, handle_xy))
    return float(min_distance)


def choose_door_edge_arc_delta(center_xy, radius, start_vec, goal_vec, hinge_xy, handle_xy):
    start_angle = math.atan2(float(start_vec[1]), float(start_vec[0]))
    goal_angle = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
    shortest = math.atan2(math.sin(goal_angle - start_angle), math.cos(goal_angle - start_angle))
    candidates = [shortest]
    candidates.append(shortest - math.copysign(2.0 * math.pi, shortest if abs(shortest) > 1.0e-6 else 1.0))
    best_delta = candidates[0]
    best_score = -float("inf")
    best_clearance = 0.0
    for delta in candidates:
        clearance = sample_arc_min_distance(center_xy, radius, start_angle, delta, hinge_xy, handle_xy)
        score = clearance - 0.015 * abs(float(delta))
        if score > best_score:
            best_score = score
            best_delta = float(delta)
            best_clearance = float(clearance)
    return float(start_angle), float(best_delta), float(best_clearance)


def door_edge_inner_arc_pose(
    gym,
    env,
    door_actor,
    door,
    handle_goal,
    fallback_dir,
    start_pos,
    t,
    args,
    heading,
):
    hinge_pos, _hinge_quat = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
    inner_contact, push_dir = door_inner_contact_pose(gym, env, door_actor, door, handle_goal, fallback_dir, args)
    inner_precontact = inner_contact - push_dir * float(args.inner_precontact_offset)
    goal_xy = ensure_inside_xy(
        inner_precontact[:2],
        args,
        heading,
        float(args.inner_entry_distance),
    )
    center_xy = np.asarray(handle_goal[:2], dtype=np.float32)
    start_xy = np.asarray(start_pos[:2], dtype=np.float32)
    start_vec = start_xy - center_xy
    goal_vec = goal_xy - center_xy
    if np.linalg.norm(start_vec) < 1.0e-5:
        start_vec = -dc.normalize(push_dir[:2]) * float(args.inner_arc_clearance)
    if np.linalg.norm(goal_vec) < 1.0e-5:
        goal_vec = dc.normalize(heading) * float(args.inner_arc_clearance)
    radius = max(
        float(args.inner_arc_clearance),
        float(np.linalg.norm(start_vec)),
        float(np.linalg.norm(goal_vec)),
    )
    radius = min(float(args.inner_arc_max_radius), radius)
    start_dir = dc.normalize(start_vec)
    goal_dir = dc.normalize(goal_vec)
    start_angle, arc_delta, arc_clearance = choose_door_edge_arc_delta(
        center_xy,
        radius,
        start_dir,
        goal_dir,
        hinge_pos[:2],
        handle_goal[:2],
    )
    arc_t = dc.smoothstep(float(t))
    angle = start_angle + arc_delta * arc_t
    arc_xy = center_xy + np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32) * radius
    final_xy = dc.lerp(arc_xy, goal_xy, dc.smoothstep(max(0.0, float(t) - 0.78) / 0.22))
    lift_z = max(
        float(start_pos[2]),
        float(handle_goal[2]) + float(args.inner_arc_z_clearance),
        float(inner_precontact[2]) + 0.04,
    )
    if t < 0.80:
        z = dc.lerp(np.array([float(start_pos[2])], dtype=np.float32), np.array([lift_z], dtype=np.float32), dc.smoothstep(t / 0.80))[0]
    else:
        z = dc.lerp(np.array([lift_z], dtype=np.float32), np.array([float(inner_precontact[2])], dtype=np.float32), dc.smoothstep((t - 0.80) / 0.20))[0]
    target = np.asarray([float(final_xy[0]), float(final_xy[1]), float(z)], dtype=np.float32)
    tangent = np.asarray([-math.sin(angle), math.cos(angle), 0.0], dtype=np.float32) * np.sign(arc_delta)
    face_dir = dc.normalize(dc.lerp(tangent, push_dir, dc.smoothstep(max(0.0, float(t) - 0.55) / 0.45)))
    return target, inner_contact.astype(np.float32), push_dir.astype(np.float32), face_dir.astype(np.float32), float(arc_clearance)


def step_towards(current, desired, max_step):
    current = np.asarray(current, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    delta = desired - current
    distance = float(np.linalg.norm(delta))
    if distance <= max(1.0e-6, float(max_step)):
        return desired.copy()
    return (current + delta / distance * float(max_step)).astype(np.float32)


def step_scalar_towards(current, desired, max_step):
    current = float(current)
    desired = float(desired)
    max_step = max(0.0, float(max_step))
    delta = desired - current
    if abs(delta) <= max_step:
        return desired
    return current + math.copysign(max_step, delta)


def uniform_forward_base_step(current_xy, goal_xy, heading, step_size):
    current_xy = np.asarray(current_xy, dtype=np.float32)
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    heading = dc.normalize(np.asarray(heading, dtype=np.float32))
    remaining_forward = max(0.0, float(np.dot(goal_xy - current_xy, heading)))
    if remaining_forward <= 1.0e-6:
        return current_xy.copy()
    step_distance = min(float(step_size), remaining_forward)
    return (current_xy + heading * step_distance).astype(np.float32)


def maybe_limit_simple_base_motion(args, traj, step, phase, start_xy, desired_xy, yaw, hinge_xy, handle_xy):
    if bool(getattr(args, "simple_post_release_use_base_motion_limit", False)):
        return limit_live_base_motion(
            args,
            traj,
            step,
            phase,
            start_xy,
            desired_xy,
            yaw,
            hinge_xy,
            handle_xy,
        )
    return np.asarray(desired_xy, dtype=np.float32), {"limited": False}


def limit_live_base_motion(args, traj, step, phase, start_xy, desired_xy, yaw, hinge_xy, handle_xy):
    limited_xy, info = dc.limit_base_motion_by_door_segment(
        args=args,
        start_xy=start_xy,
        desired_xy=desired_xy,
        yaw=yaw,
        hinge_xy=hinge_xy,
        tip_xy=handle_xy,
        clearance=float(args.base_motion_door_clearance),
        samples=int(args.base_motion_limit_samples),
    )
    if info.get("limited", False):
        log_key = f"{phase}_base_motion_limit_log_step"
        interval = max(1, int(getattr(args, "collision_log_interval", 30)))
        if int(step) - int(traj.get(log_key, -10**9)) >= interval:
            traj[log_key] = int(step)
            print(
                f"[BaseMotionLimit] env={int(getattr(args, 'parallel_env_id', 0))} "
                f"step={int(step)} phase={phase} "
                f"fraction={float(info.get('motion_fraction', 0.0)):.3f} "
                f"clearance_distance={float(info.get('clearance_distance', 0.0)):.4f} "
                f"required_clearance={float(info.get('clearance', 0.0)):.3f}",
                flush=True,
            )
    return limited_xy.astype(np.float32), info


def simple_post_release_reach_target(args, base_xy, yaw, handle_goal, progress):
    reach_t = dc.smoothstep(float(np.clip(progress, 0.0, 1.0)))
    local_y = float(dc.lerp(
        np.array([float(args.simple_reach_right_y)], dtype=np.float32),
        np.array([float(args.simple_reach_left_y)], dtype=np.float32),
        reach_t,
    )[0])
    local = np.asarray(
        [
            float(args.simple_reach_base_x),
            local_y,
            float(handle_goal[2]) + float(args.simple_reach_z_offset) - float(args.robot_z),
        ],
        dtype=np.float32,
    )
    target = dc.base_pos_to_world(local, base_xy, args.robot_z, yaw)
    left_dir = np.asarray([-math.sin(float(yaw)), math.cos(float(yaw)), 0.0], dtype=np.float32)
    return target.astype(np.float32), dc.normalize(left_dir).astype(np.float32)


def debug_post_release_command(step, phase, args, traj, ik_state, base_xy, yaw, target_pos, handle_goal):
    interval = int(getattr(args, "post_release_debug_interval", 0))
    if interval <= 0 or phase not in ("move_to_inner_contact", "brace_open", "pass_through"):
        return
    if int(step) % max(1, interval) != 0:
        return
    target_local = dc.world_pos_to_base(target_pos, base_xy, args.robot_z, yaw)
    ee_pos = ik_state.current_pos_np
    ee_local = None if ee_pos is None else dc.world_pos_to_base(ee_pos, base_xy, args.robot_z, yaw)
    door_pos = traj.get("last_door_pos_for_debug")
    open_deg = dc.door_open_degrees(door_pos, args) if door_pos is not None else 0.0
    inner_contact = traj.get("inner_contact_pos")
    inner_push = traj.get("inner_push_dir")
    print(
        f"[PostReleaseDebug] env={int(getattr(args, 'parallel_env_id', 0))} "
        f"step={int(step)} phase={phase} open_deg={open_deg:.1f} "
        f"base={np.round(base_xy, 4).tolist()} "
        f"target={np.round(target_pos, 4).tolist()} "
        f"target_local={np.round(target_local, 4).tolist()} "
        f"ee={np.round(ee_pos, 4).tolist() if ee_pos is not None else None} "
        f"ee_local={np.round(ee_local, 4).tolist() if ee_local is not None else None} "
        f"ik_pos_err={float(getattr(ik_state, 'last_pos_error', 0.0)):.4f} "
        f"handle={np.round(handle_goal, 4).tolist()} "
        f"inner_contact={np.round(inner_contact, 4).tolist() if inner_contact is not None else None} "
        f"inner_push={np.round(inner_push, 4).tolist() if inner_push is not None else None}",
        flush=True,
    )


def ee_quat_facing_point(target_pos, face_pos, args, fallback_quat=None, roll_about_x=0.0):
    if args.ik_position_only:
        return None
    forward = np.asarray(face_pos, dtype=np.float32) - np.asarray(target_pos, dtype=np.float32)
    forward[2] = 0.0
    return dc.ee_quat_from_forward_xy(
        args,
        forward[:2],
        roll_about_x=roll_about_x,
        fallback_quat=fallback_quat,
    )


def ee_quat_facing_direction(forward_dir, args, fallback_quat=None, roll_about_x=0.0):
    if args.ik_position_only:
        return None
    forward = np.asarray(forward_dir, dtype=np.float32)
    forward[2] = 0.0
    return dc.ee_quat_from_forward_xy(
        args,
        forward[:2],
        roll_about_x=roll_about_x,
        fallback_quat=fallback_quat,
    )


def handle_contact_quat(handle_quat, traj, fallback_quat, args, target_pos=None, handle_goal=None, roll_about_x=None):
    if args.ik_position_only:
        return None
    _ = target_pos, handle_goal
    _ = roll_about_x
    local_quat = traj.get("handle_contact_quat_local")
    if local_quat is None or not bool(getattr(args, "pull_follow_orientation", False)):
        return None if fallback_quat is None else base_ik.normalize_quat(fallback_quat).astype(np.float32)
    return base_ik.normalize_quat(base_ik.quat_multiply(handle_quat, local_quat)).astype(np.float32)


def door_contact_quat(handle_quat, traj, fallback_quat, args, push_dir=None):
    if args.ik_position_only:
        return None
    if push_dir is not None:
        return ee_quat_facing_direction(push_dir, args, fallback_quat=fallback_quat)
    return handle_contact_quat(handle_quat, traj, fallback_quat, args)


def base_right_vector(yaw):
    return np.asarray([math.sin(float(yaw)), -math.cos(float(yaw))], dtype=np.float32)


def door_avoid_side_sign(gym, env, door_actor, door, base_xy, yaw):
    hinge_pos, _hinge_quat = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
    handle_pos, handle_quat = dc.get_body_pose(gym, env, door_actor, door.handle_body_index)
    handle_goal = dc.quat_apply(handle_quat, door.handle_goal_offset) + handle_pos
    hinge_local = dc.world_xy_to_base_local(hinge_pos[:2], base_xy, yaw)
    handle_local = dc.world_xy_to_base_local(handle_goal[:2], base_xy, yaw)
    mean_y = 0.5 * (float(hinge_local[1]) + float(handle_local[1]))
    return 1.0 if mean_y >= 0.0 else -1.0


def apply_base_lateral_offset(base_xy, yaw, side_sign, offset):
    return np.asarray(base_xy, dtype=np.float32) + base_right_vector(yaw) * float(side_sign) * float(offset)


def forward_only_base_target(start_xy, goal_xy, heading):
    start_xy = np.asarray(start_xy, dtype=np.float32)
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    heading = dc.normalize(np.asarray(heading, dtype=np.float32))
    forward = max(0.0, float(np.dot(goal_xy - start_xy, heading)))
    return (start_xy + heading * forward).astype(np.float32)


def post_release_base_goal(start_xy, nominal_goal_xy, heading, args):
    if bool(getattr(args, "allow_post_release_base_lateral", False)):
        return np.asarray(nominal_goal_xy, dtype=np.float32)
    return forward_only_base_target(start_xy, nominal_goal_xy, heading)


def post_release_clearance_yaw(yaw_pull, avoid_side_sign, args):
    return float(yaw_pull) - float(avoid_side_sign) * float(
        getattr(args, "post_release_yaw_clearance_delta", 0.0)
    )


def compute_reachable_inner_base_target(
    args,
    base_pull_xy,
    base_pass_xy,
    heading,
    yaw,
    side_sign,
    hinge_xy,
    handle_xy,
    inner_target_xy,
    min_progress_ratio,
):
    base_pull_xy = np.asarray(base_pull_xy, dtype=np.float32)
    base_pass_xy = np.asarray(base_pass_xy, dtype=np.float32)
    heading = dc.normalize(np.asarray(heading, dtype=np.float32))
    right = base_right_vector(yaw) * float(side_sign)
    inner_target_xy = np.asarray(inner_target_xy, dtype=np.float32)
    total_forward = max(0.0, float(np.dot(base_pass_xy - base_pull_xy, heading)))
    target_forward = float(np.dot(inner_target_xy - base_pull_xy, heading))
    reach_x = max(0.05, float(args.inner_base_reach_x_max))
    needed_forward = max(
        total_forward * float(np.clip(min_progress_ratio, 0.0, 1.0)),
        target_forward - reach_x + float(args.inner_base_reach_extra),
    )
    needed_forward = float(np.clip(needed_forward, 0.0, total_forward))
    lateral_max = max(0.0, float(args.inner_base_lateral_search_max))
    if not bool(getattr(args, "allow_post_release_base_lateral", False)):
        lateral_max = 0.0
    samples = max(2, int(args.inner_base_lateral_samples))
    clearance = float(args.base_motion_door_clearance)
    best_xy = base_pull_xy + heading * needed_forward
    best_distance = -float("inf")
    chosen_lateral = 0.0
    chosen_safe = False
    for lateral in np.linspace(0.0, lateral_max, samples):
        candidate = base_pull_xy + heading * needed_forward + right * float(lateral)
        distance = dc.door_segment_distance_to_base(
            hinge_xy,
            handle_xy,
            candidate,
            yaw,
            float(args.base_collision_front_extent),
            float(args.base_collision_rear_extent),
            float(args.base_collision_half_width),
            clearance=clearance,
        )
        score = float(distance) - 0.01 * float(lateral)
        if distance > 0.0:
            best_xy = candidate
            best_distance = float(distance)
            chosen_lateral = float(lateral)
            chosen_safe = True
            break
        if score > best_distance:
            best_xy = candidate
            best_distance = score
            chosen_lateral = float(lateral)
    info = {
        "needed_forward": float(needed_forward),
        "target_forward": float(target_forward),
        "reach_x": float(reach_x),
        "lateral": float(chosen_lateral),
        "safe": bool(chosen_safe),
        "clearance_distance": float(best_distance),
    }
    return best_xy.astype(np.float32), info


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
    yaw_start,
    yaw_pull,
    traj,
):
    handle_pos, handle_quat = dc.get_body_pose(gym, env, door_actor, door.handle_body_index)
    handle_goal = dc.quat_apply(handle_quat, door.handle_goal_offset) + handle_pos
    base_xy_current = traj.get("base_xy", base_start)
    approach_dir = np.array([base_xy_current[0], base_xy_current[1], args.robot_z], dtype=np.float32) - handle_goal
    approach_dir[2] = 0.0
    approach_dir = dc.normalize(approach_dir)
    if np.linalg.norm(approach_dir) < 1.0e-5:
        approach_dir = np.array([math.cos(yaw_start), math.sin(yaw_start), 0.0], dtype=np.float32)

    pregrasp = handle_goal + approach_dir * args.pregrasp_offset
    grasp = handle_goal + approach_dir * args.grasp_offset
    pregrasp[0] += args.grasp_x_offset
    pregrasp[2] += args.grasp_z_offset
    grasp[0] += args.grasp_x_offset
    grasp[2] += args.grasp_z_offset
    goal_quat = dc.forward_ee_quat(args, yaw_start)

    rotate_offset = np.zeros(3, dtype=np.float32)
    rotate_offset[1] = args.handle_rotate_right_distance
    rotate_offset[2] = -args.handle_rotate_down_distance
    rotate_pos = grasp + rotate_offset

    pull_dir = dc.quat_axis(handle_quat, axis=2)
    pull_dir[2] = 0.0
    pull_dir = dc.normalize(pull_dir)
    if np.linalg.norm(pull_dir) < 1.0e-5:
        pull_dir = approach_dir.copy()
    if float(np.dot(pull_dir, approach_dir)) < 0.0:
        pull_dir = -pull_dir
    pull_pos = rotate_pos + pull_dir * args.door_pull_distance

    walk_end = args.walk_steps
    initial_end = walk_end + args.initial_hold_steps
    grasp_end = initial_end + args.grasp_steps
    grasp_hold_end = grasp_end + args.grasp_hold_steps
    close_end = grasp_hold_end + args.gripper_close_steps
    rotate_end = close_end + args.handle_rotate_steps
    retreat_end = rotate_end + args.base_retreat_steps
    pull_end = retreat_end + args.door_pull_steps
    if "pull_complete_step" in traj:
        pull_end = min(pull_end, max(retreat_end + 1, int(traj["pull_complete_step"])))
    open_hold_end = pull_end + args.open_hold_steps
    release_end = open_hold_end + args.release_handle_steps
    inner_contact_end = release_end + args.move_to_inner_contact_steps
    if "inner_extend_until" in traj:
        inner_contact_end = max(inner_contact_end, int(traj["inner_extend_until"]))
    brace_open_end = inner_contact_end + args.brace_open_steps
    if "brace_extend_until" in traj:
        brace_open_end = max(brace_open_end, int(traj["brace_extend_until"]))
    if "brace_complete_step" in traj:
        brace_open_end = min(brace_open_end, max(inner_contact_end + 1, int(traj["brace_complete_step"])))
    pass_end = brace_open_end + args.pass_through_steps
    if "pass_extend_until" in traj:
        pass_end = max(pass_end, int(traj["pass_extend_until"]))
    return_home_end = pass_end + args.return_home_steps

    gripper_closed = args.gripper_open + (args.gripper_closed - args.gripper_open) * args.gripper_close_ratio
    heading = np.asarray([math.cos(yaw_start), math.sin(yaw_start)], dtype=np.float32)
    avoid_side_sign = float(traj.get("pull_avoid_side_sign", 1.0))
    fallback_base_pull_lateral = apply_base_lateral_offset(
        base_pull,
        yaw_start,
        avoid_side_sign,
        getattr(args, "pull_base_lateral_offset", 0.0),
    )
    base_pull_lateral = np.asarray(
        traj.get("base_pull_lateral", fallback_base_pull_lateral),
        dtype=np.float32,
    )
    nominal_base_pass = apply_base_lateral_offset(
        dc.compute_base_pass_target(args, heading),
        yaw_start,
        avoid_side_sign,
        getattr(args, "pass_base_lateral_offset", 0.0),
    )
    fallback_base_pass = post_release_base_goal(base_pull_lateral, nominal_base_pass, heading, args)
    base_pass = np.asarray(traj.get("base_pass", fallback_base_pass), dtype=np.float32)
    yaw_clearance = post_release_clearance_yaw(yaw_pull, avoid_side_sign, args)
    inner_precontact, _inner_contact_seed, _inner_push_seed = door_inner_precontact_pose(
        gym,
        env,
        door_actor,
        door,
        handle_goal,
        pull_dir,
        args,
        traj,
    )
    inner_reach_xy = ensure_inside_xy(
        inner_precontact[:2],
        args,
        heading,
        float(args.inner_entry_distance),
    )
    hinge_pos_for_base, _ = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
    base_inner, inner_base_info = compute_reachable_inner_base_target(
        args,
        base_pull_lateral,
        base_pass,
        heading,
        yaw_clearance,
        avoid_side_sign,
        hinge_pos_for_base[:2],
        handle_goal[:2],
        inner_reach_xy,
        float(np.clip(args.inner_base_advance_ratio, 0.0, 0.85)),
    )
    base_brace, brace_base_info = compute_reachable_inner_base_target(
        args,
        base_pull_lateral,
        base_pass,
        heading,
        yaw_clearance,
        avoid_side_sign,
        hinge_pos_for_base[:2],
        handle_goal[:2],
        inner_reach_xy,
        float(np.clip(args.brace_base_advance_ratio, 0.0, 0.95)),
    )
    traj["base_pull_lateral"] = base_pull_lateral.copy()
    traj["base_pass"] = base_pass.copy()
    traj["base_inner"] = base_inner.copy()
    traj["base_brace"] = base_brace.copy()
    traj["inner_base_info"] = inner_base_info
    traj["brace_base_info"] = brace_base_info
    target_pos = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else pregrasp.copy()
    target_quat = None if args.ik_position_only else goal_quat.copy()
    gripper = args.gripper_open
    base_xy = base_start.copy()
    yaw = yaw_start
    phase = "walk"

    if "home_ee_base_pos" not in traj and ik_state.current_pos_np is not None:
        home_base_xy = traj.get("base_xy", base_start)
        home_yaw = float(traj.get("yaw", yaw_start))
        traj["home_ee_base_pos"] = dc.world_pos_to_base(
            ik_state.current_pos_np,
            home_base_xy,
            args.robot_z,
            home_yaw,
        )
        if ik_state.current_quat_np is not None:
            traj["home_ee_base_quat"] = dc.world_quat_to_base(ik_state.current_quat_np, home_yaw)

    if step < walk_end:
        t = dc.smoothstep((step + 1) / max(1, args.walk_steps))
        base_xy = dc.lerp(base_start, base_stop, t)
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
            closed_hinge_pos, _ = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
            traj["closed_hinge_xy"] = closed_hinge_pos[:2].copy()
            traj["closed_handle_vec_xy"] = (handle_goal[:2] - closed_hinge_pos[:2]).astype(np.float32).copy()
            traj["rotate_base_pos"] = dc.world_pos_to_base(rotate_pos, base_stop, args.robot_z, yaw_start)
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
                t = dc.smoothstep((initial_step + 1) / move_steps)
                target_pos = dc.lerp(traj["initial_hold_start_pos"], traj["pregrasp"], t)
                if not args.ik_position_only:
                    target_quat = dc.quat_nlerp(traj["initial_hold_start_quat"], traj["goal_quat"], t)
        elif step < grasp_end:
            t = dc.smoothstep((step - initial_end + 1) / max(1, args.grasp_steps))
            target_pos = dc.lerp(traj["pregrasp"], traj["grasp"], t)
            phase = "grasp"
        elif step < grasp_hold_end:
            target_pos = traj["grasp"].copy()
            phase = "grasp_hold"
        elif step < close_end:
            t = dc.smoothstep((step - grasp_hold_end + 1) / max(1, args.gripper_close_steps))
            target_pos = traj["grasp"].copy()
            gripper = args.gripper_open + (gripper_closed - args.gripper_open) * t
            phase = "close_gripper"
        elif step < rotate_end:
            rotate_progress = (step - close_end + 1) / max(1, args.handle_rotate_steps)
            t = dc.smoothstep(rotate_progress)
            target_pos = dc.lerp(traj["grasp"], traj["rotate"], t)
            turned_quat = None if args.ik_position_only else base_ik.quat_multiply(
                traj["goal_quat"],
                dc.quat_from_angle_axis(-t * args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            target_quat = handle_contact_quat(
                handle_quat,
                traj,
                turned_quat,
                args,
                target_pos=target_pos,
                handle_goal=handle_goal,
                roll_about_x=-t * args.handle_rotate_angle,
            )
            base_xy = base_stop.copy()
            yaw = yaw_start
            gripper = gripper_closed
            phase = "rotate_handle"
        elif step < retreat_end:
            t = dc.smoothstep((step - rotate_end + 1) / max(1, args.base_retreat_steps))
            turned_quat = base_ik.quat_multiply(
                traj["goal_quat"],
                dc.quat_from_angle_axis(-args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            traj["turned_quat"] = turned_quat.copy()
            if "handle_contact_offset_local" not in traj:
                traj["handle_contact_offset_local"] = dc.quat_apply(
                    base_ik.quat_conjugate(handle_quat),
                    traj["rotate"] - handle_goal,
                )
                traj["handle_contact_quat_local"] = base_ik.quat_multiply(
                    base_ik.quat_conjugate(handle_quat),
                    turned_quat,
                )
            if "handle_contact_offset_local" in traj:
                live_pull_dir = traj["pull_dir"].copy()
                handle_contact_offset = dc.quat_apply(handle_quat, traj["handle_contact_offset_local"])
                target_pos = handle_goal + handle_contact_offset + live_pull_dir * args.pull_contact_bias
            else:
                target_pos = traj["rotate"].copy()
            base_xy = dc.lerp(base_stop, base_pull, t)
            yaw = float(dc.lerp(np.array([yaw_start], dtype=np.float32), np.array([yaw_pull], dtype=np.float32), t)[0])
            target_quat = handle_contact_quat(
                handle_quat,
                traj,
                turned_quat,
                args,
                target_pos=target_pos,
                handle_goal=handle_goal,
            )
            gripper = gripper_closed
            phase = "retreat_base"
        elif step < pull_end:
            t = dc.smoothstep((step - retreat_end + 1) / max(1, args.door_pull_steps))
            turned_quat = base_ik.quat_multiply(
                traj["goal_quat"],
                dc.quat_from_angle_axis(-args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            if "handle_contact_offset_local" not in traj:
                traj["handle_contact_offset_local"] = dc.quat_apply(
                    base_ik.quat_conjugate(handle_quat),
                    traj["rotate"] - handle_goal,
                )
                traj["handle_contact_quat_local"] = base_ik.quat_multiply(
                    base_ik.quat_conjugate(handle_quat),
                    turned_quat,
                )

            live_pull_dir = traj["pull_dir"].copy()
            if door.open_stage:
                live_pull_dir = dc.handle_open_tangent_dir(
                    gym,
                    env,
                    door_actor,
                    door,
                    handle_goal,
                    traj["pull_dir"],
                    args,
                )

            if "handle_contact_offset_local" in traj:
                handle_contact_offset = dc.quat_apply(handle_quat, traj["handle_contact_offset_local"])
                handle_target_pos = handle_goal + handle_contact_offset + live_pull_dir * args.pull_contact_bias
            else:
                handle_target_pos = handle_goal + live_pull_dir * args.pull_contact_bias

            door_pos, _ = dc.get_actor_dof_state(gym, env, door_actor)
            door_open_ratio = dc.door_hinge_open_ratio(door, float(door_pos[0]), args) if len(door_pos) > 0 else 0.0
            door_open_deg = dc.door_open_degrees(door_pos, args)
            if (
                door.open_stage
                and door_open_deg >= float(args.pre_inner_open_angle_deg)
                and "pull_complete_step" not in traj
            ):
                traj["pull_complete_step"] = int(step) + max(0, int(args.pull_settle_steps))
                print(
                    f"[PullDoorPhase] env={int(getattr(args, 'parallel_env_id', 0))} "
                    f"pre_inner_open_angle reached at step={int(step)} "
                    f"open_deg={door_open_deg:.1f}; release starts after "
                    f"{max(0, int(args.pull_settle_steps))} settle steps.",
                    flush=True,
                )
            follow_handle = (
                door.open_stage
                and t <= args.handle_follow_pull_ratio
                and "handle_contact_offset_local" in traj
            )
            if door.open_stage and door_open_deg >= float(args.pre_inner_open_angle_deg) and ik_state.current_pos_np is not None:
                target_pos = ik_state.current_pos_np.copy()
                target_quat = handle_contact_quat(
                    handle_quat,
                    traj,
                    turned_quat,
                    args,
                    target_pos=target_pos,
                    handle_goal=handle_goal,
                )
            elif follow_handle:
                target_pos = handle_target_pos.copy()
                target_quat = handle_contact_quat(
                    handle_quat,
                    traj,
                    turned_quat,
                    args,
                    target_pos=target_pos,
                    handle_goal=handle_goal,
                )
            elif door.open_stage:
                target_pos = handle_target_pos + live_pull_dir * min(
                    float(args.door_pull_distance) * t,
                    float(args.pull_target_max_distance),
                )
                blend_start = min(float(args.door_freeze_blend_start_ratio), float(args.door_freeze_target_ratio) - 1.0e-4)
                if door_open_ratio >= blend_start:
                    blend_t = dc.smoothstep(
                        (door_open_ratio - blend_start)
                        / max(1.0e-4, float(args.door_freeze_target_ratio) - blend_start)
                    )
                    target_pos = dc.lerp(target_pos, handle_target_pos, blend_t)
                target_quat = handle_contact_quat(
                    handle_quat,
                    traj,
                    turned_quat,
                    args,
                    target_pos=target_pos,
                    handle_goal=handle_goal,
                )
            elif args.unidoor_style_pull and ik_state.current_pos_np is not None:
                pull_step_pos = ik_state.current_pos_np + traj["pull_dir"] * args.lever_step_size
                pull_max_pos = traj["rotate"] + traj["pull_dir"] * args.door_pull_distance
                progress = float(np.dot(pull_step_pos - traj["rotate"], traj["pull_dir"]))
                target_pos = pull_max_pos if progress > args.door_pull_distance else pull_step_pos
                target_quat = handle_contact_quat(
                    handle_quat,
                    traj,
                    turned_quat,
                    args,
                    target_pos=target_pos,
                    handle_goal=handle_goal,
                )
            else:
                target_pos = dc.lerp(traj["rotate"], traj["pull"], t)
                target_quat = handle_contact_quat(
                    handle_quat,
                    traj,
                    turned_quat,
                    args,
                    target_pos=target_pos,
                    handle_goal=handle_goal,
                )
            base_xy_linear = base_pull.copy()
            lateral_start = float(args.pull_lateral_start_open_deg)
            lateral_end = max(lateral_start + 1.0e-4, float(args.pull_lateral_end_open_deg))
            lateral_t = dc.smoothstep((door_open_deg - lateral_start) / (lateral_end - lateral_start))
            base_xy = dc.lerp(
                base_xy_linear,
                apply_base_lateral_offset(
                    base_xy_linear,
                    yaw_start,
                    avoid_side_sign,
                    args.pull_base_lateral_offset,
                ),
                lateral_t,
            )
            yaw = yaw_pull
            gripper = gripper_closed
            phase = "pull_door"
        elif step < open_hold_end:
            live_pull_dir = dc.handle_open_tangent_dir(
                gym,
                env,
                door_actor,
                door,
                handle_goal,
                traj["pull_dir"],
                args,
            )
            door_pos, _ = dc.get_actor_dof_state(gym, env, door_actor)
            open_enough = (
                len(door_pos) > 0
                and dc.door_open_degrees(door_pos, args) >= float(args.pre_inner_open_angle_deg)
            )
            if open_enough and ik_state.current_pos_np is not None:
                target_pos = ik_state.current_pos_np.copy()
            elif ik_state.current_pos_np is not None:
                target_pos = ik_state.current_pos_np + live_pull_dir * float(args.open_hold_step_size)
            else:
                target_pos = traj["pull"].copy()
            target_quat = handle_contact_quat(
                handle_quat,
                traj,
                traj.get("turned_quat", traj["goal_quat"]),
                args,
                target_pos=target_pos,
                handle_goal=handle_goal,
            )
            base_xy = base_pull_lateral.copy()
            yaw = yaw_pull
            gripper = gripper_closed
            phase = "open_hold"
        elif step < release_end:
            release_progress = (step - open_hold_end + 1) / max(1, args.release_handle_steps)
            if "release_start_pos" not in traj:
                traj["release_start_pos"] = (
                    ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else traj["pull"].copy()
                )
            open_ratio = float(np.clip(args.gripper_open_stage_ratio, 0.05, 0.9))
            target_pos = traj["release_start_pos"].copy()
            target_quat = handle_contact_quat(
                handle_quat,
                traj,
                traj.get("turned_quat", traj.get("goal_quat")),
                args,
                target_pos=target_pos,
                handle_goal=handle_goal,
            )
            base_xy = base_pull_lateral.copy()
            yaw = yaw_pull
            open_t = dc.smoothstep(release_progress / open_ratio)
            gripper = dc.lerp(np.array([gripper_closed], dtype=np.float32), np.array([args.gripper_open], dtype=np.float32), open_t)[0]
            phase = "release_handle"
        elif step < inner_contact_end:
            raw_t = float((step - release_end + 1) / max(1, args.move_to_inner_contact_steps))
            t = dc.smoothstep(raw_t)
            if "inner_move_start_pos" not in traj:
                traj["inner_move_start_pos"] = (
                    ik_state.current_pos_np.copy()
                    if ik_state.current_pos_np is not None
                    else traj.get("release_start_pos", traj["pull"]).copy()
                )
                traj["inner_move_start_base_xy"] = traj.get("base_xy", base_pull_lateral).copy()
                traj["inner_arc_logged"] = False
                traj["inner_extension_steps"] = 0
                away, back_out, right_safe, far_right, escape_info = post_release_escape_waypoints(
                    gym,
                    env,
                    door_actor,
                    door,
                    handle_goal,
                    traj["inner_move_start_pos"],
                    heading,
                    yaw_pull,
                    args,
                    traj,
                )
                traj["post_release_away"] = away.copy()
                traj["post_release_back_out"] = back_out.copy()
                traj["post_release_right_safe"] = right_safe.copy()
                traj["post_release_far_right"] = far_right.copy()
                traj["post_release_escape_info"] = escape_info
                print(
                    f"[InnerBasePlan] env={int(getattr(args, 'parallel_env_id', 0))} "
                    f"inner_base={np.round(base_inner, 3).tolist()} "
                    f"brace_base={np.round(base_brace, 3).tolist()} "
                    f"inner_info={traj.get('inner_base_info', {})}",
                    flush=True,
                )
            if not traj.get("inner_arc_logged", False):
                traj["inner_arc_logged"] = True
                print(
                    f"[PostReleaseEscape] env={int(getattr(args, 'parallel_env_id', 0))} "
                    f"info={traj.get('post_release_escape_info', {})} "
                    f"start={np.round(traj['inner_move_start_pos'], 3).tolist()}",
                    flush=True,
                )
            escape_start = traj["inner_move_start_pos"]
            escape_away = traj["post_release_away"]
            escape_back = traj["post_release_back_out"]
            escape_right = traj["post_release_right_safe"]
            escape_far_right = traj["post_release_far_right"]
            base_xy = traj["inner_move_start_base_xy"].copy()
            yaw = yaw_pull
            face_dir = np.asarray(handle_goal, dtype=np.float32) - np.asarray(escape_right, dtype=np.float32)
            face_dir[2] = 0.0
            inner_push_dir = traj.get("inner_push_dir", traj["pull_dir"]).copy()
            if t < 0.22:
                seg_t = dc.smoothstep(t / 0.22)
                target_pos = dc.lerp(escape_start, escape_away, seg_t)
                face_dir = np.asarray(handle_goal, dtype=np.float32) - np.asarray(target_pos, dtype=np.float32)
                face_dir[2] = 0.0
            elif t < 0.50:
                seg_t = dc.smoothstep((t - 0.22) / 0.28)
                target_pos = dc.lerp(escape_away, escape_back, seg_t)
                face_dir = np.asarray(handle_goal, dtype=np.float32) - np.asarray(target_pos, dtype=np.float32)
                face_dir[2] = 0.0
            elif t < 0.75:
                seg_t = dc.smoothstep((t - 0.50) / 0.25)
                target_pos = dc.lerp(escape_back, escape_right, seg_t)
                face_dir = np.asarray(handle_goal, dtype=np.float32) - np.asarray(target_pos, dtype=np.float32)
                face_dir[2] = 0.0
            else:
                seg_t = dc.smoothstep((t - 0.75) / 0.25)
                target_pos = dc.lerp(escape_right, escape_far_right, seg_t)
                face_dir = np.asarray(handle_goal, dtype=np.float32) - np.asarray(target_pos, dtype=np.float32)
                face_dir[2] = 0.0
                traj["inner_contact_pos"] = target_pos.copy()
                traj["inner_push_dir"] = face_dir.copy()
            target_quat = door_contact_quat(
                handle_quat,
                traj,
                traj.get("goal_quat"),
                args,
                push_dir=face_dir,
            )
            if ik_state.current_pos_np is not None:
                target_pos = step_towards(
                    ik_state.current_pos_np,
                    target_pos,
                    float(args.inner_arc_approach_step),
                )
            yaw = yaw_pull
            gripper = args.gripper_open
            phase = "move_to_inner_contact"
            current_for_ready = ik_state.current_pos_np if ik_state.current_pos_np is not None else target_pos
            ee_escape_dist = float(np.linalg.norm(current_for_ready[:2] - target_pos[:2]))
            target_escape_dist = float(np.linalg.norm(target_pos[:2] - traj.get("last_target_pos", target_pos)[:2]))
            inner_ready = raw_t >= 0.98 and ee_escape_dist <= float(args.inner_fan_ready_distance)
            if inner_ready and "inner_ready_step" not in traj:
                traj["inner_ready_step"] = int(step) + 1
                print(
                    f"[InnerReady] env={int(getattr(args, 'parallel_env_id', 0))} "
                    f"step={int(step)} ee_escape_dist={ee_escape_dist:.3f} "
                    f"target_delta={target_escape_dist:.3f}",
                    flush=True,
                )
            elif not inner_ready and step >= inner_contact_end - 2 and "inner_ready_step" not in traj:
                extension_used = int(traj.get("inner_extension_steps", 0))
                extension_left = max(0, int(args.inner_max_extra_steps) - extension_used)
                if extension_left > 0:
                    extension = min(80, extension_left)
                    traj["inner_extension_steps"] = extension_used + extension
                    traj["inner_extend_until"] = int(step) + extension
                    print(
                        f"[InnerExtend] env={int(getattr(args, 'parallel_env_id', 0))} "
                        f"step={int(step)} ee_escape_dist={ee_escape_dist:.3f} "
                        f"target_delta={target_escape_dist:.3f} "
                        f"required={float(args.inner_fan_ready_distance):.3f} "
                        f"extend={extension} total_extra={traj['inner_extension_steps']}",
                        flush=True,
                    )
        elif step < brace_open_end:
            simple_post_release = bool(getattr(args, "simple_post_release_reach", True))
            if "brace_start_base_xy" not in traj:
                traj["brace_start_base_xy"] = traj.get("base_xy", base_inner).copy()
                traj["brace_extension_steps"] = 0
                traj["simple_reach_progress"] = float(traj.get("simple_reach_progress", 0.0))
            door_pos, _ = dc.get_actor_dof_state(gym, env, door_actor)
            door_open_deg = dc.door_open_degrees(door_pos, args)
            brace_t = dc.smoothstep(
                (step - inner_contact_end + 1)
                / max(1.0, float(args.brace_open_steps) * float(args.brace_base_time_scale))
            )
            yaw = float(dc.lerp(
                np.array([yaw_pull], dtype=np.float32),
                np.array([yaw_clearance], dtype=np.float32),
                brace_t,
            )[0])
            hinge_pos_for_limit, _ = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
            if simple_post_release:
                yaw = yaw_clearance
                desired_base_xy = uniform_forward_base_step(
                    traj.get("base_xy", traj["brace_start_base_xy"]),
                    base_pass,
                    heading,
                    float(args.simple_post_release_base_step),
                )
                base_xy, _base_limit_info = maybe_limit_simple_base_motion(
                    args,
                    traj,
                    step,
                    "brace_open",
                    traj.get("base_xy", traj["brace_start_base_xy"]),
                    desired_base_xy,
                    yaw,
                    hinge_pos_for_limit[:2],
                    handle_goal[:2],
                )
                simple_reach_progress = step_scalar_towards(
                    traj.get("simple_reach_progress", 0.0),
                    1.0,
                    float(args.simple_reach_progress_step),
                )
                traj["simple_reach_progress"] = simple_reach_progress
                desired_contact_target, inner_push_dir = simple_post_release_reach_target(
                    args,
                    base_xy,
                    yaw,
                    handle_goal,
                    simple_reach_progress,
                )
                inner_contact = desired_contact_target.copy()
            else:
                desired_base_xy = dc.lerp(
                    traj["brace_start_base_xy"],
                    base_brace,
                    brace_t,
                )
                base_xy, _base_limit_info = limit_live_base_motion(
                    args,
                    traj,
                    step,
                    "brace_open",
                    traj.get("base_xy", traj["brace_start_base_xy"]),
                    desired_base_xy,
                    yaw,
                    hinge_pos_for_limit[:2],
                    handle_goal[:2],
                )
                inner_contact, inner_push_dir = door_inner_sliding_contact_pose(
                    gym,
                    env,
                    door_actor,
                    door,
                    handle_goal,
                    traj.get("inner_push_dir", traj["pull_dir"]),
                    base_xy,
                    yaw,
                    args,
                    traj,
                    reach_push_bias=float(args.brace_contact_push_bias),
                )
                push_bias = (
                    float(args.brace_contact_hold_bias)
                    if door_open_deg >= float(args.brace_target_open_angle_deg)
                    else float(args.brace_contact_push_bias)
                )
                desired_contact_target = inner_contact + inner_push_dir * push_bias
                desired_contact_target = dc.clamp_world_pos_to_base_box(
                    desired_contact_target,
                    base_xy,
                    args.robot_z,
                    yaw,
                    min_xyz=np.array([0.20, -0.42, 0.04], dtype=np.float32),
                    max_xyz=np.array([0.82, 0.42, 0.58], dtype=np.float32),
                )
            traj["inner_contact_pos"] = inner_contact.copy()
            traj["inner_push_dir"] = inner_push_dir.copy()
            contact_step_size = (
                float(args.simple_reach_approach_step)
                if simple_post_release
                else float(args.brace_contact_approach_step)
            )
            if ik_state.current_pos_np is not None:
                target_pos = step_towards(
                    ik_state.current_pos_np,
                    desired_contact_target,
                    contact_step_size,
                )
            else:
                target_pos = desired_contact_target.copy()
            target_quat = door_contact_quat(
                handle_quat,
                traj,
                traj.get("goal_quat"),
                args,
                push_dir=inner_push_dir,
            )
            gripper = args.gripper_open
            phase = "brace_open"
            brace_elapsed = int(step) - int(inner_contact_end)
            can_finish_brace = brace_elapsed >= max(0, int(args.brace_min_contact_steps))
            brace_done = door_open_deg >= float(args.brace_target_open_angle_deg) or bool(
                getattr(args, "simple_brace_complete_on_steps", True)
            )
            if can_finish_brace and brace_done and "brace_complete_step" not in traj:
                traj["brace_complete_step"] = int(step) + max(0, int(args.brace_settle_steps))
                print(
                    f"[BraceOpen] env={int(getattr(args, 'parallel_env_id', 0))} "
                    f"step={int(step)} open_deg={door_open_deg:.1f}; "
                    f"pass starts after {max(0, int(args.brace_settle_steps))} settle steps.",
                    flush=True,
                )
            elif step >= brace_open_end - 2 and "brace_complete_step" not in traj:
                extension_used = int(traj.get("brace_extension_steps", 0))
                extension_left = max(0, int(args.brace_max_extra_steps) - extension_used)
                if extension_left > 0:
                    extension = min(120, extension_left)
                    traj["brace_extension_steps"] = extension_used + extension
                    traj["brace_extend_until"] = int(step) + extension
                    print(
                        f"[BraceExtend] env={int(getattr(args, 'parallel_env_id', 0))} "
                        f"step={int(step)} open_deg={door_open_deg:.1f} "
                        f"target={float(args.brace_target_open_angle_deg):.1f} "
                        f"extend={extension} total_extra={traj['brace_extension_steps']}",
                        flush=True,
                    )
        elif step < pass_end:
            simple_post_release = bool(getattr(args, "simple_post_release_reach", True))
            if "pass_start_base_xy" not in traj:
                traj["pass_start_base_xy"] = traj.get("base_xy", base_brace).copy()
                traj["pass_progress"] = 0.0
            door_pos, _ = dc.get_actor_dof_state(gym, env, door_actor)
            door_open_deg = dc.door_open_degrees(door_pos, args)
            yaw = yaw_clearance
            hinge_pos_for_limit, _ = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
            if simple_post_release:
                desired_base_xy = uniform_forward_base_step(
                    traj.get("base_xy", traj["pass_start_base_xy"]),
                    base_pass,
                    heading,
                    float(args.simple_post_release_base_step),
                )
                base_xy, _base_limit_info = maybe_limit_simple_base_motion(
                    args,
                    traj,
                    step,
                    "pass_through",
                    traj.get("base_xy", traj["pass_start_base_xy"]),
                    desired_base_xy,
                    yaw,
                    hinge_pos_for_limit[:2],
                    handle_goal[:2],
                )
                traj["pass_progress"] = dc.base_path_progress_ratio(traj["pass_start_base_xy"], base_pass, base_xy)
                simple_reach_progress = step_scalar_towards(
                    traj.get("simple_reach_progress", 0.0),
                    1.0,
                    float(args.simple_reach_progress_step),
                )
                traj["simple_reach_progress"] = simple_reach_progress
            else:
                if door_open_deg >= float(args.pass_min_open_angle_deg):
                    progress_scale = 1.0
                else:
                    progress_scale = float(np.clip(args.pass_closed_progress_scale, 0.0, 1.0))
                old_pass_progress = float(traj.get("pass_progress", 0.0))
                desired_pass_progress = min(
                    1.0,
                    old_pass_progress + progress_scale / max(1, int(args.pass_through_steps)),
                )
                lateral_stage = float(np.clip(args.pass_lateral_stage_ratio, 0.0, 0.8))
                pass_side_xy = np.asarray(
                    [
                        float(traj["pass_start_base_xy"][0]),
                        float(base_pass[1]),
                    ],
                    dtype=np.float32,
                )
                if lateral_stage > 1.0e-4 and desired_pass_progress < lateral_stage:
                    side_t = dc.smoothstep(desired_pass_progress / lateral_stage)
                    desired_base_xy = dc.lerp(traj["pass_start_base_xy"], pass_side_xy, side_t)
                elif lateral_stage <= 1.0e-4:
                    desired_base_xy = dc.lerp(traj["pass_start_base_xy"], base_pass, dc.smoothstep(desired_pass_progress))
                else:
                    forward_t = (
                        1.0
                        if lateral_stage >= 0.999
                        else dc.smoothstep((desired_pass_progress - lateral_stage) / max(1.0e-4, 1.0 - lateral_stage))
                    )
                    desired_base_xy = dc.lerp(pass_side_xy, base_pass, forward_t)
                base_xy, base_limit_info = limit_live_base_motion(
                    args,
                    traj,
                    step,
                    "pass_through",
                    traj.get("base_xy", traj["pass_start_base_xy"]),
                    desired_base_xy,
                    yaw,
                    hinge_pos_for_limit[:2],
                    handle_goal[:2],
                )
                if base_limit_info.get("limited", False):
                    actual_path_progress = dc.base_path_progress_ratio(traj["pass_start_base_xy"], base_pass, base_xy)
                    traj["pass_progress"] = min(
                        desired_pass_progress,
                        max(old_pass_progress, actual_path_progress),
                    )
                else:
                    traj["pass_progress"] = desired_pass_progress
            if (
                traj["pass_progress"] < 0.999
                or float(np.linalg.norm(np.asarray(base_xy, dtype=np.float32) - base_pass)) > 0.05
            ) and step >= pass_end - 2:
                traj["pass_extend_until"] = int(step) + 120
            if simple_post_release:
                inner_contact, inner_push_dir = simple_post_release_reach_target(
                    args,
                    base_xy,
                    yaw,
                    handle_goal,
                    traj.get("simple_reach_progress", 0.0),
                )
                pass_contact = inner_contact.copy()
            else:
                inner_contact, inner_push_dir = door_inner_sliding_contact_pose(
                    gym,
                    env,
                    door_actor,
                    door,
                    handle_goal,
                    traj.get("inner_push_dir", traj["pull_dir"]),
                    base_xy,
                    yaw,
                    args,
                    traj,
                    reach_push_bias=float(args.pass_contact_push_bias) + float(args.pass_reopen_push_extra),
                )
                reopen_t = float(
                    np.clip(
                        (float(args.pass_min_open_angle_deg) + 4.0 - door_open_deg) / 8.0,
                        0.0,
                        1.0,
                    )
                )
                pass_push_bias = float(args.pass_contact_push_bias) + float(args.pass_reopen_push_extra) * reopen_t
                pass_contact = inner_contact + inner_push_dir * pass_push_bias
                pass_contact = dc.clamp_world_pos_to_base_box(
                    pass_contact,
                    base_xy,
                    args.robot_z,
                    yaw,
                    min_xyz=np.array([0.18, -0.44, 0.04], dtype=np.float32),
                    max_xyz=np.array([0.84, 0.44, 0.58], dtype=np.float32),
                )
            traj["inner_contact_pos"] = inner_contact.copy()
            traj["inner_push_dir"] = inner_push_dir.copy()
            ee_inside_progress = -float("inf")
            if ik_state.current_pos_np is not None:
                ee_inside_progress = inside_progress_along_heading(
                    ik_state.current_pos_np[:2],
                    args,
                    heading,
                )
            if (
                ik_state.current_pos_np is not None
                and ee_inside_progress >= float(args.pass_home_retract_ee_inside_distance)
                and "return_home_start_step" not in traj
            ):
                traj["return_home_start_step"] = int(step)
                traj["pass_home_retract_active"] = True
                print(
                    f"[PassHomeRetract] env={int(getattr(args, 'parallel_env_id', 0))} "
                    f"step={int(step)} ee_inside={ee_inside_progress:.3f} "
                    f"threshold={float(args.pass_home_retract_ee_inside_distance):.3f} "
                    f"base_dist={float(np.linalg.norm(np.asarray(base_xy, dtype=np.float32) - base_pass)):.3f}",
                    flush=True,
                )

            if bool(traj.get("pass_home_retract_active", False)):
                start_step = int(traj.get("return_home_start_step", step))
                return_home_t = dc.smoothstep(
                    max(0.0, float(step - start_step)) / max(1.0, float(args.return_home_steps))
                )
                target_pos, target_quat = dc.chase_target_to_current_ee(
                    traj,
                    ik_state,
                    args,
                    pass_contact.copy(),
                    door_contact_quat(
                        handle_quat,
                        traj,
                        traj.get("goal_quat"),
                        args,
                        push_dir=inner_push_dir,
                    ),
                )
                gripper = args.gripper_open
                traj["return_home_alpha"] = return_home_t
                phase = "return_home"
            elif ik_state.current_pos_np is not None:
                target_pos = step_towards(
                    ik_state.current_pos_np,
                    pass_contact,
                    float(args.simple_reach_approach_step)
                    if simple_post_release
                    else float(args.pass_contact_approach_step),
                )
            else:
                target_pos = pass_contact.copy()
            if not bool(traj.get("pass_home_retract_active", False)):
                target_quat = door_contact_quat(
                    handle_quat,
                    traj,
                    traj.get("goal_quat"),
                    args,
                    push_dir=inner_push_dir,
                )
                gripper = args.gripper_open
                phase = "pass_through"
        elif step < return_home_end:
            if "return_home_start_step" not in traj:
                traj["return_home_start_step"] = int(step)
            start_step = int(traj["return_home_start_step"])
            t = dc.smoothstep(max(0.0, float(step - start_step)) / max(1.0, float(args.return_home_steps)))
            if bool(traj.get("pass_home_retract_active", False)):
                base_xy = traj.get("base_xy", base_pass).copy()
                yaw = yaw_clearance
            else:
                if "return_home_start_base_xy" not in traj:
                    traj["return_home_start_base_xy"] = traj.get("base_xy", base_pass).copy()
                    traj["return_home_start_yaw"] = float(traj.get("yaw", yaw_pull))
                base_xy = dc.lerp(traj["return_home_start_base_xy"], base_pass, t)
                yaw = float(dc.lerp(
                    np.array([traj["return_home_start_yaw"]], dtype=np.float32),
                    np.array([yaw_pull], dtype=np.float32),
                    t,
                )[0])
            target_pos, target_quat = dc.chase_target_to_current_ee(
                traj,
                ik_state,
                args,
                traj.get("inner_contact_pos", traj["pull"]).copy(),
                traj.get("goal_quat"),
            )
            gripper = args.gripper_open
            traj["return_home_alpha"] = t
            phase = "return_home"
        else:
            home_base_pos = traj.get("home_ee_base_pos")
            fallback_pos = (
                dc.base_pos_to_world(home_base_pos, base_pass, args.robot_z, yaw_pull)
                if home_base_pos is not None
                else traj.get("inner_contact_pos", traj["pull"]).copy()
            )
            fallback_quat = None
            if not args.ik_position_only and "home_ee_base_quat" in traj:
                fallback_quat = dc.base_quat_to_world(traj["home_ee_base_quat"], yaw_pull)
            target_pos, target_quat = dc.chase_target_to_current_ee(
                traj,
                ik_state,
                args,
                fallback_pos,
                fallback_quat,
            )
            base_xy = base_pass.copy()
            yaw = yaw_pull
            gripper = args.gripper_open
            traj["return_home_alpha"] = 1.0
            phase = "hold_home"

    if int(getattr(args, "post_release_debug_interval", 0)) > 0:
        try:
            door_pos_debug, _door_vel_debug = dc.get_actor_dof_state(gym, env, door_actor)
            traj["last_door_pos_for_debug"] = door_pos_debug
        except Exception:
            traj["last_door_pos_for_debug"] = None
        debug_post_release_command(step, phase, args, traj, ik_state, base_xy, yaw, target_pos, handle_goal)

    traj["base_xy"] = base_xy.copy()
    traj["yaw"] = float(yaw)
    traj["last_target_pos"] = np.asarray(target_pos, dtype=np.float32).copy()
    traj["last_target_quat"] = (
        None
        if target_quat is None
        else base_ik.normalize_quat(np.asarray(target_quat, dtype=np.float32)).astype(np.float32)
    )
    return phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal


def initialize_parallel_env_state(
    gym,
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
    avoid_side_sign = door_avoid_side_sign(gym, env, door_actor, door, base_stop, yaw_start)
    hinge_pos, _hinge_quat = dc.get_body_pose(gym, env, door_actor, door.door_body_index)
    handle_pos, handle_quat = dc.get_body_pose(gym, env, door_actor, door.handle_body_index)
    handle_goal = dc.quat_apply(handle_quat, door.handle_goal_offset) + handle_pos
    base_pull, sweep_retreat_info = dc.compute_safe_base_retreat_for_door_sweep(
        args=args,
        base_stop=base_stop,
        heading=heading,
        yaw=yaw_start,
        hinge_xy=hinge_pos[:2],
        closed_tip_xy=handle_goal[:2],
        max_open_angle_deg=float(args.safe_sweep_open_angle_deg),
    )
    base_pull_lateral = apply_base_lateral_offset(
        base_pull,
        yaw_start,
        avoid_side_sign,
        args.pull_base_lateral_offset,
    )
    nominal_base_pass = apply_base_lateral_offset(
        dc.compute_base_pass_target(args, heading),
        yaw_start,
        avoid_side_sign,
        args.pass_base_lateral_offset,
    )
    base_pass = post_release_base_goal(base_pull_lateral, nominal_base_pass, heading, args)
    return dc.ParallelDoorEnvState(
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
        base_goal=base_pull,
        yaw_start=yaw_start,
        yaw_goal=yaw_start + args.pull_base_yaw_delta,
        traj={
            "base_xy": base_start.copy(),
            "base_pull_lateral": base_pull_lateral.copy(),
            "base_pass": base_pass.copy(),
            "pull_avoid_side_sign": float(avoid_side_sign),
            "sweep_retreat_info": sweep_retreat_info,
        },
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
    envs_per_row = max(1, int(math.ceil(math.sqrt(float(args.num_envs)))))
    created = []
    for env_index in range(int(args.num_envs)):
        door_template = door_templates[env_index % len(door_templates)]
        door = dc.clone_door_runtime(door_template)
        env_args = make_env_args(args, env_index)
        env, arm_actor, actor_handles, door_actor, _ = dc.create_parallel_env_actors(
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
        if env_args.show_camera_images and (env_args.enable_wrist_camera or env_args.enable_front_camera):
            camera_handles = dc.create_low_level_cameras(gym, env, arm_actor, actor_handles, env_args)
        ik_state = base_ik.setup_ik_controller(gym, sim, env, arm_actor, arm_asset, dof_names, lower, upper, env_args)
        env_states.append(
            initialize_parallel_env_state(
                gym,
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
            )
        )
    shown_randomization = [json.loads(st.args.ikpull_randomization_json) for st in env_states[: min(4, len(env_states))]]
    print(
        f"ikpull per-env randomization seed={int(args.seed)} "
        f"enabled={bool(getattr(args, 'ikpull_env_randomization', False)) and not bool(args.no_ikpull_env_randomization)} "
        f"door_cycle={[door.spec.get('name', '') for door in door_templates]} "
        f"sample_envs={shown_randomization}",
        flush=True,
    )
    return env_states


def run_parallel_demo(gym, sim, env_states, viewer, args, dt, dof_names):
    if not env_states:
        raise RuntimeError("No parallel envs were created.")
    num_arm_dofs = len(env_states[0].dof_positions)
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    gripper_idx = dof_dict.get("jointGripper")
    max_steps = args.steps if args.steps > 0 else 3000
    start = time.time()
    step = 0
    print(
        f"Parallel float_ik pull run: num_envs={len(env_states)} steps={max_steps}",
        flush=True,
    )
    first = env_states[0]
    print(
        "base_start:",
        first.base_start.tolist(),
        "base_stop:",
        first.base_stop.tolist(),
        "base_pull:",
        first.base_goal.tolist(),
        "base_pull_lateral:",
        first.traj.get("base_pull_lateral", []).tolist(),
        "base_pass:",
        first.traj.get("base_pass", []).tolist(),
        "avoid_side:",
        float(first.traj.get("pull_avoid_side_sign", 1.0)),
        "sweep_retreat:",
        first.traj.get("sweep_retreat_info", {}),
    )
    print("Close viewer to exit.")

    while step < max_steps:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        gym.refresh_rigid_body_state_tensor(sim)
        for st in env_states:
            dc.current_ee_pose_from_refreshed_tensors(st.ik_state)

        for st in env_states:
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
                st.base_goal,
                st.yaw_start,
                st.yaw_goal,
                st.traj,
            )
            st.last_phase = phase
            st.last_handle_goal = handle_goal
            st.last_target_pos = np.asarray(target_pos, dtype=np.float32).copy()
            st.last_target_quat = None if target_quat is None else np.asarray(target_quat, dtype=np.float32).copy()
            st.last_gripper = float(gripper)

            dc.monitor_command_jumps(gym, step, st, base_xy, yaw, target_pos, phase)
            dc.set_robot_base_pose(gym, st.env, st.actor_handles, base_xy, st.args.robot_z, yaw)
            if phase == "return_home":
                if "return_home_start_dofs" not in st.traj:
                    st.traj["return_home_start_dofs"] = np.asarray(st.dof_positions, dtype=np.float32).copy()
                alpha = float(st.traj.get("return_home_alpha", 0.0))
                st.dof_positions[:] = dc.lerp(st.traj["return_home_start_dofs"], st.home_positions, alpha)
                st.ik_state.last_pos_error = 0.0
            elif phase == "hold_home":
                st.dof_positions[:] = st.home_positions
                st.ik_state.last_pos_error = 0.0
            else:
                dc.set_ik_target(st.ik_state, target_pos, target_quat)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            if st.last_phase not in ("return_home", "hold_home"):
                dc.update_arm_ik_targets_for_env(
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
            door_pos, door_vel = dc.get_actor_dof_state(gym, st.env, st.door_actor)
            st.last_door_pos = door_pos
            door_efforts = dc.compute_door_efforts(st.door, door_pos, door_vel, st.args)
            if len(door_efforts) > 0:
                gym.apply_actor_dof_efforts(st.env, st.door_actor, door_efforts)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        need_camera_render = bool(
            any(st.camera_handles for st in env_states) and args.show_camera_images
        )
        if viewer is not None and need_camera_render and (args.draw_ik_target or args.draw_camera_axes):
            gym.clear_lines(viewer)
        if viewer is not None or need_camera_render:
            gym.step_graphics(sim)
        if args.show_camera_images and env_states[0].camera_handles and step % max(1, args.camera_display_interval) == 0:
            dc.show_camera_handle_images(gym, sim, env_states[0].env, env_states[0].camera_handles, env_states[0].args)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            door_pos_record, _door_vel_record = dc.get_actor_dof_state(gym, st.env, st.door_actor)
            st.last_door_pos = door_pos_record
            st.success = st.success or dc.door_success(door_pos_record, st.args)
            dc.monitor_base_door_collision(gym, step, st)
            st.prev_base_xy = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32).copy()
            st.prev_yaw = float(st.traj.get("yaw", st.yaw_start))

        if viewer is not None:
            if args.draw_ik_target or args.draw_camera_axes:
                gym.clear_lines(viewer)
            for st in env_states[: min(4, len(env_states))]:
                if args.draw_camera_axes:
                    dc.draw_low_level_camera_axes(gym, viewer, st.env, st.arm_actor, st.actor_handles, st.args)
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
            open_deg = [
                round(dc.door_open_degrees(st.last_door_pos, st.args), 1) if st.last_door_pos is not None else 0.0
                for st in shown
            ]
            phases = [st.last_phase for st in shown]
            successes = sum(st.success for st in env_states)
            print(
                f"[{step:04d}] phases={phases} door_deg={door_deg} "
                f"open_deg={open_deg} success={successes}/{len(env_states)}",
                flush=True,
            )
        step += 1

    elapsed = time.time() - start
    successes = sum(st.success for st in env_states)
    collision_envs = sum(bool(getattr(st, "base_door_collision_detected", False)) for st in env_states)
    print(
        f"Done after {step} steps ({elapsed:.2f}s). "
        f"pull_success={successes}/{len(env_states)} "
        f"base_door_collision_envs={collision_envs}/{len(env_states)}"
    )


def main():
    args = parse_args()
    seed = dc.resolve_seed(args)
    print(f"ikpull seed={seed}", flush=True)
    gym = gymapi.acquire_gym()
    sim, dt = base_ik.create_sim(gym, args)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    with tempfile.TemporaryDirectory(prefix="b1z1_float_ik_pull_door_assets_") as temp_dir:
        base_asset, arm_asset = base_ik.load_robot_assets(gym, sim, args, Path(temp_dir))
        door_templates = dc.load_door_assets(gym, sim, args)
        if base_asset is not None:
            base_ik.print_collision_summary(gym, base_asset, "base visual actor", verbose=args.print_collision_summary)
        base_ik.print_collision_summary(gym, arm_asset, "arm articulated actor", verbose=args.print_collision_summary)
        dof_data = base_ik.configure_dofs(gym, arm_asset, args)
        dof_names, dof_props, dof_states, dof_positions, lower, upper, defaults, speeds, selected = dof_data
        _ = speeds, selected
        if "jointGripper" in dof_names:
            dof_states["pos"][dof_names.index("jointGripper")] = args.gripper_open
            dof_positions[dof_names.index("jointGripper")] = args.gripper_open
        env_states = create_parallel_env_states(
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
        viewer = dc.setup_viewer(gym, sim, args)
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
