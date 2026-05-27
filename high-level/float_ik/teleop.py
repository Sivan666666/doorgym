#!/usr/bin/env python3
"""Keyboard teleop for the float-base B1Z1 arm and door scene.

This script reuses the asset loading, door setup, and damped-LS IK helpers from
isaacgym_float_ik_b1z1_basearn_push_door.py, but replaces the scripted
door-push trajectory with viewer keyboard control.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import isaacgym_float_ik_b1z1_basearn_push_door as push_door


base_ik = push_door.base_ik
gymapi = push_door.gymapi
gymutil = push_door.gymutil
cv2 = push_door.cv2


CONTINUOUS_ACTIONS = {
    "base_forward",
    "base_back",
    "base_left",
    "base_right",
    "base_yaw_left",
    "base_yaw_right",
    "ee_x_plus",
    "ee_x_minus",
    "ee_y_plus",
    "ee_y_minus",
    "ee_z_plus",
    "ee_z_minus",
    "ee_roll_plus",
    "ee_roll_minus",
    "ee_pitch_plus",
    "ee_pitch_minus",
    "ee_yaw_plus",
    "ee_yaw_minus",
    "gripper_open",
    "gripper_close",
    "speed_fast",
}


@dataclass
class TeleopState:
    base_xy: np.ndarray
    base_z: float
    base_yaw: float
    ee_target_pos: np.ndarray
    ee_target_quat: np.ndarray | None
    gripper: float
    initial_base_xy: np.ndarray
    initial_base_yaw: float
    initial_ee_target_pos: np.ndarray
    initial_ee_target_quat: np.ndarray | None
    initial_dofs: np.ndarray
    pressed: set[str] = field(default_factory=set)
    paused: bool = False
    quit: bool = False
    reset_requested: bool = False


def parse_args():
    args = gymutil.parse_arguments(
        description="B1Z1 float-base arm + door keyboard teleop.",
        headless=True,
        no_graphics=True,
        custom_parameters=[
            {"name": "--asset_root", "type": str, "default": str(base_ik.DEFAULT_ASSET_ROOT)},
            {"name": "--asset_file", "type": str, "default": base_ik.DEFAULT_ASSET_FILE},
            {"name": "--rl_device", "type": str, "default": "cuda:0"},
            {"name": "--steps", "type": int, "default": 0, "help": "0 means run until the viewer is closed."},
            {"name": "--door_cfg", "type": str, "default": str(push_door.DEFAULT_DOOR_CFG)},
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
            {"name": "--gripper_open", "type": float, "default": -1.5707963267948966},
            {"name": "--gripper_closed", "type": float, "default": 0.0},
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
            {"name": "--base_linear_speed", "type": float, "default": 0.45, "help": "Base XY teleop speed in m/s."},
            {"name": "--base_yaw_speed", "type": float, "default": 0.9, "help": "Base yaw teleop speed in rad/s."},
            {"name": "--ee_linear_speed", "type": float, "default": 0.18, "help": "EE target translation speed in m/s."},
            {"name": "--ee_rot_speed", "type": float, "default": 0.8, "help": "EE target rotation speed in rad/s."},
            {"name": "--gripper_speed", "type": float, "default": 1.5, "help": "Gripper target speed in rad/s."},
            {"name": "--fast_multiplier", "type": float, "default": 3.0},
            {"name": "--ee_control_frame", "type": str, "default": "base", "help": "base or world."},
            {
                "name": "--ee_fixed_world",
                "action": "store_true",
                "help": "Keep the EE target fixed in world coordinates while the base moves.",
            },
            {"name": "--ee_min_z", "type": float, "default": 0.05},
            {"name": "--ee_max_z", "type": float, "default": 1.6},
            {"name": "--log_interval", "type": int, "default": 30},
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
        ],
    )

    args.ee_control_frame = args.ee_control_frame.lower().strip()
    if args.ee_control_frame not in {"base", "world"}:
        raise ValueError("--ee_control_frame must be 'base' or 'world'")

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

    # Attributes consumed by the shared base/IK helper module.
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
    args.ee_follow_base = not bool(args.ee_fixed_world)
    args.rgb = False

    return args


def subscribe_if_available(gym, viewer, key_name, action):
    key = getattr(gymapi, key_name, None)
    if key is not None:
        gym.subscribe_viewer_keyboard_event(viewer, key, action)


def subscribe_teleop_keyboard(gym, viewer):
    if viewer is None:
        return

    bindings = [
        ("KEY_W", "base_forward"),
        ("KEY_S", "base_back"),
        ("KEY_A", "base_left"),
        ("KEY_D", "base_right"),
        ("KEY_Q", "base_yaw_left"),
        ("KEY_E", "base_yaw_right"),
        ("KEY_UP", "ee_x_plus"),
        ("KEY_DOWN", "ee_x_minus"),
        ("KEY_LEFT", "ee_y_plus"),
        ("KEY_RIGHT", "ee_y_minus"),
        ("KEY_PAGE_UP", "ee_z_plus"),
        ("KEY_PAGE_DOWN", "ee_z_minus"),
        ("KEY_U", "ee_roll_plus"),
        ("KEY_O", "ee_roll_minus"),
        ("KEY_I", "ee_pitch_plus"),
        ("KEY_K", "ee_pitch_minus"),
        ("KEY_J", "ee_yaw_plus"),
        ("KEY_L", "ee_yaw_minus"),
        ("KEY_Z", "gripper_open"),
        ("KEY_X", "gripper_close"),
        ("KEY_LEFT_SHIFT", "speed_fast"),
        ("KEY_RIGHT_SHIFT", "speed_fast"),
    ]
    for key_name, action in bindings:
        subscribe_if_available(gym, viewer, key_name, action)


def print_controls(args):
    frame = "base frame" if args.ee_control_frame == "base" else "world frame"
    print("Keyboard teleop controls:")
    print("  Base: W/S forward/back, A/D strafe, Q/E yaw")
    print(f"  EE target translation ({frame}): ArrowUp/Down x, ArrowLeft/Right y, PageUp/PageDown z")
    print("  EE target rotation: U/O roll, I/K pitch, J/L yaw")
    print("  Gripper: Z open, X close")
    print("  Shift: faster, Space: pause, R: reset, Esc: quit")


def yaw_rotate(vec, yaw):
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]], dtype=np.float32)


def get_current_ee_pose(gym, sim, ik_state):
    gym.refresh_rigid_body_state_tensor(sim)
    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    pos = eef_state[:3].detach().cpu().numpy().copy()
    quat = eef_state[3:7].detach().cpu().numpy().copy()
    return pos, base_ik.normalize_quat(quat)


def get_handle_goal(gym, env, door_actor, door):
    handle_pos, handle_quat = push_door.get_body_pose(gym, env, door_actor, door.handle_body_index)
    return push_door.quat_apply(handle_quat, door.handle_goal_offset) + handle_pos


def poll_viewer_events(gym, viewer, state):
    if viewer is None:
        return

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action in CONTINUOUS_ACTIONS:
            if evt.value > 0:
                state.pressed.add(evt.action)
            else:
                state.pressed.discard(evt.action)
        elif evt.action == "pause" and evt.value > 0:
            state.paused = not state.paused
            print("Teleop paused." if state.paused else "Teleop resumed.")
        elif evt.action == "reset" and evt.value > 0:
            state.reset_requested = True
        elif evt.action == "quit" and evt.value > 0:
            state.quit = True


def apply_base_controls(state, args, dt):
    prev_xy = state.base_xy.copy()
    prev_yaw = float(state.base_yaw)
    multiplier = float(args.fast_multiplier) if "speed_fast" in state.pressed else 1.0
    heading = np.array([math.cos(state.base_yaw), math.sin(state.base_yaw)], dtype=np.float32)
    left = np.array([-heading[1], heading[0]], dtype=np.float32)
    move = np.zeros(2, dtype=np.float32)

    if "base_forward" in state.pressed:
        move += heading
    if "base_back" in state.pressed:
        move -= heading
    if "base_left" in state.pressed:
        move += left
    if "base_right" in state.pressed:
        move -= left

    norm = float(np.linalg.norm(move))
    if norm > 1.0:
        move /= norm
    state.base_xy += move * float(args.base_linear_speed) * multiplier * dt

    yaw_dir = 0.0
    if "base_yaw_left" in state.pressed:
        yaw_dir += 1.0
    if "base_yaw_right" in state.pressed:
        yaw_dir -= 1.0
    state.base_yaw += yaw_dir * float(args.base_yaw_speed) * multiplier * dt

    if args.ee_follow_base:
        delta_yaw = float(state.base_yaw - prev_yaw)
        prev_origin = np.array([prev_xy[0], prev_xy[1], state.base_z], dtype=np.float32)
        next_origin = np.array([state.base_xy[0], state.base_xy[1], state.base_z], dtype=np.float32)
        state.ee_target_pos = next_origin + yaw_rotate(state.ee_target_pos - prev_origin, delta_yaw)
        if state.ee_target_quat is not None:
            state.ee_target_quat = base_ik.normalize_quat(
                base_ik.quat_multiply(base_ik.yaw_quat(delta_yaw), state.ee_target_quat)
            )


def apply_ee_controls(state, args, dt):
    multiplier = float(args.fast_multiplier) if "speed_fast" in state.pressed else 1.0
    pos_delta = np.zeros(3, dtype=np.float32)

    if "ee_x_plus" in state.pressed:
        pos_delta[0] += 1.0
    if "ee_x_minus" in state.pressed:
        pos_delta[0] -= 1.0
    if "ee_y_plus" in state.pressed:
        pos_delta[1] += 1.0
    if "ee_y_minus" in state.pressed:
        pos_delta[1] -= 1.0
    if "ee_z_plus" in state.pressed:
        pos_delta[2] += 1.0
    if "ee_z_minus" in state.pressed:
        pos_delta[2] -= 1.0

    norm = float(np.linalg.norm(pos_delta))
    if norm > 1.0:
        pos_delta /= norm
    if norm > 0.0:
        if args.ee_control_frame == "base":
            pos_delta = yaw_rotate(pos_delta, state.base_yaw)
        state.ee_target_pos += pos_delta * float(args.ee_linear_speed) * multiplier * dt
        state.ee_target_pos[2] = np.clip(state.ee_target_pos[2], float(args.ee_min_z), float(args.ee_max_z))

    if state.ee_target_quat is not None:
        rot_delta = np.zeros(3, dtype=np.float32)
        if "ee_roll_plus" in state.pressed:
            rot_delta[0] += 1.0
        if "ee_roll_minus" in state.pressed:
            rot_delta[0] -= 1.0
        if "ee_pitch_plus" in state.pressed:
            rot_delta[1] += 1.0
        if "ee_pitch_minus" in state.pressed:
            rot_delta[1] -= 1.0
        if "ee_yaw_plus" in state.pressed:
            rot_delta[2] += 1.0
        if "ee_yaw_minus" in state.pressed:
            rot_delta[2] -= 1.0

        angle_scale = float(args.ee_rot_speed) * multiplier * dt
        for axis, direction in enumerate(rot_delta):
            if direction == 0.0:
                continue
            axis_vec = np.zeros(3, dtype=np.float32)
            axis_vec[axis] = 1.0
            if args.ee_control_frame == "base":
                axis_vec = yaw_rotate(axis_vec, state.base_yaw)
            delta_q = push_door.quat_from_angle_axis(float(direction) * angle_scale, axis_vec)
            state.ee_target_quat = base_ik.normalize_quat(base_ik.quat_multiply(delta_q, state.ee_target_quat))

    if "gripper_close" in state.pressed:
        state.gripper += float(args.gripper_speed) * multiplier * dt
    if "gripper_open" in state.pressed:
        state.gripper -= float(args.gripper_speed) * multiplier * dt
    lower = min(float(args.gripper_open), float(args.gripper_closed))
    upper = max(float(args.gripper_open), float(args.gripper_closed))
    state.gripper = float(np.clip(state.gripper, lower, upper))


def reset_door(gym, env, door_actor, door, args):
    door.open_stage = False
    states = gym.get_actor_dof_states(env, door_actor, gymapi.STATE_ALL)
    if len(states) == 0:
        return

    states["pos"][:] = 0.0
    states["vel"][:] = 0.0
    if len(states) >= 1:
        states["pos"][0] = door.dof_upper[0] if args.door_motion_sign < 0.0 else door.dof_lower[0]
    if len(states) >= 2:
        states["pos"][1] = door.dof_lower[1]
    gym.set_actor_dof_states(env, door_actor, states, gymapi.STATE_ALL)


def apply_reset(
    gym,
    env,
    arm_actor,
    actor_handles,
    door_actor,
    door,
    dof_states,
    dof_positions,
    ik_state,
    state,
    args,
):
    state.base_xy = state.initial_base_xy.copy()
    state.base_yaw = float(state.initial_base_yaw)
    state.ee_target_pos = state.initial_ee_target_pos.copy()
    state.ee_target_quat = None if state.initial_ee_target_quat is None else state.initial_ee_target_quat.copy()
    state.gripper = float(args.gripper_open)
    state.pressed.clear()

    dof_positions[:] = state.initial_dofs
    dof_states["vel"][:] = 0.0
    push_door.set_robot_base_pose(gym, env, actor_handles, state.base_xy, state.base_z, state.base_yaw)
    push_door.set_ik_target(ik_state, state.ee_target_pos, state.ee_target_quat)
    gym.set_actor_dof_states(env, arm_actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, arm_actor, dof_positions)
    reset_door(gym, env, door_actor, door, args)
    state.reset_requested = False
    print("Teleop reset.")


def setup_viewer(gym, sim, args):
    viewer = push_door.setup_viewer(gym, sim, args)
    subscribe_teleop_keyboard(gym, viewer)
    return viewer


def run_teleop(
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
    dof_states,
    dof_positions,
    defaults,
    ik_state,
    initial_base_xy,
):
    num_arm_dofs = len(dof_positions)
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    gripper_idx = dof_dict.get("jointGripper")
    if gripper_idx is not None:
        dof_positions[gripper_idx] = np.clip(
            args.gripper_open,
            ik_state.lower[gripper_idx].item(),
            ik_state.upper[gripper_idx].item(),
        )
        gym.set_actor_dof_position_targets(env, arm_actor, dof_positions)

    ee_pos, ee_quat = get_current_ee_pose(gym, sim, ik_state)
    initial_quat = None if args.ik_position_only else ee_quat.copy()
    initial_dofs = np.asarray(defaults, dtype=np.float32).copy()
    if gripper_idx is not None:
        initial_dofs[gripper_idx] = dof_positions[gripper_idx]

    state = TeleopState(
        base_xy=np.asarray(initial_base_xy, dtype=np.float32).copy(),
        base_z=float(args.robot_z),
        base_yaw=float(args.robot_yaw),
        ee_target_pos=ee_pos.copy(),
        ee_target_quat=initial_quat,
        gripper=float(dof_positions[gripper_idx]) if gripper_idx is not None else float(args.gripper_open),
        initial_base_xy=np.asarray(initial_base_xy, dtype=np.float32).copy(),
        initial_base_yaw=float(args.robot_yaw),
        initial_ee_target_pos=ee_pos.copy(),
        initial_ee_target_quat=None if initial_quat is None else initial_quat.copy(),
        initial_dofs=initial_dofs,
    )
    push_door.set_ik_target(ik_state, state.ee_target_pos, state.ee_target_quat)

    print_controls(args)
    if viewer is None:
        print("Viewer is disabled; keyboard teleop needs a viewer. Use --steps N for a headless load test.")

    max_steps = int(args.steps) if args.steps > 0 else math.inf
    if viewer is None and max_steps == math.inf:
        max_steps = 1

    start = time.time()
    step = 0
    while step < max_steps:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        poll_viewer_events(gym, viewer, state)
        if state.quit:
            break
        if state.reset_requested:
            apply_reset(
                gym,
                env,
                arm_actor,
                actor_handles,
                door_actor,
                door,
                dof_states,
                dof_positions,
                ik_state,
                state,
                args,
            )

        if not state.paused:
            apply_base_controls(state, args, dt)
            apply_ee_controls(state, args, dt)
            push_door.set_robot_base_pose(gym, env, actor_handles, state.base_xy, state.base_z, state.base_yaw)
            push_door.set_ik_target(ik_state, state.ee_target_pos, state.ee_target_quat)
            push_door.update_arm_ik_targets(gym, sim, dof_positions, ik_state, args, num_arm_dofs)
            if gripper_idx is not None:
                dof_positions[gripper_idx] = np.clip(
                    state.gripper,
                    ik_state.lower[gripper_idx].item(),
                    ik_state.upper[gripper_idx].item(),
                )
            gym.set_actor_dof_position_targets(env, arm_actor, dof_positions)

            push_door.enforce_locked_door_hinge(gym, env, door_actor, door)
            door_pos, door_vel = push_door.get_actor_dof_state(gym, env, door_actor)
            door_efforts = push_door.compute_door_efforts(door, door_pos, door_vel, args)
            if len(door_efforts) > 0:
                gym.apply_actor_dof_efforts(env, door_actor, door_efforts)

            gym.simulate(sim)
            gym.fetch_results(sim, True)
        else:
            door_pos, _ = push_door.get_actor_dof_state(gym, env, door_actor)

        if viewer is not None or (args.show_camera_images and camera_handles):
            gym.step_graphics(sim)
        if args.show_camera_images and camera_handles and step % max(1, args.camera_display_interval) == 0:
            push_door.show_camera_handle_images(gym, sim, env, camera_handles, args)

        if viewer is not None:
            if args.draw_ik_target or args.draw_camera_axes:
                gym.clear_lines(viewer)
            if args.draw_camera_axes:
                push_door.draw_low_level_camera_axes(gym, viewer, env, arm_actor, actor_handles, args)
            if args.draw_ik_target:
                push_door.refresh_current_ee_pose(gym, sim, ik_state)
                base_ik.draw_ik_target(gym, viewer, env, ik_state)
                handle_goal = get_handle_goal(gym, env, door_actor, door)
                goal_sphere = gymutil.WireframeSphereGeometry(
                    radius=0.035,
                    num_lats=8,
                    num_lons=8,
                    color=(0.0, 1.0, 0.2),
                    color2=(0.0, 0.7, 0.2),
                )
                gymutil.draw_lines(goal_sphere, gym, viewer, env, base_ik.transform_from_arrays(handle_goal))
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"[{step:04d}] teleop "
                f"base=({state.base_xy[0]:.2f},{state.base_xy[1]:.2f},{math.degrees(state.base_yaw):.1f}deg) "
                f"ee_target=({state.ee_target_pos[0]:.2f},{state.ee_target_pos[1]:.2f},{state.ee_target_pos[2]:.2f}) "
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

    with tempfile.TemporaryDirectory(prefix="b1z1_float_ik_teleop_assets_") as temp_dir:
        base_asset, arm_asset = base_ik.load_robot_assets(gym, sim, args, Path(temp_dir))
        door = push_door.load_door_asset(gym, sim, args)
        if base_asset is not None:
            base_ik.print_collision_summary(gym, base_asset, "base visual actor", verbose=args.print_collision_summary)
        base_ik.print_collision_summary(gym, arm_asset, "arm articulated actor", verbose=args.print_collision_summary)
        dof_data = base_ik.configure_dofs(gym, arm_asset, args)
        dof_names, dof_props, dof_states, dof_positions, lower, upper, defaults, speeds, selected = dof_data
        if "jointGripper" in dof_names:
            dof_states["pos"][dof_names.index("jointGripper")] = args.gripper_open
            dof_positions[dof_names.index("jointGripper")] = args.gripper_open
        env, arm_actor, actor_handles, door_actor, initial_base_xy = push_door.create_env_actors(
            gym, sim, base_asset, arm_asset, door, dof_props, dof_states, args
        )
        viewer = setup_viewer(gym, sim, args)
        camera_handles = {}
        if args.show_camera_images and (args.enable_wrist_camera or args.enable_front_camera):
            camera_handles = push_door.create_low_level_cameras(gym, env, arm_actor, actor_handles, args)
        ik_state = base_ik.setup_ik_controller(gym, sim, env, arm_actor, arm_asset, dof_names, lower, upper, args)
        try:
            run_teleop(
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
                dof_states,
                dof_positions,
                defaults,
                ik_state,
                initial_base_xy,
            )
        finally:
            if args.show_camera_images and cv2 is not None:
                cv2.destroyAllWindows()
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
