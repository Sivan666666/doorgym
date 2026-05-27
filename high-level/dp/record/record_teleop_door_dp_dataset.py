#!/usr/bin/env python3
"""Record keyboard teleop demonstrations into Door DP raw .npz episodes.

This keeps the scene/control path from high-level/float_ik/teleop.py and only
adds a small recording layer. Press C in the Isaac Gym viewer to start
recording, press C again to save the current episode.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DP_RECORD_ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = DP_RECORD_ROOT.parent
FLOAT_IK_ROOT = HIGH_LEVEL_ROOT / "float_ik"
for _path in (str(DP_RECORD_ROOT), str(FLOAT_IK_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import teleop  # noqa: E402
import door_common as dc  # noqa: E402
from door_dp_common import RawDoorDPRecorder, normalize_vision_mode  # noqa: E402


push_door = teleop.push_door
base_ik = teleop.base_ik
gymapi = teleop.gymapi
gymutil = teleop.gymutil
cv2 = teleop.cv2
TELEOP_STATE_MODE = dc.FLOAT_DP_STATE_MODE_PI05_CURRENT_STATE10


@dataclass
class RecordingControl:
    active: bool = False
    recorder: RawDoorDPRecorder | None = None
    prev_base_xy: np.ndarray | None = None
    prev_yaw: float | None = None
    last_dp_action: np.ndarray | None = None
    last_wrist_mask_rgb: np.ndarray | None = None
    last_wrist_second_rgb: np.ndarray | None = None
    last_front_mask_rgb: np.ndarray | None = None
    last_front_second_rgb: np.ndarray | None = None
    record_steps: int = 0
    success: bool = False
    warned_no_camera: bool = False
    stop_requested: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record Door DP raw .npz episodes from the float-IK keyboard teleop scene. "
            "Recorder flags are parsed here; all remaining flags are forwarded to teleop.py."
        ),
        add_help=True,
    )
    parser.add_argument(
        "--dp_raw_root",
        type=str,
        default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "teleop_door_dp"),
        help="Output directory for raw Door DP .npz episodes.",
    )
    parser.add_argument("--dp_task", type=str, default="push lever door open")
    parser.add_argument("--dp_fps", type=int, default=50)
    parser.add_argument(
        "--camera_fps",
        type=float,
        default=25.0,
        help="Actual camera capture rate. State/action stay at --dp_fps; intermediate frames hold the last image.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Record RGB+mask instead of depth+mask. Default is depth+mask.",
    )
    parser.add_argument(
        "--record_key",
        type=str,
        default="KEY_C",
        help="Isaac Gym key name used to toggle recording. Default: KEY_C.",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default="push_door",
        help="Subtask name from the scripted DP phase list, or an integer id. Default: push_door.",
    )
    parser.add_argument(
        "--require_success",
        action="store_true",
        help="Only save the episode if the door reaches --pass_open_angle_deg before recording stops.",
    )
    parser.add_argument("--pass_open_angle_deg", type=float, default=75.0)
    parser.add_argument(
        "--allow_missing_front_camera",
        action="store_true",
        help="Allow missing front images and let RawDoorDPRecorder fill zeros. Default requires both cameras.",
    )
    parser.add_argument(
        "--start_recording_immediately",
        action="store_true",
        help="Start recording as soon as the sim loop begins. Useful for automated tests.",
    )
    parser.add_argument(
        "--stop_after_record_steps",
        type=int,
        default=0,
        help="Automatically save after this many recorded frames. 0 disables auto-stop.",
    )

    record_args, teleop_argv = parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + teleop_argv
        teleop_args = teleop.parse_args()
    finally:
        sys.argv = old_argv

    teleop_args.rgb = bool(record_args.rgb)
    teleop_args.sim_dt = 1.0 / float(record_args.dp_fps)
    return record_args, teleop_args


def subtask_id(name_or_id):
    value = str(name_or_id).strip()
    try:
        return int(value)
    except ValueError:
        pass
    if value not in push_door.DP_PHASE_ID:
        names = ", ".join(push_door.DP_PHASE_NAMES)
        raise ValueError(f"Unknown --subtask {value!r}; expected one of: {names}, or an integer id.")
    return int(push_door.DP_PHASE_ID[value])


def subscribe_recording_keyboard(gym, viewer, record_args):
    if viewer is None:
        return
    teleop.subscribe_if_available(gym, viewer, str(record_args.record_key), "toggle_recording")


def print_recording_controls(record_args):
    print("Recording controls:")
    print(f"  {record_args.record_key.replace('KEY_', '')}: start/stop raw Door DP recording")
    print("  Output root:", str(Path(record_args.dp_raw_root).expanduser().resolve()))
    print("  Vision:", "rgb+mask" if record_args.rgb else "depth+mask")
    stride = camera_sample_stride(record_args)
    print(f"  State/action rate: {int(record_args.dp_fps)} Hz")
    print(f"  Camera capture rate: {float(record_args.dp_fps) / float(stride):.2f} Hz")


def poll_viewer_events(gym, viewer, state, recording):
    if viewer is None:
        return
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action in teleop.CONTINUOUS_ACTIONS:
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
        elif evt.action == "toggle_recording" and evt.value > 0:
            recording.stop_requested = recording.active
            if not recording.active:
                recording.active = True


def make_recorder(record_args, teleop_args, door):
    vision_mode = normalize_vision_mode("rgb" if record_args.rgb else "depth")
    camera_stride = camera_sample_stride(record_args)
    return RawDoorDPRecorder(
        raw_root=record_args.dp_raw_root,
        fps=record_args.dp_fps,
        state_feature_names=list(dc.PI05_CURRENT_STATE10_NAMES),
        task=record_args.dp_task,
        vision_mode=vision_mode,
        metadata={
            "door_asset_index": int(teleop_args.door_index) if int(teleop_args.door_index) >= 0 else 0,
            "door_asset_name": door.spec.get("name", ""),
            "door_asset_path": door.spec.get("path", ""),
            "door_cfg": str(teleop_args.door_cfg),
            "source_script": Path(__file__).name,
            "door_dp_mode": "teleop",
            "controller_mode": "keyboard_teleop",
            "action_frame": "base",
            "action_pose_frame": "base",
            "target_pose_frame": "base",
            "ikpush_state_version": push_door.IKPUSH_STATE_VERSION,
            "state_format": TELEOP_STATE_MODE,
            "state_source": "current_vx_yaw_rate_ee_base_gripper",
            "state_normalized": False,
            "pi05_state_action_aligned": True,
            "camera_fps": float(record_args.dp_fps) / float(camera_stride),
            "camera_sample_stride": int(camera_stride),
            "camera_hold_last_frame": True,
        },
    )


def start_recording(recording, record_args, teleop_args, door, base_xy, yaw):
    recording.recorder = make_recorder(record_args, teleop_args, door)
    recording.prev_base_xy = np.asarray(base_xy, dtype=np.float32).copy()
    recording.prev_yaw = float(yaw)
    recording.last_dp_action = np.zeros(10, dtype=np.float32)
    recording.last_wrist_mask_rgb = None
    recording.last_wrist_second_rgb = None
    recording.last_front_mask_rgb = None
    recording.last_front_second_rgb = None
    recording.record_steps = 0
    recording.success = False
    recording.warned_no_camera = False
    recording.stop_requested = False
    recording.active = True
    print(
        f"Started teleop raw recording: root={record_args.dp_raw_root} "
        f"task={record_args.dp_task!r} state_mode={TELEOP_STATE_MODE} state_dim=10",
        flush=True,
    )


def finish_recording(recording, record_args, reason):
    if recording.recorder is None:
        recording.active = False
        recording.stop_requested = False
        return

    frames = int(recording.recorder.frame_count)
    should_save = frames > 0 and (recording.success or not record_args.require_success)
    if should_save:
        recording.recorder.save_episode()
        print(
            f"Finished teleop raw recording: saved=1 frames={frames} reason={reason}",
            flush=True,
        )
    elif frames == 0:
        print(
            f"Finished teleop raw recording: saved=0 frames=0 reason={reason}; no camera frames were captured.",
            flush=True,
        )
    else:
        print(
            f"Finished teleop raw recording: saved=0 frames={frames} reason=door did not reach "
            f"{record_args.pass_open_angle_deg} deg",
            flush=True,
        )
    recording.recorder.finalize()
    recording.active = False
    recording.recorder = None
    recording.prev_base_xy = None
    recording.prev_yaw = None
    recording.last_dp_action = None
    recording.last_wrist_mask_rgb = None
    recording.last_wrist_second_rgb = None
    recording.last_front_mask_rgb = None
    recording.last_front_second_rgb = None
    recording.record_steps = 0
    recording.success = False
    recording.warned_no_camera = False
    recording.stop_requested = False


def camera_sample_stride(record_args):
    if int(record_args.dp_fps) <= 0:
        raise ValueError("--dp_fps must be positive")
    if float(record_args.camera_fps) <= 0.0:
        raise ValueError("--camera_fps must be positive")
    return max(1, int(round(float(record_args.dp_fps) / float(record_args.camera_fps))))


def camera_images_for_record_frame(gym, sim, env, camera_handles, recording, record_args, teleop_args):
    stride = camera_sample_stride(record_args)
    should_capture = (
        recording.last_wrist_mask_rgb is None
        or recording.last_wrist_second_rgb is None
        or (recording.record_steps % stride) == 0
    )
    if should_capture:
        camera_images = push_door.capture_dp_camera_images(gym, sim, env, camera_handles, teleop_args)
        wrist_mask_rgb, wrist_second_rgb, front_mask_rgb, front_second_rgb = push_door.dp_image_inputs_from_cpu_cameras(
            camera_images,
            teleop_args,
        )
        missing_camera = wrist_mask_rgb is None or wrist_second_rgb is None
        if not record_args.allow_missing_front_camera:
            missing_camera = missing_camera or front_mask_rgb is None or front_second_rgb is None
        if missing_camera:
            return None, None, None, None
        recording.last_wrist_mask_rgb = np.asarray(wrist_mask_rgb, dtype=np.uint8).copy()
        recording.last_wrist_second_rgb = np.asarray(wrist_second_rgb, dtype=np.uint8).copy()
        recording.last_front_mask_rgb = None if front_mask_rgb is None else np.asarray(front_mask_rgb, dtype=np.uint8).copy()
        recording.last_front_second_rgb = (
            None if front_second_rgb is None else np.asarray(front_second_rgb, dtype=np.uint8).copy()
        )
    return (
        recording.last_wrist_mask_rgb,
        recording.last_wrist_second_rgb,
        recording.last_front_mask_rgb,
        recording.last_front_second_rgb,
    )


def record_one_frame(
    gym,
    sim,
    env,
    arm_actor,
    camera_handles,
    door,
    door_actor,
    dof_names,
    gripper_idx,
    ik_state,
    state,
    recording,
    record_args,
    teleop_args,
    dt,
    subtask_index,
):
    if recording.recorder is None:
        return

    ee_pos, ee_quat = push_door.current_ee_pose(gym, sim, ik_state)
    dof_pos_actual, dof_vel_actual = push_door.get_actor_dof_state(gym, env, arm_actor)
    door_pos, door_vel = push_door.get_actor_dof_state(gym, env, door_actor)
    gripper_actual = (
        float(dof_pos_actual[gripper_idx])
        if gripper_idx is not None and gripper_idx < len(dof_pos_actual)
        else float(state.gripper)
    )
    vx_cmd, yaw_rate_cmd = push_door.base_command_from_targets(
        state.base_xy,
        state.base_yaw,
        recording.prev_base_xy,
        recording.prev_yaw,
        dt,
    )
    target_quat = push_door.target_quat_for_dp(state.ee_target_quat, ik_state, ee_quat)
    wrist_mask_rgb, wrist_second_rgb, front_mask_rgb, front_second_rgb = camera_images_for_record_frame(
        gym,
        sim,
        env,
        camera_handles,
        recording,
        record_args,
        teleop_args,
    )
    missing_camera = wrist_mask_rgb is None or wrist_second_rgb is None
    if not record_args.allow_missing_front_camera:
        missing_camera = missing_camera or front_mask_rgb is None or front_second_rgb is None
    if missing_camera:
        if not recording.warned_no_camera:
            print(
                "Camera unavailable: skipped teleop DP frame because wrist/front mask/depth images are missing.",
                flush=True,
            )
            recording.warned_no_camera = True
        return

    dp_state = dc.make_float_dp_observation_state(
        dof_names,
        dof_pos_actual,
        dof_vel_actual,
        ee_pos,
        ee_quat,
        state.base_xy,
        state.base_z,
        state.base_yaw,
        yaw_rate_cmd,
        gripper_actual,
        recording.last_dp_action,
        vx=vx_cmd,
        state_mode=TELEOP_STATE_MODE,
    )
    dp_action = dc.make_float_dp_action(
        vx_cmd,
        yaw_rate_cmd,
        state.ee_target_pos,
        target_quat,
        state.gripper,
        state.base_xy,
        state.base_z,
        state.base_yaw,
    )
    recording.recorder.add_frame(
        dp_state,
        wrist_mask_rgb,
        wrist_second_rgb,
        dp_action,
        subtask_index,
        front_mask_rgb=front_mask_rgb,
        front_second_rgb=front_second_rgb,
        replay_snapshot=push_door.make_float_replay_snapshot(
            teleop_args,
            door,
            dof_names,
            dof_pos_actual,
            dof_vel_actual,
            door_pos,
            door_vel,
            ee_pos,
            ee_quat,
            state.base_xy,
            state.base_yaw,
            vx_cmd,
            yaw_rate_cmd,
        ),
    )
    recording.last_dp_action = dp_action.copy()
    recording.prev_base_xy = np.asarray(state.base_xy, dtype=np.float32).copy()
    recording.prev_yaw = float(state.base_yaw)
    recording.record_steps += 1

    if len(door_pos) > 0:
        signed_open_deg = math.degrees(float(teleop_args.door_motion_sign) * float(door_pos[0]))
        recording.success = recording.success or signed_open_deg >= float(record_args.pass_open_angle_deg)


def setup_viewer(gym, sim, teleop_args, record_args):
    viewer = teleop.setup_viewer(gym, sim, teleop_args)
    subscribe_recording_keyboard(gym, viewer, record_args)
    return viewer


def run_recording_teleop(
    gym,
    sim,
    env,
    arm_actor,
    actor_handles,
    door,
    door_actor,
    viewer,
    camera_handles,
    record_args,
    teleop_args,
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
            teleop_args.gripper_open,
            ik_state.lower[gripper_idx].item(),
            ik_state.upper[gripper_idx].item(),
        )
        gym.set_actor_dof_position_targets(env, arm_actor, dof_positions)

    ee_pos, ee_quat = teleop.get_current_ee_pose(gym, sim, ik_state)
    initial_quat = None if teleop_args.ik_position_only else ee_quat.copy()
    initial_dofs = np.asarray(defaults, dtype=np.float32).copy()
    if gripper_idx is not None:
        initial_dofs[gripper_idx] = dof_positions[gripper_idx]

    state = teleop.TeleopState(
        base_xy=np.asarray(initial_base_xy, dtype=np.float32).copy(),
        base_z=float(teleop_args.robot_z),
        base_yaw=float(teleop_args.robot_yaw),
        ee_target_pos=ee_pos.copy(),
        ee_target_quat=initial_quat,
        gripper=float(dof_positions[gripper_idx]) if gripper_idx is not None else float(teleop_args.gripper_open),
        initial_base_xy=np.asarray(initial_base_xy, dtype=np.float32).copy(),
        initial_base_yaw=float(teleop_args.robot_yaw),
        initial_ee_target_pos=ee_pos.copy(),
        initial_ee_target_quat=None if initial_quat is None else initial_quat.copy(),
        initial_dofs=initial_dofs,
    )
    push_door.set_ik_target(ik_state, state.ee_target_pos, state.ee_target_quat)

    recording = RecordingControl()
    subtask_index = subtask_id(record_args.subtask)
    teleop.print_controls(teleop_args)
    print_recording_controls(record_args)
    if viewer is None:
        print("Viewer is disabled; key-triggered recording needs a viewer unless --start_recording_immediately is set.")

    max_steps = int(teleop_args.steps) if teleop_args.steps > 0 else math.inf
    if viewer is None and max_steps == math.inf:
        max_steps = 1

    start = time.time()
    step = 0
    while step < max_steps:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        poll_viewer_events(gym, viewer, state, recording)
        if record_args.start_recording_immediately and step == 0 and not recording.active:
            recording.active = True
        if recording.active and recording.recorder is None:
            start_recording(recording, record_args, teleop_args, door, state.base_xy, state.base_yaw)
        if state.quit:
            break
        if state.reset_requested:
            if recording.active:
                finish_recording(recording, record_args, "reset")
            teleop.apply_reset(
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
                teleop_args,
            )

        if not state.paused:
            teleop.apply_base_controls(state, teleop_args, dt)
            teleop.apply_ee_controls(state, teleop_args, dt)
            push_door.set_robot_base_pose(gym, env, actor_handles, state.base_xy, state.base_z, state.base_yaw)
            push_door.set_ik_target(ik_state, state.ee_target_pos, state.ee_target_quat)
            push_door.update_arm_ik_targets(gym, sim, dof_positions, ik_state, teleop_args, num_arm_dofs)
            if gripper_idx is not None:
                dof_positions[gripper_idx] = np.clip(
                    state.gripper,
                    ik_state.lower[gripper_idx].item(),
                    ik_state.upper[gripper_idx].item(),
                )
            gym.set_actor_dof_position_targets(env, arm_actor, dof_positions)

            push_door.enforce_locked_door_hinge(gym, env, door_actor, door)
            door_pos, door_vel = push_door.get_actor_dof_state(gym, env, door_actor)
            door_efforts = push_door.compute_door_efforts(door, door_pos, door_vel, teleop_args)
            if len(door_efforts) > 0:
                gym.apply_actor_dof_efforts(env, door_actor, door_efforts)

            gym.simulate(sim)
            gym.fetch_results(sim, True)
        else:
            door_pos, _door_vel = push_door.get_actor_dof_state(gym, env, door_actor)

        need_camera_render = bool(camera_handles and (teleop_args.show_camera_images or recording.active))
        if viewer is not None and need_camera_render and (teleop_args.draw_ik_target or teleop_args.draw_camera_axes):
            gym.clear_lines(viewer)
        if viewer is not None or need_camera_render:
            gym.step_graphics(sim)
        if teleop_args.show_camera_images and camera_handles and step % max(1, teleop_args.camera_display_interval) == 0:
            push_door.show_camera_handle_images(gym, sim, env, camera_handles, teleop_args)

        if recording.active and not state.paused:
            record_one_frame(
                gym,
                sim,
                env,
                arm_actor,
                camera_handles,
                door,
                door_actor,
                dof_names,
                gripper_idx,
                ik_state,
                state,
                recording,
                record_args,
                teleop_args,
                dt,
                subtask_index,
            )

        if (
            recording.active
            and record_args.stop_after_record_steps > 0
            and recording.record_steps >= int(record_args.stop_after_record_steps)
        ):
            recording.stop_requested = True
        if recording.stop_requested:
            finish_recording(recording, record_args, "manual_stop")

        if viewer is not None:
            if teleop_args.draw_ik_target or teleop_args.draw_camera_axes:
                gym.clear_lines(viewer)
            if teleop_args.draw_camera_axes:
                push_door.draw_low_level_camera_axes(gym, viewer, env, arm_actor, actor_handles, teleop_args)
            if teleop_args.draw_ik_target:
                push_door.refresh_current_ee_pose(gym, sim, ik_state)
                base_ik.draw_ik_target(gym, viewer, env, ik_state)
                handle_goal = teleop.get_handle_goal(gym, env, door_actor, door)
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

        if teleop_args.log_interval > 0 and step % teleop_args.log_interval == 0:
            rec_status = "recording" if recording.active else "idle"
            print(
                f"[{step:04d}] teleop_record {rec_status} "
                f"frames={recording.record_steps} "
                f"base=({state.base_xy[0]:.2f},{state.base_xy[1]:.2f},{math.degrees(state.base_yaw):.1f}deg) "
                f"ee_target=({state.ee_target_pos[0]:.2f},{state.ee_target_pos[1]:.2f},{state.ee_target_pos[2]:.2f}) "
                f"ik_pos_err={ik_state.last_pos_error:.4f} "
                f"door={math.degrees(float(door_pos[0])) if len(door_pos) else 0.0:.1f}deg "
                f"handle={math.degrees(float(door_pos[1])) if len(door_pos) > 1 else 0.0:.1f}deg "
                f"open_stage={door.open_stage}",
                flush=True,
            )
        step += 1

    if recording.active:
        finish_recording(recording, record_args, "program_exit")
    print(f"Done after {step} steps ({time.time() - start:.2f}s).")


def main():
    record_args, teleop_args = parse_args()
    record_args.dp_raw_root = str(Path(record_args.dp_raw_root).expanduser().resolve())
    gym = gymapi.acquire_gym()
    sim, dt = base_ik.create_sim(gym, teleop_args)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    with tempfile.TemporaryDirectory(prefix="b1z1_float_ik_teleop_record_assets_") as temp_dir:
        base_asset, arm_asset = base_ik.load_robot_assets(gym, sim, teleop_args, Path(temp_dir))
        door = push_door.load_door_asset(gym, sim, teleop_args)
        if base_asset is not None:
            base_ik.print_collision_summary(gym, base_asset, "base visual actor", verbose=teleop_args.print_collision_summary)
        base_ik.print_collision_summary(gym, arm_asset, "arm articulated actor", verbose=teleop_args.print_collision_summary)
        dof_data = base_ik.configure_dofs(gym, arm_asset, teleop_args)
        dof_names, dof_props, dof_states, dof_positions, lower, upper, defaults, _speeds, _selected = dof_data
        if "jointGripper" in dof_names:
            dof_states["pos"][dof_names.index("jointGripper")] = teleop_args.gripper_open
            dof_positions[dof_names.index("jointGripper")] = teleop_args.gripper_open
        env, arm_actor, actor_handles, door_actor, initial_base_xy = push_door.create_env_actors(
            gym,
            sim,
            base_asset,
            arm_asset,
            door,
            dof_props,
            dof_states,
            teleop_args,
        )
        viewer = setup_viewer(gym, sim, teleop_args, record_args)
        camera_handles = {}
        if teleop_args.enable_wrist_camera or teleop_args.enable_front_camera:
            camera_handles = push_door.create_low_level_cameras(gym, env, arm_actor, actor_handles, teleop_args)
        ik_state = base_ik.setup_ik_controller(
            gym,
            sim,
            env,
            arm_actor,
            arm_asset,
            dof_names,
            lower,
            upper,
            teleop_args,
        )
        try:
            run_recording_teleop(
                gym,
                sim,
                env,
                arm_actor,
                actor_handles,
                door,
                door_actor,
                viewer,
                camera_handles,
                record_args,
                teleop_args,
                dt,
                dof_names,
                dof_states,
                dof_positions,
                defaults,
                ik_state,
                initial_base_xy,
            )
        finally:
            if teleop_args.show_camera_images and cv2 is not None:
                cv2.destroyAllWindows()
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
