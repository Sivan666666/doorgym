#!/usr/bin/env python3
"""Cinematic many-environment A2W+Z1 door-opening video recorder.

This is an isolated extension of the existing parallel overview recorder.  It
keeps the current scripted A2W trajectory, gives every environment its own
flush wall/floor material, colors the complete robot uniformly, and flies the
viewer camera along a smooth path while all environments run concurrently.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import sys
from pathlib import Path

import numpy as np

import isaacgym_float_ik_b1z1_basearn_push_door_parallel_video as video


# Save the original moving-recorder class before video_change installs its
# fixed, single-environment recorder.  We reuse video_change only for its
# validated non-overlapping walls and UV-mapped material surfaces.
_ORIGINAL_OVERVIEW_RECORDER = video.OverviewVideoRecorder
_ORIGINAL_TRAJECTORY_TARGETS = video.trajectory_targets
import video_change as material_video  # noqa: E402


def _parse_rgb(text: str) -> tuple[float, float, float]:
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 3 or not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("RGB must contain three comma-separated values in [0, 1]")
    return values


def _parse_array_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--array_robot_rgb", type=str, default="0.06,0.20,0.48")
    parser.add_argument("--array_material_variants", type=int, default=12)
    parser.add_argument("--array_floor_size", type=float, default=5.0)
    parser.add_argument("--array_wall_clearance", type=float, default=0.0)
    parser.add_argument("--array_camera_total_steps", type=int, default=1600)
    parser.add_argument("--array_camera_end_height", type=float, default=28.0)
    array_args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    array_args.robot_rgb = _parse_rgb(array_args.array_robot_rgb)
    if array_args.array_material_variants <= 0:
        raise ValueError("--array_material_variants must be positive")
    return array_args


ARRAY = _parse_array_args()

# Configure video_change's material helpers for a tiled multi-env scene.
material_video.CHANGE.robot_rgb = ARRAY.robot_rgb
material_video.CHANGE.change_floor_size = float(ARRAY.array_floor_size)
material_video.CHANGE.change_wall_clearance = float(ARRAY.array_wall_clearance)
material_video.CHANGE.change_floor_saturation = 0.26
material_video.CHANGE.change_floor_value = 0.70


def _fixed_robot_colors(_args, _env_index):
    """Use one deep-blue color for the full A2W base and Z1 arm."""
    return {"robot": ARRAY.robot_rgb}


def _create_array_material_scene(
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
    # Limit unique material assets while retaining visible wall/floor variety.
    # Environment creation is sequential, so this temporary selector is safe.
    previous_index = int(material_video.CHANGE.change_scene_index)
    previous_disable_self_collisions = bool(getattr(args, "disable_self_collisions", False))
    previous_spec = door.spec
    # The diverse legacy door pool contains several collision meshes that can
    # crash PhysX GPU narrowphase.  This paper-video-only mode filters robot ↔
    # door contact; motion remains the current scripted trajectory and the door
    # articulation is driven below for deterministic visualization.
    args.disable_self_collisions = True
    door.spec = dict(door.spec)
    door.spec["door_actor_collision_filter"] = 1
    material_video.CHANGE.change_scene_index = int(env_index) % int(ARRAY.array_material_variants)
    try:
        result = material_video._create_actors_with_material_scene(
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
        )
        env, _arm_actor, _actor_handles, door_actor, _robot_xy = result
        properties = gym.get_actor_dof_properties(env, door_actor)
        if len(properties):
            properties["driveMode"][:] = video.gymapi.DOF_MODE_POS
            properties["stiffness"][:] = 6000.0
            properties["damping"][:] = 300.0
            gym.set_actor_dof_properties(env, door_actor, properties)
        return result
    finally:
        material_video.CHANGE.change_scene_index = previous_index
        args.disable_self_collisions = previous_disable_self_collisions
        # Preserve the copied per-video collision setting on this cloned door.
        # Restoring the original mapping would also remove layout normalization
        # metadata that is intentionally local to this runtime clone.
        if door.spec is previous_spec:
            door.spec = dict(previous_spec)


def _drive_cinematic_door(step, args, door, gym, env, door_actor):
    """Drive handle and hinge smoothly for the isolated visualization mode."""
    states = gym.get_actor_dof_states(env, door_actor, video.gymapi.STATE_ALL)
    if len(states) == 0:
        return

    walk_end = int(args.walk_steps)
    initial_end = walk_end + int(args.initial_hold_steps)
    grasp_end = initial_end + int(args.grasp_steps)
    grasp_hold_end = grasp_end + int(args.grasp_hold_steps)
    close_end = grasp_hold_end + int(args.gripper_close_steps)
    rotate_end = close_end + int(args.handle_rotate_steps)
    push_end = rotate_end + int(args.door_push_steps)

    hinge_index = int(door.door_dof_index)
    closed = float(video.dc.closed_hinge_angle(door, args))
    if float(args.door_motion_sign) < 0.0:
        opened = float(door.dof_lower[hinge_index])
    else:
        opened = float(door.dof_upper[hinge_index])
    push_alpha = video.smoothstep(
        np.clip((float(step) - float(rotate_end)) / max(1.0, float(push_end - rotate_end)), 0.0, 1.0)
    )
    states["pos"][hinge_index] = closed + (opened - closed) * push_alpha
    states["vel"][hinge_index] = 0.0

    handle_index = door.handle_dof_index
    if handle_index is not None:
        rest = (
            float(door.handle_rest_angle)
            if door.handle_rest_angle is not None
            else float(door.dof_lower[handle_index])
        )
        direction = -1.0 if float(door.handle_unlock_direction_sign) < 0.0 else 1.0
        rotate_alpha = video.smoothstep(
            np.clip((float(step) - float(close_end)) / max(1.0, float(rotate_end - close_end)), 0.0, 1.0)
        )
        handle_target = rest + direction * max(float(door.handle_unlock_threshold), math.radians(40.0))
        handle_target = float(np.clip(handle_target, door.dof_lower[handle_index], door.dof_upper[handle_index]))
        states["pos"][handle_index] = rest + (handle_target - rest) * rotate_alpha
        states["vel"][handle_index] = 0.0

    if step >= rotate_end:
        door.open_stage = True
    gym.set_actor_dof_states(env, door_actor, states, video.gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, door_actor, np.asarray(states["pos"], dtype=np.float32))


def _cinematic_trajectory_targets(
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
    _drive_cinematic_door(step, args, door, gym, env, door_actor)
    return _ORIGINAL_TRAJECTORY_TARGETS(
        step, args, door, gym, env, door_actor, ik_state,
        base_start, base_stop, base_push, yaw_start, yaw_push, traj,
    )


def _zero_door_efforts(_door, dof_pos, _dof_vel, _args):
    return np.zeros(len(dof_pos), dtype=np.float32)


def _no_locked_hinge_enforcement(_gym, _env, _door_actor, _door, _args):
    return None


def _bezier(points: tuple[np.ndarray, ...], t: float) -> np.ndarray:
    """Evaluate a cubic Bezier curve."""
    omt = 1.0 - t
    return (
        omt**3 * points[0]
        + 3.0 * omt**2 * t * points[1]
        + 3.0 * omt * t**2 * points[2]
        + t**3 * points[3]
    )


class ArrayFlythroughVideoRecorder(_ORIGINAL_OVERVIEW_RECORDER):
    """Continuously fly from env0 close-up to a broad array overview."""

    def __init__(self, gym, sim, viewer, env_states, args):
        super().__init__(gym, sim, viewer, env_states, args)
        self.camera_total_steps = max(1, int(ARRAY.array_camera_total_steps))
        self.camera_curves = self._compute_flythrough_curves() if self.enabled else None
        self.light = self._set_scene_light() if self.enabled else None

    def _focuses(self) -> np.ndarray:
        return np.stack(
            [self._env_door_focus_world(state, z=0.92) for state in self.env_states],
            axis=0,
        ).astype(np.float32)

    def _compute_flythrough_curves(self):
        focuses = self._focuses()
        low = focuses.min(axis=0)
        high = focuses.max(axis=0)
        span = np.maximum(high - low, np.array([5.0, 5.0, 0.0], dtype=np.float32))
        center = 0.5 * (low + high)
        first = self.env_states[0]
        first_origin = video.vec3_to_np(self.gym.get_env_origin(first.env))

        p0 = first_origin + np.array(
            [float(first.args.robot_x) + 1.20, float(first.args.robot_y) + 1.15, 1.72],
            dtype=np.float32,
        )
        p1 = np.array(
            [low[0] + 0.10 * span[0], low[1] - 0.18 * span[1], 5.5],
            dtype=np.float32,
        )
        p2 = np.array(
            [low[0] + 0.78 * span[0], low[1] - 0.22 * span[1], 14.0],
            dtype=np.float32,
        )
        p3 = np.array(
            [high[0] + 0.16 * span[0], high[1] + 0.16 * span[1], float(ARRAY.array_camera_end_height)],
            dtype=np.float32,
        )

        q0 = focuses[0].copy()
        q1 = np.array(
            [low[0] + 0.16 * span[0], low[1] + 0.12 * span[1], 0.95],
            dtype=np.float32,
        )
        q2 = np.array(
            [low[0] + 0.67 * span[0], low[1] + 0.48 * span[1], 1.08],
            dtype=np.float32,
        )
        q3 = center.copy()
        q3[2] = 1.10

        curves = {"position": (p0, p1, p2, p3), "target": (q0, q1, q2, q3)}
        print(
            "Array flythrough camera: "
            f"steps={self.camera_total_steps} "
            f"p0={p0.round(2).tolist()} p3={p3.round(2).tolist()} "
            f"target0={q0.round(2).tolist()} target3={q3.round(2).tolist()}",
            flush=True,
        )
        return curves

    def _set_scene_light(self):
        intensity = video.gymapi.Vec3(1.00, 0.98, 0.94)
        ambient = video.gymapi.Vec3(0.42, 0.43, 0.46)
        direction = video.gymapi.Vec3(-0.48, 0.30, -1.0)
        self.gym.set_light_parameters(self.sim, 0, intensity, ambient, direction)
        return {
            "intensity": [float(intensity.x), float(intensity.y), float(intensity.z)],
            "ambient": [float(ambient.x), float(ambient.y), float(ambient.z)],
            "direction": [float(direction.x), float(direction.y), float(direction.z)],
        }

    def update_camera(self, step):
        if not self.enabled or self.camera_curves is None:
            return
        raw_t = np.clip(float(step) / float(self.camera_total_steps), 0.0, 1.0)
        # Smooth acceleration/deceleration while retaining one continuous path.
        t = float(raw_t * raw_t * (3.0 - 2.0 * raw_t))
        position = _bezier(self.camera_curves["position"], t)
        target = _bezier(self.camera_curves["target"], t)
        self.gym.viewer_camera_look_at(
            self.viewer,
            None,
            video.np_to_vec3(position),
            video.np_to_vec3(target),
        )

    def close(self):
        path = self.path
        final_angles_by_door = defaultdict(list)
        for state in self.env_states:
            angle_deg = None
            if state.last_door_pos is not None and len(state.last_door_pos):
                hinge_index = int(state.door.door_dof_index)
                hinge_angle = float(state.last_door_pos[hinge_index])
                closed_angle = float(video.dc.closed_hinge_angle(state.door, state.args))
                # Report travel relative to the configured closed pose.  Wrap
                # legacy continuous-joint coordinates to their shortest angle.
                travel = math.atan2(
                    math.sin(hinge_angle - closed_angle),
                    math.cos(hinge_angle - closed_angle),
                )
                angle_deg = abs(math.degrees(travel))
            if angle_deg is not None:
                final_angles_by_door[str(state.door.spec.get("name", ""))].append(angle_deg)
        final_angle_summary = {
            name: {
                "count": len(values),
                "min_deg": float(min(values)),
                "mean_deg": float(np.mean(values)),
                "max_deg": float(max(values)),
            }
            for name, values in sorted(final_angles_by_door.items())
        }
        metadata = {
            "camera_mode": "continuous_cubic_bezier_flythrough",
            "camera_total_steps": int(self.camera_total_steps),
            "num_envs": len(self.env_states),
            "robot_rgb": [float(value) for value in ARRAY.robot_rgb],
            "material_variants": int(ARRAY.array_material_variants),
            "floor_size": float(ARRAY.array_floor_size),
            "wall_clearance": float(ARRAY.array_wall_clearance),
            "frame_count": int(self.frame_count),
            "doors": sorted({str(state.door.spec.get("name", "")) for state in self.env_states}),
            "unique_door_count": len(final_angles_by_door),
            "final_abs_door_angle_summary": final_angle_summary,
            "final_open_ge_80_count": int(sum(
                value >= 80.0 for values in final_angles_by_door.values() for value in values
            )),
            "light": self.light,
        }
        super().close()
        if path:
            manifest = Path(path).with_suffix(".scene.json")
            manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Array scene manifest saved: {manifest}", flush=True)


# Install the isolated behavior and delegate the remaining simulation setup to
# the mature parallel video script.
video.sample_video_asset_colors = _fixed_robot_colors
video.create_parallel_env_actors = _create_array_material_scene
video.OverviewVideoRecorder = ArrayFlythroughVideoRecorder
video.trajectory_targets = _cinematic_trajectory_targets
video.compute_door_efforts = _zero_door_efforts
video.dc.enforce_locked_door_hinge = _no_locked_hinge_enforcement


if __name__ == "__main__":
    video.main()
