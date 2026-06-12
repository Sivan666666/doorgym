#!/usr/bin/env python3
"""Minimal Isaac Gym loader for generated door assets."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = HIGH_LEVEL_ROOT.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import door_common as dc


gymapi = dc.gymapi
gymutil = dc.gymutil


def parse_args():
    return gymutil.parse_arguments(
        description="Load generated door assets and save a viewer screenshot.",
        headless=False,
        no_graphics=False,
        custom_parameters=[
            {"name": "--door_cfg", "type": str, "default": str(dc.DEFAULT_DOOR_CFG)},
            {"name": "--door_name", "type": str, "default": ""},
            {"name": "--door_index", "type": int, "default": -1},
            {"name": "--door_selection", "type": str, "default": "all"},
            {"name": "--door_prefer_name", "type": str, "default": ""},
            {"name": "--door_max_unique_assets", "type": int, "default": 0},
            {"name": "--num_envs", "type": int, "default": 0},
            {"name": "--steps", "type": int, "default": 120},
            {"name": "--screenshot_dir", "type": str, "default": str(HIGH_LEVEL_ROOT / "logs" / "door_asset_smoke")},
            {"name": "--door_actor_scale", "type": float, "default": 1.0},
            {"name": "--door_x", "type": float, "default": 0.0},
            {"name": "--door_y", "type": float, "default": 0.0},
            {"name": "--door_z_offset", "type": float, "default": 0.01},
            {"name": "--door_vhacd_resolution", "type": int, "default": 100000},
            {"name": "--handle_unlock_ratio", "type": float, "default": 40.0 / 45.0},
            {"name": "--door_motion_sign", "type": float, "default": -1.0},
            {"name": "--door_joint_friction", "type": float, "default": 0.0},
            {"name": "--door_joint_damping", "type": float, "default": 0.0},
            {"name": "--handle_joint_friction", "type": float, "default": 0.05},
            {"name": "--handle_joint_damping", "type": float, "default": 0.05},
            {"name": "--no_door_side_walls", "action": "store_true"},
        ],
    )


def np_list(values):
    return [float(v) for v in np.asarray(values, dtype=np.float32).reshape(-1)]


def main() -> None:
    args = parse_args()
    screenshot_dir = Path(args.screenshot_dir).expanduser()
    if not screenshot_dir.is_absolute():
        screenshot_dir = (REPO_ROOT / screenshot_dir).resolve()
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    gym = gymapi.acquire_gym()
    sim, _dt = dc.base_ik.create_sim(gym, args)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    viewer = None
    summary = []
    try:
        doors = dc.load_door_assets(gym, sim, args)
        env_count = len(doors) if int(args.num_envs) <= 0 else min(int(args.num_envs), len(doors))
        envs_per_row = int(math.ceil(math.sqrt(max(1, env_count))))
        envs = []
        for env_index, door_template in enumerate(doors[:env_count]):
            door = dc.clone_door_runtime(door_template)
            dc.apply_door_runtime_overrides(args, door)
            env = gym.create_env(
                sim,
                gymapi.Vec3(-2.0, -2.0, 0.0),
                gymapi.Vec3(2.0, 2.0, 2.5),
                envs_per_row,
            )
            door_pose = gymapi.Transform()
            actor_offset = door.actor_position_offset
            door_pose.p = gymapi.Vec3(
                float(args.door_x + actor_offset[0]),
                float(args.door_y + actor_offset[1]),
                float(-door.bounding["min"][2] * door.actor_scale + args.door_z_offset + actor_offset[2]),
            )
            door_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, float(door.actor_yaw))
            actor = gym.create_actor(env, door.asset, door_pose, f"door_{env_index}", env_index, 0, 1)
            if abs(door.actor_scale - 1.0) > 1.0e-6:
                gym.set_actor_scale(env, actor, door.actor_scale)
            dc.configure_door_actor_dofs(gym, env, actor, door, args)
            envs.append(env)
            summary.append(
                {
                    "env_index": env_index,
                    "asset_index": int(door.asset_index),
                    "name": door.spec.get("name", ""),
                    "path": door.spec.get("path", ""),
                    "variant": door.spec.get("generated_variant", ""),
                    "dof_names": list(door.dof_names),
                    "body_names": list(door.body_names),
                    "door_body_index": int(door.door_body_index),
                    "handle_body_index": int(door.handle_body_index),
                    "handle_goal_offset": np_list(door.handle_goal_offset),
                    "door_motion_sign": float(args.door_motion_sign),
                    "door_limits": {"lower": np_list(door.dof_lower), "upper": np_list(door.dof_upper)},
                }
            )

        if not bool(getattr(args, "headless", False)):
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError("Failed to create viewer.")
            gym.viewer_camera_look_at(
                viewer,
                None,
                gymapi.Vec3(3.0, -4.0, 2.0),
                gymapi.Vec3(0.0, 0.0, 0.9),
            )

        for step in range(max(1, int(args.steps))):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                if step == 5:
                    image_path = screenshot_dir / "door_assets_overview.png"
                    gym.write_viewer_image_to_file(viewer, str(image_path))
                    print(f"Saved screenshot: {image_path}")
                gym.sync_frame_time(sim)
                if gym.query_viewer_has_closed(viewer):
                    break
        summary_path = screenshot_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Loaded {len(summary)} door asset(s).")
        print(f"Wrote summary: {summary_path}")
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
