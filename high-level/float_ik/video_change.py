#!/usr/bin/env python3
"""Fixed-view, single-environment scene recorder for the paper video.

This is intentionally a thin, isolated extension of
``isaacgym_float_ik_b1z1_basearn_push_door_parallel_video.py``.  Each process
contains exactly one door/wall/robot scene.  A shell driver records multiple
doors with the same camera composition and joins the resulting clips.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

import isaacgym_float_ik_b1z1_basearn_push_door_parallel_video as video


def _parse_rgb(text: str) -> tuple[float, float, float]:
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 3 or not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("RGB must contain three comma-separated values in [0, 1]")
    return values


def _parse_change_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--change_scene_index", type=int, default=0)
    parser.add_argument("--change_robot_rgb", type=str, default="0.48,0.72,0.92")
    parser.add_argument("--change_floor_saturation", type=float, default=0.22)
    parser.add_argument("--change_floor_value", type=float, default=0.72)
    parser.add_argument("--change_floor_size", type=float, default=10.0)
    parser.add_argument("--change_wall_clearance", type=float, default=0.08)
    change_args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    change_args.robot_rgb = _parse_rgb(change_args.change_robot_rgb)
    return change_args


CHANGE = _parse_change_args()
_ORIGINAL_CREATE_ACTORS = video.create_parallel_env_actors
_FLOOR_ASSET = None
_WALL_ASSET_CACHE = {}
_TEXTURE_CACHE = {}
_MATERIAL_SURFACE_CACHE = {}


def _palette_rgb(index: int, saturation: float, value: float) -> tuple[float, float, float]:
    import colorsys

    hue = (0.09 + 0.17320508075688773 * float(index)) % 1.0
    return tuple(float(x) for x in colorsys.hsv_to_rgb(hue, saturation, value))


def _fixed_robot_colors(_args, _env_index):
    return {"robot": CHANGE.robot_rgb}


def _texture_buffer(base_rgb, style: str, seed: int, size: int = 256):
    """Create a subdued procedural RGBA material texture.

    Isaac Gym's legacy viewer does not expose a full metallic/roughness PBR
    stack.  Textured albedo plus varied directional/ambient light gives a much
    more material-like result than a flat rigid-body color while remaining
    reproducible and self-contained.
    """
    rng = np.random.default_rng(int(seed) % (2**32))
    base = np.asarray(base_rgb, dtype=np.float32).reshape(1, 1, 3)
    yy, xx = np.mgrid[0:size, 0:size]
    fine = rng.normal(0.0, 0.035, (size, size, 1)).astype(np.float32)
    broad = (
        0.025 * np.sin(xx[..., None] * (2.0 * math.pi / 73.0))
        + 0.018 * np.sin(yy[..., None] * (2.0 * math.pi / 109.0))
    ).astype(np.float32)
    image = np.clip(base * (1.0 + fine + broad), 0.0, 1.0)

    if style == "brick":
        row = yy // 48
        mortar = ((yy % 48) < 5) | (((xx + (row % 2) * 32) % 64) < 5)
        brick_tone = 0.92 + 0.10 * ((row + xx // 64) % 3)[..., None]
        image *= brick_tone
        image[mortar] *= 0.48
    elif style == "panel":
        seams = ((xx % 64) < 5) | ((yy % 128) < 5)
        image *= 0.94 + 0.08 * ((xx // 64) % 2)[..., None]
        image[seams] *= 0.48
    elif style == "plaster":
        mottling = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        for _ in range(4):
            mottling = (
                mottling
                + np.roll(mottling, 1, 0) + np.roll(mottling, -1, 0)
                + np.roll(mottling, 1, 1) + np.roll(mottling, -1, 1)
            ) / 5.0
        image *= np.clip(1.0 + 0.10 * mottling[..., None], 0.82, 1.18)
    elif style == "wood":
        grain = 0.16 * np.sin(yy[..., None] * (2.0 * math.pi / 21.0) + 0.8 * np.sin(xx[..., None] / 31.0))
        image *= 1.0 + grain
        seams = (yy % 64) < 5
        image[seams] *= 0.45
    elif style == "tile":
        grout = ((xx % 64) < 5) | ((yy % 64) < 5)
        image *= 0.95 + 0.07 * (((xx // 64) + (yy // 64)) % 2)[..., None]
        image[grout] *= 0.42
    elif style == "stone":
        veins = np.abs(np.sin((xx + 0.7 * yy) * (2.0 * math.pi / 91.0))) < 0.045
        image[veins] *= 0.68

    image = np.clip(image, 0.0, 1.0)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(image * 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba.reshape(size, size * 4))


def _material_texture(gym, sim, base_rgb, style: str, scene_index: int):
    key = (style, int(scene_index), tuple(round(float(x), 4) for x in base_rgb))
    if key not in _TEXTURE_CACHE:
        buffer = _texture_buffer(base_rgb, style, 9176 + scene_index * 7919)
        handle = gym.create_texture_from_buffer(sim, 256, 256, buffer)
        if handle == video.gymapi.INVALID_HANDLE:
            raise RuntimeError(f"Failed to create procedural {style} texture")
        _TEXTURE_CACHE[key] = handle
    return _TEXTURE_CACHE[key]


def _set_actor_texture(gym, env, actor, texture):
    count = int(gym.get_actor_rigid_body_count(env, actor))
    for body_index in range(count):
        gym.set_rigid_body_texture(
            env,
            actor,
            body_index,
            video.gymapi.MESH_VISUAL,
            texture,
        )


def _wall_asset(gym, sim, dimensions):
    key = tuple(round(float(x), 5) for x in dimensions)
    if key not in _WALL_ASSET_CACHE:
        options = video.gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        _WALL_ASSET_CACHE[key] = gym.create_box(sim, *dimensions, options)
    return _WALL_ASSET_CACHE[key]


def _material_surface_asset(gym, sim, kind, dimensions, base_rgb, style, scene_index):
    """Create a visual-only UV-mapped wall/floor surface.

    Isaac Gym primitive boxes do not reliably expose texture coordinates.  A
    thin visual mesh is therefore placed just above the physical box/plane;
    it has no collision geometry and cannot affect the scripted trajectory.
    """
    rounded_dimensions = tuple(round(float(x), 4) for x in dimensions)
    key = (kind, rounded_dimensions, style, int(scene_index), tuple(round(float(x), 4) for x in base_rgb))
    if key in _MATERIAL_SURFACE_CACHE:
        return _MATERIAL_SURFACE_CACHE[key]

    import cv2

    root = Path(tempfile.gettempdir()) / "video_change_uv_materials"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{kind}_{style}_{scene_index}_{'_'.join(str(x).replace('.', 'p') for x in rounded_dimensions)}"
    texture_name = f"{stem}.png"
    mtl_name = f"{stem}.mtl"
    obj_name = f"{stem}.obj"
    urdf_name = f"{stem}.urdf"

    rgba = _texture_buffer(base_rgb, style, 19273 + int(scene_index) * 104729, size=512).reshape(512, 512, 4)
    if not cv2.imwrite(str(root / texture_name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
        raise RuntimeError(f"Failed to write material texture: {root / texture_name}")
    (root / mtl_name).write_text(
        "newmtl surface\nKa 0.16 0.16 0.16\nKd 0.94 0.94 0.94\n"
        "Ks 0.10 0.10 0.10\nNs 18\nd 1.0\nillum 2\n"
        f"map_Kd {texture_name}\n",
        encoding="utf-8",
    )

    if kind == "wall_yz":
        width, height = rounded_dimensions
        vertices = [
            (0.0, -0.5 * width, -0.5 * height),
            (0.0, 0.5 * width, -0.5 * height),
            (0.0, 0.5 * width, 0.5 * height),
            (0.0, -0.5 * width, 0.5 * height),
        ]
        uv_max = (max(2.0, width / 0.75), max(2.0, height / 0.75))
    elif kind == "wall_xz":
        width, height = rounded_dimensions
        vertices = [
            (-0.5 * width, 0.0, -0.5 * height),
            (0.5 * width, 0.0, -0.5 * height),
            (0.5 * width, 0.0, 0.5 * height),
            (-0.5 * width, 0.0, 0.5 * height),
        ]
        uv_max = (max(2.0, width / 0.75), max(2.0, height / 0.75))
    elif kind == "floor_xy":
        width, depth = rounded_dimensions
        vertices = [
            (-0.5 * width, -0.5 * depth, 0.0),
            (0.5 * width, -0.5 * depth, 0.0),
            (0.5 * width, 0.5 * depth, 0.0),
            (-0.5 * width, 0.5 * depth, 0.0),
        ]
        uv_max = (max(2.0, width / 0.75), max(2.0, depth / 0.75))
    else:
        raise ValueError(f"Unsupported material surface kind: {kind}")

    vertex_lines = "\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    u_max, v_max = uv_max
    (root / obj_name).write_text(
        f"mtllib {mtl_name}\n{vertex_lines}\n"
        f"vt 0 0\nvt {u_max:.6f} 0\nvt {u_max:.6f} {v_max:.6f}\nvt 0 {v_max:.6f}\n"
        "usemtl surface\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
        "f 3/3 2/2 1/1\nf 4/4 3/3 1/1\n",
        encoding="utf-8",
    )
    (root / urdf_name).write_text(
        "<?xml version=\"1.0\"?>\n<robot name=\"video_material_surface\">\n"
        "  <link name=\"surface\">\n    <visual><geometry>"
        f"<mesh filename=\"{obj_name}\" scale=\"1 1 1\"/>"
        "</geometry></visual>\n  </link>\n</robot>\n",
        encoding="utf-8",
    )
    options = video.gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    options.use_mesh_materials = True
    asset = gym.load_asset(sim, str(root), urdf_name, options)
    _MATERIAL_SURFACE_CACHE[key] = asset
    return asset


def _create_nonoverlapping_side_walls(gym, sim, env, door, args, env_index):
    # Use the actual actor yaw and world-space bounds.  The previous generic
    # video helper applied a family-level +90-degree bbox override, which was
    # wrong for several custom doors and placed the wall inside the door panel.
    is_numeric = video.dc.is_partnet_numeric_door(door)
    bounds_yaw_offset = (
        video.dc.door_wall_bounds_yaw_offset(door) if is_numeric else 0.0
    )
    min_x, max_x, min_y, max_y = video.dc.door_world_xy_bounds_from_bbox(
        args, door, bounds_yaw_offset=bounds_yaw_offset
    )
    extent_x, extent_y = max_x - min_x, max_y - min_y
    # Original numeric PartNet meshes use a bbox convention rotated by 90 deg
    # relative to their rendered actor.  Custom/reconstructed doors do not.
    # Handle the convention only for true numeric assets, never by a broad
    # family label that also catches custom doors.
    width_axis = "y" if is_numeric else ("x" if extent_x >= extent_y else "y")
    clearance = max(0.0, float(CHANGE.change_wall_clearance))
    side_width = max(2.0, float(getattr(args, "door_wall_side_width", 6.0)))
    thickness = max(0.04, float(getattr(args, "door_wall_thickness", 0.08)))
    height = max(2.5, float(getattr(args, "door_wall_height", 4.0)))
    center_x, center_y = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    if width_axis == "x":
        dimensions = (side_width, thickness, height)
        centers = (
            (min_x - clearance - 0.5 * side_width, center_y),
            (max_x + clearance + 0.5 * side_width, center_y),
        )
        opening_min, opening_max = min_x, max_x
    else:
        dimensions = (thickness, side_width, height)
        centers = (
            (center_x, min_y - clearance - 0.5 * side_width),
            (center_x, max_y + clearance + 0.5 * side_width),
        )
        opening_min, opening_max = min_y, max_y

    wall_rgb = (
        float(getattr(args, "door_wall_color_r", 0.72)),
        float(getattr(args, "door_wall_color_g", 0.72)),
        float(getattr(args, "door_wall_color_b", 0.70)),
    )
    wall_styles = ("plaster", "brick", "panel", "plaster")
    wall_style = wall_styles[int(CHANGE.change_scene_index) % len(wall_styles)]
    texture = _material_texture(gym, sim, wall_rgb, wall_style, int(CHANGE.change_scene_index))
    asset = _wall_asset(gym, sim, dimensions)
    actors = []
    for side, (x, y) in zip(("left", "right"), centers):
        pose = video.gymapi.Transform()
        pose.p = video.gymapi.Vec3(float(x), float(y), 0.5 * height)
        actor = gym.create_actor(
            env, asset, pose, f"video_change_wall_{env_index}_{side}", int(env_index), 0, 0
        )
        video.set_actor_visual_color(gym, env, actor, (1.0, 1.0, 1.0))
        _set_actor_texture(gym, env, actor, texture)
        actors.append(actor)

        surface_kind = "wall_yz" if width_axis == "y" else "wall_xz"
        surface_dimensions = (side_width, height)
        surface_asset = _material_surface_asset(
            gym, sim, surface_kind, surface_dimensions, wall_rgb, wall_style,
            int(CHANGE.change_scene_index),
        )
        surface_pose = video.gymapi.Transform()
        surface_pose.p = video.gymapi.Vec3(
            float(x + (0.5 * thickness + 0.003 if width_axis == "y" else 0.0)),
            float(y + (0.5 * thickness + 0.003 if width_axis == "x" else 0.0)),
            0.5 * height,
        )
        surface_actor = gym.create_actor(
            env, surface_asset, surface_pose,
            f"video_change_wall_material_{env_index}_{side}", int(env_index), 0, 0,
        )
        actors.append(surface_actor)

    layout = {
        "width_axis": width_axis,
        "door_world_bounds_xy": [min_x, max_x, min_y, max_y],
        "opening_bounds": [opening_min, opening_max],
        "clearance_m": clearance,
        "wall_centers_xy": [list(center) for center in centers],
        "wall_dimensions": list(dimensions),
        "wall_style": wall_style,
        "bounds_yaw_offset_rad": float(bounds_yaw_offset),
        "partnet_numeric_bbox_convention": bool(is_numeric),
    }
    args.video_change_wall_layout = layout
    print(
        f"video_change_wall door={door.spec.get('name')} width_axis={width_axis} "
        f"bounds=({opening_min:.3f},{opening_max:.3f}) clearance={clearance:.3f} "
        f"centers={centers}",
        flush=True,
    )
    return actors


def _create_actors_with_material_scene(
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
    global _FLOOR_ASSET
    previous_no_walls = bool(getattr(args, "no_door_side_walls", False))
    args.no_door_side_walls = True
    try:
        result = _ORIGINAL_CREATE_ACTORS(
            gym, sim, base_asset, arm_asset, door, dof_props, dof_states,
            args, env_index, envs_per_row,
        )
    finally:
        args.no_door_side_walls = previous_no_walls
    env = result[0]
    _create_nonoverlapping_side_walls(gym, sim, env, door, args, env_index)
    size = max(4.0, float(CHANGE.change_floor_size))
    if _FLOOR_ASSET is None:
        options = video.gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        _FLOOR_ASSET = gym.create_box(sim, size, size, 0.03, options)
    pose = video.gymapi.Transform()
    # The colored top is 1 mm above the default plane.  This avoids z-fighting
    # while changing the support height by a physically negligible amount.
    pose.p = video.gymapi.Vec3(float(args.door_x), float(args.door_y), -0.014)
    actor = gym.create_actor(
        env,
        _FLOOR_ASSET,
        pose,
        f"video_change_floor_{env_index}",
        int(env_index),
        0,
        0,
    )
    floor_rgb = _palette_rgb(
        CHANGE.change_scene_index + 3,
        max(0.0, min(1.0, float(CHANGE.change_floor_saturation))),
        max(0.0, min(1.0, float(CHANGE.change_floor_value))),
    )
    floor_styles = ("wood", "tile", "stone", "tile", "wood")
    floor_style = floor_styles[int(CHANGE.change_scene_index) % len(floor_styles)]
    floor_texture = _material_texture(
        gym, sim, floor_rgb, floor_style, int(CHANGE.change_scene_index) + 100
    )
    video.set_actor_visual_color(gym, env, actor, (1.0, 1.0, 1.0))
    _set_actor_texture(gym, env, actor, floor_texture)
    floor_surface_asset = _material_surface_asset(
        gym, sim, "floor_xy", (size, size), floor_rgb, floor_style,
        int(CHANGE.change_scene_index) + 100,
    )
    floor_surface_pose = video.gymapi.Transform()
    floor_surface_pose.p = video.gymapi.Vec3(float(args.door_x), float(args.door_y), 0.003)
    gym.create_actor(
        env, floor_surface_asset, floor_surface_pose,
        f"video_change_floor_material_{env_index}", int(env_index), 0, 0,
    )
    args.video_change_floor_rgb = floor_rgb
    args.video_change_floor_style = floor_style
    return result


class FixedSceneVideoRecorder(video.OverviewVideoRecorder):
    """Record one isolated scene using a camera fixed relative to env0."""

    def __init__(self, gym, sim, viewer, env_states, args):
        if len(env_states) != 1:
            raise ValueError(
                "video_change.py requires --num_envs 1 so no second door/robot can enter the frame"
            )
        super().__init__(gym, sim, viewer, env_states, args)
        self.light = self._set_scene_light()

    def _set_scene_light(self):
        idx = int(CHANGE.change_scene_index)
        warm = 0.06 * math.sin(0.9 * idx)
        intensity = video.gymapi.Vec3(0.92 + warm, 0.91, 0.92 - warm)
        ambient_level = 0.34 + 0.05 * ((idx % 3) - 1)
        ambient = video.gymapi.Vec3(ambient_level, ambient_level, ambient_level)
        angle = 0.55 + idx * 0.47
        direction = video.gymapi.Vec3(
            float(0.55 * math.cos(angle)),
            float(0.55 * math.sin(angle)),
            -1.0,
        )
        self.gym.set_light_parameters(self.sim, 0, intensity, ambient, direction)
        return {
            "intensity": [float(intensity.x), float(intensity.y), float(intensity.z)],
            "ambient": [float(ambient.x), float(ambient.y), float(ambient.z)],
            "direction": [float(direction.x), float(direction.y), float(direction.z)],
        }

    def update_camera(self, _step):
        if not self.enabled or self.anchors is None:
            return
        # Never interpolate or orbit.  Every scene uses the same local camera
        # offset and target, so only the environment appearance changes.
        self.gym.viewer_camera_look_at(
            self.viewer,
            None,
            video.np_to_vec3(self.anchors["start_pos"]),
            video.np_to_vec3(self.anchors["start_target"]),
        )

    def close(self):
        path = self.path
        state = self.env_states[0] if self.env_states else None
        metadata = {
            "scene_index": int(CHANGE.change_scene_index),
            "door_name": str(state.door.spec.get("name", "")) if state is not None else "",
            "camera_mode": "fixed_relative_to_single_env",
            "num_envs": len(self.env_states),
            "robot_rgb": [float(x) for x in CHANGE.robot_rgb],
            "wall_rgb": [
                float(getattr(state.args, "door_wall_color_r", 0.0)),
                float(getattr(state.args, "door_wall_color_g", 0.0)),
                float(getattr(state.args, "door_wall_color_b", 0.0)),
            ] if state is not None else [],
            "floor_rgb": [float(x) for x in getattr(state.args, "video_change_floor_rgb", ())]
            if state is not None else [],
            "floor_style": str(getattr(state.args, "video_change_floor_style", ""))
            if state is not None else "",
            "wall_layout": getattr(state.args, "video_change_wall_layout", {})
            if state is not None else {},
            "light": self.light,
            "frame_count": int(self.frame_count),
            "capture_start_step": int(getattr(state.args, "video_capture_start_step", 0))
            if state is not None else 0,
            "capture_stride": int(getattr(state.args, "video_capture_stride", 1))
            if state is not None else 1,
        }
        super().close()
        if path:
            manifest = Path(path).with_suffix(".scene.json")
            manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Scene manifest saved: {manifest}", flush=True)


video.sample_video_asset_colors = _fixed_robot_colors
video.create_parallel_env_actors = _create_actors_with_material_scene
video.OverviewVideoRecorder = FixedSceneVideoRecorder


if __name__ == "__main__":
    video.main()
