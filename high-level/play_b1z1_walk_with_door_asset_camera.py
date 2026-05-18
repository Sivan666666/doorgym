import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

try:
    import cv2  # noqa: E402
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = ROOT / "high-level"
LOW_LEVEL_ROOT = ROOT / "low-level"
if str(HIGH_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_ROOT))
if str(LOW_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOW_LEVEL_ROOT))

from isaacgym import gymapi, gymtorch, gymutil  # noqa: E402
import isaacgym  # noqa: F401,E402
from isaacgym.torch_utils import *  # noqa: F401,F403,E402
import torch  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.envs.manip_loco.manip_loco import ManipLoco  # noqa: E402
from legged_gym.utils import task_registry  # noqa: E402

try:
    from dp.door_dp_common import (
        DoorDPJsonlLogger,
        DoorDPPolicyController,
        RawDoorDPRecorder,
        apply_door_dp_action,
        dp_image_inputs_from_camera_tensors,
        get_door_dp_action,
        get_door_dp_state,
        images_from_camera_tensors,
        make_door_dp_log_record,
        make_door_dp_replay_snapshot,
        make_state_feature_names,
        print_door_dp_log_record,
    )
except Exception:
    DoorDPJsonlLogger = None
    DoorDPPolicyController = None
    RawDoorDPRecorder = None
    apply_door_dp_action = None
    dp_image_inputs_from_camera_tensors = None
    get_door_dp_action = None
    get_door_dp_state = None
    images_from_camera_tensors = None
    make_door_dp_log_record = None
    make_door_dp_replay_snapshot = None
    make_state_feature_names = None
    print_door_dp_log_record = None


DOOR_RUNTIME = {}
DEFAULT_DOOR_ASSET_NAMES = (
    "99650089960001",
    "99650089960006",
    "99655039960001",
    "99655039960006",
)


class PositionErrorPlotter:
    def __init__(self, env_id=0, window=600):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.env_id = int(env_id)
        self.window = int(window)
        self.steps = []
        self.series = {
            "pos_norm": [],
            "pos_x": [],
            "pos_y": [],
            "pos_z": [],
            "orn_norm": [],
        }
        self.plt.ion()
        self.fig, axes = self.plt.subplots(5, 1, sharex=True, num="Scripted EE Tracking Error", figsize=(8, 9))
        self.axes = list(np.asarray(axes).reshape(-1))
        specs = [
            ("pos_norm", "Position error norm", "m", "{:.4f}"),
            ("pos_x", "Position error x", "m", "{:.4f}"),
            ("pos_y", "Position error y", "m", "{:.4f}"),
            ("pos_z", "Position error z", "m", "{:.4f}"),
            ("orn_norm", "Orientation error norm", "deg", "{:.2f}"),
        ]
        self.title_specs = {}
        self.lines = {}
        for ax, (name, title, ylabel, fmt) in zip(self.axes, specs):
            (line,) = ax.plot([], [], label=name)
            self.lines[name] = line
            self.title_specs[name] = (ax, title, ylabel, fmt)
            ax.set_title(f"{title}: -- {ylabel}")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend(loc="upper right")
        self.axes[-1].set_xlabel("policy step")
        self.fig.tight_layout()

    def update(self, step, target_pos, ee_pos, target_quat, ee_quat):
        env_id = int(np.clip(self.env_id, 0, target_pos.shape[0] - 1))
        pos_err = (target_pos[env_id] - ee_pos[env_id]).detach().cpu().numpy()
        quat_target = target_quat[env_id : env_id + 1]
        quat_current = ee_quat[env_id : env_id + 1]
        orn_err = orientation_error(
            quat_target,
            quat_current / torch.clamp(torch.norm(quat_current, dim=-1, keepdim=True), min=1e-6),
        )[0]
        self.steps.append(int(step))
        self.series["pos_norm"].append(float(np.linalg.norm(pos_err)))
        self.series["pos_x"].append(float(pos_err[0]))
        self.series["pos_y"].append(float(pos_err[1]))
        self.series["pos_z"].append(float(pos_err[2]))
        self.series["orn_norm"].append(float(torch.rad2deg(torch.norm(orn_err)).detach().cpu().item()))
        if len(self.steps) > self.window:
            self.steps = self.steps[-self.window :]
            for name in self.series:
                self.series[name] = self.series[name][-self.window :]
        for name, line in self.lines.items():
            line.set_data(self.steps, self.series[name])
            ax, title, ylabel, fmt = self.title_specs[name]
            ax.set_title(f"{title}: {fmt.format(self.series[name][-1])} {ylabel}")
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


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


def _sorted_asset_items(asset_dict):
    def _key(item):
        key = item[0]
        return (0, int(key)) if key.isdigit() else (1, key)

    return [item[1] for item in sorted(asset_dict.items(), key=_key)]


def _load_door_runtime(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    asset_cfg = cfg["env"]["asset"]
    train_assets = asset_cfg["trainAssets"]
    load_block = asset_cfg.get("load_block") or next(iter(train_assets.keys()))
    door_asset_specs = _sorted_asset_items(train_assets[load_block])

    asset_root = HIGH_LEVEL_ROOT / asset_cfg["assetRoot"]
    asset_file_door = asset_cfg["assetFileDoor"]
    door_set_root = asset_root / asset_file_door

    door_bounding_data = []
    handle_bounding_data = []
    for spec in door_asset_specs:
        with open(door_set_root / spec["bounding_box"], "r", encoding="utf-8") as f:
            door_bounding_data.append(json.load(f))
        with open(door_set_root / spec["handle_bounding"], "r", encoding="utf-8") as f:
            handle_bounding_data.append(json.load(f))

    env_cfg = cfg.get("env", {})
    sensor_cfg = cfg.get("sensor", {})
    return {
        "asset_root": str(asset_root),
        "asset_file_door": asset_file_door,
        "door_asset_specs": door_asset_specs,
        "door_asset_names": [item["name"] for item in door_asset_specs],
        "door_bounding_data": door_bounding_data,
        "handle_bounding_data": handle_bounding_data,
        "door_lock_force": float(env_cfg.get("doorLockForce", 150.0)),
        "door_open_resistance": float(env_cfg.get("doorOpenResistance", 3.0)),
        "door_open_damping": float(env_cfg.get("doorOpenDamping", 0.5)),
        "handle_unlock_ratio": float(env_cfg.get("handleOpenThresholdRatio", 0.65)),
        "handle_spring_stiffness": float(env_cfg.get("handleSpringStiffness", 40.0)),
        "handle_spring_damping": float(env_cfg.get("handleSpringDamping", 2.0)),
        "door_joint_friction": env_cfg.get("doorJointFriction", [6.0, 18.0]),
        "door_joint_damping": env_cfg.get("doorJointDamping", [3.0, 10.0]),
        "door_joint_effort": env_cfg.get("doorJointEffort", [200.0, 200.0]),
        "wrist_camera_cfg": sensor_cfg.get(
            "wrist_camera",
            {
                "horizontal_fov": 69,
                "resolution": [96, 54],
                "position": [0.0955, 0.22, -0.03175],
                "rotation": [-1.57, 0.0, -0.87],
                "rand_position": [0.0, 0.0, 0.0],
            },
        ),
        "front_camera_cfg": sensor_cfg.get(
            "onboard_camera",
            {
                "horizontal_fov": 69,
                "resolution": [96, 54],
                "position": [0.425, 0.04, 0.12],
                "rotation": [0.0, 0.0, 0.0],
                "rand_position": [0.0, 0.0, 0.0],
            },
        ),
    }


def _compute_robot_y_by_spec(robot_y, door_y, door_bounding_data, handle_bounding_data, door_actor_scale=1.0):
    robot_y_by_spec = []
    for handle_bounds in handle_bounding_data:
        handle_center_y = 0.5 * (float(handle_bounds["handle_min"][1]) + float(handle_bounds["handle_max"][1]))
        # Door actors are rotated 180 deg around Z, so asset-local handle Y contributes with a flipped sign in world Y.
        handle_center_world_y = door_y - door_actor_scale * handle_center_y
        robot_y_by_spec.append(float(robot_y + handle_center_world_y))
    return robot_y_by_spec


_LOG_COLORS = {
    "reset": "\033[0m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "magenta": "\033[95m",
    "gray": "\033[90m",
}


def _c(text, color):
    return f"{_LOG_COLORS[color]}{text}{_LOG_COLORS['reset']}"


def _round_for_log(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_round_for_log(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_round_for_log(v) for v in value)
    return value


def _door_dimension_summary(door_asset_names, door_bounding_data, door_actor_scale):
    dims = []
    for name, bounds in zip(door_asset_names, door_bounding_data):
        extents = [float(bounds["max"][i]) - float(bounds["min"][i]) for i in range(3)]
        dims.append(
            {
                "name": name,
                "width_m": round(extents[0] * door_actor_scale, 4),
                "thickness_m": round(extents[1] * door_actor_scale, 4),
                "height_m": round(extents[2] * door_actor_scale, 4),
            }
        )
    return dims


def _filter_door_runtime_by_names(runtime, names):
    selected = []
    available = {spec["name"]: i for i, spec in enumerate(runtime["door_asset_specs"])}
    missing = [name for name in names if name not in available]
    if missing:
        print("Warning: default door assets missing from cfg:", missing)
    for name in names:
        if name in available:
            selected.append(available[name])
    if not selected:
        return
    for key in ("door_asset_specs", "door_asset_names", "door_bounding_data", "handle_bounding_data"):
        runtime[key] = [runtime[key][i] for i in selected]


def _format_step_log(step, data):
    groups = [
        ("command", "cyan", ["command_x", "command_yaw", "base_lin_vel_x", "base_ang_vel_z"]),
        ("base", "green", ["base_height", "base_xy", "door_xy", "front_to_door_distance", "traveled_xy"]),
        (
            "phase",
            "yellow",
            [
                "stopped_by_door",
                "arm_phase",
                "pass_started",
                "pass_backoff_done",
                "pass_arm_zero_snapped",
                "pass_backoff_dist",
                "pass_align_done",
                "pass_done",
            ],
        ),
        ("ee", "magenta", ["ee_target", "ee_pos", "ee_z_error"]),
        ("door", "blue", ["door_dof", "door_open_deg", "signed_push_open_deg", "door_open_stage", "door_open_ratio", "handle_open_ratio"]),
        ("episode", "gray", ["episode_length", "time_out", "reset"]),
    ]
    lines = [f"\n{_c(f'[step {step:04d}]', 'blue')}"]
    used = set()
    for title, color, keys in groups:
        present = [key for key in keys if key in data]
        if not present:
            continue
        lines.append(f"  {_c(title, color)}")
        for key in present:
            used.add(key)
            lines.append(f"    {key}: {_round_for_log(data[key])}")
    extra = [key for key in data if key not in used]
    if extra:
        lines.append(f"  {_c('extra', 'gray')}")
        for key in extra:
            lines.append(f"    {key}: {_round_for_log(data[key])}")
    return "\n".join(lines)


class ManipLocoDoorAsset(ManipLoco):
    """Low-level B1Z1 locomotion/manipulation env with an extra door actor."""

    def handle_viewer_action_event(self, evt):
        if evt.action == "free_cam" and evt.value > 0:
            self.free_cam = not self.free_cam
            return
        super().handle_viewer_action_event(evt)

    def _apply_gripper_shape_contact_props(self, robot_asset, shape_props):
        try:
            shape_ranges = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        except Exception:
            return shape_props

        gripper_body_names = ("gripperStator", "gripperMover", self.cfg.asset.gripper_name)
        for body_name in gripper_body_names:
            body_idx = self.body_names_to_idx.get(body_name)
            if body_idx is None or body_idx >= len(shape_ranges):
                continue
            shape_range = shape_ranges[body_idx]
            start = getattr(shape_range, "start", None)
            count = getattr(shape_range, "count", None)
            if start is None or count is None:
                continue
            for shape_id in range(start, start + count):
                prop = shape_props[shape_id]
                prop.friction = DOOR_RUNTIME["gripper_shape_friction"]
                if hasattr(prop, "rolling_friction"):
                    prop.rolling_friction = DOOR_RUNTIME["gripper_shape_friction"]
                if hasattr(prop, "torsion_friction"):
                    prop.torsion_friction = DOOR_RUNTIME["gripper_shape_friction"]
                if hasattr(prop, "rest_offset"):
                    prop.rest_offset = DOOR_RUNTIME["gripper_shape_rest_offset"]
                if hasattr(prop, "contact_offset"):
                    prop.contact_offset = DOOR_RUNTIME["gripper_shape_contact_offset"]
                if hasattr(prop, "physx"):
                    if hasattr(prop.physx, "rest_offset"):
                        prop.physx.rest_offset = DOOR_RUNTIME["gripper_shape_rest_offset"]
                    if hasattr(prop.physx, "contact_offset"):
                        prop.physx.contact_offset = DOOR_RUNTIME["gripper_shape_contact_offset"]
        return shape_props

    def _get_env_origins(self):
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        cols = max(1, int(math.ceil(math.sqrt(self.num_envs))))
        ids = torch.arange(self.num_envs, device=self.device)
        rows = torch.div(ids, cols, rounding_mode="floor")
        cols_idx = ids % cols
        spacing = DOOR_RUNTIME["layout_spacing"]
        self.env_origins[:, 0] = rows.to(torch.float) * spacing
        self.env_origins[:, 1] = cols_idx.to(torch.float) * spacing
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_mesh_materials = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = DOOR_RUNTIME["robot_vhacd_resolution"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset["driveMode"][12:].fill(gymapi.DOF_MODE_POS)
        dof_props_asset["stiffness"][12:].fill(400.0)
        dof_props_asset["damping"][12:].fill(40.0)
        gripper_dofs = self.cfg.env.num_gripper_joints
        if gripper_dofs > 0:
            dof_props_asset["stiffness"][-gripper_dofs:].fill(DOOR_RUNTIME["gripper_stiffness"])
            dof_props_asset["damping"][-gripper_dofs:].fill(DOOR_RUNTIME["gripper_damping"])
            dof_props_asset["friction"][-gripper_dofs:].fill(DOOR_RUNTIME["gripper_joint_friction"])
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        rigid_shape_props_asset = self._apply_gripper_shape_contact_props(robot_asset, rigid_shape_props_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_wo_gripper_names = self.dof_names[:-self.cfg.env.num_gripper_joints]
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)

        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            body_names = [s for s in self.body_names if name in s]
            if len(body_names) == 0:
                raise Exception(f"No body found with name {name}")
            penalized_contact_names.extend(body_names)
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            body_names = [s for s in self.body_names if name in s]
            if len(body_names) == 0:
                raise Exception(f"No body found with name {name}")
            termination_contact_names.extend(body_names)

        self.sensor_indices = []
        for name in feet_names:
            foot_idx = self.body_names_to_idx[name]
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.05))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)
            self.sensor_indices.append(sensor_idx)

        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.gripper_name]

        box_opts = gymapi.AssetOptions()
        box_opts.density = 1000
        box_opts.fix_base_link = False
        box_opts.disable_gravity = False
        box_asset = self.gym.create_box(self.sim, self.cfg.box.box_size, self.cfg.box.box_size, self.cfg.box.box_size, box_opts)

        self._load_door_assets()

        print("------------------------------------------------------")
        print(f"num_actions: {self.num_actions}")
        print(f"num_torques: {self.num_torques}")
        print(f"num_dofs: {self.num_dofs}")
        print(f"num_bodies: {self.num_bodies}")
        print(f"door_assets: {self.door_asset_names}")
        print(f"door_dofs: {self.num_door_dofs}")
        print(f"door_bodies: {self.num_door_bodies}")
        print(f"penalized_contact_names: {penalized_contact_names}")
        print(f"termination_contact_names: {termination_contact_names}")
        print(f"feet_names: {feet_names}")
        print(f"EE Gripper index: {self.gripper_idx}")

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        box_start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.box_actor_handles = []
        self.door_handles = []
        self.door_actor_spec_ids = []
        self.wrist_camera_handles = []
        self.front_camera_handles = []
        self.envs = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            pos = self.env_origins[i].clone()
            pos[0] += DOOR_RUNTIME["robot_x"]
            pos[1] += DOOR_RUNTIME["robot_y_by_spec"][i % len(self.door_asset_specs)]
            pos[2] += DOOR_RUNTIME["robot_z"]
            pos[0] += torch_rand_float(
                -self.cfg.init_state.origin_perturb_range,
                self.cfg.init_state.origin_perturb_range,
                (1, 1),
                device=self.device,
            ).squeeze()
            robot_yaw = DOOR_RUNTIME["robot_yaw"] + self.cfg.init_state.rand_yaw_range * np.random.uniform(-1, 1)
            rand_yaw_quat = gymapi.Quat.from_euler_zyx(0.0, 0.0, robot_yaw)
            start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_dog_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot_dog", i, self.cfg.asset.self_collisions, 0)
            self.actor_handles.append(robot_dog_handle)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_dog_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_dog_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_dog_handle, body_props, recomputeInertia=True)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device)
            if DOOR_RUNTIME["enable_wrist_camera"]:
                self.wrist_camera_handles.append(self._create_wrist_camera(env_handle, robot_dog_handle, i))
            if DOOR_RUNTIME["enable_front_camera"]:
                self.front_camera_handles.append(self._create_front_camera(env_handle, robot_dog_handle, i))

            box_pos = self.env_origins[i].clone()
            box_pos[0] += DOOR_RUNTIME["box_x"]
            box_pos[1] += DOOR_RUNTIME["box_y"]
            box_pos[2] += self.cfg.box.box_env_origins_z
            box_start_pose.p = gymapi.Vec3(*box_pos)
            box_handle = self.gym.create_actor(env_handle, box_asset, box_start_pose, "box", i, self.cfg.asset.self_collisions, 0)
            self.box_actor_handles.append(box_handle)

            box_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_handle)
            box_body_props, _ = self._box_process_rigid_body_props(box_body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, box_handle, box_body_props, recomputeInertia=True)

            self._create_door_actor(env_handle, i)

        assert np.all(np.array(self.actor_handles) == 0)
        assert np.all(np.array(self.box_actor_handles) == 1)
        assert np.all(np.array(self.door_handles) == 2)
        self.num_actors_per_env = 3
        self.full_bodies_per_env = self.num_bodies + 1 + self.num_door_bodies
        self.dof_per_env = self.num_dofs + self.num_door_dofs
        self.robot_actor_indices = torch.arange(0, self.num_actors_per_env * self.num_envs, self.num_actors_per_env, device=self.device)
        self.box_actor_indices = self.robot_actor_indices + 1
        self.door_actor_indices = self.robot_actor_indices + 2

        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)

        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch.cat(
                [
                    torch_rand_float(
                        self.cfg.domain_rand.leg_motor_strength_range[0],
                        self.cfg.domain_rand.leg_motor_strength_range[1],
                        (self.num_envs, 12),
                        device=self.device,
                    ),
                    torch_rand_float(
                        self.cfg.domain_rand.arm_motor_strength_range[0],
                        self.cfg.domain_rand.arm_motor_strength_range[1],
                        (self.num_envs, 6),
                        device=self.device,
                    ),
                ],
                dim=1,
            )
        else:
            self.motor_strength = torch.ones(self.num_envs, self.num_torques, device=self.device)

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        print(f"penalized_contact_indices: {self.penalized_contact_indices}")
        print(f"termination_contact_indices: {self.termination_contact_indices}")
        print(f"feet_indices: {self.feet_indices}")

        if self.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720
            camera_props.height = 480
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                cam_pos = np.array([0, 1, 0.5])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0 * cam_pos))

    def _load_door_assets(self):
        self.door_asset_specs = DOOR_RUNTIME["door_asset_specs"]
        self.door_asset_names = DOOR_RUNTIME["door_asset_names"]
        self.door_bounding_data = DOOR_RUNTIME["door_bounding_data"]
        self.handle_bounding_data = DOOR_RUNTIME["handle_bounding_data"]
        self.door_asset_list = []
        self.door_asset_body_names = []
        self.door_asset_dof_names = []
        self.door_asset_dof_limits_lower = []
        self.door_asset_dof_limits_upper = []

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
        door_opts.vhacd_params.resolution = DOOR_RUNTIME["door_vhacd_resolution"]

        for spec in self.door_asset_specs:
            door_asset = self.gym.load_asset(
                self.sim,
                DOOR_RUNTIME["asset_root"],
                os.path.join(DOOR_RUNTIME["asset_file_door"], spec["path"]),
                door_opts,
            )
            self.door_asset_list.append(door_asset)
            self.door_asset_body_names.append(self.gym.get_asset_rigid_body_names(door_asset))
            self.door_asset_dof_names.append(self.gym.get_asset_dof_names(door_asset))
            door_dof_props = self.gym.get_asset_dof_properties(door_asset)
            if len(door_dof_props["upper"]) >= 2:
                door_dof_props["upper"][1] = min(float(door_dof_props["upper"][1]), math.pi / 4)
            self.door_asset_dof_limits_lower.append(torch.tensor(door_dof_props["lower"], device=self.device, dtype=torch.float))
            self.door_asset_dof_limits_upper.append(torch.tensor(door_dof_props["upper"], device=self.device, dtype=torch.float))

            door_shape_props = self.gym.get_asset_rigid_shape_properties(door_asset)
            for prop in door_shape_props:
                prop.friction = 2.0
            self.gym.set_asset_rigid_shape_properties(door_asset, door_shape_props)

        self.num_door_dofs = self.gym.get_asset_dof_count(self.door_asset_list[0])
        self.num_door_bodies = self.gym.get_asset_rigid_body_count(self.door_asset_list[0])
        for door_asset in self.door_asset_list[1:]:
            if self.gym.get_asset_dof_count(door_asset) != self.num_door_dofs:
                raise RuntimeError("Door assets must have the same DOF count for tensor views.")
            if self.gym.get_asset_rigid_body_count(door_asset) != self.num_door_bodies:
                raise RuntimeError("Door assets must have the same rigid body count for tensor views.")

    def _create_door_actor(self, env_handle, env_i):
        spec_id = env_i % len(self.door_asset_specs)
        door_asset = self.door_asset_list[spec_id]
        door_bounds = self.door_bounding_data[spec_id]

        door_pose = gymapi.Transform()
        origin = self.env_origins[env_i]
        door_pose.p = gymapi.Vec3(
            float(origin[0] + DOOR_RUNTIME["door_x"]),
            float(origin[1] + DOOR_RUNTIME["door_y"]),
            float(origin[2] - door_bounds["min"][2] * DOOR_RUNTIME["door_actor_scale"] + DOOR_RUNTIME["door_z_offset"]),
        )
        door_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        door_handle = self.gym.create_actor(env_handle, door_asset, door_pose, "door", env_i, 0, 1)
        if abs(DOOR_RUNTIME["door_actor_scale"] - 1.0) > 1e-6:
            self.gym.set_actor_scale(env_handle, door_handle, DOOR_RUNTIME["door_actor_scale"])
        handle_body_local_idx = len(self.door_asset_body_names[spec_id]) - 1
        try:
            self.gym.set_rigid_body_segmentation_id(
                env_handle,
                door_handle,
                handle_body_local_idx,
                DOOR_RUNTIME["handle_seg_id"],
            )
        except AttributeError:
            if not getattr(self, "_segmentation_warned", False):
                print("set_rigid_body_segmentation_id is not available; handle mask will fall back to the door actor segmentation.")
                self._segmentation_warned = True

        door_dof_props = self.gym.get_asset_dof_properties(door_asset)
        door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        num_cfg_dofs = min(self.gym.get_asset_dof_count(door_asset), len(DOOR_RUNTIME["door_joint_damping"]))
        door_dof_props["damping"][:num_cfg_dofs] = np.asarray(DOOR_RUNTIME["door_joint_damping"][:num_cfg_dofs], dtype=np.float32)
        door_dof_props["friction"][:num_cfg_dofs] = np.asarray(DOOR_RUNTIME["door_joint_friction"][:num_cfg_dofs], dtype=np.float32)
        door_dof_props["effort"][:num_cfg_dofs] = np.asarray(DOOR_RUNTIME["door_joint_effort"][:num_cfg_dofs], dtype=np.float32)
        if len(door_dof_props["upper"]) >= 2:
            door_dof_props["upper"][1] = min(float(door_dof_props["upper"][1]), math.pi / 4)
        self.gym.set_actor_dof_properties(env_handle, door_handle, door_dof_props)

        self.door_handles.append(door_handle)
        self.door_actor_spec_ids.append(spec_id)

    def _robot_spawn_y(self, env_ids):
        if hasattr(self, "initial_handle_center_y"):
            return self.initial_handle_center_y[env_ids] + DOOR_RUNTIME["robot_y"]
        if hasattr(self, "door_asset_indices"):
            spec_ids = self.door_asset_indices[env_ids]
        else:
            spec_ids = torch.tensor([self.door_actor_spec_ids[int(env_id)] for env_id in env_ids], device=self.device, dtype=torch.long)
        return self.env_origins[env_ids, 1] + torch.tensor(DOOR_RUNTIME["robot_y_by_spec"], device=self.device, dtype=torch.float)[spec_ids]

    def _create_wrist_camera(self, env_handle, actor_handle, env_i):
        wrist_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "link06")
        if wrist_handle < 0:
            raise RuntimeError("Cannot attach wrist camera: rigid body 'link06' was not found.")

        wrist_cfg = DOOR_RUNTIME["wrist_camera_cfg"]
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = int(wrist_cfg.get("resolution", [96, 54])[0])
        camera_props.height = int(wrist_cfg.get("resolution", [96, 54])[1])
        if wrist_cfg.get("horizontal_fov", None) is not None:
            horizontal_fov = wrist_cfg["horizontal_fov"]
            camera_props.horizontal_fov = (
                np.random.uniform(horizontal_fov[0], horizontal_fov[1])
                if isinstance(horizontal_fov, (list, tuple))
                else horizontal_fov
            )

        local_pos = list(wrist_cfg.get("position", [0.0955, 0.22, -0.03175]))
        rand_position = wrist_cfg.get("rand_position", [0.0, 0.0, 0.0])
        local_pos[0] += np.random.uniform(-rand_position[0], rand_position[0])
        local_pos[1] += np.random.uniform(-rand_position[1], rand_position[1])
        local_pos[2] += np.random.uniform(-rand_position[2], rand_position[2])
        local_rot = list(wrist_cfg.get("rotation", [-1.57, 0.0, -0.87]))
        local_rot[2] -= DOOR_RUNTIME["wrist_camera_down_tilt"]

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*local_pos)
        local_transform.r = gymapi.Quat.from_euler_zyx(*local_rot)
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        if camera_handle < 0:
            if not getattr(self, "_wrist_camera_create_warned", False):
                print("⚠️📷 Wrist camera sensor creation failed; camera tensors will be disabled for this run.", flush=True)
                self._wrist_camera_create_warned = True
            return camera_handle
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            wrist_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        if not hasattr(self, "wrist_camera_local_pos"):
            self.wrist_camera_local_pos = []
            self.wrist_camera_local_quat = []
        self.wrist_camera_local_pos.append(local_pos)
        self.wrist_camera_local_quat.append([local_transform.r.x, local_transform.r.y, local_transform.r.z, local_transform.r.w])
        return camera_handle

    def _create_front_camera(self, env_handle, actor_handle, env_i):
        front_body_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "trunk")
        if front_body_handle < 0:
            front_body_handle = actor_handle

        front_cfg = DOOR_RUNTIME["front_camera_cfg"]
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = int(front_cfg.get("resolution", [96, 54])[0])
        camera_props.height = int(front_cfg.get("resolution", [96, 54])[1])
        if front_cfg.get("horizontal_fov", None) is not None:
            horizontal_fov = front_cfg["horizontal_fov"]
            camera_props.horizontal_fov = (
                np.random.uniform(horizontal_fov[0], horizontal_fov[1])
                if isinstance(horizontal_fov, (list, tuple))
                else horizontal_fov
            )

        local_pos = list(front_cfg.get("position", [0.425, 0.04, 0.12]))
        rand_position = front_cfg.get("rand_position", [0.0, 0.0, 0.0])
        local_pos[0] += np.random.uniform(-rand_position[0], rand_position[0])
        local_pos[1] += np.random.uniform(-rand_position[1], rand_position[1])
        local_pos[2] += np.random.uniform(-rand_position[2], rand_position[2])
        local_rot = list(front_cfg.get("rotation", [0.0, 0.0, 0.0]))
        # gymapi.Quat.from_euler_zyx expects [yaw_z, pitch_y, roll_x].
        local_rot[0] = np.deg2rad(DOOR_RUNTIME["front_camera_yaw_deg"])
        local_rot[1] = np.deg2rad(DOOR_RUNTIME["front_camera_pitch_deg"])
        local_rot[2] = np.deg2rad(DOOR_RUNTIME["front_camera_roll_deg"])

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*local_pos)
        local_transform.r = gymapi.Quat.from_euler_zyx(*local_rot)
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        if camera_handle < 0:
            if not getattr(self, "_front_camera_create_warned", False):
                print("⚠️📷 Front camera sensor creation failed; front camera tensors will be disabled for this run.", flush=True)
                self._front_camera_create_warned = True
            return camera_handle
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            front_body_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        if not hasattr(self, "front_camera_local_pos"):
            self.front_camera_local_pos = []
            self.front_camera_local_quat = []
        self.front_camera_local_pos.append(local_pos)
        self.front_camera_local_quat.append([local_transform.r.x, local_transform.r.y, local_transform.r.z, local_transform.r.w])
        return camera_handle

    def _init_wrist_camera_tensors(self):
        self.wrist_camera_tensors = {}
        self.front_camera_tensors = {}
        if DOOR_RUNTIME["enable_wrist_camera"] and any(camera_handle < 0 for camera_handle in self.wrist_camera_handles):
            if not getattr(self, "_wrist_camera_tensor_warned", False):
                print("⚠️📷 Wrist camera handles are invalid; skipping wrist camera tensor access.", flush=True)
                self._wrist_camera_tensor_warned = True
            self.wrist_camera_handles = []
        if DOOR_RUNTIME["enable_front_camera"] and any(camera_handle < 0 for camera_handle in self.front_camera_handles):
            if not getattr(self, "_front_camera_tensor_warned", False):
                print("⚠️📷 Front camera handles are invalid; skipping front camera tensor access.", flush=True)
                self._front_camera_tensor_warned = True
            self.front_camera_handles = []
        image_types = {
            "rgb": gymapi.IMAGE_COLOR,
            "depth": gymapi.IMAGE_DEPTH,
            "seg": gymapi.IMAGE_SEGMENTATION,
        }
        for name, image_type in image_types.items():
            if not DOOR_RUNTIME[f"camera_{name}"]:
                continue
            if DOOR_RUNTIME["enable_wrist_camera"]:
                self.wrist_camera_tensors[name] = [
                    gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.wrist_camera_handles[env_i],
                            image_type,
                        )
                    )
                    for env_i, env_handle in enumerate(self.envs)
                ]
            if DOOR_RUNTIME["enable_front_camera"]:
                self.front_camera_tensors[name] = [
                    gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.front_camera_handles[env_i],
                            image_type,
                        )
                    )
                    for env_i, env_handle in enumerate(self.envs)
                ]

    def capture_wrist_camera_images(self):
        if not getattr(self, "wrist_camera_tensors", None) and not getattr(self, "front_camera_tensors", None):
            return {}

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        images = {}
        try:
            for name, tensors in self.wrist_camera_tensors.items():
                images[name] = torch.stack([tensor.clone() for tensor in tensors], dim=0)
                images[f"wrist_{name}"] = images[name]
            for name, tensors in self.front_camera_tensors.items():
                images[f"front_{name}"] = torch.stack([tensor.clone() for tensor in tensors], dim=0)
        finally:
            self.gym.end_access_image_tensors(self.sim)
        for prefix in ("wrist", "front"):
            seg_key = "seg" if prefix == "wrist" and "seg" in images else f"{prefix}_seg"
            depth_key = "depth" if prefix == "wrist" and "depth" in images else f"{prefix}_depth"
            if seg_key not in images:
                continue
            handle_mask = images[seg_key] == DOOR_RUNTIME["handle_seg_id"]
            images[f"{prefix}_handle_mask"] = handle_mask.to(torch.float32)
            if prefix == "wrist":
                images["handle_mask"] = images[f"{prefix}_handle_mask"]
            if depth_key in images:
                depth_image = images[depth_key].clone().to(torch.float32)
                depth_image = torch.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
                depth_image = torch.abs(depth_image)
                depth_image[depth_image < DOOR_RUNTIME["camera_depth_clip_lower"]] = 0
                depth_image = torch.clamp(depth_image, 0.0, DOOR_RUNTIME["camera_depth_clip_far"])
                normalized_depth = depth_image / DOOR_RUNTIME["camera_depth_clip_far"]
                images[f"{prefix}_depth_meters"] = depth_image
                images[f"{prefix}_normalized_depth"] = normalized_depth
                images[f"{prefix}_handle_masked_depth"] = depth_image * images[f"{prefix}_handle_mask"]
                if prefix == "wrist":
                    images["depth_meters"] = depth_image
                    images["normalized_depth"] = normalized_depth
                    images["handle_masked_depth"] = images[f"{prefix}_handle_masked_depth"]
        self.last_wrist_camera_images = images
        return images

    def show_wrist_seg(self, images, env_id=0):
        if not DOOR_RUNTIME["show_seg"]:
            return
        if cv2 is None:
            if not getattr(self, "_cv2_missing_warned", False):
                print("cv2 is not available; wrist segmentation image will not be displayed.")
                self._cv2_missing_warned = True
            return

        mask_key = "wrist_handle_mask" if "wrist_handle_mask" in images else "handle_mask"
        if mask_key not in images:
            return
        env_id = int(np.clip(env_id, 0, images[mask_key].shape[0] - 1))
        display_scale = max(1, int(DOOR_RUNTIME["camera_display_scale"]))
        for prefix, title in (("wrist", "Wrist"), ("front", "Front")):
            local_mask_key = f"{prefix}_handle_mask" if f"{prefix}_handle_mask" in images else ("handle_mask" if prefix == "wrist" else None)
            local_depth_key = (
                f"{prefix}_handle_masked_depth"
                if f"{prefix}_handle_masked_depth" in images
                else ("handle_masked_depth" if prefix == "wrist" else None)
            )
            if local_mask_key is None or local_mask_key not in images:
                continue
            mask_image = np.squeeze(images[local_mask_key][env_id].detach().cpu().numpy()).astype(np.float32)
            mask_vis = (255.0 * mask_image).astype(np.uint8)
            if display_scale > 1:
                mask_vis = cv2.resize(mask_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"{title} Handle Mask", mask_vis)
            if local_depth_key is None or local_depth_key not in images:
                continue
            masked_depth = np.squeeze(images[local_depth_key][env_id].detach().cpu().numpy()).astype(np.float32)
            masked_depth_vis = np.zeros_like(masked_depth, dtype=np.uint8)
            valid_depth = masked_depth[mask_image > 0.5]
            valid_depth = valid_depth[np.isfinite(valid_depth) & (valid_depth > 0.0)]
            if valid_depth.size > 0:
                depth_min = float(valid_depth.min())
                depth_max = float(valid_depth.max())
                if depth_max - depth_min < 1e-4:
                    depth_scaled = masked_depth / max(depth_max, 1e-4)
                else:
                    depth_scaled = (masked_depth - depth_min) / (depth_max - depth_min)
                masked_depth_vis = (255.0 * np.clip(depth_scaled, 0.0, 1.0) * mask_image).astype(np.uint8)
            if display_scale > 1:
                masked_depth_vis = cv2.resize(masked_depth_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"{title} Handle Masked Depth", masked_depth_vis)
        cv2.waitKey(1)

    def draw_wrist_camera_axes(self, scale=0.10, thickness=0.004):
        if getattr(self, "viewer", None) is None:
            return
        camera_specs = []
        if DOOR_RUNTIME["enable_wrist_camera"] and hasattr(self, "wrist_camera_local_pos"):
            camera_specs.append(("link06", self.wrist_camera_local_pos, self.wrist_camera_local_quat))
        if DOOR_RUNTIME["enable_front_camera"] and hasattr(self, "front_camera_local_pos"):
            camera_specs.append(("trunk", self.front_camera_local_pos, self.front_camera_local_quat))
        for body_name, local_pos_list, local_quat_list in camera_specs:
            body_idx = self.body_names_to_idx.get(body_name)
            if body_idx is None:
                continue
            local_pos = torch.tensor(local_pos_list, device=self.device, dtype=torch.float32)
            local_quat = torch.tensor(local_quat_list, device=self.device, dtype=torch.float32)
            body_pos = self.rigid_body_state[:, body_idx, :3]
            body_quat = self.rigid_body_state[:, body_idx, 3:7]
            camera_pos = body_pos + quat_apply(body_quat, local_pos)
            camera_quat = quat_mul(body_quat, local_quat)
            for env_i in range(self.num_envs):
                pos = camera_pos[env_i].detach().cpu().tolist()
                quat = camera_quat[env_i].detach().cpu().tolist()
                pose = gymapi.Transform(gymapi.Vec3(*pos), gymapi.Quat(*quat))
                axes_geom = ThickAxesGeometry(scale=scale, thickness=thickness, pose=pose)
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[env_i], gymapi.Transform())

    def _init_buffers(self):
        self.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.device)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot_dog")
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)
        self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, self.num_actors_per_env, 13)
        self.root_states = self._root_states[:, 0, :]
        self.box_root_state = self._root_states[:, 1, :]
        self.door_root_state = self._root_states[:, 2, :]
        self.initial_door_root_state = self.door_root_state.clone()

        self._full_dof_state_flat = gymtorch.wrap_tensor(dof_state_tensor)
        self._full_dof_state = self._full_dof_state_flat.view(self.num_envs, self.dof_per_env, 2)
        self.dof_state = self._full_dof_state[:, : self.num_dofs, :]
        self.dof_pos = self._full_dof_state[:, : self.num_dofs, 0]
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.cfg.env.num_gripper_joints]
        self.dof_vel = self._full_dof_state[:, : self.num_dofs, 1]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.cfg.env.num_gripper_joints]
        self._door_dof_pos = self._full_dof_state[:, self.num_dofs :, 0]
        self._door_dof_vel = self._full_dof_state[:, self.num_dofs :, 1]

        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.arm_base_offset = torch.tensor([0.3, 0.0, 0.09], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)

        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)

        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.full_bodies_per_env, 3)
        self.contact_forces = self._contact_forces[:, : self.num_bodies, :]
        self.box_contact_force = self._contact_forces[:, self.num_bodies, :]
        self.door_contact_forces = self._contact_forces[:, self.num_bodies + 1 :, :]

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.full_bodies_per_env, 13)
        self.rigid_body_state = self._rigid_body_state[:, : self.num_bodies, :]
        self.box_rigid_body_state = self._rigid_body_state[:, self.num_bodies, :]
        self.door_rigid_body_state = self._rigid_body_state[:, self.num_bodies + 1 :, :]

        self.jacobian_whole = gymtorch.wrap_tensor(jacobian_tensor)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        self.ee_pos = self.rigid_body_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_state[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_body_state[:, self.gripper_idx, 7:]
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6, -(6 + self.cfg.env.num_gripper_joints) : -self.cfg.env.num_gripper_joints]
        self._init_wrist_camera_tensors()

        self.box_pos = self.box_root_state[:, 0:3]
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        self.traj_timesteps = (
            torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze(1)
            / self.dt
        )
        self.traj_total_timesteps = self.traj_timesteps + (
            torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze(1)
            / self.dt
        )
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.add_noise = self.cfg.noise.add_noise
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert self.cfg.goal_ee.command_mode in ["cart", "sphere"]
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.ee_goal_center_offset = torch.tensor(
            [
                self.cfg.goal_ee.sphere_center.x_offset,
                self.cfg.goal_ee.sphere_center.y_offset,
                self.cfg.goal_ee.sphere_center.z_invariant_offset,
            ],
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.curr_ee_goal_cart_world = self._get_ee_goal_spherical_center() + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        self._init_door_tensors()

        print("------------------------------------------------------")
        print(f"root_states shape: {self.root_states.shape}")
        print(f"full root_states shape: {self._root_states.shape}")
        print(f"dof_state shape: {self.dof_state.shape}")
        print(f"full dof_state shape: {self._full_dof_state.shape}")
        print(f"force_sensor_tensor shape: {self.force_sensor_tensor.shape}")
        print(f"contact_forces shape: {self.contact_forces.shape}")
        print(f"rigid_body_state shape: {self.rigid_body_state.shape}")
        print(f"door_rigid_body_state shape: {self.door_rigid_body_state.shape}")
        print(f"jacobian_whole shape: {self.jacobian_whole.shape}")
        print(f"box_root_state shape: {self.box_root_state.shape}")
        print(f"door_root_state shape: {self.door_root_state.shape}")
        print("------------------------------------------------------")

        self.common_step_counter = 0
        self.extras = {}
        self.extras["episode"] = {}
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.full_torques = torch.zeros(self.num_envs, self.dof_per_env, dtype=torch.float, device=self.device, requires_grad=False)
        self.full_pos_targets = torch.zeros(self.num_envs, self.dof_per_env, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )[: self.cfg.commands.num_commands]
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.cfg.env.num_gripper_joints, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]

        for i in range(self.num_torques):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    raise Exception(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:-self.cfg.env.num_gripper_joints]
        self.default_ee_orn_local_quat = self._compute_default_ee_orn_local_quat()
        self.global_steps = 0

    def _init_door_tensors(self):
        body_names = self.door_asset_body_names[0]
        self.door_body_name = body_names[-2] if len(body_names) >= 2 else body_names[-1]
        self.handle_body_name = body_names[-1]
        self.door_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.door_handles[0], self.door_body_name, gymapi.DOMAIN_ENV)
        self.handle_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.door_handles[0], self.handle_body_name, gymapi.DOMAIN_ENV)
        self.initial_handle_center_y = self._rigid_body_state[:, self.handle_body_idx, 1].clone()
        self.door_asset_indices = torch.tensor(self.door_actor_spec_ids, device=self.device, dtype=torch.long)
        self.door_hinge_limits_lower = torch.stack([limits[0] for limits in self.door_asset_dof_limits_lower], dim=0)
        self.door_hinge_limits_upper = torch.stack([limits[0] for limits in self.door_asset_dof_limits_upper], dim=0)
        self.handle_limits_lower = torch.stack([limits[1] for limits in self.door_asset_dof_limits_lower], dim=0)
        self.handle_limits_upper = torch.stack([limits[1] for limits in self.door_asset_dof_limits_upper], dim=0)
        goal_pos_offsets = [[DOOR_RUNTIME["door_actor_scale"] * float(v) for v in item["goal_pos"]] for item in self.handle_bounding_data]
        self.goal_pos_offset_tensor = torch.tensor(goal_pos_offsets, device=self.device, dtype=torch.float)[self.door_asset_indices]
        self.handle_unlock_threshold = (
            DOOR_RUNTIME["handle_unlock_ratio"]
            * (self.handle_limits_upper[self.door_asset_indices] - self.handle_limits_lower[self.door_asset_indices])
        )
        self.open_door_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.door_open_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.handle_open_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _door_torques(self):
        door_torques = torch.zeros(self.num_envs, self.num_door_dofs, dtype=torch.float, device=self.device)
        if self.num_door_dofs < 2:
            return door_torques

        door_angle = self._door_dof_pos[:, 0]
        handle_angle_from_lower = self._door_dof_pos[:, 1] - self.handle_limits_lower[self.door_asset_indices]
        unlock_now = (handle_angle_from_lower >= self.handle_unlock_threshold) | (torch.abs(door_angle) > 0.01)
        self.open_door_stage[:] = self.open_door_stage | unlock_now
        hinge_range = torch.clamp(
            torch.maximum(
                torch.abs(self.door_hinge_limits_upper[self.door_asset_indices]),
                torch.abs(self.door_hinge_limits_lower[self.door_asset_indices]),
            ),
            min=1e-3,
        )
        auto_open_active = torch.abs(door_angle) < DOOR_RUNTIME["door_auto_open_target_ratio"] * hinge_range
        auto_open_torque = torch.where(
            auto_open_active,
            torch.full_like(door_angle, DOOR_RUNTIME["door_auto_open_force"] * DOOR_RUNTIME["door_auto_open_sign"]),
            torch.zeros_like(door_angle),
        )
        door_torques[:, 0] = torch.where(
            self.open_door_stage,
            auto_open_torque - DOOR_RUNTIME["door_open_resistance"] * door_angle - DOOR_RUNTIME["door_open_damping"] * self._door_dof_vel[:, 0],
            -torch.full_like(door_angle, DOOR_RUNTIME["door_lock_force"]),
        )
        door_torques[:, 1] = (
            -DOOR_RUNTIME["handle_spring_stiffness"] * handle_angle_from_lower
            - DOOR_RUNTIME["handle_spring_damping"] * self._door_dof_vel[:, 1]
        )

        handle_range = torch.clamp(
            self.handle_limits_upper[self.door_asset_indices] - self.handle_limits_lower[self.door_asset_indices],
            min=1e-3,
        )
        self.door_open_ratio[:] = torch.clamp(torch.abs(door_angle) / hinge_range, 0.0, 1.5)
        self.handle_open_ratio[:] = torch.clamp(handle_angle_from_lower / handle_range, 0.0, 1.5)
        return door_torques

    def _update_curr_ee_goal(self):
        if getattr(self, "external_ee_goal_control", False):
            self.goal_timer += 1
            return
        super()._update_curr_ee_goal()

    def _draw_ee_goal_traj(self):
        if getattr(self, "external_ee_goal_control", False):
            return
        super()._draw_ee_goal_traj()

    def step(self, actions):
        actions[:, 12:] = 0.0
        actions = self._reindex_all(actions)
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions).to(self.device)
        self.render()
        if self.action_delay != -1:
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], actions[:, None, :]], dim=1)
        if self.global_steps < 10000 * 24:
            actions = self.action_history_buf[:, -1]
        else:
            actions = self.action_history_buf[:, -2]

        self.actions = actions.clone()

        dpos = self.curr_ee_goal_cart_world - self.ee_pos
        if getattr(self, "external_ee_goal_control", False):
            dpos *= getattr(self, "external_pos_gain", 1.0)
        drot = orientation_error(self.ee_goal_orn_quat, self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))
        if getattr(self, "external_ee_goal_control", False):
            external_orn_gain = getattr(self, "external_orn_gain", 0.0)
            drot *= external_orn_gain
        dpose = torch.cat([dpos, drot], -1).unsqueeze(-1)
        arm_pos_targets = self._control_ik(dpose) + self.dof_pos[:, -(6 + self.cfg.env.num_gripper_joints) : -self.cfg.env.num_gripper_joints]
        freeze_arm_default = getattr(self, "freeze_arm_default", None)
        if freeze_arm_default is not None and torch.any(freeze_arm_default):
            arm_slice = slice(-(6 + self.cfg.env.num_gripper_joints), -self.cfg.env.num_gripper_joints)
            arm_pos_targets[freeze_arm_default] = self.default_dof_pos[arm_slice].unsqueeze(0)
        freeze_arm_zero = getattr(self, "freeze_arm_zero", None)
        if freeze_arm_zero is not None and torch.any(freeze_arm_zero):
            arm_pos_targets[freeze_arm_zero] = 0.0
        all_pos_targets = torch.zeros_like(self.dof_pos)
        all_pos_targets[:, -(6 + self.cfg.env.num_gripper_joints) : -self.cfg.env.num_gripper_joints] = arm_pos_targets
        gripper_target = getattr(self, "external_gripper_target", None)
        if gripper_target is None:
            all_pos_targets[:, -self.cfg.env.num_gripper_joints :] = self.default_dof_pos[-self.cfg.env.num_gripper_joints :].unsqueeze(0)
        else:
            all_pos_targets[:, -self.cfg.env.num_gripper_joints :] = gripper_target

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            self.full_pos_targets.zero_()
            self.full_pos_targets[:, : self.num_dofs] = all_pos_targets
            self.full_pos_targets[:, self.num_dofs :] = self._door_dof_pos
            self.full_torques.zero_()
            self.full_torques[:, : self.num_dofs] = self.torques
            self.full_torques[:, self.num_dofs :] = self._door_torques()
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.full_pos_targets))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.full_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.global_steps += 1
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.arm_rew_buf, self.reset_buf, self.extras

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.0
        self._door_dof_pos[env_ids] = 0.0
        self._door_dof_vel[env_ids] = 0.0
        self.open_door_stage[env_ids] = False
        self.door_open_ratio[env_ids] = 0.0
        self.handle_open_ratio[env_ids] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._full_dof_state_flat))
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, 0] = self.env_origins[env_ids, 0] + DOOR_RUNTIME["robot_x"]
        self.root_states[env_ids, 1] = self._robot_spawn_y(env_ids)
        self.root_states[env_ids, 2] = self.env_origins[env_ids, 2] + DOOR_RUNTIME["robot_z"]
        self.root_states[env_ids, 0] += torch_rand_float(
            -self.cfg.init_state.origin_perturb_range,
            self.cfg.init_state.origin_perturb_range,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.box_root_state[env_ids, 0] = self.env_origins[env_ids, 0] + DOOR_RUNTIME["box_x"]
        self.box_root_state[env_ids, 1] = self.env_origins[env_ids, 1] + DOOR_RUNTIME["box_y"]
        self.box_root_state[env_ids, 2] = self.env_origins[env_ids, 2] + self.cfg.box.box_env_origins_z
        self.door_root_state[env_ids] = self.initial_door_root_state[env_ids]

        rand_yaw = DOOR_RUNTIME["robot_yaw"] + self.cfg.init_state.rand_yaw_range * torch_rand_float(
            -1, 1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        quat = quat_from_euler_xyz(0 * rand_yaw, 0 * rand_yaw, rand_yaw)
        self.root_states[env_ids, 3:7] = quat[:, :]
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -self.cfg.init_state.init_vel_perturb_range,
            self.cfg.init_state.init_vel_perturb_range,
            (len(env_ids), 6),
            device=self.device,
        )

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.root_states[:, 7:9] = torch.where(
            self.commands.sum(dim=1).unsqueeze(-1) == 0,
            self.root_states[:, 7:9] * 2.5,
            self.root_states[:, 7:9],
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episode_length_s", type=float, default=10000.0)
    parser.add_argument("--speed_min", type=float, default=0.65)
    parser.add_argument("--speed_max", type=float, default=0.80)
    parser.add_argument("--yaw_min", type=float, default=0.0)
    parser.add_argument("--yaw_max", type=float, default=0.0)
    parser.add_argument("--resample_interval", type=int, default=90)
    parser.add_argument("--stop_distance", type=float, default=0.75)
    parser.add_argument("--robot_front_offset", type=float, default=0.55)
    parser.add_argument("--pregrasp_offset", type=float, default=0.15)
    parser.add_argument("--grasp_offset", type=float, default=0.0)
    parser.add_argument("--grasp_x_offset", type=float, default=-0.03)
    parser.add_argument("--grasp_z_offset", type=float, default=-0.03)
    parser.add_argument("--initial_hold_seconds", type=float, default=5.0)
    parser.add_argument("--approach_steps", type=int, default=240)
    parser.add_argument("--grasp_steps", type=int, default=150)
    parser.add_argument("--grasp_hold_steps", type=int, default=100)
    parser.add_argument("--gripper_close_steps", type=int, default=120)
    parser.add_argument("--handle_rotate_steps", type=int, default=300)
    parser.add_argument("--door_pull_steps", type=int, default=960)
    parser.add_argument("--lever_step_size", type=float, default=0.06)
    parser.add_argument("--pull_base_vx", type=float, default=-0.3)
    parser.add_argument("--pull_base_yaw", type=float, default=0.25)
    parser.add_argument("--pass_open_angle_deg", type=float, default=80.0)
    parser.add_argument("--pass_backoff_distance", type=float, default=0.50)
    parser.add_argument("--pass_backoff_vx", type=float, default=-0.30)
    parser.add_argument("--pass_front_distance", type=float, default=0.50)
    parser.add_argument("--pass_through_distance", type=float, default=1.40)
    parser.add_argument("--pass_align_vx", type=float, default=0.35)
    parser.add_argument("--pass_forward_vx", type=float, default=0.45)
    parser.add_argument("--pass_min_vx", type=float, default=0.12)
    parser.add_argument("--pass_move_yaw_tol", type=float, default=0.35)
    parser.add_argument("--pass_yaw_gain", type=float, default=1.5)
    parser.add_argument("--pass_yaw_clip", type=float, default=0.6)
    parser.add_argument("--pass_align_pos_tol", type=float, default=0.12)
    parser.add_argument("--pass_yaw_tol", type=float, default=0.25)
    parser.add_argument("--pass_center_y_offset", type=float, default=0.0)
    parser.add_argument("--pass_left_offset", type=float, default=0.10)
    parser.add_argument("--unidoor_style_pull", dest="unidoor_style_pull", action="store_true", default=True)
    parser.add_argument("--no_unidoor_style_pull", dest="unidoor_style_pull", action="store_false")
    parser.add_argument("--handle_rotate_distance", type=float, default=0.28)
    parser.add_argument("--handle_rotate_right_distance", type=float, default=0.03)
    parser.add_argument("--handle_rotate_down_distance", type=float, default=0.03)
    parser.add_argument("--handle_rotate_angle", type=float, default=1.05)
    parser.add_argument("--handle_arc_radius", type=float, default=0.18)
    parser.add_argument("--door_pull_distance", type=float, default=1.10)
    parser.add_argument("--gripper_open", type=float, default=-1.5707963267948966)
    parser.add_argument("--gripper_closed", type=float, default=0.0)
    parser.add_argument("--gripper_close_ratio", type=float, default=0.8)
    parser.add_argument("--gripper_stiffness", type=float, default=160.0)
    parser.add_argument("--gripper_damping", type=float, default=16.0)
    parser.add_argument("--gripper_joint_friction", type=float, default=120.0)
    parser.add_argument("--handle_spring_stiffness", type=float, default=0.5)
    parser.add_argument("--handle_spring_damping", type=float, default=0.1)
    parser.add_argument("--handle_unlock_ratio", type=float, default=0.35)
    parser.add_argument("--handle_joint_friction", type=float, default=0.05)
    parser.add_argument("--handle_joint_damping", type=float, default=0.05)
    parser.add_argument("--door_open_resistance", type=float, default=0.2)
    parser.add_argument("--door_open_damping", type=float, default=0.05)
    parser.add_argument("--door_joint_friction", type=float, default=0.5)
    parser.add_argument("--door_joint_damping", type=float, default=0.2)
    parser.add_argument("--door_auto_open_force", type=float, default=0.0)
    parser.add_argument("--door_auto_open_sign", type=float, default=1.0)
    parser.add_argument("--door_auto_open_target_ratio", type=float, default=0.95)
    parser.add_argument("--robot_vhacd_resolution", type=int, default=300000)
    parser.add_argument("--gripper_shape_contact_offset", type=float, default=0.018)
    parser.add_argument("--gripper_shape_rest_offset", type=float, default=0.003)
    parser.add_argument("--gripper_shape_friction", type=float, default=8.0)
    parser.add_argument("--door_vhacd_resolution", type=int, default=100000)
    parser.add_argument("--sim_substeps", type=int, default=2)
    parser.add_argument("--sim_position_iterations", type=int, default=12)
    parser.add_argument("--sim_velocity_iterations", type=int, default=4)
    parser.add_argument("--sim_contact_offset", type=float, default=0.02)
    parser.add_argument("--sim_rest_offset", type=float, default=0.002)
    parser.add_argument("--sim_max_depenetration_velocity", type=float, default=0.5)
    parser.add_argument("--external_pos_gain", type=float, default=1)
    parser.add_argument("--external_orn_gain", type=float, default=1)
    parser.add_argument("--forward_ee_roll", type=float, default=math.pi / 2)
    parser.add_argument("--forward_ee_pitch", type=float, default=0.0)
    parser.add_argument("--gripper_red_axis_rot", type=float, default=-math.pi / 2)
    parser.add_argument("--preview_trajectory_at_spawn", dest="preview_trajectory_at_spawn", action="store_true", default=False)
    parser.add_argument("--no_preview_trajectory_at_spawn", dest="preview_trajectory_at_spawn", action="store_false")
    parser.add_argument("--fixed_vx", type=float, default=None)
    parser.add_argument("--fixed_yaw", type=float, default=None)
    parser.add_argument("--layout_spacing", type=float, default=5.0)
    parser.add_argument("--robot_x", type=float, default=4.1)
    parser.add_argument("--robot_y", type=float, default=0.0)
    parser.add_argument("--robot_z", type=float, default=0.5)
    parser.add_argument("--robot_yaw", type=float, default=math.pi)
    parser.add_argument("--door_x", type=float, default=2.5)
    parser.add_argument("--door_y", type=float, default=0.0)
    parser.add_argument("--door_z_offset", type=float, default=0.01)
    parser.add_argument("--door_actor_scale", type=float, default=1.2)
    parser.add_argument("--box_x", type=float, default=-3.0)
    parser.add_argument("--box_y", type=float, default=-3.0)
    parser.add_argument("--door_cfg", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "cfg" / "b1z1_opendoor.yaml"))
    parser.add_argument("--use_all_door_assets", action="store_true")
    parser.add_argument("--log_dir", type=str, default=str(LOW_LEVEL_ROOT / "logs" / "b1z1-low" / "b1z1_locomanip"))
    parser.add_argument("--checkpoint", type=int, default=45000)
    parser.add_argument("--enable_wrist_camera", dest="enable_wrist_camera", action="store_true", default=True)
    parser.add_argument("--no_enable_wrist_camera", dest="enable_wrist_camera", action="store_false")
    parser.add_argument("--enable_front_camera", dest="enable_front_camera", action="store_true", default=True)
    parser.add_argument("--no_enable_front_camera", dest="enable_front_camera", action="store_false")
    parser.add_argument("--camera_rgb", action="store_true")
    parser.add_argument("--camera_depth", dest="camera_depth", action="store_true", default=True)
    parser.add_argument("--no_camera_depth", dest="camera_depth", action="store_false")
    parser.add_argument("--camera_seg", dest="camera_seg", action="store_true", default=True)
    parser.add_argument("--no_camera_seg", dest="camera_seg", action="store_false")
    parser.add_argument("--show_seg", dest="show_seg", action="store_true", default=True)
    parser.add_argument("--no_show_seg", dest="show_seg", action="store_false")
    parser.add_argument("--camera_env_id", type=int, default=0)
    parser.add_argument("--handle_seg_id", type=int, default=2)
    parser.add_argument("--camera_depth_clip_lower", type=float, default=0.02)
    parser.add_argument("--camera_depth_clip_far", type=float, default=2.0)
    parser.add_argument("--camera_display_scale", type=int, default=5)
    parser.add_argument("--wrist_camera_down_tilt", type=float, default=0.20)
    parser.add_argument("--front_camera_yaw_deg", type=float, default=0.0)
    parser.add_argument("--front_camera_pitch_deg", type=float, default=-60.0)
    parser.add_argument("--front_camera_roll_deg", type=float, default=0.0)
    parser.add_argument("--camera_axis_scale", type=float, default=0.10)
    parser.add_argument("--camera_axis_thickness", type=float, default=0.004)
    parser.add_argument("--record_dp_dataset", action="store_true")
    parser.add_argument("--dp_raw_root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp"))
    parser.add_argument("--dp_dataset_root", type=str, default=None)
    parser.add_argument("--dp_repo_id", type=str, default="local/door_dp")
    parser.add_argument("--dp_task", type=str, default="pull lever door open")
    parser.add_argument("--dp_record_env_id", type=int, default=0)
    parser.add_argument("--dp_record_all_envs", dest="dp_record_all_envs", action="store_true", default=True)
    parser.add_argument("--no_dp_record_all_envs", dest="dp_record_all_envs", action="store_false")
    parser.add_argument("--dp_fps", type=int, default=50)
    parser.add_argument("--dp_policy_checkpoint", type=str, default=None)
    parser.add_argument("--dp_control_env_id", type=int, default=0)
    parser.add_argument("--dp_inference_steps", type=int, default=100)
    parser.add_argument("--dp_action_horizon", type=int, default=None)
    parser.add_argument("--dp_log_path", type=str, default=None)
    parser.add_argument("--dp_log_interval", type=int, default=25)
    parser.add_argument("--no_dp_print", dest="dp_print", action="store_false", default=True)
    parser.add_argument("--plot_pos_error", action="store_true")
    parser.add_argument("--plot_env_id", type=int, default=0)
    parser.add_argument("--plot_interval", type=int, default=1)
    parser.add_argument("--plot_window", type=int, default=600)
    return parser.parse_args()


def build_low_level_args(args):
    use_gpu = args.sim_device.startswith("cuda")
    return SimpleNamespace(
        task="b1z1_door_asset",
        resume=True,
        experiment_name=None,
        run_name=None,
        load_run="",
        checkpoint=args.checkpoint,
        stop_update_goal=False,
        observe_gait_commands=True,
        exptid="b1z1_try",
        debug=False,
        proj_name="b1z1-low",
        resumeid=None,
        headless=args.headless,
        horovod=False,
        rl_device=args.rl_device,
        num_envs=args.num_envs,
        seed=1,
        max_iterations=None,
        stochastic=False,
        use_jit=False,
        record_video=False,
        stand_by=False,
        flat_terrain=True,
        pitch_control=False,
        vel_obs=False,
        rows=None,
        cols=None,
        test=True,
        sim_device=args.sim_device,
        sim_device_id=0,
        physics_engine=gymapi.SIM_PHYSX,
        device="cuda" if use_gpu else "cpu",
        use_gpu=use_gpu,
        use_gpu_pipeline=use_gpu,
        subscenes=0,
        num_threads=4,
    )


def main():
    args = parse_args()
    os.chdir(ROOT)

    DOOR_RUNTIME.update(_load_door_runtime(args.door_cfg))
    if not args.use_all_door_assets:
        _filter_door_runtime_by_names(DOOR_RUNTIME, DEFAULT_DOOR_ASSET_NAMES)
    total_door_assets = len(DOOR_RUNTIME["door_asset_specs"])
    door_asset_count = min(max(1, args.num_envs), total_door_assets)
    for key in ("door_asset_specs", "door_asset_names", "door_bounding_data", "handle_bounding_data"):
        DOOR_RUNTIME[key] = DOOR_RUNTIME[key][:door_asset_count]
    DOOR_RUNTIME["total_door_asset_count"] = total_door_assets
    DOOR_RUNTIME["loaded_door_asset_count"] = door_asset_count
    DOOR_RUNTIME["layout_spacing"] = args.layout_spacing
    DOOR_RUNTIME["robot_x"] = args.robot_x
    DOOR_RUNTIME["robot_y"] = args.robot_y
    DOOR_RUNTIME["robot_z"] = args.robot_z
    DOOR_RUNTIME["robot_yaw"] = args.robot_yaw
    DOOR_RUNTIME["door_x"] = args.door_x
    DOOR_RUNTIME["door_y"] = args.door_y
    DOOR_RUNTIME["door_actor_scale"] = args.door_actor_scale
    DOOR_RUNTIME["robot_y_by_spec"] = _compute_robot_y_by_spec(
        args.robot_y,
        args.door_y,
        DOOR_RUNTIME["door_bounding_data"],
        DOOR_RUNTIME["handle_bounding_data"],
        args.door_actor_scale,
    )
    DOOR_RUNTIME["door_z_offset"] = args.door_z_offset
    DOOR_RUNTIME["box_x"] = args.box_x
    DOOR_RUNTIME["box_y"] = args.box_y
    DOOR_RUNTIME["gripper_stiffness"] = args.gripper_stiffness
    DOOR_RUNTIME["gripper_damping"] = args.gripper_damping
    DOOR_RUNTIME["gripper_joint_friction"] = args.gripper_joint_friction
    DOOR_RUNTIME["handle_spring_stiffness"] = args.handle_spring_stiffness
    DOOR_RUNTIME["handle_spring_damping"] = args.handle_spring_damping
    DOOR_RUNTIME["handle_unlock_ratio"] = args.handle_unlock_ratio
    DOOR_RUNTIME["door_open_resistance"] = args.door_open_resistance
    DOOR_RUNTIME["door_open_damping"] = args.door_open_damping
    DOOR_RUNTIME["door_auto_open_force"] = args.door_auto_open_force
    DOOR_RUNTIME["door_auto_open_sign"] = args.door_auto_open_sign
    DOOR_RUNTIME["door_auto_open_target_ratio"] = args.door_auto_open_target_ratio
    DOOR_RUNTIME["door_joint_friction"][0] = args.door_joint_friction
    DOOR_RUNTIME["door_joint_damping"][0] = args.door_joint_damping
    DOOR_RUNTIME["door_joint_friction"][1] = args.handle_joint_friction
    DOOR_RUNTIME["door_joint_damping"][1] = args.handle_joint_damping
    DOOR_RUNTIME["robot_vhacd_resolution"] = args.robot_vhacd_resolution
    DOOR_RUNTIME["gripper_shape_contact_offset"] = args.gripper_shape_contact_offset
    DOOR_RUNTIME["gripper_shape_rest_offset"] = args.gripper_shape_rest_offset
    DOOR_RUNTIME["gripper_shape_friction"] = args.gripper_shape_friction
    DOOR_RUNTIME["door_vhacd_resolution"] = args.door_vhacd_resolution
    DOOR_RUNTIME["enable_wrist_camera"] = args.enable_wrist_camera
    DOOR_RUNTIME["enable_front_camera"] = args.enable_front_camera
    DOOR_RUNTIME["camera_rgb"] = args.camera_rgb
    DOOR_RUNTIME["camera_depth"] = args.camera_depth
    DOOR_RUNTIME["camera_seg"] = args.camera_seg
    DOOR_RUNTIME["show_seg"] = bool(args.show_seg and not args.headless)
    if args.headless and args.show_seg:
        print(
            "⚠️📷 Headless mode disables OpenCV camera preview windows; DP recording still captures camera tensors.",
            flush=True,
        )
    DOOR_RUNTIME["handle_seg_id"] = args.handle_seg_id
    DOOR_RUNTIME["camera_depth_clip_lower"] = args.camera_depth_clip_lower
    DOOR_RUNTIME["camera_depth_clip_far"] = args.camera_depth_clip_far
    DOOR_RUNTIME["camera_display_scale"] = args.camera_display_scale
    DOOR_RUNTIME["wrist_camera_down_tilt"] = args.wrist_camera_down_tilt
    DOOR_RUNTIME["front_camera_yaw_deg"] = args.front_camera_yaw_deg
    DOOR_RUNTIME["front_camera_pitch_deg"] = args.front_camera_pitch_deg
    DOOR_RUNTIME["front_camera_roll_deg"] = args.front_camera_roll_deg

    low_args = build_low_level_args(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name="b1z1")
    task_registry.register("b1z1_door_asset", ManipLocoDoorAsset, env_cfg, train_cfg, "b1z1")

    env_cfg.sim.substeps = args.sim_substeps
    env_cfg.sim.physx.num_position_iterations = args.sim_position_iterations
    env_cfg.sim.physx.num_velocity_iterations = args.sim_velocity_iterations
    env_cfg.sim.physx.contact_offset = args.sim_contact_offset
    env_cfg.sim.physx.rest_offset = args.sim_rest_offset
    env_cfg.sim.physx.max_depenetration_velocity = args.sim_max_depenetration_velocity

    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = args.episode_length_s
    env_cfg.env.enable_headless_camera = bool(
        args.headless and (args.enable_wrist_camera or args.enable_front_camera or args.record_dp_dataset or args.dp_policy_checkpoint)
    )
    terrain_side = max(2, int(math.ceil(math.sqrt(args.num_envs))))
    env_cfg.terrain.num_rows = terrain_side
    env_cfg.terrain.num_cols = terrain_side
    env_cfg.terrain.height = [0.0, 0.0]
    env_cfg.commands.curriculum = False
    env_cfg.env.observe_gait_commands = True
    env_cfg.commands.ranges.lin_vel_x = [args.speed_min, args.speed_max]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.lin_vel_x_clip = min(env_cfg.commands.lin_vel_x_clip, max(0.01, args.speed_min * 0.5))
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.noise.add_noise = False
    env_cfg.init_state.rand_yaw_range = 0.0
    env_cfg.init_state.origin_perturb_range = 0.0
    env_cfg.init_state.init_vel_perturb_range = 0.0

    env, _ = task_registry.make_env(name="b1z1_door_asset", args=low_args, env_cfg=env_cfg)
    obs = env.get_observations()

    ppo_runner, _, _, _ = task_registry.make_alg_runner(
        log_root=args.log_dir,
        env=env,
        name="b1z1",
        args=low_args,
        train_cfg=train_cfg,
        return_log_dir=True,
    )
    policy = ppo_runner.get_inference_policy(device=env.device, stochastic=False)

    print("Loaded low-level walking policy from:", os.path.join(args.log_dir, f"model_{args.checkpoint}.pt"))
    print("Loaded door assets:", env.door_asset_names)
    print(
        "Door asset selection:",
        {"loaded": DOOR_RUNTIME["loaded_door_asset_count"], "available": DOOR_RUNTIME["total_door_asset_count"]},
    )
    print("Door dimensions in simulator:", _door_dimension_summary(env.door_asset_names, env.door_bounding_data, args.door_actor_scale))
    print("Door actor count:", len(env.door_handles))
    print("Door DOF count:", env.num_door_dofs)
    print("Door body / handle body:", env.door_body_name, env.handle_body_name)
    print(
        "Wrist camera:",
        {
            "enabled": args.enable_wrist_camera,
            "front_enabled": args.enable_front_camera,
            "rgb": args.camera_rgb,
            "depth": args.camera_depth,
            "seg": args.camera_seg,
            "show_seg": DOOR_RUNTIME["show_seg"],
            "display_env": args.camera_env_id,
            "handle_seg_id": args.handle_seg_id,
            "down_tilt": args.wrist_camera_down_tilt,
            "front_yaw_deg": args.front_camera_yaw_deg,
            "front_pitch_deg": args.front_camera_pitch_deg,
            "front_roll_deg": args.front_camera_roll_deg,
            "display_scale": args.camera_display_scale,
            "depth_clip": [args.camera_depth_clip_lower, args.camera_depth_clip_far],
            "cfg": DOOR_RUNTIME["wrist_camera_cfg"],
            "front_cfg": DOOR_RUNTIME["front_camera_cfg"],
        },
    )
    print(
        "Layout offsets:",
        {
            "spacing": args.layout_spacing,
            "robot": [args.robot_x, args.robot_y, args.robot_z, args.robot_yaw],
            "robot_y_by_spec": DOOR_RUNTIME["robot_y_by_spec"],
            "door": [args.door_x, args.door_y, args.door_z_offset],
            "door_actor_scale": args.door_actor_scale,
            "hidden_box": [args.box_x, args.box_y],
        },
    )
    print(
        "Door force settings:",
        {
            "lock_force": DOOR_RUNTIME["door_lock_force"],
            "unlock_handle_ratio": DOOR_RUNTIME["handle_unlock_ratio"],
            "open_resistance": DOOR_RUNTIME["door_open_resistance"],
            "handle_spring": DOOR_RUNTIME["handle_spring_stiffness"],
            "handle_damping": DOOR_RUNTIME["handle_spring_damping"],
        },
    )
    print("Forward command range:", env_cfg.commands.ranges.lin_vel_x)
    print("Yaw command range:", env_cfg.commands.ranges.ang_vel_yaw)
    print("Episode length:", {"seconds": env_cfg.env.episode_length_s, "policy_steps": int(env.max_episode_length)})
    print(
        "Stop rule:",
        {
            "robot_front_offset": args.robot_front_offset,
            "front_to_door_distance": args.stop_distance,
        },
    )
    print(
        "Arm trajectory:",
        {
            "pregrasp_offset": args.pregrasp_offset,
            "grasp_offset": args.grasp_offset,
            "grasp_x_offset": args.grasp_x_offset,
            "grasp_z_offset": args.grasp_z_offset,
            "initial_hold_seconds": args.initial_hold_seconds,
            "approach_steps": args.approach_steps,
            "grasp_steps": args.grasp_steps,
            "grasp_hold_steps": args.grasp_hold_steps,
            "gripper_close_steps": args.gripper_close_steps,
            "handle_rotate_steps": args.handle_rotate_steps,
            "door_pull_steps": args.door_pull_steps,
            "lever_step_size": args.lever_step_size,
            "pull_base_vx": args.pull_base_vx,
            "pull_base_yaw": args.pull_base_yaw,
            "pass_open_angle_deg": args.pass_open_angle_deg,
            "pass_backoff_distance": args.pass_backoff_distance,
            "pass_backoff_vx": args.pass_backoff_vx,
            "pass_front_distance": args.pass_front_distance,
            "pass_through_distance": args.pass_through_distance,
            "pass_align_vx": args.pass_align_vx,
            "pass_forward_vx": args.pass_forward_vx,
            "pass_min_vx": args.pass_min_vx,
            "pass_move_yaw_tol": args.pass_move_yaw_tol,
            "pass_yaw_gain": args.pass_yaw_gain,
            "pass_yaw_clip": args.pass_yaw_clip,
            "pass_align_pos_tol": args.pass_align_pos_tol,
            "pass_yaw_tol": args.pass_yaw_tol,
            "pass_center_y_offset": args.pass_center_y_offset,
            "pass_left_offset": args.pass_left_offset,
            "unidoor_style_pull": args.unidoor_style_pull,
            "handle_rotate_distance": args.handle_rotate_distance,
            "handle_rotate_right_distance": args.handle_rotate_right_distance,
            "handle_rotate_down_distance": args.handle_rotate_down_distance,
            "handle_rotate_angle": args.handle_rotate_angle,
            "handle_arc_radius": args.handle_arc_radius,
            "door_pull_distance": args.door_pull_distance,
            "gripper_open": args.gripper_open,
            "gripper_closed": args.gripper_closed,
            "gripper_close_ratio": args.gripper_close_ratio,
            "gripper_stiffness": args.gripper_stiffness,
            "gripper_damping": args.gripper_damping,
            "gripper_joint_friction": args.gripper_joint_friction,
            "handle_spring_stiffness": args.handle_spring_stiffness,
            "handle_spring_damping": args.handle_spring_damping,
            "door_open_resistance": args.door_open_resistance,
            "door_open_damping": args.door_open_damping,
            "door_joint_friction": args.door_joint_friction,
            "door_joint_damping": args.door_joint_damping,
            "door_auto_open_force": args.door_auto_open_force,
            "door_auto_open_sign": args.door_auto_open_sign,
            "door_auto_open_target_ratio": args.door_auto_open_target_ratio,
            "handle_joint_friction": args.handle_joint_friction,
            "handle_joint_damping": args.handle_joint_damping,
            "robot_vhacd_resolution": args.robot_vhacd_resolution,
            "gripper_shape_contact_offset": args.gripper_shape_contact_offset,
            "gripper_shape_rest_offset": args.gripper_shape_rest_offset,
            "gripper_shape_friction": args.gripper_shape_friction,
            "door_vhacd_resolution": args.door_vhacd_resolution,
            "sim_substeps": args.sim_substeps,
            "sim_position_iterations": args.sim_position_iterations,
            "sim_velocity_iterations": args.sim_velocity_iterations,
            "sim_contact_offset": args.sim_contact_offset,
            "sim_rest_offset": args.sim_rest_offset,
            "sim_max_depenetration_velocity": args.sim_max_depenetration_velocity,
            "external_pos_gain": args.external_pos_gain,
            "external_orn_gain": args.external_orn_gain,
            "forward_ee_roll": args.forward_ee_roll,
            "forward_ee_pitch": args.forward_ee_pitch,
            "gripper_red_axis_rot": args.gripper_red_axis_rot,
            "preview_trajectory_at_spawn": args.preview_trajectory_at_spawn,
        },
    )

    env.reset()
    env.external_ee_goal_control = True
    env.external_pos_gain = args.external_pos_gain
    env.external_orn_gain = args.external_orn_gain
    initial_hold_steps = max(0, int(round(args.initial_hold_seconds / env.dt)))
    print("Direct pregrasp hold:", {"seconds": args.initial_hold_seconds, "policy_steps": initial_hold_steps, "dt": env.dt})
    start_xy = env.root_states[:, :2].clone()
    commanded_vx = torch.zeros(args.num_envs, device=env.device)
    commanded_yaw = torch.zeros(args.num_envs, device=env.device)
    stopped_by_door = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    front_to_door_distance = torch.full((args.num_envs,), float("inf"), device=env.device)
    forward_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(args.num_envs, 1)
    manip_started = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    manip_step = torch.zeros(args.num_envs, device=env.device, dtype=torch.long)
    pass_started = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    pass_backoff_done = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    pass_arm_zero_snapped = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    pass_align_done = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    pass_done = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    pass_side = torch.ones(args.num_envs, device=env.device)
    pass_backoff_start_xy = env.root_states[:, :2].clone()
    pass_pre_xy = env.root_states[:, :2].clone()
    pass_through_xy = env.root_states[:, :2].clone()
    env.freeze_arm_default = torch.ones(args.num_envs, device=env.device, dtype=torch.bool)
    env.freeze_arm_zero = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    env.external_gripper_target = torch.full(
        (args.num_envs, env.cfg.env.num_gripper_joints),
        args.gripper_open,
        device=env.device,
    )
    traj_anchor_pos = env.ee_pos.clone()
    manip_target_pos = env.ee_pos.clone()
    manip_target_quat = env.ee_orn.clone()
    traj_pregrasp_pos = env.ee_pos.clone()
    traj_grasp_pos = env.ee_pos.clone()
    traj_rotate_center = env.ee_pos.clone()
    traj_rotate_pos = env.ee_pos.clone()
    traj_pull_pos = env.ee_pos.clone()
    traj_rotate_tangent_dir = torch.zeros_like(env.ee_pos)
    traj_down_arc_dir = torch.zeros_like(env.ee_pos)
    traj_goal_quat = env.ee_orn.clone()
    gripper_closed_target = args.gripper_open + (args.gripper_closed - args.gripper_open) * args.gripper_close_ratio
    phase_name = [
        "walk_default",
        "initial_hold",
        "approach",
        "grasp",
        "grasp_hold_open",
        "close_gripper",
        "rotate_handle",
        "pull_door",
        "hold",
        "pass_backoff",
        "pass_align",
        "pass_through",
        "pass_done",
    ]
    phase_id = torch.zeros(args.num_envs, device=env.device, dtype=torch.long)
    dp_recorders = {}
    dp_record_success = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    dp_record_closed = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    pos_error_plotter = None
    if args.plot_pos_error:
        try:
            pos_error_plotter = PositionErrorPlotter(env_id=args.plot_env_id, window=args.plot_window)
            print(
                f"Plotting scripted EE position error for env {args.plot_env_id} "
                f"every {max(1, args.plot_interval)} step(s).",
                flush=True,
            )
        except Exception as exc:
            print(f"Warning: failed to start matplotlib position-error plot: {exc}", flush=True)
    if args.record_dp_dataset:
        if RawDoorDPRecorder is None:
            raise RuntimeError(
                "DP raw recording requires door_dp_common.py. It should not require LeRobot in the b1z1 environment."
            )
        if args.headless:
            print(
                "⚠️📷 Headless DP recording needs camera tensors; if the graphics device cannot render cameras, "
                "raw episodes will be discarded with a camera-unavailable message.",
                flush=True,
            )
        if args.dp_record_env_id < 0 or args.dp_record_env_id >= args.num_envs:
            raise ValueError(f"--dp_record_env_id must be in [0, {args.num_envs - 1}]")
        state_names = make_state_feature_names(env.num_dofs, env.num_actions, phase_name)
        record_env_ids = list(range(args.num_envs)) if args.dp_record_all_envs else [args.dp_record_env_id]
        for record_env_id in record_env_ids:
            door_asset_index = int(env.door_asset_indices[record_env_id].detach().cpu().item())
            door_asset_spec = env.door_asset_specs[door_asset_index]
            dp_recorders[record_env_id] = RawDoorDPRecorder(
                raw_root=args.dp_raw_root,
                fps=args.dp_fps,
                state_feature_names=state_names,
                task=args.dp_task,
                metadata={
                    "door_asset_index": door_asset_index,
                    "door_asset_name": env.door_asset_names[door_asset_index],
                    "door_asset_path": door_asset_spec.get("path", ""),
                    "door_cfg": str(args.door_cfg),
                },
            )
        print(
            f"Recording raw Door DP dataset to {args.dp_raw_root} task={args.dp_task!r} "
            f"env_ids={record_env_ids} success_angle_deg={args.pass_open_angle_deg}"
        )
    dp_controller = None
    dp_logger = None
    if args.dp_policy_checkpoint:
        if DoorDPPolicyController is None:
            raise RuntimeError("DP inference requires diffusers and the Door DP model code.")
        if args.dp_control_env_id < 0 or args.dp_control_env_id >= args.num_envs:
            raise ValueError(f"--dp_control_env_id must be in [0, {args.num_envs - 1}]")
        dp_controller = DoorDPPolicyController(
            args.dp_policy_checkpoint,
            device=args.rl_device,
            num_inference_steps=args.dp_inference_steps,
            action_horizon=args.dp_action_horizon,
        )
        print(f"Loaded Door DP policy from {args.dp_policy_checkpoint}")
        print(
            f"Door DP controls only env {args.dp_control_env_id}; other envs keep the scripted target trajectory.",
            flush=True,
        )
        if args.dp_log_path:
            if DoorDPJsonlLogger is None:
                raise RuntimeError("DP logging requires door_dp_common.py")
            dp_logger = DoorDPJsonlLogger(args.dp_log_path)
            print(f"Door DP log: {args.dp_log_path}", flush=True)

    def sample_commands():
        if args.fixed_vx is not None:
            commanded_vx[:] = args.fixed_vx
        else:
            commanded_vx[:] = torch.empty(args.num_envs, device=env.device).uniform_(args.speed_min, args.speed_max)
        if args.fixed_yaw is not None:
            commanded_yaw[:] = args.fixed_yaw
        else:
            commanded_yaw[:] = torch.empty(args.num_envs, device=env.device).uniform_(args.yaw_min, args.yaw_max)

    def update_stop_mask():
        heading = quat_apply(env.root_states[:, 3:7], forward_axis)
        heading_xy = heading[:, :2]
        heading_xy = heading_xy / torch.clamp(torch.norm(heading_xy, dim=-1, keepdim=True), min=1e-6)
        robot_front_xy = env.root_states[:, :2] + heading_xy * args.robot_front_offset
        to_door_xy = env.door_root_state[:, :2] - robot_front_xy
        front_to_door_distance[:] = torch.sum(to_door_xy * heading_xy, dim=-1)
        stopped_by_door[:] |= front_to_door_distance <= args.stop_distance

    def quat_axis(q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1.0
        return quat_apply(q, basis_vec)

    def lerp(a, b, t):
        return a + (b - a) * t.unsqueeze(-1)

    def smoothstep(x):
        x = torch.clamp(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def wrap_angle(x):
        return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi

    def snap_arm_to_zero(env_mask):
        if not torch.any(env_mask):
            return
        env_ids = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        num_gripper = env.cfg.env.num_gripper_joints
        arm_start = env.num_dofs - num_gripper - 6
        arm_end = env.num_dofs - num_gripper
        arm_ids = torch.arange(arm_start, arm_end, device=env.device)
        env.dof_pos[env_ids[:, None], arm_ids.unsqueeze(0)] = 0.0
        env.dof_vel[env_ids[:, None], arm_ids.unsqueeze(0)] = 0.0
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env._full_dof_state_flat))
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_jacobian_tensors(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

    def door_center_xy(env_mask):
        center_xy = env.door_root_state[env_mask, :2].clone()
        center_xy[:, 1] = env.door_root_state[env_mask, 1] + args.pass_center_y_offset
        return center_xy

    def update_pass_state():
        nonlocal pass_backoff_start_xy, pass_pre_xy, pass_through_xy

        door_open_enough = torch.abs(env._door_dof_pos[:, 0]) >= math.radians(args.pass_open_angle_deg)
        newly_passing = door_open_enough & (~pass_started)
        if torch.any(newly_passing):
            pass_started[newly_passing] = True
            pass_backoff_done[newly_passing] = False
            pass_arm_zero_snapped[newly_passing] = False
            pass_align_done[newly_passing] = False
            pass_done[newly_passing] = False
            pass_backoff_start_xy[newly_passing] = env.root_states[newly_passing, :2]
            side = torch.sign(env.root_states[newly_passing, 0] - env.door_root_state[newly_passing, 0])
            side = torch.where(torch.abs(side) < 1e-6, torch.ones_like(side), side)
            pass_side[newly_passing] = side

            pre_xy = door_center_xy(newly_passing)
            pre_xy[:, 0] += side * (args.pass_front_distance + args.robot_front_offset)
            pre_xy[:, 1] -= side * args.pass_left_offset
            pass_pre_xy[newly_passing] = pre_xy

            through_xy = door_center_xy(newly_passing)
            through_xy[:, 0] -= side * args.pass_through_distance
            through_xy[:, 1] -= side * args.pass_left_offset
            pass_through_xy[newly_passing] = through_xy

        if torch.any(pass_done):
            env.freeze_arm_default[pass_done] = False
            env.freeze_arm_zero[pass_done] = True
            env.external_gripper_target[pass_done] = args.gripper_open
            env.curr_ee_goal_cart_world[pass_done] = env.ee_pos[pass_done]
            env.ee_goal_orn_quat[pass_done] = env.ee_orn[pass_done]
            env.ee_goal_orn_delta_rpy[pass_done] = 0.0
            phase_id[pass_done] = 12

        active = pass_started & (~pass_done)
        if not torch.any(active):
            return

        backoff = active & (~pass_backoff_done)
        if torch.any(backoff):
            backoff_dist = torch.norm(env.root_states[backoff, :2] - pass_backoff_start_xy[backoff], dim=-1)
            done_now = backoff_dist >= args.pass_backoff_distance
            backoff_ids = torch.nonzero(backoff, as_tuple=False).squeeze(-1)
            pass_backoff_done[backoff_ids[done_now]] = True
            phase_id[backoff] = 9

        post_backoff = active & pass_backoff_done
        if not torch.any(post_backoff):
            return

        env.freeze_arm_default[post_backoff] = False
        env.freeze_arm_zero[post_backoff] = True
        env.external_gripper_target[post_backoff] = args.gripper_open
        snap_now = post_backoff & (~pass_arm_zero_snapped)
        if torch.any(snap_now):
            snap_arm_to_zero(snap_now)
            pass_arm_zero_snapped[snap_now] = True
        env.curr_ee_goal_cart_world[post_backoff] = env.ee_pos[post_backoff]
        env.ee_goal_orn_quat[post_backoff] = env.ee_orn[post_backoff]
        env.ee_goal_orn_delta_rpy[post_backoff] = 0.0

        to_pre = pass_pre_xy[post_backoff] - env.root_states[post_backoff, :2]
        pre_dist = torch.norm(to_pre, dim=-1)
        reached_pre = pre_dist <= args.pass_align_pos_tol
        post_backoff_ids = torch.nonzero(post_backoff, as_tuple=False).squeeze(-1)
        pass_align_done[post_backoff_ids[reached_pre]] = True

        align = post_backoff & (~pass_align_done)
        pass_through = post_backoff & pass_align_done
        phase_id[align] = 10
        phase_id[pass_through] = 11

        if torch.any(pass_through):
            through_dist = torch.norm(pass_through_xy[pass_through] - env.root_states[pass_through, :2], dim=-1)
            done_now = through_dist <= args.pass_align_pos_tol
            pass_ids = torch.nonzero(pass_through, as_tuple=False).squeeze(-1)
            pass_done[pass_ids[done_now]] = True
            phase_id[pass_ids[done_now]] = 12
        phase_id[pass_done] = 12

    def navigation_command(target_xy, vx):
        to_target = target_xy - env.root_states[:, :2]
        desired_yaw = torch.atan2(to_target[:, 1], to_target[:, 0])
        base_yaw = euler_from_quat(env.root_states[:, 3:7])[2]
        yaw_error = wrap_angle(desired_yaw - base_yaw)
        yaw_cmd = torch.clamp(args.pass_yaw_gain * yaw_error, -args.pass_yaw_clip, args.pass_yaw_clip)
        dist = torch.norm(to_target, dim=-1)
        yaw_abs = torch.abs(yaw_error)
        yaw_scale = 1.0 - torch.clamp(yaw_abs / max(args.pass_move_yaw_tol, 1e-4), 0.0, 1.0)
        min_vx = min(args.pass_min_vx, vx)
        vx_cmd = min_vx + (vx - min_vx) * yaw_scale
        vx_cmd = torch.where(yaw_abs <= args.pass_move_yaw_tol, vx_cmd, torch.zeros_like(vx_cmd))
        vx_cmd = torch.where(dist <= args.pass_align_pos_tol, torch.zeros_like(vx_cmd), vx_cmd)
        return vx_cmd, yaw_cmd

    def forward_ee_quat():
        base_yaw = euler_from_quat(env.root_states[:, 3:7])[2]
        roll = torch.full_like(base_yaw, args.forward_ee_roll)
        pitch = torch.full_like(base_yaw, args.forward_ee_pitch)
        base_quat = quat_from_euler_xyz(roll, pitch, base_yaw)
        red_axis_rot = torch.full_like(base_yaw, args.gripper_red_axis_rot)
        zeros = torch.zeros_like(base_yaw)
        red_axis_quat = quat_from_euler_xyz(red_axis_rot, zeros, zeros)
        return quat_mul(base_quat, red_axis_quat)

    def ee_goal_delta_rpy_from_quat(target_pos, target_quat, env_ids=None):
        goal_roll, goal_pitch, goal_yaw = euler_from_quat(target_quat)
        center = env._get_ee_goal_spherical_center()
        if env_ids is not None:
            center = center[env_ids]
        elif center.shape[0] != target_pos.shape[0]:
            center = center[: target_pos.shape[0]]
        target_cart = target_pos - center
        target_xy_len = torch.norm(target_cart[:, :2], dim=-1)
        target_sphere_pitch = torch.atan2(target_cart[:, 2], target_xy_len)
        default_pitch = -target_sphere_pitch + env.cfg.goal_ee.arm_induced_pitch
        default_yaw = torch.atan2(target_cart[:, 1], target_cart[:, 0])
        return torch.stack(
            (
                wrap_angle(goal_roll - math.pi / 2),
                wrap_angle(goal_pitch - default_pitch),
                wrap_angle(goal_yaw - default_yaw),
            ),
            dim=-1,
        )

    def update_ee_trajectory():
        nonlocal traj_anchor_pos, manip_target_pos, manip_target_quat
        nonlocal traj_pregrasp_pos, traj_grasp_pos, traj_rotate_center, traj_rotate_pos, traj_pull_pos
        nonlocal traj_rotate_tangent_dir, traj_down_arc_dir, traj_goal_quat

        handle_state = env._rigid_body_state[:, env.handle_body_idx, :]
        handle_pos = handle_state[:, :3]
        handle_rot = handle_state[:, 3:7]
        handle_goal = quat_apply(handle_rot, env.goal_pos_offset_tensor) + handle_pos
        approach_dir = env.root_states[:, :3] - handle_goal
        approach_dir[:, 2] = 0.0
        approach_dir = approach_dir / torch.clamp(torch.norm(approach_dir, dim=-1, keepdim=True), min=1e-6)
        pregrasp_pos = handle_goal + approach_dir * args.pregrasp_offset
        handle_arc_radius = max(args.handle_arc_radius, 1e-4)
        grasp_pos = handle_goal + approach_dir * args.grasp_offset
        pregrasp_pos[:, 0] += args.grasp_x_offset
        pregrasp_pos[:, 2] += args.grasp_z_offset
        grasp_pos[:, 0] += args.grasp_x_offset
        grasp_pos[:, 2] += args.grasp_z_offset
        goal_quat = forward_ee_quat()
        handle_rot_axis = torch.zeros(args.num_envs, 3, device=env.device)
        handle_rot_axis[:, 0] = 1.0
        handle_turned_quat = quat_mul(
            traj_goal_quat,
            quat_from_angle_axis(torch.full((args.num_envs,), -args.handle_rotate_angle, device=env.device), handle_rot_axis),
        )
        pull_dir = quat_axis(handle_rot, axis=2)
        pull_dir[:, 2] = 0.0
        fallback_pull_dir = torch.zeros_like(pull_dir)
        fallback_pull_dir[:, 0:2] = approach_dir[:, 0:2]
        pull_dir_norm = torch.norm(pull_dir, dim=-1, keepdim=True)
        pull_dir = torch.where(pull_dir_norm > 1e-4, pull_dir / torch.clamp(pull_dir_norm, min=1e-6), fallback_pull_dir)
        pull_sign = torch.where(
            torch.sum(pull_dir * approach_dir, dim=-1, keepdim=True) < 0.0,
            -torch.ones_like(pull_dir_norm),
            torch.ones_like(pull_dir_norm),
        )
        pull_dir = pull_dir * pull_sign
        rotate_tangent_dir = torch.zeros_like(pull_dir)
        rotate_tangent_dir[:, 1] = args.handle_rotate_right_distance
        rotate_tangent_dir[:, 2] = -args.handle_rotate_down_distance
        rotate_tangent_norm = torch.norm(rotate_tangent_dir, dim=-1, keepdim=True)
        rotate_tangent_dir = rotate_tangent_dir / torch.clamp(rotate_tangent_norm, min=1e-6)
        down_arc_dir = torch.zeros_like(pull_dir)
        down_arc_dir[:, 2] = -1.0
        rotate_offset = torch.zeros_like(grasp_pos)
        rotate_offset[:, 1] = args.handle_rotate_right_distance
        rotate_offset[:, 2] = -args.handle_rotate_down_distance
        rotate_center = grasp_pos
        rotate_pos = grasp_pos + rotate_offset
        pull_pos = rotate_pos + pull_dir * args.door_pull_distance

        newly_started = stopped_by_door & (~manip_started)
        if torch.any(newly_started):
            manip_started[newly_started] = True
            manip_step[newly_started] = 0
            traj_anchor_pos[newly_started] = pregrasp_pos[newly_started]
            manip_target_pos[newly_started] = pregrasp_pos[newly_started]
            manip_target_quat[newly_started] = goal_quat[newly_started]
            traj_pregrasp_pos[newly_started] = pregrasp_pos[newly_started]
            traj_grasp_pos[newly_started] = grasp_pos[newly_started]
            traj_rotate_center[newly_started] = rotate_center[newly_started]
            traj_rotate_pos[newly_started] = rotate_pos[newly_started]
            traj_pull_pos[newly_started] = pull_pos[newly_started]
            traj_rotate_tangent_dir[newly_started] = rotate_tangent_dir[newly_started]
            traj_down_arc_dir[newly_started] = down_arc_dir[newly_started]
            traj_goal_quat[newly_started] = goal_quat[newly_started]

        walking = ~stopped_by_door
        if torch.any(walking):
            env.freeze_arm_default[walking] = True
            env.external_gripper_target[walking] = args.gripper_open
            env.curr_ee_goal_cart_world[walking] = env.ee_pos[walking]
            env.ee_goal_orn_quat[walking] = env.ee_orn[walking]
            env.ee_goal_orn_delta_rpy[walking] = 0.0
            phase_id[walking] = 0

        active = stopped_by_door & manip_started
        if not torch.any(active):
            return

        env.freeze_arm_default[active] = False
        i_end = initial_hold_steps
        a_end = i_end
        g_end = a_end + args.grasp_steps
        h_end = g_end + args.grasp_hold_steps
        c_end = h_end + args.gripper_close_steps
        r_end = c_end + args.handle_rotate_steps
        p_end = r_end + args.door_pull_steps

        initial_hold = active & (manip_step < i_end)
        if torch.any(initial_hold):
            manip_target_pos[initial_hold] = traj_pregrasp_pos[initial_hold]
            manip_target_quat[initial_hold] = traj_goal_quat[initial_hold]
            env.external_gripper_target[initial_hold] = args.gripper_open
            phase_id[initial_hold] = 1

        approach = active & (manip_step >= i_end) & (manip_step < a_end)
        if torch.any(approach):
            denom = max(1, args.approach_steps)
            t = smoothstep(((manip_step[approach] - i_end).to(torch.float) + 1.0) / denom)
            manip_target_pos[approach] = lerp(traj_anchor_pos[approach], traj_pregrasp_pos[approach], t)
            manip_target_quat[approach] = traj_goal_quat[approach]
            env.external_gripper_target[approach] = args.gripper_open
            phase_id[approach] = 2

        grasp = active & (manip_step >= a_end) & (manip_step < g_end)
        if torch.any(grasp):
            denom = max(1, args.grasp_steps)
            t = smoothstep(((manip_step[grasp] - a_end).to(torch.float) + 1.0) / denom)
            manip_target_pos[grasp] = lerp(traj_pregrasp_pos[grasp], traj_grasp_pos[grasp], t)
            manip_target_quat[grasp] = traj_goal_quat[grasp]
            env.external_gripper_target[grasp] = args.gripper_open
            phase_id[grasp] = 3

        grasp_hold = active & (manip_step >= g_end) & (manip_step < h_end)
        if torch.any(grasp_hold):
            manip_target_pos[grasp_hold] = traj_grasp_pos[grasp_hold]
            manip_target_quat[grasp_hold] = traj_goal_quat[grasp_hold]
            env.external_gripper_target[grasp_hold] = args.gripper_open
            phase_id[grasp_hold] = 4

        close_gripper = active & (manip_step >= h_end) & (manip_step < c_end)
        if torch.any(close_gripper):
            manip_target_pos[close_gripper] = traj_grasp_pos[close_gripper]
            manip_target_quat[close_gripper] = traj_goal_quat[close_gripper]
            denom = max(1, args.gripper_close_steps)
            t = smoothstep(((manip_step[close_gripper] - h_end).to(torch.float) + 1.0) / denom)
            gripper_target = args.gripper_open + (gripper_closed_target - args.gripper_open) * t
            env.external_gripper_target[close_gripper] = gripper_target.unsqueeze(-1)
            phase_id[close_gripper] = 5

        rotate = active & (manip_step >= c_end) & (manip_step < r_end)
        if torch.any(rotate):
            env.external_gripper_target[rotate] = gripper_closed_target
            denom = max(1, args.handle_rotate_steps)
            t = smoothstep(((manip_step[rotate] - c_end).to(torch.float) + 1.0) / denom)
            theta = t * args.handle_rotate_angle
            manip_target_pos[rotate] = lerp(traj_grasp_pos[rotate], traj_rotate_pos[rotate], t)
            manip_target_quat[rotate] = quat_mul(
                traj_goal_quat[rotate],
                quat_from_angle_axis(-theta, handle_rot_axis[rotate]),
            )
            phase_id[rotate] = 6

        pull = active & (manip_step >= r_end) & (manip_step < p_end)
        if torch.any(pull):
            env.external_gripper_target[pull] = gripper_closed_target
            denom = max(1, args.door_pull_steps)
            t = smoothstep(((manip_step[pull] - r_end).to(torch.float) + 1.0) / denom)
            if args.unidoor_style_pull:
                pull_step_pos = env.ee_pos[pull] + pull_dir[pull] * args.lever_step_size
                pull_max_pos = traj_rotate_pos[pull] + pull_dir[pull] * args.door_pull_distance
                pull_progress = torch.sum((pull_step_pos - traj_rotate_pos[pull]) * pull_dir[pull], dim=-1)
                manip_target_pos[pull] = torch.where(
                    (pull_progress > args.door_pull_distance).unsqueeze(-1),
                    pull_max_pos,
                    pull_step_pos,
                )
            else:
                manip_target_pos[pull] = lerp(traj_rotate_pos[pull], traj_pull_pos[pull], t)
            manip_target_quat[pull] = handle_turned_quat[pull]
            phase_id[pull] = 7

        hold = active & (manip_step >= p_end)
        if torch.any(hold):
            manip_target_quat[hold] = handle_turned_quat[hold]
            env.external_gripper_target[hold] = gripper_closed_target
            phase_id[hold] = 8

        env.curr_ee_goal_cart_world[active] = manip_target_pos[active]
        env.ee_goal_orn_quat[active] = manip_target_quat[active]
        env.ee_goal_orn_delta_rpy[active] = ee_goal_delta_rpy_from_quat(manip_target_pos, manip_target_quat)[active]
        manip_step[active] += 1

    red_target_geom = gymutil.WireframeSphereGeometry(0.035, 8, 8, None, color=(1, 0, 0))

    def draw_ee_target():
        if getattr(env, "viewer", None) is None:
            return
        for env_i in range(min(args.num_envs, 16)):
            target = env.curr_ee_goal_cart_world[env_i].detach().cpu().tolist()
            pose = gymapi.Transform(gymapi.Vec3(target[0], target[1], target[2]), r=None)
            gymutil.draw_lines(red_target_geom, env.gym, env.viewer, env.envs[env_i], pose)

    def close_dp_recording(env_id, reason):
        recorder = dp_recorders[env_id]
        if bool(dp_record_closed[env_id].item()):
            return
        if bool(dp_record_success[env_id].item()) and recorder.frame_count > 0:
            recorder.save_episode()
            print(
                f"Saved successful raw Door DP episode env={env_id} "
                f"frames={recorder.frame_count} reason={reason}",
                flush=True,
            )
        elif recorder.frame_count == 0 and str(reason).startswith("camera_unavailable"):
            print(
                f"⚠️📷 Discarded raw Door DP episode env={env_id} "
                f"frames=0 reason={reason}: camera images were unavailable, so no DP frames could be saved.",
                flush=True,
            )
        elif bool(dp_record_success[env_id].item()):
            print(
                f"Discarded successful raw Door DP episode env={env_id} "
                f"frames=0 reason={reason}: no DP frames were recorded.",
                flush=True,
            )
        else:
            print(
                f"Discarded failed raw Door DP episode env={env_id} "
                f"frames={recorder.frame_count} reason={reason}: "
                f"door did not reach {args.pass_open_angle_deg} deg",
                flush=True,
            )
        recorder.finalize()
        dp_record_closed[env_id] = True

    dp_record_warned_no_camera = False
    sample_commands()
    for step in range(args.steps):
        if step % args.resample_interval == 0:
            sample_commands()
        if args.preview_trajectory_at_spawn:
            stopped_by_door[:] = True
            front_to_door_distance[:] = args.stop_distance
        else:
            update_stop_mask()
        update_ee_trajectory()
        update_pass_state()

        pass_backoff = pass_started & (~pass_backoff_done) & (~pass_done)
        pass_align = pass_started & pass_backoff_done & (~pass_align_done) & (~pass_done)
        pass_through = pass_started & pass_backoff_done & pass_align_done & (~pass_done)
        align_vx, align_yaw = navigation_command(pass_pre_xy, args.pass_align_vx)
        through_vx, through_yaw = navigation_command(pass_through_xy, args.pass_forward_vx)
        pull_base = (phase_id == 7) & (~pass_started)
        env.commands[:, 0] = torch.where(
            pass_backoff,
            torch.full_like(commanded_vx, args.pass_backoff_vx),
            torch.where(
                pass_align,
                align_vx,
                torch.where(
                    pass_through,
                    through_vx,
                    torch.where(
                        pass_done,
                        torch.zeros_like(commanded_vx),
                        torch.where(
                            pull_base,
                            torch.full_like(commanded_vx, args.pull_base_vx),
                            torch.where(stopped_by_door, torch.zeros_like(commanded_vx), commanded_vx),
                        ),
                    ),
                ),
            ),
        )
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = torch.where(
            pass_backoff,
            torch.zeros_like(commanded_yaw),
            torch.where(
                pass_align,
                align_yaw,
                torch.where(
                    pass_through,
                    through_yaw,
                    torch.where(
                        pass_done,
                        torch.zeros_like(commanded_yaw),
                        torch.where(
                            pull_base,
                            torch.full_like(commanded_yaw, args.pull_base_yaw),
                            torch.where(stopped_by_door, torch.zeros_like(commanded_yaw), commanded_yaw),
                        ),
                    ),
                ),
            ),
        )
        if dp_controller is not None:
            dp_env_id = args.dp_control_env_id
            dp_camera_images = env.capture_wrist_camera_images()
            mask_rgb, masked_depth_rgb, front_mask_rgb, front_masked_depth_rgb = dp_image_inputs_from_camera_tensors(dp_camera_images, dp_env_id)
            if mask_rgb is not None and masked_depth_rgb is not None:
                dp_action = dp_controller.act(
                    get_door_dp_state(env, phase_id, phase_name, dp_env_id),
                    mask_rgb,
                    masked_depth_rgb,
                    front_mask_rgb,
                    front_masked_depth_rgb,
                )
                apply_door_dp_action(env, dp_action, dp_env_id, ee_goal_delta_rpy_from_quat)
                if make_door_dp_log_record is not None:
                    dp_extra = {
                        "stopped_by_door": bool(stopped_by_door[dp_env_id].item()),
                        "front_to_door_distance": float(front_to_door_distance[dp_env_id].item()),
                        "pass_started": bool(pass_started[dp_env_id].item()),
                        "pass_done": bool(pass_done[dp_env_id].item()),
                        "door_open_stage": bool(env.open_door_stage[dp_env_id].item()),
                        "handle_open_ratio": float(env.handle_open_ratio[dp_env_id].item()),
                        "door_open_ratio": float(env.door_open_ratio[dp_env_id].item()),
                    }
                    dp_record = make_door_dp_log_record(
                        env,
                        step,
                        dp_action,
                        dp_env_id,
                        phase_id=phase_id,
                        phase_names=phase_name,
                        extra=dp_extra,
                    )
                    if dp_logger is not None:
                        dp_logger.write(dp_record)
                    if args.dp_print and step % max(1, args.dp_log_interval) == 0:
                        print_door_dp_log_record(dp_record)
        actions = policy(obs.detach(), hist_encoding=True)
        obs, _, _, _, dones, infos = env.step(actions.detach())
        if pos_error_plotter is not None and step % max(1, args.plot_interval) == 0:
            pos_error_plotter.update(step, env.curr_ee_goal_cart_world, env.ee_pos, env.ee_goal_orn_quat, env.ee_orn)
        camera_images = env.capture_wrist_camera_images()
        env.show_wrist_seg(camera_images, args.camera_env_id)
        if dp_recorders:
            dp_record_success |= pass_started
            for env_id, recorder in dp_recorders.items():
                if bool(dp_record_closed[env_id].item()):
                    continue
                mask_rgb, masked_depth_rgb, front_mask_rgb, front_masked_depth_rgb = dp_image_inputs_from_camera_tensors(camera_images, env_id)
                should_close = bool(pass_done[env_id].item()) or step == args.steps - 1
                if mask_rgb is None or masked_depth_rgb is None:
                    if not dp_record_warned_no_camera:
                        print(
                            "⚠️📷 Camera unavailable: skipped DP frame because wrist camera mask/depth images are missing. "
                            "No raw DP frame can be saved until camera tensors are available.",
                            flush=True,
                        )
                        dp_record_warned_no_camera = True
                    if should_close:
                        close_dp_recording(
                            env_id,
                            "camera_unavailable_" + ("pass_done" if bool(pass_done[env_id].item()) else "max_steps"),
                        )
                    continue
                recorder.add_frame(
                    get_door_dp_state(env, phase_id, phase_name, env_id),
                    mask_rgb,
                    masked_depth_rgb,
                    get_door_dp_action(env, env_id),
                    int(phase_id[env_id].detach().cpu().item()),
                    front_mask_rgb=front_mask_rgb,
                    front_masked_depth_rgb=front_masked_depth_rgb,
                    replay_snapshot=(
                        make_door_dp_replay_snapshot(env, env_id)
                        if make_door_dp_replay_snapshot is not None
                        else None
                    ),
                )
                if should_close:
                    close_dp_recording(env_id, "pass_done" if bool(pass_done[env_id].item()) else "max_steps")
        env.draw_wrist_camera_axes(args.camera_axis_scale, args.camera_axis_thickness)
        draw_ee_target()
        if torch.any(dones):
            if dp_recorders:
                for env_id in dp_recorders:
                    if bool(dones[env_id].item()):
                        close_dp_recording(env_id, "env_reset")
            stopped_by_door[dones] = False
            manip_started[dones] = False
            manip_step[dones] = 0
            pass_started[dones] = False
            pass_backoff_done[dones] = False
            pass_arm_zero_snapped[dones] = False
            pass_align_done[dones] = False
            pass_done[dones] = False
            env.freeze_arm_default[dones] = True
            env.freeze_arm_zero[dones] = False
            env.external_gripper_target[dones] = args.gripper_open
        if step % 30 == 0:
            shown_phase = [phase_name[int(i)] for i in phase_id[: min(args.num_envs, 4)].detach().cpu().tolist()]
            traveled = torch.norm(env.root_states[:, :2] - start_xy, dim=-1)
            print(
                _format_step_log(
                    step,
                    {
                    "command_x": env.commands[: min(args.num_envs, 4), 0].detach().cpu().tolist(),
                    "command_yaw": env.commands[: min(args.num_envs, 4), 2].detach().cpu().tolist(),
                    "base_lin_vel_x": env.base_lin_vel[: min(args.num_envs, 4), 0].detach().cpu().tolist(),
                    "base_ang_vel_z": env.base_ang_vel[: min(args.num_envs, 4), 2].detach().cpu().tolist(),
                    "base_height": env.root_states[: min(args.num_envs, 4), 2].detach().cpu().tolist(),
                    "base_xy": env.root_states[: min(args.num_envs, 4), :2].detach().cpu().tolist(),
                    "door_xy": env.door_root_state[: min(args.num_envs, 4), :2].detach().cpu().tolist(),
                    "front_to_door_distance": front_to_door_distance[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "stopped_by_door": stopped_by_door[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "arm_phase": shown_phase,
                    "pass_started": pass_started[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "pass_backoff_done": pass_backoff_done[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "pass_arm_zero_snapped": pass_arm_zero_snapped[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "pass_backoff_dist": torch.norm(
                        env.root_states[: min(args.num_envs, 4), :2] - pass_backoff_start_xy[: min(args.num_envs, 4)],
                        dim=-1,
                    ).detach().cpu().tolist(),
                    "pass_align_done": pass_align_done[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "pass_done": pass_done[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "pass_pre_xy": pass_pre_xy[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "pass_through_xy": pass_through_xy[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "ee_target": env.curr_ee_goal_cart_world[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "ee_pos": env.ee_pos[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "ee_z_error": (
                        env.curr_ee_goal_cart_world[: min(args.num_envs, 4), 2] - env.ee_pos[: min(args.num_envs, 4), 2]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "door_dof": env._door_dof_pos[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "door_open_deg": torch.rad2deg(env._door_dof_pos[: min(args.num_envs, 4), 0]).detach().cpu().tolist(),
                    "door_open_stage": env.open_door_stage[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "door_open_ratio": env.door_open_ratio[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "handle_open_ratio": env.handle_open_ratio[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "episode_length": env.episode_length_buf[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "time_out": env.time_out_buf[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "traveled_xy": traveled[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "reset": dones[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    },
                )
            )
        if dp_recorders and torch.all(dp_record_closed[list(dp_recorders.keys())]):
            saved_count = int(torch.sum(dp_record_success[list(dp_recorders.keys())]).detach().cpu().item())
            print(f"Finished raw Door DP recording: saved_successful={saved_count}/{len(dp_recorders)}")
            break
    if dp_logger is not None:
        dp_logger.close()


if __name__ == "__main__":
    main()
