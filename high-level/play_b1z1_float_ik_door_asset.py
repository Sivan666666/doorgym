import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = ROOT / "high-level"
LOW_LEVEL_ROOT = ROOT / "low-level"
if str(HIGH_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_ROOT))
if str(LOW_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOW_LEVEL_ROOT))

from isaacgym import gymapi, gymtorch  # noqa: E402
from isaacgym.torch_utils import (  # noqa: E402
    euler_from_quat,
    orientation_error,
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
)
import torch  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg  # noqa: E402
from legged_gym.utils.terrain import Terrain  # noqa: E402


DEFAULT_DOOR_ASSET_NAMES = (
    "99650089960001",
    "99650089960006",
    "99655039960001",
    "99655039960006",
)
DOOR_RUNTIME = {}

DEFAULT_ARM_POSE = {
    "z1_waist": 0.0,
    "z1_shoulder": 1.48,
    "z1_elbow": -0.63,
    "z1_wrist_angle": -0.84,
    "z1_forearm_roll": 0.0,
    "z1_wrist_rotate": 1.57,
    "z1_jointGripper": -0.785,
}


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
        "door_joint_friction": list(env_cfg.get("doorJointFriction", [6.0, 18.0])),
        "door_joint_damping": list(env_cfg.get("doorJointDamping", [3.0, 10.0])),
        "door_joint_effort": list(env_cfg.get("doorJointEffort", [200.0, 200.0])),
    }


def _filter_door_runtime_by_names(runtime, names):
    selected = []
    available = {spec["name"]: i for i, spec in enumerate(runtime["door_asset_specs"])}
    for name in names:
        if name in available:
            selected.append(available[name])
    if not selected:
        return
    for key in ("door_asset_specs", "door_asset_names", "door_bounding_data", "handle_bounding_data"):
        runtime[key] = [runtime[key][i] for i in selected]


def _compute_robot_y_by_spec(robot_y, door_y, door_bounding_data, handle_bounding_data, door_actor_scale=1.0):
    robot_y_by_spec = []
    for handle_bounds in handle_bounding_data:
        handle_center_y = 0.5 * (float(handle_bounds["handle_min"][1]) + float(handle_bounds["handle_max"][1]))
        robot_y_by_spec.append(float(robot_y + door_y - door_actor_scale * handle_center_y))
    return robot_y_by_spec


def smoothstep(x):
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return a + (b - a) * t.unsqueeze(-1)


def quat_axis(q, axis=0):
    basis = torch.zeros(q.shape[0], 3, device=q.device)
    basis[:, axis] = 1.0
    return quat_apply(q, basis)


def make_sim(args, terrain_cfg):
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = args.dt
    sim_params.substeps = args.sim_substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.sim_device.startswith("cuda")
    sim_params.physx.use_gpu = args.sim_device.startswith("cuda")
    sim_params.physx.num_position_iterations = args.sim_position_iterations
    sim_params.physx.num_velocity_iterations = args.sim_velocity_iterations
    sim_params.physx.contact_offset = args.sim_contact_offset
    sim_params.physx.rest_offset = args.sim_rest_offset
    sim_params.physx.max_depenetration_velocity = args.sim_max_depenetration_velocity

    sim_device_id = int(args.sim_device.split(":")[-1]) if args.sim_device.startswith("cuda") else 0
    graphics_id = -1 if args.headless else args.graphics_device_id
    sim = gym.create_sim(sim_device_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")
    terrain = Terrain(terrain_cfg)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = terrain.vertices.shape[0]
    tm_params.nb_triangles = terrain.triangles.shape[0]
    tm_params.transform.p.x = -terrain.cfg.border_size
    tm_params.transform.p.y = -terrain.cfg.border_size
    tm_params.transform.p.z = 0.0
    tm_params.static_friction = terrain_cfg.static_friction
    tm_params.dynamic_friction = terrain_cfg.dynamic_friction
    tm_params.restitution = terrain_cfg.restitution
    print("Adding trimesh to simulation...")
    gym.add_triangle_mesh(sim, terrain.vertices.flatten(order="C"), terrain.triangles.flatten(order="C"), tm_params)
    print("Trimesh added")
    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(4.0, -3.0, 1.5), gymapi.Vec3(2.5, 0.0, 0.7))
    return gym, sim, viewer, terrain


class FloatIKDoorWorld:
    def __init__(self, args):
        global DOOR_RUNTIME
        self.args = args
        self.device = torch.device(args.sim_device if torch.cuda.is_available() or not args.sim_device.startswith("cuda") else "cpu")
        self.env_cfg = B1Z1RoughCfg
        self.env_cfg.terrain.num_rows = max(2, int(math.ceil(math.sqrt(args.num_envs))))
        self.env_cfg.terrain.num_cols = max(2, int(math.ceil(math.sqrt(args.num_envs))))
        self.env_cfg.terrain.height = [0.0, 0.0]
        self.env_cfg.domain_rand.randomize_friction = False
        self.env_cfg.init_state.rand_yaw_range = 0.0
        self.env_cfg.init_state.origin_perturb_range = 0.0
        self.env_cfg.init_state.init_vel_perturb_range = 0.0
        self.gym, self.sim, self.viewer, self.terrain = make_sim(args, self.env_cfg.terrain)
        DOOR_RUNTIME.clear()
        DOOR_RUNTIME.update(_load_door_runtime(args.door_cfg))
        if not args.use_all_door_assets:
            _filter_door_runtime_by_names(DOOR_RUNTIME, DEFAULT_DOOR_ASSET_NAMES)
        DOOR_RUNTIME["layout_spacing"] = args.layout_spacing
        DOOR_RUNTIME["robot_x"] = args.robot_x
        DOOR_RUNTIME["robot_y"] = args.robot_y
        DOOR_RUNTIME["robot_z"] = args.robot_z
        DOOR_RUNTIME["robot_yaw"] = args.robot_yaw
        DOOR_RUNTIME["door_x"] = args.door_x
        DOOR_RUNTIME["door_y"] = args.door_y
        DOOR_RUNTIME["door_actor_scale"] = args.door_actor_scale
        DOOR_RUNTIME["door_z_offset"] = args.door_z_offset
        DOOR_RUNTIME["box_x"] = args.box_x
        DOOR_RUNTIME["box_y"] = args.box_y
        DOOR_RUNTIME["handle_seg_id"] = args.handle_seg_id
        DOOR_RUNTIME["handle_spring_stiffness"] = args.handle_spring_stiffness
        DOOR_RUNTIME["handle_spring_damping"] = args.handle_spring_damping
        DOOR_RUNTIME["handle_unlock_ratio"] = args.handle_unlock_ratio
        DOOR_RUNTIME["door_open_resistance"] = args.door_open_resistance
        DOOR_RUNTIME["door_open_damping"] = args.door_open_damping
        DOOR_RUNTIME["door_joint_friction"][0] = args.door_joint_friction
        DOOR_RUNTIME["door_joint_damping"][0] = args.door_joint_damping
        DOOR_RUNTIME["door_joint_friction"][1] = args.handle_joint_friction
        DOOR_RUNTIME["door_joint_damping"][1] = args.handle_joint_damping
        DOOR_RUNTIME["door_vhacd_resolution"] = args.door_vhacd_resolution
        keep = min(max(1, args.num_envs), len(DOOR_RUNTIME["door_asset_specs"]))
        for key in ("door_asset_specs", "door_asset_names", "door_bounding_data", "handle_bounding_data"):
            DOOR_RUNTIME[key] = DOOR_RUNTIME[key][:keep]
        self.num_envs = min(args.num_envs, keep)
        self.runtime = DOOR_RUNTIME
        self.robot_y_by_spec = _compute_robot_y_by_spec(
            args.robot_y,
            args.door_y,
            DOOR_RUNTIME["door_bounding_data"],
            DOOR_RUNTIME["handle_bounding_data"],
            args.door_actor_scale,
        )
        DOOR_RUNTIME["robot_y_by_spec"] = self.robot_y_by_spec
        self._load_assets()
        self._create_envs()
        self._init_tensors()
        self.reset()

    def _load_assets(self):
        robot_opts = gymapi.AssetOptions()
        robot_opts.default_dof_drive_mode = self.env_cfg.asset.default_dof_drive_mode
        # Keep fixed visual links in the float diagnostic so the loaded robot is visually
        # identical to the URDF instead of looking like only the tiny dummy base survived.
        robot_opts.collapse_fixed_joints = False
        robot_opts.replace_cylinder_with_capsule = self.env_cfg.asset.replace_cylinder_with_capsule
        robot_opts.flip_visual_attachments = self.env_cfg.asset.flip_visual_attachments
        robot_opts.fix_base_link = False
        robot_opts.density = self.env_cfg.asset.density
        robot_opts.angular_damping = self.env_cfg.asset.angular_damping
        robot_opts.linear_damping = self.env_cfg.asset.linear_damping
        robot_opts.max_angular_velocity = self.env_cfg.asset.max_angular_velocity
        robot_opts.max_linear_velocity = self.env_cfg.asset.max_linear_velocity
        robot_opts.armature = self.env_cfg.asset.armature
        robot_opts.thickness = self.env_cfg.asset.thickness
        robot_opts.disable_gravity = self.env_cfg.asset.disable_gravity
        robot_opts.use_mesh_materials = True
        robot_opts.vhacd_enabled = True
        robot_opts.vhacd_params = gymapi.VhacdParams()
        robot_opts.vhacd_params.resolution = self.args.robot_vhacd_resolution
        robot_root = HIGH_LEVEL_ROOT / "data" / "asset" / "b1z1-base-arm-from-lowlevel"
        robot_file = "urdf/b1z1_base_arm.urdf"
        self.robot_asset = self.gym.load_asset(self.sim, str(robot_root), robot_file, robot_opts)
        if self.robot_asset is None:
            raise RuntimeError(f"Failed to load {robot_root / robot_file}")
        self.robot_dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.robot_body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.robot_body_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)
        self.robot_dof_dict = self.gym.get_asset_dof_dict(self.robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        print("------------------------------------------------------")
        print("float_robot_asset:", robot_root / robot_file)
        print("float_robot_dofs:", self.robot_dof_names)
        print("float_robot_bodies:", self.robot_body_names)
        print("------------------------------------------------------")
        self.arm_dof_ids = torch.tensor([self.robot_dof_dict[name] for name in self.robot_dof_names if name != "z1_jointGripper"], device=self.device)
        self.gripper_dof_id = self.robot_dof_dict["z1_jointGripper"]
        self.ee_body_idx = self.robot_body_dict["ee_gripper_link"]

        box_opts = gymapi.AssetOptions()
        box_opts.density = 1000
        box_opts.fix_base_link = False
        box_opts.disable_gravity = False
        self.box_asset = self.gym.create_box(
            self.sim,
            self.env_cfg.box.box_size,
            self.env_cfg.box.box_size,
            self.env_cfg.box.box_size,
            box_opts,
        )

        self._load_door_assets()

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

            shape_props = self.gym.get_asset_rigid_shape_properties(door_asset)
            for prop in shape_props:
                prop.friction = 2.0
            self.gym.set_asset_rigid_shape_properties(door_asset, shape_props)

        self.num_door_dofs = self.gym.get_asset_dof_count(self.door_asset_list[0])
        self.num_door_bodies = self.gym.get_asset_rigid_body_count(self.door_asset_list[0])
        print("door_asset_names:", self.runtime["door_asset_names"])
        print("door_asset_body_names:", self.door_asset_body_names[0])
        print("door_asset_dof_names:", self.door_asset_dof_names[0])
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

    def _create_envs(self):
        self.envs = []
        self.robot_handles = []
        self.box_handles = []
        self.door_handles = []
        self.door_actor_spec_ids = []
        lower = gymapi.Vec3(0.0, 0.0, 0.0)
        upper = gymapi.Vec3(0.0, 0.0, 0.0)
        cols = max(1, int(math.ceil(math.sqrt(self.num_envs))))
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        ids = torch.arange(self.num_envs, device=self.device)
        rows = torch.div(ids, cols, rounding_mode="floor")
        cols_idx = ids % cols
        self.env_origins[:, 0] = rows.to(torch.float) * DOOR_RUNTIME["layout_spacing"]
        self.env_origins[:, 1] = cols_idx.to(torch.float) * DOOR_RUNTIME["layout_spacing"]
        for env_id in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, cols)
            self.envs.append(env)
            origin = self.env_origins[env_id]
            spec_id = env_id % len(self.door_asset_list)

            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(
                float(origin[0] + DOOR_RUNTIME["robot_x"]),
                float(origin[1] + self.robot_y_by_spec[spec_id]),
                float(origin[2] + DOOR_RUNTIME["robot_z"]),
            )
            robot_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, DOOR_RUNTIME["robot_yaw"])
            robot_handle = self.gym.create_actor(env, self.robot_asset, robot_pose, "robot_dog", env_id, self.env_cfg.asset.self_collisions, 0)
            self.robot_handles.append(robot_handle)
            self._set_robot_dof_props(env, robot_handle)

            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(
                float(origin[0] + DOOR_RUNTIME["box_x"]),
                float(origin[1] + DOOR_RUNTIME["box_y"]),
                float(origin[2] + self.env_cfg.box.box_env_origins_z),
            )
            box_handle = self.gym.create_actor(env, self.box_asset, box_pose, "box", env_id, self.env_cfg.asset.self_collisions, 0)
            self.box_handles.append(box_handle)

            self._create_door_actor(env, env_id)

        self.num_actors_per_env = 3
        self.dof_per_env = self.num_robot_dofs + self.num_door_dofs
        self.bodies_per_env = self.num_robot_bodies + 1 + self.num_door_bodies
        self.robot_actor_ids = torch.arange(0, self.num_envs * self.num_actors_per_env, self.num_actors_per_env, device=self.device, dtype=torch.int32)
        self.box_actor_ids = self.robot_actor_ids + 1
        self.door_actor_ids = self.robot_actor_ids + 2
        self.door_asset_indices = torch.tensor(self.door_actor_spec_ids, device=self.device, dtype=torch.long)

    def _set_robot_dof_props(self, env, handle):
        props = self.gym.get_asset_dof_properties(self.robot_asset)
        props["driveMode"][:] = gymapi.DOF_MODE_POS
        props["stiffness"][:].fill(self.args.arm_stiffness)
        props["damping"][:].fill(self.args.arm_damping)
        props["stiffness"][self.gripper_dof_id] = self.args.gripper_stiffness
        props["damping"][self.gripper_dof_id] = self.args.gripper_damping
        if "friction" in props.dtype.names:
            props["friction"][self.gripper_dof_id] = self.args.gripper_joint_friction
        self.gym.set_actor_dof_properties(env, handle, props)

    def _set_door_dof_props(self, env, handle, asset):
        props = self.gym.get_asset_dof_properties(asset)
        props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        n = min(len(props), len(self.runtime["door_joint_damping"]))
        props["damping"][:n] = np.asarray(self.runtime["door_joint_damping"][:n], dtype=np.float32)
        if "friction" in props.dtype.names:
            props["friction"][:n] = np.asarray(self.runtime["door_joint_friction"][:n], dtype=np.float32)
        if "effort" in props.dtype.names:
            props["effort"][:n] = np.asarray(self.runtime["door_joint_effort"][:n], dtype=np.float32)
        if len(props["upper"]) >= 2:
            props["upper"][1] = min(float(props["upper"][1]), math.pi / 4)
        self.gym.set_actor_dof_properties(env, handle, props)

    def _init_tensors(self):
        self.gym.prepare_sim(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(self.num_envs, self.num_actors_per_env, 13).to(self.device)
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, self.dof_per_env, 2).to(self.device)
        self.rigid_body_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.bodies_per_env, 13).to(self.device)
        self.jacobian = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "robot_dog")).to(self.device)
        self.force_targets = torch.zeros(self.num_envs, self.dof_per_env, device=self.device)
        self.pos_targets = torch.zeros(self.num_envs, self.dof_per_env, device=self.device)
        self.default_dof_pos = torch.zeros(self.num_robot_dofs, device=self.device)
        for name, value in DEFAULT_ARM_POSE.items():
            if name in self.robot_dof_dict:
                self.default_dof_pos[self.robot_dof_dict[name]] = float(value)
        goal_offsets = [[self.args.door_actor_scale * float(v) for v in item["goal_pos"]] for item in self.runtime["handle_bounding_data"]]
        self.goal_pos_offset_tensor = torch.tensor(goal_offsets, device=self.device, dtype=torch.float32)[self.door_asset_indices]
        self.door_hinge_limits_lower = torch.stack([self.door_asset_dof_limits_lower[i][0] for i in self.door_actor_spec_ids]).to(self.device)
        self.door_hinge_limits_upper = torch.stack([self.door_asset_dof_limits_upper[i][0] for i in self.door_actor_spec_ids]).to(self.device)
        self.handle_limits_lower = torch.stack([self.door_asset_dof_limits_lower[i][1] for i in self.door_actor_spec_ids]).to(self.device)
        self.handle_limits_upper = torch.stack([self.door_asset_dof_limits_upper[i][1] for i in self.door_actor_spec_ids]).to(self.device)
        self.handle_unlock_threshold = self.args.handle_unlock_ratio * (self.handle_limits_upper - self.handle_limits_lower)
        self.open_door_stage = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self):
        self.refresh()
        self.initial_root_states = self.root_state_tensor.clone()
        self.initial_door_root_states = self.root_state_tensor[:, 2].clone()
        self.dof_state_tensor[:, : self.num_robot_dofs, 0] = self.default_dof_pos
        self.dof_state_tensor[:, : self.num_robot_dofs, 1] = 0.0
        self.dof_state_tensor[:, self.num_robot_dofs :, :] = 0.0
        self.root_state_tensor[:, :, 7:13] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
        self.pos_targets[:, : self.num_robot_dofs] = self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_targets))
        self.refresh()

    def refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.robot_root = self.root_state_tensor[:, 0]
        self.door_root = self.root_state_tensor[:, 2]
        self.robot_dof_pos = self.dof_state_tensor[:, : self.num_robot_dofs, 0]
        self.door_dof_pos = self.dof_state_tensor[:, self.num_robot_dofs :, 0]
        self.door_dof_vel = self.dof_state_tensor[:, self.num_robot_dofs :, 1]
        self.ee_pos = self.rigid_body_tensor[:, self.ee_body_idx, :3]
        self.ee_quat = self.rigid_body_tensor[:, self.ee_body_idx, 3:7]
        self.handle_body_idx = self.num_robot_bodies + 1 + (self.num_door_bodies - 1)
        self.handle_pos = self.rigid_body_tensor[:, self.handle_body_idx, :3]
        self.handle_quat = self.rigid_body_tensor[:, self.handle_body_idx, 3:7]

    def door_torques(self):
        torques = torch.zeros(self.num_envs, self.num_door_dofs, device=self.device)
        if self.num_door_dofs < 2:
            return torques
        door_angle = self.door_dof_pos[:, 0]
        handle_angle = self.door_dof_pos[:, 1] - self.handle_limits_lower
        unlock_now = (handle_angle >= self.handle_unlock_threshold) | (torch.abs(door_angle) > 0.01)
        self.open_door_stage |= unlock_now
        hinge_range = torch.clamp(torch.maximum(torch.abs(self.door_hinge_limits_upper), torch.abs(self.door_hinge_limits_lower)), min=1e-3)
        auto_active = torch.abs(door_angle) < self.args.door_auto_open_target_ratio * hinge_range
        auto_torque = torch.where(auto_active, torch.full_like(door_angle, self.args.door_auto_open_force * self.args.door_auto_open_sign), torch.zeros_like(door_angle))
        torques[:, 0] = torch.where(
            self.open_door_stage,
            auto_torque - self.args.door_open_resistance * door_angle - self.args.door_open_damping * self.door_dof_vel[:, 0],
            torch.zeros_like(door_angle),
        )
        torques[:, 1] = -self.args.handle_spring_stiffness * handle_angle - self.args.handle_spring_damping * self.door_dof_vel[:, 1]
        return torques

    def enforce_locked_door_state(self):
        locked = ~self.open_door_stage
        if not torch.any(locked):
            return
        self.dof_state_tensor[locked, self.num_robot_dofs + 0, 0] = 0.0
        self.dof_state_tensor[locked, self.num_robot_dofs + 0, 1] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor))

    def set_base_pose(self, base_xy, yaw):
        self.root_state_tensor[:, 0, 0:2] = base_xy
        self.root_state_tensor[:, 0, 2] = self.args.robot_z
        self.root_state_tensor[:, 0, 3:7] = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            yaw,
        )
        self.root_state_tensor[:, 0, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(self.robot_actor_ids),
            self.num_envs,
        )

    def solve_ik_targets(self, target_pos, target_quat, gripper_target):
        dpos = target_pos - self.ee_pos
        target_quat = target_quat / torch.clamp(torch.norm(target_quat, dim=-1, keepdim=True), min=1e-6)
        current_quat = self.ee_quat / torch.clamp(torch.norm(self.ee_quat, dim=-1, keepdim=True), min=1e-6)
        drot = orientation_error(target_quat, current_quat)
        dpose = torch.cat([self.args.ik_pos_gain * dpos, self.args.ik_orn_gain * drot], dim=-1).unsqueeze(-1)
        j = self.jacobian[:, self.ee_body_idx, :6, : self.num_robot_dofs][:, :, self.arm_dof_ids]
        jt = torch.transpose(j, 1, 2)
        damping = (self.args.ik_damping ** 2) * torch.eye(6, device=self.device).unsqueeze(0)
        dq = torch.bmm(jt, torch.linalg.solve(torch.bmm(j, jt) + damping, dpose)).squeeze(-1)
        dq = torch.clamp(dq, -self.args.ik_step_clip, self.args.ik_step_clip)
        next_pos = self.robot_dof_pos.clone()
        next_pos[:, self.arm_dof_ids] = self.robot_dof_pos[:, self.arm_dof_ids] + dq
        next_pos[:, self.gripper_dof_id] = gripper_target
        self.pos_targets[:, : self.num_robot_dofs] = next_pos
        self.pos_targets[:, self.num_robot_dofs :] = self.door_dof_pos

    def simulate_step(self):
        self.force_targets.zero_()
        self.force_targets[:, self.num_robot_dofs :] = self.door_torques()
        self.enforce_locked_door_state()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_targets))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.force_targets))
        self.gym.simulate(self.sim)
        if self.args.sim_device == "cpu":
            self.gym.fetch_results(self.sim, True)
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        self.refresh()

    def handle_goal(self):
        return quat_apply(self.handle_quat, self.goal_pos_offset_tensor) + self.handle_pos

    def forward_ee_quat(self):
        yaw = euler_from_quat(self.robot_root[:, 3:7])[2]
        base_quat = quat_from_euler_xyz(
            torch.full_like(yaw, self.args.forward_ee_roll),
            torch.full_like(yaw, self.args.forward_ee_pitch),
            yaw,
        )
        red_axis_quat = quat_from_euler_xyz(
            torch.full_like(yaw, self.args.gripper_red_axis_rot),
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
        )
        return quat_mul(base_quat, red_axis_quat)

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def parse_args():
    parser = argparse.ArgumentParser(description="Float-base B1Z1/Z1 IK door-opening diagnostic script.")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2200)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--sim_substeps", type=int, default=2)
    parser.add_argument("--sim_position_iterations", type=int, default=12)
    parser.add_argument("--sim_velocity_iterations", type=int, default=4)
    parser.add_argument("--sim_contact_offset", type=float, default=0.02)
    parser.add_argument("--sim_rest_offset", type=float, default=0.002)
    parser.add_argument("--sim_max_depenetration_velocity", type=float, default=0.5)
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
    parser.add_argument("--stop_distance", type=float, default=0.75)
    parser.add_argument("--robot_front_offset", type=float, default=0.55)
    parser.add_argument("--walk_steps", type=int, default=260)
    parser.add_argument("--initial_hold_steps", type=int, default=150)
    parser.add_argument("--grasp_steps", type=int, default=150)
    parser.add_argument("--grasp_hold_steps", type=int, default=80)
    parser.add_argument("--gripper_close_steps", type=int, default=120)
    parser.add_argument("--handle_rotate_steps", type=int, default=300)
    parser.add_argument("--door_pull_steps", type=int, default=700)
    parser.add_argument("--hold_steps", type=int, default=300)
    parser.add_argument("--pregrasp_offset", type=float, default=0.15)
    parser.add_argument("--grasp_offset", type=float, default=0.0)
    parser.add_argument("--grasp_x_offset", type=float, default=-0.03)
    parser.add_argument("--grasp_z_offset", type=float, default=-0.03)
    parser.add_argument("--handle_rotate_right_distance", type=float, default=0.03)
    parser.add_argument("--handle_rotate_down_distance", type=float, default=0.03)
    parser.add_argument("--handle_rotate_angle", type=float, default=1.05)
    parser.add_argument("--door_pull_distance", type=float, default=1.10)
    parser.add_argument("--lever_step_size", type=float, default=0.04)
    parser.add_argument("--pull_base_distance", type=float, default=-0.35)
    parser.add_argument("--pull_base_yaw_delta", type=float, default=0.18)
    parser.add_argument("--gripper_open", type=float, default=-1.5707963267948966)
    parser.add_argument("--gripper_closed", type=float, default=0.0)
    parser.add_argument("--gripper_close_ratio", type=float, default=0.8)
    parser.add_argument("--arm_stiffness", type=float, default=400.0)
    parser.add_argument("--arm_damping", type=float, default=40.0)
    parser.add_argument("--gripper_stiffness", type=float, default=160.0)
    parser.add_argument("--gripper_damping", type=float, default=16.0)
    parser.add_argument("--gripper_joint_friction", type=float, default=120.0)
    parser.add_argument("--ik_pos_gain", type=float, default=1.0)
    parser.add_argument("--ik_orn_gain", type=float, default=0.5)
    parser.add_argument("--ik_damping", type=float, default=0.05)
    parser.add_argument("--ik_step_clip", type=float, default=0.06)
    parser.add_argument("--forward_ee_roll", type=float, default=math.pi / 2)
    parser.add_argument("--forward_ee_pitch", type=float, default=0.0)
    parser.add_argument("--gripper_red_axis_rot", type=float, default=-math.pi / 2)
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
    parser.add_argument("--handle_seg_id", type=int, default=2)
    parser.add_argument("--robot_vhacd_resolution", type=int, default=300000)
    parser.add_argument("--door_vhacd_resolution", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    world = FloatIKDoorWorld(args)
    try:
        base_start = world.robot_root[:, :2].clone()
        base_yaw_start = euler_from_quat(world.robot_root[:, 3:7])[2].clone()
        front_offset = torch.tensor([args.robot_front_offset, 0.0], device=world.device).repeat(world.num_envs, 1)
        heading = torch.stack([torch.cos(base_yaw_start), torch.sin(base_yaw_start)], dim=-1)
        robot_front = base_start + heading * args.robot_front_offset
        door_xy = world.door_root[:, :2].clone()
        front_to_door = torch.sum((door_xy - robot_front) * heading, dim=-1)
        walk_dist = torch.clamp(front_to_door - args.stop_distance, min=0.0)
        base_stop = base_start + heading * walk_dist.unsqueeze(-1)
        base_pull = base_stop + heading * args.pull_base_distance
        base_pull_yaw = base_yaw_start + args.pull_base_yaw_delta

        gripper_closed_target = args.gripper_open + (args.gripper_closed - args.gripper_open) * args.gripper_close_ratio
        traj_pregrasp = world.ee_pos.clone()
        traj_grasp = world.ee_pos.clone()
        traj_rotate = world.ee_pos.clone()
        traj_goal_quat = world.ee_quat.clone()
        phase = "walk"

        for step in range(args.steps):
            if world.viewer is not None and world.gym.query_viewer_has_closed(world.viewer):
                break

            world.refresh()
            handle_goal = world.handle_goal()
            approach_dir = world.robot_root[:, :3] - handle_goal
            approach_dir[:, 2] = 0.0
            approach_dir = approach_dir / torch.clamp(torch.norm(approach_dir, dim=-1, keepdim=True), min=1e-6)
            pregrasp = handle_goal + approach_dir * args.pregrasp_offset
            grasp = handle_goal + approach_dir * args.grasp_offset
            pregrasp[:, 0] += args.grasp_x_offset
            pregrasp[:, 2] += args.grasp_z_offset
            grasp[:, 0] += args.grasp_x_offset
            grasp[:, 2] += args.grasp_z_offset
            goal_quat = world.forward_ee_quat()

            walk_end = args.walk_steps
            ih_end = walk_end + args.initial_hold_steps
            grasp_end = ih_end + args.grasp_steps
            gh_end = grasp_end + args.grasp_hold_steps
            close_end = gh_end + args.gripper_close_steps
            rotate_end = close_end + args.handle_rotate_steps
            pull_end = rotate_end + args.door_pull_steps

            if step < walk_end:
                t = smoothstep(torch.full((world.num_envs,), (step + 1) / max(1, args.walk_steps), device=world.device))
                base_xy = lerp(base_start, base_stop, t)
                yaw = base_yaw_start
                target_pos = world.ee_pos.clone()
                target_quat = world.ee_quat.clone()
                gripper = torch.full((world.num_envs,), args.gripper_open, device=world.device)
                phase = "walk"
            else:
                if step == walk_end:
                    traj_pregrasp = pregrasp.clone()
                    traj_grasp = grasp.clone()
                    rotate_offset = torch.zeros_like(grasp)
                    rotate_offset[:, 1] = args.handle_rotate_right_distance
                    rotate_offset[:, 2] = -args.handle_rotate_down_distance
                    traj_rotate = grasp + rotate_offset
                    traj_goal_quat = goal_quat.clone()
                base_xy = base_stop.clone()
                yaw = base_yaw_start.clone()
                target_pos = traj_pregrasp.clone()
                target_quat = traj_goal_quat.clone()
                gripper = torch.full((world.num_envs,), args.gripper_open, device=world.device)
                phase = "initial_hold"
                if step >= ih_end and step < grasp_end:
                    t = smoothstep(torch.full((world.num_envs,), (step - ih_end + 1) / max(1, args.grasp_steps), device=world.device))
                    target_pos = lerp(traj_pregrasp, traj_grasp, t)
                    phase = "grasp"
                elif step >= grasp_end and step < gh_end:
                    target_pos = traj_grasp.clone()
                    phase = "grasp_hold"
                elif step >= gh_end and step < close_end:
                    t = smoothstep(torch.full((world.num_envs,), (step - gh_end + 1) / max(1, args.gripper_close_steps), device=world.device))
                    target_pos = traj_grasp.clone()
                    gripper = torch.full((world.num_envs,), args.gripper_open, device=world.device) + (
                        gripper_closed_target - args.gripper_open
                    ) * t
                    phase = "close_gripper"
                elif step >= close_end and step < rotate_end:
                    t = smoothstep(torch.full((world.num_envs,), (step - close_end + 1) / max(1, args.handle_rotate_steps), device=world.device))
                    target_pos = lerp(traj_grasp, traj_rotate, t)
                    axis = torch.zeros(world.num_envs, 3, device=world.device)
                    axis[:, 0] = 1.0
                    target_quat = quat_mul(traj_goal_quat, quat_from_angle_axis(-t * args.handle_rotate_angle, axis))
                    gripper = torch.full((world.num_envs,), gripper_closed_target, device=world.device)
                    phase = "rotate_handle"
                elif step >= rotate_end and step < pull_end:
                    t = smoothstep(torch.full((world.num_envs,), (step - rotate_end + 1) / max(1, args.door_pull_steps), device=world.device))
                    pull_dir = quat_axis(world.handle_quat, axis=2)
                    pull_dir[:, 2] = 0.0
                    pull_dir = pull_dir / torch.clamp(torch.norm(pull_dir, dim=-1, keepdim=True), min=1e-6)
                    pull_sign = torch.where(
                        torch.sum(pull_dir * approach_dir, dim=-1, keepdim=True) < 0.0,
                        -torch.ones(world.num_envs, 1, device=world.device),
                        torch.ones(world.num_envs, 1, device=world.device),
                    )
                    pull_dir = pull_dir * pull_sign
                    target_pos = world.ee_pos + pull_dir * args.lever_step_size
                    max_pos = traj_rotate + pull_dir * args.door_pull_distance
                    progress = torch.sum((target_pos - traj_rotate) * pull_dir, dim=-1)
                    target_pos = torch.where((progress > args.door_pull_distance).unsqueeze(-1), max_pos, target_pos)
                    axis = torch.zeros(world.num_envs, 3, device=world.device)
                    axis[:, 0] = 1.0
                    target_quat = quat_mul(traj_goal_quat, quat_from_angle_axis(-torch.full((world.num_envs,), args.handle_rotate_angle, device=world.device), axis))
                    base_xy = lerp(base_stop, base_pull, t)
                    yaw = lerp(base_yaw_start.unsqueeze(-1), base_pull_yaw.unsqueeze(-1), t).squeeze(-1)
                    gripper = torch.full((world.num_envs,), gripper_closed_target, device=world.device)
                    phase = "pull_door"
                elif step >= pull_end:
                    target_pos = world.ee_pos.clone()
                    gripper = torch.full((world.num_envs,), gripper_closed_target, device=world.device)
                    base_xy = base_pull.clone()
                    yaw = base_pull_yaw.clone()
                    phase = "hold"

            world.set_base_pose(base_xy, yaw)
            world.solve_ik_targets(target_pos, target_quat, gripper)
            world.simulate_step()

            if step % max(1, args.log_interval) == 0:
                pos_err = torch.norm(target_pos - world.ee_pos, dim=-1)
                orn_err = torch.rad2deg(
                    torch.norm(
                        orientation_error(
                            target_quat / torch.clamp(torch.norm(target_quat, dim=-1, keepdim=True), min=1e-6),
                            world.ee_quat / torch.clamp(torch.norm(world.ee_quat, dim=-1, keepdim=True), min=1e-6),
                        ),
                        dim=-1,
                    )
                )
                print(
                    f"[step {step:04d}]",
                    {
                        "phase": phase,
                        "pos_err_m": pos_err[: min(4, world.num_envs)].detach().cpu().tolist(),
                        "orn_err_deg": orn_err[: min(4, world.num_envs)].detach().cpu().tolist(),
                        "door_deg": torch.rad2deg(world.door_dof_pos[: min(4, world.num_envs), 0]).detach().cpu().tolist(),
                        "handle_ratio": (
                            (world.door_dof_pos[: min(4, world.num_envs), 1] - world.handle_limits_lower[: min(4, world.num_envs)])
                            / torch.clamp(world.handle_limits_upper[: min(4, world.num_envs)] - world.handle_limits_lower[: min(4, world.num_envs)], min=1e-6)
                        )
                        .detach()
                        .cpu()
                        .tolist(),
                    },
                    flush=True,
                )
    finally:
        world.close()


if __name__ == "__main__":
    main()
