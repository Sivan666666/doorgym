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

ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = ROOT / "high-level"
LOW_LEVEL_ROOT = ROOT / "low-level"
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


DOOR_RUNTIME = {}


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

    return {
        "asset_root": str(asset_root),
        "asset_file_door": asset_file_door,
        "door_asset_specs": door_asset_specs,
        "door_asset_names": [item["name"] for item in door_asset_specs],
        "door_bounding_data": door_bounding_data,
        "handle_bounding_data": handle_bounding_data,
        "door_lock_force": 150.0,
        "door_open_resistance": 3.0,
        "handle_unlock_ratio": 0.65,
        "handle_spring_stiffness": 3.5,
        "handle_spring_damping": 1.0,
    }


class ManipLocoDoorAsset(ManipLoco):
    """Low-level B1Z1 locomotion/manipulation env with an extra door actor."""

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

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset["driveMode"][12:].fill(gymapi.DOF_MODE_POS)
        dof_props_asset["stiffness"][12:].fill(400.0)
        dof_props_asset["damping"][12:].fill(40.0)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
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
        self.envs = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            pos = self.env_origins[i].clone()
            pos[0] += DOOR_RUNTIME["robot_x"]
            pos[1] += DOOR_RUNTIME["robot_y"]
            pos[2] += DOOR_RUNTIME["robot_z"]
            pos[:2] += torch_rand_float(
                -self.cfg.init_state.origin_perturb_range,
                self.cfg.init_state.origin_perturb_range,
                (2, 1),
                device=self.device,
            ).squeeze(1)
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
        door_opts.vhacd_params.resolution = 2048

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
            float(origin[2] - door_bounds["min"][2] + DOOR_RUNTIME["door_z_offset"]),
        )
        door_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        door_handle = self.gym.create_actor(env_handle, door_asset, door_pose, "door", env_i, 0, 1)

        door_dof_props = self.gym.get_asset_dof_properties(door_asset)
        door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        if len(door_dof_props["upper"]) >= 2:
            door_dof_props["upper"][1] = min(float(door_dof_props["upper"][1]), math.pi / 4)
        self.gym.set_actor_dof_properties(env_handle, door_handle, door_dof_props)

        self.door_handles.append(door_handle)
        self.door_actor_spec_ids.append(spec_id)

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
        self.global_steps = 0

    def _init_door_tensors(self):
        body_names = self.door_asset_body_names[0]
        self.door_body_name = body_names[-2] if len(body_names) >= 2 else body_names[-1]
        self.handle_body_name = body_names[-1]
        self.door_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.door_handles[0], self.door_body_name, gymapi.DOMAIN_ENV)
        self.handle_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.door_handles[0], self.handle_body_name, gymapi.DOMAIN_ENV)
        self.door_asset_indices = torch.tensor(self.door_actor_spec_ids, device=self.device, dtype=torch.long)
        self.door_hinge_limits_lower = torch.stack([limits[0] for limits in self.door_asset_dof_limits_lower], dim=0)
        self.door_hinge_limits_upper = torch.stack([limits[0] for limits in self.door_asset_dof_limits_upper], dim=0)
        self.handle_limits_lower = torch.stack([limits[1] for limits in self.door_asset_dof_limits_lower], dim=0)
        self.handle_limits_upper = torch.stack([limits[1] for limits in self.door_asset_dof_limits_upper], dim=0)
        goal_pos_offsets = [item["goal_pos"] for item in self.handle_bounding_data]
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
        self.open_door_stage[:] = (handle_angle_from_lower >= self.handle_unlock_threshold) | (torch.abs(door_angle) > 0.01)
        door_torques[:, 0] = torch.where(
            self.open_door_stage,
            -DOOR_RUNTIME["door_open_resistance"] * door_angle,
            -torch.full_like(door_angle, DOOR_RUNTIME["door_lock_force"]),
        )
        door_torques[:, 1] = (
            -DOOR_RUNTIME["handle_spring_stiffness"] * handle_angle_from_lower
            - DOOR_RUNTIME["handle_spring_damping"] * self._door_dof_vel[:, 1]
        )

        hinge_range = torch.clamp(
            torch.maximum(
                torch.abs(self.door_hinge_limits_upper[self.door_asset_indices]),
                torch.abs(self.door_hinge_limits_lower[self.door_asset_indices]),
            ),
            min=1e-3,
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
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._full_dof_state_flat))
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, 0] = self.env_origins[env_ids, 0] + DOOR_RUNTIME["robot_x"]
        self.root_states[env_ids, 1] = self.env_origins[env_ids, 1] + DOOR_RUNTIME["robot_y"]
        self.root_states[env_ids, 2] = self.env_origins[env_ids, 2] + DOOR_RUNTIME["robot_z"]
        self.root_states[env_ids, :2] += torch_rand_float(
            -self.cfg.init_state.origin_perturb_range,
            self.cfg.init_state.origin_perturb_range,
            (len(env_ids), 2),
            device=self.device,
        )

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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episode_length_s", type=float, default=10000.0)
    parser.add_argument("--speed_min", type=float, default=0.65)
    parser.add_argument("--speed_max", type=float, default=0.80)
    parser.add_argument("--yaw_min", type=float, default=0.0)
    parser.add_argument("--yaw_max", type=float, default=0.0)
    parser.add_argument("--resample_interval", type=int, default=90)
    parser.add_argument("--stop_distance", type=float, default=0.50)
    parser.add_argument("--robot_front_offset", type=float, default=0.55)
    parser.add_argument("--pregrasp_offset", type=float, default=0.22)
    parser.add_argument("--grasp_offset", type=float, default=0.0)
    parser.add_argument("--approach_steps", type=int, default=240)
    parser.add_argument("--grasp_steps", type=int, default=180)
    parser.add_argument("--grasp_hold_steps", type=int, default=100)
    parser.add_argument("--gripper_close_steps", type=int, default=120)
    parser.add_argument("--handle_rotate_steps", type=int, default=420)
    parser.add_argument("--door_pull_steps", type=int, default=480)
    parser.add_argument("--lever_step_size", type=float, default=0.06)
    parser.add_argument("--handle_rotate_distance", type=float, default=0.28)
    parser.add_argument("--handle_rotate_angle", type=float, default=1.05)
    parser.add_argument("--handle_arc_radius", type=float, default=0.18)
    parser.add_argument("--door_pull_distance", type=float, default=0.45)
    parser.add_argument("--gripper_open", type=float, default=-1.5707963267948966)
    parser.add_argument("--gripper_closed", type=float, default=0.0)
    parser.add_argument("--external_orn_gain", type=float, default=0.00)
    parser.add_argument("--forward_ee_roll", type=float, default=math.pi / 2)
    parser.add_argument("--forward_ee_pitch", type=float, default=0.0)
    parser.add_argument("--preview_trajectory_at_spawn", dest="preview_trajectory_at_spawn", action="store_true", default=True)
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
    parser.add_argument("--door_z_offset", type=float, default=0)
    parser.add_argument("--box_x", type=float, default=-3.0)
    parser.add_argument("--box_y", type=float, default=-3.0)
    parser.add_argument("--door_cfg", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "cfg" / "b1z1_opendoor.yaml"))
    parser.add_argument("--log_dir", type=str, default=str(LOW_LEVEL_ROOT / "logs" / "b1z1-low" / "b1z1_locomanip"))
    parser.add_argument("--checkpoint", type=int, default=45000)
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
    DOOR_RUNTIME["layout_spacing"] = args.layout_spacing
    DOOR_RUNTIME["robot_x"] = args.robot_x
    DOOR_RUNTIME["robot_y"] = args.robot_y
    DOOR_RUNTIME["robot_z"] = args.robot_z
    DOOR_RUNTIME["robot_yaw"] = args.robot_yaw
    DOOR_RUNTIME["door_x"] = args.door_x
    DOOR_RUNTIME["door_y"] = args.door_y
    DOOR_RUNTIME["door_z_offset"] = args.door_z_offset
    DOOR_RUNTIME["box_x"] = args.box_x
    DOOR_RUNTIME["box_y"] = args.box_y

    low_args = build_low_level_args(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name="b1z1")
    task_registry.register("b1z1_door_asset", ManipLocoDoorAsset, env_cfg, train_cfg, "b1z1")

    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = args.episode_length_s
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
    print("Door actor count:", len(env.door_handles))
    print("Door DOF count:", env.num_door_dofs)
    print("Door body / handle body:", env.door_body_name, env.handle_body_name)
    print(
        "Layout offsets:",
        {
            "spacing": args.layout_spacing,
            "robot": [args.robot_x, args.robot_y, args.robot_z, args.robot_yaw],
            "door": [args.door_x, args.door_y, args.door_z_offset],
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
            "approach_steps": args.approach_steps,
            "grasp_steps": args.grasp_steps,
            "grasp_hold_steps": args.grasp_hold_steps,
            "gripper_close_steps": args.gripper_close_steps,
            "handle_rotate_steps": args.handle_rotate_steps,
            "door_pull_steps": args.door_pull_steps,
            "lever_step_size": args.lever_step_size,
            "handle_rotate_angle": args.handle_rotate_angle,
            "handle_arc_radius": args.handle_arc_radius,
            "door_pull_distance": args.door_pull_distance,
            "gripper_open": args.gripper_open,
            "gripper_closed": args.gripper_closed,
            "external_orn_gain": args.external_orn_gain,
            "forward_ee_roll": args.forward_ee_roll,
            "forward_ee_pitch": args.forward_ee_pitch,
            "preview_trajectory_at_spawn": args.preview_trajectory_at_spawn,
        },
    )

    env.reset()
    env.external_ee_goal_control = True
    env.external_orn_gain = args.external_orn_gain
    start_xy = env.root_states[:, :2].clone()
    commanded_vx = torch.zeros(args.num_envs, device=env.device)
    commanded_yaw = torch.zeros(args.num_envs, device=env.device)
    stopped_by_door = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    front_to_door_distance = torch.full((args.num_envs,), float("inf"), device=env.device)
    forward_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(args.num_envs, 1)
    manip_started = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    manip_step = torch.zeros(args.num_envs, device=env.device, dtype=torch.long)
    env.freeze_arm_default = torch.ones(args.num_envs, device=env.device, dtype=torch.bool)
    env.external_gripper_target = torch.full(
        (args.num_envs, env.cfg.env.num_gripper_joints),
        args.gripper_open,
        device=env.device,
    )
    traj_anchor_pos = env.ee_pos.clone()
    manip_target_pos = env.ee_pos.clone()
    manip_target_quat = env.ee_orn.clone()
    phase_name = ["walk_default", "approach", "grasp", "grasp_hold_open", "close_gripper", "rotate_handle", "pull_door", "hold"]
    phase_id = torch.zeros(args.num_envs, device=env.device, dtype=torch.long)

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

    def forward_ee_quat():
        base_yaw = euler_from_quat(env.root_states[:, 3:7])[2]
        roll = torch.full_like(base_yaw, args.forward_ee_roll)
        pitch = torch.full_like(base_yaw, args.forward_ee_pitch)
        return quat_from_euler_xyz(roll, pitch, base_yaw)

    def update_ee_trajectory():
        nonlocal traj_anchor_pos, manip_target_pos, manip_target_quat

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
        goal_quat = forward_ee_quat()
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
        rotate_tangent_dir = pull_dir.clone()
        rotate_tangent_dir[:, 0] = 0.0
        rotate_tangent_norm = torch.norm(rotate_tangent_dir, dim=-1, keepdim=True)
        rotate_tangent_fallback = torch.zeros_like(rotate_tangent_dir)
        rotate_tangent_fallback[:, 1] = torch.where(pull_dir[:, 1] < 0.0, -1.0, 1.0)
        rotate_tangent_dir = torch.where(
            rotate_tangent_norm > 1e-4,
            rotate_tangent_dir / torch.clamp(rotate_tangent_norm, min=1e-6),
            rotate_tangent_fallback,
        )
        down_arc_dir = torch.zeros_like(pull_dir)
        down_arc_dir[:, 2] = 1.0
        rotate_center = grasp_pos - rotate_tangent_dir * handle_arc_radius
        rotate_pos = rotate_center + handle_arc_radius * (
            math.cos(args.handle_rotate_angle) * rotate_tangent_dir - math.sin(args.handle_rotate_angle) * down_arc_dir
        )
        pull_pos = rotate_pos + pull_dir * args.door_pull_distance

        newly_started = stopped_by_door & (~manip_started)
        if torch.any(newly_started):
            manip_started[newly_started] = True
            manip_step[newly_started] = 0
            traj_anchor_pos[newly_started] = env.ee_pos[newly_started]
            manip_target_pos[newly_started] = env.ee_pos[newly_started]
            manip_target_quat[newly_started] = goal_quat[newly_started]

        walking = ~stopped_by_door
        if torch.any(walking):
            env.freeze_arm_default[walking] = True
            env.external_gripper_target[walking] = args.gripper_open
            env.curr_ee_goal_cart_world[walking] = env.ee_pos[walking]
            env.ee_goal_orn_quat[walking] = env.ee_orn[walking]
            phase_id[walking] = 0

        active = stopped_by_door & manip_started
        if not torch.any(active):
            return

        env.freeze_arm_default[active] = False
        a_end = args.approach_steps
        g_end = a_end + args.grasp_steps
        h_end = g_end + args.grasp_hold_steps
        c_end = h_end + args.gripper_close_steps
        r_end = c_end + args.handle_rotate_steps
        p_end = r_end + args.door_pull_steps

        approach = active & (manip_step < a_end)
        if torch.any(approach):
            denom = max(1, args.approach_steps)
            t = smoothstep((manip_step[approach].to(torch.float) + 1.0) / denom)
            manip_target_pos[approach] = lerp(traj_anchor_pos[approach], pregrasp_pos[approach], t)
            manip_target_quat[approach] = goal_quat[approach]
            env.external_gripper_target[approach] = args.gripper_open
            phase_id[approach] = 1

        grasp = active & (manip_step >= a_end) & (manip_step < g_end)
        if torch.any(grasp):
            denom = max(1, args.grasp_steps)
            t = smoothstep(((manip_step[grasp] - a_end).to(torch.float) + 1.0) / denom)
            manip_target_pos[grasp] = lerp(pregrasp_pos[grasp], grasp_pos[grasp], t)
            manip_target_quat[grasp] = goal_quat[grasp]
            env.external_gripper_target[grasp] = args.gripper_open
            phase_id[grasp] = 2

        grasp_hold = active & (manip_step >= g_end) & (manip_step < h_end)
        if torch.any(grasp_hold):
            manip_target_pos[grasp_hold] = grasp_pos[grasp_hold]
            manip_target_quat[grasp_hold] = goal_quat[grasp_hold]
            env.external_gripper_target[grasp_hold] = args.gripper_open
            phase_id[grasp_hold] = 3

        close_gripper = active & (manip_step >= h_end) & (manip_step < c_end)
        if torch.any(close_gripper):
            manip_target_pos[close_gripper] = grasp_pos[close_gripper]
            manip_target_quat[close_gripper] = goal_quat[close_gripper]
            env.external_gripper_target[close_gripper] = args.gripper_closed
            phase_id[close_gripper] = 4

        rotate = active & (manip_step >= c_end) & (manip_step < r_end)
        if torch.any(rotate):
            env.external_gripper_target[rotate] = args.gripper_closed
            denom = max(1, args.handle_rotate_steps)
            t = smoothstep(((manip_step[rotate] - c_end).to(torch.float) + 1.0) / denom)
            theta = t * args.handle_rotate_angle
            manip_target_pos[rotate] = rotate_center[rotate] + handle_arc_radius * (
                torch.cos(theta)[:, None] * rotate_tangent_dir[rotate] - torch.sin(theta)[:, None] * down_arc_dir[rotate]
            )
            manip_target_quat[rotate] = goal_quat[rotate]
            phase_id[rotate] = 5

        pull = active & (manip_step >= r_end) & (manip_step < p_end)
        if torch.any(pull):
            env.external_gripper_target[pull] = args.gripper_closed
            denom = max(1, args.door_pull_steps)
            t = smoothstep(((manip_step[pull] - r_end).to(torch.float) + 1.0) / denom)
            manip_target_pos[pull] = lerp(rotate_pos[pull], pull_pos[pull], t)
            manip_target_quat[pull] = goal_quat[pull]
            phase_id[pull] = 6

        hold = active & (manip_step >= p_end)
        if torch.any(hold):
            manip_target_quat[hold] = goal_quat[hold]
            env.external_gripper_target[hold] = args.gripper_closed
            phase_id[hold] = 7

        env.curr_ee_goal_cart_world[active] = manip_target_pos[active]
        env.ee_goal_orn_quat[active] = manip_target_quat[active]
        manip_step[active] += 1

    red_target_geom = gymutil.WireframeSphereGeometry(0.035, 8, 8, None, color=(1, 0, 0))

    def draw_ee_target():
        if getattr(env, "viewer", None) is None:
            return
        for env_i in range(min(args.num_envs, 16)):
            target = env.curr_ee_goal_cart_world[env_i].detach().cpu().tolist()
            pose = gymapi.Transform(gymapi.Vec3(target[0], target[1], target[2]), r=None)
            gymutil.draw_lines(red_target_geom, env.gym, env.viewer, env.envs[env_i], pose)

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
        env.commands[:, 0] = torch.where(stopped_by_door, torch.zeros_like(commanded_vx), commanded_vx)
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = torch.where(stopped_by_door, torch.zeros_like(commanded_yaw), commanded_yaw)
        actions = policy(obs.detach(), hist_encoding=True)
        obs, _, _, _, dones, infos = env.step(actions.detach())
        draw_ee_target()
        if torch.any(dones):
            stopped_by_door[dones] = False
            manip_started[dones] = False
            manip_step[dones] = 0
            env.freeze_arm_default[dones] = True
            env.external_gripper_target[dones] = args.gripper_open
        if step % 30 == 0:
            shown_phase = [phase_name[int(i)] for i in phase_id[: min(args.num_envs, 4)].detach().cpu().tolist()]
            traveled = torch.norm(env.root_states[:, :2] - start_xy, dim=-1)
            print(
                f"[step {step:04d}]",
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
                    "ee_target": env.curr_ee_goal_cart_world[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "door_dof": env._door_dof_pos[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "door_open_stage": env.open_door_stage[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "door_open_ratio": env.door_open_ratio[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "handle_open_ratio": env.handle_open_ratio[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "episode_length": env.episode_length_buf[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "time_out": env.time_out_buf[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "traveled_xy": traveled[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "reset": dones[: min(args.num_envs, 4)].detach().cpu().tolist(),
                },
            )


if __name__ == "__main__":
    main()
