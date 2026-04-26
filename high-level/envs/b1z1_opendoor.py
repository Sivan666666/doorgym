import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from .b1z1_base import B1Z1Base, LIN_VEL_X_CLIP, reindex_all, torch_rand_int


def _sorted_asset_items(asset_dict: Dict[str, Dict]) -> List[Dict]:
    def _key(item):
        key = item[0]
        return (0, int(key)) if key.isdigit() else (1, key)

    return [item[1] for item in sorted(asset_dict.items(), key=_key)]


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1.0
    return quat_rotate(q, basis_vec)


def wrap_to_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class B1Z1OpenDoor(B1Z1Base):
    def __init__(self, table_height=None, *args, **kwargs):
        self.num_actors = 2
        cfg = kwargs["cfg"]
        self.base_door_distance_threshold = cfg["env"].get("baseDoorDisThreshold", 0.85)
        self.handle_press_threshold_ratio = cfg["env"].get("handleOpenThresholdRatio", 0.65)
        self.door_open_success_threshold = cfg["env"].get("doorOpenSuccessThreshold", 0.45)
        self.door_open_hold_steps = cfg["env"].get("doorOpenHoldSteps", 5)
        self.ee_far_threshold = cfg["env"].get("eeFarThreshold", 1.1)
        self.door_lock_force = cfg["env"].get("doorLockForce", 150.0)
        self.door_open_resistance = cfg["env"].get("doorOpenResistance", 3.0)
        self.door_open_damping = cfg["env"].get("doorOpenDamping", 0.5)
        self.door_close_torque_sign = cfg["env"].get("doorCloseTorqueSign", 1.0)
        self.hard_lock_before_handle_threshold = cfg["env"].get("hardLockBeforeHandleThreshold", True)
        self.handle_spring_stiffness = cfg["env"].get("handleSpringStiffness", 40.0)
        self.handle_spring_damping = cfg["env"].get("handleSpringDamping", 2.0)
        self.door_joint_friction = cfg["env"].get("doorJointFriction", [2.0, 8.0])
        self.door_joint_damping = cfg["env"].get("doorJointDamping", [2.0, 6.0])
        self.door_joint_effort = cfg["env"].get("doorJointEffort", [200.0, 200.0])
        self.door_robot_start_pose = tuple(cfg["env"].get("doorRobotStartPose", [1.75, 0.0, 0.55]))
        self.door_robot_yaw = cfg["env"].get("doorRobotYaw", math.pi)
        super().__init__(*args, **kwargs)

    def _extra_env_settings(self):
        asset_cfg = self.cfg["env"]["asset"]
        load_block = asset_cfg.get("load_block")
        train_assets = asset_cfg["trainAssets"]
        if load_block is None:
            load_block = next(iter(train_assets.keys()))
        self.load_block = load_block
        self.door_asset_specs = _sorted_asset_items(train_assets[self.load_block])
        self.door_asset_names = [item["name"] for item in self.door_asset_specs]
        self.num_features = 0

        self.door_bounding_data = []
        self.handle_bounding_data = []
        asset_root = os.path.join(asset_cfg["assetRoot"], asset_cfg["assetFileDoor"])
        for spec in self.door_asset_specs:
            bounding_path = os.path.join(asset_root, spec["bounding_box"])
            handle_path = os.path.join(asset_root, spec["handle_bounding"])
            with open(bounding_path, "r", encoding="utf-8") as f:
                self.door_bounding_data.append(json.load(f))
            with open(handle_path, "r", encoding="utf-8") as f:
                self.handle_bounding_data.append(json.load(f))

    def _setup_obs_and_action_info(self):
        num_action = 9
        if self.floating_base:
            num_action += 1
        if self.pitch_control:
            num_action += 1

        dof_dim = 12
        if self.use_roboinfo and (not self.floating_base):
            dof_dim = 37
        core_obs_dim = 45 + dof_dim
        self.cfg["env"]["numObservations"] = core_obs_dim + num_action
        self.cfg["env"]["numActions"] = num_action

    def _create_envs(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file_door = self.cfg["env"]["asset"]["assetFileDoor"]

        self.door_asset_list = []
        self.door_asset_body_names = []
        self.door_asset_dof_names = []
        self.door_asset_dof_limits_lower = []
        self.door_asset_dof_limits_upper = []

        door_opts = gymapi.AssetOptions()
        door_opts.fix_base_link = True
        door_opts.collapse_fixed_joints = False
        door_opts.use_mesh_materials = True
        door_opts.override_com = True
        door_opts.override_inertia = True
        door_opts.disable_gravity = True

        for spec in self.door_asset_specs:
            door_asset = self.gym.load_asset(self.sim, asset_root, os.path.join(asset_file_door, spec["path"]), door_opts)
            self.door_asset_list.append(door_asset)
            self.door_asset_body_names.append(self.gym.get_asset_rigid_body_names(door_asset))
            self.door_asset_dof_names.append(self.gym.get_asset_dof_names(door_asset))
            door_dof_props = self.gym.get_asset_dof_properties(door_asset)
            self.door_asset_dof_limits_lower.append(torch.tensor(door_dof_props["lower"], device=self.device, dtype=torch.float))
            self.door_asset_dof_limits_upper.append(torch.tensor(door_dof_props["upper"], device=self.device, dtype=torch.float))

            door_shape_props = self.gym.get_asset_rigid_shape_properties(door_asset)
            for prop in door_shape_props:
                prop.friction = 2.0
            self.gym.set_asset_rigid_shape_properties(door_asset, door_shape_props)

        self.door_handles = []
        self.door_actor_spec_ids = []

        super()._create_envs()

    def _create_extra(self, env_i):
        env_ptr = self.envs[env_i]
        col_group = env_i
        col_filter = 0

        spec_id = env_i % len(self.door_asset_specs)
        spec = self.door_asset_specs[spec_id]
        door_asset = self.door_asset_list[spec_id]
        door_bounds = self.door_bounding_data[spec_id]

        door_pose = gymapi.Transform()
        door_pose.p = gymapi.Vec3(0.0, 0.0, -door_bounds["min"][2] + 0.1)
        door_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        door_handle = self.gym.create_actor(env_ptr, door_asset, door_pose, "door", col_group, col_filter, 1)

        door_dof_props = self.gym.get_asset_dof_properties(door_asset)
        door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        door_dof_props["stiffness"][:] = 0.0
        door_dof_props["damping"][:] = np.asarray(self.door_joint_damping, dtype=door_dof_props["damping"].dtype)
        if "effort" in door_dof_props.dtype.names:
            door_dof_props["effort"][:] = np.asarray(self.door_joint_effort, dtype=door_dof_props["effort"].dtype)
        if "friction" in door_dof_props.dtype.names:
            door_dof_props["friction"][:] = np.asarray(self.door_joint_friction, dtype=door_dof_props["friction"].dtype)
        if len(door_dof_props["upper"]) >= 2:
            door_dof_props["upper"][1] = min(float(door_dof_props["upper"][1]), math.pi / 4)
        self.gym.set_actor_dof_properties(env_ptr, door_handle, door_dof_props)

        self.door_handles.append(door_handle)
        self.door_actor_spec_ids.append(spec_id)

    def _init_tensors(self):
        super()._init_tensors()

        root_view = self._root_states.view(self.num_envs, self.num_actors, self._actor_root_state.shape[-1])
        self._door_root_states = root_view[..., 1, :]
        self._initial_door_root_states = self._door_root_states.clone()
        self._initial_door_root_states[:, 7:13] = 0.0
        self._door_actor_ids = self.num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 1

        full_dof_state = self._dof_state.view(self.num_envs, self.dof_per_env, 2)
        self._full_dof_pos = full_dof_state[..., :, 0]
        self.num_door_dofs = self.dof_per_env - self.num_dofs
        self._door_dof_pos = full_dof_state[..., self.num_dofs:, 0]
        self._door_dof_vel = full_dof_state[..., self.num_dofs:, 1]

        body_names = self.door_asset_body_names[0]
        self.door_body_name = body_names[-2] if len(body_names) >= 2 else body_names[-1]
        self.handle_body_name = body_names[-1]
        self.door_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.door_handles[0], self.door_body_name, gymapi.DOMAIN_ENV)
        self.handle_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.door_handles[0], self.handle_body_name, gymapi.DOMAIN_ENV)

        self.door_hinge_limits_lower = torch.stack([limits[0] for limits in self.door_asset_dof_limits_lower], dim=0)
        self.door_hinge_limits_upper = torch.stack([limits[0] for limits in self.door_asset_dof_limits_upper], dim=0)
        self.handle_limits_lower = torch.stack([limits[1] for limits in self.door_asset_dof_limits_lower], dim=0)
        self.handle_limits_upper = torch.stack([limits[1] for limits in self.door_asset_dof_limits_upper], dim=0)

        self.door_asset_indices = torch.tensor(self.door_actor_spec_ids, device=self.device, dtype=torch.long)
        goal_pos_offsets = [item["goal_pos"] for item in self.handle_bounding_data]
        handle_min = [item["handle_min"] for item in self.handle_bounding_data]
        handle_max = [item["handle_max"] for item in self.handle_bounding_data]
        self.goal_pos_offset_tensor = torch.tensor(goal_pos_offsets, device=self.device, dtype=torch.float)[self.door_asset_indices]
        self.handle_min_tensor = torch.tensor(handle_min, device=self.device, dtype=torch.float)[self.door_asset_indices]
        self.handle_max_tensor = torch.tensor(handle_max, device=self.device, dtype=torch.float)[self.door_asset_indices]

        self.best_handle_open_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.best_door_open_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.door_open_hold_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_recorded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.door_open_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.handle_open_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.door_open_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.handle_target_rot = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)
        self.grasp_goal_world = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.pregrasp_goal_world = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.handle_approach_dir_world = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.handle_rotate_dir_world = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.door_open_dir_world = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.goal_pos_local_yaw = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.base_door_dis = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.handle_press_threshold = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.open_door_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._update_door_state()
        self._update_curr_dist()

    def _reset_actors(self, env_ids):
        if len(env_ids) == 0:
            return

        self._robot_root_states[env_ids] = self._initial_robot_root_states[env_ids]
        self._robot_root_states[env_ids, 0] = self.door_robot_start_pose[0]
        self._robot_root_states[env_ids, 1] = self.door_robot_start_pose[1]
        self._robot_root_states[env_ids, 2] = self.door_robot_start_pose[2]
        self._robot_root_states[env_ids, :2] += torch_rand_float(-0.005, 0.005, (len(env_ids), 2), device=self.device)
        rand_yaw_robot = torch_rand_float(-0.03, 0.03, (len(env_ids), 1), device=self.device).squeeze(1)
        base_yaw = self.door_robot_yaw + rand_yaw_robot
        self._robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(base_yaw),
            torch.zeros_like(base_yaw),
            base_yaw,
        )
        self._robot_root_states[env_ids, 7:13] = 0.0

        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self.dof_pos_gripper[env_ids] += torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_gripper_joints), device=self.device)

        self._door_root_states[env_ids] = self._initial_door_root_states[env_ids]
        self._door_root_states[env_ids, 1] += torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device).squeeze(1)
        self._door_root_states[env_ids, 2] = self._initial_door_root_states[env_ids, 2]
        self._door_root_states[env_ids, 3:7] = self._initial_door_root_states[env_ids, 3:7]
        self._door_root_states[env_ids, 7:13] = 0.0

        self._door_dof_pos[env_ids] = 0.0
        self._door_dof_vel[env_ids] = 0.0
        self.update_roboinfo()
        self.last_ee_pos = quat_rotate_inverse(self._robot_root_states[:, 3:7], self.ee_pos - self.arm_base)

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if len(env_ids) == 0:
            return

        robot_ids_int32 = self._robot_actor_ids[env_ids]
        door_ids_int32 = self._door_actor_ids[env_ids]
        actor_ids = torch.cat([robot_ids_int32, door_ids_int32], dim=0)

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

        self.best_handle_open_ratio[env_ids] = 0.0
        self.best_door_open_ratio[env_ids] = 0.0
        self.door_open_hold_counter[env_ids] = 0
        self.success_recorded[env_ids] = False
        self.door_open_success[env_ids] = False

    def _reset_envs(self, env_ids):
        if len(env_ids) > 0 and (not self.cfg["env"].get("wandb", False)):
            self._print_success_rates()
        super()._reset_envs(env_ids)

    def _print_success_rates(self):
        stats = []
        for asset_id, asset_name in enumerate(self.door_asset_names):
            env_mask = self.door_asset_indices == asset_id
            success = self.success_counter[env_mask].sum().item()
            episodes = max(self.episode_counter[env_mask].sum().item(), 0)
            success_rate = min(success, episodes) / max(episodes, 1)
            stats.append(f"{asset_name}: {success_rate:.3f} ({success}/{max(episodes, 0)})")
        total_success = self.success_counter.sum().item()
        total_episodes = max(self.episode_counter.sum().item(), 0)
        total_rate = min(total_success, total_episodes) / max(total_episodes, 1)
        print({"door_success_rate": total_rate})
        print(", ".join(stats))

    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        self._update_door_state()
        self._enforce_locked_door_state()
        self._update_door_state()
        self.update_roboinfo()
        self._update_curr_dist()

    def _enforce_locked_door_state(self):
        if not self.hard_lock_before_handle_threshold:
            return

        locked_mask = ~self.open_door_stage
        if not torch.any(locked_mask):
            return

        # Keep the door leaf exactly closed until the lever has been pressed far enough.
        self._door_dof_pos[locked_mask, 0] = 0.0
        self._door_dof_vel[locked_mask, 0] = 0.0
        actor_ids = self._door_actor_ids[locked_mask]
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _update_door_state(self):
        handle_pos = self._rigid_body_pos[:, self.handle_body_idx]
        handle_rot = self._rigid_body_rot[:, self.handle_body_idx]

        self.handle_approach_dir_world[:] = quat_axis(handle_rot, axis=0)
        self.handle_rotate_dir_world[:] = quat_axis(handle_rot, axis=1)
        self.door_open_dir_world[:] = quat_axis(handle_rot, axis=2)

        self.grasp_goal_world[:] = quat_apply(handle_rot, self.goal_pos_offset_tensor) + handle_pos
        self.pregrasp_goal_world[:] = self.grasp_goal_world + self.handle_approach_dir_world * 0.18

        down_q = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.handle_target_rot[:] = quat_mul(handle_rot, down_q)

        hinge_range = torch.clamp(torch.maximum(torch.abs(self.door_hinge_limits_upper[self.door_asset_indices]),
                                                torch.abs(self.door_hinge_limits_lower[self.door_asset_indices])), min=1e-3)
        handle_range = torch.clamp(self.handle_limits_upper[self.door_asset_indices] - self.handle_limits_lower[self.door_asset_indices], min=1e-3)

        self.handle_press_threshold[:] = self.handle_press_threshold_ratio * handle_range
        self.door_open_ratio[:] = torch.clamp(torch.abs(self._door_dof_pos[:, 0]) / hinge_range, 0.0, 1.5)
        self.handle_open_ratio[:] = torch.clamp((self._door_dof_pos[:, 1] - self.handle_limits_lower[self.door_asset_indices]) / handle_range, 0.0, 1.5)
        self.open_door_stage[:] = self._door_dof_pos[:, 1] >= self.handle_press_threshold
        self.door_open_success[:] = self.door_open_ratio >= self.door_open_success_threshold

    def update_roboinfo(self):
        super().update_roboinfo()
        base_yaw = euler_from_quat(self._robot_root_states[:, 3:7])[2]
        base_yaw_quat = quat_from_euler_xyz(torch.zeros_like(base_yaw), torch.zeros_like(base_yaw), base_yaw)
        self.goal_pos_local_yaw[:] = quat_rotate_inverse(base_yaw_quat, self.grasp_goal_world - self.arm_base)
        self.goal_pos_local_yaw[:, 2] = self.grasp_goal_world[:, 2]
        self.base_door_dis[:] = torch.norm(self.grasp_goal_world[:, :2] - self.arm_base[:, :2], dim=-1)

    def _update_curr_dist(self):
        d = torch.norm(self.ee_pos - self.grasp_goal_world, dim=-1)
        self.curr_dist[:] = d
        self.closest_dist = torch.where(self.closest_dist < 0, self.curr_dist, torch.minimum(self.closest_dist, self.curr_dist))

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(range(self.num_envs), device=self.device, dtype=torch.long)

        obs = self._compute_robot_obs(env_ids)
        if self.cfg["env"].get("lastCommands", False):
            self.obs_buf[env_ids] = torch.cat([obs, self.command_history_buf[env_ids, -1]], dim=-1)
        else:
            self.obs_buf[env_ids] = torch.cat([obs, self.action_history_buf[env_ids, -1]], dim=-1)

    def _compute_robot_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        robot_root_state = self._robot_root_states[env_ids]
        door_root_state = self._door_root_states[env_ids]
        body_pos = self._rigid_body_pos[env_ids]
        body_rot = self._rigid_body_rot[env_ids]
        dof_pos = self._dof_pos[env_ids]
        dof_vel = self._dof_vel[env_ids]
        base_quat_yaw = self.base_yaw_quat[env_ids]
        commands = self.commands[env_ids]

        handle_pos = body_pos[:, self.handle_body_idx]
        handle_rot = body_rot[:, self.handle_body_idx]
        ee_pos = body_pos[:, self.gripper_idx]
        ee_rot = body_rot[:, self.gripper_idx]
        arm_base = self.arm_base[env_ids]

        if self.use_roboinfo and (not self.floating_base):
            obs_dof_pos = reindex_all(dof_pos)
            obs_dof_vel = reindex_all(dof_vel)[:, :-self.num_gripper_joints] * 0.05
        else:
            obs_dof_pos = dof_pos[:, 12:-self.num_gripper_joints]
            obs_dof_vel = dof_vel[:, 12:-self.num_gripper_joints] * 0.05

        base_quat = robot_root_state[:, 3:7]
        handle_pos_local = quat_rotate_inverse(base_quat_yaw, handle_pos - arm_base)
        handle_pos_local[:, 2] = handle_pos[:, 2]
        goal_pos_local = quat_rotate_inverse(base_quat_yaw, self.grasp_goal_world[env_ids] - arm_base)
        goal_pos_local[:, 2] = self.grasp_goal_world[env_ids, 2]
        door_root_local = quat_rotate_inverse(base_quat_yaw, door_root_state[:, :3] - arm_base)
        door_root_local[:, 2] = door_root_state[:, 2]

        ee_pos_local = quat_rotate_inverse(base_quat, ee_pos - arm_base)
        ee_rot_local = quat_mul(quat_conjugate(base_quat), ee_rot)
        ee_rot_local_rpy = torch.stack(euler_from_quat(ee_rot_local), dim=-1)
        handle_rot_local = quat_mul(quat_conjugate(base_quat_yaw), handle_rot)
        handle_rot_local_rpy = torch.stack(euler_from_quat(handle_rot_local), dim=-1)
        ee_to_goal_local = quat_rotate_inverse(base_quat, self.grasp_goal_world[env_ids] - ee_pos)

        approach_dir_local = quat_rotate_inverse(base_quat_yaw, self.handle_approach_dir_world[env_ids])
        rotate_dir_local = quat_rotate_inverse(base_quat_yaw, self.handle_rotate_dir_world[env_ids])
        open_dir_local = quat_rotate_inverse(base_quat_yaw, self.door_open_dir_world[env_ids])

        robot_vel_local = quat_rotate_inverse(base_quat_yaw, robot_root_state[:, 7:10])

        obs = torch.cat(
            (
                goal_pos_local,
                handle_pos_local,
                handle_rot_local_rpy,
                approach_dir_local,
                rotate_dir_local,
                open_dir_local,
                ee_pos_local,
                ee_rot_local_rpy,
                ee_to_goal_local,
                door_root_local,
                self._door_dof_pos[env_ids, 0:1],
                self._door_dof_pos[env_ids, 1:2],
                self.door_open_ratio[env_ids].unsqueeze(-1),
                obs_dof_pos,
                obs_dof_vel,
                commands,
                self.curr_ee_goal_cart[env_ids],
                self.curr_ee_goal_orn_rpy[env_ids],
                robot_vel_local,
            ),
            dim=-1,
        )
        return obs

    def _get_seg_id(self):
        return 1

    def check_termination(self):
        super().check_termination()

        self.door_open_hold_counter = torch.where(
            self.door_open_success,
            self.door_open_hold_counter + 1,
            torch.zeros_like(self.door_open_hold_counter),
        )
        self.reset_buf |= self.door_open_hold_counter >= self.door_open_hold_steps

        ee_far = (self.progress_buf > 40) & (self.curr_dist > self.ee_far_threshold)
        base_far = (self.progress_buf > 80) & (self.base_door_dis > 2.0)
        self.reset_buf |= ee_far | base_far

    def get_grasp_goal_world(self):
        return self.grasp_goal_world.clone()

    def get_pregrasp_goal_world(self, offset=0.18):
        return self.grasp_goal_world + self.handle_approach_dir_world * offset

    def scripted_actions_from_world_targets(self, target_pos_world, target_rot_world, gripper_open, base_x_cmd, yaw_cmd):
        actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        target_local = quat_rotate_inverse(self._robot_root_states[:, 3:7], target_pos_world - self.arm_base)
        actions[:, :3] = torch.clamp(target_local - self.curr_ee_goal_cart, -0.02, 0.02)

        target_local_rot = quat_mul(quat_conjugate(self._robot_root_states[:, 3:7]), target_rot_world)
        target_rpy = torch.stack(euler_from_quat(target_local_rot), dim=-1)
        actions[:, 3:6] = torch.clamp(wrap_to_pi(target_rpy - self.curr_ee_goal_orn_rpy), -0.06, 0.06)

        actions[:, 6] = torch.where(gripper_open, torch.ones_like(actions[:, 6]), -torch.ones_like(actions[:, 6]))
        actions[:, 7] = torch.clamp(base_x_cmd, -0.35, 0.35)
        actions[:, 8] = torch.clamp(yaw_cmd, -0.5, 0.5)
        if self.pitch_control:
            actions[:, 9] = 0.0
        return actions

    def get_all_pos_targets(self, ee_goal_cart, ee_goal_orn_quat):
        robot_targets = super().get_all_pos_targets(ee_goal_cart, ee_goal_orn_quat)
        all_targets = torch.zeros(self.num_envs, self.dof_per_env, device=self.device, dtype=robot_targets.dtype)
        all_targets[:, :self.num_dofs] = robot_targets
        return all_targets

    def get_torques(self):
        robot_torques = super().get_torques()
        all_torques = torch.zeros(self.num_envs, self.dof_per_env, device=self.device, dtype=robot_torques.dtype)
        all_torques[:, :self.num_dofs] = robot_torques
        door_torque = torch.where(
            self.open_door_stage,
            self.door_close_torque_sign * (
                self.door_open_resistance * self._door_dof_pos[:, 0] + self.door_open_damping * self._door_dof_vel[:, 0]
            ),
            torch.full_like(self._door_dof_pos[:, 0], self.door_close_torque_sign * self.door_lock_force),
        )
        handle_torque = -self.handle_spring_stiffness * self._door_dof_pos[:, 1] - self.handle_spring_damping * self._door_dof_vel[:, 1]
        all_torques[:, self.num_dofs + 0] = door_torque
        all_torques[:, self.num_dofs + 1] = handle_torque
        return all_torques

    # ----------------------------- reward functions -----------------------------
    def _reward_approach_handle(self):
        dist_delta = self.closest_dist - self.curr_dist
        self.closest_dist = torch.minimum(self.closest_dist, self.curr_dist)
        dist_delta = torch.clamp(dist_delta, 0.0, 10.0)
        reward = torch.tanh(10.0 * dist_delta)
        reward *= ~self.door_open_success
        return reward, reward

    def _reward_ee_align_handle(self):
        orn_err = orientation_error(self.handle_target_rot, self.ee_orn / torch.norm(self.ee_orn, dim=-1, keepdim=True))
        metric = torch.norm(orn_err, dim=-1)
        reward = torch.exp(-3.0 * metric)
        reward *= (self.curr_dist < 0.25).float()
        return reward, metric

    def _reward_lever_press(self):
        progress = self.handle_open_ratio - self.best_handle_open_ratio
        self.best_handle_open_ratio = torch.maximum(self.best_handle_open_ratio, self.handle_open_ratio)
        progress = torch.clamp(progress, min=0.0, max=1.0)
        reward = torch.tanh(5.0 * progress)
        reward *= (self.curr_dist < 0.18).float()
        reward *= (self.handle_open_ratio < self.handle_press_threshold_ratio + 0.05).float()
        return reward, self.handle_open_ratio

    def _reward_door_open_progress(self):
        progress = self.door_open_ratio - self.best_door_open_ratio
        self.best_door_open_ratio = torch.maximum(self.best_door_open_ratio, self.door_open_ratio)
        progress = torch.clamp(progress, min=0.0, max=1.0)
        reward = torch.tanh(5.0 * progress)
        reward *= (self.handle_open_ratio >= self.handle_press_threshold_ratio).float()
        return reward, self.door_open_ratio

    def _reward_door_open_success(self):
        newly_successful = self.door_open_success & (~self.success_recorded)
        self.success_counter[newly_successful] += 1
        self.success_recorded |= self.door_open_success
        reward = newly_successful.float()
        return reward, self.door_open_success.float()

    def _reward_base_command_penalty(self):
        penalty = torch.where(
            self.base_door_dis < self.base_door_distance_threshold,
            torch.norm(self.commands[:, :], dim=-1),
            torch.zeros_like(self.reset_buf, dtype=torch.float, device=self.device),
        )
        return penalty, penalty
