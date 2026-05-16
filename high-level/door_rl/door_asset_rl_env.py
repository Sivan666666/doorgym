import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

ROOT = Path(__file__).resolve().parents[2]
HIGH_LEVEL_ROOT = ROOT / "high-level"
LOW_LEVEL_ROOT = ROOT / "low-level"
if str(HIGH_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_ROOT))
if str(LOW_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOW_LEVEL_ROOT))

from isaacgym import gymapi, gymtorch  # noqa: E402
import torch  # noqa: E402
from isaacgym.torch_utils import (  # noqa: E402
    euler_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
    torch_rand_float,
)
from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.utils import task_registry  # noqa: E402
from skrl.envs.wrappers.torch import Wrapper  # noqa: E402

import play_b1z1_push_with_door_asset_camera as door_play  # noqa: E402


ACTION_DIM = 9
REWARD_CURRICULA = ("reach", "grasp", "handle", "open", "pass", "full")
PHASE_NAMES = ("reach", "grasp", "handle_press", "door_open", "pass")
TEACHER_OBS_DIM = 96
PROPRIO_DIM = 72
IMAGE_H = 54
IMAGE_W = 96
IMAGE_CHANNELS = 12
STUDENT_OBS_DIM = IMAGE_CHANNELS * IMAGE_H * IMAGE_W + PROPRIO_DIM


def _wrap_to_pi(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def build_default_runtime_args(**overrides):
    values = dict(
        door_cfg=str(HIGH_LEVEL_ROOT / "data" / "cfg" / "b1z1_opendoor.yaml"),
        use_all_door_assets=False,
        num_envs=4,
        layout_spacing=5.0,
        robot_x=4.1,
        robot_y=0.0,
        robot_z=0.5,
        robot_yaw=math.pi,
        door_x=2.5,
        door_y=0.0,
        door_z_offset=0.01,
        door_actor_scale=1.2,
        box_x=-3.0,
        box_y=-3.0,
        gripper_stiffness=160.0,
        gripper_damping=16.0,
        gripper_joint_friction=120.0,
        handle_spring_stiffness=0.5,
        handle_spring_damping=0.1,
        handle_unlock_ratio=0.35,
        door_open_resistance=0.2,
        door_open_damping=0.05,
        door_auto_open_force=0.0,
        door_auto_open_sign=1.0,
        door_auto_open_target_ratio=0.95,
        door_joint_friction=0.5,
        door_joint_damping=0.2,
        handle_joint_friction=0.05,
        handle_joint_damping=0.05,
        robot_vhacd_resolution=300000,
        gripper_shape_contact_offset=0.018,
        gripper_shape_rest_offset=0.003,
        gripper_shape_friction=8.0,
        door_vhacd_resolution=100000,
        enable_wrist_camera=True,
        enable_front_camera=True,
        camera_rgb=False,
        camera_depth=True,
        camera_seg=True,
        show_seg=False,
        handle_seg_id=2,
        camera_depth_clip_lower=0.02,
        camera_depth_clip_far=2.0,
        camera_display_scale=5,
        wrist_camera_down_tilt=0.20,
        front_camera_yaw_deg=0.0,
        front_camera_pitch_deg=-60.0,
        front_camera_roll_deg=0.0,
        sim_substeps=2,
        sim_position_iterations=12,
        sim_velocity_iterations=4,
        sim_contact_offset=0.02,
        sim_rest_offset=0.002,
        sim_max_depenetration_velocity=0.5,
        episode_length_s=20.0,
        mode="both",
        checkpoint=45000,
        log_dir=str(LOW_LEVEL_ROOT / "logs" / "b1z1-low" / "b1z1_locomanip"),
        headless=True,
        rl_device="cuda:0",
        sim_device="cuda:0",
        reward_curriculum="full",
        stagewise_log_path=None,
        stagewise_log_interval=24,
        reach_success_dist=0.08,
        reach_close_dist=0.15,
        reach_hold_steps=10,
        base_approach_dist=0.50,
        base_stop_dist=0.60,
        base_stop_hold_gain=1.0,
        base_stop_hold_max_vx=0.20,
        base_approach_min_vx=0.30,
        base_approach_max_vx=0.55,
        base_approach_vx_gain=0.60,
        base_heading_sigma=0.35,
        stagewise_action_assist=False,
        grasp_entry_dist=0.16,
        grasp_success_dist=0.12,
        grasp_hold_steps=10,
        open_success_angle_deg=20.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def configure_door_runtime(args):
    runtime = door_play.DOOR_RUNTIME
    runtime.clear()
    runtime.update(door_play._load_door_runtime(args.door_cfg))
    if not args.use_all_door_assets:
        door_play._filter_door_runtime_by_names(runtime, door_play.DEFAULT_DOOR_ASSET_NAMES)

    total_door_assets = len(runtime["door_asset_specs"])
    door_asset_count = min(max(1, args.num_envs), total_door_assets)
    for key in ("door_asset_specs", "door_asset_names", "door_bounding_data", "handle_bounding_data"):
        runtime[key] = runtime[key][:door_asset_count]

    runtime.update(
        total_door_asset_count=total_door_assets,
        loaded_door_asset_count=door_asset_count,
        layout_spacing=args.layout_spacing,
        robot_x=args.robot_x,
        robot_y=args.robot_y,
        robot_z=args.robot_z,
        robot_yaw=args.robot_yaw,
        door_x=args.door_x,
        door_y=args.door_y,
        door_actor_scale=args.door_actor_scale,
        robot_y_by_spec=door_play._compute_robot_y_by_spec(
            args.robot_y,
            args.door_y,
            runtime["door_bounding_data"],
            runtime["handle_bounding_data"],
            args.door_actor_scale,
        ),
        door_z_offset=args.door_z_offset,
        box_x=args.box_x,
        box_y=args.box_y,
        gripper_stiffness=args.gripper_stiffness,
        gripper_damping=args.gripper_damping,
        gripper_joint_friction=args.gripper_joint_friction,
        handle_spring_stiffness=args.handle_spring_stiffness,
        handle_spring_damping=args.handle_spring_damping,
        handle_unlock_ratio=args.handle_unlock_ratio,
        door_open_resistance=args.door_open_resistance,
        door_open_damping=args.door_open_damping,
        door_auto_open_force=args.door_auto_open_force,
        door_auto_open_sign=args.door_auto_open_sign,
        door_auto_open_target_ratio=args.door_auto_open_target_ratio,
        door_motion_sign=1.0,
        robot_vhacd_resolution=args.robot_vhacd_resolution,
        gripper_shape_contact_offset=args.gripper_shape_contact_offset,
        gripper_shape_rest_offset=args.gripper_shape_rest_offset,
        gripper_shape_friction=args.gripper_shape_friction,
        door_vhacd_resolution=args.door_vhacd_resolution,
        enable_wrist_camera=args.enable_wrist_camera,
        enable_front_camera=args.enable_front_camera,
        camera_rgb=args.camera_rgb,
        camera_depth=args.camera_depth,
        camera_seg=args.camera_seg,
        show_seg=args.show_seg,
        handle_seg_id=args.handle_seg_id,
        camera_depth_clip_lower=args.camera_depth_clip_lower,
        camera_depth_clip_far=args.camera_depth_clip_far,
        camera_display_scale=args.camera_display_scale,
        wrist_camera_down_tilt=args.wrist_camera_down_tilt,
        front_camera_yaw_deg=args.front_camera_yaw_deg,
        front_camera_pitch_deg=args.front_camera_pitch_deg,
        front_camera_roll_deg=args.front_camera_roll_deg,
        rl_mode=args.mode,
    )
    runtime["door_joint_friction"][0] = args.door_joint_friction
    runtime["door_joint_damping"][0] = args.door_joint_damping
    runtime["door_joint_friction"][1] = args.handle_joint_friction
    runtime["door_joint_damping"][1] = args.handle_joint_damping
    return runtime


def build_low_level_args(args):
    use_gpu = args.sim_device.startswith("cuda")
    return _ns(
        task="b1z1_door_asset_rl",
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


class B1Z1DoorAssetRLEnv(door_play.ManipLocoDoorAsset):
    """Working door-asset sim class with signed pull/push hinge physics.

    This class is intentionally derived from the camera play script's
    ``ManipLocoDoorAsset`` and not from ``B1Z1OpenDoor``.
    """

    def _create_door_actor(self, env_handle, env_i):
        super()._create_door_actor(env_handle, env_i)
        door_handle = self.door_handles[-1]
        door_asset = self.door_asset_list[env_i % len(self.door_asset_specs)]
        door_dof_props = self.gym.get_asset_dof_properties(door_asset)
        actor_dof_props = self.gym.get_actor_dof_properties(env_handle, door_handle)
        actor_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        n = min(self.gym.get_asset_dof_count(door_asset), len(door_play.DOOR_RUNTIME["door_joint_damping"]))
        actor_dof_props["damping"][:n] = np.asarray(door_play.DOOR_RUNTIME["door_joint_damping"][:n], dtype=np.float32)
        actor_dof_props["friction"][:n] = np.asarray(door_play.DOOR_RUNTIME["door_joint_friction"][:n], dtype=np.float32)
        actor_dof_props["effort"][:n] = np.asarray(door_play.DOOR_RUNTIME["door_joint_effort"][:n], dtype=np.float32)
        if len(actor_dof_props["lower"]) >= 1:
            actor_dof_props["lower"][0] = -math.pi / 2
            actor_dof_props["upper"][0] = math.pi / 2
        if len(actor_dof_props["upper"]) >= 2:
            actor_dof_props["upper"][1] = min(float(actor_dof_props["upper"][1]), math.pi / 4)
        self.gym.set_actor_dof_properties(env_handle, door_handle, actor_dof_props)

    def _init_door_tensors(self):
        super()._init_door_tensors()
        self.door_motion_sign = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.signed_door_angle = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.signed_door_open_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
        self.signed_door_angle[:] = self.door_motion_sign * door_angle
        self.signed_door_open_ratio[:] = torch.clamp(self.signed_door_angle / hinge_range, -1.5, 1.5)

        auto_open_active = self.signed_door_angle < door_play.DOOR_RUNTIME["door_auto_open_target_ratio"] * hinge_range
        auto_open_torque = torch.where(
            auto_open_active,
            self.door_motion_sign * door_play.DOOR_RUNTIME["door_auto_open_force"] * door_play.DOOR_RUNTIME["door_auto_open_sign"],
            torch.zeros_like(door_angle),
        )
        door_torques[:, 0] = torch.where(
            self.open_door_stage,
            auto_open_torque
            - door_play.DOOR_RUNTIME["door_open_resistance"] * door_angle
            - door_play.DOOR_RUNTIME["door_open_damping"] * self._door_dof_vel[:, 0],
            -self.door_motion_sign * door_play.DOOR_RUNTIME["door_lock_force"],
        )
        door_torques[:, 1] = (
            -door_play.DOOR_RUNTIME["handle_spring_stiffness"] * handle_angle_from_lower
            - door_play.DOOR_RUNTIME["handle_spring_damping"] * self._door_dof_vel[:, 1]
        )

        handle_range = torch.clamp(
            self.handle_limits_upper[self.door_asset_indices] - self.handle_limits_lower[self.door_asset_indices],
            min=1e-3,
        )
        self.door_open_ratio[:] = torch.clamp(torch.abs(door_angle) / hinge_range, 0.0, 1.5)
        self.handle_open_ratio[:] = torch.clamp(handle_angle_from_lower / handle_range, 0.0, 1.5)
        return door_torques

    def reset_idx(self, env_ids, *args, **kwargs):
        super().reset_idx(env_ids, *args, **kwargs)
        if hasattr(self, "door_motion_sign"):
            mode = door_play.DOOR_RUNTIME.get("rl_mode", "both")
            if mode == "pull":
                signs = torch.ones(len(env_ids), device=self.device)
            elif mode == "push":
                signs = -torch.ones(len(env_ids), device=self.device)
            else:
                # Balanced per-env sampling. Pull=+1, push=-1.
                signs = torch.where(
                    torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(-1) < 0.5,
                    torch.ones(len(env_ids), device=self.device),
                    -torch.ones(len(env_ids), device=self.device),
                )
            self.door_motion_sign[env_ids] = signs


class DoorAssetRLVecEnv:
    """High-level RL wrapper around the working low-level door asset env."""

    def __init__(self, args, mode="teacher", eval_mode=False):
        self.args = args
        self.mode = mode
        self.eval_mode = eval_mode
        self.device = torch.device(args.rl_device)
        self.rl_device = args.rl_device
        self.num_agents = 1
        self.num_observations = TEACHER_OBS_DIM
        self.num_states = STUDENT_OBS_DIM if mode == "student" else 0
        self.num_actions = ACTION_DIM
        self.obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.num_observations,), dtype=np.float32)
        self.state_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.num_states,), dtype=np.float32)
        self.act_space = gym.spaces.Box(-1.0, 1.0, shape=(self.num_actions,), dtype=np.float32)

        configure_door_runtime(args)
        low_args = build_low_level_args(args)
        env_cfg, train_cfg = task_registry.get_cfgs(name="b1z1")
        task_registry.register("b1z1_door_asset_rl", B1Z1DoorAssetRLEnv, env_cfg, train_cfg, "b1z1")

        env_cfg.sim.substeps = args.sim_substeps
        env_cfg.sim.physx.num_position_iterations = args.sim_position_iterations
        env_cfg.sim.physx.num_velocity_iterations = args.sim_velocity_iterations
        env_cfg.sim.physx.contact_offset = args.sim_contact_offset
        env_cfg.sim.physx.rest_offset = args.sim_rest_offset
        env_cfg.sim.physx.max_depenetration_velocity = args.sim_max_depenetration_velocity
        env_cfg.env.num_envs = args.num_envs
        env_cfg.env.episode_length_s = args.episode_length_s
        side = max(2, int(math.ceil(math.sqrt(args.num_envs))))
        env_cfg.terrain.num_rows = side
        env_cfg.terrain.num_cols = side
        env_cfg.terrain.height = [0.0, 0.0]
        env_cfg.commands.curriculum = False
        env_cfg.env.observe_gait_commands = True
        env_cfg.commands.ranges.lin_vel_x = [-args.max_vx, args.max_vx]
        env_cfg.commands.ranges.ang_vel_yaw = [-args.max_yaw, args.max_yaw]
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.noise.add_noise = False
        env_cfg.init_state.rand_yaw_range = args.init_yaw_noise
        env_cfg.init_state.origin_perturb_range = args.init_xy_noise
        env_cfg.init_state.init_vel_perturb_range = 0.0

        self.low_env, _ = task_registry.make_env(name="b1z1_door_asset_rl", args=low_args, env_cfg=env_cfg)
        self.num_envs = self.low_env.num_envs
        self.max_episode_length = int(self.low_env.max_episode_length)
        runner, _, _, _ = task_registry.make_alg_runner(
            log_root=args.log_dir,
            env=self.low_env,
            name="b1z1",
            args=low_args,
            train_cfg=train_cfg,
            return_log_dir=True,
        )
        self.low_policy = runner.get_inference_policy(device=self.low_env.device, stochastic=False)
        self.low_env.external_ee_goal_control = True
        self.low_env.external_pos_gain = args.external_pos_gain
        self.low_env.external_orn_gain = args.external_orn_gain
        self.low_env.freeze_arm_default = torch.zeros(self.num_envs, device=self.low_env.device, dtype=torch.bool)
        self.low_env.freeze_arm_zero = torch.zeros(self.num_envs, device=self.low_env.device, dtype=torch.bool)
        self.low_env.external_gripper_target = torch.full(
            (self.num_envs, self.low_env.cfg.env.num_gripper_joints),
            args.gripper_open,
            device=self.low_env.device,
        )
        if args.reward_curriculum not in REWARD_CURRICULA:
            raise ValueError(f"Unknown reward_curriculum: {args.reward_curriculum}")
        self.last_high_action = torch.zeros(self.num_envs, ACTION_DIM, device=self.low_env.device)
        self.prev_high_action = torch.zeros(self.num_envs, ACTION_DIM, device=self.low_env.device)
        self.prev_signed_angle = torch.zeros(self.num_envs, device=self.low_env.device)
        self.prev_handle_ratio = torch.zeros(self.num_envs, device=self.low_env.device)
        self.prev_open_ratio = torch.zeros(self.num_envs, device=self.low_env.device)
        self.prev_pass_distance = torch.zeros(self.num_envs, device=self.low_env.device)
        self.prev_ee_to_handle = torch.zeros(self.num_envs, device=self.low_env.device)
        self.prev_ee_goal_to_handle = torch.zeros(self.num_envs, device=self.low_env.device)
        self.prev_arm_base_to_handle = torch.zeros(self.num_envs, device=self.low_env.device)
        self.best_ee_to_handle = torch.full((self.num_envs,), float("inf"), device=self.low_env.device)
        self.best_ee_goal_to_handle = torch.full((self.num_envs,), float("inf"), device=self.low_env.device)
        self.best_arm_base_to_handle = torch.full((self.num_envs,), float("inf"), device=self.low_env.device)
        self.phase_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.low_env.device)
        self.phase_hold_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.low_env.device)
        self.reach_hold_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.low_env.device)
        self.grasp_hold_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.low_env.device)
        self.phase_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.low_env.device)
        self.reach_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.low_env.device)
        self.grasp_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.low_env.device)
        self.base_stop_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.low_env.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.low_env.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.low_env.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.low_env.device)
        self.stagewise_global_step = 0
        self.ee_goal_clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.low_env.device)
        self._image_history = torch.zeros(self.num_envs, IMAGE_CHANNELS, IMAGE_H, IMAGE_W, device=self.low_env.device)
        self.stagewise_log_interval = max(0, int(getattr(args, "stagewise_log_interval", 0) or 0))
        self.stagewise_log_path = Path(args.stagewise_log_path) if getattr(args, "stagewise_log_path", None) else None
        if self.stagewise_log_path is not None:
            self.stagewise_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_reward_info = None
        print("Door RL env uses play script ManipLocoDoorAsset, not B1Z1OpenDoor.")
        print("Door assets:", self.low_env.door_asset_names)
        print("Door RL reward curriculum:", args.reward_curriculum)

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    def reset(self):
        self.low_env.reset()
        low_obs = self.low_env.get_observations()
        self.low_env.external_ee_goal_control = True
        self.low_env.curr_ee_goal_cart_world[:] = self.low_env.ee_pos
        self.low_env.ee_goal_orn_quat[:] = self.low_env.ee_orn
        self.low_env.ee_goal_orn_delta_rpy[:] = 0.0
        self.low_env.external_gripper_target[:] = self.args.gripper_open
        self.progress_buf.zero_()
        self.reset_buf.zero_()
        self.success_buf.zero_()
        self.ee_goal_clamped.zero_()
        self.prev_signed_angle[:] = getattr(self.low_env, "signed_door_angle", self.low_env._door_dof_pos[:, 0])
        self.prev_high_action.zero_()
        self.last_high_action.zero_()
        self._reset_phase_state()
        if self.mode == "student":
            self._update_image_history()
        return self._obs_dict()

    def step(self, actions):
        actions = torch.clamp(actions.to(self.low_env.device), -1.0, 1.0)
        self._apply_high_level_action(actions)
        low_obs = self.low_env.get_observations()
        low_actions = self.low_policy(low_obs.detach(), hist_encoding=True)
        low_obs, _, _, _, dones, infos = self.low_env.step(low_actions.detach())
        self.progress_buf += 1
        self.stagewise_global_step += 1
        reward, reward_info = self._compute_reward(actions)
        timeout = self.progress_buf >= min(self.max_episode_length, self.args.max_episode_steps)
        terminated = dones.to(torch.bool) | timeout | self.success_buf
        door_metrics = {
            "signed_angle_deg": reward_info["signed_angle_deg"],
            "signed_open_ratio": reward_info["signed_open_ratio"],
            "handle_open_ratio": reward_info["handle"],
            "ee_to_handle_m": reward_info["ee_to_handle_m"],
            "ee_goal_to_handle_m": reward_info["ee_goal_to_handle_m"],
            "ee_goal_clamped_rate": reward_info["ee_goal_clamped"],
            "arm_base_to_handle_m": reward_info["arm_base_to_handle_m"],
            "base_tilt": reward_info["base_tilt"],
            "action_l2": reward_info["action_l2"],
            "base_vx_cmd": reward_info["base_vx_cmd"],
            "base_yaw_cmd": reward_info["base_yaw_cmd"],
        }
        success_rate = {
            "Success / reach": reward_info["reach_success"].detach().float().cpu().numpy(),
            "Success / grasp": reward_info["grasp_success"].detach().float().cpu().numpy(),
            "Success / phase": reward_info["phase_success"].detach().float().cpu().numpy(),
            "Success / open_80deg": reward_info["open_bonus"].detach().float().cpu().numpy(),
            "Success / passed_door": reward_info["pass_bonus"].detach().float().cpu().numpy(),
        }
        self.reset_buf[:] = terminated
        if torch.any(terminated):
            reset_ids = torch.nonzero(terminated, as_tuple=False).squeeze(-1)
            self.low_env.reset_idx(reset_ids)
            self.progress_buf[terminated] = 0
            self.success_buf[terminated] = False
            self.low_env.curr_ee_goal_cart_world[terminated] = self.low_env.ee_pos[terminated]
            self.low_env.ee_goal_orn_quat[terminated] = self.low_env.ee_orn[terminated]
            self.low_env.ee_goal_orn_delta_rpy[terminated] = 0.0
            self.low_env.external_gripper_target[terminated] = self.args.gripper_open
            self._reset_phase_state(reset_ids)
        if self.mode == "student":
            self._update_image_history()
        info = {
            "reward_terms": reward_info,
            "door_metrics": door_metrics,
            "phase_metrics": {
                "id": reward_info["phase_id"],
                "reach_success_rate": reward_info["reach_success"],
                "grasp_success_rate": reward_info["grasp_success"],
                "phase_success_rate": reward_info["phase_success"],
            },
            "success_rate": success_rate,
            "lifted_now": self.success_buf.to(torch.float32).view(-1, 1),
        }
        self._write_stagewise_log(reward_info)
        return self._obs_dict(), reward, terminated.to(torch.float32), info

    def render(self, *args, **kwargs):
        return self.low_env.render(*args, **kwargs)

    def close(self):
        pass

    def _apply_high_level_action(self, actions):
        self.prev_high_action[:] = self.last_high_action
        self.last_high_action[:] = actions
        base_assisted_curriculum = self.args.reward_curriculum in ("reach", "grasp", "handle", "open")
        use_action_assist = base_assisted_curriculum and bool(getattr(self.args, "stagewise_action_assist", False))
        if use_action_assist:
            handle_goal = self._handle_goal()
            arm_base_pos = self._arm_base_pos()
            arm_base_to_handle = torch.norm(arm_base_pos[:, :2] - handle_goal[:, :2], dim=-1)
            ee_to_handle_for_action = torch.norm(self.low_env.ee_pos - handle_goal, dim=-1)
            self.base_stop_latch[:] = self.base_stop_latch | (arm_base_to_handle <= self.args.base_stop_dist)
            approach_active = ~self.base_stop_latch
            close_allowed = (~approach_active) & (ee_to_handle_for_action <= self.args.grasp_entry_dist)
            handle_in_base = quat_rotate_inverse(self.low_env.root_states[:, 3:7], handle_goal - arm_base_pos)
            stop_hold_vx = torch.clamp(
                (handle_in_base[:, 0] - self.args.base_stop_dist) * self.args.base_stop_hold_gain,
                min=-self.args.base_stop_hold_max_vx,
                max=self.args.base_stop_hold_max_vx,
            )
            target_vx_far = torch.clamp(
                self.args.base_approach_min_vx
                + (arm_base_to_handle - self.args.base_stop_dist) * self.args.base_approach_vx_gain,
                min=self.args.base_approach_min_vx,
                max=min(self.args.base_approach_max_vx, self.args.max_vx),
            )
            approach_target_vx = torch.where(approach_active, target_vx_far, torch.zeros_like(target_vx_far))
        else:
            approach_active = torch.zeros(self.num_envs, device=self.low_env.device, dtype=torch.bool)
            close_allowed = torch.ones(self.num_envs, device=self.low_env.device, dtype=torch.bool)
            approach_target_vx = torch.zeros(self.num_envs, device=self.low_env.device)
        active_reach = ~approach_active

        pos_delta = actions[:, :3] * self.args.ee_delta_scale
        orn_delta = actions[:, 3:6] * self.args.orn_delta_scale
        if use_action_assist:
            pos_delta = torch.where(approach_active.unsqueeze(-1), torch.zeros_like(pos_delta), pos_delta)
            orn_delta = torch.where(approach_active.unsqueeze(-1), torch.zeros_like(orn_delta), orn_delta)

        if use_action_assist and torch.any(approach_active):
            self.low_env.freeze_arm_default[approach_active] = True
            self.low_env.freeze_arm_zero[approach_active] = False
            self.low_env.curr_ee_goal_cart_world[approach_active] = self.low_env.ee_pos[approach_active]
            self.low_env.ee_goal_orn_quat[approach_active] = self.low_env.ee_orn[approach_active]
            self.low_env.ee_goal_orn_delta_rpy[approach_active] = 0.0
        if torch.any(active_reach):
            self.low_env.freeze_arm_default[active_reach] = False
            self.low_env.freeze_arm_zero[active_reach] = False

        self.low_env.curr_ee_goal_cart_world[:] = self.low_env.curr_ee_goal_cart_world + pos_delta
        center = self.low_env._get_ee_goal_spherical_center()
        rel = self.low_env.curr_ee_goal_cart_world - center
        rel_norm = torch.norm(rel, dim=-1, keepdim=True)
        too_far = rel_norm.squeeze(-1) > self.args.ee_max_radius
        self.ee_goal_clamped[:] = too_far
        rel = torch.where(too_far.unsqueeze(-1), rel / torch.clamp(rel_norm, min=1e-6) * self.args.ee_max_radius, rel)
        self.low_env.curr_ee_goal_cart_world[:] = center + rel
        self.low_env.ee_goal_orn_delta_rpy[:] = torch.clamp(
            self.low_env.ee_goal_orn_delta_rpy + orn_delta,
            -self.args.max_orn_delta,
            self.args.max_orn_delta,
        )
        base_quat = self.low_env.base_quat / torch.clamp(torch.norm(self.low_env.base_quat, dim=-1, keepdim=True), min=1e-6)
        default_orn = quat_mul(base_quat, self.low_env.default_ee_orn_local_quat)
        delta_orn = quat_from_euler_xyz(
            self.low_env.ee_goal_orn_delta_rpy[:, 0],
            self.low_env.ee_goal_orn_delta_rpy[:, 1],
            self.low_env.ee_goal_orn_delta_rpy[:, 2],
        )
        self.low_env.ee_goal_orn_quat[:] = quat_mul(default_orn, delta_orn)
        gripper_action = actions[:, 6:7]
        if use_action_assist:
            force_gripper_open = approach_active
            if self.args.reward_curriculum in ("grasp", "handle", "open"):
                force_gripper_open = force_gripper_open | (~close_allowed)
            force_gripper_close = (
                close_allowed
                if self.args.reward_curriculum in ("grasp", "handle", "open")
                else torch.zeros_like(close_allowed)
            )
            gripper_action = torch.where(force_gripper_open.unsqueeze(-1), -torch.ones_like(gripper_action), gripper_action)
            gripper_action = torch.where(force_gripper_close.unsqueeze(-1), torch.ones_like(gripper_action), gripper_action)
        gripper = (gripper_action + 1.0) * 0.5
        self.low_env.external_gripper_target[:] = self.args.gripper_open + gripper * (
            self.args.gripper_closed - self.args.gripper_open
        )
        policy_vx_cmd = actions[:, 7] * self.args.max_vx
        if use_action_assist:
            hold_vx = stop_hold_vx if self.args.reward_curriculum == "reach" else torch.zeros_like(approach_target_vx)
            self.low_env.commands[:, 0] = torch.where(approach_active, approach_target_vx, hold_vx)
        else:
            self.low_env.commands[:, 0] = policy_vx_cmd
        self.low_env.commands[:, 1] = 0.0
        if use_action_assist:
            self.low_env.commands[:, 2] = 0.0
        else:
            self.low_env.commands[:, 2] = actions[:, 8] * self.args.max_yaw

    def _handle_goal(self):
        handle_state = self.low_env._rigid_body_state[:, self.low_env.handle_body_idx, :]
        handle_pos = handle_state[:, :3]
        handle_rot = handle_state[:, 3:7]
        return quat_apply(handle_rot, self.low_env.goal_pos_offset_tensor) + handle_pos

    def _arm_base_pos(self):
        arm_base_pos = self.low_env.base_pos
        if hasattr(self.low_env, "arm_base_offset"):
            arm_base_pos = self.low_env.base_pos + quat_apply(self.low_env.base_yaw_quat, self.low_env.arm_base_offset)
        return arm_base_pos

    def _reset_phase_state(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.low_env.device)
        handle_goal = self._handle_goal()
        arm_base_pos = self._arm_base_pos()
        ee_to_handle = torch.norm(self.low_env.ee_pos[env_ids] - handle_goal[env_ids], dim=-1)
        ee_goal_to_handle = torch.norm(self.low_env.curr_ee_goal_cart_world[env_ids] - handle_goal[env_ids], dim=-1)
        arm_base_to_handle = torch.norm(arm_base_pos[env_ids, :2] - handle_goal[env_ids, :2], dim=-1)
        self.prev_ee_to_handle[env_ids] = ee_to_handle
        self.prev_ee_goal_to_handle[env_ids] = ee_goal_to_handle
        self.prev_arm_base_to_handle[env_ids] = arm_base_to_handle
        self.best_ee_to_handle[env_ids] = ee_to_handle
        self.best_ee_goal_to_handle[env_ids] = ee_goal_to_handle
        self.best_arm_base_to_handle[env_ids] = arm_base_to_handle
        self.prev_signed_angle[env_ids] = getattr(self.low_env, "signed_door_angle", self.low_env._door_dof_pos[:, 0])[env_ids]
        self.prev_handle_ratio[env_ids] = self.low_env.handle_open_ratio[env_ids]
        open_ratio = getattr(self.low_env, "signed_door_open_ratio", self.low_env.door_open_ratio)
        self.prev_open_ratio[env_ids] = open_ratio[env_ids]
        self.prev_pass_distance[env_ids] = self._pass_distance()[env_ids]
        self.phase_id[env_ids] = 0
        self.phase_hold_buf[env_ids] = 0
        self.reach_hold_buf[env_ids] = 0
        self.grasp_hold_buf[env_ids] = 0
        self.phase_success_buf[env_ids] = False
        self.reach_success_buf[env_ids] = False
        self.grasp_success_buf[env_ids] = False
        self.base_stop_latch[env_ids] = False

    def _pass_center(self):
        pass_center = self.low_env.door_root_state[:, :2].clone()
        pass_center[:, 1] += self.args.pass_left_offset * torch.sign(
            self.low_env.root_states[:, 0] - self.low_env.door_root_state[:, 0]
        )
        return pass_center

    def _pass_distance(self):
        return torch.norm(self._pass_center() - self.low_env.root_states[:, :2], dim=-1)

    def _gripper_reward(self, target):
        denom = max(abs(self.args.gripper_closed - self.args.gripper_open), 1e-6)
        gripper_error = torch.abs(self.low_env.external_gripper_target.mean(dim=-1) - target) / denom
        return torch.clamp(1.0 - gripper_error, 0.0, 1.0)

    def _actual_gripper_closed(self):
        gripper_pos = self.low_env.dof_pos[:, -self.low_env.cfg.env.num_gripper_joints :].mean(dim=-1)
        denom = self.args.gripper_closed - self.args.gripper_open
        if abs(denom) < 1e-6:
            return torch.zeros_like(gripper_pos)
        return torch.clamp((gripper_pos - self.args.gripper_open) / denom, 0.0, 1.0)

    def _body_pos(self, name, fallback=None):
        body_map = getattr(self.low_env, "body_names_to_idx", {})
        idx = body_map.get(name)
        if idx is None:
            return self.low_env.ee_pos if fallback is None else fallback
        return self.low_env._rigid_body_state[:, idx, :3]

    def _update_stagewise_phase(self, ee_to_handle, handle_progress, open_ratio):
        if self.args.reward_curriculum == "reach":
            self.phase_id[:] = 0
            return
        if self.args.reward_curriculum == "grasp":
            self.phase_id[:] = torch.where(
                ee_to_handle < self.args.grasp_entry_dist,
                torch.ones_like(self.phase_id),
                torch.zeros_like(self.phase_id),
            )
            return
        if self.args.reward_curriculum == "handle":
            grasp_phase = torch.ones_like(self.phase_id)
            reach_phase = torch.zeros_like(self.phase_id)
            handle_phase = torch.full_like(self.phase_id, 2)
            self.phase_id[:] = torch.where(
                handle_progress > 0.05,
                handle_phase,
                torch.where(ee_to_handle < self.args.grasp_entry_dist, grasp_phase, reach_phase),
            )
            return
        if self.args.reward_curriculum == "open":
            reach_phase = torch.zeros_like(self.phase_id)
            grasp_phase = torch.ones_like(self.phase_id)
            handle_phase = torch.full_like(self.phase_id, 2)
            open_phase = torch.full_like(self.phase_id, 3)
            self.phase_id[:] = torch.where(
                open_ratio > 0.05,
                open_phase,
                torch.where(
                    handle_progress > 0.05,
                    handle_phase,
                    torch.where(ee_to_handle < self.args.grasp_entry_dist, grasp_phase, reach_phase),
                ),
            )
            return
        if self.args.reward_curriculum == "pass":
            open_enough = getattr(self.low_env, "signed_door_angle", self.low_env._door_dof_pos[:, 0]) >= math.radians(
                self.args.open_success_angle_deg
            )
            reach_phase = torch.zeros_like(self.phase_id)
            open_phase = torch.full_like(self.phase_id, 3)
            pass_phase = torch.full_like(self.phase_id, 4)
            self.phase_id[:] = torch.where(open_enough, pass_phase, torch.where(open_ratio > 0.05, open_phase, reach_phase))

    def _compute_reward(self, actions):
        handle_goal = self._handle_goal()
        arm_base_pos = self._arm_base_pos()
        ee_to_handle = torch.norm(self.low_env.ee_pos - handle_goal, dim=-1)
        ee_goal_to_handle = torch.norm(self.low_env.curr_ee_goal_cart_world - handle_goal, dim=-1)
        arm_base_to_handle = torch.norm(arm_base_pos[:, :2] - handle_goal[:, :2], dim=-1)
        to_handle_xy = handle_goal[:, :2] - self.low_env.root_states[:, :2]
        desired_base_yaw = torch.atan2(to_handle_xy[:, 1], to_handle_xy[:, 0])
        base_yaw = euler_from_quat(self.low_env.root_states[:, 3:7])[2]
        base_heading_error = torch.abs(_wrap_to_pi(desired_base_yaw - base_yaw))
        base_heading_align = torch.exp(-base_heading_error / max(self.args.base_heading_sigma, 1e-4))
        handle_in_base = quat_rotate_inverse(self.low_env.root_states[:, 3:7], handle_goal - self.low_env.root_states[:, :3])
        base_lateral_error = torch.abs(handle_in_base[:, 1])
        base_lateral_align = torch.exp(-5.0 * base_lateral_error)
        arm_handle_in_base = quat_rotate_inverse(self.low_env.root_states[:, 3:7], handle_goal - arm_base_pos)
        stop_hold_vx = torch.clamp(
            (arm_handle_in_base[:, 0] - self.args.base_stop_dist) * self.args.base_stop_hold_gain,
            min=-self.args.base_stop_hold_max_vx,
            max=self.args.base_stop_hold_max_vx,
        )
        reach_dense = torch.exp(-4.0 * ee_to_handle)
        reach_progress = torch.clamp(self.prev_ee_to_handle - ee_to_handle, min=-0.02, max=0.05)
        reach_goal_dense = torch.exp(-4.0 * ee_goal_to_handle)
        reach_goal_progress = torch.clamp(self.prev_ee_goal_to_handle - ee_goal_to_handle, min=-0.02, max=0.05)
        base_reachable_margin = torch.clamp(arm_base_to_handle - self.args.ee_max_radius * 0.8, min=0.0)
        base_reach_dense = torch.exp(-3.0 * base_reachable_margin)
        base_reach_progress = torch.clamp(self.prev_arm_base_to_handle - arm_base_to_handle, min=-0.02, max=0.05)
        base_stop_release = arm_base_to_handle > (self.args.base_stop_dist + 0.20)
        self.base_stop_latch[:] = torch.where(
            base_stop_release,
            torch.zeros_like(self.base_stop_latch),
            self.base_stop_latch | (arm_base_to_handle <= self.args.base_stop_dist),
        )
        base_approach_active = ~self.base_stop_latch
        base_approach_active_f = base_approach_active.to(torch.float32)
        base_reach_active_f = 1.0 - base_approach_active_f
        target_vx_far = torch.clamp(
            self.args.base_approach_min_vx
            + (arm_base_to_handle - self.args.base_stop_dist) * self.args.base_approach_vx_gain,
            min=self.args.base_approach_min_vx,
            max=min(self.args.base_approach_max_vx, self.args.max_vx),
        )
        hold_target_vx = stop_hold_vx if self.args.reward_curriculum == "reach" else torch.zeros_like(stop_hold_vx)
        target_vx = torch.where(base_approach_active, target_vx_far, hold_target_vx)
        vx_cmd = self.low_env.commands[:, 0]
        yaw_cmd = self.low_env.commands[:, 2]
        policy_vx_cmd = actions[:, 7] * self.args.max_vx
        policy_yaw_cmd = actions[:, 8] * self.args.max_yaw
        base_vx_tracking = torch.exp(-5.0 * torch.abs(vx_cmd - target_vx))
        policy_base_vx_tracking = torch.exp(-5.0 * torch.abs(policy_vx_cmd - target_vx))
        policy_base_vx_signed = torch.clamp(policy_vx_cmd / max(self.args.max_vx, 1e-4), -1.0, 1.0) * torch.clamp(
            target_vx / max(self.args.max_vx, 1e-4),
            0.0,
            1.0,
        )
        policy_base_yaw_stop = torch.exp(-10.0 * torch.abs(policy_yaw_cmd))
        base_vx_stop = torch.exp(-8.0 * torch.abs(vx_cmd))
        base_yaw_stop = torch.exp(-10.0 * torch.abs(yaw_cmd))
        base_reach_hold = torch.exp(-4.0 * torch.clamp(arm_base_to_handle - self.args.base_stop_dist, min=0.0))
        distance_over_base_stop = torch.clamp(arm_base_to_handle - self.args.base_stop_dist, min=0.0)
        base_under_70cm = (arm_base_to_handle < 0.70).to(torch.float32)
        base_under_60cm = (arm_base_to_handle < 0.60).to(torch.float32)
        base_under_50cm = (arm_base_to_handle < self.args.base_approach_dist).to(torch.float32)
        base_under_stop_dist = (arm_base_to_handle < self.args.base_stop_dist).to(torch.float32)
        self.best_ee_to_handle[:] = torch.minimum(self.best_ee_to_handle, ee_to_handle)
        self.best_ee_goal_to_handle[:] = torch.minimum(self.best_ee_goal_to_handle, ee_goal_to_handle)
        self.best_arm_base_to_handle[:] = torch.minimum(self.best_arm_base_to_handle, arm_base_to_handle)
        reach_close_bonus = (ee_to_handle < self.args.reach_close_dist).to(torch.float32)
        reach_under_25cm = (ee_to_handle < 0.25).to(torch.float32)
        reach_under_20cm = (ee_to_handle < 0.20).to(torch.float32)
        reach_under_15cm = (ee_to_handle < 0.15).to(torch.float32)
        reach_under_12cm = (ee_to_handle < 0.12).to(torch.float32)
        reach_under_08cm = (ee_to_handle < 0.08).to(torch.float32)
        reach_now = ee_to_handle < self.args.reach_success_dist
        self.reach_hold_buf[:] = torch.where(reach_now, self.reach_hold_buf + 1, torch.zeros_like(self.reach_hold_buf))
        reach_success = (self.reach_hold_buf >= self.args.reach_hold_steps).to(torch.float32)
        self.reach_success_buf[:] = reach_success > 0.5

        handle_progress = self.low_env.handle_open_ratio.clamp(0.0, 1.0)
        signed_angle = getattr(self.low_env, "signed_door_angle", self.low_env._door_dof_pos[:, 0])
        signed_progress = signed_angle - self.prev_signed_angle
        open_ratio = getattr(self.low_env, "signed_door_open_ratio", self.low_env.door_open_ratio).clamp(-1.0, 1.2)
        door_open_abs_ratio = self.low_env.door_open_ratio.clamp(0.0, 1.2)
        open_reward = door_open_abs_ratio
        open_bonus = (torch.abs(signed_angle) >= math.radians(self.args.open_success_angle_deg)).to(torch.float32)
        to_pass = self._pass_center() - self.low_env.root_states[:, :2]
        pass_distance = torch.norm(to_pass, dim=-1)
        pass_progress = torch.clamp(self.prev_pass_distance - pass_distance, min=-0.02, max=0.05)
        pass_align = torch.exp(-torch.norm(to_pass, dim=-1))
        pass_success_x = self.low_env.door_root_state[:, 0] - self.low_env.root_states[:, 0]
        pass_bonus = ((torch.abs(signed_angle) >= math.radians(self.args.open_success_angle_deg)) & (pass_success_x > 0.7)).to(torch.float32)
        action_penalty = torch.sum(actions * actions, dim=-1)
        action_pos_l2 = torch.norm(actions[:, :3], dim=-1)
        action_ee_pose_l2 = torch.norm(actions[:, :6], dim=-1)
        ee_pose_action_stop = torch.exp(-4.0 * action_ee_pose_l2)
        ee_action_to_handle = handle_goal - self.low_env.ee_pos
        desired_ee_pos_action = torch.clamp(ee_action_to_handle / max(self.args.ee_delta_scale, 1e-4), -1.0, 1.0)
        ee_pos_action_tracking = torch.exp(-2.0 * torch.norm(actions[:, :3] - desired_ee_pos_action, dim=-1))
        ee_action_norm = torch.norm(actions[:, :3], dim=-1, keepdim=True)
        ee_to_handle_norm = torch.norm(ee_action_to_handle, dim=-1, keepdim=True)
        ee_pos_action_alignment = torch.sum(
            actions[:, :3] / torch.clamp(ee_action_norm, min=1e-6)
            * ee_action_to_handle / torch.clamp(ee_to_handle_norm, min=1e-6),
            dim=-1,
        ).clamp(min=0.0)
        ee_goal_clamp_penalty = self.ee_goal_clamped.to(torch.float32)
        tilt = torch.abs(self.low_env.projected_gravity[:, 0]) + torch.abs(self.low_env.projected_gravity[:, 1])
        gripper_open = self._gripper_reward(self.args.gripper_open)
        gripper_closed = self._gripper_reward(self.args.gripper_closed)
        actual_gripper_closed = self._actual_gripper_closed()
        actual_gripper_open = 1.0 - actual_gripper_closed

        handle_state = self.low_env._rigid_body_state[:, self.low_env.handle_body_idx, :]
        handle_rot = handle_state[:, 3:7]
        ee_orn = self.low_env.ee_orn / torch.clamp(torch.norm(self.low_env.ee_orn, dim=-1, keepdim=True), min=1e-6)
        handle_rot = handle_rot / torch.clamp(torch.norm(handle_rot, dim=-1, keepdim=True), min=1e-6)
        ee_handle_alignment = torch.abs(torch.sum(ee_orn * handle_rot, dim=-1))

        unit_x = torch.zeros_like(handle_goal)
        unit_y = torch.zeros_like(handle_goal)
        unit_z = torch.zeros_like(handle_goal)
        unit_x[:, 0] = 1.0
        unit_y[:, 1] = 1.0
        unit_z[:, 2] = 1.0
        handle_x = quat_apply(handle_rot, unit_x)
        handle_y = quat_apply(handle_rot, unit_y)
        handle_z_axis = quat_apply(handle_rot, unit_z)
        ee_x = quat_apply(ee_orn, unit_x)
        ee_z = quat_apply(ee_orn, unit_z)
        align_z = torch.sum(ee_z * -handle_x, dim=-1)
        align_x = torch.sum(ee_x * -handle_y, dim=-1)
        align_ee_handle = 0.5 * (torch.sign(align_z) * align_z * align_z + torch.sign(align_x) * align_x * align_x)
        approach_ee_handle = torch.pow(1.0 / (1.0 + ee_to_handle * ee_to_handle), 2)
        approach_ee_handle = torch.where(ee_to_handle <= 0.20, 2.0 * approach_ee_handle, approach_ee_handle)

        stator_pos = self._body_pos("gripperStator", self.low_env.ee_pos)
        mover_pos = self._body_pos("gripperMover", self.low_env.ee_pos)
        finger_mid = 0.5 * (stator_pos + mover_pos)
        finger_mid_to_handle = torch.norm(finger_mid - handle_goal, dim=-1)
        finger_xy_to_handle = torch.norm(finger_mid[:, :2] - handle_goal[:, :2], dim=-1)
        finger_low_z = torch.minimum(stator_pos[:, 2], mover_pos[:, 2])
        finger_high_z = torch.maximum(stator_pos[:, 2], mover_pos[:, 2])
        handle_z = handle_goal[:, 2]
        grasp_z_margin = 0.035
        grasp_xy_margin = 0.12
        grasp_around = (
            (finger_low_z <= handle_z + grasp_z_margin)
            & (finger_high_z >= handle_z - grasp_z_margin)
            & (finger_xy_to_handle <= grasp_xy_margin)
        ).to(torch.float32)
        approach_gripper_handle = torch.exp(-8.0 * finger_mid_to_handle)
        finger_close_bonus = (finger_mid_to_handle <= 0.10).to(torch.float32)
        gripper_close_action = torch.clamp((actions[:, 6] + 1.0) * 0.5, min=0.0, max=1.0)
        gripper_open_action = 1.0 - gripper_close_action
        grasp_close_score = 0.5 * gripper_closed + 0.5 * actual_gripper_closed
        grasp_handle_reward = (ee_to_handle <= self.args.grasp_success_dist).to(torch.float32) * grasp_close_score
        hold_close = torch.exp(-8.0 * ee_to_handle)

        self._update_stagewise_phase(ee_to_handle, handle_progress, open_reward)
        handle_delta = torch.clamp(handle_progress - self.prev_handle_ratio, min=-0.01, max=0.03)
        open_delta = torch.clamp(open_reward - self.prev_open_ratio, min=-0.01, max=0.03)
        abs_signed_angle = torch.abs(signed_angle)
        open_easy = (abs_signed_angle > math.radians(2.0)).to(torch.float32) * 0.5
        open_medium = (abs_signed_angle > math.radians(10.0)).to(torch.float32) * grasp_around
        open_hard = (abs_signed_angle > math.radians(self.args.open_success_angle_deg)).to(torch.float32) * grasp_around
        multi_stage_open = open_easy + open_medium + open_hard
        base_still_reward = 0.5 * (base_vx_stop + base_yaw_stop)
        policy_base_still_reward = 0.5 * (
            torch.exp(-8.0 * torch.abs(policy_vx_cmd)) + torch.exp(-10.0 * torch.abs(policy_yaw_cmd))
        )
        action_rate_l2 = torch.sum((actions - self.prev_high_action) * (actions - self.prev_high_action), dim=-1)
        approach_dir = self.low_env.root_states[:, :3] - handle_goal
        approach_dir[:, 2] = 0.0
        approach_dir = approach_dir / torch.clamp(torch.norm(approach_dir, dim=-1, keepdim=True), min=1e-6)
        pull_dir = handle_z_axis.clone()
        pull_dir[:, 2] = 0.0
        pull_dir_norm = torch.norm(pull_dir, dim=-1, keepdim=True)
        pull_dir = torch.where(pull_dir_norm > 1e-4, pull_dir / torch.clamp(pull_dir_norm, min=1e-6), approach_dir)
        pull_sign = torch.where(
            torch.sum(pull_dir * approach_dir, dim=-1, keepdim=True) < 0.0,
            -torch.ones_like(pull_dir_norm),
            torch.ones_like(pull_dir_norm),
        )
        pull_dir = pull_dir * pull_sign
        desired_pull_action = torch.clamp(pull_dir / max(self.args.ee_delta_scale, 1e-4), -1.0, 1.0)
        open_pull_action_tracking = torch.exp(-2.0 * torch.norm(actions[:, :3] - desired_pull_action, dim=-1))
        open_pull_action_alignment = torch.sum(
            actions[:, :3] / torch.clamp(torch.norm(actions[:, :3], dim=-1, keepdim=True), min=1e-6) * pull_dir,
            dim=-1,
        ).clamp(min=0.0)
        reach_stage = (ee_to_handle > self.args.grasp_entry_dist).to(torch.float32)
        grasp_ready = 1.0 - reach_stage
        grasped_stage = (
            (ee_to_handle <= max(self.args.grasp_success_dist, 0.12))
            & (gripper_closed > 0.75)
            & (actual_gripper_closed > 0.35)
        ).to(torch.float32)
        grasp_stage = grasp_ready * (1.0 - grasped_stage)
        open_stage = grasp_ready * grasped_stage
        distance_over_grasp = torch.clamp(ee_to_handle - self.args.grasp_entry_dist, min=0.0)
        distance_over_hold = torch.clamp(ee_to_handle - max(self.args.grasp_success_dist, 0.12), min=0.0)
        base_approach_reward = (
            10.0 * base_vx_tracking
            + 80.0 * policy_base_vx_tracking
            + 30.0 * policy_base_vx_signed
            + 180.0 * base_reach_progress
            + 12.0 * base_heading_align
            + 4.0 * base_lateral_align
            + 8.0 * base_yaw_stop
            + 30.0 * policy_base_yaw_stop
            + 30.0 * ee_pose_action_stop
            + 4.0 * base_reach_dense
            + 4.0 * gripper_open
            + 16.0 * gripper_open_action
            + 6.0 * actual_gripper_open
            - 8.0 * torch.abs(yaw_cmd)
            - 1.5 * action_ee_pose_l2
            - self.args.rew_action_penalty * action_penalty
            - self.args.rew_tilt_penalty * tilt
        )
        stage_reach_reward = (
            12.0 * reach_dense
            + 180.0 * reach_progress
            + 70.0 * ee_pos_action_tracking
            + 25.0 * ee_pos_action_alignment
            + 25.0 * reach_close_bonus
            + 8.0 * reach_goal_dense
            + 80.0 * reach_goal_progress
            + 10.0 * base_reach_hold
            + 8.0 * base_still_reward
            + 6.0 * policy_base_still_reward
            + 8.0 * gripper_open
            + 12.0 * gripper_open_action
            + 6.0 * actual_gripper_open
            - 12.0 * ee_goal_clamp_penalty
            - 4.0 * distance_over_grasp
            - 8.0 * distance_over_base_stop
            - self.args.rew_action_penalty * action_penalty
            - 0.01 * action_rate_l2
            - self.args.rew_tilt_penalty * tilt
        )
        stage_grasp_reward = (
            4.0 * approach_ee_handle
            + 0.5 * align_ee_handle
            + 24.0 * hold_close
            + 18.0 * approach_gripper_handle
            + 10.0 * finger_close_bonus
            + 4.0 * grasp_around
            + 24.0 * gripper_close_action
            + 16.0 * gripper_closed
            + 24.0 * actual_gripper_closed
            + 45.0 * grasp_handle_reward
            + 8.0 * base_still_reward
            + 14.0 * policy_base_still_reward
            - 12.0 * ee_goal_clamp_penalty
            - 15.0 * distance_over_hold
            - self.args.rew_action_penalty * action_penalty
            - 0.01 * action_rate_l2
            - self.args.rew_tilt_penalty * tilt
        )
        stage_open_reward = (
            3.0 * approach_ee_handle
            + 0.5 * align_ee_handle
            + 10.0 * hold_close
            + 6.0 * approach_gripper_handle
            + 6.0 * grasp_around
            + 8.0 * gripper_close_action
            + 8.0 * gripper_closed
            + 8.0 * actual_gripper_closed
            + 80.0 * handle_delta
            + 6.0 * handle_progress
            + 120.0 * open_delta
            + 12.0 * open_reward
            + 8.0 * multi_stage_open
            + 35.0 * open_bonus
            + 80.0 * open_pull_action_tracking
            + 35.0 * open_pull_action_alignment
            + 10.0 * base_still_reward
            + 8.0 * policy_base_still_reward
            - 12.0 * ee_goal_clamp_penalty
            - 12.0 * distance_over_hold
            - self.args.rew_action_penalty * action_penalty
            - 0.01 * action_rate_l2
            - self.args.rew_tilt_penalty * tilt
        )
        if self.args.reward_curriculum == "full":
            reward = (
                self.args.rew_reach * reach_dense
                + self.args.rew_handle * handle_progress
                + self.args.rew_open_progress * torch.clamp(signed_progress, min=-0.01, max=0.03)
                + self.args.rew_open * open_reward
                + self.args.rew_pass_align * pass_align
                + self.args.rew_open_bonus * open_bonus
                + self.args.rew_pass_bonus * pass_bonus
                - self.args.rew_action_penalty * action_penalty
                - self.args.rew_tilt_penalty * tilt
            )
            self.success_buf[:] = pass_bonus > 0.5
        elif self.args.reward_curriculum == "reach":
            self.phase_hold_buf[:] = self.reach_hold_buf
            ee_reach_reward = (
                10.0 * reach_dense
                + 150.0 * reach_progress
                + 50.0 * ee_pos_action_tracking
                + 20.0 * ee_pos_action_alignment
                + 20.0 * reach_close_bonus
                + 50.0 * reach_success
                + 8.0 * gripper_open
                + 12.0 * gripper_open_action
                + 6.0 * actual_gripper_open
                + 3.0 * reach_goal_dense
                + 50.0 * reach_goal_progress
                + 3.0 * base_reach_dense
                + 40.0 * base_reach_progress
                + 12.0 * base_reach_hold
                + 5.0 * base_vx_stop
                + 2.0 * base_yaw_stop
                - 5.0 * ee_goal_clamp_penalty
                - 10.0 * distance_over_base_stop
                - 0.02 * action_pos_l2
                - self.args.rew_action_penalty * action_penalty
                - self.args.rew_tilt_penalty * tilt
            )
            reward = base_approach_active_f * base_approach_reward + base_reach_active_f * ee_reach_reward
            self.phase_success_buf[:] = reach_success > 0.5
            self.success_buf[:] = self.phase_success_buf
        elif self.args.reward_curriculum == "grasp":
            grasp_now = (
                (ee_to_handle < self.args.grasp_success_dist)
                & (gripper_closed > 0.65)
                & (actual_gripper_closed > 0.35)
            )
            grasp_hold = torch.where(grasp_now, self.grasp_hold_buf + 1, torch.zeros_like(self.grasp_hold_buf))
            grasp_success = (grasp_hold >= self.args.grasp_hold_steps).to(torch.float32)
            self.grasp_hold_buf[:] = grasp_hold
            self.grasp_success_buf[:] = grasp_success > 0.5
            self.phase_hold_buf[:] = self.grasp_hold_buf
            grasp_reward = reach_stage * stage_reach_reward + grasp_ready * (stage_grasp_reward + 25.0 * grasp_success)
            reward = base_approach_active_f * base_approach_reward + base_reach_active_f * grasp_reward
            self.phase_success_buf[:] = self.grasp_success_buf
            self.success_buf[:] = self.phase_success_buf
        elif self.args.reward_curriculum == "handle":
            handle_reward = reach_stage * stage_reach_reward + grasp_stage * stage_grasp_reward + open_stage * (
                stage_open_reward + 80.0 * handle_delta + 10.0 * handle_progress
            )
            reward = base_approach_active_f * base_approach_reward + base_reach_active_f * handle_reward
            self.success_buf[:] = handle_progress > self.low_env.handle_unlock_threshold
            self.phase_success_buf[:] = self.success_buf
        elif self.args.reward_curriculum == "open":
            open_bonus = (torch.abs(signed_angle) >= math.radians(self.args.open_success_angle_deg)).to(torch.float32)
            open_reward_terms = reach_stage * stage_reach_reward + grasp_stage * stage_grasp_reward + open_stage * stage_open_reward
            reward = base_approach_active_f * base_approach_reward + base_reach_active_f * open_reward_terms
            self.success_buf[:] = open_bonus > 0.5
            self.phase_success_buf[:] = self.success_buf
        else:
            reward = (
                5.0 * open_reward
                + 50.0 * pass_progress
                + 5.0 * pass_align
                + 50.0 * pass_bonus
                - self.args.rew_action_penalty * action_penalty
                - self.args.rew_tilt_penalty * tilt
            )
            self.success_buf[:] = pass_bonus > 0.5
            self.phase_success_buf[:] = self.success_buf

        self.prev_ee_to_handle[:] = ee_to_handle
        self.prev_ee_goal_to_handle[:] = ee_goal_to_handle
        self.prev_arm_base_to_handle[:] = arm_base_to_handle
        self.prev_signed_angle[:] = signed_angle
        self.prev_handle_ratio[:] = handle_progress
        self.prev_open_ratio[:] = open_reward
        self.prev_pass_distance[:] = pass_distance
        terms = {
            "reach": reach_dense.detach(),
            "reach_dense": reach_dense.detach(),
            "reach_progress": reach_progress.detach(),
            "reach_goal_dense": reach_goal_dense.detach(),
            "reach_goal_progress": reach_goal_progress.detach(),
            "base_reach_dense": base_reach_dense.detach(),
            "base_reach_progress": base_reach_progress.detach(),
            "base_approach_active": base_approach_active_f.detach(),
            "base_stop_latched": self.base_stop_latch.detach().to(torch.float32),
            "base_heading_error_rad": base_heading_error.detach(),
            "base_heading_align": base_heading_align.detach(),
            "base_lateral_error_m": base_lateral_error.detach(),
            "base_lateral_align": base_lateral_align.detach(),
            "base_vx_target": target_vx.detach(),
            "base_stop_hold_vx": stop_hold_vx.detach(),
            "base_vx_tracking": base_vx_tracking.detach(),
            "policy_base_vx_cmd": policy_vx_cmd.detach(),
            "policy_base_vx_tracking": policy_base_vx_tracking.detach(),
            "policy_base_vx_signed": policy_base_vx_signed.detach(),
            "policy_base_yaw_cmd": policy_yaw_cmd.detach(),
            "policy_base_yaw_stop": policy_base_yaw_stop.detach(),
            "base_vx_stop": base_vx_stop.detach(),
            "base_yaw_stop": base_yaw_stop.detach(),
            "base_reach_hold": base_reach_hold.detach(),
            "distance_over_base_stop": distance_over_base_stop.detach(),
            "base_under_70cm": base_under_70cm.detach(),
            "base_under_60cm": base_under_60cm.detach(),
            "base_under_50cm": base_under_50cm.detach(),
            "base_under_stop_dist": base_under_stop_dist.detach(),
            "reach_close_bonus": reach_close_bonus.detach(),
            "reach_under_25cm": reach_under_25cm.detach(),
            "reach_under_20cm": reach_under_20cm.detach(),
            "reach_under_15cm": reach_under_15cm.detach(),
            "reach_under_12cm": reach_under_12cm.detach(),
            "reach_under_08cm": reach_under_08cm.detach(),
            "reach_success": reach_success.detach(),
            "grasp_success": self.grasp_success_buf.detach().to(torch.float32),
            "gripper_open": gripper_open.detach(),
            "gripper_closed": gripper_closed.detach(),
            "actual_gripper_open": actual_gripper_open.detach(),
            "actual_gripper_closed": actual_gripper_closed.detach(),
            "ee_handle_alignment": ee_handle_alignment.detach(),
            "approach_ee_handle": approach_ee_handle.detach(),
            "align_ee_handle": align_ee_handle.detach(),
            "approach_gripper_handle": approach_gripper_handle.detach(),
            "finger_mid_to_handle_m": finger_mid_to_handle.detach(),
            "finger_close_bonus": finger_close_bonus.detach(),
            "grasp_around_handle": grasp_around.detach(),
            "grasp_handle": grasp_handle_reward.detach(),
            "gripper_close_action": gripper_close_action.detach(),
            "gripper_open_action": gripper_open_action.detach(),
            "handle": handle_progress.detach(),
            "handle_delta": handle_delta.detach(),
            "open": open_reward.detach(),
            "open_delta": open_delta.detach(),
            "multi_stage_open": multi_stage_open.detach(),
            "open_pull_action_tracking": open_pull_action_tracking.detach(),
            "open_pull_action_alignment": open_pull_action_alignment.detach(),
            "pass_align": pass_align.detach(),
            "pass_progress": pass_progress.detach(),
            "open_bonus": open_bonus.detach(),
            "pass_bonus": pass_bonus.detach(),
            "phase_id": self.phase_id.detach().to(torch.float32),
            "phase_hold": self.phase_hold_buf.detach().to(torch.float32),
            "phase_success": self.phase_success_buf.detach().to(torch.float32),
            "reach_stage": reach_stage.detach(),
            "grasp_stage": grasp_stage.detach(),
            "open_stage": open_stage.detach(),
            "signed_angle_deg": torch.rad2deg(signed_angle).detach(),
            "signed_open_ratio": open_ratio.detach(),
            "ee_to_handle_m": ee_to_handle.detach(),
            "ee_goal_to_handle_m": ee_goal_to_handle.detach(),
            "ee_goal_clamped": self.ee_goal_clamped.detach().to(torch.float32),
            "ee_goal_clamp_penalty": ee_goal_clamp_penalty.detach(),
            "best_ee_to_handle_m": self.best_ee_to_handle.detach(),
            "best_ee_goal_to_handle_m": self.best_ee_goal_to_handle.detach(),
            "arm_base_to_handle_m": arm_base_to_handle.detach(),
            "best_arm_base_to_handle_m": self.best_arm_base_to_handle.detach(),
            "base_tilt": tilt.detach(),
            "action_l2": action_penalty.detach(),
            "action_rate_l2": action_rate_l2.detach(),
            "action_pos_abs_mean": torch.mean(torch.abs(actions[:, :3]), dim=-1).detach(),
            "action_pos_l2": action_pos_l2.detach(),
            "action_ee_pose_l2": action_ee_pose_l2.detach(),
            "ee_pose_action_stop": ee_pose_action_stop.detach(),
            "ee_pos_action_tracking": ee_pos_action_tracking.detach(),
            "ee_pos_action_alignment": ee_pos_action_alignment.detach(),
            "base_still_reward": base_still_reward.detach(),
            "policy_base_still_reward": policy_base_still_reward.detach(),
            "base_vx_cmd": self.low_env.commands[:, 0].detach(),
            "base_yaw_cmd": self.low_env.commands[:, 2].detach(),
        }
        self._last_reward_info = terms
        return reward, terms

    def _write_stagewise_log(self, reward_info):
        if self.stagewise_log_path is None or self.stagewise_log_interval <= 0:
            return
        step = int(self.stagewise_global_step)
        if step == 0 or step % self.stagewise_log_interval != 0:
            return

        def mean_value(key):
            value = reward_info.get(key)
            if value is None:
                return None
            if torch.is_tensor(value):
                return float(value.detach().float().mean().cpu().item())
            return float(np.asarray(value, dtype=np.float32).mean())

        record = {
            "global_step": step,
            "episode_step_max": int(self.progress_buf.max().detach().cpu().item()) if self.progress_buf.numel() else 0,
            "reward_curriculum": self.args.reward_curriculum,
            "phase_id_mean": mean_value("phase_id"),
            "phase_success_rate": mean_value("phase_success"),
            "reach_success_rate": mean_value("reach_success"),
            "grasp_success_rate": mean_value("grasp_success"),
            "reach_stage_rate": mean_value("reach_stage"),
            "grasp_stage_rate": mean_value("grasp_stage"),
            "open_stage_rate": mean_value("open_stage"),
            "ee_to_handle_m": mean_value("ee_to_handle_m"),
            "ee_goal_to_handle_m": mean_value("ee_goal_to_handle_m"),
            "ee_goal_clamped_rate": mean_value("ee_goal_clamped"),
            "ee_goal_clamp_penalty": mean_value("ee_goal_clamp_penalty"),
            "best_ee_to_handle_m": mean_value("best_ee_to_handle_m"),
            "best_ee_goal_to_handle_m": mean_value("best_ee_goal_to_handle_m"),
            "arm_base_to_handle_m": mean_value("arm_base_to_handle_m"),
            "best_arm_base_to_handle_m": mean_value("best_arm_base_to_handle_m"),
            "reach_dense": mean_value("reach_dense"),
            "reach_progress": mean_value("reach_progress"),
            "reach_goal_dense": mean_value("reach_goal_dense"),
            "reach_goal_progress": mean_value("reach_goal_progress"),
            "base_reach_dense": mean_value("base_reach_dense"),
            "base_reach_progress": mean_value("base_reach_progress"),
            "base_approach_active_rate": mean_value("base_approach_active"),
            "base_stop_latched_rate": mean_value("base_stop_latched"),
            "base_heading_error_rad": mean_value("base_heading_error_rad"),
            "base_heading_align": mean_value("base_heading_align"),
            "base_lateral_error_m": mean_value("base_lateral_error_m"),
            "base_lateral_align": mean_value("base_lateral_align"),
            "base_vx_target": mean_value("base_vx_target"),
            "base_stop_hold_vx": mean_value("base_stop_hold_vx"),
            "base_vx_tracking": mean_value("base_vx_tracking"),
            "policy_base_vx_cmd": mean_value("policy_base_vx_cmd"),
            "policy_base_vx_tracking": mean_value("policy_base_vx_tracking"),
            "policy_base_vx_signed": mean_value("policy_base_vx_signed"),
            "policy_base_yaw_cmd": mean_value("policy_base_yaw_cmd"),
            "policy_base_yaw_stop": mean_value("policy_base_yaw_stop"),
            "base_vx_stop": mean_value("base_vx_stop"),
            "base_yaw_stop": mean_value("base_yaw_stop"),
            "base_reach_hold": mean_value("base_reach_hold"),
            "distance_over_base_stop": mean_value("distance_over_base_stop"),
            "base_under_70cm_rate": mean_value("base_under_70cm"),
            "base_under_60cm_rate": mean_value("base_under_60cm"),
            "base_under_50cm_rate": mean_value("base_under_50cm"),
            "base_under_stop_dist_rate": mean_value("base_under_stop_dist"),
            "reach_close_bonus": mean_value("reach_close_bonus"),
            "reach_under_25cm_rate": mean_value("reach_under_25cm"),
            "reach_under_20cm_rate": mean_value("reach_under_20cm"),
            "reach_under_15cm_rate": mean_value("reach_under_15cm"),
            "reach_under_12cm_rate": mean_value("reach_under_12cm"),
            "reach_under_08cm_rate": mean_value("reach_under_08cm"),
            "gripper_open": mean_value("gripper_open"),
            "gripper_closed": mean_value("gripper_closed"),
            "actual_gripper_open": mean_value("actual_gripper_open"),
            "actual_gripper_closed": mean_value("actual_gripper_closed"),
            "approach_ee_handle": mean_value("approach_ee_handle"),
            "align_ee_handle": mean_value("align_ee_handle"),
            "approach_gripper_handle": mean_value("approach_gripper_handle"),
            "finger_mid_to_handle_m": mean_value("finger_mid_to_handle_m"),
            "finger_close_bonus_rate": mean_value("finger_close_bonus"),
            "grasp_around_handle_rate": mean_value("grasp_around_handle"),
            "grasp_handle": mean_value("grasp_handle"),
            "gripper_close_action": mean_value("gripper_close_action"),
            "gripper_open_action": mean_value("gripper_open_action"),
            "handle_open_ratio": mean_value("handle"),
            "handle_delta": mean_value("handle_delta"),
            "signed_open_ratio": mean_value("signed_open_ratio"),
            "signed_angle_deg": mean_value("signed_angle_deg"),
            "open_delta": mean_value("open_delta"),
            "multi_stage_open": mean_value("multi_stage_open"),
            "open_pull_action_tracking": mean_value("open_pull_action_tracking"),
            "open_pull_action_alignment": mean_value("open_pull_action_alignment"),
            "open_bonus_rate": mean_value("open_bonus"),
            "action_pos_abs_mean": mean_value("action_pos_abs_mean"),
            "action_pos_l2": mean_value("action_pos_l2"),
            "action_rate_l2": mean_value("action_rate_l2"),
            "action_ee_pose_l2": mean_value("action_ee_pose_l2"),
            "ee_pose_action_stop": mean_value("ee_pose_action_stop"),
            "ee_pos_action_tracking": mean_value("ee_pos_action_tracking"),
            "ee_pos_action_alignment": mean_value("ee_pos_action_alignment"),
            "base_still_reward": mean_value("base_still_reward"),
            "policy_base_still_reward": mean_value("policy_base_still_reward"),
            "base_vx_cmd": mean_value("base_vx_cmd"),
            "base_yaw_cmd": mean_value("base_yaw_cmd"),
            "action_l2": mean_value("action_l2"),
        }
        with self.stagewise_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _teacher_obs(self):
        env = self.low_env
        handle_goal = self._handle_goal()
        arm_base_pos = env.base_pos
        if hasattr(env, "arm_base_offset"):
            arm_base_pos = env.base_pos + quat_apply(env.base_yaw_quat, env.arm_base_offset)
        ee_pos_base = quat_rotate_inverse(env.root_states[:, 3:7], env.ee_pos - arm_base_pos)
        handle_base = quat_rotate_inverse(env.root_states[:, 3:7], handle_goal - arm_base_pos)
        door_base = quat_rotate_inverse(env.root_states[:, 3:7], env.door_root_state[:, :3] - env.root_states[:, :3])
        roll, pitch, yaw = euler_from_quat(env.root_states[:, 3:7])
        signed_angle = getattr(env, "signed_door_angle", env._door_dof_pos[:, 0]).unsqueeze(-1)
        signed_ratio = getattr(env, "signed_door_open_ratio", env.door_open_ratio).unsqueeze(-1)
        gripper_pos = env.dof_pos[:, -env.cfg.env.num_gripper_joints :].mean(dim=-1, keepdim=True)
        parts = [
            torch.stack([roll, pitch, yaw], dim=-1),
            env.base_lin_vel,
            env.base_ang_vel,
            env.commands,
            env.dof_pos,
            env.dof_vel,
            ee_pos_base,
            env.ee_orn,
            handle_base,
            door_base,
            env._door_dof_pos,
            env._door_dof_vel,
            signed_angle,
            signed_ratio,
            env.handle_open_ratio.unsqueeze(-1),
            env.door_motion_sign.unsqueeze(-1),
            self.last_high_action,
            gripper_pos,
        ]
        obs = torch.cat(parts, dim=-1)
        if obs.shape[1] < TEACHER_OBS_DIM:
            obs = torch.cat([obs, torch.zeros(obs.shape[0], TEACHER_OBS_DIM - obs.shape[1], device=obs.device)], dim=-1)
        return obs[:, :TEACHER_OBS_DIM]

    def _proprio_obs(self):
        env = self.low_env
        roll, pitch, _ = euler_from_quat(env.root_states[:, 3:7])
        arm_base_pos = env.base_pos
        if hasattr(env, "arm_base_offset"):
            arm_base_pos = env.base_pos + quat_apply(env.base_yaw_quat, env.arm_base_offset)
        ee_pos_base = quat_rotate_inverse(env.root_states[:, 3:7], env.ee_pos - arm_base_pos)
        gripper_pos = env.dof_pos[:, -env.cfg.env.num_gripper_joints :].mean(dim=-1, keepdim=True)
        foot_contacts = env._reindex_feet(env.foot_contacts_from_sensor).to(torch.float32)
        obs = torch.cat(
            [
                torch.stack([roll, pitch], dim=-1),
                env.base_ang_vel,
                env.dof_pos,
                env.dof_vel,
                env.last_actions,
                foot_contacts,
                ee_pos_base,
                env.ee_orn,
                gripper_pos,
                self.last_high_action,
            ],
            dim=-1,
        )
        if obs.shape[1] < PROPRIO_DIM:
            obs = torch.cat([obs, torch.zeros(obs.shape[0], PROPRIO_DIM - obs.shape[1], device=obs.device)], dim=-1)
        return obs[:, :PROPRIO_DIM]

    def _update_image_history(self):
        images = self.low_env.capture_wrist_camera_images()

        def _single(key):
            img = images.get(key)
            if img is None:
                return torch.zeros(self.num_envs, IMAGE_H, IMAGE_W, device=self.low_env.device)
            if img.ndim == 4:
                img = img[..., 0]
            return img.to(torch.float32) / 255.0

        channels = torch.stack(
            [
                _single("front_handle_mask"),
                _single("front_handle_masked_depth"),
                _single("wrist_handle_mask"),
                _single("wrist_handle_masked_depth"),
            ],
            dim=1,
        )
        self._image_history[:, :-4] = self._image_history[:, 4:].clone()
        self._image_history[:, -4:] = channels

    def _student_obs(self):
        return torch.cat([self._image_history.reshape(self.num_envs, -1), self._proprio_obs()], dim=-1)

    def _obs_dict(self):
        obs = {"obs": self._teacher_obs()}
        if self.mode == "student":
            obs["states"] = self._student_obs()
        return obs


class DoorAssetSKRLWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._obs_dict = None

    @property
    def num_states(self):
        return self._env.num_states

    @property
    def num_agents(self):
        return self._env.num_agents

    def reset(self):
        self._obs_dict = self._env.reset()
        if self.num_states:
            return self._obs_dict, {}
        return self._obs_dict["obs"], {}

    def step(self, actions):
        self._obs_dict, reward, terminated, info = self._env.step(actions)
        truncated = torch.zeros_like(terminated)
        if self.num_states:
            return self._obs_dict, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info
        return self._obs_dict["obs"], reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


def make_door_rl_env(args, mode="teacher", eval_mode=False):
    env = DoorAssetRLVecEnv(args, mode=mode, eval_mode=eval_mode)
    return DoorAssetSKRLWrapper(env)
