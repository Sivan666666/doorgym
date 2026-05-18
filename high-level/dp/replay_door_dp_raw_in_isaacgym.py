import argparse
import importlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent
REPO_ROOT = HIGH_LEVEL_ROOT.parent
LOW_LEVEL_ROOT = REPO_ROOT / "low-level"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay raw Door DP .npz episodes in Isaac Gym. "
            "Use --replay_mode state to inspect recorded joint/state data, or "
            "--replay_mode action to re-run the recorded high-level actions through the low-level policy."
        )
    )
    parser.add_argument("--raw_episode", type=str, default=None, help="Path to one episode_*.npz file.")
    parser.add_argument(
        "--raw_root",
        type=str,
        default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp"),
        help="Directory containing episode_*.npz files, used when --raw_episode is omitted.",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--mode", choices=["auto", "pull", "push"], default="auto")
    parser.add_argument("--replay_mode", choices=["state", "action"], default="state")
    parser.add_argument("--control_env_id", type=int, default=0)
    parser.add_argument("--broadcast_all_envs", action="store_true")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--real_time", dest="real_time", action="store_true", default=True)
    parser.add_argument("--no_real_time", dest="real_time", action="store_false")
    parser.add_argument("--replay_fps", type=float, default=None)

    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episode_length_s", type=float, default=10000.0)
    parser.add_argument("--max_abs_vx", type=float, default=1.0)
    parser.add_argument("--max_abs_yaw", type=float, default=1.0)

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
    parser.add_argument(
        "--door_asset_name",
        type=str,
        default=None,
        help="Override recorded door_asset_name; useful for replaying old raw episodes without door metadata.",
    )
    parser.add_argument(
        "--door_asset_index",
        type=int,
        default=None,
        help="Override recorded door_asset_index after YAML/default filtering; useful for old raw episodes.",
    )
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

    parser.add_argument("--gripper_open", type=float, default=-1.5707963267948966)
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
    parser.add_argument("--external_pos_gain", type=float, default=1.5)
    parser.add_argument("--external_orn_gain", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--dp_log_path", type=str, default=None)
    parser.add_argument("--stop_on_done", dest="stop_on_done", action="store_true", default=True)
    parser.add_argument("--no_stop_on_done", dest="stop_on_done", action="store_false")
    parser.add_argument("--no_print", dest="print_logs", action="store_false", default=True)
    return parser.parse_args()


def _resolve_existing_path(path):
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    candidates = [REPO_ROOT / raw, HIGH_LEVEL_ROOT / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_episode(args):
    if args.raw_episode:
        episode = _resolve_existing_path(args.raw_episode)
        if not episode.exists():
            raise FileNotFoundError(f"raw episode not found: {episode}")
        return episode
    raw_root = _resolve_existing_path(args.raw_root)
    episodes = sorted(raw_root.glob("episode_*.npz"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.npz files under {raw_root}")
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise IndexError(f"--episode_index must be in [0, {len(episodes) - 1}]")
    return episodes[args.episode_index]


def scalar_to_str(value):
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
    elif arr.size > 0:
        value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def scalar_to_int(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return int(arr.item())
    return int(arr.reshape(-1)[0])


def door_asset_selection_from_episode(data, args):
    name = args.door_asset_name
    index = args.door_asset_index
    path = None
    if name is None and "door_asset_name" in data.files:
        name = scalar_to_str(data["door_asset_name"])
    if index is None and "door_asset_index" in data.files:
        index = scalar_to_int(data["door_asset_index"])
    if "door_asset_path" in data.files:
        path = scalar_to_str(data["door_asset_path"])
    if name is None and index is None and path is None:
        return None
    return {"name": name, "index": index, "path": path}


def infer_mode(data, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    task = scalar_to_str(data["task"]) if "task" in data.files else ""
    task_l = task.lower()
    if "push" in task_l:
        return "push"
    if "pull" in task_l or "walk" in task_l:
        return "pull"
    raise ValueError("cannot infer --mode from raw episode task; pass --mode pull or --mode push")


def load_play_module(mode):
    for path in (str(HIGH_LEVEL_ROOT), str(LOW_LEVEL_ROOT), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    module_name = "play_b1z1_push_with_door_asset_camera" if mode == "push" else "play_b1z1_walk_with_door_asset_camera"
    return importlib.import_module(module_name)


def select_single_door_asset(runtime, selection):
    if not selection:
        return None
    selected_idx = None
    if selection.get("name"):
        for idx, spec in enumerate(runtime["door_asset_specs"]):
            if spec.get("name") == selection["name"]:
                selected_idx = idx
                break
        if selected_idx is None:
            raise ValueError(f"door_asset_name={selection['name']!r} was not found in --door_cfg")
    elif selection.get("path"):
        for idx, spec in enumerate(runtime["door_asset_specs"]):
            if spec.get("path") == selection["path"]:
                selected_idx = idx
                break
        if selected_idx is None:
            raise ValueError(f"door_asset_path={selection['path']!r} was not found in --door_cfg")
    elif selection.get("index") is not None:
        selected_idx = int(selection["index"])
        if selected_idx < 0 or selected_idx >= len(runtime["door_asset_specs"]):
            raise ValueError(
                f"door_asset_index={selected_idx} is outside available range "
                f"[0, {len(runtime['door_asset_specs']) - 1}]"
            )
    if selected_idx is None:
        return None
    for key in ("door_asset_specs", "door_asset_names", "door_bounding_data", "handle_bounding_data"):
        runtime[key] = [runtime[key][selected_idx]]
    spec = runtime["door_asset_specs"][0]
    return {"index": selected_idx, "name": spec.get("name"), "path": spec.get("path")}


def configure_door_runtime(base, args, mode, door_asset_selection=None):
    cfg_path = _resolve_existing_path(args.door_cfg)
    runtime = base._load_door_runtime(str(cfg_path))
    base.DOOR_RUNTIME.clear()
    base.DOOR_RUNTIME.update(runtime)
    selected_door_asset = None
    if door_asset_selection and (door_asset_selection.get("name") or door_asset_selection.get("path")):
        selected_door_asset = select_single_door_asset(base.DOOR_RUNTIME, door_asset_selection)
    elif not args.use_all_door_assets:
        base._filter_door_runtime_by_names(base.DOOR_RUNTIME, base.DEFAULT_DOOR_ASSET_NAMES)
    if selected_door_asset is None:
        selected_door_asset = select_single_door_asset(base.DOOR_RUNTIME, door_asset_selection)

    total_door_assets = len(base.DOOR_RUNTIME["door_asset_specs"])
    door_asset_count = min(max(1, args.num_envs), total_door_assets)
    for key in ("door_asset_specs", "door_asset_names", "door_bounding_data", "handle_bounding_data"):
        base.DOOR_RUNTIME[key] = base.DOOR_RUNTIME[key][:door_asset_count]
    base.DOOR_RUNTIME["total_door_asset_count"] = total_door_assets
    base.DOOR_RUNTIME["loaded_door_asset_count"] = door_asset_count
    base.DOOR_RUNTIME["selected_replay_door_asset"] = selected_door_asset
    base.DOOR_RUNTIME["layout_spacing"] = args.layout_spacing
    base.DOOR_RUNTIME["robot_x"] = args.robot_x
    base.DOOR_RUNTIME["robot_y"] = args.robot_y
    base.DOOR_RUNTIME["robot_z"] = args.robot_z
    base.DOOR_RUNTIME["robot_yaw"] = args.robot_yaw
    base.DOOR_RUNTIME["door_x"] = args.door_x
    base.DOOR_RUNTIME["door_y"] = args.door_y
    base.DOOR_RUNTIME["door_actor_scale"] = args.door_actor_scale
    base.DOOR_RUNTIME["robot_y_by_spec"] = base._compute_robot_y_by_spec(
        args.robot_y,
        args.door_y,
        base.DOOR_RUNTIME["door_bounding_data"],
        base.DOOR_RUNTIME["handle_bounding_data"],
        args.door_actor_scale,
    )
    base.DOOR_RUNTIME["door_z_offset"] = args.door_z_offset
    base.DOOR_RUNTIME["box_x"] = args.box_x
    base.DOOR_RUNTIME["box_y"] = args.box_y
    base.DOOR_RUNTIME["gripper_stiffness"] = args.gripper_stiffness
    base.DOOR_RUNTIME["gripper_damping"] = args.gripper_damping
    base.DOOR_RUNTIME["gripper_joint_friction"] = args.gripper_joint_friction
    base.DOOR_RUNTIME["handle_spring_stiffness"] = args.handle_spring_stiffness
    base.DOOR_RUNTIME["handle_spring_damping"] = args.handle_spring_damping
    base.DOOR_RUNTIME["handle_unlock_ratio"] = args.handle_unlock_ratio
    base.DOOR_RUNTIME["door_open_resistance"] = args.door_open_resistance
    base.DOOR_RUNTIME["door_open_damping"] = args.door_open_damping
    base.DOOR_RUNTIME["door_auto_open_force"] = args.door_auto_open_force
    base.DOOR_RUNTIME["door_auto_open_sign"] = args.door_auto_open_sign
    base.DOOR_RUNTIME["door_auto_open_target_ratio"] = args.door_auto_open_target_ratio
    if mode == "push":
        base.DOOR_RUNTIME["door_motion_sign"] = -1.0
    base.DOOR_RUNTIME["door_joint_friction"][0] = args.door_joint_friction
    base.DOOR_RUNTIME["door_joint_damping"][0] = args.door_joint_damping
    base.DOOR_RUNTIME["door_joint_friction"][1] = args.handle_joint_friction
    base.DOOR_RUNTIME["door_joint_damping"][1] = args.handle_joint_damping
    base.DOOR_RUNTIME["robot_vhacd_resolution"] = args.robot_vhacd_resolution
    base.DOOR_RUNTIME["gripper_shape_contact_offset"] = args.gripper_shape_contact_offset
    base.DOOR_RUNTIME["gripper_shape_rest_offset"] = args.gripper_shape_rest_offset
    base.DOOR_RUNTIME["gripper_shape_friction"] = args.gripper_shape_friction
    base.DOOR_RUNTIME["door_vhacd_resolution"] = args.door_vhacd_resolution
    base.DOOR_RUNTIME["enable_wrist_camera"] = args.enable_wrist_camera
    base.DOOR_RUNTIME["enable_front_camera"] = args.enable_front_camera
    base.DOOR_RUNTIME["camera_rgb"] = args.camera_rgb
    base.DOOR_RUNTIME["camera_depth"] = args.camera_depth
    base.DOOR_RUNTIME["camera_seg"] = args.camera_seg
    base.DOOR_RUNTIME["show_seg"] = args.show_seg
    base.DOOR_RUNTIME["handle_seg_id"] = args.handle_seg_id
    base.DOOR_RUNTIME["camera_depth_clip_lower"] = args.camera_depth_clip_lower
    base.DOOR_RUNTIME["camera_depth_clip_far"] = args.camera_depth_clip_far
    base.DOOR_RUNTIME["camera_display_scale"] = args.camera_display_scale
    base.DOOR_RUNTIME["wrist_camera_down_tilt"] = args.wrist_camera_down_tilt
    base.DOOR_RUNTIME["front_camera_yaw_deg"] = args.front_camera_yaw_deg
    base.DOOR_RUNTIME["front_camera_pitch_deg"] = args.front_camera_pitch_deg
    base.DOOR_RUNTIME["front_camera_roll_deg"] = args.front_camera_roll_deg
    return cfg_path


def make_env(base, args, need_policy):
    low_args = base.build_low_level_args(args)
    task_name = "b1z1_door_raw_replay"
    low_args.task = task_name
    env_cfg, train_cfg = base.task_registry.get_cfgs(name="b1z1")
    base.task_registry.register(task_name, base.ManipLocoDoorAsset, env_cfg, train_cfg, "b1z1")

    env_cfg.sim.substeps = args.sim_substeps
    env_cfg.sim.physx.num_position_iterations = args.sim_position_iterations
    env_cfg.sim.physx.num_velocity_iterations = args.sim_velocity_iterations
    env_cfg.sim.physx.contact_offset = args.sim_contact_offset
    env_cfg.sim.physx.rest_offset = args.sim_rest_offset
    env_cfg.sim.physx.max_depenetration_velocity = args.sim_max_depenetration_velocity

    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = args.episode_length_s
    terrain_side = max(2, int(math.ceil(math.sqrt(args.num_envs))))
    env_cfg.terrain.num_rows = terrain_side
    env_cfg.terrain.num_cols = terrain_side
    env_cfg.terrain.height = [0.0, 0.0]
    env_cfg.commands.curriculum = False
    env_cfg.env.observe_gait_commands = True
    env_cfg.commands.ranges.lin_vel_x = [-args.max_abs_vx, args.max_abs_vx]
    env_cfg.commands.ranges.ang_vel_yaw = [-args.max_abs_yaw, args.max_abs_yaw]
    env_cfg.commands.lin_vel_x_clip = min(env_cfg.commands.lin_vel_x_clip, max(0.01, args.max_abs_vx))
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.noise.add_noise = False
    env_cfg.init_state.rand_yaw_range = 0.0
    env_cfg.init_state.origin_perturb_range = 0.0
    env_cfg.init_state.init_vel_perturb_range = 0.0

    env, _ = base.task_registry.make_env(name=task_name, args=low_args, env_cfg=env_cfg)
    policy = None
    if need_policy:
        ppo_runner, _, _, _ = base.task_registry.make_alg_runner(
            log_root=args.log_dir,
            env=env,
            name="b1z1",
            args=low_args,
            train_cfg=train_cfg,
            return_log_dir=True,
        )
        policy = ppo_runner.get_inference_policy(device=env.device, stochastic=False)
    return env, policy


def init_external_control(base, env, args):
    torch = base.torch
    env.reset()
    env.external_ee_goal_control = True
    env.external_pos_gain = args.external_pos_gain
    env.external_orn_gain = args.external_orn_gain
    env.commands[:] = 0.0
    env.curr_ee_goal_cart_world[:] = env.ee_pos
    env.ee_goal_orn_quat[:] = env.ee_orn
    env.ee_goal_orn_delta_rpy[:] = 0.0
    env.freeze_arm_default = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    env.freeze_arm_zero = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    env.external_gripper_target = torch.full(
        (args.num_envs, env.cfg.env.num_gripper_joints),
        args.gripper_open,
        device=env.device,
    )
    return env.get_observations()


def wrap_angle(base, x):
    return base.torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def make_delta_rpy_fn(base, env):
    def ee_goal_delta_rpy_from_quat(target_pos, target_quat, env_ids=None):
        torch = base.torch
        goal_roll, goal_pitch, goal_yaw = base.euler_from_quat(target_quat)
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
                wrap_angle(base, goal_roll - math.pi / 2.0),
                wrap_angle(base, goal_pitch - default_pitch),
                wrap_angle(base, goal_yaw - default_yaw),
            ),
            dim=-1,
        )

    return ee_goal_delta_rpy_from_quat


def feature_names_from_data(data):
    if "state_feature_names" not in data.files:
        return []
    return [scalar_to_str(item) for item in np.asarray(data["state_feature_names"]).tolist()]


def prefixed_feature_indices(names, prefix):
    result = []
    for idx, name in enumerate(names):
        if not name.startswith(prefix):
            continue
        try:
            order = int(name[len(prefix) :])
        except ValueError:
            order = len(result)
        result.append((order, idx))
    return [idx for _, idx in sorted(result)]


def named_feature_index(names, name):
    try:
        return names.index(name)
    except ValueError:
        return None


def assign_vector_from_episode(base, target, value):
    length = min(target.numel(), int(np.asarray(value).size))
    tensor = base.torch.as_tensor(np.asarray(value).reshape(-1)[:length], dtype=target.dtype, device=target.device)
    if target.ndim != 1:
        target = target.reshape(-1)
    target[:length].copy_(tensor)


def apply_snapshot_state(base, env, data, frame_idx, env_id):
    if "replay_root_state" in data.files:
        assign_vector_from_episode(base, env.root_states[env_id], data["replay_root_state"][frame_idx])
    if "replay_door_root_state" in data.files and hasattr(env, "door_root_state"):
        assign_vector_from_episode(base, env.door_root_state[env_id], data["replay_door_root_state"][frame_idx])
    if "replay_box_root_state" in data.files and hasattr(env, "box_root_state"):
        assign_vector_from_episode(base, env.box_root_state[env_id], data["replay_box_root_state"][frame_idx])
    if "replay_dof_pos" in data.files:
        assign_vector_from_episode(base, env.dof_pos[env_id], data["replay_dof_pos"][frame_idx])
    if "replay_dof_vel" in data.files:
        assign_vector_from_episode(base, env.dof_vel[env_id], data["replay_dof_vel"][frame_idx])
    if "replay_door_dof_pos" in data.files and hasattr(env, "_door_dof_pos"):
        assign_vector_from_episode(base, env._door_dof_pos[env_id], data["replay_door_dof_pos"][frame_idx])
    if "replay_door_dof_vel" in data.files and hasattr(env, "_door_dof_vel"):
        assign_vector_from_episode(base, env._door_dof_vel[env_id], data["replay_door_dof_vel"][frame_idx])
    if "replay_ee_pos" in data.files:
        assign_vector_from_episode(base, env.curr_ee_goal_cart_world[env_id], data["replay_ee_pos"][frame_idx])
    if "replay_ee_quat" in data.files:
        assign_vector_from_episode(base, env.ee_goal_orn_quat[env_id], data["replay_ee_quat"][frame_idx])


def apply_legacy_observation_state(base, env, data, names, frame_idx, env_id):
    if "state" not in data.files or not names:
        return False
    row = np.asarray(data["state"][frame_idx], dtype=np.float32)
    dof_pos_indices = prefixed_feature_indices(names, "dof_pos_")
    dof_vel_indices = prefixed_feature_indices(names, "dof_vel_")
    if dof_pos_indices:
        assign_vector_from_episode(base, env.dof_pos[env_id], row[dof_pos_indices])
    if dof_vel_indices:
        assign_vector_from_episode(base, env.dof_vel[env_id], row[dof_vel_indices])

    roll_idx = named_feature_index(names, "base_roll")
    pitch_idx = named_feature_index(names, "base_pitch")
    if roll_idx is not None or pitch_idx is not None:
        torch = base.torch
        cur_roll, cur_pitch, cur_yaw = base.euler_from_quat(env.root_states[env_id : env_id + 1, 3:7])
        roll = torch.tensor([float(row[roll_idx])], dtype=torch.float32, device=env.device) if roll_idx is not None else cur_roll
        pitch = torch.tensor([float(row[pitch_idx])], dtype=torch.float32, device=env.device) if pitch_idx is not None else cur_pitch
        env.root_states[env_id : env_id + 1, 3:7] = base.quat_from_euler_xyz(roll, pitch, cur_yaw)

    ang_indices = [named_feature_index(names, f"base_ang_vel_{axis}") for axis in ("x", "y", "z")]
    if all(idx is not None for idx in ang_indices):
        assign_vector_from_episode(base, env.root_states[env_id, 10:13], row[ang_indices])
    return bool(dof_pos_indices or dof_vel_indices)


def push_state_to_sim(base, env, advance_sim=False):
    if hasattr(env, "full_pos_targets"):
        env.full_pos_targets.zero_()
        env.full_pos_targets[:, : env.num_dofs] = env.dof_pos
        if hasattr(env, "_door_dof_pos"):
            env.full_pos_targets[:, env.num_dofs :] = env._door_dof_pos
        env.gym.set_dof_position_target_tensor(env.sim, base.gymtorch.unwrap_tensor(env.full_pos_targets))
    if hasattr(env, "full_torques"):
        env.full_torques.zero_()
        env.gym.set_dof_actuation_force_tensor(env.sim, base.gymtorch.unwrap_tensor(env.full_torques))
    env.gym.set_actor_root_state_tensor(env.sim, base.gymtorch.unwrap_tensor(env._root_states))
    env.gym.set_dof_state_tensor(env.sim, base.gymtorch.unwrap_tensor(env._full_dof_state_flat))
    if advance_sim:
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_jacobian_tensors(env.sim)


def apply_state_frame(base, env, data, names, frame_idx, env_ids):
    has_snapshot = any(key in data.files for key in ("replay_root_state", "replay_dof_pos", "replay_door_dof_pos"))
    for env_id in env_ids:
        if has_snapshot:
            apply_snapshot_state(base, env, data, frame_idx, env_id)
        else:
            apply_legacy_observation_state(base, env, data, names, frame_idx, env_id)
    push_state_to_sim(base, env, advance_sim=True)
    return has_snapshot


def draw_replay_markers(base, env, env_ids):
    if getattr(env, "viewer", None) is None:
        return
    target_geom = base.gymutil.WireframeSphereGeometry(0.035, 8, 8, None, color=(1, 0, 0))
    axes_geom = base.ThickAxesGeometry(scale=0.12, thickness=0.004)
    for env_id in env_ids[:16]:
        target = env.curr_ee_goal_cart_world[env_id].detach().cpu().tolist()
        target_quat = env.ee_goal_orn_quat[env_id].detach().cpu().tolist()
        ee_pos = env.ee_pos[env_id].detach().cpu().tolist()
        ee_quat = env.ee_orn[env_id].detach().cpu().tolist()
        target_pose = base.gymapi.Transform(
            base.gymapi.Vec3(target[0], target[1], target[2]),
            base.gymapi.Quat(target_quat[0], target_quat[1], target_quat[2], target_quat[3]),
        )
        ee_pose = base.gymapi.Transform(
            base.gymapi.Vec3(ee_pos[0], ee_pos[1], ee_pos[2]),
            base.gymapi.Quat(ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]),
        )
        base.gymutil.draw_lines(target_geom, env.gym, env.viewer, env.envs[env_id], target_pose)
        base.gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[env_id], target_pose)
        base.gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[env_id], ee_pose)


def _raw_frame_image_tensor(base, data, key, frame_idx, num_envs):
    if key not in data.files:
        return None
    image = np.asarray(data[key][frame_idx])
    if image.ndim == 3 and image.shape[-1] == 3:
        image = image[..., 0]
    image = image.astype(np.float32)
    if image.size > 0 and float(np.nanmax(image)) > 1.5:
        image /= 255.0
    tensor = base.torch.as_tensor(image, dtype=base.torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1 and num_envs > 1:
        tensor = tensor.repeat(num_envs, 1, 1)
    return tensor


def raw_camera_images_from_episode(base, data, frame_idx, num_envs):
    images = {}
    wrist_mask = _raw_frame_image_tensor(base, data, "wrist_handle_mask", frame_idx, num_envs)
    wrist_depth = _raw_frame_image_tensor(base, data, "wrist_masked_depth", frame_idx, num_envs)
    front_mask = _raw_frame_image_tensor(base, data, "front_handle_mask", frame_idx, num_envs)
    front_depth = _raw_frame_image_tensor(base, data, "front_masked_depth", frame_idx, num_envs)
    if wrist_mask is not None:
        images["wrist_handle_mask"] = wrist_mask
        images["handle_mask"] = wrist_mask
    if wrist_depth is not None:
        images["wrist_handle_masked_depth"] = wrist_depth
        images["handle_masked_depth"] = wrist_depth
    if front_mask is not None:
        images["front_handle_mask"] = front_mask
    if front_depth is not None:
        images["front_handle_masked_depth"] = front_depth
    return images or None


def render_state_frame(base, env, args, env_ids, camera_images=None):
    if getattr(env, "viewer", None) is not None and env.gym.query_viewer_has_closed(env.viewer):
        return False
    images = camera_images if camera_images is not None else env.capture_wrist_camera_images()
    env.show_wrist_seg(images, args.camera_env_id)
    if getattr(env, "viewer", None) is not None:
        env.gym.clear_lines(env.viewer)
    env.draw_wrist_camera_axes(args.camera_axis_scale, args.camera_axis_thickness)
    draw_replay_markers(base, env, env_ids)
    if getattr(env, "viewer", None) is not None:
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
        env.gym.sync_frame_time(env.sim)
    return True


def frame_indices(args, total_frames):
    if args.start_step < 0 or args.start_step >= total_frames:
        raise IndexError(f"--start_step must be in [0, {total_frames - 1}]")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    available = list(range(args.start_step, total_frames, args.stride))
    if args.steps is not None:
        available = available[: max(0, args.steps)]
    return available


def print_replay_log(data, frame_idx, replay_step, replay_mode, env, env_id, action=None, has_snapshot=False):
    pieces = [
        f"[DoorDPReplay] mode={replay_mode}",
        f"step={replay_step}",
        f"frame={frame_idx}",
        f"env={env_id}",
    ]
    if "subtask_index" in data.files:
        pieces.append(f"subtask={int(np.asarray(data['subtask_index'][frame_idx]).reshape(-1)[0])}")
    if action is not None:
        pieces.append(
            "action="
            f"vx:{float(action[0]):.3f} yaw:{float(action[1]):.3f} "
            f"ee:[{float(action[2]):.3f},{float(action[3]):.3f},{float(action[4]):.3f}] "
            f"grip:{float(action[9]):.3f}"
        )
    if hasattr(env, "_door_dof_pos"):
        door = env._door_dof_pos[env_id].detach().cpu().numpy()
        pieces.append(f"door_dof={np.round(door, 4).tolist()}")
    pieces.append(f"snapshot={has_snapshot}")
    print(" ".join(pieces), flush=True)


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    episode_path = resolve_episode(args)
    data = np.load(episode_path, allow_pickle=True)
    mode = infer_mode(data, args.mode)
    base = load_play_module(mode)
    from dp.door_dp_common import (
        DoorDPJsonlLogger,
        apply_door_dp_action,
        make_door_dp_log_record,
        print_door_dp_log_record,
    )

    door_asset_selection = door_asset_selection_from_episode(data, args)
    cfg_path = configure_door_runtime(base, args, mode, door_asset_selection=door_asset_selection)
    if args.control_env_id < 0 or args.control_env_id >= args.num_envs:
        raise ValueError(f"--control_env_id must be in [0, {args.num_envs - 1}]")
    env_ids = list(range(args.num_envs)) if args.broadcast_all_envs else [args.control_env_id]

    actions = data["action"].astype(np.float32) if "action" in data.files else None
    if args.replay_mode == "action" and (actions is None or actions.ndim != 2 or actions.shape[1] < 10):
        raise ValueError("action replay requires raw episode field action with shape [T, 10]")
    total_frames = int(actions.shape[0] if actions is not None else data["state"].shape[0])
    indices = frame_indices(args, total_frames)
    if not indices:
        raise ValueError("no frames selected for replay")

    need_policy = args.replay_mode == "action"
    env, policy = make_env(base, args, need_policy=need_policy)
    obs = init_external_control(base, env, args)
    delta_rpy_fn = make_delta_rpy_fn(base, env)
    names = feature_names_from_data(data)
    logger = DoorDPJsonlLogger(args.dp_log_path) if args.dp_log_path else None
    replay_fps = float(args.replay_fps or (int(np.asarray(data["fps"]).item()) if "fps" in data.files else 50))

    print(
        "Replay raw Door DP episode:",
        {
            "episode": str(episode_path),
            "task": scalar_to_str(data["task"]) if "task" in data.files else None,
            "mode": mode,
            "replay_mode": args.replay_mode,
            "frames": total_frames,
            "selected_frames": len(indices),
            "door_cfg": str(cfg_path),
            "door_asset_selection": base.DOOR_RUNTIME.get("selected_replay_door_asset"),
            "env_ids": env_ids,
        },
        flush=True,
    )
    if args.replay_mode == "state" and not any(
        key in data.files for key in ("replay_root_state", "replay_dof_pos", "replay_door_dof_pos")
    ):
        print(
            "Warning: this raw episode has legacy observation-only state. "
            "State replay will set robot dof_pos/dof_vel and base roll/pitch only; "
            "root xy/yaw and door state were not recorded.",
            flush=True,
        )
    if args.replay_mode == "action":
        print("Loaded low-level walking policy from:", os.path.join(args.log_dir, f"model_{args.checkpoint}.pt"), flush=True)

    try:
        replay_step = 0
        while True:
            for frame_idx in indices:
                loop_start = time.time()
                action = actions[frame_idx, :10] if actions is not None else None
                if action is not None and args.replay_mode == "action":
                    for env_id in env_ids:
                        apply_door_dp_action(env, action, env_id, delta_rpy_fn)

                has_snapshot = False
                if args.replay_mode == "state":
                    has_snapshot = apply_state_frame(base, env, data, names, frame_idx, env_ids)
                    raw_camera_images = raw_camera_images_from_episode(base, data, frame_idx, args.num_envs)
                    keep_running = render_state_frame(base, env, args, env_ids, camera_images=raw_camera_images)
                    if not keep_running:
                        return
                else:
                    low_actions = policy(obs.detach(), hist_encoding=True)
                    obs, _, _, _, dones, _ = env.step(low_actions.detach())
                    images = env.capture_wrist_camera_images()
                    env.show_wrist_seg(images, args.camera_env_id)
                    if getattr(env, "viewer", None) is not None:
                        env.gym.clear_lines(env.viewer)
                    env.draw_wrist_camera_axes(args.camera_axis_scale, args.camera_axis_thickness)
                    draw_replay_markers(base, env, env_ids)
                    if args.stop_on_done and bool(dones[args.control_env_id].item()):
                        print(f"Stopped: env {args.control_env_id} reset at replay step {replay_step}", flush=True)
                        return

                if logger is not None and action is not None:
                    record = make_door_dp_log_record(
                        env,
                        replay_step,
                        action,
                        args.control_env_id,
                        extra={
                            "replay_mode": args.replay_mode,
                            "raw_episode": str(episode_path),
                            "raw_frame": int(frame_idx),
                            "has_replay_snapshot": bool(has_snapshot),
                        },
                    )
                    logger.write(record)
                    if args.print_logs and replay_step % max(1, args.log_interval) == 0:
                        print_door_dp_log_record(record)
                elif args.print_logs and replay_step % max(1, args.log_interval) == 0:
                    print_replay_log(data, frame_idx, replay_step, args.replay_mode, env, args.control_env_id, action, has_snapshot)

                replay_step += 1
                if args.replay_mode == "state" and args.real_time:
                    elapsed = time.time() - loop_start
                    time.sleep(max(0.0, (1.0 / max(1e-6, replay_fps)) - elapsed))
            if not args.loop:
                break
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
