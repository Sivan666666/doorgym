import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

ROOT = Path(__file__).resolve().parents[2]
HIGH_LEVEL_ROOT = ROOT / "high-level"
if str(HIGH_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_ROOT))

from door_rl.door_asset_rl_env import build_default_runtime_args, make_door_rl_env  # noqa: E402

import torch  # noqa: E402
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG  # noqa: E402
from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: E402
from skrl.resources.schedulers.torch import KLAdaptiveRL  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

from learning.dagger import DAGGER_DEFAULT_CONFIG, DAgger  # noqa: E402
from learning.dagger_trainer import DAggerTrainer  # noqa: E402

from door_rl.models import StudentPolicy, TeacherPolicy, TeacherValue  # noqa: E402


ACTION_NAMES = [
    "delta_ee_x",
    "delta_ee_y",
    "delta_ee_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
    "vx",
    "yaw",
]


def _mean_item(value):
    if torch.is_tensor(value):
        return value.detach().float().mean().item()
    try:
        return torch.as_tensor(value, dtype=torch.float32).mean().item()
    except Exception:
        return None


def _track_door_infos(agent, infos):
    if not isinstance(infos, dict):
        return
    for group_name, prefix in (
        ("reward_terms", "Reward terms"),
        ("door_metrics", "Door metrics"),
        ("phase_metrics", "Phase"),
    ):
        values = infos.get(group_name, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            mean_value = _mean_item(value)
            if mean_value is not None:
                agent.track_data(f"{prefix} / {key}", mean_value)
    success_values = infos.get("success_rate", {})
    if isinstance(success_values, dict):
        for key, value in success_values.items():
            mean_value = _mean_item(value)
            if mean_value is not None:
                agent.track_data(key, mean_value)


def _track_action_stats(agent, actions, prefix="Action"):
    if not torch.is_tensor(actions):
        return
    actions = actions.detach().float()
    agent.track_data(f"{prefix} / l2", torch.norm(actions, dim=-1).mean().item())
    for idx, name in enumerate(ACTION_NAMES):
        if idx < actions.shape[-1]:
            agent.track_data(f"{prefix} / {name} mean", actions[:, idx].mean().item())
            agent.track_data(f"{prefix} / {name} abs mean", actions[:, idx].abs().mean().item())


class DoorPPO(PPO):
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        if self.write_interval > 0:
            _track_door_infos(self, infos)
            _track_action_stats(self, actions, prefix="Teacher action")


class DoorDAgger(DAgger):
    def record_transition(
        self,
        student_obs,
        teacher_obs,
        actions,
        teacher_actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        super().record_transition(
            student_obs=student_obs,
            teacher_obs=teacher_obs,
            actions=actions,
            teacher_actions=teacher_actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )
        if self.write_interval > 0:
            _track_door_infos(self, infos)
            _track_action_stats(self, actions, prefix="Student action")
            _track_action_stats(self, teacher_actions, prefix="Teacher action")
            self.track_data("Loss / Online teacher-student MSE", torch.mean((actions - teacher_actions) ** 2).item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("teacher", "student", "eval_teacher", "eval_student"), default="teacher")
    parser.add_argument("--mode", choices=("both", "pull", "push"), default="both")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--debug_cycle_timesteps", type=int, default=2000)
    parser.add_argument("--headless", dest="headless", action="store_true", default=None)
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--run_name", type=str, default="door_asset_rl")
    parser.add_argument("--log_root", type=str, default=str(HIGH_LEVEL_ROOT / "logs" / "door-rl"))
    parser.add_argument("--teacher_ckpt_path", type=str, default=None)
    parser.add_argument("--student_ckpt_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=45000)
    parser.add_argument("--low_log_dir", type=str, default=str(ROOT / "low-level" / "logs" / "b1z1-low" / "b1z1_locomanip"))
    parser.add_argument("--door_cfg", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "cfg" / "b1z1_opendoor.yaml"))
    parser.add_argument("--use_all_door_assets", action="store_true")
    parser.add_argument("--episode_length_s", type=float, default=20.0)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--enable_camera", dest="enable_camera", action="store_true", default=None)
    parser.add_argument("--disable_camera", dest="enable_camera", action="store_false")
    parser.add_argument("--front_camera_yaw_deg", type=float, default=0.0)
    parser.add_argument("--front_camera_pitch_deg", type=float, default=-60.0)
    parser.add_argument("--front_camera_roll_deg", type=float, default=0.0)
    parser.add_argument("--wrist_camera_down_tilt", type=float, default=0.20)
    parser.add_argument("--teacher_lr", type=float, default=5e-4)
    parser.add_argument("--student_lr", type=float, default=5e-5)
    parser.add_argument("--teacher_initial_log_std", type=float, default=None)
    parser.add_argument("--rollouts", type=int, default=24)
    parser.add_argument("--mini_batches", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--write_interval", type=int, default=24)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="door-rl")
    parser.add_argument("--reward_curriculum", choices=("reach", "grasp", "handle", "open", "pass", "full"), default="full")
    parser.add_argument("--reach_success_dist", type=float, default=0.08)
    parser.add_argument("--reach_close_dist", type=float, default=0.15)
    parser.add_argument("--reach_hold_steps", type=int, default=10)
    parser.add_argument("--base_approach_dist", type=float, default=0.50)
    parser.add_argument("--base_stop_dist", type=float, default=0.60)
    parser.add_argument("--base_stop_hold_gain", type=float, default=1.0)
    parser.add_argument("--base_stop_hold_max_vx", type=float, default=0.20)
    parser.add_argument("--base_approach_min_vx", type=float, default=0.30)
    parser.add_argument("--base_approach_max_vx", type=float, default=0.55)
    parser.add_argument("--base_approach_vx_gain", type=float, default=0.60)
    parser.add_argument("--base_heading_sigma", type=float, default=0.35)
    parser.add_argument("--stagewise_action_assist", action="store_true", help="Use scripted action overrides for debugging only")
    parser.add_argument("--init_yaw_noise", type=float, default=None)
    parser.add_argument("--grasp_entry_dist", type=float, default=0.16)
    parser.add_argument("--grasp_success_dist", type=float, default=0.12)
    parser.add_argument("--grasp_hold_steps", type=int, default=10)
    parser.add_argument("--open_success_angle_deg", type=float, default=20.0)
    parser.add_argument("--stagewise_log_path", type=str, default=None)
    parser.add_argument("--print_high_level_commands", dest="print_high_level_commands", action="store_true", default=True)
    parser.add_argument("--no_print_high_level_commands", dest="print_high_level_commands", action="store_false")
    parser.add_argument("--print_high_level_command_interval", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.headless is None:
        args.headless = args.stage not in ("eval_teacher", "eval_student")
    if args.reward_curriculum != "full" and args.timesteps == 100000:
        args.timesteps = args.debug_cycle_timesteps
    if args.teacher_initial_log_std is None and args.reward_curriculum != "full":
        args.teacher_initial_log_std = -2.0
    if args.stagewise_log_path is None:
        args.stagewise_log_path = str(Path(args.log_root) / args.run_name / "stagewise_debug.jsonl")
    return args


def _runtime_args(args, stage):
    enable_camera = args.stage in ("student", "eval_student") if args.enable_camera is None else args.enable_camera
    return build_default_runtime_args(
        num_envs=args.num_envs,
        mode=args.mode,
        headless=args.headless,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        checkpoint=args.checkpoint,
        log_dir=args.low_log_dir,
        door_cfg=args.door_cfg,
        use_all_door_assets=args.use_all_door_assets,
        episode_length_s=args.episode_length_s,
        enable_wrist_camera=enable_camera,
        enable_front_camera=enable_camera,
        camera_depth=enable_camera,
        camera_seg=enable_camera,
        show_seg=False,
        front_camera_yaw_deg=args.front_camera_yaw_deg,
        front_camera_pitch_deg=args.front_camera_pitch_deg,
        front_camera_roll_deg=args.front_camera_roll_deg,
        wrist_camera_down_tilt=args.wrist_camera_down_tilt,
        max_episode_steps=args.max_episode_steps,
        max_vx=0.55,
        max_yaw=0.9,
        ee_delta_scale=0.035,
        orn_delta_scale=0.08,
        ee_max_radius=1.25,
        max_orn_delta=1.5,
        init_yaw_noise=0.0 if args.init_yaw_noise is None and args.reward_curriculum == "reach" else (0.05 if args.init_yaw_noise is None else args.init_yaw_noise),
        init_xy_noise=0.03,
        external_pos_gain=1.5,
        external_orn_gain=1.0,
        gripper_open=-1.5707963267948966,
        gripper_closed=0.0,
        open_success_angle_deg=args.open_success_angle_deg,
        pass_left_offset=0.10,
        rew_reach=1.0,
        rew_handle=0.5,
        rew_open_progress=20.0,
        rew_open=1.0,
        rew_pass_align=0.2,
        rew_open_bonus=2.0,
        rew_pass_bonus=10.0,
        rew_action_penalty=0.005,
        rew_tilt_penalty=0.2,
        reward_curriculum=args.reward_curriculum,
        stagewise_log_path=args.stagewise_log_path,
        stagewise_log_interval=args.write_interval,
        reach_success_dist=args.reach_success_dist,
        reach_close_dist=args.reach_close_dist,
        reach_hold_steps=args.reach_hold_steps,
        base_approach_dist=args.base_approach_dist,
        base_stop_dist=args.base_stop_dist,
        base_stop_hold_gain=args.base_stop_hold_gain,
        base_stop_hold_max_vx=args.base_stop_hold_max_vx,
        base_approach_min_vx=args.base_approach_min_vx,
        base_approach_max_vx=args.base_approach_max_vx,
        base_approach_vx_gain=args.base_approach_vx_gain,
        base_heading_sigma=args.base_heading_sigma,
        stagewise_action_assist=args.stagewise_action_assist,
        grasp_entry_dist=args.grasp_entry_dist,
        grasp_success_dist=args.grasp_success_dist,
        grasp_hold_steps=args.grasp_hold_steps,
    )


def make_teacher_agent(env, args, deterministic=False):
    device = env.rl_device
    models = {
        "policy": TeacherPolicy(env.observation_space, env.action_space, device, deterministic=deterministic),
        "value": TeacherValue(env.observation_space, env.action_space, device),
    }
    if args.teacher_initial_log_std is not None and not deterministic:
        with torch.no_grad():
            models["policy"].log_std_parameter.fill_(float(args.teacher_initial_log_std))
    memory = RandomMemory(memory_size=args.rollouts, num_envs=env.num_envs, device=device)
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = args.rollouts
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = args.mini_batches
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = args.teacher_lr
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"]["write_interval"] = args.write_interval
    cfg["experiment"]["checkpoint_interval"] = args.save_interval
    cfg["experiment"]["directory"] = args.log_root
    cfg["experiment"]["experiment_name"] = f"{args.run_name}/teacher"
    cfg["experiment"]["wandb"] = args.wandb
    if args.wandb:
        cfg["experiment"]["wandb_kwargs"] = {
            "project": args.wandb_project,
            "name": f"{args.run_name}_teacher",
            "tensorboard": False,
            "config": vars(args),
        }
    return DoorPPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )


def make_student_agent(env, args):
    device = env.rl_device
    models = {"policy": StudentPolicy(env.state_space, env.action_space, device)}
    memory = RandomMemory(memory_size=args.rollouts, num_envs=env.num_envs, device=device)
    cfg = DAGGER_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = args.rollouts
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = max(1, args.mini_batches // 2)
    cfg["learning_rate"] = args.student_lr
    cfg["grad_norm_clip"] = 1.0
    cfg["experiment"]["write_interval"] = args.write_interval
    cfg["experiment"]["checkpoint_interval"] = args.save_interval
    cfg["experiment"]["directory"] = args.log_root
    cfg["experiment"]["experiment_name"] = f"{args.run_name}/student"
    cfg["experiment"]["wandb"] = args.wandb
    cfg["fixed_base"] = False
    cfg["reach_only"] = False
    cfg["pred_success"] = False
    if args.wandb:
        cfg["experiment"]["wandb_kwargs"] = {
            "project": args.wandb_project,
            "name": f"{args.run_name}_student",
            "tensorboard": False,
            "config": vars(args),
        }
    return DoorDAgger(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        state_space=env.state_space,
        device=device,
    )


def _save_wandb_artifacts(args, stage):
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        return
    for rel_path in (
        "door_rl/train_door_rl.py",
        "door_rl/door_asset_rl_env.py",
        "door_rl/models.py",
        "play_b1z1_walk_with_door_asset_camera.py",
        "play_b1z1_push_with_door_asset_camera.py",
    ):
        path = HIGH_LEVEL_ROOT / rel_path
        if path.exists():
            wandb.save(str(path), policy="now")
    door_cfg = Path(args.door_cfg)
    if door_cfg.exists():
        wandb.save(str(door_cfg), policy="now")
    wandb.config.update({"door_rl_stage": stage}, allow_val_change=True)


def train_teacher(args):
    env = make_door_rl_env(_runtime_args(args, "teacher"), mode="teacher", eval_mode=False)
    agent = make_teacher_agent(env, args)
    if args.dry_run:
        obs, _ = env.reset()
        print("teacher obs:", obs.shape, "action_space:", env.action_space)
        print("reward_curriculum:", args.reward_curriculum, "timesteps:", args.timesteps)
        return
    if args.teacher_ckpt_path:
        agent.load(args.teacher_ckpt_path)
    _append_stagewise_run_record(args, "teacher_start")
    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": args.headless}, env=env, agents=agent)
    _save_wandb_artifacts(args, "teacher")
    trainer.train()
    _append_stagewise_run_record(args, "teacher_end")


def _append_stagewise_run_record(args, event):
    path = Path(args.stagewise_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "stage": args.stage,
        "reward_curriculum": args.reward_curriculum,
        "mode": args.mode,
        "num_envs": args.num_envs,
        "timesteps": args.timesteps,
        "run_name": args.run_name,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def train_student(args):
    if not args.teacher_ckpt_path and not args.dry_run:
        raise ValueError("--teacher_ckpt_path is required for --stage student")
    if args.headless:
        print(
            "Warning: student camera tensors can fail in low-level headless mode because BaseTask sets graphics_device_id=-1. "
            "Use --viewer for real student visual training.",
            flush=True,
        )
    env = make_door_rl_env(_runtime_args(args, "student"), mode="student", eval_mode=False)
    if args.dry_run:
        obs, _ = env.reset()
        print("teacher obs:", obs["obs"].shape, "student obs:", obs["states"].shape, "action_space:", env.action_space)
        return
    teacher = make_teacher_agent(env, args, deterministic=True)
    teacher.load(args.teacher_ckpt_path)
    student = make_student_agent(env, args)
    if args.student_ckpt_path:
        student.load(args.student_ckpt_path)
    trainer = DAggerTrainer(
        cfg={"timesteps": args.timesteps, "headless": args.headless, "pretrain_timesteps": 0},
        env=env,
        agents=student,
        teacher_agents=teacher,
    )
    _save_wandb_artifacts(args, "student")
    trainer.train()


def _print_high_level_commands(step, actions, runtime_args, actual_commands=None, reward_info=None):
    if actions is None:
        return
    actions_cpu = actions.detach().float().cpu()
    actual_cpu = actual_commands.detach().float().cpu() if actual_commands is not None else None
    gripper_span = runtime_args.gripper_closed - runtime_args.gripper_open
    for env_id, action in enumerate(actions_cpu):
        raw = action.tolist()
        gripper_alpha = (float(action[6]) + 1.0) * 0.5
        scaled_policy_command = {
            "ee_delta_m": (action[:3] * runtime_args.ee_delta_scale).tolist(),
            "orn_delta_rad": (action[3:6] * runtime_args.orn_delta_scale).tolist(),
            "gripper_target": runtime_args.gripper_open + gripper_alpha * gripper_span,
            "base_vx_cmd": float(action[7]) * runtime_args.max_vx,
            "base_yaw_cmd": float(action[8]) * runtime_args.max_yaw,
        }
        actual_command = None
        if actual_cpu is not None:
            actual_command = {
                "base_vx_cmd": float(actual_cpu[env_id, 0]),
                "base_y_cmd": float(actual_cpu[env_id, 1]),
                "base_yaw_cmd": float(actual_cpu[env_id, 2]),
            }
        metrics = {}
        if reward_info:
            for key in ("base_stop_latched", "base_approach_active", "arm_base_to_handle_m", "ee_to_handle_m"):
                value = reward_info.get(key)
                if value is not None:
                    metrics[key] = float(value.detach().float().cpu()[env_id])
        record = {
            "type": "high_level_command",
            "step": step,
            "env_id": env_id,
            "raw_action": {
                "ee_dx": raw[0],
                "ee_dy": raw[1],
                "ee_dz": raw[2],
                "orn_dr": raw[3],
                "orn_dp": raw[4],
                "orn_dy": raw[5],
                "gripper": raw[6],
                "base_vx": raw[7],
                "base_yaw": raw[8],
            },
            "scaled_policy_command": scaled_policy_command,
            "actual_command": actual_command,
            "metrics": metrics,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)


def eval_policy(args, student=False):
    runtime_args = _runtime_args(args, "eval_student" if student else "eval_teacher")
    env = make_door_rl_env(runtime_args, mode="student" if student else "teacher", eval_mode=True)
    if student:
        if not args.student_ckpt_path:
            raise ValueError("--student_ckpt_path is required for eval_student")
        agent = make_student_agent(env, args)
        agent.load(args.student_ckpt_path)
    else:
        ckpt = args.teacher_ckpt_path or args.student_ckpt_path
        if not ckpt:
            raise ValueError("--teacher_ckpt_path is required for eval_teacher")
        agent = make_teacher_agent(env, args, deterministic=True)
        agent.load(ckpt)
    agent.set_running_mode("eval")
    states, _ = env.reset()
    for step in range(args.timesteps):
        with torch.no_grad():
            policy_obs = states["states"] if student else states
            actions = agent.act(policy_obs, timestep=step, timesteps=args.timesteps)[0]
            interval = max(1, int(args.print_high_level_command_interval))
            states, rewards, terminated, truncated, infos = env.step(actions)
            if args.print_high_level_commands and step % interval == 0:
                actual_commands = env._env.low_env.commands if hasattr(env, "_env") else None
                reward_info = getattr(env._env, "_last_reward_info", None) if hasattr(env, "_env") else None
                _print_high_level_commands(step, actions, runtime_args, actual_commands, reward_info)
            if not args.headless:
                env.render()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.stage == "teacher":
        train_teacher(args)
    elif args.stage == "student":
        train_student(args)
    elif args.stage == "eval_teacher":
        eval_policy(args, student=False)
    elif args.stage == "eval_student":
        eval_policy(args, student=True)


if __name__ == "__main__":
    main()
