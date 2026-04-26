import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_ROOT = ROOT / "low-level"
if str(LOW_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOW_LEVEL_ROOT))

from isaacgym import gymapi  # noqa: E402
import isaacgym  # noqa: F401,E402
import torch  # noqa: E402
from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.utils import task_registry  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--speed_min", type=float, default=0.65)
    parser.add_argument("--speed_max", type=float, default=0.80)
    parser.add_argument("--yaw_min", type=float, default=0.0)
    parser.add_argument("--yaw_max", type=float, default=0.0)
    parser.add_argument("--resample_interval", type=int, default=90)
    parser.add_argument("--fixed_vx", type=float, default=None)
    parser.add_argument("--fixed_yaw", type=float, default=None)
    parser.add_argument("--log_dir", type=str, default=str(LOW_LEVEL_ROOT / "logs" / "b1z1-low" / "b1z1_locomanip"))
    parser.add_argument("--checkpoint", type=int, default=45000)
    return parser.parse_args()


def build_low_level_args(args):
    use_gpu = args.sim_device.startswith("cuda")
    return SimpleNamespace(
        task="b1z1",
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

    low_args = build_low_level_args(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name="b1z1")
    env_cfg.env.num_envs = args.num_envs
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 3
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

    env, _ = task_registry.make_env(name="b1z1", args=low_args, env_cfg=env_cfg)
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
    print("Forward command range:", env_cfg.commands.ranges.lin_vel_x)
    print("Yaw command range:", env_cfg.commands.ranges.ang_vel_yaw)

    env.reset()
    start_xy = env.root_states[:, :2].clone()
    commanded_vx = torch.zeros(args.num_envs, device=env.device)
    commanded_yaw = torch.zeros(args.num_envs, device=env.device)

    def sample_commands():
        if args.fixed_vx is not None:
            commanded_vx[:] = args.fixed_vx
        else:
            commanded_vx[:] = torch.empty(args.num_envs, device=env.device).uniform_(args.speed_min, args.speed_max)
        if args.fixed_yaw is not None:
            commanded_yaw[:] = args.fixed_yaw
        else:
            commanded_yaw[:] = torch.empty(args.num_envs, device=env.device).uniform_(args.yaw_min, args.yaw_max)

    sample_commands()
    for step in range(args.steps):
        if step % args.resample_interval == 0:
            sample_commands()
        env.commands[:, 0] = commanded_vx
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = commanded_yaw
        actions = policy(obs.detach(), hist_encoding=True)
        obs, _, _, _, dones, infos = env.step(actions.detach())
        if step % 30 == 0:
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
                    "traveled_xy": traveled[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "reset": dones[: min(args.num_envs, 4)].detach().cpu().tolist(),
                },
            )


if __name__ == "__main__":
    main()
