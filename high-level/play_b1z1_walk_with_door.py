import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

from isaacgym import gymapi  # noqa: F401
import isaacgym  # noqa: F401
from envs import B1Z1OpenDoor
import torch
from utils.config import load_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--speed_min", type=float, default=0.65)
    parser.add_argument("--speed_max", type=float, default=0.80)
    parser.add_argument("--yaw_min", type=float, default=0.0)
    parser.add_argument("--yaw_max", type=float, default=0.0)
    parser.add_argument("--resample_interval", type=int, default=90)
    parser.add_argument("--fixed_vx", type=float, default=None)
    parser.add_argument("--fixed_yaw", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)

    cfg = load_cfg("data/cfg/b1z1_opendoor.yaml")
    cfg["env"]["numEnvs"] = args.num_envs
    cfg["env"]["maxEpisodeLength"] = max(cfg["env"]["maxEpisodeLength"], args.steps + 20)
    cfg["env"]["smallValueSetZero"] = False
    if args.sim_device == "cpu":
        cfg["sim"]["use_gpu_pipeline"] = False
        cfg["sim"]["physx"]["use_gpu"] = False

    env = B1Z1OpenDoor(
        cfg=cfg,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=args.headless,
        use_roboinfo=True,
        observe_gait_commands=True,
        no_feature=True,
        robot_start_pose=(1.75, 0.0, 0.55),
        eval=True,
    )
    env.reset()

    env.curr_ee_goal_cart[:] = env.init_ee_goal_cart[:]
    env.curr_ee_goal_orn_rpy[:] = torch.tensor([torch.pi / 2, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)

    start_xy = env._robot_root_states[:, :2].clone()
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    actions[:, 6] = 1.0  # keep gripper open
    commanded_vx = torch.zeros(env.num_envs, device=env.device)
    commanded_yaw = torch.zeros(env.num_envs, device=env.device)

    def sample_commands():
        if args.fixed_vx is not None:
            commanded_vx[:] = args.fixed_vx
        else:
            commanded_vx[:] = torch.empty(env.num_envs, device=env.device).uniform_(args.speed_min, args.speed_max)
        if args.fixed_yaw is not None:
            commanded_yaw[:] = args.fixed_yaw
        else:
            commanded_yaw[:] = torch.empty(env.num_envs, device=env.device).uniform_(args.yaw_min, args.yaw_max)

    sample_commands()

    print("Loaded door assets:", env.door_asset_names)
    print("Loaded low-level path:", cfg["env"]["low_policy_path"])
    print("Forward command range:", [args.speed_min, args.speed_max] if args.fixed_vx is None else [args.fixed_vx, args.fixed_vx])
    print("Yaw command range:", [args.yaw_min, args.yaw_max] if args.fixed_yaw is None else [args.fixed_yaw, args.fixed_yaw])

    for step in range(args.steps):
        if step % args.resample_interval == 0:
            sample_commands()
        actions[:, 7] = commanded_vx
        actions[:, 8] = commanded_yaw
        _, _, done, _ = env.step(actions)

        if step % 30 == 0:
            traveled = torch.norm(env._robot_root_states[:, :2] - start_xy, dim=-1)
            print(
                f"[step {step:04d}]",
                {
                    "command_x": env.commands[: min(args.num_envs, 4), 0].detach().cpu().tolist(),
                    "command_yaw": env.commands[: min(args.num_envs, 4), 2].detach().cpu().tolist(),
                    "base_lin_vel_x": env._robot_root_states[: min(args.num_envs, 4), 7].detach().cpu().tolist(),
                    "base_ang_vel_z": env._robot_root_states[: min(args.num_envs, 4), 12].detach().cpu().tolist(),
                    "base_height": env._robot_root_states[: min(args.num_envs, 4), 2].detach().cpu().tolist(),
                    "base_xy": env._robot_root_states[: min(args.num_envs, 4), :2].detach().cpu().tolist(),
                    "traveled_xy": traveled[: min(args.num_envs, 4)].detach().cpu().tolist(),
                    "door_dof": env._door_dof_pos[: min(args.num_envs, 4), 0].detach().cpu().tolist(),
                    "reset": done[: min(args.num_envs, 4)].detach().cpu().tolist(),
                },
            )


if __name__ == "__main__":
    main()
