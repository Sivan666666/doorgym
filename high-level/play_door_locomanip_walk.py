import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

from isaacgym import gymapi  # noqa: F401
from isaacgym.torch_utils import euler_from_quat
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse

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
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--resample_interval", type=int, default=60)
    parser.add_argument("--speed_min", type=float, default=0.05)
    parser.add_argument("--speed_max", type=float, default=0.25)
    return parser.parse_args()


def _sample_forward_actions(num_envs, device, speed_min, speed_max):
    return torch.empty(num_envs, device=device).uniform_(speed_min, speed_max)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)

    cfg = load_cfg("data/cfg/b1z1_opendoor.yaml")
    cfg["env"]["numEnvs"] = args.num_envs
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

    # Keep the arm at its current pose so this script isolates walking behavior.
    ee_local_pos = quat_rotate_inverse(env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
    ee_local_orn = quat_mul(quat_conjugate(env._robot_root_states[:, 3:7]), env.ee_orn)
    env.curr_ee_goal_cart[:] = ee_local_pos
    env.curr_ee_goal_orn_rpy[:] = torch.stack(euler_from_quat(ee_local_orn), dim=-1)

    side_metric = torch.sum((env.arm_base - env.grasp_goal_world) * env.door_open_dir_world, dim=-1)
    print("Loaded door assets:", env.door_asset_names)
    print("Door-side metric (>0 means robot is on the handle/open side):", side_metric.detach().cpu().tolist())

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    forward_actions = _sample_forward_actions(env.num_envs, env.device, args.speed_min, args.speed_max)
    actions[:, 7] = forward_actions

    for step in range(args.steps):
        if step % args.resample_interval == 0:
            forward_actions = _sample_forward_actions(env.num_envs, env.device, args.speed_min, args.speed_max)
            actions[:, 7] = forward_actions

        env.step(actions)

        if step % 30 == 0:
            print(
                f"[step {step:04d}]",
                {
                    "forward_action": actions[: min(4, env.num_envs), 7].detach().cpu().tolist(),
                    "command_x": env.commands[: min(4, env.num_envs), 0].detach().cpu().tolist(),
                    "base_lin_vel_x": env._robot_root_states[: min(4, env.num_envs), 7].detach().cpu().tolist(),
                    "base_height": env._robot_root_states[: min(4, env.num_envs), 2].detach().cpu().tolist(),
                    "base_door_dis": env.base_door_dis[: min(4, env.num_envs)].detach().cpu().tolist(),
                },
            )


if __name__ == "__main__":
    main()
