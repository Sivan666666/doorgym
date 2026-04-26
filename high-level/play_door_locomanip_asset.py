import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

from isaacgym import gymapi  # noqa: F401
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz
from envs import B1Z1OpenDoor
import torch
from utils.config import load_cfg


def _use_nominal_low_level_stance(env):
    if getattr(env, "floating_base", False):
        return

    def _zero_low_policy(obs, hist_encoding=True):
        return torch.zeros(env.num_envs, 18, device=env.device, dtype=torch.float32)

    env.low_level_policy = _zero_low_policy
    env.last_low_actions.zero_()


def _disable_resets(env):
    env.max_episode_length = 10**9
    env.progress_buf.zero_()
    env.reset_buf.zero_()
    if hasattr(env, "timeout_buf"):
        env.timeout_buf.zero_()
    env.check_termination = lambda: env.reset_buf.zero_()


def _snap_robot_to_nominal_pose(env):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env._robot_root_states[:] = env._initial_robot_root_states[:]
    env._robot_root_states[:, 0] = env.door_robot_start_pose[0]
    env._robot_root_states[:, 1] = env.door_robot_start_pose[1]
    env._robot_root_states[:, 2] = env.door_robot_start_pose[2]
    base_yaw = torch.full((env.num_envs,), env.door_robot_yaw, device=env.device)
    env._robot_root_states[:, 3:7] = quat_from_euler_xyz(
        torch.zeros_like(base_yaw),
        torch.zeros_like(base_yaw),
        base_yaw,
    )
    env._robot_root_states[:, 7:13] = 0.0

    env._dof_pos[:] = env._initial_dof_pos[:]
    env._dof_vel[:] = 0.0
    env.commands.zero_()
    env.last_actions.zero_()
    env.actions.zero_()
    env.clipped_actions.zero_()
    env.last_low_actions.zero_()

    robot_ids_int32 = env._robot_actor_ids[env_ids]
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._dof_state),
        gymtorch.unwrap_tensor(robot_ids_int32),
        len(robot_ids_int32),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._root_states),
        gymtorch.unwrap_tensor(robot_ids_int32),
        len(robot_ids_int32),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env._refresh_sim_tensors()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--reset_every_s", type=float, default=5.0)
    return parser.parse_args()


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
    _use_nominal_low_level_stance(env)
    _disable_resets(env)
    _snap_robot_to_nominal_pose(env)
    env.curr_ee_goal_cart[:] = env.init_ee_goal_cart[:]
    env.curr_ee_goal_orn_rpy[:] = torch.tensor([torch.pi / 2, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    reset_interval_steps = max(1, int(round(args.reset_every_s / env.dt)))

    side_metric = torch.sum((env.arm_base - env.grasp_goal_world) * env.door_open_dir_world, dim=-1)

    print("Loaded door assets:", env.door_asset_names)
    print("Door actor count:", len(env.door_handles))
    print("Door DOF count:", env.num_door_dofs)
    print("Door body / handle body:", env.door_body_name, env.handle_body_name)
    print("Rigid body indices:", {"door_body_idx": env.door_body_idx, "handle_body_idx": env.handle_body_idx})
    print(
        "Tensor shapes:",
        {
            "door_root": tuple(env._door_root_states.shape),
            "door_dof_pos": tuple(env._door_dof_pos.shape),
            "goal_pos_offset": tuple(env.goal_pos_offset_tensor.shape),
            "grasp_goal_world": tuple(env.grasp_goal_world.shape),
        },
    )
    print("Door-side metric (>0 means robot is on the handle/open side):", side_metric.detach().cpu().tolist())
    print(f"Resetting robot+arm to nominal pose every {args.reset_every_s:.2f}s ({reset_interval_steps} control steps)")

    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for step in range(args.steps):
        if step > 0 and step % reset_interval_steps == 0:
            _snap_robot_to_nominal_pose(env)
            env.curr_ee_goal_cart[:] = env.init_ee_goal_cart[:]
            env.curr_ee_goal_orn_rpy[:] = torch.tensor([torch.pi / 2, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
            env.progress_buf.zero_()
            env.reset_buf.zero_()
            if hasattr(env, "timeout_buf"):
                env.timeout_buf.zero_()
        env.step(zero_actions)
        if step % 60 == 0:
            print(
                f"[step {step:04d}]",
                {
                    "base_door_dis": env.base_door_dis[: min(4, env.num_envs)].detach().cpu().tolist(),
                    "door_dof": env._door_dof_pos[: min(4, env.num_envs), 0].detach().cpu().tolist(),
                    "handle_dof": env._door_dof_pos[: min(4, env.num_envs), 1].detach().cpu().tolist(),
                    "door_open_ratio": env.door_open_ratio[: min(4, env.num_envs)].detach().cpu().tolist(),
                    "handle_open_ratio": env.handle_open_ratio[: min(4, env.num_envs)].detach().cpu().tolist(),
                    "door_side_metric": torch.sum((env.arm_base - env.grasp_goal_world) * env.door_open_dir_world, dim=-1)[: min(4, env.num_envs)].detach().cpu().tolist(),
                },
            )


if __name__ == "__main__":
    main()
