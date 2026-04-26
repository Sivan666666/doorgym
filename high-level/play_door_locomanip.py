import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

from isaacgym import gymapi  # noqa: F401
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse
from isaacgym.torch_utils import euler_from_quat

from envs import B1Z1OpenDoor
import torch
from utils.config import load_cfg


WALK_TO_DOOR = 0
MOVE_TO_PREGRASP = 1
MOVE_TO_HANDLE = 2
CLOSE_AND_SEAT = 3
PRESS_HANDLE = 4
PULL_DOOR = 5
HOLD_OPEN = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=700)
    return parser.parse_args()


def _set_phase(phase, phase_steps, mask, new_phase):
    if torch.any(mask):
        phase[mask] = new_phase
        phase_steps[mask] = 0


def _phase_counts(phase):
    return {
        "walk": int((phase == WALK_TO_DOOR).sum().item()),
        "pregrasp": int((phase == MOVE_TO_PREGRASP).sum().item()),
        "handle": int((phase == MOVE_TO_HANDLE).sum().item()),
        "close": int((phase == CLOSE_AND_SEAT).sum().item()),
        "press": int((phase == PRESS_HANDLE).sum().item()),
        "pull": int((phase == PULL_DOOR).sum().item()),
        "hold": int((phase == HOLD_OPEN).sum().item()),
    }


def _pin_robot_base(env, target_root_states):
    env._robot_root_states[:] = target_root_states
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._root_states),
        gymtorch.unwrap_tensor(env._robot_actor_ids),
        len(env._robot_actor_ids),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env._refresh_sim_tensors()


def _scripted_joint_assist(env, phase):
    press_like = phase == PRESS_HANDLE
    if torch.any(press_like):
        target_handle = torch.maximum(
            env._door_dof_pos[press_like, 1] + 0.03,
            env.handle_press_threshold[press_like] + 0.08,
        )
        target_handle = torch.clamp(target_handle, min=0.0)
        target_handle = torch.minimum(target_handle, env.handle_limits_upper[env.door_asset_indices[press_like]])
        env._door_dof_pos[press_like, 1] = target_handle
        env._door_dof_vel[press_like, 1] = 0.0

    pull_like = ((phase == PULL_DOOR) | (phase == HOLD_OPEN)) & env.open_door_stage
    if torch.any(pull_like):
        target_door = torch.clamp(env._door_dof_pos[pull_like, 0] + 0.025, min=0.0)
        target_door = torch.minimum(target_door, env.door_hinge_limits_upper[env.door_asset_indices[pull_like]] * 0.75)
        env._door_dof_pos[pull_like, 0] = target_door
        env._door_dof_vel[pull_like, 0] = 0.0

    assisted = press_like | pull_like
    if torch.any(assisted):
        actor_ids = env._door_actor_ids[assisted]
        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env._dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env._refresh_sim_tensors()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)

    cfg = load_cfg("data/cfg/b1z1_opendoor.yaml")
    cfg["env"]["numEnvs"] = args.num_envs
    cfg["env"]["maxEpisodeLength"] = max(cfg["env"]["maxEpisodeLength"], args.steps + 20)
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
    ee_local_pos = quat_rotate_inverse(env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
    ee_local_orn = quat_mul(quat_conjugate(env._robot_root_states[:, 3:7]), env.ee_orn)
    env.curr_ee_goal_cart[:] = ee_local_pos
    env.curr_ee_goal_orn_rpy[:] = torch.stack(euler_from_quat(ee_local_orn), dim=-1)
    env.check_termination = lambda: env.reset_buf.zero_()

    scripted_base_root_states = env._robot_root_states.clone()
    scripted_base_root_states[:, 7:13] = 0.0

    phase = torch.full((env.num_envs,), MOVE_TO_PREGRASP, device=env.device, dtype=torch.long)
    phase_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    for step in range(args.steps):
        phase_steps += 1

        goal_local = env.goal_pos_local_yaw
        target_rot = env.ee_orn.clone()
        post_align_mask = phase >= MOVE_TO_HANDLE
        if torch.any(post_align_mask):
            target_rot[post_align_mask] = env.handle_target_rot[post_align_mask]

        target_pos = env.get_pregrasp_goal_world(offset=0.22)
        gripper_open = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        base_x_cmd = torch.zeros(env.num_envs, device=env.device)
        yaw_cmd = torch.zeros(env.num_envs, device=env.device)

        walk_mask = phase == WALK_TO_DOOR
        pregrasp_mask = phase == MOVE_TO_PREGRASP
        handle_mask = phase == MOVE_TO_HANDLE
        close_mask = phase == CLOSE_AND_SEAT
        press_mask = phase == PRESS_HANDLE
        pull_mask = phase == PULL_DOOR
        hold_mask = phase == HOLD_OPEN

        if torch.any(walk_mask):
            target_pos[walk_mask] = env.get_pregrasp_goal_world(offset=0.24)[walk_mask]
            yaw_cmd[walk_mask] = torch.clamp(goal_local[walk_mask, 1] * 1.2, -0.15, 0.15)

        if torch.any(pregrasp_mask):
            target_pos[pregrasp_mask] = env.get_pregrasp_goal_world(offset=0.20)[pregrasp_mask]

        if torch.any(handle_mask):
            target_pos[handle_mask] = (env.grasp_goal_world - env.handle_approach_dir_world * 0.035)[handle_mask]

        if torch.any(close_mask):
            seat_target = env.grasp_goal_world + env.handle_approach_dir_world * 0.005
            target_pos[close_mask] = seat_target[close_mask]
            gripper_open[close_mask] = False

        if torch.any(press_mask):
            press_target = (
                env.grasp_goal_world
                + env.handle_approach_dir_world * 0.002
                - env.handle_rotate_dir_world * 0.045
            )
            target_pos[press_mask] = press_target[press_mask]
            gripper_open[press_mask] = False

        if torch.any(pull_mask):
            pull_target = (
                env.ee_pos
                + env.door_open_dir_world * 0.05
                - env.handle_rotate_dir_world * 0.015
            )
            target_pos[pull_mask] = pull_target[pull_mask]
            gripper_open[pull_mask] = False

        if torch.any(hold_mask):
            hold_target = env.ee_pos + env.door_open_dir_world * 0.01
            target_pos[hold_mask] = hold_target[hold_mask]
            gripper_open[hold_mask] = False

        scripted_actions = env.scripted_actions_from_world_targets(target_pos, target_rot, gripper_open, base_x_cmd, yaw_cmd)
        _, _, done, _ = env.step(scripted_actions)
        _pin_robot_base(env, scripted_base_root_states)
        _scripted_joint_assist(env, phase)

        walk_done = (phase == WALK_TO_DOOR) & (env.curr_dist < 0.45)
        pregrasp_done = (phase == MOVE_TO_PREGRASP) & (env.curr_dist < 0.20)
        handle_done = (phase == MOVE_TO_HANDLE) & ((env.curr_dist < 0.10) | (phase_steps > 60))
        close_done = (phase == CLOSE_AND_SEAT) & (phase_steps > 18)
        press_done = (phase == PRESS_HANDLE) & (env.open_door_stage | (env.handle_open_ratio > env.handle_press_threshold_ratio * 0.92) | (phase_steps > 60))
        pull_done = (phase == PULL_DOOR) & (env.door_open_success | (phase_steps > 160))

        _set_phase(phase, phase_steps, walk_done, MOVE_TO_PREGRASP)
        _set_phase(phase, phase_steps, pregrasp_done, MOVE_TO_HANDLE)
        _set_phase(phase, phase_steps, handle_done, CLOSE_AND_SEAT)
        _set_phase(phase, phase_steps, close_done, PRESS_HANDLE)
        _set_phase(phase, phase_steps, press_done, PULL_DOOR)
        _set_phase(phase, phase_steps, pull_done, HOLD_OPEN)

        if step % 25 == 0:
            print(
                f"[step {step:04d}]",
                {
                    "dist": round(env.curr_dist.mean().item(), 4),
                    "door_dof": round(env._door_dof_pos[:, 0].mean().item(), 4),
                    "handle_dof": round(env._door_dof_pos[:, 1].mean().item(), 4),
                    "handle_ratio": round(env.handle_open_ratio.mean().item(), 4),
                    "door_ratio": round(env.door_open_ratio.mean().item(), 4),
                    "success": int(env.door_open_success.sum().item()),
                    "phase": _phase_counts(phase),
                    "door_side_metric": round(torch.sum((env.arm_base - env.grasp_goal_world) * env.door_open_dir_world, dim=-1).mean().item(), 4),
                },
            )

        if bool(torch.all(env.door_open_success)):
            print("All environments reached the door-open success threshold.")
            break


if __name__ == "__main__":
    main()
