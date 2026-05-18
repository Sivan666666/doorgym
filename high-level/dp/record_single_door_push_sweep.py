import argparse
import random
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_SCRIPT = REPO_ROOT / "high-level" / "dp" / "record_door_dp_dataset.py"
DEFAULT_DOOR_CFG = REPO_ROOT / "high-level" / "experiments" / "isaacgym" / "b1z1_opendoor_single_99650089960001.yaml"
DEFAULT_RAW_ROOT = REPO_ROOT / "high-level" / "data" / "door_dp_raw" / "single_99650089960001_push_sweep"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record push Door DP raw episodes on one door asset by repeatedly launching "
            "the scripted recorder with randomized but conservative play parameters."
        )
    )
    parser.add_argument("--target_episodes", type=int, default=128, help="Stop once raw_root contains this many .npz episodes.")
    parser.add_argument("--max_runs", type=int, default=30, help="Maximum simulator launches before stopping.")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel envs per launch.")
    parser.add_argument("--steps", type=int, default=1080)
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--door_cfg", type=Path, default=DEFAULT_DOOR_CFG)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", dest="headless", action="store_true", default=True, help="Forward --headless. On by default.")
    parser.add_argument("--no_headless", "--no-headless", dest="headless", action="store_false", help="Open the Isaac Gym viewer.")
    parser.add_argument("--preview_cameras", action="store_true", help="Show OpenCV camera preview windows.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without launching Isaac Gym.")
    parser.add_argument(
        "extra_play_args",
        nargs=argparse.REMAINDER,
        help="Extra play-script args after --, for example: -- --pass_open_angle_deg 75",
    )
    return parser.parse_args()


def count_episodes(raw_root):
    return len(list(Path(raw_root).glob("episode_*.npz")))


def uniform(rng, lo, hi, digits=4):
    return f"{rng.uniform(lo, hi):.{digits}f}"


def sample_play_args(rng):
    speed_min = rng.uniform(0.60, 0.70)
    speed_max = rng.uniform(max(speed_min + 0.08, 0.72), 0.86)
    robot_yaw = rng.uniform(3.10, 3.18)
    return [
        "--speed_min",
        f"{speed_min:.4f}",
        "--speed_max",
        f"{speed_max:.4f}",
        "--yaw_min",
        uniform(rng, -0.08, -0.02),
        "--yaw_max",
        uniform(rng, 0.02, 0.08),
        "--robot_y",
        uniform(rng, -0.07, 0.07),
        "--robot_yaw",
        f"{robot_yaw:.4f}",
        "--pregrasp_offset",
        uniform(rng, 0.12, 0.20),
        "--grasp_x_offset",
        uniform(rng, -0.055, -0.010),
        "--grasp_z_offset",
        uniform(rng, -0.055, -0.005),
        "--push_base_vx",
        uniform(rng, 0.20, 0.32),
        "--door_push_distance",
        uniform(rng, 0.95, 1.20),
        "--handle_rotate_angle",
        uniform(rng, 0.95, 1.15),
        "--door_joint_friction",
        uniform(rng, 0.35, 0.75),
        "--door_joint_damping",
        uniform(rng, 0.12, 0.30),
    ]


def build_command(args, play_args, num_envs):
    raw_root = args.raw_root.expanduser().resolve()
    door_cfg = args.door_cfg.expanduser().resolve()
    cmd = [
        sys.executable,
        str(RECORD_SCRIPT),
        "--mode",
        "push",
        "--num_envs",
        str(num_envs),
        "--num_rollouts",
        "1",
        "--steps",
        str(args.steps),
        "--raw_root",
        str(raw_root),
        "--rl_device",
        args.rl_device,
        "--sim_device",
        args.sim_device,
    ]
    if args.graphics_device_id is not None:
        cmd += ["--graphics_device_id", str(args.graphics_device_id)]
    if args.headless:
        cmd.append("--headless")

    extra = args.extra_play_args[1:] if args.extra_play_args[:1] == ["--"] else args.extra_play_args
    cmd += [
        "--",
        "--door_cfg",
        str(door_cfg),
    ]
    if not args.preview_cameras:
        cmd.append("--no_show_seg")
    cmd += play_args + extra
    return cmd


def main():
    args = parse_args()
    rng = random.SystemRandom()
    args.raw_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    before_total = count_episodes(args.raw_root.expanduser().resolve())
    print(
        f"Recording single-door push sweep to {args.raw_root} "
        f"target_episodes={args.target_episodes} current={before_total} num_envs={args.num_envs}",
        flush=True,
    )

    for run_idx in range(args.max_runs):
        current = count_episodes(args.raw_root.expanduser().resolve())
        if current >= args.target_episodes:
            break
        play_args = sample_play_args(rng)
        launch_envs = args.num_envs
        cmd = build_command(args, play_args, launch_envs)
        print(
            f"\n=== sweep run {run_idx + 1}/{args.max_runs} "
            f"current={current}/{args.target_episodes} launch_envs={launch_envs} ===",
            flush=True,
        )
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        after = count_episodes(args.raw_root.expanduser().resolve())
        print(f"Saved this run: {after - current}; total: {after}", flush=True)

    final_total = count_episodes(args.raw_root.expanduser().resolve())
    print(f"\nDone. Episodes in {args.raw_root}: {final_total}", flush=True)


if __name__ == "__main__":
    main()
