import argparse
import subprocess
import sys
from pathlib import Path


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record pull/push door expert play rollouts into raw .npz episodes. "
            "Only envs that open the door to the scripted pass threshold are saved."
        )
    )
    parser.add_argument("--mode", choices=["pull", "push", "both"], default="both")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help=(
            "Target number of parallel attempts per mode. If --num_envs is not set, "
            "this value is used as --num_envs so recording happens in one simulator launch."
        ),
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel Isaac Gym envs to launch. Defaults to --num_episodes.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="How many simulator launches to run per mode. Keep this at 1 for fastest parallel recording.",
    )
    parser.add_argument("--raw_root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp"))
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record_env_id", type=int, default=0)
    parser.add_argument("--record_all_envs", dest="record_all_envs", action="store_true", default=True)
    parser.add_argument("--no_record_all_envs", dest="record_all_envs", action="store_false")
    parser.add_argument("--no_preview_trajectory_at_spawn", action="store_true", default=True)
    parser.add_argument(
        "play_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the underlying play script after --.",
    )
    return parser.parse_args()


def script_for_mode(mode):
    if mode == "pull":
        return HIGH_LEVEL_ROOT / "play_b1z1_walk_with_door_asset_camera.py", "pull lever door open"
    if mode == "push":
        return HIGH_LEVEL_ROOT / "play_b1z1_push_with_door_asset_camera.py", "push lever door open"
    raise ValueError(mode)


def run_one(mode, rollout_idx, args):
    script, task = script_for_mode(mode)
    cmd = [
        sys.executable,
        str(script),
        "--rl_device",
        args.rl_device,
        "--sim_device",
        args.sim_device,
        "--num_envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--enable_wrist_camera",
        "--camera_seg",
        "--camera_depth",
        "--record_dp_dataset",
        "--dp_raw_root",
        args.raw_root,
        "--dp_task",
        task,
        "--dp_record_env_id",
        str(args.record_env_id),
        "--dp_fps",
        str(args.fps),
    ]
    if args.graphics_device_id is not None:
        cmd += ["--graphics_device_id", str(args.graphics_device_id)]
    if args.headless:
        cmd.append("--headless")
    if args.no_preview_trajectory_at_spawn:
        cmd.append("--no_preview_trajectory_at_spawn")
    if not args.record_all_envs:
        cmd.append("--no_dp_record_all_envs")
    extra = args.play_args[1:] if args.play_args[:1] == ["--"] else args.play_args
    cmd += extra
    attempts = args.num_envs if args.record_all_envs else 1
    print(
        f"\n=== Recording {mode} rollout {rollout_idx + 1}/{args.num_rollouts} "
        f"({attempts} parallel attempt{'s' if attempts != 1 else ''}): {' '.join(cmd)} ===",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(HIGH_LEVEL_ROOT), check=True)


def main():
    args = parse_args()
    if args.num_envs is None:
        args.num_envs = args.num_episodes
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be positive")
    modes = ["pull", "push"] if args.mode == "both" else [args.mode]
    if args.headless:
        print("Warning: Isaac Gym wrist camera tensors may be unavailable in headless mode.", flush=True)
    if args.record_all_envs:
        print(
            f"Raw recording uses all {args.num_envs} envs in parallel for {args.num_rollouts} rollout(s) per mode; "
            "failed envs are discarded.",
            flush=True,
        )
    else:
        print(f"Raw recording uses only env {args.record_env_id}; failed rollouts are discarded.", flush=True)
    for mode in modes:
        for rollout_idx in range(args.num_rollouts):
            run_one(mode, rollout_idx, args)
    print(
        "\nDone. Only successful env rollouts were saved as raw episodes. "
        "Convert raw episodes in a Python>=3.10 environment with:\n"
        f"  python high-level/dp/convert_door_raw_to_lerobot.py --raw_root {args.raw_root} --root data/lerobot --repo_id local/door_dp",
        flush=True,
    )


if __name__ == "__main__":
    main()
