import argparse
import random
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RECORD_SCRIPT = REPO_ROOT / "high-level" / "dp" / "record" / "record_door_dp_dataset.py"
DEFAULT_DOOR_CFG = REPO_ROOT / "high-level" / "experiments" / "isaacgym" / "b1z1_opendoor_single_door0.yaml"
DEFAULT_RAW_ROOTS = {
    "ikpush": REPO_ROOT / "high-level" / "data" / "door_dp_raw" / "single_door0_push_sweep",
    "ikpull": REPO_ROOT / "high-level" / "data" / "door_dp_raw" / "single_door0_pull_sweep",
}
DEFAULT_STEPS = {"ikpush": 2210, "ikpull": 4300}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record float_ik Door DP raw episodes on one door asset by repeatedly launching "
            "the float_ik recorder with a new seed for each run."
        )
    )
    parser.add_argument("--mode", choices=["ikpush", "ikpull"], default="ikpush")
    parser.add_argument("--target_episodes", type=int, default=128, help="Stop once raw_root contains this many .npz episodes.")
    parser.add_argument("--max_runs", type=int, default=30, help="Maximum simulator launches before stopping.")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel float_ik envs per sweep run.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--raw_root", type=Path, default=None)
    parser.add_argument("--door_cfg", type=Path, default=DEFAULT_DOOR_CFG)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", dest="headless", action="store_true", default=True, help="Forward --headless. On by default.")
    parser.add_argument("--no_headless", "--no-headless", dest="headless", action="store_false", help="Open the Isaac Gym viewer.")
    parser.add_argument("--rgb", action="store_true", help="Record RGB+mask Door DP data.")
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


def build_command(args, play_args, num_envs, seed):
    raw_root = args.raw_root.expanduser().resolve()
    door_cfg = args.door_cfg.expanduser().resolve()
    cmd = [
        sys.executable,
        str(RECORD_SCRIPT),
        "--mode",
        args.mode,
        "--num_envs",
        str(num_envs),
        "--num_rollouts",
        "1",
        "--steps",
        str(args.steps),
        "--seed",
        str(int(seed)),
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
    if args.rgb:
        cmd.append("--rgb")

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
    if args.steps is None:
        args.steps = DEFAULT_STEPS[args.mode]
    if args.raw_root is None:
        args.raw_root = DEFAULT_RAW_ROOTS[args.mode]
    rng = random.SystemRandom()
    args.raw_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    before_total = count_episodes(args.raw_root.expanduser().resolve())
    print(
        f"Recording single-door {args.mode} sweep to {args.raw_root} "
        f"target_episodes={args.target_episodes} current={before_total} num_envs={args.num_envs}",
        flush=True,
    )

    for run_idx in range(args.max_runs):
        current = count_episodes(args.raw_root.expanduser().resolve())
        if current >= args.target_episodes:
            break
        run_start = current
        run_seed = rng.randrange(0, 2**31 - 1)
        cmd = build_command(args, [], args.num_envs, run_seed)
        print(
            f"\n=== sweep run {run_idx + 1}/{args.max_runs} "
            f"current={current}/{args.target_episodes} parallel_envs={args.num_envs} seed={run_seed} ===",
            flush=True,
        )
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        after = count_episodes(args.raw_root.expanduser().resolve())
        print(f"Saved this run: {after - run_start}; total: {after}", flush=True)

    final_total = count_episodes(args.raw_root.expanduser().resolve())
    print(f"\nDone. Episodes in {args.raw_root}: {final_total}", flush=True)


if __name__ == "__main__":
    main()
