import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


DP_ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = DP_ROOT.parent
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from depth_camera_aug import add_depth_aug_args, add_depth_aug_command_args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record ikpush/ikpull float-IK door expert rollouts into raw .npz episodes with "
            "a PI0.5-friendly 10D current-state observation. Only envs that open the door "
            "to the scripted pass threshold are saved."
        )
    )
    parser.add_argument("--mode", choices=["ikpush", "ikpull"], default="ikpush")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help=(
            "Target number of parallel attempts per mode. If --num_envs is not set, "
            "this value is used as --num_envs."
        ),
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel Isaac Gym envs/attempts.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="How many simulator launches to run per mode. Keep this at 1 for fastest parallel recording.",
    )
    parser.add_argument(
        "--success_per_door",
        type=int,
        default=0,
        help=(
            "If >0, keep launching float_ik rollouts until each door asset in --door_cfg has at least this "
            "many saved successful raw episodes. This quota mode ignores --num_rollouts."
        ),
    )
    parser.add_argument(
        "--max_quota_rollouts",
        type=int,
        default=100,
        help="Safety cap for --success_per_door quota mode.",
    )
    parser.add_argument("--raw_root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp"))
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--camera_fps", type=float, default=25.0)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Simulator steps. Defaults to 2405 for ikpush, 4300 for ikpull, and 2500 for pull/push.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Base seed for float_ik recording. When --num_rollouts > 1, rollout_idx is added.",
    )
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Record RGB+mask vision for push/ikpush/ikpull data instead of full depth+mask.")
    parser.add_argument("--depth_only", action="store_true", help="Record only wrist/front depth images, without mask images.")
    add_depth_aug_args(parser)
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
        return HIGH_LEVEL_ROOT / "play_b1z1_walk_with_door_asset_camera.py", "pull lever door open", False
    if mode == "push":
        return HIGH_LEVEL_ROOT / "play_b1z1_push_with_door_asset_camera.py", "push lever door open", False
    if mode == "ikpush":
        return HIGH_LEVEL_ROOT / "float_ik" / "isaacgym_float_ik_b1z1_basearn_push_door_parallel.py", "push lever door open", True
    if mode == "ikpull":
        return HIGH_LEVEL_ROOT / "float_ik" / "isaacgym_float_ik_b1z1_basearn_pull_door_parallel.py", "pull lever door open", True
    raise ValueError(mode)


def default_float_ik_scripted_args(mode):
    if mode == "ikpush":
        return [
            "--initial_hold_steps",
            "150",
            "--initial_hold_move_steps",
            "100",
            "--grasp_hold_steps",
            "5",
            "--gripper_close_steps",
            "100",
            "--door_wall_opening_width",
            "0.0",
            "--door_wall_gap",
            "0.0",
        ]
    return []


def forwarded_play_args(args):
    return args.play_args[1:] if args.play_args[:1] == ["--"] else args.play_args


def find_forwarded_arg(extra_args, name):
    prefix = name + "="
    for i, value in enumerate(extra_args):
        if value == name and i + 1 < len(extra_args):
            return extra_args[i + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def resolve_cli_path(value):
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for root in (Path.cwd(), HIGH_LEVEL_ROOT, HIGH_LEVEL_ROOT.parent):
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    return path


def door_names_from_cfg(cfg_path):
    if cfg_path is None or not Path(cfg_path).exists():
        return []
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    asset_cfg = cfg.get("env", {}).get("asset", {})
    load_block = asset_cfg.get("load_block")
    train_assets = asset_cfg.get("trainAssets", {})
    block_assets = train_assets.get(load_block, {}) if load_block is not None else {}

    def key_fn(item):
        key = item[0]
        return (0, int(key)) if str(key).isdigit() else (1, str(key))

    names = []
    for _key, spec in sorted(block_assets.items(), key=key_fn):
        name = spec.get("name")
        if name:
            names.append(str(name))
    return names


def npz_scalar_to_str(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0].item())
    return str(arr.tolist())


def count_saved_episodes_by_door(raw_root):
    counts = {}
    raw_root = Path(raw_root)
    for episode_path in sorted(raw_root.glob("episode_*.npz")):
        try:
            with np.load(episode_path, allow_pickle=True) as data:
                if "door_asset_name" not in data:
                    continue
                name = npz_scalar_to_str(data["door_asset_name"])
        except Exception as exc:
            print(f"Warning: failed to inspect {episode_path}: {exc}", flush=True)
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def format_door_quota_counts(counts, door_names):
    if door_names:
        return ", ".join(f"{name}:{int(counts.get(name, 0))}" for name in door_names)
    return ", ".join(f"{name}:{int(count)}" for name, count in sorted(counts.items()))


def quota_complete(counts, door_names, target):
    if not door_names:
        return False
    return all(int(counts.get(name, 0)) >= int(target) for name in door_names)


def run_one(mode, rollout_idx, args):
    script, task, is_float_ik = script_for_mode(mode)
    attempts = args.num_envs if args.record_all_envs else 1
    if is_float_ik:
        extra = forwarded_play_args(args)
        parallel_envs = args.num_envs if args.record_all_envs else max(1, args.record_env_id + 1)
        cmd = [
            sys.executable,
            str(script),
            "--rl_device",
            args.rl_device,
            "--sim_device",
            args.sim_device,
            "--num_envs",
            str(parallel_envs),
            "--steps",
            str(args.steps),
            "--enable_wrist_camera",
            "--enable_front_camera",
            "--camera_seg",
            "--record_dp_dataset",
            "--dp_raw_root",
            args.raw_root,
            "--dp_task",
            task,
            "--dp_record_env_id",
            str(args.record_env_id),
            "--dp_fps",
            str(args.fps),
            "--camera_fps",
            str(args.camera_fps),
            "--dp_record_state_mode",
            "pi05_current_state10",
        ]
        if args.rgb and args.depth_only:
            raise ValueError("--rgb and --depth_only are mutually exclusive.")
        if args.rgb:
            cmd += ["--rgb", "--camera_rgb", "--no_camera_depth"]
        else:
            cmd.append("--camera_depth")
            if args.depth_only:
                cmd.append("--depth_only")
        if args.graphics_device_id is not None:
            cmd += ["--graphics_device_id", str(args.graphics_device_id)]
        if args.headless:
            cmd.append("--headless")
        if args.no_preview_trajectory_at_spawn:
            cmd.append("--no_preview_trajectory_at_spawn")
        if args.record_all_envs:
            cmd.append("--dp_record_all_envs")
        else:
            cmd.append("--no_dp_record_all_envs")
        if args.seed >= 0:
            cmd += ["--seed", str(int(args.seed) + int(rollout_idx))]
        cmd += default_float_ik_scripted_args(mode)
        add_depth_aug_command_args(cmd, args)
        cmd += extra
        print(
            f"\n=== Recording {mode} rollout {rollout_idx + 1}/{args.num_rollouts} "
            f"({attempts} parallel float_ik env{'s' if attempts != 1 else ''}): {' '.join(cmd)} ===",
            flush=True,
        )
        subprocess.run(cmd, cwd=str(HIGH_LEVEL_ROOT), check=True)
        return

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
    if args.rgb and args.depth_only:
        raise ValueError("--rgb and --depth_only are mutually exclusive.")
    if args.rgb:
        cmd += ["--rgb", "--camera_rgb", "--no_camera_depth"]
    else:
        cmd.append("--camera_depth")
        if args.depth_only:
            cmd.append("--depth_only")
    if args.graphics_device_id is not None:
        cmd += ["--graphics_device_id", str(args.graphics_device_id)]
    if args.headless:
        cmd.append("--headless")
    if args.no_preview_trajectory_at_spawn:
        cmd.append("--no_preview_trajectory_at_spawn")
    if not args.record_all_envs:
        cmd.append("--no_dp_record_all_envs")
    extra = forwarded_play_args(args)
    add_depth_aug_command_args(cmd, args)
    cmd += extra
    print(
        f"\n=== Recording {mode} rollout {rollout_idx + 1}/{args.num_rollouts} "
        f"({attempts} parallel attempt{'s' if attempts != 1 else ''}): {' '.join(cmd)} ===",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(HIGH_LEVEL_ROOT), check=True)


def main():
    args = parse_args()
    args.raw_root = str(Path(args.raw_root).expanduser().resolve())
    if args.num_envs is None:
        args.num_envs = args.num_episodes
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be positive")
    if args.max_quota_rollouts <= 0:
        raise ValueError("--max_quota_rollouts must be positive")
    modes = ["pull", "push"] if args.mode == "both" else [args.mode]
    if args.steps is None:
        args.steps = 2405 if modes == ["ikpush"] else (4300 if modes == ["ikpull"] else 2500)
    if args.rgb and any(mode not in ("push", "ikpush", "ikpull") for mode in modes):
        raise ValueError("--rgb recording is only wired for push/ikpush/ikpull mode.")
    if args.headless:
        print(
            "⚠️📷 Headless raw recording requested. If Isaac Gym cannot render camera tensors, "
            "the play script will print a camera-unavailable warning and discard empty episodes.",
            flush=True,
        )
    if args.record_all_envs and len(modes) == 1 and modes[0] in ("ikpush", "ikpull"):
        print(
            f"{modes[0]} raw recording uses the parallel float_ik recorder with {args.num_envs} env(s) "
            f"for {args.num_rollouts} rollout(s); failed attempts are discarded.",
            flush=True,
        )
    elif args.record_all_envs:
        print(
            f"Raw recording uses all {args.num_envs} envs in parallel for {args.num_rollouts} rollout(s) per mode; "
            "failed envs are discarded.",
            flush=True,
        )
    else:
        print(f"Raw recording uses only env {args.record_env_id}; failed rollouts are discarded.", flush=True)
    if int(args.success_per_door) > 0:
        if len(modes) != 1 or modes[0] not in ("ikpush", "ikpull"):
            raise ValueError("--success_per_door quota mode is only supported for a single float_ik mode: ikpush or ikpull.")
        if not args.record_all_envs:
            raise ValueError("--success_per_door requires --record_all_envs so every door can be sampled.")
        extra = forwarded_play_args(args)
        door_cfg = resolve_cli_path(find_forwarded_arg(extra, "--door_cfg"))
        door_names = door_names_from_cfg(door_cfg)
        if not door_names:
            raise ValueError("--success_per_door requires a readable forwarded --door_cfg with door asset names.")
        print(
            f"Success quota mode: target={int(args.success_per_door)} successful episode(s) per door, "
            f"doors={door_names}, max_rollouts={int(args.max_quota_rollouts)}",
            flush=True,
        )
        mode = modes[0]
        rollout_idx = 0
        while True:
            counts = count_saved_episodes_by_door(args.raw_root)
            print(
                f"[quota] before rollout {rollout_idx}: {format_door_quota_counts(counts, door_names)}",
                flush=True,
            )
            if quota_complete(counts, door_names, args.success_per_door):
                break
            if rollout_idx >= int(args.max_quota_rollouts):
                raise RuntimeError(
                    f"Reached --max_quota_rollouts={args.max_quota_rollouts} before quota completed: "
                    f"{format_door_quota_counts(counts, door_names)}"
                )
            run_one(mode, rollout_idx, args)
            rollout_idx += 1
        counts = count_saved_episodes_by_door(args.raw_root)
        print(f"[quota] complete: {format_door_quota_counts(counts, door_names)}", flush=True)
    else:
        for mode in modes:
            for rollout_idx in range(args.num_rollouts):
                run_one(mode, rollout_idx, args)
    print(
        "\nDone. Only successful env rollouts were saved as raw episodes. "
        "The raw observation.state is physical 10D current state, so convert it once with:\n"
        f"  python high-level/dp/convert_door_raw_to_lerobot.py --raw_root {args.raw_root} "
        f"--root data/lerobot --repo_id local/door_pi05_state10{' --rgb' if args.rgb else ''}",
        flush=True,
    )


if __name__ == "__main__":
    main()
