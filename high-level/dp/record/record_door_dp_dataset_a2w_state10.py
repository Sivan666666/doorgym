import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


DP_ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = DP_ROOT.parent
REPO_ROOT = HIGH_LEVEL_ROOT.parent
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from depth_camera_aug import add_depth_aug_args, add_depth_aug_command_args, apply_depth_aug_config_defaults
from door_dp_common import DEFAULT_KEYFRAME_LOSS_RADIUS, DEFAULT_KEYFRAME_LOSS_WEIGHT


A2W_IKPUSH_SCRIPT = (
    HIGH_LEVEL_ROOT / "float_ik" / "isaacgym_float_ik_a2w_basearn_push_door_parallel.py"
)
A2W_RAW_ROOT = HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp_a2w_state10"
A2W_LEROBOT_REPO_ID = "local/door_a2w_state10"
A2W_EE_ACTION10_NAMES = ["vx", "yaw", "ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw", "gripper"]
A2W_JOINT_ACTION9_NAMES = ["vx", "yaw", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record A2W+Z1 ikpush float-IK door expert rollouts into raw .npz episodes "
            "with a PI0.5-friendly 10D state whose first two dimensions are the last "
            "commanded vx/vyaw. Recording frequency and the remaining state dimensions "
            "match record_door_dp_dataset_pi05_state10.py. Only successful envs are saved."
        )
    )
    parser.add_argument("--mode", choices=["ikpush"], default="ikpush")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help=(
            "Target total number of successful raw episodes in --raw_root. The script "
            "keeps launching batches until this count is reached."
        ),
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Maximum number of parallel Isaac Gym envs in each recording batch.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="Deprecated compatibility option; normal recording now runs until --num_episodes successes.",
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
        help="Safety cap on simulator launches for success-target and per-door quota modes.",
    )
    parser.add_argument("--raw_root", type=str, default=str(A2W_RAW_ROOT))
    parser.add_argument(
        "--state_action_mode",
        choices=["ee_state10", "joint_state9"],
        default="ee_state10",
        help=(
            "ee_state10 keeps the old 10D state/action schema "
            "[last_command_vx, last_command_vyaw, EE pose, gripper] + EE target action. "
            "joint_state9 records 9D state/action "
            "[last_command_vx, last_command_vyaw, joint1..joint6, jointGripper] "
            "and uses joint targets as actions."
        ),
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--camera_fps", type=float, default=25.0)
    parser.add_argument("--camera_depth_clip_lower", type=float, default=0.2)
    parser.add_argument("--camera_depth_clip_far", type=float, default=1.5)
    parser.add_argument(
        "--keyframe_loss_weight",
        type=float,
        default=DEFAULT_KEYFRAME_LOSS_WEIGHT,
        help="Action loss weight λ applied to frames within --keyframe_loss_radius of extracted keyframes.",
    )
    parser.add_argument(
        "--keyframe_loss_radius",
        type=int,
        default=DEFAULT_KEYFRAME_LOSS_RADIUS,
        help="Frame radius δ around each extracted keyframe that receives --keyframe_loss_weight.",
    )
    parser.add_argument(
        "--no_keyframe_loss_weights",
        action="store_true",
        help="Record keyframe indices but keep action_loss_weight at 1.0 everywhere.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Simulator steps. Defaults to 1000 for A2W dataset recording.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Base seed for float_ik recording. The internal batch index is added for each launch.",
    )
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Record A2W ikpush RGB+mask vision instead of depth-only vision.")
    parser.add_argument("--depth_only", dest="depth_only", action="store_true", default=True, help="Record only wrist/front depth images, without mask images.")
    parser.add_argument("--no_depth_only", dest="depth_only", action="store_false", help="Record legacy depth+mask image inputs.")
    add_depth_aug_args(parser)
    parser.add_argument("--record_env_id", type=int, default=0)
    parser.add_argument("--record_all_envs", dest="record_all_envs", action="store_true", default=True)
    parser.add_argument("--no_record_all_envs", dest="record_all_envs", action="store_false")
    parser.add_argument("--no_preview_trajectory_at_spawn", action="store_true", default=True)
    parser.add_argument(
        "--run_log_root",
        type=str,
        default=None,
        help="Directory for per-run command/log/git metadata. Defaults to <raw_root>/_run_logs.",
    )
    parser.add_argument(
        "--no_save_run_metadata",
        dest="save_run_metadata",
        action="store_false",
        default=True,
        help="Disable automatic command, args, git, and terminal log capture.",
    )
    parser.add_argument(
        "play_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the underlying play script after --.",
    )
    return parser.parse_args()


def script_for_mode(mode):
    if mode == "ikpush":
        return A2W_IKPUSH_SCRIPT, "push lever door open", True
    raise ValueError(mode)


def dp_record_state_mode_for_schema(schema):
    if schema == "joint_state9":
        return "a2w_last_command_joint_state9"
    if schema == "ee_state10":
        return "pi05_last_command_state10"
    raise ValueError(schema)


def expected_action_names_for_schema(schema):
    if schema == "joint_state9":
        return list(A2W_JOINT_ACTION9_NAMES)
    if schema == "ee_state10":
        return list(A2W_EE_ACTION10_NAMES)
    raise ValueError(schema)


def assert_raw_root_schema_compatible(raw_root, schema):
    sidecar_path = Path(raw_root) / "door_dp_feature_names.json"
    if not sidecar_path.exists():
        return
    with sidecar_path.open("r", encoding="utf-8") as f:
        sidecar = json.load(f)
    existing_action = list(sidecar.get("action", []))
    expected_action = expected_action_names_for_schema(schema)
    if existing_action and existing_action != expected_action:
        raise ValueError(
            f"Existing raw_root {raw_root} has action schema {existing_action}, "
            f"but --state_action_mode {schema} expects {expected_action}. "
            "Please use a new --raw_root or remove the old raw data."
        )


def default_float_ik_scripted_args(mode):
    if mode == "ikpush":
        return [
            "--initial_hold_steps",
            "150",
            "--initial_hold_move_steps",
            "100",
            "--grasp_steps",
            "50",
            "--grasp_hold_steps",
            "0",
            "--gripper_close_steps",
            "50",
            "--handle_rotate_steps",
            "100",
            "--door_push_steps",
            "300",
            "--return_home_steps",
            "150",
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


def count_saved_episodes(raw_root):
    return len(list(Path(raw_root).glob("episode_*.npz")))


def format_door_quota_counts(counts, door_names):
    if door_names:
        return ", ".join(f"{name}:{int(counts.get(name, 0))}" for name in door_names)
    return ", ".join(f"{name}:{int(count)}" for name, count in sorted(counts.items()))


def quota_complete(counts, door_names, target):
    if not door_names:
        return False
    return all(int(counts.get(name, 0)) >= int(target) for name in door_names)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def unique_run_dir(root):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(root) / f"{stamp}_pid{os.getpid()}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def capture_command(cmd, cwd):
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return f"failed to run {' '.join(cmd)}: {exc}\n"
    output = result.stdout or ""
    if result.stderr:
        output += result.stderr
    if result.returncode != 0:
        output += f"\n[exit_code={result.returncode}]\n"
    return output


def write_run_metadata(args):
    if not bool(getattr(args, "save_run_metadata", True)):
        args.run_metadata_dir = ""
        return None, None

    root = Path(args.run_log_root).expanduser() if args.run_log_root else Path(args.raw_root) / "_run_logs"
    run_dir = unique_run_dir(root)
    args.run_metadata_dir = str(run_dir)

    top_level_cmd = [sys.executable] + sys.argv
    (run_dir / "command.txt").write_text(shlex.join(top_level_cmd) + "\n", encoding="utf-8")
    (run_dir / "argv.json").write_text(json.dumps(top_level_cmd, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(
        capture_command(["git", "rev-parse", "HEAD"], REPO_ROOT),
        encoding="utf-8",
    )
    (run_dir / "git_status.txt").write_text(
        capture_command(["git", "status", "--short"], REPO_ROOT),
        encoding="utf-8",
    )
    (run_dir / "git_diff.patch").write_text(
        capture_command(["git", "diff"], REPO_ROOT),
        encoding="utf-8",
    )
    (run_dir / "subprocess_commands.txt").write_text("", encoding="utf-8")

    terminal_log = (run_dir / "terminal.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, terminal_log)
    sys.stderr = TeeStream(sys.__stderr__, terminal_log)

    print(f"Run metadata will be saved to: {run_dir}", flush=True)
    return run_dir, terminal_log


def run_logged_subprocess(cmd, args, cwd, check=True):
    run_dir = getattr(args, "run_metadata_dir", "")
    if run_dir:
        with (Path(run_dir) / "subprocess_commands.txt").open("a", encoding="utf-8") as f:
            f.write(shlex.join([str(item) for item in cmd]) + "\n")

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    returncode = process.wait()
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return returncode


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
            "--camera_depth_clip_lower",
            str(args.camera_depth_clip_lower),
            "--camera_depth_clip_far",
            str(args.camera_depth_clip_far),
            "--dp_record_state_mode",
            dp_record_state_mode_for_schema(args.state_action_mode),
            "--keyframe_loss_weight",
            str(args.keyframe_loss_weight),
            "--keyframe_loss_radius",
            str(args.keyframe_loss_radius),
        ]
        if args.no_keyframe_loss_weights:
            cmd.append("--no_keyframe_loss_weights")
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
            f"\n=== Recording {mode} batch {rollout_idx + 1} "
            f"({attempts} parallel float_ik env{'s' if attempts != 1 else ''}): {' '.join(cmd)} ===",
            flush=True,
        )
        run_logged_subprocess(cmd, args, cwd=HIGH_LEVEL_ROOT, check=True)
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
        f"\n=== Recording {mode} batch {rollout_idx + 1} "
        f"({attempts} parallel attempt{'s' if attempts != 1 else ''}): {' '.join(cmd)} ===",
        flush=True,
    )
    run_logged_subprocess(cmd, args, cwd=HIGH_LEVEL_ROOT, check=True)


def record_until_target_successes(args, mode):
    target = int(args.num_episodes)
    batch_size = int(args.num_envs)
    rollout_idx = 0

    while True:
        saved_before = count_saved_episodes(args.raw_root)
        if saved_before >= target:
            print(f"[target] complete: successful episodes={saved_before}/{target}", flush=True)
            break
        if rollout_idx >= int(args.max_quota_rollouts):
            raise RuntimeError(
                f"Reached --max_quota_rollouts={args.max_quota_rollouts} with "
                f"{saved_before}/{target} successful episodes."
            )

        remaining = target - saved_before
        args.num_envs = min(batch_size, remaining) if args.record_all_envs else 1
        print(
            f"[target] before batch {rollout_idx + 1}: successful={saved_before}/{target} "
            f"remaining={remaining} batch_envs={args.num_envs}",
            flush=True,
        )
        run_one(mode, rollout_idx, args)
        saved_after = count_saved_episodes(args.raw_root)
        print(
            f"[target] after batch {rollout_idx + 1}: successful={saved_after}/{target} "
            f"new={saved_after - saved_before}",
            flush=True,
        )
        rollout_idx += 1

    args.num_envs = batch_size


def main():
    args = parse_args()
    apply_depth_aug_config_defaults(args, sys.argv[1:])
    args.raw_root = str(Path(args.raw_root).expanduser().resolve())
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be positive")
    if args.max_quota_rollouts <= 0:
        raise ValueError("--max_quota_rollouts must be positive")
    modes = [args.mode]
    if args.steps is None:
        args.steps = 1000
    if args.rgb:
        args.depth_only = False
    assert_raw_root_schema_compatible(args.raw_root, args.state_action_mode)

    run_dir, terminal_log = write_run_metadata(args)
    _ = run_dir, terminal_log

    if args.headless:
        print(
            "⚠️📷 Headless raw recording requested. If Isaac Gym cannot render camera tensors, "
            "the play script will print a camera-unavailable warning and discard empty episodes.",
            flush=True,
        )
    if args.record_all_envs:
        print(
            f"A2W ikpush recording target={args.num_episodes} successful episode(s), "
            f"batch_size={args.num_envs} env(s), schema={args.state_action_mode}; failed attempts are discarded.",
            flush=True,
        )
    else:
        print(f"Raw recording uses only env {args.record_env_id}; failed rollouts are discarded.", flush=True)
    if int(args.success_per_door) > 0:
        if modes != ["ikpush"]:
            raise ValueError("--success_per_door quota mode is only supported for A2W ikpush.")
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
        record_until_target_successes(args, modes[0])
    if args.state_action_mode == "joint_state9":
        schema_desc = (
            "The raw observation.state/action are both 9D "
            "[last_command_vx, last_command_vyaw, joint1..joint6, jointGripper]"
        )
        repo_hint = "local/door_a2w_joint_state9"
    else:
        schema_desc = (
            "The raw observation.state is 10D "
            "[last_command_vx, last_command_vyaw, current EE pose, gripper]"
        )
        repo_hint = A2W_LEROBOT_REPO_ID
    print(
        "\nDone. Only successful env rollouts were saved as raw episodes. "
        f"{schema_desc}. Each raw episode also includes keyframe_indices/keyframe_names/"
        "action_loss_weight for weighted ACT training, so convert it once with:\n"
        f"  python high-level/dp/convert_door_raw_to_lerobot.py --raw_root {args.raw_root} "
        f"--root data/lerobot --repo_id {repo_hint}{' --rgb' if args.rgb else ''}",
        flush=True,
    )


if __name__ == "__main__":
    main()
