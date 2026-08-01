import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


DP_ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = DP_ROOT.parent
PROJECT_ROOT = HIGH_LEVEL_ROOT.parent
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from depth_camera_aug import add_depth_aug_args, add_depth_aug_command_args


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def official_lerobot_policy_dir(checkpoint_path):
    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_file() and path.parent.name == "pretrained_model":
        policy_dir = path.parent
        step_dir = policy_dir.parent
    elif path.name == "pretrained_model":
        policy_dir = path
        step_dir = path.parent
    else:
        step_dir = path
        policy_dir = path / "pretrained_model"
    if (policy_dir / "config.json").is_file() and (policy_dir / "train_config.json").is_file():
        return step_dir, policy_dir
    return None, None


def lerobot_python_command():
    explicit = os.environ.get("DOOR_DP_LEROBOT_PYTHON")
    if explicit:
        return explicit.split()
    env_name = os.environ.get("DOOR_DP_LEROBOT_CONDA_ENV", "b1z1_lerobot")
    conda_exe = shutil.which("conda")
    if conda_exe:
        return [conda_exe, "run", "--no-capture-output", "-n", env_name, "python"]
    return [sys.executable]


def resolve_train_dataset_root(train_config):
    dataset = train_config.get("dataset") or {}
    repo_id = dataset.get("repo_id")
    root = dataset.get("root")
    if not repo_id or not root:
        raise ValueError(
            "Official LeRobot checkpoint is missing dataset.root or dataset.repo_id in "
            "pretrained_model/train_config.json, so Door metadata cannot be located automatically."
        )
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    else:
        root = root.resolve()
    return root, str(repo_id)


def auto_wrap_official_lerobot_checkpoint(checkpoint_path, args):
    step_dir, policy_dir = official_lerobot_policy_dir(checkpoint_path)
    if policy_dir is None:
        return checkpoint_path

    policy_config = load_json(policy_dir / "config.json")
    policy_type = str(policy_config.get("type", ""))
    if policy_type not in ("act", "diffusion", "pi05", "pi05_evo"):
        raise ValueError(
            f"Direct play of official LeRobot policy type={policy_type!r} is not wired yet. "
            "Currently auto-wrapping supports official ACT, Diffusion, and pi0.5 checkpoints."
        )

    train_config = load_json(policy_dir / "train_config.json")
    dataset_root, repo_id = resolve_train_dataset_root(train_config)
    sidecar = dataset_root / "door_dp_feature_names.json"
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"Could not find Door dataset sidecar for official LeRobot checkpoint: {sidecar}\n"
            "This file stores action_frame, state/action preprocess, and image mode for Door play."
        )
    sidecar_data = load_json(sidecar)
    dataset_vision_mode = str(sidecar_data.get("vision_mode", "depth")).lower().replace("-", "_")
    if dataset_vision_mode == "depth_only":
        args.depth_only = True
    elif dataset_vision_mode == "rgb":
        args.rgb = True

    run_name = str(train_config.get("job_name") or step_dir.parent.parent.name or "official_lerobot")
    step_name = step_dir.name
    cache_root = DP_ROOT / "logs" / "door-auto-wrapped" / run_name / step_name
    out_dir = cache_root / "model_latest"
    manifest_path = cache_root / "model_latest.pt"
    cached_meta_path = out_dir / "door_policy_meta.json"
    if cached_meta_path.is_file() and manifest_path.is_file():
        cached_meta = load_json(cached_meta_path)
        cached_policy = cached_meta.get("policy_config") or {}
        cached_end = bool(cached_policy.get("end_signal_prediction", False))
        requested_end = bool(policy_config.get("end_signal_prediction", False))
        cached_interaction = bool(cached_policy.get("interaction_state_conditioning", False))
        requested_interaction = bool(policy_config.get("interaction_state_conditioning", False))
        cached_interaction_mode = str(
            cached_policy.get("interaction_state_prediction_mode", "encoder_current")
        )
        requested_interaction_mode = str(
            policy_config.get("interaction_state_prediction_mode", "encoder_current")
        )
        if (
            cached_end == requested_end
            and cached_interaction == requested_interaction
            and cached_interaction_mode == requested_interaction_mode
        ):
            print(f"Using cached Door-wrapped checkpoint: {manifest_path}", flush=True)
            return manifest_path.resolve()
        print(
            "Cached Door wrapper has stale auxiliary-head metadata; rebuilding it from the official checkpoint.",
            flush=True,
        )

    if policy_type == "act":
        export_script = DP_ROOT / "export_official_lerobot_act_to_door_checkpoint.py"
    elif policy_type == "diffusion":
        export_script = DP_ROOT / "export_official_lerobot_diffusion_to_door_checkpoint.py"
    else:
        export_script = DP_ROOT / "export_official_lerobot_pi05_to_door_checkpoint.py"
    cmd = lerobot_python_command() + [
        str(export_script),
        "--official_checkpoint",
        str(step_dir),
        "--root",
        str(dataset_root),
        "--repo_id",
        repo_id,
        "--out_dir",
        str(out_dir),
        "--manifest_name",
        "model_latest.pt",
        "--device",
        args.rl_device,
    ]
    if policy_type == "diffusion":
        cmd += [
            "--num_inference_steps",
            str(args.dp_inference_steps),
            "--noise_scheduler_type",
            args.dp_noise_scheduler_type,
        ]
    elif policy_type in ("pi05", "pi05_evo"):
        cmd += [
            "--num_inference_steps",
            str(args.dp_inference_steps),
        ]
        if args.dp_action_horizon is not None:
            cmd += ["--action_horizon", str(args.dp_action_horizon)]
    if args.rgb:
        cmd.append("--rgb")
    elif getattr(args, "depth_only", False):
        cmd.append("--depth_only")
    env = os.environ.copy()
    py_paths = [str(HIGH_LEVEL_ROOT / "lerobot" / "src"), str(DP_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(py_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    print(
        f"Detected official LeRobot {policy_type} checkpoint; wrapping it for Door play:\n"
        f"  official: {step_dir}\n"
        f"  dataset:  {dataset_root}\n"
        f"  output:   {manifest_path}",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    return manifest_path.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained Door LeRobot policy in the door asset scene.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--expert_action_replay_raw_episode",
        type=str,
        default=None,
        help=(
            "Use one raw episode's fixed action sequence instead of a learned checkpoint. "
            "The current randomized simulator state is retained."
        ),
    )
    parser.add_argument("--mode", choices=["ikpush", "ikpull", "pull", "push"], default="ikpush")
    parser.add_argument(
        "--robot_body",
        "--robot",
        dest="robot_body",
        choices=["b1z1", "a2wz1"],
        default="b1z1",
        help="Robot play script to run. a2wz1 supports --mode ikpush and --mode ikpull.",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Run a RGB+mask Door policy checkpoint. Push/ikpush/ikpull modes only.")
    parser.add_argument("--depth_only", dest="depth_only", action="store_true", default=True, help="Run a Door policy checkpoint trained with wrist/front depth only.")
    parser.add_argument("--no_depth_only", dest="depth_only", action="store_false", help="Run a legacy depth+mask Door policy checkpoint.")
    add_depth_aug_args(parser)
    parser.add_argument("--show_seg", dest="show_seg", action="store_true", default=True)
    parser.add_argument("--no_show_seg", dest="show_seg", action="store_false")
    parser.add_argument("--camera_display_scale", type=int, default=1)
    parser.add_argument(
        "--debug_visuals",
        action="store_true",
        help="Keep viewer-only debug markers such as EE target spheres and camera axes.",
    )
    parser.add_argument("--dp_inference_steps", type=int, default=10)
    parser.add_argument("--dp_noise_scheduler_type", type=str.upper, choices=["DDIM", "DDPM"], default="DDIM")
    parser.add_argument("--dp_action_horizon", type=int, default=None)
    parser.add_argument(
        "--dp_temporal_ensemble",
        action="store_true",
        help="Use NX-style action chunk overlap fusion for 10D EE actions during local sim play.",
    )
    parser.add_argument("--dp_temporal_prefetch_actions", type=int, default=3)
    parser.add_argument("--dp_temporal_old_weight", type=float, default=0.3)
    parser.add_argument("--dp_temporal_new_weight", type=float, default=0.7)
    parser.add_argument("--dp_end_signal_monitor", action="store_true")
    parser.add_argument("--dp_end_signal_threshold", type=float, default=0.8)
    parser.add_argument("--dp_end_signal_consecutive_steps", type=int, default=10)
    parser.add_argument("--dp_control_env_id", type=int, default=0)
    parser.add_argument("--dp_control_all_envs", dest="dp_control_all_envs", action="store_true", default=True)
    parser.add_argument("--no_dp_control_all_envs", dest="dp_control_all_envs", action="store_false")
    parser.add_argument("--dp_log_path", type=str, default=None)
    parser.add_argument("--dp_log_interval", type=int, default=25)
    parser.add_argument(
        "--dp_log_replay_snapshot",
        action="store_true",
        help="Include replay-style simulator snapshot fields in each policy JSONL record.",
    )
    parser.add_argument("--no_dp_print", dest="dp_print", action="store_false", default=True)
    parser.add_argument("--dp_warmstart", action="store_true", help="Initialize ikpush policy play from a raw expert frame.")
    parser.add_argument("--dp_warmstart_raw_episode", type=str, default=None)
    parser.add_argument("--dp_warmstart_step", type=int, default=None)
    parser.add_argument(
        "--dp_warmstart_expert_obs",
        dest="dp_warmstart_expert_obs",
        action="store_true",
        default=True,
        help="Prefill the policy observation buffer from raw expert observations before closed-loop play.",
    )
    parser.add_argument(
        "--no_dp_warmstart_expert_obs",
        dest="dp_warmstart_expert_obs",
        action="store_false",
        help="Warm-start simulator state only; do not prefill the DP observation buffer.",
    )
    parser.add_argument(
        "play_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the underlying camera play script after --.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.checkpoint) == bool(args.expert_action_replay_raw_episode):
        raise ValueError(
            "Specify exactly one action source: --checkpoint or --expert_action_replay_raw_episode."
        )
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
        checkpoint_path = auto_wrap_official_lerobot_checkpoint(checkpoint_path, args)
        replay_path = None
    else:
        checkpoint_path = None
        replay_path = Path(args.expert_action_replay_raw_episode).expanduser()
        if not replay_path.is_absolute():
            replay_path = (Path.cwd() / replay_path).resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"Expert action replay episode not found: {replay_path}")
    if args.steps is None:
        args.steps = 4300 if args.mode == "ikpull" else (2405 if args.mode == "ikpush" else 2500)
    if args.rgb:
        args.depth_only = False
    if args.rgb and args.mode not in ("push", "ikpush", "ikpull"):
        raise ValueError("--rgb Door policy play is only wired for push/ikpush/ikpull mode.")
    if args.expert_action_replay_raw_episode and (args.robot_body != "a2wz1" or args.mode != "ikpush"):
        raise ValueError(
            "--expert_action_replay_raw_episode currently supports only --robot_body a2wz1 --mode ikpush."
        )
    warmstart_params = [
        args.dp_warmstart_raw_episode is not None,
        args.dp_warmstart_step is not None,
        "--dp_warmstart_expert_obs" in sys.argv or "--no_dp_warmstart_expert_obs" in sys.argv,
    ]
    if args.dp_control_all_envs and args.num_envs > 1 and args.dp_warmstart:
        raise ValueError("--dp_warmstart currently supports a single controlled env; add --no_dp_control_all_envs.")
    if not args.dp_warmstart and any(warmstart_params):
        raise ValueError("Warm-start options require --dp_warmstart.")
    if args.dp_warmstart:
        if args.mode != "ikpush":
            raise ValueError("--dp_warmstart is only wired for --mode ikpush.")
        if args.dp_warmstart_raw_episode is None:
            raise ValueError("--dp_warmstart requires --dp_warmstart_raw_episode.")
        if args.dp_warmstart_step is None:
            raise ValueError("--dp_warmstart requires --dp_warmstart_step.")
        if args.dp_warmstart_step < 0:
            raise ValueError("--dp_warmstart_step must be non-negative.")
        warmstart_raw_path = Path(args.dp_warmstart_raw_episode).expanduser()
        if not warmstart_raw_path.is_absolute():
            warmstart_raw_path = (Path.cwd() / warmstart_raw_path).resolve()
    else:
        warmstart_raw_path = None
    if args.robot_body == "a2wz1":
        if args.mode == "ikpush":
            script = HIGH_LEVEL_ROOT / "float_ik" / "isaacgym_float_ik_a2w_basearn_push_door_parallel.py"
        elif args.mode == "ikpull":
            script = HIGH_LEVEL_ROOT / "float_ik" / "isaacgym_float_ik_a2w_basearn_pull_door_parallel.py"
        else:
            raise ValueError("--robot_body a2wz1 supports only --mode ikpush or --mode ikpull.")
    elif args.mode == "ikpush":
        script = HIGH_LEVEL_ROOT / "float_ik" / "isaacgym_float_ik_b1z1_basearn_push_door_parallel.py"
    elif args.mode == "ikpull":
        script = HIGH_LEVEL_ROOT / "float_ik" / "isaacgym_float_ik_b1z1_basearn_pull_door_parallel.py"
    elif args.mode == "pull":
        script = HIGH_LEVEL_ROOT / "play_b1z1_walk_with_door_asset_camera.py"
    else:
        script = HIGH_LEVEL_ROOT / "play_b1z1_push_with_door_asset_camera.py"
    dp_log_path = args.dp_log_path
    if dp_log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dp_log_path = str(
            HIGH_LEVEL_ROOT / "logs" / "door-policy-play" / f"{args.robot_body}_{args.mode}_{timestamp}.jsonl"
        )

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
        "--camera_display_scale",
        str(args.camera_display_scale),
        "--dp_control_env_id",
        str(args.dp_control_env_id),
        "--dp_log_path",
        dp_log_path,
        "--dp_log_interval",
        str(args.dp_log_interval),
        "--no_preview_trajectory_at_spawn",
    ]
    if checkpoint_path is not None:
        cmd += [
            "--dp_policy_checkpoint",
            str(checkpoint_path),
            "--dp_inference_steps",
            str(args.dp_inference_steps),
            "--dp_noise_scheduler_type",
            args.dp_noise_scheduler_type,
        ]
    else:
        cmd += ["--expert_action_replay_raw_episode", str(replay_path)]
    if args.dp_log_replay_snapshot:
        cmd.append("--dp_log_replay_snapshot")
    if args.mode in ("ikpush", "ikpull"):
        cmd.append("--enable_front_camera")
        if not args.debug_visuals:
            cmd += ["--no_draw_ik_target", "--no_draw_camera_axes"]
    elif not args.debug_visuals:
        cmd += ["--no_draw_ee_target", "--no_draw_camera_axes"]
    if args.rgb:
        cmd += ["--rgb", "--camera_rgb", "--no_camera_depth"]
    else:
        cmd.append("--camera_depth")
        if args.depth_only:
            cmd.append("--depth_only")
    if not args.dp_print:
        cmd.append("--no_dp_print")
    if args.dp_control_all_envs:
        cmd.append("--dp_control_all_envs")
    else:
        cmd.append("--no_dp_control_all_envs")
    if args.dp_action_horizon is not None:
        cmd += ["--dp_action_horizon", str(args.dp_action_horizon)]
    if args.dp_temporal_ensemble:
        cmd.append("--dp_temporal_ensemble")
        cmd += ["--dp_temporal_prefetch_actions", str(args.dp_temporal_prefetch_actions)]
        cmd += ["--dp_temporal_old_weight", str(args.dp_temporal_old_weight)]
        cmd += ["--dp_temporal_new_weight", str(args.dp_temporal_new_weight)]
    if args.dp_end_signal_monitor:
        cmd.append("--dp_end_signal_monitor")
        cmd += ["--dp_end_signal_threshold", str(args.dp_end_signal_threshold)]
        cmd += ["--dp_end_signal_consecutive_steps", str(args.dp_end_signal_consecutive_steps)]
    if args.dp_warmstart:
        cmd += [
            "--dp_warmstart",
            "--dp_warmstart_raw_episode",
            str(warmstart_raw_path),
            "--dp_warmstart_step",
            str(args.dp_warmstart_step),
        ]
        if args.dp_warmstart_expert_obs:
            cmd.append("--dp_warmstart_expert_obs")
        else:
            cmd.append("--no_dp_warmstart_expert_obs")
    if args.graphics_device_id is not None:
        cmd += ["--graphics_device_id", str(args.graphics_device_id)]
    if args.headless:
        cmd.append("--headless")
    if not args.show_seg:
        cmd.append("--no_show_seg")
    add_depth_aug_command_args(cmd, args)
    extra = args.play_args[1:] if args.play_args[:1] == ["--"] else args.play_args
    cmd += extra
    source_description = (
        f"policy checkpoint {checkpoint_path}"
        if checkpoint_path is not None
        else f"expert action replay {replay_path}"
    )
    print(f"Running Door control source ({source_description}): {' '.join(cmd)}", flush=True)
    print(
        f"Door policy log will be saved to: {dp_log_path}\n"
        + (
            "All envs are controlled by the selected action source."
            if args.dp_control_all_envs
            else f"Only env {args.dp_control_env_id} is externally controlled; other envs keep scripted targets."
        ),
        flush=True,
    )
    subprocess.run(cmd, cwd=str(HIGH_LEVEL_ROOT), check=True)


if __name__ == "__main__":
    main()
