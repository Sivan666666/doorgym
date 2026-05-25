import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


DP_ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = DP_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained Door LeRobot policy in the door asset scene.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["ikpush", "a2wpush", "pull", "push"], default="ikpush")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Run a RGB+mask Door policy checkpoint. Push/ikpush/a2wpush modes only.")
    parser.add_argument("--show_seg", dest="show_seg", action="store_true", default=True)
    parser.add_argument("--no_show_seg", dest="show_seg", action="store_false")
    parser.add_argument("--camera_display_scale", type=int, default=5)
    parser.add_argument(
        "--debug_visuals",
        action="store_true",
        help="Keep viewer-only debug markers such as EE target spheres and camera axes.",
    )
    parser.add_argument("--dp_inference_steps", type=int, default=10)
    parser.add_argument("--dp_noise_scheduler_type", type=str.upper, choices=["DDIM", "DDPM"], default="DDIM")
    parser.add_argument("--dp_action_horizon", type=int, default=None)
    parser.add_argument("--dp_control_env_id", type=int, default=0)
    parser.add_argument("--dp_control_all_envs", dest="dp_control_all_envs", action="store_true", default=True)
    parser.add_argument("--no_dp_control_all_envs", dest="dp_control_all_envs", action="store_false")
    parser.add_argument("--dp_log_path", type=str, default=None)
    parser.add_argument("--dp_log_interval", type=int, default=25)
    parser.add_argument("--no_dp_print", dest="dp_print", action="store_false", default=True)
    parser.add_argument("--dp_warmstart", action="store_true", help="Initialize ikpush/a2wpush policy play from a raw expert frame.")
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
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if args.rgb and args.mode not in ("push", "ikpush", "a2wpush"):
        raise ValueError("--rgb Door policy play is only wired for push/ikpush/a2wpush mode.")
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
        if args.mode not in ("ikpush", "a2wpush"):
            raise ValueError("--dp_warmstart is only wired for --mode ikpush or --mode a2wpush.")
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
    if args.mode == "ikpush":
        script = HIGH_LEVEL_ROOT / "float_ik" / "isaacgym_float_ik_b1z1_basearn_push_door_parallel.py"
    elif args.mode == "a2wpush":
        script = HIGH_LEVEL_ROOT / "a2w_ik" / "isaacgym_a2w_ik_push_door_parallel.py"
    elif args.mode == "pull":
        script = HIGH_LEVEL_ROOT / "play_b1z1_walk_with_door_asset_camera.py"
    else:
        script = HIGH_LEVEL_ROOT / "play_b1z1_push_with_door_asset_camera.py"
    dp_log_path = args.dp_log_path
    if dp_log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dp_log_path = str(HIGH_LEVEL_ROOT / "logs" / "door-policy-play" / f"{args.mode}_{timestamp}.jsonl")

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
        "--dp_policy_checkpoint",
        str(checkpoint_path),
        "--dp_inference_steps",
        str(args.dp_inference_steps),
        "--dp_noise_scheduler_type",
        args.dp_noise_scheduler_type,
        "--dp_control_env_id",
        str(args.dp_control_env_id),
        "--dp_log_path",
        dp_log_path,
        "--dp_log_interval",
        str(args.dp_log_interval),
        "--no_preview_trajectory_at_spawn",
    ]
    if args.mode in ("ikpush", "a2wpush"):
        cmd.append("--enable_front_camera")
        if not args.debug_visuals:
            cmd += ["--no_draw_ik_target", "--no_draw_camera_axes"]
    elif not args.debug_visuals:
        cmd += ["--no_draw_ee_target", "--no_draw_camera_axes"]
    if args.rgb:
        cmd += ["--rgb", "--camera_rgb", "--no_camera_depth"]
    else:
        cmd.append("--camera_depth")
    if not args.dp_print:
        cmd.append("--no_dp_print")
    if args.dp_control_all_envs:
        cmd.append("--dp_control_all_envs")
    else:
        cmd.append("--no_dp_control_all_envs")
    if args.dp_action_horizon is not None:
        cmd += ["--dp_action_horizon", str(args.dp_action_horizon)]
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
    extra = args.play_args[1:] if args.play_args[:1] == ["--"] else args.play_args
    cmd += extra
    print(f"Running Door policy: {' '.join(cmd)}", flush=True)
    print(
        f"Door policy log will be saved to: {dp_log_path}\n"
        + (
            "All envs are controlled by the learned policy."
            if args.dp_control_all_envs
            else f"Only env {args.dp_control_env_id} is controlled by the learned policy; other envs keep scripted targets."
        ),
        flush=True,
    )
    subprocess.run(cmd, cwd=str(HIGH_LEVEL_ROOT), check=True)


if __name__ == "__main__":
    main()
