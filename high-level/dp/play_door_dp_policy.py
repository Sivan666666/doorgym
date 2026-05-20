import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained Door Diffusion Policy in the door asset scene.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["ikpush", "pull", "push"], default="ikpush")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Run a RGB+mask Door DP checkpoint. Push mode only.")
    parser.add_argument("--show_seg", dest="show_seg", action="store_true", default=True)
    parser.add_argument("--no_show_seg", dest="show_seg", action="store_false")
    parser.add_argument("--camera_display_scale", type=int, default=5)
    parser.add_argument("--dp_inference_steps", type=int, default=100)
    parser.add_argument("--dp_action_horizon", type=int, default=None)
    parser.add_argument("--dp_control_env_id", type=int, default=0)
    parser.add_argument("--dp_log_path", type=str, default=None)
    parser.add_argument("--dp_log_interval", type=int, default=25)
    parser.add_argument("--no_dp_print", dest="dp_print", action="store_false", default=True)
    parser.add_argument(
        "play_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the underlying camera play script after --.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "ikpush":
        raise ValueError(
            "--mode ikpush is the float_ik recorder/replay environment. "
            "DP policy execution is not wired in that script yet; use replay for ikpush raw data, "
            "or pass --mode push/--mode pull for the old DP play environments."
        )
    if args.rgb and args.mode != "push":
        raise ValueError("--rgb Door DP policy play is only wired for push mode.")
    script = (
        HIGH_LEVEL_ROOT / "play_b1z1_walk_with_door_asset_camera.py"
        if args.mode == "pull"
        else HIGH_LEVEL_ROOT / "play_b1z1_push_with_door_asset_camera.py"
    )
    dp_log_path = args.dp_log_path
    if dp_log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dp_log_path = str(HIGH_LEVEL_ROOT / "logs" / "door-dp-play" / f"{args.mode}_{timestamp}.jsonl")

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
        args.checkpoint,
        "--dp_inference_steps",
        str(args.dp_inference_steps),
        "--dp_control_env_id",
        str(args.dp_control_env_id),
        "--dp_log_path",
        dp_log_path,
        "--dp_log_interval",
        str(args.dp_log_interval),
        "--no_preview_trajectory_at_spawn",
    ]
    if args.rgb:
        cmd += ["--rgb", "--camera_rgb", "--no_camera_depth"]
    else:
        cmd.append("--camera_depth")
    if not args.dp_print:
        cmd.append("--no_dp_print")
    if args.dp_action_horizon is not None:
        cmd += ["--dp_action_horizon", str(args.dp_action_horizon)]
    if args.graphics_device_id is not None:
        cmd += ["--graphics_device_id", str(args.graphics_device_id)]
    if args.headless:
        cmd.append("--headless")
    if not args.show_seg:
        cmd.append("--no_show_seg")
    extra = args.play_args[1:] if args.play_args[:1] == ["--"] else args.play_args
    cmd += extra
    print(f"Running Door DP policy: {' '.join(cmd)}", flush=True)
    print(
        f"Door DP log will be saved to: {dp_log_path}\n"
        f"Only env {args.dp_control_env_id} is controlled by DP; other envs keep scripted targets.",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(HIGH_LEVEL_ROOT), check=True)


if __name__ == "__main__":
    main()
