import argparse
import json
import sys
from pathlib import Path

import torch


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from door_dp_common import ACTION_NAMES, normalize_vision_mode  # noqa: E402
from door_policy_backend import (  # noqa: E402
    ACTION,
    BACKEND_LEROBOT_DIFFUSION,
    OBS_STATE,
    DoorPolicySequenceDataset,
    LeRobotDiffusionDoorPolicyBackend,
    _feature_dim_from_stats,
    _resolve_lerobot_root,
    import_lerobot_policy_modules,
    load_lerobot_policy_normalizer_stats,
    make_lerobot_diffusion_config,
    merge_lerobot_processor_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Wrap an official LeRobot Diffusion checkpoint into the Door policy checkpoint format "
            "used by high-level/dp/play/play_door_policy.py."
        )
    )
    parser.add_argument(
        "--official_checkpoint",
        required=True,
        help="Official LeRobot checkpoint dir, e.g. .../checkpoints/100000 or .../checkpoints/100000/pretrained_model.",
    )
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest_name", type=str, default="model_latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--noise_scheduler_type", choices=["DDPM", "DDIM"], default=None)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_pretrained_model(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.parent.name == "pretrained_model":
        policy_dir = path.parent
        official_step_dir = policy_dir.parent
    elif path.name == "pretrained_model":
        policy_dir = path
        official_step_dir = path.parent
    else:
        official_step_dir = path
        policy_dir = path / "pretrained_model"
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Could not find official pretrained_model directory under: {path}")
    config_path = policy_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Official LeRobot policy config missing: {config_path}")
    return official_step_dir, policy_dir, load_json(config_path)


def load_sidecar(dataset_root):
    sidecar_path = Path(dataset_root) / "door_dp_feature_names.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Door sidecar missing: {sidecar_path}. "
            "This file is required for state/action preprocess alignment."
        )
    return load_json(sidecar_path), sidecar_path


def _config_value(config, key, default=None):
    return config[key] if key in config else default


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    official_step_dir, policy_dir, policy_config = resolve_pretrained_model(args.official_checkpoint)

    if policy_config.get("type") != "diffusion":
        raise ValueError(f"Expected official Diffusion checkpoint, got policy type={policy_config.get('type')!r}")

    dataset_root = _resolve_lerobot_root(args.root, args.repo_id)
    sidecar_data, sidecar_path = load_sidecar(dataset_root)
    if args.rgb and args.depth_only:
        raise ValueError("--rgb and --depth_only are mutually exclusive.")
    vision_mode = "rgb" if args.rgb else ("depth_only" if args.depth_only else "depth")
    dataset_vision_mode = normalize_vision_mode(sidecar_data.get("vision_mode", "depth"))
    if dataset_vision_mode != vision_mode:
        raise ValueError(f"Dataset vision_mode={dataset_vision_mode!r}, export expected {vision_mode!r}.")

    obs_horizon = int(policy_config["n_obs_steps"])
    horizon = int(policy_config["horizon"])
    action_horizon = int(args.action_horizon if args.action_horizon is not None else policy_config["n_action_steps"])
    num_inference_steps = (
        args.num_inference_steps if args.num_inference_steps is not None else policy_config.get("num_inference_steps")
    )
    noise_scheduler_type = args.noise_scheduler_type or policy_config.get("noise_scheduler_type", "DDPM")

    action_frame = str(sidecar_data.get("action_frame", sidecar_data.get("action_pose_frame", "world"))).lower()
    controller_mode = str(sidecar_data.get("door_dp_mode", sidecar_data.get("controller_mode", "legacy")))

    dataset = DoorPolicySequenceDataset(
        dataset_root,
        args.repo_id,
        obs_horizon,
        horizon,
        vision_mode=vision_mode,
    )
    processor_stats = load_lerobot_policy_normalizer_stats(policy_dir)
    stats = merge_lerobot_processor_stats(dataset.stats, processor_stats)
    state_dim = _feature_dim_from_stats(stats, OBS_STATE)
    action_dim = _feature_dim_from_stats(stats, ACTION)
    if action_dim != len(ACTION_NAMES):
        raise ValueError(f"Door policy expects 10D actions, but dataset action_dim={action_dim}.")

    modules = import_lerobot_policy_modules()
    DiffusionPolicy = modules["DiffusionPolicy"]
    image_keys = list(dataset.image_keys)
    config = make_lerobot_diffusion_config(
        state_dim=state_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        horizon=horizon,
        action_horizon=action_horizon,
        image_keys=image_keys,
        device=str(device),
        normalization_mapping=policy_config.get("normalization_mapping"),
        vision_backbone=policy_config.get("vision_backbone", "resnet18"),
        resize_shape=policy_config.get("resize_shape"),
        crop_ratio=float(policy_config.get("crop_ratio", 1.0)),
        crop_is_random=bool(policy_config.get("crop_is_random", True)),
        pretrained_backbone_weights=policy_config.get("pretrained_backbone_weights"),
        use_group_norm=bool(policy_config.get("use_group_norm", True)),
        spatial_softmax_num_keypoints=int(policy_config.get("spatial_softmax_num_keypoints", 32)),
        use_separate_rgb_encoder_per_camera=bool(policy_config.get("use_separate_rgb_encoder_per_camera", False)),
        down_dims=policy_config.get("down_dims", (512, 1024, 2048)),
        kernel_size=int(policy_config.get("kernel_size", 5)),
        n_groups=int(policy_config.get("n_groups", 8)),
        diffusion_step_embed_dim=int(policy_config.get("diffusion_step_embed_dim", 128)),
        use_film_scale_modulation=bool(policy_config.get("use_film_scale_modulation", True)),
        noise_scheduler_type=noise_scheduler_type,
        num_train_timesteps=int(policy_config.get("num_train_timesteps", 100)),
        beta_schedule=policy_config.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=float(policy_config.get("beta_start", 0.0001)),
        beta_end=float(policy_config.get("beta_end", 0.02)),
        prediction_type=policy_config.get("prediction_type", "epsilon"),
        clip_sample=bool(policy_config.get("clip_sample", True)),
        clip_sample_range=float(policy_config.get("clip_sample_range", 1.0)),
        num_inference_steps=None if num_inference_steps is None else int(num_inference_steps),
        compile_model=bool(policy_config.get("compile_model", False)),
        compile_mode=policy_config.get("compile_mode", "reduce-overhead"),
        do_mask_loss_for_padding=bool(policy_config.get("do_mask_loss_for_padding", False)),
    )
    policy = DiffusionPolicy.from_pretrained(
        policy_dir,
        config=config,
        local_files_only=True,
        strict=True,
    ).to(str(device))
    backend = LeRobotDiffusionDoorPolicyBackend(
        policy=policy,
        config=config,
        stats=stats,
        vision_mode=vision_mode,
        action_frame=action_frame,
        sidecar_config=sidecar_data,
        device=device,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = out_dir.parent / args.manifest_name
    train_config = {
        "backend": BACKEND_LEROBOT_DIFFUSION,
        "source_official_checkpoint": str(official_step_dir),
        "source_official_policy_dir": str(policy_dir),
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "vision_mode": vision_mode,
        "action_frame": action_frame,
        "ikpush_state_version": str(sidecar_data.get("ikpush_state_version", "legacy")),
        "door_dp_mode": controller_mode,
        "controller_mode": controller_mode,
        "stats_source": "official_lerobot_preprocessor" if processor_stats else "dataset",
    }
    backend.save_checkpoint(out_dir, optimizer=None, extra_config=train_config, manifest_path=manifest_path)
    print(f"Wrapped official Diffusion checkpoint:\n  source: {policy_dir}\n  door checkpoint: {out_dir}\n  manifest: {manifest_path}")
    print(f"Sidecar aligned from: {sidecar_path}")


if __name__ == "__main__":
    main()
