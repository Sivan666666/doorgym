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
    BACKEND_LEROBOT_ACT,
    OBS_STATE,
    DoorPolicyChunkDataset,
    LeRobotActDoorPolicyBackend,
    _feature_dim_from_stats,
    _resolve_lerobot_root,
    load_lerobot_policy_normalizer_stats,
    merge_lerobot_processor_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Wrap an official LeRobot ACT checkpoint into the Door policy checkpoint format "
            "used by high-level/dp/play/play_door_policy.py."
        )
    )
    parser.add_argument(
        "--official_checkpoint",
        required=True,
        help="Official LeRobot checkpoint dir, e.g. .../checkpoints/last or .../checkpoints/042000/pretrained_model.",
    )
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest_name", type=str, default="model_latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--action_horizon", type=int, default=None)
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


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    official_step_dir, policy_dir, policy_config = resolve_pretrained_model(args.official_checkpoint)

    if policy_config.get("type") != "act":
        raise ValueError(f"Expected official ACT checkpoint, got policy type={policy_config.get('type')!r}")

    dataset_root = _resolve_lerobot_root(args.root, args.repo_id)
    sidecar_data, sidecar_path = load_sidecar(dataset_root)
    vision_mode = "rgb" if args.rgb else "depth"
    dataset_vision_mode = normalize_vision_mode(sidecar_data.get("vision_mode", "depth"))
    if dataset_vision_mode != vision_mode:
        raise ValueError(f"Dataset vision_mode={dataset_vision_mode!r}, export expected {vision_mode!r}.")

    action_frame = str(sidecar_data.get("action_frame", sidecar_data.get("action_pose_frame", "world"))).lower()
    controller_mode = str(sidecar_data.get("door_dp_mode", sidecar_data.get("controller_mode", "legacy")))
    chunk_size = int(policy_config["chunk_size"])
    action_horizon = int(args.action_horizon if args.action_horizon is not None else policy_config["n_action_steps"])

    dataset = DoorPolicyChunkDataset(dataset_root, args.repo_id, chunk_size, vision_mode=vision_mode)
    processor_stats = load_lerobot_policy_normalizer_stats(policy_dir)
    stats = merge_lerobot_processor_stats(dataset.stats, processor_stats)
    state_dim = _feature_dim_from_stats(stats, OBS_STATE)
    action_dim = _feature_dim_from_stats(stats, ACTION)
    if action_dim != len(ACTION_NAMES):
        raise ValueError(f"Door policy expects 10D actions, but dataset action_dim={action_dim}.")

    backend = LeRobotActDoorPolicyBackend.create(
        stats=stats,
        vision_mode=vision_mode,
        action_frame=action_frame,
        sidecar_config=sidecar_data,
        device=device,
        chunk_size=chunk_size,
        action_horizon=action_horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        pretrained_path=str(policy_dir),
        normalization_mapping=policy_config.get("normalization_mapping"),
        vision_backbone=policy_config.get("vision_backbone", "resnet18"),
        pretrained_backbone_weights=policy_config.get("pretrained_backbone_weights"),
        replace_final_stride_with_dilation=bool(policy_config.get("replace_final_stride_with_dilation", False)),
        freeze_vision_backbone=bool(policy_config.get("freeze_vision_backbone", False)),
        dinov2_image_size=int(policy_config.get("dinov2_image_size", 224)),
        dinov2_feature_grid_size=int(policy_config.get("dinov2_feature_grid_size", 6)),
        dinov2_normalize_inputs=bool(policy_config.get("dinov2_normalize_inputs", True)),
        pre_norm=bool(policy_config.get("pre_norm", False)),
        dim_model=int(policy_config.get("dim_model", 512)),
        n_heads=int(policy_config.get("n_heads", 8)),
        dim_feedforward=int(policy_config.get("dim_feedforward", 3200)),
        feedforward_activation=policy_config.get("feedforward_activation", "relu"),
        n_encoder_layers=int(policy_config.get("n_encoder_layers", 4)),
        n_decoder_layers=int(policy_config.get("n_decoder_layers", 1)),
        use_vae=bool(policy_config.get("use_vae", True)),
        latent_dim=int(policy_config.get("latent_dim", 32)),
        n_vae_encoder_layers=int(policy_config.get("n_vae_encoder_layers", 4)),
        temporal_ensemble_coeff=policy_config.get("temporal_ensemble_coeff"),
        dropout=float(policy_config.get("dropout", 0.1)),
        kl_weight=float(policy_config.get("kl_weight", 10.0)),
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = out_dir.parent / args.manifest_name
    train_config = {
        "backend": BACKEND_LEROBOT_ACT,
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
    print(f"Wrapped official ACT checkpoint:\n  source: {policy_dir}\n  door checkpoint: {out_dir}\n  manifest: {manifest_path}")
    print(f"Sidecar aligned from: {sidecar_path}")


if __name__ == "__main__":
    main()
