import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

DP_ROOT = Path(__file__).resolve().parents[1]
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
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Door policy with LeRobot's ACTPolicy.")
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--run_name", type=str, default="act_debug")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--rgb", action="store_true", help="Train on RGB+mask image fields instead of depth+mask.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Optional LeRobot ACT checkpoint to initialize from.")

    parser.add_argument("--vision_backbone", type=str, default="resnet18")
    parser.add_argument("--pretrained_backbone_weights", type=str, default="ResNet18_Weights.IMAGENET1K_V1")
    parser.add_argument("--no_pretrained_backbone", dest="pretrained_backbone_weights", action="store_const", const=None)
    parser.add_argument("--replace_final_stride_with_dilation", action="store_true")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--dim_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--feedforward_activation", type=str, default="relu")
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_decoder_layers", type=int, default=1)
    parser.add_argument("--no_vae", dest="use_vae", action="store_false", default=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--n_vae_encoder_layers", type=int, default=4)
    parser.add_argument("--temporal_ensemble_coeff", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--visual_normalization", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], default="MEAN_STD")
    parser.add_argument("--state_normalization", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], default="MEAN_STD")
    parser.add_argument("--action_normalization", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], default="MEAN_STD")

    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="door-act")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_save_checkpoints", action="store_true")
    return parser.parse_args()


def select_device(device_arg):
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def load_sidecar(dataset_root):
    sidecar = Path(dataset_root) / "door_dp_feature_names.json"
    if not sidecar.exists():
        return {}, sidecar
    with sidecar.open("r", encoding="utf-8") as f:
        return json.load(f), sidecar


def save_latest(backend, optimizer, ckpt_dir, train_config, wandb_run=None, upload_to_wandb=False):
    latest_dir = ckpt_dir / "model_latest"
    manifest_path = ckpt_dir / "model_latest.pt"
    backend.save_checkpoint(latest_dir, optimizer=optimizer, extra_config=train_config, manifest_path=manifest_path)
    if wandb_run is not None and upload_to_wandb:
        wandb_run.save(str(manifest_path))
        wandb_run.save(str(latest_dir / "door_policy_meta.json"))
        wandb_run.save(str(latest_dir / "door_policy_stats.pt"))
        wandb_run.save(str(latest_dir / "policy" / "config.json"))
        wandb_run.save(str(latest_dir / "policy" / "model.safetensors"))
    return manifest_path


def main():
    args = parse_args()
    device = select_device(args.device)
    dataset_root = _resolve_lerobot_root(args.root, args.repo_id)
    vision_mode = "rgb" if args.rgb else "depth"
    sidecar_data, sidecar_path = load_sidecar(dataset_root)
    if args.rgb and not sidecar_data:
        raise FileNotFoundError(f"RGB training requires {sidecar_path} with vision_mode='rgb'.")
    dataset_vision_mode = normalize_vision_mode(sidecar_data.get("vision_mode", "depth"))
    if dataset_vision_mode != vision_mode:
        raise ValueError(f"Dataset vision_mode={dataset_vision_mode!r}, train expected {vision_mode!r}.")
    action_frame = str(sidecar_data.get("action_frame", sidecar_data.get("action_pose_frame", "world"))).lower()

    dataset = DoorPolicyChunkDataset(dataset_root, args.repo_id, args.chunk_size, vision_mode=vision_mode)
    stats = dataset.stats
    state_dim = _feature_dim_from_stats(stats, OBS_STATE)
    action_dim = _feature_dim_from_stats(stats, ACTION)
    if action_dim != len(ACTION_NAMES):
        raise ValueError(f"Door policy expects 10D actions, but dataset action_dim={action_dim}.")

    normalization_mapping = {
        "VISUAL": args.visual_normalization,
        "STATE": args.state_normalization,
        "ACTION": args.action_normalization,
    }
    backend = LeRobotActDoorPolicyBackend.create(
        stats=stats,
        vision_mode=vision_mode,
        action_frame=action_frame,
        sidecar_config=sidecar_data,
        device=device,
        chunk_size=args.chunk_size,
        action_horizon=args.action_horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        pretrained_path=args.pretrained_path,
        normalization_mapping=normalization_mapping,
        vision_backbone=args.vision_backbone,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
        replace_final_stride_with_dilation=args.replace_final_stride_with_dilation,
        pre_norm=args.pre_norm,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        feedforward_activation=args.feedforward_activation,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        use_vae=args.use_vae,
        latent_dim=args.latent_dim,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        dropout=args.dropout,
        kl_weight=args.kl_weight,
    )
    backend.config.optimizer_lr_backbone = float(args.lr_backbone)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        backend.policy.get_optim_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    run_dir = DP_ROOT / "logs" / "door-act" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    train_config = vars(args).copy()
    train_config.update(
        {
            "backend": BACKEND_LEROBOT_ACT,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "vision_mode": vision_mode,
            "action_frame": action_frame,
            "mode": str(sidecar_data.get("mode", "ikpush")),
            "ikpush_state_version": str(sidecar_data.get("ikpush_state_version", "legacy")),
            "a2wpush_state_version": str(sidecar_data.get("a2wpush_state_version", "legacy")),
            "dataset_root": str(dataset_root),
            "repo_id": args.repo_id,
        }
    )
    for key in (
        "state_dof_names",
        "a2wz1_asset_root",
        "a2wz1_asset_file",
        "a2w_wheel_radius",
        "a2w_track_width",
        "a2w_wheel_velocity_sign",
    ):
        if key in sidecar_data:
            train_config[key] = sidecar_data[key]
    print(
        f"Training backend={BACKEND_LEROBOT_ACT} state_dim={state_dim} action_dim={action_dim} "
        f"chunk_size={args.chunk_size} action_horizon={args.action_horizon} vision_mode={vision_mode}",
        flush=True,
    )

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.run_name,
            config=train_config,
            dir=str(run_dir),
        )

    step = 0
    t0 = time.time()
    latest_manifest = None
    while step < args.steps:
        for batch in loader:
            loss = backend.compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(backend.policy.parameters(), float(args.grad_clip_norm))
            optimizer.step()
            step += 1
            if step % args.log_interval == 0:
                elapsed = time.time() - t0
                print(f"[step {step}] loss={loss.item():.6f} elapsed={elapsed:.1f}s", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/step": step, "train/loss": float(loss.item()), "train/elapsed_sec": elapsed})
            if step % args.save_interval == 0 or step == args.steps:
                train_config["step"] = int(step)
                latest_manifest = save_latest(backend, optimizer, ckpt_dir, train_config, wandb_run, args.wandb_save_checkpoints)
            if step >= args.steps:
                break
    train_config["step"] = int(step)
    latest_manifest = save_latest(backend, optimizer, ckpt_dir, train_config, wandb_run, args.wandb_save_checkpoints)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved {latest_manifest} and {ckpt_dir / 'model_latest'}", flush=True)


if __name__ == "__main__":
    main()
