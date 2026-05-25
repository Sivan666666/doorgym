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
    BACKEND_LEROBOT_DIFFUSION,
    OBS_STATE,
    DoorPolicySequenceDataset,
    LeRobotDiffusionDoorPolicyBackend,
    _feature_dim_from_stats,
    _resolve_lerobot_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Door policy with LeRobot's built-in DiffusionPolicy.")
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--policy_backend", choices=[BACKEND_LEROBOT_DIFFUSION], default=BACKEND_LEROBOT_DIFFUSION)
    parser.add_argument("--obs_horizon", type=int, default=16)
    parser.add_argument(
        "--pred_horizon",
        type=int,
        default=48,
        help=(
            "LeRobot DP horizon, counted from the first observation frame. "
            "Default 48 with obs_horizon=16 gives 33 current/future action slots."
        ),
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--num_diffusion_iters", type=int, default=100)
    parser.add_argument("--clip_sample_range", type=float, default=1.0)
    parser.add_argument("--rgb", action="store_true", help="Train on RGB+mask image fields instead of depth+mask.")

    parser.add_argument("--vision_backbone", type=str, default="resnet18")
    parser.add_argument("--down_dims", type=str, default="512,1024,2048")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--n_groups", type=int, default=8)
    parser.add_argument("--diffusion_step_embed_dim", type=int, default=128)
    parser.add_argument("--spatial_softmax_num_keypoints", type=int, default=32)
    parser.add_argument("--pretrained_backbone_weights", type=str, default=None)
    parser.add_argument("--use_separate_rgb_encoder_per_camera", action="store_true")
    parser.add_argument("--no_group_norm", dest="use_group_norm", action="store_false", default=True)
    parser.add_argument("--no_film_scale_modulation", dest="use_film_scale_modulation", action="store_false", default=True)
    parser.add_argument("--resize_shape", type=int, nargs=2, default=None)
    parser.add_argument("--crop_ratio", type=float, default=1.0)
    parser.add_argument("--no_random_crop", dest="crop_is_random", action="store_false", default=True)
    parser.add_argument("--noise_scheduler_type", choices=["DDPM", "DDIM"], default="DDPM")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--prediction_type", choices=["epsilon", "sample"], default="epsilon")
    parser.add_argument("--no_clip_sample", dest="clip_sample", action="store_false", default=True)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--do_mask_loss_for_padding", action="store_true")
    parser.add_argument("--visual_normalization", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], default="MEAN_STD")
    parser.add_argument("--state_normalization", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], default="MIN_MAX")
    parser.add_argument("--action_normalization", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], default="MIN_MAX")

    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="door-dp")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_save_checkpoints", action="store_true", help="Upload checkpoint files to wandb.")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


def parse_down_dims(value):
    dims = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not dims:
        raise ValueError("--down_dims must contain at least one integer.")
    return tuple(dims)


def load_sidecar(dataset_root):
    sidecar = Path(dataset_root) / "door_dp_feature_names.json"
    if not sidecar.exists():
        return {}, sidecar
    with sidecar.open("r", encoding="utf-8") as f:
        return json.load(f), sidecar


def select_device(device_arg):
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def save_latest(backend, optimizer, ckpt_dir, train_config, wandb_run=None, upload_to_wandb=False):
    latest_dir = ckpt_dir / "model_latest"
    manifest_path = ckpt_dir / "model_latest.pt"
    backend.save_checkpoint(
        latest_dir,
        optimizer=optimizer,
        extra_config=train_config,
        manifest_path=manifest_path,
    )
    if wandb_run is not None and upload_to_wandb:
        wandb_run.save(str(manifest_path))
        wandb_run.save(str(latest_dir / "door_policy_meta.json"))
        wandb_run.save(str(latest_dir / "door_policy_stats.pt"))
        wandb_run.save(str(latest_dir / "policy" / "config.json"))
        wandb_run.save(str(latest_dir / "policy" / "model.safetensors"))
    return manifest_path


def main():
    args = parse_args()
    if args.policy_backend != BACKEND_LEROBOT_DIFFUSION:
        raise ValueError(f"Unsupported policy backend: {args.policy_backend}")

    device = select_device(args.device)
    dataset_root = _resolve_lerobot_root(args.root, args.repo_id)
    vision_mode = "rgb" if args.rgb else "depth"
    print(f"Using LeRobot dataset root: {dataset_root}", flush=True)

    sidecar_data, sidecar_path = load_sidecar(dataset_root)
    if args.rgb and not sidecar_data:
        raise FileNotFoundError(f"RGB training requires {sidecar_path} with vision_mode='rgb'.")
    dataset_vision_mode = normalize_vision_mode(sidecar_data.get("vision_mode", "depth"))
    if dataset_vision_mode != vision_mode:
        raise ValueError(
            f"LeRobot dataset vision_mode={dataset_vision_mode!r}, but train was run with "
            f"{'--rgb' if args.rgb else 'depth mode'}."
        )
    action_frame = str(sidecar_data.get("action_frame", sidecar_data.get("action_pose_frame", "world"))).lower()
    if action_frame not in ("world", "base"):
        raise ValueError(f"LeRobot dataset action_frame={action_frame!r}; expected 'world' or 'base'.")

    dataset = DoorPolicySequenceDataset(
        dataset_root,
        args.repo_id,
        args.obs_horizon,
        args.pred_horizon,
        vision_mode=vision_mode,
    )
    stats = dataset.stats
    state_dim = _feature_dim_from_stats(stats, OBS_STATE)
    action_dim = _feature_dim_from_stats(stats, ACTION)
    if action_dim != len(ACTION_NAMES):
        raise ValueError(f"Door DP expects 10D actions, but dataset action_dim={action_dim}.")
    print(
        f"Training backend={args.policy_backend} state_dim={state_dim} action_dim={action_dim} "
        f"obs_horizon={args.obs_horizon} horizon={args.pred_horizon} action_horizon={args.action_horizon} "
        f"vision_mode={vision_mode}",
        flush=True,
    )

    normalization_mapping = {
        "VISUAL": args.visual_normalization,
        "STATE": args.state_normalization,
        "ACTION": args.action_normalization,
    }
    backend = LeRobotDiffusionDoorPolicyBackend.create(
        stats=stats,
        vision_mode=vision_mode,
        action_frame=action_frame,
        sidecar_config=sidecar_data,
        device=device,
        obs_horizon=args.obs_horizon,
        horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        normalization_mapping=normalization_mapping,
        vision_backbone=args.vision_backbone,
        resize_shape=args.resize_shape,
        crop_ratio=args.crop_ratio,
        crop_is_random=args.crop_is_random,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
        use_group_norm=args.use_group_norm,
        spatial_softmax_num_keypoints=args.spatial_softmax_num_keypoints,
        use_separate_rgb_encoder_per_camera=args.use_separate_rgb_encoder_per_camera,
        down_dims=parse_down_dims(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        use_film_scale_modulation=args.use_film_scale_modulation,
        noise_scheduler_type=args.noise_scheduler_type,
        num_train_timesteps=args.num_diffusion_iters,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
        num_inference_steps=args.num_inference_steps,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        do_mask_loss_for_padding=args.do_mask_loss_for_padding,
    )

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
    run_dir = DP_ROOT / "logs" / "door-dp" / args.run_name
    ckpt_dir = run_dir / "checkpoints"

    train_config = vars(args).copy()
    train_config.update(
        {
            "backend": args.policy_backend,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "vision_mode": vision_mode,
            "action_frame": action_frame,
            "action_pose_frame": action_frame,
            "target_pose_frame": action_frame,
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

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("Install wandb in the training env, e.g. `pip install wandb`.") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.run_name,
            config=train_config,
            dir=str(run_dir),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

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
                    wandb_run.log(
                        {
                            "train/step": step,
                            "train/loss": float(loss.item()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/elapsed_sec": float(elapsed),
                        }
                    )

            if step % args.save_interval == 0 or step == args.steps:
                train_config["step"] = int(step)
                latest_manifest = save_latest(
                    backend,
                    optimizer,
                    ckpt_dir,
                    train_config,
                    wandb_run=wandb_run,
                    upload_to_wandb=args.wandb_save_checkpoints,
                )

            if step >= args.steps:
                break

    train_config["step"] = int(step)
    latest_manifest = save_latest(
        backend,
        optimizer,
        ckpt_dir,
        train_config,
        wandb_run=wandb_run,
        upload_to_wandb=args.wandb_save_checkpoints,
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved {latest_manifest} and {ckpt_dir / 'model_latest'}", flush=True)


if __name__ == "__main__":
    main()
