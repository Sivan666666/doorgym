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
    BACKEND_LEROBOT_PI05,
    OBS_STATE,
    DoorPolicyChunkDataset,
    LeRobotPI05DoorPolicyBackend,
    _feature_dim_from_stats,
    _resolve_lerobot_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Door policy with LeRobot's PI05Policy.")
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--run_name", type=str, default="pi05_debug")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--rgb", action="store_true", help="Train on RGB+mask image fields instead of depth+mask.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Optional LeRobot pi0.5/OpenPI checkpoint.")

    parser.add_argument("--task_prompt", type=str, default="open the door")
    parser.add_argument("--tokenizer_name", type=str, default="google/paligemma-3b-pt-224")
    parser.add_argument("--paligemma_variant", choices=["gemma_300m", "gemma_2b"], default="gemma_2b")
    parser.add_argument("--action_expert_variant", choices=["gemma_300m", "gemma_2b"], default="gemma_300m")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--max_state_dim", type=int, default=128)
    parser.add_argument("--max_action_dim", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--image_resolution", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--tokenizer_max_length", type=int, default=200)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="max-autotune")
    parser.add_argument("--freeze_vision_encoder", action="store_true")
    parser.add_argument("--train_expert_only", action="store_true")
    parser.add_argument("--visual_normalization", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], default="IDENTITY")
    parser.add_argument("--state_normalization", choices=["QUANTILES", "QUANTILE10", "MEAN_STD", "MIN_MAX", "IDENTITY"], default="QUANTILES")
    parser.add_argument("--action_normalization", choices=["QUANTILES", "QUANTILE10", "MEAN_STD", "MIN_MAX", "IDENTITY"], default="QUANTILES")

    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="door-pi05")
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
    backend = LeRobotPI05DoorPolicyBackend.create(
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
        task_prompt=args.task_prompt,
        tokenizer_name=args.tokenizer_name,
        normalization_mapping=normalization_mapping,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        dtype=args.dtype,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        num_inference_steps=args.num_inference_steps,
        image_resolution=args.image_resolution,
        tokenizer_max_length=args.tokenizer_max_length,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
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
    run_dir = DP_ROOT / "logs" / "door-pi05" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    train_config = vars(args).copy()
    train_config.update(
        {
            "backend": BACKEND_LEROBOT_PI05,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "vision_mode": vision_mode,
            "action_frame": action_frame,
            "ikpush_state_version": str(sidecar_data.get("ikpush_state_version", "legacy")),
            "dataset_root": str(dataset_root),
            "repo_id": args.repo_id,
        }
    )
    print(
        f"Training backend={BACKEND_LEROBOT_PI05} state_dim={state_dim} action_dim={action_dim} "
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
