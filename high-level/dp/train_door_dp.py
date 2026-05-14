import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .door_dp_common import ACTION_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH, import_lerobot_or_raise
    from .models.door_diffusion_policy import DoorDiffusionPolicy
except ImportError:
    from door_dp_common import ACTION_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH, import_lerobot_or_raise
    from models.door_diffusion_policy import DoorDiffusionPolicy


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a wrist-mask/depth conditioned Door Diffusion Policy.")
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--obs_horizon", type=int, default=16)
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--num_diffusion_iters", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="door-dp")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_save_checkpoints", action="store_true", help="Upload checkpoint files to wandb.")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


def _load_lerobot_dataset(root, repo_id):
    import_lerobot_or_raise()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = _resolve_lerobot_root(root, repo_id)
    try:
        return LeRobotDataset(repo_id=repo_id, root=str(root))
    except TypeError:
        return LeRobotDataset(repo_id, root=str(root))


def _resolve_lerobot_root(root, repo_id):
    root = Path(root).expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return root
    nested = root / repo_id
    if (nested / "meta" / "info.json").is_file():
        return nested
    return root


def _field(frame, key):
    if key in frame:
        return frame[key]
    if hasattr(frame, "__getitem__"):
        return frame[key]
    raise KeyError(key)


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    try:
        from PIL import Image

        if isinstance(x, Image.Image):
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)


def _as_chw_u8(x):
    arr = _as_numpy(x)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        chw = arr
    else:
        chw = np.transpose(arr[..., :3], (2, 0, 1))
    if chw.shape[-2:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        import cv2

        hwc = np.transpose(chw, (1, 2, 0))
        hwc = cv2.resize(hwc, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        chw = np.transpose(hwc, (2, 0, 1))
    return torch.from_numpy(chw.astype(np.uint8))


def _image_or_zeros(frame, key):
    try:
        return _as_chw_u8(_field(frame, key))
    except Exception:
        return torch.zeros(3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.uint8)


def _scalar_int(x, default=0):
    if x is None:
        return int(default)
    arr = _as_numpy(x)
    return int(arr.reshape(-1)[0])


class DoorDPSequenceDataset(Dataset):
    def __init__(self, root, repo_id, obs_horizon, pred_horizon):
        self.dataset = _load_lerobot_dataset(root, repo_id)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.length = len(self.dataset)
        if self.length < self.obs_horizon + self.pred_horizon:
            raise ValueError("Dataset is too short for the requested horizons.")
        self.episodes = []
        for i in range(self.length):
            frame = self.dataset[i]
            try:
                ep = _scalar_int(_field(frame, "episode_index"))
            except Exception:
                ep = 0
            self.episodes.append(ep)
        self.indices = self._build_indices()

    def _build_indices(self):
        indices = []
        start = 0
        while start < self.length:
            ep = self.episodes[start]
            end = start + 1
            while end < self.length and self.episodes[end] == ep:
                end += 1
            first = start + self.obs_horizon - 1
            last = end - self.pred_horizon
            indices.extend(range(first, max(last + 1, first)))
            start = end
        if not indices:
            raise ValueError("No valid train sequences found. Record longer episodes or reduce horizons.")
        return indices

    def _frame(self, idx):
        return self.dataset[int(idx)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        center = self.indices[item]
        obs_ids = range(center - self.obs_horizon + 1, center + 1)
        action_ids = range(center, center + self.pred_horizon)
        states, masks, depths, front_masks, front_depths = [], [], [], [], []
        for idx in obs_ids:
            frame = self._frame(idx)
            states.append(torch.as_tensor(_as_numpy(_field(frame, "observation.state")), dtype=torch.float32))
            masks.append(_image_or_zeros(frame, "observation.images.wrist_handle_mask"))
            depths.append(_image_or_zeros(frame, "observation.images.wrist_masked_depth"))
            front_masks.append(_image_or_zeros(frame, "observation.images.front_handle_mask"))
            front_depths.append(_image_or_zeros(frame, "observation.images.front_masked_depth"))
        actions = []
        for idx in action_ids:
            frame = self._frame(idx)
            actions.append(torch.as_tensor(_as_numpy(_field(frame, "action")), dtype=torch.float32))
        return {
            "state": torch.stack(states, dim=0),
            "mask": torch.stack(masks, dim=0),
            "masked_depth": torch.stack(depths, dim=0),
            "front_mask": torch.stack(front_masks, dim=0),
            "front_masked_depth": torch.stack(front_depths, dim=0),
            "action": torch.stack(actions, dim=0),
        }


def compute_stats(seq_dataset):
    base = seq_dataset.dataset
    states, actions = [], []
    for i in range(len(base)):
        frame = base[i]
        states.append(torch.as_tensor(_as_numpy(_field(frame, "observation.state")), dtype=torch.float32))
        actions.append(torch.as_tensor(_as_numpy(_field(frame, "action")), dtype=torch.float32))
    states = torch.stack(states, dim=0)
    actions = torch.stack(actions, dim=0)
    return {
        "state_mean": states.mean(dim=0),
        "state_std": states.std(dim=0).clamp_min(1e-6),
        "action_mean": actions.mean(dim=0),
        "action_std": actions.std(dim=0).clamp_min(1e-6),
    }


def normalize_batch(batch, stats, device):
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    state = (state - stats["state_mean"].to(device)) / stats["state_std"].to(device)
    action = (action - stats["action_mean"].to(device)) / stats["action_std"].to(device)
    return (
        state,
        batch["mask"].to(device),
        batch["masked_depth"].to(device),
        batch["front_mask"].to(device),
        batch["front_masked_depth"].to(device),
        action,
    )


def save_checkpoint(path, model, optimizer, stats, config, feature_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": {k: v.cpu() for k, v in stats.items()},
            "config": config,
            "feature_names": feature_names,
            "action_names": ACTION_NAMES,
        },
        path,
    )


def main():
    args = parse_args()
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dataset_root = _resolve_lerobot_root(args.root, args.repo_id)
    print(f"Using LeRobot dataset root: {dataset_root}", flush=True)
    dataset = DoorDPSequenceDataset(dataset_root, args.repo_id, args.obs_horizon, args.pred_horizon)
    stats = compute_stats(dataset)
    sidecar = dataset_root / "door_dp_feature_names.json"
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as f:
            feature_names = json.load(f).get("state", [])
    else:
        feature_names = [f"state_{i}" for i in range(stats["state_mean"].numel())]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = DoorDiffusionPolicy(
        state_dim=stats["state_mean"].numel(),
        action_dim=len(ACTION_NAMES),
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
    ).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    run_dir = Path(__file__).resolve().parent / "logs" / "door-dp" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    config = vars(args).copy()
    config.update(
        {
            "state_dim": stats["state_mean"].numel(),
            "action_dim": len(ACTION_NAMES),
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
        }
    )
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
            config=config,
            dir=str(run_dir),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

    step = 0
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            state, mask, masked_depth, front_mask, front_masked_depth, action = normalize_batch(batch, stats, device)
            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (action.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
            pred = model(noisy_action, timesteps, state, mask, masked_depth, front_mask, front_masked_depth)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                save_checkpoint(ckpt_dir / "model_latest.pt", model, optimizer, stats, config, feature_names)
                if wandb_run is not None and args.wandb_save_checkpoints:
                    wandb_run.save(str(ckpt_dir / "model_latest.pt"))
            if step >= args.steps:
                break
    save_checkpoint(ckpt_dir / "model_latest.pt", model, optimizer, stats, config, feature_names)
    if wandb_run is not None and args.wandb_save_checkpoints:
        wandb_run.save(str(ckpt_dir / "model_latest.pt"))
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved {ckpt_dir / 'model_latest.pt'}")


if __name__ == "__main__":
    main()
