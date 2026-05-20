import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .door_dp_common import (
        ACTION_NAMES,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        import_lerobot_or_raise,
        lerobot_image_keys_for_vision_mode,
        normalize_vision_mode,
    )
    from .models.door_diffusion_policy import DoorDiffusionPolicy
except ImportError:
    from door_dp_common import (
        ACTION_NAMES,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        import_lerobot_or_raise,
        lerobot_image_keys_for_vision_mode,
        normalize_vision_mode,
    )
    from models.door_diffusion_policy import DoorDiffusionPolicy


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a wrist-mask/depth conditioned Door Diffusion Policy.")
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--obs_horizon", type=int, default=16)
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--num_diffusion_iters", type=int, default=100)
    parser.add_argument("--clip_sample_range", type=float, default=7.0)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--rgb", action="store_true", help="Train on RGB+mask image fields instead of masked depth+mask.")
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


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _image_required(frame, key):
    return _as_chw_u8(_field(frame, key))


def _scalar_int(x, default=0):
    if x is None:
        return int(default)
    arr = _as_numpy(x)
    return int(arr.reshape(-1)[0])


def _to_1d_int_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.int64).reshape(-1)
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _episode_ranges_from_index(index, total_length):
    if not isinstance(index, dict):
        return None
    start = index.get("from", index.get("start", index.get("starts")))
    end = index.get("to", index.get("end", index.get("ends")))
    if start is None or end is None:
        return None
    starts = _to_1d_int_array(start)
    ends = _to_1d_int_array(end)
    if len(starts) != len(ends) or len(starts) == 0:
        return None
    ranges = [(int(s), int(e)) for s, e in zip(starts, ends) if 0 <= int(s) < int(e) <= total_length]
    return ranges or None


def _episode_ranges_from_dataset_attrs(dataset, total_length):
    candidates = [getattr(dataset, "episode_data_index", None)]
    meta = getattr(dataset, "meta", None)
    if meta is not None:
        candidates.append(getattr(meta, "episode_data_index", None))
    for candidate in candidates:
        ranges = _episode_ranges_from_index(candidate, total_length)
        if ranges:
            return ranges
    return None


def _read_arrow_column(table, name):
    if name not in table.column_names:
        return None
    return table[name].to_pylist()


def _episode_ranges_from_metadata(dataset_root, total_length):
    episode_files = sorted((Path(dataset_root) / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        return None
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    rows = []
    for path in episode_files:
        table = pq.read_table(path)
        names = set(table.column_names)
        episode_index = _read_arrow_column(table, "episode_index") or list(range(table.num_rows))
        lengths = None
        for key in ("length", "episode_length", "num_frames", "frame_count"):
            values = _read_arrow_column(table, key)
            if values is not None:
                lengths = [int(v) for v in values]
                break
        starts = ends = None
        for start_key, end_key in (
            ("from", "to"),
            ("start", "end"),
            ("frame_index_from", "frame_index_to"),
            ("start_frame", "end_frame"),
        ):
            if start_key in names and end_key in names:
                starts = [int(v) for v in _read_arrow_column(table, start_key)]
                ends = [int(v) for v in _read_arrow_column(table, end_key)]
                break
        for row_idx, ep in enumerate(episode_index):
            row = {"episode_index": int(ep)}
            if lengths is not None:
                row["length"] = lengths[row_idx]
            if starts is not None and ends is not None:
                row["start"] = starts[row_idx]
                row["end"] = ends[row_idx]
            rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda item: item["episode_index"])
    if all("start" in row and "end" in row for row in rows):
        ranges = [(row["start"], row["end"]) for row in rows]
    elif all("length" in row for row in rows):
        ranges = []
        start = 0
        for row in rows:
            end = start + int(row["length"])
            ranges.append((start, end))
            start = end
    else:
        return None
    if ranges and ranges[-1][1] == total_length and all(0 <= s < e <= total_length for s, e in ranges):
        return [(int(s), int(e)) for s, e in ranges]
    return None


def _episode_ranges_by_frame_scan(dataset, total_length):
    episodes = []
    for i in range(total_length):
        frame = dataset[i]
        try:
            ep = _scalar_int(_field(frame, "episode_index"))
        except Exception:
            ep = 0
        episodes.append(ep)
    ranges = []
    start = 0
    while start < total_length:
        ep = episodes[start]
        end = start + 1
        while end < total_length and episodes[end] == ep:
            end += 1
        ranges.append((start, end))
        start = end
    return ranges


def _load_episode_ranges(dataset_root, dataset, total_length):
    for loader_name, loader in (
        ("metadata", lambda: _episode_ranges_from_metadata(dataset_root, total_length)),
        ("dataset index", lambda: _episode_ranges_from_dataset_attrs(dataset, total_length)),
    ):
        try:
            ranges = loader()
        except Exception as exc:
            print(f"Warning: failed to build episode ranges from {loader_name}: {exc}", flush=True)
            ranges = None
        if ranges:
            print(f"Loaded {len(ranges)} episode ranges from {loader_name}; skipped full image frame scan.", flush=True)
            return ranges
    print("Warning: falling back to full dataset frame scan to build episode ranges.", flush=True)
    return _episode_ranges_by_frame_scan(dataset, total_length)


def _lerobot_data_files(dataset_root):
    return sorted((Path(dataset_root) / "data").glob("**/*.parquet"))


def _arrow_column_to_2d_numpy(table, name):
    if name not in table.column_names:
        raise KeyError(f"Parquet table is missing column {name!r}")
    values = table[name].to_pylist()
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _compute_stats_from_parquet_columns(dataset_root):
    files = _lerobot_data_files(dataset_root)
    if not files:
        return None
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    count = 0
    state_sum = action_sum = state_sumsq = action_sumsq = None
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = _arrow_column_to_2d_numpy(table, "observation.state")
        actions = _arrow_column_to_2d_numpy(table, "action")
        if states.shape[0] != actions.shape[0]:
            raise ValueError(f"{path} has mismatched state/action rows.")
        if state_sum is None:
            state_sum = np.zeros(states.shape[1], dtype=np.float64)
            action_sum = np.zeros(actions.shape[1], dtype=np.float64)
            state_sumsq = np.zeros(states.shape[1], dtype=np.float64)
            action_sumsq = np.zeros(actions.shape[1], dtype=np.float64)
        count += int(states.shape[0])
        state_sum += states.sum(axis=0)
        action_sum += actions.sum(axis=0)
        state_sumsq += np.square(states).sum(axis=0)
        action_sumsq += np.square(actions).sum(axis=0)
    if count <= 0:
        return None

    def finish(total, total_sq):
        mean = total / float(count)
        if count > 1:
            var = (total_sq - np.square(total) / float(count)) / float(count - 1)
        else:
            var = np.zeros_like(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        return torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)

    state_mean, state_std = finish(state_sum, state_sumsq)
    action_mean, action_std = finish(action_sum, action_sumsq)
    print(f"Computed state/action stats from parquet columns only: frames={count}", flush=True)
    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def _compute_stats_from_stats_json(dataset_root):
    path = Path(dataset_root) / "meta" / "stats.json"
    if not path.is_file():
        return None
    data = _read_json(path)
    try:
        state = data["observation.state"]
        action = data["action"]
        print("Loaded state/action stats from meta/stats.json; skipped full image frame scan.", flush=True)
        return {
            "state_mean": torch.as_tensor(state["mean"], dtype=torch.float32),
            "state_std": torch.as_tensor(state["std"], dtype=torch.float32).clamp_min(1e-6),
            "action_mean": torch.as_tensor(action["mean"], dtype=torch.float32),
            "action_std": torch.as_tensor(action["std"], dtype=torch.float32).clamp_min(1e-6),
        }
    except Exception:
        return None


class DoorDPSequenceDataset(Dataset):
    def __init__(self, root, repo_id, obs_horizon, pred_horizon, vision_mode="depth"):
        self.dataset_root = _resolve_lerobot_root(root, repo_id)
        self.dataset = _load_lerobot_dataset(self.dataset_root, repo_id)
        self.vision_mode = normalize_vision_mode(vision_mode)
        self.image_keys = lerobot_image_keys_for_vision_mode(self.vision_mode)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.length = len(self.dataset)
        if self.length < self.obs_horizon + self.pred_horizon:
            raise ValueError("Dataset is too short for the requested horizons.")
        self.episode_ranges = _load_episode_ranges(self.dataset_root, self.dataset, self.length)
        self.indices = self._build_indices()

    def _build_indices(self):
        indices = []
        for start, end in self.episode_ranges:
            first = start + self.obs_horizon - 1
            last = end - self.pred_horizon
            indices.extend(range(first, max(last + 1, first)))
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
        states, masks, second_images, front_masks, front_second_images = [], [], [], [], []
        image_reader = _image_required if self.vision_mode == "rgb" else _image_or_zeros
        for idx in obs_ids:
            frame = self._frame(idx)
            states.append(torch.as_tensor(_as_numpy(_field(frame, "observation.state")), dtype=torch.float32))
            masks.append(image_reader(frame, self.image_keys[0]))
            second_images.append(image_reader(frame, self.image_keys[1]))
            front_masks.append(image_reader(frame, self.image_keys[2]))
            front_second_images.append(image_reader(frame, self.image_keys[3]))
        actions = []
        for idx in action_ids:
            frame = self._frame(idx)
            actions.append(torch.as_tensor(_as_numpy(_field(frame, "action")), dtype=torch.float32))
        return {
            "state": torch.stack(states, dim=0),
            "mask": torch.stack(masks, dim=0),
            "masked_depth": torch.stack(second_images, dim=0),
            "front_mask": torch.stack(front_masks, dim=0),
            "front_masked_depth": torch.stack(front_second_images, dim=0),
            "action": torch.stack(actions, dim=0),
        }


def compute_stats(seq_dataset):
    for loader in (
        lambda: _compute_stats_from_parquet_columns(seq_dataset.dataset_root),
        lambda: _compute_stats_from_stats_json(seq_dataset.dataset_root),
    ):
        try:
            stats = loader()
        except Exception as exc:
            print(f"Warning: optimized stats loading failed: {exc}", flush=True)
            stats = None
        if stats is not None:
            return stats

    print("Warning: falling back to full dataset frame scan to compute stats.", flush=True)
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
    vision_mode = "rgb" if args.rgb else "depth"
    print(f"Using LeRobot dataset root: {dataset_root}", flush=True)
    sidecar = dataset_root / "door_dp_feature_names.json"
    sidecar_data = {}
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as f:
            sidecar_data = json.load(f)
    if args.rgb and not sidecar_data:
        raise FileNotFoundError(f"RGB training requires {sidecar} with vision_mode='rgb'.")
    dataset_vision_mode = normalize_vision_mode(sidecar_data.get("vision_mode", "depth"))
    if dataset_vision_mode != vision_mode:
        raise ValueError(
            f"LeRobot dataset vision_mode={dataset_vision_mode!r}, but train was run with "
            f"{'--rgb' if args.rgb else 'depth mode'}."
        )
    dataset = DoorDPSequenceDataset(dataset_root, args.repo_id, args.obs_horizon, args.pred_horizon, vision_mode=vision_mode)
    stats = compute_stats(dataset)
    if sidecar_data:
        feature_names = sidecar_data.get("state", [])
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
        clip_sample_range=float(args.clip_sample_range),
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
            "vision_mode": vision_mode,
            "image_features": lerobot_image_keys_for_vision_mode(vision_mode),
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
