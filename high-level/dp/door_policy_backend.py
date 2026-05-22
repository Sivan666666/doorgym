"""Door policy backend abstraction.

This module keeps the Door DP input/output contract stable:

  observation.state [73] + four image streams -> action [10]

The current implementation uses LeRobot's built-in DiffusionPolicy.  The
wrapper is deliberately narrow so ACT / pi0.5 can later plug into the same
controller and train/eval scripts without touching Isaac Gym play code.
"""

from __future__ import annotations

import json
import os
import pickle
import shlex
import shutil
import struct
import subprocess
import sys
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .door_dp_common import (
        ACTION_NAMES,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        lerobot_image_keys_for_vision_mode,
        normalize_vision_mode,
    )
except ImportError:
    from door_dp_common import (
        ACTION_NAMES,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        lerobot_image_keys_for_vision_mode,
        normalize_vision_mode,
    )


OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
ACTION = "action"
ACTION_IS_PAD = "action_is_pad"
BACKEND_LEROBOT_DIFFUSION = "lerobot_diffusion"
BACKEND_LEROBOT_ACT = "lerobot_act"
BACKEND_LEROBOT_PI05 = "lerobot_pi05"
CHECKPOINT_META = "door_policy_meta.json"
CHECKPOINT_STATS = "door_policy_stats.pt"
CHECKPOINT_POLICY_DIR = "policy"
CHECKPOINT_OPTIMIZER = "optimizer.pt"


def _ensure_hf_cache_env() -> None:
    """Keep local LeRobot/HF cache writes inside writable temp dirs by default."""

    os.environ.setdefault("HF_HOME", "/tmp/hf-door-dp")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-door-dp/datasets")


def import_lerobot_policy_modules():
    _ensure_hf_cache_env()
    try:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except Exception as exc:
        raise RuntimeError(
            "LeRobot DiffusionPolicy is required for Door DP. Use a Python>=3.10 environment "
            "with `pip install -r high-level/dp/requirements_dp.txt`. The old custom DoorDiffusionPolicy "
            "runtime is no longer used by train/play/eval."
        ) from exc
    return {
        "FeatureType": FeatureType,
        "NormalizationMode": NormalizationMode,
        "PolicyFeature": PolicyFeature,
        "LeRobotDataset": LeRobotDataset,
        "DiffusionConfig": DiffusionConfig,
        "DiffusionPolicy": DiffusionPolicy,
    }


def import_lerobot_act_modules():
    _ensure_hf_cache_env()
    try:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except Exception as exc:
        raise RuntimeError(
            "LeRobot ACTPolicy is required for Door ACT. Use a Python>=3.10 environment "
            "with LeRobot and torchvision/transformers dependencies installed."
        ) from exc
    return {
        "FeatureType": FeatureType,
        "NormalizationMode": NormalizationMode,
        "PolicyFeature": PolicyFeature,
        "LeRobotDataset": LeRobotDataset,
        "ACTConfig": ACTConfig,
        "ACTPolicy": ACTPolicy,
    }


def import_lerobot_pi05_modules():
    _ensure_hf_cache_env()
    try:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    except Exception as exc:
        raise RuntimeError(
            "LeRobot PI05Policy is required for Door pi0.5. Use a Python>=3.10 environment "
            "with LeRobot transformers dependencies installed."
        ) from exc
    return {
        "FeatureType": FeatureType,
        "NormalizationMode": NormalizationMode,
        "PolicyFeature": PolicyFeature,
        "LeRobotDataset": LeRobotDataset,
        "PI05Config": PI05Config,
        "PI05Policy": PI05Policy,
        "OBS_LANGUAGE_TOKENS": OBS_LANGUAGE_TOKENS,
        "OBS_LANGUAGE_ATTENTION_MASK": OBS_LANGUAGE_ATTENTION_MASK,
    }


def _resolve_lerobot_root(root: Union[str, Path], repo_id: str) -> Path:
    root = Path(root).expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return root
    nested = root / repo_id
    if (nested / "meta" / "info.json").is_file():
        return nested
    return root


def _load_lerobot_dataset(root: Union[str, Path], repo_id: str):
    modules = import_lerobot_policy_modules()
    dataset_root = _resolve_lerobot_root(root, repo_id)
    LeRobotDataset = modules["LeRobotDataset"]
    try:
        return LeRobotDataset(repo_id=repo_id, root=str(dataset_root))
    except TypeError:
        return LeRobotDataset(repo_id, root=str(dataset_root))


def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _field(frame: Mapping[str, Any], key: str) -> Any:
    if key in frame:
        return frame[key]
    if hasattr(frame, "__getitem__"):
        return frame[key]
    raise KeyError(key)


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    try:
        from PIL import Image

        if isinstance(x, Image.Image):
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)


def _to_1d_int_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.int64).reshape(-1)
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _scalar_int(x: Any, default: int = 0) -> int:
    if x is None:
        return int(default)
    arr = _as_numpy(x)
    return int(arr.reshape(-1)[0])


def _episode_ranges_from_index(index: Any, total_length: int) -> Optional[List[Tuple[int, int]]]:
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


def _episode_ranges_from_dataset_attrs(dataset: Any, total_length: int) -> Optional[List[Tuple[int, int]]]:
    candidates = [getattr(dataset, "episode_data_index", None)]
    meta = getattr(dataset, "meta", None)
    if meta is not None:
        candidates.append(getattr(meta, "episode_data_index", None))
    for candidate in candidates:
        ranges = _episode_ranges_from_index(candidate, total_length)
        if ranges:
            return ranges
    return None


def _read_arrow_column(table: Any, name: str) -> Optional[List[Any]]:
    if name not in table.column_names:
        return None
    return table[name].to_pylist()


def _episode_ranges_from_metadata(dataset_root: Union[str, Path], total_length: int) -> Optional[List[Tuple[int, int]]]:
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


def _episode_ranges_by_frame_scan(dataset: Any, total_length: int) -> List[Tuple[int, int]]:
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


def load_episode_ranges(dataset_root: Union[str, Path], dataset: Any, total_length: int) -> List[Tuple[int, int]]:
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
            print(f"Loaded {len(ranges)} episode ranges from {loader_name}; skipped full frame scan.", flush=True)
            return ranges
    print("Warning: falling back to full dataset frame scan to build episode ranges.", flush=True)
    return _episode_ranges_by_frame_scan(dataset, total_length)


def _image_to_chw_float(x: Any, *, required: bool = True) -> torch.Tensor:
    if x is None:
        if required:
            raise ValueError("Missing required image.")
        return torch.zeros(3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)

    if isinstance(x, torch.Tensor):
        tensor = x.detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            pass
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(tensor.shape)}")
        tensor = tensor.to(torch.float32)
        if tensor.max().item() > 1.5:
            tensor = tensor / 255.0
    else:
        arr = _as_numpy(x)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            chw = arr
        elif arr.ndim == 3:
            chw = np.transpose(arr[..., :3], (2, 0, 1))
        else:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")
        tensor = torch.as_tensor(chw, dtype=torch.float32)
        if np.issubdtype(arr.dtype, np.integer) or tensor.max().item() > 1.5:
            tensor = tensor / 255.0

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[-2:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            mode="nearest",
        ).squeeze(0)
    return tensor.clamp(0.0, 1.0).contiguous()


def _image_from_frame(frame: Mapping[str, Any], key: str, *, required: bool) -> torch.Tensor:
    try:
        return _image_to_chw_float(_field(frame, key), required=required)
    except Exception:
        if required:
            raise
        return torch.zeros(3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)


def _tensor_to_device(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        tensor = x.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(x, dtype=dtype, device=device)


def _move_stats_to_device(stats: Mapping[str, Any], device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in stats.items():
        if not isinstance(value, Mapping):
            continue
        out[key] = {}
        for stat_name, stat_value in value.items():
            try:
                out[key][stat_name] = torch.as_tensor(stat_value, dtype=torch.float32, device=device)
            except Exception:
                continue
    return out


def _stats_to_cpu(stats: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in stats.items():
        out[key] = {}
        for stat_name, stat_value in value.items():
            if isinstance(stat_value, torch.Tensor):
                out[key][stat_name] = stat_value.detach().cpu()
            else:
                out[key][stat_name] = torch.as_tensor(stat_value, dtype=torch.float32).cpu()
    return out


def _feature_dim_from_stats(stats: Mapping[str, Mapping[str, Any]], key: str) -> int:
    if key not in stats:
        raise KeyError(f"Dataset stats are missing {key!r}")
    for stat_name in ("mean", "min", "max", "std"):
        if stat_name in stats[key]:
            arr = torch.as_tensor(stats[key][stat_name])
            if arr.ndim == 0:
                return 1
            return int(arr.reshape(-1).shape[0])
    raise KeyError(f"Dataset stats for {key!r} have no usable tensor.")


def _mode_name(mode: Any) -> str:
    if hasattr(mode, "value"):
        return str(mode.value)
    return str(mode)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu()
        if arr.numel() == 1:
            return arr.item()
        return arr.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class DoorPolicySequenceDataset(Dataset):
    """LeRobotDataset wrapper that emits LeRobot DiffusionPolicy training batches."""

    def __init__(
        self,
        root: Union[str, Path],
        repo_id: str,
        obs_horizon: int,
        horizon: int,
        vision_mode: str = "depth",
    ):
        self.dataset_root = _resolve_lerobot_root(root, repo_id)
        self.dataset = _load_lerobot_dataset(self.dataset_root, repo_id)
        self.vision_mode = normalize_vision_mode(vision_mode)
        self.image_keys = lerobot_image_keys_for_vision_mode(self.vision_mode)
        self.obs_horizon = int(obs_horizon)
        self.horizon = int(horizon)
        if self.horizon < self.obs_horizon:
            raise ValueError(
                f"LeRobot DP horizon={self.horizon} must be >= obs_horizon={self.obs_horizon}. "
                "The action chunk starts at index obs_horizon - 1 inside the trajectory."
            )
        self.length = len(self.dataset)
        if self.length < self.horizon:
            raise ValueError("Dataset is too short for the requested horizons.")
        self.episode_ranges = load_episode_ranges(self.dataset_root, self.dataset, self.length)
        self.indices = self._build_indices()
        meta = getattr(self.dataset, "meta", None)
        self.stats = getattr(meta, "stats", None)
        if self.stats is None:
            raise ValueError("LeRobotDataset has no meta.stats; run conversion/stat generation before training.")

    def _build_indices(self) -> List[int]:
        indices: List[int] = []
        for start, end in self.episode_ranges:
            first = int(start) + self.obs_horizon - 1
            last = int(end) - self.horizon + self.obs_horizon - 1
            if last >= first:
                indices.extend(range(first, last + 1))
        if not indices:
            raise ValueError("No valid train sequences found. Record longer episodes or reduce horizons.")
        return indices

    def _frame(self, idx: int) -> Mapping[str, Any]:
        return self.dataset[int(idx)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        center = self.indices[int(item)]
        obs_start = center - self.obs_horizon + 1
        obs_ids = range(obs_start, center + 1)
        action_ids = range(obs_start, obs_start + self.horizon)
        image_required = self.vision_mode == "rgb"

        states: List[torch.Tensor] = []
        images: Dict[str, List[torch.Tensor]] = {key: [] for key in self.image_keys}
        for idx in obs_ids:
            frame = self._frame(idx)
            states.append(torch.as_tensor(_as_numpy(_field(frame, OBS_STATE)), dtype=torch.float32))
            for key in self.image_keys:
                images[key].append(_image_from_frame(frame, key, required=image_required))

        actions: List[torch.Tensor] = []
        for idx in action_ids:
            frame = self._frame(idx)
            actions.append(torch.as_tensor(_as_numpy(_field(frame, ACTION)), dtype=torch.float32))

        sample: Dict[str, torch.Tensor] = {
            OBS_STATE: torch.stack(states, dim=0),
            ACTION: torch.stack(actions, dim=0),
            ACTION_IS_PAD: torch.zeros(self.horizon, dtype=torch.bool),
        }
        for key, values in images.items():
            sample[key] = torch.stack(values, dim=0)
        return sample


class DoorPolicyChunkDataset(Dataset):
    """LeRobotDataset wrapper for single-observation chunking policies such as ACT and pi0.5."""

    def __init__(
        self,
        root: Union[str, Path],
        repo_id: str,
        chunk_size: int,
        vision_mode: str = "depth",
    ):
        self.dataset_root = _resolve_lerobot_root(root, repo_id)
        self.dataset = _load_lerobot_dataset(self.dataset_root, repo_id)
        self.vision_mode = normalize_vision_mode(vision_mode)
        self.image_keys = lerobot_image_keys_for_vision_mode(self.vision_mode)
        self.chunk_size = int(chunk_size)
        self.length = len(self.dataset)
        if self.length < self.chunk_size:
            raise ValueError("Dataset is too short for the requested action chunk size.")
        self.episode_ranges = load_episode_ranges(self.dataset_root, self.dataset, self.length)
        self.indices = self._build_indices()
        meta = getattr(self.dataset, "meta", None)
        self.stats = getattr(meta, "stats", None)
        if self.stats is None:
            raise ValueError("LeRobotDataset has no meta.stats; run conversion/stat generation before training.")

    def _build_indices(self) -> List[int]:
        indices: List[int] = []
        for start, end in self.episode_ranges:
            last = int(end) - self.chunk_size
            if last >= int(start):
                indices.extend(range(int(start), last + 1))
        if not indices:
            raise ValueError("No valid train chunks found. Record longer episodes or reduce chunk_size.")
        return indices

    def _frame(self, idx: int) -> Mapping[str, Any]:
        return self.dataset[int(idx)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        center = self.indices[int(item)]
        frame = self._frame(center)
        image_required = self.vision_mode == "rgb"
        sample: Dict[str, torch.Tensor] = {
            OBS_STATE: torch.as_tensor(_as_numpy(_field(frame, OBS_STATE)), dtype=torch.float32),
            ACTION_IS_PAD: torch.zeros(self.chunk_size, dtype=torch.bool),
        }
        for key in self.image_keys:
            sample[key] = _image_from_frame(frame, key, required=image_required)
        actions: List[torch.Tensor] = []
        for idx in range(center, center + self.chunk_size):
            action_frame = self._frame(idx)
            actions.append(torch.as_tensor(_as_numpy(_field(action_frame, ACTION)), dtype=torch.float32))
        sample[ACTION] = torch.stack(actions, dim=0)
        return sample


class DoorPolicyNormalizer:
    """Small equivalent of LeRobot normalization for the direct policy API path."""

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, Any]],
        normalization_mapping: Mapping[str, Any],
        image_keys: Sequence[str],
        device: torch.device,
        eps: float = 1e-8,
    ):
        self.stats = _move_stats_to_device(stats, device)
        self.normalization_mapping = {str(k): _mode_name(v) for k, v in normalization_mapping.items()}
        self.image_keys = list(image_keys)
        self.device = device
        self.eps = float(eps)

    def _feature_type(self, key: str) -> str:
        if key == ACTION:
            return "ACTION"
        if key == OBS_STATE:
            return "STATE"
        if key in self.image_keys or key.startswith("observation.images."):
            return "VISUAL"
        return "IDENTITY"

    def _mode_for_key(self, key: str) -> str:
        return self.normalization_mapping.get(self._feature_type(key), "IDENTITY")

    def _stat(self, key: str, name: str) -> torch.Tensor:
        if key not in self.stats or name not in self.stats[key]:
            raise KeyError(f"Missing LeRobot stat {key}.{name}")
        return self.stats[key][name]

    def apply(self, tensor: torch.Tensor, key: str, inverse: bool = False) -> torch.Tensor:
        mode = self._mode_for_key(key)
        if mode == "IDENTITY" or key not in self.stats:
            return tensor

        if mode == "MEAN_STD":
            mean = self._stat(key, "mean").to(device=tensor.device, dtype=tensor.dtype)
            std = self._stat(key, "std").to(device=tensor.device, dtype=tensor.dtype)
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / (std + self.eps)

        if mode == "MIN_MAX":
            min_val = self._stat(key, "min").to(device=tensor.device, dtype=tensor.dtype)
            max_val = self._stat(key, "max").to(device=tensor.device, dtype=tensor.dtype)
            denom = max_val - min_val
            denom = torch.where(
                denom == 0,
                torch.as_tensor(self.eps, device=tensor.device, dtype=tensor.dtype),
                denom,
            )
            if inverse:
                return (tensor + 1.0) / 2.0 * denom + min_val
            return 2.0 * (tensor - min_val) / denom - 1.0

        if mode == "QUANTILES":
            lo = self._stat(key, "q01").to(device=tensor.device, dtype=tensor.dtype)
            hi = self._stat(key, "q99").to(device=tensor.device, dtype=tensor.dtype)
            denom = hi - lo
            denom = torch.where(
                denom == 0,
                torch.as_tensor(self.eps, device=tensor.device, dtype=tensor.dtype),
                denom,
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + lo
            return 2.0 * (tensor - lo) / denom - 1.0

        if mode == "QUANTILE10":
            lo = self._stat(key, "q10").to(device=tensor.device, dtype=tensor.dtype)
            hi = self._stat(key, "q90").to(device=tensor.device, dtype=tensor.dtype)
            denom = hi - lo
            denom = torch.where(
                denom == 0,
                torch.as_tensor(self.eps, device=tensor.device, dtype=tensor.dtype),
                denom,
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + lo
            return 2.0 * (tensor - lo) / denom - 1.0

        raise ValueError(f"Unsupported normalization mode for {key}: {mode}")

    def normalize_batch(self, batch: Mapping[str, Any], include_action: bool = True) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            if key == ACTION_IS_PAD:
                out[key] = value.to(device=self.device, dtype=torch.bool)
                continue
            tensor = value.to(device=self.device)
            if key == ACTION and not include_action:
                continue
            if key == ACTION or key == OBS_STATE or key in self.image_keys:
                tensor = tensor.to(dtype=torch.float32)
                out[key] = self.apply(tensor, key, inverse=False)
            else:
                out[key] = tensor
        return out

    def normalize_state(self, state: Any) -> torch.Tensor:
        state_tensor = _tensor_to_device(state, self.device, torch.float32)
        return self.apply(state_tensor, OBS_STATE, inverse=False)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.apply(action, ACTION, inverse=True)


def make_lerobot_diffusion_config(
    *,
    state_dim: int,
    action_dim: int,
    obs_horizon: int,
    horizon: int,
    action_horizon: int,
    image_keys: Sequence[str],
    device: str,
    normalization_mapping: Optional[Mapping[str, Any]] = None,
    vision_backbone: str = "resnet18",
    resize_shape: Optional[Sequence[int]] = None,
    crop_ratio: float = 1.0,
    crop_is_random: bool = True,
    pretrained_backbone_weights: Optional[str] = None,
    use_group_norm: bool = True,
    spatial_softmax_num_keypoints: int = 32,
    use_separate_rgb_encoder_per_camera: bool = False,
    down_dims: Sequence[int] = (512, 1024, 2048),
    kernel_size: int = 5,
    n_groups: int = 8,
    diffusion_step_embed_dim: int = 128,
    use_film_scale_modulation: bool = True,
    noise_scheduler_type: str = "DDPM",
    num_train_timesteps: int = 100,
    beta_schedule: str = "squaredcos_cap_v2",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    prediction_type: str = "epsilon",
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
    num_inference_steps: Optional[int] = None,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    do_mask_loss_for_padding: bool = False,
):
    modules = import_lerobot_policy_modules()
    DiffusionConfig = modules["DiffusionConfig"]
    FeatureType = modules["FeatureType"]
    NormalizationMode = modules["NormalizationMode"]
    PolicyFeature = modules["PolicyFeature"]

    if action_horizon > horizon - obs_horizon + 1:
        raise ValueError(
            f"action_horizon={action_horizon} is too long for LeRobot DP horizon={horizon} "
            f"and obs_horizon={obs_horizon}. It must be <= horizon - obs_horizon + 1."
        )

    if normalization_mapping is None:
        normalization_mapping = {
            "VISUAL": "MEAN_STD",
            "STATE": "MIN_MAX",
            "ACTION": "MIN_MAX",
        }
    norm_map = {
        str(k): v if hasattr(v, "value") else NormalizationMode(str(v))
        for k, v in normalization_mapping.items()
    }

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(int(state_dim),)),
    }
    for key in image_keys:
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH))
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(int(action_dim),)),
    }

    resize_tuple = None if resize_shape is None else tuple(int(x) for x in resize_shape)
    config = DiffusionConfig(
        n_obs_steps=int(obs_horizon),
        input_features=input_features,
        output_features=output_features,
        device=str(device),
        push_to_hub=False,
        horizon=int(horizon),
        n_action_steps=int(action_horizon),
        normalization_mapping=norm_map,
        vision_backbone=str(vision_backbone),
        resize_shape=resize_tuple,
        crop_ratio=float(crop_ratio),
        crop_is_random=bool(crop_is_random),
        pretrained_backbone_weights=pretrained_backbone_weights,
        use_group_norm=bool(use_group_norm),
        spatial_softmax_num_keypoints=int(spatial_softmax_num_keypoints),
        use_separate_rgb_encoder_per_camera=bool(use_separate_rgb_encoder_per_camera),
        down_dims=tuple(int(x) for x in down_dims),
        kernel_size=int(kernel_size),
        n_groups=int(n_groups),
        diffusion_step_embed_dim=int(diffusion_step_embed_dim),
        use_film_scale_modulation=bool(use_film_scale_modulation),
        noise_scheduler_type=str(noise_scheduler_type),
        num_train_timesteps=int(num_train_timesteps),
        beta_schedule=str(beta_schedule),
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        prediction_type=str(prediction_type),
        clip_sample=bool(clip_sample),
        clip_sample_range=float(clip_sample_range),
        num_inference_steps=None if num_inference_steps is None else int(num_inference_steps),
        compile_model=bool(compile_model),
        compile_mode=str(compile_mode),
        do_mask_loss_for_padding=bool(do_mask_loss_for_padding),
    )
    return config


def make_lerobot_act_config(
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    action_horizon: int,
    image_keys: Sequence[str],
    device: str,
    normalization_mapping: Optional[Mapping[str, Any]] = None,
    vision_backbone: str = "resnet18",
    pretrained_backbone_weights: Optional[str] = "ResNet18_Weights.IMAGENET1K_V1",
    replace_final_stride_with_dilation: bool = False,
    pre_norm: bool = False,
    dim_model: int = 512,
    n_heads: int = 8,
    dim_feedforward: int = 3200,
    feedforward_activation: str = "relu",
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 1,
    use_vae: bool = True,
    latent_dim: int = 32,
    n_vae_encoder_layers: int = 4,
    temporal_ensemble_coeff: Optional[float] = None,
    dropout: float = 0.1,
    kl_weight: float = 10.0,
):
    modules = import_lerobot_act_modules()
    ACTConfig = modules["ACTConfig"]
    FeatureType = modules["FeatureType"]
    NormalizationMode = modules["NormalizationMode"]
    PolicyFeature = modules["PolicyFeature"]

    if action_horizon > chunk_size:
        raise ValueError(f"action_horizon={action_horizon} must be <= ACT chunk_size={chunk_size}.")
    if normalization_mapping is None:
        normalization_mapping = {
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    norm_map = {
        str(k): v if hasattr(v, "value") else NormalizationMode(str(v))
        for k, v in normalization_mapping.items()
    }
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(int(state_dim),)),
    }
    for key in image_keys:
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH))
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(int(action_dim),)),
    }
    return ACTConfig(
        n_obs_steps=1,
        input_features=input_features,
        output_features=output_features,
        device=str(device),
        push_to_hub=False,
        chunk_size=int(chunk_size),
        n_action_steps=int(action_horizon),
        normalization_mapping=norm_map,
        vision_backbone=str(vision_backbone),
        pretrained_backbone_weights=pretrained_backbone_weights,
        replace_final_stride_with_dilation=bool(replace_final_stride_with_dilation),
        pre_norm=bool(pre_norm),
        dim_model=int(dim_model),
        n_heads=int(n_heads),
        dim_feedforward=int(dim_feedforward),
        feedforward_activation=str(feedforward_activation),
        n_encoder_layers=int(n_encoder_layers),
        n_decoder_layers=int(n_decoder_layers),
        use_vae=bool(use_vae),
        latent_dim=int(latent_dim),
        n_vae_encoder_layers=int(n_vae_encoder_layers),
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        dropout=float(dropout),
        kl_weight=float(kl_weight),
    )


def make_lerobot_pi05_config(
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    action_horizon: int,
    image_keys: Sequence[str],
    device: str,
    normalization_mapping: Optional[Mapping[str, Any]] = None,
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    dtype: str = "float32",
    max_state_dim: int = 128,
    max_action_dim: int = 32,
    num_inference_steps: int = 10,
    image_resolution: Sequence[int] = (224, 224),
    tokenizer_max_length: int = 200,
    gradient_checkpointing: bool = False,
    compile_model: bool = False,
    compile_mode: str = "max-autotune",
    freeze_vision_encoder: bool = False,
    train_expert_only: bool = False,
):
    modules = import_lerobot_pi05_modules()
    PI05Config = modules["PI05Config"]
    FeatureType = modules["FeatureType"]
    NormalizationMode = modules["NormalizationMode"]
    PolicyFeature = modules["PolicyFeature"]

    if action_horizon > chunk_size:
        raise ValueError(f"action_horizon={action_horizon} must be <= pi0.5 chunk_size={chunk_size}.")
    if max_state_dim < state_dim:
        raise ValueError(f"max_state_dim={max_state_dim} must be >= state_dim={state_dim}.")
    if max_action_dim < action_dim:
        raise ValueError(f"max_action_dim={max_action_dim} must be >= action_dim={action_dim}.")
    if normalization_mapping is None:
        normalization_mapping = {
            "VISUAL": "IDENTITY",
            "STATE": "QUANTILES",
            "ACTION": "QUANTILES",
        }
    norm_map = {
        str(k): v if hasattr(v, "value") else NormalizationMode(str(v))
        for k, v in normalization_mapping.items()
    }
    image_resolution = tuple(int(x) for x in image_resolution)
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(int(state_dim),)),
    }
    for key in image_keys:
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_resolution[0], image_resolution[1]))
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(int(action_dim),)),
    }
    return PI05Config(
        n_obs_steps=1,
        input_features=input_features,
        output_features=output_features,
        device=str(device),
        push_to_hub=False,
        paligemma_variant=str(paligemma_variant),
        action_expert_variant=str(action_expert_variant),
        dtype=str(dtype),
        chunk_size=int(chunk_size),
        n_action_steps=int(action_horizon),
        max_state_dim=int(max_state_dim),
        max_action_dim=int(max_action_dim),
        num_inference_steps=int(num_inference_steps),
        image_resolution=image_resolution,
        tokenizer_max_length=int(tokenizer_max_length),
        normalization_mapping=norm_map,
        gradient_checkpointing=bool(gradient_checkpointing),
        compile_model=bool(compile_model),
        compile_mode=str(compile_mode),
        freeze_vision_encoder=bool(freeze_vision_encoder),
        train_expert_only=bool(train_expert_only),
    )


class LeRobotDiffusionDoorPolicyBackend:
    """Backend implementation backed by LeRobot's DiffusionPolicy."""

    backend_name = BACKEND_LEROBOT_DIFFUSION
    uses_observation_sequence = True

    def __init__(
        self,
        policy: Any,
        config: Any,
        stats: Mapping[str, Mapping[str, Any]],
        vision_mode: str,
        action_frame: str,
        sidecar_config: Optional[Mapping[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.policy = policy
        self.policy.eval()
        self.config = config
        self.device = torch.device(device or getattr(config, "device", "cpu"))
        self.policy.to(self.device)
        self.vision_mode = normalize_vision_mode(vision_mode)
        self.image_keys = lerobot_image_keys_for_vision_mode(self.vision_mode)
        self.action_frame = str(action_frame or "world").lower()
        self.sidecar_config = dict(sidecar_config or {})
        self.stats = _stats_to_cpu(stats)
        self.normalizer = DoorPolicyNormalizer(
            self.stats,
            getattr(config, "normalization_mapping", {}),
            self.image_keys,
            self.device,
        )

    @property
    def obs_horizon(self) -> int:
        return int(self.config.n_obs_steps)

    @property
    def horizon(self) -> int:
        return int(self.config.horizon)

    @property
    def action_horizon(self) -> int:
        return int(self.config.n_action_steps)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

    @classmethod
    def create(
        cls,
        *,
        stats: Mapping[str, Mapping[str, Any]],
        vision_mode: str,
        action_frame: str,
        sidecar_config: Optional[Mapping[str, Any]],
        device: Union[str, torch.device],
        obs_horizon: int,
        horizon: int,
        action_horizon: int,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        **policy_kwargs: Any,
    ) -> "LeRobotDiffusionDoorPolicyBackend":
        modules = import_lerobot_policy_modules()
        DiffusionPolicy = modules["DiffusionPolicy"]
        vision_mode = normalize_vision_mode(vision_mode)
        image_keys = lerobot_image_keys_for_vision_mode(vision_mode)
        state_dim = int(state_dim or _feature_dim_from_stats(stats, OBS_STATE))
        action_dim = int(action_dim or _feature_dim_from_stats(stats, ACTION))
        device_str = str(device)
        config = make_lerobot_diffusion_config(
            state_dim=state_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            horizon=horizon,
            action_horizon=action_horizon,
            image_keys=image_keys,
            device=device_str,
            **policy_kwargs,
        )
        policy = DiffusionPolicy(config).to(device_str)
        return cls(
            policy=policy,
            config=config,
            stats=stats,
            vision_mode=vision_mode,
            action_frame=action_frame,
            sidecar_config=sidecar_config,
            device=device_str,
        )

    @classmethod
    def load(
        cls,
        checkpoint: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        num_inference_steps: Optional[int] = None,
        action_horizon: Optional[int] = None,
    ) -> "LeRobotDiffusionDoorPolicyBackend":
        modules = import_lerobot_policy_modules()
        DiffusionPolicy = modules["DiffusionPolicy"]
        ckpt_dir = resolve_checkpoint_dir(checkpoint)
        meta_path = ckpt_dir / CHECKPOINT_META
        stats_path = ckpt_dir / CHECKPOINT_STATS
        policy_dir = ckpt_dir / CHECKPOINT_POLICY_DIR
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing Door policy metadata: {meta_path}")
        if not stats_path.is_file():
            raise FileNotFoundError(f"Missing Door policy stats: {stats_path}")
        if not policy_dir.is_dir():
            raise FileNotFoundError(f"Missing LeRobot policy directory: {policy_dir}")

        meta = _read_json(meta_path)
        if meta.get("backend") != BACKEND_LEROBOT_DIFFUSION:
            raise ValueError(f"Unsupported Door policy backend: {meta.get('backend')!r}")
        cfg = dict(meta.get("policy_config", {}))
        if device is not None:
            cfg["device"] = str(device)
        runtime_device = torch.device(cfg.get("device", "cpu"))
        if action_horizon is not None:
            cfg["action_horizon"] = int(action_horizon)
        if num_inference_steps is not None:
            cfg["num_inference_steps"] = int(num_inference_steps)

        config = make_lerobot_diffusion_config(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            obs_horizon=int(cfg["obs_horizon"]),
            horizon=int(cfg["horizon"]),
            action_horizon=int(cfg["action_horizon"]),
            image_keys=cfg["image_features"],
            device=str(runtime_device),
            normalization_mapping=cfg.get("normalization_mapping"),
            vision_backbone=cfg.get("vision_backbone", "resnet18"),
            resize_shape=cfg.get("resize_shape"),
            crop_ratio=float(cfg.get("crop_ratio", 1.0)),
            crop_is_random=bool(cfg.get("crop_is_random", True)),
            pretrained_backbone_weights=cfg.get("pretrained_backbone_weights"),
            use_group_norm=bool(cfg.get("use_group_norm", True)),
            spatial_softmax_num_keypoints=int(cfg.get("spatial_softmax_num_keypoints", 32)),
            use_separate_rgb_encoder_per_camera=bool(cfg.get("use_separate_rgb_encoder_per_camera", False)),
            down_dims=cfg.get("down_dims", (512, 1024, 2048)),
            kernel_size=int(cfg.get("kernel_size", 5)),
            n_groups=int(cfg.get("n_groups", 8)),
            diffusion_step_embed_dim=int(cfg.get("diffusion_step_embed_dim", 128)),
            use_film_scale_modulation=bool(cfg.get("use_film_scale_modulation", True)),
            noise_scheduler_type=cfg.get("noise_scheduler_type", "DDPM"),
            num_train_timesteps=int(cfg.get("num_train_timesteps", 100)),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            beta_start=float(cfg.get("beta_start", 0.0001)),
            beta_end=float(cfg.get("beta_end", 0.02)),
            prediction_type=cfg.get("prediction_type", "epsilon"),
            clip_sample=bool(cfg.get("clip_sample", True)),
            clip_sample_range=float(cfg.get("clip_sample_range", 1.0)),
            num_inference_steps=cfg.get("num_inference_steps"),
            compile_model=bool(cfg.get("compile_model", False)),
            compile_mode=cfg.get("compile_mode", "reduce-overhead"),
            do_mask_loss_for_padding=bool(cfg.get("do_mask_loss_for_padding", False)),
        )
        policy = DiffusionPolicy.from_pretrained(
            policy_dir,
            config=config,
            local_files_only=True,
            strict=True,
        )
        stats = torch.load(stats_path, map_location="cpu")
        sidecar_config = meta.get("sidecar_config", {})
        return cls(
            policy=policy,
            config=config,
            stats=stats,
            vision_mode=meta.get("vision_mode", cfg.get("vision_mode", "depth")),
            action_frame=meta.get("action_frame", cfg.get("action_frame", "world")),
            sidecar_config=sidecar_config,
            device=runtime_device,
        )

    def train(self, mode: bool = True):
        self.policy.train(mode)
        return self

    def eval(self):
        self.policy.eval()
        return self

    def optimizer_parameters(self) -> Iterable[torch.nn.Parameter]:
        params = self.policy.get_optim_params()
        if isinstance(params, Mapping):
            return params.values()
        return params

    def compute_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        batch_norm = self.normalizer.normalize_batch(batch, include_action=True)
        self.policy.train()
        loss, _ = self.policy(batch_norm)
        return loss

    @torch.no_grad()
    def predict_action_chunks_from_batch(
        self,
        batch: Mapping[str, Any],
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_norm = self.normalizer.normalize_batch(batch, include_action=False)
        batch_norm[OBS_IMAGES] = torch.stack([batch_norm[key] for key in self.image_keys], dim=-4)
        if noise is not None:
            noise = noise.to(device=self.device, dtype=torch.float32)
        self.policy.eval()
        actions = self.policy.diffusion.generate_actions(batch_norm, noise=noise)
        return self.normalizer.denormalize_action(actions)

    def metadata(self, extra_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        policy_config = {
            "state_dim": int(self.config.robot_state_feature.shape[0]),
            "action_dim": int(self.config.action_feature.shape[0]),
            "obs_horizon": int(self.config.n_obs_steps),
            "horizon": int(self.config.horizon),
            "pred_horizon": int(self.config.horizon),
            "action_horizon": int(self.config.n_action_steps),
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "vision_mode": self.vision_mode,
            "image_features": list(self.image_keys),
            "action_frame": self.action_frame,
            "action_pose_frame": self.action_frame,
            "target_pose_frame": self.action_frame,
            "normalization_mapping": _json_safe(getattr(self.config, "normalization_mapping", {})),
            "vision_backbone": self.config.vision_backbone,
            "resize_shape": None if self.config.resize_shape is None else list(self.config.resize_shape),
            "crop_ratio": float(self.config.crop_ratio),
            "crop_is_random": bool(self.config.crop_is_random),
            "pretrained_backbone_weights": self.config.pretrained_backbone_weights,
            "use_group_norm": bool(self.config.use_group_norm),
            "spatial_softmax_num_keypoints": int(self.config.spatial_softmax_num_keypoints),
            "use_separate_rgb_encoder_per_camera": bool(self.config.use_separate_rgb_encoder_per_camera),
            "down_dims": list(self.config.down_dims),
            "kernel_size": int(self.config.kernel_size),
            "n_groups": int(self.config.n_groups),
            "diffusion_step_embed_dim": int(self.config.diffusion_step_embed_dim),
            "use_film_scale_modulation": bool(self.config.use_film_scale_modulation),
            "noise_scheduler_type": self.config.noise_scheduler_type,
            "num_train_timesteps": int(self.config.num_train_timesteps),
            "num_diffusion_iters": int(self.config.num_train_timesteps),
            "beta_schedule": self.config.beta_schedule,
            "beta_start": float(self.config.beta_start),
            "beta_end": float(self.config.beta_end),
            "prediction_type": self.config.prediction_type,
            "clip_sample": bool(self.config.clip_sample),
            "clip_sample_range": float(self.config.clip_sample_range),
            "num_inference_steps": self.config.num_inference_steps,
            "compile_model": bool(self.config.compile_model),
            "compile_mode": self.config.compile_mode,
            "do_mask_loss_for_padding": bool(self.config.do_mask_loss_for_padding),
        }
        if "ikpush_state_version" in self.sidecar_config:
            policy_config["ikpush_state_version"] = str(self.sidecar_config["ikpush_state_version"])
        elif extra_config and "ikpush_state_version" in extra_config:
            policy_config["ikpush_state_version"] = str(extra_config["ikpush_state_version"])
        else:
            policy_config["ikpush_state_version"] = "legacy"
        if extra_config:
            for key, value in extra_config.items():
                if key not in policy_config:
                    policy_config[key] = _json_safe(value)
        return {
            "backend": BACKEND_LEROBOT_DIFFUSION,
            "format_version": 1,
            "policy_config": policy_config,
            "config": policy_config,
            "vision_mode": self.vision_mode,
            "action_frame": self.action_frame,
            "sidecar_config": _json_safe(self.sidecar_config),
            "action_names": list(ACTION_NAMES),
        }

    def save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_config: Optional[Mapping[str, Any]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy_dir = checkpoint_dir / CHECKPOINT_POLICY_DIR
        policy_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(policy_dir, push_to_hub=False)

        meta = self.metadata(extra_config=extra_config)
        with (checkpoint_dir / CHECKPOINT_META).open("w", encoding="utf-8") as f:
            json.dump(_json_safe(meta), f, indent=2, ensure_ascii=False)
        torch.save(_stats_to_cpu(self.stats), checkpoint_dir / CHECKPOINT_STATS)
        if optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict()}, checkpoint_dir / CHECKPOINT_OPTIMIZER)
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backend": BACKEND_LEROBOT_DIFFUSION,
                    "format_version": 1,
                    "checkpoint_dir": str(checkpoint_dir),
                    "checkpoint_dir_name": checkpoint_dir.name,
                    "policy_dir": str(policy_dir),
                    "config": meta["policy_config"],
                    "action_names": list(ACTION_NAMES),
                },
                manifest_path,
            )
        return checkpoint_dir


class LeRobotActDoorPolicyBackend:
    """Backend implementation backed by LeRobot's ACTPolicy."""

    backend_name = BACKEND_LEROBOT_ACT
    uses_observation_sequence = False

    def __init__(
        self,
        policy: Any,
        config: Any,
        stats: Mapping[str, Mapping[str, Any]],
        vision_mode: str,
        action_frame: str,
        sidecar_config: Optional[Mapping[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.policy = policy
        self.policy.eval()
        self.config = config
        self.device = torch.device(device or getattr(config, "device", "cpu"))
        self.policy.to(self.device)
        self.vision_mode = normalize_vision_mode(vision_mode)
        self.image_keys = lerobot_image_keys_for_vision_mode(self.vision_mode)
        self.action_frame = str(action_frame or "world").lower()
        self.sidecar_config = dict(sidecar_config or {})
        self.stats = _stats_to_cpu(stats)
        self.normalizer = DoorPolicyNormalizer(
            self.stats,
            getattr(config, "normalization_mapping", {}),
            self.image_keys,
            self.device,
        )

    @property
    def obs_horizon(self) -> int:
        return 1

    @property
    def horizon(self) -> int:
        return int(self.config.chunk_size)

    @property
    def action_horizon(self) -> int:
        return int(self.config.n_action_steps)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

    @classmethod
    def create(
        cls,
        *,
        stats: Mapping[str, Mapping[str, Any]],
        vision_mode: str,
        action_frame: str,
        sidecar_config: Optional[Mapping[str, Any]],
        device: Union[str, torch.device],
        chunk_size: int,
        action_horizon: int,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        **policy_kwargs: Any,
    ) -> "LeRobotActDoorPolicyBackend":
        modules = import_lerobot_act_modules()
        ACTPolicy = modules["ACTPolicy"]
        vision_mode = normalize_vision_mode(vision_mode)
        image_keys = lerobot_image_keys_for_vision_mode(vision_mode)
        state_dim = int(state_dim or _feature_dim_from_stats(stats, OBS_STATE))
        action_dim = int(action_dim or _feature_dim_from_stats(stats, ACTION))
        config = make_lerobot_act_config(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            action_horizon=action_horizon,
            image_keys=image_keys,
            device=str(device),
            **policy_kwargs,
        )
        if pretrained_path:
            policy = ACTPolicy.from_pretrained(pretrained_path, config=config, strict=False)
        else:
            policy = ACTPolicy(config)
        policy.to(str(device))
        return cls(policy, config, stats, vision_mode, action_frame, sidecar_config, device)

    @classmethod
    def load(
        cls,
        checkpoint: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        num_inference_steps: Optional[int] = None,
        action_horizon: Optional[int] = None,
    ) -> "LeRobotActDoorPolicyBackend":
        modules = import_lerobot_act_modules()
        ACTPolicy = modules["ACTPolicy"]
        ckpt_dir = resolve_checkpoint_dir(checkpoint)
        meta = _read_json(ckpt_dir / CHECKPOINT_META)
        cfg = dict(meta.get("policy_config", {}))
        if device is not None:
            cfg["device"] = str(device)
        if action_horizon is not None:
            cfg["action_horizon"] = int(action_horizon)
        config = make_lerobot_act_config(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            chunk_size=int(cfg["chunk_size"]),
            action_horizon=int(cfg["action_horizon"]),
            image_keys=cfg["image_features"],
            device=cfg.get("device", "cpu"),
            normalization_mapping=cfg.get("normalization_mapping"),
            vision_backbone=cfg.get("vision_backbone", "resnet18"),
            pretrained_backbone_weights=cfg.get("pretrained_backbone_weights"),
            replace_final_stride_with_dilation=bool(cfg.get("replace_final_stride_with_dilation", False)),
            pre_norm=bool(cfg.get("pre_norm", False)),
            dim_model=int(cfg.get("dim_model", 512)),
            n_heads=int(cfg.get("n_heads", 8)),
            dim_feedforward=int(cfg.get("dim_feedforward", 3200)),
            feedforward_activation=cfg.get("feedforward_activation", "relu"),
            n_encoder_layers=int(cfg.get("n_encoder_layers", 4)),
            n_decoder_layers=int(cfg.get("n_decoder_layers", 1)),
            use_vae=bool(cfg.get("use_vae", True)),
            latent_dim=int(cfg.get("latent_dim", 32)),
            n_vae_encoder_layers=int(cfg.get("n_vae_encoder_layers", 4)),
            temporal_ensemble_coeff=cfg.get("temporal_ensemble_coeff"),
            dropout=float(cfg.get("dropout", 0.1)),
            kl_weight=float(cfg.get("kl_weight", 10.0)),
        )
        policy = ACTPolicy.from_pretrained(
            ckpt_dir / CHECKPOINT_POLICY_DIR,
            config=config,
            local_files_only=True,
            strict=True,
        )
        stats = torch.load(ckpt_dir / CHECKPOINT_STATS, map_location="cpu")
        return cls(
            policy,
            config,
            stats,
            meta.get("vision_mode", cfg.get("vision_mode", "depth")),
            meta.get("action_frame", cfg.get("action_frame", "world")),
            meta.get("sidecar_config", {}),
            device=config.device,
        )

    def train(self, mode: bool = True):
        self.policy.train(mode)
        return self

    def eval(self):
        self.policy.eval()
        return self

    def compute_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        batch_norm = self.normalizer.normalize_batch(batch, include_action=True)
        self.policy.train()
        loss, _ = self.policy(batch_norm)
        return loss

    @torch.no_grad()
    def predict_action_chunks_from_batch(self, batch: Mapping[str, Any], noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_norm = self.normalizer.normalize_batch(batch, include_action=False)
        self.policy.eval()
        actions = self.policy.predict_action_chunk(batch_norm)
        return self.normalizer.denormalize_action(actions)

    def metadata(self, extra_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        policy_config = {
            "state_dim": int(self.config.robot_state_feature.shape[0]),
            "action_dim": int(self.config.action_feature.shape[0]),
            "obs_horizon": 1,
            "chunk_size": int(self.config.chunk_size),
            "horizon": int(self.config.chunk_size),
            "pred_horizon": int(self.config.chunk_size),
            "action_horizon": int(self.config.n_action_steps),
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "vision_mode": self.vision_mode,
            "image_features": list(self.image_keys),
            "action_frame": self.action_frame,
            "action_pose_frame": self.action_frame,
            "target_pose_frame": self.action_frame,
            "normalization_mapping": _json_safe(getattr(self.config, "normalization_mapping", {})),
            "vision_backbone": self.config.vision_backbone,
            "pretrained_backbone_weights": self.config.pretrained_backbone_weights,
            "replace_final_stride_with_dilation": bool(self.config.replace_final_stride_with_dilation),
            "pre_norm": bool(self.config.pre_norm),
            "dim_model": int(self.config.dim_model),
            "n_heads": int(self.config.n_heads),
            "dim_feedforward": int(self.config.dim_feedforward),
            "feedforward_activation": self.config.feedforward_activation,
            "n_encoder_layers": int(self.config.n_encoder_layers),
            "n_decoder_layers": int(self.config.n_decoder_layers),
            "use_vae": bool(self.config.use_vae),
            "latent_dim": int(self.config.latent_dim),
            "n_vae_encoder_layers": int(self.config.n_vae_encoder_layers),
            "temporal_ensemble_coeff": self.config.temporal_ensemble_coeff,
            "dropout": float(self.config.dropout),
            "kl_weight": float(self.config.kl_weight),
            "ikpush_state_version": str(self.sidecar_config.get("ikpush_state_version", "legacy")),
        }
        if extra_config:
            policy_config.update({k: _json_safe(v) for k, v in extra_config.items() if k not in policy_config})
        return {
            "backend": BACKEND_LEROBOT_ACT,
            "format_version": 1,
            "policy_config": policy_config,
            "config": policy_config,
            "vision_mode": self.vision_mode,
            "action_frame": self.action_frame,
            "sidecar_config": _json_safe(self.sidecar_config),
            "action_names": list(ACTION_NAMES),
        }

    def save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_config: Optional[Mapping[str, Any]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy_dir = checkpoint_dir / CHECKPOINT_POLICY_DIR
        policy_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(policy_dir, push_to_hub=False)
        meta = self.metadata(extra_config=extra_config)
        with (checkpoint_dir / CHECKPOINT_META).open("w", encoding="utf-8") as f:
            json.dump(_json_safe(meta), f, indent=2, ensure_ascii=False)
        torch.save(_stats_to_cpu(self.stats), checkpoint_dir / CHECKPOINT_STATS)
        if optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict()}, checkpoint_dir / CHECKPOINT_OPTIMIZER)
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backend": BACKEND_LEROBOT_ACT,
                    "format_version": 1,
                    "checkpoint_dir": str(checkpoint_dir),
                    "checkpoint_dir_name": checkpoint_dir.name,
                    "policy_dir": str(policy_dir),
                    "config": meta["policy_config"],
                    "action_names": list(ACTION_NAMES),
                },
                manifest_path,
            )
        return checkpoint_dir


class LeRobotPI05DoorPolicyBackend:
    """Backend implementation backed by LeRobot's PI05Policy."""

    backend_name = BACKEND_LEROBOT_PI05
    uses_observation_sequence = False

    def __init__(
        self,
        policy: Any,
        config: Any,
        stats: Mapping[str, Mapping[str, Any]],
        vision_mode: str,
        action_frame: str,
        sidecar_config: Optional[Mapping[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        task_prompt: str = "open the door",
        tokenizer_name: str = "google/paligemma-3b-pt-224",
    ):
        self.policy = policy
        self.policy.eval()
        self.config = config
        self.device = torch.device(device or getattr(config, "device", "cpu"))
        self.policy.to(self.device)
        self.vision_mode = normalize_vision_mode(vision_mode)
        self.image_keys = lerobot_image_keys_for_vision_mode(self.vision_mode)
        self.action_frame = str(action_frame or "world").lower()
        self.sidecar_config = dict(sidecar_config or {})
        self.stats = _stats_to_cpu(stats)
        self.task_prompt = str(task_prompt)
        self.tokenizer_name = str(tokenizer_name)
        self._tokenizer = None
        self.normalizer = DoorPolicyNormalizer(
            self.stats,
            getattr(config, "normalization_mapping", {}),
            self.image_keys,
            self.device,
        )
        self._pi05_modules = import_lerobot_pi05_modules()

    @property
    def obs_horizon(self) -> int:
        return 1

    @property
    def horizon(self) -> int:
        return int(self.config.chunk_size)

    @property
    def action_horizon(self) -> int:
        return int(self.config.n_action_steps)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

    @classmethod
    def create(
        cls,
        *,
        stats: Mapping[str, Mapping[str, Any]],
        vision_mode: str,
        action_frame: str,
        sidecar_config: Optional[Mapping[str, Any]],
        device: Union[str, torch.device],
        chunk_size: int,
        action_horizon: int,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        task_prompt: str = "open the door",
        tokenizer_name: str = "google/paligemma-3b-pt-224",
        **policy_kwargs: Any,
    ) -> "LeRobotPI05DoorPolicyBackend":
        modules = import_lerobot_pi05_modules()
        PI05Policy = modules["PI05Policy"]
        vision_mode = normalize_vision_mode(vision_mode)
        image_keys = lerobot_image_keys_for_vision_mode(vision_mode)
        state_dim = int(state_dim or _feature_dim_from_stats(stats, OBS_STATE))
        action_dim = int(action_dim or _feature_dim_from_stats(stats, ACTION))
        config = make_lerobot_pi05_config(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            action_horizon=action_horizon,
            image_keys=image_keys,
            device=str(device),
            **policy_kwargs,
        )
        if pretrained_path:
            policy = PI05Policy.from_pretrained(pretrained_path, config=config, strict=False)
        else:
            policy = PI05Policy(config)
        policy.to(str(device))
        return cls(policy, config, stats, vision_mode, action_frame, sidecar_config, device, task_prompt, tokenizer_name)

    @classmethod
    def load(
        cls,
        checkpoint: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        num_inference_steps: Optional[int] = None,
        action_horizon: Optional[int] = None,
    ) -> "LeRobotPI05DoorPolicyBackend":
        modules = import_lerobot_pi05_modules()
        PI05Policy = modules["PI05Policy"]
        ckpt_dir = resolve_checkpoint_dir(checkpoint)
        meta = _read_json(ckpt_dir / CHECKPOINT_META)
        cfg = dict(meta.get("policy_config", {}))
        if device is not None:
            cfg["device"] = str(device)
        if action_horizon is not None:
            cfg["action_horizon"] = int(action_horizon)
        if num_inference_steps is not None:
            cfg["num_inference_steps"] = int(num_inference_steps)
        config = make_lerobot_pi05_config(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            chunk_size=int(cfg["chunk_size"]),
            action_horizon=int(cfg["action_horizon"]),
            image_keys=cfg["image_features"],
            device=cfg.get("device", "cpu"),
            normalization_mapping=cfg.get("normalization_mapping"),
            paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
            dtype=cfg.get("dtype", "float32"),
            max_state_dim=int(cfg.get("max_state_dim", 128)),
            max_action_dim=int(cfg.get("max_action_dim", 32)),
            num_inference_steps=int(cfg.get("num_inference_steps", 10)),
            image_resolution=cfg.get("image_resolution", (224, 224)),
            tokenizer_max_length=int(cfg.get("tokenizer_max_length", 200)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
            compile_model=bool(cfg.get("compile_model", False)),
            compile_mode=cfg.get("compile_mode", "max-autotune"),
            freeze_vision_encoder=bool(cfg.get("freeze_vision_encoder", False)),
            train_expert_only=bool(cfg.get("train_expert_only", False)),
        )
        policy = PI05Policy.from_pretrained(
            ckpt_dir / CHECKPOINT_POLICY_DIR,
            config=config,
            local_files_only=True,
            strict=False,
        )
        stats = torch.load(ckpt_dir / CHECKPOINT_STATS, map_location="cpu")
        return cls(
            policy,
            config,
            stats,
            meta.get("vision_mode", cfg.get("vision_mode", "depth")),
            meta.get("action_frame", cfg.get("action_frame", "world")),
            meta.get("sidecar_config", {}),
            device=config.device,
            task_prompt=cfg.get("task_prompt", "open the door"),
            tokenizer_name=cfg.get("tokenizer_name", "google/paligemma-3b-pt-224"),
        )

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def _add_language_tokens(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens_key = self._pi05_modules["OBS_LANGUAGE_TOKENS"]
        mask_key = self._pi05_modules["OBS_LANGUAGE_ATTENTION_MASK"]
        state = batch[OBS_STATE]
        if state.ndim != 2:
            state = state.reshape(state.shape[0], -1)
        bsz = state.shape[0]
        padded = torch.zeros(bsz, int(self.config.max_state_dim), dtype=state.dtype, device=state.device)
        width = min(state.shape[1], padded.shape[1])
        padded[:, :width] = state[:, :width]
        bins = torch.linspace(-1.0, 1.0, 257, device=state.device, dtype=state.dtype)[:-1]
        discretized = torch.bucketize(padded.clamp(-1.0, 1.0), bins).sub(1).clamp(0, 255).cpu().numpy()
        prompts = []
        for row in discretized:
            state_str = " ".join(str(int(v)) for v in row)
            prompts.append(f"Task: {self.task_prompt.strip()}, State: {state_str};\nAction: ")
        tokenized = self._get_tokenizer()(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=int(self.config.tokenizer_max_length),
            return_tensors="pt",
        )
        batch[tokens_key] = tokenized["input_ids"].to(self.device)
        batch[mask_key] = tokenized["attention_mask"].to(self.device, dtype=torch.bool)
        return batch

    def train(self, mode: bool = True):
        self.policy.train(mode)
        return self

    def eval(self):
        self.policy.eval()
        return self

    def compute_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        batch_norm = self.normalizer.normalize_batch(batch, include_action=True)
        batch_norm = self._add_language_tokens(batch_norm)
        self.policy.train()
        loss, _ = self.policy(batch_norm)
        return loss

    @torch.no_grad()
    def predict_action_chunks_from_batch(self, batch: Mapping[str, Any], noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_norm = self.normalizer.normalize_batch(batch, include_action=False)
        batch_norm = self._add_language_tokens(batch_norm)
        self.policy.eval()
        actions = self.policy.predict_action_chunk(batch_norm)
        return self.normalizer.denormalize_action(actions)

    def metadata(self, extra_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        policy_config = {
            "state_dim": int(self.config.input_features[OBS_STATE].shape[0]),
            "action_dim": int(self.config.action_feature.shape[0]),
            "obs_horizon": 1,
            "chunk_size": int(self.config.chunk_size),
            "horizon": int(self.config.chunk_size),
            "pred_horizon": int(self.config.chunk_size),
            "action_horizon": int(self.config.n_action_steps),
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "vision_mode": self.vision_mode,
            "image_features": list(self.image_keys),
            "action_frame": self.action_frame,
            "action_pose_frame": self.action_frame,
            "target_pose_frame": self.action_frame,
            "normalization_mapping": _json_safe(getattr(self.config, "normalization_mapping", {})),
            "paligemma_variant": self.config.paligemma_variant,
            "action_expert_variant": self.config.action_expert_variant,
            "dtype": self.config.dtype,
            "max_state_dim": int(self.config.max_state_dim),
            "max_action_dim": int(self.config.max_action_dim),
            "num_inference_steps": int(self.config.num_inference_steps),
            "image_resolution": list(self.config.image_resolution),
            "tokenizer_max_length": int(self.config.tokenizer_max_length),
            "tokenizer_name": self.tokenizer_name,
            "task_prompt": self.task_prompt,
            "gradient_checkpointing": bool(self.config.gradient_checkpointing),
            "compile_model": bool(self.config.compile_model),
            "compile_mode": self.config.compile_mode,
            "freeze_vision_encoder": bool(self.config.freeze_vision_encoder),
            "train_expert_only": bool(self.config.train_expert_only),
            "ikpush_state_version": str(self.sidecar_config.get("ikpush_state_version", "legacy")),
        }
        if extra_config:
            policy_config.update({k: _json_safe(v) for k, v in extra_config.items() if k not in policy_config})
        return {
            "backend": BACKEND_LEROBOT_PI05,
            "format_version": 1,
            "policy_config": policy_config,
            "config": policy_config,
            "vision_mode": self.vision_mode,
            "action_frame": self.action_frame,
            "sidecar_config": _json_safe(self.sidecar_config),
            "action_names": list(ACTION_NAMES),
        }

    def save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_config: Optional[Mapping[str, Any]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy_dir = checkpoint_dir / CHECKPOINT_POLICY_DIR
        policy_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(policy_dir, push_to_hub=False)
        meta = self.metadata(extra_config=extra_config)
        with (checkpoint_dir / CHECKPOINT_META).open("w", encoding="utf-8") as f:
            json.dump(_json_safe(meta), f, indent=2, ensure_ascii=False)
        torch.save(_stats_to_cpu(self.stats), checkpoint_dir / CHECKPOINT_STATS)
        if optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict()}, checkpoint_dir / CHECKPOINT_OPTIMIZER)
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backend": BACKEND_LEROBOT_PI05,
                    "format_version": 1,
                    "checkpoint_dir": str(checkpoint_dir),
                    "checkpoint_dir_name": checkpoint_dir.name,
                    "policy_dir": str(policy_dir),
                    "config": meta["policy_config"],
                    "action_names": list(ACTION_NAMES),
                },
                manifest_path,
            )
        return checkpoint_dir


def resolve_checkpoint_dir(checkpoint: Union[str, Path]) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir():
        return path
    if not path.is_file():
        raise FileNotFoundError(f"Door policy checkpoint not found: {path}")
    manifest = torch.load(path, map_location="cpu")
    if not isinstance(manifest, Mapping) or manifest.get("backend") not in (
        BACKEND_LEROBOT_DIFFUSION,
        BACKEND_LEROBOT_ACT,
        BACKEND_LEROBOT_PI05,
    ):
        raise ValueError(
            f"{path} is not a supported LeRobot Door policy checkpoint manifest. "
            "Retrain with high-level/dp/train_door_dp.py after the LeRobot backend migration."
        )
    candidates = []
    if manifest.get("checkpoint_dir"):
        candidates.append(Path(manifest["checkpoint_dir"]).expanduser())
    if manifest.get("checkpoint_dir_name"):
        candidates.append(path.parent / str(manifest["checkpoint_dir_name"]))
    if manifest.get("policy_dir"):
        candidates.append(Path(manifest["policy_dir"]).expanduser().parent)
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / CHECKPOINT_META).is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve Door policy checkpoint directory from manifest: {path}")


def checkpoint_backend_name(checkpoint: Union[str, Path]) -> str:
    ckpt_dir = resolve_checkpoint_dir(checkpoint)
    meta = _read_json(ckpt_dir / CHECKPOINT_META)
    backend = str(meta.get("backend", ""))
    if backend not in (BACKEND_LEROBOT_DIFFUSION, BACKEND_LEROBOT_ACT, BACKEND_LEROBOT_PI05):
        raise ValueError(f"Unsupported Door policy backend in {ckpt_dir}: {backend!r}")
    return backend


def load_door_policy_backend(
    checkpoint: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    num_inference_steps: Optional[int] = None,
    action_horizon: Optional[int] = None,
):
    backend = checkpoint_backend_name(checkpoint)
    if backend == BACKEND_LEROBOT_DIFFUSION:
        return LeRobotDiffusionDoorPolicyBackend.load(
            checkpoint,
            device=device,
            num_inference_steps=num_inference_steps,
            action_horizon=action_horizon,
        )
    if backend == BACKEND_LEROBOT_ACT:
        return LeRobotActDoorPolicyBackend.load(
            checkpoint,
            device=device,
            num_inference_steps=num_inference_steps,
            action_horizon=action_horizon,
        )
    if backend == BACKEND_LEROBOT_PI05:
        return LeRobotPI05DoorPolicyBackend.load(
            checkpoint,
            device=device,
            num_inference_steps=num_inference_steps,
            action_horizon=action_horizon,
        )
    raise ValueError(f"Unsupported Door policy backend: {backend!r}")


class DoorPolicyController:
    """Runtime controller with the old DoorDPPolicyController call contract."""

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        num_inference_steps: Optional[int] = None,
        action_horizon: Optional[int] = None,
    ):
        self.backend = load_door_policy_backend(
            checkpoint,
            device=device,
            num_inference_steps=num_inference_steps,
            action_horizon=action_horizon,
        )
        self.device = self.backend.device
        self.config = dict(self.backend.metadata()["policy_config"])
        self.vision_mode = self.backend.vision_mode
        self.action_frame = self.backend.action_frame
        self.image_keys = list(self.backend.image_keys)
        self.obs_horizon = self.backend.obs_horizon
        self.pred_horizon = self.backend.horizon
        self.action_horizon = self.backend.action_horizon
        self.action_dim = self.backend.action_dim
        self.obs_buffer: deque = deque(maxlen=self.obs_horizon)
        self.action_queue: deque = deque()
        self.multi_obs_buffers: Dict[int, deque] = {}
        self.multi_action_queues: Dict[int, deque] = {}

    def reset(self) -> None:
        self.obs_buffer.clear()
        self.action_queue.clear()
        self.multi_obs_buffers.clear()
        self.multi_action_queues.clear()

    def reset_envs(self, env_ids: Optional[Sequence[int]] = None) -> None:
        if env_ids is None:
            self.multi_obs_buffers.clear()
            self.multi_action_queues.clear()
            return
        for env_id in env_ids:
            env_id = int(env_id)
            self.multi_obs_buffers.pop(env_id, None)
            self.multi_action_queues.pop(env_id, None)

    def _ensure_env_buffers(self, env_id: int) -> Tuple[deque, deque]:
        env_id = int(env_id)
        obs_buffer = self.multi_obs_buffers.get(env_id)
        if obs_buffer is None:
            obs_buffer = deque(maxlen=self.obs_horizon)
            self.multi_obs_buffers[env_id] = obs_buffer
        action_queue = self.multi_action_queues.get(env_id)
        if action_queue is None:
            action_queue = deque()
            self.multi_action_queues[env_id] = action_queue
        return obs_buffer, action_queue

    def _normalize_state(self, state: Any) -> torch.Tensor:
        return self.backend.normalizer.normalize_state(state)

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.backend.normalizer.denormalize_action(action)

    def _make_item(
        self,
        state: Any,
        mask_rgb: Any,
        second_rgb: Any,
        front_mask_rgb: Any = None,
        front_second_rgb: Any = None,
    ) -> Dict[str, torch.Tensor]:
        if front_mask_rgb is None:
            front_mask_rgb = np.zeros_like(mask_rgb)
        if front_second_rgb is None:
            front_second_rgb = np.zeros_like(second_rgb)
        return {
            OBS_STATE: _tensor_to_device(state, self.device, torch.float32),
            self.image_keys[0]: _image_to_chw_float(mask_rgb, required=True).to(self.device),
            self.image_keys[1]: _image_to_chw_float(second_rgb, required=True).to(self.device),
            self.image_keys[2]: _image_to_chw_float(front_mask_rgb, required=True).to(self.device),
            self.image_keys[3]: _image_to_chw_float(front_second_rgb, required=True).to(self.device),
        }

    def append_observation(
        self,
        state: Any,
        mask_rgb: Any,
        masked_depth_rgb: Any,
        front_mask_rgb: Any = None,
        front_masked_depth_rgb: Any = None,
    ) -> None:
        item = self._make_item(state, mask_rgb, masked_depth_rgb, front_mask_rgb, front_masked_depth_rgb)
        if len(self.obs_buffer) == 0:
            for _ in range(self.obs_horizon):
                self.obs_buffer.append(item)
        else:
            self.obs_buffer.append(item)

    def append_observation_for_env(
        self,
        env_id: int,
        state: Any,
        mask_rgb: Any,
        masked_depth_rgb: Any,
        front_mask_rgb: Any = None,
        front_masked_depth_rgb: Any = None,
    ) -> None:
        item = self._make_item(state, mask_rgb, masked_depth_rgb, front_mask_rgb, front_masked_depth_rgb)
        obs_buffer, _ = self._ensure_env_buffers(int(env_id))
        if len(obs_buffer) == 0:
            for _ in range(self.obs_horizon):
                obs_buffer.append(item)
        else:
            obs_buffer.append(item)

    def _batch_from_windows(self, windows: Sequence[Sequence[Mapping[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, List[torch.Tensor]] = {OBS_STATE: []}
        for key in self.image_keys:
            batch[key] = []
        for window in windows:
            if getattr(self.backend, "uses_observation_sequence", True):
                if len(window) != self.obs_horizon:
                    raise ValueError(f"Expected obs window length {self.obs_horizon}, got {len(window)}")
                batch[OBS_STATE].append(torch.stack([item[OBS_STATE] for item in window], dim=0))
                for key in self.image_keys:
                    batch[key].append(torch.stack([item[key] for item in window], dim=0))
            else:
                item = window[-1]
                batch[OBS_STATE].append(item[OBS_STATE])
                for key in self.image_keys:
                    batch[key].append(item[key])
        return {key: torch.stack(values, dim=0) for key, values in batch.items()}

    def _current_batch(self) -> Dict[str, torch.Tensor]:
        if len(self.obs_buffer) != self.obs_horizon:
            raise RuntimeError("Observation buffer is not initialized.")
        return self._batch_from_windows([list(self.obs_buffer)])

    @torch.no_grad()
    def predict_action_chunks_from_batch(
        self,
        batch: Mapping[str, Any],
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.backend.predict_action_chunks_from_batch(batch, noise=noise)

    @torch.no_grad()
    def predict_action_chunks_from_windows(
        self,
        windows: Sequence[Sequence[Mapping[str, torch.Tensor]]],
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_action_chunks_from_batch(self._batch_from_windows(windows), noise=noise)

    @torch.no_grad()
    def sample_action_chunks_for_envs(
        self,
        env_ids: Sequence[int],
        noise: Optional[torch.Tensor] = None,
    ) -> None:
        env_ids = [int(env_id) for env_id in env_ids]
        if not env_ids:
            return
        windows = []
        for env_id in env_ids:
            obs_buffer, _ = self._ensure_env_buffers(env_id)
            if len(obs_buffer) != self.obs_horizon:
                raise RuntimeError(f"Observation buffer for env {env_id} is not initialized.")
            windows.append(list(obs_buffer))
        actions = self.predict_action_chunks_from_windows(windows, noise=noise)
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        for row_idx, env_id in enumerate(env_ids):
            _, action_queue = self._ensure_env_buffers(env_id)
            action_queue.clear()
            for row in actions_np[row_idx, : self.action_horizon]:
                action_queue.append(row)

    @torch.no_grad()
    def sample_action_chunk(self, noise: Optional[torch.Tensor] = None) -> None:
        actions = self.backend.predict_action_chunks_from_batch(self._current_batch(), noise=noise)
        actions_np = actions[0].detach().cpu().numpy().astype(np.float32)
        self.action_queue.clear()
        for row in actions_np[: self.action_horizon]:
            self.action_queue.append(row)

    def act(
        self,
        state: Any,
        mask_rgb: Any,
        masked_depth_rgb: Any,
        front_mask_rgb: Any = None,
        front_masked_depth_rgb: Any = None,
    ) -> np.ndarray:
        self.append_observation(state, mask_rgb, masked_depth_rgb, front_mask_rgb, front_masked_depth_rgb)
        if not self.action_queue:
            self.sample_action_chunk()
        return self.action_queue.popleft()

    def act_batch(
        self,
        env_ids: Sequence[int],
        states: Any,
        mask_rgbs: Any,
        masked_depth_rgbs: Any,
        front_mask_rgbs: Any = None,
        front_masked_depth_rgbs: Any = None,
    ) -> np.ndarray:
        env_ids = [int(env_id) for env_id in env_ids]
        if not env_ids:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        states_seq = list(states)
        mask_seq = list(mask_rgbs)
        second_seq = list(masked_depth_rgbs)
        if len(states_seq) != len(env_ids) or len(mask_seq) != len(env_ids) or len(second_seq) != len(env_ids):
            raise ValueError("act_batch inputs must have the same length as env_ids.")
        front_mask_seq = [None] * len(env_ids) if front_mask_rgbs is None else list(front_mask_rgbs)
        front_second_seq = [None] * len(env_ids) if front_masked_depth_rgbs is None else list(front_masked_depth_rgbs)
        if len(front_mask_seq) != len(env_ids) or len(front_second_seq) != len(env_ids):
            raise ValueError("act_batch front image inputs must have the same length as env_ids.")

        for idx, env_id in enumerate(env_ids):
            self.append_observation_for_env(
                env_id,
                states_seq[idx],
                mask_seq[idx],
                second_seq[idx],
                front_mask_seq[idx],
                front_second_seq[idx],
            )

        empty_env_ids = [env_id for env_id in env_ids if not self._ensure_env_buffers(env_id)[1]]
        if empty_env_ids:
            self.sample_action_chunks_for_envs(empty_env_ids)
        actions = []
        for env_id in env_ids:
            _, action_queue = self._ensure_env_buffers(env_id)
            if not action_queue:
                raise RuntimeError(f"Action queue for env {env_id} is empty after sampling.")
            actions.append(action_queue.popleft())
        return np.stack(actions, axis=0).astype(np.float32)


def _write_pickle_message(stream: Any, payload: Mapping[str, Any]) -> None:
    data = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def _read_exact(stream: Any, n_bytes: int) -> bytes:
    chunks = []
    remaining = int(n_bytes)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Door policy worker closed the protocol stream.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_pickle_message(stream: Any) -> Mapping[str, Any]:
    header = _read_exact(stream, 4)
    size = struct.unpack(">I", header)[0]
    return pickle.loads(_read_exact(stream, size))


def _worker_python_command() -> List[str]:
    worker = Path(__file__).resolve().parent / "door_policy_worker.py"
    explicit = os.environ.get("DOOR_DP_LEROBOT_PYTHON")
    if explicit:
        return shlex.split(explicit) + [str(worker)]

    env_name = os.environ.get("DOOR_DP_LEROBOT_CONDA_ENV", "door_dp")
    conda_exe = shutil.which("conda")
    if conda_exe:
        return [conda_exe, "run", "--no-capture-output", "-n", env_name, "python", str(worker)]

    raise RuntimeError(
        "Could not find a LeRobot Python for subprocess inference. Set DOOR_DP_LEROBOT_PYTHON "
        "to a Python>=3.10 executable or DOOR_DP_LEROBOT_CONDA_ENV to a conda env name."
    )


class _RemoteObsBufferProxy:
    def __init__(self, controller: "DoorPolicySubprocessController"):
        self.controller = controller

    def clear(self) -> None:
        self.controller.reset()

    def __len__(self) -> int:
        return 0


class _RemoteActionQueueProxy:
    def __init__(self, controller: "DoorPolicySubprocessController"):
        self.controller = controller

    def clear(self) -> None:
        self.controller._request({"cmd": "clear_action_queue"})

    def __len__(self) -> int:
        return 0


class DoorPolicySubprocessController:
    """Controller proxy used when LeRobot cannot be imported in the Isaac Gym process."""

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        num_inference_steps: Optional[int] = None,
        action_horizon: Optional[int] = None,
        startup_error: Optional[BaseException] = None,
    ):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self._startup_error = startup_error
        self._proc = subprocess.Popen(
            _worker_python_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        try:
            response = self._request(
                {
                    "cmd": "init",
                    "checkpoint": self.checkpoint,
                    "device": None if device is None else str(device),
                    "num_inference_steps": num_inference_steps,
                    "action_horizon": action_horizon,
                }
            )
        except Exception:
            self.close()
            raise
        self._set_metadata(response["metadata"])
        self.obs_buffer = _RemoteObsBufferProxy(self)
        self.action_queue = _RemoteActionQueueProxy(self)

    def _set_metadata(self, meta: Mapping[str, Any]) -> None:
        self.config = dict(meta.get("config", {}))
        self.vision_mode = str(meta["vision_mode"])
        self.action_frame = str(meta["action_frame"])
        self.image_keys = list(meta["image_keys"])
        self.obs_horizon = int(meta["obs_horizon"])
        self.pred_horizon = int(meta["pred_horizon"])
        self.action_horizon = int(meta["action_horizon"])
        self.action_dim = int(meta["action_dim"])
        self.device = torch.device(str(meta.get("device", "cpu")))

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Door policy worker pipes are closed.")
        _write_pickle_message(self._proc.stdin, payload)
        response = _read_pickle_message(self._proc.stdout)
        if not response.get("ok", False):
            error = response.get("error", "Unknown Door policy worker error")
            if self._startup_error is not None:
                error = f"{error}\n\nLocal LeRobot import/load error was:\n{self._startup_error}"
            raise RuntimeError(error)
        return response

    def reset(self) -> None:
        self._request({"cmd": "reset"})

    def reset_envs(self, env_ids: Optional[Sequence[int]] = None) -> None:
        self._request({"cmd": "reset_envs", "env_ids": None if env_ids is None else [int(x) for x in env_ids]})

    def append_observation(
        self,
        state: Any,
        mask_rgb: Any,
        masked_depth_rgb: Any,
        front_mask_rgb: Any = None,
        front_masked_depth_rgb: Any = None,
    ) -> None:
        self._request(
            {
                "cmd": "append_observation",
                "state": np.asarray(state, dtype=np.float32),
                "mask_rgb": np.asarray(mask_rgb),
                "masked_depth_rgb": np.asarray(masked_depth_rgb),
                "front_mask_rgb": None if front_mask_rgb is None else np.asarray(front_mask_rgb),
                "front_masked_depth_rgb": None
                if front_masked_depth_rgb is None
                else np.asarray(front_masked_depth_rgb),
            }
        )

    def append_observation_for_env(
        self,
        env_id: int,
        state: Any,
        mask_rgb: Any,
        masked_depth_rgb: Any,
        front_mask_rgb: Any = None,
        front_masked_depth_rgb: Any = None,
    ) -> None:
        self._request(
            {
                "cmd": "append_observation_for_env",
                "env_id": int(env_id),
                "state": np.asarray(state, dtype=np.float32),
                "mask_rgb": np.asarray(mask_rgb),
                "masked_depth_rgb": np.asarray(masked_depth_rgb),
                "front_mask_rgb": None if front_mask_rgb is None else np.asarray(front_mask_rgb),
                "front_masked_depth_rgb": None
                if front_masked_depth_rgb is None
                else np.asarray(front_masked_depth_rgb),
            }
        )

    def sample_action_chunk(self, noise: Optional[torch.Tensor] = None) -> None:
        self._request(
            {
                "cmd": "sample_action_chunk",
                "noise": None if noise is None else noise.detach().cpu(),
            }
        )

    def act(
        self,
        state: Any,
        mask_rgb: Any,
        masked_depth_rgb: Any,
        front_mask_rgb: Any = None,
        front_masked_depth_rgb: Any = None,
    ) -> np.ndarray:
        response = self._request(
            {
                "cmd": "act",
                "state": np.asarray(state, dtype=np.float32),
                "mask_rgb": np.asarray(mask_rgb),
                "masked_depth_rgb": np.asarray(masked_depth_rgb),
                "front_mask_rgb": None if front_mask_rgb is None else np.asarray(front_mask_rgb),
                "front_masked_depth_rgb": None
                if front_masked_depth_rgb is None
                else np.asarray(front_masked_depth_rgb),
            }
        )
        return np.asarray(response["action"], dtype=np.float32)

    def act_batch(
        self,
        env_ids: Sequence[int],
        states: Any,
        mask_rgbs: Any,
        masked_depth_rgbs: Any,
        front_mask_rgbs: Any = None,
        front_masked_depth_rgbs: Any = None,
    ) -> np.ndarray:
        response = self._request(
            {
                "cmd": "act_batch",
                "env_ids": [int(env_id) for env_id in env_ids],
                "states": np.asarray(states, dtype=np.float32),
                "mask_rgbs": np.asarray(mask_rgbs),
                "masked_depth_rgbs": np.asarray(masked_depth_rgbs),
                "front_mask_rgbs": None if front_mask_rgbs is None else np.asarray(front_mask_rgbs),
                "front_masked_depth_rgbs": None
                if front_masked_depth_rgbs is None
                else np.asarray(front_masked_depth_rgbs),
            }
        )
        return np.asarray(response["actions"], dtype=np.float32)

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None and proc.stdout is not None:
                    _write_pickle_message(proc.stdin, {"cmd": "close"})
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                proc.terminate()
        self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
