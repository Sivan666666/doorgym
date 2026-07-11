#!/usr/bin/env python
from __future__ import annotations

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterator

import torch
from torch.utils.data import WeightedRandomSampler


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def _as_1d_tensor(values, *, dtype: torch.dtype) -> torch.Tensor:
    """Convert HuggingFace-dataset column values to a flat torch tensor."""
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype).reshape(-1)
    if hasattr(values, "to_list"):
        values = values.to_list()
    if isinstance(values, list) and len(values) > 0 and isinstance(values[0], torch.Tensor):
        return torch.stack([v.reshape(-1)[0] for v in values]).to(dtype=dtype).reshape(-1)
    return torch.as_tensor(values, dtype=dtype).reshape(-1)


def _as_int_list(values) -> list[int]:
    if hasattr(values, "to_list"):
        values = values.to_list()
    return [int(v.item() if isinstance(v, torch.Tensor) else v) for v in values]


def _episode_allowed_mask(dataset, drop_n_first_frames: int = 0, drop_n_last_frames: int = 0) -> torch.Tensor:
    """Return relative dataset rows that are valid after optional episode-boundary dropping."""
    num_rows = len(dataset)
    if drop_n_first_frames <= 0 and drop_n_last_frames <= 0:
        return torch.ones(num_rows, dtype=torch.bool)

    abs_indices = _as_1d_tensor(dataset.hf_dataset["index"], dtype=torch.long)
    episode_indices = _as_1d_tensor(dataset.hf_dataset["episode_index"], dtype=torch.long)
    starts = _as_int_list(dataset.meta.episodes["dataset_from_index"])
    ends = _as_int_list(dataset.meta.episodes["dataset_to_index"])

    allowed = torch.zeros(num_rows, dtype=torch.bool)
    for ep_idx in torch.unique(episode_indices).tolist():
        ep_idx = int(ep_idx)
        if ep_idx < 0 or ep_idx >= len(starts):
            continue
        start = starts[ep_idx] + max(0, int(drop_n_first_frames))
        end = ends[ep_idx] - max(0, int(drop_n_last_frames))
        if end <= start:
            continue
        in_episode = episode_indices == ep_idx
        allowed |= in_episode & (abs_indices >= start) & (abs_indices < end)
    return allowed


def make_keyframe_window_sampler(
    dataset,
    *,
    keyframe_sampling_ratio: float,
    weight_feature: str = "loss.action_weight",
    keyframe_threshold: float = 1.0,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
    seed: int | None = None,
) -> tuple[WeightedRandomSampler | None, dict]:
    """Build a sampler that draws a fixed probability mass from keyframe-adjacent frames.

    The keyframe window is detected from ``weight_feature``. For Door ACT datasets this is
    ``loss.action_weight``: frames inside keyframe windows have values greater than 1, and
    ordinary frames are 1. A ratio of 0.2 means roughly 20% of sampled chunk anchors come
    from keyframe windows and 80% from ordinary frames. Use ratio=0 to keep the original
    DataLoader sampling behavior.
    """
    ratio = float(keyframe_sampling_ratio)
    if ratio <= 0.0:
        return None, {"enabled": False, "reason": "ratio<=0"}
    if ratio >= 1.0:
        ratio = 1.0

    features = getattr(dataset, "features", {})
    if weight_feature not in features:
        return None, {
            "enabled": False,
            "reason": f"missing feature {weight_feature!r}",
            "weight_feature": weight_feature,
        }

    if hasattr(dataset, "_ensure_hf_dataset_loaded"):
        dataset._ensure_hf_dataset_loaded()

    try:
        weight_values = _as_1d_tensor(dataset.hf_dataset[weight_feature], dtype=torch.float64)
    except Exception as exc:
        return None, {
            "enabled": False,
            "reason": f"failed to read feature {weight_feature!r}: {exc}",
            "weight_feature": weight_feature,
        }

    if weight_values.numel() != len(dataset):
        return None, {
            "enabled": False,
            "reason": f"feature length {weight_values.numel()} != dataset length {len(dataset)}",
            "weight_feature": weight_feature,
        }

    allowed = _episode_allowed_mask(
        dataset,
        drop_n_first_frames=drop_n_first_frames,
        drop_n_last_frames=drop_n_last_frames,
    )
    keyframe_mask = allowed & (weight_values > float(keyframe_threshold))
    ordinary_mask = allowed & ~keyframe_mask

    allowed_count = int(allowed.sum().item())
    keyframe_count = int(keyframe_mask.sum().item())
    ordinary_count = int(ordinary_mask.sum().item())
    if allowed_count <= 0:
        return None, {
            "enabled": False,
            "reason": "no eligible frames after episode-boundary dropping",
            "weight_feature": weight_feature,
        }
    if keyframe_count <= 0:
        return None, {
            "enabled": False,
            "reason": "no keyframe-window frames found",
            "weight_feature": weight_feature,
            "eligible_frames": allowed_count,
        }

    sample_weights = torch.zeros(len(dataset), dtype=torch.double)
    if ordinary_count > 0 and ratio < 1.0:
        sample_weights[ordinary_mask] = (1.0 - ratio) / ordinary_count
    if keyframe_count > 0:
        keyframe_mass = ratio if ordinary_count > 0 else 1.0
        sample_weights[keyframe_mask] = keyframe_mass / keyframe_count

    # If ratio==1, ordinary rows intentionally get zero probability. If numerical or
    # degenerate conditions emptied weights, fall back safely.
    if float(sample_weights.sum().item()) <= 0.0:
        sample_weights[allowed] = 1.0 / allowed_count

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=allowed_count,
        replacement=True,
        generator=generator,
    )
    stats = {
        "enabled": True,
        "ratio": ratio,
        "ordinary_ratio": 1.0 - ratio if ordinary_count > 0 else 0.0,
        "keyframe_ratio": ratio if ordinary_count > 0 else 1.0,
        "eligible_frames": allowed_count,
        "ordinary_frames": ordinary_count,
        "keyframe_window_frames": keyframe_count,
        "weight_feature": weight_feature,
        "keyframe_threshold": float(keyframe_threshold),
        "drop_n_first_frames": int(drop_n_first_frames),
        "drop_n_last_frames": int(drop_n_last_frames),
    }
    return sampler, stats


def make_recovery_sampler(
    dataset,
    *,
    recovery_sampling_ratio: float,
    recovery_feature: str = "aux.is_recovery",
    recovery_threshold: float = 0.5,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
    seed: int | None = None,
) -> tuple[WeightedRandomSampler | None, dict]:
    """Draw fixed probability mass from verified recovery versus expert frames.

    ``recovery_sampling_ratio=0.8`` assigns 80% total sampling probability to
    rows where ``recovery_feature > recovery_threshold`` and 20% to all other
    eligible rows. This is frame/chunk-anchor sampling with replacement; it
    does not duplicate videos or alter the behavior-cloning loss.
    """
    sampler, stats = make_keyframe_window_sampler(
        dataset,
        keyframe_sampling_ratio=recovery_sampling_ratio,
        weight_feature=recovery_feature,
        keyframe_threshold=recovery_threshold,
        drop_n_first_frames=drop_n_first_frames,
        drop_n_last_frames=drop_n_last_frames,
        seed=seed,
    )
    stats = dict(stats)
    stats["sampler_type"] = "recovery"
    if stats.get("enabled"):
        stats["recovery_ratio"] = stats.pop("keyframe_ratio")
        stats["expert_ratio"] = stats.pop("ordinary_ratio")
        stats["recovery_frames"] = stats.pop("keyframe_window_frames")
        stats["expert_frames"] = stats.pop("ordinary_frames")
        stats["recovery_feature"] = stats.pop("weight_feature")
        stats["recovery_threshold"] = stats.pop("keyframe_threshold")
    return sampler, stats
