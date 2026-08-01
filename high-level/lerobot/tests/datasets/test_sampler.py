#!/usr/bin/env python

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
from datasets import Dataset
import pytest
import torch

from lerobot.datasets.push_dataset_to_hub.utils import calculate_episode_data_index
from lerobot.datasets.sampler import EpisodeAwareSampler, make_recovery_sampler
from lerobot.datasets.utils import (
    hf_transform_to_torch,
)


def test_drop_n_first_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], drop_n_first_frames=1)
    assert sampler.indices == [1, 4, 5]
    assert len(sampler) == 3
    assert list(sampler) == [1, 4, 5]


def test_drop_n_last_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], drop_n_last_frames=1)
    assert sampler.indices == [0, 3, 4]
    assert len(sampler) == 3
    assert list(sampler) == [0, 3, 4]


def test_episode_indices_to_use():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(
        episode_data_index["from"], episode_data_index["to"], episode_indices_to_use=[0, 2]
    )
    assert sampler.indices == [0, 1, 3, 4, 5]
    assert len(sampler) == 5
    assert list(sampler) == [0, 1, 3, 4, 5]


def test_shuffle():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], shuffle=False)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert list(sampler) == [0, 1, 2, 3, 4, 5]
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], shuffle=True)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert set(sampler) == {0, 1, 2, 3, 4, 5}


def test_shuffle_generator_isolated_from_global_rng():
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    sampler_a = EpisodeAwareSampler([0], [10], shuffle=True, generator=generator_a)
    sampler_b = EpisodeAwareSampler([0], [10], shuffle=True, generator=generator_b)

    torch.manual_seed(999)
    _ = torch.rand(1000)
    order_a = list(sampler_a)
    torch.manual_seed(1)
    _ = torch.rand(3)
    order_b = list(sampler_b)
    assert order_a == order_b


class _RecoverySamplerDataset:
    def __init__(self, values):
        self.features = {"aux.is_recovery": {}}
        self.hf_dataset = {"aux.is_recovery": values}
        self._length = len(values)

    def __len__(self):
        return self._length


def test_recovery_sampler_assigns_requested_probability_mass():
    dataset = _RecoverySamplerDataset([[0.0], [0.0], [1.0], [1.0]])
    sampler, stats = make_recovery_sampler(
        dataset,
        recovery_sampling_ratio=0.8,
        recovery_feature="aux.is_recovery",
        seed=123,
    )

    assert sampler is not None
    assert sampler.weights[:2].sum().item() == pytest.approx(0.2)
    assert sampler.weights[2:].sum().item() == pytest.approx(0.8)
    assert stats["expert_frames"] == 2
    assert stats["recovery_frames"] == 2
    assert stats["expert_ratio"] == pytest.approx(0.2)
    assert stats["recovery_ratio"] == pytest.approx(0.8)


def test_recovery_sampler_requires_indicator_feature():
    dataset = _RecoverySamplerDataset([[0.0], [1.0]])
    dataset.features = {}

    sampler, stats = make_recovery_sampler(dataset, recovery_sampling_ratio=0.8)

    assert sampler is None
    assert stats["enabled"] is False
    assert "missing feature" in stats["reason"]
