#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from safetensors import safe_open
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import _transformers_available


DINOV2_MODEL_ALIASES = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
}

DEFM_MODEL_ALIASES = {
    "defm-vit-l14": "defm_vit_l14",
    "defm_vit_l14": "defm_vit_l14",
    "defm-vit-l/14": "defm_vit_l14",
}

DEFM_DEPTH_MAX_C1 = 100.0
DEFM_DEPTH_MAX_C2 = 9.0
DEFM_MEAN = [0.248880, 0.495620, 0.492858]
DEFM_STD = [0.139357, 0.271314, 0.297177]


def is_dinov2_backbone(vision_backbone: str) -> bool:
    name = str(vision_backbone).lower()
    return name.startswith("dinov2") or name.startswith("facebook/dinov2")


def is_defm_backbone(vision_backbone: str) -> bool:
    return str(vision_backbone).lower() in DEFM_MODEL_ALIASES


def resolve_dinov2_model_name(vision_backbone: str) -> str:
    name = str(vision_backbone)
    return DINOV2_MODEL_ALIASES.get(name.lower(), name)


def resolve_defm_model_name(vision_backbone: str) -> str:
    return DEFM_MODEL_ALIASES[str(vision_backbone).lower()]


class ACTDINOv2Backbone(nn.Module):
    """DINOv2 wrapper that exposes a ResNet-like {"feature_map": tensor} interface for ACT."""

    def __init__(
        self,
        vision_backbone: str,
        image_size: int,
        feature_grid_size: int,
        normalize_inputs: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if not _transformers_available:
            raise ImportError(
                "DINOv2 ACT backbone requires the `transformers` package. "
                "Install the Door DP requirements or run in the b1z1_lerobot environment."
            )
        from transformers import AutoModel

        self.model_name = resolve_dinov2_model_name(vision_backbone)
        self.image_size = int(image_size)
        self.feature_grid_size = int(feature_grid_size)
        self.normalize_inputs = bool(normalize_inputs)
        self.freeze = bool(freeze)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.out_channels = int(getattr(self.model.config, "hidden_size"))
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "ACTDINOv2Backbone":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.normalize_inputs:
            x = (x - self.image_mean.to(dtype=x.dtype)) / self.image_std.to(dtype=x.dtype)
        context: Any
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            outputs = self.model(pixel_values=x)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        batch_size, num_tokens, channels = patch_tokens.shape
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                f"DINOv2 patch token count must be square to form a feature map; got {num_tokens} tokens."
            )
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        if self.feature_grid_size != grid_size:
            feature_map = F.adaptive_avg_pool2d(feature_map, (self.feature_grid_size, self.feature_grid_size))
        return {"feature_map": feature_map}


class ACTDeFMBackbone(nn.Module):
    """DeFM depth backbone exposed through ACT's {"feature_map": tensor} interface."""

    def __init__(
        self,
        vision_backbone: str,
        image_size: int,
        patch_size: int,
        feature_grid_size: int,
        depth_lower: float,
        depth_far: float,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = resolve_defm_model_name(vision_backbone)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.feature_grid_size = int(feature_grid_size)
        self.depth_lower = float(depth_lower)
        self.depth_far = float(depth_far)
        self.freeze = bool(freeze)
        hub_kwargs: dict[str, Any] = {"pretrained": bool(pretrained)}
        if pretrained_path:
            hub_kwargs["pretrained_path"] = str(pretrained_path)
        try:
            self.model = torch.hub.load(
                "leggedrobotics/defm:main",
                self.model_name,
                trust_repo=True,
                **hub_kwargs,
            )
        except TypeError as exc:
            if "trust_repo" not in str(exc):
                raise
            self.model = torch.hub.load("leggedrobotics/defm:main", self.model_name, **hub_kwargs)
        self.out_channels = self._infer_out_channels()
        self.register_buffer(
            "defm_mean",
            torch.tensor(DEFM_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "defm_std",
            torch.tensor(DEFM_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model.eval()

    def _infer_out_channels(self) -> int:
        for attr in ("embed_dim", "num_features"):
            value = getattr(self.model, attr, None)
            if value is not None:
                return int(value)
        norm = getattr(self.model, "norm", None)
        normalized_shape = getattr(norm, "normalized_shape", None)
        if normalized_shape:
            return int(normalized_shape[0])
        raise ValueError("Could not infer DeFM output channel count from the loaded model.")

    def train(self, mode: bool = True) -> "ACTDeFMBackbone":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _depth_visual_to_metric(self, x: Tensor) -> Tensor:
        gray = x[:, :1].to(dtype=torch.float32).clamp(0.0, 1.0)
        return torch.where(
            gray > 0.0,
            gray * (self.depth_far - self.depth_lower) + self.depth_lower,
            torch.zeros_like(gray),
        )

    def _preprocess_depth(self, x: Tensor) -> Tensor:
        depth = self._depth_visual_to_metric(x)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.clamp(depth, min=0.0, max=DEFM_DEPTH_MAX_C1)
        log_depth = torch.log1p(depth)
        c1 = log_depth / math.log1p(DEFM_DEPTH_MAX_C1)
        c2 = torch.clamp(log_depth / math.log1p(DEFM_DEPTH_MAX_C2), min=0.0, max=1.0)
        batch_size = depth.shape[0]
        flat = log_depth.reshape(batch_size, -1)
        min_log = flat.min(dim=1).values.view(batch_size, 1, 1, 1)
        max_log = flat.max(dim=1).values.view(batch_size, 1, 1, 1)
        denom = max_log - min_log
        denom_safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
        c3 = (log_depth - min_log) / denom_safe
        c3 = torch.where(denom > 0.0, c3, torch.zeros_like(c3))
        x = torch.cat([c1, c2, c3], dim=1)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (x - self.defm_mean.to(dtype=x.dtype)) / self.defm_std.to(dtype=x.dtype)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self._preprocess_depth(x)
        context: Any = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            outputs = self.model.get_intermediate_layers(
                x,
                n=1,
                reshape=True,
                return_class_token=True,
            )
        feature_map = outputs[0][0]
        if self.feature_grid_size != feature_map.shape[-1] or self.feature_grid_size != feature_map.shape[-2]:
            feature_map = F.adaptive_avg_pool2d(feature_map, (self.feature_grid_size, self.feature_grid_size))
        return {"feature_map": feature_map}


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    @staticmethod
    def _local_saved_keys(pretrained_name_or_path) -> set[str]:
        """Read local checkpoint keys without constructing a potentially incompatible model."""
        path = Path(pretrained_name_or_path).expanduser()
        model_file = path / "model.safetensors" if path.is_dir() else path
        if not model_file.is_file():
            return set()
        with safe_open(str(model_file), framework="pt", device="cpu") as handle:
            return set(handle.keys())

    @staticmethod
    def _legacy_point_cloud_mode(saved_keys: set[str]) -> str | None:
        """Identify the two early PointCloud ACT encoders from their state-dict layout."""
        prefix = "model.point_cloud_encoder."
        has_early_point_mlp = any(key.startswith(prefix + "point_mlp.") for key in saved_keys)
        if not has_early_point_mlp:
            return None
        if any(key.startswith(prefix + "projection.") for key in saved_keys):
            return "dp3_global_legacy"
        if any(key.startswith(prefix + "local_projection.") for key in saved_keys):
            return "obsbench_local_legacy"
        return None

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict=False, **kwargs):
        """Load ACT while preserving compatibility with opt-in heads and early PCD encoders."""
        saved_keys = cls._local_saved_keys(pretrained_name_or_path)
        legacy_point_cloud_mode = cls._legacy_point_cloud_mode(saved_keys)
        if legacy_point_cloud_mode is not None:
            if config is not None:
                configured_mode = str(getattr(config, "point_cloud_encoder_mode", ""))
                if configured_mode in {"dp3_global", "obsbench_local", "obsbench_post_pointnet"}:
                    config.point_cloud_encoder_mode = legacy_point_cloud_mode
            else:
                cli_overrides = list(kwargs.pop("cli_overrides", []))
                cli_overrides.append(f"--point_cloud_encoder_mode={legacy_point_cloud_mode}")
                kwargs["cli_overrides"] = cli_overrides
        policy = super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            strict=strict,
            **kwargs,
        )
        optional_warmstart = bool(getattr(policy.config, "end_signal_prediction", False)) or bool(
            getattr(policy.config, "interaction_state_conditioning", False)
        )
        if strict or not optional_warmstart:
            return policy
        if not saved_keys:
            return policy
        expected_keys = set(policy.state_dict().keys())
        missing = expected_keys - saved_keys
        unexpected = saved_keys - expected_keys
        allowed_missing = set()
        if bool(getattr(policy.config, "end_signal_prediction", False)):
            allowed_missing.update({"model.end_signal_head.weight", "model.end_signal_head.bias"})
        if bool(getattr(policy.config, "interaction_state_conditioning", False)):
            allowed_missing.update(
                key
                for key in expected_keys
                if key.startswith("model.interaction_state_head.")
                or key.startswith("model.interaction_state_chunk_head.")
                or key.startswith("model.interaction_conditioner.")
            )
        if missing - allowed_missing or unexpected:
            raise RuntimeError(
                "Old-checkpoint warm start only permits parameters from explicitly enabled new ACT heads "
                f"parameters to be missing. missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        return policy

    def __init__(
        self,
        config: ACTConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)
        self._last_separate_auxiliary_loss: Tensor | None = None

        if config.interaction_state_probe_freeze_main:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(
                    name.startswith("interaction_state_head.")
                    or name.startswith("interaction_state_chunk_head.")
                )

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    @torch.no_grad()
    def predict_action_chunk_with_end_signal(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        """Predict motion actions and an optional future-aligned end probability chunk."""
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions = self.model(batch)[0]
        logits = self.model._last_end_signal_logits
        probabilities = None if logits is None else torch.sigmoid(logits)
        return actions, probabilities

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        self._last_separate_auxiliary_loss = None
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        valid = ~batch["action_is_pad"].unsqueeze(-1)
        valid_f = valid.to(dtype=actions_hat.dtype)
        l1_per_elem = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_elem_count = valid_f.expand_as(l1_per_elem).sum(dim=(1, 2))
        l1_per_sample = (l1_per_elem * valid_f).sum(dim=(1, 2)) / torch.clamp(valid_elem_count, min=1.0)
        action_loss_weight = (
            batch.get("loss.action_weight") if bool(getattr(self.config, "use_action_loss_weight", True)) else None
        )
        action_weight = None
        if action_loss_weight is not None:
            action_weight = self._action_loss_weight_to_timestep_weight(
                action_loss_weight,
                target_horizon=batch[ACTION].shape[1],
                device=l1_per_elem.device,
                dtype=l1_per_elem.dtype,
            )
            # Per-timestep keyframe-consistent weighting:
            #   L = Σ_{b,h,d} w_{b,h} * valid_{b,h} * |a_hat - a|
            #       / Σ_{b,h,d} w_{b,h} * valid_{b,h}
            # New datasets provide w as (B, H, 1), aligned with the action chunk.
            # Old datasets that provide (B, 1) still work by broadcasting the scalar
            # anchor weight across the whole chunk.
            weighted_valid = valid_f * action_weight
            weighted_valid_elem = weighted_valid.expand_as(l1_per_elem)
            weighted_elem_sum = (l1_per_elem * weighted_valid).sum(dim=(1, 2))
            weighted_elem_count = weighted_valid_elem.sum(dim=(1, 2))
            l1_per_sample = weighted_elem_sum / torch.clamp(weighted_elem_count, min=1.0)
            l1_loss = weighted_elem_sum.sum() / torch.clamp(weighted_elem_count.sum(), min=1.0)
            mean_action_loss_weight = (
                (action_weight * valid_f).sum() / torch.clamp(valid_f.sum(), min=1.0)
            )
        else:
            l1_loss = l1_per_sample.mean()
            mean_action_loss_weight = None

        loss_dict = {"l1_loss": l1_loss.item()}
        if mean_action_loss_weight is not None:
            loss_dict["action_loss_weight_mean"] = float(mean_action_loss_weight.detach().cpu())
        if self.model._last_camera_gates is not None:
            loss_dict["camera_gate_front"] = float(self.model._last_camera_gates[:, 0].mean().cpu())
            loss_dict["camera_gate_wrist"] = float(self.model._last_camera_gates[:, 1].mean().cpu())
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            kld_per_sample = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1)
            mean_kld = kld_per_sample.mean()
            loss_dict["kld_loss"] = mean_kld.item()
            per_sample_loss = l1_per_sample + kld_per_sample * self.config.kl_weight
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            per_sample_loss = l1_per_sample
            loss = l1_loss

        handle_latent_loss = self.model._last_handle_latent_loss
        if handle_latent_loss is not None:
            handle_weight = float(self.config.handle_latent_loss_weight)
            loss = loss + handle_weight * handle_latent_loss
            per_sample_loss = per_sample_loss + handle_weight * handle_latent_loss
            loss_dict["handle_latent_loss"] = float(handle_latent_loss.detach().cpu())
            loss_dict["handle_latent_loss_weight"] = handle_weight
            if self.model._last_handle_latent_front_loss is not None:
                loss_dict["handle_latent_front_loss"] = float(
                    self.model._last_handle_latent_front_loss.detach().cpu()
                )
            if self.model._last_handle_latent_wrist_loss is not None:
                loss_dict["handle_latent_wrist_loss"] = float(
                    self.model._last_handle_latent_wrist_loss.detach().cpu()
                )
            if self.model._last_handle_latent_valid_count is not None:
                loss_dict["handle_latent_valid_count"] = float(
                    self.model._last_handle_latent_valid_count.detach().cpu()
                )

        if self.config.interaction_state_conditioning:
            logits = self.model._last_interaction_state_logits
            probabilities = self.model._last_interaction_state_probabilities
            if logits is None or probabilities is None or logits.shape[-1] != 3:
                raise RuntimeError("ACT interaction-state conditioning did not produce three-state predictions.")
            target_keys = (
                str(self.config.interaction_contact_target_key),
                str(self.config.interaction_handle_target_key),
                str(self.config.interaction_door_target_key),
            )
            missing = [key for key in target_keys if key not in batch]
            if missing:
                raise KeyError(
                    f"ACT interaction-state conditioning requires {missing}. "
                    "Re-convert the dataset with --add_interaction_state."
                )
            mode = str(self.config.interaction_state_prediction_mode)
            if mode == "encoder_current":
                if logits.ndim != 2:
                    raise RuntimeError(
                        "encoder_current interaction mode expects Bx3 predictions; "
                        f"got {tuple(logits.shape)}."
                    )
                targets = []
                for key in target_keys:
                    target = (
                        batch[key]
                        .to(device=logits.device, dtype=logits.dtype)
                        .reshape(logits.shape[0], -1)[:, 0]
                    )
                    if (
                        not torch.all(torch.isfinite(target))
                        or torch.any(target < 0.0)
                        or torch.any(target > 1.0)
                    ):
                        raise ValueError(f"Interaction target {key!r} must contain finite values in [0, 1].")
                    targets.append(target)
                contact_per_sample = F.binary_cross_entropy_with_logits(
                    logits[:, 0], targets[0], reduction="none"
                )
                handle_per_sample = F.smooth_l1_loss(
                    probabilities[:, 1], targets[1], reduction="none"
                )
                door_per_sample = F.smooth_l1_loss(
                    probabilities[:, 2], targets[2], reduction="none"
                )
                contact_loss = contact_per_sample.mean()
                handle_loss = handle_per_sample.mean()
                door_loss = door_per_sample.mean()
                probability_means = probabilities.mean(dim=0)
            elif mode == "decoder_chunk":
                if logits.ndim != 3:
                    raise RuntimeError(
                        "decoder_chunk interaction mode expects BxHx3 predictions; "
                        f"got {tuple(logits.shape)}."
                    )
                batch_size, horizon, _ = logits.shape
                targets = []
                valid_interaction = ~batch["action_is_pad"].to(device=logits.device, dtype=torch.bool)
                if valid_interaction.shape != (batch_size, horizon):
                    raise ValueError(
                        "Interaction/action padding shape mismatch: "
                        f"action_is_pad={tuple(valid_interaction.shape)}, logits={tuple(logits.shape)}."
                    )
                for key in target_keys:
                    target = batch[key].to(device=logits.device, dtype=logits.dtype)
                    if target.ndim == 3 and target.shape[-1] == 1:
                        target = target.squeeze(-1)
                    if target.shape != (batch_size, horizon):
                        raise ValueError(
                            f"decoder_chunk interaction target {key!r} must have shape "
                            f"(B, H, 1) or (B, H); got {tuple(batch[key].shape)} for "
                            f"prediction shape {tuple(logits.shape)}. Ensure future delta timestamps "
                            "are configured for all three interaction targets."
                        )
                    if (
                        not torch.all(torch.isfinite(target))
                        or torch.any(target < 0.0)
                        or torch.any(target > 1.0)
                    ):
                        raise ValueError(f"Interaction target {key!r} must contain finite values in [0, 1].")
                    target_pad_key = f"{key}_is_pad"
                    if target_pad_key in batch:
                        target_is_pad = batch[target_pad_key].to(device=logits.device, dtype=torch.bool)
                        if target_is_pad.ndim == 3 and target_is_pad.shape[-1] == 1:
                            target_is_pad = target_is_pad.squeeze(-1)
                        if target_is_pad.shape != (batch_size, horizon):
                            raise ValueError(
                                f"Interaction target padding {target_pad_key!r} has shape "
                                f"{tuple(batch[target_pad_key].shape)}; expected (B, H)."
                            )
                        valid_interaction = valid_interaction & ~target_is_pad
                    targets.append(target)

                contact_per_timestep = F.binary_cross_entropy_with_logits(
                    logits[..., 0], targets[0], reduction="none"
                )
                handle_per_timestep = F.smooth_l1_loss(
                    probabilities[..., 1], targets[1], reduction="none"
                )
                door_per_timestep = F.smooth_l1_loss(
                    probabilities[..., 2], targets[2], reduction="none"
                )
                valid_interaction_f = valid_interaction.to(dtype=logits.dtype)
                count_per_sample = valid_interaction_f.sum(dim=1)

                def masked_per_sample(value: Tensor) -> Tensor:
                    return (value * valid_interaction_f).sum(dim=1) / torch.clamp(
                        count_per_sample, min=1.0
                    )

                contact_per_sample = masked_per_sample(contact_per_timestep)
                handle_per_sample = masked_per_sample(handle_per_timestep)
                door_per_sample = masked_per_sample(door_per_timestep)
                valid_count = torch.clamp(valid_interaction_f.sum(), min=1.0)
                contact_loss = (contact_per_timestep * valid_interaction_f).sum() / valid_count
                handle_loss = (handle_per_timestep * valid_interaction_f).sum() / valid_count
                door_loss = (door_per_timestep * valid_interaction_f).sum() / valid_count
                probability_means = (
                    probabilities * valid_interaction_f.unsqueeze(-1)
                ).sum(dim=(0, 1)) / valid_count
            else:
                raise RuntimeError(f"Unsupported interaction_state_prediction_mode={mode!r}.")

            contact_weight = float(self.config.interaction_contact_loss_weight)
            handle_weight = float(self.config.interaction_handle_loss_weight)
            door_weight = float(self.config.interaction_door_loss_weight)
            interaction_per_sample = (
                contact_weight * contact_per_sample
                + handle_weight * handle_per_sample
                + door_weight * door_per_sample
            )
            interaction_loss = (
                contact_weight * contact_loss
                + handle_weight * handle_loss
                + door_weight * door_loss
            )
            if (
                self.config.interaction_state_probe_only
                and self.config.interaction_state_probe_separate_backward
            ):
                self._last_separate_auxiliary_loss = interaction_loss
            else:
                per_sample_loss = per_sample_loss + interaction_per_sample
                loss = loss + interaction_loss
            loss_dict.update(
                {
                    "interaction_contact_loss": float(contact_loss.detach().cpu()),
                    "interaction_handle_loss": float(handle_loss.detach().cpu()),
                    "interaction_door_loss": float(door_loss.detach().cpu()),
                    "interaction_contact_probability": float(probability_means[0].detach().cpu()),
                    "interaction_handle_progress": float(probability_means[1].detach().cpu()),
                    "interaction_door_progress": float(probability_means[2].detach().cpu()),
                }
            )

        if self.config.end_signal_prediction:
            target_key = str(self.config.end_signal_target_key)
            if target_key not in batch:
                raise KeyError(
                    f"ACT end-signal prediction requires future-aligned target {target_key!r} in the batch."
                )
            logits = self.model._last_end_signal_logits
            if logits is None:
                raise RuntimeError("ACT end-signal head is enabled but did not produce logits.")
            targets = batch[target_key].to(device=logits.device, dtype=logits.dtype)
            if targets.ndim == 2:
                targets = targets.unsqueeze(-1)
            if targets.shape != logits.shape:
                raise ValueError(
                    f"End-signal target/logit shape mismatch: target={tuple(targets.shape)} "
                    f"logits={tuple(logits.shape)}."
                )
            end_per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            end_valid = valid_f.to(dtype=end_per_elem.dtype)
            end_count_per_sample = end_valid.sum(dim=(1, 2))
            end_per_sample = (end_per_elem * end_valid).sum(dim=(1, 2)) / torch.clamp(
                end_count_per_sample, min=1.0
            )
            end_loss = (end_per_elem * end_valid).sum() / torch.clamp(end_valid.sum(), min=1.0)
            end_weight = float(self.config.end_signal_loss_weight)
            loss = loss + end_weight * end_loss
            per_sample_loss = per_sample_loss + end_weight * end_per_sample
            loss_dict["end_signal_loss"] = float(end_loss.detach().cpu())
            loss_dict["end_signal_loss_weight"] = end_weight
            loss_dict["end_signal_probability_mean"] = float(torch.sigmoid(logits).mean().detach().cpu())

        if reduction == "none":
            return per_sample_loss, loss_dict
        if reduction != "mean":
            raise ValueError(f"Unsupported ACT loss reduction={reduction!r}; expected 'mean' or 'none'.")
        return loss, loss_dict

    @staticmethod
    def _action_loss_weight_to_timestep_weight(
        action_loss_weight: Tensor,
        *,
        target_horizon: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Normalize action-loss weights to shape (B, H, 1).

        Supported inputs:
        - (B,) or (B, 1): legacy scalar anchor weight; broadcast to all H steps.
        - (B, H): per-timestep chunk weights.
        - (B, H, 1) or (B, H, D): per-timestep weights; D is reduced to its first channel.

        The output is clamped to non-negative values because it is used as a loss
        weight, not as a signed target.
        """
        weight = action_loss_weight.to(device=device, dtype=dtype)
        batch_size = int(weight.shape[0])
        horizon = int(target_horizon)

        if weight.ndim == 1:
            weight = weight.reshape(batch_size, 1, 1)
        elif weight.ndim == 2:
            if int(weight.shape[1]) == horizon:
                weight = weight.unsqueeze(-1)
            else:
                weight = weight.reshape(batch_size, -1)[:, :1].reshape(batch_size, 1, 1)
        else:
            # Keep the first per-timestep weight channel.  If the horizon axis is
            # not present, fall back to a scalar legacy weight.
            if int(weight.shape[1]) == horizon:
                weight = weight.reshape(batch_size, horizon, -1)[..., :1]
            else:
                weight = weight.reshape(batch_size, -1)[:, :1].reshape(batch_size, 1, 1)

        if int(weight.shape[1]) == 1 and horizon != 1:
            weight = weight.expand(batch_size, horizon, 1)
        elif int(weight.shape[1]) != horizon:
            weight = weight[:, :1].expand(batch_size, horizon, 1)

        return torch.clamp(weight, min=0.0)


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACTCameraInputGating(nn.Module):
    """Predict identity-initialized front/wrist gates from visual features and robot state."""

    def __init__(
        self,
        dim_model: int,
        robot_state_dim: int,
        hidden_dim: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.mlp = nn.Sequential(
            nn.Linear(2 * int(dim_model) + int(robot_state_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        # Zero logits give softmax([0, 0]) = [0.5, 0.5]. Multiplying by 2
        # initializes both gates to exactly 1, preserving original ACT inputs.
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        front_features: Tensor,
        wrist_features: Tensor,
        robot_state: Tensor,
    ) -> Tensor:
        if front_features.ndim != 4 or wrist_features.ndim != 4:
            raise ValueError(
                "Camera gating expects projected feature maps shaped (batch, channels, height, width)."
            )
        if front_features.shape[:2] != wrist_features.shape[:2]:
            raise ValueError(
                "Front and wrist projected feature maps must have matching batch/channel dimensions. "
                f"Got front={tuple(front_features.shape)} and wrist={tuple(wrist_features.shape)}."
            )
        batch_size = front_features.shape[0]
        robot_state = robot_state.to(
            device=front_features.device,
            dtype=front_features.dtype,
        ).reshape(batch_size, -1)
        front_global = F.adaptive_avg_pool2d(front_features, output_size=1).flatten(1)
        wrist_global = F.adaptive_avg_pool2d(wrist_features, output_size=1).flatten(1)
        logits = self.mlp(torch.cat([front_global, wrist_global, robot_state], dim=-1))
        return 2.0 * torch.softmax(logits / self.temperature, dim=-1)


def _parse_plucker_encoder_channels(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, str):
        channels = [int(x.strip()) for x in value.split(",") if x.strip()]
    else:
        channels = [int(x) for x in value]
    if not channels or any(channel <= 0 for channel in channels):
        raise ValueError(f"Invalid Plücker encoder channel list: {value!r}")
    return channels


def make_camera_local_unit_rays(
    *,
    height: int,
    width: int,
    horizontal_fov_deg: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create normalized optical-frame rays at full image resolution.

    Pixel coordinates use pixel centers: u=x+0.5, v=y+0.5.  The optical frame
    convention is x-right, y-down, z-forward, matching the pinhole projection
    used by depth images.
    """
    height = int(height)
    width = int(width)
    hfov = math.radians(float(horizontal_fov_deg))
    fx = float(width) / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )
    u = xs + 0.5
    v = ys + 0.5
    rays = torch.stack(
        [
            (u - cx) / fx,
            (v - cy) / fy,
            torch.ones_like(u),
        ],
        dim=0,
    )
    return F.normalize(rays, dim=0)


def make_camera_local_unit_rays_from_intrinsics(
    *,
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create rays using calibrated OpenCV pixel coordinates (u=x, v=y)."""
    ys, xs = torch.meshgrid(
        torch.arange(int(height), dtype=dtype),
        torch.arange(int(width), dtype=dtype),
        indexing="ij",
    )
    rays = torch.stack(
        [
            (xs - float(cx)) / float(fx),
            (ys - float(cy)) / float(fy),
            torch.ones_like(xs),
        ],
        dim=0,
    )
    return F.normalize(rays, dim=0)


def quat_xyzw_to_matrix(quat: Tensor) -> Tensor:
    """Convert normalized xyzw quaternions to rotation matrices."""
    quat = F.normalize(quat, dim=-1)
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def make_plucker_map(camera_pose_base: Tensor, camera_local_unit_rays: Tensor) -> Tensor:
    """Create B×6×H×W Plücker ray-maps in robot base coordinates.

    Args:
        camera_pose_base: B×7 pose [x, y, z, qx, qy, qz, qw] of the camera
            optical frame in the robot base frame.
        camera_local_unit_rays: 3×H×W normalized optical-frame ray directions.

    Returns:
        B×6×H×W map [d_x, d_y, d_z, m_x, m_y, m_z], where d is the unit ray
        direction in base frame and m = C × d with C the camera center.
    """
    if camera_pose_base.ndim != 2 or camera_pose_base.shape[-1] != 7:
        raise ValueError(f"Expected camera poses shaped (B,7), got {tuple(camera_pose_base.shape)}.")
    rays = camera_local_unit_rays.to(device=camera_pose_base.device, dtype=camera_pose_base.dtype)
    center = camera_pose_base[:, :3]
    rot = quat_xyzw_to_matrix(camera_pose_base[:, 3:7])
    directions = torch.einsum("bij,jhw->bihw", rot, rays)
    directions = F.normalize(directions, dim=1)
    centers = center[:, :, None, None].expand_as(directions)
    moments = torch.cross(centers, directions, dim=1)
    return torch.cat([directions, moments], dim=1)


class ACTPluckerEncoder(nn.Module):
    """Small CNN that encodes full-resolution 6D Plücker maps to visual-feature resolution."""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: list[int],
        deterministic_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.deterministic_pooling = bool(deterministic_pooling)
        layers: list[nn.Module] = []
        in_channels = 6
        for hidden in hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, int(hidden), kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                ]
            )
            in_channels = int(hidden)
        layers.extend(
            [
                nn.Conv2d(in_channels, int(out_channels), kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            ]
        )
        self.encoder = nn.Sequential(*layers)

    def forward(self, plucker_map: Tensor, target_hw: tuple[int, int]) -> Tensor:
        geom = self.encoder(plucker_map)
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if self.deterministic_pooling:
            source_h, source_w = int(geom.shape[-2]), int(geom.shape[-1])
            if source_h % target_h != 0 or source_w % target_w != 0:
                raise ValueError(
                    "Deterministic Plucker pooling requires integer spatial ratios, got "
                    f"source={(source_h, source_w)} target={(target_h, target_w)}."
                )
            return F.avg_pool2d(
                geom,
                kernel_size=(source_h // target_h, source_w // target_w),
                stride=(source_h // target_h, source_w // target_w),
            )
        return F.adaptive_avg_pool2d(geom, output_size=(target_h, target_w))


class ACTHandleLatentHead(nn.Module):
    """Predict a frozen-teacher handle latent from one camera view's image tokens."""

    def __init__(self, dim_model: int, n_heads: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Embedding(1, int(dim_model))
        self.cross_attn = nn.MultiheadAttention(int(dim_model), int(n_heads), dropout=float(dropout))
        self.head = nn.Sequential(
            nn.LayerNorm(int(dim_model)),
            nn.Linear(int(dim_model), int(dim_model)),
            nn.GELU(),
            nn.Linear(int(dim_model), int(latent_dim)),
        )

    def forward(self, image_tokens: Tensor) -> Tensor:
        """Return B×latent_dim from image tokens shaped S×B×D."""
        if image_tokens.ndim != 3:
            raise ValueError(
                "ACTHandleLatentHead expects image tokens shaped (sequence, batch, dim); "
                f"got {tuple(image_tokens.shape)}."
            )
        batch_size = int(image_tokens.shape[1])
        query = self.query.weight.unsqueeze(1).expand(-1, batch_size, -1)
        handle_token = self.cross_attn(query=query, key=image_tokens, value=image_tokens)[0][0]
        return self.head(handle_token)


class ACTInteractionStateHead(nn.Module):
    """Pool non-VAE encoder tokens and predict contact/handle/door logits."""

    def __init__(self, dim_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Embedding(1, int(dim_model))
        self.cross_attn = nn.MultiheadAttention(int(dim_model), int(n_heads), dropout=float(dropout))
        self.head = nn.Sequential(nn.LayerNorm(int(dim_model)), nn.Linear(int(dim_model), 3))

    def forward(self, encoder_tokens_without_vae: Tensor) -> Tensor:
        if encoder_tokens_without_vae.ndim != 3 or encoder_tokens_without_vae.shape[0] <= 0:
            raise ValueError(
                "ACTInteractionStateHead expects non-empty tokens shaped (sequence, batch, dim); "
                f"got {tuple(encoder_tokens_without_vae.shape)}."
            )
        batch_size = int(encoder_tokens_without_vae.shape[1])
        query = self.query.weight.unsqueeze(1).expand(-1, batch_size, -1)
        pooled = self.cross_attn(
            query=query,
            key=encoder_tokens_without_vae,
            value=encoder_tokens_without_vae,
        )[0][0]
        return self.head(pooled)


class ACTInteractionStateChunkHead(nn.Module):
    """Predict one interaction-state triplet from every ACT decoder token."""

    def __init__(self, dim_model: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(int(dim_model)), nn.Linear(int(dim_model), 3))

    def forward(self, decoder_tokens: Tensor) -> Tensor:
        if decoder_tokens.ndim != 3:
            raise ValueError(
                "ACTInteractionStateChunkHead expects decoder tokens shaped (batch, horizon, dim); "
                f"got {tuple(decoder_tokens.shape)}."
            )
        return self.head(decoder_tokens)


def _parse_xyz_bounds(value: str, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in str(value).split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must contain three comma-separated floats, got {value!r}.") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values, got {value!r}.")
    return values


def _batched_index(points: Tensor, indices: Tensor) -> Tensor:
    """Gather BxNxC values with Bx... integer indices."""
    if points.ndim != 3 or indices.ndim < 2 or points.shape[0] != indices.shape[0]:
        raise ValueError(f"Invalid batched gather shapes points={points.shape}, indices={indices.shape}.")
    batch_shape = [points.shape[0]] + [1] * (indices.ndim - 1)
    batch = torch.arange(points.shape[0], device=points.device).view(*batch_shape).expand_as(indices)
    return points[batch, indices]


def deterministic_farthest_point_indices(xyz: Tensor, num_samples: int) -> Tensor:
    """Pure-PyTorch deterministic FPS with the point farthest from the centroid as seed."""
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"FPS expects BxNx3, got {tuple(xyz.shape)}.")
    batch_size, point_count, _ = xyz.shape
    if not 1 <= int(num_samples) <= int(point_count):
        raise ValueError(f"num_samples must be in [1,{point_count}], got {num_samples}.")
    centroid = xyz.mean(dim=1, keepdim=True)
    farthest = ((xyz - centroid) ** 2).sum(dim=-1).argmax(dim=1)
    selected = torch.empty(batch_size, int(num_samples), dtype=torch.long, device=xyz.device)
    min_distance = torch.full(
        (batch_size, point_count), torch.finfo(xyz.dtype).max, dtype=xyz.dtype, device=xyz.device
    )
    batch = torch.arange(batch_size, device=xyz.device)
    for sample_index in range(int(num_samples)):
        selected[:, sample_index] = farthest
        center = xyz[batch, farthest].unsqueeze(1)
        distance = ((xyz - center) ** 2).sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        farthest = min_distance.argmax(dim=1)
    return selected


class ACTPointCloudLegacyGlobalEncoder(nn.Module):
    """Early DP3-style prototype retained for old PointCloud ACT checkpoints."""

    def __init__(self, dim_model: int, global_dim: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(256, int(global_dim)), nn.ReLU(), nn.Linear(int(global_dim), int(dim_model))
        )
        self.position = nn.Parameter(torch.zeros(1, 1, int(dim_model)))

    def forward(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.point_mlp(xyz).amax(dim=1, keepdim=True)
        token = self.projection(feature)
        return token, self.position.expand(xyz.shape[0], -1, -1).to(dtype=token.dtype)


class ACTDP3PointNetEncoderXYZ(nn.Module):
    """Exact XYZ PointNet used by the original single-view DP3 policy.

    This mirrors ``PointNetEncoderXYZ`` from 3D-Diffusion-Policy: three
    Linear/LayerNorm/ReLU blocks, global max pooling, and a final
    Linear/LayerNorm projection. The previous Door DP3 runs used
    ``out_channels=64``, ``use_layernorm=true`` and
    ``final_norm=layernorm``.
    """

    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.final_projection = nn.Sequential(
            nn.Linear(256, int(out_channels)),
            nn.LayerNorm(int(out_channels)),
        )
        self.out_channels = int(out_channels)

    def forward(self, xyz: Tensor) -> Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"DP3 PointNet expects [B,N,3], got {tuple(xyz.shape)}.")
        point_features = self.mlp(xyz)
        global_feature = torch.max(point_features, dim=1)[0]
        return self.final_projection(global_feature)


class ACTPointCloudGlobalEncoder(nn.Module):
    """Original DP3 XYZ PointNet followed by the required ACT token adapter."""

    def __init__(self, dim_model: int, global_dim: int) -> None:
        super().__init__()
        self.extractor = ACTDP3PointNetEncoderXYZ(out_channels=int(global_dim))
        # DP3 feeds the 64D latent to its diffusion condition encoder. ACT
        # instead requires a dim_model visual token, so this is the only
        # intentionally ACT-specific layer after the exact DP3 extractor.
        self.token_projection = nn.Linear(int(global_dim), int(dim_model))
        self.position = nn.Parameter(torch.zeros(1, 1, int(dim_model)))

    def forward(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.extractor(xyz)
        token = self.token_projection(feature).unsqueeze(1)
        return token, self.position.expand(xyz.shape[0], -1, -1).to(dtype=token.dtype)


class ACTPointCloudLegacyLocalEncoder(nn.Module):
    """Legacy local-token prototype retained only for old point-cloud checkpoints."""

    def __init__(
        self,
        dim_model: int,
        num_tokens: int,
        knn_k: int,
        workspace_min: tuple[float, float, float],
        workspace_max: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.knn_k = int(knn_k)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
        )
        self.local_projection = nn.Linear(512 + 3, int(dim_model))
        self.local_norm = nn.BatchNorm1d(int(dim_model))
        self.register_buffer("workspace_min", torch.tensor(workspace_min, dtype=torch.float32), persistent=True)
        self.register_buffer("workspace_max", torch.tensor(workspace_max, dtype=torch.float32), persistent=True)

    def _position_embedding(self, xyz: Tensor, dim_model: int) -> Tensor:
        # Normalize metric XYZ to [0, 1], then allocate an equal Fourier block to each axis.
        lo = self.workspace_min.to(device=xyz.device, dtype=xyz.dtype)
        hi = self.workspace_max.to(device=xyz.device, dtype=xyz.dtype)
        normalized = ((xyz - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
        axis_dim = max(2, int(math.ceil(dim_model / 3)))
        pair_dim = max(1, axis_dim // 2)
        frequencies = torch.arange(pair_dim, device=xyz.device, dtype=xyz.dtype)
        frequencies = 2.0 ** frequencies
        blocks = []
        for axis in range(3):
            angles = normalized[..., axis : axis + 1] * math.pi * frequencies
            blocks.extend((angles.sin(), angles.cos()))
        embedding = torch.cat(blocks, dim=-1)
        if embedding.shape[-1] < dim_model:
            embedding = F.pad(embedding, (0, dim_model - embedding.shape[-1]))
        return embedding[..., :dim_model]

    def forward(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        point_feature = self.point_mlp(xyz)
        seed_indices = deterministic_farthest_point_indices(xyz, self.num_tokens)
        seed_xyz = _batched_index(xyz, seed_indices)
        # cdist/topk has deterministic tie-breaking for a fixed input/order and avoids optional CUDA extensions.
        knn_indices = torch.cdist(seed_xyz, xyz).topk(self.knn_k, dim=-1, largest=False, sorted=True).indices
        neighbor_xyz = _batched_index(xyz, knn_indices)
        neighbor_feature = _batched_index(point_feature, knn_indices)
        local = torch.cat((neighbor_feature, neighbor_xyz - seed_xyz.unsqueeze(2)), dim=-1)
        local = self.local_projection(local)
        batch_size, token_count, neighbor_count, channels = local.shape
        local = local.reshape(batch_size * token_count * neighbor_count, channels)
        local = self.local_norm(local)
        local = F.relu(local).reshape(batch_size, token_count, neighbor_count, channels)
        tokens = local.amax(dim=2)
        positions = self._position_embedding(seed_xyz, channels).to(dtype=tokens.dtype)
        return tokens, positions


class ACTOBSBenchPointNet(nn.Module):
    """Dense equivalent of OBSBench's sparse 1x1 PointNet backbone.

    OBSBench represents the point set with ``spconv.SparseConvTensor`` and applies
    five bias-free ``SubMConv3d(kernel_size=1)`` blocks.  A 1x1 sparse convolution
    cannot exchange information between points, so for a fixed-size dense point
    tensor it is mathematically equivalent to ``Conv1d(kernel_size=1)``.  Keeping
    the channel widths and BatchNorm hyperparameters identical avoids requiring
    spconv while preserving the PointNet computation.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        channels = (int(in_channels), 64, 64, 64, 128, 512)
        blocks: list[nn.Module] = []
        for in_dim, out_dim in zip(channels[:-1], channels[1:], strict=True):
            blocks.extend(
                (
                    nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_dim, eps=1.0e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )
        self.network = nn.Sequential(*blocks)
        self.in_channels = int(in_channels)
        self.num_channels = 512

    def forward(self, point_features: Tensor) -> Tensor:
        if point_features.ndim != 3 or point_features.shape[-1] != self.in_channels:
            raise ValueError(
                f"OBSBench PointNet expects [B,N,{self.in_channels}], got {tuple(point_features.shape)}."
            )
        return self.network(point_features.transpose(1, 2)).transpose(1, 2).contiguous()


def obsbench_farthest_point_indices(xyz: Tensor, num_samples: int) -> Tensor:
    """Pure-PyTorch FPS matching OBSBench pointops' deterministic first seed.

    The pointops CUDA kernel always starts each point set at local index zero.
    Subsequent seeds maximize the minimum squared distance to selected seeds.
    """

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected point cloud [B,N,3], got {tuple(xyz.shape)}.")
    batch_size, point_count, _ = xyz.shape
    if num_samples <= 0 or num_samples > point_count:
        raise ValueError(f"num_samples must be in [1,{point_count}], got {num_samples}.")
    selected = torch.empty(batch_size, int(num_samples), dtype=torch.long, device=xyz.device)
    min_distance = torch.full(
        (batch_size, point_count), torch.finfo(xyz.dtype).max, dtype=xyz.dtype, device=xyz.device
    )
    farthest = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)
    batch = torch.arange(batch_size, device=xyz.device)
    for sample_index in range(int(num_samples)):
        selected[:, sample_index] = farthest
        center = xyz[batch, farthest].unsqueeze(1)
        distance = ((xyz - center) ** 2).sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        farthest = min_distance.argmax(dim=1)
    return selected


class ACTPointCloudLocalEncoder(nn.Module):
    """OBSBench PointNet post-sampling local-token encoder.

    This follows ``ACTPCD(pre_sample=False)``: PointNet first computes a
    feature for every input point, FPS selects seeds afterwards, KNN groups
    encoded neighboring features plus relative XYZ, and max pooling emits one
    token per seed.
    """

    def __init__(
        self,
        dim_model: int,
        num_tokens: int,
        knn_k: int,
        workspace_min: tuple[float, float, float],
        workspace_max: tuple[float, float, float],
    ) -> None:
        super().__init__()
        del workspace_min, workspace_max  # OBSBench uses unnormalized metric XYZ for its sine embedding.
        self.num_tokens = int(num_tokens)
        self.knn_k = int(knn_k)
        self.pointnet = ACTOBSBenchPointNet(in_channels=3)
        self.local_projection = nn.Linear(3 + self.pointnet.num_channels, int(dim_model), bias=False)
        self.local_norm = nn.BatchNorm1d(int(dim_model))
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _position_embedding(coord: Tensor, dim_model: int, temperature: float = 10000.0) -> Tensor:
        """OBSBench ACTPCD.coord_embedding_sine for batched metric XYZ."""

        num_pos_feats = int(dim_model) // 3
        num_pad_feats = int(dim_model) - num_pos_feats * 3
        dim_t = torch.arange(num_pos_feats, dtype=coord.dtype, device=coord.device)
        dim_t = float(temperature) ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

        blocks = []
        for axis in range(3):
            angles = coord[..., axis : axis + 1] / dim_t
            sin = angles[..., 0::2].sin()
            cos = angles[..., 1::2].cos()
            pair_count = min(sin.shape[-1], cos.shape[-1])
            paired = torch.stack((sin[..., :pair_count], cos[..., :pair_count]), dim=-1).flatten(-2)
            if paired.shape[-1] < num_pos_feats:
                paired = F.pad(paired, (0, num_pos_feats - paired.shape[-1]))
            blocks.append(paired[..., :num_pos_feats])
        position = torch.cat(blocks, dim=-1)
        if num_pad_feats:
            position = F.pad(position, (0, num_pad_feats))
        return position

    def forward(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        # OBSBench default post-sampling: encode the complete preprocessed cloud first.
        point_feature = self.pointnet(xyz)
        seed_indices = obsbench_farthest_point_indices(xyz, self.num_tokens)
        seed_xyz = _batched_index(xyz, seed_indices)

        # Pure-PyTorch equivalent of pointops.knn_query_and_group(..., with_xyz=True).
        knn_indices = torch.cdist(seed_xyz, xyz).topk(
            self.knn_k, dim=-1, largest=False, sorted=True
        ).indices
        neighbor_xyz = _batched_index(xyz, knn_indices)
        neighbor_feature = _batched_index(point_feature, knn_indices)
        grouped = torch.cat((neighbor_xyz - seed_xyz.unsqueeze(2), neighbor_feature), dim=-1)

        # OBSBench: Linear -> BN -> ReLU -> MaxPool1d(nsample).
        local = self.local_projection(grouped)
        batch_size, token_count, neighbor_count, channels = local.shape
        local = local.reshape(batch_size * token_count, neighbor_count, channels).transpose(1, 2)
        local = self.relu(self.local_norm(local))
        tokens = F.max_pool1d(local, kernel_size=neighbor_count).squeeze(-1)
        tokens = tokens.reshape(batch_size, token_count, channels)
        positions = self._position_embedding(seed_xyz, channels).to(dtype=tokens.dtype)
        return tokens, positions


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            if is_defm_backbone(config.vision_backbone):
                self.backbone = ACTDeFMBackbone(
                    config.vision_backbone,
                    image_size=config.defm_image_size,
                    patch_size=config.defm_patch_size,
                    feature_grid_size=config.defm_feature_grid_size,
                    depth_lower=config.defm_depth_lower,
                    depth_far=config.defm_depth_far,
                    pretrained=config.defm_pretrained,
                    pretrained_path=config.defm_pretrained_path,
                    freeze=bool(config.freeze_vision_backbone),
                )
                backbone_out_channels = self.backbone.out_channels
            elif is_dinov2_backbone(config.vision_backbone):
                self.backbone = ACTDINOv2Backbone(
                    config.vision_backbone,
                    image_size=config.dinov2_image_size,
                    feature_grid_size=config.dinov2_feature_grid_size,
                    normalize_inputs=config.dinov2_normalize_inputs,
                    freeze=config.freeze_vision_backbone,
                )
                backbone_out_channels = self.backbone.out_channels
            else:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
                # feature map).
                # Note: The forward method of this returns a dict: {"feature_map": output}.
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
                backbone_out_channels = backbone_model.fc.in_features
                if config.freeze_vision_backbone:
                    for parameter in self.backbone.parameters():
                        parameter.requires_grad_(False)
            if self.config.plucker_conditioning:
                self.plucker_pose_keys = [
                    self._plucker_pose_key_for_image_key(image_key)
                    for image_key in list(self.config.image_features)
                ]
                self.plucker_encoder = ACTPluckerEncoder(
                    out_channels=int(backbone_out_channels),
                    hidden_channels=_parse_plucker_encoder_channels(self.config.plucker_encoder_channels),
                    deterministic_pooling=self.config.plucker_deterministic_pooling,
                )
                if getattr(self.config, "plucker_intrinsics_mode", "legacy_shared_fov") == "per_camera":
                    for camera_name in ("front", "wrist"):
                        self.register_buffer(
                            f"plucker_{camera_name}_camera_local_unit_rays",
                            make_camera_local_unit_rays_from_intrinsics(
                                height=int(self.config.plucker_image_height),
                                width=int(self.config.plucker_image_width),
                                fx=float(getattr(self.config, f"plucker_{camera_name}_fx")),
                                fy=float(getattr(self.config, f"plucker_{camera_name}_fy")),
                                cx=float(getattr(self.config, f"plucker_{camera_name}_cx")),
                                cy=float(getattr(self.config, f"plucker_{camera_name}_cy")),
                            ),
                            persistent=False,
                        )
                else:
                    # Keep the historical buffer and ray construction byte-for-byte unchanged.
                    self.register_buffer(
                        "plucker_camera_local_unit_rays",
                        make_camera_local_unit_rays(
                            height=int(self.config.plucker_image_height),
                            width=int(self.config.plucker_image_width),
                            horizontal_fov_deg=float(self.config.plucker_horizontal_fov_deg),
                        ),
                        persistent=False,
                    )

        if self.config.point_cloud_conditioning:
            workspace_min = _parse_xyz_bounds(
                self.config.point_cloud_workspace_min, "point_cloud_workspace_min"
            )
            workspace_max = _parse_xyz_bounds(
                self.config.point_cloud_workspace_max, "point_cloud_workspace_max"
            )
            if any(upper <= lower for lower, upper in zip(workspace_min, workspace_max, strict=True)):
                raise ValueError(
                    f"Invalid point-cloud workspace bounds: min={workspace_min}, max={workspace_max}."
                )
            if self.config.point_cloud_encoder_mode == "dp3_global":
                self.point_cloud_encoder = ACTPointCloudGlobalEncoder(
                    dim_model=config.dim_model,
                    global_dim=config.point_cloud_global_dim,
                )
            elif self.config.point_cloud_encoder_mode == "dp3_global_legacy":
                self.point_cloud_encoder = ACTPointCloudLegacyGlobalEncoder(
                    dim_model=config.dim_model,
                    global_dim=config.point_cloud_global_dim,
                )
            elif self.config.point_cloud_encoder_mode == "obsbench_local_legacy":
                self.point_cloud_encoder = ACTPointCloudLegacyLocalEncoder(
                    dim_model=config.dim_model,
                    num_tokens=config.point_cloud_num_tokens,
                    knn_k=config.point_cloud_knn_k,
                    workspace_min=workspace_min,
                    workspace_max=workspace_max,
                )
            else:
                self.point_cloud_encoder = ACTPointCloudLocalEncoder(
                    dim_model=config.dim_model,
                    num_tokens=config.point_cloud_num_tokens,
                    knn_k=config.point_cloud_knn_k,
                    workspace_min=workspace_min,
                    workspace_max=workspace_max,
                )

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            img_proj_in_channels = int(backbone_out_channels)
            if self.config.plucker_conditioning:
                img_proj_in_channels += int(backbone_out_channels)
            self.encoder_img_feat_input_proj = nn.Conv2d(
                img_proj_in_channels, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        self.end_signal_head = None
        if self.config.end_signal_prediction:
            context = (
                torch.random.fork_rng(devices=[])
                if self.config.auxiliary_head_rng_isolation
                else nullcontext()
            )
            with context:
                self.end_signal_head = nn.Linear(config.dim_model, 1)
        if self.config.interaction_state_conditioning:
            context = (
                torch.random.fork_rng(devices=[])
                if self.config.auxiliary_head_rng_isolation
                else nullcontext()
            )
            with context:
                if config.interaction_state_prediction_mode == "decoder_chunk":
                    self.interaction_state_chunk_head = ACTInteractionStateChunkHead(config.dim_model)
                else:
                    interaction_dropout = (
                        float(config.interaction_state_probe_dropout)
                        if (
                            config.interaction_state_probe_only
                            or config.interaction_state_auxiliary_only
                        )
                        else float(config.dropout)
                    )
                    self.interaction_state_head = ACTInteractionStateHead(
                        config.dim_model,
                        config.n_heads,
                        interaction_dropout,
                    )
                    self.interaction_conditioner = nn.Sequential(
                        nn.Linear(3, config.dim_model),
                        nn.GELU(),
                        nn.Linear(config.dim_model, config.dim_model),
                    )

        self._reset_parameters()
        self._last_camera_gates: Tensor | None = None
        self._last_handle_latent_loss: Tensor | None = None
        self._last_handle_latent_front_loss: Tensor | None = None
        self._last_handle_latent_wrist_loss: Tensor | None = None
        self._last_handle_latent_valid_count: Tensor | None = None
        self._last_end_signal_logits: Tensor | None = None
        self._last_interaction_state_logits: Tensor | None = None
        self._last_interaction_state_probabilities: Tensor | None = None
        if self.config.camera_input_gating:
            self._init_camera_input_gating()
        if self.config.handle_latent_aux:
            self._init_handle_latent_aux()
        if self.end_signal_head is not None:
            nn.init.zeros_(self.end_signal_head.weight)
            init_probability = float(self.config.end_signal_init_probability)
            nn.init.constant_(self.end_signal_head.bias, math.log(init_probability / (1.0 - init_probability)))
        if (
            self.config.interaction_state_conditioning
            and self.config.interaction_state_prediction_mode == "encoder_current"
        ):
            nn.init.zeros_(self.interaction_conditioner[-1].weight)
            nn.init.zeros_(self.interaction_conditioner[-1].bias)

    def _plucker_pose_key_for_image_key(self, image_key: str) -> str:
        image_key_lower = str(image_key).lower()
        if "front" in image_key_lower:
            return str(self.config.plucker_front_pose_key)
        if "wrist" in image_key_lower:
            return str(self.config.plucker_wrist_pose_key)
        raise ValueError(
            "Could not infer Plücker camera pose key from image feature "
            f"{image_key!r}; expected the key to contain 'front' or 'wrist'."
        )

    @staticmethod
    def _plucker_camera_name_for_image_key(image_key: str) -> str:
        image_key_lower = str(image_key).lower()
        if "front" in image_key_lower:
            return "front"
        if "wrist" in image_key_lower:
            return "wrist"
        raise ValueError(f"Could not infer Plücker camera name from image feature {image_key!r}.")

    @staticmethod
    def _camera_pose_tensor_from_batch(batch: dict[str, Tensor], pose_key: str, batch_size: int) -> Tensor:
        if pose_key not in batch:
            raise KeyError(
                f"Plücker conditioning requires camera pose feature {pose_key!r}, but it is missing from the batch."
            )
        pose = batch[pose_key]
        if pose.ndim >= 3:
            pose = pose.reshape(int(batch_size), -1, 7)[:, -1]
        else:
            pose = pose.reshape(int(batch_size), 7)
        return pose

    def _init_handle_latent_aux(self) -> None:
        image_keys = list(self.config.image_features)

        def resolve_camera_index(camera_name: str) -> int:
            candidates = [index for index, key in enumerate(image_keys) if camera_name in str(key).lower()]
            if len(candidates) != 1:
                raise ValueError(
                    f"Could not uniquely infer the {camera_name} camera for handle-latent aux from ACT image "
                    f"features {image_keys}."
                )
            return candidates[0]

        self.handle_latent_front_index = resolve_camera_index("front")
        self.handle_latent_wrist_index = resolve_camera_index("wrist")
        if self.handle_latent_front_index == self.handle_latent_wrist_index:
            raise ValueError("Front and wrist handle-latent aux keys must refer to different image features.")
        self.front_handle_latent_head = ACTHandleLatentHead(
            dim_model=self.config.dim_model,
            n_heads=self.config.n_heads,
            latent_dim=self.config.handle_latent_dim,
            dropout=self.config.dropout,
        )
        self.wrist_handle_latent_head = ACTHandleLatentHead(
            dim_model=self.config.dim_model,
            n_heads=self.config.n_heads,
            latent_dim=self.config.handle_latent_dim,
            dropout=self.config.dropout,
        )

    def _handle_latent_target_from_batch(
        self,
        batch: dict[str, Tensor],
        *,
        target_key: str,
        valid_key: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor] | tuple[None, None]:
        missing = [key for key in (target_key, valid_key) if key not in batch]
        if missing:
            if self.training:
                raise KeyError(
                    "ACT handle-latent aux is enabled, but the training batch is missing "
                    f"{missing}. Re-convert the dataset with --add_handle_latent, or disable "
                    "--policy.handle_latent_aux."
                )
            return None, None
        target = batch[target_key].to(device=device, dtype=dtype).reshape(int(batch_size), -1)
        if target.shape[-1] != int(self.config.handle_latent_dim):
            raise ValueError(
                f"Handle latent target {target_key!r} must have dim {self.config.handle_latent_dim}; "
                f"got shape {tuple(target.shape)}."
            )
        valid = batch[valid_key].to(device=device, dtype=dtype).reshape(int(batch_size), -1)[:, 0]
        valid = torch.clamp(valid, min=0.0, max=1.0)
        return target, valid

    def _compute_handle_latent_aux_loss(
        self,
        batch: dict[str, Tensor],
        encoder_out: Tensor,
        camera_token_spans: dict[int, tuple[int, int]],
        *,
        batch_size: int,
    ) -> None:
        self._last_handle_latent_loss = None
        self._last_handle_latent_front_loss = None
        self._last_handle_latent_wrist_loss = None
        self._last_handle_latent_valid_count = None
        self._last_interaction_state_logits = None
        self._last_interaction_state_probabilities = None
        self._last_end_signal_logits = None

        if not self.config.handle_latent_aux or not self.training:
            return

        device = encoder_out.device
        dtype = encoder_out.dtype
        numerator = encoder_out.new_zeros(())
        denom = encoder_out.new_zeros(())

        def view_loss(
            *,
            camera_index: int,
            head: ACTHandleLatentHead,
            target_key: str,
            valid_key: str,
        ) -> tuple[Tensor, Tensor, Tensor]:
            if camera_index not in camera_token_spans:
                raise RuntimeError(
                    f"Internal ACT error: no image token span recorded for camera index {camera_index}."
                )
            start, end = camera_token_spans[camera_index]
            pred = head(encoder_out[start:end])
            target, valid = self._handle_latent_target_from_batch(
                batch,
                target_key=target_key,
                valid_key=valid_key,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            if target is None or valid is None:
                zero = encoder_out.new_zeros(())
                return zero, zero, zero
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)
            per_sample = 1.0 - (pred * target).sum(dim=-1)
            valid_sum = valid.sum()
            weighted_sum = (valid * per_sample).sum()
            mean_valid_loss = weighted_sum / torch.clamp(valid_sum, min=1.0)
            return weighted_sum, valid_sum, mean_valid_loss

        front_sum, front_valid_sum, front_loss = view_loss(
            camera_index=self.handle_latent_front_index,
            head=self.front_handle_latent_head,
            target_key=str(self.config.handle_latent_front_key),
            valid_key=str(self.config.handle_latent_front_valid_key),
        )
        wrist_sum, wrist_valid_sum, wrist_loss = view_loss(
            camera_index=self.handle_latent_wrist_index,
            head=self.wrist_handle_latent_head,
            target_key=str(self.config.handle_latent_wrist_key),
            valid_key=str(self.config.handle_latent_wrist_valid_key),
        )
        numerator = numerator + front_sum + wrist_sum
        denom = denom + front_valid_sum + wrist_valid_sum
        self._last_handle_latent_loss = numerator / torch.clamp(denom, min=1.0)
        # If no handle target is valid in the batch, explicitly keep the loss at
        # zero without letting invalid samples dilute valid-sample batches.
        self._last_handle_latent_loss = torch.where(
            denom > 0.0,
            self._last_handle_latent_loss,
            numerator.detach() * 0.0,
        )
        self._last_handle_latent_front_loss = front_loss
        self._last_handle_latent_wrist_loss = wrist_loss
        self._last_handle_latent_valid_count = denom.detach()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_camera_input_gating(self) -> None:
        image_keys = list(self.config.image_features)
        if len(image_keys) != 2:
            raise ValueError(
                "ACT camera input gating currently requires exactly two image features "
                f"(front and wrist); got {image_keys}."
            )
        if self.config.robot_state_feature is None:
            raise ValueError("ACT camera input gating requires an observation.state feature.")

        def resolve_camera_index(explicit_key: str | None, camera_name: str) -> int:
            if explicit_key:
                if explicit_key not in image_keys:
                    raise ValueError(
                        f"Configured {camera_name} camera key {explicit_key!r} is not present in "
                        f"ACT image features {image_keys}."
                    )
                return image_keys.index(explicit_key)
            candidates = [index for index, key in enumerate(image_keys) if camera_name in key.lower()]
            if len(candidates) != 1:
                raise ValueError(
                    f"Could not uniquely infer the {camera_name} camera from ACT image features {image_keys}. "
                    f"Set --policy.camera_input_gating_{camera_name}_key explicitly."
                )
            return candidates[0]

        self.camera_input_gating_front_index = resolve_camera_index(
            self.config.camera_input_gating_front_key,
            "front",
        )
        self.camera_input_gating_wrist_index = resolve_camera_index(
            self.config.camera_input_gating_wrist_key,
            "wrist",
        )
        if self.camera_input_gating_front_index == self.camera_input_gating_wrist_index:
            raise ValueError("Front and wrist camera gating keys must refer to different image features.")
        self.camera_input_gate = ACTCameraInputGating(
            dim_model=self.config.dim_model,
            robot_state_dim=self.config.robot_state_feature.shape[0],
            hidden_dim=self.config.camera_input_gating_hidden_dim,
            temperature=self.config.camera_input_gating_temperature,
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )
        self._last_camera_gates = None
        self._last_handle_latent_loss = None
        self._last_handle_latent_front_loss = None
        self._last_handle_latent_wrist_loss = None
        self._last_handle_latent_valid_count = None

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        elif self.config.point_cloud_conditioning and self.config.point_cloud_key in batch:
            batch_size = batch[self.config.point_cloud_key].shape[0]
        elif OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
        elif OBS_STATE in batch:
            batch_size = batch[OBS_STATE].shape[0]
        else:
            raise KeyError("ACT batch has no image, point cloud, environment state, or robot state.")

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        camera_token_spans: dict[int, tuple[int, int]] = {}
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            camera_feature_maps = []
            camera_pos_embeds = []
            for camera_index, img in enumerate(batch[OBS_IMAGES]):
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                if self.config.plucker_conditioning:
                    pose_key = self.plucker_pose_keys[camera_index]
                    camera_pose_base = self._camera_pose_tensor_from_batch(batch, pose_key, batch_size)
                    camera_pose_base = camera_pose_base.to(device=cam_features.device, dtype=cam_features.dtype)
                    if getattr(self.config, "plucker_intrinsics_mode", "legacy_shared_fov") == "per_camera":
                        camera_name = self._plucker_camera_name_for_image_key(
                            list(self.config.image_features)[camera_index]
                        )
                        camera_local_unit_rays = getattr(
                            self, f"plucker_{camera_name}_camera_local_unit_rays"
                        )
                    else:
                        camera_local_unit_rays = self.plucker_camera_local_unit_rays
                    plucker_map = make_plucker_map(camera_pose_base, camera_local_unit_rays)
                    geom_features = self.plucker_encoder(plucker_map, target_hw=cam_features.shape[-2:])
                    cam_features = torch.cat([cam_features, geom_features.to(dtype=cam_features.dtype)], dim=1)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                camera_feature_maps.append(cam_features)
                camera_pos_embeds.append(cam_pos_embed)

            camera_gates = None
            if self.config.camera_input_gating:
                camera_gates = self.camera_input_gate(
                    camera_feature_maps[self.camera_input_gating_front_index],
                    camera_feature_maps[self.camera_input_gating_wrist_index],
                    batch[OBS_STATE],
                )
                # Canonical order is always [front, wrist], independent of image-feature order.
                self._last_camera_gates = camera_gates.detach()

            for camera_index, (cam_features, cam_pos_embed) in enumerate(
                zip(camera_feature_maps, camera_pos_embeds, strict=True)
            ):
                if camera_gates is not None:
                    gate_index = 0 if camera_index == self.camera_input_gating_front_index else 1
                    cam_features = cam_features * camera_gates[:, gate_index].view(-1, 1, 1, 1)
                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                token_start = len(encoder_in_tokens)
                token_end = token_start + int(cam_features.shape[0])
                camera_token_spans[int(camera_index)] = (token_start, token_end)

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        if self.config.point_cloud_conditioning:
            point_cloud_key = self.config.point_cloud_key
            if point_cloud_key not in batch:
                raise KeyError(
                    f"Point-cloud ACT requires batch feature {point_cloud_key!r}; got {list(batch)}."
                )
            point_cloud = batch[point_cloud_key]
            expected_shape = (batch_size, self.config.point_cloud_num_points, 3)
            if tuple(point_cloud.shape) != expected_shape:
                raise ValueError(
                    f"Point-cloud ACT expected {point_cloud_key!r} shape {expected_shape}, "
                    f"got {tuple(point_cloud.shape)}."
                )
            if not torch.is_floating_point(point_cloud):
                point_cloud = point_cloud.float()
            point_tokens, point_positions = self.point_cloud_encoder(point_cloud)
            point_tokens = point_tokens.transpose(0, 1)
            point_positions = point_positions.transpose(0, 1)
            # Legacy 1-D token positions are stored as 1xD and normally
            # broadcast inside attention. Metric point positions vary per
            # batch element, so make that batch dimension explicit here.
            encoder_in_pos_embed = [
                position.expand(batch_size, -1) if position.shape[0] == 1 else position
                for position in encoder_in_pos_embed
            ]
            encoder_in_tokens.extend(list(point_tokens))
            encoder_in_pos_embed.extend(list(point_positions))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        self._compute_handle_latent_aux_loss(
            batch,
            encoder_out,
            camera_token_spans,
            batch_size=batch_size,
        )
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        if (
            self.config.interaction_state_conditioning
            and self.config.interaction_state_prediction_mode == "encoder_current"
        ):
            # Token zero is the ACT VAE latent. Excluding it prevents the
            # interaction predictor from reading future ground-truth actions in training.
            interaction_feature = encoder_out[1:]
            if self.config.interaction_state_probe_only:
                # Probe-only is a read-only diagnostic head: its auxiliary
                # losses cannot change the ACT encoder or visual backbone.
                interaction_feature = interaction_feature.detach()
            interaction_logits = self.interaction_state_head(interaction_feature)
            interaction_probabilities = torch.sigmoid(interaction_logits)
            self._last_interaction_state_logits = interaction_logits
            self._last_interaction_state_probabilities = interaction_probabilities
            if not (
                self.config.interaction_state_probe_only
                or self.config.interaction_state_auxiliary_only
            ):
                interaction_residual = self.interaction_conditioner(interaction_probabilities)
                decoder_in = decoder_in + interaction_residual.unsqueeze(0)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)
        if (
            self.config.interaction_state_conditioning
            and self.config.interaction_state_prediction_mode == "decoder_chunk"
        ):
            # Each decoder token is already aligned with one future action
            # timestep, so predict [contact, handle progress, door progress]
            # directly for the same t:t+H horizon. Probe-only keeps this as a
            # read-only diagnostic of decoder features.
            interaction_feature = (
                decoder_out.detach() if self.config.interaction_state_probe_only else decoder_out
            )
            interaction_logits = self.interaction_state_chunk_head(interaction_feature)
            self._last_interaction_state_logits = interaction_logits
            self._last_interaction_state_probabilities = torch.sigmoid(interaction_logits)
        if self.end_signal_head is not None:
            end_signal_feature = (
                decoder_out.detach()
                if self.config.end_signal_detach_decoder_feature
                else decoder_out
            )
            self._last_end_signal_logits = self.end_signal_head(end_signal_feature)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
