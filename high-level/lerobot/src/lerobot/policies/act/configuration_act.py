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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        camera_input_gating: Enable learned front/wrist camera gating between the CNN and transformer encoder.
            Disabled by default to preserve the original ACT architecture and checkpoint behavior.
        camera_input_gating_hidden_dim: Hidden dimension of the camera-gating MLP.
        camera_input_gating_temperature: Softmax temperature for the two camera gates.
        camera_input_gating_front_key: Optional exact front-camera feature key. If unset, the key containing
            "front" is selected automatically.
        camera_input_gating_wrist_key: Optional exact wrist-camera feature key. If unset, the key containing
            "wrist" is selected automatically.
        plucker_conditioning: Enable per-pixel Plücker ray conditioning for dual-depth ACT. Disabled by
            default to preserve the original ACT architecture and checkpoint behavior.
        plucker_front_pose_key: Batch key containing front camera pose in robot base frame as
            [x, y, z, qx, qy, qz, qw].
        plucker_wrist_pose_key: Batch key containing wrist camera pose in robot base frame as
            [x, y, z, qx, qy, qz, qw].
        plucker_encoder_channels: Comma-separated hidden channel sizes for the Plücker CNN before the final
            output layer.
        plucker_image_width / plucker_image_height: Full-resolution image size used to generate the ray-map.
        plucker_horizontal_fov_deg: Horizontal FOV used to derive default pinhole intrinsics. v1 assumes front
            and wrist depth cameras share this intrinsics model.
        handle_latent_aux: Enable the training-only DINOv2 handle-latent reconstruction auxiliary loss.
            Disabled by default; inference does not require the aux latent targets.
        handle_latent_dim: Dimensionality of the frozen DINOv2-small patch-mean latent target.
        handle_latent_loss_weight: Scalar multiplier λ for the handle-latent auxiliary loss.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    freeze_vision_backbone: bool | None = None
    dinov2_image_size: int = 224
    dinov2_feature_grid_size: int = 6
    dinov2_normalize_inputs: bool = True
    defm_image_size: int = 224
    defm_patch_size: int = 14
    defm_feature_grid_size: int = 6
    defm_depth_lower: float = 0.02
    defm_depth_far: float = 2.0
    defm_pretrained: bool = True
    defm_pretrained_path: str | None = None
    # Optional learned front/wrist input gating. The gate is initialized to
    # [1, 1], so enabling it starts from the original ACT visual-token path.
    camera_input_gating: bool = False
    camera_input_gating_hidden_dim: int = 128
    camera_input_gating_temperature: float = 1.0
    camera_input_gating_front_key: str | None = None
    camera_input_gating_wrist_key: str | None = None
    # Optional Plücker-conditioned dual-depth input. The ResNet still receives
    # the original 3-channel depth image; a small separate CNN encodes the 6D
    # per-pixel ray-map and fuses it with the ResNet feature map afterwards.
    plucker_conditioning: bool = False
    plucker_front_pose_key: str = "observation.camera_pose.front"
    plucker_wrist_pose_key: str = "observation.camera_pose.wrist"
    plucker_encoder_channels: str = "32,64"
    plucker_image_width: int = 640
    plucker_image_height: int = 480
    plucker_horizontal_fov_deg: float = 69.0
    # Optional training-only DINOv2 handle-latent reconstruction auxiliary task.
    handle_latent_aux: bool = False
    handle_latent_dim: int = 384
    handle_latent_loss_weight: float = 0.1
    handle_latent_front_key: str = "aux.front_handle_latent"
    handle_latent_front_valid_key: str = "aux.front_handle_latent_valid"
    handle_latent_wrist_key: str = "aux.wrist_handle_latent"
    handle_latent_wrist_valid_key: str = "aux.wrist_handle_latent_valid"
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0
    use_action_loss_weight: bool = True

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        vision_backbone = str(self.vision_backbone).lower()
        is_resnet = vision_backbone.startswith("resnet")
        is_dinov2 = vision_backbone.startswith("dinov2") or vision_backbone.startswith("facebook/dinov2")
        is_defm = vision_backbone in {"defm-vit-l14", "defm_vit_l14", "defm-vit-l/14"}
        if self.freeze_vision_backbone is None:
            self.freeze_vision_backbone = bool(is_defm)
        if not (is_resnet or is_dinov2 or is_defm):
            raise ValueError(
                "`vision_backbone` must be one of the ResNet, DINOv2, or DeFM variants "
                "('dinov2-small/base/large', 'facebook/dinov2-*', or 'defm-vit-l14'). "
                f"Got {self.vision_backbone}."
            )
        if is_dinov2 or is_defm:
            found_visual_norm = False
            for key in list(self.normalization_mapping):
                if key == "VISUAL" or getattr(key, "value", None) == "VISUAL" or str(key).endswith(".VISUAL"):
                    self.normalization_mapping[key] = NormalizationMode.IDENTITY
                    found_visual_norm = True
            if not found_visual_norm:
                self.normalization_mapping["VISUAL"] = NormalizationMode.IDENTITY
        if is_dinov2:
            if self.dinov2_image_size <= 0:
                raise ValueError(f"`dinov2_image_size` must be positive. Got {self.dinov2_image_size}.")
            if self.dinov2_feature_grid_size <= 0:
                raise ValueError(
                    f"`dinov2_feature_grid_size` must be positive. Got {self.dinov2_feature_grid_size}."
                )
        if is_defm:
            non_depth_keys = [
                key for key in self.image_features if "depth" not in str(key).lower()
            ]
            if non_depth_keys:
                raise ValueError(
                    "DeFM ACT backbone only supports depth image features. "
                    f"Non-depth image features were configured: {non_depth_keys}."
                )
            if self.defm_image_size <= 0:
                raise ValueError(f"`defm_image_size` must be positive. Got {self.defm_image_size}.")
            if self.defm_patch_size <= 0:
                raise ValueError(f"`defm_patch_size` must be positive. Got {self.defm_patch_size}.")
            if self.defm_feature_grid_size <= 0:
                raise ValueError(
                    f"`defm_feature_grid_size` must be positive. Got {self.defm_feature_grid_size}."
                )
            if self.defm_depth_lower < 0.0:
                raise ValueError(f"`defm_depth_lower` must be non-negative. Got {self.defm_depth_lower}.")
            if self.defm_depth_far <= self.defm_depth_lower:
                raise ValueError(
                    "`defm_depth_far` must be greater than `defm_depth_lower`. "
                    f"Got far={self.defm_depth_far}, lower={self.defm_depth_lower}."
                )
        if self.camera_input_gating_hidden_dim <= 0:
            raise ValueError(
                "`camera_input_gating_hidden_dim` must be positive. "
                f"Got {self.camera_input_gating_hidden_dim}."
            )
        if self.camera_input_gating_temperature <= 0.0:
            raise ValueError(
                "`camera_input_gating_temperature` must be positive. "
                f"Got {self.camera_input_gating_temperature}."
            )
        if self.plucker_conditioning:
            if not is_resnet:
                raise ValueError(
                    "ACT Plücker conditioning v1 only supports ResNet backbones because it fuses after the "
                    f"ResNet feature map. Got vision_backbone={self.vision_backbone!r}."
                )
            # `input_features` are usually inferred from the dataset later in
            # `make_policy()`, after the CLI config has already been decoded.
            # Therefore, only validate feature names in `validate_features()`,
            # where dataset-derived features are available.
            if not self.plucker_front_pose_key or not self.plucker_wrist_pose_key:
                raise ValueError("Plücker conditioning requires front and wrist camera pose keys.")
            if self.plucker_image_width <= 0 or self.plucker_image_height <= 0:
                raise ValueError(
                    "`plucker_image_width` and `plucker_image_height` must be positive. "
                    f"Got {self.plucker_image_width}x{self.plucker_image_height}."
                )
            if self.plucker_horizontal_fov_deg <= 0.0 or self.plucker_horizontal_fov_deg >= 180.0:
                raise ValueError(
                    "`plucker_horizontal_fov_deg` must be in (0, 180). "
                    f"Got {self.plucker_horizontal_fov_deg}."
                )
            try:
                channels = [int(x.strip()) for x in str(self.plucker_encoder_channels).split(",") if x.strip()]
            except ValueError as exc:
                raise ValueError(
                    "`plucker_encoder_channels` must be a comma-separated list of positive integers. "
                    f"Got {self.plucker_encoder_channels!r}."
                ) from exc
            if not channels or any(channel <= 0 for channel in channels):
                raise ValueError(
                    "`plucker_encoder_channels` must contain at least one positive channel size. "
                    f"Got {self.plucker_encoder_channels!r}."
                )
        if self.handle_latent_aux:
            if self.handle_latent_dim <= 0:
                raise ValueError(f"`handle_latent_dim` must be positive. Got {self.handle_latent_dim}.")
            if self.handle_latent_loss_weight < 0.0:
                raise ValueError(
                    "`handle_latent_loss_weight` must be non-negative. "
                    f"Got {self.handle_latent_loss_weight}."
                )
            for key_name in (
                "handle_latent_front_key",
                "handle_latent_front_valid_key",
                "handle_latent_wrist_key",
                "handle_latent_wrist_valid_key",
            ):
                if not str(getattr(self, key_name, "")):
                    raise ValueError(f"`{key_name}` must be non-empty when handle_latent_aux is enabled.")
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.plucker_conditioning:
            image_keys = list(self.image_features)
            if len(image_keys) != 2:
                raise ValueError(
                    "ACT Plücker conditioning v1 expects exactly two image features (front and wrist). "
                    f"Got {image_keys}."
                )
            lower_keys = [str(key).lower() for key in image_keys]
            if not any("front" in key for key in lower_keys) or not any("wrist" in key for key in lower_keys):
                raise ValueError(
                    "ACT Plücker conditioning needs one front image key and one wrist image key so it can "
                    f"select the matching camera pose. Got {image_keys}."
                )
            missing_pose_keys = [
                key
                for key in (self.plucker_front_pose_key, self.plucker_wrist_pose_key)
                if key not in (self.input_features or {})
            ]
            if missing_pose_keys:
                raise ValueError(
                    "ACT Plücker conditioning requires camera pose features in the dataset. "
                    f"Missing {missing_pose_keys}; available input features are {list((self.input_features or {}).keys())}."
                )
        if self.handle_latent_aux:
            image_keys = list(self.image_features)
            if len(image_keys) != 2:
                raise ValueError(
                    "ACT handle-latent auxiliary loss expects exactly two image features (front and wrist). "
                    f"Got {image_keys}."
                )
            lower_keys = [str(key).lower() for key in image_keys]
            if not any("front" in key for key in lower_keys) or not any("wrist" in key for key in lower_keys):
                raise ValueError(
                    "ACT handle-latent auxiliary loss needs one front image key and one wrist image key. "
                    f"Got {image_keys}."
                )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
