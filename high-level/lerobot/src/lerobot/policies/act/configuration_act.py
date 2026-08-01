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
import math
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
            "POINT_CLOUD": NormalizationMode.IDENTITY,
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
    # Optional metric point-cloud input. When disabled, no point-cloud module
    # is constructed and the legacy image ACT state dict/forward path is unchanged.
    point_cloud_conditioning: bool = False
    point_cloud_key: str = "observation.point_cloud"
    point_cloud_views: str = "front"
    # dp3_global: exact DP3 XYZ PointNet plus an ACT token adapter.
    # obsbench_local: OBSBench PointNet post-sampling. The *_legacy modes only
    # exist to load early PointCloud ACT prototypes.
    point_cloud_encoder_mode: str = "obsbench_local"
    point_cloud_num_points: int = 1024
    point_cloud_num_tokens: int = 256
    point_cloud_knn_k: int = 16
    point_cloud_global_dim: int = 64
    point_cloud_frame: str = "robot_base"
    point_cloud_workspace_min: str = "0.20,-1.00,0.00"
    point_cloud_workspace_max: str = "2.00,1.00,1.80"
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
    # Use exact integer avg-pooling instead of adaptive pooling when the
    # geometry grid is evenly divisible by the ResNet feature grid. The
    # default preserves legacy checkpoint numerics; strict CUDA ablations can
    # opt in because adaptive_avg_pool2d backward is nondeterministic.
    plucker_deterministic_pooling: bool = False
    plucker_image_width: int = 640
    plucker_image_height: int = 480
    plucker_horizontal_fov_deg: float = 69.0
    # legacy_shared_fov preserves the original shared-FOV ray construction,
    # including its pixel-center convention. per_camera uses calibrated
    # OpenCV pinhole intrinsics independently for front and wrist.
    plucker_intrinsics_mode: str = "legacy_shared_fov"
    plucker_front_fx: float = 0.0
    plucker_front_fy: float = 0.0
    plucker_front_cx: float = 0.0
    plucker_front_cy: float = 0.0
    plucker_wrist_fx: float = 0.0
    plucker_wrist_fy: float = 0.0
    plucker_wrist_cx: float = 0.0
    plucker_wrist_cy: float = 0.0
    # Optional training-only DINOv2 handle-latent reconstruction auxiliary task.
    handle_latent_aux: bool = False
    handle_latent_dim: int = 384
    handle_latent_loss_weight: float = 0.1
    handle_latent_front_key: str = "aux.front_handle_latent"
    handle_latent_front_valid_key: str = "aux.front_handle_latent_valid"
    handle_latent_wrist_key: str = "aux.wrist_handle_latent"
    handle_latent_wrist_valid_key: str = "aux.wrist_handle_latent_valid"
    # Optional dense autonomous-termination auxiliary target. Motion actions remain
    # unchanged; a separate sigmoid head predicts this future-aligned scalar.
    end_signal_prediction: bool = False
    end_signal_target_key: str = "aux.end_signal"
    end_signal_loss_weight: float = 0.1
    end_signal_init_probability: float = 0.01
    # Treat the ACT decoder feature as read-only for autonomous termination.
    # This lets the end head learn from L_end without sending its gradients
    # into the motion decoder, transformer, or visual encoder.
    end_signal_detach_decoder_feature: bool = True
    # Keep optional auxiliary-head construction from advancing the global RNG.
    # This is required for strict from-scratch ablations: enabling a detached
    # head must not silently change the ACT encoder/decoder initialization or
    # the later DataLoader random stream.
    auxiliary_head_rng_isolation: bool = True
    # Optional privileged interaction supervision. The legacy mode predicts the
    # current state from encoder tokens; decoder_chunk predicts the full future
    # state sequence aligned with the action chunk.
    interaction_state_conditioning: bool = False
    # Where interaction states are predicted:
    # - encoder_current: legacy Bx3 current-frame prediction from encoder tokens.
    # - decoder_chunk: BxHx3 future prediction aligned with the H-step action chunk.
    # The legacy value remains the default so old configs/checkpoints keep their
    # original architecture and numerics.
    interaction_state_prediction_mode: str = "encoder_current"
    # Probe-only keeps the interaction prediction task for diagnostics while
    # treating encoder features as read-only and disabling decoder feedback.
    # It preserves the original ACT motion path exactly.
    interaction_state_probe_only: bool = False
    # Train interaction supervision into the encoder, but do not feed the
    # predicted state back to the action decoder. This isolates representation
    # shaping from the decoder-conditioning mechanism.
    interaction_state_auxiliary_only: bool = False
    # A diagnostic probe should not consume RNG before the motion decoder. A
    # non-zero value is retained as an explicit research override only.
    interaction_state_probe_dropout: float = 0.0
    # Backpropagate the detached probe loss only after the motion backward has
    # completed. This avoids CUDA scheduling/numerical coupling between two
    # disconnected graphs in strict ablations.
    interaction_state_probe_separate_backward: bool = True
    # Freeze every ACT parameter except the interaction-state probe. This is
    # intended for fitting a read-only probe on an already-trained checkpoint.
    interaction_state_probe_freeze_main: bool = False
    interaction_contact_target_key: str = "aux.interaction_contact"
    interaction_handle_target_key: str = "aux.interaction_handle_progress"
    interaction_door_target_key: str = "aux.interaction_door_progress"
    interaction_contact_loss_weight: float = 0.1
    interaction_handle_loss_weight: float = 0.1
    interaction_door_loss_weight: float = 0.1
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
        self.point_cloud_views = ",".join(
            part.strip().lower() for part in str(self.point_cloud_views).split(",") if part.strip()
        )
        if self.point_cloud_views not in {"front", "wrist", "front,wrist"}:
            raise ValueError(
                "point_cloud_views must be 'front', 'wrist', or 'front,wrist', got "
                f"{self.point_cloud_views!r}."
            )
        self.point_cloud_encoder_mode = str(self.point_cloud_encoder_mode).strip().lower()
        if self.point_cloud_encoder_mode not in {
            "dp3_global",
            "dp3_global_legacy",
            "obsbench_local",
            "obsbench_post_pointnet",
            "obsbench_local_legacy",
        }:
            raise ValueError(
                "point_cloud_encoder_mode must be 'dp3_global', 'dp3_global_legacy', 'obsbench_local' "
                "('obsbench_post_pointnet' is an explicit alias), or "
                "'obsbench_local_legacy', got "
                f"{self.point_cloud_encoder_mode!r}."
            )
        if self.point_cloud_conditioning:
            if self.image_features:
                raise ValueError("ACT v1 image and point-cloud inputs are mutually exclusive.")
            if self.plucker_conditioning or self.camera_input_gating or self.handle_latent_aux:
                raise ValueError(
                    "Point-cloud ACT is incompatible with Plucker conditioning, camera gating, and image handle latent."
                )
            if not str(self.point_cloud_key):
                raise ValueError("point_cloud_key must be non-empty.")
            if self.point_cloud_num_points <= 0:
                raise ValueError("point_cloud_num_points must be positive.")
            if self.point_cloud_num_tokens <= 0 or self.point_cloud_num_tokens > self.point_cloud_num_points:
                raise ValueError("point_cloud_num_tokens must be in [1, point_cloud_num_points].")
            if self.point_cloud_knn_k <= 0 or self.point_cloud_knn_k > self.point_cloud_num_points:
                raise ValueError("point_cloud_knn_k must be in [1, point_cloud_num_points].")
            if self.point_cloud_global_dim <= 0:
                raise ValueError("point_cloud_global_dim must be positive.")
            if self.point_cloud_frame != "robot_base":
                raise ValueError("Point-cloud ACT v1 only supports point_cloud_frame='robot_base'.")
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
            if self.plucker_intrinsics_mode not in ("legacy_shared_fov", "per_camera"):
                raise ValueError(
                    "`plucker_intrinsics_mode` must be 'legacy_shared_fov' or 'per_camera'. "
                    f"Got {self.plucker_intrinsics_mode!r}."
                )
            if self.plucker_intrinsics_mode == "legacy_shared_fov":
                if self.plucker_horizontal_fov_deg <= 0.0 or self.plucker_horizontal_fov_deg >= 180.0:
                    raise ValueError(
                        "`plucker_horizontal_fov_deg` must be in (0, 180). "
                        f"Got {self.plucker_horizontal_fov_deg}."
                    )
            else:
                calibrated = {
                    "front_fx": self.plucker_front_fx,
                    "front_fy": self.plucker_front_fy,
                    "front_cx": self.plucker_front_cx,
                    "front_cy": self.plucker_front_cy,
                    "wrist_fx": self.plucker_wrist_fx,
                    "wrist_fy": self.plucker_wrist_fy,
                    "wrist_cx": self.plucker_wrist_cx,
                    "wrist_cy": self.plucker_wrist_cy,
                }
                if not all(math.isfinite(float(value)) for value in calibrated.values()):
                    raise ValueError(f"Per-camera Plücker intrinsics must be finite: {calibrated}.")
                if self.plucker_front_fx <= 0.0 or self.plucker_front_fy <= 0.0:
                    raise ValueError(f"Front Plücker focal lengths must be positive: {calibrated}.")
                if self.plucker_wrist_fx <= 0.0 or self.plucker_wrist_fy <= 0.0:
                    raise ValueError(f"Wrist Plücker focal lengths must be positive: {calibrated}.")
                if not 0.0 <= self.plucker_front_cx < self.plucker_image_width or not 0.0 <= self.plucker_front_cy < self.plucker_image_height:
                    raise ValueError(f"Front Plücker principal point is outside the image: {calibrated}.")
                if not 0.0 <= self.plucker_wrist_cx < self.plucker_image_width or not 0.0 <= self.plucker_wrist_cy < self.plucker_image_height:
                    raise ValueError(f"Wrist Plücker principal point is outside the image: {calibrated}.")
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
        if self.end_signal_prediction:
            if not str(self.end_signal_target_key):
                raise ValueError("`end_signal_target_key` must be non-empty when end prediction is enabled.")
            if self.end_signal_loss_weight < 0.0:
                raise ValueError("`end_signal_loss_weight` must be non-negative.")
            if not 0.0 < self.end_signal_init_probability < 1.0:
                raise ValueError("`end_signal_init_probability` must be in (0, 1).")
        if self.interaction_state_probe_only and not self.interaction_state_conditioning:
            raise ValueError(
                "`interaction_state_probe_only` requires `interaction_state_conditioning=true`."
            )
        if self.interaction_state_auxiliary_only and not self.interaction_state_conditioning:
            raise ValueError(
                "`interaction_state_auxiliary_only` requires `interaction_state_conditioning=true`."
            )
        if self.interaction_state_probe_only and self.interaction_state_auxiliary_only:
            raise ValueError(
                "`interaction_state_probe_only` and `interaction_state_auxiliary_only` are mutually exclusive."
            )
        if self.interaction_state_probe_freeze_main and not self.interaction_state_probe_only:
            raise ValueError(
                "`interaction_state_probe_freeze_main` requires `interaction_state_probe_only=true`."
            )
        if not 0.0 <= float(self.interaction_state_probe_dropout) < 1.0:
            raise ValueError(
                "interaction_state_probe_dropout must be in [0, 1), got "
                f"{self.interaction_state_probe_dropout}."
            )
        interaction_prediction_mode = str(self.interaction_state_prediction_mode).strip().lower()
        if interaction_prediction_mode not in {"encoder_current", "decoder_chunk"}:
            raise ValueError(
                "interaction_state_prediction_mode must be 'encoder_current' or 'decoder_chunk', got "
                f"{self.interaction_state_prediction_mode!r}."
            )
        self.interaction_state_prediction_mode = interaction_prediction_mode
        if self.interaction_state_conditioning:
            for key_name in (
                "interaction_contact_target_key",
                "interaction_handle_target_key",
                "interaction_door_target_key",
            ):
                if not str(getattr(self, key_name, "")):
                    raise ValueError(f"`{key_name}` must be non-empty when interaction conditioning is enabled.")
            for weight_name in (
                "interaction_contact_loss_weight",
                "interaction_handle_loss_weight",
                "interaction_door_loss_weight",
            ):
                if float(getattr(self, weight_name)) < 0.0:
                    raise ValueError(f"`{weight_name}` must be non-negative.")
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
        if not self.image_features and not self.env_state_feature and not self.point_cloud_features:
            raise ValueError("You must provide an image, point cloud, or environment state among the inputs.")
        if self.point_cloud_conditioning:
            point_cloud_features = self.point_cloud_features
            if list(point_cloud_features) != [self.point_cloud_key]:
                raise ValueError(
                    f"Point-cloud ACT requires exactly {self.point_cloud_key!r}; got {list(point_cloud_features)}."
                )
            if tuple(point_cloud_features[self.point_cloud_key].shape) != (self.point_cloud_num_points, 3):
                raise ValueError(
                    f"{self.point_cloud_key!r} must have shape ({self.point_cloud_num_points}, 3), got "
                    f"{point_cloud_features[self.point_cloud_key].shape}."
                )
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
        # Auxiliary targets such as ``aux.end_signal`` are intentionally not
        # policy inputs. They remain in the training batch and are validated
        # in ACTPolicy.forward(), while inference needs observations only.

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
