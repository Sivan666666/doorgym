import os

import pytest
import torch

os.environ.setdefault("LEROBOT_MINIMAL_ACT_IMPORTS", "1")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import (
    ACT,
    ACTDP3PointNetEncoderXYZ,
    ACTOBSBenchPointNet,
    ACTPointCloudGlobalEncoder,
    ACTPointCloudLegacyGlobalEncoder,
    ACTPointCloudLegacyLocalEncoder,
    ACTPointCloudLocalEncoder,
    deterministic_farthest_point_indices,
    obsbench_farthest_point_indices,
)
from lerobot.utils.constants import ACTION, OBS_STATE


POINT_KEY = "observation.point_cloud"


def make_config(
    enabled=True,
    mode="obsbench_local",
    *,
    interaction_state=False,
):
    inputs = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))}
    if enabled:
        inputs[POINT_KEY] = PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(32, 3))
    else:
        inputs["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(4,))
    return ACTConfig(
        input_features=inputs,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
        point_cloud_conditioning=enabled,
        point_cloud_encoder_mode=mode,
        point_cloud_num_points=32,
        point_cloud_num_tokens=8,
        point_cloud_knn_k=4,
        interaction_state_conditioning=interaction_state,
        interaction_state_prediction_mode="decoder_chunk",
        interaction_state_probe_only=False,
        pretrained_backbone_weights=None,
        use_vae=False,
        chunk_size=4,
        n_action_steps=2,
        dim_model=48,
        n_heads=8,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )


def test_disabled_mode_has_no_point_cloud_parameters():
    model = ACT(make_config(enabled=False))
    assert not hasattr(model, "point_cloud_encoder")
    assert not any("point_cloud" in key for key in model.state_dict())


def test_global_and_local_token_shapes_and_deterministic_fps():
    torch.manual_seed(4)
    xyz = torch.randn(2, 32, 3)
    first = deterministic_farthest_point_indices(xyz, 8)
    second = deterministic_farthest_point_indices(xyz, 8)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    global_encoder = ACTPointCloudGlobalEncoder(dim_model=48, global_dim=16).eval()
    global_token, global_pos = global_encoder(xyz)
    assert global_token.shape == global_pos.shape == (2, 1, 48)

    local_encoder = ACTPointCloudLocalEncoder(
        dim_model=48,
        num_tokens=8,
        knn_k=4,
        workspace_min=(-2.0, -2.0, -2.0),
        workspace_max=(2.0, 2.0, 2.0),
    ).eval()
    local_token_1, local_pos_1 = local_encoder(xyz)
    local_token_2, local_pos_2 = local_encoder(xyz)
    assert local_token_1.shape == local_pos_1.shape == (2, 8, 48)
    torch.testing.assert_close(local_token_1, local_token_2, rtol=0, atol=0)
    torch.testing.assert_close(local_pos_1, local_pos_2, rtol=0, atol=0)


def test_dp3_global_matches_original_pointnet_xyz_topology():
    encoder = ACTDP3PointNetEncoderXYZ(out_channels=64)
    linears = [module for module in encoder.modules() if isinstance(module, torch.nn.Linear)]
    layer_norms = [module for module in encoder.modules() if isinstance(module, torch.nn.LayerNorm)]
    assert [(module.in_features, module.out_features) for module in linears] == [
        (3, 64),
        (64, 128),
        (128, 256),
        (256, 64),
    ]
    assert [tuple(module.normalized_shape) for module in layer_norms] == [(64,), (128,), (256,), (64,)]
    assert encoder(torch.randn(2, 32, 3)).shape == (2, 64)


def test_legacy_global_encoder_remains_available_for_old_point_cloud_checkpoints():
    encoder = ACTPointCloudLegacyGlobalEncoder(dim_model=48, global_dim=16).eval()
    tokens, positions = encoder(torch.randn(2, 32, 3))
    assert tokens.shape == positions.shape == (2, 1, 48)


def test_obsbench_pointnet_matches_default_backbone_topology():
    pointnet = ACTOBSBenchPointNet(in_channels=3)
    convolutions = [module for module in pointnet.modules() if isinstance(module, torch.nn.Conv1d)]
    batch_norms = [module for module in pointnet.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    assert [(module.in_channels, module.out_channels) for module in convolutions] == [
        (3, 64),
        (64, 64),
        (64, 64),
        (64, 128),
        (128, 512),
    ]
    assert all(module.kernel_size == (1,) and module.bias is None for module in convolutions)
    assert len(batch_norms) == 5
    assert all(module.eps == pytest.approx(1.0e-3) for module in batch_norms)
    assert all(module.momentum == pytest.approx(0.01) for module in batch_norms)
    output = pointnet.eval()(torch.randn(2, 32, 3))
    assert output.shape == (2, 32, 512)


def test_obsbench_fps_starts_at_zero_like_pointops():
    xyz = torch.tensor(
        [[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    )
    indices = obsbench_farthest_point_indices(xyz, 3)
    assert indices.tolist() == [[0, 2, 1]]


def test_legacy_local_encoder_remains_available_for_old_point_cloud_checkpoints():
    encoder = ACTPointCloudLegacyLocalEncoder(
        dim_model=48,
        num_tokens=8,
        knn_k=4,
        workspace_min=(-2.0, -2.0, -2.0),
        workspace_max=(2.0, 2.0, 2.0),
    ).eval()
    tokens, positions = encoder(torch.randn(2, 32, 3))
    assert tokens.shape == positions.shape == (2, 8, 48)


@pytest.mark.parametrize(
    "mode",
    [
        "dp3_global",
        "dp3_global_legacy",
        "obsbench_local",
        "obsbench_post_pointnet",
        "obsbench_local_legacy",
    ],
)
def test_point_cloud_act_forward_shape(mode):
    model = ACT(make_config(mode=mode)).eval()
    batch = {OBS_STATE: torch.randn(2, 10), POINT_KEY: torch.randn(2, 32, 3)}
    with torch.no_grad():
        actions = model(batch)[0]
    assert actions.shape == (2, 4, 10)


def test_point_cloud_and_images_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ACTConfig(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
                POINT_KEY: PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(1024, 3)),
                "observation.images.front_masked_depth": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 64, 64)
                ),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
            point_cloud_conditioning=True,
        )


def test_obsbench_local_supports_decoder_chunk_interaction_prediction():
    model = ACT(make_config(mode="obsbench_local", interaction_state=True)).train()
    batch = {OBS_STATE: torch.randn(2, 10), POINT_KEY: torch.randn(2, 32, 3)}

    actions = model(batch)[0]
    probabilities = model._last_interaction_state_probabilities

    assert actions.shape == (2, 4, 10)
    assert probabilities is not None and probabilities.shape == (2, 4, 3)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))

    model.zero_grad(set_to_none=True)
    model(batch)
    model._last_interaction_state_logits.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.decoder.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.point_cloud_encoder.parameters()
    )


def test_old_obsbench_local_checkpoint_warmstarts_decoder_chunk_head_bit_exactly():
    torch.manual_seed(719)
    original = ACT(make_config(mode="obsbench_local", interaction_state=False)).eval()
    augmented = ACT(make_config(mode="obsbench_local", interaction_state=True)).eval()
    incompatible = augmented.load_state_dict(original.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        key.startswith("interaction_state_chunk_head.") for key in incompatible.missing_keys
    )

    batch = {OBS_STATE: torch.randn(2, 10), POINT_KEY: torch.randn(2, 32, 3)}
    with torch.no_grad():
        original_actions = original(batch)[0]
        augmented_actions = augmented(batch)[0]
    torch.testing.assert_close(augmented_actions, original_actions, rtol=0.0, atol=0.0)
