import os

import pytest
import torch

os.environ.setdefault("LEROBOT_MINIMAL_ACT_IMPORTS", "1")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


IMAGE_KEY = "observation.images.wrist_masked_depth"
CONTACT_KEY = "aux.interaction_contact"
HANDLE_KEY = "aux.interaction_handle_progress"
DOOR_KEY = "aux.interaction_door_progress"


def make_config(
    enabled: bool,
    *,
    prediction_mode: str = "encoder_current",
    end_signal: bool = False,
    probe_only: bool = False,
    auxiliary_only: bool = False,
    freeze_main: bool = False,
) -> ACTConfig:
    return ACTConfig(
        input_features={
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
        device="cpu",
        pretrained_backbone_weights=None,
        use_vae=False,
        chunk_size=4,
        n_action_steps=4,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
        interaction_state_conditioning=enabled,
        interaction_state_prediction_mode=prediction_mode,
        interaction_state_probe_only=probe_only,
        interaction_state_auxiliary_only=auxiliary_only,
        interaction_state_probe_freeze_main=freeze_main,
        end_signal_prediction=end_signal,
    )


def model_batch(batch_size: int = 2):
    return {
        OBS_IMAGES: [torch.randn(batch_size, 3, 64, 64)],
        OBS_STATE: torch.randn(batch_size, 10),
    }


def policy_batch(batch_size: int = 2, *, chunk_targets: bool = False):
    batch = {
        IMAGE_KEY: torch.randn(batch_size, 3, 64, 64),
        OBS_STATE: torch.randn(batch_size, 10),
        ACTION: torch.randn(batch_size, 4, 10),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        CONTACT_KEY: torch.tensor([[0.0], [1.0]]),
        HANDLE_KEY: torch.tensor([[0.25], [0.75]]),
        DOOR_KEY: torch.tensor([[0.1], [0.9]]),
    }
    if chunk_targets:
        batch[CONTACT_KEY] = torch.tensor(
            [[[0.0], [0.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [0.0]]]
        )
        batch[HANDLE_KEY] = torch.tensor(
            [[[0.0], [0.2], [0.6], [1.0]], [[0.4], [0.6], [0.8], [1.0]]]
        )
        batch[DOOR_KEY] = torch.tensor(
            [[[0.0], [0.0], [0.1], [0.3]], [[0.0], [0.2], [0.5], [0.9]]]
        )
        for key in (CONTACT_KEY, HANDLE_KEY, DOOR_KEY):
            batch[f"{key}_is_pad"] = torch.zeros(batch_size, 4, dtype=torch.bool)
    return batch


def test_disabled_mode_has_no_interaction_parameters():
    model = ACT(make_config(False))
    assert not hasattr(model, "interaction_state_head")
    assert not hasattr(model, "interaction_conditioner")
    assert not any("interaction_" in key for key in model.state_dict())


def test_old_motion_output_is_bit_exact_after_zero_residual_warmstart():
    torch.manual_seed(31)
    original = ACT(make_config(False)).eval()
    augmented = ACT(make_config(True)).eval()
    incompatible = augmented.load_state_dict(original.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        key.startswith("interaction_state_head.") or key.startswith("interaction_conditioner.")
        for key in incompatible.missing_keys
    )
    batch = model_batch()
    with torch.no_grad():
        original_action = original(batch)[0]
        augmented_action = augmented(batch)[0]
    torch.testing.assert_close(augmented_action, original_action, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(augmented.interaction_conditioner[-1].weight) == 0
    assert torch.count_nonzero(augmented.interaction_conditioner[-1].bias) == 0


def test_interaction_predictions_losses_and_gradients_are_finite():
    policy = ACTPolicy(make_config(True))
    loss, metrics = policy.forward(policy_batch())
    loss.backward()
    assert torch.isfinite(loss)
    for key in (
        "interaction_contact_loss",
        "interaction_handle_loss",
        "interaction_door_loss",
        "interaction_contact_probability",
        "interaction_handle_progress",
        "interaction_door_progress",
    ):
        assert key in metrics
        assert metrics[key] == pytest.approx(metrics[key])
    probabilities = policy.model._last_interaction_state_probabilities
    assert probabilities is not None and probabilities.shape == (2, 3)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert policy.model.interaction_conditioner[-1].weight.grad is not None
    assert torch.count_nonzero(policy.model.interaction_conditioner[-1].weight.grad) > 0


def test_probe_only_is_read_only_and_does_not_feed_back_into_motion_decoder():
    model = ACT(make_config(True, probe_only=True)).eval()
    batch = model_batch()
    with torch.no_grad():
        action_before = model(batch)[0]
        for parameter in model.interaction_conditioner.parameters():
            parameter.fill_(10.0)
        action_after = model(batch)[0]
    torch.testing.assert_close(action_after, action_before, rtol=0.0, atol=0.0)

    model.train()
    model.zero_grad(set_to_none=True)
    model(batch)
    logits = model._last_interaction_state_logits
    assert logits is not None
    interaction_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], torch.tensor([0.0, 1.0])
        )
        + torch.nn.functional.smooth_l1_loss(torch.sigmoid(logits[:, 1]), torch.tensor([0.25, 0.75]))
        + torch.nn.functional.smooth_l1_loss(torch.sigmoid(logits[:, 2]), torch.tensor([0.1, 0.9]))
    )
    interaction_loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.interaction_state_head.parameters()
    )
    for name, parameter in model.named_parameters():
        if name.startswith("interaction_state_head."):
            continue
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name

    model.zero_grad(set_to_none=True)
    motion = model(batch)[0]
    motion.square().mean().backward()
    assert model.action_head.weight.grad is not None
    assert torch.count_nonzero(model.action_head.weight.grad) > 0
    for name, parameter in model.interaction_state_head.named_parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name
    for name, parameter in model.interaction_conditioner.named_parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name


def test_probe_only_preserves_from_scratch_main_initialization_and_train_motion_rng():
    torch.manual_seed(1000)
    original = ACT(make_config(False))
    torch.manual_seed(1000)
    probe = ACT(make_config(True, probe_only=True))

    original_state = original.state_dict()
    probe_state = probe.state_dict()
    for key, value in original_state.items():
        assert key in probe_state
        torch.testing.assert_close(probe_state[key], value, rtol=0.0, atol=0.0)

    batch = model_batch()
    original.train()
    probe.train()
    torch.manual_seed(777)
    original_action = original(batch)[0]
    torch.manual_seed(777)
    probe_action = probe(batch)[0]
    torch.testing.assert_close(probe_action, original_action, rtol=0.0, atol=0.0)


def test_auxiliary_only_preserves_initial_train_motion_path_but_allows_encoder_gradients():
    torch.manual_seed(1000)
    original = ACT(make_config(False))
    torch.manual_seed(1000)
    auxiliary = ACT(make_config(True, auxiliary_only=True))

    original_state = original.state_dict()
    auxiliary_state = auxiliary.state_dict()
    for key, value in original_state.items():
        torch.testing.assert_close(auxiliary_state[key], value, rtol=0.0, atol=0.0)

    batch = model_batch()
    original.train()
    auxiliary.train()
    torch.manual_seed(777)
    original_action = original(batch)[0]
    torch.manual_seed(777)
    auxiliary_action = auxiliary(batch)[0]
    torch.testing.assert_close(auxiliary_action, original_action, rtol=0.0, atol=0.0)

    logits = auxiliary._last_interaction_state_logits
    assert logits is not None
    logits.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in auxiliary.encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in auxiliary.decoder.parameters())


def test_frozen_probe_only_trains_only_interaction_head():
    policy = ACTPolicy(make_config(True, probe_only=True, freeze_main=True))
    trainable = [name for name, parameter in policy.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("interaction_state_head." in name for name in trainable)


def test_interaction_and_end_signal_heads_can_coexist_without_changing_action_dim():
    model = ACT(make_config(True, end_signal=True)).eval()
    with torch.no_grad():
        action = model(model_batch())[0]
    assert action.shape == (2, 4, 10)
    assert model._last_interaction_state_probabilities.shape == (2, 3)
    assert model._last_end_signal_logits.shape == (2, 4, 1)


def test_decoder_chunk_predicts_one_interaction_state_per_action_timestep():
    model = ACT(make_config(True, prediction_mode="decoder_chunk")).eval()
    with torch.no_grad():
        action = model(model_batch())[0]
    assert action.shape == (2, 4, 10)
    assert not hasattr(model, "interaction_state_head")
    assert not hasattr(model, "interaction_conditioner")
    probabilities = model._last_interaction_state_probabilities
    assert probabilities is not None and probabilities.shape == (2, 4, 3)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_decoder_chunk_loss_ignores_episode_tail_padding():
    torch.manual_seed(123)
    policy = ACTPolicy(make_config(True, prediction_mode="decoder_chunk")).eval()
    batch = policy_batch(chunk_targets=True)
    batch["action_is_pad"][:, 2:] = True
    for key in (CONTACT_KEY, HANDLE_KEY, DOOR_KEY):
        batch[f"{key}_is_pad"][:, 2:] = True

    changed_padded_targets = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    for key in (CONTACT_KEY, HANDLE_KEY, DOOR_KEY):
        changed_padded_targets[key][:, 2:] = 1.0 - changed_padded_targets[key][:, 2:]

    loss_a, metrics_a = policy.forward(batch)
    loss_b, metrics_b = policy.forward(changed_padded_targets)
    torch.testing.assert_close(loss_a, loss_b, rtol=0.0, atol=0.0)
    for key in (
        "interaction_contact_loss",
        "interaction_handle_loss",
        "interaction_door_loss",
    ):
        assert metrics_a[key] == pytest.approx(metrics_b[key], abs=0.0, rel=0.0)


def test_decoder_chunk_interaction_loss_updates_decoder_without_probe_detach():
    model = ACT(make_config(True, prediction_mode="decoder_chunk")).train()
    model.zero_grad(set_to_none=True)
    model(model_batch())
    logits = model._last_interaction_state_logits
    assert logits is not None and logits.shape == (2, 4, 3)
    logits.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.interaction_state_chunk_head.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.decoder.parameters()
    )


def test_decoder_chunk_probe_only_updates_only_chunk_head():
    model = ACT(make_config(True, prediction_mode="decoder_chunk", probe_only=True)).train()
    model.zero_grad(set_to_none=True)
    model(model_batch())
    logits = model._last_interaction_state_logits
    assert logits is not None
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith("interaction_state_chunk_head."):
            assert parameter.grad is not None
        else:
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name


def test_decoder_chunk_warmstart_does_not_change_old_motion_output():
    torch.manual_seed(131)
    original = ACT(make_config(False)).eval()
    augmented = ACT(make_config(True, prediction_mode="decoder_chunk")).eval()
    incompatible = augmented.load_state_dict(original.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(key.startswith("interaction_state_chunk_head.") for key in incompatible.missing_keys)
    batch = model_batch()
    with torch.no_grad():
        original_action = original(batch)[0]
        augmented_action = augmented(batch)[0]
    torch.testing.assert_close(augmented_action, original_action, rtol=0.0, atol=0.0)
