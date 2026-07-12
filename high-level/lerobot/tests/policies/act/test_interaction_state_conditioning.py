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


def make_config(enabled: bool, *, end_signal: bool = False) -> ACTConfig:
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
        end_signal_prediction=end_signal,
    )


def model_batch(batch_size: int = 2):
    return {
        OBS_IMAGES: [torch.randn(batch_size, 3, 64, 64)],
        OBS_STATE: torch.randn(batch_size, 10),
    }


def policy_batch(batch_size: int = 2):
    return {
        IMAGE_KEY: torch.randn(batch_size, 3, 64, 64),
        OBS_STATE: torch.randn(batch_size, 10),
        ACTION: torch.randn(batch_size, 4, 10),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        CONTACT_KEY: torch.tensor([[0.0], [1.0]]),
        HANDLE_KEY: torch.tensor([[0.25], [0.75]]),
        DOOR_KEY: torch.tensor([[0.1], [0.9]]),
    }


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


def test_interaction_and_end_signal_heads_can_coexist_without_changing_action_dim():
    model = ACT(make_config(True, end_signal=True)).eval()
    with torch.no_grad():
        action = model(model_batch())[0]
    assert action.shape == (2, 4, 10)
    assert model._last_interaction_state_probabilities.shape == (2, 3)
    assert model._last_end_signal_logits.shape == (2, 4, 1)
