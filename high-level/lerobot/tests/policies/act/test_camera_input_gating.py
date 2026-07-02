import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTCameraInputGating, ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


WRIST_KEY = "observation.images.wrist_masked_depth"
FRONT_KEY = "observation.images.front_masked_depth"


def make_act_config(camera_input_gating: bool) -> ACTConfig:
    return ACTConfig(
        input_features={
            # Keep the real door-dataset order to verify semantic key resolution.
            WRIST_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            FRONT_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
        },
        device="cpu",
        pretrained_backbone_weights=None,
        camera_input_gating=camera_input_gating,
        camera_input_gating_hidden_dim=16,
        use_vae=False,
        chunk_size=4,
        n_action_steps=4,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )


def make_batch(batch_size: int = 2) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    return {
        # OBS_IMAGES follows config.image_features order: wrist, then front.
        OBS_IMAGES: [
            torch.randn(batch_size, 3, 64, 64),
            torch.randn(batch_size, 3, 64, 64),
        ],
        OBS_STATE: torch.randn(batch_size, 10),
    }


def test_camera_input_gating_is_disabled_by_default():
    config = make_act_config(camera_input_gating=False)
    model = ACT(config)

    assert config.camera_input_gating is False
    assert not hasattr(model, "camera_input_gate")
    assert not any(key.startswith("camera_input_gate.") for key in model.state_dict())


def test_enabled_camera_gating_starts_exactly_as_original_act():
    torch.manual_seed(7)
    original = ACT(make_act_config(camera_input_gating=False)).eval()
    torch.manual_seed(7)
    gated = ACT(make_act_config(camera_input_gating=True)).eval()

    # An original checkpoint can initialize the gated model with strict=False;
    # only the new identity-initialized gating parameters are missing.
    incompatible = gated.load_state_dict(original.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(key.startswith("camera_input_gate.") for key in incompatible.missing_keys)

    batch = make_batch()
    with torch.no_grad():
        original_actions = original(batch)[0]
        gated_actions = gated(batch)[0]

    torch.testing.assert_close(gated_actions, original_actions, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        gated._last_camera_gates,
        torch.ones_like(gated._last_camera_gates),
        rtol=0.0,
        atol=0.0,
    )
    assert gated.camera_input_gating_wrist_index == 0
    assert gated.camera_input_gating_front_index == 1


def test_two_times_softmax_gates_are_learnable_and_sum_to_two():
    gate = ACTCameraInputGating(dim_model=8, robot_state_dim=3, hidden_dim=4)
    front = torch.randn(2, 8, 3, 3)
    wrist = torch.randn(2, 8, 3, 3)
    state = torch.randn(2, 3)

    initial = gate(front, wrist, state)
    torch.testing.assert_close(initial, torch.ones_like(initial), rtol=0.0, atol=0.0)
    initial[:, 0].sum().backward()
    assert gate.mlp[-1].weight.grad is not None
    assert torch.count_nonzero(gate.mlp[-1].weight.grad) > 0

    with torch.no_grad():
        gate.mlp[-1].bias.copy_(torch.tensor([1.0, -1.0]))
    learned = gate(front, wrist, state)
    torch.testing.assert_close(learned.sum(dim=-1), torch.full((2,), 2.0))
    assert torch.all(learned[:, 0] > 1.0)
    assert torch.all(learned[:, 1] < 1.0)


def test_policy_training_metrics_report_front_and_wrist_gates():
    policy = ACTPolicy(make_act_config(camera_input_gating=True))
    batch_size = 2
    batch = {
        WRIST_KEY: torch.randn(batch_size, 3, 64, 64),
        FRONT_KEY: torch.randn(batch_size, 3, 64, 64),
        OBS_STATE: torch.randn(batch_size, 10),
        ACTION: torch.randn(batch_size, 4, 10),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }

    loss, metrics = policy.forward(batch)

    assert torch.isfinite(loss)
    assert metrics["camera_gate_front"] == pytest.approx(1.0)
    assert metrics["camera_gate_wrist"] == pytest.approx(1.0)


def test_camera_gating_requires_unambiguous_front_and_wrist_keys():
    config = make_act_config(camera_input_gating=True)
    config.input_features = {
        "observation.images.camera_0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.images.camera_1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
    }

    with pytest.raises(ValueError, match="Could not uniquely infer the front camera"):
        ACT(config)
