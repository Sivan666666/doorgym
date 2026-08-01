import math
import os

import pytest
import torch

os.environ.setdefault("LEROBOT_MINIMAL_ACT_IMPORTS", "1")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


IMAGE_KEY = "observation.images.wrist_masked_depth"
END_KEY = "aux.end_signal"


def make_config(enabled: bool) -> ACTConfig:
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
        end_signal_prediction=enabled,
        end_signal_init_probability=0.01,
    )


def make_batch() -> dict[str, torch.Tensor]:
    return {
        IMAGE_KEY: torch.randn(2, 3, 64, 64),
        OBS_STATE: torch.randn(2, 10),
        ACTION: torch.randn(2, 4, 10),
        "action_is_pad": torch.tensor([[False, False, True, True], [False, True, True, True]]),
        END_KEY: torch.tensor(
            [
                [[0.0], [1.0], [1.0], [1.0]],
                [[1.0], [0.0], [0.0], [0.0]],
            ]
        ),
    }


def test_disabled_mode_has_no_new_state_dict_keys():
    model = ACT(make_config(False))
    assert model.end_signal_head is None
    assert not any("end_signal_head" in key for key in model.state_dict())


def test_head_initializes_to_requested_probability():
    model = ACT(make_config(True)).eval()
    assert model.end_signal_head is not None
    assert torch.count_nonzero(model.end_signal_head.weight) == 0
    assert float(model.end_signal_head.bias) == pytest.approx(math.log(0.01 / 0.99))


def test_old_motion_weights_remain_bit_exact_when_head_is_added():
    torch.manual_seed(17)
    old_model = ACT(make_config(False)).eval()
    new_model = ACT(make_config(True)).eval()
    incompatible = new_model.load_state_dict(old_model.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "end_signal_head.weight",
        "end_signal_head.bias",
    }
    images = [torch.randn(2, 3, 64, 64)]
    state = torch.randn(2, 10)
    with torch.no_grad():
        old_motion = old_model({"observation.images": images, OBS_STATE: state})[0]
        new_motion = new_model({"observation.images": images, OBS_STATE: state})[0]
    torch.testing.assert_close(new_motion, old_motion, rtol=0.0, atol=0.0)


def test_masked_bce_ignores_episode_padding():
    policy = ACTPolicy(make_config(True))
    batch = make_batch()
    loss, metrics = policy.forward(batch)

    # Only targets [0, 1, 1] are valid. The five padded values must not
    # dilute the BCE denominator.
    logit = math.log(0.01 / 0.99)
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([logit, logit, logit]),
            torch.tensor([0.0, 1.0, 1.0]),
        )
        .item()
    )
    assert torch.isfinite(loss)
    assert metrics["end_signal_loss"] == pytest.approx(expected, rel=1.0e-6)


def test_end_signal_loss_only_updates_end_head_when_decoder_feature_is_detached():
    model = ACT(make_config(True)).train()
    assert model.config.end_signal_detach_decoder_feature
    assert model.end_signal_head is not None
    # Use non-zero weights so this test would send a gradient into the ACT
    # decoder if the feature were not detached.
    with torch.no_grad():
        model.end_signal_head.weight.fill_(0.1)

    batch = make_batch()
    model({"observation.images": [batch[IMAGE_KEY]], OBS_STATE: batch[OBS_STATE]})
    logits = model._last_end_signal_logits
    assert logits is not None
    end_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch[END_KEY])
    end_loss.backward()

    assert model.end_signal_head.weight.grad is not None
    assert torch.count_nonzero(model.end_signal_head.weight.grad) > 0
    assert model.end_signal_head.bias.grad is not None
    for name, parameter in model.named_parameters():
        if name.startswith("end_signal_head."):
            continue
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name


def test_detached_end_head_preserves_from_scratch_motion_initialization():
    torch.manual_seed(1000)
    original = ACT(make_config(False))
    torch.manual_seed(1000)
    augmented = ACT(make_config(True))
    for key, value in original.state_dict().items():
        torch.testing.assert_close(augmented.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_inference_returns_separate_probability_chunk():
    policy = ACTPolicy(make_config(True))
    batch = make_batch()
    motion, end_probability = policy.predict_action_chunk_with_end_signal(
        {IMAGE_KEY: batch[IMAGE_KEY], OBS_STATE: batch[OBS_STATE]}
    )
    assert motion.shape == (2, 4, 10)
    assert end_probability is not None
    assert end_probability.shape == (2, 4, 1)
    assert torch.all((end_probability >= 0.0) & (end_probability <= 1.0))
