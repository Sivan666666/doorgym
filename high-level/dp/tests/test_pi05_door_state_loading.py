import json
from types import SimpleNamespace

import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file

from dp.door_policy_backend import (
    LeRobotPI05DoorPolicyBackend,
    make_lerobot_pi05_config,
    validate_pi05_state_projection,
    validate_native_pi05_evo_weights,
)
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep


def check_state_flag_reaches_config(enabled):
    config = make_lerobot_pi05_config(
        state_dim=9, action_dim=9, chunk_size=100, action_horizon=25,
        image_keys=["observation.images.front_rgb"], device="cpu",
        state_conditioning=enabled, max_state_dim=32,
    )
    assert config.state_conditioning is enabled


def make_policy(tmp_path):
    policy = torch.nn.Module()
    policy.config = SimpleNamespace(state_conditioning=True)
    policy.model = torch.nn.Module()
    policy.model.state_proj = torch.nn.Linear(32, 4)
    (tmp_path / "config.json").write_text(json.dumps({"state_conditioning": True}))
    save_file(policy.state_dict(), str(tmp_path / "model.safetensors"))
    return policy


def check_projection_weights_are_loaded_exactly(tmp_path):
    policy = make_policy(tmp_path)
    validate_pi05_state_projection(policy, tmp_path)
    with torch.no_grad():
        policy.model.state_proj.bias.add_(1)
    with unittest.TestCase().assertRaisesRegex(ValueError, "not loaded exactly"):
        validate_pi05_state_projection(policy, tmp_path)


def check_dropped_state_flag_is_an_error(tmp_path):
    policy = make_policy(tmp_path)
    policy.config.state_conditioning = False
    with unittest.TestCase().assertRaisesRegex(ValueError, "mismatch"):
        validate_pi05_state_projection(policy, tmp_path)


def check_missing_projection_is_an_error(tmp_path):
    policy = make_policy(tmp_path)
    save_file({"other": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
    with unittest.TestCase().assertRaisesRegex(ValueError, "missing"):
        validate_pi05_state_projection(policy, tmp_path)


def check_prompt_processor_serializes_behavior(include):
    processor = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=64, include_state_in_prompt=include)
    restored = Pi05PrepareStateTokenizerProcessorStep(**processor.get_config())
    assert restored.max_state_dim == 64
    assert restored.include_state_in_prompt is include


def check_saved_processor_receives_raw_unclipped_state_and_correct_task():
    backend = object.__new__(LeRobotPI05DoorPolicyBackend)
    backend.task_prompt = "pull lever door open"
    backend.preprocessor = lambda x: x
    state = torch.tensor([[10.0] * 9, [-10.0] * 9])
    inputs = {"observation.state": state, "action": torch.zeros(2, 100, 9)}
    output = backend._prepare_batch(inputs)
    assert output["observation.state"] is state
    assert output["task"] == [backend.task_prompt] * 2
    assert "action" not in output
    assert "action" in inputs


class Pi05DoorStateTests(unittest.TestCase):
    def test_native_evo_partial_load_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            policy = torch.nn.Module()
            policy.config = SimpleNamespace()
            policy.model = torch.nn.Linear(2, 2)
            policy._fix_pytorch_state_dict_keys = lambda weights, cfg: weights
            save_file(policy.state_dict(), str(root / 'model.safetensors'))
            validate_native_pi05_evo_weights(policy, root)
            with torch.no_grad():
                policy.model.weight.add_(1)
            with self.assertRaisesRegex(ValueError, 'not loaded exactly'):
                validate_native_pi05_evo_weights(policy, root)
            save_file({'model.weight': policy.model.weight.detach()}, str(root / 'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'keys mismatch'):
                validate_native_pi05_evo_weights(policy, root)

    def test_config(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                check_state_flag_reaches_config(enabled)

    def test_exact_weights(self):
        with tempfile.TemporaryDirectory() as path:
            check_projection_weights_are_loaded_exactly(Path(path))

    def test_dropped_flag(self):
        with tempfile.TemporaryDirectory() as path:
            check_dropped_state_flag_is_an_error(Path(path))

    def test_missing_weights(self):
        with tempfile.TemporaryDirectory() as path:
            check_missing_projection_is_an_error(Path(path))

    def test_stale_wrapper_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({"state_conditioning": True}))
            (root / "door_policy_meta.json").write_text(json.dumps({
                "policy_config": {"source_official_policy_dir": str(source), "state_conditioning": False}
            }))
            with self.assertRaisesRegex(ValueError, "Stale PI05 wrapper"):
                LeRobotPI05DoorPolicyBackend.load(root)

    def test_processor_serialization(self):
        for include in (False, True):
            with self.subTest(include=include):
                check_prompt_processor_serializes_behavior(include)

    def test_raw_input(self):
        check_saved_processor_receives_raw_unclipped_state_and_correct_task()


if __name__ == "__main__":
    unittest.main()
