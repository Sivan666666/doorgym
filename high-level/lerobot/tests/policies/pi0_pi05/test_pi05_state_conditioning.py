#!/usr/bin/env python

import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_STATE


def _make_transition():
    return {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, 9)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pull lever door open"]},
    }


def test_state_conditioning_is_disabled_by_default():
    assert PI05Config().state_conditioning is False


def test_legacy_prompt_keeps_quantized_state():
    step = Pi05PrepareStateTokenizerProcessorStep(
        max_state_dim=32,
        include_state_in_prompt=True,
    )
    prompt = step(_make_transition())[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    assert "State:" in prompt


def test_continuous_state_mode_uses_task_only_prompt():
    step = Pi05PrepareStateTokenizerProcessorStep(
        max_state_dim=32,
        include_state_in_prompt=False,
    )
    prompt = step(_make_transition())[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    assert prompt == "Task: pull lever door open;\nAction: "
    assert "State:" not in prompt
