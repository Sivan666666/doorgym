import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "door_act_shadow.py"
SPEC = importlib.util.spec_from_file_location("door_act_shadow_overlap_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_action(value: float, quat: np.ndarray, gripper: float) -> np.ndarray:
    action = np.zeros(10, dtype=np.float32)
    action[:5] = float(value)
    action[5:9] = np.asarray(quat, dtype=np.float32)
    action[9] = float(gripper)
    return action


def test_step6_observation_aligns_new_action0_to_step6():
    old_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    new_quat = np.asarray([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32)
    old_chunk = np.stack([make_action(i, old_quat, -1.0) for i in range(10)])
    new_chunk = np.stack([make_action(100 + i, -new_quat, -0.25) for i in range(10)])

    buffer = MODULE.EEActionOverlapBuffer(old_weight=0.3, new_weight=0.7)
    first = buffer.ingest(old_chunk, start_timestep=0, current_timestep=0)
    assert first["appended"] == 10
    for step in range(7):
        assert buffer.pop(expected_timestep=step).timestep == step

    # Observation at step 6 predicts new action[0] for global step 6. If the
    # result is only ingested at step 7, new action[0] is already stale.
    merged = buffer.ingest(new_chunk, start_timestep=6, current_timestep=7)
    assert merged["stale_skipped"] == 1
    assert merged["overlap_blended"] == 3
    assert merged["appended"] == 6

    action7 = buffer.pop(expected_timestep=7)
    np.testing.assert_allclose(action7.action[:5], 0.3 * 7.0 + 0.7 * 101.0, atol=1.0e-6)
    np.testing.assert_allclose(action7.action[9], -0.25, atol=1.0e-6)
    # -new_quat represents the same rotation. Shortest-path SLERP must flip
    # that sign before interpolation and return a normalized quaternion.
    assert abs(float(np.linalg.norm(action7.action[5:9])) - 1.0) < 1.0e-6
    assert action7.action[8] >= 0.0
    assert action7.source == "overlap_0.3_old_0.7_new"


def test_late_chunk_skips_expired_prefix_and_blends_current_step():
    quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    old_chunk = np.stack([make_action(i, quat, -1.0) for i in range(10)])
    new_chunk = np.stack([make_action(100 + i, quat, -0.5) for i in range(10)])

    buffer = MODULE.EEActionOverlapBuffer(old_weight=0.3, new_weight=0.7)
    buffer.ingest(old_chunk, start_timestep=0, current_timestep=0)
    for step in range(8):
        buffer.pop(expected_timestep=step)

    # The step-6 request starts at global step 6, but inference only becomes
    # available at step 8. new_chunk[0:2] are expired and must be discarded.
    merged = buffer.ingest(new_chunk, start_timestep=6, current_timestep=8)
    assert merged["stale_skipped"] == 2
    assert merged["overlap_blended"] == 2
    assert merged["appended"] == 6

    action8 = buffer.pop(expected_timestep=8)
    np.testing.assert_allclose(action8.action[:5], 0.3 * 8.0 + 0.7 * 102.0, atol=1.0e-6)
    np.testing.assert_allclose(action8.action[9], -0.5, atol=1.0e-6)
