import numpy as np

from dp.door_dp_common import (
    INTERACTION_CONTACT_FEATURE,
    INTERACTION_DOOR_PROGRESS_FEATURE,
    INTERACTION_HANDLE_PROGRESS_FEATURE,
    make_interaction_state_targets,
)


def test_contact_requires_three_causal_frames_and_resets_immediately():
    contact = [0, 1, 1, 1, 1, 0, 1, 1, 1]
    dof = np.zeros((len(contact), 2), dtype=np.float32)
    targets = make_interaction_state_targets(contact, dof, contact_min_consecutive_frames=3)
    assert targets[INTERACTION_CONTACT_FEATURE][:, 0].tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 1]


def test_progress_is_sign_invariant_cumulative_and_clipped():
    contact = np.zeros(6, dtype=np.float32)
    dof = np.asarray(
        [
            [0.0, 0.0],
            [-0.25, 0.2],
            [-0.5, 0.5],
            [-1.0, 0.8],
            [-0.7, 0.1],
            [-1.8, 0.0],
        ],
        dtype=np.float32,
    )
    targets = make_interaction_state_targets(
        contact,
        dof,
        handle_unlock_delta_rad=0.7,
        door_goal_delta_rad=1.57,
    )
    handle = targets[INTERACTION_HANDLE_PROGRESS_FEATURE][:, 0]
    door = targets[INTERACTION_DOOR_PROGRESS_FEATURE][:, 0]
    assert np.all(np.diff(handle) >= 0.0)
    assert np.all(np.diff(door) >= 0.0)
    assert handle[3] == 1.0 and handle[-1] == 1.0
    assert door[-1] == 1.0
    assert np.all((handle >= 0.0) & (handle <= 1.0))
    assert np.all((door >= 0.0) & (door <= 1.0))
