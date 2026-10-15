"""Stable contracts shared by PhysX, MeisterRender and LeRobot processes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ACTION_NAMES = ['base_vx', 'base_vy', 'base_vyaw'] + [f'joint{i}' for i in range(1, 7)] + ['jointGripper']
STATE_NAMES = ['previous_command_vx', 'previous_command_vy', 'previous_command_vyaw'] + ACTION_NAMES[3:]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def indices(data, fps):
    source_fps = data['fps']
    assert fps > 0 and source_fps % fps == 0, 'Output FPS must divide physics FPS'
    assert len(data['frames']) >= 2
    return list(range(0, len(data['frames']) - 1, source_fps // fps))


def transition(data, i):
    """POST-step observation i paired with the ACTUAL command for step i+1."""
    import numpy as np
    current = data['frames'][i]
    following = data['frames'][i + 1]
    assert following['step'] == current['step'] + 1
    prev = current['control_record']
    command = following['control_record']
    state = np.array(prev['applied_action10'][:3] + prev['measured_joint7'], dtype=np.float32)
    action = np.array(command['applied_action10'], dtype=np.float32)
    assert state.shape == action.shape == (10,)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    return state, action


def validate_rollout(path, expected_steps, min_door_deg=25, min_traverse_m=.5):
    import math
    import numpy as np
    data = read(path)
    assert len(data['frames']) == expected_steps, 'Incomplete physical rollout'
    for i, frame in enumerate(data['frames']):
        assert frame['step'] == i
        c = frame['control_record']
        assert len(c['applied_action10']) == 10
        values = [c['commanded_dof_positions'][c['dof_names'].index(n)] for n in ACTION_NAMES[3:]]
        assert np.allclose(values, c['applied_action10'][3:], atol=1e-7), 'Action is not the sent joint target'
        if i + 1 < len(data['frames']):
            transition(data, i)
    maximum = max(abs(math.degrees(f['replay_door_dof_pos'][0])) for f in data['frames'])
    first, last = data['frames'][0], data['frames'][-1]
    # This WC adapter pulls along simulation-world -X. Do not reuse this gate
    # for other door orientations without replacing it with an asset adapter.
    traverse = first['replay_door_root_state'][0] - last['replay_root_state'][0]
    assert maximum >= min_door_deg and traverse >= min_traverse_m, (maximum, traverse)
    assert last['phase'] == 'hold_home', 'Incomplete controller episode'
    return dict(passed=True, frames=len(data['frames']), max_door_deg=maximum,
                traverse_m=traverse, source_sha256=sha(path), actions_are_command_targets=True)
