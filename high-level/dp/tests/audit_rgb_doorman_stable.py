"""Compare stable RGB recordings against the earlier same-seed physical rollout."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--previews', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--physics_reference_dump')
    parser.add_argument('--physics_current_dump')
    args = parser.parse_args()
    fields = ['state', 'action', 'subtask_index', 'replay_root_state', 'replay_dof_pos', 'replay_dof_vel',
              'replay_ee_pos', 'replay_ee_quat', 'replay_door_root_state', 'replay_door_dof_pos',
              'replay_door_dof_vel', 'replay_door_open_stage', 'scripted_target_ee_pos_world',
              'scripted_target_ee_quat_world', 'front_camera_pose_base', 'wrist_camera_pose_base',
              'gripper_handle_contact_score', 'gripper_handle_contact_count',
              'gripper_handle_contact_any', 'gripper_handle_contact_both']
    with np.load(args.reference, allow_pickle=True) as old, np.load(args.raw, allow_pickle=True) as new:
        length = len(new['state'])
        checks = {}
        for key in fields:
            a, b = old[key][:length], new[key]
            checks[key] = dict(exact=bool(np.array_equal(a, b, equal_nan=True)),
                               max_abs_diff=float(np.max(abs(a.astype(float)-b.astype(float)))))
        before = old['rgb_visual_episode'].item()['physics_before']
        current = new['rgb_visual_episode'].item()
        result = dict(reference=args.reference, raw=args.raw, compared_frames=length, fields=checks,
            legacy_raw_byte_hashes_equal=before == current['physics_before'],
            appearance_setters_physics_hash_unchanged=current['physics_before'] == current['physics_after'],
            material_selection=current['materials'], visual_seed=current['seed'], physics_seed=int(new['seed']),
            opening_deg=float(new['pull_max_open_deg']), traversal_m=float(new['pull_max_traversal_m']))
    if args.physics_reference_dump and args.physics_current_dump:
        a = json.loads(Path(args.physics_reference_dump).read_text())
        b = json.loads(Path(args.physics_current_dump).read_text())
        result['actor_physics_properties_exact'] = a == b
        result['actor_counts'] = [len(a), len(b)]
        result['physics_validation'] = dict(reference=args.physics_reference_dump, current=args.physics_current_dump,
            method='Named DOF/body/shape fields, excluding uninitialised structured-array padding bytes')
    else:
        result['actor_physics_properties_exact'] = result['legacy_raw_byte_hashes_equal']
    previews = [json.loads(p.read_text()) for p in Path(args.previews).glob('env00_front_*.json')]
    assert len(previews) >= 3
    result['light_slots'] = sorted(set(p['lighting_slot'] for p in previews))
    result['lighting_fixed'] = len(set(json.dumps([p['hdr'],p['intensity']], sort_keys=True) for p in previews)) == 1
    result['photometric_parameters_fixed'] = len(set(json.dumps(p['parameters'],sort_keys=True) for p in previews)) == 1
    result['all_core_arrays_exact'] = all(c['exact'] for c in checks.values())
    Path(args.out).write_text(json.dumps(result, indent=2))
    assert result['all_core_arrays_exact'] and result['actor_physics_properties_exact'], result
    assert result['appearance_setters_physics_hash_unchanged'], result
    assert result['lighting_fixed'] and result['photometric_parameters_fixed'], result
    print(json.dumps({k:v for k,v in result.items() if k not in ('fields','material_selection')}), flush=True)


if __name__ == '__main__':
    main()
