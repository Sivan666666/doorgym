"""Opt-in expert snapshot handoff; never used by normal push/pull rollouts.

Restore generalized state and latch, randomized dynamics and camera extrinsics.
PhysX contact solver caches are not serialized by the original recorder.
No expert future actions/images are used after the handoff.
"""
import json
from pathlib import Path

import numpy as np


def load_manifest(args):
    with Path(args.dp_expert_handoff_manifest).open() as f:
        result = json.load(f)
    if len(result['entries']) != int(args.num_envs):
        raise ValueError('Handoff manifest must contain exactly num_envs entries')
    return result


def prepare_env_args(env_args, args, index):
    entry = load_manifest(args)['entries'][index]
    # These are recorded sampled values, not new draws. Avoid replacing CLI
    # policy settings, the physical env index, or the evaluation RNG seed.
    sampled = entry['randomization']
    for key, value in sampled.items():
        if key not in {'seed', 'env_seed', 'enabled'} and hasattr(env_args, key):
            if isinstance(value, (bool, int, float, str)):
                setattr(env_args, key, value)
    for key, value in entry.get('scene', {}).items():
        if hasattr(env_args, key):
            setattr(env_args, key, value)
    env_args.ikpush_randomization_json = json.dumps(sampled, sort_keys=True)
    env_args._expert_handoff_entry = entry


def restore_batch(gym, sim, states, args, controller, dof_names, api):
    if controller is None or args.dp_warmstart:
        raise ValueError('Expert handoff requires a policy and cannot combine with dp_warmstart')
    manifest = load_manifest(args)
    audits = []
    for st in states:
        entry = st.args._expert_handoff_entry
        with np.load(entry['snapshot'], allow_pickle=False) as d:
            # Actor creation applies asset-specific friction multipliers. The
            # saved values are already post-multiplier: restore them afterwards
            # to avoid applying the multiplier twice.
            api.apply_recovery_randomization(gym, st, {'ikpush_randomization': entry['randomization']})
            st.args.ikpush_randomization_json = json.dumps(entry['randomization'], sort_keys=True)
            props = gym.get_actor_dof_properties(st.env, st.door_actor)
            physics = {}
            for name, dof_index in (('door',st.door.door_dof_index), ('handle',st.door.handle_dof_index)):
                if dof_index is None:
                    continue
                for field in ('friction','damping'):
                    key = name+'_joint_'+field
                    actual = float(props[field][dof_index])
                    expected = float(entry['randomization'][key])
                    if not np.isclose(actual, expected, rtol=1e-6, atol=1e-7):
                        raise RuntimeError(f'Handoff dynamics mismatch: {key} {actual} != {expected}')
                    physics[key] = actual
            api.apply_warmstart_state(gym, sim, st, d, 0, dof_names)
            # Joint9 state starts with previous commands, not measured velocity.
            # action in the snapshot is the previous command, never the future label.
            st.last_dp_action = np.array(d['action'][0], dtype=np.float32)
            st.last_dp_action[:2] = d['state'][0, :2]
            st.args.robot_z = float(d['replay_root_state'][0, 2])
            st.args.robot_pitch = float(api.pitch_from_quat_xyzw(d['replay_root_state'][0, 3:7]))
            # Restore actuator targets held just before this observation.
            for i, name in enumerate(dof_names):
                if name in api.Z1_ARM_JOINT_NAMES:
                    st.dof_positions[i] = float(d['action'][0, 2 + api.Z1_ARM_JOINT_NAMES.index(name)])
                elif name == 'jointGripper':
                    st.dof_positions[i] = float(d['action'][0, -1])
            gym.set_actor_dof_position_targets(st.env, st.arm_actor, st.dof_positions)
            # Attach each camera at the recorded optical pose relative to the
            # restored base. Infer its local attachment from restored link FK.
            base_actor = st.actor_handles[0] if len(st.actor_handles) > 1 else st.arm_actor
            base_pos, base_quat = api.get_body_pose(gym, st.env, base_actor, 0)
            for camera in ('front', 'wrist'):
                pose = d[camera + '_camera_pose_base'][0]
                world_pos, world_quat = api.dc.compose_pose_np(base_pos, base_quat, pose[:3], pose[3:])
                actor = base_actor if camera == 'front' else st.arm_actor
                body_index = 0 if camera == 'front' else api.dc.get_actor_body_index(gym, st.env, actor, 'link06')
                body_pos, body_quat = api.get_body_pose(gym, st.env, actor, body_index)
                local_pos, local_quat = api.dc.relative_pose_np(body_pos, body_quat, world_pos, world_quat)
                handle = gym.get_actor_rigid_body_handle(st.env, actor, body_index)
                gym.attach_camera_to_body(st.camera_handles[camera], st.env, handle,
                    api.local_camera_transform_from_pose(local_pos, local_quat), api.gymapi.FOLLOW_TRANSFORM)
                api.camera_axis_pose_cache(st.args)[camera] = dict(local_pos=local_pos, local_quat=local_quat)
            q, _ = api.get_actor_dof_state(gym, st.env, st.arm_actor)
            state = api.dc.make_a2w_last_command_joint_state9(st.last_dp_action, dof_names, q)
            door, _ = api.get_actor_dof_state(gym, st.env, st.door_actor)
            front, wrist = api.dc.float_camera_pose_base(gym, st)
            ee, _ = api.current_ee_pose(gym, sim, st.ik_state)
            row = dict(env_id=st.index, episode=entry['episode'], source_frame=entry['source_frame'],
                dynamics=physics,
                phase=entry['phase'], state_error=float(np.max(np.abs(state-d['state'][0]))),
                door_error=float(np.max(np.abs(door-d['replay_door_dof_pos'][0]))),
                ee_error_m=float(np.linalg.norm(np.asarray(ee)-d['replay_ee_pos'][0])),
                front_pose_error=float(np.max(np.abs(front-d['front_camera_pose_base'][0]))),
                wrist_pose_error=float(np.max(np.abs(wrist-d['wrist_camera_pose_base'][0]))),
                camera_local_poses={k:{a:np.asarray(v).tolist() for a,v in p.items()}
                                    for k,p in api.camera_axis_pose_cache(st.args).items()},
                latch_open=bool(st.door.open_stage))
            for key in ('state_error', 'door_error', 'ee_error_m', 'front_pose_error', 'wrist_pose_error'):
                if row[key] > 1e-3:
                    raise RuntimeError('Handoff restoration failed: ' + json.dumps(row))
            audits.append(row)
    controller.reset_envs([st.index for st in states])
    Path(manifest['restore_audit']).write_text(json.dumps(audits, indent=2))
    print('EXPERT_HANDOFF_RESTORED ' + json.dumps(audits), flush=True)
