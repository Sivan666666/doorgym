"""Audit completed 0907 runs against raw rollout logs and the old evaluation setup."""
import json
import argparse
import hashlib
import math
from pathlib import Path

ARCHIVE = Path('/mnt/sda/ps_archive/doorgym/high-level')
PIPE = ARCHIVE / 'logs/0907_pull_rgb_joint9_pi05_evo'
REFERENCE = ARCHIVE / 'logs/door-policy-success' / (
    'pi05_pull_fixed15_g080_rgb50_joint9_contstate_seed1000_30k_pull_open60_traverse080_'
    'seed615455575_0907_pull_rgb_pi05_contstate_evalfix_v1') / 'summary.json'


def scan(path):
    first, last, dimensions = {}, {}, set()
    with Path(path).open() as file:
        for line in file:
            row = json.loads(line)
            env = int(row['controlled_env_id'])
            first.setdefault(env, row)
            last[env] = int(row['step'])
            dimensions.add((len(row['state']), len(row['dp_action_raw']), row['action_mode']))
            assert all(math.isfinite(x) for x in row['state'] + row['dp_action_raw']), (path, env)
    assert set(first) == set(range(16))
    assert set(last.values()) == {2599}, last
    assert dimensions == {(9, 9, 'a2w_joint_action9')}, dimensions
    return first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    reference = json.loads(REFERENCE.read_text())
    reference_first = [scan(b['log_path']) for b in reference['batch_logs']]
    fields = ['mode', 'robot_body', 'num_envs', 'steps', 'pass_open_angle_deg',
              'pull_traversal_distance_m', 'dp_fps', 'dp_action_horizon', 'dp_inference_steps',
              'base_seed', 'success_criterion', 'success_metric', 'rgb']
    results = []
    for path in sorted(PIPE.glob('*.result.json')):
        data = json.loads(path.read_text())
        assert data['trials'] == 64 and len(data['trials_detail']) == 64
        assert len(data['batch_logs']) == 4
        for key in fields:
            assert data[key] == reference[key], (path, key, data[key], reference[key])
        for index, batch in enumerate(data['batch_logs']):
            first = scan(batch['log_path'])
            for env in range(16):
                for key in ('ikpush_randomization', 'pull_task_geometry'):
                    assert first[env][key] == reference_first[index][env][key], (path, index, env, key)
        for key, total_key in [('door_open_success', 'door_open_successes'),
                               ('traversal_success', 'traversal_successes'), ('success', 'successes')]:
            assert sum(bool(t[key]) for t in data['trials_detail']) == data[total_key]
            assert sum(b[total_key] for b in data['batch_logs']) == data[total_key]
        for trial in data['trials_detail']:
            assert bool(trial['success']) == (bool(trial['door_open_success']) and bool(trial['traversal_success']))
        results.append(dict(name=path.name, trials=64, opening=data['door_open_successes'],
                            traversal=data['traversal_successes'], joint=data['successes'],
                            all_final_steps=2599, state_action_dim=9,
                            initial_randomization_and_geometry_match_reference=True))
    out = dict(passed=True, reference=str(REFERENCE), completed_models=len(results), results=results)
    if args.require_complete:
        assert len(results) == 4
        smoke = json.loads((PIPE / 'evo_native_smoke.json').read_text())
        for filename, expected in smoke['source_hashes'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
        configs = []
        for result in results:
            name = result['name'].removesuffix('.result.json')
            checkpoint = ARCHIVE / 'dp/logs/lerobot-train-pi05' / name / 'checkpoints/030000'
            assert json.loads((checkpoint / 'training_state/training_step.json').read_text())['step'] == 30000
            cfg = json.loads((checkpoint / 'pretrained_model/config.json').read_text())
            train = json.loads((checkpoint / 'pretrained_model/train_config.json').read_text())
            assert train['seed'] == 1000 and train['steps'] == 30000 and train['batch_size'] == 16
            assert cfg['chunk_size'] == 100 and cfg['n_action_steps'] == 50
            assert cfg['input_features']['observation.state']['shape'] == [9]
            assert cfg['output_features']['action']['shape'] == [9]
            assert bool(cfg.get('state_conditioning', False)) == name.startswith('pi05_contstate')
            configs.append(dict(name=name, checkpoint=str(checkpoint), config=cfg,
                                dataset=train['dataset'], seed=train['seed']))
        (PIPE / 'final_training_config_audit.json').write_text(json.dumps(configs, indent=2))
        out.update(all_four_30k_checkpoints_verified=True, native_policy_sources_unchanged=True,
                   all_logged_state_and_action_values_finite=True)
    (PIPE / 'completed_evaluation_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
