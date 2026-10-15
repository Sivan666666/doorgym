"""Read-only experiment audit; writes its derived evidence beside the results."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

A = Path('/mnt/sda/ps_archive/doorgym/high-level')
OUT = A / 'dp/result/pull_evo_trajectory_20260910'
OLD = A / 'logs/0907_pull_rgb_joint9_pi05_evo'
NAMES = ('smooth15', 'steady15', 'trimtail15')


def scan(path):
    first, last, counts, maxima = {}, {}, Counter(), {}
    with Path(path).open() as stream:
        for line in stream:
            r = json.loads(line)
            e = int(r['controlled_env_id'])
            first.setdefault(e, r)
            last[e] = r['step']
            counts[e] += 1
            assert len(r['state']) == len(r['dp_action_raw']) == 9
            assert r['action_mode'] == 'a2w_joint_action9'
            assert all(math.isfinite(x) for x in r['state'] + r['dp_action_raw'])
            g = r['pull_task_geometry']
            distance = sum((x - p) * v for x, p, v in zip(
                r['base']['xy'], g['door_plane_xy'], g['traversal_direction_xy']))
            distance /= math.sqrt(sum(v * v for v in g['traversal_direction_xy']))
            angle = abs(math.degrees(r['door']['dof'][0]))
            assert math.isfinite(angle) and math.isfinite(distance)
            old = maxima.get(e, (0, -math.inf, False))
            maxima[e] = (max(old[0], angle), max(old[1], distance),
                         old[2] or (angle >= 60 and distance >= .8))
    assert set(first) == set(range(16))
    assert set(last.values()) == {2599}
    assert set(counts.values()) == {2600}
    return first, maxima


def mcnemar(a, b):
    gains = sum(not x and y for x, y in zip(a, b))
    losses = sum(x and not y for x, y in zip(a, b))
    n = gains + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n)
    return dict(gains=gains, losses=losses, exact_two_sided_p=p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    original = json.loads((OLD / 'pi05evo_native_pull_rgb50_joint9_g080_seed1000_30k_0907.result.json').read_text())
    original_train = json.loads((A / 'dp/logs/lerobot-train-pi05' /
        'pi05evo_native_pull_rgb50_joint9_g080_seed1000_30k_0907' /
        'checkpoints/030000/pretrained_model/train_config.json').read_text())
    initial = [scan(b['log_path'])[0] for b in original['batch_logs']]
    fields = ('mode', 'robot_body', 'num_envs', 'steps', 'pass_open_angle_deg',
              'pull_traversal_distance_m', 'dp_fps', 'dp_action_horizon', 'dp_inference_steps',
              'base_seed', 'success_criterion', 'success_metric', 'rgb', 'dp_temporal_ensemble')
    results = []
    for name in NAMES:
        path = OUT / f'{name}.result.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        assert data['trials'] == len(data['trials_detail']) == 64
        assert len(data['batch_logs']) == 4
        for key in fields:
            assert data[key] == original[key], (name, key)
        for i, batch in enumerate(data['batch_logs']):
            first, maxima = scan(batch['log_path'])
            for e in range(16):
                for key in ('ikpush_randomization', 'pull_task_geometry'):
                    assert first[e][key] == initial[i][e][key], (name, i, e, key)
                trial = data['trials_detail'][i * 16 + e]
                angle, distance, joint = maxima[e]
                assert math.isclose(angle, trial['max_open_deg'], abs_tol=1e-8)
                assert math.isclose(distance, trial['max_pull_traversal_m'], abs_tol=1e-8)
                assert (angle >= 60) == trial['door_open_success']
                assert (distance >= .8) == trial['traversal_success']
                assert joint == trial['success']
            oldcmd, newcmd = list(original['batch_logs'][i]['command']), list(batch['command'])
            for cmd in (oldcmd, newcmd):
                for flag in ('--checkpoint', '--dp_log_path'):
                    cmd[cmd.index(flag) + 1] = '<artifact>'
            assert oldcmd == newcmd, (name, i, 'evaluation argv differs')
        for single, total in (('success', 'successes'), ('door_open_success', 'door_open_successes'),
                              ('traversal_success', 'traversal_successes')):
            assert sum(t[single] for t in data['trials_detail']) == data[total]
            assert sum(b[total] for b in data['batch_logs']) == data[total]
        parity = json.loads((OUT / f'{name}.parity.json').read_text())
        assert parity['passed'] and parity['max_action_error'] == 0
        ckpt = Path(parity['official_checkpoint'])
        assert json.loads((ckpt.parent / 'training_state/training_step.json').read_text())['step'] == 30000
        cfg = json.loads((ckpt / 'config.json').read_text())
        train = json.loads((ckpt / 'train_config.json').read_text())
        for key in ('seed', 'steps', 'batch_size', 'optimizer', 'scheduler', 'num_workers'):
            assert train[key] == original_train[key], (name, 'training', key)
        assert cfg['type'] == 'pi05_evo' and not cfg.get('state_conditioning', False)
        assert cfg['chunk_size'] == 100 and cfg['n_action_steps'] == 50
        assert cfg['input_features']['observation.state']['shape'] == cfg['output_features']['action']['shape'] == [9]
        assert train['seed'] == 1000 and train['steps'] == 30000 and train['batch_size'] == 16
        info = json.loads((Path(train['dataset']['root']) / 'meta/info.json').read_text())
        assert info['total_episodes'] == 50
        results.append(dict(name=name, opening=data['door_open_successes'],
                            traversal=data['traversal_successes'], joint=data['successes'],
                            failures=data['failure_categories'], frames=info['total_frames'],
                            checkpoint=str(ckpt), dataset=train['dataset'],
                            paired_vs_original50=mcnemar(
                                [t['success'] for t in original['trials_detail']],
                                [t['success'] for t in data['trials_detail']]),
                            config=cfg, train_config=train))
    smoke = json.loads((OLD / 'evo_native_smoke.json').read_text())
    for filename, expected in smoke['source_hashes'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    if args.require_complete:
        assert len(results) == 3
    out = dict(passed=True, completed_models=len(results), all_complete=len(results) == 3,
               native_sources_unchanged=True, raw_rollout_metrics_recomputed=True,
               training_seed_budget_optimizer_scheduler_match_original=True,
               every_trial_has_2600_finite_joint9_records=True,
               evaluation_commands_and_initial_geometry_identical=True, results=results)
    (OUT / 'final_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({**out, 'results': [{k: v for k, v in r.items()
                      if k not in ('config', 'train_config')} for r in results]}, indent=2))


if __name__ == '__main__':
    main()
