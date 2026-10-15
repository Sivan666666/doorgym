"""Append 300 successful RGB episodes to the existing 200; never overwrite old data."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REPO = Path('/home/ps/workspace/txc/doorgym')
ARCHIVE = Path('/mnt/sda/ps_archive/doorgym/high-level')
TAG = '0907_pull_rgb_joint9_pi05_evo'
PIPE = ARCHIVE / 'logs' / TAG
DATASET = 'local/door_a2w_0907_wc_nomenkuang_fixed15_gripper080_rgb_500_joint9'
PYTHON = '/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'


def roots():
    return [REPO / 'high-level/data/door_dp_raw' / (
        f'a2w_0829_wc_nomenkuang_fixed15_gripper080_rgb_v1_shard{i}_50_seed{615455575+i*100000}'
        if i < 4 else f'a2w_0907_wc_nomenkuang_fixed15_gripper080_rgb_shard{i}_50_seed{615455575+i*100000}')
        for i in range(10)]


def status(value):
    text = f'{time.strftime("%F %T")} {value}\n'
    print(text, end='', flush=True)
    (PIPE / 'record_status.txt').write_text(text)


def run(command, name):
    (PIPE / f'{name}.command.json').write_text(json.dumps(command, indent=2))
    with (PIPE / f'{name}.log').open('a') as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def validate():
    import numpy as np
    manifest = []
    for root in roots():
        paths = sorted(root.glob('episode_*.npz'))
        assert [p.name for p in paths] == [f'episode_{i:06d}.npz' for i in range(50)], root
        for p in paths:
            with np.load(p, allow_pickle=True) as d:
                assert d['state'].shape == d['action'].shape == (1300, 9), p
                for k in ('front_rgb', 'wrist_rgb'):
                    assert d[k].shape == (1300, 480, 640, 3), (p, k)
                for k in ('front_camera_pose_base', 'wrist_camera_pose_base',
                          'replay_door_dof_pos', 'gripper_handle_contact_both'):
                    assert k in d, (p, k)
                release = float(d['pull_release_angle_observed_deg'].reshape(-1)[0])
                opened = float(d['pull_max_open_deg'].reshape(-1)[0])
                traversed = float(d['pull_max_traversal_m'].reshape(-1)[0])
                assert 14.5 <= release <= 16 and opened >= 80 and traversed >= .8, p
                manifest.append(dict(index=len(manifest), path=str(p.resolve()), frames=1300,
                                     release_deg=release, open_deg=opened, traversal_m=traversed))
        status(f'VALIDATED {len(manifest)}/500')
    (PIPE / 'raw500_manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    PIPE.mkdir(parents=True, exist_ok=True)
    temp = PIPE / 'tmp'; temp.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TMPDIR=str(temp), WANDB_MODE='disabled',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.validate_only:
        validate(); return
    source = REPO / 'high-level/dp/record/run_wc_nomenkuang_fixed15_gripper080_rgb_pi05_4way_ps1.sh'
    trajectory = shlex.split(source.read_text().split('TRAJECTORY_ARGS=(', 1)[1].split('\n)', 1)[0])
    assert trajectory[trajectory.index('--gripper_close_ratio') + 1] == '0.8'
    for i, root in enumerate(roots()):
        if i < 4:
            assert len(list(root.glob('episode_*.npz'))) == 50, root
            continue
        marker = PIPE / f'record_shard{i}.done'
        if marker.exists():
            assert len(list(root.glob('episode_*.npz'))) == 50
            continue
        if len(list(root.glob('episode_*.npz'))) == 50:
            marker.touch(); continue
        # Recorder has its own resumable quota accounting; never delete partial shards.
        status(f'RECORD shard={i} existing={len(list(root.glob("episode_*.npz")))}')
        command = ['/home/ps/miniconda3/bin/conda', 'run', '--no-capture-output', '-n', 'txc_vbc',
                   'python', 'high-level/dp/record/record_door_dp_dataset_a2w_state10.py',
                   '--mode', 'ikpull', '--num_episodes', '50', '--num_envs', '4',
                   '--max_quota_rollouts', '1000', '--steps', '2600', '--seed', str(615455575+i*100000),
                   '--continue_after_cleanup_sigsegv', '--headless', '--raw_root', str(root),
                   '--state_action_mode', 'joint_state9', '--rl_device', 'cuda:0', '--sim_device', 'cuda:0',
                   '--graphics_device_id', '0', '--rgb_only', '--camera_depth_clip_lower', '0.2',
                   '--camera_depth_clip_far', '1.5', '--enable_depth_camera_randomization',
                   '--depth_camera_pos_rand_m', '0.02', '--depth_camera_rot_rand_deg', '5',
                   '--keyframe_loss_weight', '3', '--keyframe_loss_radius', '3', '--record_camera_pose',
                   '--record_handle_bbox', '--record_gripper_handle_contact',
                   '--filter_gripper_handle_contact', '--filter_gripper_handle_contact_min_frames', '5',
                   '--filter_gripper_handle_contact_phase_names', 'close_gripper,rotate_handle',
                   '--filter_gripper_handle_contact_require_both', '--'] + trajectory
        run(command, f'record_shard{i}')
        assert len(list(root.glob('episode_*.npz'))) == 50
        marker.touch()
    validate()
    destination = ARCHIVE / 'data/lerobot' / DATASET
    if not (PIPE / 'dataset500_ready.json').exists():
        assert not destination.exists(), f'Inspect partial conversion before resuming: {destination}'
        status('CONVERT 500')
        command = [PYTHON, 'high-level/dp/convert_door_raw_to_lerobot.py', '--raw_root', str(roots()[0])]
        for root in roots()[1:]:
            command.extend(['--additional_raw_root', str(root)])
        command.extend(['--root', str(ARCHIVE / 'data/lerobot'), '--repo_id', DATASET,
                        '--state_action_mode', 'joint_state9', '--rgb_only', '--image_storage', 'video',
                        '--video_codec', 'h264', '--num_workers', '4'])
        run(command, 'convert500')
        info = json.loads((destination / 'meta/info.json').read_text())
        assert info['total_episodes'] == 500 and info['total_frames'] == 650000
        for k in ('observation.state', 'action'):
            assert info['features'][k]['shape'] == [9]
        (PIPE / 'dataset500_ready.json').write_text(json.dumps(dict(root=str(destination), repo_id=DATASET,
                                                                  episodes=500, frames=650000), indent=2))
    status('DATASET500_READY')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if PIPE.exists():
            status(f'FAILED {type(exc).__name__}: {exc}')
        raise
