"""Same-seed original pull trajectory; two-env legacy/visual recording validation.

Runs only the selected stage. Does not train, change contact/success rules, or
delete outputs. Large outputs go to HDD. Recorded successful episodes only.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

REPO = Path('/home/ps/workspace/txc/doorgym')
A = Path('/mnt/sda/ps_archive/doorgym/high-level')
OUT = A / 'dp/result/rgb_doorman_20260911'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['legacy', 'doorman'], required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--steps', type=int, default=2600)
    p.add_argument('--tag', default='full')
    p.add_argument('--num_envs', type=int, default=2)
    p.add_argument('--overview', action='store_true', help='Also record an external viewer camera from this same rollout')
    p.add_argument('--visual_seed', type=int, default=-1)
    p.add_argument('--visual_config', default='')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.mode + '_' + args.tag
    raw = A / 'data/door_dp_raw' / ('rgb_doorman_smoke_' + tag)
    if raw.exists():
        raise RuntimeError('Existing smoke data preserved; choose a new tag: ' + str(raw))
    text = (REPO / 'high-level/dp/record/run_wc_nomenkuang_fixed15_gripper080_rgb_pi05_4way_ps1.sh').read_text()
    trajectory = shlex.split(text.split('TRAJECTORY_ARGS=(', 1)[1].split('\n)', 1)[0])
    cmd = ['/home/ps/miniconda3/envs/txc_vbc/bin/python',
        'high-level/dp/record/record_door_dp_dataset_a2w_state10.py', '--mode', 'ikpull',
        '--num_episodes', str(args.num_envs), '--num_envs', str(args.num_envs), '--max_quota_rollouts', '1', '--steps', str(args.steps),
        '--seed', '615455575', '--continue_after_cleanup_sigsegv', '--raw_root', str(raw),
        '--state_action_mode', 'joint_state9', '--rl_device', 'cuda:0', '--sim_device', 'cuda:0',
        '--graphics_device_id', '0', '--rgb_only', '--enable_depth_camera_randomization',
        '--depth_camera_pos_rand_m', '.02', '--depth_camera_rot_rand_deg', '5',
        '--no_enable_depth_noise', '--no_enable_depth_gaussian_blur',
        '--keyframe_loss_weight', '3', '--keyframe_loss_radius', '3',
        '--record_camera_pose', '--record_handle_bbox', '--record_gripper_handle_contact',
        '--filter_gripper_handle_contact', '--filter_gripper_handle_contact_min_frames', '5',
        '--filter_gripper_handle_contact_phase_names', 'close_gripper,rotate_handle',
        '--filter_gripper_handle_contact_require_both', '--'] + trajectory
    cmd += ['--rgb_visual_mode', args.mode, '--rgb_visual_assets', str(A / 'data/visual_randomization/doorman'),
            '--rgb_visual_seed', str(args.visual_seed),
            '--rgb_visual_preview_dir', str(OUT / (tag + '_previews'))]
    if args.visual_config:
        config = Path(args.visual_config)
        cmd += ['--rgb_visual_config', str(config if config.is_absolute() else REPO / config)]
    if args.overview:
        cmd += ['--overview_video_path', str(OUT / (tag + '_overview.mp4')),
                '--overview_video_fps', '25', '--overview_video_capture_stride', '2',
                '--no_draw_scripted_trajectory', '--no_preview_trajectory_at_spawn',
                '--no_draw_ik_target', '--no_draw_camera_axes', '--no_show_seg']
    else:
        cmd.insert(cmd.index('--'), '--headless')
    env = os.environ.copy()
    tmp = A / 'data/visual_randomization/tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TMPDIR=str(tmp))
    if args.overview:
        env.update(DISPLAY=':1', XAUTHORITY='/run/user/1000/gdm/Xauthority')
    (OUT / (tag + '.command.json')).write_text(json.dumps(dict(argv=cmd, gpu=args.gpu, raw_root=str(raw)), indent=2))
    with (OUT / (tag + '.log')).open('w') as log:
        result = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    episodes = list(raw.glob('episode_*.npz'))
    # Gym may continue simulating after silently dropping a visual mesh. A
    # successful trajectory/process exit is NOT a successful rendering test.
    log_text=(OUT / (tag + '.log')).read_text(errors='replace')
    render_errors=[line for line in log_text.splitlines()
                   if '[Error]' in line or 'Could not load visual mesh' in line]
    info = dict(returncode=result.returncode, episodes=len(episodes), raw_root=str(raw),
                render_errors=render_errors)
    (OUT / (tag + '.result.json')).write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    if result.returncode != 0:
        raise RuntimeError('Inspect smoke log: ' + str(OUT / (tag + '.log')))
    if render_errors:
        raise RuntimeError('Rendering validation failed: '+ '\n'.join(render_errors))


if __name__ == '__main__':
    main()
