"""Play the existing fixed-15deg pull recipe with optional visual-only DoorMan.

Reads the original trajectory argument array as text. NEVER executes/sources the
training pipeline shell script. Uses the current Python interpreter for Gym.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def command(args):
    source = ROOT / 'high-level/dp/record/run_wc_nomenkuang_fixed15_gripper080_rgb_pi05_4way_ps1.sh'
    trajectory = shlex.split(source.read_text().split('TRAJECTORY_ARGS=(', 1)[1].split('\n)', 1)[0])
    cmd = [sys.executable, 'high-level/float_ik/isaacgym_float_ik_a2w_basearn_pull_door_parallel.py',
           '--num_envs', str(args.num_envs), '--steps', str(args.steps), '--seed', str(args.seed),
           '--rl_device', 'cuda:0', '--sim_device', 'cuda:0', '--graphics_device_id', '0',
           '--enable_wrist_camera', '--enable_front_camera', '--rgb', '--camera_rgb', '--no_camera_depth',
           '--show_camera_images', '--camera_fps', '25',
           '--enable_depth_camera_randomization', '--depth_camera_pos_rand_m', '.02',
           '--depth_camera_rot_rand_deg', '5', '--no_enable_depth_noise', '--no_enable_depth_gaussian_blur',
           '--rgb_visual_mode', args.mode, '--rgb_visual_assets', str(Path(args.assets).resolve()),
           '--rgb_visual_seed', str(args.visual_seed)] + trajectory
    return cmd


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=['legacy', 'doorman'], default='doorman')
    p.add_argument('--num_envs', type=int, default=1)
    p.add_argument('--steps', type=int, default=2600)
    p.add_argument('--seed', type=int, default=615455575)
    p.add_argument('--visual_seed', type=int, default=-1, help='Only appearance seed; never changes the physical seed')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--assets', default=str(ROOT / 'high-level/data/visual_randomization/doorman'))
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
