"""Durable two-GPU queue: original dual-state RGB500; native Evo RGB50/200/500.

Start after native smoke audit. GPU0 waits for the HDD recorder/converter.
GPU1 runs Evo datasets in sequence. GPU2/3 are never touched.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path('/home/ps/workspace/txc/doorgym')
ARCHIVE = Path('/mnt/sda/ps_archive/doorgym/high-level')
TAG = '0907_pull_rgb_joint9_pi05_evo'
PIPE = ARCHIVE / 'logs' / TAG
PYTHON = '/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'
CONDA = '/home/ps/miniconda3/bin/conda'
TRAIN_ROOT = ARCHIVE / 'dp/logs/lerobot-train-pi05'


def status(name, phase):
    entry = dict(time=time.strftime('%F %T'), name=name, phase=phase)
    (PIPE / f'{name}.status.json').write_text(json.dumps(entry, indent=2))
    print(json.dumps(entry), flush=True)


def command_run(args, env, name):
    if shutil.disk_usage(ARCHIVE).free < 150 * 1024**3:
        raise RuntimeError('HDD free space under 150GiB; do not delete existing data automatically')
    (PIPE / f'{name}.command.json').write_text(json.dumps(dict(argv=list(map(str, args)),
        env={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'TMPDIR')}), indent=2))
    with (PIPE / f'{name}.log').open('a') as log:
        subprocess.run(list(map(str, args)), cwd=REPO, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def dataset(n):
    repo_id = ('local/door_a2w_0907_wc_nomenkuang_fixed15_gripper080_rgb_500_joint9' if n == 500 else
               f'local/door_a2w_0829_wc_nomenkuang_fixed15_gripper080_rgb_v1_{n}_joint9')
    if n == 500:
        while not (PIPE / 'dataset500_ready.json').exists():
            if (PIPE / 'record_status.txt').exists() and 'FAILED' in (PIPE / 'record_status.txt').read_text():
                raise RuntimeError('500 recording/conversion failed; inspect record logs')
            time.sleep(60)
    root = ARCHIVE / 'data/lerobot' / repo_id
    info = json.loads((root / 'meta/info.json').read_text())
    assert info['total_episodes'] == n and info['total_frames'] == n * 1300
    for key in ('observation.state', 'action'):
        assert info['features'][key]['shape'] == [9]
    return root, repo_id


def running_training(name):
    """Find only our exact accelerate training command, not unrelated GPU users."""
    matches = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
            if 'accelerate.commands.launch' in argv and f'--job_name={name}' in argv:
                matches.append(int(proc.name))
        except (OSError, UnicodeError):
            continue
    return matches


def job(n, evo, gpu):
    name = f'{"pi05evo_native" if evo else "pi05_contstate"}_pull_rgb{n}_joint9_g080_seed1000_30k_0907'
    try:
        if (PIPE / f'{name}.result.json').exists():
            return
        status(name, 'WAIT_DATASET')
        root, repo_id = dataset(n)
        src = REPO / ('high-level/hwk/lerobot/src' if evo else 'high-level/lerobot/src')
        env = os.environ.copy()
        # Python multiprocessing uses AF_UNIX sockets below TMPDIR (108-byte limit).
        temp = Path('/mnt/sda/ps/dp_tmp0907') / f'{"evo" if evo else "pi05"}{n}'
        temp.mkdir(parents=True, exist_ok=True)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=f'{src}:{REPO / "high-level"}',
            TMPDIR=str(temp), WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            DOOR_DP_VIDEO_BACKEND='torchcodec', DOOR_DP_LEROBOT_PYTHON=PYTHON,
            PYTHONUNBUFFERED='1')
        native_launcher = [PYTHON, '-u', str(REPO / 'high-level/dp/train/run_native_pi05_evo.py')]
        launcher = native_launcher if evo else [PYTHON, '-u']
        base = (ARCHIVE / 'dp/models/pi05_evo_native_base_0907' if evo else
                REPO / 'high-level/dp/models/pi05_base_cached_localtokenizer')
        out = TRAIN_ROOT / name
        official = out / 'checkpoints/030000/pretrained_model'
        active = running_training(name)
        if active:
            assert os.environ.get('ADOPT_RUNNING') == '1', f'Training already active: {active}'
            assert len(active) == 1, active
            status(name, f'ADOPT_RUNNING_GPU{gpu}_PID{active[0]}')
            while running_training(name):
                time.sleep(60)
            assert (official / 'model.safetensors').exists(), 'Adopted training stopped before 30K'
        if not (official / 'model.safetensors').exists():
            assert not out.exists(), f'Partial training exists; inspect/resume explicitly: {out}'
            status(name, f'TRAIN_GPU{gpu}')
            train_script = src / 'lerobot/scripts/lerobot_train.py'
            args = [PYTHON, '-m', 'accelerate.commands.launch', '--num_processes=1', '--mixed_precision=bf16']
            if evo:
                args.append(REPO / 'high-level/dp/train/run_native_pi05_evo.py')
            args += [train_script, f'--dataset.root={root}', f'--dataset.repo_id={repo_id}',
                     '--dataset.video_backend=torchcodec', '--dataset.use_imagenet_stats=false',
                     f'--policy.type={"pi05_evo" if evo else "pi05"}', '--policy.device=cuda',
                     f'--policy.pretrained_path={base}', '--policy.dtype=bfloat16', '--policy.push_to_hub=false',
                     '--policy.gradient_checkpointing=true', '--policy.chunk_size=100',
                     '--policy.n_action_steps=50', '--policy.num_inference_steps=10',
                     '--batch_size=16', '--steps=30000', '--num_workers=4', '--save_freq=10000',
                     '--log_freq=50', '--seed=1000', f'--job_name={name}', f'--output_dir={out}',
                     '--wandb.enable=false', '--wandb.disable_artifact=true']
            if not evo:
                args.append('--policy.state_conditioning=true')
            command_run(args, env, f'{name}.train')
        cfg = json.loads((official / 'config.json').read_text())
        assert cfg['input_features']['observation.state']['shape'] == [9]
        assert cfg['output_features']['action']['shape'] == [9]
        assert bool(cfg.get('state_conditioning', False)) == (not evo)
        wrap = ARCHIVE / 'dp/logs/door-auto-wrapped' / TAG / name / '030000/model_latest'
        if not wrap.with_suffix('.pt').exists():
            assert not wrap.exists(), f'Inspect partial export: {wrap}'
            status(name, 'EXPORT')
            command_run(launcher + [REPO / 'high-level/dp/export_official_lerobot_pi05_to_door_checkpoint.py',
                '--official_checkpoint', official, '--root', root, '--repo_id', repo_id, '--out_dir', wrap,
                '--device', 'cuda:0', '--rgb', '--action_horizon', '25', '--num_inference_steps', '10'],
                env, f'{name}.export')
        status(name, 'PARITY')
        command_run(launcher + [REPO / 'high-level/dp/eval/verify_pi05_saved_processor_parity.py',
            '--official_checkpoint', official, '--door_checkpoint', wrap, '--dataset_root', root,
            '--repo_id', repo_id, '--device', 'cuda:0', '--output', PIPE / f'{name}.parity.json'],
            env, f'{name}.parity')
        ev = ARCHIVE / 'logs/door-policy-success' / f'{name}_open60_traversal080_seed615455575'
        assert not ev.exists(), f'Inspect partial evaluation: {ev}'
        status(name, 'EVAL64')
        args = [CONDA, 'run', '--no-capture-output', '-n', 'txc_vbc', 'python',
            'high-level/dp/eval/eval_door_policy_success.py', '--checkpoint', wrap.with_suffix('.pt'),
            '--mode', 'ikpull', '--robot_body', 'a2wz1', '--door_cfg', 'high-level/data/cfg/b1z1_opendoor.yaml',
            '--num_envs', '16', '--total_trials', '64', '--steps', '2600', '--base_seed', '615455575',
            '--pass_open_angle_deg', '60', '--pull_traversal_distance_m', '0.8',
            '--dp_action_horizon', '25', '--dp_inference_steps', '10', '--dp_fps', '25', '--headless',
            '--parallel_batches', '1', '--run_root', ev, '--rgb', '--', '--door_name', 'wc_nomenkuang',
            '--wc4_disable_door_open_resistance', '--door_auto_open_force', '0', '--gripper_close_ratio', '0.8',
            '--robot_pitch', '0', '--ikpush_robot_pitch_rand_min', '0', '--ikpush_robot_pitch_rand_max', '0',
            '--enable_depth_camera_randomization', '--depth_camera_pos_rand_m', '0.02',
            '--depth_camera_rot_rand_deg', '5', '--no_enable_depth_noise', '--no_enable_depth_gaussian_blur',
            '--no_show_seg', '--dp_rgb_without_masks']
        command_run(args, env, f'{name}.eval')
        result = json.loads((ev / 'summary.json').read_text())
        assert result['trials'] == 64 and len(result['batch_logs']) == 4
        (PIPE / f'{name}.result.json').write_text(json.dumps(result, indent=2))
        status(name, 'COMPLETE')
    except BaseException as exc:
        status(name, f'FAILED {type(exc).__name__}: {exc}')
        raise


def evo_lane():
    for n in ((50, 500) if os.environ.get('GPU_EVO200') else (50, 200, 500)):
        job(n, evo=True, gpu=1)


def monitor():
    entries = [json.loads(p.read_text()) for p in PIPE.glob('*.status.json')]
    entry = dict(time=time.strftime('%F %T'), statuses=entries,
                 hdd_free_gib=round(shutil.disk_usage(ARCHIVE).free / 1024**3, 1))
    for key, args in [('gpu', ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader']),
                      ('processes', ['ps', '-eo', 'pid,etime,args'])]:
        output = subprocess.run(args, capture_output=True, text=True).stdout
        if key == 'processes':
            output = '\n'.join(x for x in output.splitlines() if any(w in x for w in
                ('lerobot_train', 'extend_pull_rgb', 'record_door_dp', 'eval_door_policy_success')))
        entry[key] = output
    entry['record'] = (PIPE / 'record_status.txt').read_text() if (PIPE / 'record_status.txt').exists() else ''
    with (PIPE / 'monitor_30min.jsonl').open('a') as file:
        file.write(json.dumps(entry) + '\n')
    print('MONITOR ' + json.dumps(entry), flush=True)


def main():
    PIPE.mkdir(parents=True, exist_ok=True)
    assert json.loads((PIPE / 'evo_native_smoke.json').read_text())['passed']
    # Exclusive lock prevents accidental duplicate queues on reconnect.
    import fcntl
    lock = (PIPE / 'training_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(job, 500, False, 0), pool.submit(evo_lane)]
        if os.environ.get('GPU_EVO200'):
            futures.append(pool.submit(job, 200, True, int(os.environ['GPU_EVO200'])))
        while True:
            monitor()
            done, pending = concurrent.futures.wait(futures, timeout=1800,
                                                    return_when=concurrent.futures.ALL_COMPLETED)
            if not pending:
                for f in done:
                    f.result()
                break
    status('pipeline', 'ALL_FOUR_TRAINED_AND_EVALUATED')


if __name__ == '__main__':
    main()
