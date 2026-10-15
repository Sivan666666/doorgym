"""Local/remote batch driver with per-episode locks, retries and resumable RGB."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import random
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from common import read, save, sha, digest, validate_rollout

HERE = Path(__file__).resolve().parent


def run(argv, log, env, timeout):
    with Path(log).open('w') as stream:
        process = subprocess.Popen([str(v) for v in argv], stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if status:
        raise RuntimeError(f'Exit {status}: see {log}')


def fingerprint(cfg):
    code = {f.name: sha(f) for f in HERE.glob('*.py')}
    for key in ('recorder','scene_builder'):
        path = Path(cfg['tools'][key])
        code[key] = sha(path)
    code['room_helper'] = sha(Path(cfg['tools']['scene_builder']).with_name('furnish_wc_blender_room.py'))
    if cfg['tools'].get('rgb_augmenter'):
        code['rgb_augmenter'] = sha(cfg['tools']['rgb_augmenter'])
    if cfg['tools'].get('photo_room_builder'):
        code['photo_room_builder'] = sha(cfg['tools']['photo_room_builder'])
    return digest(dict(config=cfg, code=code)), code


def reuse_recording(source_root, eid, destination, physics):
    source = Path(source_root) / 'episodes' / f'{eid:06d}' / 'status.json'
    previous = read(source)
    assert previous['stages']['record']['status'] == 'complete'
    raw = Path(previous['raw'])
    assert sha(raw/'trajectory.json') == previous['raw_sha256']
    assert previous['physics_seed'] == physics['seed'] + eid
    destination.mkdir()
    # Copy only raw artifacts; never write scenes into the old episode folder.
    for item in raw.iterdir():
        if item.is_file():
            shutil.copy2(item, destination/item.name)
    assert sha(destination/'trajectory.json') == previous['raw_sha256']
    provenance = dict(reused=True, source_status=str(source), source_status_sha256=sha(source),
                      source_trajectory=str(raw/'trajectory.json'), source_sha256=previous['raw_sha256'])
    save(destination/'reuse_provenance.json', provenance)
    return provenance


def episode(cfg, config_path, root, eid, gpu, signature, retry_failed, prepare_only=False):
    folder = root / 'episodes' / f'{eid:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = folder / 'status.json'
        state = read(status_path) if status_path.exists() else dict(id=eid, signature=signature, stages={},
            physics_seed=cfg['physics']['seed']+eid, scene_seed=cfg['scene']['seed']+eid)
        assert state['signature'] == signature, 'Configuration drift: use a new run root'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        for stage in (('record','scene') if prepare_only else ('record','scene','render')):
            info = state['stages'].get(stage, {})
            if info.get('status') == 'complete':
                # Completion markers must still refer to intact stage artifacts.
                if stage == 'record':
                    assert sha(Path(state['raw'])/'trajectory.json') == state['raw_sha256']
                elif stage == 'scene':
                    assert sha(state['scene']) == read(Path(state['scene']).parent/'scene_verified.json')['scene_sha256']
                else:
                    assert (Path(state['render'])/'render_complete.json').is_file()
                continue
            if info.get('status') == 'failed' and not retry_failed:
                raise RuntimeError(f'Episode {eid} failed before; inspect logs then use --retry-failed')
            success = False
            for _ in range(cfg['batch'].get('stage_attempts', {}).get(stage, cfg['batch']['max_attempts'])):
                attempt = info.get('attempt', 0) + 1
                info = dict(status='running', attempt=attempt, started=time.time(), gpu=gpu)
                state['stages'][stage] = info
                save(status_path, state)
                log = folder / f'{stage}_{attempt:03d}.log'
                try:
                    if stage == 'record':
                        # Retry failed physical candidates reproducibly, without
                        # replacing any failed trajectory or its diagnostic log.
                        state['physics_seed'] = cfg['physics']['seed'] + eid + (attempt-1)*cfg['physics'].get('retry_seed_stride', 0)
                        save(status_path, state)
                        name = f'raw_{attempt:03d}'
                        command = [cfg['tools']['physics_python'], cfg['tools']['recorder'], '--name', name,
                                   '--output-root', folder, '--steps', cfg['physics']['steps'],
                                   '--seed', state['physics_seed'], '--record-actions', '--base-y',
                                   random.Random(state['physics_seed']).uniform(*cfg['physics']['base_y_range'])]
                        reused = None
                        reuse_root = cfg['physics'].get('reuse_episodes', {}).get(str(eid), cfg['physics'].get('reuse_run_root'))
                        if reuse_root:
                            reused = reuse_recording(reuse_root, eid, folder/name, cfg['physics'])
                            save(log, reused)
                        else:
                            run(command, log, env, cfg['batch']['physics_timeout_s'])
                        state['raw'] = str(folder/name)
                        state['qc'] = validate_rollout(folder/name/'trajectory.json', cfg['physics']['steps'],
                            cfg['physics']['min_open_deg'], cfg['physics']['min_traverse_m'])
                        state['raw_sha256'] = state['qc']['source_sha256']
                        if reused:
                            state['qc']['recording_reuse'] = reused
                    elif stage == 'scene':
                        name = f'scene_{attempt:03d}'
                        run([cfg['tools']['render_python'], HERE/'build_scene.py', '--config', config_path,
                             '--raw', state['raw'], '--name', name, '--seed', state['scene_seed']], log, env,
                            cfg['batch']['scene_timeout_s'])
                        state['scene'] = str(Path(state['raw'])/name/'scene.blend')
                    else:
                        state['render'] = str(folder/'render')
                        run([cfg['tools']['render_python'], HERE/'render_meister.py', '--config', config_path,
                             '--scene', state['scene'], '--out', state['render']], log, env,
                            cfg['batch']['render_timeout_s'])
                    info.update(status='complete', finished=time.time(), log=str(log))
                    save(status_path, state)
                    success = True
                    break
                except Exception as error:
                    info.update(status='failed', finished=time.time(), error=str(error), log=str(log))
                    save(status_path, state)
                    print('FAILED', eid, stage, str(error), flush=True)
            if not success:
                raise RuntimeError(f'Episode {eid} stage {stage} failed')
            print('COMPLETE', eid, stage, flush=True)
        return state


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--stage', choices=['all','prepare','episodes','export','status'], default='all')
    p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    cfg = read(a.config)
    root = Path(cfg['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    if a.stage == 'status':
        for f in sorted(root.glob('episodes/*/status.json')):
            s = read(f)
            print(s['id'], {k:v['status'] for k,v in s['stages'].items()})
        return
    signature, code = fingerprint(cfg)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = root/'run.json'
        if identity.exists():
            assert read(identity)['signature'] == signature, 'Configuration/code changed; use a new run root'
        else:
            save(identity, dict(signature=signature, config=cfg, source_hashes=code))
        save(root/'config.json', cfg)
        config_path = root/'config.json'
        if a.stage in ('all','prepare','episodes'):
            gpus = cfg['batch']['gpus']
            assert gpus and len(gpus) == len(set(gpus))
            workers_per_gpu = cfg['batch'].get('workers_per_gpu', 1)
            assert isinstance(workers_per_gpu, int) and 1 <= workers_per_gpu <= 2
            slots = gpus * workers_per_gpu
            def worker(index):
                return [episode(cfg,config_path,root,eid,slots[index],signature,a.retry_failed,a.stage=='prepare')
                        for eid in range(index,cfg['batch']['episodes'],len(slots))]
            with ThreadPoolExecutor(max_workers=len(slots)) as executor:
                list(executor.map(worker, range(len(slots))))
        if a.stage in ('all','export'):
            states = [read(root/'episodes'/f'{eid:06d}'/'status.json') for eid in range(cfg['batch']['episodes'])]
            assert all(all(s['stages'][k]['status']=='complete' for k in ('record','scene','render')) for s in states)
            if cfg['dataset'].get('require_unique_trajectories'):
                assert len({s['raw_sha256'] for s in states}) == len(states), 'Duplicate physical trajectories: do not count visual variants as independent recordings'
            manifest = root/'manifest.json'
            save(manifest, dict(signature=signature, episodes=[{k:s[k] for k in
                ('id','raw','render','scene','physics_seed','scene_seed','raw_sha256','qc')} for s in states]))
            run([cfg['tools']['dataset_python'], HERE/'export_lerobot.py', '--config', config_path,
                 '--manifest', manifest, '--out', root/'dataset'], root/'export.log',
                dict(os.environ, OMP_NUM_THREADS='4'), cfg['batch']['export_timeout_s'])
            print('PIPELINE_COMPLETE', root/'dataset', flush=True)


if __name__ == '__main__':
    main()
