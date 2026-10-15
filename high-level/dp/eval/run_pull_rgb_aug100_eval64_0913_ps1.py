"""30K augmented-RGB Pi05Evo: export/parity gate and 64 held-out closed-loop trials.

Four isolated one-environment simulators on two GPUs; never share a Gym scene
or its global lights between trials. No expert handoff, training, or retuning.
"""
import concurrent.futures
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO=Path('/home/ps/workspace/txc/doorgym')
A=Path('/mnt/sda/ps_archive/doorgym/high-level')
TRAIN_OUT=A/'dp/result/0912_pull_rgb_aug100_joint9_g080_fixed15'
OUT=A/'dp/result/0913_pull_rgb_aug100_pi05_eval64'
RUNS=A/'logs/door-policy-success/0913_pull_rgb_aug100_pi05_eval64'
RID='local/door_a2w_0912_pull_rgb_aug100_joint9_g080_fixed15'
DATA=A/'data/lerobot'/RID
OFFICIAL=A/'dp/logs/lerobot-train-pi05/pi05evo_0912_pull_rgb_aug100_joint9_g080_fixed15_seed1000_30k/checkpoints/030000/pretrained_model'
WRAP=A/'dp/logs/door-auto-wrapped/0913_pull_rgb_aug100_eval/030000/model_latest'
SRC=REPO/'high-level/hwk/lerobot/src'
PY='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'
SIM='/home/ps/miniconda3/envs/txc_vbc/bin/python'
OLD=A/'logs/0907_pull_rgb_joint9_pi05_evo/pi05evo_native_pull_rgb50_joint9_g080_seed1000_30k_0907.eval.command.json'
GPUS=(1,3,1,3)
BASE_SEED=715455575
VISUAL_SEED=815455575
UNLOCK_DEG=40
TRAVERSAL_M=.8
STOP=threading.Event()
LOCK=threading.Lock()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def status(phase,**kw):
    d=dict(time=time.strftime('%F %T'),phase=phase,**kw)
    save(OUT/'status.json',d);print(json.dumps(d),flush=True)


def environment(gpu):
    env=os.environ.copy();tmp=Path('/mnt/sda/ps/tmp/eval100');tmp.mkdir(parents=True,exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=f'{SRC}:{REPO}/high-level',
        TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),HF_HOME='/home/ps/.cache/huggingface',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        DOOR_DP_VIDEO_BACKEND='torchcodec',DOOR_DP_LEROBOT_PYTHON=PY)
    return env


def run(cmd,env,name,trial_root=None):
    cmd=list(map(str,cmd));logpath=OUT/(name+'.log')
    assert not logpath.exists(),f'Preserve previous run: {logpath}'
    save(OUT/(name+'.command.json'),dict(argv=cmd,env={k:env[k] for k in
        ('CUDA_VISIBLE_DEVICES','PYTHONPATH','TMPDIR','DOOR_DP_LEROBOT_PYTHON')}))
    with logpath.open('x') as log:
        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        last_change=time.monotonic();last_size=-1
        try:
            while child.poll() is None:
                if STOP.is_set():raise RuntimeError('Another lane failed; stopping this task only')
                if shutil.disk_usage(A).free<100*1024**3:raise RuntimeError('HDD free space below 100GiB')
                files=[logpath]+(list(trial_root.glob('*.jsonl')) if trial_root else [])
                size=sum(p.stat().st_size for p in files if p.exists())
                if size!=last_size:last_size=size;last_change=time.monotonic()
                if time.monotonic()-last_change>1800:raise RuntimeError(name+': no progress for 30 minutes')
                save(OUT/(name+'.status.json'),dict(time=time.strftime('%F %T'),pid=child.pid,
                    phase='RUNNING',log=str(logpath),bytes=size))
                time.sleep(15)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGKILL)
                child.wait()
            raise
    if child.returncode:raise RuntimeError(f'{name} exit={child.returncode}; {logpath}')
    save(OUT/(name+'.status.json'),dict(time=time.strftime('%F %T'),phase='COMPLETE',log=str(logpath)))


def evaluation_command(i):
    old=json.loads(OLD.read_text())['argv']
    cmd=[SIM,'high-level/dp/eval/eval_door_policy_success.py',*old[7:]]
    values={'--checkpoint':str(WRAP.with_suffix('.pt')),'--num_envs':'1','--total_trials':'1',
        '--base_seed':str(BASE_SEED+i),'--run_root':str(RUNS/f'trial_{i:03d}')}
    for flag,value in values.items():cmd[cmd.index(flag)+1]=value
    cmd[cmd.index('--'):cmd.index('--')]=['--graphics_device_id','0','--dp_log_interval','1','--no_progress']
    cmd += ['--rgb_visual_mode','doorman','--rgb_visual_config',str(OUT/'rgb_config.yaml'),
        '--rgb_visual_assets',str(A/'data/visual_randomization/doorman'),
        '--rgb_visual_seed',str(VISUAL_SEED+i),'--rgb_visual_preview_dir',str(OUT/'previews'/f'trial_{i:03d}'),
        '--no_draw_scripted_trajectory','--no_draw_ik_target','--no_draw_camera_axes',
        '--no_preview_trajectory_at_spawn','--expected_handle_unlock_angle_deg','40']
    assert '--expert_action_replay_raw_episode' not in cmd
    return cmd


def validate_trial(i):
    root=RUNS/f'trial_{i:03d}'
    summary=json.loads((root/'summary.json').read_text())
    assert summary['trials']==1 and summary['steps']==2600
    assert summary['pass_open_angle_deg']==60 and summary['pull_traversal_distance_m']==TRAVERSAL_M
    batch=summary['batch_logs'][0]
    stdout=Path(batch['stdout_path']).read_text(errors='replace')
    assert '[Error]' not in stdout and 'Could not load visual mesh' not in stdout
    assert 'AF_UNIX path too long' not in stdout and 'CUDA out of memory' not in stdout
    if 'Traceback (most recent call last)' in stdout:
        assert 'SIGSEGV' in stdout and 'Done after 2600 steps' in stdout,stdout[-2000:]
    unlock=[json.loads(line.split('] ',1)[1]) for line in stdout.splitlines() if line.startswith('[PullUnlockAudit] ')]
    assert len(unlock)==1 and len(unlock[0])==1 and abs(unlock[0][0]['handle_unlock_angle_deg']-UNLOCK_DEG)<1e-4
    count=0
    with Path(batch['log_path']).open() as stream:
        for line in stream:
            r=json.loads(line)
            assert r['controlled_env_id']==0 and r['step']==count
            for k in ('state','dp_action_raw'):
                assert len(r[k])==9 and all(math.isfinite(float(x)) for x in r[k]),(i,count,k)
            count+=1
    assert count==2600,(i,count)
    previews=OUT/'previews'/f'trial_{i:03d}'
    scenes=[json.loads(p.read_text()) for p in sorted(previews.glob('env00_front_*.json'))]
    assert len(scenes)>=3,(i,'Missing augmented policy camera samples')
    first=scenes[0]
    for scene in scenes:
        assert scene['episode']['seed']==VISUAL_SEED+i
        assert scene['episode']['physics_before']==scene['episode']['physics_after']
        assert scene['parameters']==first['parameters'] and scene['intensity']==first['intensity']
        assert scene['metadata']['config']['robot_appearance']['enabled']
    row=summary['trials_detail'][0]
    return dict(**row,trial_id=i,physical_seed=BASE_SEED+i,visual_seed=VISUAL_SEED+i,
        lighting_intensity=first['intensity'],image_parameters=first['parameters'],
        audited_full_rollout=True,physical_unlock_deg=UNLOCK_DEG)


def report(rows):
    rows=sorted(rows,key=lambda r:r['trial_id']);n=len(rows)
    d=dict(complete=n==64,trials=n,total_trials=64,checkpoint=str(OFFICIAL),
        opening_threshold_deg=60,traversal_distance_m=TRAVERSAL_M,physical_unlock_deg=UNLOCK_DEG,
        trials_detail=rows)
    for key in ('door_open_success','traversal_success','success'):
        count=sum(bool(r[key]) for r in rows)
        d[key+'_count']=count;d[key+'_rate']=count/n if n else None
    save(OUT/'summary.json',d)
    status('COMPLETE' if n==64 else 'EVALUATING',completed_trials=n,total_trials=64,
        opening=d['door_open_success_count'],traversal=d['traversal_success_count'],successes=d['success_count'])


def lane(lane_id,rows):
    try:
        env=environment(GPUS[lane_id])
        for i in range(lane_id,64,len(GPUS)):
            if STOP.is_set():return
            run(evaluation_command(i),env,f'trial_{i:03d}',RUNS/f'trial_{i:03d}')
            row=validate_trial(i)
            save(OUT/f'trial_{i:03d}.result.json',row)
            with LOCK:rows.append(row);report(rows)
    except BaseException as exc:
        STOP.set();save(OUT/f'lane_{lane_id}.error.json',dict(error=repr(exc),time=time.strftime('%F %T')))
        raise


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists() and not RUNS.exists(),'Existing results preserved; inspect before retry'
    try:
        assert json.loads((TRAIN_OUT/'status.json').read_text())['phase']=='COMPLETE'
        assert json.loads((OFFICIAL/'config.json').read_text())['type']=='pi05_evo'
        train_manifest=json.loads((TRAIN_OUT/'raw_manifest.json').read_text())
        assert {r['seed'] for r in train_manifest}.isdisjoint(range(BASE_SEED,BASE_SEED+64))
        assert {r['visual_seed'] for r in train_manifest}.isdisjoint(range(VISUAL_SEED,VISUAL_SEED+64))
        shutil.copy2(TRAIN_OUT/'rgb_config.yaml',OUT/'rgb_config.yaml')
        shutil.copy2(Path(__file__),OUT/'pipeline.source.py')
        save(OUT/'protocol.json',dict(trials=64,steps=2600,gpus=GPUS,num_envs_per_process=1,
            physical_seed_base=BASE_SEED,visual_seed_base=VISUAL_SEED,action_horizon=25,inference_steps=10,
            policy_fps=25,door='wc_nomenkuang',opening_threshold_deg=60,traversal_distance_m=.8,
            physical_unlock_deg=40,checkpoint=str(OFFICIAL),rgb_augmentation='same frozen training config',
            caveat='Distance-only traversal is not a collision-free passage certification; no exact paired comparison to legacy visual scenes. Policy diffusion RNG uses existing backend behavior.'))
        env=environment(GPUS[0]);launcher=[PY,REPO/'high-level/dp/train/run_native_pi05_evo.py']
        status('EXPORTING')
        run(launcher+[REPO/'high-level/dp/export_official_lerobot_pi05_to_door_checkpoint.py',
            '--official_checkpoint',OFFICIAL,'--root',DATA,'--repo_id',RID,'--out_dir',WRAP,
            '--device','cuda:0','--rgb','--action_horizon','25','--num_inference_steps','10'],env,'export')
        status('VERIFYING_PROCESSOR_PARITY')
        run(launcher+[REPO/'high-level/dp/eval/verify_pi05_saved_processor_parity.py',
            '--official_checkpoint',OFFICIAL,'--door_checkpoint',WRAP,'--dataset_root',DATA,'--repo_id',RID,
            '--device','cuda:0','--output',OUT/'parity.json'],env,'parity')
        assert json.loads((OUT/'parity.json').read_text())['passed']
        # Save each final command before launching; success thresholds never depend on outcomes.
        save(OUT/'planned_trials.json',[evaluation_command(i) for i in range(64)])
        rows=[];report(rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
            futures=[pool.submit(lane,j,rows) for j in range(len(GPUS))]
            for future in concurrent.futures.as_completed(futures):future.result()
        assert len(rows)==64
        report(rows)
    except BaseException as exc:
        STOP.set();status('FAILED',error=repr(exc));raise


if __name__=='__main__':main()
