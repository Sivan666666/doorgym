"""PS1: current-contact Gym -> augmented Blender RGB100 -> native pi05 Evo.

Own-process watchdog, 60-second status and 1800-second audit snapshots.
Never terminate unrelated GPU jobs, overwrite old models, or upload data.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

JOB=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_dim60_aug100_pi05_20261008')
REPO=Path('/home/ps/workspace/txc/doorgym')
SRC=REPO/'high-level/hwk/lerobot/src'
PY='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'
SIM=JOB/'sim_data'
CFG=SIM/'configs/rgb_photo_office_dim60_aug100.json'
ROOT=JOB/'run'
BASE=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/models/pi05_evo_native_base_0907')
TRAIN=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/logs/lerobot-train-pi05/pi05evo_wc_dim60_aug100_joint10_seed1000_30k')


def read(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def environment(gpu):
    env=os.environ.copy();cache=JOB/'cache';tmp=Path('/mnt/sda/ps/tmp/wc100')
    tmp.mkdir(parents=True,exist_ok=True)
    for name in ('datasets','xdg','torch','triton'):(cache/name).mkdir(parents=True,exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=f'{SRC}:{REPO}/high-level',
        TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),HF_DATASETS_CACHE=str(cache/'datasets'),
        XDG_CACHE_HOME=str(cache/'xdg'),TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),
        HF_HOME=env.get('HF_HOME','/home/ps/.cache/huggingface'),
        WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return env


def progress():
    counts={k:0 for k in ('record','scene','render')};frames=0;failures=[];attempts=0
    for p in ROOT.glob('episodes/*/status.json'):
        s=read(p)
        for name,st in s['stages'].items():
            if st['status']=='complete':counts[name]+=1
            if st['status']=='failed':failures.append(dict(episode=s['id'],stage=name,error=st.get('error'),log=st.get('log')))
        attempts+=s['stages'].get('record',{}).get('attempt',0)
        frames+=sum(1 for _ in (p.parent/'render/frames').glob('*.json'))
    return dict(episodes=counts,rendered_frame_groups=frames,target_frame_groups=130000,
        recording_attempts=attempts,failures=failures)


def status(phase,**kw):
    value=dict(time=time.strftime('%F %T'),unix_time=time.time(),phase=phase,
        supervisor_pid=os.getpid(),
        **progress(),**kw,hdd_free_gib=shutil.disk_usage(JOB).free/1024**3)
    save(JOB/'status.json',value)
    return value


def snapshot(value):
    value=dict(value,gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader'],text=True))
    with (JOB/'monitor_30min.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    print('MONITOR_30MIN',json.dumps(value),flush=True)


def stop(child):
    with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGKILL)
        child.wait()


def run(command,env,name,training=False,log_freq=50,timeout=172800):
    stamp=time.strftime('%Y%m%d_%H%M%S');log=JOB/f'{name}_{stamp}.log'
    save(log.with_suffix('.command.json'),dict(argv=list(map(str,command)),gpu=env.get('CUDA_VISIBLE_DEVICES')))
    started=last_progress=time.monotonic();last_snapshot=0.;last_key=None;last_loss=None;max_loss=0.;steps=0;logs=0;pending=''
    with log.open('x') as stream:
        child=subprocess.Popen(list(map(str,command)),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            with log.open() as reader:
                while True:
                    text=reader.read();pending+=text
                    lines=pending.split('\n');pending=lines.pop()
                    for line in lines:
                        match=re.search(r'\bstep:(\S+).*?\bloss:(\S+)',line)
                        if match:
                            last_loss=float(match[2]);assert math.isfinite(last_loss),line
                            max_loss=max(max_loss,last_loss)
                            if last_loss>100:raise RuntimeError('Abnormally large normalized loss: '+line)
                            logs+=1;steps=logs*log_freq;last_progress=time.monotonic()
                        if training and any(x in line for x in ('Traceback (most recent call last)',
                                'CUDA out of memory','AF_UNIX path too long','DataLoader worker (pid')):
                            raise RuntimeError(line)
                    value=status(name,pid=child.pid,log=str(log),training_steps=steps,loss=last_loss,
                        elapsed_s=round(time.monotonic()-started),target_training_steps=30000)
                    key=(value['rendered_frame_groups'],tuple(value['episodes'].values()),value['recording_attempts'])
                    if not training and key!=last_key:last_progress=time.monotonic();last_key=key
                    now=time.monotonic()
                    if now-last_snapshot>=1800:snapshot(value);last_snapshot=now
                    code=child.poll()
                    if code is not None:
                        if code:raise RuntimeError(f'{name} exit={code}; {log}')
                        break
                    if now-started>timeout:raise TimeoutError(f'{name} exceeded timeout')
                    if now-last_progress>(1800 if training else 7200):
                        raise RuntimeError(f'{name}: no real progress; {log}')
                    if value['hdd_free_gib']<150:raise RuntimeError('Low disk: stopping only this task')
                    time.sleep(60)
        except BaseException:
            stop(child);raise
    report=dict(passed=True,log=str(log),finite_loss_logs=logs,confirmed_training_steps=steps,
        last_loss=last_loss,max_loss=max_loss,elapsed_s=round(time.monotonic()-started))
    if training:assert logs and last_loss is not None,report
    save(JOB/f'{name}_process_complete.json',report)
    return report


def train_command(data,rid,out,steps):
    return [PY,'-m','accelerate.commands.launch','--num_processes=1','--mixed_precision=bf16',
        JOB/'code/run_native_pi05_evo.py',SRC/'lerobot/scripts/lerobot_train.py',
        f'--dataset.root={data}',f'--dataset.repo_id={rid}','--dataset.video_backend=torchcodec',
        '--dataset.use_imagenet_stats=false','--dataset.image_transforms.enable=false',
        '--policy.type=pi05_evo','--policy.device=cuda',f'--policy.pretrained_path={BASE}',
        '--policy.dtype=bfloat16','--policy.push_to_hub=false','--policy.gradient_checkpointing=true',
        '--policy.normalization_mapping={"VISUAL":"IDENTITY","STATE":"MIN_MAX","ACTION":"MIN_MAX"}',
        '--policy.chunk_size=100','--policy.n_action_steps=50','--policy.num_inference_steps=10',
        '--batch_size=16',f'--steps={steps}','--num_workers=4',f'--save_freq={3 if steps==3 else 10000}',
        f'--log_freq={1 if steps==3 else 50}','--seed=1000',f'--job_name={out.name}',f'--output_dir={out}',
        '--wandb.enable=false','--wandb.disable_artifact=true']


def verify_checkpoint(out,steps):
    checkpoint=out/'checkpoints'/f'{steps:06d}'
    model=checkpoint/'pretrained_model';cfg=read(model/'config.json')
    assert cfg['type']=='pi05_evo'
    assert cfg['normalization_mapping']=={'VISUAL':'IDENTITY','STATE':'MIN_MAX','ACTION':'MIN_MAX'}
    assert cfg['input_features']['observation.state']['shape']==[10]
    assert cfg['output_features']['action']['shape']==[10]
    assert {k for k,v in cfg['input_features'].items() if v['type']=='VISUAL'}=={
        'observation.images.front','observation.images.wrist'}
    assert set(cfg['input_features'])=={'observation.state','observation.images.front','observation.images.wrist'}
    ts=read(checkpoint/'training_state/training_step.json');assert ts['step']==steps,ts
    weights=list(model.glob('*.safetensors'));assert weights
    return dict(passed=True,steps=steps,checkpoint=str(checkpoint),config=cfg,
        weights_sha256={p.name:sha(p) for p in weights})


def dataset_audit():
    cfg=read(CFG);p=ROOT/'dataset';info=read(p/'meta/info.json');provenance=read(p/'pipeline_provenance.json')
    assert cfg['render']['fps']==25
    assert info['total_episodes']==100 and info['total_frames']==130000 and info['fps']==25
    assert info['features']['observation.state']['shape']==info['features']['action']['shape']==[10]
    assert provenance['readback_verified'] and provenance['video_frames_decoded']==260000
    assert len({e['raw_sha256'] for e in provenance['episodes']})==100
    for rel,expected in read(p/'file_sha256.json').items():assert sha(p/rel)==expected,rel
    means={v:[] for v in cfg['render']['views']};clips={v:[] for v in means}
    for ep in provenance['episodes']:
        render=Path(ep['source_render']);rr=read(render/'render_complete.json')
        for v in means:
            means[v].extend(x['images'][v]['mean'] for x in rr['frames'])
            clips[v].extend(x['images'][v]['clipped_high_fraction'] for x in rr['frames'])
        scene=Path(read(render.parent/'status.json')['scene']).parent
        clearance=read(scene/'room_clearance.json')
        assert not clearance['clearance_warnings'] and clearance['original_assets_unchanged']
        assert len(rr['frames'])==1300
        assert [r['source_index'] for r in rr['frames']]==list(range(0,2600,2))
        assert all(abs(r['timestamp_s']-i/25)<1e-9 for i,r in enumerate(rr['frames']))
        for v in means:
            assert sum(x['images'][v]['clipped_high_fraction'] for x in rr['frames'])/1300<.10,(ep['episode'],v)
    report=dict(passed=True,episodes=100,unique_trajectories=100,rows=130000,fps=25,
        mean_rgb={v:sum(values)/len(values) for v,values in means.items()},
        clipped_high={v:sum(values)/len(values) for v,values in clips.items()})
    save(JOB/'dataset_training_gate.json',report)
    return cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-train',action='store_true');p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'.supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.preflight_train:
            out=JOB/'preflight_train_minmax';assert not out.exists(),'Preserve previous preflight'
            result=run(train_command(JOB/'preflight_dataset','local/wc_dim60_pi05_preflight',out,3),
                environment(1),'PREFLIGHT_TRAIN',True,1,3600)
            save(JOB/'preflight_training_verified.json',dict(process=result,checkpoint=verify_checkpoint(out,3),passed=True))
            status('PREFLIGHT_PASSED');return
        assert read(JOB/'preflight_training_verified.json')['passed']
        assert read(JOB/'preflight_training_verified.json')['checkpoint']['config']['normalization_mapping']['ACTION']=='MIN_MAX'
        if not (ROOT/'dataset/pipeline_provenance.json').exists():
            command=[PY,SIM/'pipeline.py','--config',CFG]
            if a.retry_failed:command.append('--retry-failed')
            # No outer CUDA restriction: the pipeline assigns GPUs per worker.
            env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4');env.pop('CUDA_VISIBLE_DEVICES',None)
            run(command,env,'RENDER_DATASET',timeout=345600)
        cfg=dataset_audit()
        complete=JOB/'training_complete.json'
        if complete.exists():
            assert read(complete)['passed'];status('COMPLETE',checkpoint=read(complete)['checkpoint']);return
        assert not TRAIN.exists(),'Existing training preserved; inspect verified checkpoint before resuming'
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=40000:break
            snapshot(status('WAIT_TRAIN_GPU',gpu=1,free_mib=free));time.sleep(1800)
        result=run(train_command(ROOT/'dataset',cfg['dataset']['repo_id'],TRAIN,30000),environment(1),'TRAINING',True,50,172800)
        verified=verify_checkpoint(TRAIN,30000);verified['process']=result
        save(complete,verified);snapshot(status('COMPLETE',checkpoint=verified['checkpoint']))


if __name__=='__main__':
    def interrupt(signum,frame):raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:main()
    except BaseException as e:
        status('FAILED',error=repr(e));raise
