"""Single-sim RGB DR collection -> audited Joint9 dataset -> native Pi05Evo 30K.

All outputs on HDD. One environment per fresh simulator gives each episode its
own global Gym lighting and avoids overlap between the visual-only room meshes.
No trajectory/physics changes, no automatic evaluation or deletion of old runs.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import math
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time

import numpy as np
import yaml

REPO=Path('/home/ps/workspace/txc/doorgym')
A=Path('/mnt/sda/ps_archive/doorgym/high-level')
TAG='0912_pull_rgb_aug100_joint9_g080_fixed15'
OUT=A/'dp/result'/TAG
RAW=A/'data/door_dp_raw'/TAG
RID='local/door_a2w_'+TAG
DATA=A/'data/lerobot'/RID
JOB='pi05evo_'+TAG+'_seed1000_30k'
TRAIN=A/'dp/logs/lerobot-train-pi05'/JOB
PY='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'
SIM='/home/ps/miniconda3/envs/txc_vbc/bin/python'
SRC=REPO/'high-level/hwk/lerobot/src'
BASE=A/'dp/models/pi05_evo_native_base_0907'


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def state(phase,**kw):
    value=dict(time=time.strftime('%F %T'),phase=phase,**kw)
    save(OUT/'status.json',value);print(json.dumps(value),flush=True)


def environment(gpu):
    env=os.environ.copy()
    cache=OUT/'cache'
    for name in ('datasets','xdg','torch','triton'):(cache/name).mkdir(parents=True,exist_ok=True)
    # multiprocessing adds /pymp-XXXXXXXX/listener-XXXXXXXX to TMPDIR.
    # Linux AF_UNIX pathname capacity is 108 bytes INCLUDING the trailing NUL.
    tmp=Path('/mnt/sda/ps/tmp/rgb100')
    tmp.mkdir(parents=True,exist_ok=True)
    assert len(os.fsencode(str(tmp)+'/pymp-XXXXXXXX/listener-XXXXXXXX'))<108
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=f'{SRC}:{REPO}/high-level',
        TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),
        HF_DATASETS_CACHE=str(cache/'datasets'),XDG_CACHE_HOME=str(cache/'xdg'),
        HF_HOME=os.environ.get('HF_HOME','/home/ps/.cache/huggingface'),
        TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),
        WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
        DOOR_DP_VIDEO_BACKEND='torchcodec',DOOR_DP_LEROBOT_PYTHON=PY,
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    # Keep the existing read-only HF model/tokenizer cache available offline.
    return env


def snapshot(phase,**kw):
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader'],capture_output=True,text=True).stdout
    d=dict(time=time.strftime('%F %T'),phase=phase,gpu=gpu,
        raw_files=len(list(RAW.glob('episode_*.npz'))),hdd_free_gib=shutil.disk_usage(A).free/1024**3,**kw)
    with (OUT/'monitor.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')


class TrainingWatch:
    """Progress is a finite-loss optimizer log, not merely a living process."""
    def __init__(self,now=None):
        self.last_progress=time.monotonic() if now is None else now
        self.pending=''
        self.latest=None
        self.log_count=0

    def feed(self,text,now=None):
        self.pending+=text
        lines=self.pending.split('\n');self.pending=lines.pop()
        for line in lines:
            if any(marker in line for marker in ('Traceback (most recent call last)',
                'AF_UNIX path too long','CUDA out of memory','DataLoader worker (pid')):
                raise RuntimeError('Training worker/runtime failure: '+line.strip())
            match=re.search(r'\bstep:(\S+).*?\bloss:(\S+)',line)
            if match:
                loss=float(match[2])
                if not math.isfinite(loss):raise RuntimeError('Non-finite training loss: '+line)
                self.log_count+=1
                self.latest=dict(step_display=match[1],loss=loss,log_line=line.strip(),
                    # This launcher only supports fresh training, log_freq=50.
                    confirmed_steps=self.log_count*50)
                self.last_progress=time.monotonic() if now is None else now

    def check_stalled(self,now=None):
        now=time.monotonic() if now is None else now
        if now-self.last_progress>1800:
            raise RuntimeError('Training stalled: no finite-loss step log for 30 minutes')


def run(cmd,env,name,quota_failure_ok=False,training=False,stall_timeout_s=None):
    assert shutil.disk_usage(A).free>200*1024**3,'HDD free space below 200GiB'
    cmd=list(map(str,cmd));logpath=OUT/(name+'.log')
    assert not logpath.exists(),f'Preserve existing stage log: {logpath}'
    save(OUT/(name+'.command.json'),dict(argv=cmd,env={k:env[k] for k in
        ('CUDA_VISIBLE_DEVICES','PYTHONPATH','TMPDIR','HF_DATASETS_CACHE','XDG_CACHE_HOME')}))
    with logpath.open('w') as log:
        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(OUT/'active_process.json',dict(pid=child.pid,phase=name,log=str(logpath)))
        last=0;heartbeat=0;watch=TrainingWatch() if training else None
        last_output=time.monotonic()
        try:
            with logpath.open() as reader:
                while child.poll() is None:
                    text=reader.read()
                    if text:last_output=time.monotonic()
                    if stall_timeout_s is not None and time.monotonic()-last_output>stall_timeout_s:
                        raise RuntimeError(f'{name}: no stage output for {stall_timeout_s}s')
                    if watch:
                        watch.feed(text);watch.check_stalled()
                        if time.monotonic()-heartbeat>=60:
                            state('TRAINING' if watch.latest else 'TRAIN_STARTING',
                                gpu=env['CUDA_VISIBLE_DEVICES'],target_steps=30000,seed=1000,
                                pid=child.pid,log=str(logpath),progress=watch.latest,
                                seconds_since_progress=round(time.monotonic()-watch.last_progress))
                            heartbeat=time.monotonic()
                    if '[Error]' in text or 'Could not load visual mesh' in text:
                        raise RuntimeError('Simulator/render error; inspect '+str(logpath))
                    if time.time()-last>=1800:
                        snapshot(name,pid=child.pid);last=time.time()
                    if shutil.disk_usage(A).free<100*1024**3:
                        raise RuntimeError('Stopping this task before HDD exhausts')
                    time.sleep(10)
                if watch:watch.feed(reader.read()+'\n')
        except BaseException:
            with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGKILL)
                child.wait()
            raise
    text=logpath.read_text(errors='replace')
    assert '[Error]' not in text and 'Could not load visual mesh' not in text,logpath
    if child.returncode:
        if quota_failure_ok and 'Reached --max_quota_rollouts=1 with' in text:return False
        raise RuntimeError(f'{name} exit={child.returncode}: {logpath}')
    if watch:
        assert watch.latest and watch.latest['confirmed_steps']==30000,watch.latest
    return True


def recording_command(target,attempt,preview):
    source=REPO/'high-level/dp/record/run_wc_nomenkuang_fixed15_gripper080_rgb_pi05_4way_ps1.sh'
    trajectory=shlex.split(source.read_text().split('TRAJECTORY_ARGS=(',1)[1].split('\n)',1)[0])
    return [SIM,'high-level/dp/record/record_door_dp_dataset_a2w_state10.py',
        '--mode','ikpull','--num_episodes',str(target),'--num_envs','1','--max_quota_rollouts','1',
        '--steps','2600','--seed',str(615455575+attempt),'--continue_after_cleanup_sigsegv','--headless',
        '--raw_root',RAW,'--state_action_mode','joint_state9','--rl_device','cuda:0','--sim_device','cuda:0',
        '--graphics_device_id','0','--rgb_only','--no_enable_depth_noise','--no_enable_depth_gaussian_blur',
        '--enable_depth_camera_randomization','--depth_camera_pos_rand_m','0.02','--depth_camera_rot_rand_deg','5',
        '--keyframe_loss_weight','3','--keyframe_loss_radius','3','--record_camera_pose','--record_handle_bbox',
        '--record_gripper_handle_contact','--filter_gripper_handle_contact',
        '--filter_gripper_handle_contact_min_frames','5','--filter_gripper_handle_contact_phase_names',
        'close_gripper,rotate_handle','--filter_gripper_handle_contact_require_both','--',*trajectory,
        '--rgb_visual_mode','doorman','--rgb_visual_config',OUT/'rgb_config.yaml',
        '--rgb_visual_assets',A/'data/visual_randomization/doorman','--rgb_visual_seed','-1',
        '--rgb_visual_preview_dir',preview,'--no_draw_scripted_trajectory','--no_draw_ik_target',
        '--no_draw_camera_axes','--no_preview_trajectory_at_spawn','--no_show_seg']


def audit_episode(path,attempt,preview):
    with np.load(path,allow_pickle=True) as d:
        s,a=d['state'],d['action'];n=len(s)
        assert s.shape==a.shape==(n,9) and n>0
        assert np.isfinite(s).all() and np.isfinite(a).all()
        np.testing.assert_array_equal(s[:,2:],d['replay_dof_pos'][:,-7:])
        assert str(d['task'])=='pull lever door open'
        assert float(d['pull_max_open_deg'])>=80 and float(d['pull_max_traversal_m'])>=.8
        assert abs(float(d['pull_release_angle_observed_deg'])-15)<2
        for key in ('front_camera_pose_base','wrist_camera_pose_base','gripper_handle_contact_both',
                    'replay_door_dof_pos'):assert len(d[key])==n,key
        episode=d['rgb_visual_episode'].item();meta=d['rgb_visual_randomization'].item()
        assert episode['physics_before']==episode['physics_after']
        assert meta['config']==yaml.safe_load((OUT/'rgb_config.yaml').read_text())
        image_stats={}
        for key in ('front_rgb','wrist_rgb'):
            frames=d[key]
            assert frames.shape==(n,480,640,3) and frames.dtype==np.uint8
            means=frames.mean(axis=(1,2,3))
            assert means.min()>1 and means.max()<254,(key,means.min(),means.max())
            image_stats[key]=dict(shape=list(frames.shape),mean_min=float(means.min()),mean_max=float(means.max()))
            del frames
        samples=[json.loads(p.read_text()) for p in sorted(preview.glob('env00_front_*.json'))]
        assert len(samples)>=3
        assert len({json.dumps(p['parameters'],sort_keys=True) for p in samples})==1
        assert len({(p['intensity'],p['hdr']['prepared_sha256']) for p in samples})==1
        return dict(path=str(path),attempt=attempt,frames=n,seed=int(d['seed']),visual_seed=episode['seed'],
            images=image_stats,parameters=samples[0]['parameters'],intensity=samples[0]['intensity'],
            hdr=samples[0]['hdr'],materials=episode['materials'],physics_unchanged=True,
            opening=float(d['pull_max_open_deg']),traversal=float(d['pull_max_traversal_m']))


def train_command():
    return [PY,'-m','accelerate.commands.launch','--num_processes=1','--mixed_precision=bf16',
        REPO/'high-level/dp/train/run_native_pi05_evo.py',SRC/'lerobot/scripts/lerobot_train.py',
        f'--dataset.root={DATA}',f'--dataset.repo_id={RID}','--dataset.video_backend=torchcodec',
        '--dataset.use_imagenet_stats=false','--dataset.image_transforms.enable=false',
        '--policy.type=pi05_evo','--policy.device=cuda',f'--policy.pretrained_path={BASE}',
        '--policy.dtype=bfloat16','--policy.push_to_hub=false','--policy.gradient_checkpointing=true',
        '--policy.chunk_size=100','--policy.n_action_steps=50','--policy.num_inference_steps=10',
        '--batch_size=16','--steps=30000','--num_workers=4','--save_freq=10000','--log_freq=50',
        '--seed=1000',f'--job_name={JOB}',f'--output_dir={TRAIN}',
        '--wandb.enable=false','--wandb.disable_artifact=true']


def train_validated_dataset(env,gpu,name):
    assert not TRAIN.exists(),'Existing model output preserved; verified checkpoint resume needed'
    manifest=json.loads((OUT/'raw_manifest.json').read_text())
    info=json.loads((DATA/'meta/info.json').read_text())
    assert info==json.loads((OUT/'dataset_ready.json').read_text())
    assert len(manifest)==info['total_episodes']==100
    assert info['total_frames']==sum(m['frames'] for m in manifest)==130000
    features=info['features']
    assert features['observation.state']['shape']==features['action']['shape']==[9]
    assert {k for k,v in features.items() if v['dtype'] in ('video','image')}=={
        'observation.images.front_rgb','observation.images.wrist_rgb'}
    # Exercise the ACTUAL training interpreter's worker->parent tensor transfer.
    run([PY,REPO/'high-level/dp/tests/test_rgb_aug100_training_runtime.py','--ipc-only'],
        env,name+'_ipc_preflight')
    state('WAIT_TRAIN_GPU',gpu=gpu)
    while True:
        free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free>=38000:break
        snapshot('WAIT_TRAIN_GPU',free_mib=free);time.sleep(300)
    state('TRAIN_STARTING',gpu=gpu,target_steps=30000,seed=1000,log=str(OUT/(name+'.log')))
    run(train_command(),env,name,training=True)
    ckpt=TRAIN/'checkpoints/030000/pretrained_model'
    assert (ckpt/'model.safetensors').is_file()
    cfg=json.loads((ckpt/'config.json').read_text())
    assert cfg['type']=='pi05_evo'
    assert cfg['input_features']['observation.state']['shape']==cfg['output_features']['action']['shape']==[9]
    processor=json.loads((ckpt/'policy_preprocessor.json').read_text())
    assert any(s['registry_name']=='pi05_evo_prepare_state_tokenizer_processor_step' for s in processor['steps'])
    save(OUT/'completed.json',dict(checkpoint=str(ckpt),episodes=100,steps=30000,config=cfg))
    state('COMPLETE',checkpoint=str(ckpt),evaluation='not requested / not run')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--train-only',action='store_true',help='Reuse audited data; restart only before any checkpoint exists')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.train_only:
        name='train_restart_'+time.strftime('%Y%m%d_%H%M%S')
        try:
            previous=json.loads((OUT/'status.json').read_text())
            save(OUT/(name+'.previous_status.json'),previous)
            planned=json.loads((OUT/'planned_train.command.json').read_text())['argv']
            assert planned==list(map(str,train_command())),'Do not silently change training hyperparameters'
            env=environment(args.gpu)
            save(OUT/(name+'.repair.json'),dict(reason='AF_UNIX TMPDIR too long; user authorized restart',
                tmpdir=env['TMPDIR'],dataset_unchanged=True,training_argv_unchanged=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            shutil.copy2(Path(__file__),OUT/(name+'.source.py'))
            train_validated_dataset(env,args.gpu,name)
        except BaseException as exc:
            state('FAILED',error=repr(exc));raise
        return
    assert not (OUT/'status.json').exists(),'Existing run preserved; inspect status before resuming'
    assert not RAW.exists() and not DATA.exists() and not TRAIN.exists(),'Use fresh experiment paths'
    env=environment(args.gpu)
    try:
        state('PREFLIGHT',gpu=args.gpu)
        assert (BASE/'model.safetensors').is_file()
        shutil.copy2(REPO/'high-level/dp/config/rgb_doorman_stable.yaml',OUT/'rgb_config.yaml')
        sources=['float_ik/rgb_robot_materials.py','float_ik/rgb_doorman.py','float_ik/rgb_visual_scene.py',
            'float_ik/rgb_door_materials.py','float_ik/door_common.py',
            'float_ik/isaacgym_float_ik_a2w_basearn_pull_door_parallel.py',
            'dp/train/run_pull_rgb_aug100_pi05_0912_ps1.py','dp/train/run_native_pi05_evo.py',
            'dp/record/record_door_dp_dataset_a2w_state10.py',
            'dp/record/run_wc_nomenkuang_fixed15_gripper080_rgb_pi05_4way_ps1.sh',
            'dp/convert_door_raw_to_lerobot.py']
        hashes={}
        for name in sources:
            src=REPO/'high-level'/name;dest=OUT/'source_snapshot'/name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
        save(OUT/'experiment.json',dict(gpu=args.gpu,episodes=100,num_envs=1,steps=2600,
            raw=str(RAW),dataset=str(DATA),train=str(TRAIN),source_sha256=hashes,
            augmentation='recorded once; no second training color jitter',
            policy='native hwk Pi05Evo; normalized/quantized Joint9 + task tokens; dual RGB',
            state='last base vx/yaw commands + measured six arm joints and gripper',
            action='base vx/yaw commands + six arm joint targets and final gripper command'))
        save(OUT/'planned_train.command.json',dict(argv=list(map(str,train_command()))))
        run([SIM,'high-level/dp/tests/test_rgb_robot_materials.py'],env,'preflight_mesh_tests')
        run([PY,REPO/'high-level/dp/train/run_native_pi05_evo.py',SRC/'lerobot/scripts/lerobot_train.py','--help'],
            env,'preflight_native_trainer')
        manifest=[]
        for attempt in range(500):
            count=len(manifest)
            if count==100:break
            state('RECORDING',validated_episodes=count,target=100,attempt=attempt,gpu=args.gpu)
            preview=OUT/'previews'/f'attempt_{attempt:04d}'
            ok=run(recording_command(count+1,attempt,preview),env,f'record_{attempt:04d}',quota_failure_ok=True)
            paths=sorted(RAW.glob('episode_*.npz'))
            assert len(paths) in (count,count+1)
            if len(paths)==count+1:
                assert ok,'Saved partial episode with unsuccessful recorder exit; inspect manually'
                assert paths[-1].name==f'episode_{count:06d}.npz'
                state('AUDIT_EPISODE',episode=count,attempt=attempt)
                manifest.append(audit_episode(paths[-1],attempt,preview))
                save(OUT/'raw_manifest.json',manifest)
        assert len(manifest)==100,'Exceeded recording attempt limit'
        assert len({m['visual_seed'] for m in manifest})==100
        assert len({m['intensity'] for m in manifest})>10
        assert sum('contrast' in m['parameters'] for m in manifest)>0
        assert sum('brightness' in m['parameters'] for m in manifest)>0
        state('CONVERTING',episodes=100)
        ce=env.copy();ce['PYTHONPATH']=f'{REPO}/high-level/lerobot/src:{REPO}/high-level'
        run([PY,'high-level/dp/convert_door_raw_to_lerobot.py','--raw_root',RAW,
            '--root',A/'data/lerobot','--repo_id',RID,'--state_action_mode','joint_state9',
            '--rgb_only','--image_storage','video','--video_codec','h264','--num_workers','4'],ce,'convert')
        info=json.loads((DATA/'meta/info.json').read_text())
        assert info['total_episodes']==100 and info['total_frames']==sum(m['frames'] for m in manifest)
        features=info['features']
        assert features['observation.state']['shape']==features['action']['shape']==[9]
        images={k for k,v in features.items() if v['dtype'] in ('video','image')}
        assert images=={'observation.images.front_rgb','observation.images.wrist_rgb'}
        save(DATA/'rgb_visual_augmentation_manifest.json',dict(config=yaml.safe_load((OUT/'rgb_config.yaml').read_text()),episodes=manifest))
        save(OUT/'dataset_ready.json',info)
        train_validated_dataset(env,args.gpu,'train')
    except BaseException as exc:
        state('FAILED',error=repr(exc));raise


if __name__=='__main__':main()
