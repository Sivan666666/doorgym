"""Nested RGB DR500: verified original100 + new400 -> native Evo30K -> eval64.

Fresh experiment paths; the original raw data/checkpoints/results are read-only.
Four independent one-environment recording processes preserve per-episode lights.
Only COMPLETE after the final 25deg/0.5m evaluation and independent log audit.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np
import yaml
import run_pull_rgb_aug100_pi05_0912_ps1 as r

REPO=r.REPO
A=r.A
OLD_OUT=r.OUT
TAG='0914_pull_rgb_aug500_joint9_g080_fixed15'
OUT=A/'dp/result'/TAG
RAW=A/'data/door_dp_raw'/TAG
SHARDS=A/'data/door_dp_raw'/(TAG+'_new400')
RID='local/door_a2w_'+TAG
DATA=A/'data/lerobot'/RID
JOB='pi05evo_'+TAG+'_seed1000_30k'
TRAIN=A/'dp/logs/lerobot-train-pi05'/JOB
OFFICIAL=TRAIN/'checkpoints/030000/pretrained_model'
EVAL_OUT=OUT/'eval64_unlock25_traverse050'
GPUS=(1,3,1,3)
NEW_SEED=615456575


def install_signal_handlers():
    def stop(signum,frame):raise InterruptedError(f'Task termination signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def configure(out=OUT,raw=RAW):
    r.OUT=out;r.RAW=raw;r.DATA=DATA;r.RID=RID;r.JOB=JOB;r.TRAIN=TRAIN


def environment(gpu):
    env=r.environment(gpu)
    tmp=Path('/mnt/sda/ps/tmp/rgb500');tmp.mkdir(parents=True,exist_ok=True)
    env.update(TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp))
    return env


def source_checks():
    old=json.loads((OLD_OUT/'experiment.json').read_text())['source_sha256']
    hashes={k:v for k,v in old.items() if k.startswith('float_ik/') or k.startswith('dp/record/')
        or k in ('dp/convert_door_raw_to_lerobot.py','dp/train/run_native_pi05_evo.py')}
    for name,value in hashes.items():
        assert hashlib.sha256((REPO/'high-level'/name).read_bytes()).hexdigest()==value,('Source changed',name)
    return hashes


def link_episode(source,dest):
    assert source.is_file(),source
    if dest.exists() or dest.is_symlink():assert os.path.samefile(source,dest),(source,dest)
    else:dest.symlink_to(source)


def reuse100():
    marker=OUT/'reuse100.audit.json'
    if marker.exists():return json.loads(marker.read_text())['episodes']
    old=json.loads((OLD_OUT/'raw_manifest.json').read_text())
    assert len(old)==100 and sum(x['frames'] for x in old)==130000
    RAW.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((OUT/'rgb_config.yaml').read_text());rows=[]
    for i,item in enumerate(old):
        src=Path(item['path'])
        with np.load(src,allow_pickle=True) as d:
            s,a=d['state'],d['action']
            assert s.shape==a.shape==(1300,9) and np.isfinite(s).all() and np.isfinite(a).all()
            assert int(d['seed'])==item['seed']
            assert d['rgb_visual_randomization'].item()['config']==cfg
            assert d['rgb_visual_episode'].item()['seed']==item['visual_seed']
            assert str(d['task'])=='pull lever door open'
        # Original full pixel/physics/quality audits are retained. Recheck array
        # headers without decompressing 240GB of already-validated image pixels.
        with zipfile.ZipFile(src) as archive:
            for key in ('front_rgb','wrist_rgb'):
                with archive.open(key+'.npy') as f:
                    version=np.lib.format.read_magic(f)
                    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                    shape,fortran,dtype=reader(f)
                assert shape==(1300,480,640,3) and dtype==np.uint8 and not fortran
                assert item['images'][key]['mean_min']>1 and item['images'][key]['mean_max']<254
        dest=RAW/f'episode_{i:06d}.npz';link_episode(src,dest)
        rows.append(dict(item,path=str(dest),source_path=str(src),global_episode=i,origin='original_aug100'))
    source_root=Path(old[0]['path']).parent
    shutil.copy2(source_root/'door_dp_feature_names.json',RAW/'door_dp_feature_names.json')
    r.save(marker,dict(passed=True,episodes=rows,nested_first100=True,
        verification='same-file links; schema/state/seeds rechecked; original full image/physics audits retained'))
    return rows


def recording_command(lane,count,attempt):
    preview=OUT/'lanes'/f'lane_{lane}'/'previews'/f'attempt_{attempt:04d}'
    cmd=list(r.recording_command(count+1,attempt,preview))
    cmd[cmd.index('--seed')+1]=str(NEW_SEED+lane*10000+attempt)
    cmd[cmd.index('--raw_root')+1]=SHARDS/f'lane_{lane}'
    cmd[cmd.index('--rgb_visual_config')+1]=OUT/'rgb_config.yaml'
    return cmd,preview


def record_lane(lane,gpu):
    lane_out=OUT/'lanes'/f'lane_{lane}';lane_out.mkdir(parents=True,exist_ok=True)
    raw=SHARDS/f'lane_{lane}'
    configure(lane_out,raw)
    lock=(lane_out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shutil.copy2(OUT/'rgb_config.yaml',lane_out/'rgb_config.yaml')
    manifest_path=lane_out/'raw_manifest.json'
    rows=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    assert len(rows)<=100
    env=environment(gpu)
    try:
        # If interrupted between saving an NPZ and saving its audit, audit that
        # episode before resuming. Never silently discard or double-count it.
        paths=sorted(raw.glob('episode_*.npz'))
        assert len(paths) in (len(rows),len(rows)+1)
        if len(paths)>len(rows):
            with np.load(paths[-1],allow_pickle=True) as d:seed=int(d['seed'])
            attempt=seed-(NEW_SEED+lane*10000)
            assert 0<=attempt<1000
            log=(lane_out/f'record_{attempt:04d}.log').read_text(errors='replace')
            assert 'Done after 2600 steps' in log and '[Error]' not in log
            row=r.audit_episode(paths[-1],attempt,lane_out/'previews'/f'attempt_{attempt:04d}')
            rows.append(row);r.save(manifest_path,rows)
        prior_attempts=[int(p.stem.split('_')[1]) for p in lane_out.glob('record_*.log')]
        start=max(prior_attempts,default=-1)+1
        for attempt in range(start,1000):
            if len(rows)==100:break
            source_checks()
            r.state('RECORDING',lane=lane,gpu=gpu,validated_episodes=len(rows),target=100,attempt=attempt)
            cmd,preview=recording_command(lane,len(rows),attempt)
            ok=r.run(cmd,env,f'record_{attempt:04d}',quota_failure_ok=True,stall_timeout_s=3600)
            paths=sorted(raw.glob('episode_*.npz'))
            assert len(paths) in (len(rows),len(rows)+1)
            if len(paths)>len(rows):
                assert ok and paths[-1].name==f'episode_{len(rows):06d}.npz'
                r.state('AUDIT_EPISODE',lane=lane,episode=len(rows),attempt=attempt)
                row=r.audit_episode(paths[-1],attempt,preview)
                assert row['seed']==row['visual_seed']==NEW_SEED+lane*10000+attempt
                assert row['frames']==1300
                rows.append(row);r.save(manifest_path,rows)
        assert len(rows)==100,'Recording attempt limit exceeded'
        r.state('COMPLETE',lane=lane,validated_episodes=100)
    except BaseException as exc:
        r.state('FAILED',lane=lane,error=repr(exc));raise


def collect400():
    children=[]
    try:
        for lane,gpu in enumerate(GPUS):
            meta=OUT/'lanes'/f'lane_{lane}'/'raw_manifest.json'
            if meta.exists() and len(json.loads(meta.read_text()))==100:continue
            cmd=[r.SIM,'-u',str(Path(__file__).resolve()),'--record-lane',str(lane),'--gpu',str(gpu)]
            stamp=time.strftime('%Y%m%d_%H%M%S')
            with (OUT/f'lane_{lane}.launch_{stamp}.log').open('x') as log:
                proc=subprocess.Popen(cmd,cwd=REPO,env=environment(gpu),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((lane,proc))
        r.save(OUT/'record_processes.json',[dict(lane=j,pid=p.pid,gpu=GPUS[j]) for j,p in children])
        while True:
            counts=[]
            for lane in range(4):
                meta=OUT/'lanes'/f'lane_{lane}'/'raw_manifest.json'
                counts.append(len(json.loads(meta.read_text())) if meta.exists() else 0)
            r.state('RECORDING',validated_episodes=100+sum(counts),target=500,reused=100,new_by_lane=counts)
            for lane,proc in children:
                if proc.poll() is not None and proc.returncode!=0:raise RuntimeError(f'Record lane{lane} failed: {proc.returncode}')
            if all(proc.poll() is not None for _,proc in children):break
            if shutil.disk_usage(A).free<150*1024**3:raise RuntimeError('Low HDD free space')
            time.sleep(60)
        assert counts==[100]*4,counts
    except BaseException:
        for _,proc in children:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGTERM)
        for _,proc in children:
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
        raise


def assemble500(old):
    rows=list(old)
    for lane in range(4):
        new=json.loads((OUT/'lanes'/f'lane_{lane}'/'raw_manifest.json').read_text())
        assert len(new)==100
        for local_id,item in enumerate(new):
            i=100+lane*100+local_id;src=Path(item['path']);dest=RAW/f'episode_{i:06d}.npz'
            link_episode(src,dest)
            rows.append(dict(item,path=str(dest),source_path=str(src),global_episode=i,origin=f'new_lane_{lane}'))
    assert len(rows)==500 and sum(r['frames'] for r in rows)==650000
    assert len({r['seed'] for r in rows})==len({r['visual_seed'] for r in rows})==500
    assert {r['seed'] for r in rows}.isdisjoint(range(715455575,715455639))
    assert {r['visual_seed'] for r in rows}.isdisjoint(range(815455575,815455639))
    assert [p.name for p in sorted(RAW.glob('episode_*.npz'))]==[f'episode_{i:06d}.npz' for i in range(500)]
    r.save(OUT/'raw_manifest.json',rows)
    r.save(OUT/'raw500_ready.json',dict(episodes=500,frames=650000,nested_original100=True,new_episodes=400))
    return rows


def training_command():
    old=json.loads((OLD_OUT/'planned_train.command.json').read_text())['argv']
    replacements={'--dataset.root':str(DATA),'--dataset.repo_id':RID,'--job_name':JOB,'--output_dir':str(TRAIN)}
    cmd=list(old)
    for i,arg in enumerate(cmd):
        key=arg.split('=',1)[0]
        if key in replacements:cmd[i]=key+'='+replacements[key]
    assert f'--policy.pretrained_path={r.BASE}' in cmd
    assert '--steps=30000' in cmd and '--seed=1000' in cmd
    return cmd


def checkpoint_audit():
    assert (OFFICIAL/'model.safetensors').is_file()
    cfg=json.loads((OFFICIAL/'config.json').read_text())
    assert cfg['type']=='pi05_evo'
    assert cfg['input_features']['observation.state']['shape']==cfg['output_features']['action']['shape']==[9]
    tc=json.loads((OFFICIAL/'train_config.json').read_text())
    assert tc['dataset']['repo_id']==RID and str(tc['dataset']['root'])==str(DATA)
    assert tc['steps']==30000 and tc['seed']==1000
    pre=json.loads((OFFICIAL/'policy_preprocessor.json').read_text())
    assert any(s['registry_name']=='pi05_evo_prepare_state_tokenizer_processor_step' for s in pre['steps'])
    r.save(OUT/'training_complete.json',dict(checkpoint=str(OFFICIAL),episodes=500,steps=30000,config=cfg))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--record-lane',type=int,choices=range(4));parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();install_signal_handlers()
    if args.record_lane is not None:return record_lane(args.record_lane,args.gpu)
    OUT.mkdir(parents=True,exist_ok=True);configure()
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.resume:
        assert not (OUT/'status.json').exists() and not RAW.exists() and not DATA.exists() and not TRAIN.exists(),'Use --resume after inspecting existing task artifacts'
    try:
        r.state('PREFLIGHT',target_episodes=500,train_gpu=args.gpu)
        assert shutil.disk_usage(A).free>800*1024**3,'Need at least800GiB HDD headroom'
        hashes=source_checks()
        cfg=OUT/'rgb_config.yaml'
        if cfg.exists():assert cfg.read_bytes()==(OLD_OUT/'rgb_config.yaml').read_bytes()
        else:shutil.copy2(OLD_OUT/'rgb_config.yaml',cfg)
        for name in hashes:
            dest=OUT/'source_snapshot'/name
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/'high-level'/name,dest)
        script_copy=OUT/('pipeline.source_'+time.strftime('%Y%m%d_%H%M%S')+'.py')
        shutil.copy2(Path(__file__),script_copy)
        r.save(OUT/'experiment.json',dict(target_episodes=500,reused_episodes=100,new_episodes=400,
            raw=str(RAW),dataset=str(DATA),train=str(TRAIN),train_gpu=args.gpu,record_gpus=GPUS,
            train_steps=30000,train_seed=1000,initialization=str(r.BASE),source_sha256=hashes,
            evaluation=dict(trials=64,physical_unlock_deg=25,open_deg=60,traversal_m=.5,
                paired_reference='0913_pull_rgb_aug100_unlock25_traverse050_eval64')))
        r.save(OUT/'planned_train.command.json',dict(argv=training_command()))
        env=environment(args.gpu)
        if not (OUT/'preflight_tests.ok.json').exists():
            r.run([r.PY,REPO/'high-level/dp/tests/test_rgb_aug100_training_runtime.py'],env,'preflight_tests')
            r.run([r.PY,REPO/'high-level/dp/tests/test_rgb_aug500_pipeline.py'],env,'preflight_pipeline_tests')
            r.save(OUT/'preflight_tests.ok.json',dict(passed=True))
        old=reuse100()
        collect400()
        rows=assemble500(old)
        source_checks()
        if not (OUT/'dataset_ready.json').exists():
            if DATA.exists():
                partial=DATA.with_name(DATA.name+'.partial_'+time.strftime('%Y%m%d_%H%M%S'))
                DATA.rename(partial);r.save(OUT/'preserved_partial_conversion.json',dict(path=str(partial)))
            r.state('CONVERTING',episodes=500,frames=650000)
            ce=env.copy();ce['PYTHONPATH']=f'{REPO}/high-level/lerobot/src:{REPO}/high-level'
            r.run([r.PY,'high-level/dp/convert_door_raw_to_lerobot.py','--raw_root',RAW,
                '--root',A/'data/lerobot','--repo_id',RID,'--state_action_mode','joint_state9',
                '--rgb_only','--image_storage','video','--video_codec','h264','--num_workers','4'],ce,
                'convert_'+time.strftime('%Y%m%d_%H%M%S'),stall_timeout_s=3600)
            info=json.loads((DATA/'meta/info.json').read_text())
            assert info['total_episodes']==500 and info['total_frames']==650000
            f=info['features'];assert f['observation.state']['shape']==f['action']['shape']==[9]
            assert {k for k,v in f.items() if v['dtype'] in ('video','image')}=={'observation.images.front_rgb','observation.images.wrist_rgb'}
            r.save(DATA/'rgb_visual_augmentation_manifest.json',dict(config=yaml.safe_load(cfg.read_text()),episodes=rows))
            r.save(OUT/'dataset_ready.json',info)
        info=json.loads((DATA/'meta/info.json').read_text())
        assert info==json.loads((OUT/'dataset_ready.json').read_text()) and info['total_episodes']==500
        if not (OFFICIAL/'model.safetensors').exists():
            assert not TRAIN.exists(),'Partial training preserved: verify optimizer/checkpoint resume before proceeding'
            r.state('WAIT_TRAIN_GPU',gpu=args.gpu)
            while int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}',
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<38000:
                r.snapshot('WAIT_TRAIN_GPU');time.sleep(300)
            r.state('TRAIN_STARTING',gpu=args.gpu,target_steps=30000)
            r.run(training_command(),env,'train_'+time.strftime('%Y%m%d_%H%M%S'),training=True)
        checkpoint_audit()
        r.state('EVALUATING',checkpoint=str(OFFICIAL),evaluation_status=str(EVAL_OUT/'status.json'))
        final=EVAL_OUT/'final_audit.json'
        if not final.exists():
            r.run([r.SIM,'-u',REPO/'high-level/dp/eval/run_pull_rgb_aug500_eval64_0914_ps1.py'],env,
                'evaluate_'+time.strftime('%Y%m%d_%H%M%S'))
        result=json.loads((EVAL_OUT/'summary.json').read_text());checked=json.loads(final.read_text())
        assert checked['passed'] and checked['records']==166400
        assert result['complete'] and result['trials']==64 and result['checkpoint']==str(OFFICIAL)
        assert result['physical_unlock_deg']==25 and result['traversal_distance_m']==.5
        r.save(OUT/'completed.json',dict(episodes=500,train_steps=30000,checkpoint=str(OFFICIAL),
            evaluation=result,final_audit=str(final)))
        r.state('COMPLETE',episodes=500,steps=30000,eval_trials=64,successes=result['success_count'],
            success_rate=result['success_rate'],checkpoint=str(OFFICIAL))
    except BaseException as exc:
        r.state('FAILED',error=repr(exc));raise


if __name__=='__main__':main()
