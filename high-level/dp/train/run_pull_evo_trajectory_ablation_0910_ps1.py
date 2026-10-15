"""HDD-only, three-lane native PI05Evo trajectory ablation with 30min monitoring.

No changes to native policy/state/language processing or the old data. New expert
variants pass a small recording gate before their fifty-episode production runs.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import numpy as np

REPO=Path('/home/ps/workspace/txc/doorgym')
A=Path('/mnt/sda/ps_archive/doorgym/high-level')
OUT=A/'dp/result/pull_evo_trajectory_20260910'
OLD=A/'logs/0907_pull_rgb_joint9_pi05_evo'
PY='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'
SIM='/home/ps/miniconda3/envs/txc_vbc/bin/python'
SRC=REPO/'high-level/hwk/lerobot/src'
VARIANTS={
    'smooth15': ['--pull_expert_arm_command_max_speed','1.5','--panel_postpush_home_steps','100'],
    'steady15': ['--pull_expert_arm_command_max_speed','1.0','--panel_postpush_home_steps','150',
        '--panel_reposition_steps','350','--pull_panel_low_angle_reposition_steps','350'],
    'release30': ['--pull_panel_release_angle_deg','30'],
    'release30compact': ['--pull_panel_release_angle_deg','30',
        '--pull_base_retreat_distance','0.36','--panel_right_arc_radius','0.12',
        '--panel_edge_clearance','0.12','--panel_right_arc_outward_clearance','0.20',
        '--panel_reposition_steps','260','--release_clearance_base_retreat_extra','0.15'],
    'trimtail15': [],
}


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def status(name,phase,**extra):
    d=dict(time=time.strftime('%F %T'),name=name,phase=phase,**extra)
    save(OUT/f'{name}.status.json',d);print(json.dumps(d),flush=True)


def environment(gpu,name):
    env=os.environ.copy();tmp=Path('/mnt/sda/ps/dp_tmp0910')/name;tmp.mkdir(parents=True,exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=f'{SRC}:{REPO}/high-level',TMPDIR=str(tmp),
        WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DOOR_DP_VIDEO_BACKEND='torchcodec',DOOR_DP_LEROBOT_PYTHON=PY,PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    return env


def run(cmd,env,name):
    assert shutil.disk_usage(A).free>180*1024**3,'HDD free space below 180GiB'
    save(OUT/f'{name}.command.json',dict(argv=list(map(str,cmd)),env={k:env[k] for k in
        ('CUDA_VISIBLE_DEVICES','PYTHONPATH','TMPDIR')}))
    with (OUT/f'{name}.log').open('a') as log:
        subprocess.run(list(map(str,cmd)),cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def trajectory(name):
    source=REPO/'high-level/dp/record/run_wc_nomenkuang_fixed15_gripper080_rgb_pi05_4way_ps1.sh'
    args=shlex.split(source.read_text().split('TRAJECTORY_ARGS=(',1)[1].split('\n)',1)[0])
    for flag,value in zip(VARIANTS[name][::2],VARIANTS[name][1::2]):
        if flag in args:args[args.index(flag)+1]=value
        else:args.extend([flag,value])
    return args


def record(name,env,root,n,smoke=False):
    if len(list(root.glob('episode_*.npz'))) == n:return
    # Fresh production roots, or safely resume with non-overlapping seed block.
    count=len(list(root.glob('episode_*.npz')))
    seed=615455575 if count==0 else 625455575+count*10000
    cmd=[SIM,'high-level/dp/record/record_door_dp_dataset_a2w_state10.py',
        '--mode','ikpull','--num_episodes',str(n),'--num_envs','4','--max_quota_rollouts','6' if smoke else '100',
        '--steps','2600','--seed',str(seed),'--continue_after_cleanup_sigsegv','--headless',
        '--raw_root',root,'--state_action_mode','joint_state9','--rl_device','cuda:0','--sim_device','cuda:0',
        '--graphics_device_id','0','--rgb_only','--camera_depth_clip_lower','0.2','--camera_depth_clip_far','1.5',
        '--enable_depth_camera_randomization','--depth_camera_pos_rand_m','0.02','--depth_camera_rot_rand_deg','5',
        '--keyframe_loss_weight','3','--keyframe_loss_radius','3','--record_camera_pose','--record_handle_bbox',
        '--record_gripper_handle_contact','--filter_gripper_handle_contact',
        '--filter_gripper_handle_contact_min_frames','5','--filter_gripper_handle_contact_phase_names',
        'close_gripper,rotate_handle','--filter_gripper_handle_contact_require_both','--']+trajectory(name)
    run(cmd,env,name+('.smoke' if smoke else '.record'))


def validate_raw(root,n,name):
    paths=sorted(root.glob('episode_*.npz'))
    assert [p.name for p in paths]==[f'episode_{i:06d}.npz' for i in range(n)]
    report=[]
    for p in paths:
        with np.load(p,allow_pickle=True) as d:
            state,action=d['state'],d['action']
            assert state.shape==action.shape and state.shape[1]==9
            assert np.isfinite(state).all() and np.isfinite(action).all()
            np.testing.assert_array_equal(state[:,2:],d['replay_dof_pos'][:,-7:])
            assert str(d['task'])=='pull lever door open'
            for key in ('front_rgb','wrist_rgb','front_camera_pose_base','wrist_camera_pose_base',
                        'gripper_handle_contact_both','replay_door_dof_pos'):assert key in d,key
            release=float(d['pull_release_angle_observed_deg'])
            assert abs(release-(30 if name.startswith('release30') else 15))<2.0
            assert float(d['pull_max_open_deg'])>=80 and float(d['pull_max_traversal_m'])>=.8
            names=list(map(str,d['phase_names']));phases=d['subtask_index'].reshape(-1)
            delta=np.max(np.abs(np.diff(action[:,2:8],axis=0)),axis=1)
            mask=np.isin(phases[1:],[i for i,k in enumerate(names) if k in
                ['pull_door','release_handle','panel_reposition','panel_contact','panel_push','pass_through','return_home','hold_home']])
            if name in ('smooth15','steady15'):
                limit=.06001 if name=='smooth15' else .04001
                assert delta[mask].max()<=limit,(p,float(delta[mask].max()))
            report.append(dict(path=str(p),frames=len(state),release_deg=release,
                opening=float(d['pull_max_open_deg']),traversal=float(d['pull_max_traversal_m']),
                arm_jump_max=float(delta[mask].max()),hold_frames=int((phases==names.index('hold_home')).sum())))
    save(OUT/f'{name}.raw{n}_audit.json',report)
    return report


def trim_original(root):
    """Only shorten the trailing hold; all retained state/action/image samples are unchanged."""
    manifest=json.loads((OLD/'raw500_manifest.json').read_text())[:50]
    root.mkdir(parents=True,exist_ok=True)
    source_root=Path(manifest[0]['path']).parent
    for sidecar in source_root.glob('*.json'):
        if not (root/sidecar.name).exists():shutil.copy2(sidecar,root/sidecar.name)
    audit=[]
    for i,entry in enumerate(manifest):
        dest=root/f'episode_{i:06d}.npz'
        with np.load(entry['path'],allow_pickle=True) as d:
            phases=d['subtask_index'].reshape(-1);names=list(map(str,d['phase_names']))
            hold=np.flatnonzero(phases==names.index('hold_home'))
            assert len(hold)>50
            end=int(hold[0])+50 # keep two seconds of stable terminal hold
            n=len(phases)
            if not dest.exists():
                ki=np.asarray(d['keyframe_indices']);km=ki<end
                arrays={}
                for k in d.files:
                    value=d[k]
                    if value.ndim>0 and value.shape[0]==n:value=value[:end]
                    elif k in ('keyframe_indices','keyframe_names','keyframe_target_phase_names','keyframe_extraction_rules'):
                        value=value[km]
                    arrays[k]=value
                arrays['trim_source_episode']=np.asarray(entry['path'])
                arrays['trim_original_frames']=np.asarray(n)
                temp=dest.with_suffix('.partial.npz')
                np.savez_compressed(temp,**arrays);temp.replace(dest)
                del arrays
            with np.load(dest,allow_pickle=True) as new:
                np.testing.assert_array_equal(new['state'],d['state'][:end])
                np.testing.assert_array_equal(new['action'],d['action'][:end])
                assert len(new['state'])==end
            audit.append(dict(source=entry['path'],output=str(dest),old_frames=n,new_frames=end,
                              identical_retained_state_action=True))
        print(f'TRIM {i+1}/50 frames={end}',flush=True)
    save(OUT/'trimtail15.trim_manifest.json',audit)


def train_eval(name,gpu,env,root,repo_id):
    job=f'pi05evo_pull_{name}_rgb50_joint9_g080_seed1000_30k_0910'
    output=A/'dp/logs/lerobot-train-pi05'/job
    official=output/'checkpoints/030000/pretrained_model'
    launcher=[PY,'-u',str(REPO/'high-level/dp/train/run_native_pi05_evo.py')]
    if not (official/'model.safetensors').exists():
        status(name,'TRAIN30K',gpu=gpu,job=job)
        assert not output.exists(),f'Partial run requires verified resume: {output}'
        cmd=[PY,'-m','accelerate.commands.launch','--num_processes=1','--mixed_precision=bf16',
            REPO/'high-level/dp/train/run_native_pi05_evo.py',SRC/'lerobot/scripts/lerobot_train.py',
            f'--dataset.root={root}',f'--dataset.repo_id={repo_id}',
            '--dataset.video_backend=torchcodec','--dataset.use_imagenet_stats=false',
            '--policy.type=pi05_evo','--policy.device=cuda',
            f'--policy.pretrained_path={A}/dp/models/pi05_evo_native_base_0907',
            '--policy.dtype=bfloat16','--policy.push_to_hub=false','--policy.gradient_checkpointing=true',
            '--policy.chunk_size=100','--policy.n_action_steps=50','--policy.num_inference_steps=10',
            '--batch_size=16','--steps=30000','--num_workers=4','--save_freq=10000','--log_freq=50',
            '--seed=1000',f'--job_name={job}',f'--output_dir={output}',
            '--wandb.enable=false','--wandb.disable_artifact=true']
        run(cmd,env,name+'.train')
    cfg=json.loads((official/'config.json').read_text())
    assert cfg['type']=='pi05_evo' and not cfg.get('state_conditioning',False)
    assert cfg['input_features']['observation.state']['shape']==cfg['output_features']['action']['shape']==[9]
    wrap=A/'dp/logs/door-auto-wrapped/0910_pull_evo_trajectory'/job/'030000/model_latest'
    if not wrap.with_suffix('.pt').exists():
        status(name,'EXPORT')
        run(launcher+[REPO/'high-level/dp/export_official_lerobot_pi05_to_door_checkpoint.py',
            '--official_checkpoint',official,'--root',root,'--repo_id',repo_id,'--out_dir',wrap,
            '--device','cuda:0','--rgb','--action_horizon','25','--num_inference_steps','10'],env,name+'.export')
    status(name,'PARITY')
    run(launcher+[REPO/'high-level/dp/eval/verify_pi05_saved_processor_parity.py',
        '--official_checkpoint',official,'--door_checkpoint',wrap,'--dataset_root',root,'--repo_id',repo_id,
        '--device','cuda:0','--output',OUT/f'{name}.parity.json'],env,name+'.parity')
    ev=A/'logs/door-policy-success'/f'{job}_open60_traversal080_seed615455575'
    if not (ev/'summary.json').exists():
        assert not ev.exists(),f'Inspect partial evaluation {ev}'
        status(name,'EVAL64')
        old=json.loads((OLD/'pi05evo_native_pull_rgb50_joint9_g080_seed1000_30k_0907.eval.command.json').read_text())['argv']
        cmd=list(old)
        cmd[cmd.index('--checkpoint')+1]=str(wrap.with_suffix('.pt'))
        cmd[cmd.index('--run_root')+1]=str(ev)
        run(cmd,env,name+'.eval')
    result=json.loads((ev/'summary.json').read_text())
    assert result['trials']==64 and len(result['trials_detail'])==64
    assert result['pass_open_angle_deg']==60 and result['pull_traversal_distance_m']==.8
    save(OUT/f'{name}.result.json',result)
    status(name,'COMPLETE',opening=result['door_open_successes'],traversal=result['traversal_successes'],joint=result['successes'])


def lane(name,gpu):
    try:
        if (OUT/f'{name}.result.json').exists():return
        env=environment(gpu,name)
        raw=Path('/mnt/sda/ps/doorgym_data/door_dp_raw')/f'a2w_0910_pull_{name}_rgb50_joint9'
        save(OUT/f'{name}.experiment.json',dict(variant=name,gpu=gpu,trajectory=trajectory(name),
            policy='native hwk pi05_evo',train_seed=1000,steps=30000,eval_seed=615455575,
            trim_only=name=='trimtail15',raw=str(raw)))
        if name=='trimtail15':
            status(name,'TRIM_ORIGINAL50');trim_original(raw)
        else:
            smoke=raw.with_name(raw.name+'_smoke4')
            status(name,'SMOKE4');record(name,env,smoke,4,smoke=True);validate_raw(smoke,4,name)
            validation=re.findall(r'completed_traverse=(\d+)/(\d+)',(OUT/f'{name}.smoke.log').read_text())
            attempts=sum(int(total) for _,total in validation)
            successes=sum(int(good) for good,_ in validation)
            assert attempts>0 and successes/attempts>=.6, f'Expert smoke acceptance rate too low: {successes}/{attempts}'
            status(name,'RECORD50');record(name,env,raw,50)
        audit=validate_raw(raw,50,name)
        repo_id=f'local/door_a2w_0910_pull_{name}_rgb50_joint9'
        root=A/'data/lerobot'/repo_id
        if not (OUT/f'{name}.dataset_ready.json').exists():
            if root.exists():
                assert root.name.startswith('door_a2w_0910_pull_')
                failed=root.with_name(root.name+'.partial_'+time.strftime('%Y%m%d_%H%M%S'))
                root.rename(failed)
                save(OUT/f'{name}.conversion_retry.json',dict(preserved_partial=str(failed),
                    reason='Use established main LeRobot dataset writer; native Evo writer API is incompatible with shared recorder'))
            status(name,'CONVERT50')
            convert_env=env.copy()
            # Same main-fork writer as the original50/200/500 conversion. Native
            # Evo reads this format; its fork's writer differs in scalar handling.
            convert_env['PYTHONPATH']=f'{REPO}/high-level/lerobot/src:{REPO}/high-level'
            run([PY,'high-level/dp/convert_door_raw_to_lerobot.py','--raw_root',raw,
                '--root',A/'data/lerobot','--repo_id',repo_id,'--state_action_mode','joint_state9',
                '--rgb_only','--image_storage','video','--video_codec','h264','--num_workers','4'],convert_env,name+'.convert')
            info=json.loads((root/'meta/info.json').read_text())
            assert info['total_episodes']==50 and info['total_frames']==sum(x['frames'] for x in audit)
            assert info['features']['action']['shape']==info['features']['observation.state']['shape']==[9]
            save(OUT/f'{name}.dataset_ready.json',dict(root=str(root),repo_id=repo_id,info=info))
        if name=='trimtail15':
            status(name,'WAIT_VIDEO_GPU0')
            while not (OUT/'videos/complete.json').exists():time.sleep(60)
        train_eval(name,gpu,env,root,repo_id)
    except BaseException as exc:
        status(name,'FAILED',error=repr(exc));raise


def monitor():
    states=[json.loads(p.read_text()) for p in OUT.glob('*.status.json')]
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
    event=dict(time=time.strftime('%F %T'),states=states,gpu=gpu,hdd_free_gib=shutil.disk_usage(A).free/1024**3)
    with (OUT/'monitor_30min.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    print('MONITOR '+json.dumps(event),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--only',choices=list(VARIANTS));p.add_argument('--gpu',type=int,default=1)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/('pipeline.lock' if not args.only else args.only+'.lock')).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[(args.only,args.gpu)] if args.only else [('smooth15',1),('steady15',3),('trimtail15',0)]
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
        fs=[pool.submit(lane,name,gpu) for name,gpu in jobs]
        while True:
            monitor()
            done,pending=concurrent.futures.wait(fs,timeout=1800,return_when=concurrent.futures.ALL_COMPLETED)
            if not pending:
                for f in done:f.result()
                break
    if args.only:
        status(args.only+'.queue','LANE_TRAINED_AND_EVALUATED')
    else:
        status('pipeline','ALL_THREE_TRAINED_AND_EVALUATED')


if __name__=='__main__':main()
