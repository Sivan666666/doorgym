"""Read-only source-data audit and independently labelled native-Evo videos on PS1."""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path('/home/ps/workspace/txc/doorgym')
ARCHIVE = Path('/mnt/sda/ps_archive/doorgym/high-level')
OLD = ARCHIVE / 'logs/0907_pull_rgb_joint9_pi05_evo'
OUT = ARCHIVE / 'dp/result/pull_evo_trajectory_20260910'
NATIVE = 'pi05evo_native_pull_rgb500_joint9_g080_seed1000_30k_0907'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def describe(x):
    a = np.asarray(x, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()), min=float(a.min()))


def audit():
    manifest = json.loads((OLD / 'raw500_manifest.json').read_text())
    rows = []
    for i, entry in enumerate(manifest):
        with np.load(entry['path'], allow_pickle=True) as d:
            state, action = d['state'], d['action']
            names = list(map(str, d['phase_names']))
            phases = np.asarray(d['subtask_index']).reshape(-1)
            dt = 1 / float(d['fps'])
            door = np.abs(np.rad2deg(d['replay_door_dof_pos'][:, 0]))
            contact = d['gripper_handle_contact_both'].reshape(-1) > .5
            ee = d['replay_ee_pos']
            target = d['scripted_target_ee_pos_world']
            boundaries = np.flatnonzero(np.diff(phases) != 0) + 1
            phase_stats = {}
            for p in np.unique(phases):
                mask = phases == p
                ids = np.flatnonzero(mask)
                internal = ids[1:][np.diff(ids) == 1]
                phase_stats[names[int(p)]] = dict(frames=int(mask.sum()),
                    start=int(ids[0]), end=int(ids[-1]), seconds=float(mask.sum()*dt),
                    door_start=float(door[ids[0]]), door_end=float(door[ids[-1]]),
                    door_max=float(door[mask].max()), contact_fraction=float(contact[mask].mean()),
                    gripper_command_median=float(np.median(action[mask, -1])),
                    gripper_state_median=float(np.median(state[mask, -1])),
                    joint_target_error_p95=float(np.percentile(np.abs(action[mask, 2:8]-state[mask, 2:8]),95)),
                    ee_target_error_p95=float(np.percentile(np.linalg.norm(target[mask]-ee[mask],axis=1),95)),
                    max_arm_action_jump=float(np.abs(action[internal,2:8]-action[internal-1,2:8]).max()) if len(internal) else 0.,
                    stationary_fraction=float((np.linalg.norm(ee[internal]-ee[internal-1],axis=1)<.0005).mean()) if len(internal) else 0.)
            rows.append(dict(index=i, path=entry['path'], phase=phase_stats,
                finite=bool(np.isfinite(state).all() and np.isfinite(action).all()),
                state_action_shape=list(state.shape), task=str(d['task']),
                state_source=str(d['state_source']), action_source=str(d['action_source']),
                aligned=bool(d['pi05_state_action_aligned']),
                joint_state_vs_replay_max_error=float(np.max(np.abs(state[:,2:]-d['replay_dof_pos'][:,-7:]))),
                arm_command_differs_from_state_fraction=float(np.mean(np.abs(action[:,2:8]-state[:,2:8])>.01)),
                boundaries=[dict(frame=int(t), before=names[int(phases[t-1])], after=names[int(phases[t])],
                    max_arm_command_jump=float(np.abs(action[t,2:8]-action[t-1,2:8]).max()),
                    gripper_jump=float(action[t,-1]-action[t-1,-1])) for t in boundaries],
                opened=float(door.max()), traversal=float(d['pull_max_traversal_m'])))
        if (i+1) % 50 == 0:
            print(f'AUDIT {i+1}/500', flush=True)
    save(OUT/'raw_episode_audit.json', rows)
    summaries = {}
    for n in (50,200,500):
        selected = rows[:n]
        phase_names = sorted(set(k for r in selected for k in r['phase']))
        summary = {}
        for phase in phase_names:
            ps = [r['phase'][phase] for r in selected if phase in r['phase']]
            summary[phase] = dict(episodes=len(ps), **{k:describe([p[k] for p in ps]) for k in
                ('frames','start','seconds','contact_fraction','gripper_command_median','gripper_state_median',
                 'joint_target_error_p95','ee_target_error_p95','max_arm_action_jump','stationary_fraction')})
        summaries[n] = dict(phase=summary, frames=sum(sum(p['frames'] for p in r['phase'].values()) for r in selected),
                           max_boundary_arm_jump=describe([b['max_arm_command_jump'] for r in selected for b in r['boundaries']]))
    save(OUT/'raw_phase_summary.json', summaries)
    evals = {}
    for n in (50,200,500):
        d=json.loads((OLD/f'pi05evo_native_pull_rgb{n}_joint9_g080_seed1000_30k_0907.result.json').read_text())
        trials=d['trials_detail']
        evals[n]=dict(opening=d['door_open_successes'],traversal=d['traversal_successes'],joint=d['successes'],
            failures=d['failure_categories'],
            fail_angle=describe([t['max_open_deg'] for t in trials if not t['success']]),
            opening_time_success=describe([t['first_door_open_step']/50 for t in trials if t['door_open_success']]),
            joint_flags=[int(t['success']) for t in trials])
    save(OUT/'eval_comparison.json',evals)
    print(json.dumps({n:{p:round(v['seconds']['mean'],2) for p,v in s['phase'].items()} for n,s in summaries.items()},indent=2),flush=True)


def video(gpu):
    # Keep all trial attempts; selection is for visualization, NOT a new SR estimate.
    sys.path.insert(0,str(REPO/'high-level/dp/eval'))
    from eval_door_policy_success import summarize_log
    original=json.loads((OLD/f'{NATIVE}.result.json').read_text())
    env=os.environ.copy()
    tmp=Path('/mnt/sda/ps/dp_tmp0910/video');tmp.mkdir(parents=True,exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),DISPLAY=':1',XAUTHORITY='/run/user/1000/gdm/Xauthority',
        PYTHONPATH=f'{REPO}/high-level/hwk/lerobot/src:{REPO}/high-level',TMPDIR=str(tmp),
        DOOR_DP_LEROBOT_PYTHON='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
    videos=OUT/'videos';videos.mkdir(parents=True,exist_ok=True)
    accepted=[]
    counts=collections.Counter()
    for attempt in range(32):
        if counts['success']>=3 and counts['failure']>=3:break
        seed=615455575+attempt
        stem=videos/f'attempt{attempt:02d}_seed{seed}'
        meta=stem.with_suffix('.json')
        if meta.exists():
            result=json.loads(meta.read_text())
        else:
            cmd=list(original['batch_logs'][0]['command'])
            cmd.remove('--headless')
            for flag,value in [('--num_envs','1'),('--dp_log_path',str(stem.with_suffix('.jsonl'))),('--seed',str(seed))]:
                cmd[cmd.index(flag)+1]=value
            cmd+=['--overview_video_path',str(stem.with_suffix('.mp4')),
                  '--overview_video_fps','25','--overview_video_capture_stride','2',
                  '--no_draw_scripted_trajectory','--no_preview_trajectory_at_spawn']
            save(stem.with_suffix('.command.json'),cmd)
            with stem.with_suffix('.log').open('w') as log:
                run=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            if run.returncode != 0:
                # Isaac Gym sometimes exits139 after successful cleanup; never accept incomplete recordings.
                tail=stem.with_suffix('.log').read_text()[-5000:]
                assert 'frames=1300' in tail and run.returncode in (-11,139), tail
            result=summarize_log(stem.with_suffix('.jsonl'),1,60.,'abs',-1.,mode='ikpull',pull_traversal_distance_m=.8)[0]
            assert result['records']>=100 and stem.with_suffix('.mp4').stat().st_size>100000
            result.update(seed=seed,checkpoint=original['checkpoint'],attempt=attempt,
                          video=str(stem.with_suffix('.mp4')),criterion='opening>=60deg AND traversal>=0.8m')
            save(meta,result)
        category='success' if result['success'] else 'failure'
        if counts[category]<3:
            counts[category]+=1
            dest=videos/f'{category}_{counts[category]:02d}_seed{seed}.mp4'
            if not dest.exists():
                import imageio_ffmpeg
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',result['video'],
                    '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],
                    check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            accepted.append(dict(category=category,path=str(dest),metrics=result))
            save(videos/'selected.json',accepted)
        print(f'VIDEO {attempt} {category} {dict(counts)}',flush=True)
    assert counts['success']==counts['failure']==3,counts
    save(videos/'complete.json',dict(complete=True,selected=accepted))


def compare_push():
    root=Path('/mnt/sda/ps/doorgym_data/door_dp_raw/a2w_wc4_jointstate9_gymjacobian_realK_render60_200')
    values=collections.defaultdict(list);records=[]
    for path in sorted(root.glob('episode_*.npz')):
        with np.load(path,allow_pickle=True) as d:
            n=len(d['state']);fps=float(d['fps']);phases=d['subtask_index'].reshape(-1)
            names=list(map(str,d['phase_names']))
            durations={k:float((phases==i).sum()/fps) for i,k in enumerate(names)}
            for k,v in durations.items():values[k].append(v)
            records.append(dict(path=str(path),frames=n,seconds=n/fps,phase_seconds=durations))
    assert len(records)==200
    save(OUT/'push200_phase_audit.json',dict(episodes=records,phase_seconds={k:describe(v) for k,v in values.items()}))
    comparison={}
    for n in (50,200,500):
        repo_id=(f'local/door_a2w_0829_wc_nomenkuang_fixed15_gripper080_rgb_v1_{n}_joint9' if n<500 else
                 'local/door_a2w_0907_wc_nomenkuang_fixed15_gripper080_rgb_500_joint9')
        stats=json.loads((ARCHIVE/'data/lerobot'/repo_id/'meta/stats.json').read_text())
        comparison[n]={key:{k:stats[key].get(k) for k in ('min','max','q01','q99','mean','std')} for key in ('observation.state','action')}
    save(OUT/'dataset_normalization_comparison.json',comparison)
    r=json.loads((OUT/'eval_comparison.json').read_text())
    tests={}
    from scipy.stats import binomtest
    for a,b in [('50','200'),('50','500'),('200','500')]:
        x=np.array(r[a]['joint_flags']);y=np.array(r[b]['joint_flags'])
        gain=int(((x==0)&(y==1)).sum());loss=int(((x==1)&(y==0)).sum())
        tests[f'{a}_vs_{b}']=dict(paired_gain=gain,paired_loss=loss,
            exact_mcnemar_p=float(binomtest(min(gain,loss),gain+loss,.5).pvalue) if gain+loss else 1.)
    save(OUT/'paired_evaluation_tests.json',tests)
    print(json.dumps(dict(push_mean_phase_seconds={k:float(np.mean(v)) for k,v in values.items()},paired_tests=tests),indent=2))


def audit_rgb():
    """Inspect actual saved observations at phase transitions across all ten shards."""
    import cv2
    manifest=json.loads((OLD/'raw500_manifest.json').read_text())
    report=[]; images=OUT/'raw_rgb_checks';images.mkdir(exist_ok=True)
    for index in [0,49,50,149,199,200,249,299,349,399,449,499]:
        with np.load(manifest[index]['path'],allow_pickle=True) as d:
            phases=d['subtask_index'].reshape(-1);names=list(map(str,d['phase_names']))
            samples=[]
            for phase in ['initial_hold','grasp','rotate_handle','pull_door','release_handle',
                          'panel_reposition','panel_contact','panel_push','pass_through','hold_home']:
                ids=np.flatnonzero(phases==names.index(phase))
                if len(ids):samples.append((phase,int(ids[len(ids)//2])))
            grids=[];stats={}
            for key in ['front_rgb','wrist_rgb']:
                rgb=d[key]
                assert rgb.shape==(1300,480,640,3) and rgb.dtype==np.uint8
                row=[];stats[key]=[]
                for phase,frame in samples:
                    a=rgb[frame]
                    stats[key].append(dict(phase=phase,frame=frame,mean=float(a.mean()),std=float(a.std()),
                                           low_texture=bool(float(a.std())<=2.),
                                           all_black_fraction=float(np.mean(np.max(a,axis=2)==0))))
                    thumb=cv2.resize(cv2.cvtColor(a,cv2.COLOR_RGB2BGR),(192,144))
                    thumb=cv2.copyMakeBorder(thumb,20,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                    cv2.putText(thumb,phase,(3,14),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1)
                    row.append(thumb)
                grids.append(np.concatenate(row,axis=1))
                spatial=rgb[:,::16,::16].astype(np.float32)
                frame_std=spatial.reshape(len(rgb),-1).std(axis=1)
                frame_mean=spatial.reshape(len(rgb),-1).mean(axis=1)
                stats[key+'_all_frame_low_texture_by_phase']={phase:dict(
                    frames=int(np.sum(phases==pid)),
                    low_texture=int(np.sum((phases==pid)&(frame_std<=2.))),
                    fully_black=int(np.sum((phases==pid)&(frame_mean==0.))))
                    for pid,phase in enumerate(names) if np.any(phases==pid)}
                del rgb,spatial
            cv2.imwrite(str(images/f'episode{index:03d}_front_top_wrist_bottom.jpg'),np.concatenate(grids,axis=0))
            report.append(dict(index=index,path=manifest[index]['path'],stats=stats,
                intrinsics=d['camera_intrinsics'].item(),
                rgb_masks_included=bool(d['rgb_masks_included']) if 'rgb_masks_included' in d else None))
        print(f'RGB_AUDIT {index}',flush=True)
    save(OUT/'sampled_rgb_audit.json',dict(structural_checks_passed=True,episodes=report,
        note='Sampled transition frames and spatially subsampled per-frame texture in12 episodes across ten shards. Low texture is an observation, not proof of corrupt rendering.'))


def transitions():
    manifest=json.loads((OLD/'raw500_manifest.json').read_text())
    sets={f'original{n}':[Path(x['path']) for x in manifest[:n]] for n in (50,200,500)}
    for name in ('smooth15','steady15'):
        sets[name]=sorted(Path(f'/mnt/sda/ps/doorgym_data/door_dp_raw/a2w_0910_pull_{name}_rgb50_joint9').glob('episode_*.npz'))
        assert len(sets[name])==50
    report={}
    for name,paths in sets.items():
        events={60:collections.Counter(),80:collections.Counter()};frames=collections.Counter();jumps=[];rows=[]
        for path in paths:
            with np.load(path,allow_pickle=True) as d:
                phases=d['subtask_index'].reshape(-1);names=list(map(str,d['phase_names']))
                angles=np.abs(np.rad2deg(d['replay_door_dof_pos'][:,0]))
                row={}
                for angle in (60,80):
                    hit=np.flatnonzero(angles>=angle)
                    phase=names[int(phases[int(hit[0])])] if len(hit) else 'not_reached'
                    events[angle][phase]+=1;row[f'first{angle}_phase']=phase
                for i,k in enumerate(names):frames[k]+=int(np.sum(phases==i))
                a=d['action'];js=np.abs(np.diff(a[:,2:8],axis=0)).max(axis=1)
                jump=float(js.max());jumps.append(jump)
                rows.append(dict(path=str(path),max_arm_jump=jump,**row))
        report[name]=dict(episodes=len(paths),opening_phase_histograms=events,
            phase_mean_seconds={k:float(v/len(paths)/25) for k,v in frames.items()},
            max_arm_jump=describe(jumps),rows=rows)
    save(OUT/'opening_transition_audit.json',report)
    print(json.dumps({k:{'opening':v['opening_phase_histograms'],'arm_jump':v['max_arm_jump']} for k,v in report.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','video','compare_push','audit_rgb','transitions']);p.add_argument('--gpu',default='0')
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='audit':audit()
    elif a.mode=='compare_push':compare_push()
    elif a.mode=='audit_rgb':audit_rgb()
    elif a.mode=='transitions':transitions()
    else:video(a.gpu)
