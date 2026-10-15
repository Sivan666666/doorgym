"""Re-evaluate native Evo RGB500/30K from standard starts with a physical 25deg latch."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

from eval_door_policy_success import summarize_log

REPO=Path('/home/ps/workspace/txc/doorgym')
ARCHIVE=Path('/mnt/sda/ps_archive/doorgym/high-level')
MODEL='pi05evo_native_pull_rgb500_joint9_g080_seed1000_30k_0907'
SOURCE=ARCHIVE/f'logs/0907_pull_rgb_joint9_pi05_evo/{MODEL}.result.json'
OUT=ARCHIVE/'dp/result/pi05evo_rgb500_unlock25_open60_traverse050_20260910'


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def metrics(rows):
    n=len(rows)
    return dict(trials=n, **{k:sum(bool(r[k]) for r in rows) for k in
        ('door_open_success','traversal_success','success')})


def read_checked_log(path):
    first={};last={};counts={}
    with path.open() as f:
        for line in f:
            r=json.loads(line);i=int(r['controlled_env_id'])
            first.setdefault(i,r);last[i]=int(r['step']);counts[i]=counts.get(i,0)+1
            for k in ('state','dp_action_raw'):
                assert len(r[k])==9 and all(math.isfinite(x) for x in r[k]), (path,i,k)
    assert set(first)==set(range(16)) and set(last.values())=={2599}, (path,last)
    assert set(counts.values())=={2600}, (path,counts)
    return first,last


def report(rows,old,complete):
    lines=['# Native π0.5 Evo RGB500 / 30K: physical handle unlock 25°', '',
        f'Complete: {complete}. Standard starts, not expert handoff. Same 4×16 env seeds '
        '615455575–615455578, 2600 steps (52s), action horizon25, inference steps10, policy25Hz.', '',
        'Physical latch: `--handle_unlock_ratio=0.5555555555555556` (25/45). '
        '`--expected_handle_unlock_angle_deg=25` verifies each actual instantiated door threshold. '
        'Default recording/push rules and original checkpoints are unchanged.', '',
        'Success: same-frame absolute door angle>=60° AND signed base-plane progress>=0.5m. '
        'Distance-only crossing is not collision-validated traversal.', '',
        '| Physical handle unlock | Trials | Opening | Distance >=0.5m | Joint success |',
        '|---|---:|---:|---:|---:|']
    for label,rs in [('40° (historical logs rescored at0.5m)',old),('25° (new rollouts)',rows)]:
        n=len(rs);m=metrics(rs)
        vals=[f'{m[k]}/{n} = {100*m[k]/n:.2f}%' if n else 'Pending' for k in
              ('door_open_success','traversal_success','success')]
        lines.append(f'| {label} | {n}/64 | '+' | '.join(vals)+' |')
    lines+=['', 'The 40° comparator is the original standard-start64-trial evaluation, not '
        'the different expert-frame0 diagnostic. Model weights, image/state/task preprocessing and '
        'environment randomization settings are unchanged. Actual first-frame scene randomization '
        'and traversal geometry are checked against the corresponding historical batch.', '',
        'Policy diffusion noise has no additional separately fixed sampler seed in the inherited '
        'pipeline. Therefore one repeat cannot isolate the threshold effect from policy sampling variance.', '',
        'Failure classification uses25° in this report; it is separate from the physical latch change. '
        'No retraining or expert-state restoration is used. Complete logs, finite Joint9 values and '
        'runtime threshold checks are required; a known post-completion Isaac Gym cleanup SIGSEGV '
        'is recorded but incomplete rollouts are never counted.', '']
    (OUT/'README.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=json.loads(SOURCE.read_text());old=[];rows=[];batches=[]
    for b in src['batch_logs']:
        old+=summarize_log(Path(b['log_path']),16,60.,'abs',-1.,mode='ikpull',pull_traversal_distance_m=.5)
    assert len(old)==64
    save(OUT/'reference_40deg_rescored_050.json',dict(**metrics(old),trials_detail=old,source=str(SOURCE)))
    report(rows,old,False)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='2',PYTHONPATH=f'{REPO}/high-level/hwk/lerobot/src:{REPO}/high-level',
        DOOR_DP_LEROBOT_PYTHON='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
    for i,b in enumerate(src['batch_logs']):
        root=OUT/f'batch_{i:02d}';root.mkdir(exist_ok=True)
        result=root/'result.json'
        if result.exists():
            data=json.loads(result.read_text())
        else:
            cmd=list(b['command']);cmd[cmd.index('--dp_log_path')+1]=str(root/'rollout.jsonl')
            cmd+=['--handle_unlock_ratio',str(25./45.),'--expected_handle_unlock_angle_deg','25']
            save(root/'command.json',cmd)
            with (root/'stdout.log').open('w') as f:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
                save(OUT/'status.json',dict(state='running',batch=i,pid=child.pid,root=str(root),time=time.time()))
                print('RUN',i,flush=True)
                while child.poll() is None:
                    time.sleep(30)
                    with (OUT/'monitor.jsonl').open('a') as m:
                        m.write(json.dumps(dict(time=time.time(),batch=i,pid=child.pid,
                            log_bytes=(root/'rollout.jsonl').stat().st_size if (root/'rollout.jsonl').exists() else 0))+'\n')
            stdout=(root/'stdout.log').read_text()
            cleanup=child.returncode in (1,139,-11) and 'Done after 2600 steps' in stdout and 'SIGSEGV' in stdout
            assert child.returncode==0 or cleanup, (i,child.returncode,str(root))
            threshold_lines=[l for l in stdout.splitlines() if l.startswith('[PullUnlockAudit] ')]
            assert len(threshold_lines)==1
            threshold=json.loads(threshold_lines[0].split('] ',1)[1])
            assert len(threshold)==16 and all(abs(t['handle_unlock_angle_deg']-25)<1e-4 for t in threshold)
            first,last=read_checked_log(root/'rollout.jsonl')
            ref_first,_=read_checked_log(Path(b['log_path']))
            for eid in range(16):
                for k in ('ikpush_randomization','pull_task_geometry'):
                    assert first[eid][k]==ref_first[eid][k], (i,eid,k)
            trials=summarize_log(root/'rollout.jsonl',16,60.,'abs',-1.,mode='ikpull',pull_traversal_distance_m=.5)
            for t in trials:
                t['failure_category']=(None if t['success'] else 'opened_not_traversed' if t['door_open_success']
                    else 'grasp_failure' if t['max_handle_or_secondary_dof_deg']<25. else 'insufficient_opening')
                t.update(batch=i,seed=615455575+i)
            data=dict(**metrics(trials),trials_detail=trials,runtime_unlock_audit=threshold,
                first_scene_matches_reference=True,final_steps=last,returncode=child.returncode)
            save(result,data)
        rows+=data['trials_detail'];batches.append(data)
        save(OUT/'summary.json',dict(complete=len(rows)==64,checkpoint=src['checkpoint'],
            physical_unlock_deg=25.,opening_threshold_deg=60.,traversal_distance_m=.5,
            **metrics(rows),trials_detail=rows,batches=batches,reference40=metrics(old)))
        report(rows,old,len(rows)==64)
        print('DONE',i,metrics(data['trials_detail']),flush=True)
    save(OUT/'status.json',dict(state='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S')))


if __name__=='__main__':
    try: main()
    except Exception as e:
        save(OUT/'status.json',dict(state='error',error=repr(e)))
        raise
