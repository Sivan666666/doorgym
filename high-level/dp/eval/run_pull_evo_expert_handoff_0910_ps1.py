"""Matched expert-snapshot diagnostic and non-destructive 0.5m rescoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path('/home/ps/workspace/txc/doorgym')
ARCHIVE = Path('/mnt/sda/ps_archive/doorgym/high-level')
OLD = ARCHIVE/'logs/0907_pull_rgb_joint9_pi05_evo'
OUT = ARCHIVE/'dp/result/pull_evo_expert_handoff_20260910'
MODEL = 'pi05evo_native_pull_rgb500_joint9_g080_seed1000_30k_0907'
STAGES = ('start', 'pull_door', 'panel_contact', 'pass_through')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def summary(rows):
    return {k:sum(bool(r[k]) for r in rows) for k in
            ('door_open_success', 'traversal_success', 'success')} | {'trials':len(rows)}


def write_report(results, complete=False):
    names={'start':'Expert frame 0', 'pull_door':'Start pulling (after handle rotation)',
           'panel_contact':'Start panel contact (after release/reposition)',
           'pass_through':'Start traversal (door already open)'}
    lines=['# Native PI05 Evo: expert-state handoff diagnostic', '',
        'Complete: '+str(complete)+'. Model: `'+MODEL+'`.', '',
        'Success: at least one logged frame simultaneously has absolute door angle >=60 deg '
        'and signed base progress >=0.5 m beyond the door plane. Push rules unchanged.', '',
        '| Starting state | Completed / planned | Opening | Traversal | Joint success |',
        '|---|---:|---:|---:|---:|']
    for stage in STAGES:
        r=results.get(stage)
        if r:
            count=r['trials']
            cells=[f"{r[k]}/{count} = {100*r[k]/count:.2f}%" for k in
                   ('door_open_success','traversal_success','success')]
            lines.append('| '+names[stage]+f' | {count}/64 | '+' | '.join(cells)+' |')
        else:
            lines.append('| '+names[stage]+' | 0/64 | Pending | Pending | Pending |')
    lines += ['', '## Interpretation and limitations', '',
        '- The same 64 evenly spaced episode indices from the existing 500 training demonstrations '
        'are used in all conditions. These are seen-state diagnostics, not zero-shot or held-out success rates.',
        '- All conditions receive 2600 continuation steps (52 s), with the original policy, saved task '
        'processor, dual RGB, Joint9 input and Joint9 command output. No model retraining.',
        '- Initial door angle already exceeds 60 deg in 0/64 start, 0/64 pull, 35/64 panel-contact '
        'and 64/64 traversal snapshots. Later-stage opening SR is therefore not evidence of learning to open.',
        '- Robot and door positions/velocities, latch, previous command, randomized dynamics and '
        'camera attachment poses are restored. Every batch has a numerical restoration audit.',
        '- Original data do not serialize PhysX contact solver caches. State restoration does not '
        'guarantee bitwise reproduction of contact impulses. Raw first-frame RGB is not fed as future input.',
        '- Environment batch seeds: 615455575..615455578. Policy diffusion noise sampling follows '
        'the existing pipeline without adding a separate sampler seed; repeated actions need not be bitwise deterministic.',
        '- Original 0.8 m summaries are preserved. `historical_rescore_050.json` rescored the original '
        'logs, not a new rollout: RGB50 joint46/64 (71.88%), RGB200 joint46/64 (71.88%), '
        'RGB500 joint47/64 (73.44%). The disputed video passes at step1322 under 0.5 m.',
        '- Controller shutdown may emit SIGSEGV after a complete rollout. Results require all envs '
        'to reach step2599 with finite Joint9 states/actions and a successful restoration audit.', '']
    (OUT/'README.md').write_text('\n'.join(lines))


def rescore():
    from eval_door_policy_success import summarize_log
    output = {}
    for n in (50, 200, 500):
        model = f'pi05evo_native_pull_rgb{n}_joint9_g080_seed1000_30k_0907'
        original = json.loads((OLD/f'{model}.result.json').read_text())
        rows = []
        for batch in original['batch_logs']:
            r = summarize_log(Path(batch['log_path']), batch['envs'], 60., 'abs', -1.,
                              mode='ikpull', pull_traversal_distance_m=.5)
            for x in r:
                x.update(batch=batch['batch'], log_path=batch['log_path'])
            rows.extend(r)
        assert len(rows) == 64
        output[str(n)] = dict(**summary(rows), trials_detail=rows,
                             original_080_successes=original['successes'])
    video_log = ARCHIVE/'dp/result/pull_evo_trajectory_20260910/videos/attempt06_seed615455581.jsonl'
    video = summarize_log(video_log, 1, 60., 'abs', -1., mode='ikpull', pull_traversal_distance_m=.5)[0]
    save(OUT/'historical_rescore_050.json', dict(criterion='same-frame abs angle>=60deg AND base progress>=0.5m',
         models=output, disputed_video=video, historical_files_unchanged=True))
    print('RESCORE', {n:summary(v['trials_detail']) for n,v in output.items()}, flush=True)


def prepare():
    manifest = json.loads((OLD/'raw500_manifest.json').read_text())
    indices = np.linspace(0, 499, 64, dtype=int).tolist()
    entries = {s:[] for s in STAGES}
    frame_keys = ['state', 'replay_root_state', 'replay_dof_pos', 'replay_dof_vel',
        'replay_door_root_state', 'replay_door_dof_pos', 'replay_door_dof_vel',
        'replay_door_open_stage', 'replay_ee_pos', 'replay_ee_quat',
        'front_camera_pose_base', 'wrist_camera_pose_base']
    for index in indices:
        source = manifest[index]['path']
        with np.load(source, allow_pickle=True) as d:
            names = list(map(str, d['phase_names']))
            labels = d['subtask_index'].reshape(-1)
            sampled = json.loads(str(d['ikpush_randomization']))
            scene = {k:d[k].item() for k in ('door_x','door_y','door_z_offset','door_wall_x_offset',
                     'door_wall_y_offset','door_side_walls') if k in d.files}
            arrays = {k:d[k] for k in frame_keys}
            actions = d['action']
            for stage in STAGES:
                frame = 0 if stage == 'start' else int(np.flatnonzero(labels == names.index(stage))[0])
                snapshot = OUT/'snapshots'/f'episode{index:03d}_{stage}.npz'
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                selected = {k:v[frame:frame+1] for k,v in arrays.items()}
                selected['action'] = actions[max(0, frame-1):max(0, frame-1)+1].copy()
                selected['action'][0,:2] = selected['state'][0,:2]
                np.savez(snapshot, **selected)
                entries[stage].append(dict(episode=index, source=source, source_frame=frame,
                    source_sim_step=frame*int(d['record_sample_stride']), phase=stage,
                    source_time_s=frame/float(d['fps']), snapshot=str(snapshot),
                    randomization=sampled, scene=scene, initial_open_deg=float(abs(np.rad2deg(selected['replay_door_dof_pos'][0,0])))))
        print('PREPARE', index, flush=True)
    save(OUT/'design.json', dict(model=MODEL, indices=indices, stages=STAGES, entries=entries,
        criterion='same-frame abs angle>=60deg AND base progress>=0.5m',
        continuation_steps=2600, continuation_seconds=52, policy_seed='existing pipeline: env seeded; no extra sampler seed',
        notes=['Seen training expert snapshots; conditional diagnostic, not held-out/generalization SR.',
               'Same 64 episodes in every stage; first occurrence of each phase, not picked by outcome.',
               'Full 52s continuation for every stage; more total elapsed task time than from-start evaluation.',
               'Physics contact solver caches are unavailable in original NPZ.',
               'No future expert commands or expert images fed to the policy.']))


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=f'{REPO}/high-level/hwk/lerobot/src:{REPO}/high-level',
        DOOR_DP_LEROBOT_PYTHON='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4')
    return env


def run_batch(stage, batch, entries, gpu, steps=2600, smoke=False):
    from eval_door_policy_success import summarize_log
    name = f'{stage}_batch{batch:02d}' + ('_smoke' if smoke else '')
    root = OUT/name
    root.mkdir(parents=True, exist_ok=True)
    result = root/'result.json'
    if result.exists():
        return json.loads(result.read_text())
    mf = root/'manifest.json'
    save(mf, dict(entries=entries, restore_audit=str(root/'restore_audit.json')))
    old = json.loads((OLD/f'{MODEL}.result.json').read_text())
    cmd = list(old['batch_logs'][0]['command'])
    for flag, value in [('--num_envs',len(entries)), ('--steps',steps),
                        ('--dp_log_path',root/'rollout.jsonl'), ('--seed',615455575+batch),
                        ('--dp_log_interval',1)]:
        cmd[cmd.index(flag)+1] = str(value)
    cmd += ['--dp_expert_handoff_manifest', str(mf)]
    save(root/'command.json', cmd)
    print('RUN',name,flush=True)
    started = time.time()
    with (root/'stdout.log').open('w') as log:
        child = subprocess.Popen(cmd,cwd=REPO,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
        save(OUT/'status.json', dict(state='running',stage=stage,batch=batch,pid=child.pid,
             root=str(root),started=time.strftime('%Y-%m-%d %H:%M:%S')))
        while child.poll() is None:
            time.sleep(30)
            with (OUT/'monitor.jsonl').open('a') as monitor:
                monitor.write(json.dumps(dict(time=time.time(),pid=child.pid,stage=stage,batch=batch,
                    log_bytes=(root/'rollout.jsonl').stat().st_size if (root/'rollout.jsonl').exists() else 0))+'\n')
    if not (root/'restore_audit.json').exists():
        raise RuntimeError(f'Restore failed; inspect {root}/stdout.log')
    rows = summarize_log(root/'rollout.jsonl',len(entries),60.,'abs',-1.,mode='ikpull',pull_traversal_distance_m=.5)
    last = {i:-1 for i in range(len(entries))}
    with (root/'rollout.jsonl').open() as log:
        for line in log:
            r=json.loads(line); last[int(r['controlled_env_id'])]=int(r['step'])
            for key in ('state','dp_action_raw'):
                assert len(r[key])==9 and np.isfinite(r[key]).all(), (name,key,r['step'])
    tail=(root/'stdout.log').read_text()[-12000:]
    cleanup_segv=(child.returncode in (1,139,-11) and f'Done after {steps} steps' in tail
                  and ('SIGSEGV' in tail or child.returncode in (139,-11)))
    if any(v != steps-1 for v in last.values()) or not (child.returncode==0 or cleanup_segv):
        raise RuntimeError(f'Incomplete rollout {name}: rc={child.returncode}, last={last}')
    for row, entry in zip(rows,entries):
        row.update(episode=entry['episode'],source_frame=entry['source_frame'],phase=stage)
    data=dict(**summary(rows), trials_detail=rows,elapsed_s=time.time()-started,returncode=child.returncode,
        final_steps=last, log_path=str(root/'rollout.jsonl'), smoke_only=smoke)
    save(result,data)
    print('DONE',name,summary(rows),flush=True)
    return data


def run(gpu, smoke):
    design = json.loads((OUT/'design.json').read_text())
    if smoke:
        # Include a grasped state and an already released state in the same sim.
        selected = [design['entries'][s][0] for s in STAGES]
        return run_batch('mixed',0,selected,gpu,steps=50,smoke=True)
    results={}
    write_report(results)
    for stage in STAGES:
        rows=[]
        for batch in range(4):
            entries=design['entries'][stage][batch*16:(batch+1)*16]
            rows.extend(run_batch(stage,batch,entries,gpu)['trials_detail'])
            results[stage]=dict(**summary(rows),trials_detail=rows)
            save(OUT/'summary.json',dict(complete=False,conditions=results,design=str(OUT/'design.json')))
            write_report(results)
    save(OUT/'summary.json',dict(complete=True,conditions=results,design=str(OUT/'design.json')))
    write_report(results,complete=True)
    save(OUT/'status.json',dict(state='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S')))


def analyze():
    design=json.loads((OUT/'design.json').read_text())
    aggregate={}
    rows_by_stage={}
    for stage in STAGES:
        rows=[];audits=[]
        for batch in range(4):
            root=OUT/f'{stage}_batch{batch:02d}'
            data=json.loads((root/'result.json').read_text())
            assert data['trials']==16 and set(data['final_steps'].values())=={2599}
            rows+=data['trials_detail']
            audits+=json.loads((root/'restore_audit.json').read_text())
        assert [r['episode'] for r in rows]==design['indices']
        adjusted=[]
        for row,entry in zip(rows,design['entries'][stage]):
            deadline=2599-entry['source_sim_step']
            adjusted.append({k:row[step] is not None and row[step]<=deadline for k,step in
                [('door_open_success','first_door_open_step'),('traversal_success','first_traversal_step'),
                 ('success','first_success_step')]})
        failures=[r for r in rows if not r['success']]
        aggregate[stage]=dict(full_continuation=summary(rows),
            original_52s_total_budget=summary(adjusted),
            failure_open_angles_deg=[r['max_open_deg'] for r in failures],
            failed_episodes=[r['episode'] for r in failures],
            max_restore_errors={k:max(a[k] for a in audits) for k in
                ('state_error','door_error','ee_error_m','front_pose_error','wrist_pose_error')},
            initial_open_ge60=sum(e['initial_open_deg']>=60 for e in design['entries'][stage]))
        rows_by_stage[stage]=rows
    for stage in STAGES[1:]:
        pairs=list(zip(rows_by_stage['start'],rows_by_stage[stage]))
        aggregate[stage]['paired_vs_start']=dict(
            gains=sum(not a['success'] and b['success'] for a,b in pairs),
            losses=sum(a['success'] and not b['success'] for a,b in pairs))
    save(OUT/'analysis.json',aggregate)
    write_report({s:dict(**summary(rows),trials_detail=rows) for s,rows in rows_by_stage.items()},True)
    lines=['', '## Completed audit and time-budget sensitivity', '',
        '| Stage | Full 52s continuation joint SR | Remaining original 52s budget joint SR |',
        '|---|---:|---:|']
    for s,a in aggregate.items():
        n=a['full_continuation']['success'];m=a['original_52s_total_budget']['success']
        lines.append(f'| {s} | {n}/64 = {n/64*100:.2f}% | {m}/64 = {m/64*100:.2f}% |')
    lines += ['', 'For the sensitivity check, success must occur by simulation step '
              '`2599 - source_sim_step` after handoff. This avoids crediting additional task time '
              'to later starts. Source time is based on the recorded sample cadence.', '',
              'The first start batch has stale pre-restore friction values in its JSONL randomization '
              'metadata. The executed handoff restored the recorded physical DOF properties after '
              'asset creation; subsequent batches also directly audit those properties and log '
              'the corrected metadata. No historical logs were rewritten.', '',
              'See `analysis.json` for failed episode IDs, maximum opening angles, paired gains/losses '
              'and restoration errors. Do not interpret seen-expert-state conditional SR as full-task '
              'policy performance or proof that a single trajectory parameter causes failures.', '']
    with (OUT/'README.md').open('a') as f: f.write('\n'.join(lines))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','rescore','smoke','run','pipeline','analyze']);p.add_argument('--gpu',type=int,default=2)
    a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        if a.action=='prepare': prepare()
        elif a.action=='rescore': rescore()
        elif a.action=='analyze': analyze()
        elif a.action=='pipeline':
            run(a.gpu,True)
            run(a.gpu,False)
        else: run(a.gpu,a.action=='smoke')
    except Exception as e:
        save(OUT/'status.json',dict(state='error',error=repr(e),action=a.action))
        raise
