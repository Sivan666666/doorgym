"""Paired augmented-RGB100 evaluation: physical unlock25, traverse0.5m.

Replay the prior64 scene seeds, not prior actions. No retraining or changes to
model, images, state, prompt, trajectory duration or policy action settings.
"""
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import run_pull_rgb_aug100_eval64_0913_ps1 as p
import audit_pull_rgb_aug100_eval64_0913_ps1 as audit
from eval_door_policy_success import summarize_log

REFERENCE=p.OUT
OUT=p.A/'dp/result/0913_pull_rgb_aug100_unlock25_traverse050_eval64'
RUNS=p.A/'logs/door-policy-success/0913_pull_rgb_aug100_unlock25_traverse050_eval64'


def command(i):
    old=json.loads((REFERENCE/'planned_trials.json').read_text())[i]
    cmd=list(old)
    replacements={
        '--run_root':str(RUNS/f'trial_{i:03d}'),
        '--rgb_visual_config':str(OUT/'rgb_config.yaml'),
        '--rgb_visual_preview_dir':str(OUT/'previews'/f'trial_{i:03d}'),
        '--pull_traversal_distance_m':'0.5',
        '--expected_handle_unlock_angle_deg':'25',
    }
    for flag,value in replacements.items():cmd[cmd.index(flag)+1]=value
    # Physical latch change, distinct from scoring/failure-label thresholds.
    assert '--handle_unlock_ratio' not in old and '--pull_handle_unlock_angle_deg' not in old
    cmd += ['--handle_unlock_ratio',str(25./45.)]
    index=cmd.index('--');cmd[index:index]=['--pull_handle_unlock_angle_deg','25']
    # Exact argv comparison: no hidden changes beyond requested thresholds/paths.
    restored=list(cmd)
    for flag in ('--handle_unlock_ratio','--pull_handle_unlock_angle_deg'):
        index=restored.index(flag);del restored[index:index+2]
    for flag in replacements:restored[restored.index(flag)+1]=old[old.index(flag)+1]
    assert restored==old
    return cmd


def paired_scene_audit(i):
    old=json.loads((REFERENCE/f'trial_{i:03d}.result.json').read_text())
    new_summary=json.loads((RUNS/f'trial_{i:03d}'/'summary.json').read_text())
    assert new_summary['pull_handle_unlock_angle_deg']==25
    new_log=Path(new_summary['batch_logs'][0]['log_path'])
    with Path(old['log_path']).open() as f:old_first=json.loads(next(f))
    with new_log.open() as f:new_first=json.loads(next(f))
    for key in ('ikpush_randomization','pull_task_geometry'):
        assert old_first[key]==new_first[key],(i,key)
    def first_scene(root):
        return json.loads(sorted((root/'previews'/f'trial_{i:03d}').glob('env00_front_*.json'))[0].read_text())
    a,b=first_scene(REFERENCE),first_scene(OUT)
    for key in ('metadata','parameters','intensity','hdr'):
        assert a[key]==b[key],(i,key)
    for key in ('seed','env_id','materials','indoor_room'):
        assert a['episode'][key]==b['episode'][key],(i,key)
    return dict(physical_start_randomization_matches=True,scene_geometry_matches=True,
        visual_configuration_and_seed_match=True)


def lane(j,rows):
    try:
        env=p.environment(p.GPUS[j])
        for i in range(j,64,len(p.GPUS)):
            if p.STOP.is_set():return
            p.run(command(i),env,f'trial_{i:03d}',RUNS/f'trial_{i:03d}')
            row=p.validate_trial(i)
            row['global_trial']=i
            row['paired_scene_audit']=paired_scene_audit(i)
            p.save(OUT/f'trial_{i:03d}.result.json',row)
            with p.LOCK:rows.append(row);p.report(rows)
    except BaseException as exc:
        p.STOP.set();p.save(OUT/f'lane_{j}.error.json',dict(error=repr(exc),time=time.strftime('%F %T')))
        raise


def reference_rescore():
    old=json.loads((REFERENCE/'summary.json').read_text());rows=[]
    assert old['complete'] and old['trials']==64
    for original in old['trials_detail']:
        row=summarize_log(Path(original['log_path']),1,60.,'abs',-1.,mode='ikpull',pull_traversal_distance_m=.5)[0]
        assert row['records']==2600
        rows.append(dict(**row,trial_id=original['trial_id'],log_path=original['log_path']))
    totals={key:sum(bool(r[key]) for r in rows) for key in ('door_open_success','traversal_success','success')}
    p.save(OUT/'reference_40deg_050_rescored.json',dict(trials=64,physical_unlock_deg=40,traversal_m=.5,
        source=str(REFERENCE/'summary.json'),counts=totals,trials_detail=rows))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists() and not RUNS.exists(),'Preserve existing results'
    p.OUT=OUT;p.RUNS=RUNS;p.UNLOCK_DEG=25;p.TRAVERSAL_M=.5
    try:
        assert json.loads((REFERENCE/'final_audit.json').read_text())['passed']
        assert json.loads((REFERENCE/'parity.json').read_text())['passed']
        shutil.copy2(REFERENCE/'rgb_config.yaml',OUT/'rgb_config.yaml')
        for src in (Path(__file__),Path(p.__file__),Path(audit.__file__)):
            shutil.copy2(src,OUT/src.name)
        old_protocol=json.loads((REFERENCE/'protocol.json').read_text())
        protocol={**old_protocol,'physical_unlock_deg':25,'traversal_distance_m':.5,
            'paired_reference':str(REFERENCE),'physical_handle_unlock_ratio':25./45.,
            'caveat':'Same scene seeds and model as augmented-RGB100 reference. Policy diffusion RNG remains unfixed as in that run; not a fully deterministic paired-policy-noise experiment. 100 training episodes still differs from historical RGB500 result.'}
        p.save(OUT/'protocol.json',protocol)
        p.save(OUT/'planned_trials.json',[command(i) for i in range(64)])
        p.status('PARITY_AND_REFERENCE_RESCORE')
        reference_rescore()
        env=p.environment(p.GPUS[0])
        p.run([p.PY,p.REPO/'high-level/dp/train/run_native_pi05_evo.py',
            p.REPO/'high-level/dp/eval/verify_pi05_saved_processor_parity.py',
            '--official_checkpoint',p.OFFICIAL,'--door_checkpoint',p.WRAP,
            '--dataset_root',p.DATA,'--repo_id',p.RID,'--device','cuda:0','--output',OUT/'parity.json'],env,'parity')
        assert json.loads((OUT/'parity.json').read_text())['passed']
        rows=[];p.report(rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(p.GPUS)) as pool:
            futures=[pool.submit(lane,j,rows) for j in range(len(p.GPUS))]
            for future in concurrent.futures.as_completed(futures):future.result()
        assert len(rows)==64
        p.status('FINAL_AUDIT',completed_trials=64)
        audit.main(out=OUT,traversal_m=.5)
        p.report(rows)
    except BaseException as exc:
        p.STOP.set();p.status('FAILED',error=repr(exc));raise


if __name__=='__main__':main()
