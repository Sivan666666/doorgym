"""Automatically evaluate native Evo RGB500/30K on the RGB100 paired64 benchmark."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'train'))
import run_pull_rgb_aug500_pi05_0914_ps1 as t
import run_pull_rgb_aug100_eval64_0913_ps1 as p
import run_pull_rgb_aug100_unlock25_050_0913_ps1 as paired
import audit_pull_rgb_aug100_eval64_0913_ps1 as audit

REFERENCE=t.A/'dp/result/0913_pull_rgb_aug100_unlock25_traverse050_eval64'
OUT=t.EVAL_OUT
RUNS=t.A/'logs/door-policy-success/0914_pull_rgb_aug500_unlock25_traverse050_eval64'
WRAP=t.A/'dp/logs/door-auto-wrapped/0914_pull_rgb_aug500_eval/030000/model_latest'


def command(i):
    old=json.loads((REFERENCE/'planned_trials.json').read_text())[i]
    cmd=list(old)
    replacements={'--checkpoint':str(WRAP.with_suffix('.pt')),
        '--run_root':str(RUNS/f'trial_{i:03d}'),'--rgb_visual_config':str(OUT/'rgb_config.yaml'),
        '--rgb_visual_preview_dir':str(OUT/'previews'/f'trial_{i:03d}')}
    for flag,value in replacements.items():cmd[cmd.index(flag)+1]=value
    restored=list(cmd)
    for flag in replacements:restored[restored.index(flag)+1]=old[old.index(flag)+1]
    assert restored==old
    assert cmd[cmd.index('--expected_handle_unlock_angle_deg')+1]=='25'
    assert float(cmd[cmd.index('--pull_traversal_distance_m')+1])==.5
    assert '--expert_action_replay_raw_episode' not in cmd
    return cmd


def environment(gpu):
    env=p.environment(gpu);tmp=Path('/mnt/sda/ps/tmp/eval500');tmp.mkdir(parents=True,exist_ok=True)
    env.update(TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp))
    return env


def lane(j,rows):
    try:
        env=environment(p.GPUS[j])
        for i in range(j,64,len(p.GPUS)):
            if p.STOP.is_set():return
            result=OUT/f'trial_{i:03d}.result.json'
            if result.exists():
                row=json.loads(result.read_text())
                # Resume completed trials only after revalidating actual logs.
                checked=p.validate_trial(i)
                assert all(checked[k]==row[k] for k in ('success','door_open_success','traversal_success'))
                paired.paired_scene_audit(i)
            else:
                root=RUNS/f'trial_{i:03d}'
                if root.exists():raise RuntimeError(f'Incomplete trial preserved; inspect before rerun: {root}')
                p.run(command(i),env,f'trial_{i:03d}',root)
                row=p.validate_trial(i);row['global_trial']=i
                row['paired_scene_audit']=paired.paired_scene_audit(i)
                p.save(result,row)
            with p.LOCK:rows.append(row);p.report(rows)
    except BaseException as exc:
        p.STOP.set();p.save(OUT/f'lane_{j}.error.json',dict(error=repr(exc),time=time.strftime('%F %T')))
        raise


def main():
    def stop(signum,frame):
        p.STOP.set();raise InterruptedError(f'Evaluation termination signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p.OUT=OUT;p.RUNS=RUNS;p.OFFICIAL=t.OFFICIAL;p.WRAP=WRAP;p.DATA=t.DATA;p.RID=t.RID
    p.UNLOCK_DEG=25;p.TRAVERSAL_M=.5
    paired.OUT=OUT;paired.RUNS=RUNS;paired.REFERENCE=REFERENCE
    try:
        t.source_checks()
        train=json.loads((t.OUT/'training_complete.json').read_text())
        assert train['episodes']==500 and train['steps']==30000 and train['checkpoint']==str(t.OFFICIAL)
        assert json.loads((t.DATA/'meta/info.json').read_text())['total_episodes']==500
        assert json.loads((REFERENCE/'final_audit.json').read_text())['passed']
        assert (t.OUT/'rgb_config.yaml').read_bytes()==(REFERENCE/'rgb_config.yaml').read_bytes()
        shutil.copy2(t.OUT/'rgb_config.yaml',OUT/'rgb_config.yaml')
        for src in (Path(__file__),Path(p.__file__),Path(paired.__file__),Path(audit.__file__)):
            dest=OUT/src.name
            if not dest.exists():shutil.copy2(src,dest)
        previous=json.loads((REFERENCE/'protocol.json').read_text())
        p.save(OUT/'protocol.json',dict(previous,checkpoint=str(t.OFFICIAL),training_episodes=500,
            paired_reference=str(REFERENCE),physical_unlock_deg=25,traversal_distance_m=.5,
            caveat='Same64 scene seeds and visual settings as RGB100. Different data size, same30K training budget. Policy diffusion noise not independently fixed.'))
        p.save(OUT/'planned_trials.json',[command(i) for i in range(64)])
        p.status('WAIT_EVAL_GPUS',gpus=list(set(p.GPUS)))
        while True:
            free={gpu:int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for gpu in set(p.GPUS)}
            if min(free.values())>=26000:break
            p.status('WAIT_EVAL_GPUS',free_mib=free);time.sleep(300)
        env=environment(p.GPUS[0]);launcher=[p.PY,p.REPO/'high-level/dp/train/run_native_pi05_evo.py']
        if not WRAP.with_suffix('.pt').exists():
            p.status('EXPORTING')
            p.run(launcher+[p.REPO/'high-level/dp/export_official_lerobot_pi05_to_door_checkpoint.py',
                '--official_checkpoint',t.OFFICIAL,'--root',t.DATA,'--repo_id',t.RID,'--out_dir',WRAP,
                '--device','cuda:0','--rgb','--action_horizon','25','--num_inference_steps','10'],env,'export')
        if not (OUT/'parity.json').exists():
            p.status('VERIFYING_PROCESSOR_PARITY')
            p.run(launcher+[p.REPO/'high-level/dp/eval/verify_pi05_saved_processor_parity.py',
                '--official_checkpoint',t.OFFICIAL,'--door_checkpoint',WRAP,'--dataset_root',t.DATA,'--repo_id',t.RID,
                '--device','cuda:0','--output',OUT/'parity.json'],env,'parity')
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
