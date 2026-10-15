"""Task-scoped watchdog: 30min snapshots, immediate failure records, safe API retry.

Only retries the diagnosed, pre-training converter API mismatch. Unknown failures
are surfaced for inspection, never hidden by deleting data or restarting training.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

REPO=Path('/home/ps/workspace/txc/doorgym')
OUT=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/pull_evo_trajectory_20260910')
PY='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'
NAMES={'smooth15':1,'steady15':3,'trimtail15':0}


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def main():
    lock=(OUT/'watchdog.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last_monitor=0; last_alert={}
    while True:
        states={n:json.loads((OUT/f'{n}.status.json').read_text()) for n in NAMES if (OUT/f'{n}.status.json').exists()}
        now=time.time()
        for name,state in states.items():
            if state['phase']!='FAILED':continue
            if last_alert.get(name)!=state.get('time'):
                with (OUT/'watchdog_alerts.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
                last_alert[name]=state.get('time')
            marker=OUT/f'{name}.auto_converter_retry.json'
            log=OUT/f'{name}.convert.log'
            error=state.get('error','')
            if ('convert_door_raw_to_lerobot.py' in error and log.exists()
                and 'only 0-dimensional arrays' in log.read_text() and not marker.exists()):
                # Existing per-lane process must exit and release its lock before retry.
                time.sleep(5)
                cmd=[PY,'-u',str(REPO/'high-level/dp/train/run_pull_evo_trajectory_ablation_0910_ps1.py'),
                     '--only',name,'--gpu',str(NAMES[name])]
                with (OUT/f'{name}.auto_retry.pipeline.log').open('a') as f:
                    p=subprocess.Popen(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL,start_new_session=True)
                write(marker,dict(time=time.strftime('%F %T'),pid=p.pid,command=cmd,
                                  reason='Known Evo writer scalar API incompatibility; main converter restored'))
        if now-last_monitor>=1800:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
            snapshot=dict(time=time.strftime('%F %T'),states=states,gpu=gpu)
            with (OUT/'watchdog_30min.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
            print(json.dumps(snapshot),flush=True);last_monitor=now
        completed=all((OUT/f'{name}.result.json').exists() for name in NAMES) and (OUT/'videos/complete.json').exists()
        if completed:
            counts={}
            for name in NAMES:
                d=json.loads((OUT/f'{name}.result.json').read_text())
                assert d['trials']==64 and len(d['trials_detail'])==64
                assert (OUT/f'{name}.parity.json').exists()
                counts[name]={k:d[k] for k in ('door_open_successes','traversal_successes','successes')}
            write(OUT/'all_experiments_complete.json',dict(time=time.strftime('%F %T'),results=counts,
                note='All training/evaluations done; final assistant review/report still required'))
            return
        time.sleep(60)


if __name__=='__main__':main()
