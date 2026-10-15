"""Check small start-offset variations with current WC collision assets; no RGB."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess

JOB=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_dim60_aug100_pi05_20261008')
PY='/home/ps/miniconda3/envs/txc_vbc/bin/python'
REC='/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_meister_rgb_aug5_20260921/code/run_wc_contact_physics.py'

def run(item):
    index,gpu,y=item
    out=JOB/'pilot';out.mkdir(parents=True,exist_ok=True)
    with (out/f'offset_{index}.log').open('w') as stream:
        result=subprocess.run([PY,REC,'--name',f'offset_{index}','--output-root',str(out),
            '--steps','2601','--seed',str(2026100800+index),'--record-actions','--base-y',str(y)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4'),
            stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    p=out/f'offset_{index}'/'summary.json'
    report=dict(index=index,gpu=gpu,base_y=y,returncode=result.returncode,
        summary=json.loads(p.read_text()) if p.exists() else None)
    print(json.dumps(report),flush=True)
    return report

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(run,[(0,0,.125),(1,1,.15),(2,3,.175)]))
    (JOB/'pilot/results.json').write_text(json.dumps(rows,indent=2))
