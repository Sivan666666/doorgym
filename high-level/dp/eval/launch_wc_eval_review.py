"""Durable three-GPU render queue; never touches existing GPU processes."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_dim60_blender_physx_eval64_20261010')
OUT=ROOT/'review_videos'
PY='/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_meister_pipeline_20260920/runtime/bin/python'
FF='/home/ps/miniconda3/envs/b1z1_lerobot/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'


def main():
    import fcntl
    lock=(OUT/'launcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def command(trial):
        return [PY,str(OUT/'code/render_wc_eval_review.py'),'--eval-root',str(ROOT),
                '--out',str(OUT),'--trial',str(trial),'--ffmpeg',FF]
    def render(index):
        gpu=[0,1,3][index]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for trial in [0,2,1,4]:
            marker=OUT/f'trial_{trial:03d}/shard_{index}_complete.json'
            if marker.exists():continue
            with (OUT/f'trial_{trial:03d}_shard_{index}.log').open('a') as f:
                subprocess.run(command(trial)+['--shard-index',str(index),'--shard-count','3'],
                               env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            print('SHARD_COMPLETE',trial,index,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(render,range(3)))
    for trial in [0,2,1,4]:
        with (OUT/f'trial_{trial:03d}_encode.log').open('a') as f:
            subprocess.run(command(trial)+['--encode-only'],stdout=f,stderr=subprocess.STDOUT,check=True)
        print('VIDEO_COMPLETE',trial,flush=True)
    report=dict(complete=True,trials=[0,2,1,4],time=time.strftime('%Y-%m-%d %H:%M:%S'))
    (OUT/'complete.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':main()
