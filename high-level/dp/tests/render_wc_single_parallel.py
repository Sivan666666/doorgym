"""Split one baked episode over GPUs, then validate cached frames via the pipeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--sim-data',type=Path,required=True)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    root=Path(cfg['run_root'])
    assert cfg['batch']['episodes']==1
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.parallel_render.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([sys.executable,str(a.sim_data/'pipeline.py'),'--config',str(a.config),'--stage','prepare'],check=True)
        folder=root/'episodes/000000'
        state=json.loads((folder/'status.json').read_text())
        scene=Path(state['scene'])
        report=json.loads((scene.parent/'scene_verified.json').read_text())
        count=len(report['source_indices'])
        gpus=cfg['batch']['gpus']
        # Initialize shared cache/calibration once, and leave real completion
        # publishing to the normal full-frame validation path below.
        def render(indices,gpu,log):
            command=[cfg['tools']['render_python'],str(a.sim_data/'render_meister.py'),
                     '--config',str(root/'config.json'),'--scene',str(scene),'--out',str(folder/'render'),
                     '--preview-frames',','.join(map(str,indices))]
            with log.open('w') as stream:
                subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4'),
                               stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=cfg['batch']['render_timeout_s'])
        render([0],gpus[0],folder/'parallel_init.log')
        def worker(index):
            indices=range(index,count,len(gpus))
            render(indices,gpus[index],folder/f'parallel_gpu{gpus[index]}.log')
            print('SHARD_COMPLETE',index,flush=True)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker,range(len(gpus))))
        # Verifies all 520 marker signatures and image checksums, skips cached
        # images, writes render_complete, then performs official LeRobot export.
        subprocess.run([sys.executable,str(a.sim_data/'pipeline.py'),'--config',str(a.config)],check=True)


if __name__=='__main__':main()
