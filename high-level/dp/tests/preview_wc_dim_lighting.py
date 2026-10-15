"""Render representative relit frames and compare with the bright source run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def label(image, text):
    image = cv2.resize(image,(400,300))
    image = cv2.copyMakeBorder(image,28,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(image,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--renderer',required=True,type=Path)
    args = parser.parse_args()
    cfg = read(args.root/'config.json')
    previous = Path(cfg['physics']['reuse_run_root'])
    gpus = cfg['batch']['gpus']
    def worker(worker_id):
        for eid in range(worker_id,cfg['batch']['episodes'],len(gpus)):
            folder = args.root/'episodes'/f'{eid:06d}'
            state = read(folder/'status.json')
            with (folder/'preview.log').open('w') as log:
                subprocess.run([cfg['tools']['render_python'],str(args.renderer),'--config',str(args.root/'config.json'),
                    '--scene',state['scene'],'--out',str(folder/'render'),'--preview-frames','0,240,490'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[worker_id]),OMP_NUM_THREADS='4'),
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker,range(len(gpus))))
    out = args.root/'preview'
    out.mkdir(exist_ok=True)
    comparisons = []
    for eid in range(cfg['batch']['episodes']):
        folder = args.root/'episodes'/f'{eid:06d}'
        old = previous/'episodes'/f'{eid:06d}'
        new_aug = read(folder/'render/augmentation.json')
        old_aug = read(old/'render/augmentation.json')
        assert new_aug['lighting']['material_sha256'] == old_aug['lighting']['material_sha256']
        for number in (0,240,490):
            old_images, new_images = [], []
            for view in cfg['render']['views']:
                relative = Path('render/rgb')/view/f'{number:06d}.png'
                before, after = cv2.imread(str(old/relative)),cv2.imread(str(folder/relative))
                assert before is not None and after is not None
                row = dict(episode=eid,frame=number,view=view,mean_before=float(before.mean()),mean_after=float(after.mean()),
                           high_clip_before=float((before==255).mean()),high_clip_after=float((after==255).mean()))
                assert row['mean_after'] < row['mean_before']-5, row
                assert row['high_clip_after'] < .005, row
                comparisons.append(row)
                old_images.append(label(before,f'BEFORE ep{eid} {view} frame{number}'))
                new_images.append(label(after,f'DIM LIGHT ep{eid} {view} frame{number}'))
            assert cv2.imwrite(str(out/f'episode_{eid:03d}_frame_{number:03d}_before_after.jpg'),
                               np.vstack([np.hstack(old_images),np.hstack(new_images)]))
    report = dict(passed=True,images_compared=len(comparisons),materials_unchanged_from_source=True,comparisons=comparisons)
    (out/'preview_qc.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DIM_PREVIEW_VERIFIED',len(comparisons),flush=True)


if __name__ == '__main__':
    main()
