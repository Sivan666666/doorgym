"""Compare the approved dark preview with slightly brighter final point lights."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def read(p):return json.loads(Path(p).read_text())


def label(im,text):
    im=cv2.resize(im,(480,360))
    im=cv2.copyMakeBorder(im,28,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(im,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1,cv2.LINE_AA)
    return im


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--previous',required=True,type=Path)
    p.add_argument('--preview',action='store_true')
    a=p.parse_args()
    cfg=read(a.root/'config.json')
    old=read(a.previous/'settings.json')
    folder=a.root/'episodes/000000/render'
    new=read(folder/'augmentation.json')
    lights=[l for l in new['lighting']['lights'] if l['energy_after']>0]
    assert len(lights)==2 and all(l['type']=='POINT' for l in lights)
    for before,after in zip(sorted(old['lights'],key=lambda l:l['position'][1]),sorted(lights,key=lambda l:l['location'][1])):
        np.testing.assert_allclose(before['position'],after['location'],atol=1e-6)
        assert after['energy_after']/before['energy']==1.25
    assert old['material_sha256']==new['lighting']['material_sha256']
    assert all(w['strength_after']==0 for w in new['lighting']['world'])
    selected=[0,240,490] if a.preview else old['frames']
    out=a.root/'review';out.mkdir(exist_ok=True)
    report=dict(passed=True,preview_only=a.preview,positions_unchanged=True,materials_unchanged=True,
                power_before=12,power_after=15,views={},resolution_note='Old 480x360/4 samples versus new 640x480/16 samples; pixel means are descriptive, not a controlled pixelwise exposure test.')
    for view in cfg['render']['views']:
        before_means,after_means=[],[]
        for i in selected:
            before=cv2.imread(str(a.previous/view/f'{i:06d}.png'))
            after=cv2.imread(str(folder/'rgb'/view/f'{i:06d}.png'))
            assert before is not None and after is not None
            before_means.append(float(before.mean()));after_means.append(float(after.mean()))
        report['views'][view]=dict(images_compared=len(selected),old_mean=float(np.mean(before_means)),new_mean=float(np.mean(after_means)))
        assert np.mean(after_means)>np.mean(before_means), report['views'][view]
        assert np.mean(after_means)<np.mean(before_means)*1.6, 'Brightness changed more than intended'
    before_row=[];after_row=[]
    for view in cfg['render']['views']:
        before_row.append(label(cv2.imread(str(a.previous/view/'000240.png')),f'Previous / {view} / 12'))
        after_row.append(label(cv2.imread(str(folder/'rgb'/view/'000240.png')),f'Final / {view} / 15'))
    assert cv2.imwrite(str(out/'point_light_before_after.jpg'),np.vstack([np.hstack(before_row),np.hstack(after_row)]))
    name='point_brightness_preview.json' if a.preview else 'point_brightness_validation.json'
    (out/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
