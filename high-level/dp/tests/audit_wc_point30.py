"""Audit power-30 video against the prior otherwise-identical power-15 run."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def read(p):return json.loads(Path(p).read_text())


def label(im,text):
    im=cv2.resize(im,(480,360))
    im=cv2.copyMakeBorder(im,28,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(im,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1,cv2.LINE_AA)
    return im


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    a=p.parse_args()
    cfg=read(a.root/'config.json')
    old_root=Path(cfg['physics']['reuse_run_root'])
    old_folder=old_root/'episodes/000000'
    folder=a.root/'episodes/000000'
    state,old_state=read(folder/'status.json'),read(old_folder/'status.json')
    assert state['raw_sha256']==old_state['raw_sha256']
    new,old=read(folder/'render/render_complete.json'),read(old_folder/'render/render_complete.json')
    assert new['config']==old['config']
    assert new['cameras']==old['cameras']
    assert len(new['frames'])==len(old['frames'])==520
    nl,ol=new['augmentation']['lighting'],old['augmentation']['lighting']
    assert nl['material_sha256']==ol['material_sha256']
    active=lambda report: {l['name']:l for l in report['lights'] if l['energy_after']>0}
    na,oa=active(nl),active(ol)
    assert set(na)==set(oa) and len(na)==2
    for name in na:
        assert na[name]['type']==oa[name]['type']=='POINT'
        for k in ('location','color_after','radius_m'):
            np.testing.assert_allclose(na[name][k],oa[name][k],rtol=0,atol=1e-6)
        assert na[name]['energy_after']==30 and oa[name]['energy_after']==15
    assert all(w['strength_after']==0 for w in nl['world'])
    report=dict(structural_checks_passed=True,raw_unchanged=True,materials_unchanged=True,cameras_unchanged=True,
                light_positions_unchanged=True,light_power_before=15,light_power_after=30,
                frames=520,views={},dark_shadow_warnings=[],highlight_warnings=[])
    for view in cfg['render']['views']:
        before,after,high,low=[],[],[],[]
        for nf,of in zip(new['frames'],old['frames']):
            assert nf['source_index']==of['source_index']
            np.testing.assert_allclose(nf['camera_c2w'][view],of['camera_c2w'][view],atol=1e-6,rtol=0)
            n,o=nf['images'][view],of['images'][view]
            before.append(o['mean']);after.append(n['mean'])
            high.append(n['clipped_high_fraction']);low.append(n['clipped_low_fraction'])
            for key,limit,target in [('clipped_low_fraction',.05,'dark_shadow_warnings'),('clipped_high_fraction',.005,'highlight_warnings')]:
                if n[key]>=limit:
                    report[target].append(dict(view=view,frame=nf['frame_index'],fraction=n[key]))
        assert np.mean(after)>np.mean(before)
        report['views'][view]=dict(mean_before=float(np.mean(before)),mean_after=float(np.mean(after)),
                                  maximum_high_clip=max(high),maximum_black_clip=max(low))
    report['photometric_thresholds_passed']=not(report['dark_shadow_warnings'] or report['highlight_warnings'])
    out=a.root/'review';out.mkdir(exist_ok=True)
    rows=[]
    for base,title in [(old_folder,'Previous: power 15'),(folder,'New: power 30')]:
        ims=[]
        for view in cfg['render']['views']:
            im=cv2.imread(str(base/'render/rgb'/view/'000240.png'))
            assert im is not None
            ims.append(label(im,title+' / '+view))
        rows.append(np.hstack(ims))
    assert cv2.imwrite(str(out/'power30_vs15.jpg'),np.vstack(rows))
    (out/'power30_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
