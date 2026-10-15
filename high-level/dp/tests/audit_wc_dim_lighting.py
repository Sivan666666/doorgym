"""Full-sequence photometric and unchanged-geometry audit of a relit run."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    a = p.parse_args()
    cfg = read(a.root/'config.json')
    previous = Path(cfg['physics']['reuse_run_root'])
    report = dict(passed=True,episodes=[],images_checked=0,dark_shadow_warnings=[],
                  limits=dict(max_high_clipping_per_image=.005,max_black_clipping_per_image=.05))
    for eid in range(cfg['batch']['episodes']):
        folder = a.root/'episodes'/f'{eid:06d}'
        old_folder = previous/'episodes'/f'{eid:06d}'
        state, old_state = read(folder/'status.json'),read(old_folder/'status.json')
        assert state['raw_sha256'] == old_state['raw_sha256']
        new, old = read(folder/'render/render_complete.json'),read(old_folder/'render/render_complete.json')
        assert new['augmentation']['lighting']['material_sha256'] == old['augmentation']['lighting']['material_sha256']
        assert new['cameras'] == old['cameras']
        assert len(new['frames']) == len(old['frames']) == 520
        results = {}
        for view in cfg['render']['views']:
            before, after, clipping, black = [], [], [], []
            for nf, of in zip(new['frames'],old['frames']):
                assert nf['source_index'] == of['source_index']
                np.testing.assert_allclose(nf['camera_c2w'][view],of['camera_c2w'][view],atol=1e-6,rtol=0)
                n, o = nf['images'][view],of['images'][view]
                before.append(o['mean']);after.append(n['mean'])
                clipping.append(n['clipped_high_fraction']);black.append(n['clipped_low_fraction'])
                assert n['clipped_high_fraction'] < .005, (eid,view,nf['frame_index'],n)
                if n['clipped_low_fraction'] >= .05:
                    report['dark_shadow_warnings'].append(dict(episode=eid,view=view,frame=nf['frame_index'],
                        clipped_low_fraction=n['clipped_low_fraction']))
                report['images_checked'] += 1
            assert np.mean(after) < np.mean(before)-5
            results[view] = dict(mean_before=float(np.mean(before)),mean_after=float(np.mean(after)),
                maximum_high_clip_fraction=max(clipping),maximum_black_clip_fraction=max(black))
        report['episodes'].append(dict(id=eid,raw_unchanged=True,materials_unchanged=True,cameras_unchanged=True,views=results))
    report['passed'] = not report['dark_shadow_warnings']
    report['geometry_and_highlight_checks_passed'] = True
    target = a.root/'review/lighting_validation.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print('DIM_LIGHT_AUDIT',report['images_checked'],'dark_shadow_warnings',len(report['dark_shadow_warnings']),flush=True)


if __name__ == '__main__':
    main()
