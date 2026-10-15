"""Reuse rendered frames only after exact equality of their physical snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--stop',type=int,default=260)
    a=p.parse_args()
    source=json.loads((a.source/'trajectory.json').read_text())
    target=json.loads((a.target/'trajectory.json').read_text())
    report=json.loads((a.source/'blender'/f'render_complete_0_{a.stop}.json').read_text())
    assert source['fps']==target['fps']
    stride=round(source['fps']/report['output_fps'])
    assert report['frames']==list(range(0,a.stop*stride,stride))
    assert source['frames'][:a.stop*stride]==target['frames'][:a.stop*stride]
    copied=[]
    for index in range(a.stop):
        for view in report['views']:
            relative=Path('frames')/view/f'{index:04d}.png'
            old=a.source/'blender'/relative;new=a.target/'blender'/relative
            assert old.is_file(),old
            new.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256(old.read_bytes()).hexdigest()
            if new.exists():assert hashlib.sha256(new.read_bytes()).hexdigest()==digest,new
            else:shutil.copy2(old,new)
            copied.append(dict(file=str(relative),sha256=digest))
    report.update(source=str(a.target/'trajectory.json'),raw_source_frames=len(target['frames']),
                  end_step=len(target['frames']),omitted_idle_frames=0,
                  cache_origin=str(a.source),exact_snapshot_prefix_checked=a.stop*stride)
    (a.target/'blender'/f'render_complete_0_{a.stop}.json').write_text(json.dumps(report,indent=2))
    (a.target/'blender'/'verified_render_cache.json').write_text(json.dumps(dict(
        source=str(a.source),target=str(a.target),all_snapshot_fields_equal=True,
        physics_frames_compared=a.stop*stride,images=copied),indent=2))
    print('Reused',len(copied),'images after exact equality of',a.stop*stride,'full physics snapshots.')


if __name__=='__main__':main()
