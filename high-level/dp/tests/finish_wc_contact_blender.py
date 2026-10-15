"""Validate physical replay frames, summarize contacts, and encode RGB videos.

The report describes the rollout; it does not equate rendering success with
physical task success or claim a fully dynamic robot base.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--views',default='overview,contact')
    p.add_argument('--end-step',type=int)
    p.add_argument('--output-name',default='blender')
    p.add_argument('--ffmpeg',default='/home/ps/miniconda3/envs/lerobot/bin/ffmpeg')
    a=p.parse_args()
    assert Path(a.output_name).name==a.output_name
    out=a.trial/a.output_name
    trajectory=json.loads((a.trial/'trajectory.json').read_text())
    audit=json.loads((a.trial/'contact_audit.json').read_text())
    source_fps=trajectory['fps'];fps=source_fps/a.stride
    end=len(trajectory['frames']) if a.end_step is None else a.end_step
    assert 0<end<=len(trajectory['frames'])
    assert all(v['phase']=='hold_home' for v in trajectory['frames'][end:])
    indices=list(range(0,end,a.stride));n=len(indices)
    source_coverage=set()
    reports=list(out.glob('render_complete_*.json'))
    wall_modes=set()
    scene_modes=set()
    for path in reports:
        report=json.loads(path.read_text())
        assert report['actual_physics_body_poses'] and report['geometry_scale']==[1,1,1]
        assert report['output_fps']==fps
        wall_modes.add(report.get('white_walls',False))
        scene_modes.add((report.get('scene_variant'), report.get('room_clearance_sha256')))
        source_coverage.update(report['frames'])
    assert len(wall_modes)==1,'Mixed wall scene variants'
    assert len(scene_modes)==1,'Mixed room scene variants'
    assert source_coverage==set(indices),'Incomplete physical trajectory render coverage'
    images=[];videos=[]
    common=[a.ffmpeg,'-hide_banner','-loglevel','error','-n','-filter_complex_threads','2']
    encode=['-an','-c:v','libx264','-crf','18','-preset','fast','-threads','4',
            '-pix_fmt','yuv420p','-movflags','+faststart']
    for view in a.views.split(','):
        paths=sorted((out/'frames'/view).glob('*.png'))
        assert [v.name for v in paths]==[f'{i:04d}.png' for i in range(n)],view
        dimensions=None
        for path in paths:
            im=cv2.imread(str(path));assert im is not None and im.std()>3,path
            if dimensions is None:dimensions=im.shape
            assert im.shape==dimensions
            images.append(dict(file=str(path.relative_to(out)),sha256=digest(path)))
        video=out/f'{view}_physical_cycles.mp4'
        if not video.exists():
            subprocess.run(common+['-framerate',str(fps),'-i',str(out/'frames'/view/'%04d.png')]+encode+[str(video)],check=True)
        videos.append(video)
    if {'overview','contact'}.issubset(a.views.split(',')):
        video=out/'overview_contact_physical_cycles.mp4'
        font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        filters="[0:v][1:v]hstack=inputs=2,drawtext=fontfile="+font+":text='PHYSX CONTACT ROLLOUT / BLENDER RGB - KINEMATIC BASE':x=14:y=14:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6"
        if not video.exists():
            subprocess.run(common+['-i',str(out/'overview_physical_cycles.mp4'),'-i',str(out/'contact_physical_cycles.mp4'),'-filter_complex',filters]+encode+[str(video)],check=True)
        videos.append(video)
    video_specs=[(v,n,0.) for v in videos]
    if 'contact' in a.views.split(','):
        grasp=[v['step'] for v in audit if v['phase']=='grasp']
        release=[v['step'] for v in audit if v['phase']=='release_handle']
        if grasp and release:
            start=max(0.,math.floor((min(grasp)/source_fps-1)*fps)/fps)
            stop=min(n/fps,math.ceil((max(release)/source_fps+.5)*fps)/fps)
            count=round((stop-start)*fps);video=out/'grasp_detail_physical_cycles.mp4'
            if not video.exists():
                subprocess.run(common+['-ss',str(start),'-i',str(out/'contact_physical_cycles.mp4'),'-frames:v',str(count)]+encode+[str(video)],check=True)
            video_specs.append((video,count,start))
    verified=[]
    for video,expected,start in video_specs:
        cap=cv2.VideoCapture(str(video));assert cap.isOpened(),video
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==expected
        assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<1e-5
        size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im is not None and im.std()>3
            decoded+=1
        cap.release();assert decoded==expected
        verified.append(dict(file=video.name,frames=expected,fps=fps,duration_s=expected/fps,
                             source_start_s=start,dimensions=size,sha256=digest(video)))
    longest=current=0;phases={}
    for row in audit:
        current=current+1 if row['both'] else 0;longest=max(longest,current)
        item=phases.setdefault(row['phase'],dict(frames=0,bilateral_contact_frames=0))
        item['frames']+=1;item['bilateral_contact_frames']+=int(row['both'])
    report=dict(render_verified=True,source_frames=len(audit),source_fps=source_fps,
                rendered_frames_per_view=n,source_sampling_stride=a.stride,time_speedup=1,
                end_step=end,omitted_idle_tail_s=(len(audit)-end)/source_fps,
                max_door_open_deg=max(abs(v['door_deg']) for v in audit),
                max_handle_depression_deg=max(v['handle_deg'] for v in audit),
                longest_bilateral_contact_s=longest/source_fps,contacts_by_phase=phases,
                base_kinematic=True,door_state_replay=False,geometry_scale=[1,1,1],
                white_walls=next(iter(wall_modes)),walls_render_only=next(iter(wall_modes)),
                scene_variant=next(iter(scene_modes))[0],room_clearance_sha256=next(iter(scene_modes))[1],
                caveats=['Assumed masses/friction; not calibrated hardware.',
                         'Blender renders recorded physics poses; no second contact simulation.',
                         'Triangle intersection audit is separate from PhysX contact forces.'],
                videos=verified)
    (out/'frame_sha256.json').write_text(json.dumps(images,indent=2))
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
