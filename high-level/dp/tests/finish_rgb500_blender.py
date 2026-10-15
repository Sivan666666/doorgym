"""Audit recorded-pose FK and rendered frames, then encode review videos."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import cv2
import imageio_ffmpeg
from scipy.spatial.transform import Rotation

OUT=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/rgb500_blender_review_20260918')


def transform(pos,rot):
    m=np.eye(4);m[:3,3]=pos;m[:3,:3]=rot;return m


def main():
    result=json.loads((OUT/'render_complete.json').read_text())
    assert result['source_frames']==list(range(150,390,2))
    data=json.loads((OUT/'scene_data.json').read_text());ep=data['episode']
    errors=[]
    for i in result['source_frames']:
        root=ep['replay_root_state'][i]
        poses={'world':transform(root[:3],Rotation.from_quat(root[3:7]).as_matrix())}
        q=dict(zip(['joint1','joint2','joint3','joint4','joint5','joint6','jointGripper'],ep['replay_dof_pos'][i][-7:]))
        todo=list(data['models']['robot']['joints'])
        while todo:
            count=len(todo)
            for j in todo[:]:
                if j['parent'] not in poses:continue
                o=j['origin'];amount=q.get(j['name'],0.);rot=np.eye(3)
                if j['type'] in ('revolute','continuous'):
                    rot=Rotation.from_rotvec(np.asarray(j['axis'])*amount).as_matrix()
                poses[j['child']]=poses[j['parent']]@transform(o['xyz'],Rotation.from_euler('xyz',o['rpy']).as_matrix())@transform([0,0,0],rot)
                todo.remove(j)
            assert len(todo)<count
        errors.append(float(np.linalg.norm(poses['ee_gripper_link'][:3,3]-ep['replay_ee_pos'][i])))
    assert max(errors)<1e-4, max(errors)
    for name in ('front','wrist','overview'):
        paths=sorted((OUT/'frames'/name).glob('*.png'))
        assert [p.name for p in paths]==[f'{i:04d}.png' for i in range(120)]
        for path in paths:
            im=cv2.imread(str(path));assert im is not None and im.shape==(480,640,3) and im.std()>3,path
    # The imageio static build lacks drawtext; use the installed full build.
    ff='/home/ps/miniconda3/envs/lerobot/bin/ffmpeg'
    common=[ff,'-hide_banner','-loglevel','error','-n']
    encode=['-an','-c:v','libx264','-crf','18','-preset','fast','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart']
    def run_once(command, target):
        if target.exists():
            cap=cv2.VideoCapture(str(target))
            assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==120, target
            cap.release()
            return
        subprocess.run(command+[str(target)],check=True)
    for name in ('front','wrist','overview'):
        run_once(common+['-framerate','12.5','-i',str(OUT/'frames'/name/'%04d.png')]+encode,OUT/(name+'_cycles.mp4'))
    run_once(common+['-i',str(OUT/'front_cycles.mp4'),'-i',str(OUT/'wrist_cycles.mp4'),
        '-filter_complex','[0:v][1:v]hstack=inputs=2']+encode,OUT/'dual_rgb_cycles.mp4')
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    labels=f",drawtext=fontfile={font}:text='BLENDER CYCLES - KINEMATIC REPLAY':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65"
    labels+=f",drawtext=fontfile={font}:text='FRONT RGB':x=970:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65"
    labels+=f",drawtext=fontfile={font}:text='WRIST RGB':x=970:y=372:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65"
    filt='[0:v]scale=960:720[o];[1:v]scale=480:360[f];[2:v]scale=480:360[w];[f][w]vstack=inputs=2[r];[o][r]hstack=inputs=2'+labels
    run_once(common+['-i',str(OUT/'overview_cycles.mp4'),'-i',str(OUT/'front_cycles.mp4'),'-i',str(OUT/'wrist_cycles.mp4'),
        '-filter_complex',filt]+encode,OUT/'three_rgb_cycles.mp4')
    videos=[]
    for path in sorted(OUT.glob('*.mp4')):
        cap=cv2.VideoCapture(str(path));assert cap.isOpened()
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        assert count==120 and abs(fps-12.5)<1e-6,(path,count,fps)
        for index in (0,60,119):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();assert ok and im.std()>3
        cap.release()
        videos.append(dict(file=path.name,frames=count,fps=fps,duration_s=count/fps,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(videos)==5
    audit=dict(passed=True,rendered_images=360,source_frame_range=[150,388],source_time_s=[6,15.52],
        max_ee_position_error_m=max(errors),mean_ee_position_error_m=float(np.mean(errors)),
        intrinsics=ep['camera_intrinsics'],resolution=[640,480],horizontal_fov_deg=55,
        videos=videos,renderer=result['renderer'],
        scope='Appearance preview, kinematic replay of recorded expert data; NOT a policy rollout or physics test.',
        changes='PBR material response, lighting, AgX, small primitive bevels; room envelope approximated. No post-render RGB augmentation.',
        camera_note='Original stored attachment poses and focal length. Gym +X-forward convention converted to Blender.')
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
