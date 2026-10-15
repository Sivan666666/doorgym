"""Create ordered reference comparisons and verify the full walkthrough video."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def tile(path,label,width=540):
    im=cv2.imread(str(path));assert im is not None,path
    h=round(im.shape[0]*width/im.shape[1]);im=cv2.resize(im,(width,h))
    im=cv2.copyMakeBorder(im,30,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(im,label,(10,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
    return im


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--references',required=True,type=Path)
    p.add_argument('--previews',required=True,type=Path)
    p.add_argument('--ffmpeg',required=True)
    p.add_argument('--comparisons-only',action='store_true')
    a=p.parse_args();out=a.root/'review';out.mkdir(parents=True,exist_ok=True)
    pairs=[]
    for i in range(1,7):
        pair=np.hstack([tile(a.references/f'{i}.jpg',f'Photo {i} / real reference'),
            tile(a.previews/f'photo_{i}_layout.png',f'Photo {i} / estimated Blender layout')])
        assert cv2.imwrite(str(out/f'photo_{i}_comparison.jpg'),pair)
        pairs.append(pair)
    contact=np.vstack([np.hstack(pairs[i:i+2]) for i in (0,2,4)])
    assert cv2.imwrite(str(out/'six_photo_comparison.jpg'),contact)
    if a.comparisons_only:return
    cfg=read(a.root/'config.json');ep=a.root/'episodes/000000'
    status=read(ep/'status.json');scene=Path(status['scene'])
    report=read(scene.parent/'room_clearance.json')
    assert report['original_assets_unchanged'] and not report['clearance_warnings']
    assert len(report['reference_images'])==6 and report['frames_checked']==2601
    frames=len(read(ep/'render/render_complete.json')['frames'])
    target=out/'photo_room_full_walkthrough.mp4'
    subprocess.run([a.ffmpeg,'-hide_banner','-loglevel','error','-n',
        '-framerate',str(cfg['render']['fps']),'-i',str(ep/'render/rgb/overview/%06d.png'),
        '-frames:v',str(frames),'-an','-c:v','libx264','-preset','fast','-crf','18',
        '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(target)],check=True)
    cap=cv2.VideoCapture(str(target));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(cfg['render']['height'],cfg['render']['width'],3)
        decoded+=1
    cap.release();assert decoded==frames and fps==cfg['render']['fps']
    (out/'photo_reconstruction_verified.json').write_text(json.dumps(dict(
        passed=True,frames=decoded,fps=fps,duration_s=decoded/fps,walkthrough_sha256=sha(target),
        source_sha256=report['source_sha256'],original_assets_sha256=report['original_assets_sha256'],
        estimated_dimensions_m=report['dimensions_m'],lights=report['lights'],
        min_aabb_gap_m=report['minimum_aabb_gap_m'],clearance_method=report['clearance_method'],
        layout_previews=read(a.previews/'preview.json'),
        notes=['References 3 and 4 are identical files.',
               'Front/wrist cameras and recorded physical trajectory retained; overview is a human-height walkthrough.',
               'Room/props/lighting reconstructed by eye, not measured/calibrated. No new room collision simulation.',
               'Phone layout comparisons hide the robot only; full video includes it.',
               'Diffuse appearance and furniture detail approximate, not exact texture recovery.']),indent=2))
    print('PHOTO_REVIEW_VERIFIED',target,decoded,flush=True)


if __name__=='__main__':main()
