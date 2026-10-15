"""Audit completed RGB trial and create side-by-side review media, not training inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labeled(image, label, width=480):
    image = cv2.resize(image, (width, int(image.shape[0]*width/image.shape[1])))
    image = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(image, label, (8,19), cv2.FONT_HERSHEY_SIMPLEX, .45, (240,240,240),1,cv2.LINE_AA)
    return image


def video(ffmpeg, render, out, views, fps, count, width, height):
    if not out.exists():
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-threads', '2']
        for view in views:
            command += ['-framerate', str(fps), '-i', str(render/'rgb'/view/'%06d.png')]
        command += ['-filter_complex_threads', '2', '-filter_complex',
                    ''.join(f'[{i}:v]' for i in range(len(views))) + f'hstack=inputs={len(views)}[out]',
                    '-map','[out]','-frames:v',str(count),'-an','-c:v','libx264','-crf','20',
                    '-preset','fast','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(out)]
        subprocess.run(command, check=True)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (height,width*len(views),3)
        decoded += 1
    cap.release()
    assert decoded == count, (out,decoded,count)
    return dict(path=out.name, sha256=sha(out), frames=decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--ffmpeg',default='ffmpeg')
    args = parser.parse_args()
    root = args.root
    cfg = read(root/'config.json')
    r = cfg['render']
    out = root/'review'
    out.mkdir(exist_ok=True)
    dataset = root/'dataset'
    provenance = read(dataset/'pipeline_provenance.json')
    assert provenance['readback_verified'] and len(provenance['episodes']) == cfg['batch']['episodes']
    for path, expected in read(dataset/'file_sha256.json').items():
        assert sha(dataset/path) == expected, path
    report = dict(episodes=[],quality_warnings=[],dataset_sha256_verified=True,
                  comparison_note='Clean previews already include randomized lighting; before/after here isolates photometric augmentation.',
                  temporal_note=provenance['temporal_note'])
    rows = []
    material_hashes = set()
    for episode in sorted((root/'episodes').iterdir()):
        status = read(episode/'status.json')
        assert all(s['status']=='complete' for s in status['stages'].values())
        render = episode/'render'
        rendered = read(render/'render_complete.json')
        aug = read(render/'augmentation.json')
        light = aug['lighting']
        assert light['materials_unchanged']
        material_hashes.add(light['material_sha256'])
        count = len(rendered['frames'])
        item = dict(id=status['id'], physics_qc=status['qc'], augmentation=aug, frames=count, views={})
        for view in r['views']:
            values = [f['images'][view] for f in rendered['frames']]
            item['views'][view] = {key:float(np.mean([v[key] for v in values])) for key in
                                   ('mean','std','clipped_low_fraction','clipped_high_fraction')}
            clipping = item['views'][view]['clipped_high_fraction']
            if clipping > 0.1:
                report['quality_warnings'].append(dict(episode=status['id'],view=view,
                    reason='More than 10% of RGB channel samples saturated at 255, averaged over the episode.',
                    average_clipped_high_fraction=clipping))
        item['three_view_video'] = video(args.ffmpeg,render,out/f'episode_{status["id"]:03d}_three_views.mp4',
                                        r['views'],r['fps'],count,r['width'],r['height'])
        if status['id'] == 0:
            item['two_view_video'] = video(args.ffmpeg,render,out/'episode_000_two_views.mp4',
                                           r['views'][:2],r['fps'],count,r['width'],r['height'])
        columns = []
        for view in r['views']:
            clean = cv2.imread(str(render/'clean_preview'/view/'000240.png'))
            augmented = cv2.imread(str(render/'rgb'/view/'000240.png'))
            assert clean is not None and augmented is not None
            columns.append(np.vstack([labeled(clean,f'ep{status["id"]} {view} / lighting only',320),
                                       labeled(augmented,f'ep{status["id"]} {view} / +RGB augmentation',320)]))
        comparison = np.hstack(columns)
        assert cv2.imwrite(str(out/f'episode_{status["id"]:03d}_comparison.jpg'),comparison)
        params = aug['photometric']['parameters']
        front = cv2.imread(str(render/'rgb/front/000240.png'))
        wrist = cv2.imread(str(render/'rgb/wrist/000240.png'))
        overview = cv2.imread(str(render/'rgb/overview/000240.png'))
        rows.append(np.hstack([labeled(im,f'ep{status["id"]} {view} B={params["brightness"]:.2f} C={params["contrast"]:.2f}',320)
                               for im,view in zip((front,wrist,overview),r['views'])]))
        report['episodes'].append(item)
    assert len(material_hashes) == 1
    unique_trajectories = len({e['physics_qc']['source_sha256'] for e in report['episodes']})
    report.update(structural_checks_passed=True,materials_identical_across_episodes=True,
                  photometric_qc_passed=not report['quality_warnings'], training_ready=False,
                  unique_physical_trajectories=unique_trajectories,
                  trajectory_note='See physical_qc.recording_reuse for reused recordings. Do not split identical motions across train/test.',
                  material_sha256=next(iter(material_hashes)), total_frames=sum(e['frames'] for e in report['episodes']))
    collage = 'five_episodes_rgb.jpg' if len(rows)==5 else 'episodes_rgb.jpg'
    assert cv2.imwrite(str(out/collage),np.vstack(rows))
    (out/'quality_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('REVIEW_VERIFIED',out,flush=True)


if __name__ == '__main__':
    main()
