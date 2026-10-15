"""Verify all full-trajectory Cycles images and encode RGB videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='/home/ps/miniconda3/envs/lerobot/bin/ffmpeg')
    a = parser.parse_args()
    out = a.output
    reports = [json.loads((out / f'render_complete_{start}_{stop}.json').read_text())
               for start, stop in ((0, 325), (325, 650))]
    assert reports[0]['source_frames'] + reports[1]['source_frames'] == list(range(0, 1300, 2))
    assert all(r['output_fps'] == 12.5 and r['duration_s'] == 52 for r in reports)
    assert all(r['max_ee_error_m'] < 1e-4 and r['max_spindle_error_m'] < 1e-4 for r in reports)
    images = []
    for name, shape in [('overview', (720, 960, 3)), ('front', (480, 640, 3)), ('wrist', (480, 640, 3))]:
        paths = sorted((out / 'frames' / name).glob('*.png'))
        assert [p.name for p in paths] == [f'{i:04d}.png' for i in range(650)], name
        for path in paths:
            im = cv2.imread(str(path))
            assert im is not None and im.shape == shape and im.std() > 3, path
            images.append(dict(file=str(path.relative_to(out)), sha256=sha(path)))
    print('VERIFIED all 1950 rendered PNGs and full trajectory coverage', flush=True)
    (out / 'frame_sha256.json').write_text(json.dumps(images, indent=2))
    common = [a.ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-filter_complex_threads', '2']
    encode = ['-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-threads', '4',
              '-pix_fmt', 'yuv420p', '-movflags', '+faststart']

    def run(command, name):
        path = out / name
        if path.exists():
            cap = cv2.VideoCapture(str(path))
            assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 650, path
            cap.release()
        else:
            subprocess.run(command + [str(path)], check=True)
        print('ENCODED', name, flush=True)

    for name in ('overview', 'front', 'wrist'):
        run(common + ['-framerate', '12.5', '-i', str(out / 'frames' / name / '%04d.png')] + encode,
            name + '_full_cycles.mp4')
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def label(text, x, y, size=18):
        return f",drawtext=fontfile={font}:text='{text}':x={x}:y={y}:fontsize={size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
    dual = '[0:v][1:v]hstack=inputs=2' + label('FRONT RGB', 14, 14) + label('WRIST RGB', 654, 14)
    run(common + ['-i', str(out / 'front_full_cycles.mp4'), '-i', str(out / 'wrist_full_cycles.mp4'),
                  '-filter_complex', dual] + encode, 'dual_rgb_full_cycles.mp4')
    triple = '[1:v]scale=480:360[f];[2:v]scale=480:360[w];[f][w]vstack=inputs=2[r];[0:v][r]hstack=inputs=2'
    triple += label('BLENDER CYCLES - EXPERT TRAJECTORY REPLAY', 14, 14)
    triple += label('OVERVIEW (WALL CUTAWAY)', 14, 688, 16)
    triple += label('FRONT RGB', 974, 14) + label('WRIST RGB', 974, 374)
    run(common + ['-i', str(out / 'overview_full_cycles.mp4'), '-i', str(out / 'front_full_cycles.mp4'),
                  '-i', str(out / 'wrist_full_cycles.mp4'), '-filter_complex', triple] + encode,
        'three_rgb_full_cycles.mp4')

    videos = []
    for path in sorted(out.glob('*_full_cycles.mp4')):
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), path
        fps, declared = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dims = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        assert fps == 12.5 and declared == 650, (path, fps, declared)
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame is not None and frame.std() > 3, (path, decoded)
            decoded += 1
        cap.release()
        assert decoded == 650, (path, decoded)
        videos.append(dict(file=path.name, frames=decoded, fps=fps, duration_s=decoded / fps,
                           dimensions=dims, bytes=path.stat().st_size, sha256=sha(path)))
    assert len(videos) == 5

    tiles = []
    for index, label_text in [(0, '0.0s - APPROACH'), (115, '9.2s - HANDLE'),
                              (140, '11.2s - PULL'), (275, '22.0s - OPEN'),
                              (325, '26.0s - PASS THROUGH'), (649, '51.9s - END HOLD')]:
        im = cv2.imread(str(out / 'frames' / 'overview' / f'{index:04d}.png'))
        im = cv2.resize(im, (480, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (480, 32), (35, 35, 35), -1)
        cv2.putText(im, label_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    assert cv2.imwrite(str(out / 'trajectory_contact_sheet.jpg'), sheet)
    summary = dict(passed=True, rendered_images=1950, source_frames=1300, rendered_frames_per_view=650,
                   full_duration_s=52, fps=12.5, source_sampling='Every second source frame; no time speedup, no synthesized motion.',
                   max_ee_error_m=max(r['max_ee_error_m'] for r in reports),
                   max_spindle_error_m=max(r['max_spindle_error_m'] for r in reports),
                   door_render_copy_axis_scale=reports[0]['door_render_copy_axis_scale'],
                   source=reports[0]['source'], notes=reports[0]['notes'], videos=videos)
    (out / 'verification.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
