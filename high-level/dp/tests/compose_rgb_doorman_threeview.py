"""Synchronize same-rollout viewer video with the two actual recorded RGB inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--overview', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with np.load(args.raw, allow_pickle=True) as data:
        front, wrist = data['front_rgb'], data['wrist_rgb']
        fps = float(data['fps'])
        cap = cv2.VideoCapture(args.overview)
        assert cap.isOpened()
        assert len(front) == len(wrist) == int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 1e-5
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pixel_format', 'bgr24', '-video_size', '1280x540', '-framerate', str(fps),
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '21',
               '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(len(front)):
                ok, overview = cap.read()
                assert ok, i
                frame = np.zeros((540, 1280, 3), np.uint8)
                frame[:, :960] = cv2.resize(overview, (960, 540), interpolation=cv2.INTER_AREA)
                frame[30:270, 960:] = cv2.resize(front[i][..., ::-1], (320, 240), interpolation=cv2.INTER_AREA)
                frame[300:540, 960:] = cv2.resize(wrist[i][..., ::-1], (320, 240), interpolation=cv2.INTER_AREA)
                cv2.rectangle(frame, (0, 0), (490, 30), (0, 0, 0), -1)
                for text, position in [('THIRD-PERSON | scripted rollout', (10, 22)),
                                       ('FRONT RGB input', (972, 22)), ('WRIST RGB input', (972, 292))]:
                    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                encoder.stdin.write(frame.tobytes())
        finally:
            encoder.stdin.close()
            cap.release()
            code = encoder.wait()
        assert code == 0
        metadata = dict(raw=args.raw, overview=args.overview, output=str(output), frames=len(front), fps=fps,
                        duration_s=len(front)/fps, synchronization='same run; both captured on sim steps 0,2,4,...',
                        layout='left third-person; top right front RGB; bottom right wrist RGB',
                        rgb_source='actual saved post-augmentation input, downsampled only for preview',
                        policy_rollout=False, seed=int(data['seed']),
                        pull_max_open_deg=float(data['pull_max_open_deg']),
                        pull_max_traversal_m=float(data['pull_max_traversal_m']))
        output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
