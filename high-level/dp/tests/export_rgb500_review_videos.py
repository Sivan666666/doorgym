"""Read-only dataset previews and a fresh policy rollout with RGB capture.

No training/evaluation inputs are overwritten. Inference hooks only copy the
already prepared camera inputs; neither camera rendering nor augmentation is
invoked a second time by the capture hook.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

REPO = Path('/home/ps/workspace/txc/doorgym')
A = Path('/mnt/sda/ps_archive/doorgym/high-level')
EXPERIMENT = A / 'dp/result/0914_pull_rgb_aug500_joint9_g080_fixed15'
OUT = A / 'dp/result/rgb500_review_videos_20260918'
SIM = '/home/ps/miniconda3/envs/txc_vbc/bin/python'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class Writer:
    def __init__(self, path, width, height, fps=25):
        if path.exists():
            raise FileExistsError(path)
        self.proc = subprocess.Popen([
            ffmpeg(), '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
            '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '20', '-threads', '2', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, rgb):
        self.proc.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError('Video encoder failed')


def dataset():
    import numpy as np
    rows = json.loads((EXPERIMENT / 'raw_manifest.json').read_text())
    for idx in (100, 102):
        row = rows[idx]
        assert row['global_episode'] == idx
        prefix = OUT / f'train_episode_{idx:06d}'
        if Path(str(prefix) + '.json').exists():
            assert all(Path(str(prefix) + suffix).is_file() for suffix in ('_front.mp4', '_wrist.mp4', '_dual_rgb.mp4'))
            print(f'Already exported training episode {idx}', flush=True)
            continue
        with np.load(row['path'], allow_pickle=True) as data:
            # Bound ZIP reads explicitly; some old NumPy/Python ZIP paths are
            # extremely slow for multi-GB incompressible noisy RGB arrays.
            with zipfile.ZipFile(row['path']) as archive:
                front, wrist = [read_rgb_member(archive, key) for key in ('front_rgb', 'wrist_rgb')]
            assert front.shape == wrist.shape == (1300, 480, 640, 3)
            assert front.dtype == wrist.dtype == np.uint8
            assert float(data['fps']) == 25
            writers = [Writer(Path(str(prefix) + suffix), width, 480)
                       for suffix, width in (('_front.mp4', 640), ('_wrist.mp4', 640), ('_dual_rgb.mp4', 1280))]
            try:
                for f, w in zip(front, wrist):
                    for writer, rgb in zip(writers, (f, w, np.concatenate((f, w), axis=1))):
                        writer.write(rgb)
            finally:
                for writer in writers:
                    writer.close()
        save(Path(str(prefix) + '.json'), dict(
            source=row, frames=1300, fps=25, duration_s=52,
            layout='left FRONT RGB, right WRIST RGB',
            provenance='Actual saved post-augmentation training RGB; not simulator replay; H264 preview encoding'))
        print(f'EXPORTED training episode {idx}', flush=True)


def read_rgb_member(archive, key):
    import numpy as np
    with archive.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = { (1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0 }[version]
        shape, fortran, dtype = reader(stream)
        assert dtype == np.uint8 and not fortran
        array = np.empty(shape, dtype=dtype)
        buffer = memoryview(array).cast('B')
        offset = 0
        while offset < len(buffer):
            end = min(offset + 8 * 1024 * 1024, len(buffer))
            count = stream.readinto(buffer[offset:end])
            if not count:
                raise EOFError(key)
            offset += count
        assert not stream.read(1)
        print(f'Decoded {key}: {shape}', flush=True)
        return array


def inference(visual_config=None, module_setup=None):
    # Import Isaac Gym before torch via the original simulation module.
    path = REPO / 'high-level/float_ik/isaacgym_float_ik_a2w_basearn_pull_door_parallel.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('rgb500_capture_pull', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    import numpy as np

    reference = A / 'logs/door-policy-success/0914_pull_rgb_aug500_unlock25_traverse050_eval64/trial_000/batch_0000.out'
    line = reference.read_text().splitlines()[0]
    argv = shlex.split(line[line.index(SIM):])
    assert argv[1] == str(path)
    argv.remove('--headless')
    for key, value in {
        '--dp_log_path': OUT / 'inference.jsonl',
        '--rgb_visual_preview_dir': OUT / 'inference_previews',
    }.items():
        argv[argv.index(key) + 1] = str(value)
    if visual_config is not None:
        argv[argv.index('--rgb_visual_config') + 1] = str(visual_config)
    argv += ['--overview_video_path', str(OUT / 'inference_overview_source.mp4'),
             '--overview_video_fps', '25', '--overview_video_capture_stride', '2',
             '--overview_video_width', '1280', '--overview_video_height', '720']
    save(OUT / 'inference.command.json', dict(argv=argv, fresh_rollout=True,
         note='Same physical/visual seed as eval trial000, diffusion noise not independently fixed'))
    writers = [Writer(OUT / name, width, 480)
               for name, width in (('inference_front.mp4', 640), ('inference_wrist.mp4', 640),
                                   ('inference_dual_rgb.mp4', 1280))]
    active = [False]
    original_collect = mod.dc.collect_float_dp_policy_actions
    original_images = mod.dc.dp_image_inputs_from_cpu_cameras

    def capture_images(images, args):
        result = original_images(images, args)
        if active[0]:
            f, w = images['front_rgb'], images['wrist_rgb']
            assert f.shape == w.shape == (480, 640, 3)
            assert f.dtype == w.dtype == np.uint8
            assert result[1] is w and result[3] is f
            for writer, rgb in zip(writers, (f, w, np.concatenate((f, w), axis=1))):
                writer.write(rgb)
        return result

    def collect(*args, **kwargs):
        active[0] = True
        try:
            return original_collect(*args, **kwargs)
        finally:
            active[0] = False

    mod.dc.dp_image_inputs_from_cpu_cameras = capture_images
    mod.dc.collect_float_dp_policy_actions = collect
    if module_setup is not None:
        module_setup(mod)
    sys.argv = argv[1:]
    try:
        mod.main()
    finally:
        for writer in writers:
            writer.close()
    assert all(w.frames == 1300 for w in writers), [w.frames for w in writers]
    save(OUT / 'inference_capture.json', dict(frames=1300, fps=25,
         source='Already augmented RGB passed to the policy observation pipeline; no second augmentation',
         overview_timing='Overview at end of same sim step, policy cameras before physics; 20ms offset',
         physical_seed=715455575, visual_seed=815455575, chunk_size=100, action_horizon=25,
         inference_steps=10, fresh_rollout=True))


def compose():
    import cv2
    import numpy as np
    paths = ['inference_overview_source.mp4', 'inference_front.mp4', 'inference_wrist.mp4']
    caps = [cv2.VideoCapture(str(OUT / name)) for name in paths]
    assert all(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1300 for cap in caps)
    assert all(abs(cap.get(cv2.CAP_PROP_FPS) - 25) < 1e-5 for cap in caps)
    writer = Writer(OUT / 'inference_three_rgb.mp4', 1600, 720)
    try:
        for _ in range(1300):
            frames = [cap.read() for cap in caps]
            assert all(ok for ok, _ in frames)
            overview, front, wrist = [frame for _, frame in frames]
            frame = np.zeros((720, 1600, 3), np.uint8)
            frame[45:675, :1120] = cv2.resize(overview, (1120, 630))
            frame[:360, 1120:] = cv2.resize(front, (480, 360))
            frame[360:, 1120:] = cv2.resize(wrist, (480, 360))
            for title, xy in [('PI0.5 EVO | THIRD-PERSON', (15, 30)), ('FRONT RGB', (1130, 25)),
                              ('WRIST RGB', (1130, 385))]:
                cv2.putText(frame, title, xy, cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, title, xy, cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame[..., ::-1].copy())
    finally:
        writer.close()
        for cap in caps:
            cap.release()
    print('COMPOSED 1300 frames / 52 seconds', flush=True)


def verify():
    import cv2
    import numpy as np
    records = [json.loads(line) for line in (OUT / 'inference.jsonl').read_text().splitlines()]
    assert len(records) == 2600
    ref = A / 'logs/door-policy-success/0914_pull_rgb_aug500_unlock25_traverse050_eval64/trial_000/batch_0000.jsonl'
    with ref.open() as stream:
        ref_first = json.loads(next(stream))
    for key in ('ikpush_randomization', 'pull_task_geometry'):
        assert records[0][key] == ref_first[key], key
    ref_scene = json.loads(sorted((EXPERIMENT / 'eval64_unlock25_traverse050/previews/trial_000').glob('env00_front_*.json'))[0].read_text())
    new_scene = json.loads(sorted((OUT / 'inference_previews').glob('env00_front_*.json'))[0].read_text())
    for key in ('metadata', 'parameters', 'intensity', 'hdr'):
        assert ref_scene[key] == new_scene[key], key
    for key in ('seed', 'env_id', 'materials', 'indoor_room'):
        assert ref_scene['episode'][key] == new_scene['episode'][key], key
    max_open, max_distance = 0., -math.inf
    first_success = None
    for i, record in enumerate(records):
        assert record['step'] == i and record['controlled_env_id'] == 0
        for key in ('state', 'dp_action_raw'):
            assert len(record[key]) == 9 and np.isfinite(record[key]).all()
        angle = abs(math.degrees(record['door']['dof'][0]))
        geom = record['pull_task_geometry']
        direction = np.asarray(geom['traversal_direction_xy'][:2])
        distance = float(np.dot(np.asarray(record['base']['xy'])[:2] -
                              np.asarray(geom['door_plane_xy'])[:2], direction) / np.linalg.norm(direction))
        max_open = max(max_open, angle)
        max_distance = max(max_distance, distance)
        if angle >= 60 and distance >= .5 and first_success is None:
            first_success = i
    logs = (OUT / 'inference.log').read_text()
    assert 'Done after 2600 steps' in logs
    assert not any(s in logs for s in ('[Error]', 'Traceback (most recent call last)', 'Could not load visual mesh'))
    videos = []
    for path in sorted(OUT.glob('*.mp4')):
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened()
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        assert frames == 1300 and abs(fps - 25) < 1e-5, (path, frames, fps)
        samples = []
        for index in (0, 500, 1000, 1299):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            assert ok and frame.std() > 5 and 1 < frame.mean() < 254, (path, index)
            samples.append(dict(index=index, mean=float(frame.mean()), std=float(frame.std())))
        cap.release()
        videos.append(dict(name=path.name, frames=frames, fps=fps, duration_s=frames/fps,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(), samples=samples))
    assert len(videos) == 11, len(videos)
    save(OUT / 'verification.json', dict(passed=True, videos=videos,
        inference=dict(records=len(records), max_open_deg=max_open, max_traversal_m=max_distance,
                       success=first_success is not None, first_success_step=first_success),
        original_eval_success_rate=46/64,
        reference_scene_matches=True,
        note='Fresh one-episode video demonstration, not a new success-rate evaluation'))
    print('VERIFIED all 11 videos and full 2600-step policy log', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['dataset', 'inference', 'compose', 'verify'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((EXPERIMENT / 'experiment.json').read_text())['source_sha256']
    for name, expected in hashes.items():
        assert hashlib.sha256((REPO / 'high-level' / name).read_bytes()).hexdigest() == expected, name
    {'dataset': dataset, 'inference': inference, 'compose': compose, 'verify': verify}[args.stage]()


if __name__ == '__main__':
    main()
