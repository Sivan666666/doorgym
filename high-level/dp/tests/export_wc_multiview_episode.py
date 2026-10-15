"""Package one synchronized Blender RGB observation episode and review videos.

This samples an existing physical rollout, not a fresh rollout or policy run.
The archive has observations and states, not invented training action labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import cv2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--trial', required=True, type=Path)
    p.add_argument('--output-name',default='blender_white_walls_multiview')
    p.add_argument('--overview-name',default='blender_white_walls')
    p.add_argument('--ffmpeg', default='/home/ps/miniconda3/envs/lerobot/bin/ffmpeg')
    a = p.parse_args()
    source = a.trial / 'trajectory.json'
    trajectory = read(source)
    assert Path(a.output_name).name == a.output_name
    assert Path(a.overview_name).name == a.overview_name
    multi = a.trial / a.output_name
    overview = a.trial / a.overview_name
    dirs = {'front': multi, 'wrist': multi, 'overview': overview}
    reports = {name: read(folder / 'render_complete_0_None.json') for name, folder in dirs.items()}
    reference = reports['front']
    room_files=[]
    if reference.get('room_clearance_sha256'):
        assert digest(multi / 'room_clearance.json') == reference['room_clearance_sha256']
        room_files=['room_clearance.json']
    indices = reference['frames']
    n = len(indices)
    fps = reference['output_fps']
    stride = round(trajectory['fps'] / fps)
    assert indices == list(range(0, len(trajectory['frames']), stride))
    for name, report in reports.items():
        assert name in report['views']
        for key in ('source', 'frames', 'output_fps', 'source_fps', 'geometry_scale',
                    'actual_physics_body_poses', 'white_walls', 'render_only_walls', 'samples'):
            assert report[key] == reference[key], (name, key)
        assert report['white_walls'] and report['actual_physics_body_poses']
        for key in ('scene_variant','room_clearance_sha256'):
            assert report.get(key) == reference.get(key), (name, key)
    provenance = []
    for name, folder in dirs.items():
        verification = read(folder / 'verification.json')
        spec = next(v for v in verification['videos'] if v['file'] == f'{name}_physical_cycles.mp4')
        assert spec['fps'] == fps and spec['frames'] == n
        assert digest(folder / spec['file']) == spec['sha256']
        recorded = {v['file']: v['sha256'] for v in read(folder / 'frame_sha256.json')}
        paths = sorted((folder / 'frames' / name).glob('*.png'))
        assert [v.name for v in paths] == [f'{i:04d}.png' for i in range(n)]
        for path in paths:
            sha = digest(path)
            assert sha == recorded[str(path.relative_to(folder))], path
            provenance.append({'file': f'rgb/{name}/{path.name}', 'sha256': sha})

    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    videos = []
    for label, views in [('dual_rgb', ['front', 'wrist']),
                         ('three_rgb', ['front', 'wrist', 'overview'])]:
        path = multi / f'episode_000_{label}.mp4'
        inputs = []
        filters = []
        for i, view in enumerate(views):
            inputs += ['-i', str(dirs[view] / f'{view}_physical_cycles.mp4')]
            filters.append(f'[{i}:v]scale=640:480,setsar=1,drawtext=fontfile={font}:'
                           f'text={view.upper()}:x=10:y=10:fontsize=20:fontcolor=white:'
                           f'box=1:boxcolor=black@0.5[v{i}]')
        filters.append(''.join(f'[v{i}]' for i in range(len(views))) +
                       f'hstack=inputs={len(views)}[out]')
        if not path.exists():
            subprocess.run([a.ffmpeg, '-hide_banner', '-loglevel', 'error', '-n',
                            '-filter_complex_threads', '2', *inputs,
                            '-filter_complex', ';'.join(filters), '-map', '[out]',
                            '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                            '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            str(path)], check=True)
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        assert abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 1e-5
        decoded = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            assert im.shape == (480, 640 * len(views), 3) and im.std() > 3
            if decoded == 240:
                assert cv2.imwrite(str(multi / f'episode_000_{label}_preview.jpg'), im)
            decoded += 1
        cap.release()
        assert decoded == n
        videos.append(dict(file=path.name, views_left_to_right=views, frames=n,
                           fps=fps, duration_s=n/fps, dimensions=[640 * len(views), 480],
                           sha256=digest(path)))

    states = [dict(frame_index=i, source_step=step, timestamp_s=step/trajectory['fps'],
                   rgb={view: f'rgb/{view}/{i:04d}.png' for view in dirs},
                   state=trajectory['frames'][step]) for i, step in enumerate(indices)]
    metadata = dict(episode_id='wc_'+reference.get('scene_variant','white_walls')+'_000', source_trajectory_sha256=digest(source),
                    source_trajectory=str(source), source_fps=trajectory['fps'], fps=fps,
                    frames=n, duration_s=n/fps, sampling_stride=stride,
                    rgb_dimensions={'front': [640, 480], 'wrist': [640, 480],
                                    'overview': [reports['overview']['width'], reports['overview']['width']*3//4]},
                    scene_variant=reference.get('scene_variant','white_walls'),
                    room_clearance_sha256=reference.get('room_clearance_sha256'),
                    timestamp_origin='First recorded physical snapshot is t=0',
                    fresh_physics_rollout=False, fresh_policy_inference=False,
                    rgb_post_augmentation=False, blender_white_walls=True, walls_render_only=True,
                    actions_included=False, train_ready_dataset=False,
                    overview_reuse='Same white-wall scene, exact source steps; PNG hashes verified.',
                    state_pose_convention='Original simulation world; quaternion xyzw. Camera poses relative to robot base.',
                    render_world='Inverse of first replay_door_root_state applied to all physical world poses.',
                    camera_note='Unchanged recorded front/wrist mounts; front tilted up and wrist partially self-occluded.',
                    limitations=['Base motion is prescribed, not full-body dynamics.',
                                 'White walls were not added to the physical collision simulation.',
                                 'This observation sample does not contain controller action labels.'],
                    videos=videos)
    save(multi / 'episode_000.json', metadata)
    save(multi / 'episode_000_states.json', states)
    save(multi / 'episode_000_rgb_sha256.json', provenance)
    archive = multi / 'episode_000_rgb_states.zip'
    assert not archive.exists(), 'Do not overwrite an existing episode archive'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name in ['episode_000.json', 'episode_000_states.json', 'episode_000_rgb_sha256.json', *room_files]:
            z.write(multi / name, name)
        for view, folder in dirs.items():
            for i in range(n):
                z.write(folder / 'frames' / view / f'{i:04d}.png', f'rgb/{view}/{i:04d}.png')
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == 3 + len(room_files) + n * 3
        assert z.testzip() is None
    save(multi / 'episode_000_verification.json', dict(
        source_steps_aligned=True, png_hashes_verified=n * 3, archive_crc_verified=True,
        archive=dict(file=archive.name, bytes=archive.stat().st_size, sha256=digest(archive)),
        videos=videos))
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
