"""Transactional LeRobot v3 export; no Hub uploads and no fabricated actions."""
import argparse
import os
from pathlib import Path
import uuid
from common import read, save, sha, transition, ACTION_NAMES, STATE_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    cfg, manifest = read(a.config), read(a.manifest)
    if a.out.exists():
        previous = read(a.out / 'pipeline_provenance.json')
        assert previous['manifest_sha256'] == sha(a.manifest), 'Existing dataset has different input episodes'
        print('DATASET_ALREADY_VERIFIED', a.out, flush=True)
        return
    import cv2
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
    assert CODEBASE_VERSION == 'v3.0', CODEBASE_VERSION
    r = cfg['render']
    views = r['views']
    camera_metadata = cfg['dataset'].get('include_camera_metadata', True)
    door_state = cfg['dataset'].get('include_door_state', True)
    features = {
        'observation.state': dict(dtype='float32', shape=(10,), names=STATE_NAMES),
        'action': dict(dtype='float32', shape=(10,), names=ACTION_NAMES),
        'source_step': dict(dtype='int64', shape=(1,), names=['physical_step']),
        'next.done': dict(dtype='bool', shape=(1,), names=['done']),
    }
    if door_state:
        features['observation.door_state'] = dict(dtype='float32', shape=(4,), names=['hinge_q','handle_q','hinge_qvel','handle_qvel'])
    for view in views:
        features['observation.images.' + view] = dict(dtype='video', shape=(r['height'], r['width'], 3), names=['height','width','channels'])
        if camera_metadata:
            features['observation.camera_c2w.' + view] = dict(dtype='float32', shape=(16,), names=None)
            features['observation.camera_K.' + view] = dict(dtype='float32', shape=(9,), names=None)
    staging = a.out.with_name(a.out.name + '.partial.' + uuid.uuid4().hex[:8])
    dataset = LeRobotDataset.create(repo_id=cfg['dataset']['repo_id'], root=staging, fps=r['fps'],
        robot_type='a2w_z1_prescribed_base_joint10', features=features,
        image_writer_threads=4, vcodec='h264', video_backend='pyav')
    expected = []
    provenance = []
    try:
        for entry in manifest['episodes']:
            raw = Path(entry['raw'])
            render_dir = Path(entry['render'])
            data, rendered = read(raw / 'trajectory.json'), read(render_dir / 'render_complete.json')
            assert rendered['passed'] and rendered['renderer'].startswith('MeisterRender/')
            assert sha(raw / 'trajectory.json') == entry['raw_sha256']
            for index, record in enumerate(rendered['frames']):
                i = record['source_index']
                state, action = transition(data, i)
                current = data['frames'][i]
                frame = {'observation.state': state, 'action': action,
                         'source_step': np.array([i], dtype=np.int64),
                         'next.done': np.array([index == len(rendered['frames'])-1], dtype=np.bool_),
                         'task': cfg['dataset']['task']}
                if door_state:
                    frame['observation.door_state'] = np.array(current['replay_door_dof_pos']+current['replay_door_dof_vel'], dtype=np.float32)
                for view in views:
                    info = record['images'][view]
                    path = render_dir / info['path']
                    assert sha(path) == info['sha256'], path
                    image = cv2.imread(str(path))
                    assert image.shape == (r['height'], r['width'], 3)
                    frame['observation.images.' + view] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if camera_metadata:
                        frame['observation.camera_c2w.' + view] = np.array(record['camera_c2w'][view], dtype=np.float32).ravel()
                        frame['observation.camera_K.' + view] = np.array(rendered['cameras'][view]['K'], dtype=np.float32).ravel()
                dataset.add_frame(frame)
                expected.append((state.copy(), action.copy(), i))
            dataset.save_episode()
            provenance.append(dict(episode=entry['id'], physics_seed=entry['physics_seed'], scene_seed=entry['scene_seed'],
                raw_sha256=entry['raw_sha256'], render_sha256=sha(render_dir / 'render_complete.json'),
                source_episode=str(raw), source_render=str(render_dir), physical_qc=entry['qc'],
                augmentation=rendered.get('augmentation')))
        if cfg['render'].get('image_augmentation', {}).get('enabled'):
            hashes = {p['augmentation']['lighting']['material_sha256'] for p in provenance}
            assert len(hashes) == 1, 'Material definitions differ between episodes'
            assert all(p['augmentation']['lighting']['materials_unchanged'] for p in provenance)
        dataset.finalize()
    finally:
        dataset.stop_image_writer()
    # Verify through the installed official reader, not just a hand-made schema.
    check = LeRobotDataset(cfg['dataset']['repo_id'], root=staging, video_backend='pyav')
    assert len(check) == len(expected)
    assert check.num_episodes == len(provenance)
    for j, (state, action, i) in enumerate(expected):
        row = check.hf_dataset[j]
        assert np.allclose(np.asarray(row['observation.state']), state, atol=1e-7)
        assert np.allclose(np.asarray(row['action']), action, atol=1e-7)
        assert int(np.asarray(row['source_step']).item()) == i
    for i in sorted({0, len(check)//2, len(check)-1}):
        sample = check[i]
        for view in views:
            assert tuple(sample['observation.images.' + view].shape) == (3, r['height'], r['width'])
    video_frames = 0
    for video in staging.glob('videos/**/*.mp4'):
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-r['fps']) < 1e-5
        n = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape == (r['height'], r['width'], 3)
            n += 1
        cap.release()
        video_frames += n
    assert video_frames == len(expected)*len(views)
    save(staging / 'pipeline_provenance.json', dict(
        manifest_sha256=sha(a.manifest), format=CODEBASE_VERSION, episodes=provenance,
        action_semantics='State at post-step i; actual joint targets + prescribed base vx/vy/vyaw for step i+1. Joint targets are absolute radians.',
        temporal_note='50 Hz controls sampled at output FPS; labels are the next 20 ms command, not a held-action replay claim for the full RGB interval. Full 50 Hz controls are retained in raw trajectories.',
        state_semantics='Previous applied base command followed by measured joint1..6 and gripper.',
        camera_semantics='c2w: row-major Blender camera-to-render-world, +X right/+Y up/-Z forward; K uses pixel-center convention.',
        limitations=['Prescribed floating base, not full-body wheel dynamics.',
                    'Room props are render-only; full-recorded-trajectory AABB clearance checked.',
                    'Joint10 requires matching policy/deployment configuration; not interchangeable with Joint9.'],
        readback_verified=True, numeric_rows_checked=len(expected), video_frames_decoded=video_frames,
        production_resolution=(r['width'] >= 640 and r['height'] >= 480)))
    save(staging / 'file_sha256.json', {str(f.relative_to(staging)): sha(f) for f in sorted(staging.rglob('*')) if f.is_file()})
    os.rename(staging, a.out)
    print('LEROBOT_VERIFIED', len(expected), len(provenance), a.out, flush=True)


if __name__ == '__main__':
    main()
