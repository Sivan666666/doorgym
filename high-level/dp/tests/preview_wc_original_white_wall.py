"""Isolated WC-original-color / white-wall preview; no training changes.

Only the material and lighting methods of this preview process are overridden.
Original atlas pixels are cropped at fixed UV-safe coordinates, never recolored.
Physics geometry, camera poses, actions and source material files are untouched.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import export_rgb500_review_videos as capture

OUT = capture.A / 'dp/result/rgb500_wc_original_whitewall_20260918'
SOURCE = OUT / 'source_textures'
EXPECTED = {
    'door_leaf_opaque_basecolor.png': '0a9abb67bf95c084035f913872e76c11645337b62ad6c2598a22bda79b4243f1',
    'frame_basecolor.png': '7ab7b67dbec202b294434c31d1a6e689c1f4206d18746a4ddfcbfbd57603666e',
    'lever_handle_basecolor.png': 'f7b041394e4ccabdce49e34dd06a6db8238e0d8b840e1b8f1f4409a185f24037',
}


def reference_scene():
    root = capture.EXPERIMENT / 'eval64_unlock25_traverse050/previews/trial_000'
    return json.loads(sorted(root.glob('env00_front_*.json'))[0].read_text())


def materials():
    desc = {}
    for group, name, crop in (
        ('panel', 'door_leaf_opaque_basecolor.png', [140, 110, 380, 740]),
        ('frame', 'frame_basecolor.png', [12, 260, 48, 930]),
        ('handle', 'lever_handle_basecolor.png', [225, 270, 330, 450]),
    ):
        desc[group] = dict(kind='wc_original_pixels', path=str(SOURCE / name),
                           sha256=EXPECTED[name], crop_xyxy_1024=crop,
                           recolor=False, randomization=False, name='fixed_original_wc_' + group)
    for group in ('wall', 'far_wall'):
        desc[group] = dict(kind='fixed_rgb', name='fixed_white_' + group,
                           rgb=[255, 255, 255], randomization=False)
    # Keep the previous floor appearance exactly, but freeze its descriptor;
    # no floor sampling is performed by this preview, even with another seed.
    desc['floor'] = copy.deepcopy(reference_scene()['episode']['materials']['floor'])
    return desc


def prepare():
    import yaml
    OUT.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
    cfg = yaml.safe_load((capture.EXPERIMENT / 'rgb_config.yaml').read_text())
    cfg['image_augmentation']['enabled'] = False
    cfg['lighting'].update(enabled=False, intensity=[3000, 3000], rotation_deg=[0, 0])
    cfg['materials']['wc_reference']['texture_root'] = str(SOURCE)
    room = reference_scene()['episode']['indoor_room']
    cfg['scene']['indoor_room']['far_wall_distance_m'] = [room['far_wall_distance_m']] * 2
    cfg['scene']['indoor_room']['ceiling_gray_range'] = [255, 256]
    cfg['fixed_preview'] = dict(
        material_sampling=False, materials=materials(),
        lighting='Fixed neutral white key/fill; no HDR illumination',
        original_source='/home/sivan/whole_body/visual_whole_body/doortwin/wc/assets/textures',
        texture_mapping='Fixed clean atlas crops on original WC_nomenkuang geometry; no recoloring',
        physics_and_cameras_unchanged=True,
    )
    (OUT / 'rgb_fixed.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    capture.save(OUT / 'appearance.json', cfg['fixed_preview'])
    return OUT / 'rgb_fixed.yaml'


def install_fixed_appearance(mod):
    import cv2
    import numpy as np
    import rgb_doorman
    cls = rgb_doorman.VisualRandomizer
    original_texture, original_update = cls.texture, cls.update
    descriptions = materials()

    def fixed_materials(self, index):
        if index not in self.envs:
            self.envs[index] = copy.deepcopy(descriptions)
        return self.envs[index]

    def fixed_texture(self, desc):
        kind = desc.get('kind')
        if kind == 'fixed_rgb':
            return np.full((4, 4, 3), desc['rgb'], dtype=np.uint8)
        if kind != 'wc_original_pixels':
            return original_texture(self, desc)
        key = json.dumps(desc, sort_keys=True)
        if key not in self.textures:
            path = Path(desc['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == desc['sha256']
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            x, y, X, Y = desc['crop_xyxy_1024']
            crop = image[y*h//1024:Y*h//1024, x*w//1024:X*w//1024]
            assert crop.size and crop.mean() > 1
            self.textures[key] = cv2.resize(crop, (512, 1024), interpolation=cv2.INTER_LINEAR)
        return self.textures[key]

    def fixed_update(self, gym, sim, gymapi, step, dt):
        original_update(self, gym, sim, gymapi, step, dt)
        if not getattr(self, '_fixed_neutral_light_installed', False):
            for i, (color, ambient, direction) in enumerate((
                ((.65, .65, .65), (.45, .45, .45), (1., -.5, 1.)),
                ((.25, .25, .25), (0., 0., 0.), (-1., .5, 1.)),
                ((0., 0., 0.), (0., 0., 0.), (0., 0., 1.)),
                ((0., 0., 0.), (0., 0., 0.), (0., 0., 1.)),
            )):
                gym.set_light_parameters(sim, i, gymapi.Vec3(*color),
                                         gymapi.Vec3(*ambient), gymapi.Vec3(*direction))
            self._fixed_neutral_light_installed = True

    cls.environment_materials = fixed_materials
    cls.texture = fixed_texture
    cls.update = fixed_update


def verify():
    import cv2
    import numpy as np
    capture_metadata = json.loads((OUT / 'inference_capture.json').read_text())
    capture_metadata.update(
        source='Rendered policy camera RGB; image augmentation disabled for fixed-material preview',
        appearance_config=str(OUT / 'rgb_fixed.yaml'),
        appearance_manifest=str(OUT / 'appearance.json'),
    )
    capture.save(OUT / 'inference_capture.json', capture_metadata)
    records = [json.loads(s) for s in (OUT / 'inference.jsonl').read_text().splitlines()]
    assert len(records) == 2600
    ref = capture.A / 'logs/door-policy-success/0914_pull_rgb_aug500_unlock25_traverse050_eval64/trial_000/batch_0000.jsonl'
    with ref.open() as f:
        first = json.loads(next(f))
    for key in ('ikpush_randomization', 'pull_task_geometry'):
        assert records[0][key] == first[key], key
    opened, traversed, success_step = 0., -math.inf, None
    for i, r in enumerate(records):
        assert r['step'] == i and r['controlled_env_id'] == 0
        for key in ('state', 'dp_action_raw'):
            assert len(r[key]) == 9 and np.isfinite(r[key]).all()
        angle = abs(math.degrees(r['door']['dof'][0]))
        g = r['pull_task_geometry']
        d = np.asarray(g['traversal_direction_xy'][:2])
        distance = float(np.dot(np.asarray(r['base']['xy'])[:2] - np.asarray(g['door_plane_xy'])[:2], d) / np.linalg.norm(d))
        opened, traversed = max(opened, angle), max(traversed, distance)
        if angle >= 60 and distance >= .5 and success_step is None:
            success_step = i
    scenes = sorted((OUT / 'inference_previews').glob('env00_*_*.json'))
    assert len(scenes) >= 6
    for p in scenes:
        scene = json.loads(p.read_text())
        assert scene['parameters'] == {}
        assert scene['episode']['physics_before'] == scene['episode']['physics_after']
        assert scene['episode']['materials'] == materials()
        assert scene['episode']['indoor_room']['ceiling_rgb'] == [255] * 3
    log = (OUT / 'inference.log').read_text()
    assert 'Done after 2600 steps' in log
    assert not any(t in log for t in ('[Error]', 'Traceback (most recent call last)', 'Could not load visual mesh'))
    videos = []
    for path in sorted(OUT.glob('*.mp4')):
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1300
        assert abs(cap.get(cv2.CAP_PROP_FPS) - 25) < 1e-5
        for frame in (0, 500, 1000, 1299):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            assert ok and image.std() > 5 and 1 < image.mean() < 254
        cap.release()
        videos.append(dict(name=path.name, frames=1300, fps=25,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(videos) == 5
    capture.save(OUT / 'verification.json', dict(passed=True, videos=videos,
        fixed_materials=True, white_walls=True, rgb_augmentation=False, neutral_fixed_lighting=True,
        unchanged_physical_start_and_task_geometry=True,
        max_open_deg=opened, max_traversal_m=traversed, success=success_step is not None,
        first_success_step=success_step, training_changed=False))
    print('VERIFIED fixed materials, white walls, source pixels, physics, logs and five videos', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'inference', 'compose', 'verify'])
    args = parser.parse_args()
    capture.OUT = OUT
    for name, expected in json.loads((capture.EXPERIMENT / 'experiment.json').read_text())['source_sha256'].items():
        assert hashlib.sha256((capture.REPO / 'high-level' / name).read_bytes()).hexdigest() == expected, name
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'inference':
        config = prepare()
        capture.inference(visual_config=config, module_setup=install_fixed_appearance)
    elif args.stage == 'compose':
        capture.compose()
    else:
        verify()


if __name__ == '__main__':
    main()
