"""MeisterRender Pipeline adapter. Upstream source is pinned and unmodified.

Runs its BlendLoader, processor, Cycles renderer and compositor synchronously,
so Python exceptions propagate (the upstream queue API can wait indefinitely
after a child failure). Our lazy-init override makes render settings explicit.
"""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import types
from common import read, save, sha, digest
from augmentation import doorman, photo_plan, apply_photo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--scene', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--preview-frames', help='Comma-separated output frame indices; never writes render_complete.json')
    a = p.parse_args()
    cfg = read(a.config)
    r = cfg['render']
    pin = subprocess.check_output(['git', '-C', cfg['tools']['meister_root'], 'rev-parse', 'HEAD'], text=True).strip()
    assert pin == cfg['tools']['meister_commit'], 'Unexpected MeisterRender revision'
    scene_report = read(a.scene.parent / 'scene_verified.json')
    preview = set(map(int, a.preview_frames.split(','))) if a.preview_frames else None
    if preview is not None:
        assert preview and min(preview) >= 0 and max(preview) < len(scene_report['source_indices'])
    assert sha(a.scene) == scene_report['scene_sha256']
    augmentation = photo_plan(cfg, scene_report['config']['seed'])
    augmenter = doorman(cfg) if augmentation['enabled'] else None
    signature = digest(dict(scene=sha(a.scene), render=r, meister=pin, augmentation=augmentation))
    a.out.mkdir(parents=True, exist_ok=True)
    save(a.out / 'augmentation.json', dict(photometric=augmentation,
         lighting=scene_report.get('lighting_augmentation'), settings=r.get('image_augmentation')))
    existing = a.out / 'render_identity.json'
    if existing.exists():
        assert read(existing)['signature'] == signature, 'Refuse stale frame cache'
    else:
        save(existing, dict(signature=signature, meister_commit=pin, settings=r))
    sys.path.insert(0, cfg['tools']['meister_root'])
    import bpy
    import cv2
    import numpy as np
    from omegaconf import OmegaConf
    from pipeline import Pipeline
    from data import InputPack
    from utils.blender_utils.renderer_utils import init_renderer_output
    pipeline_cfg = OmegaConf.load(str(Path(cfg['tools']['meister_root']) / 'configs/blender_workbench.yml')).engine.pipeline
    pipeline_cfg.renderer.type = 'BlenderPathtracingRenderer'
    pipe = Pipeline(pipeline_cfg)

    def initialize(renderer):
        s = bpy.context.scene
        s.render.engine = 'CYCLES'
        s.render.threads_mode = 'FIXED'
        s.render.threads = r.get('cpu_threads', 4)
        s.render.use_persistent_data = True
        s.cycles.samples = r['samples']
        s.cycles.use_denoising = r.get('denoise', True)
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'OPTIX'
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == 'OPTIX']
        assert devices, 'No OptiX device visible'
        for d in prefs.devices:
            d.use = d == devices[0]
        s.cycles.device = 'GPU'
        init_renderer_output(renderer.shared_memory_path)
        s.view_settings.view_transform = r.get('view_transform', 'AgX')
        renderer.first_frame_init = True
    pipe.renderer._lazy_render_init = types.MethodType(initialize, pipe.renderer)

    # Explicit scene-only load avoids upstream Load_scene rendering an implicit
    # first frame with default 100x100 intrinsics before camera configuration.
    pack = InputPack()
    pack.scene_only = True
    pack.scene_file_path['blend'] = str(a.scene)
    pipe.scene = pipe.loader.scene_load(None, pack)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = r['width'], r['height']
    cameras = {}
    for name, camera in pipe.scene.all_cameras.items():
        view = name.removeprefix('Replay ')
        assert view in r['views'], name
        camera.image_width, camera.image_height = r['width'], r['height']
        cameras[view] = camera
    assert set(cameras) == set(r['views'])
    calibration = {v: dict(K=c.get_intrinsics_as_K_matrix().tolist(),
                           width=r['width'], height=r['height'],
                           pose_convention='Blender camera-to-world: +X right, +Y up, -Z viewing direction')
                   for v, c in cameras.items()}
    save(a.out / 'cameras.json', calibration)
    frames = []
    try:
        for number, source_index in enumerate(scene_report['source_indices']):
            if preview is not None and number not in preview:
                continue
            marker = a.out / 'frames' / f'{number:06d}.json'
            if marker.exists():
                saved = read(marker)
                if (saved['signature'] == signature and saved['source_index'] == source_index
                        and all((a.out / im['path']).is_file() and sha(a.out / im['path']) == im['sha256']
                                for im in saved['images'].values())):
                    frames.append(saved)
                    continue
            scene.frame_set(number + 1)
            bpy.context.view_layer.update()
            poses = {v: c.get_matrix_world().tolist() for v, c in cameras.items()}
            # Preserve baked quaternion animations. Empty action leaves their
            # evaluated transforms intact; the processor's frame interval is
            # explicitly one recorded snapshot, never an interpolated subframe.
            scene.frame_start = number + 1
            scene.frame_end = number + 2
            pack = InputPack()
            pack.action_only = True
            result = pipe.run(pack)
            images = {}
            for v in r['views']:
                rgb = result.rgb_image['Replay ' + v][0]
                assert rgb.shape == (r['height'], r['width'], 3) and rgb.dtype == np.uint8
                assert rgb.std() > 1 and np.isfinite(rgb).all()
                if augmentation['enabled'] and number in r.get('clean_preview_frames', [0, 240, 490]):
                    clean = a.out / 'clean_preview' / v / f'{number:06d}.png'
                    clean.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(clean), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                rgb = apply_photo(augmenter, rgb, augmentation, number, v)
                path = a.out / 'rgb' / v / f'{number:06d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.stem + '.tmp.png')
                assert cv2.imwrite(str(temporary), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                os.replace(temporary, path)
                images[v] = dict(path=str(path.relative_to(a.out)), sha256=sha(path),
                                 mean=float(rgb.mean()), std=float(rgb.std()),
                                 clipped_low_fraction=float((rgb == 0).mean()),
                                 clipped_high_fraction=float((rgb == 255).mean()))
            saved = dict(signature=signature, frame_index=number, source_index=source_index,
                         timestamp_s=number/r['fps'], camera_c2w=poses, images=images)
            save(marker, saved)
            frames.append(saved)
            if number % 25 == 0:
                print('MEISTER_FRAME', number, len(scene_report['source_indices']), flush=True)
    finally:
        if Path(pipe.renderer.shared_memory_path).exists():
            shutil.rmtree(pipe.renderer.shared_memory_path)
    save(a.out / ('preview_complete.json' if preview is not None else 'render_complete.json'), dict(passed=True, signature=signature,
         meister_commit=pin, renderer='MeisterRender/BlenderPathtracingRenderer',
         frames=frames, config=r, cameras=calibration, augmentation=read(a.out / 'augmentation.json'),
         scene_sha256=sha(a.scene)))


if __name__ == '__main__':
    main()
