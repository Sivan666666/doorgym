"""Bake physical poses into a packed scene using the existing asset adapter."""
import argparse
from pathlib import Path
import runpy
import sys
from common import read, save, indices, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    cfg = read(a.config)
    data = read(a.raw / 'trajectory.json')
    selected = indices(data, cfg['render']['fps'])
    out = a.raw / a.name
    out.mkdir()
    scene_cfg = dict(cfg['scene'], seed=a.seed)
    save(out / 'scene_config.json', scene_cfg)
    sys.argv = [cfg['tools']['scene_builder'], '--', '--trial', str(a.raw),
                '--bake-only', '--white-walls', '--furnished-room', '--output-name', a.name,
                '--base-scene', cfg['scene']['base_blend'], '--scene-config', str(out / 'scene_config.json'),
                '--stride', str(data['fps'] // cfg['render']['fps']), '--end-step', str(len(data['frames']) - 1),
                '--views', ','.join(cfg['render']['views']), '--samples', str(cfg['render']['samples']),
                '--width', str(cfg['render']['width'])]
    runpy.run_path(cfg['tools']['scene_builder'], run_name='__main__')
    import bpy
    from mathutils import Quaternion, Vector
    from math import acos
    def pose(v):
        m = Quaternion((v[6], *v[3:6])).to_matrix().to_4x4()
        m.translation = Vector(v[:3])
        return m
    scene = bpy.context.scene
    keep = {'Replay ' + n for n in cfg['render']['views']}
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA' and obj.name not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.resolution_x = cfg['render']['width']
    scene.render.resolution_y = cfg['render']['height']
    scene.render.resolution_percentage = 100
    from augmentation import randomize_lighting
    lighting_report = randomize_lighting(bpy, scene_cfg, a.seed)
    save(out / 'lighting_augmentation.json', lighting_report)
    convert = pose(data['frames'][0]['replay_door_root_state']).inverted()
    max_error = max_angle = 0.
    for number, i in enumerate(selected, 1):
        scene.frame_set(number)
        bpy.context.view_layer.update()
        for role, records in data['frames'][i]['bodies'].items():
            for name, v in records.items():
                key = name if role == 'door' else 'robot/' + name
                if key not in bpy.data.objects:
                    continue
                expected = convert @ pose(v)
                actual = bpy.data.objects[key].matrix_world
                error = (expected.translation - actual.translation).length
                angle = 2 * acos(min(1., abs(expected.to_quaternion().normalized().dot(actual.to_quaternion().normalized()))))
                max_error = max(max_error, error)
                max_angle = max(max_angle, angle)
                assert error < 1e-4 and angle < .002, (number, key, error, angle)
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'))
    save(out / 'scene_verified.json', dict(passed=True, frames=len(selected),
        source_sha256=sha(a.raw / 'trajectory.json'), scene_sha256=sha(out / 'scene.blend'),
        source_indices=selected, max_position_error_m=max_error, max_orientation_error_rad=max_angle,
        config=scene_cfg, lighting_augmentation=lighting_report, blender_version=bpy.app.version_string))


if __name__ == '__main__':
    main()
