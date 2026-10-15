"""Check a reopened baked Blender animation against all recorded poses."""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


def pose(v):
    m = Quaternion((v[6], *v[3:6])).to_matrix().to_4x4()
    m.translation = Vector(v[:3])
    return m


source = Path(sys.argv[sys.argv.index('--') + 1])
ep = json.loads(source.read_text())['episode']
convert = pose(ep['replay_door_root_state'][0]).inverted()
errors = []
for i in range(len(ep['replay_root_state'])):
    bpy.context.scene.frame_set(i + 1)
    bpy.context.view_layer.update()
    ee = bpy.data.objects['robot/ee_gripper_link'].matrix_world.translation
    expected = convert @ Vector(ep['replay_ee_pos'][i])
    spindle = bpy.data.objects['lever_handle'].matrix_world.translation
    target = (convert @ pose(ep['replay_door_root_state'][i]) @ Matrix.Translation(Vector((-.463, 0, 0))) @
              Matrix.Rotation(ep['replay_door_dof_pos'][i][0], 4, 'Z') @ Vector((.868, -.058, .985)))
    errors.append([(ee - expected).length, (spindle - target).length])
assert max(max(row) for row in errors) < 1e-4
images = [im for im in bpy.data.images if im.source == 'FILE']
assert len(images) == 3 and all(im.packed_file for im in images)
result = dict(passed=True, frames_checked=len(errors),
              max_ee_error_m=max(row[0] for row in errors),
              max_spindle_error_m=max(row[1] for row in errors), packed_textures=len(images))
(Path(bpy.data.filepath).parent / 'animation_verification.json').write_text(json.dumps(result, indent=2))
print('ANIMATION_VERIFIED', result, flush=True)
