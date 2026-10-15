"""Full recorded expert pull trajectory, re-rendered with UV-mapped WC assets.

Blender-only kinematic appearance replay; not a physics or policy evaluation.
The rendering copy of the new door is fitted to the old hinge/spindle geometry.
Source assets and recorded robot/camera trajectories remain unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


def pose(v):
    m = Quaternion((v[6], *v[3:6])).to_matrix().to_4x4()
    m.translation = Vector(v[:3])
    return m


def origin(v):
    m = Euler(v['rpy'], 'XYZ').to_matrix().to_4x4()
    m.translation = Vector(v['xyz'])
    return m


def material(name, color, metal, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Metallic'].default_value = metal
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--scene-data', type=Path, required=True)
    p.add_argument('--door-blend', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--bake-only', action='store_true', help='Save all 1300 original poses as an editable 25 fps animation.')
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    a.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(a.scene_data.read_text())
    ep, model = data['episode'], data['models']['robot']
    bpy.ops.wm.open_mainfile(filepath=str(a.door_blend))
    scene = bpy.context.scene
    scene.cycles.samples = a.samples
    scene.cycles.max_bounces = 6
    scene.cycles.transmission_bounces = 4
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    chosen = next(d for d in prefs.devices if d.type == 'OPTIX')
    for device in prefs.devices:
        device.use = device == chosen
    scene.cycles.device = 'GPU'

    # Fit the *rendering copy*, not source assets. Scale link-local coordinates
    # before rotations so opening never shears or changes the panel's size.
    sx, sy, sz = .868 / .8613, .058 / .043, .985 / .922
    scale = Vector((sx, sy, sz))
    frame = bpy.data.objects['frame']
    leaf = bpy.data.objects['door_leaf']
    mount = bpy.data.objects['handle_mount']
    handle = bpy.data.objects['lever_handle']
    for link in (frame, leaf, mount, handle):
        for obj in link.children:
            if obj.type == 'MESH':
                for v in obj.data.vertices:
                    v.co = Vector((v.co.x * sx, v.co.y * sy, v.co.z * sz))
        link.location = Vector((link.location.x * sx, link.location.y * sy, link.location.z * sz))
    old_hinge = Vector((-.463, 0, 0))
    frame.location = old_hinge - leaf.location
    leaf_origin = leaf.matrix_basis.copy()
    handle_origin = handle.matrix_basis.copy()
    fixed_world = pose(ep['replay_door_root_state'][0])
    world_to_display = fixed_world.inverted()

    # Enlarge the neutral room; retain the static preview's lighting/materials.
    for obj in list(bpy.data.objects):
        if obj.name.startswith('Preview wall') or obj.name == 'Preview rear wall':
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.objects['Preview floor'].location.z = -.04
    plaster = bpy.data.materials['Preview only - neutral white plaster']
    def cube(name, position, size):
        bpy.ops.mesh.primitive_cube_add(size=1, location=position)
        obj = bpy.context.object
        obj.name, obj.scale = name, size
        obj.data.materials.append(plaster)
        return obj
    bpy.context.view_layer.update()
    bounds = [o.matrix_world @ Vector(c) for o in frame.children if o.type == 'MESH' for c in o.bound_box]
    lo, hi = min(p.x for p in bounds), max(p.x for p in bounds)
    top = max(p.z for p in bounds)
    partitions = [cube('Cutaway side wall L', ((lo - 4) / 2, .1, 1.65), (lo + 4, .12, 3.3)),
                  cube('Cutaway side wall R', ((hi + 4) / 2, .1, 1.65), (4 - hi, .12, 3.3)),
                  cube('Cutaway header', ((hi + lo) / 2, .1, (top + 3.3) / 2), (hi - lo, .12, 3.3 - top))]
    cube('Rear room wall', (0, 4.3, 1.65), (12, .12, 3.3))

    materials = dict(silver=material('Robot brushed silver', (.52, .55, .58), .82, .27),
                     rubber=material('Robot rubber', (.015, .018, .022), 0, .78),
                     black=material('Robot camera mounts', (.014, .016, .018), .45, .3))
    links = {}
    roots = set(model['links']) - {j['child'] for j in model['joints']}
    for name in model['links']:
        obj = bpy.data.objects.new('robot/' + name, None)
        scene.collection.objects.link(obj)
        links[name] = obj
    for joint in model['joints']:
        obj = links[joint['child']]
        obj.parent = links[joint['parent']]
        obj.matrix_basis = origin(joint['origin'])
    for name, visuals in model['links'].items():
        for v in visuals:
            if 'mesh' in v:
                bpy.ops.wm.ply_import(filepath=v['mesh'])
                obj, sizes = bpy.context.object, v['scale']
                for poly in obj.data.polygons:
                    poly.use_smooth = True
            elif 'box' in v:
                bpy.ops.mesh.primitive_cube_add(size=1)
                obj, sizes = bpy.context.object, v['box']
            elif 'cylinder' in v:
                bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=v['cylinder'][0], depth=v['cylinder'][1])
                obj, sizes = bpy.context.object, [1] * 3
            elif 'sphere' in v:
                bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=v['sphere'])
                obj, sizes = bpy.context.object, [1] * 3
            else:
                raise ValueError(v)
            obj.name = 'robot/' + name + '/' + v['name']
            obj.parent = links[name]
            obj.matrix_basis = origin(v['origin']) @ Matrix.Diagonal((*sizes, 1))
            key = 'rubber' if 'wheel' in name.lower() else 'black' if 'mount' in name else 'silver'
            obj.data.materials.clear()
            obj.data.materials.append(materials[key])

    cameras = {}
    for name in ('overview', 'front', 'wrist'):
        camera = bpy.data.cameras.new('Replay ' + name)
        camera.sensor_fit = 'HORIZONTAL'
        camera.sensor_width = 36
        camera.lens = 36 / (2 * math.tan(math.radians(55) / 2))
        camera.clip_start, camera.clip_end = .01, 100
        obj = bpy.data.objects.new('Replay ' + name, camera)
        scene.collection.objects.link(obj)
        cameras[name] = obj
    cameras['overview'].location = (3.5, -4.8, 3.0)
    cameras['overview'].rotation_euler = (Vector((0, .05, 1.0)) - cameras['overview'].location).to_track_quat('-Z', 'Y').to_euler()
    cameras['overview'].data.lens = 43
    axes = Matrix(((0, 0, -1, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))

    def set_frame(i):
        base = world_to_display @ pose(ep['replay_root_state'][i])
        for root in roots:
            links[root].matrix_world = base
        q = dict(zip(['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper'], ep['replay_dof_pos'][i][-7:]))
        for limb in ('FL', 'FR', 'RL', 'RR'):
            q.update({limb + '_hip_joint': 0., limb + '_thigh_joint': .8, limb + '_calf_joint': -1.5})
        for j in model['joints']:
            amount = q.get(j['name'], 0.)
            if j['mimic']:
                m = j['mimic']
                amount = q.get(m['joint'], 0.) * float(m.get('multiplier', 1)) + float(m.get('offset', 0))
            motion = Matrix.Identity(4)
            if j['type'] in ('revolute', 'continuous'):
                motion = Matrix.Rotation(amount, 4, Vector(j['axis']))
            elif j['type'] == 'prismatic':
                motion = Matrix.Translation(Vector(j['axis']) * amount)
            links[j['child']].matrix_basis = origin(j['origin']) @ motion
        doorq, handleq = ep['replay_door_dof_pos'][i]
        leaf.matrix_basis = leaf_origin @ Matrix.Rotation(doorq, 4, 'Z')
        handle.matrix_basis = handle_origin @ Matrix.Rotation(handleq, 4, Vector((0, -1, 0)))
        for name in ('front', 'wrist'):
            cameras[name].matrix_world = base @ pose(ep[name + '_camera_pose_base'][i]) @ axes
        bpy.context.view_layer.update()
        ee_error = (links['ee_gripper_link'].matrix_world.translation - world_to_display @ Vector(ep['replay_ee_pos'][i])).length
        expected_spindle = (world_to_display @ pose(ep['replay_door_root_state'][i]) @
                            Matrix.Translation(old_hinge) @ Matrix.Rotation(doorq, 4, 'Z') @
                            Vector((.868, -.058, .985)))
        spindle_error = (handle.matrix_world.translation - expected_spindle).length
        assert ee_error < 1e-4 and spindle_error < 1e-4, (i, ee_error, spindle_error)
        return ee_error, spindle_error

    count = len(ep['replay_root_state'])
    if a.bake_only:
        animated = [links[name] for name in roots]
        animated += [links[j['child']] for j in model['joints']
                     if j['name'] in ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper')]
        animated += [leaf, handle, cameras['front'], cameras['wrist']]
        for obj in animated:
            obj.rotation_mode = 'QUATERNION'
        scene.frame_start, scene.frame_end = 1, count
        scene.render.fps, scene.render.fps_base = round(float(ep['fps'])), 1
        errors, previous = [], {}
        for i in range(count):
            errors.append(set_frame(i))
            for obj in animated:
                if obj.name in previous and obj.rotation_quaternion.dot(previous[obj.name]) < 0:
                    obj.rotation_quaternion.negate()
                previous[obj.name] = obj.rotation_quaternion.copy()
                obj.keyframe_insert(data_path='location', frame=i + 1, group='Recorded trajectory')
                obj.keyframe_insert(data_path='rotation_quaternion', frame=i + 1, group='Recorded trajectory')
        # Blender 5 stores curves in layered action channel bags.
        for action in bpy.data.actions:
            for layer in action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for curve in bag.fcurves:
                            for key in curve.keyframe_points:
                                key.interpolation = 'LINEAR'
        for wall in partitions:
            wall.hide_render = True
        scene.camera = cameras['overview']
        scene.render.resolution_x, scene.render.resolution_y = (960, 720)
        scene.render.filepath = '//animation_frames/'
        scene.frame_set(1)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(a.output / 'wc_full_pull_animated.blend'))
        (a.output / 'animation_complete.json').write_text(json.dumps(dict(
            frames=count, fps=scene.render.fps, animated_objects=[o.name for o in animated],
            max_ee_error_m=max(v[0] for v in errors), max_spindle_error_m=max(v[1] for v in errors),
            interpolation='LINEAR', packed_textures=True,
            overview_wall_cutaway=True, all_source_poses_baked=True), indent=2))
        print('FULL_ANIMATION_BAKED', flush=True)
        return
    frames = list(range(0, count, a.stride))
    fps = float(ep['fps']) / a.stride
    indices = [0, 230, 550, 650, 1299] if a.probe else frames[a.start:a.stop]
    errors = []
    start = time.monotonic()
    for i in indices:
        errors.append(set_frame(i))
        for name, camera in cameras.items():
            # External cutaway keeps the full traversal visible. Robot cameras
            # retain the complete white partition on both sides of the doorway.
            for wall in partitions:
                wall.hide_render = name == 'overview'
            scene.camera = camera
            scene.render.resolution_x, scene.render.resolution_y = (960, 720) if name == 'overview' else (640, 480)
            folder = a.output / ('probe' if a.probe else 'frames') / name
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f'{i if a.probe else i // a.stride:04d}.png'
            if not target.exists():
                scene.render.filepath = str(target)
                bpy.ops.render.render(write_still=True)
        print('FRAME', i, 'of', count, 'elapsed_s', round(time.monotonic() - start, 1), flush=True)
    report = dict(source=data['raw'], source_scene_sha256=hashlib.sha256(a.scene_data.read_bytes()).hexdigest(),
                  source_frames=indices, source_total_frames=count, stride=a.stride,
                  source_fps=float(ep['fps']), output_fps=fps, duration_s=count / float(ep['fps']),
                  samples=a.samples, renderer='Blender Cycles OptiX',
                  max_ee_error_m=max(v[0] for v in errors), max_spindle_error_m=max(v[1] for v in errors),
                  door_render_copy_axis_scale=list(scale), original_door_asset='doortwin/wc/model.urdf',
                  notes=['Original UVs and basecolor images; rendering-copy dimensions fit old hinge and handle.',
                         'Recorded robot joints/base and camera poses are unchanged, up to rigid world coordinate conversion.',
                         'Negative hinge angles come from pull recording, not the new asset URDF positive limits.',
                         'Fixed visual leg pose matches recording loader; wheel rolling is not inferred.',
                         'Overview hides side partition to reveal full passage; front/wrist retain white walls.',
                         'Kinematic expert trajectory replay, not new physics or policy evaluation.'])
    target = 'probe_complete.json' if a.probe else f'render_complete_{a.start}_{a.stop}.json'
    (a.output / target).write_text(json.dumps(report, indent=2))
    if a.probe:
        set_frame(0)
        for wall in partitions:
            wall.hide_render = True
        scene.camera = cameras['overview']
        scene.render.resolution_x, scene.render.resolution_y = (960, 720)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(a.output / 'pull_replay_setup.blend'))


if __name__ == '__main__':
    main()
