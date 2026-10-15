"""Render-only room dressing, with conservative full-rollout clearance checks."""
import hashlib
import json
import random

import bpy
import numpy as np
from mathutils import Vector


def furnish_room(scene, data, convert, pose, source, out, settings=None):
    settings=settings or {}
    rng=random.Random(int(settings.get('seed',0)))
    randomize=bool(settings.get('randomize',False))
    minimum=float(settings.get('minimum_robot_clearance_m',.8))
    assert minimum>=.8
    # Caller has applied physical frame zero. Associate every mesh with its
    # nearest recorded rigid-body parent, including all fingers and mounts.
    owners = {}
    for role, records in data['frames'][0]['bodies'].items():
        for name in records:
            key = name if role == 'door' else 'robot/' + name
            if key in bpy.data.objects:
                owners[key] = (role, name)
    bounds = {group: [np.full(3, np.inf), np.full(3, -np.inf)] for group in ('robot', 'door')}
    counts = dict(robot=0, door=0)
    matrices = {}
    for obj in list(bpy.data.objects):
        if obj.type != 'MESH':
            continue
        parent = obj.parent
        while parent is not None and parent.name not in owners:
            parent = parent.parent
        if parent is None:
            assert not obj.name.startswith('robot/'), f'Untracked robot mesh: {obj.name}'
            continue
        role, name = owners[parent.name]
        group = 'door' if role == 'door' else 'robot'
        relative = parent.matrix_world.inverted() @ obj.matrix_world
        points = np.array([(*tuple(relative @ Vector(c)), 1.) for c in obj.bound_box])
        if parent.name not in matrices:
            matrices[parent.name] = np.array([np.array(convert @ pose(f['bodies'][role][name]))
                                              for f in data['frames']])
        world = np.einsum('tij,kj->tki', matrices[parent.name], points)[..., :3]
        bounds[group][0] = np.minimum(bounds[group][0], world.min(axis=(0, 1)))
        bounds[group][1] = np.maximum(bounds[group][1], world.max(axis=(0, 1)))
        counts[group] += 1
    assert counts['robot'] >= 20 and counts['door'] >= 3, counts
    robot_lo, robot_hi = bounds['robot']
    door_lo, door_hi = bounds['door']
    assert robot_hi[2] < 2.45, 'Raise the ceiling for this trajectory'
    objects = []

    def material(name, color, roughness=.7, metallic=0):
        m = bpy.data.materials.new('Room / ' + name)
        m.diffuse_color = (*color, 1)
        m.use_nodes = True
        shader = m.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = (*color, 1)
        shader.inputs['Roughness'].default_value = roughness
        shader.inputs['Metallic'].default_value = metallic
        return m

    white = material('white plaster', (.85, .85, .85), .85)
    steel = material('shelf powder coat', (.21, .25, .28), .5, .3)
    wood = material('shelf oak', (.34, .22, .12))
    cardboard = material('cardboard', rng.choice([(.42,.27,.13),(.50,.35,.20),(.32,.23,.15)]) if randomize else (.42,.27,.13), .95)
    tape = material('packing tape', (.62, .46, .25), .55)
    blue = material('blue storage bin', rng.choice([(.09,.23,.32),(.18,.28,.20),(.30,.20,.13)]) if randomize else (.09,.23,.32), .5)
    cream = material('cream storage bin', (.63, .60, .51), .65)

    def distance(lo, hi, a, b):
        return float(np.linalg.norm(np.maximum(np.maximum(lo - b, a - hi), 0)))

    def cube(name, center, size, mat, bevel=0):
        lo = np.array(center) - np.array(size) / 2
        hi = np.array(center) + np.array(size) / 2
        robot_gap = distance(lo, hi, robot_lo, robot_hi)
        door_gap = distance(lo, hi, door_lo, door_hi)
        assert robot_gap >= minimum, (name, 'robot clearance', robot_gap)
        assert door_gap >= .5, (name, 'door clearance', door_gap)
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        obj = bpy.context.object
        obj.name = 'Room / ' + name
        obj.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.data.materials.append(mat)
        if bevel:
            modifier = obj.modifiers.new('small edge bevel', 'BEVEL')
            modifier.width = bevel
            modifier.segments = 2
        objects.append(dict(name=obj.name, center=list(center), dimensions=list(size),
                            robot_swept_aabb_clearance_lower_bound_m=robot_gap,
                            door_swept_aabb_clearance_lower_bound_m=door_gap))
        return obj

    # White roof spans both sides of the doorway; its underside is at 3.3 m.
    cube('white ceiling', (0, -.7, 3.36), (12, 10.12, .12), white)
    cube('left room wall', (-4, 2.23, 1.65), (.12, 4.14, 3.3), white)
    cube('right room wall', (4, 2.23, 1.65), (.12, 4.14, 3.3), white)
    # The old rear wall stays in place; audit it too and unify the plaster.
    rear = bpy.data.objects['Rear room wall']
    rear.hide_render = True
    cube('white rear wall', (0, 4.3, 1.65), (8.12, .12, 3.3), white)
    # These side storage areas leave a wide central corridor clear.
    left_x = min(-2.15, float(robot_lo[0]) - 1.4, float(door_lo[0]) - 1.1)
    right_x = max(2.15, float(robot_hi[0]) + 1.4, float(door_hi[0]) + 1.1)
    if randomize:
        left_x-=rng.uniform(0,.15)
        right_x+=rng.uniform(0,.15)
    assert left_x > -3.3 and right_x < 3.3, 'Room is too narrow for safe side storage'
    for x in (left_x - .52, left_x + .52):
        for y in (2.68, 3.32):
            cube('shelf upright', (x, y, .90), (.045, .045, 1.8), steel, .006)
    for z in (.16, .70, 1.24, 1.78):
        cube('shelf board', (left_x, 3., z), (1.12, .72, .045), wood, .008)
    for j, (x, y, bottom, w, d, h, mat) in enumerate([
        (left_x-.27, 3., .183, .42, .54, .37, blue),
        (left_x+.25, 3., .183, .40, .48, .30, cream),
        (left_x-.23, 3., .723, .43, .46, .33, cardboard),
        (left_x+.28, 3., .723, .40, .50, .40, cardboard),
        (left_x, 3., 1.263, .60, .46, .28, blue),
        (right_x, 2.75, -.01, .86, .68, .52, cardboard),
        (right_x-.10, 2.77, .51, .58, .52, .40, cardboard),
        (right_x+.36, 3.65, -.01, .56, .48, .42, cream),
    ]):
        cube(f'storage box {j}', (x, y, bottom+h/2), (w, d, h), mat, .012)
        if mat == cardboard:
            cube(f'tape {j}', (x, y, bottom+h+.001), (.065, d-.03, .002), tape)
    cube('short side partition', (right_x+.70, 2.35, .70), (.10, 1.75, 1.40), white)

    # Existing studio lights above the roof cannot illuminate the interior.
    # Add soft ceiling panels under the slab without changing camera mounts.
    for label, y, energy in [('entrance', -2.4, 500), ('back room', 2.5, 300)]:
        lamp = bpy.data.lights.new('Room / ' + label + ' ceiling light', 'AREA')
        lamp.energy = energy*(rng.uniform(.85,1.15) if randomize else 1.)
        lamp.shape = 'DISK'
        lamp.size = 2.5
        obj = bpy.data.objects.new(lamp.name, lamp)
        scene.collection.objects.link(obj)
        obj.location = (0, y, 3.24)
        obj.rotation_euler = (0, 0, 0)  # Area lights emit along local -Z.

    report = dict(scene_variant='white_ceiling_furnished_room_v1',settings=settings,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  frames_checked=len(data['frames']), meshes_checked=counts,
                  robot_swept_aabb=[v.tolist() for v in bounds['robot']],
                  door_swept_aabb=[v.tolist() for v in bounds['door']],
                  min_robot_clearance_lower_bound_m=min(v['robot_swept_aabb_clearance_lower_bound_m'] for v in objects),
                  min_door_clearance_lower_bound_m=min(v['door_swept_aabb_clearance_lower_bound_m'] for v in objects),
                  added_objects=objects, render_only=True,
                  method='World AABB enclosing every transformed visual mesh bounding box at all 2600 recorded physical frames; distance to each added box is a conservative lower bound.',
                  scope='Added ceiling, side/rear walls, partition and props only. Existing door-surround walls excluded.',
                  limitations='Recorded frames only, not continuous collision detection or new physics simulation; not a guarantee for changed trajectories.')
    (out / 'room_clearance.json').write_text(json.dumps(report, indent=2))
    scene['room_note'] = json.dumps({k: report[k] for k in ('scene_variant', 'min_robot_clearance_lower_bound_m', 'render_only')})
    print('ROOM_CLEARANCE', report['min_robot_clearance_lower_bound_m'], flush=True)
    return report
