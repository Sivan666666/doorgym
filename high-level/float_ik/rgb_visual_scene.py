"""Visual-only replacement meshes for the two EXISTING non-colliding wall actors.

Adds indoor surfaces, a lintel and ground without adding actors, joints or collision
shapes. The original fixed wall body's inertial definition is preserved.
"""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def box_quads(lo, hi):
    x, y, z = lo
    X, Y, Z = hi
    vertices = np.asarray([(x,y,z),(X,y,z),(X,Y,z),(x,Y,z),
                           (x,y,Z),(X,y,Z),(X,Y,Z),(x,Y,Z)], float)
    return [vertices[list(face)] for face in ((0,3,2,1),(4,5,6,7),(0,1,5,4),
                                              (3,7,6,2),(0,4,7,3),(1,2,6,5))]


def mesh_geometry(pose, axis, side, width, opening_width, gap, height, top, thickness, floor_extent, center):
    tangent = 0 if axis == 'x' else 1
    normal = 1 - tangent
    p = np.asarray(pose, float)
    lo, hi = p.copy(), p.copy()
    lo[tangent] -= width/2
    hi[tangent] += width/2
    lo[normal] -= thickness/2
    hi[normal] += thickness/2
    lo[2], hi[2] = 0., height
    walls = box_quads(lo, hi)
    # The original opening width/gap is not reduced; only bridge ABOVE the door.
    inner = center[tangent]
    outer = p[tangent] + side*width/2
    lo[tangent], hi[tangent] = min(inner, outer), max(inner, outer)
    lo[2], hi[2] = top, height
    walls.extend(box_quads(lo, hi))
    low = np.asarray([center[0]-floor_extent, center[1]-floor_extent, .001])
    high = np.asarray([center[0]+floor_extent, center[1]+floor_extent, .001])
    if side < 0:
        high[tangent] = center[tangent]
    else:
        low[tangent] = center[tangent]
    floor = np.asarray([(low[0],low[1],.001),(high[0],low[1],.001),
                        (high[0],high[1],.001),(low[0],high[1],.001)])
    return walls, floor


def indoor_parameters(seed, index, cfg, axis, yaw, center, height):
    """Independent visual RNG; distance measured from the doorway, along travel."""
    tangent = 0 if axis == 'x' else 1
    normal = 1 - tangent
    heading = np.asarray([np.cos(yaw), np.sin(yaw)])
    if abs(heading[normal]) < .5:
        raise ValueError('Indoor room needs a robot heading across the doorway')
    r = np.random.default_rng(np.random.SeedSequence([int(seed), 748, int(index)]))
    bounds = cfg['far_wall_distance_m']
    if not 0 < bounds[0] <= bounds[1]:
        raise ValueError('Invalid far-wall distance range')
    distance = float(r.uniform(*bounds))
    return dict(version='indoor_visual_v1', center=list(map(float, center)),
                normal_axis=normal, travel_sign=1 if heading[normal] > 0 else -1,
                far_wall_distance_m=distance, half_width_m=float(cfg['half_width_m']),
                ceiling_height_m=float(height), approach_ceiling_extent_m=float(cfg['approach_ceiling_extent_m']),
                ceiling_rgb=[int(r.integers(*cfg['ceiling_gray_range']))]*3,
                collision_shapes_added=0, visible_photo_background=False)


def indoor_geometry(room, side):
    """One half-room per EXISTING wall actor. All coordinates are env-local.

    Far wall + rear side walls enclose the destination room. The ceiling also
    covers the approach, so the upward-facing front camera never needs a sky
    photograph. Thin visual boxes are visible from either side.
    """
    n, sign = room['normal_axis'], room['travel_sign']
    t = 1 - n
    center = np.asarray([*room['center'], 0.])
    distance, width, height = room['far_wall_distance_m'], room['half_width_m'], room['ceiling_height_m']
    far = center[n] + sign * distance
    lo, hi = center.copy(), center.copy()
    lo[t], hi[t] = sorted([center[t], center[t] + side * width])
    # Far wall: the inward face, not its midpoint, is exactly distance metres away.
    lo[n], hi[n] = sorted([far, far + sign*.08])
    lo[2], hi[2] = 0., height
    walls = box_quads(lo, hi)
    # One side wall on the far side of the doorway, not across the robot path.
    lo[n], hi[n] = sorted([center[n], far])
    edge = center[t] + side * width
    lo[t], hi[t] = sorted([edge, edge + side*.08])
    walls.extend(box_quads(lo, hi))
    lo[t], hi[t] = sorted([center[t], edge])
    lo[n], hi[n] = sorted([center[n] - sign*room['approach_ceiling_extent_m'], far])
    lo[2], hi[2] = height, height+.08
    return walls, box_quads(lo, hi)


def tiled_quads(quad, tile):
    """Gym clamps imported texture UVs: explicitly tessellate repeated tiles."""
    normal = np.cross(quad[1]-quad[0], quad[2]-quad[0])
    normal /= np.linalg.norm(normal)
    constant = int(np.argmax(abs(normal)))
    axes = [i for i in range(3) if i != constant]
    grids = []
    for axis in axes:
        lo, hi = float(quad[:,axis].min()), float(quad[:,axis].max())
        inside = np.arange(math_floor(lo/tile)+1, math_ceil(hi/tile))*tile
        grids.append([lo, *[float(v) for v in inside if lo+1e-7 < v < hi-1e-7], hi])
    for a, A in zip(grids[0][:-1], grids[0][1:]):
        for b, B in zip(grids[1][:-1], grids[1][1:]):
            positions = np.repeat(quad[0:1], 4, axis=0)
            positions[:,axes] = [(a,b),(A,b),(A,B),(a,B)]
            if np.dot(np.cross(positions[1]-positions[0], positions[2]-positions[0]), normal) < 0:
                positions = positions[::-1]
            origin = np.floor(np.asarray([(a+A)/2, (b+B)/2])/tile)
            uv = np.clip(positions[:,axes]/tile-origin, 0., 1.)
            yield positions, uv, normal


def math_floor(x):
    return int(np.floor(x))


def math_ceil(x):
    return int(np.ceil(x))


def write_asset(root, pose, walls, floor, wall_tex, floor_tex, wall_tile, floor_tile, extras=()):
    root = Path(root)
    cv2.imwrite(str(root/'wall.png'), wall_tex[..., ::-1])
    cv2.imwrite(str(root/'floor.png'), floor_tex[..., ::-1])
    lines = ['mtllib scene.mtl']
    offset = 0
    groups = [('wall', walls, wall_tile), ('floor', [floor], floor_tile)]
    for name, quads, texture, tile in extras:
        cv2.imwrite(str(root/(name+'.png')), texture[..., ::-1])
        groups.append((name, quads, tile))
    for material, quads, tile in groups:
        lines.append('usemtl ' + material)
        for quad, uv, normal in (part for q in quads for part in tiled_quads(q, tile)):
            for xyz in quad - np.asarray(pose):
                lines.append('v ' + ' '.join(f'{v:.8f}' for v in xyz))
            for xy in uv:
                lines.append('vt ' + ' '.join(f'{v:.8f}' for v in xy))
            for _ in quad:
                lines.append('vn ' + ' '.join(f'{v:.8f}' for v in normal))
            for tri in ((1,2,3),(1,3,4)):
                lines.append('f ' + ' '.join(f'{offset+i}/{offset+i}/{offset+i}' for i in tri))
            offset += 4
    (root/'scene.obj').write_text('\n'.join(lines)+'\n')
    (root/'scene.mtl').write_text('\n'.join(
        f'newmtl {name}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 2\nmap_Kd {name}.png\n'
        for name, _, _ in groups))
    urdf = '''<robot name="door_side_wall"><link name="wall">
<visual><geometry><mesh filename="scene.obj"/></geometry></visual>
<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
<inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/></inertial>
</link></robot>'''
    assert not ET.fromstring(urdf).findall('.//collision')
    assert len(ET.fromstring(urdf).findall('link')) == 1
    (root/'scene.urdf').write_text(urdf)


def make_wall_asset(gym, sim, gymapi, runtime, args, door, env_index, side_name,
                    pose, axis, opening_width, center):
    key = (id(sim), env_index, side_name)
    if key in runtime.visual_assets:
        return runtime.visual_assets[key]
    cfg = runtime.config['scene']
    top = max(float(args.door_wall_height), float(args.door_z_offset) +
              float(door.actor_position_offset[2]) + float(door.bounding['max'][2])*float(door.actor_scale))
    top += float(cfg['opening_top_clearance'])
    height = max(float(cfg['wall_height']), top + .2)
    p = [float(pose.p.x), float(pose.p.y), float(pose.p.z)]
    floor_extent = float(cfg['floor_half_extent'])
    if int(getattr(args, 'parallel_num_envs', getattr(args, 'num_envs', 1))) > 1:
        # Adjacent Gym environments must not draw coplanar floors over each other.
        floor_extent = min(floor_extent, float(getattr(args, 'env_spacing', 5.0))/2 - .002)
    walls, floor = mesh_geometry(p, axis, -1 if side_name == 'left' else 1,
        float(args.door_wall_side_width), opening_width, float(args.door_wall_gap), height, top,
        float(args.door_wall_thickness), floor_extent, center)
    root = Path(tempfile.mkdtemp(prefix=f'doorman_visual_{runtime.seed}_{env_index}_{side_name}_'))
    descriptions = runtime.environment_materials(env_index)
    extras = []
    room_cfg = cfg.get('indoor_room', {})
    if room_cfg.get('enabled'):
        room = indoor_parameters(runtime.seed, env_index, room_cfg, axis, float(args.robot_yaw), center, height)
        room_walls, ceiling = indoor_geometry(room, -1 if side_name == 'left' else 1)
        extras = [('far_wall', room_walls, runtime.texture(descriptions['far_wall']), cfg['wall_texture_tile_m']),
                  ('ceiling', ceiling, np.full((4,4,3), room['ceiling_rgb'], np.uint8), 1.5)]
        runtime.scene_geometry[env_index] = room
    write_asset(root, p, walls, floor, runtime.texture(descriptions['wall']),
                runtime.texture(descriptions['floor']), cfg['wall_texture_tile_m'], cfg['floor_texture_tile_m'], extras)
    opts = gymapi.AssetOptions()
    opts.fix_base_link = True
    opts.disable_gravity = True
    opts.use_mesh_materials = True
    asset = gym.load_asset(sim, str(root), 'scene.urdf', opts)
    if asset is None:
        raise RuntimeError('Could not load visual-only scene mesh')
    assert gym.get_asset_rigid_shape_count(asset) == 0, 'Visual wall must NEVER add collision shapes'
    assert gym.get_asset_rigid_body_count(asset) == 1
    assert gym.get_asset_dof_count(asset) == 0
    runtime.visual_assets[key] = asset
    return asset
