"""Create an isolated, metrically faithful collision asset for the WC door.

Run with the geometry Python environment (numpy/trimesh/yaml installed).
Never changes doortwin/wc or the training assets.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml


def text(v):
    return ' '.join(f'{x:.9g}' for x in v)


def components(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return list(mesh.split(only_watertight=False))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    dest = a.output / 'assets/wc_contact'
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(a.source / 'assets', dest / 'assets', dirs_exist_ok=True)
    root = ET.parse(a.source / 'model.urdf').getroot()
    root.set('name', 'wc_original_contact_pull')
    all_boxes = {}
    for link in root.findall('link'):
        for old in list(link.findall('collision')):
            link.remove(old)
        boxes = []
        for visual in link.findall('visual'):
            path = a.source / visual.find('geometry/mesh').get('filename')
            for mesh in components(path):
                if not mesh.is_watertight:
                    assert np.min(mesh.extents) < 1e-7  # zero-thickness decal
                    continue
                lo, hi = mesh.bounds
                if len(mesh.vertices) == 16:  # exact rectangular panel with glazing opening
                    assert path.name == 'door_leaf_opaque.obj'
                    hole_lo = np.array([.6255, -.021, .205])
                    hole_hi = np.array([.7805, .021, 1.935])
                    panel_boxes = [(lo, [hole_lo[0], hi[1], hi[2]]),
                                   ([hole_hi[0], lo[1], lo[2]], hi),
                                   ([hole_lo[0], lo[1], lo[2]], [hole_hi[0], hi[1], hole_lo[2]]),
                                   ([hole_lo[0], lo[1], hole_hi[2]], [hole_hi[0], hi[1], hi[2]])]
                    assert abs(sum(np.prod(np.asarray(h)-np.asarray(l)) for l,h in panel_boxes) - mesh.volume) < 1e-7
                    boxes.extend(panel_boxes)
                else:
                    assert len(mesh.vertices) == 8 and abs(mesh.volume - np.prod(hi - lo)) < 1e-7
                    boxes.append((lo, hi))
        all_boxes[link.get('name')] = []
        for i, (low, high) in enumerate(boxes):
            low, high = np.asarray(low), np.asarray(high)
            c = ET.SubElement(link, 'collision', name=f'exact_component_{i}')
            ET.SubElement(c, 'origin', xyz=text((low + high)/2), rpy='0 0 0')
            ET.SubElement(ET.SubElement(c, 'geometry'), 'box', size=text(high-low))
            all_boxes[link.get('name')].append(dict(low=low.tolist(), high=high.tolist()))

    # The original centered hinge can sweep the leaf into the 4 mm jamb gap
    # when pulling. Move its axis to the front edge, while preserving every
    # visible surface in the closed pose (no rescaling of the door or handle).
    hinge = root.find("joint[@name='frame_to_door_leaf']")
    hinge.find('origin').set('xyz', '-0.443226 -0.036 0.004')
    hinge.find('limit').set('lower', str(-math.pi/2))
    hinge.find('limit').set('upper', '0')
    leaf = root.find("link[@name='door_leaf']")
    for tag in ('visual', 'collision'):
        for node in leaf.findall(tag):
            origin = node.find('origin')
            if origin is None:
                origin = ET.SubElement(node, 'origin', xyz='0 0 0', rpy='0 0 0')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            xyz[1] += .021
            origin.set('xyz', text(xyz))
    root.find("joint[@name='door_to_handle_mount']/origin").set('xyz', '.8613 0 .922')

    # Express the identical handle rotation as local Z to match the controller's
    # spindle-frame convention. Inverse-rotate visuals and collisions: world
    # geometry is unchanged for all joint angles, not just the closed pose.
    joint = root.find("joint[@name='mount_to_handle']")
    joint.find('origin').set('rpy', f'{math.pi/2} 0 0')
    joint.find('axis').set('xyz', '0 0 1')
    handle = root.find("link[@name='lever_handle']")
    for tag in ('visual', 'collision'):
        for node in handle.findall(tag):
            origin = node.find('origin')
            if origin is None:
                origin = ET.SubElement(node, 'origin', xyz='0 0 0')
            x, y, z = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            origin.set('xyz', text((x, z, -y)))
            origin.set('rpy', f'{-math.pi/2} 0 0')

    masses = {'frame': 15., 'door_leaf': 25., 'handle_mount': .02, 'lever_handle': .18}
    for link in root.findall('link'):
        name = link.get('name')
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass', value=str(masses[name]))
        if name == 'door_leaf':
            center, dimensions = [.443226, .021, 1.065], [.886452, .042, 2.13]
        elif name == 'lever_handle':
            center, dimensions = [-.054, 0, .01], [.1295, .03, .021]
        else:
            center, dimensions = [0, 0, 0], [.1, .1, .1]
        ET.SubElement(inertial, 'origin', xyz=text(center), rpy='0 0 0')
        x, y, z = dimensions
        vals = masses[name] / 12 * np.array([y*y+z*z, x*x+z*z, x*x+y*y])
        ET.SubElement(inertial, 'inertia', ixx=str(vals[0]), iyy=str(vals[1]), izz=str(vals[2]), ixy='0', ixz='0', iyz='0')
    ET.indent(root)
    ET.ElementTree(root).write(dest / 'model.urdf', encoding='utf-8', xml_declaration=True)
    (dest / 'bounding_box.json').write_text(json.dumps(dict(min=[-.55348, -.05, -.000005], max=[.55348, .0925, 2.25089])))
    (dest / 'handle_bounding.json').write_text(json.dumps(dict(handle_min=[-.1185,-.015,-.0005], handle_max=[.011,.015,.0205], goal_pos=[-.080,0,.010])))
    old = yaml.safe_load(a.config.read_text())
    entries = old['env']['asset']['trainAssets'][old['env']['asset']['load_block']]
    spec = copy.deepcopy(next(v for v in entries.values() if v['name'] == 'wc_nomenkuang'))
    spec.update(name='wc_contact', path='wc_contact/model.urdf', bounding_box='wc_contact/bounding_box.json',
                handle_bounding='wc_contact/handle_bounding.json', door_body_name='door_leaf',
                door_dof_name='frame_to_door_leaf', handle_dof_name='mount_to_handle',
                handle_dof_upper_override=.95, handle_unlock_angle=math.radians(25),
                door_wall_x_reference_offset=.146)
    spec.pop('controller_multipliers', None)
    spec.pop('controller_overrides', None)
    cfg = {'env': {'asset': {'assetRoot': str(a.output / 'assets'), 'assetFileDoor': '',
                            'load_block': 'wc_contact', 'trainAssets': {'wc_contact': {'0': spec}}}}}
    (a.output / 'door_contact.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    report = dict(source_sha256={str(p.relative_to(a.source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(a.source.rglob('*')) if p.is_file()},
                  collision_boxes_source_coordinates=all_boxes, mass_kg=masses,
                  mass_note='Assumed engineering parameters, not measured real-door mass properties.',
                  hinge_axis_front_shift_m=.021, closed_visual_geometry_preserved=True,
                  geometry_scale=[1,1,1], handle_frame_rotation_deg=90,
                  excluded_collision='Zero-thickness decal only; glazing included as solid 6 mm box.')
    (a.output / 'asset_preparation.json').write_text(json.dumps(report, indent=2))
    robot_collision = a.output/'assets/robot_collision'
    robot_collision.mkdir(exist_ok=True)
    robot_mesh_root = a.output.parent/'rgb500_blender_review_20260918/meshes'
    for name in ('gripperStator','gripperMover'):
        mesh=trimesh.load(robot_mesh_root/f'robot_{name}_0.ply',force='mesh',process=False)
        mesh.export(robot_collision/f'{name}.stl')
    print('PREPARED', dest, {k:len(v) for k,v in all_boxes.items()})


if __name__ == '__main__':
    main()
