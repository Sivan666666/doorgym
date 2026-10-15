"""Render the actual doortwin/wc URDF and UV-mapped OBJ assets in Blender.

Usage: blender -b --python-exit-code 1 --python render_wc_original_blender.py
       -- --source /path/to/wc --output /path/to/output
Source assets are read-only. No physics, training, or policy inference is run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import bpy
from mathutils import Euler, Matrix, Vector


def origin(element):
    if element is None:
        return Matrix.Identity(4)
    xyz = [float(v) for v in element.get('xyz', '0 0 0').split()]
    rpy = [float(v) for v in element.get('rpy', '0 0 0').split()]
    return Matrix.Translation(Vector(xyz)) @ Euler(rpy, 'XYZ').to_matrix().to_4x4()


def material(name, color, roughness, transmission=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Transmission Weight'].default_value = transmission
    shader.inputs['IOR'].default_value = 1.45
    return mat


def box(name, position, size, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    obj.data.materials.append(mat)
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    device = next(d for d in prefs.devices if d.type == 'OPTIX')
    for d in prefs.devices:
        d.use = d == device
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 8
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'AgX'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.8, .85, 1, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .25

    tree = ET.parse(source / 'model.urdf').getroot()
    links = {}
    for link in tree.findall('link'):
        obj = bpy.data.objects.new(link.get('name'), None)
        scene.collection.objects.link(obj)
        links[link.get('name')] = obj
    for joint in tree.findall('joint'):
        child = links[joint.find('child').get('link')]
        child.parent = links[joint.find('parent').get('link')]
        child.matrix_basis = origin(joint.find('origin'))
        child['urdf_joint'] = joint.get('name')
        child['joint_angle_rad'] = 0.0

    glass = material('Frosted glass - URDF tint, Cycles transmission', (.72, .84, .84), .42, .8)
    meshes = []
    for link in tree.findall('link'):
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            path = source / mesh.get('filename')
            before = set(bpy.data.objects)
            # OBJ coordinates already use URDF Z-up; disable axis conversion.
            bpy.ops.wm.obj_import(filepath=str(path), forward_axis='Y', up_axis='Z')
            for obj in set(bpy.data.objects) - before:
                obj.parent = links[link.get('name')]
                obj.matrix_basis = origin(visual.find('origin')) @ Matrix.Diagonal(
                    (*[float(v) for v in mesh.get('scale', '1 1 1').split()], 1))
                if visual.get('name') == 'door_leaf_frosted_lite':
                    obj.data.materials.clear()
                    obj.data.materials.append(glass)
                meshes.append(dict(name=obj.name, vertices=len(obj.data.vertices),
                                   polygons=len(obj.data.polygons), uv_layers=len(obj.data.uv_layers),
                                   source=mesh.get('filename')))
    bpy.context.view_layer.update()
    points = [obj.matrix_world @ Vector(corner) for obj in bpy.data.objects
              if obj.type == 'MESH' for corner in obj.bound_box]
    low = [min(p[i] for p in points) for i in range(3)]
    high = [max(p[i] for p in points) for i in range(3)]
    assert high[2] - low[2] > 2 and high[1] - low[1] < .5, (low, high)
    texture_nodes = [n for m in bpy.data.materials if m.use_nodes
                     for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
    assert len(texture_nodes) == 3, len(texture_nodes)
    assert all(Path(bpy.path.abspath(n.image.filepath)).is_file() for n in texture_nodes)

    wall = material('Preview only - neutral white plaster', (.78, .78, .76), .85)
    floor = material('Preview only - matte warm grey floor', (.42, .40, .37), .8)
    box('Preview floor', (0, 0, -.045), (12, 12, .08), floor)
    # Opening is outside the complete original door frame, with no occlusion.
    for sign in (-1, 1):
        box('Preview wall side', (sign * 1.65, .075, 1.6), (2.18, .12, 3.2), wall)
    box('Preview wall above', (0, .075, 2.728), (1.12, .12, .94), wall)
    box('Preview rear wall', (0, 2.3, 1.6), (6, .1, 3.2), wall)
    for name, position, energy, size, target in (
        ('Large soft key', (-2, -3, 3.6), 400, 2.5, (0, 0, 1)),
        ('Soft front fill', (2.3, -1.6, 2.4), 150, 2, (0, 0, 1)),
        ('Interior soft light', (-.8, 1.3, 2.8), 150, 1.5, (0, 0, 1)),
    ):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.shape, data.size = energy, 'DISK', size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = position
        aim(obj, target)

    views = (
        ('front', (0, -4.3, 1.2), (0, 0, 1.125), 58, (1200, 1500)),
        ('three_quarter', (1.7, -3.9, 1.8), (0, 0, 1.12), 53, (1400, 1400)),
        ('handle_detail', (.63, -.83, 1.10), (.35, -.06, .97), 55, (1400, 1000)),
    )
    cameras = []
    for name, location, target, lens, resolution in views:
        data = bpy.data.cameras.new(name)
        data.lens = lens
        data.clip_start = .01
        data.clip_end = 100
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        aim(obj, target)
        cameras.append(obj)
        scene.camera = obj
        scene.render.resolution_x, scene.render.resolution_y = resolution
        scene.render.filepath = str(out / (name + '.png'))
        bpy.ops.render.render(write_still=True)
    scene.camera = cameras[0]
    scene.render.resolution_x, scene.render.resolution_y = views[0][-1]
    scene.render.filepath = '//front.png'
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'wc_original.blend'))
    manifest = dict(source='doortwin/wc/model.urdf', blender=bpy.app.version_string,
                    engine='Cycles', samples=96, device=device.name, meshes=meshes,
                    source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(source.rglob('*')) if p.is_file()},
                    bounds_m=dict(low=low, high=high), original_geometry_and_uv=True,
                    original_basecolor_textures=True, textures_packed=True,
                    glass='URDF tint; roughness 0.42, transmission 0.8, IOR 1.45',
                    notes='Closed door, static appearance preview. Added neutral room and soft lights. No simulation or inference.',
                    outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(out.iterdir()) if p.suffix in ('.png', '.blend')})
    (out / 'verification.json').write_text(json.dumps(manifest, indent=2))
    print('WC_RENDER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
