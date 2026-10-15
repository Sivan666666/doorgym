"""Export trusted raw episode kinematics and original URDF visuals for rendering."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
import sys
import numpy as np
import trimesh

REPO = Path('/home/ps/workspace/txc/doorgym')
A = Path('/mnt/sda/ps_archive/doorgym/high-level')
OUT = A / 'dp/result/rgb500_blender_review_20260918'
RAW = A / 'data/door_dp_raw/0914_pull_rgb_aug500_joint9_g080_fixed15/episode_000100.npz'


def vec(element, key, default='0 0 0'):
    return [float(v) for v in (element.get(key, default) if element is not None else default).split()]


def origin(element):
    return dict(xyz=vec(element, 'xyz'), rpy=vec(element, 'rpy'))


def export_urdf(path, role):
    root = ET.parse(path).getroot()
    model = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), links={}, joints=[])
    for link in root.findall('link'):
        name = link.get('name'); visuals = []
        for i, visual in enumerate(link.findall('visual')):
            o = origin(visual.find('origin'))
            if role == 'robot' and name in ('wrist_up_mount', 'wrist_down_mount'):
                # Same explicit visual correction as project's reviewed Blender exporter.
                o['rpy'] = vec(link.find('collision/origin'), 'rpy')
            item = dict(name=visual.get('name', f'{name}_{i}'), origin=o,
                        color=vec(visual.find('material/color'), 'rgba', '.65 .67 .69 1'))
            geo = visual.find('geometry'); mesh = geo.find('mesh')
            if mesh is not None:
                source = (path.parent / mesh.get('filename')).resolve()
                target = OUT / 'meshes' / f'{role}_{name}_{i}.ply'
                if not target.exists():
                    obj = trimesh.load_scene(source, process=False).to_geometry()
                    assert isinstance(obj, trimesh.Trimesh), source
                    obj.export(target)
                item.update(mesh=str(target), scale=vec(mesh, 'scale', '1 1 1'), source_mesh=str(source))
            elif geo.find('box') is not None:
                item['box'] = vec(geo.find('box'), 'size')
            elif geo.find('cylinder') is not None:
                item['cylinder'] = [float(geo.find('cylinder').get(k)) for k in ('radius', 'length')]
            elif geo.find('sphere') is not None:
                item['sphere'] = float(geo.find('sphere').get('radius'))
            else:
                raise ValueError(ET.tostring(geo))
            visuals.append(item)
        model['links'][name] = visuals
    for joint in root.findall('joint'):
        mimic = joint.find('mimic')
        model['joints'].append(dict(name=joint.get('name'), type=joint.get('type'),
            parent=joint.find('parent').get('link'), child=joint.find('child').get('link'),
            origin=origin(joint.find('origin')), axis=vec(joint.find('axis'), 'xyz', '1 0 0'),
            mimic=dict(mimic.attrib) if mimic is not None else None))
    return model


def main():
    (OUT / 'meshes').mkdir(parents=True, exist_ok=True)
    with np.load(RAW, allow_pickle=True) as data:
        keys = [k for k in data.files if k not in ('front_rgb', 'wrist_rgb')]
        episode = {k: data[k].tolist() for k in keys}
    sys.path.insert(0,str(REPO/'high-level/float_ik'))
    from rgb_door_materials import reference_texture
    import cv2
    textures=OUT/'textures';textures.mkdir(exist_ok=True)
    for key in ('panel','handle'):
        rgb=reference_texture(episode['rgb_visual_episode']['materials'][key],512)
        assert cv2.imwrite(str(textures/(key+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    models = dict(robot=export_urdf(REPO/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf', 'robot'),
                  door=export_urdf(REPO/'high-level/data/asset/door_set/wc_nomenkuang/model.urdf', 'door'))
    (OUT/'scene_data.json').write_text(json.dumps(dict(raw=str(RAW), episode=episode, models=models)))
    print('EXPORTED kinematics and original mesh geometry', flush=True)


if __name__ == '__main__':
    main()
