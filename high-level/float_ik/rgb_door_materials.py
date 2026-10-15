"""Reference WC wood/black-handle textures and UV-only visual asset copies.

Never replaces collision geometry, vertices, joints, inertia, or trajectories.
The reference PNGs are UV atlases: use clean surface crops, not an atlas pasted
as a photograph (which would paint fake glass cutouts and locks on the slab).
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def reference_sources(repo, cfg):
    root = Path(cfg['texture_root'])
    if not root.is_absolute():
        root = Path(repo) / root
    sources = {}
    for group, name in [('panel', 'door_leaf_opaque_basecolor.png'), ('handle', 'lever_handle_basecolor.png')]:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError('WC reference texture missing: ' + str(path))
        sources[group] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return sources


def reference_descriptors(seed, index, cfg, sources):
    r = np.random.default_rng(np.random.SeedSequence([int(seed) % 2**32, 750, int(index)]))
    palettes = cfg['panel_palettes']
    palette = palettes[int(r.integers(len(palettes)))]
    # Clean wood region in the 1024-square reference atlas, avoiding glass/lock.
    x, y = int(r.integers(135, 180)), int(r.integers(110, 190))
    width, height = int(r.integers(160, 210)), int(r.integers(450, 550))
    panel = dict(kind='wc_reference_v1', name='wc_wood_' + palette['name'],
        **sources['panel'], crop_xyxy_1024=[x,y,x+width,y+height],
        base_rgb=palette['rgb'], contrast=float(r.uniform(*cfg['grain_contrast'])),
        flip_u=bool(r.integers(2)), flip_v=bool(r.integers(2)))
    handle = dict(kind='wc_reference_v1', name='wc_black_handle', **sources['handle'],
        crop_xyxy_1024=[225,270,330,450], base_rgb=cfg['handle_rgb'],
        contrast=.35, flip_u=False, flip_v=False)
    return dict(panel=panel, handle=handle)


def reference_texture(desc, size):
    path = Path(desc['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != desc['sha256']:
        raise ValueError('WC reference texture changed during run: ' + str(path))
    source = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    x, y, X, Y = desc['crop_xyxy_1024']
    h, w = source.shape[:2]
    crop = source[y*h//1024:Y*h//1024, x*w//1024:X*w//1024]
    luminance = crop.astype(np.float32).mean(-1)
    relative = 1 + desc['contrast'] * (luminance / max(float(luminance.mean()), 1.) - 1)
    rgb = np.clip(relative[...,None] * np.asarray(desc['base_rgb']), 0, 255).astype(np.uint8)
    if desc['flip_u']:
        rgb = rgb[:,::-1]
    if desc['flip_v']:
        rgb = rgb[::-1]
    return cv2.resize(rgb, (size, size*2), interpolation=cv2.INTER_LINEAR)


def add_planar_uv(source, destination):
    """Preserve every vertex and face index; append per-face UV coordinates."""
    lines = Path(source).read_text().splitlines()
    if any(row.startswith('vt ') for row in lines):
        raise ValueError('Refusing to overwrite existing mesh UVs: ' + str(source))
    vertices = np.asarray([list(map(float, row.split()[1:4])) for row in lines if row.startswith('v ')])
    low, span = vertices.min(0), np.maximum(np.ptp(vertices, axis=0), 1e-8)
    uv, faces = [], []
    for row in lines:
        if not row.startswith('f '):
            continue
        refs = row.split()[1:]
        indices = [int(ref.split('/')[0]) for ref in refs]
        points = vertices[[i-1 if i > 0 else i for i in indices]]
        normal = np.cross(points[1]-points[0], points[2]-points[0])
        axes = [i for i in range(3) if i != int(np.argmax(abs(normal)))]
        tokens = []
        for ref, point in zip(refs, points):
            coords = (point[axes]-low[axes])/span[axes]
            uv.append('vt ' + ' '.join(f'{v:.9f}' for v in coords))
            fields = ref.split('/')
            tokens.append(f'{fields[0]}/{len(uv)}' + ('/'+fields[2] if len(fields)>2 and fields[2] else ''))
        faces.append('f ' + ' '.join(tokens))
    kept = [row for row in lines if not row.startswith('f ')]
    Path(destination).write_text('\n'.join(kept+uv+faces)+'\n')


def prepare_visual_door(runtime, asset_root, door_file):
    """Private generated URDF: edit only missing slab visual UVs, not physics."""
    if not runtime.config['materials'].get('enabled', True):
        return asset_root, door_file
    if runtime.config['materials'].get('preserve_subvisual_materials', False):
        return prepare_per_visual_door(runtime, asset_root, door_file)
    if not runtime.config['materials'].get('wc_reference', {}).get('enabled'):
        return asset_root, door_file
    source = (Path(asset_root)/door_file).resolve()
    tree = ET.parse(source)
    targets = [m for m in tree.findall(".//link[@name='door_panel']/visual[@name='door_slab']/geometry/mesh")
               if Path(m.get('filename')).suffix.lower() == '.obj'
               and not any(row.startswith('vt ') for row in (source.parent/m.get('filename')).read_text().splitlines())]
    if not targets:
        return asset_root, door_file
    # Absolute paths preserve the exact original collision mesh and cache key.
    for node in tree.findall('.//mesh') + tree.findall('.//texture'):
        if node.get('filename'):
            node.set('filename', str((source.parent/node.get('filename')).resolve()))
    baseline = copy.deepcopy(tree.getroot())
    root = Path(tempfile.mkdtemp(prefix='wc_reference_visual_uv_'))
    for i, mesh in enumerate(targets):
        mapped = root/f'slab_uv_{i}.obj'
        add_planar_uv(mesh.get('filename'), mapped)
        mesh.set('filename', str(mapped))
    def nonvisual(element):
        element = copy.deepcopy(element)
        for link in element.findall('link'):
            for visual in link.findall('visual'):
                link.remove(visual)
        return ET.tostring(element)
    assert nonvisual(baseline) == nonvisual(tree.getroot()), 'UV edit changed nonvisual URDF'
    tree.write(str(root/'door.urdf'))
    (root/'source.json').write_text(json.dumps(dict(source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), visual_uv_only=True), indent=2))
    return str(root), 'door.urdf'


def nonvisual_xml(element):
    element = copy.deepcopy(element)
    for link in element.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
    return ET.tostring(element)


def wood_visual_group(visual):
    """Explicit wood allowlist: never classify glass/gaskets by rigid body."""
    material = visual.find('material')
    name = material.get('name', '') if material is not None else ''
    if name in ('pale_whitewashed_wood', 'pale_oak'):
        return 'panel'
    if name in ('light_frame_wood', 'frame_oak'):
        return 'frame'
    if visual.get('name') in ('door_slab', 'door_leaf_opaque'):
        return 'panel'
    return None


def write_box_obj(path, size):
    x, y, z = np.asarray(size, float) / 2
    vertices = [(-x,-y,-z),(x,-y,-z),(x,y,-z),(-x,y,-z),
                (-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)]
    faces = [(1,4,3,2),(5,6,7,8),(1,2,6,5),(4,8,7,3),(1,5,8,4),(2,3,7,6)]
    lines = ['v ' + ' '.join(map(str, v)) for v in vertices]
    for a,b,c,d in faces:
        lines.extend((f'f {a} {b} {c}', f'f {a} {c} {d}'))
    Path(path).write_text('\n'.join(lines) + '\n')


def prepare_per_visual_door(runtime, asset_root, door_file):
    """Bake only wooden visual meshes; preserve nonvisual XML byte-for-byte.

    Asset loading precedes environment creation. A varying per-env texture
    cannot be baked into one shared asset, so this path deliberately requires
    the single-env rollout used by recording/evaluation instead of silently
    assigning environment zero's materials to every environment.
    """
    if getattr(runtime, 'num_envs', 1) != 1:
        raise ValueError('Per-visual door materials require one environment per simulator process')
    source = (Path(asset_root) / door_file).resolve()
    tree = ET.parse(source)
    for node in tree.findall('.//mesh') + tree.findall('.//texture'):
        if node.get('filename'):
            node.set('filename', str((source.parent / node.get('filename')).resolve()))
    baseline = copy.deepcopy(tree.getroot())
    root = Path(tempfile.mkdtemp(prefix='wc_per_visual_'))
    descriptions = runtime.environment_materials(0)
    changed, preserved = [], []
    for link in tree.findall('link'):
        for visual in link.findall('visual'):
            group = wood_visual_group(visual)
            tag = link.get('name') + '/' + str(visual.get('name'))
            if group is None:
                preserved.append(tag)
                continue
            geometry = visual.find('geometry')
            mesh, box = geometry.find('mesh'), geometry.find('box')
            idx = len(changed)
            if mesh is not None:
                original = Path(mesh.get('filename'))
                if original.suffix.lower() != '.obj':
                    raise ValueError('Wood visual requires OBJ: ' + str(original))
            elif box is not None:
                original = root / f'wood_box_{idx}.obj'
                write_box_obj(original, list(map(float, box.get('size').split())))
                geometry.remove(box)
                mesh = ET.SubElement(geometry, 'mesh')
            else:
                raise ValueError('Unsupported wooden visual geometry: ' + tag)
            mapped = root / f'wood_{idx}.obj'
            if any(s.startswith('vt ') for s in original.read_text().splitlines()):
                mapped.write_text(original.read_text())
            else:
                add_planar_uv(original, mapped)
            lines = [s for s in mapped.read_text().splitlines()
                     if not s.startswith(('mtllib ', 'usemtl '))]
            mapped.write_text(f'mtllib {group}.mtl\nusemtl wood\n' + '\n'.join(lines) + '\n')
            mesh.set('filename', str(mapped))
            for material in list(visual.findall('material')):
                visual.remove(material)
            texture = runtime.texture(descriptions[group])
            assert cv2.imwrite(str(root / f'{group}.png'), texture[..., ::-1])
            (root / f'{group}.mtl').write_text(
                'newmtl wood\nKa 1 1 1\nKd 1 1 1\nKs 0.04 0.04 0.04\n'
                f'd 1\nillum 2\nmap_Kd {group}.png\n')
            changed.append(tag)
    if not changed:
        raise ValueError('No recognized wooden visual parts: ' + str(source))
    assert nonvisual_xml(baseline) == nonvisual_xml(tree.getroot()), 'Per-visual materials changed physics'
    tree.write(str(root / 'door.urdf'))
    manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    modified_wood_visuals=changed, preserved_visuals=preserved,
                    nonvisual_unchanged=True, rigid_body_texture_override=False,
                    original_glass_alpha_preserved=True)
    (root / 'source.json').write_text(json.dumps(manifest, indent=2))
    runtime.per_visual_door_materials = manifest
    runtime.per_visual_door_root = str(root)
    return str(root), 'door.urdf'
