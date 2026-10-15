"""Unitree-inspired visual materials on the EXISTING A2W/Z1 mesh geometry.

Official references: https://www.unitree.com/z1/ and https://www.unitree.com/b1/
Z1 CAD https://github.com/unitreerobotics/unitree_ros/tree/master/robots/z1_description
The official DAE has geometry but no material/image library. This is a material
reconstruction, not official production textures or an exact B1 robot model.
"""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

NS = 'http://www.collada.org/2005/11/COLLADASchema'
N = {'c': NS}
ET.register_namespace('', NS)

MATERIALS = {
    'silver': ([.55,.59,.66], [.65,.65,.68], 64),
    'cover': ([.16,.19,.24], [.22,.22,.24], 30),
    'black': ([.022,.025,.030], [.06,.06,.06], 12),
    'rubber': ([.025,.028,.035], [.015,.015,.015], 4),
}


def q(tag):
    return '{'+NS+'}'+tag


def logo_texture(path):
    """Reconstructed white UNITREE lettering, not a claimed official decal."""
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGB', (1024,256), (180,188,202))
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    if not font_path.is_file():
        raise FileNotFoundError('Install DejaVuSans-Bold to reproduce the robot label')
    draw = ImageDraw.Draw(canvas)
    draw.text((512,128), 'UNITREE', font=ImageFont.truetype(str(font_path), 63),
              fill=(245,245,245), anchor='mm')
    canvas.save(path)


def decorate_dae(source, dest):
    tree = ET.parse(source)
    root = tree.getroot()
    geometry_before = [ET.tostring(x) for x in root.findall('c:library_geometries', N)]
    effects = ET.SubElement(root, q('library_effects'))
    materials = ET.SubElement(root, q('library_materials'))
    has_logo = Path(source).stem == 'z1_Link02'
    if has_logo:
        logo_texture(Path(dest).parent/'unitree_lettering.png')
        images = root.find('c:library_images', N)
        if images is None:
            images = ET.SubElement(root, q('library_images'))
        im = ET.SubElement(images, q('image'), id='unitree_lettering')
        ET.SubElement(im, q('init_from')).text = 'unitree_lettering.png'
    for name, (diffuse, specular, power) in MATERIALS.items():
        fx = ET.SubElement(effects, q('effect'), id=name+'_fx')
        profile = ET.SubElement(fx, q('profile_COMMON'))
        if has_logo and name == 'silver':
            surf = ET.SubElement(ET.SubElement(profile,q('newparam'),sid='label_surface'),q('surface'),type='2D')
            ET.SubElement(surf,q('init_from')).text = 'unitree_lettering'
            sampler = ET.SubElement(ET.SubElement(profile,q('newparam'),sid='label_sampler'),q('sampler2D'))
            ET.SubElement(sampler,q('source')).text = 'label_surface'
        phong = ET.SubElement(ET.SubElement(profile,q('technique'),sid='common'), q('phong'))
        for tag, color in [('ambient',diffuse),('specular',specular)]:
            ET.SubElement(ET.SubElement(phong,q(tag)),q('color')).text = ' '.join(map(str,[*color,1]))
        d = ET.SubElement(phong,q('diffuse'))
        if has_logo and name == 'silver':
            ET.SubElement(d,q('texture'),texture='label_sampler',texcoord='UVMap')
        else:
            ET.SubElement(d,q('color')).text = ' '.join(map(str,[*diffuse,1]))
        ET.SubElement(ET.SubElement(phong,q('shininess')),q('float')).text = str(power)
        # Fully opaque. Avoid old glassy/white default material fallback.
        ET.SubElement(ET.SubElement(phong,q('transparency')),q('float')).text = '1'
        mat = ET.SubElement(materials,q('material'),id=name+'_mat',name=name)
        ET.SubElement(mat,q('instance_effect'),url='#'+name+'_fx')
    for geom in root.findall('c:library_geometries/c:geometry',N):
        mesh = geom.find('c:mesh',N)
        vinput = mesh.find('c:vertices/c:input[@semantic="POSITION"]',N)
        position_source = mesh.find('c:source[@id="'+vinput.get('source')[1:]+'"]',N)
        pos = np.fromstring(position_source.find('c:float_array',N).text,sep=' ').reshape(-1,3)
        if has_logo:
            lo, span = pos.min(0), np.ptp(pos,axis=0)
            uv = (pos[:,[0,2]]-lo[[0,2]])/span[[0,2]]
            sid = geom.get('id')+'-label-uv'
            src = ET.SubElement(mesh,q('source'),id=sid)
            # COLLADA sources precede vertices/primitives. Keep importer order
            # valid, not just the XML tree and numeric arrays.
            mesh.remove(src)
            mesh.insert(list(mesh).index(mesh.find('c:vertices',N)),src)
            ET.SubElement(src,q('float_array'),id=sid+'-array',count=str(uv.size)).text = ' '.join(f'{x:.9g}' for x in uv.flat)
            acc = ET.SubElement(ET.SubElement(src,q('technique_common')),q('accessor'),source='#'+sid+'-array',count=str(len(uv)),stride='2')
            for name in ('S','T'):
                ET.SubElement(acc,q('param'),name=name,type='float')
        for tri in list(mesh.findall('c:triangles',N)):
            inputs = tri.findall('c:input',N)
            stride = max(int(x.get('offset')) for x in inputs)+1
            vi = int(next(x.get('offset') for x in inputs if x.get('semantic')=='VERTEX'))
            data = np.fromstring(tri.find('c:p',N).text,sep=' ',dtype=int).reshape(-1,3,stride)
            labels = np.zeros(len(data),int)
            if 'Gripper' in Path(source).name:
                # End portions of the existing finger mesh: dark gripping pads.
                centers = pos[data[:,:,vi]].mean(1)
                labels[centers[:,0] > pos[:,0].min()+.68*np.ptp(pos[:,0])] = 1
            for label, material in [(0,'silver'),(1,'rubber')]:
                subset = data[labels==label]
                if not len(subset):
                    continue
                target = copy.deepcopy(tri)
                target.set('material',material)
                target.set('count',str(len(subset)))
                if has_logo:
                    # Assimp reads <p> immediately using inputs seen so far.
                    # Appending TEXCOORD after <p> makes it parse 3-wide indices
                    # as 2-wide and reject the entire Link02 visual mesh.
                    uv_input = ET.Element(q('input'),semantic='TEXCOORD',source='#'+sid,offset=str(stride),set='0')
                    target.insert(list(target).index(target.find('c:p',N)),uv_input)
                    subset = np.concatenate((subset,subset[:,:,vi:vi+1]),axis=2)
                target.find('c:p',N).text = ' '.join(map(str,subset.flat))
                mesh.append(target)
            mesh.remove(tri)
    for instance in root.findall('.//c:instance_geometry',N):
        old = instance.find('c:bind_material',N)
        if old is not None:
            instance.remove(old)
        common = ET.SubElement(ET.SubElement(instance,q('bind_material')),q('technique_common'))
        for material in MATERIALS:
            inst = ET.SubElement(common,q('instance_material'),symbol=material,target='#'+material+'_mat')
            if has_logo:
                ET.SubElement(inst,q('bind_vertex_input'),semantic='UVMap',input_semantic='TEXCOORD',input_set='0')
    # Libraries must precede the scene in COLLADA 1.4.
    scene = root.find('c:scene',N)
    if scene is not None:
        root.remove(scene)
        root.append(scene)
    tree.write(dest, encoding='utf-8', xml_declaration=True)
    return dict(original_geometry_sha256=hashlib.sha256(b''.join(geometry_before)).hexdigest(),
                logo_reconstructed=has_logo, vertex_positions_unchanged=True)


def decorate_stl(source, dest, link):
    data = Path(source).read_bytes()
    count = int.from_bytes(data[80:84],'little')
    if len(data) != 84+50*count:
        raise ValueError('Expected binary STL: '+str(source))
    facets = np.frombuffer(data, dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]),offset=84,count=count)
    vertices, normals = facets['v'], facets['normal']
    center = vertices.mean(1)
    labels = np.full(count,'silver',dtype='<U8')
    if 'mount' in link:
        labels[:] = 'black'
    elif link.endswith('_wheel') or link.endswith('Link4'):
        labels[:] = 'rubber'
    elif link == 'base_link':
        low, high = vertices.reshape(-1,3).min(0), vertices.reshape(-1,3).max(0)
        # Top and sides use exactly the same silver as the legs. Only the
        # underside remains dark; the robot's upper shell is NOT black.
        labels[center[:,2] < low[2]+.16*(high[2]-low[2])] = 'cover'
    dest = Path(dest)
    mtl = dest.with_suffix('.mtl')
    mtl.write_text('\n'.join(f'newmtl {name}\nKa {" ".join(map(str,d))}\nKd {" ".join(map(str,d))}\nKs {" ".join(map(str,s))}\nNs {power}\nd 1\nillum 2\n'
                             for name,(d,s,power) in MATERIALS.items()))
    with dest.open('w') as out:
        out.write('mtllib '+mtl.name+'\n')
        for v in vertices.reshape(-1,3):
            out.write('v '+' '.join(f'{x:.9g}' for x in v)+'\n')
        for normal in normals:
            out.write('vn '+' '.join(f'{x:.9g}' for x in normal)+'\n')
        previous = None
        for i, name in enumerate(labels):
            if name != previous:
                out.write('usemtl '+name+'\n')
                previous = name
            out.write('f '+' '.join(f'{3*i+j}//{i+1}' for j in (1,2,3))+'\n')
    return dict(triangle_count=count, vertex_positions_unchanged=True,
                material_faces={str(k):int(v) for k,v in zip(*np.unique(labels,return_counts=True))})


def prepare_robot_visuals(split_root, files, runtime):
    if runtime is None or not runtime.config.get('robot_appearance',{}).get('enabled'):
        return
    root = Path(split_root)
    output = root/'unitree_visual_materials'
    output.mkdir(exist_ok=True)
    records = []
    for file in files:
        path = root/file
        tree = ET.parse(path)
        # Canonical absolute references ensure collisions still use old files.
        for mesh in tree.findall('.//mesh'):
            mesh.set('filename',str((path.parent/mesh.get('filename')).resolve()))
        before = copy.deepcopy(tree.getroot())
        for link in tree.findall('link'):
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    # Original A2W URDF includes offset red debug spheres at the
                    # wheels. Hide only these nonphysical visual markers.
                    if link.get('name','').endswith('_wheel') and visual.find('geometry/sphere') is not None:
                        link.remove(visual)
                        records.append(dict(link=link.get('name'),hidden_debug_visual='offset_sphere',
                                            collision_unchanged=True))
                    continue
                source = Path(mesh.get('filename'))
                dest = output/(source.stem + ('.dae' if source.suffix.lower()=='.dae' else '.obj'))
                if source.suffix.lower()=='.dae':
                    info = decorate_dae(source,dest)
                elif source.suffix.lower()=='.stl':
                    info = decorate_stl(source,dest,link.get('name'))
                else:
                    continue
                mesh.set('filename',str(dest))
                # Mesh submaterials must not be multiplied by old blue/white tint.
                old = visual.find('material')
                if old is not None:
                    visual.remove(old)
                ET.SubElement(ET.SubElement(visual,'material',name='unitree_visual_white'),'color',rgba='1 1 1 1')
                records.append(dict(link=link.get('name'),source=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),**info))
        def nonvisual(element):
            element=copy.deepcopy(element)
            for link in element.findall('link'):
                for visual in link.findall('visual'):
                    link.remove(visual)
            return ET.tostring(element)
        assert nonvisual(before)==nonvisual(tree.getroot()), 'Robot material edit changed nonvisual URDF'
        tree.write(path,encoding='utf-8',xml_declaration=True)
    runtime.metadata['robot_appearance'] = dict(version='unitree_visual_v1',
        base_geometry='original A2W, not B1', visual_materials_reconstructed=True,
        exact_production_texture_match=False, parts=records,
        references=['https://www.unitree.com/z1/','https://www.unitree.com/b1/',
                    'https://github.com/unitreerobotics/unitree_ros/tree/master/robots/z1_description'])
    (output/'manifest.json').write_text(json.dumps(runtime.metadata['robot_appearance'],indent=2))
