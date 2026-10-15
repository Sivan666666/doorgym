"""Cycles appearance preview using recorded poses, not a new physics rollout."""
import argparse
import json
import math
import sys
from pathlib import Path
import bpy
from mathutils import Matrix, Vector, Quaternion, Euler

OUT = Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/rgb500_blender_review_20260918')
ASSETS = Path('/mnt/sda/ps_archive/doorgym/high-level/data/visual_randomization/doorman')
p = argparse.ArgumentParser()
p.add_argument('--animation', action='store_true')
p.add_argument('--samples', type=int, default=32)
a = p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
data = json.loads((OUT/'scene_data.json').read_text()); ep = data['episode']
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
s = bpy.context.scene
s.render.engine = 'CYCLES'; s.cycles.samples = a.samples; s.cycles.use_denoising = True
s.cycles.max_bounces = 6; s.cycles.diffuse_bounces = 3; s.cycles.glossy_bounces = 3
s.render.resolution_x = 640; s.render.resolution_y = 480; s.render.resolution_percentage = 100
s.render.image_settings.file_format = 'PNG'; s.render.use_persistent_data = True
s.render.threads_mode = 'FIXED'; s.render.threads = 8
s.view_settings.view_transform = 'AgX'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
selected = next(d for d in prefs.devices if d.type == 'OPTIX')
for device in prefs.devices:
    device.use = device == selected
print('DEVICES', [(d.name, d.type, d.use) for d in prefs.devices], flush=True)
s.cycles.device = 'GPU'


def pose(v):
    m = Quaternion((v[6], v[3], v[4], v[5])).to_matrix().to_4x4()
    m.translation = Vector(v[:3]); return m


def origin(o):
    m = Euler(o['rpy'], 'XYZ').to_matrix().to_4x4()
    m.translation = Vector(o['xyz']); return m


def mat(name, color, metallic=0., roughness=.5, texture=None, scale=1., uv=False):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1.)
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    if texture:
        image = m.node_tree.nodes.new('ShaderNodeTexImage'); image.image = bpy.data.images.load(str(texture), check_existing=True)
        image.projection = 'FLAT' if uv else 'BOX'; image.projection_blend = .15
        coord = m.node_tree.nodes.new('ShaderNodeTexCoord')
        mapping = m.node_tree.nodes.new('ShaderNodeVectorMath'); mapping.operation = 'SCALE'; mapping.inputs[3].default_value = scale
        m.node_tree.links.new(coord.outputs['UV' if uv else 'Generated'], mapping.inputs[0])
        m.node_tree.links.new(mapping.outputs['Vector'], image.inputs['Vector'])
        m.node_tree.links.new(image.outputs['Color'], bsdf.inputs['Base Color'])
    return m


materials = dict(
    silver=mat('Brushed silver robot', (.52,.55,.58), .82, .27),
    rubber=mat('Rubber', (.015,.018,.022), .0, .78),
    black=mat('Black handle and camera mounts', (.014,.016,.018), .45, .3),
    leaf=mat('Recorded gray wood basecolor', (.55,.59,.60), roughness=.48,
             texture=OUT/'textures/panel.png', uv=True),
    frame=mat('Recorded frame wood', (.35,.22,.13), roughness=.48,
              texture=ASSETS/(ep['rgb_visual_episode']['materials']['frame']['name']+'.png')),
    floor=mat('Wood floor', (.44,.30,.18), roughness=.42,
              texture=ASSETS/(ep['rgb_visual_episode']['materials']['floor']['name']+'.png'), scale=6.),
    wall=mat('Recorded wall wood', (.53,.48,.42), roughness=.67,
             texture=ASSETS/(ep['rgb_visual_episode']['materials']['wall']['name']+'.png'), scale=2.),
    plaster=mat('Ceiling plaster', (.78,.78,.76), roughness=.9),
    glass=mat('Opaque source inset', (.12,.18,.17), .15, .3))


def cube(name, size, position, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object; obj.name = name; obj.scale = Vector(size)
    obj.data.materials.append(material); return obj


models = {}
for role, model in data['models'].items():
    roots = set(model['links']) - {j['child'] for j in model['joints']}
    objects = {}
    for name in model['links']:
        obj = bpy.data.objects.new(role+'/'+name, None); s.collection.objects.link(obj); objects[name] = obj
    for joint in model['joints']:
        child = objects[joint['child']]; child.parent = objects[joint['parent']]
        child.matrix_local = origin(joint['origin'])
    for name, visuals in model['links'].items():
        for v in visuals:
            if 'mesh' in v:
                bpy.ops.wm.ply_import(filepath=v['mesh']); obj = bpy.context.object
                scale = v['scale']
                for poly in obj.data.polygons:
                    poly.use_smooth = True
            elif 'box' in v:
                bpy.ops.mesh.primitive_cube_add(size=1); obj = bpy.context.object; scale = v['box']
            elif 'cylinder' in v:
                bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=v['cylinder'][0], depth=v['cylinder'][1]); obj=bpy.context.object; scale=[1]*3
            elif 'sphere' in v:
                bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=v['sphere']); obj=bpy.context.object; scale=[1]*3
            else:
                raise ValueError(v)
            obj.name=role+'/'+name+'/'+v['name']; obj.parent=objects[name]
            obj.matrix_local=origin(v['origin']) @ Matrix.Diagonal((*scale,1.))
            if role == 'robot':
                key='rubber' if ('wheel' in name.lower() or name.endswith('Link4')) else 'black' if 'mount' in name else 'silver'
            else:
                key='frame' if name=='fixed_frame' else 'leaf' if name=='door_panel' else 'black'
                if key=='leaf':
                    coords=[vert.co for vert in obj.data.vertices]
                    low=[min(v[i] for v in coords) for i in range(3)]
                    span=[max(max(v[i] for v in coords)-low[i],1e-8) for i in range(3)]
                    uv_layer=obj.data.uv_layers.active or obj.data.uv_layers.new(name='Recorded planar UV')
                    for poly in obj.data.polygons:
                        excluded=max(range(3),key=lambda i:abs(poly.normal[i]));axes=[i for i in range(3) if i!=excluded]
                        for loop_idx in poly.loop_indices:
                            point=obj.data.vertices[obj.data.loops[loop_idx].vertex_index].co
                            uv_layer.data[loop_idx].uv=[(point[i]-low[i])/span[i] for i in axes]
            obj.data.materials.clear(); obj.data.materials.append(materials[key])
            if 'box' in v and min(v['box'])>.003:
                bevel=obj.modifiers.new('Subtle edge highlight', 'BEVEL'); bevel.width=min(.001,min(v['box'])/8); bevel.segments=2
    models[role]=(model, objects, roots)

room=ep['rgb_visual_episode']['indoor_room']; cx,cy=room['center']
cube('Floor', (14,12,.08), (cx,cy,-.04), materials['floor'])
# Side-wall opening and room envelope from recording metadata; rendering only.
for side in (-1,1):
    cube('Door side wall', (.08,ep['door_wall_side_width'],ep['door_wall_height']),
         (cx,cy+side*(.525+.5*ep['door_wall_side_width']),ep['door_z_offset']+.5*ep['door_wall_height']),materials['wall'])
cube('Far room wall', (.08,6,3.2), (cx-room['far_wall_distance_m'],cy,1.6), materials['wall'])
for side in (-1,1):
    cube('Room side', (room['far_wall_distance_m']+2,.08,3.2),
         (cx+1-room['far_wall_distance_m']/2,cy+side*3,1.6),materials['plaster'])
cube('Ceiling', (room['far_wall_distance_m']+2,6,.06), (cx+1-room['far_wall_distance_m']/2,cy,3.23),materials['plaster'])

world = bpy.data.worlds.new('HDR soft environment'); s.world=world; world.use_nodes=True
world.node_tree.nodes['Background'].inputs['Strength'].default_value=.2
hdr = ASSETS/'4c3d7d706b4f0362.hdr'
if hdr.exists():
    env=world.node_tree.nodes.new('ShaderNodeTexEnvironment'); env.image=bpy.data.images.load(str(hdr))
    world.node_tree.links.new(env.outputs['Color'],world.node_tree.nodes['Background'].inputs['Color'])
for name, loc, energy, size in [('Ceiling softbox',(cx+1.,cy-1.,3.05),250,2.5),('Room fill',(cx-2,cy+1.,3.05),180,2.)]:
    light=bpy.data.lights.new(name,'AREA'); light.energy=energy; light.shape='DISK'; light.size=size
    obj=bpy.data.objects.new(name,light); s.collection.objects.link(obj); obj.location=loc

cameras={}
for name in ('front','wrist','overview'):
    cam=bpy.data.cameras.new(name); obj=bpy.data.objects.new(name,cam); s.collection.objects.link(obj)
    cam.type='PERSP'; cam.sensor_fit='HORIZONTAL'; cam.sensor_width=36
    cam.lens=36/(2*math.tan(math.radians(55)/2)); cam.clip_start=.01; cam.clip_end=100
    cameras[name]=obj
cameras['overview'].location=(cx+3.2,cy-3.4,2.5)
cameras['overview'].rotation_euler=(Vector((cx+.3,cy,1.0))-cameras['overview'].location).to_track_quat('-Z','Y').to_euler()
cameras['overview'].data.lens=34
# Recorded Gym attachment frame: +X forward, +Z up. Blender: -Z forward,+Y up.
camera_axes=Matrix(((0,0,-1,0),(-1,0,0,0),(0,1,0,0),(0,0,0,1)))


def set_frame(index):
    base=pose(ep['replay_root_state'][index]); door=pose(ep['replay_door_root_state'][index])
    for role,(model,objects,roots) in models.items():
        for root in roots:
            objects[root].matrix_world=base if role=='robot' else door
        q={f'joint{i+1}':ep['replay_dof_pos'][index][-7+i] for i in range(6)}
        q['jointGripper']=ep['replay_dof_pos'][index][-1]
        # The float Gym base is a separate visual actor held in this fixed
        # loader pose; raw replay's first 12 entries are placeholders, not legs.
        if role=='robot':
            for limb in ('FL','FR','RL','RR'):
                q.update({limb+'_hip_joint':0.,limb+'_thigh_joint':.8,limb+'_calf_joint':-1.5})
        if role=='door':
            movable=[j for j in model['joints'] if j['type'] not in ('fixed','floating')]
            assert len(movable)==len(ep['replay_door_dof_pos'][index])
            q=dict(zip([j['name'] for j in movable],ep['replay_door_dof_pos'][index]))
        for joint in model['joints']:
            amount=q.get(joint['name'],0.)
            if joint['mimic']:
                m=joint['mimic'];amount=q.get(m['joint'],0.)*float(m.get('multiplier',1))+float(m.get('offset',0))
            motion=Matrix.Identity(4)
            if joint['type'] in ('revolute','continuous'):
                motion=Matrix.Rotation(amount,4,Vector(joint['axis']))
            elif joint['type']=='prismatic':
                motion=Matrix.Translation(Vector(joint['axis'])*amount)
            objects[joint['child']].matrix_local=origin(joint['origin'])@motion
    for name in ('front','wrist'):
        cameras[name].matrix_world=base@pose(ep[name+'_camera_pose_base'][index])@camera_axes
    bpy.context.view_layer.update()


frames=list(range(150,390,2)) if a.animation else [150]
for number,index in enumerate(frames):
    set_frame(index)
    for name in cameras:
        s.camera=cameras[name]; folder=OUT/('frames' if a.animation else 'probe')/name; folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'{number:04d}.png'
        if target.exists():
            continue
        s.render.filepath=str(target); bpy.ops.render.render(write_still=True)
    print('FRAME',number,'source',index,flush=True)
if not a.animation:
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'replay_appearance.blend'))
(OUT/('render_complete.json' if a.animation else 'probe_complete.json')).write_text(json.dumps(dict(
    source=data['raw'], source_frames=frames, source_fps=25, output_fps=12.5,
    renderer='Blender 5.1 Cycles OptiX', samples=a.samples, cameras=['front','wrist','overview'],
    dimensions=[640,480], replay='kinematic only; not a new policy or physics evaluation',
    changes='PBR materials, HDR plus area lighting, AgX; original meshes and recorded poses. Room walls reconstructed.',
    camera_convention='Gym attachment +X forward/+Z up mapped to Blender -Z forward/+Y up'),indent=2))
