"""Render actual PhysX body poses with original-size WC visuals in Cycles."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

JOB=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_contact_physics_20260918')


def pose(v):
    m=Quaternion((v[6],*v[3:6])).to_matrix().to_4x4();m.translation=Vector(v[:3]);return m


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--trial',required=True)
    p.add_argument('--probe',action='store_true')
    p.add_argument('--frames',default='0,420,480,550,700')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--stop',type=int)
    p.add_argument('--stride',type=int,default=4)
    p.add_argument('--views',default='overview,contact')
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--bake-only',action='store_true')
    p.add_argument('--end-step',type=int)
    p.add_argument('--white-walls',action='store_true',help='Add render-only white walls around the physical door frame')
    p.add_argument('--furnished-room',action='store_true',help='White ceiling and distant, clearance-audited room props')
    p.add_argument('--output-name',default='blender')
    p.add_argument('--base-scene',type=Path,default=JOB.parent/'wc_pull_blender_20260918/pull_replay_setup.blend')
    p.add_argument('--scene-config',type=Path)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    assert Path(a.output_name).name==a.output_name
    assert not a.white_walls or a.output_name!='blender','Use a separate output directory for the wall scene'
    assert not a.furnished_room or (a.white_walls and a.output_name not in ('blender','blender_white_walls','blender_white_walls_multiview'))
    out=JOB/a.trial/a.output_name;out.mkdir(parents=True,exist_ok=True)
    data=json.loads((JOB/a.trial/'trajectory.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str(a.base_scene))
    s=bpy.context.scene
    # Undo the old appearance-only fitting. All new visuals are in source metres.
    scale=Vector((.868/.8613,.058/.043,.985/.922))
    for name in ('frame','door_leaf','lever_handle'):
        for obj in bpy.data.objects[name].children:
            if obj.type=='MESH':
                for v in obj.data.vertices:
                    v.co=Vector((v.co.x/scale.x,v.co.y/scale.y,v.co.z/scale.z))
                obj.matrix_basis=Matrix.Identity(4)
                if name=='door_leaf':obj.matrix_basis=Matrix.Translation(Vector((0,.021,0)))
                if name=='lever_handle':obj.matrix_basis=Matrix.Rotation(-math.pi/2,4,'X')
    s.cycles.samples=a.samples;s.cycles.use_denoising=True
    s.render.use_persistent_data=True;s.render.threads_mode='FIXED';s.render.threads=8
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    device=next(d for d in prefs.devices if d.type=='OPTIX')
    for d in prefs.devices:d.use=d==device
    s.cycles.device='GPU'
    for name in ('Cutaway side wall L','Cutaway side wall R','Cutaway header'):
        bpy.data.objects[name].hide_render=True
    convert=pose(data['frames'][0]['replay_door_root_state']).inverted()
    camera_axes=Matrix(((0,0,-1,0),(-1,0,0,0),(0,1,0,0),(0,0,0,1)))
    cameras={name:bpy.data.objects['Replay '+name] for name in ('overview','front','wrist')}
    camdata=bpy.data.cameras.new('Physical grasp closeup');camdata.lens=52
    camdata.clip_start=.01
    cam=bpy.data.objects.new('Physical grasp closeup',camdata);s.collection.objects.link(cam)
    cameras['contact']=cam
    errors=[]
    def update(i):
        f=data['frames'][i]
        base=convert@pose(f['replay_root_state'])
        bpy.data.objects['robot/world'].matrix_world=base
        bpy.context.view_layer.update()
        for role in f['bodies']:
            for name,v in f['bodies'][role].items():
                key='robot/'+name if role!='door' else name
                if key in bpy.data.objects:
                    bpy.data.objects[key].matrix_world=convert@pose(v)
                    bpy.context.view_layer.update()
        for name in ('front','wrist'):
            cameras[name].matrix_world=base@pose(f[name+'_camera_pose_base'])@camera_axes
        target=convert@pose(f['bodies']['door']['lever_handle'])@Vector((-.075,0,.01))
        cam.location=target+Vector((.32,-.55,.25))
        cam.rotation_mode='QUATERNION'
        cam.rotation_quaternion=(target-cam.location).to_track_quat('-Z','Y')
        bpy.context.view_layer.update()
        error=(bpy.data.objects['robot/ee_gripper_link'].matrix_world.translation-convert@Vector(f['replay_ee_pos'])).length
        assert error<1e-5,error
        errors.append(error)
    walls=[]
    if a.white_walls:
        # Fit the opening to the corrected, original-size frame, not the old
        # appearance-only fitted mesh. These walls do not alter the rollout.
        update(0)
        bounds=[obj.matrix_world@Vector(c) for obj in bpy.data.objects['frame'].children
                if obj.type=='MESH' for c in obj.bound_box]
        lo=min(v.x for v in bounds);hi=max(v.x for v in bounds);top=max(v.z for v in bounds)
        height=3.3;extent=4.;depth=.12;wall_y=.1
        plaster=bpy.data.materials['Preview only - neutral white plaster'].copy()
        plaster.name='White wall plaster - physical replay surround'
        plaster.diffuse_color=(.85,.85,.85,1)
        plaster.use_nodes=True
        bsdf=next(n for n in plaster.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        for key,value in [('Base Color',(.85,.85,.85,1)),('Roughness',.85),('Metallic',0.),('Alpha',1.)]:
            for link in list(bsdf.inputs[key].links):plaster.node_tree.links.remove(link)
            bsdf.inputs[key].default_value=value
        specs=[('Cutaway side wall L',((lo-extent)/2,wall_y,height/2),(lo+extent,depth,height)),
               ('Cutaway side wall R',((hi+extent)/2,wall_y,height/2),(extent-hi,depth,height)),
               ('Cutaway header',((hi+lo)/2,wall_y,(top+height)/2),(hi-lo,depth,height-top))]
        for name,location,size in specs:
            obj=bpy.data.objects[name];obj.hide_render=False;obj.hide_viewport=False
            obj.location=location;obj.rotation_euler=(0,0,0);obj.scale=size
            obj.data.materials.clear();obj.data.materials.append(plaster)
            walls.append(dict(name=name,location=list(location),dimensions=list(size)))
        bpy.context.view_layer.update()
        s['wall_note']='White surround walls are render-only; original recorded physical trajectory unchanged.'
    room=None
    if a.furnished_room:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        room_cfg=json.loads(a.scene_config.read_text()) if a.scene_config else None
        if room_cfg and room_cfg.get('variant')=='photo_office_20261008':
            from furnish_wc_photo_room import furnish_room
        else:
            from furnish_wc_blender_room import furnish_room
        room=furnish_room(s,data,convert,pose,JOB/a.trial/'trajectory.json',out,room_cfg)
    end=len(data['frames']) if a.end_step is None else a.end_step
    assert 0<end<=len(data['frames'])
    assert all(f['phase']=='hold_home' for f in data['frames'][end:]),'Only idle tail may be omitted'
    selected=[int(v) for v in a.frames.split(',')] if a.probe else list(range(0,end,a.stride))[a.start:a.stop]
    for i in selected:
        if a.bake_only:s.frame_set(i//a.stride+1)
        update(i)
        if a.bake_only:
            objects=[bpy.data.objects['robot/world'],*cameras.values()]
            for role in data['frames'][i]['bodies']:
                for name in data['frames'][i]['bodies'][role]:
                    key='robot/'+name if role!='door' else name
                    if key in bpy.data.objects:objects.append(bpy.data.objects[key])
            for obj in objects:
                obj.rotation_mode='QUATERNION'
                obj.keyframe_insert('location')
                obj.keyframe_insert('rotation_quaternion')
            continue
        for name in a.views.split(','):
            s.camera=cameras[name]
            s.render.resolution_x,s.render.resolution_y=(a.width,a.width*3//4) if name in ('overview','contact') else (640,480)
            folder=out/('probe' if a.probe else 'frames')/name;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{i if a.probe else i//a.stride:04d}.png'
            if not path.exists():
                s.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        print('RENDERED_PHYSICAL_FRAME',i,flush=True)
    if a.probe:
        update(selected[0]);s.camera=cameras['overview'];bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'physical_replay_setup.blend'))
    if a.bake_only:
        s.render.fps=round(data['fps']/a.stride);s.render.fps_base=s.render.fps/(data['fps']/a.stride)
        s.frame_start=1;s.frame_end=len(selected);s.frame_set(1);s.camera=cameras['overview']
        s.render.resolution_x=a.width;s.render.resolution_y=a.width*3//4
        s['physical_replay_note']='Actual PhysX poses at each integer frame; prescribed floating base; Blender does not re-simulate contacts.'
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'wc_physical_pull_animated.blend'))
    report=dict(source=str(JOB/a.trial/'trajectory.json'),frames=selected,source_fps=data['fps'],
                output_fps=data['fps']/a.stride,geometry_scale=[1,1,1],actual_physics_body_poses=True,
                views=a.views.split(','),samples=a.samples,width=a.width,max_ee_error_m=max(errors,default=0),
                end_step=end,raw_source_frames=len(data['frames']),omitted_idle_frames=len(data['frames'])-end,
                white_walls=a.white_walls,render_only_walls=walls,
                scene_variant=room['scene_variant'] if room else 'white_walls' if a.white_walls else 'original',
                room_clearance_sha256=hashlib.sha256((out/'room_clearance.json').read_bytes()).hexdigest() if room else None)
    report_name='bake_complete.json' if a.bake_only else ('probe_complete.json' if a.probe else f'render_complete_{a.start}_{a.stop}.json')
    (out/report_name).write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
