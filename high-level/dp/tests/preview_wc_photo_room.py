"""Render approximate phone viewpoints for layout review, plus actual RGB cameras."""
import argparse
import json
from pathlib import Path
import bpy
from mathutils import Vector


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--samples',type=int,default=12)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(a.scene))
    s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=a.samples
    s.cycles.use_denoising=True;s.render.use_persistent_data=True
    s.render.threads_mode='FIXED';s.render.threads=4
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    gpu=next(d for d in prefs.devices if d.type=='OPTIX')
    for d in prefs.devices:d.use=d==gpu
    s.cycles.device='GPU';s.view_settings.view_transform='AgX'
    s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG'
    camera=bpy.data.cameras.new('Approximate phone reference')
    camera.lens=17.5;camera.sensor_fit='VERTICAL';camera.sensor_height=32;camera.clip_start=.03
    obj=bpy.data.objects.new(camera.name,camera);s.collection.objects.link(obj)
    obj.rotation_mode='QUATERNION'
    robot=[o for o in bpy.data.objects if o.type=='MESH' and o.name.startswith('robot/')]
    hidden={o:o.hide_render for o in robot}
    for o in robot:o.hide_render=True
    records=[]
    for number,frame,loc,target in [
        (1,1,(.25,-2.35,1.62),(0,0,1.62)),
        (2,201,(.20,-1.80,1.62),(0,0,1.37)),
        (3,401,(.13,-.85,1.62),(.25,4.0,1.28)),
        (4,401,(.13,-.85,1.62),(.25,4.0,1.28)),
        (5,451,(.12,.10,1.62),(.48,4.4,1.43)),
        (6,520,(.10,.25,1.62),(.35,4.65,.95))]:
        s.frame_set(frame);s.camera=obj;obj.location=loc
        obj.rotation_quaternion=(Vector(target)-obj.location).to_track_quat('-Z','Y')
        s.render.resolution_x=540;s.render.resolution_y=720
        s.render.filepath=str(a.out/f'photo_{number}_layout.png')
        bpy.ops.render.render(write_still=True)
        records.append(dict(photo=number,blender_frame=frame,location=loc,target=target,robot_hidden=True))
    for o,hide in hidden.items():o.hide_render=hide
    s.render.resolution_x=640;s.render.resolution_y=480
    for frame in (1,241,401,520):
        s.frame_set(frame)
        for view in ('front','wrist','overview'):
            s.camera=bpy.data.objects['Replay '+view]
            s.render.filepath=str(a.out/f'{view}_{frame-1:06d}.png')
            bpy.ops.render.render(write_still=True)
    (a.out/'preview.json').write_text(json.dumps(dict(
        scene=str(a.scene),reference_views=records,
        note='Approximate human-height layout comparison, not calibrated photogrammetry. Robot hidden ONLY for photo layout comparisons; retained in RGB camera previews.'),indent=2))


if __name__=='__main__':main()
