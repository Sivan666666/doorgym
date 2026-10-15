"""Fast preview only: two distant point lights on the doorway normal axis."""
import argparse
import json
from pathlib import Path
import sys
import bpy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sim-data',required=True)
    p.add_argument('--video',action='store_true')
    a = p.parse_args()
    sys.path.insert(0,a.sim_data)
    from augmentation import material_manifest
    from common import digest, save
    a.out.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=a.scene)
    s = bpy.context.scene
    before = digest(material_manifest(bpy))
    for obj in bpy.data.objects:
        if obj.type=='LIGHT':
            obj.data.energy = 0
    if s.world and s.world.use_nodes:
        for node in s.world.node_tree.nodes:
            if node.type=='BACKGROUND':
                assert not node.inputs['Strength'].is_linked
                node.inputs['Strength'].default_value = 0
    lights = []
    for name,y in [('Front distant point',-4.0),('Back distant point',3.5)]:
        light = bpy.data.lights.new(name,'POINT')
        light.energy = 12.0
        light.color = (1.,.97,.93)
        light.shadow_soft_size = .08
        obj = bpy.data.objects.new(name,light)
        s.collection.objects.link(obj)
        obj.location = (0,y,1.6)
        lights.append(dict(name=name,position=list(obj.location),energy=light.energy,type=light.type))
    assert digest(material_manifest(bpy)) == before
    s.render.resolution_x,s.render.resolution_y = 480,360
    s.render.resolution_percentage = 100
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode = 'RGB'
    s.render.threads_mode = 'FIXED';s.render.threads = 4
    s.view_settings.view_transform = 'AgX'
    s.view_settings.exposure = 0
    s.view_settings.gamma = 1
    s.render.use_persistent_data = True
    s.render.engine = 'CYCLES'
    s.cycles.samples = 4
    s.cycles.use_denoising = True
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    device=next(d for d in prefs.devices if d.type=='OPTIX')
    for d in prefs.devices:d.use=d==device
    s.cycles.device='GPU'
    frames = range(0,520,5) if a.video else (0,240,490)
    # 2 FPS quick whole-trajectory preview: preserves 52-second duration.
    for i in frames:
        s.frame_set(i+1)
        for view in ('front','wrist','overview'):
            s.camera=bpy.data.objects['Replay '+view]
            target=a.out/view/f'{i:06d}.png'
            target.parent.mkdir(exist_ok=True)
            if target.exists():continue
            s.render.filepath=str(target)
            bpy.ops.render.render(write_still=True)
        print('PREVIEW',i,flush=True)
    save(a.out/'settings.json',dict(preview_only=True,lights=lights,
        placement='Door normal is scene Y: front y=-4 m / back y=+3.5 m; same x=0, z=1.6 m, not left/right.',
        material_sha256=before,source_scene=a.scene,frames=list(frames),width=480,height=360,samples=4,
        no_photometric_brightening=True,world_strength=0,other_lights_disabled=True))


if __name__=='__main__':main()
