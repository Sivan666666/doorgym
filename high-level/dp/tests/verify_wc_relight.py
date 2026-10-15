"""Verify a rerender changed only light power and explicitly emissive strengths."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
import numpy as np


def snapshot(material_manifest):
    s=bpy.context.scene
    s.frame_set(1)
    objects={}
    for o in bpy.data.objects:
        item=dict(type=o.type,hide_render=o.hide_render,parent=o.parent.name if o.parent else None)
        if o.type=='MESH':
            vertices=np.array([tuple(v.co) for v in o.data.vertices],dtype=np.float32)
            item.update(mesh_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
                # Blender primitive generation can enumerate identical sphere
                # faces differently between processes; preserve winding and
                # face material, but ignore enumeration order.
                polygons=sorted((tuple(p.vertices),p.material_index) for p in o.data.polygons),
                materials=[m.name if m else None for m in o.data.materials])
        if o.type=='CAMERA':
            item.update(lens=o.data.lens,sensor_width=o.data.sensor_width,sensor_height=o.data.sensor_height,
                sensor_fit=o.data.sensor_fit,clip_start=o.data.clip_start,clip_end=o.data.clip_end)
        objects[o.name]=item
    names=sorted(objects)
    initial=np.array([np.array(bpy.data.objects[n].matrix_world) for n in names])
    animated=sorted(o.name for o in bpy.data.objects if o.animation_data and o.animation_data.action)
    motion=[]
    for frame in range(s.frame_start,s.frame_end+1):
        s.frame_set(frame)
        motion.append([np.array(bpy.data.objects[n].matrix_world) for n in animated])
    lights={o.name:dict(energy=o.data.energy,color=list(o.data.color),type=o.data.type,
        shape=getattr(o.data,'shape',None),size=getattr(o.data,'size',None),size_y=getattr(o.data,'size_y',None))
        for o in bpy.data.objects if o.type=='LIGHT'}
    world=[(n.name,n.inputs['Strength'].default_value) for n in s.world.node_tree.nodes if n.type=='BACKGROUND']
    display=dict(exposure=s.view_settings.exposure,gamma=s.view_settings.gamma,
        transform=s.view_settings.view_transform,look=s.view_settings.look,
        resolution=[s.render.resolution_x,s.render.resolution_y],fps=s.render.fps)
    return dict(objects=objects,names=names,initial=initial,animated=animated,motion=np.array(motion),
        lights=lights,world=world,display=display,materials=material_manifest(bpy))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--ratio',type=float,required=True)
    p.add_argument('--sim-data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.sim_data))
    from augmentation import material_manifest
    bpy.ops.wm.open_mainfile(filepath=str(a.before));before=snapshot(material_manifest)
    bpy.ops.wm.open_mainfile(filepath=str(a.after));after=snapshot(material_manifest)
    if before['objects']!=after['objects']:
        diffs=[]
        for name in sorted(set(before['objects'])|set(after['objects'])):
            old=before['objects'].get(name,{});new=after['objects'].get(name,{})
            changed={k:[str(old.get(k))[:160],str(new.get(k))[:160]]
                for k in set(old)|set(new) if old.get(k)!=new.get(k)}
            if changed:diffs.append(dict(object=name,changes=changed))
        print('OBJECT_DIFFERENCES',json.dumps(diffs),flush=True)
    for key in ('objects','names','animated','world','display'):
        assert before[key]==after[key],key
    for key in ('initial','motion'):
        np.testing.assert_allclose(before[key],after[key],rtol=0,atol=1e-6,err_msg=key)
    report=json.loads((a.after.parent/'room_clearance.json').read_text())
    previous=json.loads((a.before.parent/'room_clearance.json').read_text())
    assert report['original_assets_sha256']==previous['original_assets_sha256']
    assert report['source_sha256']==previous['source_sha256']
    emission_names={v['material'] for v in report['emissive_materials']}
    changes=[]
    for name,old in before['lights'].items():
        new=after['lights'][name]
        assert np.isclose(new['energy'],old['energy']*a.ratio),name
        changes.append(dict(light=name,before=old['energy'],after=new['energy']))
        new['energy']=old['energy']
        assert old==new,name
    assert set(before['materials'])==set(after['materials'])
    for name,old in before['materials'].items():
        new=after['materials'][name]
        if name in emission_names:
            old_nodes={v['name']:v for v in old['nodes']}
            for node in new['nodes']:
                if node['type']=='ShaderNodeBsdfPrincipled':
                    value=old_nodes[node['name']]['inputs']['Emission Strength']
                    assert np.isclose(node['inputs']['Emission Strength'],value*a.ratio),name
                    node['inputs']['Emission Strength']=value
        assert old==new,name
    out=dict(passed=True,light_multiplier=a.ratio,lights=changes,
        emissive_materials=report['emissive_materials'],original_assets_unchanged=True,
        source_sha256=report['source_sha256'],frames_checked=len(after['motion']),
        max_transform_difference=float(np.max(np.abs(before['motion']-after['motion']))),
        geometry_camera_and_nonemissive_materials_unchanged=True,
        world_exposure_gamma_view_transform_unchanged=True,display=after['display'])
    a.out.write_text(json.dumps(out,indent=2));print('RELIGHT_VERIFIED',a.out,flush=True)


if __name__=='__main__':main()
