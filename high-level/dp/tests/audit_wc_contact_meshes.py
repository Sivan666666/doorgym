"""Independent triangle-mesh contact audit of recorded physical link poses."""
import argparse
import json
from pathlib import Path
import sys

JOB = Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_contact_physics_20260918')
sys.path.insert(0,str(JOB/'audit_deps'))
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def pose(v):
    m=np.eye(4);m[:3,:3]=Rotation.from_quat(v[3:7]).as_matrix();m[:3,3]=v[:3];return m


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('trial');p.add_argument('--stride',type=int,default=10)
    a=p.parse_args();out=JOB/a.trial
    data=json.loads((out/'trajectory.json').read_text())
    manifest=json.loads((JOB.parent/'rgb500_blender_review_20260918/scene_data.json').read_text())['models']['robot']
    hand_names={'gripperStator','gripperMover','wrist_up_mount','wrist_down_mount'}
    roles={n:role for role,bodies in data['frames'][0]['bodies'].items() if role!='door'
           for n in bodies if manifest['links'].get(n)}
    names=list(roles)
    robot=trimesh.collision.CollisionManager()
    hand=trimesh.collision.CollisionManager()
    meshes={}
    boxes=json.loads((JOB/'asset_preparation.json').read_text())['collision_boxes_source_coordinates']
    for name in names:
        pieces=[]
        for v in manifest['links'][name]:
            if 'mesh' in v:mesh=trimesh.load(v['mesh'],force='mesh',process=False)
            elif 'box' in v:mesh=trimesh.creation.box(extents=v['box'])
            elif 'sphere' in v:mesh=trimesh.creation.icosphere(subdivisions=3,radius=v['sphere'])
            else:raise ValueError(v)
            transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',v['origin']['rpy']).as_matrix();transform[:3,3]=v['origin']['xyz']
            mesh.apply_transform(transform);pieces.append(mesh)
        mesh=trimesh.util.concatenate(pieces)
        robot.add_object(name,mesh)
        meshes[name]=mesh
        if name in hand_names:hand.add_object(name,mesh)
    # Handle link coordinates were rotated by Rx(+90) in the physics URDF.
    mesh=trimesh.load(JOB/'assets/wc_contact/assets/meshes/lever_handle.obj',force='mesh',process=False)
    m=np.eye(4);m[:3,:3]=Rotation.from_euler('x',-np.pi/2).as_matrix();mesh.apply_transform(m)
    door=trimesh.collision.CollisionManager();door.add_object('handle',mesh)
    for name,files in [('frame',['frame.obj']),('door_leaf',['door_leaf_opaque.obj','door_leaf_frosted_lite.obj'])]:
        for filename in files:
            path=JOB/'assets/wc_contact/assets/meshes'/filename
            if not path.exists():
                raise FileNotFoundError(path)
            panel=trimesh.load(path,force='mesh',process=False)
            if name=='door_leaf':panel.apply_translation([0,.021,0])
            door.add_object(filename,panel)
    records=[]
    for i in range(0,len(data['frames']),a.stride):
        f=data['frames'][i]
        for name in names:
            robot.set_transform(name,pose(f['bodies'][roles[name]][name]))
            if name in hand_names:hand.set_transform(name,pose(f['bodies'][roles[name]][name]))
        door.set_transform('handle',pose(f['bodies']['door']['lever_handle']))
        door.set_transform('frame.obj',pose(f['bodies']['door']['frame']))
        for filename in ('door_leaf_opaque.obj','door_leaf_frosted_lite.obj'):
            door.set_transform(filename,pose(f['bodies']['door']['door_leaf']))
        contact, pairs, details=robot.in_collision_other(door,return_names=True,return_data=True)
        internal, internal_pairs, internal_details=hand.in_collision_internal(return_names=True,return_data=True)
        depths={}
        for c in details+internal_details:
            key='|'.join(sorted(c.names));depths[key]=max(depths.get(key,0),float(c.depth))
        vertex_depths={}
        for pair in pairs:
            robot_name=next(n for n in pair if n in meshes)
            door_name=next(n for n in pair if n not in meshes)
            link='lever_handle' if door_name=='handle' else ('frame' if door_name=='frame.obj' else 'door_leaf')
            relative=np.linalg.inv(pose(f['bodies']['door'][link]))@pose(f['bodies'][roles[robot_name]][robot_name])
            points=trimesh.transform_points(meshes[robot_name].vertices,relative)
            if link=='door_leaf':points[:,1]-=.021
            if link=='lever_handle':points=points[:,[0,2,1]]*np.array([1,-1,1])
            deepest=0.
            for box in boxes[link]:
                depth=np.minimum(points-np.asarray(box['low']),np.asarray(box['high'])-points).min(axis=1)
                deepest=max(deepest,float(depth.max(initial=0)))
            vertex_depths['|'.join(sorted(pair))]=deepest
        records.append(dict(step=i,phase=f['phase'],external=sorted([sorted(v) for v in pairs]),
                            internal=sorted([sorted(v) for v in internal_pairs]),triangle_depths=depths,
                            external_vertex_inside_depth_m=vertex_depths))
    (out/'mesh_contact_audit.json').write_text(json.dumps(records,indent=2))
    maxima={};vertex_maxima={}
    for row in records:
        for k,v in row['triangle_depths'].items():maxima[k]=max(maxima.get(k,0),v)
        for k,v in row['external_vertex_inside_depth_m'].items():vertex_maxima[k]=max(vertex_maxima.get(k,0),v)
    print(json.dumps(dict(frames=len(records),triangle_contact_depth_max=maxima,
                         external_vertex_inside_depth_max_m=vertex_maxima),indent=2))


if __name__=='__main__':main()
