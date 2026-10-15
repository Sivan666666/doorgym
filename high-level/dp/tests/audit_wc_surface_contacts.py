"""Resolve triangle-contact warnings with dense surface samples vs solid boxes."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from audit_wc_contact_meshes import pose, JOB


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('trial');p.add_argument('--body',default='wrist_down_mount')
    a=p.parse_args();out=JOB/a.trial
    frames=json.loads((out/'trajectory.json').read_text())['frames']
    audit=json.loads((out/'mesh_contact_audit.json').read_text())
    model=json.loads((JOB.parent/'rgb500_blender_review_20260918/scene_data.json').read_text())['models']['robot']
    v=model['links'][a.body][0];mesh=trimesh.load(v['mesh'],force='mesh',process=False)
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',v['origin']['rpy']).as_matrix();transform[:3,3]=v['origin']['xyz']
    mesh.apply_transform(transform)
    points,_=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=.002,max_iter=10)
    boxes=json.loads((JOB/'asset_preparation.json').read_text())['collision_boxes_source_coordinates']['door_leaf']
    records=[]
    for row in audit:
        if not any(a.body in pair and 'door_leaf_opaque.obj' in pair for pair in row['external']):continue
        f=frames[row['step']]
        relative=np.linalg.inv(pose(f['bodies']['door']['door_leaf']))@pose(f['bodies']['arm'][a.body])
        xyz=trimesh.transform_points(points,relative);xyz[:,1]-=.021
        depth=np.zeros(len(points))
        lo,hi=xyz.min(axis=0),xyz.max(axis=0)
        for box in boxes:
            if np.any(hi<np.asarray(box['low'])) or np.any(lo>np.asarray(box['high'])):continue
            depth=np.maximum(depth,np.minimum(xyz-np.asarray(box['low']),np.asarray(box['high'])-xyz).min(axis=1))
        records.append(dict(step=row['step'],phase=row['phase'],max_depth_m=float(depth.max()),samples_inside=int((depth>0).sum())))
    report=dict(body=a.body,surface_samples=len(points),max_sample_edge_m=.002,frames=records,
                max_depth_m=max((v['max_depth_m'] for v in records),default=0))
    (out/f'surface_audit_{a.body}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))
    print('Worst frames',sorted(records,key=lambda v:v['max_depth_m'],reverse=True)[:5])


if __name__=='__main__':main()
