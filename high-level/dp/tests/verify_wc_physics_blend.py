"""Reopen the baked .blend and check every frame against actual body poses."""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Quaternion, Vector


def pose(v):
    m=Quaternion((v[6],*v[3:6])).to_matrix().to_4x4()
    m.translation=Vector(v[:3]);return m


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output-name',default='blender')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    assert Path(a.output_name).name==a.output_name
    out=a.trial/a.output_name
    report=json.loads((out/'bake_complete.json').read_text())
    data=json.loads((a.trial/'trajectory.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str(out/'wc_physical_pull_animated.blend'))
    s=bpy.context.scene;convert=pose(data['frames'][0]['replay_door_root_state']).inverted()
    pos_error=rot_error=0.;checked=0
    for index,source_frame in enumerate(report['frames'],1):
        s.frame_set(index);bpy.context.view_layer.update();frame=data['frames'][source_frame]
        expected={'robot/world':convert@pose(frame['replay_root_state'])}
        for role in frame['bodies']:
            for name,value in frame['bodies'][role].items():
                key='robot/'+name if role!='door' else name
                if key in bpy.data.objects:expected[key]=convert@pose(value)
        for name,matrix in expected.items():
            actual=bpy.data.objects[name].matrix_world
            error=(actual.translation-matrix.translation).length
            dot=abs(actual.to_quaternion().normalized().dot(matrix.to_quaternion().normalized()))
            angle=2*math.acos(min(1.,dot))
            pos_error=max(pos_error,error);rot_error=max(rot_error,angle);checked+=1
            assert error<1e-4,(index,name,error)
            # mathutils uses float32; dot-product angle loses precision near 0.
            assert angle<.002,(index,name,angle)
    summary=dict(passed=True,frames=len(report['frames']),checked_body_poses=checked,
                 max_position_error_m=pos_error,max_orientation_error_rad=rot_error,
                 integer_frames_are_recorded_physics=True,blender_contact_solver=False)
    (out/'blend_pose_verification.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
