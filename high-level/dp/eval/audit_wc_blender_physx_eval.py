"""Independent read-only closed-loop audit; writes only the final audit report."""
import argparse
from pathlib import Path
import numpy as np
from wc_blender_physx_common import read,save,sha,score,wilson


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--partial',action='store_true');a=p.parse_args()
    root=a.root;protocol=read(root/'protocol.json');summary=read(root/'summary.json')
    assert protocol['trials']==64
    assert a.partial or (summary['complete'] and summary['trials']==64)
    assert sha(Path(protocol['model'])/'model.safetensors')==protocol['model_sha256']
    assert protocol['score_door_open_deg']==60 and protocol['physical_handle_unlock_deg']==25
    assert protocol['score_traversal_m']==.5 and protocol['policy_action_hz']==25
    errors=[];rows=[];requests_checked=0;actions_checked=0;body_poses_checked=0;max_pose_error=0.
    for trial in range(summary['trials']):
        folder=root/'trials'/('%03d'%trial)
        metrics=read(folder/'metrics.json');trace=read(folder/'metrics_trace.json')
        assert metrics['seed']==protocol['seeds'][trial] and metrics['steps']==2600
        assert metrics['requests']==26 and metrics['policy_ticks']==1300
        recalculated=score(trace)
        for k,v in recalculated.items():assert metrics[k]==v,(trial,k)
        data=read(folder/'trajectory.json')['frames']
        assert len(data)==2600 and all(f['phase']=='dp_policy' for f in data)
        assert [r['step'] for r in trace]==list(range(2600))
        settings=read(folder/'effective_args.json')
        raw_angles=np.rad2deg([f['replay_door_dof_pos'][0] for f in data])*settings['door_motion_sign']
        np.testing.assert_allclose(raw_angles,[r['open_deg'] for r in trace],atol=1e-6,rtol=0)
        bases=np.array([f['replay_root_state'][:3] for f in data])
        np.testing.assert_allclose(bases,[r['base'] for r in trace],atol=1e-6,rtol=0)
        initial=read(root/'ipc'/('%03d'%trial)/'000/request.json')['replay_door_root_state']
        x,y,z,w=initial[3:7]
        normal=np.array([2*(x*y-w*z),1-2*(x*x+z*z)])
        normal/=np.linalg.norm(normal)
        traversal=(bases[:,:2]-np.array(initial[:2]))@normal
        np.testing.assert_allclose(traversal,[r['traversal_m'] for r in trace],atol=1e-5,rtol=0)
        props=read(folder/'physics_properties.json')['arm']
        names=props['dof_names'];indices=[names.index('joint'+str(i)) for i in range(1,7)]+[names.index('jointGripper')]
        lo=np.array(props['dof_properties']['lower'])[indices]
        hi=np.array(props['dof_properties']['upper'])[indices]
        # configure_dofs supplies the controller with the scaled sweep limits,
        # while physics_properties retains the wider URDF limits.
        defaults=np.clip(np.zeros_like(lo),lo,hi)
        scale=np.clip(settings['range_scale'],.05,1.)
        lo=defaults-(defaults-lo)*scale;hi=defaults+(hi-defaults)*scale
        for request in range(26):
            ipc=root/'ipc'/('%03d'%trial)/('%03d'%request)
            r=read(ipc/'request.json');render=read(ipc/'render.json');infer=read(ipc/'inference.json')
            step=request*100
            assert r['physics_step']==step and r['tick']==request*50 and r['simulation_time_s']==step/50
            assert render['request_sha256']==sha(ipc/'request.json')
            assert infer['observation_sha256']==sha(ipc/'observations.npz')
            assert render['source']=='live_physx' and render['max_position_error_m']<1e-4
            max_pose_error=max(max_pose_error,render['max_position_error_m'])
            with np.load(ipc/'observations.npz',allow_pickle=False) as obs:
                np.testing.assert_allclose(obs['state'],r['state'],atol=1e-7,rtol=0)
                for view in ('front','wrist'):
                    assert sha(ipc/(view+'.png'))==render['rgb_sha256'][view]
                    assert obs[view].shape==(480,640,3) and obs[view].dtype==np.uint8
            if step:
                previous=data[step-1]
                np.testing.assert_allclose(r['state'][3:],previous['control_record']['measured_joint7'],atol=1e-7,rtol=0)
                for role,records in r['bodies'].items():
                    for name,value in records.items():
                        np.testing.assert_allclose(value,previous['bodies'][role][name],atol=1e-6,rtol=0)
                        body_poses_checked+=1
                for v in ('front_camera_pose_base','wrist_camera_pose_base'):
                    np.testing.assert_allclose(r[v],previous[v],atol=1e-6,rtol=0)
            actions=np.load(ipc/'actions.npy',allow_pickle=False)
            assert actions.shape==(100,10) and np.isfinite(actions).all()
            for j in range(100):
                applied=data[step+j]['control_record']['applied_action10']
                predicted=actions[j//2]
                # The shared recorder reports a zero derivative at its first
                # snapshot because previous_base_xy/yaw have not been set yet.
                if step+j:
                    np.testing.assert_allclose(applied[:3],predicted[:3],atol=3e-5,rtol=0,
                                               err_msg=str((trial,step+j)))
                else:
                    np.testing.assert_array_equal(applied[:3],[0.,0.,0.])
                np.testing.assert_allclose(applied[3:],np.clip(predicted[3:],lo,hi),atol=1e-6,rtol=0)
                actions_checked+=1
            requests_checked+=1
        rows.append(metrics)
    n=len(rows);wins=sum(r['success'] for r in rows);safe=sum(r['contact_free_success'] for r in rows)
    assert wins==summary['successes']
    result=dict(passed=True,partial=a.partial,trials=n,successes=wins,success_rate=wins/n,wilson_95=wilson(wins,n),
        contact_free_successes=safe,contact_free_success_rate=safe/n,contact_free_wilson_95=wilson(safe,n),
        base_door_contact_trials=sum(r['base_door_contact'] for r in rows),
        requests_checked=requests_checked,applied_physics_actions_checked=actions_checked,
        current_body_poses_checked=body_poses_checked,max_render_position_error_m=max_pose_error,
        protocol_sha256=sha(root/'protocol.json'),model_sha256=protocol['model_sha256'],
        limitations=protocol['limitations'])
    save(root/('partial_audit.json' if a.partial else 'final_audit.json'),result);print(result,flush=True)


if __name__=='__main__':main()
