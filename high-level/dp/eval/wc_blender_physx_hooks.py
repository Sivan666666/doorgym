"""Inject learned Joint10 commands into the unchanged contact-physics loop.

Never call the expert trajectory controller. The source recorder still records
actual PhysX states/contacts; render/inference pauses wall time, not sim time.
"""
import math
from pathlib import Path
import types
import numpy as np
from wc_blender_physx_common import read, save, wait_file, score


def integrate(action, xy, yaw, dt):
    a = np.asarray(action)
    assert a.shape == (10,) and np.isfinite(a).all()
    c,s = math.cos(yaw),math.sin(yaw)
    return (np.asarray(xy)+dt*np.array([c*a[0]-s*a[1],s*a[0]+c*a[1]]),
            float(yaw+dt*a[2]),dict(zip(['joint'+str(i) for i in range(1,7)]+['jointGripper'],a[3:])))


def install(mod, gym, config_path, out):
    cfg = read(config_path)
    root = Path(cfg['root'])
    trial = cfg['trial']
    ipc = root/'ipc'/('%03d'%trial)
    ipc.mkdir(parents=True, exist_ok=True)
    logs = []
    state = dict(tick=0, actions=None, prev=np.zeros(3), requests=0)
    controller = types.SimpleNamespace(action_names=list(mod.dc.A2W_JOINT_ACTION10_NAMES),action_frame='base')

    def setup(args, states, *unused):
        assert len(states)==1
        args.dp_fps=25
        args.dp_warmstart=False
        for st in states:
            st.args.dp_fps=25
            st.args.dp_gripper_latch=False
            st.dp_action_frame='base'
            assert abs(math.degrees(st.door.handle_unlock_threshold)-25)<1e-4
        return controller,None,states[0],[0],{0}

    def forbid(*args,**kw):
        raise RuntimeError('Expert trajectory fallback is forbidden during policy evaluation')

    def collect(gym,sim,states,names,gripper_idx,dt,*unused):
        st=states[0]
        tick=state['tick']
        assert abs(dt-.02)<1e-9
        q,_=mod.get_actor_dof_state(gym,st.env,st.arm_actor)
        joint_idx=[names.index(n) for n in controller.action_names[3:]]
        obs=np.concatenate([state['prev'],np.asarray(q)[joint_idx]]).astype(np.float32)
        ee,eq=mod.current_ee_pose_from_refreshed_tensors(st.ik_state)
        xy=np.asarray(st.traj.get('base_xy',st.base_start)).copy()
        yaw=float(st.traj.get('yaw',st.yaw_start))
        if tick%50==0:
            base_actor=st.actor_handles[0] if len(st.actor_handles)>1 else st.arm_actor
            bp,bq=mod.get_body_pose(gym,st.env,base_actor,0)
            front,wrist=mod.dc.float_camera_pose_base(gym,st)
            bodies={}
            for role,actor in [('base',base_actor),('arm',st.arm_actor),('door',st.door_actor)]:
                bodies[role]={n:np.concatenate(mod.get_body_pose(gym,st.env,actor,i)).tolist()
                              for i,n in enumerate(gym.get_actor_rigid_body_names(st.env,actor))}
            # Door frame is the fixed actor root in this specific corrected asset.
            door_root=bodies['door']['frame']
            request=dict(trial=trial,request=tick//50,tick=tick,physics_step=tick*2,
                         simulation_time_s=tick/25,seed=cfg['seed'],state=obs.tolist(),
                         replay_root_state=np.concatenate([bp,bq]).tolist(),
                         replay_door_root_state=door_root,bodies=bodies,
                         front_camera_pose_base=np.asarray(front).tolist(),
                         wrist_camera_pose_base=np.asarray(wrist).tolist())
            p=ipc/('%03d'%state['requests'])
            p.mkdir(exist_ok=False)
            save(p/'request.json',request)
            wait_file(p/'actions.npy',root)
            actions=np.load(p/'actions.npy',allow_pickle=False)
            assert actions.shape==(100,10) and np.isfinite(actions).all()
            state['actions']=actions
            state['requests']+=1
        action=state['actions'][tick%50].copy()
        state['prev']=action[:3].copy()
        state['tick']+=1
        hp,hq=mod.get_body_pose(gym,st.env,st.door_actor,st.door.handle_body_index)
        return {0:dict(base_xy_current=xy,yaw_current=yaw,handle_goal=hp,
                       ee_pos=ee,ee_quat=eq,dp_state=obs)}, {0:action}

    original_monitor=mod.dc.monitor_base_door_collision
    def monitor(gym,step,st):
        result=original_monitor(gym,step,st)
        dq,_=mod.get_actor_dof_state(gym,st.env,st.door_actor)
        base_actor=st.actor_handles[0] if len(st.actor_handles)>1 else st.arm_actor
        bp,_=mod.get_body_pose(gym,st.env,base_actor,0)
        direction=np.asarray(st.base_traverse)-np.asarray(st.base_stop)
        direction=direction/np.linalg.norm(direction)
        offset=np.asarray(getattr(st.door,'actor_position_offset',(0,0,0)))
        plane=np.array([st.args.door_x,st.args.door_y])+offset[:2]
        opened=float(mod.dc.door_open_degrees(dq,st.args))
        # Retain source gripper force ramp, based only on physical latch/hinge.
        if st.door.open_stage and opened>=3 and not st.traj.get('_eval_grip_tightened'):
            props=gym.get_actor_dof_properties(st.env,st.arm_actor)
            idx=gym.get_actor_dof_names(st.env,st.arm_actor).index('jointGripper')
            props['effort'][idx]=24.
            gym.set_actor_dof_properties(st.env,st.arm_actor,props)
            st.traj['_eval_grip_tightened']=True
        labels={}
        for role,actor in [('base',base_actor),('arm',st.arm_actor),('door',st.door_actor)]:
            for n in gym.get_actor_rigid_body_names(st.env,actor):
                labels[gym.find_actor_rigid_body_handle(st.env,actor,n)]=role
        contacts=gym.get_env_rigid_contacts(st.env)
        collision=any({labels.get(int(c['body0'])),labels.get(int(c['body1']))}=={'base','door'}
                      and abs(float(c['lambda']))>1e-6 for c in contacts)
        logs.append(dict(step=step,open_deg=opened,handle_deg=math.degrees(float(dq[1])),
                         traversal_m=float(np.dot(bp[:2]-plane,direction[:2])),
                         base=bp.tolist(),base_door_contact=collision,unlocked=bool(st.door.open_stage)))
        if step%100==0:
            save(out/'live.json',dict(step=step,policy_ticks=state['tick'],requests=state['requests'],**score(logs)))
        if step==2599:
            assert state['tick']==1300 and state['requests']==26
            save(out/'metrics.json',dict(complete=True,trial=trial,seed=cfg['seed'],base_y=cfg['base_y'],
                steps=len(logs),policy_ticks=state['tick'],requests=state['requests'],
                physics_hz=50,policy_hz=25,predict_steps=100,execute_steps=50,**score(logs)))
            save(out/'metrics_trace.json',logs)
        return result

    mod.dc.setup_float_dp_policy_controller=setup
    mod.dc.collect_float_dp_policy_actions=collect
    mod.trajectory_targets=forbid
    # Do not snap small lateral velocities to zero: Joint10 labels contain vy.
    mod.apply_float_dp_joint_action9=lambda a,xy,yaw,dt,include_vy:integrate(a,xy,yaw,dt)
    mod.dc.monitor_base_door_collision=monitor
    mod.apply_dp_gripper_latch=lambda st,value,dq:value
