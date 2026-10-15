"""Isolated physical WC pull rollout with actual-state/contact capture.

Uses the existing controller, but never replays old simulator states. Door and
arm move through PhysX; the floating robot base remains prescribed motion.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import sys
import xml.etree.ElementTree as ET
import yaml

ROOT = Path('/home/ps/workspace/txc/doorgym')
JOB = Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_contact_physics_20260918')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--steps', type=int, default=2600)
    p.add_argument('--grasp-x', type=float, default=-.015)
    p.add_argument('--grasp-z', type=float, default=-.020)
    p.add_argument('--goal-x', type=float, default=-.065)
    p.add_argument('--gripper-ratio', type=float, default=1.)
    p.add_argument('--gripper-effort', type=float, default=24.)
    p.add_argument('--substeps', type=int, default=8)
    p.add_argument('--stop-distance',type=float,default=.24)
    p.add_argument('--base-y',type=float,default=.15)
    p.add_argument('--pull-lead',type=float,default=.4)
    p.add_argument('--rotate-steps',type=int,default=300)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--seed',type=int,default=615456575)
    p.add_argument('--record-actions',action='store_true')
    p.add_argument('--policy-eval-config',type=Path,
                   help='Optional isolated live Blender/PhysX evaluation hook; never used by recording.')
    args = p.parse_args()
    if not 0 < args.gripper_effort <= 30:
        p.error('Gripper effort must stay within the source URDF limit (30 Nm).')
    out = (args.output_root or JOB) / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out/'controller_source.py').write_text(Path(__file__).read_text())
    config=yaml.safe_load((JOB/'door_contact.yaml').read_text())
    for block in config['env']['asset']['trainAssets'].values():
        for spec in block.values():spec.pop('controller_overrides',None)
    (out/'door_contact.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    path = ROOT / 'high-level/float_ik/isaacgym_float_ik_a2w_basearn_pull_door_parallel.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('wc_contact_pull', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    import numpy as np
    api = mod.gymapi

    class GymProxy:
        def __init__(self, gym):
            self.gym = gym
        def __getattr__(self, name):
            return getattr(self.gym, name)
        def load_asset(self, sim, root, file, options):
            if 'z1_arm' in str(file):
                tree = ET.parse(Path(root)/file)
                for name in ('gripperStator','gripperMover'):
                    link=tree.getroot().find("link[@name='%s']" % name)
                    for c in link.findall('collision'):
                        link.remove(c)
                    c=ET.SubElement(link,'collision',name='visual_matched_convex_decomposition')
                    ET.SubElement(c,'origin',xyz='0 0 0',rpy='0 0 0')
                    ET.SubElement(ET.SubElement(c,'geometry'),'mesh',filename=str(JOB/'assets/robot_collision'/(name+'.stl')))
                # Never rewrite shared/generated controller assets in place.
                for mesh in tree.getroot().iter('mesh'):
                    filename=mesh.get('filename')
                    if filename and not Path(filename).is_absolute():
                        if '://' in filename:raise ValueError('Resolve mesh URI before isolated loading: '+filename)
                        mesh.set('filename',str((Path(root)/Path(file).parent/filename).resolve()))
                    if mesh.get('filename') and not Path(mesh.get('filename')).is_file():
                        raise FileNotFoundError('Missing required arm mesh: '+mesh.get('filename'))
                file='z1_arm_contact_runtime.urdf'
                tree.write(out/file,encoding='utf-8',xml_declaration=True)
                root=str(out)
                options.vhacd_enabled = True
                options.vhacd_params.resolution = 1000000
                options.vhacd_params.max_convex_hulls = 64
                options.vhacd_params.max_num_vertices_per_ch = 64
            if str(file).endswith('wc_contact/model.urdf'):
                options.override_com = False
                options.override_inertia = False
                options.disable_gravity = False
                options.vhacd_enabled = False  # explicit exact boxes, no convex-filled doorways
            asset = self.gym.load_asset(sim, root, file, options)
            if asset is not None:
                props = self.gym.get_asset_rigid_shape_properties(asset)
                for prop in props:
                    if hasattr(prop, 'contact_offset'):
                        prop.contact_offset = .002
                    if hasattr(prop, 'rest_offset'):
                        prop.rest_offset = .0002
                if 'z1_arm' in str(file):
                    names = self.gym.get_asset_rigid_body_names(asset)
                    ranges = self.gym.get_asset_rigid_body_shape_indices(asset)
                    # These links are welded into one rigid assembly. Internal
                    # collisions cannot be resolved and must not exert forces.
                    # The moving jaw is deliberately NOT in this filter group.
                    welded = {'link06','gripperStator','wrist_up_mount','wrist_down_mount','ee_gripper_link'}
                    for name, interval in zip(names,ranges):
                        if name in welded:
                            for k in range(interval.start, interval.start+interval.count):
                                props[k].filter |= 128
                self.gym.set_asset_rigid_shape_properties(asset, props)
            return asset

    real_gym = api.acquire_gym()
    gym = GymProxy(real_gym)
    api.acquire_gym = lambda: gym
    def create_sim(gym, cli):
        params = api.SimParams()
        params.dt = .02
        params.substeps = args.substeps
        params.up_axis = api.UP_AXIS_Z
        params.gravity = api.Vec3(0,0,-9.81)
        params.use_gpu_pipeline = False
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 16
        params.physx.num_velocity_iterations = 4
        params.physx.num_threads = 4
        params.physx.use_gpu = True
        params.physx.contact_offset = .002
        params.physx.rest_offset = .0002
        params.physx.contact_collection = api.ContactCollection.CC_ALL_SUBSTEPS
        cli.use_gpu_pipeline = False
        cli.pipeline = 'CPU'
        sim = gym.create_sim(cli.compute_device_id, cli.graphics_device_id, cli.physics_engine, params)
        assert sim is not None
        return sim, params.dt
    mod.base_ik.create_sim = create_sim

    # A force-limited hinge servo models the closed latch. Remove the legacy
    # per-step hinge state teleport. Once the physical handle passes 25deg,
    # switch to passive effort mode; no opening torque is injected.
    latch_modes = {}
    def latch(gym, env, actor, door, cli):
        locked = not door.open_stage
        key = (id(env), actor)
        if latch_modes.get(key) == locked:
            return
        props = gym.get_actor_dof_properties(env, actor)
        i = door.door_dof_index
        props['driveMode'][i] = api.DOF_MODE_POS if locked else api.DOF_MODE_EFFORT
        props['stiffness'][i] = 5000 if locked else 0
        props['damping'][i] = 100 if locked else cli.door_joint_damping
        gym.set_actor_dof_properties(env, actor, props)
        if locked:
            targets = np.zeros(len(props), dtype=np.float32)
            gym.set_actor_dof_position_targets(env, actor, targets)
        latch_modes[key] = locked
    mod.dc.enforce_locked_door_hinge = latch
    original_efforts=mod.compute_door_efforts
    def passive_hinge_efforts(door,q,qv,cli):
        efforts=original_efforts(door,q,qv,cli)
        assert abs(float(efforts[door.door_dof_index]))<1e-8, 'Unexpected external door-opening torque'
        return efforts
    mod.compute_door_efforts=passive_hinge_efforts

    original_targets=mod.trajectory_targets
    def targets(step,cli,door,gym,env,door_actor,ik_state,*remaining):
        traj=remaining[-1]
        previous_base=np.asarray(traj.get('base_xy',remaining[0])).copy()
        previous_yaw=float(traj.get('yaw',remaining[4]))
        result=original_targets(step,cli,door,gym,env,door_actor,ik_state,*remaining)
        if result[0]=='pass_through':
            # Steer toward the doorway centre, rather than dragging the outer
            # wheels along the jamb at the grasp's laterally offset base pose.
            goal=np.asarray(remaining[3]).copy();goal[1]=0.
            delta=goal-previous_base;distance=float(np.linalg.norm(delta))
            error=mod.wrap_to_pi(math.atan2(delta[1],delta[0])-previous_yaw)
            yaw=previous_yaw+float(np.clip(error,-.6*.02,.6*.02))
            speed=min(.5,2*distance)*max(0.,math.cos(error))
            base=previous_base+.02*speed*np.array([math.cos(yaw),math.sin(yaw)])
            old_base=np.array([*result[1],cli.robot_z]);new_base=np.array([*base,cli.robot_z])
            rotation=mod.quat_from_angle_axis(yaw-float(result[2]),np.array([0.,0.,1.]))
            result=list(result)
            if result[3] is not None:result[3]=new_base+mod.quat_apply(rotation,np.asarray(result[3])-old_base)
            if result[4] is not None:result[4]=mod.base_ik.quat_multiply(rotation,result[4])
            result[1]=base;result[2]=yaw
            traj['base_xy']=base.copy();traj['yaw']=yaw
            return tuple(result)
        if result[0] in ('panel_reposition','panel_contact','panel_push'):
            hp,_=mod.get_body_pose(gym,env,door_actor,door.handle_body_index)
            if '_physical_panel_anchor' not in traj:
                traj['_physical_panel_anchor']=np.asarray(result[1]).copy()
                traj['_physical_panel_handle_x']=float(hp[0])
                traj['_physical_panel_base']=np.asarray(result[1]).copy()
            base=traj['_physical_panel_base'].copy()
            desired=traj['_physical_panel_anchor'][0]+.85*max(0.,float(hp[0])-traj['_physical_panel_handle_x'])
            base[0]+=np.clip(desired-base[0],-.15*.02,.15*.02)
            traj['_physical_panel_base']=base.copy();traj['base_xy']=base.copy()
            result=list(result);result[1]=base;return tuple(result)
        if result[0]=='pull_door' and door.open_stage:
            dq,_=mod.get_actor_dof_state(gym,env,door_actor)
            hp,hq=mod.get_body_pose(gym,env,door_actor,door.handle_body_index)
            if '_physical_pull_start' not in traj:
                hinge,_=mod.get_body_pose(gym,env,door_actor,door.door_body_index)
                traj['_physical_pull_start']=step
                traj['_physical_pull_hinge']=hinge.copy()
                inverse=mod.base_ik.quat_conjugate(hq)
                traj['_physical_grip_quat']=mod.base_ik.quat_multiply(inverse,np.asarray(ik_state.current_quat_np))
            ramp=mod.smoothstep((step-traj['_physical_pull_start']+1)/100.)
            angle=math.radians(args.pull_lead*ramp)
            rotation=mod.quat_from_angle_axis(float(cli.door_motion_sign)*angle,np.array([0.,0.,1.]))
            depression=np.clip(math.radians(35.)-float(dq[1]),-.1,.1)*ramp
            depressed=mod.base_ik.quat_multiply(mod.quat_from_angle_axis(depression,mod.quat_axis(hq,axis=2)),hq)
            desired_quat=mod.base_ik.quat_multiply(rotation,depressed)
            # A bounded incremental Cartesian pull generates tensile contact
            # force even after the jaws settle under load. A fixed pre-load
            # grasp offset otherwise commands the hand back toward the door.
            ee=np.asarray(ik_state.current_pos_np)
            depression_rotation=mod.quat_from_angle_axis(depression,mod.quat_axis(hq,axis=2))
            depressed_ee=hp+mod.quat_apply(depression_rotation,ee-hp)
            result=list(result)
            result[3]=traj['_physical_pull_hinge']+mod.quat_apply(rotation,depressed_ee-traj['_physical_pull_hinge'])
            result[4]=mod.base_ik.quat_multiply(desired_quat,traj['_physical_grip_quat'])
            return tuple(result)
        if result[0]=='pull_door' and not door.open_stage:
            result=list(result);result[0]='rotate_handle';result=tuple(result)
        if result[0]!='rotate_handle':
            return result
        hp,hq=mod.get_body_pose(gym,env,door_actor,door.handle_body_index)
        dq,_=mod.get_actor_dof_state(gym,env,door_actor)
        if '_contact_rotate_start' not in traj:
            traj['_contact_rotate_start']=step
            traj['_contact_rotate_q0']=float(dq[1])
            traj['_contact_rotate_offset']=np.asarray(ik_state.current_pos_np)-hp
            traj['_contact_rotate_quat']=np.asarray(ik_state.current_quat_np).copy()
            traj['goal_quat']=traj['_contact_rotate_quat'].copy()
        t=mod.smoothstep((step-traj['_contact_rotate_start']+1)/max(1,cli.handle_rotate_steps))
        delta=min(.65*t, max(0.,float(dq[1])-traj['_contact_rotate_q0'])+.25)
        axis=mod.quat_axis(hq,axis=2)
        rotation=mod.quat_from_angle_axis(delta,axis)
        position=hp+mod.quat_apply(rotation,traj['_contact_rotate_offset'])+.010*t*axis
        orientation=mod.base_ik.quat_multiply(rotation,traj['_contact_rotate_quat'])
        traj['rotate']=position.copy()
        if step-traj['_contact_rotate_start']>cli.handle_rotate_steps+200 and not door.open_stage:
            raise RuntimeError('Handle did not physically unlock; aborting failed grasp.')
        result=list(result);result[3]=position;result[4]=orientation
        return tuple(result)
    mod.trajectory_targets=targets

    # Enforce the requested gripper torque ceiling; legacy code uses max().
    original_create = mod.create_parallel_env_states
    def create_envs(*v, **kw):
        states, vision = original_create(*v, **kw)
        for st in states:
            st.door.handle_goal_offset[0] = args.goal_x
            shape_props = gym.get_actor_rigid_shape_properties(st.env,st.arm_actor)
            body_names = gym.get_actor_rigid_body_names(st.env,st.arm_actor)
            ranges = gym.get_actor_rigid_body_shape_indices(st.env,st.arm_actor)
            for name, interval in zip(body_names,ranges):
                if name in {'link06','gripperStator','wrist_up_mount','wrist_down_mount','ee_gripper_link'}:
                    for k in range(interval.start,interval.start+interval.count):
                        shape_props[k].filter |= 128
            gym.set_actor_rigid_shape_properties(st.env,st.arm_actor,shape_props)
            (out/'collision_filters.json').write_text(json.dumps(dict(
                body_names=body_names, filters=[int(v.filter) for v in gym.get_actor_rigid_shape_properties(st.env,st.arm_actor)]),indent=2))
            props = gym.get_actor_dof_properties(st.env, st.arm_actor)
            names = gym.get_actor_dof_names(st.env, st.arm_actor)
            props['effort'][names.index('jointGripper')] = min(12.,args.gripper_effort)
            gym.set_actor_dof_properties(st.env, st.arm_actor, props)
            physics={}
            for role,actor in [('arm',st.arm_actor),('door',st.door_actor)]:
                properties=gym.get_actor_dof_properties(st.env,actor)
                physics[role]=dict(dof_names=gym.get_actor_dof_names(st.env,actor),
                    dof_properties={k:properties[k].tolist() for k in properties.dtype.names},
                    shape_friction=[float(v.friction) for v in gym.get_actor_rigid_shape_properties(st.env,actor)],
                    body_masses=[float(v.mass) for v in gym.get_actor_rigid_body_properties(st.env,actor)])
            (out/'physics_properties.json').write_text(json.dumps(physics,indent=2))
            (out / 'effective_args.json').write_text(json.dumps(vars(st.args), default=str, indent=2))
            (out / 'body_names.json').write_text(json.dumps(dict(
                arm=gym.get_actor_rigid_body_names(st.env, st.arm_actor),
                door=gym.get_actor_rigid_body_names(st.env, st.door_actor)), indent=2))
        return states, vision
    mod.create_parallel_env_states = create_envs
    snapshots, audit = [], []
    original_monitor = mod.dc.monitor_base_door_collision
    def capture(gym, step, st):
        result = original_monitor(gym, step, st)
        names = gym.get_actor_dof_names(st.env, st.arm_actor)
        q, qv = mod.get_actor_dof_state(gym, st.env, st.arm_actor)
        dq, dqv = mod.get_actor_dof_state(gym, st.env, st.door_actor)
        base_actor = st.actor_handles[0] if len(st.actor_handles)>1 else st.arm_actor
        bp, bq = mod.get_body_pose(gym, st.env, base_actor, 0)
        ee_index = mod.dc.get_actor_body_index(gym, st.env, st.arm_actor, 'ee_gripper_link')
        ep, eq = mod.get_body_pose(gym, st.env, st.arm_actor, ee_index)
        snap = mod.dc.make_float_replay_snapshot(st.args, st.door, names, q, qv, dq, dqv, ep, eq,
                                                bp[:2], 0, 0, 0)
        snap['replay_root_state'][:3], snap['replay_root_state'][3:7] = bp, bq
        front, wrist = mod.dc.float_camera_pose_base(gym, st)
        snap.update(front_camera_pose_base=front, wrist_camera_pose_base=wrist)
        snap = {k:np.asarray(v).tolist() for k,v in snap.items()}
        snap['phase'] = st.last_phase
        snap['step'] = step
        if args.record_actions:
            joint_names=['joint'+str(i) for i in range(1,7)]+['jointGripper']
            joint_indices=[names.index(n) for n in joint_names]
            base_xy=np.asarray(st.traj.get('base_xy',st.base_start))
            yaw=float(st.traj.get('yaw',st.yaw_start))
            velocity=mod.dc.base_command_from_targets_with_lateral(base_xy,yaw,st.prev_base_xy,st.prev_yaw,.02)
            applied=np.concatenate([velocity,np.asarray(st.dof_positions)[joint_indices]])
            assert np.isfinite(applied).all()
            snap['control_record']=dict(
                applied_action10=applied.tolist(),measured_joint7=np.asarray(q)[joint_indices].tolist(),
                commanded_base_xy=base_xy.tolist(),commanded_base_yaw=yaw,
                commanded_dof_positions=np.asarray(st.dof_positions).tolist(),dof_names=names,
                semantics='Command applied during the step that produced this POST-step snapshot. Pair snapshot i with control_record from i+1 for supervised s_t,a_t.',
                base_command_type='Prescribed pose increment / dt in previous commanded base frame; not wheel actuator commands.')
        snap['bodies'] = {}
        for role, actor in (('base',base_actor),('arm', st.arm_actor), ('door', st.door_actor)):
            snap['bodies'][role] = {name:np.concatenate(mod.get_body_pose(gym, st.env, actor, idx)).tolist()
                for idx,name in enumerate(gym.get_actor_rigid_body_names(st.env, actor))}
        contacts = gym.get_env_rigid_contacts(st.env)
        if step == 0:
            print('CONTACT_DTYPE', contacts.dtype, flush=True)
        handle = gym.find_actor_rigid_body_handle(st.env, st.door_actor, st.door.body_names[st.door.handle_body_index])
        fingers = {gym.find_actor_rigid_body_handle(st.env, st.arm_actor,n):n for n in ('gripperStator','gripperMover')}
        body_labels = {}
        for role, actor in [('arm',st.arm_actor),('door',st.door_actor),('base',base_actor)]:
            for n in gym.get_actor_rigid_body_names(st.env,actor):
                body_labels[gym.find_actor_rigid_body_handle(st.env,actor,n)] = role+'/'+n
        finger_other = []
        handle_contacts = []
        pair_forces={}
        for c in contacts:
            b0, b1 = int(c['body0']), int(c['body1'])
            pair='|'.join(sorted([body_labels.get(b0,str(b0)),body_labels.get(b1,str(b1))]))
            pair_forces[pair]=pair_forces.get(pair,0.)+float(c['lambda'])
            if b0 in fingers or b1 in fingers:
                finger_other.append(dict(bodies=[body_labels.get(b0,str(b0)),body_labels.get(b1,str(b1))],
                    **{k:float(c[k]) for k in ('lambda','initialOverlap','minDist') if k in contacts.dtype.names}))
            if handle not in (b0,b1):
                continue
            other = b1 if b0==handle else b0
            if other not in fingers:
                continue
            handle_contacts.append(dict(finger=fingers[other],
                **{k:float(c[k]) for k in ('lambda','initialOverlap','minDist') if k in contacts.dtype.names}))
        both = len({c['finger'] for c in handle_contacts if abs(c.get('lambda',0)) > 1e-6}) == 2
        record = dict(step=step, phase=st.last_phase, door_deg=math.degrees(float(dq[0])),
                      handle_deg=math.degrees(float(dq[1])), gripper_q=float(q[-1]),
                      base=bp.tolist(), ee=ep.tolist(), contacts=handle_contacts, both=both,
                      unlocked=bool(st.door.open_stage), finger_all_contacts=finger_other,
                      all_contact_pair_lambda=pair_forces,
                      gripper_target=float(st.last_gripper))
        snapshots.append(snap)
        audit.append(record)
        if st.last_phase=='pull_door' and abs(record['door_deg'])>=3 and not st.traj.get('_physical_grip_tightened'):
            props=gym.get_actor_dof_properties(st.env,st.arm_actor)
            props['effort'][names.index('jointGripper')]=args.gripper_effort
            gym.set_actor_dof_properties(st.env,st.arm_actor,props)
            st.traj['_physical_grip_tightened']=True
            print('PHYSICAL_GRIP_TIGHTEN',step,args.gripper_effort,flush=True)
        if st.last_phase=='pull_door':
            active=any(c.get('lambda',0)>1e-6 for c in handle_contacts)
            st.traj['_physical_contact_lost_steps']=0 if active else st.traj.get('_physical_contact_lost_steps',0)+1
            if st.traj['_physical_contact_lost_steps']>25:
                raise RuntimeError('Physical grasp lost for >0.5s; aborting failed rollout, not rendering success.')
        if step % 100 == 0:
            brief = {k:v for k,v in record.items() if k not in ('finger_all_contacts','contacts')}
            brief['other_finger_pairs'] = sorted(set('/'.join(c['bodies']) for c in finger_other))
            print('CONTACT_AUDIT', json.dumps(brief), flush=True)
        return result
    mod.dc.monitor_base_door_collision = capture
    source = ROOT/'high-level/dp/record/run_wc_nomenkuang_fixed15_gripper080_rgb_pi05_4way_ps1.sh'
    argv = shlex.split(source.read_text().split('TRAJECTORY_ARGS=(',1)[1].split('\n)',1)[0])
    argv.remove('--pull_vx_monotonic_retreat')
    argv.remove('--pull_panel_adaptive_release_retreat')
    def setting(key,value):
        if key in argv:
            argv[argv.index(key)+1] = str(value)
        else:
            argv.extend([key,str(value)])
    for key,value in {'--door_cfg':out/'door_contact.yaml','--door_name':'wc_contact',
                      '--steps':args.steps,'--num_envs':1,'--seed':args.seed,
                      '--grasp_x_offset':args.grasp_x,'--grasp_z_offset':args.grasp_z,
                      '--gripper_close_ratio':args.gripper_ratio,'--pull_gripper_stiffness':100,
                      '--gripper_open':-.9,'--stiffness':200,'--damping':20,
                      '--initial_hold_steps':350,'--initial_hold_move_steps':250,
                      '--pregrasp_offset':.25,'--grasp_steps':150,
                      '--pull_gripper_damping':10,'--pull_gripper_effort_limit':args.gripper_effort,
                      '--door_lock_force':0,'--door_joint_friction':.04,'--door_joint_damping':.1,
                      '--handle_joint_friction':.02,'--handle_joint_damping':.05,
                      '--handle_spring_stiffness':1.,'--handle_spring_damping':.1,
                      '--handle_rotate_angle':.65,'--handle_rotate_steps':args.rotate_steps,
                      '--rotate_pull_overlap_steps':0,
                      '--release_clearance_base_retreat_extra':0,
                      '--release_clearance_low_angle_base_retreat_extra':0,
                      '--graphics_device_id':-1,'--sim_device':'cuda:0',
                      '--pull_gripper_contact_friction':1.5,'--pull_handle_contact_friction':1.5,
                      '--stop_distance':args.stop_distance,
                      '--robot_y_alignment':'door_y','--robot_y':args.base_y}.items():
        setting(key,value)
    argv += ['--headless','--no_ikpush_env_randomization','--no_enable_front_camera',
             '--no_enable_wrist_camera','--no_show_camera_images','--no_draw_scripted_trajectory',
             '--no_preview_trajectory_at_spawn','--no_draw_ik_target','--no_draw_camera_axes']
    (out / 'command.json').write_text(json.dumps(dict(argv=argv,options=vars(args)),indent=2,default=str))
    sys.argv = [str(path)] + argv
    if args.policy_eval_config is not None:
        from wc_blender_physx_hooks import install
        install(mod, gym, args.policy_eval_config, out)
    try:
        mod.main()
    finally:
        (out / 'trajectory.json').write_text(json.dumps(dict(fps=50, frames=snapshots)))
        (out / 'contact_audit.json').write_text(json.dumps(audit))
        if audit:
            summary = dict(frames=len(audit), both_contact_frames=sum(r['both'] for r in audit),
                           max_open_deg=max(abs(r['door_deg']) for r in audit),
                           max_handle_deg=max(r['handle_deg'] for r in audit),
                           final_base=audit[-1]['base'], phases=list(dict.fromkeys(r['phase'] for r in audit)),
                           base_kinematic=True, door_state_replay=False, latch='servo until 25deg then passive',
                           substeps=args.substeps, physical_step_s=.02/args.substeps)
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print('ROLLOUT_SUMMARY',json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
