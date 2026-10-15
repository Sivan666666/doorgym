"""PS1 closed-loop evaluation: live PhysX -> Blender RGB -> native pi05 Evo.

Three isolated runtimes communicate via atomic, numbered files. No Gym RGB,
teacher actions, replayed physical poses or training-scene seed selection.
"""
import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from wc_blender_physx_common import read,save,sha,wilson

ARCH=Path('/mnt/sda/ps_archive/doorgym/high-level')
ROOT=ARCH/'dp/result/wc_dim60_blender_physx_eval64_20261010'
TRAINJOB=ARCH/'dp/result/wc_dim60_aug100_pi05_20261008'
MODEL=ARCH/'dp/logs/lerobot-train-pi05/pi05evo_wc_dim60_aug100_joint10_seed1000_30k/checkpoints/030000/pretrained_model'
REPO=Path('/home/ps/workspace/txc/doorgym')
PY='/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'
PHYS='/home/ps/miniconda3/envs/txc_vbc/bin/python'
BLENDER=str(ARCH/'dp/result/wc_meister_pipeline_20260920/runtime/bin/python')
TEMPLATE=ARCH/'dp/result/wc_photo_office_dim60_20261008/run/episodes/000000/raw_001/scene_001/scene.blend'
MEISTER=ARCH/'dp/result/wc_meister_pipeline_20260920/SIM1MeisterRender'
TASK='Grasp the handle, pull the door open and pass through the doorway.'


def requests(root,done):
    return [p for p in sorted((root/'ipc').glob('*/*/request.json')) if not (p.parent/done).exists()]


def policy(root):
    import numpy as np
    import torch
    from lerobot.policies.pi05_evo.modeling_pi05_evo import PI05EvoPolicy
    from lerobot.policies.pi05_evo import processor_pi05_evo
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    torch.set_num_threads(4)
    cfg=PreTrainedConfig.from_pretrained(MODEL)
    assert cfg.type=='pi05_evo'
    assert cfg.input_features['observation.state'].shape==(10,) or list(cfg.input_features['observation.state'].shape)==[10]
    cfg.device='cuda';cfg.gradient_checkpointing=False
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(MODEL),
        preprocessor_overrides={'device_processor':{'device':'cuda'}},
        postprocessor_overrides={'device_processor':{'device':'cuda'}})
    model=PI05EvoPolicy.from_pretrained(MODEL,config=cfg,strict=True,local_files_only=True).to('cuda').eval()
    assert cfg.chunk_size==100 and cfg.n_action_steps==50 and cfg.num_inference_steps==10
    save(root/'policy_ready.json',dict(checkpoint=str(MODEL),device='cuda',pid=os.getpid()))
    trial=None
    while not (root/'stop').exists():
        for req in requests(root,'actions.npy'):
            folder=req.parent
            if not (folder/'observations.npz').exists():continue
            r=read(req)
            if r['trial']!=trial:
                trial=r['trial'];torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed']);model.reset()
            with np.load(folder/'observations.npz',allow_pickle=False) as d:
                batch={'observation.state':torch.from_numpy(d['state'].copy()).unsqueeze(0),'task':[TASK]}
                for v in ('front','wrist'):
                    image=d[v].copy()
                    assert image.shape==(480,640,3) and image.dtype==np.uint8
                    batch['observation.images.'+v]=torch.from_numpy(image).permute(2,0,1).float().div(255).unsqueeze(0)
            start=time.monotonic()
            with torch.inference_mode():
                action=post(model.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
            assert action.shape==(100,10) and np.isfinite(action).all()
            save(folder/'inference.json',dict(seconds=time.monotonic()-start,trial=trial,request=r['request'],
                observation_sha256=sha(folder/'observations.npz'),model=str(MODEL),
                minimum=action.min(axis=0).tolist(),maximum=action.max(axis=0).tolist()))
            with (folder/'actions.tmp').open('wb') as f:np.save(f,action)
            (folder/'actions.tmp').replace(folder/'actions.npy')
        time.sleep(.03)


def render(root):
    import types
    import math
    import numpy as np
    import cv2
    import bpy
    from mathutils import Matrix,Quaternion,Vector
    # Avoid collision with our pipeline adapter module name.
    sys.path.insert(0,str(MEISTER))
    from omegaconf import OmegaConf
    from pipeline import Pipeline
    from data import InputPack
    from utils.blender_utils.renderer_utils import init_renderer_output
    assert subprocess.check_output(['git','-C',str(MEISTER),'rev-parse','HEAD'],text=True).strip()=='c031e1f8594dab3a91832e817d884f55189a4e8a'
    cfg=OmegaConf.load(str(MEISTER/'configs/blender_workbench.yml')).engine.pipeline
    cfg.renderer.type='BlenderPathtracingRenderer'
    pipe=Pipeline(cfg)
    def init(renderer):
        s=bpy.context.scene;s.render.engine='CYCLES';s.render.threads_mode='FIXED';s.render.threads=4
        s.render.use_persistent_data=True;s.cycles.samples=24;s.cycles.use_denoising=True
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type='OPTIX';prefs.get_devices()
        devices=[d for d in prefs.devices if d.type=='OPTIX'];assert devices
        for d in prefs.devices:d.use=d==devices[0]
        s.cycles.device='GPU';init_renderer_output(renderer.shared_memory_path)
        s.view_settings.view_transform='AgX';renderer.first_frame_init=True
    pipe.renderer._lazy_render_init=types.MethodType(init,pipe.renderer)
    pack=InputPack();pack.scene_only=True;pack.scene_file_path['blend']=str(TEMPLATE)
    pipe.scene=pipe.loader.scene_load(None,pack)
    scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update()
    for obj in bpy.data.objects:
        obj.animation_data_clear()
        if obj.data and hasattr(obj.data,'animation_data_clear'):obj.data.animation_data_clear()
    scene.animation_data_clear()
    for name in list(pipe.scene.cameras_dict):
        if name not in ('Replay front','Replay wrist'):
            obj=pipe.scene.cameras_dict[name].instance
            del pipe.scene.cameras_dict[name]
            bpy.data.objects.remove(obj,do_unlink=True)
    cameras={name.removeprefix('Replay '):c for name,c in pipe.scene.all_cameras.items()}
    assert set(cameras)=={'front','wrist'}
    for i,c in enumerate(cameras.values()):c.instance.name='live_camera_tmp_'+str(i)
    for view in scene.render.views:view.use=False
    for i,c in enumerate(cameras.values()):
        c.instance.name='camera_'+str(i)
        view=scene.render.views.get('view_'+str(i)) or scene.render.views.new('view_'+str(i))
        view.camera_suffix='_'+str(i);view.file_suffix='_'+str(i);view.use=True
    scene.camera=cameras['front'].instance
    scene.render.resolution_x=640;scene.render.resolution_y=480;scene.render.resolution_percentage=100
    for c in cameras.values():c.image_width=640;c.image_height=480
    for v,c in cameras.items():
        expected=read(TEMPLATE.parent.parent.parent/'render/cameras.json')[v]['K']
        assert np.allclose(c.get_intrinsics_as_K_matrix(),expected,atol=1e-6)
    def pose(value):
        m=Quaternion((value[6],*value[3:6])).to_matrix().to_4x4();m.translation=Vector(value[:3]);return m
    axes=Matrix(((0,0,-1,0),(-1,0,0,0),(0,1,0,0),(0,0,0,1)))
    def depth(o):
        n=0
        while o.parent:n+=1;o=o.parent
        return n
    lights={o.name:float(o.data.energy) for o in bpy.data.objects if o.type=='LIGHT'}
    save(root/'render_ready.json',dict(scene=str(TEMPLATE),scene_sha256=sha(TEMPLATE),lights=lights,
        cameras={v:c.get_intrinsics_as_K_matrix().tolist() for v,c in cameras.items()},
        samples=24,photometric_augmentation=False,pid=os.getpid()))
    while not (root/'stop').exists():
        for req in requests(root,'observations.npz'):
            r=read(req);start=time.monotonic()
            convert=pose(r['replay_door_root_state']).inverted()
            base=convert@pose(r['replay_root_state'])
            expected={'robot/world':base}
            for role,records in r['bodies'].items():
                for name,value in records.items():
                    key=name if role=='door' else 'robot/'+name
                    if key in bpy.data.objects:expected[key]=convert@pose(value)
            for key in sorted(expected,key=lambda k:depth(bpy.data.objects[k])):
                bpy.data.objects[key].matrix_world=expected[key];bpy.context.view_layer.update()
            for v in cameras:
                name=cameras[v].instance.name
                expected[name]=base@pose(r[v+'_camera_pose_base'])@axes
                cameras[v].set_matrix_world(expected[name])
            bpy.context.view_layer.update()
            errors=[]
            for key,m in expected.items():
                actual=bpy.data.objects[key].matrix_world
                error=(actual.translation-m.translation).length
                angle=2*math.acos(min(1.,abs(actual.to_quaternion().normalized().dot(m.to_quaternion().normalized()))))
                assert error<1e-4 and angle<.002,(key,error,angle)
                errors.append(error)
            scene.frame_start=1;scene.frame_end=2
            pack=InputPack();pack.action_only=True
            result=pipe.run(pack)
            images={v:result.rgb_image['Replay '+v][0] for v in cameras}
            for v,rgb in images.items():
                # A policy may enter an occluded/black view. This is a valid
                # observation, not a reason to reject/reseed an evaluation.
                assert rgb.shape==(480,640,3) and rgb.dtype==np.uint8
                assert cv2.imwrite(str(req.parent/(v+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            save(req.parent/'render.json',dict(seconds=time.monotonic()-start,request_sha256=sha(req),
                max_position_error_m=max(errors),camera_c2w={v:c.get_matrix_world().tolist() for v,c in cameras.items()},
                means={v:float(im.mean()) for v,im in images.items()},
                stds={v:float(im.std()) for v,im in images.items()},
                rgb_sha256={v:sha(req.parent/(v+'.png')) for v in cameras},source='live_physx',renderer='MeisterRender/Cycles'))
            with (req.parent/'observations.tmp').open('wb') as f:
                np.savez(f,state=np.asarray(r['state'],dtype=np.float32),**images)
            (req.parent/'observations.tmp').replace(req.parent/'observations.npz')
        time.sleep(.03)


def env(gpu):
    e=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HOME='/home/ps/.cache/huggingface',PYTHONPATH=str(REPO/'high-level/hwk/lerobot/src')+':'+str(Path(__file__).parent),
        TMPDIR='/mnt/sda/ps/tmp/wc100')
    return e


def report(root,planned):
    rows=[read(p) for p in sorted((root/'trials').glob('*/metrics.json'))]
    total=len(rows);wins=sum(r['success'] for r in rows)
    result=dict(complete=total==planned,planned_trials=planned,trials=total,successes=wins,
        success_rate=wins/total if total else None,wilson_95=wilson(wins,total),
        open_successes=sum(r['door_open_success'] for r in rows),
        traversal_successes=sum(r['traversal_success'] for r in rows),
        contact_free_successes=sum(r['contact_free_success'] for r in rows),trials_detail=rows)
    save(root/'summary.json',result)
    return result


def stop(p):
    if p.poll() is None:
        with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
            p.wait()


def run(root,trials,resume=False):
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert resume == (root/'protocol.json').exists(),'Use --resume only for this existing protocol'
    finished=read(TRAINJOB/'training_complete.json');assert finished['passed']
    assert sha(MODEL/'model.safetensors')==finished['weights_sha256']['model.safetensors']
    seeds=[2026101000+i for i in range(trials)]
    training=read(TRAINJOB/'run/manifest.json')['episodes']
    used={read(Path(e['raw'])/'command.json')['options']['seed'] for e in training}
    assert not set(seeds)&used
    protocol=dict(trials=trials,seeds=seeds,base_y_range=[.15,.175],
        model=str(MODEL),model_sha256=finished['weights_sha256']['model.safetensors'],
        template=str(TEMPLATE),lighting='fixed approved 60 percent baseline',rgb_augmentation=False,
        physical_handle_unlock_deg=25,score_door_open_deg=60,score_traversal_m=.5,
        sim_hz=50,policy_action_hz=25,predict_steps=100,execute_steps=50,rollout_steps=2600,
        render_observation_every_sim_steps=100,closed_loop=True,rejected_failures=False,
        limitations=['Floating prescribed base (not wheel dynamics), matching training.',
                    'Room walls/furniture are render-only, matching training; door/arm contacts use PhysX.',
                    'Primary success requires open60 and signed traversal0.5 simultaneously at a step, matching the historical scorer; base-door-contact-free count also reported.',
                    'Held 25Hz action executes for two PhysX steps; no scripted expert assistance.'],
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    first=0
    if resume:
        old=read(root/'protocol.json')
        for k,v in protocol.items():
            if k!='code_sha256':assert old[k]==v,('Protocol changed',k)
        complete=sorted((root/'trials').glob('*/metrics.json'))
        first=len(complete)
        assert [int(p.parent.name) for p in complete]==list(range(first))
        assert first<trials
        archive=root/'interrupted_attempts'/time.strftime('%Y%m%d_%H%M%S')
        archive.mkdir(parents=True,exist_ok=False)
        for p in [root/'trials'/('%03d'%first),root/'ipc'/('%03d'%first)]:
            if p.exists():p.rename(archive/(p.parent.name+'_'+p.name))
        for name in ['render.log','policy.log','render_ready.json','policy_ready.json',
                     'render_error.json','policy_error.json','trial_%03d.log'%first]:
            p=root/name
            if p.exists():p.rename(archive/name)
        save(archive/'resume.json',dict(first_trial=first,seed=seeds[first],
            reason='Accept low-variance occluded RGB as valid observations; no action/model/seed/scene changes.',
            old_code_sha256=old['code_sha256'],new_code_sha256=protocol['code_sha256']))
    else:
        save(root/'protocol.json',protocol)
    children=[];streams=[]
    def launch(argv,gpu,name):
        f=(root/(name+'.log')).open('x');streams.append(f)
        p=subprocess.Popen(list(map(str,argv)),env=env(gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    try:
        render_p=launch([BLENDER,__file__,'render','--root',root],3,'render')
        policy_p=launch([PY,__file__,'policy','--root',root],1,'policy')
        began=time.monotonic()
        while not all((root/(name+'_ready.json')).exists() for name in ('render','policy')):
            for p in children:assert p.poll() is None,('worker startup failed',p.returncode)
            assert time.monotonic()-began<900,'worker startup timeout'
            time.sleep(2)
        for i in range(first,trials):
            seed=seeds[i]
            cfg=dict(root=str(root),trial=i,seed=seed,base_y=random.Random(seed).uniform(.15,.175))
            cp=root/'configs'/('%03d.json'%i);save(cp,cfg)
            p=launch([PHYS,Path(__file__).parent/'physics_recorder.py','--name','%03d'%i,
                '--output-root',root/'trials','--seed',seed,'--base-y',cfg['base_y'],
                '--steps','2600','--record-actions','--policy-eval-config',cp],0,'trial_%03d'%i)
            start=time.monotonic();last=time.monotonic()-1800
            while p.poll() is None:
                assert render_p.poll() is None and policy_p.poll() is None,'worker failed'
                assert time.monotonic()-start<3600,'episode infrastructure timeout'
                live=root/'trials'/('%03d'%i)/'live.json'
                s=dict(phase='EVALUATING',time=time.strftime('%F %T'),trial=i,planned_trials=trials,
                    live=read(live) if live.exists() else None,completed=report(root,trials)['trials'])
                save(root/'status.json',s)
                if time.monotonic()-last>=1800:
                    with (root/'monitor_30min.jsonl').open('a') as f:f.write(__import__('json').dumps(s)+'\n')
                    last=time.monotonic()
                time.sleep(10)
            assert p.returncode==0,('physics infrastructure failure',i,p.returncode)
            metrics=root/'trials'/('%03d'%i)/'metrics.json'
            assert metrics.exists() and read(metrics)['complete']
            # Failing task trials remain in the denominator; never resample.
            s=report(root,trials);print('TRIAL_RESULT',i,read(metrics),flush=True)
        save(root/'status.json',dict(phase='COMPLETE',time=time.strftime('%F %T'),**report(root,trials)))
    except BaseException as exc:
        save(root/'status.json',dict(phase='FAILED',error=repr(exc),time=time.strftime('%F %T')))
        raise
    finally:
        for p in reversed(children):stop(p)
        for f in streams:f.close()


def main():
    def interrupt(signum,frame):raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupt)
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['run','render','policy']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--trials',type=int,default=64);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:
        if a.mode=='run':run(a.root,a.trials,a.resume)
        else:globals()[a.mode](a.root)
    except InterruptedError:
        if a.mode=='run':raise
    except BaseException as exc:
        if a.mode!='run':save(a.root/(a.mode+'_error.json'),dict(error=repr(exc)))
        raise


if __name__=='__main__':main()
