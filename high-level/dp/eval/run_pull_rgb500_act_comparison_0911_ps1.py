"""Durable RGB500 comparison: labelled Evo videos, ACT30K, then matched64 evaluation.

All outputs live on HDD. No changes to simulation, data, or existing experiments.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from eval_door_policy_success import summarize_log

REPO = Path('/home/ps/workspace/txc/doorgym')
ARCHIVE = Path('/mnt/sda/ps_archive/doorgym/high-level')
OUT = ARCHIVE / 'dp/result/pull_rgb500_act_evo_comparison_20260911'
REFERENCE = ARCHIVE / 'dp/result/pi05evo_rgb500_unlock25_open60_traverse050_20260910'
DATA_ID = 'local/door_a2w_0907_wc_nomenkuang_fixed15_gripper080_rgb_500_joint9'
DATA = ARCHIVE / 'data/lerobot' / DATA_ID
NAME = 'act_pull_rgb500_joint9_g080_baseline_seed1000_30k_0911'
TRAIN = ARCHIVE / 'dp/logs/lerobot-train' / NAME
WRAPPED = ARCHIVE / 'dp/logs/door-auto-wrapped' / NAME / '030000'
PYTHON = '/home/ps/miniconda3/envs/b1z1_lerobot/bin/python'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def counts(rows):
    return dict(trials=len(rows), **{k: sum(bool(r[k]) for r in rows) for k in
                ('door_open_success', 'traversal_success', 'success')})


def environment(gpu, native=False, viewer=False):
    env = os.environ.copy()
    # multiprocessing UNIX sockets have a ~108-byte path limit.
    tmp = Path('/mnt/sda/ps/pull0911tmp'); tmp.mkdir(parents=True, exist_ok=True)
    fork = 'hwk/lerobot/src' if native else 'lerobot/src'
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=f'{REPO}/high-level/{fork}:{REPO}/high-level',
               DOOR_DP_LEROBOT_PYTHON=PYTHON, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TMPDIR=str(tmp))
    if viewer:
        env.update(DISPLAY=':1', XAUTHORITY='/run/user/1000/gdm/Xauthority')
    return env


def run(cmd, stem, env):
    save(stem.with_suffix('.command.json'), cmd)
    with stem.with_suffix('.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        save(OUT / 'status.json', dict(state='running',stage=stem.name,pid=child.pid,time=time.time()))
        while child.poll() is None:
            time.sleep(30)
            with (OUT / 'monitor.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(), stage=stem.name, pid=child.pid,
                                       log_bytes=stem.with_suffix('.log').stat().st_size)) + '\n')
    return child.returncode


def checked_rollout(stem, num_envs, code):
    stdout = stem.with_suffix('.log').read_text()
    cleanup = code in (1,139,-11) and 'Done after 2600 steps' in stdout and 'SIGSEGV' in stdout
    assert code == 0 or cleanup, (stem,code,stdout[-2000:])
    lines = [l for l in stdout.splitlines() if l.startswith('[PullUnlockAudit] ')]
    assert len(lines) == 1, (stem,'missing physical latch audit')
    audit = json.loads(lines[0].split('] ',1)[1])
    assert len(audit) == num_envs and all(abs(t['handle_unlock_angle_deg']-25)<1e-4 for t in audit)
    n = collections.Counter(); first = {}; last = {}
    with stem.with_suffix('.jsonl').open() as f:
        for line in f:
            r = json.loads(line); i = int(r['controlled_env_id'])
            n[i] += 1; first.setdefault(i,r); last[i] = r['step']
            for k in ('state','dp_action_raw'):
                assert len(r[k]) == 9 and all(math.isfinite(v) for v in r[k])
    assert set(n) == set(range(num_envs)) and set(n.values()) == {2600} and set(last.values()) == {2599}, (n,last)
    rows = summarize_log(stem.with_suffix('.jsonl'),num_envs,60.,'abs',-1.,mode='ikpull',pull_traversal_distance_m=.5)
    for r in rows:
        r['failure_category'] = (None if r['success'] else 'opened_not_traversed' if r['door_open_success']
                                 else 'grasp_failure' if r['max_handle_or_secondary_dof_deg'] < 25 else 'insufficient_opening')
    return rows, audit, first


def set_arg(cmd, flag, value):
    cmd[cmd.index(flag)+1] = str(value)


def videos(gpu):
    import imageio_ffmpeg
    root = OUT / 'videos'; root.mkdir(parents=True,exist_ok=True)
    selected = []; number = collections.Counter()
    for attempt in range(48):
        if number['success'] == number['failure'] == 3:
            break
        seed = 615455575 + attempt
        stem = root / f'attempt{attempt:02d}_seed{seed}'
        if stem.with_suffix('.json').exists():
            result = json.loads(stem.with_suffix('.json').read_text())
        else:
            cmd = json.loads((REFERENCE/'batch_00/command.json').read_text())
            cmd.remove('--headless')
            for flag, value in [('--num_envs',1),('--seed',seed),('--dp_log_path',stem.with_suffix('.jsonl'))]:
                set_arg(cmd,flag,value)
            cmd += ['--overview_video_path',str(stem.with_suffix('.mp4')), '--overview_video_fps','25',
                    '--overview_video_capture_stride','2','--no_draw_scripted_trajectory','--no_preview_trajectory_at_spawn']
            code = run(cmd,stem,environment(gpu,native=True,viewer=True))
            rows,audit,_ = checked_rollout(stem,1,code)
            result = dict(rows[0], seed=seed, attempt=attempt, returncode=code, runtime_unlock_audit=audit,
                          checkpoint=cmd[cmd.index('--checkpoint')+1], video=str(stem.with_suffix('.mp4')))
            assert 'frames=1300' in stem.with_suffix('.log').read_text()
            save(stem.with_suffix('.json'),result)
        category = 'success' if result['success'] else 'failure'
        if number[category] < 3:
            number[category] += 1
            dest = root / f'{category}_{number[category]:02d}_seed{seed}.mp4'
            if not dest.exists():
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',result['video'],'-c:v','libx264',
                                '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],
                               check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            reader = imageio_ffmpeg.read_frames(str(dest))
            stream = next(reader); reader.close()
            frame_count, seconds = imageio_ffmpeg.count_frames_and_secs(str(dest))
            assert stream['codec'].startswith('h264') and frame_count==1300
            assert tuple(stream['size'])==(1280,720) and abs(seconds-52)<.05
            stream.update(nb_frames=frame_count,verified_seconds=seconds)
            selected.append(dict(category=category,path=str(dest),metrics=result,video_info=stream))
            save(root/'selected.json',selected)
        print('VIDEO',attempt,category,dict(number),flush=True)
    assert number['success'] == number['failure'] == 3, number
    save(root/'complete.json',dict(complete=True,selected=selected,
        criterion='same-frame abs(door)>=60deg AND base progress>=0.5m; physical latch25deg',
        note='Balanced illustrative selection, not an SR sample. Labels come from each video rollout itself.'))


def train_and_eval(gpu):
    info = json.loads((DATA/'meta/info.json').read_text())
    sidecar = json.loads((DATA/'door_dp_feature_names.json').read_text())
    assert info['total_episodes']==500 and info['total_frames']==650000
    assert info['features']['observation.state']['shape']==info['features']['action']['shape']==[9]
    assert sidecar['vision_mode']=='rgb' and sidecar['state_action_mode']=='joint_state9'
    assert sidecar['image_features']==['observation.images.wrist_rgb','observation.images.front_rgb']
    assert sidecar['action_source']=='base_command_plus_a2w_z1_joint_targets'
    save(OUT/'dataset_audit.json',dict(root=str(DATA),info=info,sidecar=sidecar,
         sha256={p:hashlib.sha256((DATA/p).read_bytes()).hexdigest() for p in
                 ('meta/info.json','meta/stats.json','door_dp_feature_names.json')}))
    checkpoint = TRAIN/'checkpoints/030000/pretrained_model'
    if not (checkpoint/'model.safetensors').exists():
        assert not TRAIN.exists(), f'Refusing to overwrite existing partial training: {TRAIN}'
        cmd = [PYTHON,str(REPO/'high-level/lerobot/src/lerobot/scripts/lerobot_train.py'),
               f'--dataset.repo_id={DATA_ID}',f'--dataset.root={DATA}', '--dataset.video_backend=torchcodec',
               '--dataset.use_imagenet_stats=false','--dataset.image_transforms.enable=false',
               '--policy.type=act','--policy.device=cuda','--policy.push_to_hub=false','--policy.use_amp=false',
               '--policy.chunk_size=100','--policy.n_action_steps=50','--policy.vision_backbone=resnet18',
               '--policy.optimizer_lr=1e-5','--policy.optimizer_lr_backbone=1e-5',
               '--policy.use_action_loss_weight=false','--policy.plucker_conditioning=false',
               '--policy.interaction_state_conditioning=false','--policy.end_signal_prediction=false',
               '--policy.camera_input_gating=false','--policy.point_cloud_conditioning=false','--policy.handle_latent_aux=false',
               '--seed=1000','--steps=30000','--batch_size=16','--num_workers=3','--save_freq=10000',
               '--log_freq=100','--keyframe_sampling_ratio=0','--eval_freq=0',f'--output_dir={TRAIN}',
               f'--job_name={NAME}','--wandb.enable=false','--wandb.disable_artifact=true']
        assert run(cmd,OUT/'training',environment(gpu)) == 0, 'ACT training failed; see training.log'
    cfg = json.loads((checkpoint/'config.json').read_text())
    assert cfg['input_features']['observation.state']['shape']==[9] and cfg['output_features']['action']['shape']==[9]
    assert not cfg['plucker_conditioning'] and not cfg['interaction_state_conditioning']
    assert set(k for k in cfg['input_features'] if k.startswith('observation.images.'))==set(sidecar['image_features'])
    train_cfg = json.loads((checkpoint/'train_config.json').read_text())
    assert train_cfg['steps']==30000 and train_cfg['seed']==1000 and Path(train_cfg['dataset']['root'])==DATA
    save(OUT/'trained_config_audit.json',dict(config=cfg,train_config=train_cfg))
    if not (WRAPPED/'model_latest.pt').exists():
        cmd = [PYTHON,str(REPO/'high-level/dp/export_official_lerobot_act_to_door_checkpoint.py'),
               '--official_checkpoint',str(checkpoint),'--root',str(DATA),'--repo_id',DATA_ID,
               '--out_dir',str(WRAPPED/'model_latest'),'--rgb','--action_horizon','25']
        assert run(cmd,OUT/'export',environment(gpu))==0, 'ACT export failed'
    if not (OUT/'export_parity.json').exists():
        cmd=[PYTHON,str(REPO/'high-level/dp/eval/audit_pull_rgb500_act_export_0911.py'),
             '--checkpoint',str(checkpoint),'--wrapper',str(WRAPPED/'model_latest.pt'),
             '--dataset',str(DATA),'--repo_id',DATA_ID,'--out',str(OUT/'export_parity.json')]
        assert run(cmd,OUT/'export_parity',environment(gpu))==0, 'Export parity check failed'
    assert json.loads((OUT/'export_parity.json').read_text())['passed']
    rows = []
    for i in range(4):
        stem=OUT/f'act_batch{i:02d}'
        if stem.with_suffix('.json').exists():
            data=json.loads(stem.with_suffix('.json').read_text())
        else:
            cmd=json.loads((REFERENCE/f'batch_{i:02d}/command.json').read_text())
            set_arg(cmd,'--checkpoint',WRAPPED/'model_latest.pt')
            set_arg(cmd,'--dp_log_path',stem.with_suffix('.jsonl'))
            code=run(cmd,stem,environment(gpu))
            trials,audit,first=checked_rollout(stem,16,code)
            ref_first={}
            with (REFERENCE/f'batch_{i:02d}/rollout.jsonl').open() as f:
                for line in f:
                    r=json.loads(line); ref_first.setdefault(int(r['controlled_env_id']),r)
                    if len(ref_first)==16: break
            for eid in range(16):
                for key in ('ikpush_randomization','pull_task_geometry'):
                    assert first[eid][key]==ref_first[eid][key], (i,eid,key)
            data=dict(**counts(trials),trials_detail=trials,runtime_unlock_audit=audit,returncode=code,
                      first_scene_matches_evo=True,seed=615455575+i)
            save(stem.with_suffix('.json'),data)
        rows+=data['trials_detail']
        evo=json.loads((REFERENCE/'summary.json').read_text())
        save(OUT/'comparison.json',dict(complete=len(rows)==64,act=dict(checkpoint=str(WRAPPED/'model_latest.pt'),
             **counts(rows),trials_detail=rows),evo={k:evo[k] for k in
                ('checkpoint','trials','door_open_success','traversal_success','success')},
             physical_unlock_deg=25,opening_deg=60,traversal_m=.5,steps=2600,action_horizon=25,
             training_steps=30000,training_seed=1000,eval_seeds=list(range(615455575,615455579))))
        print('ACT EVAL',i,counts(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--only',choices=['videos','train_eval','all'],default='all')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    # Exclusive pipeline lock prevents accidental duplicate workers.
    import fcntl
    lock=(OUT/f'{args.only}.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if args.only in ('videos','all'): videos(args.gpu)
        if args.only in ('train_eval','all'): train_and_eval(args.gpu)
        save(OUT/'status.json',dict(state='complete',scope=args.only,time=time.time()))
    except Exception as error:
        save(OUT/'status.json',dict(state='error',error=repr(error),time=time.time()))
        raise
