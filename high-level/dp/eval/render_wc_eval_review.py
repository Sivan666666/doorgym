"""Render original closed-loop trial poses as 25 fps review videos.

Uses the frozen live-evaluation renderer and scene, without rerunning the policy
or physics. All generated files live in a separate review directory.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--eval-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trial', type=int, required=True)
    p.add_argument('--ffmpeg', required=True)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--encode-only', action='store_true')
    a = p.parse_args()
    sys.path.insert(0, str(a.eval_root / 'code'))
    from wc_blender_physx_common import read, save, sha
    import run_wc_blender_physx_eval as adapter
    import cv2
    import numpy as np

    trial = a.eval_root / 'trials' / f'{a.trial:03d}'
    out = a.out / f'trial_{a.trial:03d}'
    out.mkdir(parents=True, exist_ok=True)
    source = read(trial / 'trajectory.json')
    metrics = read(trial / 'metrics.json')
    trace = read(trial / 'metrics_trace.json')
    initial = read(a.eval_root / 'ipc' / f'{a.trial:03d}' / '000/request.json')
    assert source['fps'] == 50 and len(source['frames']) == 2600
    runtime = out / ('probe_runtime' if a.probe else 'runtime')
    runtime.mkdir(exist_ok=True)
    assert 0 <= a.shard_index < a.shard_count
    worker = runtime / f'worker_{a.shard_index}'
    worker.mkdir(exist_ok=True)
    # Only remove our own completion sentinel when resuming this review job.
    (worker / 'stop').unlink(missing_ok=True)
    indices = [0, 350, 750, 1100, 1299] if a.probe else list(range(1300))
    manifest = dict(trial=a.trial, metrics=metrics, source=str(trial),
                    trajectory_sha256=sha(trial / 'trajectory.json'),
                    scene_sha256=sha(adapter.TEMPLATE), fps=25, duration_s=52,
                    source_indices=[None if i == 0 else 2*i-1 for i in indices],
                    video_indices=indices, physics_rerun=False, policy_rerun=False,
                    pose_interpolation=False, lighting_changed=False,
                    caveat='Historical success may include base-door contact; prescribed floating base.')
    if a.shard_index == 0 and not a.encode_only:
        save(out / ('probe_manifest.json' if a.probe else 'manifest.json'), manifest)
    if not a.probe:
        indices = indices[a.shard_index::a.shard_count]
    started = time.monotonic()

    def requests(root, done):
        for i in indices:
            folder = runtime / 'frames' / f'{i:04d}'
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'render.json').exists() and all((folder/(v+'.png')).exists() for v in ('front','wrist')):
                continue
            record = dict(initial)
            if i:
                frame = source['frames'][2*i-1]
                for k in ('replay_root_state', 'replay_door_root_state', 'bodies',
                          'front_camera_pose_base', 'wrist_camera_pose_base'):
                    record[k] = frame[k]
            record.update(request=i, physics_step=2*i, tick=i, simulation_time_s=i/25,
                          source='recorded_closed_loop_physx', source_frame_index=None if i==0 else 2*i-1)
            save(folder/'request.json', record)
            yield folder/'request.json'
            assert (folder/'observations.npz').exists()
            rendered = read(folder/'render.json')
            rendered['source'] = 'recorded_closed_loop_physx'
            rendered['source_frame_index'] = record['source_frame_index']
            save(folder/'render.json', rendered)
            # The video needs RGB only; this is our newly generated scratch array.
            (folder/'observations.npz').unlink()
            if i % 50 == 0 or i == indices[-1]:
                status=dict(trial=a.trial, phase='render', frame=i, frames=1300,
                            elapsed_s=time.monotonic()-started)
                save(out/f'progress_{a.shard_index}.json', status)
                print('REVIEW_PROGRESS', json.dumps(status), flush=True)
        save(worker/'stop', dict(complete=True))

    adapter.requests = requests
    if not a.encode_only:
        adapter.render(worker)
    if a.shard_count > 1 and not a.encode_only:
        save(out/f'shard_{a.shard_index}_complete.json', dict(indices=indices, complete=True))
        return
    if a.probe:
        rows=[]
        for i in indices:
            row=np.hstack([cv2.imread(str(runtime/'frames'/f'{i:04d}'/(v+'.png'))) for v in ('front','wrist')])
            cv2.putText(row, f'trial {a.trial:03d}  t={i/25:.2f}s', (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,255),2)
            rows.append(row)
        assert cv2.imwrite(str(out/'probe.jpg'), np.vstack(rows))
        return

    label = 'SUCCESS_WITH_BASE_DOOR_CONTACT' if metrics['success'] else 'FAILURE'
    target = out / f'{label.lower()}_trial_{a.trial:03d}_dual_rgb.mp4'
    tmp = target.with_name(target.stem+'.partial.mp4')
    command=[a.ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
             '-pixel_format','bgr24','-video_size','1280x544','-framerate','25','-i','pipe:0',
             '-an','-c:v','libx264','-crf','18','-preset','fast','-threads','4',
             '-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    image_hashes=[]
    try:
        for i in range(1300):
            folder=runtime/'frames'/f'{i:04d}'
            images=[cv2.imread(str(folder/(v+'.png'))) for v in ('front','wrist')]
            assert all(im is not None and im.shape==(480,640,3) for im in images)
            metadata=read(folder/'render.json')
            assert metadata['request_sha256']==sha(folder/'request.json')
            assert metadata['max_position_error_m']<1e-4
            hashes={v:sha(folder/(v+'.png')) for v in ('front','wrist')}
            assert hashes==metadata['rgb_sha256']
            image_hashes.append(hashes)
            canvas=np.zeros((544,1280,3),np.uint8)
            canvas[64:]=np.hstack(images)
            state=trace[max(0,2*i-1)]
            title=f'Trial {a.trial:03d} | {label} | Recorded PhysX / Blender RGB | 25 fps'
            detail=(f't={i/25:05.2f}s | open={state["open_deg"]:5.1f} deg | '
                    f'traversal={state["traversal_m"]:5.2f}m | FRONT (left) / WRIST (right)')
            cv2.putText(canvas,title,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,detail,(12,51),cv2.FONT_HERSHEY_SIMPLEX,.6,(210,210,210),1,cv2.LINE_AA)
            proc.stdin.write(canvas.tobytes())
    finally:
        proc.stdin.close()
    assert proc.wait()==0
    cap=cv2.VideoCapture(str(tmp));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(544,1280,3);count+=1
    cap.release()
    assert count==1300 and fps==25,(count,fps)
    tmp.replace(target)
    save(out/'frame_hashes.json',image_hashes)
    save(out/'verification.json',dict(passed=True,file=str(target),frames=count,fps=fps,
         duration_s=count/fps,dimensions=[1280,544],sha256=sha(target),
         manifest_sha256=sha(out/'manifest.json'),elapsed_s=time.monotonic()-started))
    save(out/'progress.json',dict(trial=a.trial,phase='COMPLETE',video=str(target)))
    print('REVIEW_COMPLETE',str(target),flush=True)


if __name__ == '__main__':
    main()
