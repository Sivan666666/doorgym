# PS1: dim60 RGB100 and native pi0.5 Evo training

User request (2026-10-08): use the reviewed 60% photo-office scene, generate
100 RGB-augmented episodes by Gym physics + Blender replay, train pi0.5 on PS1,
and inspect progress every 30 minutes. Reuse compatible prior recordings.

## Reuse decision

The prior Gym RGB500 stores root/joint/camera states, but used
`wc_nomenkuang/model.urdf`, not the corrected `wc_contact/model.urdf` underlying
the approved dim60 video. Old hinge origin is `(-.463,0,0)`, handle height
`.985`; current hinge is `(-.443226,-.036,.004)`, handle offset height `.922`,
and the hinge opening sign is reversed. Blindly applying old motion would
reintroduce contact misalignment. The old files remain untouched.

The accepted dim60 physical recording is reused as episode 0. Remaining
episodes run the current collision model and controller without Gym RGB.
Failed candidates and their contact logs are preserved. Pilot base_y=.125
failed to unlock; .15 and .175 succeeded. Production samples [.15,.175].
Seeds change reproducibly on physical retry; no fabricated successful poses.
The export gate requires 100 distinct trajectory hashes. This remains one
task/controller with small start-offset variations, not 100 different skills.

## Visual/data contract

- Baseline entrance/office AREA powers: 39/81; lamp emissions .9/2.4.
- Fixed room/door/robot geometry, material colours and furnishings.
- AREA power per episode ×[.8,1.2], small warm/cool tint ±.05; world stays zero.
  Emissive materials remain at the dim60 baseline, preserving material identity.
- DoorMan-derived brightness/contrast ×[.9,1.1] at p=.25; saturation
  ×[.8,1.2] p=.25; hue ±.025 cycle p=.5; noise std [0,.015] p=.25;
  3×5 Gaussian blur p=.1. Parameters stable per episode; noise per frame/view.
- Two training views: front/wrist, 640×480, **25 fps** (matching RGB500),
  24 Cycles samples/denoise. 1300 RGB pairs/episode, total 130,000 rows /
  260,000 rendered images. Source steps 0,2,...,2598; no interpolated frames.
- Official LeRobot v3, state/action Joint10: vx, vy, vyaw + 6 arm joints + gripper.
  State has previous base command and measured joints; action is the actual
  next 20 ms command from the retained 50 Hz control stream. RGB/labels sampled
  at 25 Hz; this does not imply holding each sampled command for 40 ms.
- Privileged door state/camera poses are kept in raw/render sidecars, excluded
  from training observations. This model is NOT drop-in compatible with the
  older Joint9 policy/deployment adapter. Future execution must match timing.

## Training and monitoring

Fresh fine-tuning from existing native pi05 Evo base (not RGB500 checkpoint):
30,000 steps, batch16, bf16, gradient checkpointing, seed1000, chunk100,
n_action_steps50, inference10. Extra training-time image transforms disabled.
Three-step batch16 numerical/runtime preflight uses a separate 10 fps preview
dataset/output, not production training data. Production uses only the newly
rendered 25 fps dataset and checks every source index and timestamp before
training. At 25 fps, chunk100 covers 4 s and n_action_steps50 covers 2 s.
The first preflight exposed a zero q01–q99 span for the sparse yaw channel,
producing losses around 1e12 despite valid raw commands. State/action use
MIN_MAX normalization for this new model (images IDENTITY); original labels
and raw statistics are not edited. Updated preflight and training reject
non-finite or abnormally large (>100) normalized loss. The earlier diagnostic
checkpoint is preserved but is never used to initialize production training.
Production begins only after that preflight and recorded-motion checks pass.
No automatic closed-loop evaluation was requested for this run.

Root: `/mnt/sda/ps_archive/doorgym/high-level/dp/result/wc_dim60_aug100_pi05_20261008`.
Supervisor: `code/run_wc_dim60_aug100_pi05.py`.
Configuration: `sim_data/configs/rgb_photo_office_dim60_aug100.json`.
Model output: `/mnt/sda/ps_archive/doorgym/high-level/dp/logs/lerobot-train-pi05/pi05evo_wc_dim60_aug100_joint10_seed1000_30k`.

`status.json` refreshes every minute. `monitor_30min.jsonl` records counters,
loss/steps, disk and GPU use every 1800 seconds. Pipeline stages have bounded
timeouts/retries. Rendering uses GPUs 0/1/3 with two independent workers each
(48 CPU threads / 251 GiB host RAM checked at launch); GPU2 is left untouched.
Each worker has 4 render threads. Training uses GPU1 only after export/QC.
Pipeline stages have bounded
timeouts/retries; training requires finite loss and actual step progress.
Failure stops only this task's own child process group. Existing jobs/models
are never killed/overwritten. Completion requires all 100 episodes, full label
and video checks, the 30K checkpoint, matching feature config, and weight hashes.

## Monitoring handoff — 2026-10-08

Production started at 01:55:56 CST, supervisor PID 666498, pipeline PID 666499.
Last verified server snapshot at **13:06:52 CST**: 42 accepted physical
episodes, 39 scenes, 36 fully rendered episodes, 46,946 / 130,000 RGB pairs,
no unresolved failures, 2437.8 GiB free. Production training had not started.
The assistant checked approximately every 30 minutes until connectivity failed.
At the next check the workstation had moved off the 192.168.1.x LAN;
192.168.1.106:22 closed the connection before SSH key exchange. Configured
alternative 192.168.193.16 returned “No route to host” (checked with the
original PS1 host-key alias). Both still failed after another 30-minute wait.
No background job was stopped or restarted. Subsequent progress is UNKNOWN,
not evidence that rendering stopped. Restore PS1 connectivity before reporting
new counts or restarting anything. Read status.json, monitor_30min.jsonl and
supervisor_25fps.log first; the detached supervisor may already have advanced
to export/training. User was asked for restored connectivity or a usable PS1
SSH endpoint. Do not mutate the frozen deployed sim_data snapshot/config while
this fingerprinted run is active.

Connectivity restored at 18:29 CST on 2026-10-08 via the original
192.168.1.106 endpoint. The same supervisor/pipeline PIDs remained active;
no restart was required. Snapshot at 18:28:52: 55 accepted recordings,
54 scenes, 49 complete renders, 69,916 / 130,000 RGB pairs, no unresolved
failures, 2422.2 GiB free, production training not yet started. Server-side
30-minute monitoring continued throughout the disconnection. Assistant-side
30-minute checks resumed.

On 2026-10-09 all 100 physical episodes/scenes/renders completed (178 physical
attempts including rejected candidates). Dataset export and training gate
passed: 100 unique trajectories, 130,000 rows, 25 fps, 260,000 decoded video
frames. `dataset_training_gate.json` reports mean RGB front=128.966,
wrist=113.158 and high-clipping fractions 0.000001631 / 0.000142403.
Formal training launched at 10:37:54 CST, child PID 1293514, log
`TRAINING_20261009_103754.log`. First verified log at 10:41:12: step50,
loss0.225, grad norm2.691, 2.749 s/update, batch16; GPU1 about37.9GiB and100%
utilization. This is the production 25 fps dataset, not the preflight preview.
Continue half-hour checks until the 30K checkpoint and completion audit pass.

**Completed 2026-10-10 10:14:37 CST.** All 100 episodes and the 30,000-step
checkpoint passed the supervisor audit; training exited normally at 10:14:11.
600 finite loss logs, last loss0.003, maximum0.225, training process84,998s.
Independent final read-only check confirmed checkpoint/config/normalization,
recomputed all weight hashes, and scanned all1,032 tensors (3,616,758,136
values including normalization tensors) for NaN/Inf: all finite. Final model
SHA256 `d1cd52837c431c6ea0272ca25038ba6f7488c79380497ae377cd1a64704a1cc8`.
No closed-loop success rate was measured. Monitoring for this completed job
can stop. Local handoff and audit copies are under
`high-level/dp/result/wc_dim60_aug100_pi05_20261008/README.md`; full dataset and
model weights remain on PS1.
