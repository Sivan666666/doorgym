# A2W front-camera FoundationPose

The original `isaacgym_float_ik_a2w_basearn_push_door_parallel.py` is not
modified. Use the dedicated wrapper:

```text
isaacgym_float_ik_a2w_basearn_push_door_parallel_foundationpose.py
```

Isaac Gym runs in `b1z1`. A persistent subprocess runs FoundationPose with
`/home/sivan/miniconda3/envs/foundationpose/bin/python`, so their Python/CUDA
dependencies remain isolated.

For WC4, the wrapper uses the exact `lever_handle` visual geometry exported
from the URDF. The simulation handle segmentation initializes FoundationPose;
subsequent adjacent front RGB-D frames use `track_one` by default.

Recommended viewer command:

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n b1z1 python -u \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel_foundationpose.py \
  --num_envs 1 \
  --steps 1000 \
  --seed 1 \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_name wc4 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --enable_front_camera \
  --no_enable_wrist_camera \
  --camera_rgb \
  --camera_depth \
  --camera_seg \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --no_enable_depth_noise \
  --no_enable_depth_gaussian_blur \
  --no_enable_depth_camera_randomization \
  --foundationpose_interval 1 \
  --foundationpose_reregister_every 0 \
  --foundationpose_output_dir high-level/logs/foundationpose/wc4_front_camera_play
```

Outputs:

- `latest_pose.json`: latest estimated pose and simulation ground-truth error.
- `poses/frame_*.txt`: `T_camera_handle`, OpenCV camera frame.
- `overlays/frame_*.png`: front RGB with estimated 3D box and axes.
- `foundationpose_worker.log`: worker/model diagnostics.

Relevant modes:

- `--foundationpose_interval 1`: feed adjacent simulation frames; recommended.
- `--foundationpose_reregister_every 0`: first-frame registration followed by video tracking.
- `--foundationpose_reregister_every N`: use the simulation mask for periodic global relocalization.
- `--foundationpose_async`: let simulation continue while inference runs. This can skip large motions and is not recommended for accuracy.
- `--no_foundationpose_show_overlay`: save results without opening the overlay window.

## SAM3 mask + FoundationPose grasp evaluation

The SAM3 variant does not feed simulator segmentation into FoundationPose:

```text
front RGB
  -> SAM3 text prompt "door handle"
  -> fixed front-view ROI candidate selection
  -> binary handle mask
front RGB-D + SAM3 mask + WC4 handle mesh
  -> FoundationPose
  -> estimated initial grasp XYZ
  -> original scripted A2W controller
```

Use the independent wrapper:

```text
isaacgym_float_ik_a2w_basearn_push_door_parallel_sam3_foundationpose_grasp.py
```

SAM3 runs in `/home/sivan/miniconda3/envs/sam3` and FoundationPose runs in its
own `foundationpose` environment. The original A2W script is still imported
unchanged. Simulator segmentation is captured only after SAM3 returns, to
calculate mask IoU in the evaluation report. It is never sent to the SAM3
worker or the FoundationPose worker. Simulator pose is retained for pose-error
metrics and the same 15 mm crash guard used in the GT-mask experiment; a
rejected estimate counts as failure and is not corrected or rescued using GT.

The fixed WC4 front-view ROI is an image-space instance-selection prior. It
rejects SAM3's common false positive on the top door frame, but it does not use
the simulator handle pose, projected grasp point, or GT mask.

Single-environment smoke test result with seed `615455575`:

```text
SAM3 mask IoU (evaluation only): 0.988
FoundationPose grasp-point error: 3.0 mm
maximum door angle: 90.0 deg
success: 1/1
```

Run the reproducible 64-trial WC4 evaluation with:

```bash
high-level/float_ik/eval_sam3_foundationpose_grasp_wc4_64.sh
```

It uses seeds `615455575..615455578`, 16 virtual environments per seed,
camera position/rotation randomization of `0.02 m / 5 deg`, no depth noise,
1000 simulation steps, and an absolute door-angle success threshold of
`80 deg`. Physical simulation is run in two-env subprocesses while preserving
the original virtual environment indices; larger groups can make PhysX
unstable during post-contact withdrawal for imperfect pose estimates.

Completed result:

| Initial grasp source | Seed 575 | Seed 576 | Seed 577 | Seed 578 | Total |
|---|---:|---:|---:|---:|---:|
| Manual/simulator grasp | 16/16 | 16/16 | 16/16 | 16/16 | 64/64 = 100.00% |
| GT mask + FoundationPose | 11/16 | 9/16 | 11/16 | 11/16 | 42/64 = 65.63% |
| SAM3 mask + FoundationPose | 8/16 | 9/16 | 8/16 | 9/16 | 34/64 = 53.13% |

On the 64 exactly paired randomized environments, GT-mask and SAM3-mask
FoundationPose both succeeded 29 times and both failed 17 times. GT-mask alone
succeeded 13 times, while SAM3-mask alone succeeded 5 times. The net difference
is therefore 8 successes, or 12.50 percentage points.

For SAM3 + FoundationPose, 62 trials produced a mask large enough for pose
registration and 2 produced no usable mask. Of those 62 masks, 53 had IoU at
least 0.5 and 47 had IoU at least 0.8 against the evaluation-only simulator
mask. Mean IoU was 0.775 and median IoU was 0.884.

The simulator-only 15 mm grasp-error guard accepted 34 estimates, and every
accepted estimate opened the door. Accepted grasp-point error was 1.48 mm on
average, 1.36 mm median, and 3.44 mm maximum. Among 28 rejected pose estimates,
20 already had mask IoU at least 0.8 and 25 had rotation error over 90 degrees.
Therefore the largest remaining bottleneck is the symmetric lever-handle pose
flip in FoundationPose, not SAM3 mask quality alone.

Full aggregate report:

```text
high-level/logs/foundationpose/wc4_sam3_fp_grasp_64_seed615455575_v2/aggregate_summary.json
```

## One-shot initial handle pose

The grasp wrapper supports three backward-compatible pose modes:

| CLI | Mode | Behavior |
|---|---|---|
| no pose-mode flag | `end_of_walk_once` | Register once after reaching the approach view. This preserves the previous default. |
| `--foundationpose_initial_pose_only` | `first_valid_once` | Register on the first reliable SAM3 observation, cache the world-frame grasp point, then never track or update it. |
| `--foundationpose_track_during_walk` | `track_during_walk` | Register and continuously update the pose while approaching. This preserves the previous tracking mode. |

`--foundationpose_initial_pose_only` and `--foundationpose_track_during_walk`
are mutually exclusive. A forced simulation-frame-zero registration is not
used because the handle can be too small for reliable SAM3 segmentation at the
farthest randomized starting distance. `first_valid_once` is still a one-shot
pose estimate: after the first accepted registration for an environment, no
later RGB-D frame is sent to FoundationPose for that environment.

BundleSDF model-free example additions:

```bash
--foundationpose_mesh \
  high-level/logs/foundationpose/bundlesdf_wc4_front_sam3_no_gt_60_v2_output/textured_mesh.obj \
--foundationpose_model_grasp_point 0.00570725 0.00250802 0.01294013 \
--foundationpose_initial_pose_only \
--foundationpose_walk_track_interval 5
```

Two-environment smoke test, seed `615455575`:

```text
env 0: registered once at step 0, grasp error 5.75 mm, opened to 90 deg
env 1: registered once at step 95, grasp error 5.97 mm, opened to 90 deg
FoundationPose tracking calls after registration: 0
success: 2/2
```

Result:

```text
high-level/logs/foundationpose/wc4_bundlesdf_no_gt_sam3_fp_first_valid_once_2env_smoke/summary.json
```
