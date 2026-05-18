# Door Diffusion Policy Usage

All commands below are meant to be run from the repo root:

```bash
cd /home/sivan/whole_body/visual_whole_body
```

`b1z1` is used for Isaac Gym scripted play, raw data recording, and DP play.
`b1z1_lerobot` is used for raw-to-LeRobot conversion, Rerun visualization, and DP training.

## 1. Scripted Play

Pull-door scripted play with wrist camera:

```bash
conda run -n b1z1 python high-level/play_b1z1_walk_with_door_asset_camera.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --num_envs 1 \
  --steps 2500
```

Push-door scripted play with wrist camera:

```bash
conda run -n b1z1 python high-level/play_b1z1_push_with_door_asset_camera.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --num_envs 1 \
  --steps 2500
```

Hide OpenCV camera windows during scripted play:

```bash
conda run -n b1z1 python high-level/play_b1z1_walk_with_door_asset_camera.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --num_envs 1 \
  --steps 2500 \
  --no_show_seg
```

## 2. Record Raw Data

Raw recording now saves only successful episodes. A rollout is successful only after the door reaches
`--pass_open_angle_deg` (default `80`) and the scripted pass stage starts. Failed envs are discarded
and do not become `.npz` files.

By default, recording uses one simulator launch per mode and records all envs in parallel.
For example, if you want about 80 pull samples, use `--num_envs 80 --num_rollouts 1`.
It will attempt 80 envs at once and save only the successful ones. `--num_episodes` is kept as
a convenient alias: if `--num_envs` is not given, the script uses `--num_envs = --num_episodes`.

Record both pull and push:

```bash
conda run -n b1z1 python high-level/dp/record_door_dp_dataset.py \
  --mode both \
  --num_envs 80 \
  --num_rollouts 1 \
  --raw_root /home/sivan/whole_body/visual_whole_body/data/door_dp_raw/local_door_dp \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

Record pull only:

```bash
conda run -n b1z1 python high-level/dp/record_door_dp_dataset.py \
  --mode pull \
  --num_envs 80 \
  --num_rollouts 1 \
  --raw_root /home/sivan/whole_body/visual_whole_body/data/door_dp_raw/local_door_dp \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

Record push only:

```bash
conda run -n b1z1 python high-level/dp/record_door_dp_dataset.py \
  --mode push \
  --num_envs 80 \
  --num_rollouts 1 \
  --raw_root /home/sivan/whole_body/visual_whole_body/data/door_dp_raw/local_door_dp \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

Record only one env, useful for debugging:

```bash
conda run -n b1z1 python high-level/dp/record_door_dp_dataset.py \
  --mode pull \
  --num_episodes 1 \
  --num_envs 4 \
  --no_record_all_envs \
  --record_env_id 0 \
  --raw_root /home/sivan/whole_body/visual_whole_body/data/door_dp_raw/debug_single_env \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

Raw episodes are saved as:

```text
data/door_dp_raw/local_door_dp/
  door_dp_feature_names.json
  episode_000000.npz
  episode_000001.npz
  ...
```

Each `.npz` episode stores:

```text
state                  [T, state_dim]
action                 [T, 10]
wrist_handle_mask      [T, 54, 96, 3]
wrist_masked_depth     [T, 54, 96, 3]
front_handle_mask      [T, 54, 96, 3]
front_masked_depth     [T, 54, 96, 3]
subtask_index          [T, 1]
door_asset_index       scalar
door_asset_name        scalar
door_asset_path        scalar
door_cfg               scalar
task
fps
state_feature_names
action_names
```

`door_asset_index/name/path` identify the exact door asset used by the recorded env. Door assets
are assigned by env index as `env_i % loaded_door_asset_count`; they are not randomly sampled.
The index is the loaded asset index after the play script applies YAML/default filtering. Replay
prefers `door_asset_name` to load the same mesh.

`subtask_index` is saved only for visualization/debugging. It is not included in
`observation.state`, so the DP policy does not receive scripted phase labels at
inference time.

## 3. Replay Raw Data In Isaac Gym

Use state replay to check whether the recorded data itself looks correct. This mode does not run
the low-level policy; it restores recorded robot state each frame. New recordings include full
replay snapshots for robot root, robot joints, and door joints. Older raw files only contain
`observation.state`, so replay can restore `dof_pos/dof_vel` but not exact root xy/yaw or door state.
For new raw episodes, replay reads `door_asset_name` from the `.npz` and loads only that matching
door from `--door_cfg`.

```bash
conda run -n b1z1 python high-level/dp/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode data/door_dp_raw/local_door_dp/episode_000000.npz \
  --replay_mode state \
  --mode auto \
  --num_envs 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

For old raw episodes without door metadata, provide the door manually:

```bash
--door_asset_name 99650089960001
```

Use action replay to re-run the recorded 10D high-level actions through the low-level policy:

```bash
conda run -n b1z1 python high-level/dp/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode data/door_dp_raw/local_door_dp/episode_000000.npz \
  --replay_mode action \
  --mode auto \
  --num_envs 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

## 4. Convert Raw Data To LeRobotDataset

Convert raw `.npz` episodes to a local LeRobotDataset:

```bash
conda run -n b1z1_lerobot python high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root /home/sivan/whole_body/visual_whole_body/data/door_dp_raw/local_door_dp \
  --root /home/sivan/whole_body/visual_whole_body/data/lerobot \
  --repo_id local/door_dp \
  --overwrite
```

Converted dataset path:

```text
data/lerobot/local/door_dp/
```

## 5. View With Rerun

Directly open the local LeRobotDataset in Rerun:

```bash
conda run -n b1z1_lerobot python -m lerobot.scripts.visualize_dataset \
  --repo-id local/door_dp \
  --root /home/sivan/whole_body/visual_whole_body/data/lerobot/local/door_dp \
  --episode-index 0 \
  --mode local
```

Save a Rerun `.rrd` file for offline viewing:

```bash
conda run -n b1z1_lerobot python -m lerobot.scripts.visualize_dataset \
  --repo-id local/door_dp \
  --root /home/sivan/whole_body/visual_whole_body/data/lerobot/local/door_dp \
  --episode-index 0 \
  --mode local \
  --save 1 \
  --output-dir /home/sivan/whole_body/visual_whole_body/data/lerobot_viz
```

Open a saved `.rrd`:

```bash
conda run -n b1z1_lerobot rerun \
  /home/sivan/whole_body/visual_whole_body/data/lerobot_viz/local_door_dp_episode_0.rrd
```

## 6. Train Door DP

Short smoke training, only checks the training pipeline:

```bash
conda run -n b1z1_lerobot python high-level/dp/train_door_dp.py \
  --root /home/sivan/whole_body/visual_whole_body/data/lerobot/local/door_dp \
  --repo_id local/door_dp \
  --run_name smoke \
  --steps 2 \
  --batch_size 2 \
  --num_workers 0 \
  --device cuda:0
```

Full training example:

```bash
conda run -n b1z1_lerobot python high-level/dp/train_door_dp.py \
  --root /home/sivan/whole_body/visual_whole_body/data/lerobot/local/door_dp \
  --repo_id local/door_dp \
  --run_name door_dp_v1 \
  --steps 100000 \
  --batch_size 64 \
  --num_workers 4 \
  --device cuda:0
```

If CUDA memory is not enough, reduce:

```bash
--batch_size 32
```

Checkpoints are saved to:

```text
high-level/logs/door-dp/<run_name>/checkpoints/model_latest.pt
```

## 7. Play A Trained DP Policy

Install DP inference dependencies in `b1z1` once:

```bash
conda run -n b1z1 python -m pip install \
  "diffusers==0.24.0" \
  "huggingface-hub==0.20.3"
```

Pull-door DP play:

```bash
conda run -n b1z1 python high-level/dp/play_door_dp_policy.py \
  --checkpoint high-level/logs/door-dp/door_dp_v1/checkpoints/model_latest.pt \
  --mode pull \
  --num_envs 1 \
  --steps 2500 \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

Push-door DP play:

```bash
conda run -n b1z1 python high-level/dp/play_door_dp_policy.py \
  --checkpoint high-level/logs/door-dp/door_dp_v1/checkpoints/model_latest.pt \
  --mode push \
  --num_envs 1 \
  --steps 2500 \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

Play the smoke checkpoint, only for checking load/run:

```bash
conda run -n b1z1 python high-level/dp/play_door_dp_policy.py \
  --checkpoint high-level/logs/door-dp/smoke/checkpoints/model_latest.pt \
  --mode pull \
  --num_envs 1 \
  --steps 20 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --no_show_seg \
  --camera_display_scale 1
```

## 8. Action And Observation Format

Action is 10D:

```text
vx, yaw, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper
```

Visual observations:

```text
observation.images.wrist_handle_mask
observation.images.wrist_masked_depth
observation.images.front_handle_mask
observation.images.front_masked_depth
```

Proprioception:

```text
observation.state
```

Current state is 73D when `num_dofs=19` and `num_actions=18`:

```text
0      base_roll
1      base_pitch
2-4    base_ang_vel_x/y/z
5-23   dof_pos_0 ... dof_pos_18
24-42  dof_vel_0 ... dof_vel_18
43-60  last_low_action_0 ... last_low_action_17
61-64  foot_contact_0 ... foot_contact_3
65-67  ee_base_x/y/z
68-71  ee_qx/qy/qz/qw
72     gripper_pos
```

`wrist_masked_depth` and `front_masked_depth` are stored as 3-channel images for
LeRobot/video tooling and for the shared 3-channel CNN image encoder. They are
still grayscale masked depth visualizations internally; the three channels contain
the same value.

State/action feature names are stored in:

```text
door_dp_feature_names.json
```
