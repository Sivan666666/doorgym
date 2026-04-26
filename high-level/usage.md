# High-Level Door Usage

This file records the added scripts, smoke-test commands, and key prompts that guided the implementation.

Run commands from `high-level` unless noted otherwise.

## Added Scripts

### `play_b1z1_walk_only.py`

Purpose:

- Verify the original low-level `b1z1` policy can walk and track its regular random EE targets without any door asset.

Command:

```bash
python play_b1z1_walk_only.py --rl_device cuda:0 --sim_device cuda:0 --num_envs 4 --steps 600
```

Useful fixed-command check:

```bash
python play_b1z1_walk_only.py --rl_device cuda:0 --sim_device cuda:0 --num_envs 4 --steps 600 --fixed_vx 0.7 --fixed_yaw 0.0
```

### `play_b1z1_walk_with_door_asset.py`

Purpose:

- Use the low-level `ManipLoco` robot/arm implementation.
- Load lever-door assets from `data/cfg/b1z1_opendoor.yaml`.
- Stop the dog near the door and run a scripted EE trajectory for handle approach, grasp, lever rotation, and door pulling.
- Visualize the current scripted EE target as a red point.

Default command:

```bash
python play_b1z1_walk_with_door_asset.py --rl_device cuda:0 --sim_device cuda:0 --num_envs 4 --steps 2000
```

Position-only IK validation:

```bash
python play_b1z1_walk_with_door_asset.py --rl_device cuda:0 --sim_device cuda:0 --num_envs 4 --steps 2000 --external_orn_gain 0.0
```

Weak forward-orientation IK validation:

```bash
python play_b1z1_walk_with_door_asset.py --rl_device cuda:0 --sim_device cuda:0 --num_envs 4 --steps 2000 --external_orn_gain 0.2
```

Useful trajectory parameters:

```bash
--pregrasp_offset 0.22
--grasp_offset 0.0
--grasp_hold_steps 100
--gripper_close_steps 120
--handle_rotate_steps 420
--door_pull_steps 480
--handle_arc_radius 0.18
--door_pull_distance 0.45
--door_z_offset 0.0
```

### `play_b1z1_walk_with_door.py`

Purpose:

- Earlier low-level walking-with-door experiment entry.
- Useful as a lightweight comparison point while developing the asset-based script.

Command:

```bash
python play_b1z1_walk_with_door.py --rl_device cuda:0 --sim_device cuda:0 --num_envs 4 --steps 600
```

### `play_door_locomanip_asset.py`

Purpose:

- Visualize robot and lever-door assets together.
- Check door actor, DOF, handle-body names, and asset placement.

Command:

```bash
python play_door_locomanip_asset.py --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 --num_envs 4 --steps 240
```

### `play_door_locomanip.py`

Purpose:

- Scripted high-level door-opening check using ground-truth door and handle state.

Command:

```bash
python play_door_locomanip.py --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 --num_envs 4 --steps 700
```

### `play_door_locomanip_walk.py`

Purpose:

- Keep the door loaded while checking low-level walking stability.

Command:

```bash
python play_door_locomanip_walk.py --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 --num_envs 4 --steps 240 --resample_interval 60
```

## Training / Startup Checks

Door teacher training:

```bash
python train_multistate.py --task B1Z1OpenDoor --rl_device cuda:0 --sim_device cuda:0 --timesteps 60000 --headless --experiment_dir b1-open-door-teacher --wandb --wandb_project b1-open-door-teacher --wandb_name door-debug --roboinfo --observe_gait_commands --small_value_set_zero
```

Minimal startup check:

```bash
python train_multistate.py --task B1Z1OpenDoor --rl_device cuda:0 --sim_device cuda:0 --timesteps 1 --headless --debug --roboinfo --observe_gait_commands --small_value_set_zero
```

CPU startup check:

```bash
python train_multistate.py --task B1Z1OpenDoor --rl_device cpu --sim_device cpu --timesteps 1 --headless --debug --roboinfo --observe_gait_commands --small_value_set_zero
```

## Key Prompts / Design Requirements

- Create a new script that copies robot, arm, and terrain behavior from `play_b1z1_walk_only.py`, and copies only the door asset loading style from `play_door_locomanip_asset.py`.
- Do not reference robot/arm logic from `play_door_locomanip_asset.py`.
- Match UniDoorManip lever-door behavior:
  - The door is locked before the handle reaches an opening threshold.
  - Apply a strong opposite force before unlock.
  - Add door opening resistance after unlock.
  - Add handle spring return and damping.
- Arrange each play environment as one dog and one door in a clean fixed grid.
- Place the dog on the handle side of the door and make the walking direction face the door.
- Stop the dog when the front collision box is near the door, then zero `vx`.
- Keep the arm at default angles while walking.
- After stopping, run a scripted EE trajectory:
  - approach the handle,
  - hold at the handle with gripper open,
  - close the gripper,
  - rotate the lever downward,
  - pull the door open,
  - hold.
- Red visual target should show the active scripted EE target.
- Remove or disable the old random red trajectory visualization when using the scripted door trajectory.
- Validate whether poor red/blue tracking comes from 6D IK orientation tracking; test position-only IK by setting orientation error to zero.
- Keep `external_orn_gain` configurable because the current low-level policy was trained mainly for EE position tracking, not full 6D pose tracking.
