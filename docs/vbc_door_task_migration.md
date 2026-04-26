# VBC Door Task Migration

## Background
This migration adds a new high-level task, `B1Z1OpenDoor`, alongside the original `B1Z1PickMulti` task.

The goal is:
- keep the existing VBC control stack
- keep using the pretrained low-level executor
- avoid modifying `low-level`
- support a new lever-door loco-manipulation task in `high-level`
- focus on state-based teacher PPO only in this round

## What Was Added

### New task and config
- New environment: `high-level/envs/b1z1_opendoor.py`
- New task registration: `high-level/envs/__init__.py`
- New config: `high-level/data/cfg/b1z1_opendoor.yaml`

### New assets
- New door asset root: `high-level/data/asset/door_set/`
- Imported 4 lever-door assets from `UniDoorManip`:
  - `99650089960001`
  - `99650089960006`
  - `99655039960001`
  - `99655039960006`

Each asset directory contains:
- `mobility.urdf`
- `bounding_box.json`
- `handle_bounding.json`
- referenced `texture_dae/*` meshes and textures

### New scripts
- Asset loading validation: `high-level/play_door_locomanip_asset.py`
- Scripted door opening validation: `high-level/play_door_locomanip.py`

## Environment Design

### Scene layout
`B1Z1OpenDoor` inherits from `B1Z1Base` and uses:
- actor 0: robot
- actor 1: lever door

It does not create:
- table
- cube
- object feature buffers

### Door metadata parsed at init
For each door asset the env reads:
- door bounding box
- handle bounding box
- local `goal_pos`
- door DOF limits
- handle DOF limits

The env then builds:
- door root state tensor
- door DOF tensors
- handle rigid-body state
- door opening ratio
- handle opening ratio
- handle grasp goal in world frame
- handle approach / rotate / door-open directions

### High-level control stack
The task still uses the original VBC execution chain:
- high-level outputs EE delta goal, gripper command, base command
- low-level policy controls the legged base
- arm is executed by IK plus position targets

The only door-specific change is that the env now pads robot DOF targets and torques to the full `robot + door` DOF count before sending them to Isaac Gym.

## Observation Design

`B1Z1OpenDoor` removes object PointNet features and instead uses explicit door state observations.

The core observation includes:
- grasp goal position in robot-local frame
- handle position in robot-local frame
- handle local orientation (RPY)
- handle approach direction
- lever rotation direction
- door opening direction
- end-effector local position
- end-effector local orientation
- EE-to-goal local offset
- door root position in robot-local frame
- door hinge angle
- handle joint angle
- door opening ratio
- robot proprioception
- base commands
- current EE goal cartesian target
- current EE goal orientation target
- robot local velocity

The final high-level observation appends the last action / command history in the same style as the original VBC teacher task.

## Reward Design

The door task replaces pick/lift rewards with door-specific rewards:
- `approach_handle`
  - dense reward from progress in gripper-to-grasp-goal distance
- `ee_align_handle`
  - encourages the gripper orientation to align with the handle frame
- `lever_press`
  - rewards handle-joint progress before the door starts opening
- `door_open_progress`
  - rewards hinge-angle progress after the handle is sufficiently pressed
- `door_open_success`
  - sparse one-time success reward when the door opening ratio crosses the threshold
- `base_command_penalty`
  - suppresses unnecessary base motion near the door
- `action_rate`
  - inherited smoothness penalty
- `gripper_rate`
  - inherited gripper smoothness penalty
- `base_height`
  - inherited stability term

## Success and Termination

### Success
Door success is defined by:
- `door_open_ratio >= doorOpenSuccessThreshold`

The env also keeps a short hold counter before resetting the successful episode.

### Termination
The task resets on:
- base roll / pitch / height failure inherited from `B1Z1Base`
- IK failure inherited from `B1Z1Base`
- sustained door-open success
- end-effector drifting too far away from the handle region
- base remaining too far from the task for too long
- episode timeout

## Teacher PPO Integration

`train_multistate.py` now supports `B1Z1OpenDoor` directly:
- config file resolution works via `b1z1_opendoor.yaml`
- task registration is exposed from `envs/__init__.py`
- teacher feature encoder size is now inferred from `env.num_features`
  - this keeps object tasks unchanged
  - this also allows door task to run with `num_features = 0`

Additional robustness updates:
- `TORCH_EXTENSIONS_DIR` defaults to `/tmp/torch_extensions` in the teacher train/play entrypoints
- CPU fallback disables `use_gpu_pipeline` and `physx.use_gpu` when `--sim_device cpu`
- `B1Z1Base.create_sim()` now uses `graphics_device_id` instead of reusing the compute device id for both arguments

## Validation Scripts

### 1. Asset loading
Run:

```bash
cd high-level
python play_door_locomanip_asset.py --headless --num_envs 4
```

What it checks:
- robot and door both load
- door rigid body indices are found
- door DOF count is correct
- key door tensors are created

### 2. Scripted door opening
Run:

```bash
cd high-level
python play_door_locomanip.py --headless --num_envs 4
```

What it does:
- generates scripted high-level commands
- first walks the robot toward the door
- moves to a pre-grasp pose
- closes in on the handle
- presses the lever
- pulls along the door opening direction

This is not RL. It is only a sanity-check trajectory generator for the new task.

### 3. Teacher PPO startup
Run:

```bash
cd high-level
python train_multistate.py \
  --task B1Z1OpenDoor \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --headless \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero
```

For CPU-only smoke testing:

```bash
cd high-level
python train_multistate.py \
  --task B1Z1OpenDoor \
  --rl_device cpu \
  --sim_device cpu \
  --headless \
  --debug \
  --timesteps 1 \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero
```

## Current Scope and Limitations
- Only lever doors are supported in this round
- Only teacher PPO is supported in this round
- Student / DAgger visual policy is not adapted yet
- No PointNet object features are used in the door task
- The scripted controller is a sanity-check controller, not a polished demonstration policy

## Main Modified Files
- `high-level/envs/__init__.py`
- `high-level/envs/b1z1_base.py`
- `high-level/envs/b1z1_opendoor.py`
- `high-level/data/cfg/b1z1_opendoor.yaml`
- `high-level/train_multistate.py`
- `high-level/play_multistate.py`
- `high-level/play_door_locomanip_asset.py`
- `high-level/play_door_locomanip.py`

