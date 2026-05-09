# Low-Level Modify Notes

This file records local changes made on top of the original low-level code.

## Play Command Overrides

Files:

- `legged_gym/utils/helpers.py`
- `legged_gym/scripts/play.py`

Changes:

- Added `--fixed_vx` and `--fixed_yaw` CLI arguments.
- When either argument is provided, `legged_gym/scripts/play.py` disables command curriculum for play and sets the command ranges to the requested fixed values.
- During play, the script repeatedly writes the fixed commands into `env.commands` before and after each environment step so command resampling does not overwrite them.

Purpose:

- Make low-level locomotion debugging deterministic.
- Allow quick checks such as walking straight forward with zero yaw.
- Support the high-level door debugging workflow where the robot must approach the door from a known direction.

Example:

```bash
cd low-level
python legged_gym/scripts/play.py --task b1z1 --rl_device cuda:0 --sim_device cuda:0 --checkpoint 45000 --flat_terrain --fixed_vx 0.7 --fixed_yaw 0.0
```

## Runtime Compatibility

The play scripts in `high-level` and modified high-level training entry set:

```python
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
```

Purpose:

- Avoid writing torch extension build artifacts into restricted or shared source directories during local testing.

## End-Effector Pose Debug Visualization

File:

- `legged_gym/envs/manip_loco/manip_loco.py`

Changes:

- Added a local `ThickAxesGeometry` line geometry to make EE orientation axes easier to see in Isaac Gym viewer.
- In `_draw_ee_goal_curr`, draw thick target EE axes at `curr_ee_goal_cart_world` using `ee_goal_orn_quat`.
- Also draw thick current EE axes at the current gripper pose using `ee_orn`.

Purpose:

- When running `legged_gym/scripts/play.py`, visually compare whether the current EE frame is aligned with the target EE frame.

## 6D End-Effector Tracking Training

Files:

- `legged_gym/envs/manip_loco/manip_loco.py`
- `legged_gym/envs/manip_loco/b1z1_config.py`
- `legged_gym/envs/rewards/maniploco_rewards.py`

Changes:

- Replaced the zeroed target-orientation observation with the sampled orientation command:

```python
self.ee_goal_orn_delta_rpy
```

- Enabled orientation tracking reward:

```python
tracking_ee_orn = 0.4
```

- Reimplemented `_reward_tracking_ee_orn` by converting the current target quaternion and current EE quaternion to Euler angles, then computing wrapped Euler error.

Purpose:

- Train a low-level policy that observes and is rewarded for both end-effector position and orientation tracking.
- Keep the observation dimensionality unchanged by replacing the existing 3 zero orientation channels with the 3D orientation command.
