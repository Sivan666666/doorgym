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
