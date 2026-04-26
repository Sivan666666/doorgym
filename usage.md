# Usage

Detailed door-task commands and key prompts are recorded in:

- `high-level/usage.md`

Modification notes are recorded in:

- `high-level/modify.md`
- `low-level/modify.md`

Primary door asset play command:

```bash
cd high-level
python play_b1z1_walk_with_door_asset.py --rl_device cuda:0 --sim_device cuda:0 --num_envs 4 --steps 2000 --external_orn_gain 0.0
```

Door asset visualization command:

```bash
cd high-level
python play_door_locomanip_asset.py --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 --num_envs 4 --steps 240
```
