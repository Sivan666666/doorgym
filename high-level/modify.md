# High-Level Modify Notes

This file records local changes made on top of the original high-level code.

## Door Task Environment

Files:

- `envs/b1z1_opendoor.py`
- `envs/__init__.py`
- `data/cfg/b1z1_opendoor.yaml`
- `data/asset/door_set/`

Changes:

- Added `B1Z1OpenDoor`, a state-based high-level environment for lever-door loco-manipulation.
- Registered `B1Z1OpenDoor` in `envs/__init__.py`.
- Added lever-door task config and door assets.
- The door config includes lever-door assets, handle bounding boxes, door lock force, handle spring/damping, door opening resistance, and low-level policy path.

Purpose:

- Provide a high-level door-opening task using the pretrained B1Z1 low-level locomotion/manipulation policy.
- Support state-based teacher training and scripted validation of door interaction.

## Door Asset And Scripted Play Scripts

Files:

- `play_door_locomanip_asset.py`
- `play_door_locomanip.py`
- `play_door_locomanip_walk.py`
- `play_b1z1_walk_with_door.py`
- `play_b1z1_walk_with_door_asset.py`
- `play_b1z1_walk_only.py`

Changes:

- Added door asset visualization and scripted door-opening play entries.
- Added a low-level walking smoke test with door loaded.
- Added `play_b1z1_walk_only.py` to verify the pretrained low-level policy can walk and track random EE targets without the door.
- Added `play_b1z1_walk_with_door_asset.py`, a self-contained low-level play script that:
  - Uses `ManipLoco` from low-level as the robot/arm base.
  - Loads door assets from `b1z1_opendoor.yaml`.
  - Adds door lock force before handle threshold, handle spring return, and door opening resistance.
  - Arranges dog and door in a grid layout.
  - Stops the robot near the door.
  - Runs an external EE trajectory: approach handle, hold with gripper open, close gripper, rotate handle, pull door, hold.
  - Visualizes the active EE target as a red point.
  - Uses position-only or low-gain orientation IK for the external trajectory via `--external_orn_gain`.

## Training Entry And Utilities

Files:

- `train_multistate.py`
- `play_multistate.py`
- `envs/b1z1_base.py`
- `data/cfg/b1z1_pickmulti.yaml`
- `test_pointcloud.py`
- `README.md`

Changes:

- Set `TORCH_EXTENSIONS_DIR=/tmp/torch_extensions` before Isaac Gym/PyTorch extension loading.
- Fixed graphics device usage in `B1Z1Base.create_sim`.
- Printed the exact low-level policy path when loading a pretrained policy.
- Added CPU simulation fallback for high-level training.
- Added B1Z1OpenDoor-specific start pose handling in `train_multistate.py`.
- Made model feature dimensions read from the environment instead of hard-coded constants.
- Updated local low-level policy path in `b1z1_pickmulti.yaml`.
- Updated point-cloud test asset directory to the local object asset path.
- Extended `README.md` with door task notes and commands.

## Important Debug Findings

- The low-level policy was trained mainly for EE position tracking:
  - `tracking_ee_world = 0.8`
  - `tracking_ee_orn = 0.0`
  - target orientation observation is currently zeroed in low-level observations.
- Full 6D IK on the door handle can degrade red/blue point tracking because orientation error competes with position error.
- In `play_b1z1_walk_with_door_asset.py`, `--external_orn_gain` controls how much external door trajectory IK cares about target orientation:
  - `0.0`: position-only IK.
  - `1.0`: full orientation error.
  - small values such as `0.2-0.35`: weak orientation preference.
