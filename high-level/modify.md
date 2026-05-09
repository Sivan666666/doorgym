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
  - Synchronizes external door trajectory orientation into both `ee_goal_orn_quat` and `ee_goal_orn_delta_rpy`, so 6D low-level policies see the same orientation command in observations that the IK controller uses.
  - Adds `--gripper_red_axis_rot` to rotate the scripted gripper target around the EE red x axis. The default is `-pi/2`, which turns the gripper closing direction from left/right to up/down for lever-handle grasping.
  - Adds `--grasp_x_offset` and `--grasp_z_offset` to shift both the scripted pregrasp point and grasp point for handle alignment.
  - Applies `--grasp_z_offset` to the approach anchor as soon as manipulation starts, so the pregrasp approach stays at the same lowered height instead of dipping suddenly near the handle.
  - Reads door/handle damping, friction, effort, spring, and resistance values from `b1z1_opendoor.yaml` instead of using weak hard-coded defaults, reducing the tendency for the handle to jump to its limit after a light contact.
  - Adds `--gripper_stiffness` and `--gripper_damping`, and interpolates gripper closure over `--gripper_close_steps`, to reduce contact impulses when the gripper closes on the handle.
  - Adds `--gripper_close_ratio`, defaulting to `0.65`, so the Z1 gripper stops at a soft closed target instead of driving all the way to the tightest limit.
  - Adds play-time handle resistance overrides (`--handle_spring_stiffness`, `--handle_spring_damping`, `--handle_joint_friction`, `--handle_joint_damping`) so the handle can be made turnable for the softer B1Z1 gripper without returning to the unstable low-damping setting.
  - Overrides the viewer `F` key behavior for this play env so free-camera mode unlocks from the current camera pose instead of jumping back to the configured world-origin view.
  - Adds contact-resolution controls for door-handle debugging: higher door VHACD resolution and play-time PhysX overrides for substeps, solver iterations, contact/rest offset, and max depenetration velocity.
  - Enables VHACD for the B1Z1/Z1 robot asset in this play script and adds gripper-shape contact controls (`--robot_vhacd_resolution`, `--gripper_shape_contact_offset`, `--gripper_shape_rest_offset`, `--gripper_shape_friction`) to improve end-effector collision behavior near the door handle.
  - Adds `--gripper_joint_friction` and raises the default gripper stiffness/damping/contact friction toward the UniDoorManip Franka setup, while keeping them below the very stiff UniDoor values to reduce contact kickback.
  - Adds `--external_pos_gain` and prints `ee_pos` / `ee_z_error` in play logs to debug cases where the visual target moves but the actual EE does not follow in height.
  - During handle rotation, rotates the scripted EE target orientation around the EE red x axis together with the downward handle arc, and keeps that turned orientation during pull/hold.
  - Adds `--handle_unlock_ratio` and latches the door unlock state once the handle reaches the threshold, so the door does not immediately relock if the spring returns the handle before the door starts moving.
  - Decouples door-handle orientation following from position generation: pregrasp/grasp/rotate/pull positions are cached when manipulation starts, while later handle motion only changes the scripted EE orientation.
  - Changes the post-grasp handle-rotation position path to a UniDoorManip-style segment: after grasp, the red target moves in world `+Y/-Z` (right/down) by `--handle_rotate_right_distance` and `--handle_rotate_down_distance`, then starts the door pull from that point.
  - Changes the pull phase to a UniDoorManip-style incremental target by default (`--unidoor_style_pull`): each play step sets the target slightly ahead of the current EE along the live handle open direction, instead of interpolating along a cached straight line.
  - Adds play-time door hinge resistance overrides (`--door_open_resistance`, `--door_open_damping`, `--door_joint_friction`, `--door_joint_damping`) to separate door-resistance issues from gripper-contact issues during pull debugging.
  - Adds debug auto-open controls (`--door_auto_open_force`, `--door_auto_open_sign`, `--door_auto_open_target_ratio`) so the door can drive itself open after `door_open_stage=True`, isolating door-asset/hinge validity from gripper pulling quality.

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

- The original low-level policy was trained mainly for EE position tracking:
  - `tracking_ee_world = 0.8`
  - `tracking_ee_orn = 0.0`
  - target orientation observation was zeroed in low-level observations.
- For the newer 6D low-level policy, `play_b1z1_walk_with_door_asset.py` now updates `ee_goal_orn_delta_rpy` together with `ee_goal_orn_quat`; otherwise the policy observation would still contain the old/default orientation command.
- Full 6D IK on the door handle can degrade red/blue point tracking because orientation error competes with position error.
- In `play_b1z1_walk_with_door_asset.py`, `--external_orn_gain` controls how much external door trajectory IK cares about target orientation:
  - `0.0`: position-only IK.
  - `1.0`: full orientation error.
  - small values such as `0.2-0.5`: weak-to-medium orientation preference.
