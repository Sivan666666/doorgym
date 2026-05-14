# Door DP Folder

This folder contains the reorganized Door Diffusion Policy pipeline.

## Main Scripts

- `record_door_dp_dataset.py`  
  Run scripted pull/push experts in Isaac Gym and save successful raw `.npz` episodes.

- `convert_door_raw_to_lerobot.py`  
  Convert raw `.npz` episodes into a local LeRobotDataset for visualization and training.

- `train_door_dp.py`  
  Train the door diffusion policy from the converted LeRobotDataset.

- `play_door_dp_policy.py`  
  Load a trained Door DP checkpoint and run inference inside the scripted door environments.

## Shared Code

- `door_dp_common.py`  
  Shared helpers for state/action formatting, raw recording, LeRobot recording, camera-image conversion, and DP inference utilities.

- `models/door_diffusion_policy.py`  
  The Door Diffusion Policy model definition.

## Docs

- `docs/door_dp_usage.md`  
  End-to-end commands for scripted play, raw recording, conversion, visualization, training, and DP play.
