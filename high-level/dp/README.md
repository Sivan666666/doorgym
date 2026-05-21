# Door DP Folder

This folder contains the reorganized Door Diffusion Policy pipeline.

## Main Scripts

- `record_door_dp_dataset.py`  
  Run scripted `pull`/`push` experts or the float-IK `ikpush` expert in Isaac Gym and save successful raw `.npz` episodes.

- `convert_door_raw_to_lerobot.py`  
  Convert raw `.npz` episodes into a local LeRobotDataset for visualization and training.

- `replay_door_dp_raw_in_isaacgym.py`  
  Replay a raw `.npz` episode in Isaac Gym. `ikpush` recordings are replayed in the float-IK scene; `pull`/`push` use the old play scenes.

- `train_door_dp.py`  
  Train the door policy from the converted LeRobotDataset using LeRobot's built-in `DiffusionPolicy`.

- `train_door_act.py` / `train_door_pi05.py`
  Train the same Door input/output contract with LeRobot `ACTPolicy` or `PI05Policy`.

- `play_door_policy.py`
  Load a trained Door policy checkpoint and run inference inside the scripted door environments. The backend is inferred from checkpoint metadata.

## Shared Code

- `door_dp_common.py`  
  Shared helpers for state/action formatting, raw recording, LeRobot recording, camera-image conversion, and policy inference utilities.

- `door_policy_backend.py` / `door_policy_worker.py`
  Modular LeRobot Diffusion/ACT/pi0.5 policy backends and Python>=3.10 subprocess worker for inference from Isaac Gym runtimes.

- `models/door_diffusion_policy.py`  
  Legacy custom model definition, kept for reference but no longer used by train/play/eval.

## Docs

- `docs/door_dp_usage.md`  
  End-to-end commands for scripted play, raw recording, conversion, visualization, training, and DP play.
