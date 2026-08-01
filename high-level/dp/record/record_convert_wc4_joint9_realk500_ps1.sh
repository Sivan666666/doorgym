#!/usr/bin/env bash
set -euo pipefail

# PS1-only reproducible pipeline:
#   existing real-K WC4 Joint9 episodes 0..199
#   + newly recorded episodes 200..499
#   -> one 500-episode LeRobot dataset with interaction-state labels.

REPO=/home/ps/workspace/txc/doorgym
PY_RECORD=/home/ps/miniconda3/envs/txc_vbc/bin/python
PY_CONVERT=/home/ps/miniconda3/envs/b1z1_lerobot/bin/python
GPU=${GPU:-3}

SOURCE_RAW="$REPO/high-level/data/door_dp_raw/a2w_wc4_jointstate9_gymjacobian_realK_render60_200"
TARGET_RAW="$REPO/high-level/data/door_dp_raw/a2w_wc4_jointstate9_gymjacobian_realK_render60_500"
REPO_ID=local/door_a2w_wc4_jointstate9_gymjacobian_realK_render60_500_interaction_state
TARGET_DATASET="$REPO/high-level/data/lerobot/$REPO_ID"
LOG_ROOT="$REPO/high-level/logs/realk500"
PIPELINE_LOG="$LOG_ROOT/pipeline.log"
RECORD_LOG="$LOG_ROOT/record_wc4_joint9_realK_render60_500.log"
CONVERT_LOG="$LOG_ROOT/convert_wc4_joint9_realK_render60_500.log"

mkdir -p "$LOG_ROOT"
exec >>"$PIPELINE_LOG" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

episode_count() {
  find "$1" -maxdepth 1 -name 'episode_*.npz' | wc -l
}

echo "[$(timestamp)] Pipeline started."
echo "[$(timestamp)] Waiting for physical GPU $GPU to become idle."
while true; do
  memory_used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$memory_used" =~ ^[0-9]+$ ]] && (( memory_used < 2000 )); then
    break
  fi
  echo "[$(timestamp)] GPU $GPU still busy: ${memory_used} MiB used."
  sleep 60
done
echo "[$(timestamp)] GPU $GPU is idle; starting data pipeline."

cd "$REPO"

source_count=$(episode_count "$SOURCE_RAW")
if [[ "$source_count" -ne 200 ]]; then
  echo "Expected exactly 200 source episodes, found $source_count in $SOURCE_RAW" >&2
  exit 1
fi

mkdir -p "$TARGET_RAW"
for index in $(seq 0 199); do
  name=$(printf 'episode_%06d.npz' "$index")
  if [[ ! -e "$TARGET_RAW/$name" ]]; then
    ln "$SOURCE_RAW/$name" "$TARGET_RAW/$name"
  fi
done
if [[ -f "$SOURCE_RAW/door_dp_feature_names.json" && ! -e "$TARGET_RAW/door_dp_feature_names.json" ]]; then
  cp "$SOURCE_RAW/door_dp_feature_names.json" "$TARGET_RAW/door_dp_feature_names.json"
fi

prefill_count=$(episode_count "$TARGET_RAW")
if (( prefill_count < 200 || prefill_count > 500 )); then
  echo "Unexpected target raw count after prefill: $prefill_count" >&2
  exit 1
fi
echo "[$(timestamp)] Target raw prefilled with $prefill_count episode(s)."

if (( prefill_count < 500 )); then
  echo "[$(timestamp)] Recording until exactly 500 successful episodes (append seed=824731906)."
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_RECORD" \
    high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
    --num_episodes 500 \
    --num_envs 16 \
    --raw_root "$TARGET_RAW" \
    --state_action_mode joint_state9 \
    --steps 1000 \
    --seed 824731906 \
    --headless \
    --graphics_device_id 0 \
    --rl_device cuda:0 \
    --sim_device cuda:0 \
    --depth_only \
    --camera_depth_clip_lower 0.2 \
    --camera_depth_clip_far 1.5 \
    --camera_fps 25 \
    --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization \
    --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 \
    --keyframe_loss_weight 3 \
    --keyframe_loss_radius 3 \
    --record_camera_pose \
    --record_handle_bbox \
    --record_gripper_handle_contact \
    --camera_intrinsics_mode real_k_remap \
    --camera_intrinsics_config high-level/data/cfg/a2w_real_camera_intrinsics_640x480.yaml \
    --camera_render_horizontal_fov_deg 60 \
    -- \
    --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
    --door_name wc4 \
    --arm_ik_solver gym_jacobian \
    --ee_pose_frame robot_base_full \
    2>&1 | tee "$RECORD_LOG"
else
  echo "[$(timestamp)] Raw target already contains 500 episodes; skipping recording."
fi

echo "[$(timestamp)] Validating raw episodes."
"$PY_CONVERT" - "$TARGET_RAW" <<'PY'
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
files = sorted(root.glob("episode_*.npz"))
expected_names = [f"episode_{i:06d}.npz" for i in range(500)]
actual_names = [p.name for p in files]
if actual_names != expected_names:
    missing = sorted(set(expected_names) - set(actual_names))
    extra = sorted(set(actual_names) - set(expected_names))
    raise RuntimeError(f"Episode sequence mismatch: missing={missing[:10]} extra={extra[:10]}")

total_frames = 0
required = {
    "state",
    "action",
    "front_masked_depth",
    "wrist_masked_depth",
    "front_camera_pose_base",
    "wrist_camera_pose_base",
    "replay_door_dof_pos",
    "gripper_handle_contact_both",
}
for path in files:
    with np.load(path, allow_pickle=True) as data:
        absent = required.difference(data.files)
        if absent:
            raise RuntimeError(f"{path.name}: missing fields {sorted(absent)}")
        state = np.asarray(data["state"])
        action = np.asarray(data["action"])
        front = np.asarray(data["front_masked_depth"])
        wrist = np.asarray(data["wrist_masked_depth"])
        if state.ndim != 2 or state.shape[1] != 9:
            raise RuntimeError(f"{path.name}: state shape={state.shape}, expected [T,9]")
        if action.shape != state.shape:
            raise RuntimeError(f"{path.name}: action shape={action.shape}, state shape={state.shape}")
        if front.shape[0] != state.shape[0] or front.shape[-2:] != (480, 640):
            raise RuntimeError(f"{path.name}: front depth shape={front.shape}")
        if wrist.shape[0] != state.shape[0] or wrist.shape[-2:] != (480, 640):
            raise RuntimeError(f"{path.name}: wrist depth shape={wrist.shape}")
        mode = str(np.asarray(data["camera_intrinsics_mode"]).item())
        if mode != "real_k_remap":
            raise RuntimeError(f"{path.name}: camera_intrinsics_mode={mode!r}")
        total_frames += state.shape[0]

if total_frames != 250_000:
    raise RuntimeError(f"Expected 250000 frames, found {total_frames}")
print(f"Raw validation passed: episodes={len(files)} frames={total_frames} state/action=9D")
PY

if [[ -e "$TARGET_DATASET" ]]; then
  echo "Refusing to overwrite existing LeRobot dataset: $TARGET_DATASET" >&2
  exit 1
fi

echo "[$(timestamp)] Converting raw episodes to LeRobot."
CUDA_VISIBLE_DEVICES='' "$PY_CONVERT" \
  high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root "$TARGET_RAW" \
  --root high-level/data/lerobot \
  --repo_id "$REPO_ID" \
  --state_action_mode joint_state9 \
  --depth_only \
  --image_storage video \
  --video_codec h264 \
  --num_workers 4 \
  --add_interaction_state \
  --interaction_contact_min_consecutive_frames 3 \
  --interaction_handle_unlock_angle_deg 40 \
  --interaction_door_goal_angle_deg 90 \
  2>&1 | tee "$CONVERT_LOG"

echo "[$(timestamp)] Validating LeRobot dataset."
"$PY_CONVERT" - "$TARGET_DATASET" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 500:
    raise RuntimeError(f"total_episodes={info.get('total_episodes')}, expected 500")
if info.get("total_frames") != 250_000:
    raise RuntimeError(f"total_frames={info.get('total_frames')}, expected 250000")
features = info.get("features", {})
for key in (
    "aux.interaction_contact",
    "aux.interaction_handle_progress",
    "aux.interaction_door_progress",
):
    if features.get(key, {}).get("shape") != [1]:
        raise RuntimeError(f"Missing or invalid feature {key}: {features.get(key)}")
if features.get("observation.state", {}).get("shape") != [9]:
    raise RuntimeError("observation.state is not 9D")
if features.get("action", {}).get("shape") != [9]:
    raise RuntimeError("action is not 9D")
print("LeRobot validation passed: episodes=500 frames=250000 state/action=9D interaction=3")
PY

echo "[$(timestamp)] PIPELINE COMPLETE"
echo "[$(timestamp)] Raw: $TARGET_RAW"
echo "[$(timestamp)] LeRobot: $TARGET_DATASET"
