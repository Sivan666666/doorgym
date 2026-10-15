#!/usr/bin/env bash
set -Eeuo pipefail

# PS1 pipeline for the validated glass-door skill:
#   50 successful Joint9 demonstrations (Gym Jacobian expert)
#   -> one LeRobot depth dataset
#   -> Baseline ACT and Plucker ACT FOV55 (50K each).
#
# glass_door has a fixed pull handle and only one articulated DOF (door hinge),
# so the lever-door handle_progress supervision is intentionally not fabricated.

ROOT=${ROOT:-/home/ps/workspace/txc/doorgym}
SIM_PYTHON=${SIM_PYTHON:-/home/ps/miniconda3/envs/txc_vbc/bin/python}
TRAIN_PYTHON=${TRAIN_PYTHON:-/home/ps/miniconda3/envs/b1z1_lerobot/bin/python}
SIM_GPU=${SIM_GPU:-0}
GRAPHICS_DEVICE_ID=${GRAPHICS_DEVICE_ID:-$SIM_GPU}
BASELINE_GPU=${BASELINE_GPU:-1}
PLUCKER_GPU=${PLUCKER_GPU:-2}
GPU_START_MAX_MEMORY_MIB=${GPU_START_MAX_MEMORY_MIB:-20000}
PREPROCESS_ONLY=${PREPROCESS_ONLY:-0}

NUM_EPISODES=${NUM_EPISODES:-50}
MAX_QUOTA_ROLLOUTS=${MAX_QUOTA_ROLLOUTS:-100}
RECORD_SEED=${RECORD_SEED:-824731906}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-1100}
DOOR_NAME=${DOOR_NAME:-glass_door}
SKILL_PROGRAM_JSON=${SKILL_PROGRAM_JSON:-$ROOT/high-level/float_ik/door_twin/examples/glass_door_push_traverse_skill.json}
RAW_NAME=${RAW_NAME:-a2w_glass_door_jointstate9_gymjacobian_skill_heavier_graspyminus002_50_step1100}
DATASET_ID=${DATASET_ID:-local/door_a2w_glass_door_jointstate9_gymjacobian_skill_heavier_50}
RAW_ROOT="$ROOT/high-level/data/door_dp_raw/$RAW_NAME"
DATASET_ROOT="$ROOT/high-level/data/lerobot/$DATASET_ID"
LOG_ROOT=${LOG_ROOT:-$ROOT/high-level/logs/glass_door_joint9_skill50_pipeline_20260808}

BASELINE_RUN=${BASELINE_RUN:-leroact_a2w_glass_door_joint9_skill50_baseline_seed1000_50k_chunk100_exec50_bs16}
PLUCKER_RUN=${PLUCKER_RUN:-leroact_a2w_glass_door_joint9_skill50_plucker_fov55_seed1000_50k_chunk100_exec50_bs16}
BASELINE_OUTPUT="$ROOT/high-level/dp/logs/lerobot-train/$BASELINE_RUN"
PLUCKER_OUTPUT="$ROOT/high-level/dp/logs/lerobot-train/$PLUCKER_RUN"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
fail() { echo "[$(timestamp)] PIPELINE_ERROR: $*" >&2; exit 1; }
trap 'fail "line=$LINENO status=$?"' ERR

gpu_memory_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

wait_for_gpu() {
  local gpu=$1
  local purpose=$2
  while true; do
    local used
    used=$(gpu_memory_mib "$gpu")
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < GPU_START_MAX_MEMORY_MIB )); then
      echo "[$(timestamp)] GPU $gpu is ready for $purpose (${used} MiB used)."
      return
    fi
    echo "[$(timestamp)] GPU $gpu busy for $purpose (${used} MiB used); waiting."
    sleep 60
  done
}

mkdir -p "$RAW_ROOT" "$LOG_ROOT"
cd "$ROOT"

dataset_ready=0
if [[ -f "$DATASET_ROOT/meta/info.json" ]]; then
  dataset_ready=1
elif [[ -e "$DATASET_ROOT" ]]; then
  fail "Dataset directory exists but is incomplete: $DATASET_ROOT"
fi
if (( dataset_ready )); then
  echo "[$(timestamp)] Stages 1-3/5: reusing existing LeRobot dataset $DATASET_ROOT."
else
  existing_raw_count=$(find "$RAW_ROOT" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l)
  if (( existing_raw_count == NUM_EPISODES )); then
    echo "[$(timestamp)] Stage 1/5: reusing $existing_raw_count existing raw episodes."
  else
    echo "[$(timestamp)] Stage 1/5: recording $NUM_EPISODES successful glass_door episodes (currently $existing_raw_count)."
    wait_for_gpu "$SIM_GPU" recording
    CUDA_VISIBLE_DEVICES="$SIM_GPU" "$SIM_PYTHON" \
    high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
    --mode ikpush \
    --num_episodes "$NUM_EPISODES" \
    --num_envs 16 \
    --max_quota_rollouts "$MAX_QUOTA_ROLLOUTS" \
    --raw_root "$RAW_ROOT" \
    --state_action_mode joint_state9 \
    --steps "$ROLLOUT_STEPS" \
    --seed "$RECORD_SEED" \
    --headless \
    --graphics_device_id "$GRAPHICS_DEVICE_ID" \
    --rl_device cuda:0 \
    --sim_device cuda:0 \
    --fps 25 \
    --camera_fps 25 \
    --depth_only \
    --camera_depth_clip_lower 0.2 \
    --camera_depth_clip_far 1.5 \
    --camera_intrinsics_mode legacy \
    --camera_render_horizontal_fov_deg 55.0 \
    --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization \
    --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 \
    --record_camera_pose \
    --record_handle_bbox \
    --record_gripper_handle_contact \
    --keyframe_loss_weight 3 \
    --keyframe_loss_radius 3 \
    -- \
    --door_cfg "$ROOT/high-level/data/cfg/b1z1_opendoor.yaml" \
    --door_name "$DOOR_NAME" \
    --skill_program_json "$SKILL_PROGRAM_JSON" \
    --arm_ik_solver gym_jacobian \
    --ee_pose_frame robot_base_full \
      2>&1 | tee "$LOG_ROOT/record.log"
  fi

  echo "[$(timestamp)] Stage 2/5: validating raw data."
  "$TRAIN_PYTHON" - "$RAW_ROOT" "$NUM_EPISODES" <<'PY'
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
num_episodes = int(sys.argv[2])
paths = sorted(root.glob("episode_*.npz"))
expected = [f"episode_{i:06d}.npz" for i in range(num_episodes)]
assert [p.name for p in paths] == expected, f"expected {num_episodes} contiguous episodes, found {len(paths)}"
required = {
    "state", "action", "front_masked_depth", "wrist_masked_depth",
    "front_camera_pose_base", "wrist_camera_pose_base",
    "gripper_handle_contact_any", "replay_door_dof_pos",
}
total_frames = 0
for path in paths:
    with np.load(path, allow_pickle=True) as data:
        missing = required.difference(data.files)
        assert not missing, f"{path.name}: missing {sorted(missing)}"
        state = data["state"]
        action = data["action"]
        assert state.ndim == 2 and state.shape[1] == 9, (path.name, state.shape)
        assert action.shape == state.shape, (path.name, action.shape, state.shape)
        assert data["front_masked_depth"].shape[0] == state.shape[0]
        assert data["wrist_masked_depth"].shape[0] == state.shape[0]
        assert np.isfinite(state).all() and np.isfinite(action).all(), path.name
        total_frames += state.shape[0]
print(f"RAW_VALIDATION_OK episodes={len(paths)} frames={total_frames} state_dim=9 action_dim=9")
PY

  echo "[$(timestamp)] Stage 3/5: converting to LeRobot."
  "$TRAIN_PYTHON" high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root "$RAW_ROOT" \
  --root "$ROOT/high-level/data/lerobot" \
  --repo_id "$DATASET_ID" \
  --fps 25 \
  --depth_only \
  --image_storage video \
  --video_codec h264 \
  --num_workers 4 \
  --state_preprocess none \
  --action_preprocess none \
  --near_zero_rate_eps 1e-5 \
    2>&1 | tee "$LOG_ROOT/convert.log"
fi

echo "[$(timestamp)] Stage 4/5: validating LeRobot dataset."
"$TRAIN_PYTHON" - "$DATASET_ROOT/meta/info.json" "$NUM_EPISODES" <<'PY'
import json
from pathlib import Path
import sys

info = json.loads(Path(sys.argv[1]).read_text())
num_episodes = int(sys.argv[2])
features = info["features"]
assert info["total_episodes"] == num_episodes, info["total_episodes"]
assert features["observation.state"]["shape"] == [9]
assert features["action"]["shape"] == [9]
for key in (
    "observation.images.front_masked_depth",
    "observation.images.wrist_masked_depth",
    "observation.camera_pose.front",
    "observation.camera_pose.wrist",
):
    assert key in features, key
print(f"LEROBOT_VALIDATION_OK episodes={num_episodes} frames={info['total_frames']} state_dim=9 action_dim=9")
PY

if (( PREPROCESS_ONLY )); then
  echo "[$(timestamp)] PREPROCESS_COMPLETE: recording/conversion/validation finished."
  exit 0
fi

train_variant() {
  local gpu=$1
  local run_name=$2
  local output_dir=$3
  shift 3
  wait_for_gpu "$gpu" "$run_name"
  CUDA_VISIBLE_DEVICES="$gpu" "$TRAIN_PYTHON" \
    high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id="$DATASET_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.use_imagenet_stats=false \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=100 \
    --policy.n_action_steps=50 \
    --policy.vision_backbone=resnet18 \
    --policy.optimizer_lr=1e-5 \
    --policy.use_action_loss_weight=false \
    --policy.end_signal_prediction=false \
    --policy.interaction_state_conditioning=false \
    "$@" \
    --seed=1000 \
    --steps=50000 \
    --batch_size=16 \
    --num_workers=3 \
    --save_freq=10000 \
    --log_freq=100 \
    --keyframe_sampling_ratio=0 \
    --output_dir="$output_dir" \
    --job_name="$run_name" \
    --wandb.enable=true \
    --wandb.project=door-act \
    --wandb.disable_artifact=true
}

run_or_wait_variant() {
  local gpu=$1
  local run_name=$2
  local output_dir=$3
  shift 3
  local marker="$output_dir/checkpoints/050000/pretrained_model/config.json"
  if [[ -f "$marker" ]]; then
    echo "[$(timestamp)] Reusing completed training: $run_name"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "[$(timestamp)] Training already exists; waiting for external run: $run_name"
    while [[ ! -f "$marker" ]]; do
      if ! pgrep -f -- "--output_dir=$output_dir" >/dev/null; then
        fail "Incomplete training directory has no live process: $output_dir"
      fi
      sleep 60
    done
    echo "[$(timestamp)] External training completed: $run_name"
    return 0
  fi
  train_variant "$gpu" "$run_name" "$output_dir" "$@"
}

echo "[$(timestamp)] Stage 5/5: starting two 50K ACT trainings."
(
  set -o pipefail
  run_or_wait_variant "$BASELINE_GPU" "$BASELINE_RUN" "$BASELINE_OUTPUT" \
    --policy.plucker_conditioning=false \
    2>&1 | tee "$LOG_ROOT/train_${BASELINE_RUN}.log"
) &
baseline_pid=$!

(
  set -o pipefail
  run_or_wait_variant "$PLUCKER_GPU" "$PLUCKER_RUN" "$PLUCKER_OUTPUT" \
    --policy.plucker_conditioning=true \
    --policy.plucker_intrinsics_mode=legacy_shared_fov \
    --policy.plucker_horizontal_fov_deg=55.0 \
    2>&1 | tee "$LOG_ROOT/train_${PLUCKER_RUN}.log"
) &
plucker_pid=$!

echo "[$(timestamp)] Training PIDs: baseline=$baseline_pid plucker=$plucker_pid"
baseline_status=0
plucker_status=0
wait "$baseline_pid" || baseline_status=$?
wait "$plucker_pid" || plucker_status=$?
(( baseline_status == 0 )) || fail "Baseline training failed with status $baseline_status"
(( plucker_status == 0 )) || fail "Plucker training failed with status $plucker_status"

for checkpoint in \
  "$BASELINE_OUTPUT/checkpoints/050000/pretrained_model/config.json" \
  "$PLUCKER_OUTPUT/checkpoints/050000/pretrained_model/config.json"; do
  [[ -f "$checkpoint" ]] || fail "Missing 50K checkpoint marker: $checkpoint"
done

echo "[$(timestamp)] PIPELINE_COMPLETE"
echo "[$(timestamp)] Baseline: $BASELINE_OUTPUT/checkpoints/050000"
echo "[$(timestamp)] Plucker:  $PLUCKER_OUTPUT/checkpoints/050000"
