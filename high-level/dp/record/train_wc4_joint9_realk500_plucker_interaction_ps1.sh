#!/usr/bin/env bash
set -euo pipefail

# Wait for the real-K WC4 Joint9-500 conversion pipeline, then train the
# matching Plucker + Interaction decoder-chunk ACT from scratch on PS1.

REPO=/home/ps/workspace/txc/doorgym
PYTHON=/home/ps/miniconda3/envs/b1z1_lerobot/bin/python
GPU=${GPU:-1}
DATASET_ID=local/door_a2w_wc4_jointstate9_gymjacobian_realK_render60_500_interaction_state
DATASET_ROOT="$REPO/high-level/data/lerobot/$DATASET_ID"
PIPELINE_LOG="$REPO/high-level/logs/realk500/pipeline.log"
RUN_NAME=leroact_a2w_wc4_joint9_realK_render60_500_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16
OUTPUT_DIR="$REPO/high-level/dp/logs/lerobot-train/$RUN_NAME"
LOG_ROOT="$REPO/high-level/logs/realk500"
TRAIN_LOG="$LOG_ROOT/train_${RUN_NAME}.log"
QUEUE_LOG="$LOG_ROOT/train_queue.log"

mkdir -p "$LOG_ROOT"
exec >>"$QUEUE_LOG" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

dataset_is_complete() {
  "$PYTHON" - "$DATASET_ROOT/meta/info.json" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
info = json.loads(path.read_text())
features = info.get("features", {})
required = (
    "aux.interaction_contact",
    "aux.interaction_handle_progress",
    "aux.interaction_door_progress",
    "observation.camera_pose.front",
    "observation.camera_pose.wrist",
)
valid = (
    info.get("total_episodes") == 500
    and info.get("total_frames") == 250000
    and features.get("observation.state", {}).get("shape") == [9]
    and features.get("action", {}).get("shape") == [9]
    and all(key in features for key in required)
)
raise SystemExit(0 if valid else 1)
PY
}

echo "[$(timestamp)] Training queue started; waiting for validated LeRobot Joint9 real-K 500 dataset."
while true; do
  if grep -q 'PIPELINE COMPLETE' "$PIPELINE_LOG" 2>/dev/null && dataset_is_complete; then
    break
  fi
  echo "[$(timestamp)] Dataset conversion is not complete yet."
  sleep 120
done
echo "[$(timestamp)] Dataset is complete and validated."

if [[ -e "$OUTPUT_DIR/checkpoints/050000" ]]; then
  echo "[$(timestamp)] 50K checkpoint already exists; nothing to do: $OUTPUT_DIR/checkpoints/050000"
  exit 0
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite a partial/existing training directory: $OUTPUT_DIR" >&2
  exit 1
fi

echo "[$(timestamp)] Waiting for physical GPU $GPU to become idle."
while true; do
  memory_used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$memory_used" =~ ^[0-9]+$ ]] && (( memory_used < 2000 )); then
    break
  fi
  echo "[$(timestamp)] GPU $GPU busy: ${memory_used} MiB used."
  sleep 60
done

echo "[$(timestamp)] Starting $RUN_NAME on physical GPU $GPU."
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
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
  --policy.plucker_conditioning=true \
  --policy.plucker_intrinsics_mode=per_camera \
  --policy.plucker_front_fx=604.7375 \
  --policy.plucker_front_fy=603.8005 \
  --policy.plucker_front_cx=329.0904 \
  --policy.plucker_front_cy=240.8226 \
  --policy.plucker_wrist_fx=606.1074 \
  --policy.plucker_wrist_fy=605.9551 \
  --policy.plucker_wrist_cx=325.8074 \
  --policy.plucker_wrist_cy=259.1791 \
  --policy.interaction_state_conditioning=true \
  --policy.interaction_state_prediction_mode=decoder_chunk \
  --policy.interaction_state_probe_only=false \
  --policy.interaction_state_auxiliary_only=false \
  --policy.interaction_contact_loss_weight=0.05 \
  --policy.interaction_handle_loss_weight=0.05 \
  --policy.interaction_door_loss_weight=0.05 \
  --seed=1000 \
  --steps=50000 \
  --batch_size=16 \
  --num_workers=3 \
  --save_freq=10000 \
  --log_freq=100 \
  --keyframe_sampling_ratio=0 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  --wandb.enable=true \
  --wandb.project=door-act \
  --wandb.disable_artifact=true \
  2>&1 | tee "$TRAIN_LOG"

if [[ ! -e "$OUTPUT_DIR/checkpoints/050000" ]]; then
  echo "Training exited without a 50K checkpoint: $OUTPUT_DIR" >&2
  exit 1
fi
echo "[$(timestamp)] TRAINING COMPLETE: $OUTPUT_DIR/checkpoints/050000"
