#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
PYTHON=${PYTHON:-/home/ps/miniconda3/envs/b1z1_lerobot/bin/python}
GPU=${GPU:?Set GPU to the physical GPU index}
VARIANT=${VARIANT:?Set VARIANT to baseline or plucker_interaction}

DATASET_ID=local/door_a2w_wc_articraft2_ours_state10_200_step1400
DATASET_ROOT="$REPO/high-level/data/lerobot/$DATASET_ID"

case "$VARIANT" in
  baseline)
    RUN_NAME=leroact_a2w_wc_articraft2_200_baseline_50k_chunk100_exec50_bs16_seed1000_0728
    EXTRA_POLICY_ARGS=(
      --policy.plucker_conditioning=false
      --policy.interaction_state_conditioning=false
    )
    ;;
  plucker_interaction)
    RUN_NAME=leroact_a2w_wc_articraft2_200_plucker_fov55_interaction_decoderchunk_w005_50k_chunk100_exec50_bs16_seed1000_0728
    EXTRA_POLICY_ARGS=(
      --policy.plucker_conditioning=true
      --policy.plucker_intrinsics_mode=legacy_shared_fov
      --policy.plucker_horizontal_fov_deg=55.0
      --policy.interaction_state_conditioning=true
      --policy.interaction_state_prediction_mode=decoder_chunk
      --policy.interaction_state_probe_only=false
      --policy.interaction_state_auxiliary_only=false
      --policy.interaction_contact_loss_weight=0.05
      --policy.interaction_handle_loss_weight=0.05
      --policy.interaction_door_loss_weight=0.05
    )
    ;;
  *)
    echo "Unsupported VARIANT=$VARIANT; expected baseline or plucker_interaction." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$REPO/high-level/dp/logs/lerobot-train/$RUN_NAME"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints/050000"
LOG_ROOT="$REPO/high-level/logs/wc_articraft2_training"
TRAIN_LOG="$LOG_ROOT/${RUN_NAME}.log"
mkdir -p "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

cd "$REPO"

"$PYTHON" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing dataset metadata: {path}")
info = json.loads(path.read_text())
features = info.get("features", {})
assert info.get("total_episodes") == 200, info.get("total_episodes")
assert info.get("total_frames") == 140000, info.get("total_frames")
assert info.get("fps") == 25, info.get("fps")
assert features["observation.state"]["shape"] == [10]
assert features["action"]["shape"] == [10]
for key in (
    "observation.images.front_masked_depth",
    "observation.images.wrist_masked_depth",
    "observation.camera_pose.front",
    "observation.camera_pose.wrist",
    "aux.interaction_contact",
    "aux.interaction_handle_progress",
    "aux.interaction_door_progress",
):
    assert key in features, key
print("Dataset validation passed: 200 episodes, 140000 frames, State10, dual depth, camera poses, interaction labels.")
PY

if [[ -e "$CHECKPOINT_DIR/pretrained_model/config.json" ]]; then
  echo "[$(timestamp)] 50K checkpoint already exists: $CHECKPOINT_DIR"
  exit 0
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite partial/existing output directory: $OUTPUT_DIR" >&2
  exit 1
fi

echo "[$(timestamp)] Waiting for physical GPU $GPU to become idle."
while true; do
  memory_used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  utilization=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$memory_used" =~ ^[0-9]+$ ]] && [[ "$utilization" =~ ^[0-9]+$ ]] \
      && (( memory_used < 2000 )) && (( utilization < 10 )); then
    break
  fi
  echo "[$(timestamp)] GPU $GPU busy: memory=${memory_used}MiB utilization=${utilization}%."
  sleep 60
done

echo "[$(timestamp)] Starting $VARIANT on physical GPU $GPU."
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
  --policy.end_signal_prediction=false \
  "${EXTRA_POLICY_ARGS[@]}" \
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

if [[ ! -e "$CHECKPOINT_DIR/pretrained_model/config.json" ]]; then
  echo "Training exited without a valid 50K checkpoint: $CHECKPOINT_DIR" >&2
  exit 1
fi
echo "[$(timestamp)] TRAINING COMPLETE: $CHECKPOINT_DIR"
