#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
PY_TRAIN=/home/ps/miniconda3/envs/b1z1_lerobot/bin/python
PY_SIM=/home/ps/miniconda3/envs/txc_vbc/bin/python
GPU=${GPU:-1}
WAIT_PID=${WAIT_PID:-2971651}

DATASET_ID=local/door_a2w_wc4_jointstate9_gymjacobian_realK_render60_500_interaction_state
DATASET_ROOT="$REPO/high-level/data/lerobot/$DATASET_ID"
RUN_NAME=leroact_a2w_wc4_joint9_realK_render60_500_plucker_interaction_decoderchunk_w005_seed1000_100k_chunk100_exec50_bs16
OUTPUT_DIR="$REPO/high-level/dp/logs/lerobot-train/$RUN_NAME"
STEP_DIR="$OUTPUT_DIR/checkpoints/100000"
WRAP_ROOT="$REPO/high-level/dp/logs/door-auto-wrapped/$RUN_NAME/100000"
WRAP_DIR="$WRAP_ROOT/model_latest"
WRAP_MANIFEST="$WRAP_ROOT/model_latest.pt"
EVAL_ROOT="$REPO/high-level/logs/door-policy-success/wc4_joint9_realK_render60_500_plucker_interaction100k_seed615455575"
LOG_ROOT="$REPO/high-level/logs/realk500_100k"
PIPELINE_LOG="$LOG_ROOT/pipeline.log"
TRAIN_LOG="$LOG_ROOT/train.log"
EVAL_LOG="$LOG_ROOT/eval.log"

mkdir -p "$LOG_ROOT"
exec >>"$PIPELINE_LOG" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] 100K train/eval pipeline started."

if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "[$(timestamp)] Waiting for existing GPU $GPU evaluation PID $WAIT_PID."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
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

cd "$REPO"

"$PY_TRAIN" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json
import sys

info = json.load(open(sys.argv[1]))
assert info["total_episodes"] == 500, info["total_episodes"]
assert info["total_frames"] == 250000, info["total_frames"]
assert info["features"]["observation.state"]["shape"] == [9]
assert info["features"]["action"]["shape"] == [9]
for key in (
    "aux.interaction_contact",
    "aux.interaction_handle_progress",
    "aux.interaction_door_progress",
    "observation.camera_pose.front",
    "observation.camera_pose.wrist",
):
    assert key in info["features"], key
print("Dataset validation passed: real-K Joint9, 500 episodes, 250000 frames.")
PY

if [[ -e "$OUTPUT_DIR" && ! -e "$STEP_DIR" ]]; then
  echo "Refusing to overwrite partial output directory: $OUTPUT_DIR" >&2
  exit 1
fi

if [[ ! -e "$STEP_DIR" ]]; then
  echo "[$(timestamp)] Starting from-scratch 100K training on physical GPU $GPU."
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_TRAIN" \
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
    --steps=100000 \
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
else
  echo "[$(timestamp)] 100K checkpoint already exists; skipping training."
fi

if [[ ! -e "$STEP_DIR/pretrained_model/config.json" ]]; then
  echo "Invalid/missing 100K checkpoint: $STEP_DIR" >&2
  exit 1
fi
echo "[$(timestamp)] 100K training complete."

if [[ ! -e "$WRAP_MANIFEST" ]]; then
  echo "[$(timestamp)] Exporting official checkpoint to Door wrapper."
  export PYTHONPATH="$REPO/high-level/lerobot/src:$REPO/high-level/dp${PYTHONPATH:+:$PYTHONPATH}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_TRAIN" \
    high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
    --official_checkpoint "$STEP_DIR" \
    --root "$DATASET_ROOT" \
    --repo_id "$DATASET_ID" \
    --out_dir "$WRAP_DIR" \
    --manifest_name model_latest.pt \
    --device cuda:0 \
    --depth_only
fi

if [[ -e "$EVAL_ROOT/summary.json" ]]; then
  echo "Refusing to overwrite existing evaluation: $EVAL_ROOT" >&2
  exit 1
fi

echo "[$(timestamp)] Starting WC4 64-trial evaluation."
export DOOR_DP_LEROBOT_PYTHON="$PY_TRAIN"
CUDA_VISIBLE_DEVICES="$GPU" "$PY_SIM" \
  high-level/dp/eval/eval_door_policy_success.py \
  --checkpoint "$WRAP_MANIFEST" \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --mode ikpush \
  --robot_body a2wz1 \
  --num_envs 16 \
  --total_trials 64 \
  --steps 1000 \
  --pass_open_angle_deg 80 \
  --success_metric abs \
  --base_seed 615455575 \
  --headless \
  --graphics_device_id 0 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --dp_inference_steps 10 \
  --dp_noise_scheduler_type DDIM \
  --depth_only \
  --dp_action_horizon 25 \
  --dp_fps 25 \
  --no_enable_depth_noise \
  --no_enable_depth_gaussian_blur \
  --enable_depth_camera_randomization \
  --depth_camera_pos_rand_m 0.02 \
  --depth_camera_rot_rand_deg 5.0 \
  --run_root "$EVAL_ROOT" \
  --progress_interval 10 \
  -- \
  --door_name wc4 \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --camera_intrinsics_mode real_k_remap \
  --camera_intrinsics_config high-level/data/cfg/a2w_real_camera_intrinsics_640x480.yaml \
  --camera_render_horizontal_fov_deg 60 \
  --robot_pitch 0.0 \
  --ikpush_robot_pitch_rand_min 0.0 \
  --ikpush_robot_pitch_rand_max 0.0 \
  2>&1 | tee "$EVAL_LOG"

"$PY_TRAIN" - "$EVAL_ROOT/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
print(
    f"FINAL RESULT: {summary['successes']}/{summary['total_trials']} "
    f"= {100.0 * summary['success_rate']:.2f}%"
)
print("Batches:", [item["successes"] for item in summary["batch_logs"]])
PY
echo "[$(timestamp)] TRAIN+EVAL PIPELINE COMPLETE"
