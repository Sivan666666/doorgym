#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
SIM_ENV=${SIM_ENV:-txc_vbc}
BASE_SEED=${BASE_SEED:-615455575}
ARCHIVE=${ARCHIVE:-/mnt/sda/ps_archive/doorgym}
TAG=${TAG:-0829_wc_nomenkuang_fixed15_gripper080_rgb_v1}
PIPE="$ARCHIVE/high-level/logs/rgb_pi05_pipeline/$TAG"
TRAIN_ROOT="$ARCHIVE/high-level/dp/logs/lerobot-train-pi05"
EVAL_ROOT="$ARCHIVE/high-level/logs/door-policy-success"

RUN_NAMES=(
  pi05_pull_fixed15_g080_depth50_joint9_seed1000_30k
  pi05_pull_fixed15_g080_depth200_joint9_seed1000_30k
  pi05_pull_fixed15_g080_rgb50_joint9_seed1000_30k
  pi05_pull_fixed15_g080_rgb200_joint9_seed1000_30k
)

cd "$REPO"
mkdir -p "$PIPE" "$EVAL_ROOT"

stage() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$PIPE/eval_status.txt"
}

eval_one() {
  local gpu=$1 index=$2
  local run=${RUN_NAMES[$index]}
  local checkpoint="$TRAIN_ROOT/$run/checkpoints/030000"
  local out="$EVAL_ROOT/${run}_pull_open60_traverse080_seed${BASE_SEED}"
  local vision=(--depth_only)
  local rgb_extra=()
  if (( index >= 2 )); then
    vision=(--rgb)
    rgb_extra=(--dp_rgb_without_masks)
  fi
  [[ -f "$checkpoint/pretrained_model/model.safetensors" ]] || {
    echo "Missing 30K checkpoint: $checkpoint" >&2
    return 1
  }
  rm -rf "$out"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DOOR_AUTO_WRAP_ROOT="$ARCHIVE/high-level/dp/logs/door-auto-wrapped" \
    "$CONDA" run --no-capture-output -n "$SIM_ENV" python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "$checkpoint" --mode ikpull --robot_body a2wz1 \
    --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
    --num_envs 16 --total_trials 64 --steps 2600 --base_seed "$BASE_SEED" \
    --pass_open_angle_deg 60 --pull_traversal_distance_m 0.8 \
    --dp_action_horizon 25 --dp_inference_steps 10 --dp_fps 25 \
    --headless --parallel_batches 1 --run_root "$out" "${vision[@]}" -- \
    --door_name wc_nomenkuang \
    --wc4_disable_door_open_resistance --door_auto_open_force 0 \
    --gripper_close_ratio 0.8 --robot_pitch 0 \
    --ikpush_robot_pitch_rand_min 0 --ikpush_robot_pitch_rand_max 0 \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur --no_show_seg "${rgb_extra[@]}" \
    >"$PIPE/eval_${index}_${run}.log" 2>&1
}

stage EVAL_PI05_4WAY_RESTARTED
pids=()
for index in 0 1 2 3; do
  eval_one "$index" "$index" &
  pids+=("$!")
done

failed=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    echo "Evaluation failed: ${RUN_NAMES[$index]}" >&2
    failed=1
  fi
done
(( failed == 0 )) || { stage EVAL_PI05_4WAY_FAILED; exit 1; }

stage SUMMARIZE_PI05_4WAY
"$CONDA" run --no-capture-output -n "$SIM_ENV" python - \
  "$EVAL_ROOT" "$BASE_SEED" "${RUN_NAMES[@]}" <<'PY' | tee "$PIPE/final_results.txt"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = sys.argv[2]
for run in sys.argv[3:]:
    path = root / f"{run}_pull_open60_traverse080_seed{seed}" / "summary.json"
    summary = json.load(path.open())
    print(run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
stage COMPLETE
