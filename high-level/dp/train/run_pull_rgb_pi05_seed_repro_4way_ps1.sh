#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
TRAIN_ENV=${TRAIN_ENV:-b1z1_lerobot}
SIM_ENV=${SIM_ENV:-txc_vbc}
ARCHIVE=${ARCHIVE:-/mnt/sda/ps_archive/doorgym}
BASE_SEED=${BASE_SEED:-615455575}
PIPE="$ARCHIVE/high-level/logs/rgb_pi05_seed_repro/0831_pull_fixed15_g080_rgb_repro_v1"
TRAIN_ROOT="$ARCHIVE/high-level/dp/logs/lerobot-train-pi05"
EVAL_ROOT="$ARCHIVE/high-level/logs/door-policy-success"
WRAP_ROOT="$ARCHIVE/high-level/dp/logs/door-auto-wrapped"
PRETRAINED="$REPO/high-level/dp/models/pi05_base_cached_localtokenizer"

RGB50="local/door_a2w_0829_wc_nomenkuang_fixed15_gripper080_rgb_v1_50_joint9"
RGB200="local/door_a2w_0829_wc_nomenkuang_fixed15_gripper080_rgb_v1_200_joint9"
RGB50_ROOT="$ARCHIVE/high-level/data/lerobot/$RGB50"
RGB200_ROOT="$ARCHIVE/high-level/data/lerobot/$RGB200"

RUN_NAMES=(
  pi05_pull_fixed15_g080_rgb50_joint9_seed1000_repro2_30k
  pi05_pull_fixed15_g080_rgb50_joint9_seed2000_30k
  pi05_pull_fixed15_g080_rgb200_joint9_seed1000_repro2_30k
  pi05_pull_fixed15_g080_rgb200_joint9_seed2000_30k
)
DATASETS=("$RGB50" "$RGB50" "$RGB200" "$RGB200")
DATA_ROOTS=("$RGB50_ROOT" "$RGB50_ROOT" "$RGB200_ROOT" "$RGB200_ROOT")
SEEDS=(1000 2000 1000 2000)

cd "$REPO"
mkdir -p "$PIPE" "$TRAIN_ROOT" "$EVAL_ROOT" "$WRAP_ROOT"
exec >>"$PIPE/pipeline.log" 2>&1

stage() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$PIPE/status.txt"
}

fail() {
  stage "FAILED: $*"
  exit 1
}

for root in "$RGB50_ROOT" "$RGB200_ROOT"; do
  [[ -f "$root/meta/info.json" ]] || fail "missing dataset: $root"
done

train_one() {
  local gpu=$1 index=$2
  local run=${RUN_NAMES[$index]}
  local dataset=${DATASETS[$index]}
  local data_root=${DATA_ROOTS[$index]}
  local seed=${SEEDS[$index]}
  local out="$TRAIN_ROOT/$run"
  [[ ! -e "$out" ]] || fail "refusing to overwrite existing run: $out"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$CONDA" run --no-capture-output -n "$TRAIN_ENV" \
    accelerate launch --num-processes=1 --mixed_precision=bf16 \
    /home/ps/miniconda3/envs/b1z1_lerobot/bin/lerobot-train \
    --dataset.root="$data_root" --dataset.repo_id="$dataset" \
    --dataset.video_backend=torchcodec --dataset.use_imagenet_stats=false \
    --policy.type=pi05 --policy.device=cuda --policy.pretrained_path="$PRETRAINED" \
    --policy.dtype=bfloat16 --policy.push_to_hub=false --policy.gradient_checkpointing=true \
    --policy.chunk_size=100 --policy.n_action_steps=50 --policy.num_inference_steps=10 \
    --batch_size=16 --steps=30000 --num_workers=4 --save_freq=10000 --log_freq=50 \
    --seed="$seed" --job_name="$run" --output_dir="$out" \
    --wandb.enable=false --wandb.disable_artifact=true \
    >"$PIPE/train_${index}_${run}.log" 2>&1
}

stage TRAIN_PI05_RGB_REPRO_4WAY_START
pids=()
for index in 0 1 2 3; do
  train_one "$index" "$index" &
  pids+=("$!")
done
failed=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    echo "Training failed: ${RUN_NAMES[$index]}" >&2
    failed=1
  fi
done
(( failed == 0 )) || fail "one or more PI0.5 training jobs failed"

for run in "${RUN_NAMES[@]}"; do
  [[ -f "$TRAIN_ROOT/$run/checkpoints/030000/pretrained_model/model.safetensors" ]] \
    || fail "missing 30K checkpoint: $run"
done

eval_one() {
  local gpu=$1 index=$2
  local run=${RUN_NAMES[$index]}
  local checkpoint="$TRAIN_ROOT/$run/checkpoints/030000"
  local out="$EVAL_ROOT/${run}_pull_open60_traverse080_seed${BASE_SEED}"
  rm -rf "$out"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DOOR_AUTO_WRAP_ROOT="$WRAP_ROOT" \
    "$CONDA" run --no-capture-output -n "$SIM_ENV" python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "$checkpoint" --mode ikpull --robot_body a2wz1 \
    --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
    --num_envs 16 --total_trials 64 --steps 2600 --base_seed "$BASE_SEED" \
    --pass_open_angle_deg 60 --pull_traversal_distance_m 0.8 \
    --dp_action_horizon 25 --dp_inference_steps 10 --dp_fps 25 \
    --headless --parallel_batches 1 --run_root "$out" --rgb -- \
    --door_name wc_nomenkuang \
    --wc4_disable_door_open_resistance --door_auto_open_force 0 \
    --gripper_close_ratio 0.8 --robot_pitch 0 \
    --ikpush_robot_pitch_rand_min 0 --ikpush_robot_pitch_rand_max 0 \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur --no_show_seg --dp_rgb_without_masks \
    >"$PIPE/eval_${index}_${run}.log" 2>&1
}

stage EVAL_PI05_RGB_REPRO_4WAY_START
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
(( failed == 0 )) || fail "one or more PI0.5 evaluations failed"

stage SUMMARIZE_PI05_RGB_REPRO_4WAY
"$CONDA" run --no-capture-output -n "$SIM_ENV" python - \
  "$EVAL_ROOT" "$BASE_SEED" "${RUN_NAMES[@]}" <<'PY' | tee "$PIPE/final_results.txt"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = sys.argv[2]
for run in sys.argv[3:]:
    summary = json.load((root / f"{run}_pull_open60_traverse080_seed{seed}" / "summary.json").open())
    batches = [
        (b["door_open_successes"], b["traversal_successes"], b["successes"])
        for b in summary["batch_logs"]
    ]
    print(
        f"{run}\tbatches={batches}"
        f"\topen={summary['door_open_successes']}/64={100*summary['door_open_success_rate']:.2f}%"
        f"\ttraversal={summary['traversal_successes']}/64={100*summary['traversal_success_rate']:.2f}%"
        f"\tcombined={summary['successes']}/64={100*summary['success_rate']:.2f}%"
    )
PY
stage COMPLETE
