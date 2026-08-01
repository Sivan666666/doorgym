#!/usr/bin/env bash
set -Eeuo pipefail
REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
EVAL="$REPO/high-level/dp/eval/eval_door_dp_on_expert_obs.py"
RAW="$REPO/high-level/data/door_dp_raw/a2w_wc4_jointstate9_gymjacobian_200"
OUT="$REPO/high-level/dp/result/pull_50_200_push_comparison_20260731/offline_expert_obs"
LOG="$REPO/high-level/logs/pull-data-pipeline/pull_scaling_diagnosis_20260731/offline_expert_obs"
WRAP="$REPO/high-level/dp/logs/door-auto-wrapped"
mkdir -p "$OUT" "$LOG"
run_one() {
  local name="$1" checkpoint="$2"
  CUDA_VISIBLE_DEVICES=3 "$CONDA" run --no-capture-output -n b1z1_lerobot python "$EVAL" \
    --raw_root "$RAW" --episode_stride 10 --max_episodes 20 \
    --checkpoint "$checkpoint" --device cuda:0 --depth_only \
    --action_horizon 25 --compare_horizon 25 --stride 25 --eval_batch_size 16 --seed 615455575 \
    --output_json "$OUT/$name.json" >"$LOG/$name.log" 2>&1
}
run_one push200_baseline50k "$WRAP/leroact_a2w_wc4_jointstate9_gymjacobian_200_baseline_seed1000_50k_chunk100_exec50_bs16_0719/050000/model_latest.pt"
run_one push200_plucker_interaction50k "$WRAP/leroact_a2w_wc4_jointstate9_gymjacobian_200_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0719/050000/model_latest.pt"
printf 'COMPLETE\n' >"$LOG/push_status.txt"
