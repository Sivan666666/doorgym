#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
EVAL="$REPO/high-level/dp/eval/eval_door_dp_on_expert_obs.py"
OUT="$REPO/high-level/dp/result/pull_50_200_push_comparison_20260731/offline_expert_obs"
LOG="$REPO/high-level/logs/pull-data-pipeline/pull_scaling_diagnosis_20260731/offline_expert_obs"
WRAP="$REPO/high-level/dp/logs/door-auto-wrapped"
mkdir -p "$OUT" "$LOG"

run_one() {
    local gpu="$1" name="$2" raw="$3" checkpoint="$4" episode_stride="$5"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python "$EVAL" \
        --raw_root "$raw" --episode_stride "$episode_stride" --max_episodes 20 \
        --checkpoint "$checkpoint" --device cuda:0 --depth_only \
        --action_horizon 25 --compare_horizon 25 --stride 25 --eval_batch_size 16 --seed 615455575 \
        --output_json "$OUT/$name.json" >"$LOG/$name.log" 2>&1
}

# GPU 2: Pull checkpoints.
run_one 2 pull50old_baseline \
    "$REPO/high-level/data/door_dp_raw/a2w_wc4_pull_joint9_noresistance_noisy_rand_50_seed615455575" \
    "$WRAP/leroact_a2w_wc4_pull_joint9_noisy_rand50_baseline_seed1000_50k_chunk100_exec50_bs16_0730_1609/050000/model_latest.pt" 2
run_one 2 pull50old_plucker_interaction \
    "$REPO/high-level/data/door_dp_raw/a2w_wc4_pull_joint9_noresistance_noisy_rand_50_seed615455575" \
    "$WRAP/leroact_a2w_wc4_pull_joint9_noisy_rand50_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0730_1609/050000/model_latest.pt" 2
run_one 2 pull200_baseline50k \
    "$REPO/high-level/data/door_dp_raw/a2w_wc4_pull_joint9_noresistance_noisy_rand_200_seed615455575" \
    "$WRAP/leroact_a2w_wc4_pull_joint9_noisy_rand200_baseline_seed1000_50k_chunk100_exec50_bs16_0731/050000/model_latest.pt" 10
run_one 2 pull200_plucker_interaction50k \
    "$REPO/high-level/data/door_dp_raw/a2w_wc4_pull_joint9_noresistance_noisy_rand_200_seed615455575" \
    "$WRAP/leroact_a2w_wc4_pull_joint9_noisy_rand200_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0731/050000/model_latest.pt" 10

printf 'COMPLETE\n' >"$LOG/status.txt"
