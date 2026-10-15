#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/ps/workspace/txc/doorgym
RECORD_DIR="$ROOT/high-level/dp/record"
EVAL_DIR="$ROOT/high-level/dp/eval"
MASTER_LOG="$ROOT/high-level/logs/glass_door2_joint9_skill50_200_master_20260810"
mkdir -p "$MASTER_LOG"

SIM_GPU=2 BASELINE_GPU=2 PLUCKER_GPU=3 \
  "$RECORD_DIR/run_glass_door2_skill50_two_act_pipeline_ps1.sh" \
  >"$MASTER_LOG/pipeline_50.log" 2>&1

# GPU0/1 can be occupied by unrelated jobs. Reuse the known-free GPU2/3 only
# after the 50-episode pipeline has released them.
SIM_GPU=2 BASELINE_GPU=2 PLUCKER_GPU=3 \
  "$RECORD_DIR/run_glass_door2_skill200_two_act_pipeline_ps1.sh" \
  >"$MASTER_LOG/pipeline_200.log" 2>&1

BASE_GPU=2 PLK_GPU=3 "$EVAL_DIR/run_glass_door2_joint9_skill200_eval_ps1.sh" \
  >"$MASTER_LOG/eval_200.log" 2>&1
BASE_GPU=2 PLK_GPU=3 "$EVAL_DIR/run_glass_door2_joint9_skill50_eval_ps1.sh" \
  >"$MASTER_LOG/eval_50.log" 2>&1
echo "GLASS_DOOR2_ALL_COMPLETE"
