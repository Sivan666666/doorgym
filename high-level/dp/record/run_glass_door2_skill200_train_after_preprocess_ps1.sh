#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/home/ps/workspace/txc/doorgym}
PREPROCESS_LOG=${PREPROCESS_LOG:-$ROOT/high-level/logs/glass_door2_joint9_skill200_preprocess_parallel_20260811}
TRAIN_LOG=${TRAIN_LOG:-$ROOT/high-level/logs/glass_door2_joint9_skill200_train_parallel_20260811}
mkdir -p "$TRAIN_LOG"

preprocess_pid=$(cat "$PREPROCESS_LOG/pid")
while kill -0 "$preprocess_pid" 2>/dev/null; do
  sleep 30
done
grep -q 'PREPROCESS_COMPLETE' "$PREPROCESS_LOG/pipeline.log" || {
  echo "200-episode preprocessing did not complete successfully." >&2
  exit 1
}

env BASELINE_GPU=0 PLUCKER_GPU=1 LOG_ROOT="$TRAIN_LOG" \
  "$ROOT/high-level/dp/record/run_glass_door2_skill200_two_act_pipeline_ps1.sh"
