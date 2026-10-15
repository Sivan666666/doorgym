#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/home/ps/workspace/txc/doorgym}
LOG_ROOT=${LOG_ROOT:-$ROOT/high-level/logs/glass_door2_joint9_resume_200_train_eval_20260811}
mkdir -p "$LOG_ROOT"

env SIM_GPU=2 BASELINE_GPU=2 PLUCKER_GPU=3 \
  LOG_ROOT="$ROOT/high-level/logs/glass_door2_joint9_skill200_pipeline_resume_20260811" \
  "$ROOT/high-level/dp/record/run_glass_door2_skill200_two_act_pipeline_ps1.sh" \
  >"$LOG_ROOT/train200.log" 2>&1

env BASE_GPU=2 PLK_GPU=3 \
  "$ROOT/high-level/dp/eval/run_glass_door2_joint9_skill200_eval_ps1.sh" \
  >"$LOG_ROOT/eval200.log" 2>&1

env BASE_GPU=2 PLK_GPU=3 \
  "$ROOT/high-level/dp/eval/run_glass_door2_joint9_skill50_eval_ps1.sh" \
  >"$LOG_ROOT/eval50.log" 2>&1

echo GLASS_DOOR2_RESUME_ALL_COMPLETE
