#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DATASET_ID=local/door_a2w_glass_door_jointstate9_gymjacobian_skill_heavier_200
export LOG_ROOT=${LOG_ROOT:-/home/ps/workspace/txc/doorgym/high-level/logs/glass_door_joint9_skill200_eval_20260810}
export BASE_RUN=leroact_a2w_glass_door_joint9_skill200_baseline_seed1000_50k_chunk100_exec50_bs16
export PLK_RUN=leroact_a2w_glass_door_joint9_skill200_plucker_fov55_seed1000_50k_chunk100_exec50_bs16

exec "$SCRIPT_DIR/run_glass_door_joint9_skill50_eval_ps1.sh"
