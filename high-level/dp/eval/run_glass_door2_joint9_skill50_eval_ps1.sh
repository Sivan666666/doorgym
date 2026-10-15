#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DATASET_ID=local/door_a2w_glass_door2_jointstate9_gymjacobian_skill_glassphysics_50
export DOOR_NAME=glass_door2
export SKILL=/home/ps/workspace/txc/doorgym/high-level/float_ik/door_twin/examples/glass_door2_push_traverse_skill.json
export EVAL_STEPS=1200
export EVAL_TAG=glass_door2
export DOOR_DOF_INDEX=1
export LOG_ROOT=${LOG_ROOT:-/home/ps/workspace/txc/doorgym/high-level/logs/glass_door2_joint9_skill50_eval_20260810}
export BASE_RUN=leroact_a2w_glass_door2_joint9_skill50_baseline_seed1000_50k_chunk100_exec50_bs16
export PLK_RUN=leroact_a2w_glass_door2_joint9_skill50_plucker_fov55_seed1000_50k_chunk100_exec50_bs16

exec "$SCRIPT_DIR/run_glass_door_joint9_skill50_eval_ps1.sh"
