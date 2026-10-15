#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export NUM_EPISODES=200
export MAX_QUOTA_ROLLOUTS=400
export ROLLOUT_STEPS=1200
export DOOR_NAME=glass_door2
export SKILL_PROGRAM_JSON=/home/ps/workspace/txc/doorgym/high-level/float_ik/door_twin/examples/glass_door2_push_traverse_skill.json
export RAW_NAME=a2w_glass_door2_jointstate9_gymjacobian_skill_glassphysics_200_step1200
export DATASET_ID=local/door_a2w_glass_door2_jointstate9_gymjacobian_skill_glassphysics_200
export LOG_ROOT=${LOG_ROOT:-/home/ps/workspace/txc/doorgym/high-level/logs/glass_door2_joint9_skill200_pipeline_20260810}
export BASELINE_RUN=leroact_a2w_glass_door2_joint9_skill200_baseline_seed1000_50k_chunk100_exec50_bs16
export PLUCKER_RUN=leroact_a2w_glass_door2_joint9_skill200_plucker_fov55_seed1000_50k_chunk100_exec50_bs16

exec "$SCRIPT_DIR/run_glass_door_skill50_two_act_pipeline_ps1.sh"
